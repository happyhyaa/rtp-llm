#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

namespace rtp_llm::benchmark {
namespace {

constexpr int kBatchLimit = 500;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void require(bool condition, const std::string& reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

struct Options {
    std::string output;
    std::string group{"full"};
    std::string direction{"d2h"};
    std::string source_sha256;
    std::string binary_sha256;
    int         descriptors{50};
    int         batch_limit{kBatchLimit};
    int         samples{10};
    int         warmup{3};
    bool        correctness_only{false};
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--correctness-only") {
            options.correctness_only = true;
            continue;
        }
        require(index + 1 < argc, "missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--output") {
            options.output = value;
        } else if (key == "--group") {
            options.group = value;
        } else if (key == "--direction") {
            options.direction = value;
        } else if (key == "--descriptors") {
            options.descriptors = std::stoi(value);
        } else if (key == "--batch-limit") {
            options.batch_limit = std::stoi(value);
        } else if (key == "--samples") {
            options.samples = std::stoi(value);
        } else if (key == "--warmup") {
            options.warmup = std::stoi(value);
        } else if (key == "--source-sha256") {
            options.source_sha256 = value;
        } else if (key == "--binary-sha256") {
            options.binary_sha256 = value;
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    require(!options.output.empty(), "--output is required");
    require(options.group == "full" || options.group == "swa", "--group must be full or swa");
    require(options.direction == "d2h" || options.direction == "h2d", "--direction must be d2h or h2d");
    require(options.batch_limit == kBatchLimit, "--batch-limit must be 500");
    require(options.descriptors >= 50 && options.descriptors <= options.batch_limit && options.descriptors % 50 == 0,
            "--descriptors must be 50,100,...,500");
    require(options.samples > 0 && options.warmup >= 0, "invalid sample count");
    if (!options.correctness_only) {
        require(options.source_sha256.size() == 64 && options.binary_sha256.size() == 64,
                "timing requires source and binary SHA256 identifiers");
    }
    return options;
}

CacheConfig flashCacheConfig() {
    ModelConfig model;
    model.num_layers                          = 43;
    model.hidden_size                         = 4096;
    model.attn_config.head_num                = 64;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 512;
    model.attn_config.rope_head_dim           = 64;
    model.attn_config.sliding_window          = 128;
    model.attn_config.indexer_head_dim        = 128;
    model.attn_config.indexer_head_num        = 64;
    model.attn_config.indexer_topk            = 512;
    model.attn_config.o_groups                = 8;
    model.attn_config.o_lora_rank             = 1024;
    model.attn_config.tokens_per_block        = 128;
    model.attn_config.kernel_tokens_per_block = 128;
    model.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;
    model.attn_config.layer_compress_ratios   = {0, 0};
    for (int layer = 2; layer < 43; ++layer) {
        model.attn_config.layer_compress_ratios.push_back(layer % 2 == 0 ? 4 : 128);
    }
    model.hybrid_attention_config.enable_hybrid_attention = true;
    test::setDsv4KvCacheSpecs(model, model.attn_config.layer_compress_ratios);

    ParallelismConfig parallel;
    parallel.role_type                          = RoleType::PREFILL;
    parallel.tp_size                            = 4;
    parallel.world_size                         = 4;
    parallel.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    parallel.prefill_cp_config.kv_cache_sharded = true;
    parallel.prefill_cp_config.prefill_cp_size  = 4;
    return CacheConfigCreator::createWarmupConfig(model, parallel, 0);
}

size_t payloadBytes(const CacheConfig& config, CacheGroupType type) {
    size_t total = 0;
    for (const auto& group : config.groups()) {
        if (group.policy.enable_prefix_reuse && group.policy.group_type == type) {
            total += config.blockSizeBytesForGroup(group.tag);
        }
    }
    return total;
}

struct Setup {
    CacheConfig                     config;
    GroupSetPtr                     group;
    std::vector<DeviceBlockPoolPtr> pools;
    size_t                          payload{0};
    size_t                          pool_blocks{0};

    Setup(bool full, size_t l2, size_t descriptors): config(flashCacheConfig()) {
        const auto type = full ? CacheGroupType::FULL : CacheGroupType::SWA;
        payload         = payloadBytes(config, type);
        require(payload > 0, "empty reusable backing");
        const size_t minimum = std::max(8 * l2, (descriptors + 1) * payload);
        pool_blocks          = (minimum + payload - 1) / payload + 1;
        config.finalizeBlockNums(static_cast<uint32_t>(pool_blocks), RuntimeConfig{});
        std::vector<std::string> tags;
        for (const auto& member : config.groups()) {
            if (!member.policy.enable_prefix_reuse || member.policy.group_type != type) {
                continue;
            }
            auto pool_config = std::make_shared<DeviceBlockPoolConfig>(
                DeviceBlockPoolConfigHelper::createConfigForGroup(config, member));
            pool_config->use_device_malloc_backing = true;
            auto pool                              = std::make_shared<DeviceBlockPool>(pool_config);
            require(pool->init(), "device pool initialization failed");
            pools.push_back(std::move(pool));
            tags.push_back(member.tag);
        }
        require(!tags.empty(), "no reusable members");
        if (full) {
            group = std::make_shared<FullGroupSet>(pools, nullptr, nullptr);
        } else {
            const auto& first = config.group(tags.front());
            group             = std::make_shared<SWAGroupSet>(
                first.policy.sliding_window_size, first.cacheKeyTokenStride(), pools, nullptr, nullptr);
        }
        group->initialize(0, config.topologyPtr(), std::move(tags), payload);
        require(!group->copy3DTemplates().empty(), "production 3D strategy is not applicable to this layout");
    }
};

constexpr unsigned char kGuard = 0xd3;

size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

unsigned char pattern(size_t index) {
    uint64_t value = index ^ 0x123456789abcdef0ULL;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return static_cast<unsigned char>(value ^ (value >> 31));
}

struct PinnedHost {
    unsigned char* data{nullptr};
    explicit PinnedHost(size_t bytes) {
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&data), bytes));
    }
    ~PinnedHost() {
        if (data) {
            cudaFreeHost(data);
        }
    }
    PinnedHost(const PinnedHost&)            = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;
};

struct DeviceBuffer {
    void* data{nullptr};
    explicit DeviceBuffer(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes));
    }
    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t value{};
    Stream() {
        checkCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream() {
        cudaStreamDestroy(value);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
};

__global__ void evictCache(unsigned char* bytes, size_t count, unsigned int salt) {
    auto* data = reinterpret_cast<uint4*>(bytes);
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count / 16;
         index += size_t(gridDim.x) * blockDim.x) {
        uint4 value = data[index];
        value.x += salt;
        value.y ^= salt;
        value.z += 1;
        value.w ^= value.x;
        data[index] = value;
    }
}

struct Tile {
    void*  device;
    size_t host_offset;
    size_t bytes;
    size_t member;
};

struct Item {
    unsigned char*    host;
    std::vector<Tile> tiles;
};

struct Plan {
    std::vector<Item>               items;
    std::vector<HostBufferView>     hosts;
    std::vector<TransferDescriptor> descriptors;
    std::vector<const GroupSet*>    groups;
};

Plan createPlan(const Setup&            setup,
                const Options&          options,
                unsigned char*          host,
                size_t                  host_stride,
                const std::vector<int>& blocks,
                size_t                  rotation) {
    Plan plan;
    for (int index = 0; index < options.descriptors; ++index) {
        const int block = blocks[(rotation * options.descriptors + index) % blocks.size()];
        Item      item{host + index * host_stride, {}};
        size_t    offset = 0;
        for (size_t member = 0; member < setup.pools.size(); ++member) {
            const auto& tag       = setup.group->groupTags()[member];
            const auto  layer_ids = setup.config.layerIdsForGroup(tag);
            for (size_t layer = 0; layer < layer_ids.size(); ++layer) {
                const auto buffers = setup.pools[member]->convertIndexToBuffer(static_cast<int>(layer), block);
                for (const auto& buffer : buffers) {
                    if (buffer.size_bytes == 0) {
                        continue;
                    }
                    require(buffer.addr != nullptr && buffer.size_bytes <= setup.payload - offset,
                            "physical tile exceeds backing payload");
                    item.tiles.push_back({buffer.addr, offset, buffer.size_bytes, member});
                    offset += buffer.size_bytes;
                }
            }
        }
        require(offset == setup.payload, "prebuilt tiles do not cover the physical backing");
        plan.hosts.push_back({item.host, setup.payload, host_stride});
        plan.groups.push_back(setup.group.get());
        const std::vector<BlockIdxType> member_blocks(setup.pools.size(), block);
        plan.descriptors.push_back(options.direction == "d2h" ?
                                       TransferDescriptor::deviceToHost(0, member_blocks, index + 1) :
                                       TransferDescriptor::hostToDevice(0, index + 1, member_blocks));
        plan.items.push_back(std::move(item));
    }
    return plan;
}

std::vector<size_t> affineRows(const Plan& plan) {
    std::vector<size_t> rows_per_op;
    const auto&         tiles = plan.items.front().tiles;
    for (size_t first = 0; first < tiles.size();) {
        const auto& head  = tiles[first];
        const auto  begin = reinterpret_cast<uintptr_t>(head.device);
        size_t      rows  = 1;
        if (first + 1 < tiles.size()) {
            const auto& next         = tiles[first + 1];
            const auto  next_address = reinterpret_cast<uintptr_t>(next.device);
            if (next.bytes == head.bytes && next.host_offset == head.host_offset + head.bytes
                && next_address >= begin + head.bytes) {
                const size_t pitch = next_address - begin;
                while (first + rows < tiles.size()) {
                    const auto& tile = tiles[first + rows];
                    if (tile.bytes != head.bytes || tile.host_offset != head.host_offset + rows * head.bytes
                        || reinterpret_cast<uintptr_t>(tile.device) != begin + rows * pitch) {
                        break;
                    }
                    ++rows;
                }
            }
        }
        rows_per_op.push_back(rows);
        first += rows;
    }
    return rows_per_op;
}

class Benchmark1476Path {
public:
    Benchmark1476Path(): stream_(at::cuda::getStreamFromPool(false, 0)) {}

    void copy(const Plan& plan, const std::vector<size_t>& group_rows, bool d2h) {
        std::vector<cudaMemcpy3DBatchOp> ops;
        ops.reserve(plan.items.size() * group_rows.size());
        for (const auto& item : plan.items) {
            size_t first = 0;
            for (size_t rows : group_rows) {
                require(rows >= 1 && first + rows <= item.tiles.size(), "invalid 3D affine span");
                const auto&  head  = item.tiles[first];
                const auto   begin = reinterpret_cast<uintptr_t>(head.device);
                const size_t pitch =
                    rows == 1 ? head.bytes : reinterpret_cast<uintptr_t>(item.tiles[first + 1].device) - begin;
                require(pitch >= head.bytes, "invalid 3D device pitch");
                for (size_t row = 0; row < rows; ++row) {
                    const auto& tile = item.tiles[first + row];
                    require(tile.bytes == head.bytes && tile.host_offset == head.host_offset + row * head.bytes
                                && reinterpret_cast<uintptr_t>(tile.device) == begin + row * pitch,
                            "non-affine tile inside declared pool");
                }
                cudaMemcpy3DBatchOp op{};
                op.src.type = op.dst.type = cudaMemcpyOperandTypePointer;
                auto* host                = item.host + head.host_offset;
                op.src.op.ptr.ptr         = d2h ? head.device : host;
                op.dst.op.ptr.ptr         = d2h ? host : head.device;
                op.src.op.ptr.rowLength   = d2h ? pitch : head.bytes;
                op.dst.op.ptr.rowLength   = d2h ? head.bytes : pitch;
                op.src.op.ptr.layerHeight = op.dst.op.ptr.layerHeight = rows;
                op.extent                                             = make_cudaExtent(head.bytes, rows, 1);
                op.srcAccessOrder                                     = cudaMemcpySrcAccessOrderStream;
                ops.push_back(op);
                first += rows;
            }
            require(first == item.tiles.size(), "uncovered 3D tiles");
        }
        cudaError_t submission;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            submission = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream_.stream());
        }
        const auto completion = cudaStreamSynchronize(stream_.stream());
        checkCuda(submission);
        checkCuda(completion);
    }

private:
    at::cuda::CUDAStream stream_;
    std::mutex           mutex_;
};

void copyProduction(Cuda3DBatchDeviceHostCopyStrategy& production, const Plan& plan) {
    const auto result = production.tryExecute(plan.hosts, plan.descriptors, plan.groups);
    require(result.status == StrategyStatus::DONE, "production 1517 3D strategy was not applicable or failed");
}

size_t deviceOffset(const Setup& setup, const Tile& tile) {
    const auto* base = static_cast<const unsigned char*>(setup.pools[tile.member]->getBaseAddress());
    const auto* addr = static_cast<const unsigned char*>(tile.device);
    require(addr >= base && static_cast<size_t>(addr - base) <= setup.pools[tile.member]->getTotalSizeBytes()
                && tile.bytes <= setup.pools[tile.member]->getTotalSizeBytes() - (addr - base),
            "tile is outside its device pool");
    return static_cast<size_t>(addr - base);
}

void prepareHost(const Setup& setup, const Plan& plan, const std::vector<std::vector<unsigned char>>& oracle) {
    for (const auto& item : plan.items) {
        for (const auto& tile : item.tiles) {
            std::memcpy(
                item.host + tile.host_offset, oracle[tile.member].data() + deviceOffset(setup, tile), tile.bytes);
        }
    }
}

size_t verifyCopy(const Setup&                                   setup,
                  const Plan&                                    plan,
                  const std::vector<std::vector<unsigned char>>& oracle,
                  size_t                                         host_stride,
                  bool                                           d2h) {
    size_t checked = 0;
    if (d2h) {
        for (const auto& item : plan.items) {
            std::vector<unsigned char> expected(host_stride, kGuard);
            for (const auto& tile : item.tiles) {
                std::memcpy(expected.data() + tile.host_offset,
                            oracle[tile.member].data() + deviceOffset(setup, tile),
                            tile.bytes);
            }
            require(std::memcmp(item.host, expected.data(), host_stride) == 0, "D2H payload or padding mismatch");
            checked += host_stride;
        }
    }
    for (size_t member = 0; member < setup.pools.size(); ++member) {
        const size_t               size     = oracle[member].size();
        std::vector<unsigned char> expected = d2h ? oracle[member] : std::vector<unsigned char>(size, kGuard);
        if (!d2h) {
            for (const auto& item : plan.items) {
                for (const auto& tile : item.tiles) {
                    if (tile.member != member) {
                        continue;
                    }
                    const size_t offset = deviceOffset(setup, tile);
                    std::memcpy(expected.data() + offset, oracle[member].data() + offset, tile.bytes);
                }
            }
        }
        std::vector<unsigned char> actual(size);
        checkCuda(cudaMemcpy(actual.data(), setup.pools[member]->getBaseAddress(), size, cudaMemcpyDeviceToHost));
        if (actual != expected) {
            const auto   mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
            const size_t offset   = mismatch.first - actual.begin();
            throw std::runtime_error("device pool mismatch member=" + std::to_string(member)
                                     + " byte=" + std::to_string(offset) + " actual=" + std::to_string(*mismatch.first)
                                     + " expected=" + std::to_string(*mismatch.second));
        }
        checked += size;
    }
    return checked;
}

std::string deviceUuid(const cudaUUID_t& uuid) {
    std::ostringstream output;
    output << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) {
            output << '-';
        }
        output << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[index]));
    }
    return output.str();
}

int run(const Options& options) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    checkCuda(cudaSetDevice(0));
    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    require(prop.l2CacheSize > 0, "GPU L2 size unavailable");
    int compute_mode = 0;
    checkCuda(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, 0));
    if (!options.correctness_only) {
        require(compute_mode == cudaComputeModeExclusiveProcess, "timing requires EXCLUSIVE_PROCESS mode");
    }
    int driver_version = 0, runtime_version = 0;
    checkCuda(cudaDriverGetVersion(&driver_version));
    checkCuda(cudaRuntimeGetVersion(&runtime_version));
    require(driver_version >= 13000 && runtime_version >= 13000, "CUDA13 driver and runtime required");
    Setup        setup(options.group == "full", static_cast<size_t>(prop.l2CacheSize), options.descriptors);
    const size_t host_stride = alignUp(setup.payload + sizeof(uint32_t), 16);
    PinnedHost   host(options.descriptors * host_stride);
    std::vector<std::vector<unsigned char>> oracle;
    for (size_t member = 0; member < setup.pools.size(); ++member) {
        std::vector<unsigned char> source(setup.pools[member]->getTotalSizeBytes());
        for (size_t index = 0; index < source.size(); ++index) {
            source[index] = pattern(index + member * 131);
        }
        oracle.push_back(std::move(source));
    }
    std::vector<int> blocks(setup.pool_blocks - 1);
    std::iota(blocks.begin(), blocks.end(), 1);
    std::mt19937 random(20260922);
    std::shuffle(blocks.begin(), blocks.end(), random);
    std::vector<Plan> plans;
    const size_t      rotations = (blocks.size() + options.descriptors - 1) / options.descriptors;
    plans.reserve(rotations);
    for (size_t rotation = 0; rotation < rotations; ++rotation) {
        plans.push_back(createPlan(setup, options, host.data, host_stride, blocks, rotation));
    }
    const auto                        group_rows = affineRows(plans.front());
    Cuda3DBatchDeviceHostCopyStrategy production;
    Benchmark1476Path                 benchmark;
    const bool                        d2h = options.direction == "d2h";
    std::ofstream                     output(options.output);
    require(bool(output), "cannot open output file");
    output << std::setprecision(12);
    output << "{\"type\":\"metadata\",\"group\":\"" << options.group << "\",\"direction\":\"" << options.direction
           << "\",\"business_count\":1,\"descriptors\":" << options.descriptors
           << ",\"batch_limit\":" << options.batch_limit << ",\"payload_bytes\":" << setup.payload
           << ",\"pool_blocks\":" << setup.pool_blocks << ",\"tp_size\":4,\"cp_size\":4"
           << ",\"seq_size_per_block\":128,\"kernel_seq_size_per_block\":128,\"target_layers\":43"
           << ",\"gpu\":\"" << prop.name << "\",\"gpu_uuid\":\"" << deviceUuid(prop.uuid)
           << "\",\"driver\":" << driver_version << ",\"runtime\":" << runtime_version
           << ",\"member_pools\":" << setup.pools.size()
           << ",\"tiles_per_descriptor\":" << plans.front().items.front().tiles.size()
           << ",\"production_ops_per_descriptor\":" << setup.group->copy3DTemplates().size()
           << ",\"benchmark_ops_per_descriptor\":" << group_rows.size()
           << ",\"production_ops_per_copy_call\":" << setup.group->copy3DTemplates().size() * options.descriptors
           << ",\"benchmark_ops_per_copy_call\":" << group_rows.size() * options.descriptors
           << ",\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
           << ",\"compute_mode\":" << compute_mode << ",\"source_sha256\":\"" << options.source_sha256
           << "\",\"binary_sha256\":\"" << options.binary_sha256 << "\""
           << ",\"timing\":\"wall_per_copy_call\",\"regime\":\"cold\""
           << ",\"variants\":[\"production_1517\",\"benchmark_1476\"]}\n";
    for (int variant = 0; variant < 2; ++variant) {
        size_t checked = 0;
        for (size_t rotation = 0; rotation < std::min<size_t>(2, plans.size()); ++rotation) {
            output << "{\"type\":\"correctness_attempt\",\"variant\":\""
                   << (variant == 0 ? "production_1517" : "benchmark_1476") << "\",\"rotation\":" << rotation << "}\n";
            output.flush();
            const auto& plan = plans[rotation];
            std::memset(host.data, kGuard, options.descriptors * host_stride);
            for (size_t member = 0; member < setup.pools.size(); ++member) {
                if (d2h) {
                    checkCuda(cudaMemcpy(setup.pools[member]->getBaseAddress(),
                                         oracle[member].data(),
                                         oracle[member].size(),
                                         cudaMemcpyHostToDevice));
                } else {
                    checkCuda(cudaMemset(setup.pools[member]->getBaseAddress(), kGuard, oracle[member].size()));
                }
            }
            if (!d2h) {
                prepareHost(setup, plan, oracle);
            }
            if (variant == 0) {
                copyProduction(production, plan);
            } else {
                benchmark.copy(plan, group_rows, d2h);
            }
            checked += verifyCopy(setup, plan, oracle, host_stride, d2h);
        }
        output << "{\"type\":\"correctness\",\"variant\":\"" << (variant == 0 ? "production_1517" : "benchmark_1476")
               << "\",\"success\":true"
               << ",\"rotations_checked\":" << std::min<size_t>(2, plans.size()) << ",\"bytes_checked\":" << checked
               << "}\n";
    }
    if (options.correctness_only) {
        output << "{\"type\":\"complete\",\"success\":true}\n";
        return 0;
    }
    if (d2h) {
        for (size_t member = 0; member < setup.pools.size(); ++member) {
            checkCuda(cudaMemcpy(setup.pools[member]->getBaseAddress(),
                                 oracle[member].data(),
                                 oracle[member].size(),
                                 cudaMemcpyHostToDevice));
        }
    }
    const size_t eviction_bytes = alignUp(8 * static_cast<size_t>(prop.l2CacheSize), 16);
    DeviceBuffer eviction(eviction_bytes);
    Stream       setup_stream;
    checkCuda(cudaMemset(eviction.data, 0x5a, eviction_bytes));
    std::array<int, 2>                 order{0, 1};
    std::array<std::vector<double>, 2> latencies;
    const size_t                       bytes_per_call = setup.payload * options.descriptors;
    const int                          first_variant  = random() % 2;
    for (int round = -options.warmup; round < options.samples; ++round) {
        const size_t rotation = static_cast<size_t>(round + options.warmup) % plans.size();
        const auto&  plan     = plans[rotation];
        order = (round + options.warmup + first_variant) % 2 == 0 ? std::array<int, 2>{0, 1} : std::array<int, 2>{1, 0};
        for (size_t position = 0; position < order.size(); ++position) {
            const int variant = order[position];
            if (d2h) {
                std::memset(host.data, kGuard, options.descriptors * host_stride);
            } else {
                for (size_t member = 0; member < setup.pools.size(); ++member) {
                    checkCuda(cudaMemset(setup.pools[member]->getBaseAddress(), kGuard, oracle[member].size()));
                }
                std::memset(host.data, kGuard, options.descriptors * host_stride);
                prepareHost(setup, plan, oracle);
            }
            evictCache<<<4096, 256, 0, setup_stream.value>>>(static_cast<unsigned char*>(eviction.data),
                                                             eviction_bytes,
                                                             static_cast<unsigned int>(round + options.warmup + 1));
            checkCuda(cudaGetLastError());
            checkCuda(cudaStreamSynchronize(setup_stream.value));
            const auto begin = std::chrono::steady_clock::now();
            if (variant == 0) {
                copyProduction(production, plan);
            } else {
                benchmark.copy(plan, group_rows, d2h);
            }
            const double us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
            verifyCopy(setup, plan, oracle, host_stride, d2h);
            if (round >= 0) {
                latencies[variant].push_back(us);
                output << "{\"type\":\"sample\",\"variant\":\"" << (variant == 0 ? "production_1517" : "benchmark_1476")
                       << "\",\"round\":" << round << ",\"position\":" << position << ",\"rotation\":" << rotation
                       << ",\"us\":" << us << ",\"copy_bytes\":" << bytes_per_call
                       << ",\"throughput_gbps\":" << bytes_per_call / (us * 1000.0) << ",\"verified\":true}\n";
            }
        }
        output.flush();
    }
    for (int variant = 0; variant < 2; ++variant) {
        auto& values = latencies[variant];
        std::sort(values.begin(), values.end());
        const double p50_us = (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2.0;
        output << "{\"type\":\"summary\",\"variant\":\"" << (variant == 0 ? "production_1517" : "benchmark_1476")
               << "\",\"samples\":" << values.size() << ",\"p50_us\":" << p50_us << ",\"copy_bytes\":" << bytes_per_call
               << ",\"throughput_gbps\":" << bytes_per_call / (p50_us * 1000.0) << "}\n";
    }
    output << "{\"type\":\"complete\",\"success\":true}\n";
    require(bool(output), "failed to write benchmark output");
    return 0;
}

}  // namespace
}  // namespace rtp_llm::benchmark

int main(int argc, char** argv) {
    try {
        return rtp_llm::benchmark::run(rtp_llm::benchmark::parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
