#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace rtp_llm::crc_copy_benchmark {
namespace {
void requireDone(const StrategyResult& result, const char* name) {
    if (result.status != StrategyStatus::DONE) {
        throw std::runtime_error(std::string(name) + " did not complete; fallback is forbidden; strategy_status="
                                 + std::to_string(static_cast<int>(result.status))
                                 + " copy_status=" + std::to_string(static_cast<int>(result.copy_status)));
    }
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
    for (int layer = 2; layer < 43; ++layer)
        model.attn_config.layer_compress_ratios.push_back(layer % 2 == 0 ? 4 : 128);
    model.hybrid_attention_config.enable_hybrid_attention           = true;
    model.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    test::setDsv4KvCacheSpecs(model, model.attn_config.layer_compress_ratios);

    ParallelismConfig parallel;
    parallel.role_type  = RoleType::PREFILL;
    parallel.tp_size    = 1;
    parallel.world_size = 1;
    auto config         = CacheConfigCreator::createBasicConfig(model, parallel, false, /*gen_num_per_cycle=*/0);
    // Reference warmup geometry: sentinel plus one allocatable block.
    config.linear_step = 1;
    config.finalizeBlockNums(2, RuntimeConfig{});
    return config;
}

Layout deriveLayout(const CacheConfig& config, bool full) {
    Layout result;
    result.name            = full ? "full" : "swa";
    const auto type        = full ? CacheGroupType::FULL : CacheGroupType::SWA;
    size_t     pool_offset = 0, member = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto& group = config.topology().groupById(gid);
        if (!group.policy.enable_prefix_reuse || group.policy.group_type != type)
            continue;
        const auto  pool        = DeviceBlockPoolConfigHelper::createConfigForGroup(config, gid);
        const auto& layer_ids   = config.layerIdsForGroup(gid);
        size_t      local_layer = 0;
        for (const auto& memory : pool.memory_layouts) {
            for (size_t layer = 0; layer < memory.layer_num; ++layer, ++local_layer) {
                const auto append = [&](size_t offset, size_t width) {
                    if (!width)
                        return;
                    if (offset % pool.physical_block_count != 0)
                        throw std::runtime_error("physical pool offset cannot be scaled by capacity");
                    result.geometry.push_back({group.tag,
                                               member,
                                               local_layer,
                                               layer_ids.at(local_layer),
                                               width,
                                               pool_offset + offset / pool.physical_block_count + layer * width});
                    result.sizes.push_back(width);
                };
                append(memory.kv_cache_offset_bytes, memory.kv_block_stride_bytes);
                append(memory.kv_scale_offset_bytes, memory.kv_scale_stride_bytes);
            }
        }
        if (local_layer != layer_ids.size() || pool.total_size_bytes % pool.physical_block_count != 0)
            throw std::runtime_error("physical layer/pool geometry mismatch");
        pool_offset += pool.total_size_bytes / pool.physical_block_count;
        ++member;
    }
    auto sorted = result.geometry;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.offset_per_pool_block < b.offset_per_pool_block;
    });
    size_t covered = 0;
    for (const auto& tile : sorted) {
        if (tile.offset_per_pool_block != covered)
            throw std::runtime_error("physical pool has a gap or overlapping copy tiles");
        covered += tile.bytes;
    }
    if (covered == 0 || covered != pool_offset || covered != result.payload())
        throw std::runtime_error("physical tiles do not cover the reusable backing");
    return result;
}
}  // namespace

size_t Layout::payload() const {
    return std::accumulate(sizes.begin(), sizes.end(), size_t(0));
}

std::string Layout::geometryJson() const {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < geometry.size(); ++i) {
        const auto& tile = geometry[i];
        if (i)
            os << ',';
        os << "{\"tag\":\"" << tile.tag << "\",\"member\":" << tile.member << ",\"local_layer\":" << tile.local_layer
           << ",\"model_layer\":" << tile.model_layer << ",\"bytes\":" << tile.bytes
           << ",\"offset_per_pool_block\":" << tile.offset_per_pool_block << '}';
    }
    os << ']';
    return os.str();
}

const Layout& deepSeekV4FlashLayout(bool full) {
    static const auto config      = flashCacheConfig();
    static const auto full_layout = deriveLayout(config, true);
    static const auto swa_layout  = deriveLayout(config, false);
    return full ? full_layout : swa_layout;
}

size_t maximumLayoutTiles() {
    return std::max(deepSeekV4FlashLayout(true).sizes.size(), deepSeekV4FlashLayout(false).sizes.size());
}

size_t maximumLayoutPayload() {
    return std::max(deepSeekV4FlashLayout(true).payload(), deepSeekV4FlashLayout(false).payload());
}

struct FrameworkCopyPlan::Impl {
    DeviceHostCopyPlan store;
    DeviceHostCopyPlan load;

    Impl(const std::vector<BenchmarkCopyItem>& items, const Layout& layout) {
        store.device_to_host = true;
        load.device_to_host  = false;
        if (!items.empty()) {
            store.host = load.host = {items.front().host, items.front().payload_bytes, items.front().capacity_bytes};
        }
        size_t count = 0;
        for (const auto& item : items)
            count += item.tiles.size();
        store.copy_tiles.reserve(count);
        load.copy_tiles.reserve(count);
        for (const auto& item : items) {
            if (item.tiles.size() != layout.geometry.size())
                throw std::runtime_error("framework plan does not match derived physical geometry");
            for (size_t i = 0; i < item.tiles.size(); ++i) {
                const auto&        tile = item.tiles[i];
                DeviceHostCopyTile copy;
                copy.host_addr         = static_cast<unsigned char*>(item.host) + tile.offset;
                copy.device_addr       = tile.device;
                copy.host_offset       = tile.offset;
                copy.bytes             = tile.bytes;
                copy.device_index      = 0;
                copy.member_group_id   = layout.geometry[i].member;
                copy.local_layer_index = layout.geometry[i].local_layer;
                store.copy_tiles.push_back(copy);
                load.copy_tiles.push_back(copy);
            }
        }
    }
};

FrameworkCopyPlan::FrameworkCopyPlan(const std::vector<BenchmarkCopyItem>& items, const Layout& layout):
    impl_(std::make_unique<Impl>(items, layout)) {}
FrameworkCopyPlan::~FrameworkCopyPlan() = default;

uintptr_t FrameworkCopyPlan::touchMetadata() const {
    uintptr_t sum = 0;
    for (const auto* plan : {&impl_->store, &impl_->load}) {
        sum += plan->device_to_host ^ plan->group_set_id ^ reinterpret_cast<uintptr_t>(plan->host.base)
               ^ plan->host.payload_bytes ^ plan->host.capacity_bytes;
        for (const auto& tile : plan->copy_tiles)
            sum += reinterpret_cast<uintptr_t>(tile.host_addr) ^ reinterpret_cast<uintptr_t>(tile.device_addr)
                   ^ tile.host_offset ^ tile.bytes ^ uintptr_t(tile.device_index) ^ tile.member_group_id
                   ^ tile.local_layer_index;
    }
    return sum;
}

struct FrameworkCopies::Impl {
    CudaBatchDeviceHostCopyStrategy        batch;
    StagedSmDeviceHostCopyStrategy         staged;
    std::shared_ptr<DeviceHostCopyStreams> streams;

    explicit Impl(int device): streams(acquireDeviceHostCopyStreams(device)) {}
};

FrameworkCopies::FrameworkCopies(int device) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    impl_ = std::make_unique<Impl>(device);
}

FrameworkCopies::~FrameworkCopies() = default;

void FrameworkCopies::copyBatch(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->batch.tryExecute(store ? plan.impl_->store : plan.impl_->load,
                                        DeviceHostCopyOptions{},
                                        DeviceHostCopyExecutionContext{impl_->streams,
                                                                       store ? DeviceHostCopyDirection::D2H :
                                                                               DeviceHostCopyDirection::H2D}),
                "CUDA 1D batch");
}

void FrameworkCopies::copyStaged(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->staged.tryExecute(store ? plan.impl_->store : plan.impl_->load,
                                         DeviceHostCopyOptions{},
                                         DeviceHostCopyExecutionContext{impl_->streams,
                                                                        store ? DeviceHostCopyDirection::D2H :
                                                                                DeviceHostCopyDirection::H2D}),
                "rebased production staged without CRC");
}

struct Framework3DCopies::Impl {
    struct Plan {
        std::vector<HostBufferView>     hosts;
        std::vector<TransferDescriptor> store, load;
        std::vector<const GroupSet*>    groups;
    };
    GroupSetPtr                            group;
    Cuda3DBatchDeviceHostCopyStrategy      strategy;
    std::shared_ptr<DeviceHostCopyStreams> streams;
    std::vector<Plan>                      plans;

    Impl(const Layout& layout, size_t blocks, int device): streams(acquireDeviceHostCopyStreams(device)) {
        const auto                      config = flashCacheConfig();
        const auto                      type   = layout.name == "full" ? CacheGroupType::FULL : CacheGroupType::SWA;
        std::vector<DeviceBlockPoolPtr> pools;
        std::vector<size_t>             ids;
        for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
            const auto& g = config.topology().groupById(gid);
            if (!g.policy.enable_prefix_reuse || g.policy.group_type != type)
                continue;
            auto pc =
                std::make_shared<DeviceBlockPoolConfig>(DeviceBlockPoolConfigHelper::createConfigForGroup(config, gid));
            if (pc->memory_layouts.size() != 1 || pc->memory_layouts.front().hasScale())
                throw std::runtime_error("3D benchmark requires current Flash single-layout pools");
            auto& m = pc->memory_layouts.front();
            if (m.kv_cache_offset_bytes != 0 || m.kv_block_stride_bytes != g.kv_block_stride_bytes)
                throw std::runtime_error("3D benchmark pool layout differs from CPU oracle");
            pc->physical_block_count      = blocks;
            m.block_num                   = blocks;
            m.kv_block_pool_size_bytes    = blocks * m.layer_num * m.kv_block_stride_bytes;
            m.total_size_bytes            = m.kv_block_pool_size_bytes;
            pc->total_size_bytes          = m.total_size_bytes;
            pc->use_device_malloc_backing = true;
            pc->use_pinned_cpu_backing    = false;
            auto pool                     = std::make_shared<DeviceBlockPool>(pc);
            if (!pool->init())
                throw std::runtime_error("3D benchmark pool initialization failed");
            pools.push_back(pool);
            ids.push_back(gid);
        }
        if (type == CacheGroupType::FULL)
            group = std::make_shared<FullGroupSet>(pools, nullptr, nullptr);
        else
            group = std::make_shared<SWAGroupSet>(128, 128, pools, nullptr, nullptr);
        group->initialize(0, config.topologyPtr(), ids);
        if (group->payloadBytes() != layout.payload() || group->copy3DTemplates().size() != pools.size())
            throw std::runtime_error("3D production templates do not cover benchmark layout");
    }
};

Framework3DCopies::Framework3DCopies(const Layout& layout, size_t blocks, int device):
    impl_(std::make_unique<Impl>(layout, blocks, device)) {}
Framework3DCopies::~Framework3DCopies() = default;

size_t Framework3DCopies::addPlan(const std::vector<BenchmarkCopyItem>& items, const std::vector<int>& blocks) {
    if (items.size() != blocks.size())
        throw std::runtime_error("3D plan size mismatch");
    Impl::Plan plan;
    for (size_t i = 0; i < items.size(); ++i) {
        plan.hosts.push_back({items[i].host, items[i].payload_bytes, items[i].capacity_bytes});
        std::vector<BlockIdxType> member_blocks(impl_->group->devicePools().size(), blocks[i]);
        plan.store.push_back(TransferDescriptor::deviceToHost(0, member_blocks, i + 1));
        plan.load.push_back(TransferDescriptor::hostToDevice(0, i + 1, member_blocks));
        plan.groups.push_back(impl_->group.get());
    }
    impl_->plans.push_back(std::move(plan));
    return impl_->plans.size() - 1;
}

uintptr_t Framework3DCopies::touchMetadata(size_t index) const {
    const auto& p   = impl_->plans.at(index);
    uintptr_t   sum = 0;
    for (const auto& h : p.hosts)
        sum += reinterpret_cast<uintptr_t>(h.base) ^ h.payload_bytes ^ h.capacity_bytes;
    for (const auto* descriptors : {&p.store, &p.load})
        for (const auto& d : *descriptors)
            for (const auto b : d.blocksAt(Tier::DEVICE))
                sum += b;
    for (const auto& t : impl_->group->copy3DTemplates())
        sum +=
            reinterpret_cast<uintptr_t>(t.device_base) ^ t.host_offset ^ t.width_bytes ^ t.layer_count ^ t.device_pitch;
    return sum;
}

void Framework3DCopies::copy(size_t index, bool store) {
    const auto& p = impl_->plans.at(index);
    requireDone(impl_->strategy.tryExecute(
                    p.hosts,
                    store ? p.store : p.load,
                    p.groups,
                    {impl_->streams, store ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D}),
                "production CUDA 3D batch");
}

void Framework3DCopies::copyOracle(void* contiguous, bool into_pools) {
    auto* cursor = static_cast<unsigned char*>(contiguous);
    for (const auto& pool : impl_->group->devicePools()) {
        auto error = cudaMemcpy(into_pools ? pool->getBaseAddress() : cursor,
                                into_pools ? cursor : pool->getBaseAddress(),
                                pool->getTotalSizeBytes(),
                                cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        cursor += pool->getTotalSizeBytes();
    }
}

void Framework3DCopies::poison(unsigned char value) {
    for (const auto& pool : impl_->group->devicePools()) {
        const auto error = cudaMemset(pool->getBaseAddress(), value, pool->getTotalSizeBytes());
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
    const auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace rtp_llm::crc_copy_benchmark
