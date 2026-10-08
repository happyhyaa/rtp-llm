// Manual end-to-end cache-copy microbenchmark. See crc_copy_benchmark_test.
#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if CUDART_VERSION < 13000
#error "CrcCopyBenchmark requires --config=cuda13."
#endif

namespace rtp_llm::crc_copy_benchmark {
namespace {
constexpr int           kMaxBackings     = 32;
constexpr int           kEvictMultiplier = 8;
constexpr unsigned char deviceGuard      = 0xd3;
const char*             syncNames[]{"copy1d_batch", "staged_no_crc", "copy3d_batch"};

void requireSync(bool ok, std::string_view reason) {
    if (!ok)
        throw std::runtime_error(std::string(reason));
}
void checkedCuda(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line) + ": "
                                 + cudaGetErrorString(error));
    }
}
#define cu(expression) checkedCuda((expression), #expression, __LINE__)

struct Buffer {
    void* p{nullptr};
    bool  pinned;
    explicit Buffer(size_t bytes, bool host = false): pinned(host) {
        if (host)
            cu(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
        else
            cu(cudaMalloc(&p, bytes));
    }
    ~Buffer() {
        if (pinned)
            cudaFreeHost(p);
        else
            cudaFree(p);
    }
    Buffer(const Buffer&)                   = delete;
    Buffer&        operator=(const Buffer&) = delete;
    unsigned char* bytes() const {
        return static_cast<unsigned char*>(p);
    }
};
struct Stream {
    cudaStream_t s{};
    Stream() {
        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    ~Stream() {
        cudaStreamDestroy(s);
    }
    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;
};
size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

__host__ __device__ unsigned char pattern(uint64_t index) {
    index ^= 0x123456789abcdef0ULL;
    index ^= index >> 30;
    index *= 0xbf58476d1ce4e5b9ULL;
    index ^= index >> 27;
    index *= 0x94d049bb133111ebULL;
    index ^= index >> 31;
    return static_cast<unsigned char>(index);
}
__global__ void initRandom(unsigned char* bytes, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        bytes[i] = pattern(i);
}
__global__ void evictCache(unsigned char* bytes, size_t count, unsigned int salt) {
    auto* data = reinterpret_cast<uint4*>(bytes);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count / 16; i += size_t(gridDim.x) * blockDim.x) {
        uint4 value = data[i];
        value.x += salt;
        value.y ^= salt;
        value.z += 1;
        value.w ^= value.x;
        data[i] = value;
    }
}
const Layout& orderedLayout(bool full) {
    return deepSeekV4FlashLayout(full);
}
const Layout& full() {
    return orderedLayout(true);
}
const Layout& swa() {
    return orderedLayout(false);
}

struct Plan {
    int                                n;
    std::vector<int>                   blocks;
    std::vector<BenchmarkCopyItem>     items;
    std::unique_ptr<FrameworkCopyPlan> nativePlans;
};
struct Workspace {
    Layout                             layout;
    size_t                             p, e, hostStride, sourceBytes, evictBytes, stride;
    int                                poolBlocks;
    Stream                             stream;
    std::unique_ptr<Buffer>            source, eviction, host;
    std::vector<std::unique_ptr<Plan>> plans;
    std::vector<size_t>                planStarts;
    std::vector<int>                   rotationCounts;

    Workspace(Layout shape, size_t l2): layout(std::move(shape)) {
        p = layout.payload();
        // Preserve the reference benchmark host allocation stride; copy only payload.
        e                    = alignUp(p + 4, 16);
        stride               = alignUp(maximumLayoutPayload() + 4, 16);
        hostStride           = alignUp(e, 4096);
        const size_t minimum = std::max(kEvictMultiplier * l2, size_t(4 * kMaxBackings) * p);
        poolBlocks           = int((minimum + p - 1) / p) + 1;  // block 0 is the framework sentinel
        sourceBytes          = size_t(poolBlocks) * p;
        evictBytes           = alignUp(kEvictMultiplier * l2, 16);
        source               = std::make_unique<Buffer>(sourceBytes);
        eviction             = std::make_unique<Buffer>(evictBytes);
        host                 = std::make_unique<Buffer>(kMaxBackings * hostStride, true);
        initRandom<<<4096, 256, 0, stream.s>>>(eviction->bytes(), evictBytes);
        cu(cudaGetLastError());
        cu(cudaStreamSynchronize(stream.s));
        std::mt19937 rng(20260922);
        for (int n = 1; n <= kMaxBackings; ++n) {
            planStarts.push_back(plans.size());
            const int rotations = (poolBlocks - 1 + n - 1) / n;
            rotationCounts.push_back(rotations);
            std::vector<int> candidates(poolBlocks - 1);
            std::iota(candidates.begin(), candidates.end(), 1);
            std::shuffle(candidates.begin(), candidates.end(), rng);
            for (int rotation = 0; rotation < rotations; ++rotation) {
                auto plan = std::make_unique<Plan>();
                plan->n   = n;

                for (int b = 0; b < n; ++b) {
                    const int block = candidates[(rotation * n + b) % candidates.size()];
                    plan->blocks.push_back(block);
                    BenchmarkCopyItem item{host->bytes() + size_t(b) * hostStride, p, hostStride, {}};
                    size_t            prefix = 0;
                    for (const auto& tile : layout.geometry) {
                        auto* device = source->bytes() + tile.offset_per_pool_block * size_t(poolBlocks)
                                       + size_t(block) * tile.bytes;
                        item.tiles.push_back({device, prefix, tile.bytes});
                        prefix += tile.bytes;
                    }
                    plan->items.push_back(std::move(item));
                }
                plan->nativePlans = std::make_unique<FrameworkCopyPlan>(plan->items, layout);
                plans.push_back(std::move(plan));
            }
        }
    }
    int rotationCount(int n) const {
        return rotationCounts.at(size_t(n - 1));
    }
    size_t planIndex(int n, int rotation) const {
        return planStarts.at(size_t(n - 1)) + rotation % rotationCount(n);
    }
    void flush(unsigned int salt) {
        evictCache<<<4096, 256, 0, stream.s>>>(eviction->bytes(), evictBytes, salt + 1);
        cu(cudaGetLastError());
        cu(cudaStreamSynchronize(stream.s));
    }
};

struct CheckLayer {
    size_t offset, width;
};
__global__ void checkSyncPool(const unsigned char* actual,
                              const unsigned char* oracle,
                              const CheckLayer*    layers,
                              size_t               poolBlocks,
                              const unsigned char* selected,
                              bool                 acceptSelected,
                              unsigned int*        errors) {
    const auto   layer = layers[blockIdx.x];
    const size_t span  = layer.width * poolBlocks;
    for (size_t i = size_t(blockIdx.y) * blockDim.x + threadIdx.x; i < span; i += size_t(gridDim.y) * blockDim.x) {
        const size_t        absolute = layer.offset + i;
        const unsigned char expected = acceptSelected && selected[i / layer.width] ? oracle[absolute] : deviceGuard;
        if (actual[absolute] != expected)
            atomicExch(errors, 1U);
    }
}

struct Benchmark {
    Workspace                          w;
    FrameworkCopies                    framework;
    std::vector<unsigned char>         cpuOracle;
    std::vector<CheckLayer>            layers;
    std::unique_ptr<Framework3DCopies> threeD;
    Buffer                             deviceOracle, deviceLayers, selectedBlocks, checkErrors;
    volatile uintptr_t                 metadataSink = 0;

    Benchmark(Layout layout, size_t l2):
        w(layout, l2),
        framework(0),
        cpuOracle(w.sourceBytes),
        deviceOracle(w.sourceBytes),
        deviceLayers(layout.sizes.size() * sizeof(CheckLayer)),
        selectedBlocks(w.poolBlocks),
        checkErrors(sizeof(unsigned int)) {
        requireSync(size_t(w.poolBlocks - 1) * w.p >= 8 * l2, "active device pool smaller than 8 L2");
        cudaPointerAttributes attr{};
        cu(cudaPointerGetAttributes(&attr, w.host->p));
        requireSync(attr.type == cudaMemoryTypeHost, "final host allocation is not pinned");
        cu(cudaPointerGetAttributes(&attr, w.source->p));
        requireSync(attr.type == cudaMemoryTypeDevice, "copy pool is not device memory");
        for (size_t i = 0; i < cpuOracle.size(); ++i)
            cpuOracle[i] = pattern(i);
        for (const auto& tile : layout.geometry)
            layers.push_back({tile.offset_per_pool_block * size_t(w.poolBlocks), tile.bytes});
        cu(cudaMemcpyAsync(deviceOracle.p, cpuOracle.data(), cpuOracle.size(), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaMemcpyAsync(
            deviceLayers.p, layers.data(), layers.size() * sizeof(CheckLayer), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void initialize3D() {
        threeD = std::make_unique<Framework3DCopies>(w.layout, w.poolBlocks, 0);
        for (size_t i = 0; i < w.plans.size(); ++i)
            requireSync(threeD->addPlan(w.plans[i]->items, w.plans[i]->blocks) == i, "3D plan index mismatch");
    }

    void packCpuBlock(int block, unsigned char* output) const {
        size_t offset = 0;
        for (const auto& layer : layers) {
            std::memcpy(output + offset, cpuOracle.data() + layer.offset + size_t(block) * layer.width, layer.width);
            offset += layer.width;
        }
    }

    void restoreDeviceOracle() {
        if (threeD)
            threeD->copyOracle(deviceOracle.p, true);
        cu(cudaMemcpyAsync(w.source->p, deviceOracle.p, w.sourceBytes, cudaMemcpyDeviceToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void prepareHost(size_t ix) {
        const auto& plan = *w.plans[ix];
        std::memset(w.host->p, 0xa5, 32 * w.hostStride);
        for (int b = 0; b < plan.n; ++b) {
            auto* host = w.host->bytes() + size_t(b) * w.hostStride;
            packCpuBlock(plan.blocks[b], host);
        }
    }

    void primeCpuMetadata(size_t ix) {
        uintptr_t sum = 0;
        for (const auto& item : w.plans[ix]->items) {
            sum += reinterpret_cast<uintptr_t>(item.host) ^ item.payload_bytes ^ item.capacity_bytes;
            for (const auto& tile : item.tiles)
                sum += reinterpret_cast<uintptr_t>(tile.device) ^ tile.offset ^ tile.bytes;
        }
        // All variants receive the same CPU-only priming; never dereference
        // a host payload or a GPU pointer while warming descriptor metadata.
        sum += w.plans[ix]->nativePlans->touchMetadata();
        if (threeD)
            sum += threeD->touchMetadata(ix);
        metadataSink = sum;
    }

    void invoke(size_t ix, int variant, bool store) {
        if (variant == 0) {
            framework.copyBatch(*w.plans[ix]->nativePlans, store);
        } else if (variant == 1) {
            framework.copyStaged(*w.plans[ix]->nativePlans, store);
        } else if (variant == 2 && threeD) {
            threeD->copy(ix, store);
        } else {
            throw std::runtime_error("unknown copy variant");
        }
    }

    void poisonDevice(size_t ix) {
        if (threeD)
            threeD->poison(deviceGuard);
        std::vector<unsigned char> selected(w.poolBlocks, 0);
        for (int block : w.plans[ix]->blocks)
            selected[block] = 1;
        cu(cudaMemsetAsync(w.source->p, deviceGuard, w.sourceBytes, w.stream.s));
        cu(cudaMemcpyAsync(selectedBlocks.p, selected.data(), selected.size(), cudaMemcpyHostToDevice, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
    }

    void verifyDevice() {
        if (threeD)
            threeD->copyOracle(w.source->p, false);
        cu(cudaMemsetAsync(checkErrors.p, 0, sizeof(unsigned int), w.stream.s));
        checkSyncPool<<<dim3(unsigned(layers.size()), 32), 256, 0, w.stream.s>>>(
            w.source->bytes(),
            deviceOracle.bytes(),
            static_cast<const CheckLayer*>(deviceLayers.p),
            w.poolBlocks,
            selectedBlocks.bytes(),
            true,
            static_cast<unsigned int*>(checkErrors.p));
        cu(cudaGetLastError());
        unsigned int errors = 0;
        cu(cudaMemcpyAsync(&errors, checkErrors.p, sizeof(errors), cudaMemcpyDeviceToHost, w.stream.s));
        cu(cudaStreamSynchronize(w.stream.s));
        requireSync(errors == 0, "H2D payload or unselected guard mismatch");
    }

    void verifyHost(size_t ix) {
        const auto&                plan = *w.plans[ix];
        std::vector<unsigned char> packed(w.p);
        for (int b = 0; b < 32; ++b) {
            const auto* host = w.host->bytes() + size_t(b) * w.hostStride;
            if (b < plan.n) {
                packCpuBlock(plan.blocks[b], packed.data());
                requireSync(std::memcmp(host, packed.data(), w.p) == 0, "CPU payload oracle mismatch");
            }
            const size_t copied = b < plan.n ? w.p : 0;
            for (size_t i = copied; i < w.hostStride; ++i)
                requireSync(host[i] == 0xa5, "host non-payload guard overwritten");
        }
    }

    void verify(size_t ix, int variant, bool store) {
        try {
            if (store) {
                std::memset(w.host->p, 0xa5, 32 * w.hostStride);
                invoke(ix, variant, true);
                verifyHost(ix);
            } else {
                prepareHost(ix);
                poisonDevice(ix);
                invoke(ix, variant, false);
                verifyDevice();
                verifyHost(ix);
            }

        } catch (const std::exception& error) {
            std::fprintf(stderr,
                         "correctness failed layout=%s bs=%d variant=%s direction=%s rotation=%zu: %s\n",
                         w.layout.name.c_str(),
                         w.plans[ix]->n,
                         syncNames[variant],
                         store ? "d2h" : "h2d",
                         ix - w.planStarts[w.plans[ix]->n - 1],
                         error.what());
            throw;
        }
    }
};

struct Options {
    int         iterations      = 100;
    int         warmup          = 30;
    int         repeat          = 80;
    uint32_t    seed            = 20261004;
    bool        correctnessOnly = false;
    bool        only3d          = false;
    bool        exclude1dH2d    = false;
    bool        seedProvided    = false;
    std::string output;
};
uint32_t parseUnsigned(const std::string& text, const char* name) {
    requireSync(!text.empty() && text.front() != '-', std::string("invalid ") + name);
    size_t     end   = 0;
    const auto value = std::stoull(text, &end);
    requireSync(end == text.size() && value <= std::numeric_limits<uint32_t>::max(), std::string("invalid ") + name);
    return static_cast<uint32_t>(value);
}
Options parseOptions(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--3d-only") {
            result.only3d = true;
            continue;
        }
        if (key == "--correctness-only") {
            result.correctnessOnly = true;
            continue;
        }
        if (key == "--exclude-1d-h2d") {
            result.exclude1dH2d = true;
            continue;
        }
        if (key == "--help") {
            std::cout
                << "CrcCopyBenchmark --output FILE [--iterations 100] [--warmup 30] [--repeat 80] [--seed UINT]\n"
                   "                 [--correctness-only] [--exclude-1d-h2d] [--3d-only]\n"
                   "Two production paths, both directions, FULL/SWA, local backing BS=1..32. CUDA13 required.\n"
                   "--exclude-1d-h2d explicitly omits that path; it never substitutes another copy implementation.\n";
            std::exit(0);
        }
        std::string value;
        const auto  equal = key.find('=');
        if (equal != std::string::npos) {
            value = key.substr(equal + 1);
            key.resize(equal);
        } else {
            requireSync(i + 1 < argc, "missing option value for " + key);
            value = argv[++i];
        }
        if (key == "--output") {
            result.output = value;
            continue;
        }
        const auto number = parseUnsigned(value, key.c_str());
        if (key == "--seed") {
            result.seed         = number;
            result.seedProvided = true;
            continue;
        }
        requireSync(number <= static_cast<uint32_t>(std::numeric_limits<int>::max()), "option value exceeds INT_MAX");
        if (key == "--iterations")
            result.iterations = static_cast<int>(number);
        else if (key == "--warmup")
            result.warmup = static_cast<int>(number);
        else if (key == "--repeat")
            result.repeat = static_cast<int>(number);
        else
            throw std::runtime_error("unknown option: " + key);
    }
    requireSync(result.iterations > 0 && result.warmup <= std::numeric_limits<int>::max() - result.iterations,
                "iterations must be positive and iterations+warmup must fit INT_MAX");
    if (!result.seedProvided)
        result.seed = 20260924U + static_cast<uint32_t>(result.repeat);
    requireSync(!result.output.empty(), "--output FILE is required");
    requireSync(!result.only3d || !result.exclude1dH2d, "3D-only mode cannot exclude 1D cases");
    return result;
}

void metadata(std::ostream& os, const Options& options, const cudaDeviceProp& prop, int driver, int runtime) {
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    const auto provenance = [](const char* name) {
        const auto* value = std::getenv(name);
        return value ? value : "unavailable";
    };
    os << "{\"type\":\"metadata\",\"implementation\":\"btc_copy_benchmark_v63\",\"seed\":" << options.seed
       << ",\"source_commit\":\"" << provenance("CRC_BENCH_SOURCE_COMMIT") << "\",\"base_commit\":\""
       << provenance("CRC_BENCH_BASE_COMMIT") << "\",\"binary_sha256\":\"" << provenance("CRC_BENCH_BINARY_SHA256")
       << "\""
       << ",\"shape_source\":\"CacheConfigCreator+DeviceBlockPoolConfigHelper\""
       << ",\"staged_no_crc_source\":\"release/btc_1.0 production StagedSmDeviceHostCopyStrategy\""
       << ",\"variants\":" << (options.only3d ? "[\"copy3d_batch\"]" : "[\"copy1d_batch\",\"staged_no_crc\"]")
       << ",\"exclude_1d_h2d\":" << (options.exclude1dH2d ? "true" : "false") << ",\"excluded_cases\":"
       << (options.exclude1dH2d ?
               "[{\"direction\":\"h2d\",\"variant\":\"copy1d_batch\",\"reason\":\"explicit --exclude-1d-h2d\"}]" :
               "[]")
       << ",\"repeat\":" << options.repeat << ",\"evict_multiplier\":" << kEvictMultiplier
       << ",\"cpu_metadata\":\"warm\",\"host_gap\":0,\"descriptor_bytes\":24,\"layout_order\":\"production\""
       << ",\"l2_bytes\":" << prop.l2CacheSize << ",\"timing\":\"wall\",\"regime\":\"cold\""
       << ",\"gpu\":\"" << prop.name << "\",\"sm\":" << prop.major * 10 + prop.minor << ",\"gpu_uuid\":\"" << uuid.str()
       << "\",\"driver\":" << driver << ",\"runtime\":" << runtime << ",\"iterations\":" << options.iterations
       << ",\"warmup\":" << options.warmup << ",\"local_backing_count\":true,\"gen_num_per_cycle\":0,\"profiled\":false"
       << ",\"correctness_only\":" << (options.correctnessOnly ? "true" : "false")
       << ",\"boundary\":\"prebuilt per-API inputs; complete synchronous calls including internal descriptors, checks, locks, metadata H2D, kernels, CPU pack/unpack for staged_no_crc, data transfers, synchronization\""
       << ",\"production_source\":\"directly linked current workspace DeviceHostCopyStrategy\""
       << ",\"host_input\":\"independent CPU whole-record oracle prepared outside timing\""
       << ",\"block_counts\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]"
       << ",\"seq_size_per_block\":128,\"cp_size\":1,\"tp_size\":1,\"cp_mode\":\"NONE\""
       << ",\"h2d_host_payload\":\"recently prepared on CPU once per shared round; host cache is not claimed cold\",\"fallback_allowed\":false}\n";
}

int run(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);
    std::ofstream os(options.output);
    requireSync(bool(os), "cannot open output: " + options.output);
    os << std::setprecision(12);
    cu(cudaSetDevice(0));
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop, 0));
    requireSync(prop.l2CacheSize > 0, "known GPU L2 size is required");
    int driver = 0, runtime = 0;
    cu(cudaDriverGetVersion(&driver));
    cu(cudaRuntimeGetVersion(&runtime));
    requireSync(driver >= 13000 && runtime >= 13000, "CUDA13 runtime and driver are required");
    metadata(os, options, prop, driver, runtime);
    os.flush();
    std::mt19937 source_rng(options.seed), order_rng(options.seed ^ 0x728193U);
    for (bool is_full : {true, false}) {
        Benchmark benchmark(orderedLayout(is_full), prop.l2CacheSize);
        if (options.only3d)
            benchmark.initialize3D();
        auto& w = benchmark.w;
        os << "{\"type\":\"layout\",\"layout\":\"" << w.layout.name << "\",\"payload_bytes\":" << w.p
           << ",\"encoded_bytes\":" << w.e << ",\"host_stride\":" << w.hostStride << ",\"staging_stride\":" << w.stride
           << ",\"tiles\":" << w.layout.sizes.size() << ",\"tile_bytes\":[";
        for (size_t t = 0; t < w.layout.sizes.size(); ++t) {
            if (t)
                os << ',';
            os << w.layout.sizes[t];
        }
        os << "],\"geometry\":" << w.layout.geometryJson()
           << ",\"reserved_sentinel_blocks\":1,\"source_bytes\":" << w.sourceBytes
           << ",\"destination_bytes\":" << w.sourceBytes << ",\"pool_blocks\":" << w.poolBlocks
           << ",\"evict_bytes\":" << w.evictBytes << ",\"baseline_payload_bytes_per_backing\":" << w.p
           << ",\"host_pinned_verified\":true,\"source_device_verified\":true"
           << ",\"guard_check\":\"entire device pool against CPU oracle plus unselected sentinel\"}\n";
        benchmark.restoreDeviceOracle();
        for (bool store : {true, false}) {
            for (int n = 1; n <= kMaxBackings; ++n) {
                for (int variant = options.only3d ? 2 : 0; variant < (options.only3d ? 3 : 2); ++variant) {
                    if (!store && variant == 0 && options.exclude1dH2d) {
                        os << "{\"type\":\"excluded_correctness\",\"direction\":\"h2d\",\"layout\":\"" << w.layout.name
                           << "\",\"blocks\":" << n
                           << ",\"variant\":\"copy1d_batch\",\"reason\":\"explicit --exclude-1d-h2d\"}\n";
                        continue;
                    }
                    for (int rotation : {0, 1})
                        benchmark.verify(w.planIndex(n, rotation), variant, store);
                    os << "{\"type\":\"correctness\",\"direction\":\"" << (store ? "d2h" : "h2d") << "\",\"layout\":\""
                       << w.layout.name << "\",\"blocks\":" << n << ",\"variant\":\"" << syncNames[variant]
                       << "\",\"rotations_checked\":[0,1],\"success\":true}\n";
                }
            }
        }
        benchmark.restoreDeviceOracle();
        os.flush();
        if (options.correctnessOnly)
            continue;
        for (bool store : {true, false}) {
            for (int n = 1; n <= kMaxBackings; ++n) {
                std::vector<int> sources(w.rotationCount(n));
                std::iota(sources.begin(), sources.end(), 0);
                std::shuffle(sources.begin(), sources.end(), source_rng);
                std::vector<int> order = options.only3d ? std::vector<int>{2} : std::vector<int>{0, 1};
                if (!store && options.exclude1dH2d)
                    order.erase(order.begin());
                for (int round = -options.warmup; round < options.iterations; ++round) {
                    const int    source = sources[(round + options.warmup) % sources.size()];
                    const size_t ix     = w.planIndex(n, source);
                    if (!store)
                        benchmark.prepareHost(ix);
                    std::shuffle(order.begin(), order.end(), order_rng);
                    for (size_t position = 0; position < order.size(); ++position) {
                        const int variant = order[position];
                        w.flush(static_cast<unsigned int>(round + options.warmup) * 5U
                                + static_cast<unsigned int>(position));
                        benchmark.primeCpuMetadata(ix);
                        const auto begin = std::chrono::steady_clock::now();
                        benchmark.invoke(ix, variant, store);
                        const double us =
                            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
                        if (round >= 0)
                            os << "{\"type\":\"sample\",\"direction\":\"" << (store ? "d2h" : "h2d")
                               << "\",\"layout\":\"" << w.layout.name << "\",\"blocks\":" << n << ",\"variant\":\""
                               << syncNames[variant] << "\",\"regime\":\"cold\",\"timing\":\"wall\""
                               << ",\"repeat\":" << options.repeat << ",\"round\":" << round
                               << ",\"position\":" << position << ",\"source_plan\":" << source << ",\"us\":" << us
                               << "}\n";
                    }
                }
                os.flush();
            }
        }
    }
    os << "{\"type\":\"complete\",\"success\":true}\n";
    os.flush();
    requireSync(bool(os), "failed to write benchmark output");
    return 0;
}
}  // namespace
}  // namespace rtp_llm::crc_copy_benchmark

int main(int argc, char** argv) {
    try {
        return rtp_llm::crc_copy_benchmark::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "BTC copy benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
