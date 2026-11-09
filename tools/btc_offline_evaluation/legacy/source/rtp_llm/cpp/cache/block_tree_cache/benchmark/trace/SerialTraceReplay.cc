#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <thread>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/FullCacheManager.h"
#include "rtp_llm/cpp/cache/SWACacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/OnlineTreeScheduler.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/trace/TraceReader.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
using namespace rtp_llm;
using namespace rtp_llm::benchmark;
namespace {
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct Options {
    std::string input, output, disk_dir;
    size_t      full_device{128}, swa_device{128}, full_host{256}, swa_host{256}, full_disk{512}, swa_disk{512};
    size_t      last_n{4}, linear_step{4}, max_requests{1000}, payload_divisor{256}, timeout_ms{30000};
    int         gpu{0};
};
Options parse(int argc, char** argv) {
    Options                        o;
    std::map<std::string, size_t*> sizes = {{"--full-device", &o.full_device},
                                            {"--swa-device", &o.swa_device},
                                            {"--full-host", &o.full_host},
                                            {"--swa-host", &o.swa_host},
                                            {"--full-disk", &o.full_disk},
                                            {"--swa-disk", &o.swa_disk},
                                            {"--last-n", &o.last_n},
                                            {"--linear-step", &o.linear_step},
                                            {"--max-requests", &o.max_requests},
                                            {"--payload-divisor", &o.payload_divisor},
                                            {"--timeout-ms", &o.timeout_ms}};
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc)
            throw std::runtime_error("every flag needs a value");
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--input")
            o.input = v;
        else if (k == "--output")
            o.output = v;
        else if (k == "--disk-dir")
            o.disk_dir = v;
        else if (k == "--gpu")
            o.gpu = std::stoi(v);
        else if (sizes.count(k)) {
            if (v.empty() || v[0] == '-')
                throw std::runtime_error("negative option");
            *sizes[k] = std::stoull(v);
        } else
            throw std::runtime_error("unknown option: " + k);
    }
    if (o.input.empty() || o.output.empty() || o.disk_dir.empty() || !o.full_device || !o.swa_device
        || !o.payload_divisor || !o.timeout_ms || !o.last_n || o.last_n > 100000 || o.linear_step > 100000)
        throw std::runtime_error("invalid required options");
    if ((o.full_host == 0) != (o.swa_host == 0) || (o.full_disk == 0) != (o.swa_disk == 0))
        throw std::runtime_error("FULL/SWA tier enablement must agree");
    return o;
}
const std::array<std::string, 5> tags           = {"csa_kv", "hca_kv", "indexer_kv", "swa_kv", "csa_state"};
const std::array<size_t, 5>      original_bytes = {110592, 73728, 69632, 443520, 24576};
std::vector<uint8_t>             pattern(int64_t key, size_t member, size_t bytes) {
    std::vector<uint8_t> data(bytes);
    uint64_t             x = static_cast<uint64_t>(key) ^ ((member + 1) * 0x9e3779b97f4a7c15ULL);
    for (size_t i = 0; i < bytes; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        data[i] = static_cast<uint8_t>(x);
    }
    return data;
}
class AuditedTransferEngine: public PerRankBlockTransferEngine {
public:
    using PerRankBlockTransferEngine::PerRankBlockTransferEngine;
    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) override {
        auto                        context = PerRankBlockTransferEngine::execute(std::move(task));
        std::lock_guard<std::mutex> lock(mutex_);
        contexts_.push_back(context);
        return context;
    }
    void verifyCompleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& context : contexts_) {
            if (!context->done() || !context->success())
                throw std::runtime_error("background BTC transfer incomplete or failed");
        }
        contexts_.clear();
    }

private:
    std::mutex                                         mutex_;
    std::vector<std::shared_ptr<TransferAsyncContext>> contexts_;
};
struct Resources {
    std::shared_ptr<AuditedTransferEngine> engine;

    std::vector<DeviceBlockPoolPtr>                      pools;
    std::vector<std::unique_ptr<SingleTypeCacheManager>> managers;
    std::unique_ptr<BlockTreeCache>                      cache;
    std::array<size_t, 5>                                bytes;
    const Options&                                       options;
    explicit Resources(const Options& o): options(o) {
        std::vector<std::pair<std::string, CacheGroupType>> specs;
        std::vector<size_t>                                 strides, windows;
        for (size_t m = 0; m < 5; ++m) {
            bytes[m] = ((std::max<size_t>(64, original_bytes[m] / o.payload_divisor) + 63) / 64) * 64;
            specs.push_back({tags[m], m < 3 ? CacheGroupType::FULL : CacheGroupType::SWA});
            strides.push_back(bytes[m]);
            windows.push_back(m < 3 ? 0 : 128);
        }
        auto initial = BenchmarkFixture::createTopology(specs, strides, 1024, {}, windows);
        auto groups  = initial->groups();
        for (auto& group : groups) {
            group.policy.enable_prefix_reuse = true;
            if (group.policy.group_type == CacheGroupType::SWA) {
                group.policy.active_tail_blocks   = o.last_n;
                group.policy.validate_tail_blocks = true;
            }
        }
        auto topology = CacheTopology::create(groups, initial->layers());
        for (size_t m = 0; m < 5; ++m)
            pools.push_back(
                BenchmarkFixture::createDevicePool(bytes[m], 1, m < 3 ? o.full_device : o.swa_device, tags[m], 1024));
        std::vector<GroupSetPtr> sets;
        for (size_t gs = 0; gs < 2; ++gs) {
            size_t                          first = gs ? 3 : 0, last = gs ? 5 : 3, payload = 0;
            std::vector<DeviceBlockPoolPtr> members;
            std::vector<std::string>        names;
            for (size_t m = first; m < last; ++m) {
                members.push_back(pools[m]);
                names.push_back(tags[m]);
                payload += bytes[m];
            }
            auto host_count = gs ? o.swa_host : o.full_host, disk_count = gs ? o.swa_disk : o.full_disk;
            auto host =
                host_count ?
                    BenchmarkFixture::createHostPool(payload, host_count, gs ? "trace_host_swa" : "trace_host_full") :
                    nullptr;
            auto disk = disk_count ?
                            BenchmarkFixture::createDiskPool(
                                payload, disk_count, o.disk_dir, gs ? "trace_disk_swa" : "trace_disk_full", true) :
                            nullptr;
            sets.push_back(
                gs ? BenchmarkFixture::createSWAGroupSet(members, host, disk, gs, topology, names, 128, 1024) :
                     BenchmarkFixture::createFullGroupSet(members, host, disk, gs, topology, names));
        }
        BlockTreeCacheConfig cfg;
        cfg.enable_host_cache = o.full_host > 0;
        cfg.enable_disk_cache = o.full_disk > 0;
        cfg.watermark_device  = {.82, .90};
        cfg.watermark_host    = {.90, .94};
        cfg.watermark_disk    = {.92, .97};
        auto reporter         = std::make_shared<BlockTreeCacheMetricsReporter>(nullptr);
        engine                = std::make_shared<AuditedTransferEngine>(sets,
                                                         cfg.enable_disk_cache,
                                                         DeviceHostCopyOptions{},
                                                         cfg.device_disk_staging_block_count,
                                                         cfg.max_descriptors_per_transfer_batch,
                                                         cfg.transfer_worker_count,
                                                         cfg.transfer_queue_max_size,
                                                         reporter);
        auto dispatcher =
            std::make_unique<BlockTransferDispatcher>(engine, nullptr, cfg.max_descriptors_per_transfer_batch);
        auto tasks = std::make_unique<BlockTreeTaskPool>(4, 1000, "BTCSerialTrace");
        cache      = std::make_unique<BlockTreeCache>(
            std::make_unique<BlockTree>(sets), cfg, nullptr, std::move(dispatcher), std::move(tasks), reporter);
        if (!cache->init())
            throw std::runtime_error("cache init failed");
        for (size_t m = 0; m < 5; ++m) {
            if (m < 3)
                managers.push_back(
                    std::make_unique<FullCacheManager>(topology->group(tags[m]), pools[m], static_cast<int>(m)));
            else
                managers.push_back(
                    std::make_unique<SWACacheManager>(topology->group(tags[m]),
                                                      pools[m],
                                                      static_cast<int>(m),
                                                      o.linear_step ? static_cast<int>(o.linear_step) : 1));
            managers.back()->setEvictCallback([this, m](size_t n) { return cache->evictForGroup(tags[m], n); });
        }
    }
    bool drain() {
        return boundedPendingTaskDrain(
            [this] {
                auto&                       dispatcher = *cache->transfer_dispatcher_;
                std::lock_guard<std::mutex> lock(dispatcher.completion_mutex_);
                size_t                      pending = 0;
                for (const auto& completion : dispatcher.transfer_completions_)
                    pending += completion.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
                return pending + static_cast<size_t>(cache->task_pool_->pending_tasks_.load());
            },
            std::chrono::milliseconds(options.timeout_ms));
    }
    void requireDrained() {
        if (!drain())
            throw std::runtime_error("BTC task drain timed out");
    }
    void checkReleased() {
        engine->verifyCompleted();
        for (const auto& s : cache->poolMetricsSnapshots())
            if (s.request_ref_blocks || s.load_ref_blocks || s.store_ref_blocks || s.eviction_target_ref_blocks
                || s.frozen_ref_blocks)
                throw std::runtime_error("request or async resource references remain after drain");
    }
};
struct RequestResources {
    Resources&              owner;
    std::array<BlockIds, 5> rows;
    bool                    released{false};
    explicit RequestResources(Resources& r): owner(r) {}
    void release() {
        if (released)
            return;
        for (size_t m = 0; m < 5; ++m) {
            BlockIndicesType valid;
            for (auto b : rows[m].blocks())
                if (!isNullBlockIdx(b))
                    valid.push_back(b);
            if (!valid.empty())
                owner.pools[m]->decRef(valid);
            rows[m].assign(BlockIndicesType{});
        }
        released = true;
    }
    ~RequestResources() {
        try {
            release();
        } catch (...) {}
    }
    void place(size_t gs, size_t pos, const BlockIndicesType& ids) {
        const size_t offset  = gs ? 3 : 0;
        const size_t members = gs ? 2 : 3;
        if (ids.size() != members)
            throw std::runtime_error("group member count mismatch");
        for (size_t j = 0; j < members; ++j) {
            auto& row = rows[offset + j];
            if (row.blocksNum() <= pos)
                row.resize(pos + 1);
            if (!isNullBlockIdx(row.blocks()[pos]))
                throw std::runtime_error("duplicate request ownership");
            row.setAt(pos, ids[j]);
        }
    }
};
struct Result {
    size_t logical{0}, effective{0}, device{0}, host{0}, disk{0};
};
Result process(Resources& state, const SerialTraceRequest& request) {
    if (request.input_tokens > static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("request too long");
    Result           answer;
    RequestResources held(state);
    const auto       valid = [&](size_t n) {
        return n * 1024 <= request.input_tokens && request.input_tokens - n * 1024 >= 128;
    };
    auto matched   = state.cache->match(request.keys, valid);
    auto ticket    = matched.async_context;
    answer.device  = matched.matched_device_blocks;
    answer.logical = ticket ? ticket->matchedBlocks() : answer.device;
    for (const auto& resource : matched.matched_device_resources) {
        size_t start = answer.device - resource.node_blocks.size();
        for (size_t i = 0; i < resource.node_blocks.size(); ++i)
            held.place(resource.group_set_id, start + i, resource.node_blocks[i].second);
    }
    try {
        if (ticket) {
            answer.host        = ticket->matchedBlocks(Tier::HOST);
            answer.disk        = ticket->matchedBlocks(Tier::DISK);
            const auto& descs  = ticket->loadDescs();
            const auto& joined = ticket->joinedLoads();
            for (size_t d = 0; d < descs.size(); ++d) {
                const auto& desc = descs[d];
                if (joined[d]) {
                    held.place(desc.group_set_id, desc.path_index, desc.target_blocks);
                    continue;
                }
                if (desc.source_tier == Tier::DEVICE) {
                    held.place(desc.group_set_id, desc.path_index, desc.source_blocks);
                    continue;
                }
                const size_t     start = desc.group_set_id ? 3 : 0, end = desc.group_set_id ? 5 : 3;
                BlockIndicesType targets;
                for (size_t m = start; m < end; ++m) {
                    if (state.pools[m]->freeBlocksNum() < 1)
                        state.cache->evictForGroup(tags[m], 1);
                    auto blocks = state.pools[m]->malloc(1);
                    if (!blocks)
                        throw std::runtime_error("load target allocation failed");
                    state.pools[m]->incRef(*blocks);
                    auto& row = held.rows[m];
                    if (row.blocksNum() <= desc.path_index)
                        row.resize(desc.path_index + 1);
                    row.setAt(desc.path_index, (*blocks)[0]);
                    targets.push_back((*blocks)[0]);
                }
                ticket->setTargetBlocks(d, targets);
            }
            if (!ticket->commit())
                throw std::runtime_error("load commit failed");
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(state.options.timeout_ms);
            while (!ticket->done() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!ticket->done() || !ticket->success())
                throw std::runtime_error("load failed or timed out");
        }
        answer.effective = answer.logical;
        for (size_t m = 0; m < 5; ++m) {
            for (size_t pos = 0; pos < held.rows[m].blocksNum(); ++pos) {
                auto id = held.rows[m].blocks()[pos];
                if (isNullBlockIdx(id))
                    continue;
                auto                 expected = pattern(request.keys.at(pos), m, state.bytes[m]);
                std::vector<uint8_t> actual(expected.size());
                auto*                addr =
                    static_cast<uint8_t*>(state.pools[m]->getBaseAddress()) + static_cast<size_t>(id) * state.bytes[m];
                cudaCheck(cudaMemcpy(actual.data(), addr, actual.size(), cudaMemcpyDeviceToHost));
                if (actual != expected)
                    throw std::runtime_error("placeholder content mismatch after match/load");
            }
        }
        for (size_t m = 0; m < 5; ++m) {
            if (held.rows[m].blocksNum() > answer.effective)
                throw std::runtime_error("match returned resources beyond its effective prefix");
            held.rows[m].resize(answer.effective);
            auto before = held.rows[m].blocks();
            if (!state.managers[m]->malloc(
                    held.rows[m], static_cast<int>(request.input_tokens), m < 3 || state.options.linear_step > 0))
                throw std::runtime_error("native suffix allocation failed");
            for (size_t pos = 0; pos < held.rows[m].blocksNum(); ++pos) {
                auto id = held.rows[m].blocks()[pos];
                if (isNullBlockIdx(id) || (pos < before.size() && !isNullBlockIdx(before[pos])))
                    continue;
                auto  key  = pos < request.keys.size() ? request.keys[pos] : static_cast<int64_t>(request.timestamp_ns);
                auto  data = pattern(key, m, state.bytes[m]);
                auto* addr =
                    static_cast<uint8_t*>(state.pools[m]->getBaseAddress()) + static_cast<size_t>(id) * state.bytes[m];
                cudaCheck(cudaMemcpy(addr, data.data(), data.size(), cudaMemcpyHostToDevice));
            }
            state.managers[m]->removeSkippedBlocks(held.rows[m], m < 3 || state.options.linear_step > 0);
        }
        std::vector<std::vector<GroupSetResource>> resources(request.keys.size(), std::vector<GroupSetResource>(2));
        for (size_t pos = 0; pos < request.keys.size(); ++pos)
            for (size_t gs = 0; gs < 2; ++gs) {
                size_t           first = gs ? 3 : 0, end = gs ? 5 : 3;
                BlockIndicesType ids;
                for (size_t m = first; m < end; ++m) {
                    auto id = held.rows[m].blocks()[pos];
                    if (!isNullBlockIdx(id))
                        ids.push_back(id);
                }
                if (!ids.empty() && ids.size() != end - first)
                    throw std::runtime_error("partial group assembly");
                resources[pos][gs].device_blocks = std::move(ids);
            }
        state.cache->insert(request.keys, resources, Tier::DEVICE);
        held.release();
        state.requireDrained();
        state.checkReleased();
        return answer;
    } catch (...) {
        if (ticket && !ticket->done()) {
            ticket->abortPending();
            state.cache->shutdownLoads();
        }
        state.requireDrained();
        held.release();
        throw;
    }
}
}  // namespace
int main(int argc, char** argv) {
    std::filesystem::path output;
    bool                  output_created = false;
    try {
        if (!initLogger())
            throw std::runtime_error("logger initialization failed");
        const auto o = parse(argc, argv);
        output       = o.output;
        if (std::filesystem::exists(output))
            throw std::runtime_error("output directory already exists");
        std::filesystem::create_directories(output);
        output_created = true;
        if (std::filesystem::exists(o.disk_dir) && !std::filesystem::is_empty(o.disk_dir))
            throw std::runtime_error("disk directory must be empty");
        std::filesystem::create_directories(o.disk_dir);
        cudaCheck(cudaSetDevice(o.gpu));
        CPSlotMapper mapper(0, 4, 256);
        if (mapper.virtualBlockSize() != 1024)
            throw std::runtime_error("CP contract");
        uint64_t      chunks, expected, expected_tokens;
        std::ifstream done(std::filesystem::path(o.input) / "producer.done");
        done >> chunks >> expected >> expected_tokens;
        if (!done || !chunks || !expected)
            throw std::runtime_error("input completion marker missing");
        Resources     state(o);
        uint64_t      count = 0, tokens = 0, reused = 0, last = 0;
        std::ofstream per(output / "requests.jsonl");
        for (uint64_t c = 0; c < chunks && (o.max_requests == 0 || count < o.max_requests); ++c) {
            std::ostringstream filename;
            filename << std::setw(6) << std::setfill('0') << c << ".bin";
            std::ifstream      input(std::filesystem::path(o.input) / "chunks" / filename.str(), std::ios::binary);
            TraceChunkReader   reader(input, 1024);
            SerialTraceRequest request;
            while ((o.max_requests == 0 || count < o.max_requests) && reader.next(request)) {
                if (request.timestamp_ns < last)
                    throw std::runtime_error("trace order violation");
                last         = request.timestamp_ns;
                const auto r = process(state, request);
                ++count;
                tokens += request.input_tokens;
                reused += r.effective * 1024;
                per << "{\"ordinal\":" << count << ",\"timestamp_ns\":" << request.timestamp_ns
                    << ",\"input_tokens\":" << request.input_tokens << ",\"logical_match_blocks\":" << r.logical
                    << ",\"effective_hit_blocks\":" << r.effective << ",\"device_prefix_blocks\":" << r.device
                    << ",\"host_match_blocks\":" << r.host << ",\"disk_match_blocks\":" << r.disk << "}\n";
                if (!per)
                    throw std::runtime_error("request output failed");
                if (count % 1000 == 0) {
                    per.flush();
                    std::cerr << "completed_requests=" << count << " input_tokens=" << tokens << std::endl;
                }
            }
        }
        state.requireDrained();
        state.checkReleased();
        const auto expected_count = o.max_requests ? std::min<uint64_t>(expected, o.max_requests) : expected;
        if (count != expected_count || (count == expected && tokens != expected_tokens))
            throw std::runtime_error("full cohort mismatch");
        per.close();
        std::ofstream summary(output / "summary.json");
        summary
            << std::setprecision(17) << "{\"complete\":true,\"requests\":" << count << ",\"input_tokens\":" << tokens
            << ",\"effective_hit_tokens\":" << reused << ",\"hit_rate\":" << (tokens ? double(reused) / tokens : 0)
            << ",\"cp_size\":4,\"base_tokens\":256,\"canonical_tokens\":1024,\"linear_step\":" << o.linear_step
            << ",\"last_n\":" << o.last_n << ",\"payload_divisor\":" << o.payload_divisor
            << ",\"serial\":true,\"physical_rank_processes\":1,\"placeholder_content_verified\":true,\"request_refs_zero\":true}\n";
        summary.close();
        if (!summary)
            throw std::runtime_error("summary write");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        if (output_created) {
            std::ofstream f(output / "FAILED.txt");
            f << e.what() << '\n';
        }
        return 1;
    }
}
