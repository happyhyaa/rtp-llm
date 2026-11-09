#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/MetadataReplay.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/Dsv41Profile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeGroupLayout.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <limits>
#include <stdexcept>
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
namespace rtp_llm::benchmark {
uint64_t physicalCopyAttempts();
namespace {
using Clock = std::chrono::steady_clock;
double micros(Clock::time_point begin) {
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
CacheConfig fixtureConfig(const MetadataReplayConfig& o) {
    std::vector<std::pair<std::string, CacheGroupType>> specs = {{"full", CacheGroupType::FULL}};
    std::vector<size_t>                                 bytes = {1024}, windows = {0};
    if (o.swa) {
        require(o.last_n && o.swa_blocks && o.sliding_window, "invalid SWA config");
        specs.push_back({"swa", CacheGroupType::SWA});
        bytes.push_back(1024);
        windows.push_back(o.sliding_window);
    }
    auto topology = BenchmarkFixture::createTopology(specs, bytes, o.block_tokens, {}, windows);
    auto groups   = topology->groups();
    if (o.swa && o.cp_size > 1) {
        require(o.swa_block_tokens == o.cp_size * o.block_tokens,
                "compact SWA controller fixture requires an explicit virtual span");
        groups.back().spec = BenchmarkFixture::createTopology(
                                 {{"swa", CacheGroupType::SWA}}, {1024}, o.swa_block_tokens, {}, {o.sliding_window})
                                 ->groups()
                                 .front()
                                 .spec;
    } else if (o.swa)
        require(o.swa_block_tokens == 0 || o.swa_block_tokens == o.block_tokens, "unexpected SWA span");
    for (auto& g : groups) {
        g.policy.enable_prefix_reuse = true;  // Explicit reusable FULL/SWA fixture configuration.
        g.block_num = (g.policy.group_type == CacheGroupType::SWA ? o.swa_blocks : o.device_blocks) + 1;
        if (g.policy.group_type == CacheGroupType::SWA)
            g.policy.active_tail_blocks = o.last_n;
    }
    CacheConfig config;
    config.metadata_only           = true;
    config.dtype                   = TYPE_INT8;
    config.layer_num               = groups.size();
    config.seq_size_per_block      = o.block_tokens;
    config.reuse_cache_tail_blocks = o.last_n;
    config.cache_min_replay_tokens = o.min_replay_tokens;
    config.setTopology(std::move(groups), topology->layers());
    return config;
}

CacheConfig dsv41Config(const MetadataReplayConfig& o) {
    require(
        o.swa && o.block_tokens == 256 && o.cp_size == 4 && o.sliding_window == dsv41::kWindow
            && (!o.swa_block_tokens || o.swa_block_tokens == o.block_tokens * o.cp_size) && o.min_replay_tokens == 128
            && o.linear_step == 1 && o.swa_blocks <= o.device_blocks
            && o.last_n > 0,
        "DSv4.1 profile requires bounded replay, CP4, base/kernel 256, native SWA tail/step and equal device capacities");
    ParallelismConfig parallel;
    parallel.role_type                          = RoleType::PREFILL;
    parallel.tp_size                            = o.cp_size;
    parallel.tp_rank                            = o.cp_rank;
    parallel.prefill_cp_config.kv_cache_sharded = true;
    parallel.prefill_cp_config.prefill_cp_size  = o.cp_size;
    parallel.prefill_cp_config.method           = CPRotateMethod::PREFILL_CP;
    SpecBuildContext ctx;
    ctx.seq_size_per_block      = o.block_tokens;
    ctx.kernel_tokens_per_block = o.block_tokens;
    ctx.parallelism_config      = &parallel;
    // The historical byte audit matched all six live pool sizes with the model's five-token DSpark block.
    ctx.gen_num_per_cycle = dsv41::kDraftBlockSize;

    const auto prototypes = dsv41::descriptors();
    auto       build      = [&](const std::vector<std::vector<std::string>>& tags) {
        CacheConfig config;
        config.metadata_only                            = true;
        config.layer_num                                = tags.size();
        config.seq_size_per_block                       = o.block_tokens;
        config.cache_min_replay_tokens                  = o.min_replay_tokens;
        config.cache_key_hash_seed                      = 0x4453563431494231LL;
        config.use_typed_cache_regions                  = true;
        config.use_opaque_kv_cache_store                = true;
        config.disable_decode_first_malloc_device_reuse = true;
        config.is_sparse                                = true;
        config.dtype       = TYPE_BF16;  // Model dtype; opaque pools carry their own entry dtype.
        config.reuse_cache_tail_blocks = o.last_n;
        std::vector<GroupBase> groups;
        std::vector<LayerBase> layers;
        for (const auto& layer_tags : tags) {
            layers.push_back({static_cast<int>(layers.size()), layer_tags});
            for (const auto& tag : layer_tags) {
                if (std::any_of(groups.begin(), groups.end(), [&](const auto& g) { return g.tag == tag; }))
                    continue;
                const auto desc =
                    std::find_if(prototypes.begin(), prototypes.end(), [&](const auto& d) { return d.tag == tag; });
                require(desc != prototypes.end(), "unresolved DSv4.1 descriptor");
                auto policy = SpecBuilder::groupPolicy(*desc);
                if (policy.group_type == CacheGroupType::SWA)
                    policy.sliding_window_size = dsv41::kWindow;
                if (policy.enable_prefix_reuse && policy.group_type == CacheGroupType::SWA)
                    policy.explicit_block_num = o.swa_blocks + 1;
                groups.push_back({tag, SpecBuilder::build(*desc, ctx), policy});
            }
        }
        config.setTopology(std::move(groups), std::move(layers));
        return config;
    };
    auto config = build(dsv41::targetLayers());
    auto draft  = build(dsv41::draftLayers());
    config.mtp_sub_configs.push_back(config.mergeMTPModule(draft, 0, config.layer_num));
    config.finalizeBlockNums(o.device_blocks + 1, RuntimeConfig{});
    for (const auto& group : config.topology().groups()) {
        RTP_LLM_LOG_INFO(
            "metadata DSv4.1 tag=%s type=%s layers=%zu prefix_reuse=%d usable_blocks=%u span=%zu bytes=%zu",
            group.tag.c_str(),
            cacheGroupTypeName(group.policy.group_type),
            config.layerIdsForGroup(group.tag).size(),
            group.policy.enable_prefix_reuse,
            group.block_num - 1,
            group.seqSizePerBlock(),
            config.blockSizeBytesForGroup(group.tag));
    }
    return config;
}

class PendingTransfer final: public TransferAsyncContext {
public:
    void waitDone() override {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return done_; });
    }
    void onDone(DoneCallback callback) override {
        TransferResult result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!done_) {
                callbacks_.push_back(std::move(callback));
                return;
            }
            result = result_;
        }
        callback(result);
    }
    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }
    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_ && result_.ok();
    }
    TransferResult result() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return result_;
    }
    void finish(TransferResult result) {
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result_ = result;
            done_   = true;
            callbacks.swap(callbacks_);
        }
        cv_.notify_all();
        for (auto& callback : callbacks)
            callback(result);
    }

private:
    mutable std::mutex        mutex_;
    std::condition_variable   cv_;
    bool                      done_{false};
    TransferResult            result_;
    std::vector<DoneCallback> callbacks_;
};
class MetadataCompletion {
public:
    explicit MetadataCompletion(std::vector<GroupSetPtr> groups):
        groups_(std::move(groups)), tasks_(1, 10000, "MetadataCompletion") {
        require(tasks_.start(), "completion queue init");
    }
    ~MetadataCompletion() {
        tasks_.waitForIdle();
    }
    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) {
        auto result = std::make_shared<PendingTransfer>();
        require(tasks_.submit(BlockTreeTaskClass::BACKGROUND,
                              [this, task = std::move(task), result]() mutable {
                                  TransferResult status(TransferStatus::OK);
                                  try {
                                      if (fail_next.exchange(false))
                                          throw std::runtime_error("injected metadata transfer failure");
                                      for (const auto& d : task.descriptors()) {
                                          require(d.group_set_id < groups_.size(), "invalid transfer group");
                                          validate(*groups_[d.group_set_id], d.source_tier, d.source_blocks);
                                          validate(*groups_[d.group_set_id], d.target_tier, d.target_blocks);
                                          if (d.source_tier == Tier::DEVICE && d.target_tier == Tier::HOST)
                                              ++device_host;
                                          if (d.source_tier == Tier::HOST && d.target_tier == Tier::DISK)
                                              ++host_disk;
                                          if (d.source_tier == Tier::HOST && d.target_tier == Tier::DEVICE)
                                              ++host_device;
                                          if (d.source_tier == Tier::DISK && d.target_tier == Tier::DEVICE)
                                              ++disk_device;
                                      }
                                  } catch (const std::exception& error) {
                                      status = TransferResult(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
                                      ++failed;
                                  }
                                  ++transfers;
                                  status.expandBlockErrors(task.descriptors().size());
                                  result->finish(std::move(status));
                              }),
                "metadata completion submission failed");
        return result;
    }
    void wait() {
        tasks_.waitForIdle();
    }
    size_t pending() const {
        return tasks_.pendingTaskCount();
    }
    std::atomic<bool>     fail_next{false};
    std::atomic<uint64_t> transfers{0}, failed{0}, device_host{0}, host_disk{0}, host_device{0}, disk_device{0};

private:
    static void validate(const GroupSet& group, Tier tier, const BlockIndicesType& ids) {
        if (tier == Tier::DEVICE) {
            require(ids.size() == group.devicePools().size(), "device member count");
            for (size_t i = 0; i < ids.size(); ++i)
                require(group.devicePools()[i]->metadataOnly() && group.devicePools()[i]->isAllocated(ids[i]),
                        "device resource not allocated");
            return;
        }
        std::shared_ptr<IBlockPool> pool;
        if (tier == Tier::HOST)
            pool = group.hostPool();
        else if (tier == Tier::DISK)
            pool = group.diskPool();
        require(pool && pool->metadataOnly() && ids.size() == 1 && pool->isAllocated(ids[0]),
                "tier resource not allocated");
    }
    std::vector<GroupSetPtr> groups_;
    BlockTreeTaskPool        tasks_;
};
}  // namespace
MetadataReplayConfig::MetadataReplayConfig():
    last_n(CacheConfig{}.reuse_cache_tail_blocks),
    linear_step(1) {}

struct MetadataReplay::Impl {
    MetadataReplayConfig                     options;
    CacheConfig                              config;
    std::shared_ptr<CoordinatorCacheManager> coordinator;
    std::shared_ptr<MetadataCompletion>      completion;
    std::shared_ptr<BlockTreeCache>          cache;
    DeviceBlockPoolPtr                       decoder_pool;
    BlockTreeTaskPool*                       business{nullptr};
    BlockTransferDispatcher*                 dispatcher{nullptr};
    std::atomic<uint64_t>                    watermark_calls{0}, watermark_blocks{0}, force_calls{0}, force_blocks{0};
    std::atomic<uint64_t>                    watermark_ns{0}, force_ns{0};
    uint64_t                                 initial_copy_attempts = physicalCopyAttempts();
    double                                   init_us{0};
    uint64_t                                 last_timestamp{0};
    explicit Impl(const MetadataReplayConfig& o): options(o) {
        auto begin = Clock::now();
        require(o.block_tokens && o.cp_size && o.cp_rank < o.cp_size && o.device_blocks && o.linear_step > 0,
                "invalid metadata config");
        const size_t max_count = static_cast<size_t>(std::numeric_limits<int>::max()) - 1;
        require(o.min_replay_tokens > 0 && o.min_replay_tokens <= max_count && o.sliding_window <= max_count
                    && o.swa_block_tokens <= max_count && o.block_tokens <= max_count
                    && o.cp_size <= max_count / o.block_tokens,
                "token geometry exceeds native integer range");
        for (auto count : {o.device_blocks, o.swa_blocks, o.host_blocks, o.disk_blocks, o.linear_step})
            require(count <= max_count, "capacity exceeds native block ID range");
        for (auto levels : {std::make_pair(o.device_low, o.device_high),
                            std::make_pair(o.host_low, o.host_high),
                            std::make_pair(o.disk_low, o.disk_high)})
            require(levels.first >= 0 && levels.second <= 1
                        && ((levels.first == 0 && levels.second == 0) || levels.first < levels.second),
                    "invalid watermark levels");
        config      = o.dsv41_topology ? dsv41Config(o) : fixtureConfig(o);
        coordinator = std::make_shared<CoordinatorCacheManager>(config);
        require(coordinator->init(), "coordinator init");
        coordinator->setCPSlotMapper(std::make_shared<CPSlotMapper>(o.cp_rank, o.cp_size, o.block_tokens));
        std::vector<GroupSetPtr>              sets;
        std::vector<std::vector<std::string>> memberships;
        require(block_tree_layout::buildGroupMembers(config, memberships), "invalid BTC group membership");
        const auto topology =
            o.dsv41_topology ? block_tree_layout::projectCacheKeys(config, o.cp_size) : config.topologyPtr();
        std::map<std::string, DeviceBlockPoolPtr> pools;
        for (const auto& group : coordinator->cacheGroups())
            pools.emplace(group->tag(), group->blockPool());
        if (o.dsv41_topology)
            decoder_pool = pools.at("decoder_swa_kv");
        for (const auto& tags : memberships) {
            const auto&                     first = topology->group(tags.front());
            std::vector<DeviceBlockPoolPtr> members;
            size_t                          payload = 0;
            for (const auto& tag : tags) {
                members.push_back(pools.at(tag));
                payload += config.blockSizeBytesForGroup(tag);
            }
            const size_t                            stride = (payload + 4095) / 4096 * 4096;
            std::shared_ptr<HostBlockPool>          hp;
            std::shared_ptr<BlockTreeDiskBlockPool> kp;
            const bool swa = first.policy.group_type == CacheGroupType::SWA;
            const size_t host_blocks = swa && o.host_swa_blocks ? o.host_swa_blocks : o.host_blocks;
            const size_t disk_blocks = swa && o.disk_swa_blocks ? o.disk_swa_blocks : o.disk_blocks;
            if (host_blocks) {
                auto c                  = std::make_shared<HostBlockPoolConfig>();
                c->pool_type            = BlockPoolType::HOST;
                c->pool_name            = first.tag + "_host";
                c->metadata_only        = true;
                c->physical_block_count = host_blocks + 1;
                c->payload_bytes        = payload;
                c->stride_bytes         = stride;
                hp                      = std::make_shared<HostBlockPool>(c);
                require(hp->init(), "host metadata init");
            }
            if (disk_blocks) {
                auto c                  = std::make_shared<BlockTreeDiskBlockPoolConfig>();
                c->pool_type            = BlockPoolType::DISK;
                c->pool_name            = first.tag + "_disk";
                c->metadata_only        = true;
                c->physical_block_count = disk_blocks + 1;
                c->payload_bytes        = payload;
                c->stride_bytes         = stride;
                c->disk_size_bytes      = stride * (disk_blocks + 1);
                c->work_dir             = "metadata://no-kv-files";
                kp                      = std::make_shared<BlockTreeDiskBlockPool>(c);
                require(kp->init(), "disk metadata init");
            }
            sets.push_back(first.policy.group_type == CacheGroupType::SWA ?
                               BenchmarkFixture::createSWAGroupSet(members,
                                                                   hp,
                                                                   kp,
                                                                   sets.size(),
                                                                   topology,
                                                                   tags,
                                                                   first.policy.sliding_window_size,
                                                                   first.seqSizePerBlock()) :
                               BenchmarkFixture::createFullGroupSet(members, hp, kp, sets.size(), topology, tags));
        }
        completion = std::make_shared<MetadataCompletion>(sets);
        auto disp  = std::make_unique<BlockTransferDispatcher>(
            BlockTransferDispatcher::WithSubmitter{},
            BlockTransferDispatcher::Submitter(
                [backend = completion](TransferTask task) { return backend->execute(std::move(task)); }));
        dispatcher   = disp.get();
        auto pool    = std::make_unique<BlockTreeTaskPool>(1, 10000, "MetadataBTC");
        business     = pool.get();
        auto metrics = std::make_shared<BlockTreeCacheMetricsReporter>(nullptr);
        metrics->setEvictionTimer([this](bool force, uint64_t ns) { (force ? force_ns : watermark_ns) += ns; });
        metrics->setEvictionObserver([this](Tier, CacheGroupType, bool force, size_t, size_t scheduled) {
            if (force) {
                ++force_calls;
                force_blocks += scheduled;
            } else {
                ++watermark_calls;
                watermark_blocks += scheduled;
            }
        });
        BlockTreeCacheConfig bc;
        if (o.disk_fifo)
            bc.disk_eviction_policy = EvictionPolicy::FIFO;
        bc.enable_host_cache = o.host_blocks > 0;
        bc.enable_disk_cache = o.disk_blocks > 0;
        bc.watermark_device  = {o.device_low, o.device_high};
        bc.watermark_host    = {o.host_low, o.host_high};
        bc.watermark_disk    = {o.disk_low, o.disk_high};
        cache                = std::make_shared<BlockTreeCache>(
            std::make_unique<BlockTree>(sets), bc, nullptr, std::move(disp), std::move(pool), metrics);
        require(cache->init(), "BTC init");
        coordinator->attachBlockTreeCache(cache);
        init_us = micros(begin);
    }
    ~Impl() {
        settle();
        coordinator.reset();
        cache.reset();
        completion.reset();
    }
    void settle() {
        do {
            business->waitForIdle();
            completion->wait();
            dispatcher->drainTransfers();
            business->waitForIdle();
        } while (completion->pending() || business->pendingTaskCount() || dispatcher->hasPendingTransfers());
    }
    void check() {
        if (decoder_pool) {
            // The nonreusable decoder pool is outside the tree and its poolMetricsSnapshots().
            require(decoder_pool->usedBlocksNum() == 0, "decoder pool retained blocks after request release");
        }
        for (const auto& s : cache->poolMetricsSnapshots())
            require(!s.request_ref_blocks && !s.load_ref_blocks && !s.store_ref_blocks && !s.eviction_target_ref_blocks
                        && !s.frozen_ref_blocks,
                    "unreleased request/transfer references");
    }
    MetadataRequestResult run(const MetadataRequest& r) {
        require(r.input_tokens > 0
                    && r.prefix_keys.size() == static_cast<size_t>(r.input_tokens) / options.block_tokens,
                "complete prefix keys must match declared base granularity");
        require(r.timestamp_ns >= last_timestamp, "request order violation");
        last_timestamp = r.timestamp_ns;
        MetadataRequestResult out;
        auto                  begin  = Clock::now();
        auto                  tokens = std::make_shared<CompleteTokenIds>(1, 1, r.input_tokens, options.block_tokens);
        tokens->initCacheMetadata(r.input_tokens);
        auto resources = std::make_shared<BatchKVCacheResource>();
        resources->resetBatchSize(1);
        resources->initGroups(config.topologyPtr());
        for (auto key : r.prefix_keys)
            resources->pushBackCacheKey(0, key);
        resources->setLastBlockAligned(r.input_tokens % options.block_tokens == 0);
        resources->markCacheKeysInitialized();
        out.stage_us["request_setup"] = micros(begin);
        MallocInfo mi;
        mi.batch_kv_cache_resource      = resources;
        mi.complete_token_ids           = tokens;
        mi.request_id                   = r.id;
        mi.verbose                      = false;
        mi.enable_storage_backend       = false;
        mi.enable_remove_skipped_blocks = false;
        try {
            begin                                             = Clock::now();
            auto allocated                                    = coordinator->malloc(mi);
            out.stage_us["malloc_including_match_force_drop"] = micros(begin);
            out.stage_us["match_nested"]                      = allocated.match_cost_time_us;
            out.success                                       = allocated.success;
            out.reuse_tokens                                  = allocated.reuse_len;
            out.status                                        = allocated.success ? "ok" : "allocation_failed";
            begin                                             = Clock::now();
            if (allocated.async_context) {
                allocated.async_context->waitDone();
                out.success = out.success && allocated.async_context->success();
                if (out.success) {
                    auto load = std::dynamic_pointer_cast<LoadAsyncContext>(allocated.async_context);
                    require(load != nullptr, "unexpected load context");
                    out.reuse_tokens =
                        static_cast<int>(load->matchedBlocks()) * coordinator->cpSlotMapper()->reuseBlockTokens(config);
                } else
                    out.status = "load_failed";
            }
            settle();
            out.stage_us["load_ready_settlement"] = micros(begin);
            if (out.success) {
                begin           = Clock::now();
                size_t resident = 0;
                coordinator->insertIntoCache(InsertInfo{resources, tokens, false, Tier::DEVICE, r.id}, resident);
                out.stage_us["insert_including_watermark_plan"] = micros(begin);
            } else
                out.reuse_tokens = 0;
            begin = Clock::now();
            coordinator->free(FreeInfo{resources, tokens, r.id});
            out.stage_us["release"] = micros(begin);
            begin                   = Clock::now();
            settle();
            out.stage_us["post_request_settlement"] = micros(begin);
            begin                                   = Clock::now();
            check();
            out.stage_us["reference_validation"] = micros(begin);
            return out;
        } catch (...) {
            coordinator->free(FreeInfo{resources, tokens, r.id});
            settle();
            throw;
        }
    }
};
MetadataReplay::MetadataReplay(const MetadataReplayConfig& c): impl_(std::make_unique<Impl>(c)) {}
MetadataReplay::~MetadataReplay() = default;
MetadataRequestResult MetadataReplay::run(const MetadataRequest& r) {
    return impl_->run(r);
}
double MetadataReplay::initMicros() const {
    return impl_->init_us;
}
void MetadataReplay::failNextTransfer() {
    impl_->completion->fail_next = true;
}
MetadataReplayStats MetadataReplay::stats() const {
    const auto&         i = *impl_;
    MetadataReplayStats s;
    s.watermark_calls  = i.watermark_calls;
    s.watermark_blocks = i.watermark_blocks;
    s.force_calls      = i.force_calls;
    s.force_blocks     = i.force_blocks;
    s.transfers        = i.completion->transfers;
    s.failed_transfers = i.completion->failed;
    s.device_host      = i.completion->device_host;
    s.host_disk        = i.completion->host_disk;
    s.host_device      = i.completion->host_device;
    s.disk_device      = i.completion->disk_device;
    s.tree_nodes       = i.cache->getStats().tree_node_count;
    for (const auto& p : i.cache->poolMetricsSnapshots()) {
        if (p.tier == Tier::DEVICE)
            s.device_used += p.used_blocks;
        else if (p.tier == Tier::HOST)
            s.host_used += p.used_blocks;
        else if (p.tier == Tier::DISK)
            s.disk_used += p.used_blocks;
    }
    for (const auto& pool : i.coordinator->groupBlockPools())
        s.payload_bytes += pool->getAllocationSizeBytes();
    for (const auto& group : i.cache->groupSets()) {
        if (auto host = group->hostPool())
            s.payload_bytes += host->allocatedPayloadBytes();
        if (auto disk = group->diskPool())
            require(disk->filePath().empty() && disk->readBytes() == 0 && disk->writeBytes() == 0,
                    "physical disk backing or IO observed");
    }
    s.copy_attempts = physicalCopyAttempts() - i.initial_copy_attempts;
    s.watermark_ns  = i.watermark_ns;
    s.force_ns      = i.force_ns;
    return s;
}
bool MetadataReplay::checkPinnedForceDrop() {
    auto& i = *impl_;
    i.settle();
    auto group = i.coordinator->cacheGroups().front();
    auto pool  = group->blockPool();
    // The fixture isolates pinning from cache occupancy.
    require(pool->freeBlocksNum() == pool->totalBlocksNum(), "pin test needs an empty instance");
    i.settle();
    auto ids = pool->malloc(pool->freeBlocksNum());
    require(ids.has_value(), "pin fixture allocation");
    pool->incRef(*ids);
    auto before    = i.force_calls.load();
    bool ok        = group->ensureFreeBlocks(1);
    bool preserved = pool->referencedBlocksNum() == ids->size();
    pool->decRef(*ids);
    i.settle();
    i.check();
    return !ok && preserved && i.force_calls > before;
}
}  // namespace rtp_llm::benchmark
