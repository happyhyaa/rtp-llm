#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/MetadataCheckpoint.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"

namespace rtp_llm {
using benchmark::StateWriter;
using benchmark::StateReader;
using benchmark::stateRequire;
namespace {
void saveMeta(StateWriter& w, const CandidateMeta& m, bool normalize_time) {
    w.scalar(m.last_access_seq);
    w.scalar(m.admission_seq);
    w.scalar(m.hit_count);
    w.scalar(normalize_time ? int64_t(0) : m.tier_last_hit_time_us);
    w.scalar(normalize_time ? int64_t(0) : m.last_access_time_us);
    w.scalar(normalize_time ? int64_t(0) : m.tier_enter_time_us);
}
CandidateMeta readMeta(StateReader& r) {
    CandidateMeta m;
    m.last_access_seq       = r.scalar<uint64_t>();
    m.admission_seq         = r.scalar<uint64_t>();
    m.hit_count             = r.scalar<uint64_t>();
    m.tier_last_hit_time_us = r.scalar<int64_t>();
    m.last_access_time_us   = r.scalar<int64_t>();
    m.tier_enter_time_us    = r.scalar<int64_t>();
    return m;
}
std::vector<std::shared_ptr<IBlockPool>> pools(const BlockTree& tree) {
    std::vector<std::shared_ptr<IBlockPool>> result;
    for (const auto& group : tree.groupSets()) {
        for (const auto& pool : group->devicePools())
            result.push_back(pool);
        if (group->hostPool())
            result.push_back(group->hostPool());
        if (group->diskPool())
            result.push_back(group->diskPool());
    }
    return result;
}
}  // namespace
void BlockTreeMetadataCheckpoint::savePool(IBlockPool& p, StateWriter& w) {
    std::lock_guard<std::mutex> lock(p.mutex_);
    stateRequire(p.metadataOnly(), "checkpoint requires metadata-only pool without decode origins");
    w.text(p.poolName());
    w.scalar(uint64_t(p.totalBlocksNumNoLock()));
    w.vector(p.allocated_);
    w.vector(p.tree_refcounts_);
    for (const auto& refs : p.tree_refcounts_by_type_)
        w.vector(refs);
    for (const auto count : p.tree_referenced_block_counts_)
        w.scalar(uint64_t(count));
    w.vector(p.free_blocks_);
    w.vector(p.released_blocks_);
    w.scalar(uint64_t(p.free_head_));
    w.scalar(uint64_t(p.active_blocks_num_));
    w.scalar(uint64_t(p.available_blocks_num_));
    auto* d = dynamic_cast<DeviceBlockPool*>(&p);
    w.scalar(bool(d));
    if (d) {
        w.vector(d->refcounts_);
        w.scalar(uint64_t(d->request_referenced_blocks_num_));
    }
}
void BlockTreeMetadataCheckpoint::restorePool(IBlockPool& p, StateReader& r) {
    std::lock_guard<std::mutex> lock(p.mutex_);
    stateRequire(p.metadataOnly(), "restore requires metadata-only pool without decode origins");
    stateRequire(r.text() == p.poolName() && r.scalar<uint64_t>() == p.totalBlocksNumNoLock(),
                 "checkpoint pool geometry mismatch");
    auto allocated = r.vector<uint8_t>();
    stateRequire(allocated.size() == p.allocated_.size(), "checkpoint allocation shape mismatch");
    p.allocated_      = std::move(allocated);
    p.tree_refcounts_ = r.vector<uint32_t>();
    for (auto& refs : p.tree_refcounts_by_type_)
        refs = r.vector<uint32_t>();
    for (auto& count : p.tree_referenced_block_counts_)
        count = r.scalar<uint64_t>();
    p.free_blocks_          = r.vector<BlockIdxType>();
    p.released_blocks_      = r.vector<BlockIdxType>();
    p.free_head_            = r.scalar<uint64_t>();
    p.active_blocks_num_    = r.scalar<uint64_t>();
    p.available_blocks_num_ = r.scalar<uint64_t>();
    auto* d                 = dynamic_cast<DeviceBlockPool*>(&p);
    stateRequire(r.scalar<bool>() == bool(d), "checkpoint pool type mismatch");
    if (d) {
        d->refcounts_                     = r.vector<uint32_t>();
        d->request_referenced_blocks_num_ = r.scalar<uint64_t>();
    }
}
void BlockTreeMetadataCheckpoint::save(BlockTreeCache& cache, StateWriter& w, bool normalize_time) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    auto&                       tree    = *cache.tree_;
    auto&                       evictor = cache.evictor_;
    stateRequire(tree.pending_node_pool_.empty() && !tree.event_publisher_, "checkpoint requires quiescent local tree");
    for (auto [pool, count] : evictor.pending_release_counts_)
        stateRequire(count == 0, "pending pool releases");
    const auto ps = pools(tree);
    w.scalar(uint64_t(ps.size()));
    for (auto& pool : ps)
        savePool(*pool, w);
    w.scalar(tree.key_version_);
    w.scalar(uint64_t(tree.node_pool_.size()));
    for (const auto& node : tree.node_pool_) {
        w.scalar(node->cache_key);
        w.scalar(node->parent == tree.root_.get() ? UINT64_MAX : uint64_t(node->parent->index));
        w.scalar(node->is_resident);
        w.vector(node->token_ids);
        w.scalar(uint64_t(node->group_set_resources.size()));
        for (const auto& resource : node->group_set_resources) {
            stateRequire(resource.transfer_state == GroupSetTransferState::IDLE && !resource.transfer_detached,
                         "checkpoint contains in-flight transfer");
            w.vector(resource.device_blocks);
            w.scalar(resource.host_block);
            w.scalar(resource.disk_block);
            saveMeta(w, resource.candidate_meta, normalize_time);
        }
    }
    w.scalar(evictor.access_seq_);
    w.scalar(evictor.admission_seq_);
    for (size_t group = 0; group < evictor.heaps_.size(); ++group) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            auto                                                   candidates = evictor.heapFor(group, tier)->nodes();
            std::set<std::tuple<uint64_t, uint64_t, CacheKeyType>> ranks;
            const auto policy = tier == Tier::DEVICE ? cache.config_.device_eviction_policy :
                                tier == Tier::HOST   ? cache.config_.host_eviction_policy :
                                                       cache.config_.disk_eviction_policy;
            for (const auto* node : candidates) {
                const auto&    meta   = node->group_set_resources[group].candidate_meta;
                const uint64_t first  = policy == EvictionPolicy::LFU  ? meta.hit_count :
                                        policy == EvictionPolicy::FIFO ? meta.admission_seq :
                                                                         meta.last_access_seq;
                const uint64_t second = policy == EvictionPolicy::LFU  ? meta.last_access_seq :
                                        policy == EvictionPolicy::FIFO ? 0 :
                                                                         meta.admission_seq;
                stateRequire(ranks.emplace(first, second, node->cache_key).second,
                             "checkpoint cannot preserve native pointer-based eviction ties");
            }
            std::vector<uint64_t> indexes;
            for (const auto* node : candidates)
                indexes.push_back(node->index);
            std::sort(indexes.begin(), indexes.end());
            w.vector(indexes);
        }
        const auto& h = evictor.heaps_[group];
        stateRequire(h.pending_timings.empty(), "pending eviction timings");
        for (size_t tier = 0; tier < 3; ++tier) {
            const auto& round = h.watermark_rounds[tier];
            stateRequire(round.pending_count == 0, "pending watermark round");
            w.scalar(h.watermark_triggered[tier]);
            w.scalar(uint64_t(round.required_count));
            w.scalar(uint64_t(round.scheduled_count));
        }
    }
}
void BlockTreeMetadataCheckpoint::restore(BlockTreeCache& cache, StateReader& r) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    auto&                       tree    = *cache.tree_;
    auto&                       evictor = cache.evictor_;
    stateRequire(tree.size() == 0 && tree.pending_node_pool_.empty(), "restore requires empty tree");
    const auto ps = pools(tree);
    stateRequire(r.scalar<uint64_t>() == ps.size(), "checkpoint topology mismatch");
    for (auto& pool : ps)
        restorePool(*pool, r);
    const auto version = r.scalar<int64_t>();
    const auto count   = r.scalar<uint64_t>();
    stateRequire(count < (1ULL << 32), "invalid checkpoint node count");
    std::vector<uint64_t> parents;
    parents.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        auto node       = std::make_unique<TreeNode>();
        node->index     = i;
        node->cache_key = r.scalar<CacheKeyType>();
        parents.push_back(r.scalar<uint64_t>());
        node->is_resident = r.scalar<bool>();
        node->token_ids   = r.vector<int>();
        const auto groups = r.scalar<uint64_t>();
        stateRequire(groups == tree.groupSets().size(), "node topology mismatch");
        node->group_set_resources.resize(groups);
        for (auto& resource : node->group_set_resources) {
            resource.device_blocks  = r.vector<BlockIdxType>();
            resource.host_block     = r.scalar<BlockIdxType>();
            resource.disk_block     = r.scalar<BlockIdxType>();
            resource.candidate_meta = readMeta(r);
        }
        tree.cache_keys_.push_back(node->cache_key);
        tree.node_pool_.push_back(std::move(node));
    }
    for (uint64_t i = 0; i < count; ++i) {
        stateRequire(parents[i] == UINT64_MAX || (parents[i] < count && parents[i] != i), "invalid checkpoint parent");
        auto* node   = tree.node_pool_[i].get();
        node->parent = parents[i] == UINT64_MAX ? tree.root_.get() : tree.node_pool_[parents[i]].get();
        stateRequire(node->parent->children.emplace(node->cache_key, node).second, "duplicate child key");
    }
    tree.key_version_      = version;
    evictor.access_seq_    = r.scalar<uint64_t>();
    evictor.admission_seq_ = r.scalar<uint64_t>();
    for (size_t group = 0; group < evictor.heaps_.size(); ++group) {
        for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
            const auto indexes = r.vector<uint64_t>();
            for (const auto index : indexes) {
                stateRequire(index < count, "invalid candidate index");
                auto* node = tree.node_pool_[index].get();
                evictor.heapFor(group, tier)->upsert(node, node->group_set_resources[group].candidate_meta);
            }
        }
        auto& h = evictor.heaps_[group];
        for (size_t tier = 0; tier < 3; ++tier) {
            h.watermark_triggered[tier]              = r.scalar<bool>();
            h.watermark_rounds[tier].required_count  = r.scalar<uint64_t>();
            h.watermark_rounds[tier].scheduled_count = r.scalar<uint64_t>();
        }
    }
}
}  // namespace rtp_llm
