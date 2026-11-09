#pragma once
#include <algorithm>
#include <limits>
#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm::block_tree_layout {

inline int slidingWindowSize(const GroupBase& group) {
    RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == CacheGroupType::SWA,
                            "sliding window requested for non-SWA tag=%s",
                            group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "SWA tag=%s has null cache spec", group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(group.policy.sliding_window_size >= 0,
                            "SWA tag=%s has invalid sliding window=%d",
                            group.tag.c_str(),
                            group.policy.sliding_window_size);
    return group.policy.sliding_window_size;
}

inline bool
groupSetSemanticsCompatible(const CacheConfig& cache_config, std::string_view lhs_tag, std::string_view rhs_tag) {
    const GroupBase& lhs = cache_config.topology().group(lhs_tag);
    const GroupBase& rhs = cache_config.topology().group(rhs_tag);
    if (lhs.policy.group_type != rhs.policy.group_type || lhs.seqSizePerBlock() != rhs.seqSizePerBlock()
        || lhs.policy.cp_mapping != rhs.policy.cp_mapping) {
        return false;
    }
    if (lhs.policy.group_type == CacheGroupType::SWA) {
        if (slidingWindowSize(lhs) != slidingWindowSize(rhs)) {
            return false;
        }
    }
    if (lhs.policy.group_type == CacheGroupType::SWA || lhs.policy.group_type == CacheGroupType::LINEAR) {
        if (lhs.policy.active_tail_blocks != rhs.policy.active_tail_blocks) {
            return false;
        }
    }
    return true;
}

inline bool buildGroupMembers(const CacheConfig& cache_config, std::vector<std::vector<std::string>>& group_members) {
    group_members.clear();
    for (const GroupBase& group : cache_config.topology().groups()) {
        if (!group.policy.enable_prefix_reuse) {
            continue;
        }

        const CacheGroupType group_type = group.policy.group_type;
        if (group_type != CacheGroupType::FULL && group_type != CacheGroupType::SWA
            && group_type != CacheGroupType::LINEAR) {
            RTP_LLM_LOG_ERROR("unsupported reusable cache group type for tag=%s", group.tag.c_str());
            return false;
        }

        auto it =
            std::find_if(group_members.begin(), group_members.end(), [&](const std::vector<std::string>& members) {
                const GroupBase& first = cache_config.topology().group(members.front());
                return first.policy.group_type == group.policy.group_type;
            });
        if (it == group_members.end()) {
            group_members.push_back({group.tag});
        } else {
            if (!groupSetSemanticsCompatible(cache_config, it->front(), group.tag)) {
                RTP_LLM_LOG_ERROR("incompatible BlockTree reuse coordinates for tags=%s and %s",
                                  it->front().c_str(),
                                  group.tag.c_str());
                return false;
            }
            it->push_back(group.tag);
        }
    }
    return true;
}

inline std::shared_ptr<const CacheTopology> projectCacheKeys(const CacheConfig& cache_config, size_t cp_size) {
    auto cache_topology = cache_config.topologyPtr();
    if (cp_size > 1) {
        auto groups = cache_topology->groups();
        RTP_LLM_CHECK_WITH_INFO(cache_config.seq_size_per_block <= std::numeric_limits<size_t>::max() / cp_size,
                                "canonical CP cache key stride overflow");
        for (auto& group : groups) {
            auto spec                    = group.spec->clone();
            spec->cache_key_token_stride = cache_config.seq_size_per_block * cp_size;
            group.spec                   = std::move(spec);
        }
        cache_topology = CacheTopology::create(std::move(groups), cache_topology->layers());
    }
    return cache_topology;
}
}  // namespace rtp_llm::block_tree_layout
