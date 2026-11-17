#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace rtp_llm {

struct MatchLengths {
    uint64_t full_blocks{0};
    uint64_t joint_blocks{0};
    uint64_t key_count{0};
    uint64_t timestamp_ns{0};
    int64_t  endpoint_key{0};
    int64_t  parent_key{0};
};

struct BranchEviction {
    uint64_t node_id;
    uint64_t branch_hits;
};

struct MatchEndpoint {
    std::vector<BranchEviction> evictions;
    uint64_t                    count{0};
    uint64_t                    first_ns{0};
    uint64_t                    last_ns{0};
};

// Cache lifetime history survives node eviction. The owning cache mutex protects it.
class MatchHistory {
public:
    using Identity = std::tuple<int64_t, int64_t, uint64_t>;
    std::set<int64_t> root_context_keys;
    uint64_t          evicted_nodes{0};
    uint64_t          historical_branch_instances{0};
    void              record(int64_t key, int64_t parent, uint64_t depth, uint64_t timestamp_ns) {
        auto& entry = endpoints[{key, parent, depth}];
        if (entry.count == 0)
            entry.first_ns = timestamp_ns;
        ++entry.count;
        entry.last_ns = timestamp_ns;
    }
    MatchLengths                      latest;
    std::map<Identity, MatchEndpoint> endpoints;
};
}  // namespace rtp_llm
