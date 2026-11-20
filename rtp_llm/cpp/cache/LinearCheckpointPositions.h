#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

// One prompt's checkpoints. Depth is measured in complete BTC keys, while
// Linear allocation and publication use zero-based slots in the request.
class LinearCheckpointPositions {
public:
    static LinearCheckpointPositions select(size_t full, size_t complete, int count, int cp_scale, size_t reused) {
        LinearCheckpointPositions plan;
        plan.enabled_ = true;
        if (complete == 0) {
            return plan;
        }
        const size_t first     = full ? std::min(full, complete) : 1;
        const size_t span      = complete - first;
        const size_t intervals = std::min<size_t>(count - 1, span);
        for (size_t i = 0; i <= intervals; ++i) {
            const size_t depth = first + (intervals ? (uint64_t(i) * span + intervals / 2) / intervals : 0);
            const size_t slot  = depth * cp_scale - 1;
            plan.positions_.push_back(slot);
            if (depth > reused) {
                plan.compute_.push_back(slot);
            }
        }
        return plan;
    }

    bool enabled() const {
        return enabled_;
    }
    bool contains(size_t slot) const {
        return std::binary_search(positions_.begin(), positions_.end(), slot);
    }
    const std::vector<size_t>& toCompute() const {
        return compute_;
    }

private:
    bool                enabled_{false};
    std::vector<size_t> positions_;
    std::vector<size_t> compute_;
};

}  // namespace rtp_llm
