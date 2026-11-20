#include "rtp_llm/cpp/cache/LinearCheckpointPositions.h"

#include <cstdlib>
#include <vector>

using rtp_llm::LinearCheckpointPositions;

int main() {
    // Check every small span, including no FULL hit, coincident endpoints,
    // and more requested checkpoints than complete blocks.
    for (size_t complete = 0; complete <= 24; ++complete) {
        for (size_t full = 0; full <= complete; ++full) {
            for (size_t reused = 0; reused <= full; ++reused) {
                for (int n : {2, 3, 5, 32}) {
                    for (int cp : {1, 8}) {
                        auto                plan  = LinearCheckpointPositions::select(full, complete, n, cp, reused);
                        const size_t        start = std::max<size_t>(1, full);
                        std::vector<size_t> selected, compute;
                        for (size_t slot = 0; slot < (complete + 1) * cp; ++slot) {
                            if (!plan.contains(slot))
                                continue;
                            const size_t depth = (slot + 1) / cp;
                            if ((slot + 1) % cp || depth < start || depth > complete)
                                return EXIT_FAILURE;
                            selected.push_back(depth);
                            if (depth > reused)
                                compute.push_back(slot);
                        }
                        if (compute != plan.toCompute() || selected.size() != std::min<size_t>(n, complete + 1 - start))
                            return EXIT_FAILURE;
                        if (selected.empty())
                            continue;
                        if (selected.front() != start || selected.back() != complete)
                            return EXIT_FAILURE;
                        const size_t intervals = selected.size() - 1;
                        for (size_t i = 1; i < selected.size(); ++i) {
                            const size_t gap = selected[i] - selected[i - 1];
                            if (gap < (complete - start) / intervals
                                || gap > (complete - start + intervals - 1) / intervals)
                                return EXIT_FAILURE;
                        }
                    }
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
