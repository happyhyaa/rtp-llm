#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTask.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class EvictionTaskRunner {
public:
    EvictionTaskRunner(const BlockTransferDispatcher* transfer_dispatcher,
                       int                             memory_timeout_ms,
                       int                             disk_timeout_ms);

    EvictionTaskResult runTransfer(const EvictionTask& task) const;

private:
    struct TransferBatch {
        std::vector<TransferDescriptor> descriptors;
        std::vector<size_t>             descriptor_indices;
    };

    EvictionTaskResult runPerRankTransfer(const EvictionTask& task) const;
    EvictionTaskResult executeTransfer(const EvictionTask& task, bool multi_rank) const;
    static bool        buildTransferDescriptors(const EvictionTask&              task,
                                                std::vector<TransferDescriptor>& descriptors);
    std::vector<TransferBatch>
    partitionTransferDescriptors(const std::vector<TransferDescriptor>& descriptors) const;
    static int selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms);

    const BlockTransferDispatcher* transfer_dispatcher_{nullptr};
    int                            memory_timeout_ms_{0};
    int                            disk_timeout_ms_{0};
};

}  // namespace rtp_llm
