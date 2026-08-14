#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include <algorithm>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

EvictionTaskRunner::EvictionTaskRunner(const BlockTransferDispatcher* transfer_dispatcher,
                                       int                            memory_timeout_ms,
                                       int                            disk_timeout_ms):
    transfer_dispatcher_(transfer_dispatcher),
    memory_timeout_ms_(memory_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms) {}

EvictionTaskResult EvictionTaskRunner::runPerRankTransfer(const EvictionTask& task) const {
    return executeTransfer(task, false);
}

EvictionTaskResult EvictionTaskRunner::runTransfer(const EvictionTask& task) const {
    if (!transfer_dispatcher_->hasMultiRankEngine()) {
        return runPerRankTransfer(task);
    }
    return executeTransfer(task, true);
}

EvictionTaskResult EvictionTaskRunner::executeTransfer(const EvictionTask& task, bool multi_rank) const {
    EvictionTaskResult task_result;
    task_result.cascade_success.assign(task.cascade_descs.size(), false);

    std::vector<TransferDescriptor> descriptors;
    const bool transfer_ready = buildTransferDescriptors(task, descriptors);
    if (!transfer_ready) {
        return task_result;
    }

    const auto submit = [this, &task, multi_rank](const std::vector<TransferDescriptor>& batch) {
        if (multi_rank) {
            return transfer_dispatcher_->executeMultiRank(
                batch, selectTransferTimeoutMs(task, memory_timeout_ms_, disk_timeout_ms_));
        }
        return transfer_dispatcher_->executePerRank(batch);
    };

    const std::vector<TransferDescriptor> primary_batch{descriptors.front()};
    const auto                            primary_context = submit(primary_batch);
    primary_context->waitDone();
    task_result.primary_success = primary_context->success();
    if (!task_result.primary_success) {
        return task_result;
    }

    const std::vector<TransferDescriptor> cascade_descriptors(descriptors.begin() + 1, descriptors.end());
    const auto                            batches = partitionTransferDescriptors(cascade_descriptors);
    std::vector<std::shared_ptr<AsyncContext>> contexts;
    contexts.reserve(batches.size());
    for (const auto& batch : batches) {
        contexts.push_back(submit(batch.descriptors));
    }
    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index) {
        contexts[batch_index]->waitDone();
        const bool transfer_success = contexts[batch_index]->success();
        for (const size_t descriptor_index : batches[batch_index].descriptor_indices) {
            task_result.cascade_success[descriptor_index] = transfer_success;
        }
    }
    return task_result;
}

std::vector<EvictionTaskRunner::TransferBatch>
EvictionTaskRunner::partitionTransferDescriptors(const std::vector<TransferDescriptor>& descriptors) const {
    std::vector<TransferBatch> batches;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto& descriptor   = descriptors[descriptor_index];
        const bool  force_single = descriptor.source_tier == Tier::DEVICE && descriptor.target_tier == Tier::DISK;
        auto        batch        = batches.end();
        if (!force_single) {
            batch = std::find_if(batches.begin(), batches.end(), [&](const auto& current) {
                const auto& first = current.descriptors.front();
                return first.source_tier == descriptor.source_tier && first.target_tier == descriptor.target_tier
                    && first.group_set_id == descriptor.group_set_id;
            });
        }
        if (batch == batches.end()) {
            batches.push_back(TransferBatch{{descriptor}, {descriptor_index}});
        } else {
            batch->descriptors.push_back(descriptor);
            batch->descriptor_indices.push_back(descriptor_index);
        }
    }
    return batches;
}

bool EvictionTaskRunner::buildTransferDescriptors(const EvictionTask&              task,
                                                  std::vector<TransferDescriptor>& descriptors) {
    descriptors.clear();
    descriptors.reserve(1 + task.cascade_descs.size());

    if (!task.primary_desc.isExecutable()) {
        return false;
    }
    descriptors.push_back(task.primary_desc);

    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (!cascade_desc.isExecutable()) {
            descriptors.clear();
            return false;
        }
        descriptors.push_back(cascade_desc);
    }
    return true;
}

int EvictionTaskRunner::selectTransferTimeoutMs(const EvictionTask& task, int memory_timeout_ms, int disk_timeout_ms) {
    bool uses_disk = task.primary_desc.source_tier == Tier::DISK || task.primary_desc.target_tier == Tier::DISK;
    for (const TransferDescriptor& cascade_desc : task.cascade_descs) {
        if (cascade_desc.source_tier == Tier::DISK || cascade_desc.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    return uses_disk ? disk_timeout_ms : memory_timeout_ms;
}

}  // namespace rtp_llm
