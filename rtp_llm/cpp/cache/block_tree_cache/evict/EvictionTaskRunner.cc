#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {

EvictionTaskRunner::EvictionTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                                       const BlockTransferDispatcher*  transfer_dispatcher,
                                       int                             memory_timeout_ms,
                                       int                             disk_timeout_ms):
    group_sets_(group_sets),
    transfer_dispatcher_(transfer_dispatcher),
    memory_timeout_ms_(memory_timeout_ms),
    disk_timeout_ms_(disk_timeout_ms) {}

EvictionTaskResult EvictionTaskRunner::runTransfer(const EvictionTask&            task,
                                                   BlockTreeCacheMetricsReporter& metrics_reporter) const {
    EvictionTaskResult     task_result;
    BlockTreeTransferBytes transfer_bytes;
    int64_t                transfer_begin_time_us = 0;
    bool                   transfer_started       = false;
    const auto             finish_metrics         = [&]() {
        if (!transfer_started) {
            return;
        }
        transfer_started = false;
        metrics_reporter.reportTransferFinished(CacheTransferOperation::EVICT,
                                                task.primary_desc.source_tier,
                                                task.primary_desc.target_tier,
                                                task.cascade_descs.size() + 1,
                                                transfer_begin_time_us,
                                                task_result.primary_success,
                                                transfer_bytes);
    };

    try {
        transfer_begin_time_us = metrics_reporter.reportTransferStarted(
            CacheTransferOperation::EVICT, task.primary_desc.source_tier, task.primary_desc.target_tier);
        transfer_started = true;

        std::vector<TransferDescriptor> descriptors;
        const bool                      transfer_ready = buildTransferDescriptors(task, descriptors);
        task_result.cascade_success.assign(task.cascade_descs.size(), false);
        if (!transfer_ready) {
            finish_metrics();
            return task_result;
        }

        const int timeout_ms = selectTransferTimeoutMs(task, memory_timeout_ms_, disk_timeout_ms_);
        const auto execute = [&](const TransferDescriptor& descriptor) {
            auto context = transfer_dispatcher_->executeMultiRank({descriptor}, timeout_ms);
            context->waitDone();
            const bool transfer_success = context->success();
            if (transfer_success) {
                metrics_reporter.accumulateTransferBytes({descriptor}, group_sets_, transfer_bytes);
            }
            return transfer_success;
        };

        task_result.primary_success = execute(task.primary_desc);
        if (task_result.primary_success) {
            for (size_t i = 0; i < task.cascade_descs.size(); ++i) {
                task_result.cascade_success[i] = execute(task.cascade_descs[i]);
            }
        }
        finish_metrics();
        return task_result;
    } catch (...) {
        finish_metrics();
        throw;
    }
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
