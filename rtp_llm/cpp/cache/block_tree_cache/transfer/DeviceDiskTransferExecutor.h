#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;
class BlockTreeTaskPool;

class DeviceDiskTransferExecutor {
public:
    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count);
    ~DeviceDiskTransferExecutor() = default;

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    TransferStatus                execute(const TransferDescriptor& desc, const GroupSet& group);
    std::shared_ptr<AsyncContext> diskToDevice(const std::vector<TransferDescriptor>& descriptors,
                                               const std::vector<const GroupSet*>&    group_sets,
                                               std::shared_ptr<void>                  completion_guard = nullptr);

private:
    enum class StagingPoolState {
        FREE,
        STAGE1_FILLING,
        STAGE1_IN_FLIGHT,
        STAGE2_READY,
        STAGE2_IN_FLIGHT,
    };

    struct PipelineBatchState;
    struct PipelineSlice;

    std::optional<size_t> acquireStagingPool(std::chrono::milliseconds timeout);
    void                  setStagingPoolState(size_t pool_index, StagingPoolState state);
    void                  releaseStagingPool(size_t pool_index);

    DeviceHostTransferExecutor&                              device_host_executor_;
    HostDiskTransferExecutor&                                host_disk_executor_;
    std::array<std::unique_ptr<HostStagingBlockPool>, 2>     staging_pools_;
    size_t                                                   active_staging_pool_count_{0};
    size_t                                                   staging_pool_capacity_{0};
    size_t                                                   next_staging_pool_index_{0};
    std::array<StagingPoolState, 2>                          staging_pool_states_;
    std::mutex                                               staging_state_mutex_;
    std::condition_variable                                  staging_state_cv_;
    std::unique_ptr<BlockTreeTaskPool>                       staging_to_device_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>                       disk_to_staging_task_pool_;
};

}  // namespace rtp_llm
