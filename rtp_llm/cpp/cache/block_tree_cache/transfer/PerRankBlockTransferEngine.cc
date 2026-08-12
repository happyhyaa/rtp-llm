#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kTransferQueueWorkerCount = 2;
constexpr size_t kTransferQueueCapacity    = 1000;

struct EndpointKey {
    Tier         tier{Tier::NONE};
    const void*  pool{nullptr};
    BlockIdxType block{NULL_BLOCK_IDX};

    bool operator==(const EndpointKey& other) const {
        return tier == other.tier && pool == other.pool && block == other.block;
    }
};

struct EndpointKeyHash {
    size_t operator()(const EndpointKey& key) const {
        size_t result = std::hash<int>{}(static_cast<int>(key.tier));
        result ^= std::hash<const void*>{}(key.pool) << 1;
        result ^= std::hash<BlockIdxType>{}(key.block) << 2;
        return result;
    }
};

struct EndpointAccessState {
    std::optional<size_t> first_reader;
    std::optional<size_t> writer;
};

struct EndpointAccess {
    EndpointKey key;
    bool        write{false};
    Tier        source_tier{Tier::NONE};
    Tier        target_tier{Tier::NONE};
    size_t      descriptor_index{0};
};

const void* endpointPool(const GroupSet& group_set, Tier tier, size_t block_index) {
    if (tier == Tier::DEVICE) {
        return group_set.devicePools()[block_index].get();
    }
    if (tier == Tier::HOST) {
        return group_set.hostPool().get();
    }
    if (tier == Tier::DISK) {
        return group_set.diskPool().get();
    }
    return nullptr;
}

std::vector<EndpointAccess> buildEndpointAccesses(const std::vector<TransferDescriptor>& descriptors,
                                                  const std::vector<const GroupSet*>&    group_sets) {
    std::vector<EndpointAccess> accesses;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const TransferDescriptor& descriptor = descriptors[descriptor_index];
        const GroupSet&           group_set  = *group_sets[descriptor_index];
        const auto                append     = [&](Tier tier, const std::vector<BlockIdxType>& blocks, bool write) {
            for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
                accesses.push_back(
                    EndpointAccess{{tier, endpointPool(group_set, tier, block_index), blocks[block_index]},
                                   write,
                                   descriptor.source_tier,
                                   descriptor.target_tier,
                                   descriptor_index});
            }
        };
        append(descriptor.source_tier, descriptor.source_blocks, false);
        append(descriptor.target_tier, descriptor.target_blocks, true);
    }
    return accesses;
}

ErrorInfo validateBatchEndpointAccesses(const std::vector<EndpointAccess>& endpoint_accesses) {
    std::unordered_map<EndpointKey, EndpointAccessState, EndpointKeyHash> accesses;
    for (const EndpointAccess& access : endpoint_accesses) {
        auto&      state = accesses[access.key];
        const auto conflict_index =
            access.write ? (state.writer.has_value() ? state.writer : state.first_reader) : state.writer;
        if (conflict_index.has_value()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS,
                             "batch endpoint conflict: descriptor_index=" + std::to_string(access.descriptor_index)
                                 + ", conflicts_with_descriptor_index=" + std::to_string(*conflict_index)
                                 + ", tier=" + tierName(access.key.tier) + ", direction=" + tierName(access.source_tier)
                                 + "->" + tierName(access.target_tier) + ", block=" + std::to_string(access.key.block));
        }
        if (access.write) {
            state.writer = access.descriptor_index;
        } else if (!state.first_reader.has_value()) {
            state.first_reader = access.descriptor_index;
        }
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer queue or staging resource exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

class TransferEndpointRegistry: public std::enable_shared_from_this<TransferEndpointRegistry> {
public:
    struct ReservationResult {
        std::shared_ptr<void> guard;
        ErrorInfo             error;
    };

    ReservationResult reserve(const std::vector<EndpointAccess>& accesses) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const EndpointAccess& access : accesses) {
            const auto it = in_flight_.find(access.key);
            if (it == in_flight_.end()) {
                continue;
            }
            if (it->second.writer || (access.write && it->second.readers > 0)) {
                return {nullptr,
                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                  "RESOURCE_EXHAUSTED: transfer endpoint conflict: descriptor_index="
                                      + std::to_string(access.descriptor_index) + ", tier=" + tierName(access.key.tier)
                                      + ", direction=" + tierName(access.source_tier) + "->"
                                      + tierName(access.target_tier) + ", block=" + std::to_string(access.key.block))};
            }
        }
        for (const EndpointAccess& access : accesses) {
            auto& state = in_flight_[access.key];
            if (access.write) {
                state.writer = true;
            } else {
                ++state.readers;
            }
        }
        return {std::make_shared<Reservation>(shared_from_this(), accesses), ErrorInfo::OkStatus()};
    }

private:
    struct InFlightState {
        size_t readers{0};
        bool   writer{false};
    };

    class Reservation {
    public:
        Reservation(std::shared_ptr<TransferEndpointRegistry> registry, std::vector<EndpointAccess> accesses):
            registry_(std::move(registry)), accesses_(std::move(accesses)) {}

        ~Reservation() {
            registry_->release(accesses_);
        }

    private:
        std::shared_ptr<TransferEndpointRegistry> registry_;
        std::vector<EndpointAccess>               accesses_;
    };

    void release(const std::vector<EndpointAccess>& accesses) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const EndpointAccess& access : accesses) {
            auto it = in_flight_.find(access.key);
            RTP_LLM_CHECK(it != in_flight_.end());
            if (access.write) {
                RTP_LLM_CHECK(it->second.writer);
                it->second.writer = false;
            } else {
                RTP_LLM_CHECK(it->second.readers > 0);
                --it->second.readers;
            }
            if (!it->second.writer && it->second.readers == 0) {
                in_flight_.erase(it);
            }
        }
    }

    std::mutex                                                      mutex_;
    std::unordered_map<EndpointKey, InFlightState, EndpointKeyHash> in_flight_;
};

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count,
                                                       size_t                   max_descriptors_per_transfer_batch):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()),
    endpoint_registry_(std::make_shared<TransferEndpointRegistry>()),
    max_descriptors_per_transfer_batch_(max_descriptors_per_transfer_batch) {
    RTP_LLM_CHECK_WITH_INFO(max_descriptors_per_transfer_batch_ > 0, "max_descriptors_per_transfer_batch must be > 0");
    device_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferQueueWorkerCount, kTransferQueueCapacity, "BlockD2HTransfer");
    host_to_device_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferQueueWorkerCount, kTransferQueueCapacity, "BlockH2DTransfer");
    host_to_disk_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferQueueWorkerCount, kTransferQueueCapacity, "BlockH2DiskTransfer");
    disk_to_host_task_pool_ =
        std::make_unique<BlockTreeTaskPool>(kTransferQueueWorkerCount, kTransferQueueCapacity, "BlockDisk2HTransfer");
    RTP_LLM_CHECK_WITH_INFO(device_to_host_task_pool_->start(), "failed to start Device->Host transfer queue");
    RTP_LLM_CHECK_WITH_INFO(host_to_device_task_pool_->start(), "failed to start Host->Device transfer queue");
    RTP_LLM_CHECK_WITH_INFO(host_to_disk_task_pool_->start(), "failed to start Host->Disk transfer queue");
    RTP_LLM_CHECK_WITH_INFO(disk_to_host_task_pool_->start(), "failed to start Disk->Host transfer queue");
    const bool any_disk_pool = std::any_of(group_sets_.begin(), group_sets_.end(), [](const GroupSetPtr& group_set) {
        return group_set != nullptr && group_set->diskPool() != nullptr;
    });
    if (any_disk_pool) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(
            *device_host_executor_, *host_disk_executor_, group_sets_, device_disk_staging_block_count);
    }
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() = default;

std::shared_ptr<AsyncContext> PerRankBlockTransferEngine::submit(const std::vector<TransferDescriptor>& descriptors) {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    const Tier source_tier = descriptors.front().source_tier;
    const Tier target_tier = descriptors.front().target_tier;
    const bool device_host = (source_tier == Tier::DEVICE && target_tier == Tier::HOST)
                             || (source_tier == Tier::HOST && target_tier == Tier::DEVICE);
    const bool host_disk = (source_tier == Tier::HOST && target_tier == Tier::DISK)
                           || (source_tier == Tier::DISK && target_tier == Tier::HOST);
    const bool disk_to_device = source_tier == Tier::DISK && target_tier == Tier::DEVICE;
    const bool device_to_disk = source_tier == Tier::DEVICE && target_tier == Tier::DISK;
    if (!device_host && !host_disk && !disk_to_device && !device_to_disk) {
        return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
    }

    std::vector<const GroupSet*> group_sets;
    std::vector<HostBufferView>  hosts;
    group_sets.reserve(descriptors.size());
    hosts.reserve(descriptors.size());

    for (const auto& descriptor : descriptors) {
        if (descriptor.source_tier != source_tier || descriptor.target_tier != target_tier) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
        const GroupSet* group_set = group_sets_[descriptor.group_set_id].get();
        group_sets.push_back(group_set);
        if (!disk_to_device && !device_to_disk) {
            hosts.push_back(resolveHostView(*group_set, descriptor.singleBlockAt(Tier::HOST)));
        }
    }

    if (host_disk) {
        const auto* disk_pool = group_sets.front()->diskPool().get();
        const bool  same_pool = std::all_of(group_sets.begin(), group_sets.end(), [disk_pool](const GroupSet* group) {
            return group->diskPool().get() == disk_pool;
        });
        if (!same_pool) {
            return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(TransferStatus::INVALID_ARGS));
        }
    }
    const auto      endpoint_accesses = buildEndpointAccesses(descriptors, group_sets);
    const ErrorInfo endpoint_error    = validateBatchEndpointAccesses(endpoint_accesses);
    if (!endpoint_error.ok()) {
        RTP_LLM_LOG_WARNING("rejecting transfer batch: %s", endpoint_error.ToString().c_str());
        return std::make_shared<CompletedAsyncContext>(endpoint_error);
    }

    auto reservation = endpoint_registry_->reserve(endpoint_accesses);
    if (!reservation.error.ok()) {
        RTP_LLM_LOG_WARNING("rejecting in-flight transfer batch: %s", reservation.error.ToString().c_str());
        return std::make_shared<CompletedAsyncContext>(reservation.error);
    }

    if (disk_to_device) {
        return device_disk_executor_->diskToDevice(descriptors, group_sets, std::move(reservation.guard));
    }
    if (device_to_disk) {
        for (size_t index = 0; index < descriptors.size(); ++index) {
            const TransferStatus status = device_disk_executor_->execute(descriptors[index], *group_sets[index]);
            if (status != TransferStatus::OK) {
                const ErrorInfo error = transferStatusToErrorInfo(status);
                return std::make_shared<CompletedAsyncContext>(ErrorInfo(
                    error.code(), error.ToString() + ", descriptor_index=" + std::to_string(index)));
            }
        }
        return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    }
    auto  context   = std::make_shared<TransferBatchAsyncContext>(std::move(reservation.guard));
    auto* task_pool = taskPoolForDirection(source_tier, target_tier);
    const bool accepted = task_pool->trySubmit(
        [this, descriptors, group_sets = std::move(group_sets), hosts = std::move(hosts), context]() mutable {
            try {
                for (size_t begin = 0; begin < descriptors.size(); begin += max_descriptors_per_transfer_batch_) {
                    const size_t end = std::min(begin + max_descriptors_per_transfer_batch_, descriptors.size());
                    const std::vector<TransferDescriptor> sub_descriptors(descriptors.begin() + begin,
                                                                          descriptors.begin() + end);
                    const std::vector<const GroupSet*>    sub_group_sets(group_sets.begin() + begin,
                                                                      group_sets.begin() + end);
                    const std::vector<HostBufferView>     sub_hosts(hosts.begin() + begin, hosts.begin() + end);
                    const auto status = executeDirectBatch(sub_descriptors, sub_group_sets, sub_hosts);
                    if (status != TransferStatus::OK) {
                        const ErrorInfo base = transferStatusToErrorInfo(status);
                        context->complete(ErrorInfo(base.code(),
                                                    base.ToString() + ", physical_sub_batch=[" + std::to_string(begin)
                                                        + "," + std::to_string(end) + "), logical_descriptors="
                                                        + std::to_string(descriptors.size())));
                        return;
                    }
                }
                context->complete(ErrorInfo::OkStatus());
            } catch (const std::exception& e) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                            "direct transfer executor exception: " + std::string(e.what())
                                                + ", logical_descriptors=" + std::to_string(descriptors.size())));
            } catch (...) {
                context->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                                            "unknown direct transfer executor exception, logical_descriptors="
                                                + std::to_string(descriptors.size())));
            }
        });
    if (!accepted) {
        context->complete(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "RESOURCE_EXHAUSTED: transfer queue is full or stopped"));
    }
    return context;
}

TransferStatus PerRankBlockTransferEngine::executeDirectBatch(const std::vector<TransferDescriptor>& descriptors,
                                                              const std::vector<const GroupSet*>&    group_sets,
                                                              const std::vector<HostBufferView>&     hosts) {
    if (descriptors.front().source_tier == Tier::DEVICE || descriptors.front().target_tier == Tier::DEVICE) {
        return device_host_executor_->execute(hosts, descriptors, group_sets);
    }
    return host_disk_executor_->execute(hosts, descriptors, group_sets);
}

BlockTreeTaskPool* PerRankBlockTransferEngine::taskPoolForDirection(Tier source_tier, Tier target_tier) const {
    if (source_tier == Tier::DEVICE && target_tier == Tier::HOST) {
        return device_to_host_task_pool_.get();
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DEVICE) {
        return host_to_device_task_pool_.get();
    }
    if (source_tier == Tier::HOST && target_tier == Tier::DISK) {
        return host_to_disk_task_pool_.get();
    }
    if (source_tier == Tier::DISK && target_tier == Tier::HOST) {
        return disk_to_host_task_pool_.get();
    }
    return nullptr;
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
