#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

DeviceHostTransferExecutor::DeviceHostTransferExecutor(DeviceHostCopyOptions options): options_(std::move(options)) {
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
}

TransferStatus DeviceHostTransferExecutor::execute(HostBufferView            host,
                                                   const TransferDescriptor& desc,
                                                   const GroupSet&           group_set) {
    auto [status, plans] = generatePlan(desc, group_set, host);
    if (status != TransferStatus::OK) {
        return status;
    }
    for (const auto& plan : plans) {
        bool handled = false;
        for (auto& strategy : strategies_) {
            auto result = strategy->tryExecute(plan, options_);
            if (result.status == StrategyStatus::DONE) {
                handled = true;
                break;
            }
            if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
        }
        if (!handled) {
            RTP_LLM_LOG_WARNING("no strategy handled copy plan group_set=%zu", plan.group_set_id);
            return TransferStatus::DEVICE_IO_ERROR;
        }
    }
    return TransferStatus::OK;
}

TransferStatus DeviceHostTransferExecutor::execute(const std::vector<HostBufferView>&     hosts,
                                                   const std::vector<TransferDescriptor>& descriptors,
                                                   const std::vector<const GroupSet*>&    group_sets) {
    if (descriptors.empty() || hosts.size() != descriptors.size() || group_sets.size() != descriptors.size()) {
        return TransferStatus::INVALID_ARGS;
    }

    const Tier source_tier = descriptors.front().source_tier;
    const Tier target_tier = descriptors.front().target_tier;
    if (!((source_tier == Tier::DEVICE && target_tier == Tier::HOST)
          || (source_tier == Tier::HOST && target_tier == Tier::DEVICE))) {
        return TransferStatus::INVALID_ARGS;
    }

    std::map<int, DeviceHostCopyPlan> plans_by_device;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto& descriptor = descriptors[descriptor_index];
        if (descriptor.source_tier != source_tier || descriptor.target_tier != target_tier
            || group_sets[descriptor_index] == nullptr) {
            return TransferStatus::INVALID_ARGS;
        }
        auto [status, descriptor_plans] =
            generatePlan(descriptor, *group_sets[descriptor_index], hosts[descriptor_index]);
        if (status != TransferStatus::OK) {
            return status;
        }
        for (auto& descriptor_plan : descriptor_plans) {
            const int device_index = descriptor_plan.copy_tiles.front().device_index;
            auto&     batch_plan   = plans_by_device[device_index];
            if (batch_plan.copy_tiles.empty()) {
                batch_plan.device_to_host = descriptor_plan.device_to_host;
                batch_plan.group_set_id   = descriptor_plan.group_set_id;
                batch_plan.host           = descriptor_plan.host;
            }
            batch_plan.copy_tiles.insert(batch_plan.copy_tiles.end(),
                                         descriptor_plan.copy_tiles.begin(),
                                         descriptor_plan.copy_tiles.end());
        }
    }

    for (const auto& [_, plan] : plans_by_device) {
        bool handled = false;
        for (auto& strategy : strategies_) {
            auto result = strategy->tryExecute(plan, options_);
            if (result.status == StrategyStatus::DONE) {
                handled = true;
                break;
            }
            if (result.status == StrategyStatus::FAILED) {
                RTP_LLM_LOG_WARNING("%s batch failed status=%d descriptors=%zu",
                                    target_tier == Tier::HOST ? "D2H" : "H2D",
                                    static_cast<int>(result.copy_status),
                                    descriptors.size());
                return result.copy_status;
            }
        }
        if (!handled) {
            RTP_LLM_LOG_WARNING("no strategy handled %s batch descriptors=%zu",
                                target_tier == Tier::HOST ? "D2H" : "H2D",
                                descriptors.size());
            return TransferStatus::DEVICE_IO_ERROR;
        }
    }
    return TransferStatus::OK;
}

std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
DeviceHostTransferExecutor::generatePlan(const TransferDescriptor& desc,
                                         const GroupSet&           group_set,
                                         HostBufferView            host) const {
    const size_t required_host_bytes = group_set.payloadBytes();
    if (!isValidHostBufferView(host, required_host_bytes, required_host_bytes)) {
        RTP_LLM_LOG_WARNING("invalid host buffer group=%zu payload=%zu capacity=%zu required=%zu",
                            desc.group_set_id,
                            host.payload_bytes,
                            host.capacity_bytes,
                            required_host_bytes);
        const auto status = host.base == nullptr ? TransferStatus::DEVICE_IO_ERROR : TransferStatus::INVALID_ARGS;
        return {status, {}};
    }

    const bool                       device_to_host = desc.target_tier != Tier::DEVICE;
    const std::vector<BlockIdxType>& device_blocks = desc.blocksAt(Tier::DEVICE);
    const auto&                       device_pools  = group_set.devicePools();
    std::map<int, DeviceHostCopyPlan> plans_by_device;

    size_t host_offset = 0;
    for (size_t member_group_id = 0; member_group_id < group_set.groupIds().size(); ++member_group_id) {
        const auto& group_base        = group_set.groupAt(member_group_id);
        auto&       device_pool       = *device_pools[member_group_id];

        for (size_t local_layer_index = 0; local_layer_index < group_base.layer_ids.size(); ++local_layer_index) {
            const size_t kv_bytes        = group_base.kv_block_stride_bytes;
            const size_t scale_bytes     = group_base.kv_scale_stride_bytes;
            const size_t layer_bytes     = kv_bytes + scale_bytes;
            auto*        layer_host_addr = static_cast<uint8_t*>(host.base) + host_offset;

            const auto buffers =
                device_pool.convertIndexToBuffer(static_cast<int>(local_layer_index), device_blocks[member_group_id]);
            const auto append_tile = [&](size_t buffer_index, size_t logical_bytes, size_t layer_offset) {
                if (logical_bytes == 0) {
                    return true;
                }
                if (buffer_index >= buffers.size() || buffers[buffer_index].addr == nullptr
                    || buffers[buffer_index].size_bytes < logical_bytes) {
                    RTP_LLM_LOG_WARNING("physical buffer cannot cover logical payload group_set_id=%zu "
                                        "member_group_id=%zu group_id=%zu local_layer=%zu buffer=%zu physical=%zu "
                                        "logical=%zu block=%d",
                                        desc.group_set_id,
                                        member_group_id,
                                        group_set.groupIds()[member_group_id],
                                        local_layer_index,
                                        buffer_index,
                                        buffer_index < buffers.size() ? buffers[buffer_index].size_bytes : 0,
                                        logical_bytes,
                                        device_blocks[member_group_id]);
                    return false;
                }
                auto& plan = plans_by_device[device_pool.deviceIndex()];
                if (plan.copy_tiles.empty()) {
                    plan.device_to_host     = device_to_host;
                    plan.group_set_id       = desc.group_set_id;
                    plan.host.base          = host.base;
                    plan.host.payload_bytes = required_host_bytes;
                }
                DeviceHostCopyTile tile;
                tile.host_addr         = layer_host_addr + layer_offset;
                tile.device_addr       = buffers[buffer_index].addr;
                tile.host_offset       = host_offset + layer_offset;
                tile.bytes             = logical_bytes;
                tile.device_index      = device_pool.deviceIndex();
                tile.member_group_id   = member_group_id;
                tile.local_layer_index = local_layer_index;
                plan.copy_tiles.push_back(tile);
                return true;
            };
            if (!append_tile(0, kv_bytes, 0) || !append_tile(1, scale_bytes, kv_bytes)) {
                return {TransferStatus::INVALID_ARGS, {}};
            }
            host_offset += layer_bytes;
        }
    }

    if (host_offset != required_host_bytes) {
        RTP_LLM_LOG_WARNING("logical payload drift group_set=%zu lowered=%zu expected=%zu",
                            desc.group_set_id,
                            host_offset,
                            required_host_bytes);
        return {TransferStatus::INVALID_ARGS, {}};
    }

    if (plans_by_device.empty()) {
        RTP_LLM_LOG_WARNING("%s copy plan generated no copy tile group_set=%zu",
                            device_to_host ? "D2H" : "H2D",
                            desc.group_set_id);
        return {TransferStatus::INVALID_ARGS, {}};
    }

    std::vector<DeviceHostCopyPlan> plans;
    plans.reserve(plans_by_device.size());
    for (auto& [_, plan] : plans_by_device) {
        plans.push_back(std::move(plan));
    }
    return {TransferStatus::OK, std::move(plans)};
}

}  // namespace rtp_llm
