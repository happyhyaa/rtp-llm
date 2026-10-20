#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyGeometry.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

std::string_view configurableStrategyName(const DeviceHostCopyStrategy& strategy) {
    if (dynamic_cast<const Cuda3DBatchDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "cuda_3d_batch";
    }
    if (dynamic_cast<const CudaBatchDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "cuda_batch";
    }
    if (dynamic_cast<const StagedSmDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "sm";
    }
    if (dynamic_cast<const GenericMultiCopyDeviceHostCopyStrategy*>(&strategy) != nullptr) {
        return "generic";
    }
    return {};
}

}  // namespace

DeviceHostTransferExecutor::DeviceHostTransferExecutor(BlockTreeTaskPool&    transfer_task_pool,
                                                       size_t                max_descriptors_per_batch,
                                                       DeviceHostCopyOptions options,
                                                       std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                                                       std::shared_ptr<CrcTransferService>            crc_service,
                                                       BlockTreeTaskPool*                             load_task_pool,
                                                       const std::vector<GroupSetPtr>&                group_sets):
    TransferExecutor(transfer_task_pool, max_descriptors_per_batch, std::move(metrics_reporter), load_task_pool),
    crc_service_(std::move(crc_service)),
    options_(std::move(options)) {
    strategies_.push_back(
        std::make_unique<Cuda3DBatchDeviceHostCopyStrategy>(group_sets, options_.cuda_3d_batch_copy_enabled));
    strategies_.push_back(std::make_unique<CudaBatchDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<StagedSmDeviceHostCopyStrategy>());
    strategies_.push_back(std::make_unique<GenericMultiCopyDeviceHostCopyStrategy>());
    if (const char* selected = std::getenv("BLOCK_TREE_DEVICE_HOST_COPY_PRIORITY"); selected && *selected) {
        const auto found = std::find_if(strategies_.begin(), strategies_.end(), [selected](const auto& strategy) {
            return configurableStrategyName(*strategy) == selected;
        });
        if (found != strategies_.end()) {
            std::rotate(strategies_.begin(), found, found + 1);
        }
    }
}

TransferResult DeviceHostTransferExecutor::executeBatch(const std::vector<HostBufferView>&     hosts,
                                                        const std::vector<TransferDescriptor>& descriptors,
                                                        const std::vector<const GroupSet*>&    group_sets) {
    if (descriptors.empty() || hosts.size() != descriptors.size() || group_sets.size() != descriptors.size()) {
        return TransferStatus::INVALID_ARGS;
    }
    const Tier source = descriptors.front().source_tier;
    const Tier target = descriptors.front().target_tier;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (descriptors[i].source_tier != source || descriptors[i].target_tier != target || group_sets[i] == nullptr
            || descriptors[i].blocksAt(Tier::DEVICE).size() != group_sets[i]->devicePools().size()) {
            return TransferStatus::INVALID_ARGS;
        }
    }
    const bool enable_crc = group_sets.front()->crcEnabled();
    if (std::any_of(group_sets.begin(), group_sets.end(), [enable_crc](const GroupSet* group) {
            return group->crcEnabled() != enable_crc;
        })) {
        return TransferStatus::INVALID_ARGS;
    }
    if (enable_crc) {
        return crc_service_ ? crc_service_->copy(hosts, descriptors, group_sets) : TransferStatus::INVALID_ARGS;
    }
    std::vector<DeviceHostCopyPlan> plans;
    std::vector<bool>               handled;
    bool                            planned     = false;
    bool                            any_handled = false;
    for (auto& strategy : strategies_) {
        if (auto* copy3d = dynamic_cast<Cuda3DBatchDeviceHostCopyStrategy*>(strategy.get())) {
            // A raw-batch copy must not repeat any plan already completed by an earlier strategy.
            if (!options_.cuda_3d_batch_copy_enabled || any_handled) {
                continue;
            }
            const auto result = copy3d->tryExecute(hosts, descriptors, group_sets);
            if (result.status == StrategyStatus::DONE) {
                return TransferStatus::OK;
            }
            if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
            continue;
        }
        if (!planned) {
            auto generated = generatePlan(hosts, descriptors, group_sets);
            if (generated.first != TransferStatus::OK) {
                return generated.first;
            }
            plans = std::move(generated.second);
            handled.assign(plans.size(), false);
            planned = true;
        }
        for (size_t index = 0; index < plans.size(); ++index) {
            if (handled[index]) {
                continue;
            }
            const auto result = strategy->tryExecute(plans[index], options_);
            if (result.status == StrategyStatus::DONE) {
                handled[index] = true;
                any_handled    = true;
            } else if (result.status == StrategyStatus::FAILED) {
                return result.copy_status;
            }
        }
        if (std::all_of(handled.begin(), handled.end(), [](bool done) { return done; })) {
            return TransferStatus::OK;
        }
    }
    if (planned) {
        for (size_t index = 0; index < plans.size(); ++index) {
            if (!handled[index]) {
                RTP_LLM_LOG_WARNING("no strategy handled copy plan group_set=%zu", plans[index].group_set_id);
            }
        }
    }
    return TransferStatus::DEVICE_IO_ERROR;
}

std::pair<TransferStatus, std::vector<DeviceHostCopyPlan>>
DeviceHostTransferExecutor::generatePlan(const std::vector<HostBufferView>&     hosts,
                                         const std::vector<TransferDescriptor>& descriptors,
                                         const std::vector<const GroupSet*>&    group_sets) const {
    const bool                        device_to_host = descriptors.front().target_tier != Tier::DEVICE;
    std::map<int, DeviceHostCopyPlan> plans_by_device;
    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto&  descriptor          = descriptors[descriptor_index];
        const auto&  group_set           = *group_sets[descriptor_index];
        const auto&  host                = hosts[descriptor_index];
        const size_t required_host_bytes = group_set.payloadBytes();
        if (!isValidHostBufferView(host, required_host_bytes, required_host_bytes)) {
            RTP_LLM_LOG_WARNING(
                "invalid device-host batch item index=%zu group=%zu", descriptor_index, descriptor.group_set_id);
            return {host.base == nullptr ? TransferStatus::DEVICE_IO_ERROR : TransferStatus::INVALID_ARGS, {}};
        }

        const std::vector<BlockIdxType>&        device_blocks = descriptor.blocksAt(Tier::DEVICE);
        std::vector<DeviceHostCopyTileGeometry> tiles;
        try {
            tiles = getDeviceHostCopyTiles(group_set, device_blocks);
        } catch (const std::invalid_argument&) {
            return {TransferStatus::INVALID_ARGS, {}};
        }
        for (const auto& tile : tiles) {
            const int device = group_set.devicePools()[tile.member_group_id]->deviceIndex();
            auto&     plan   = plans_by_device[device];
            if (plan.copy_tiles.empty()) {
                plan.device_to_host = device_to_host;
                plan.group_set_id   = descriptor.group_set_id;
                plan.host           = host;
            }
            plan.copy_tiles.push_back(DeviceHostCopyTile{static_cast<uint8_t*>(host.base) + tile.host_offset,
                                                         tile.buffer.addr,
                                                         tile.buffer.size_bytes,
                                                         device,
                                                         tile.member_group_id,
                                                         tile.local_layer_index});
        }
    }

    if (plans_by_device.empty()) {
        RTP_LLM_LOG_WARNING("%s batch generated no copy tile", device_to_host ? "D2H" : "H2D");
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
