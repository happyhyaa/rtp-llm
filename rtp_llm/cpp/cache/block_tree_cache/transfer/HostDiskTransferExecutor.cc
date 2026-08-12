#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"

#include <cstring>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

const char* HostDiskTransferExecutor::blockIOStatusName(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return "OK";
        case BlockIOStatus::INVALID_BLOCK:
            return "INVALID_BLOCK";
        case BlockIOStatus::INVALID_SIZE:
            return "INVALID_SIZE";
        case BlockIOStatus::ALIGNMENT_ERROR:
            return "ALIGNMENT_ERROR";
        case BlockIOStatus::IO_ERROR:
            return "IO_ERROR";
        case BlockIOStatus::PARTIAL_FAILURE:
            return "PARTIAL_FAILURE";
    }
    return "UNKNOWN";
}

TransferStatus HostDiskTransferExecutor::blockIOStatusToTransferStatus(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return TransferStatus::OK;
        case BlockIOStatus::INVALID_BLOCK:
        case BlockIOStatus::INVALID_SIZE:
        case BlockIOStatus::ALIGNMENT_ERROR:
            return TransferStatus::INVALID_ARGS;
        case BlockIOStatus::IO_ERROR:
        case BlockIOStatus::PARTIAL_FAILURE:
            return TransferStatus::DISK_IO_ERROR;
    }
    return TransferStatus::DISK_IO_ERROR;
}

TransferStatus HostDiskTransferExecutor::execute(HostBufferView            host,
                                                 const TransferDescriptor& desc,
                                                 const GroupSet&           group_set) const {
    const bool              write_to_disk = desc.target_tier == Tier::DISK;
    const BlockIdxType      disk_block    = desc.singleBlockAt(Tier::DISK);
    BlockTreeDiskBlockPool& disk_pool     = *group_set.diskPool();
    const size_t            payload       = group_set.payloadBytes();
    const size_t            disk_stride   = disk_pool.strideBytes();
    if (!isValidHostBufferView(host, payload, disk_stride)) {
        RTP_LLM_LOG_WARNING("invalid host buffer for %s, disk=%d payload=%zu stride=%zu capacity=%zu",
                            write_to_disk ? "host->disk" : "disk->host",
                            disk_block,
                            payload,
                            disk_stride,
                            host.capacity_bytes);
        return TransferStatus::DISK_IO_ERROR;
    }
    if (write_to_disk && disk_stride > payload) {
        std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
    }
    const BlockIOStatus status = write_to_disk ? disk_pool.write(disk_block, host.base, disk_stride) :
                                                 disk_pool.read(disk_block, host.base, disk_stride);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "%s failed, disk=%d, status=%s", write_to_disk ? "write" : "read", disk_block, blockIOStatusName(status));
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

TransferStatus HostDiskTransferExecutor::execute(const std::vector<HostBufferView>&     hosts,
                                                 const std::vector<TransferDescriptor>& descriptors,
                                                 const std::vector<const GroupSet*>&    group_sets) const {
    if (descriptors.empty() || hosts.size() != descriptors.size() || group_sets.size() != descriptors.size()) {
        return TransferStatus::INVALID_ARGS;
    }

    const Tier source_tier   = descriptors.front().source_tier;
    const Tier target_tier   = descriptors.front().target_tier;
    const bool write_to_disk = source_tier == Tier::HOST && target_tier == Tier::DISK;
    if (!write_to_disk && !(source_tier == Tier::DISK && target_tier == Tier::HOST)) {
        return TransferStatus::INVALID_ARGS;
    }
    if (group_sets.front() == nullptr || group_sets.front()->diskPool() == nullptr) {
        return TransferStatus::INVALID_ARGS;
    }

    BlockTreeDiskBlockPool* disk_pool = group_sets.front()->diskPool().get();
    BlockIdList              disk_blocks;
    std::vector<const void*> sources;
    std::vector<void*>       destinations;
    disk_blocks.reserve(descriptors.size());
    sources.reserve(descriptors.size());
    destinations.reserve(descriptors.size());

    for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
        const auto& descriptor = descriptors[descriptor_index];
        const auto* group_set  = group_sets[descriptor_index];
        const auto  host       = hosts[descriptor_index];
        if (group_set == nullptr || descriptor.source_tier != source_tier || descriptor.target_tier != target_tier
            || group_set->diskPool().get() != disk_pool) {
            return TransferStatus::INVALID_ARGS;
        }
        const BlockIdxType disk_block  = descriptor.singleBlockAt(Tier::DISK);
        const size_t       payload     = group_set->payloadBytes();
        const size_t       disk_stride = disk_pool->strideBytes();
        if (!isValidHostBufferView(host, payload, disk_stride)) {
            RTP_LLM_LOG_WARNING("invalid host buffer for %s batch descriptor=%zu disk=%d payload=%zu stride=%zu "
                                "capacity=%zu",
                                write_to_disk ? "host->disk" : "disk->host",
                                descriptor_index,
                                disk_block,
                                payload,
                                disk_stride,
                                host.capacity_bytes);
            return TransferStatus::DISK_IO_ERROR;
        }
        if (write_to_disk && disk_stride > payload) {
            std::memset(static_cast<uint8_t*>(host.base) + payload, 0, disk_stride - payload);
        }
        disk_blocks.push_back(disk_block);
        if (write_to_disk) {
            sources.push_back(host.base);
        } else {
            destinations.push_back(host.base);
        }
    }

    const BlockIOStatus status = write_to_disk ? disk_pool->write(disk_blocks, sources, disk_pool->strideBytes()) :
                                                 disk_pool->read(disk_blocks, destinations, disk_pool->strideBytes());
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING(
            "batch %s failed, descriptors=%zu, status=%s",
            write_to_disk ? "write" : "read",
            descriptors.size(),
            blockIOStatusName(status));
        for (size_t descriptor_index = 0; descriptor_index < descriptors.size(); ++descriptor_index) {
            RTP_LLM_LOG_WARNING("batch %s failure candidate descriptor=%zu %s",
                                write_to_disk ? "write" : "read",
                                descriptor_index,
                                descriptors[descriptor_index].debugString().c_str());
        }
        return blockIOStatusToTransferStatus(status);
    }
    return TransferStatus::OK;
}

}  // namespace rtp_llm
