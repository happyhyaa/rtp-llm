#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace rtp_llm {

StrategyResult Cuda3DBatchDeviceHostCopyStrategy::tryExecute(const std::vector<HostBufferView>&     hosts,
                                                             const std::vector<TransferDescriptor>& descriptors,
                                                             const std::vector<const GroupSet*>&    group_sets,
                                                             const DeviceHostCopyExecutionContext&  context) {
#if USING_CUDA && CUDART_VERSION >= 12080
    int driver_version  = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess || cudaRuntimeGetVersion(&runtime_version) != cudaSuccess
        || driver_version < (CUDART_VERSION >= 13000 ? 13000 : 12080) || runtime_version < 12080
        || (runtime_version >= 13000) != (CUDART_VERSION >= 13000)) {
        return StrategyResult::notApplicable();
    }

    size_t op_count = 0;
    for (const auto* group : group_sets) {
        if (group->copy3DTemplates().empty()) {
            return StrategyResult::notApplicable();
        }
        op_count += group->copy3DTemplates().size();
    }

    std::vector<cudaMemcpy3DBatchOp> ops;
    ops.reserve(op_count);
    const bool d2h = context.direction == DeviceHostCopyDirection::D2H;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& group  = *group_sets[i];
        const auto& blocks = descriptors[i].blocksAt(Tier::DEVICE);
        for (const auto& t : group.copy3DTemplates()) {
            const auto  block = blocks[t.member_index];
            auto* host   = static_cast<uint8_t*>(hosts[i].base) + t.host_offset;
            auto* device = static_cast<uint8_t*>(t.device_base) + static_cast<size_t>(block) * t.width_bytes;
            cudaMemcpy3DBatchOp op{};
            op.src.type               = cudaMemcpyOperandTypePointer;
            op.dst.type               = cudaMemcpyOperandTypePointer;
            op.src.op.ptr.ptr         = d2h ? device : host;
            op.dst.op.ptr.ptr         = d2h ? host : device;
            op.src.op.ptr.rowLength   = d2h ? t.device_pitch : t.width_bytes;
            op.dst.op.ptr.rowLength   = d2h ? t.width_bytes : t.device_pitch;
            op.src.op.ptr.layerHeight = t.layer_count;
            op.dst.op.ptr.layerHeight = t.layer_count;
            op.extent                 = {t.width_bytes, t.layer_count, 1};
            op.srcAccessOrder         = cudaMemcpySrcAccessOrderStream;
            ops.push_back(op);
        }
    }

    c10::cuda::CUDAGuard guard(context.deviceIndex());
    auto                 stream = reinterpret_cast<cudaStream_t>(context.stream());
#if CUDART_VERSION >= 13000
    const auto submit = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
#else
    size_t     fail_index = 0;
    const auto submit     = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), &fail_index, 0, stream);
#endif
    // Drain even a failed submission: a prefix may already be in flight.
    const auto completion = cudaStreamSynchronize(stream);
    RTP_LLM_CHECK_WITH_INFO(completion == cudaSuccess,
                            "3D batch completion failed: device=%d error=%s",
                            context.deviceIndex(),
                            cudaGetErrorString(completion));
    if (submit != cudaSuccess) {
        RTP_LLM_LOG_WARNING("3D batch submission failed: device=%d ops=%zu error=%s",
                            context.deviceIndex(),
                            ops.size(),
                            cudaGetErrorString(submit));
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }
    return StrategyResult::done();
#else
    return StrategyResult::notApplicable();
#endif
}

}  // namespace rtp_llm
