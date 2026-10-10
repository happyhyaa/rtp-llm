#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm::crc_copy_benchmark {

struct PhysicalCopyTile {
    std::string tag;
    size_t      member;
    size_t      local_layer;
    int         model_layer;
    size_t      bytes;
    // Pool/layer base for capacity N is this coefficient times N; each
    // selected block contributes block_index * bytes within that layer.
    size_t offset_per_pool_block;
};

struct Layout {
    std::string                   name;
    std::vector<size_t>           sizes;
    std::vector<PhysicalCopyTile> geometry;
    size_t                        payload() const;
    std::string                   geometryJson() const;
};

struct Dsv4BenchmarkModelInfo {
    std::string name;
    uint32_t    numLayers;
    uint32_t    hiddenSize;
    uint32_t    headNum;
    uint32_t    indexerTopk;
    uint32_t    oGroups;
    uint32_t    tpSize;
    uint32_t    cpSize;
    bool        kvCacheSharded;
    std::string cpMode;
};

// Resolve model specs through main's CacheConfigCreator and physical pool
// helper. These functions describe one local, prefix-reusable backing.
Dsv4BenchmarkModelInfo dsv4BenchmarkModelInfo(const std::string& model);
Layout                 deepSeekV4Layout(const std::string& model,
                                         uint32_t           logicalTokensPerBlock,
                                         uint32_t           kernelTokensPerBlock,
                                         bool               full);

// Owns the actual production DeviceHostCopyPlan values without exposing their
// transitive framework/Torch dependencies to the CUDA translation unit.
class FrameworkCopyPlan {
public:
    FrameworkCopyPlan(const std::vector<CrcCopyItem>& items, const Layout& layout);
    ~FrameworkCopyPlan();
    FrameworkCopyPlan(const FrameworkCopyPlan&)            = delete;
    FrameworkCopyPlan& operator=(const FrameworkCopyPlan&) = delete;

    uintptr_t touchMetadata() const;

private:
    friend class FrameworkCopies;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class FrameworkCopies {
public:
    explicit FrameworkCopies(int device);
    ~FrameworkCopies();
    FrameworkCopies(const FrameworkCopies&)            = delete;
    FrameworkCopies& operator=(const FrameworkCopies&) = delete;

    void copyBatch(const FrameworkCopyPlan& plan, bool store);
    void copyStaged(const FrameworkCopyPlan& plan, bool store);

    // Production's thread-local stream is private. The experimental 3D path
    // uses a separate stream from the same nondefault Torch stream pool.
    cudaStream_t copy3dStream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm::crc_copy_benchmark
