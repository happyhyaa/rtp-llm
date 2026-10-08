#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace rtp_llm::crc_copy_benchmark {

// Test-local inputs retain the reference benchmark's prebuilt-plan boundary.
struct BenchmarkTile {
    void*  device;
    size_t offset;
    size_t bytes;
};
struct BenchmarkCopyItem {
    void*                      host;
    size_t                     payload_bytes;
    size_t                     capacity_bytes;
    std::vector<BenchmarkTile> tiles;
};

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

// Resolve model specs through main's CacheConfigCreator and physical pool
// helper. These functions describe one local, prefix-reusable backing.
const Layout& deepSeekV4FlashLayout(bool full);
size_t        maximumLayoutTiles();
size_t        maximumLayoutPayload();

// Owns the actual production DeviceHostCopyPlan values without exposing their
// transitive framework/Torch dependencies to the CUDA translation unit.
class FrameworkCopyPlan {
public:
    FrameworkCopyPlan(const std::vector<BenchmarkCopyItem>& items, const Layout& layout);
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Real production pools/templates for the separately measured 3D path.
// Preparation and oracle copies are explicitly outside the measured call.
class Framework3DCopies {
public:
    Framework3DCopies(const Layout& layout, size_t blocks, int device);
    ~Framework3DCopies();
    size_t    addPlan(const std::vector<BenchmarkCopyItem>& items, const std::vector<int>& blocks);
    uintptr_t touchMetadata(size_t plan) const;
    void      copy(size_t plan, bool store);
    void      copyOracle(void* contiguous, bool into_pools);
    void      poison(unsigned char value);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm::crc_copy_benchmark
