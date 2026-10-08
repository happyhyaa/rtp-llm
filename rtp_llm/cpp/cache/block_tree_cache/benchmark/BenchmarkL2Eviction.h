#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace rtp_llm::benchmark {

// One reusable scratch buffer; never aliases transfer source/target memory.
class BenchmarkL2Eviction {
public:
    explicit BenchmarkL2Eviction(bool enabled);
    ~BenchmarkL2Eviction();
    BenchmarkL2Eviction(const BenchmarkL2Eviction&)            = delete;
    BenchmarkL2Eviction& operator=(const BenchmarkL2Eviction&) = delete;

    void   evict();
    size_t l2Bytes() const {
        return l2_bytes_;
    }
    size_t evictionBytes() const {
        return eviction_bytes_;
    }

private:
    size_t       l2_bytes_{0};
    size_t       eviction_bytes_{0};
    void*        buffer_{nullptr};
    cudaStream_t stream_{nullptr};
};

}  // namespace rtp_llm::benchmark
