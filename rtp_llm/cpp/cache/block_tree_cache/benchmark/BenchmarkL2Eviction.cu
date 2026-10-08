#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkL2Eviction.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm::benchmark {
namespace {
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("L2 eviction: ") + cudaGetErrorString(error));
}

// Volatile read/write touches every word, rather than relying on allocation
// or a memset implementation to populate L2. This is best-effort eviction,
// not a promise that all subsequent accesses miss L2.
__global__ void sweepL2(volatile uint32_t* buffer, size_t words) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        buffer[i] = buffer[i] + 1u;
    }
}
}  // namespace

BenchmarkL2Eviction::BenchmarkL2Eviction(bool enabled) {
    if (!enabled)
        return;
    int device   = 0;
    int l2_bytes = 0;
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device));
    if (l2_bytes <= 0 || static_cast<size_t>(l2_bytes) > std::numeric_limits<size_t>::max() / 8)
        throw std::runtime_error("L2 eviction: invalid GPU L2 size");
    l2_bytes_       = static_cast<size_t>(l2_bytes);
    eviction_bytes_ = 8 * l2_bytes_;
    try {
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        checkCuda(cudaMalloc(&buffer_, eviction_bytes_));
        checkCuda(cudaMemsetAsync(buffer_, 0, eviction_bytes_, stream_));
        checkCuda(cudaStreamSynchronize(stream_));
    } catch (...) {
        if (buffer_)
            cudaFree(buffer_);
        if (stream_)
            cudaStreamDestroy(stream_);
        throw;
    }
}

BenchmarkL2Eviction::~BenchmarkL2Eviction() {
    if (buffer_)
        cudaFree(buffer_);
    if (stream_)
        cudaStreamDestroy(stream_);
}

void BenchmarkL2Eviction::evict() {
    if (!buffer_)
        return;
    // All earlier transfers must finish before disturbing the cache.
    checkCuda(cudaDeviceSynchronize());
    sweepL2<<<1024, 256, 0, stream_>>>(static_cast<volatile uint32_t*>(buffer_), eviction_bytes_ / sizeof(uint32_t));
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream_));
}
}  // namespace rtp_llm::benchmark
