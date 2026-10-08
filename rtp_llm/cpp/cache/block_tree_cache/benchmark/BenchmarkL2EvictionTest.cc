#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkL2Eviction.h"
#include <array>
#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
TEST(BenchmarkL2EvictionTest, DisabledDoesNotAllocateOrRequireCuda) {
    BenchmarkL2Eviction eviction(false);
    EXPECT_EQ(eviction.evictionBytes(), 0u);
    EXPECT_EQ(eviction.l2Bytes(), 0u);
    EXPECT_NO_THROW(eviction.evict());
}
TEST(BenchmarkL2EvictionTest, ReusesEightTimesL2WithoutTouchingTransferMemory) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int bytes = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, 0), cudaSuccess);
    ASSERT_GT(bytes, 0);
    void* sentinel = nullptr;
    ASSERT_EQ(cudaMalloc(&sentinel, 128), cudaSuccess);
    ASSERT_EQ(cudaMemset(sentinel, 0x5a, 128), cudaSuccess);
    {
        BenchmarkL2Eviction eviction(true);
        EXPECT_EQ(eviction.evictionBytes(), static_cast<size_t>(bytes) * 8);
        EXPECT_NO_THROW(eviction.evict());
        EXPECT_NO_THROW(eviction.evict());
        std::array<unsigned char, 128> result{};
        EXPECT_EQ(cudaMemcpy(result.data(), sentinel, result.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        for (const auto value : result)
            EXPECT_EQ(value, 0x5a);
    }
    EXPECT_EQ(cudaFree(sentinel), cudaSuccess);
}
}  // namespace rtp_llm::benchmark
