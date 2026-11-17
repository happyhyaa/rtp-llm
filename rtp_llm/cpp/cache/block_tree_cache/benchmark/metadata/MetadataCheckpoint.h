#pragma once
#include <istream>
#include <ostream>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/StateIO.h"
namespace rtp_llm {
class BlockTreeCache;
class IBlockPool;
class BlockTreeMetadataCheckpoint {
public:
    static void save(BlockTreeCache& cache, benchmark::StateWriter& writer, bool normalize_time = false);
    static void restore(BlockTreeCache& cache, benchmark::StateReader& reader);

public:
    static void savePool(IBlockPool& pool, benchmark::StateWriter& writer);
    static void restorePool(IBlockPool& pool, benchmark::StateReader& reader);
};
}  // namespace rtp_llm
