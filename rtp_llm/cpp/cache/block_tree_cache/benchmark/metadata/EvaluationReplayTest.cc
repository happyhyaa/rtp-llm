#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unistd.h>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/MetadataReplay.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/EvaluationStats.h"
using namespace rtp_llm::benchmark;
namespace fs = std::filesystem;
MetadataRequest request(int64_t id, std::vector<int64_t> keys) {
    MetadataRequest r;
    r.id           = id;
    r.timestamp_ns = id;
    r.input_tokens = keys.size() * 256 + 128;
    r.prefix_keys  = std::move(keys);
    return r;
}
std::string contents(const fs::path& p) {
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), {}};
}
void checkMetricAggregation() {
    EvaluationStats original;
    original.add(1, 1024, 512, 256, {});
    original.add(2, 1024, 0, 0, {});
    original.add(300000000001ULL, 1024, 1024, 1024, {});
    stateRequire(original.total.full.percentile(50) == 512 && original.total.gap.sum == 256
                     && original.total.zero_full == 1 && original.timeline.size() == 2,
                 "metric aggregation differs from known values");
    std::stringstream file;
    StateWriter       writer(file);
    original.save(writer);
    writer.finish();
    StateReader     reader(file);
    EvaluationStats restored;
    restored.restore(reader);
    reader.finish();
    stateRequire(restored.json() == original.json(), "metric restore differs");
}
int main() {
    try {
        checkMetricAggregation();
        const auto dir = fs::temp_directory_path() / ("btc-v2-contract-" + std::to_string(getpid()));
        stateRequire(fs::create_directory(dir), "temporary directory exists");
        MetadataReplayConfig c;
        c.device_blocks     = 16;
        c.swa_blocks        = 16;
        c.host_blocks       = 16;
        c.disk_blocks       = 32;
        c.last_n            = 2;
        c.min_replay_tokens = 128;
        c.device_low        = .5;
        c.device_high       = .75;
        c.host_low          = .5;
        c.host_high         = .75;
        c.disk_low          = .5;
        c.disk_high         = .75;
        MetadataReplay sparse(c);
        stateRequire(sparse.run(request(1, {1, 2, 3, 4, 5, 6, 7, 8})).success, "sparse seed failed");
        const auto gap = sparse.run(request(2, {1, 2, 3, 4, 5, 6}));
        stateRequire(gap.success && gap.full_match_tokens == 1536 && gap.joint_match_tokens == 0,
                     "SWA hole was not reflected in FULL/joint gap");
        // A continued chain is not a structural fork, and a new root child is not a fork.
        MetadataReplayConfig structure_config;
        structure_config.swa        = false;
        structure_config.device_low = structure_config.host_low = structure_config.disk_low = 0.8;
        structure_config.device_high = structure_config.host_high = structure_config.disk_high = 0.9;
        std::cout << "CHECK static tree" << std::endl;
        MetadataReplay structure(structure_config);
        stateRequire(structure.run(request(1, {1, 2})).success, "structure seed");
        stateRequire(structure.run(request(2, {1, 2, 3})).success, "structure extension");
        stateRequire(structure.run(request(3, {10, 11})).success, "second root context");
        structure.writeTreeStructure(dir.string());
        stateRequire(contents(dir / "tree-structure.json").find("\"branch_nodes\":0") != std::string::npos,
                     "chain extension or root was counted as structural fork");
        stateRequire(structure.run(request(4, {1, 2, 4})).success, "structure fork");
        structure.writeTreeStructure(dir.string());
        const auto snapshot = contents(dir / "tree-structure.csv");
        stateRequire(snapshot.find("2,1,2,2,1") != std::string::npos, "fork at key 2 missing");
        stateRequire(contents(dir / "tree-structure.json").find("\"branch_nodes\":1") != std::string::npos,
                     "expected exactly one structural fork");
        const auto before = structure.stats();
        structure.saveState((dir / "before-export.bin").string(), true);
        structure.writeTreeStructure(dir.string());
        stateRequire(contents(dir / "tree-structure.csv") == snapshot
                         && structure.stats().tree_nodes == before.tree_nodes
                         && structure.stats().watermark_blocks == before.watermark_blocks,
                     "export mutated cache");
        structure.saveState((dir / "after-export.bin").string(), true);
        stateRequire(contents(dir / "before-export.bin") == contents(dir / "after-export.bin"),
                     "export changed native tree, refs or victim priorities");
        structure.saveState((dir / "structure.bin").string());
        MetadataReplay structure_restored(structure_config);
        structure_restored.restoreState((dir / "structure.bin").string());
        structure_restored.writeTreeStructure(dir.string());
        stateRequire(contents(dir / "tree-structure.csv") == snapshot, "structure snapshot differs after restore");
        std::cout << "CHECK structure restore passed" << std::endl;
        // A prefix match can survive an allocation failure; no insertion occurred.
        std::vector<int64_t> oversized{1, 2, 3, 4, 5, 6};
        for (int64_t i = 0; i < 128; ++i)
            oversized.push_back(9000 + i);
        const auto failed = sparse.run(request(3, oversized));
        stateRequire(!failed.success && failed.full_match_tokens > 0, "expected FULL before allocation failure");
        MetadataReplay pinned(c);
        stateRequire(pinned.checkPinnedForceDrop(), "pinned allocation did not trigger native force_drop");
        std::cout << "CHECK pressure checkpoint" << std::endl;
        MetadataReplay continuous(c);
        auto           a = continuous.run(request(1, {1, 2, 3, 4, 5, 6, 7, 8}));
        auto           b = continuous.run(request(2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
        stateRequire(a.success && b.success && a.joint_match_tokens == 0 && b.full_match_tokens == 2048
                         && b.joint_match_tokens == 2048,
                     "native F/J matching failed");
        continuous.saveState((dir / "cache.bin").string());
        MetadataReplay resumed(c);
        resumed.restoreState((dir / "cache.bin").string());
        auto q = request(3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        auto x = continuous.run(q), y = resumed.run(q);
        stateRequire(x.success && y.success && x.full_match_tokens == y.full_match_tokens
                         && x.joint_match_tokens == y.joint_match_tokens,
                     "restored match differs");
        for (int id = 4; id < 35; ++id) {
            std::vector<int64_t> keys;
            for (int j = 1; j <= 8; ++j)
                keys.push_back(id * 100 + j);
            const auto next = request(id, keys);
            const auto u = continuous.run(next), v = resumed.run(next);
            stateRequire(u.success && v.success && u.full_match_tokens == v.full_match_tokens
                             && u.joint_match_tokens == v.joint_match_tokens,
                         "pressure resume changed native matching");
        }
        resumed.saveState((dir / "pressure.bin").string());
        MetadataReplay again(c);
        again.restoreState((dir / "pressure.bin").string());
        const auto next = request(35, {1, 2, 3, 4, 5, 6, 7, 8});
        const auto u = resumed.run(next), v = again.run(next);
        stateRequire(u.success && v.success && u.full_match_tokens == v.full_match_tokens
                         && u.joint_match_tokens == v.joint_match_tokens,
                     "post-pressure restore changed matching");
        auto changed   = c;
        changed.last_n = 3;
        MetadataReplay wrong(changed);
        bool           rejected = false;
        try {
            wrong.restoreState((dir / "cache.bin").string());
        } catch (const std::exception&) {
            rejected = true;
        }
        stateRequire(rejected, "different configuration accepted");
        {
            std::ofstream old(dir / "old.bin", std::ios::binary);
            StateWriter   writer(old);
            writer.text("BTCMETA2-a3fdca664-v5-branch-key");
            writer.finish();
        }
        MetadataReplay legacy(c);
        rejected = false;
        try {
            legacy.restoreState((dir / "old.bin").string());
        } catch (const std::exception& e) {
            rejected = std::string(e.what()).find("schema mismatch") != std::string::npos;
        }
        stateRequire(rejected, "legacy branch-state checkpoint must be explicitly rejected");
        auto body = contents(dir / "cache.bin");
        body[body.size() / 2] ^= 1;
        {
            std::ofstream f(dir / "bad.bin", std::ios::binary);
            f.write(body.data(), body.size());
        }
        MetadataReplay fresh(c);
        rejected = false;
        try {
            fresh.restoreState((dir / "bad.bin").string());
        } catch (const std::exception&) {
            rejected = true;
        }
        stateRequire(rejected, "corrupted checkpoint accepted");
        std::cout
            << "PASS native F/J, read-only structural analysis, pressure resume, config and corruption rejection\n";
        for (const char* name : {"cache.bin",
                                 "pressure.bin",
                                 "bad.bin",
                                 "old.bin",
                                 "structure.bin",
                                 "before-export.bin",
                                 "after-export.bin",
                                 "tree-structure.csv",
                                 "tree-structure.json"})
            ::unlink((dir / name).c_str());
        ::rmdir(dir.c_str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
