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
        const std::string csv_header = "key,parent_key,depth_blocks,count,first_ns,last_ns\n";
        const auto        observed   = dir / "endpoints.csv";
        sparse.writeMatchHistory(observed.string());
        stateRequire(contents(observed) == csv_header, "existing path must not create a branch");
        for (int64_t id : {3, 4}) {
            auto hit = sparse.run(request(id, {1, 2, 3, 4, 5, 6, 7, 8, 100 + id, 200 + id}));
            stateRequire(hit.success && hit.joint_match_tokens == 2048, "expected an eight-block prefix hit");
            sparse.writeMatchHistory(observed.string());
            const auto expected = csv_header + "8,7,8," + std::to_string(id - 2) + ",3," + std::to_string(id) + "\n";
            stateRequire(contents(observed) == expected, "history must count only the last matched block per request");
        }
        const auto           before_failed_insert = contents(observed);
        std::vector<int64_t> oversized{1, 2, 3, 4, 5, 6, 7, 8};
        for (int64_t i = 0; i < 128; ++i)
            oversized.push_back(9000 + i);
        auto failed = sparse.run(request(5, oversized));
        stateRequire(!failed.success && failed.full_match_tokens > 0,
                     "expected positive FULL match before allocation failure");
        sparse.writeMatchHistory(observed.string());
        stateRequire(contents(observed) == before_failed_insert, "match without insert counted a branch");
        sparse.writeBranchStatistics(dir.string());
        stateRequire(!fs::exists(dir / "node-heat.csv"), "obsolete heat export");
        ::unlink(observed.c_str());
        MetadataReplay pinned(c);
        stateRequire(pinned.checkPinnedForceDrop(), "pinned allocation did not trigger native force_drop");
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
        continuous.writeMatchHistory((dir / "history1.csv").string());
        resumed.writeMatchHistory((dir / "history2.csv").string());
        stateRequire(contents(dir / "history1.csv") == contents(dir / "history2.csv"), "history restore differs");
        for (int id = 4; id < 35; ++id) {
            std::vector<int64_t> keys;
            for (int j = 1; j <= 8; ++j)
                keys.push_back(id * 100 + j);
            stateRequire(resumed.run(request(id, keys)).success, "pressure request failed");
        }
        resumed.writeMatchHistory((dir / "history3.csv").string());
        stateRequire(contents(dir / "history3.csv").find("8,7,8,") != std::string::npos,
                     "eviction deleted historical endpoint");
        resumed.writeBranchStatistics(dir.string());
        stateRequire(contents(dir / "evicted-branches.csv").find(",8,7,8,1\n") != std::string::npos,
                     "missing eviction-time branch count");
        const auto eviction_summary = contents(dir / "branch-eviction-summary.json");
        stateRequire(
            eviction_summary.find("\"evicted_branch_instances\":2,\"evicted_instance_fraction\":1,")
                    != std::string::npos
                && eviction_summary.find("\"positive_evicted_branch_counts\":{\"mean\":1,\"median\":1,\"p95\":1}")
                       != std::string::npos
                && eviction_summary.find("\"median\":0,\"p95\":0") != std::string::npos,
            "unexpected branch eviction distribution");
        resumed.saveState((dir / "pressure.bin").string());
        MetadataReplay reinserted(c);
        reinserted.restoreState((dir / "pressure.bin").string());
        reinserted.writeBranchStatistics(dir.string());
        stateRequire(contents(dir / "branch-eviction-summary.json") == eviction_summary
                         && contents(dir / "evicted-branches.csv").find(",8,7,8,1\n") != std::string::npos,
                     "branch eviction evidence lost on restore");
        auto seed_again = reinserted.run(request(35, {1, 2, 3, 4, 5, 6, 7, 8}));
        stateRequire(seed_again.success && !seed_again.full_match_tokens, "expected cold reinsert");
        reinserted.writeBranchStatistics(dir.string());
        stateRequire(contents(dir / "branch-eviction-summary.json").find("\"historical_branch_instances\":2,")
                         != std::string::npos,
                     "cold reinsert counted as a branch");
        const auto rebuilt_summary = contents(dir / "branch-eviction-summary.json");
        stateRequire(rebuilt_summary.find("\"distinct_root_context_keys\":32") != std::string::npos,
                     "root reinsertion must not create a new distinct context");
        reinserted.run(request(36, {1, 2, 3, 4, 5, 6, 7, 8, 901}));
        reinserted.writeMatchHistory(observed.string());
        stateRequire(contents(observed).find("8,7,8,2,2,36\n") != std::string::npos,
                     "historical position lost on reinsert");
        ::unlink(observed.c_str());
        for (int id = 37; id < 69; ++id) {
            std::vector<int64_t> keys;
            for (int i = 1; i <= 8; ++i)
                keys.push_back(100 * id + i);
            stateRequire(reinserted.run(request(id, keys)).success, "repeat eviction pressure failed");
        }
        reinserted.writeBranchStatistics(dir.string());
        stateRequire(contents(dir / "evicted-branch-keys.csv").find("\n8,2,2,3\n") != std::string::npos,
                     "expected two evictions of key 8 with heat 1+1+1");
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
            << "PASS native F/J, history growth and eviction survival, checkpoint restore, config and corruption rejection\n";
        for (const char* name : {"cache.bin",
                                 "pressure.bin",
                                 "bad.bin",
                                 "history1.csv",
                                 "history2.csv",
                                 "history3.csv",
                                 "node-heat.csv",
                                 "evicted-branches.csv",
                                 "evicted-branch-keys.csv",
                                 "branch-eviction-summary.json"})
            ::unlink((dir / name).c_str());
        ::rmdir(dir.c_str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
