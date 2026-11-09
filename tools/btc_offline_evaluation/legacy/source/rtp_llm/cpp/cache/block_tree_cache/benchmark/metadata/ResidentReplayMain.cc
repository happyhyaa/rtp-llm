#include <climits>
// Hold a quiescent native BTC instance between approved, ordered trace segments.
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <sys/resource.h>
#include <unistd.h>
#include <rapidjson/document.h>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/MetadataReplay.h"
#include "rtp_llm/cpp/utils/Logger.h"
using namespace rtp_llm::benchmark;
namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;
namespace {
constexpr uint64_t day_start = 1793030400000000000ULL, day_end = day_start + 86400ULL * 1000000000;
constexpr uint64_t day_requests = 65740206, chunk_size = 100000;
void               check(bool ok, const char* msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
template<class T>
T scalar(std::istream& f) {
    T v{};
    check(bool(f.read(reinterpret_cast<char*>(&v), sizeof(v))), "truncated record");
    return v;
}
std::string label(std::istream& f) {
    auto n = scalar<uint32_t>(f);
    check(n <= 1048576, "invalid label size");
    std::string s(n, '\0');
    check(bool(f.read(s.data(), n)), "short label");
    return s;
}
void atomicWrite(const fs::path& p, const std::string& s) {
    auto t = p;
    t += ".tmp";
    std::ofstream f(t);
    f << s << '\n';
    f.close();
    check(bool(f), "result write failed");
    fs::rename(t, p);
}
uint64_t through(const fs::path& root) {
    std::ifstream f(root / "authorized-through");
    uint64_t      n = 0;
    check(bool(f >> n) && n <= day_requests && n > 0, "invalid authorized-through");
    return n;
}
MetadataReplayConfig config(const fs::path& path) {
    std::ifstream       f(path);
    std::string         body(std::istreambuf_iterator<char>(f), {});
    rapidjson::Document d;
    d.Parse(body.c_str());
    check(!d.HasParseError() && d.IsObject(), "invalid capacity file");
    const auto& tiers  = d["tiers"];
    auto usable = [&](const char* tier, const char* group) {
        const auto& value = tiers[tier][group];
        check(value.IsInt() && value.GetInt() > 0 && value.GetInt() < INT_MAX, "invalid tier capacity");
        return value.GetUint();
    };
    MetadataReplayConfig c;
    c.dsv41_topology    = true;
    c.block_tokens      = 256;
    c.cp_size           = 4;
    c.cp_rank           = 0;
    c.min_replay_tokens = 128;
    c.swa_block_tokens  = 1024;
    c.sliding_window    = 128;
    c.linear_step       = 1;
    if (d.HasMember("last_n")) {
        check(d["last_n"].IsUint() && d["last_n"].GetUint() > 0, "invalid last_n");
        c.last_n = d["last_n"].GetUint();
    }
    c.disk_fifo = d.HasMember("disk_fifo") && d["disk_fifo"].GetBool();
    c.device_blocks = usable("device", "full");
    c.swa_blocks = usable("device", "swa");
    c.host_blocks = usable("host", "full");
    c.host_swa_blocks = usable("host", "swa");
    c.disk_blocks = usable("disk", "full");
    c.disk_swa_blocks = usable("disk", "swa");
    c.device_low  = .82;
    c.device_high = .90;
    c.host_low    = .90;
    c.host_high   = .94;
    c.disk_low    = .92;
    c.disk_high   = .97;
    return c;
}
struct Session {
    fs::path                      input, out;
    MetadataReplay                engine;
    Clock::time_point             start = Clock::now();
    uint64_t                      n = 0, tokens = 0, hits = 0, failed = 0, last_ts = 0, limit = 0;
    std::map<std::string, double> stages;
    std::ofstream                 facts;
    Session(fs::path source, fs::path dest, const MetadataReplayConfig& c):
        input(source), out(dest), engine(c), limit(through(dest)), facts(dest / "hit-tokens.u32", std::ios::binary) {
        check(bool(facts), "cannot write hit counters");
    }
    std::string state(const char* status) {
        auto s = engine.stats();
        check(s.payload_bytes == 0 && s.copy_attempts == 0, "physical backing/copy encountered");
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        std::ostringstream o;
        o << std::setprecision(17) << "{\"status\":\"" << status << "\",\"pid\":" << getpid() << ",\"requests\":" << n
          << ",\"input_tokens\":" << tokens << ",\"hit_tokens\":" << hits
          << ",\"hit_rate\":" << (tokens ? double(hits) / tokens : 0) << ",\"failed\":" << failed
          << ",\"last_timestamp_ns\":" << last_ts << ",\"next_ordinal\":" << n + 1
          << ",\"next_chunk\":" << n / chunk_size << ",\"chunk_offset\":" << n % chunk_size
          << ",\"force_drop_blocks\":" << s.force_blocks << ",\"watermark_blocks\":" << s.watermark_blocks
          << ",\"tree_nodes\":" << s.tree_nodes << ",\"device_used\":" << s.device_used
          << ",\"host_used\":" << s.host_used << ",\"disk_used\":" << s.disk_used
          << ",\"failed_transfers\":" << s.failed_transfers << ",\"kv_payload_bytes\":" << s.payload_bytes
          << ",\"physical_copy_attempts\":" << s.copy_attempts << ",\"max_rss_kib\":" << usage.ru_maxrss
          << ",\"elapsed_s\":" << std::chrono::duration<double>(Clock::now() - start).count() << ",\"stage_us\":{";
        bool comma = false;
        for (auto& v : stages) {
            if (comma)
                o << ',';
            comma = true;
            o << '"' << v.first << "\":" << v.second;
        }
        o << "}}";
        return o.str();
    }
    void boundary() {
        if (n != limit)
            return;
        facts.flush();
        check(bool(facts), "facts flush failed");
        auto summary = state("held");
        atomicWrite(out / ("held-" + std::to_string(n) + ".json"), summary);
        atomicWrite(out / "progress.json", summary);
        // Keep the engine, workers, LRU, tier memberships, IDs and input cursor alive.
        while (limit == n) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            limit = through(out);
            check(limit >= n, "backward resume forbidden");
        }
    }
    void run() {
        MetadataRequest q;
        uint64_t        index = 0;
        for (;; ++index) {
            std::ostringstream name;
            name << std::setw(6) << std::setfill('0') << index << ".btcbin";
            auto file = input / name.str();
            while (!fs::exists(file)) {
                check(!fs::exists(input / "FAILED.json"), "input producer failed");
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
            std::ifstream f(file, std::ios::binary);
            char          magic[8];
            check(bool(f.read(magic, 8)) && !std::memcmp(magic, "BTCPFX01", 8) && scalar<uint32_t>(f) == 256,
                  "bad chunk header");
            auto count = scalar<uint64_t>(f), expected = scalar<uint64_t>(f);
            check(count == std::min<uint64_t>(chunk_size, day_requests - n), "unexpected chunk size");
            auto domain = label(f);
            check(!domain.empty(), "empty domain");
            if (index == 0)
                atomicWrite(out / "cache-domain.txt", domain);
            else {
                std::ifstream df(out / "cache-domain.txt");
                std::string   first;
                std::getline(df, first);
                check(first == domain, "domain changed");
            }
            uint64_t chunk_tokens = 0;
            for (uint64_t i = 0; i < count; ++i) {
                q.id           = n + 1;
                q.timestamp_ns = scalar<uint64_t>(f);
                auto len       = scalar<uint32_t>(f);
                check(len && len <= std::numeric_limits<int>::max(), "bad token length");
                q.input_tokens = len;
                check(q.timestamp_ns >= day_start && q.timestamp_ns < day_end && q.timestamp_ns >= last_ts,
                      "timestamp outside ordered day");
                (void)label(f);
                q.prefix_keys.resize(len / 256);
                if (!q.prefix_keys.empty())
                    check(bool(f.read(reinterpret_cast<char*>(q.prefix_keys.data()), 8 * q.prefix_keys.size())),
                          "truncated base keys");
                auto result = engine.run(q);
                check(result.reuse_tokens >= 0 && result.reuse_tokens <= q.input_tokens, "bad hit count");
                auto hit = static_cast<uint32_t>(result.reuse_tokens);
                facts.write(reinterpret_cast<const char*>(&hit), sizeof(hit));
                check(bool(facts), "facts write failed");
                ++n;
                failed += !result.success;
                tokens += len;
                chunk_tokens += len;
                hits += hit;
                last_ts = q.timestamp_ns;
                for (auto& v : result.stage_us)
                    stages[v.first] += v.second;
                if (i + 1 == count)
                    check(chunk_tokens == expected && f.peek() == std::char_traits<char>::eof(),
                          "invalid complete chunk");
                if (n % 10000 == 0)
                    atomicWrite(out / "progress.json", state("running"));
                boundary();
            }
        }
    }
};
}  // namespace
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: resident_replay CAPACITY.json CHUNKS NEW_RUN_DIR\n";
        return 2;
    }
    const fs::path out = argv[3];
    try {
        check(!fs::exists("/dev/nvidia0") && !fs::exists("/dev/nvidiactl"), "GPU device accessible");
        check(!fs::exists(out / "progress.json") && !fs::exists(out / "hit-tokens.u32"), "run is not fresh");
        rtp_llm::initLogger();
        Session session(argv[2], out, config(argv[1]));
        atomicWrite(out / "progress.json", session.state("starting"));
        session.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (!fs::exists(out / "FAILED.txt"))
            atomicWrite(out / "FAILED.txt", e.what());
        return 1;
    }
    return 0;
}
