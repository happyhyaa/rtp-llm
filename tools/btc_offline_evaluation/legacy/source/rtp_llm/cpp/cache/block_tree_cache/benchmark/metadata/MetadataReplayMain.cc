#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <sys/resource.h>
#include <random>
#include <cmath>
#include <cstring>
#include <limits>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/MetadataReplay.h"
#include "rtp_llm/cpp/utils/Logger.h"
using namespace rtp_llm::benchmark;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
rapidjson::Document json(const std::string& s) {
    rapidjson::Document d;
    d.Parse(s.c_str());
    require(!d.HasParseError() && d.IsObject(), "invalid JSON object");
    return d;
}
std::string quoted(const std::string& text) {
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.String(text.data(), static_cast<rapidjson::SizeType>(text.size()));
    return buffer.GetString();
}
struct StageTimes {
    uint64_t            count{0};
    double              total{0};
    std::vector<double> sample;
    std::mt19937_64     rng{1};
    void                add(double elapsed) {
        ++count;
        total += elapsed;
        if (sample.size() < 4096)
            sample.push_back(elapsed);
        else {
            auto index = std::uniform_int_distribution<uint64_t>(0, count - 1)(rng);
            if (index < sample.size())
                sample[index] = elapsed;
        }
    }
    double quantile(double fraction) const {
        return sample.at(static_cast<size_t>(std::ceil(fraction * sample.size())) - 1);
    }
};
// BTCPFX01 stores signed base prefix keys in little-endian order.
template<class T>
T readScalar(std::istream& input) {
    T value{};
    require(bool(input.read(reinterpret_cast<char*>(&value), sizeof(value))), "truncated prepared trace");
    return value;
}
std::string readLabel(std::istream& input) {
    auto length = readScalar<uint32_t>(input);
    require(length <= 1048576, "invalid prepared label length");
    std::string value(length, '\0');
    require(bool(input.read(value.data(), length)), "truncated prepared label");
    return value;
}
int main(int argc, char** argv) {
    std::filesystem::path out;
    bool                  created = false;
    try {
        require(argc == 4, "usage: metadata_replay CONFIG.json REQUESTS.jsonl|REQUESTS.btcbin NEW_OUTPUT_DIR");
        std::ifstream cf(argv[1]);
        require(bool(cf), "config missing");
        auto                           d = json(std::string(std::istreambuf_iterator<char>(cf), {}));
        MetadataReplayConfig           c;
        std::map<std::string, size_t*> ints = {{"block_tokens", &c.block_tokens},
                                               {"cp_size", &c.cp_size},
                                               {"cp_rank", &c.cp_rank},
                                               {"min_replay_tokens", &c.min_replay_tokens},
                                               {"device_blocks", &c.device_blocks},
                                               {"host_blocks", &c.host_blocks},
                                               {"disk_blocks", &c.disk_blocks},
                                               {"swa_block_tokens", &c.swa_block_tokens},
                                               {"swa_blocks", &c.swa_blocks},
                                               {"last_n", &c.last_n},
                                               {"linear_step", &c.linear_step},
                                               {"sliding_window", &c.sliding_window}};
        for (auto& [key, p] : ints)
            if (d.HasMember(key.c_str())) {
                require(d[key.c_str()].IsUint64(), "integer config invalid");
                *p = d[key.c_str()].GetUint64();
            }
        std::map<std::string, double*> floats = {{"device_low", &c.device_low},
                                                 {"device_high", &c.device_high},
                                                 {"host_low", &c.host_low},
                                                 {"host_high", &c.host_high},
                                                 {"disk_low", &c.disk_low},
                                                 {"disk_high", &c.disk_high}};
        for (auto& [key, p] : floats)
            if (d.HasMember(key.c_str())) {
                require(d[key.c_str()].IsNumber(), "watermark invalid");
                *p = d[key.c_str()].GetDouble();
            }
        if (d.HasMember("swa")) {
            require(d["swa"].IsBool(), "swa invalid");
            c.swa = d["swa"].GetBool();
        }
        for (auto pair : {std::make_pair(c.device_low, c.device_high),
                          std::make_pair(c.host_low, c.host_high),
                          std::make_pair(c.disk_low, c.disk_high)})
            require(pair.first >= 0 && pair.second <= 1
                        && ((pair.first == 0 && pair.second == 0) || pair.first < pair.second),
                    "invalid watermarks");
        for (const auto& member : d.GetObject()) {
            std::string key = member.name.GetString();
            require(ints.count(key) || floats.count(key) || key == "swa", "unknown config option");
        }
        out = argv[3];
        require(!std::filesystem::exists(out), "output exists");
        require(std::filesystem::create_directories(out), "output directory already claimed");
        created = true;
        rtp_llm::initLogger();
        MetadataReplay replay(c);
        std::ifstream  input(argv[2], std::ios::binary);
        require(bool(input), "trace missing");
        std::ofstream rows(out / "requests.jsonl");
        require(bool(rows), "output unavailable");
        uint64_t                          requests = 0, failed = 0, tokens = 0, hits = 0;
        std::string                       line, instance;
        std::map<std::string, StageTimes> stages;
        const bool                        prepared          = std::filesystem::path(argv[2]).extension() == ".btcbin";
        uint64_t                          expected_requests = 0, expected_tokens = 0;
        if (prepared) {
            uint32_t endian = 1;
            char     magic[8];
            require(*reinterpret_cast<char*>(&endian) == 1 && input.read(magic, 8)
                        && std::memcmp(magic, "BTCPFX01", 8) == 0,
                    "unsupported prepared format");
            require(readScalar<uint32_t>(input) == c.block_tokens, "prepared base granularity mismatch");
            expected_requests = readScalar<uint64_t>(input);
            expected_tokens   = readScalar<uint64_t>(input);
            instance          = readLabel(input);
            require(expected_requests && !instance.empty(), "invalid prepared header");
        }
        const auto run_begin = Clock::now();
        for (;;) {
            const auto      read_begin = Clock::now();
            MetadataRequest request;
            std::string     trace_id;
            if (prepared) {
                if (requests == expected_requests)
                    break;
                request.id           = static_cast<int64_t>(requests + 1);
                request.timestamp_ns = readScalar<uint64_t>(input);
                auto length          = readScalar<uint32_t>(input);
                require(request.timestamp_ns && length > 0 && length <= std::numeric_limits<int>::max(),
                        "invalid prepared record");
                request.input_tokens = static_cast<int>(length);
                trace_id             = readLabel(input);
                request.prefix_keys.resize(length / c.block_tokens);
                if (!request.prefix_keys.empty())
                    require(bool(input.read(reinterpret_cast<char*>(request.prefix_keys.data()),
                                            request.prefix_keys.size() * sizeof(int64_t))),
                            "truncated prepared keys");
                stages["prepared_read"].add(
                    std::chrono::duration<double, std::micro>(Clock::now() - read_begin).count());
            } else {
                if (!std::getline(input, line))
                    break;
                stages["read_line"].add(std::chrono::duration<double, std::micro>(Clock::now() - read_begin).count());
                const auto parse_begin = Clock::now();
                auto       r           = json(line);
                require(r.HasMember("key_form") && r["key_form"].IsString()
                            && std::string(r["key_form"].GetString()) == "prefix",
                        "require declared prefix keys; raw hash conversion is not implicit");
                require(r.HasMember("block_tokens") && r["block_tokens"].IsUint64()
                            && r["block_tokens"].GetUint64() == c.block_tokens,
                        "trace base granularity mismatch");
                require(r.HasMember("input_len") && r["input_len"].IsInt() && r["input_len"].GetInt() > 0,
                        "input_len missing/invalid");
                require(r.HasMember("keys") && r["keys"].IsArray(), "keys missing");
                require(r.HasMember("timestamp_ns") && r["timestamp_ns"].IsUint64(), "timestamp missing");
                require(r.HasMember("instance_id") && r["instance_id"].IsString(), "instance_id missing");
                std::string current = r["instance_id"].GetString();
                require(!current.empty(), "empty instance");
                if (instance.empty())
                    instance = current;
                require(instance == current, "one cache domain per run required");
                require(r["timestamp_ns"].GetUint64() > 0, "timestamp must be positive");
                trace_id = std::to_string(requests + 1);
                if (r.HasMember("request_id")) {
                    require(r["request_id"].IsString(), "request ID must be a string");
                    trace_id.assign(r["request_id"].GetString(), r["request_id"].GetStringLength());
                }
                request.id           = static_cast<int64_t>(requests + 1);
                request.timestamp_ns = r["timestamp_ns"].GetUint64();
                request.input_tokens = r["input_len"].GetInt();
                for (const auto& key : r["keys"].GetArray()) {
                    require(key.IsInt64(), "keys must be signed int64 without floating conversion");
                    request.prefix_keys.push_back(key.GetInt64());
                }
                const double parse_us = std::chrono::duration<double, std::micro>(Clock::now() - parse_begin).count();
                stages["parse_adapt"].add(parse_us);
            }
            auto result = replay.run(request);
            ++requests;
            tokens += request.input_tokens;
            failed += !result.success;
            hits += result.reuse_tokens;
            for (const auto& [stage, us] : result.stage_us)
                stages[stage].add(us);
            auto write_begin = Clock::now();
            rows << "{\"ordinal\":" << requests << ",\"request_id\":" << quoted(trace_id)
                 << ",\"timestamp_ns\":" << request.timestamp_ns << ",\"input_tokens\":" << request.input_tokens
                 << ",\"success\":" << (result.success ? "true" : "false")
                 << ",\"reuse_tokens\":" << result.reuse_tokens << ",\"status\":\"" << result.status << "\"}\n";
            require(bool(rows), "row write failed");
            stages["serialize"].add(std::chrono::duration<double, std::micro>(Clock::now() - write_begin).count());
        }
        if (prepared)
            require(requests == expected_requests && tokens == expected_tokens
                        && input.peek() == std::char_traits<char>::eof(),
                    "prepared cohort mismatch");
        require(input.eof() && !input.bad() && requests > 0, "input failure or empty trace");
        rows.close();
        require(bool(rows), "row flush failed");
        auto          s = replay.stats();
        struct rusage usage{};
        require(getrusage(RUSAGE_SELF, &usage) == 0, "RSS query failed");
        std::ofstream summary(out / "summary.json");
        summary << std::setprecision(17) << "{\"complete\":true,\"requests\":" << requests
                << ",\"linear_step\":" << c.linear_step << ",\"linear_step_source\":"
                << quoted(d.HasMember("linear_step") ? "explicit_override" : "KVCacheConfig default")
                << ",\"swa_active_tail_blocks\":" << (c.swa ? std::to_string(c.last_n) : "null")
                << ",\"swa_tail_source\":"
                << quoted(!c.swa                ? "disabled" :
                          d.HasMember("last_n") ? "explicit_override" :
                                                  "SWA group policy default")
                << ",\"failed_requests\":" << failed << ",\"input_tokens\":" << tokens << ",\"hit_tokens\":" << hits
                << ",\"hit_rate_all_submitted\":" << double(hits) / tokens
                << ",\"watermark_round_reports\":" << s.watermark_calls
                << ",\"watermark_scheduled_blocks\":" << s.watermark_blocks << ",\"force_drop_calls\":" << s.force_calls
                << ",\"force_drop_reclaimed_blocks\":" << s.force_blocks << ",\"transfers\":" << s.transfers
                << ",\"failed_transfers\":" << s.failed_transfers << ",\"device_host\":" << s.device_host
                << ",\"host_disk\":" << s.host_disk << ",\"host_device\":" << s.host_device
                << ",\"disk_device\":" << s.disk_device << ",\"tree_nodes\":" << s.tree_nodes
                << ",\"device_used\":" << s.device_used << ",\"host_used\":" << s.host_used
                << ",\"disk_used\":" << s.disk_used << ",\"kv_payload_bytes_allocated\":" << s.payload_bytes
                << ",\"physical_copy_attempts\":" << s.copy_attempts
                << ",\"watermark_check_work_us\":" << s.watermark_ns / 1000.0
                << ",\"force_drop_work_us\":" << s.force_ns / 1000.0 << ",\"init_us\":" << replay.initMicros()
                << ",\"run_us\":" << std::chrono::duration<double, std::micro>(Clock::now() - run_begin).count()
                << ",\"peak_rss_kib\":" << usage.ru_maxrss << ",\"instance_id\":" << quoted(instance)
                << ",\"stages\":{";
        bool comma = false;
        for (auto& [stage, values] : stages) {
            std::sort(values.sample.begin(), values.sample.end());
            if (comma)
                summary << ',';
            comma = true;
            summary << quoted(stage) << ":{\"count\":" << values.count << ",\"total_us\":" << values.total
                    << ",\"mean_us\":" << values.total / values.count
                    << ",\"quantile_sample_count\":" << values.sample.size() << ",\"p50_us\":" << values.quantile(0.5)
                    << ",\"p95_us\":" << values.quantile(0.95) << ",\"p99_us\":" << values.quantile(0.99) << "}";
        }
        summary << "}}\n";
        summary.close();
        require(bool(summary), "summary write failed");
        std::cout << "complete " << requests << " requests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        if (created) {
            std::ofstream f(out / "FAILED.txt");
            f << e.what() << '\n';
        }
        return 1;
    }
}
