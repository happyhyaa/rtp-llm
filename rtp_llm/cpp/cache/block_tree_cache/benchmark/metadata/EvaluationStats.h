#pragma once
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/metadata/StateIO.h"
namespace rtp_llm::benchmark {
struct LengthDistribution {
    uint64_t                     count{0}, sum{0};
    std::map<uint64_t, uint64_t> histogram;
    void                         add(uint64_t value) {
        ++count;
        sum += value;
        ++histogram[value];
    }
    uint64_t percentile(unsigned percentage) const {
        const uint64_t rank  = (count * percentage + 99) / 100;
        uint64_t       total = 0;
        for (const auto& [value, n] : histogram) {
            total += n;
            if (total >= rank)
                return value;
        }
        return 0;
    }
    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\"count\":" << count << ",\"sum\":" << sum << ",\"mean\":" << (count ? double(sum) / count : 0)
            << ",\"max\":" << (histogram.empty() ? 0 : histogram.rbegin()->first) << ",\"p50\":" << percentile(50)
            << ",\"p95\":" << percentile(95) << ",\"p99\":" << percentile(99) << '}';
        return out.str();
    }
    void save(StateWriter& w) const {
        w.scalar(count);
        w.scalar(sum);
        w.scalar(uint64_t(histogram.size()));
        for (const auto& [value, n] : histogram) {
            w.scalar(value);
            w.scalar(n);
        }
    }
    void restore(StateReader& r) {
        count           = r.scalar<uint64_t>();
        sum             = r.scalar<uint64_t>();
        const auto bins = r.scalar<uint64_t>();
        stateRequire(bins <= count, "invalid histogram size");
        for (uint64_t i = 0; i < bins; ++i) {
            const auto value = r.scalar<uint64_t>();
            histogram[value] = r.scalar<uint64_t>();
        }
    }
};
struct MatchWindow {
    uint64_t                        requests{0}, input_tokens{0}, zero_full{0};
    double                          ratio_sum{0};
    LengthDistribution              full, joint, gap;
    std::map<std::string, uint64_t> evictions;
    void add(uint64_t tokens, uint64_t f, uint64_t j, const std::map<std::string, uint64_t>& events) {
        stateRequire(j <= f && f <= tokens, "match lengths outside request");
        ++requests;
        input_tokens += tokens;
        zero_full += f == 0;
        if (f)
            ratio_sum += double(j) / f;
        full.add(f);
        joint.add(j);
        gap.add(f - j);
        for (const auto& [name, n] : events)
            evictions[name] += n;
    }
    std::string json() const {
        std::ostringstream out;
        out << std::setprecision(17);
        out << "{\"requests\":" << requests << ",\"input_tokens\":" << input_tokens
            << ",\"zero_full_requests\":" << zero_full << ",\"full_tokens\":" << full.json()
            << ",\"joint_tokens\":" << joint.json() << ",\"gap_tokens\":" << gap.json()
            << ",\"joint_full_ratio_mean\":";
        if (requests > zero_full)
            out << ratio_sum / (requests - zero_full);
        else
            out << "null";
        out << ",\"joint_full_ratio_weighted\":";
        if (full.sum)
            out << double(joint.sum) / full.sum;
        else
            out << "null";
        out << ",\"hit_rate\":" << (input_tokens ? double(joint.sum) / input_tokens : 0) << ",\"evictions\":{";
        bool comma = false;
        for (const auto& [name, n] : evictions) {
            if (comma)
                out << ',';
            comma = true;
            out << '"' << name << "\":" << n;
        }
        out << "}}";
        return out.str();
    }
    void save(StateWriter& w) const {
        w.scalar(requests);
        w.scalar(input_tokens);
        w.scalar(zero_full);
        w.scalar(ratio_sum);
        full.save(w);
        joint.save(w);
        gap.save(w);
        w.scalar(uint64_t(evictions.size()));
        for (const auto& [name, n] : evictions) {
            w.text(name);
            w.scalar(n);
        }
    }
    void restore(StateReader& r) {
        requests     = r.scalar<uint64_t>();
        input_tokens = r.scalar<uint64_t>();
        zero_full    = r.scalar<uint64_t>();
        ratio_sum    = r.scalar<double>();
        full.restore(r);
        joint.restore(r);
        gap.restore(r);
        const auto count = r.scalar<uint64_t>();
        stateRequire(count < 10000, "invalid eviction series count");
        for (uint64_t i = 0; i < count; ++i) {
            const auto name = r.text();
            evictions[name] = r.scalar<uint64_t>();
        }
    }
};
struct EvaluationStats {
    uint64_t                        window_ns{300000000000ULL};
    MatchWindow                     total;
    std::map<uint64_t, MatchWindow> timeline;
    void add(uint64_t ts, uint64_t tokens, uint64_t f, uint64_t j, const std::map<std::string, uint64_t>& events) {
        total.add(tokens, f, j, events);
        timeline[ts / window_ns * window_ns].add(tokens, f, j, events);
    }
    std::string json() const {
        std::ostringstream out;
        out << "{\"window_ns\":" << window_ns << ",\"total\":" << total.json() << ",\"timeline\":[";
        bool comma = false;
        for (const auto& [time, window] : timeline) {
            if (comma)
                out << ',';
            comma = true;
            out << "{\"start_ns\":" << time << ",\"metrics\":" << window.json() << '}';
        }
        out << "]}";
        return out.str();
    }
    void save(StateWriter& w) const {
        w.scalar(window_ns);
        total.save(w);
        w.scalar(uint64_t(timeline.size()));
        for (const auto& [ts, window] : timeline) {
            w.scalar(ts);
            window.save(w);
        }
    }
    void restore(StateReader& r) {
        window_ns = r.scalar<uint64_t>();
        stateRequire(window_ns > 0, "invalid time window");
        total.restore(r);
        const auto count = r.scalar<uint64_t>();
        stateRequire(count <= total.requests, "invalid timeline count");
        for (uint64_t i = 0; i < count; ++i) {
            const auto time = r.scalar<uint64_t>();
            timeline[time].restore(r);
        }
    }
};
}  // namespace rtp_llm::benchmark
