#include "finite_core.h"
#include "finite_io.h"

#include <iostream>
#include <thread>
#include <unistd.h>

using namespace finite_eval;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: hold_worker SOURCE EMPTY_OUTPUT");
        check_endian();
        const fs::path source(argv[1]), out(argv[2]);
        if (!fs::create_directory(out)) throw std::runtime_error("output already exists");
        constexpr uint64_t start_ns = 1793030400000000000ULL, end_ns = start_ns + 86400000000000ULL;
        constexpr size_t capacity = 3218236;
        FullCache full(capacity);
        SwaCache swa(capacity, 0, 1);
        uint64_t requests = 0, input = 0, joint = 0, full_hits = 0, first = 0, last = 0;
        uint64_t chunk = 0, row = 0;
        auto started = Clock::now();
        Clock::duration paused{};
        auto report = [&](const char* state) {
            std::ostringstream s;
            s << std::setprecision(17) << "{\"state\":\"" << state << "\",\"pid\":" << getpid()
              << ",\"requests\":" << requests << ",\"input_tokens\":" << input << ",\"hit_tokens\":" << joint
              << ",\"full_hit_tokens\":" << full_hits << ",\"hit_rate\":" << double(joint) / input
              << ",\"first_ns\":" << first << ",\"last_ns\":" << last << ",\"next_chunk\":" << chunk
              << ",\"next_record_in_chunk\":" << row << ",\"resident_full\":" << full.size()
              << ",\"resident_swa\":" << swa.size() << ",\"elapsed_active_seconds\":"
              << std::chrono::duration<double>(Clock::now() - started - paused).count() << "}\n";
            return s.str();
        };
        Record record;
        for (;;) {
            std::ifstream in(source / "chunks" / chunk_name(chunk), std::ios::binary);
            char magic[8];
            uint64_t source_capacity, count;
            in.read(magic, 8);
            if (!in || std::memcmp(magic, "FNLAST01", 8)) throw std::runtime_error("invalid chunk");
            read(in, source_capacity);
            read(in, count);
            if (chunk < 100 && count != 100000) throw std::runtime_error("unexpected cohort chunk length");
            for (row = 0; row < count; ++row) {
                read_record(in, record);
                if (record.timestamp < start_ns || record.timestamp < last) throw std::runtime_error("time order");
                if (record.timestamp >= end_ns) {
                    if (requests < 10000000) throw std::runtime_error("short day");
                    atomic_text(out / "day24.json", report("complete"));
                    return 0;
                }
                if (!requests) first = record.timestamp;
                last = record.timestamp;
                auto f = full.process(record.keys);
                auto h = swa.process(record.keys, f);
                if (h > f || f > record.keys.size()) throw std::runtime_error("invalid hit bounds");
                full_hits += f * kBlockTokens;
                joint += h * kBlockTokens;
                input += record.input_tokens;
                ++requests;
            }
            if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing data");
            full.validate();
            swa.validate();
            ++chunk;
            row = 0;
            atomic_text(out / "progress.json", report("running"));
            if (requests == 10000000) {
                atomic_text(out / "ten-million.json", report("held"));
                auto hold_start = Clock::now();
                while (!fs::exists(out / "resume-to-day-end")) std::this_thread::sleep_for(std::chrono::seconds(5));
                paused += Clock::now() - hold_start;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
