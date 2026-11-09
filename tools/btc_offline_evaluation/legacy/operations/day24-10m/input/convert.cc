#include <iostream>
#include <tuple>
#include "finite_io.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"

namespace fs = std::filesystem;
using namespace finite_eval;
using Cursor = std::tuple<uint64_t, uint64_t, uint64_t>;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct Source {
    fs::path prepared;
    std::ifstream stream;
    int index = -1;
    std::string line;
    Cursor current, previous{};
    bool next() {
        while (!stream.is_open() || !std::getline(stream, line)) {
            require(!stream.bad(), "source read failed");
            if (++index == 24) return false;
            std::ostringstream name;
            name << std::setw(3) << std::setfill('0') << index << ".sorted.tsv";
            stream = std::ifstream(prepared/name.str());
            require(stream.good(), "missing " + name.str());
        }
        auto f = split_row(line);
        current = {number(f[0]), number(f[1]), number(f[2])};
        require(std::get<0>(current) >= 1793030400000000000ULL &&
                std::get<0>(current) < 1793116800000000000ULL &&
                std::get<1>(current) == static_cast<unsigned>(index) &&
                std::get<2>(current) > 0 && current > previous, "source order/range");
        previous = current;
        return true;
    }
};

std::string cursor_json(const Cursor& c) {
    return "{\"timestamp_ns\":" + std::to_string(std::get<0>(c)) +
           ",\"file_index\":" + std::to_string(std::get<1>(c)) +
           ",\"source_line\":" + std::to_string(std::get<2>(c)) + "}";
}

int main(int argc, char** argv) {
    uint64_t requests = 0, total = 0;
    try {
        require(argc == 4, "usage: convert PREPARED REFERENCE OUTPUT");
        check_endian();
        Source src;
        src.prepared = argv[1];
        fs::path ref(argv[2]), dest(argv[3]);
        std::ifstream meta(ref/"requests.meta.tsv");
        Record canonical;
        std::string metadata;
        constexpr uint64_t per_chunk = 100000;
        for (int chunk = 0; chunk < 100; ++chunk) {
            fs::path name(chunk_name(chunk));
            std::ifstream reference(ref/"chunks"/name, std::ios::binary);
            char magic[8];
            uint64_t reserved, count;
            reference.read(magic, sizeof(magic));
            read(reference, reserved);
            read(reference, count);
            require(!std::memcmp(magic, "FNLAST01", 8) && reserved == 0 && count == per_chunk, "reference header");
            auto final = dest/name.replace_extension("btcbin");
            require(!fs::exists(final), "refuse overwrite " + final.string());
            auto tmp = final.string() + ".tmp";
            std::ofstream out(tmp, std::ios::binary);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out.write("BTCPFX01", 8);
            write(out, uint32_t(256));
            write(out, count);
            write(out, uint64_t(0));
            write(out, uint32_t(6));
            out.write("global", 6);
            uint64_t tokens = 0;
            for (uint64_t i = 0; i < count; ++i) {
                require(src.next() && bool(std::getline(meta, metadata)), "unexpected source/metadata EOF");
                auto f = split_row(src.line), m = split_row(metadata);
                require(std::equal(f.begin(), f.begin()+5, m.begin()) && m[5] == "0", "metadata mismatch");
                auto ts = std::get<0>(src.current), len = number(f[3]);
                require(len && len <= INT32_MAX && !f[4].empty() && f[4].size() <= 1048576, "length/ID");
                auto keys = kv_cache_manager::ApplyPrefixHash(parse_keys(f[5]));
                read_record(reference, canonical);
                require(canonical.timestamp == ts && canonical.input_tokens == len && !canonical.full_hit &&
                        keys.size() == len/256, "reference dimensions");
                for (size_t k = 0; k < canonical.keys.size(); ++k)
                    require(canonical.keys[k] == keys[4*k+3], "canonical key mismatch");
                write(out, ts);
                write(out, static_cast<uint32_t>(len));
                write(out, static_cast<uint32_t>(f[4].size()));
                out.write(f[4].data(), f[4].size());
                if (!keys.empty()) out.write(reinterpret_cast<const char*>(keys.data()), keys.size()*8);
                tokens += len;
                ++requests;
            }
            require(reference.peek() == EOF && !reference.bad(), "trailing reference data");
            out.seekp(20);
            write(out, tokens);
            out.close();
            total += tokens;
            fs::rename(tmp, final);
            std::cout << chunk << " " << requests << " " << total << " " << cursor_json(src.current) << std::endl;
        }
        auto last = src.current;
        require(src.next(), "missing next cursor");
        atomic_text(dest/"cohort.json","{\"requests\":" + std::to_string(requests) +
                    ",\"input_tokens\":" + std::to_string(total) + ",\"last\":" + cursor_json(last) +
                    ",\"next\":" + cursor_json(src.current) + ",\"verified_chunks\":100}\n");
    } catch (const std::exception& error) {
        std::cerr << "after " << requests << " requests: " << error.what() << std::endl;
        return 1;
    }
}
