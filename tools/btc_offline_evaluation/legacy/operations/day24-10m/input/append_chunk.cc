#include <iostream>
#include <limits>
#include <tuple>
#include "finite_io.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"

namespace fs = std::filesystem;
using namespace finite_eval;
using Cursor = std::tuple<uint64_t, uint64_t, uint64_t>;

void check(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

Cursor cursor(const std::array<std::string_view, 6>& fields) {
    return {number(fields[0]), number(fields[1]), number(fields[2])};
}

void write_cursor(std::ostream& out, const Cursor& c) {
    out << "{\"timestamp_ns\":" << std::get<0>(c) << ",\"file_index\":" << std::get<1>(c)
        << ",\"source_line\":" << std::get<2>(c) << "}";
}

class SortedSource {
public:
    SortedSource(fs::path dir, uint64_t index, uint64_t offset): dir_(dir), index_(index) {
        open();
        stream_.seekg(offset);
        check(bool(stream_), "source offset");
    }
    bool next() {
        for (;;) {
            offset = stream_.tellg();
            if (std::getline(stream_, line)) return true;
            check(stream_.eof() && !stream_.bad(), "source read error");
            if (++index_ >= 24) return false;
            open();
        }
    }
    uint64_t index() const { return index_; }
    uint64_t offset = 0;
    std::string line;
private:
    void open() {
        std::ostringstream file;
        file << std::setw(3) << std::setfill('0') << index_ << ".sorted.tsv";
        stream_ = std::ifstream(dir_/file.str(), std::ios::binary);
        check(stream_.good(), "missing sorted source");
    }
    fs::path dir_;
    uint64_t index_;
    std::ifstream stream_;
};

class Canonical {
public:
    explicit Canonical(fs::path dir): dir_(dir) {}
    const Record& at(uint64_t ordinal) {
        if (!input_.is_open() || ordinal % 100000 == 0) {
            if (input_.is_open()) check(input_.peek() == EOF && !input_.bad(), "trailing canonical bytes");
            input_ = std::ifstream(dir_/chunk_name(ordinal/100000), std::ios::binary);
            char magic[8];
            uint64_t reserved, count;
            input_.read(magic, 8);
            read(input_, reserved);
            read(input_, count);
            check(!std::memcmp(magic, "FNLAST01", 8) && reserved == 0 && count == 100000, "canonical header");
            for (uint64_t i = 0; i < ordinal%100000; ++i) read_record(input_, record_);
        }
        read_record(input_, record_);
        if ((ordinal+1)%100000 == 0)
            check(input_.peek() == EOF && !input_.bad(), "canonical boundary");
        return record_;
    }
private:
    fs::path dir_;
    std::ifstream input_;
    Record record_;
};

int main(int argc, char** argv) {
    try {
        check(argc == 18, "usage: append_chunk PREP REF STAGED DAY_TOTAL PRIOR COUNT TOKENS SOURCE_INDEX SOURCE_OFFSET META_OFFSET LAST_TS LAST_FILE LAST_LINE NEXT_TS NEXT_FILE NEXT_LINE COMPLETE_TOKENS");
        check_endian();
        const auto total_day = number(argv[4]), prior = number(argv[5]), count = number(argv[6]);
        check(count && count <= 100000 && prior >= 10000000 && prior+count <= total_day, "invalid bounds");
        check(prior%100000 == 0 && (count == 100000 || prior+count == total_day),
              "only the final day chunk may be short");
        SortedSource source(argv[1], number(argv[8]), number(argv[9]));
        fs::path ref(argv[2]);
        Canonical canonical(ref/"chunks");
        std::ifstream meta(ref/"requests.meta.tsv", std::ios::binary);
        meta.seekg(number(argv[10]));
        std::string meta_line;
        uint64_t meta_offset = 0;
        auto read_metadata = [&]() {
            meta_offset = meta.tellg();
            check(bool(std::getline(meta, meta_line)), "metadata truncated");
            return split_row(meta_line);
        };
        Cursor last{number(argv[11]),number(argv[12]),number(argv[13])};
        Cursor next{number(argv[14]),number(argv[15]),number(argv[16])};
        check(source.next() && cursor(split_row(source.line)) == last && cursor(read_metadata()) == last,
              "predecessor cursor mismatch");
        std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write("BTCPFX01", 8);
        write(out, uint32_t(256));
        write(out, count);
        write(out, uint64_t(0));
        write(out, uint32_t(6));
        out.write("global", 6);
        uint64_t tokens = 0;
        for (uint64_t i = 0; i < count; ++i) {
            check(source.next(), "source truncated");
            const auto f = split_row(source.line), m = read_metadata();
            const auto c = cursor(f);
            const auto ts = std::get<0>(c), len = number(f[3]);
            check(c > last && (i || c == next) && std::get<1>(c) == source.index() && std::get<2>(c) &&
                  ts >= 1793030400000000000ULL && ts < 1793116800000000000ULL, "source order or day mismatch");
            check(std::equal(f.begin(), f.begin()+5, m.begin()) && m[5] == "0", "request metadata mismatch");
            check(len && len <= INT32_MAX && !f[4].empty() && f[4].size() <= 1048576, "request length/ID");
            auto keys = kv_cache_manager::ApplyPrefixHash(parse_keys(f[5]));
            const auto& record = canonical.at(prior+i);
            check(record.timestamp == ts && record.input_tokens == len && !record.full_hit &&
                  keys.size() == len/256, "reference dimensions");
            for (size_t j=0; j<record.keys.size(); ++j)
                check(record.keys[j] == keys[4*j+3], "canonical key mismatch");
            write(out, ts);
            write(out, static_cast<uint32_t>(len));
            write(out, static_cast<uint32_t>(f[4].size()));
            out.write(f[4].data(), f[4].size());
            if (!keys.empty()) out.write(reinterpret_cast<const char*>(keys.data()), keys.size()*8);
            tokens += len;
            last = c;
        }
        out.seekp(20);
        write(out, tokens);
        out.close();
        auto last_meta_offset = meta_offset;
        auto cumulative = tokens + number(argv[7]);
        std::cout << "{\"requests\":" << prior+count << ",\"input_tokens\":" << cumulative
                  << ",\"chunk_requests\":" << count << ",\"chunk_input_tokens\":" << tokens
                  << ",\"source_offset\":" << source.offset << ",\"meta_offset\":" << last_meta_offset
                  << ",\"last\":";
        write_cursor(std::cout, last);
        std::cout << ",\"next\":";
        if (prior+count == total_day) {
            check(cumulative == number(argv[17]), "full day token total mismatch");
            std::cout << "null";
        } else {
            auto successor = cursor(read_metadata());
            check(successor > last && std::get<0>(successor) < 1793116800000000000ULL, "invalid successor");
            write_cursor(std::cout, successor);
        }
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
