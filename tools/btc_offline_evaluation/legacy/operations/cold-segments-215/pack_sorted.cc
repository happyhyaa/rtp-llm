// Encode the historical BTCPFX01 format. The caller verifies each SHA against the original day catalog.
#include "finite_io.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"
#include <iostream>

int main(int argc, char** argv) {
    using namespace finite_eval;
    if (argc != 2 && argc != 4) return 2;
    const unsigned first = argc == 4 ? number(argv[2]) : 0;
    const unsigned end = argc == 4 ? number(argv[3]) : 600;
    if (first >= end || end > 600) return 2;
    check_endian();
    const std::filesystem::path dest(argv[1]);
    std::string line;
    for (unsigned chunk = first; chunk < end; ++chunk) {
        auto file = dest / chunk_name(chunk);
        file.replace_extension(".btcbin");
        const auto tmp = file.string() + ".tmp";
        std::ofstream out(tmp, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.write("BTCPFX01", 8);
        write(out, uint32_t(256));
        write(out, uint64_t(100000));
        write(out, uint64_t(0));
        write(out, uint32_t(6));
        out.write("global", 6);
        uint64_t tokens = 0;
        for (unsigned i = 0; i < 100000; ++i) {
            if (!std::getline(std::cin, line)) throw std::runtime_error("short input");
            auto fields = split_row(line);
            auto length = number(fields[3]);
            auto keys = kv_cache_manager::ApplyPrefixHash(parse_keys(fields[5]));
            if (keys.size() != length / 256) throw std::runtime_error("invalid length");
            write(out, number(fields[0]));
            write(out, static_cast<uint32_t>(length));
            write(out, static_cast<uint32_t>(fields[4].size()));
            out.write(fields[4].data(), fields[4].size());
            if (!keys.empty()) out.write(reinterpret_cast<const char*>(keys.data()), keys.size()*8);
            tokens += length;
        }
        out.seekp(20);
        write(out, tokens);
        out.close();
        std::filesystem::rename(tmp, file);
    }
}
