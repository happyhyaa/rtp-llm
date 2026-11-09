#pragma once
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace finite_eval {
constexpr uint32_t kBlockTokens = 1024;
inline uint64_t number(std::string_view text, int base = 10) {
    uint64_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::runtime_error("invalid input number");
    return value;
}
template <class T>
void write(std::ostream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
template <class T>
void read(std::istream &in, T &value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in)
        throw std::runtime_error("truncated binary chunk");
}
inline void check_endian() {
    uint32_t value = 1;
    if (*reinterpret_cast<char *>(&value) != 1)
        throw std::runtime_error("little-endian host required");
}
inline std::string chunk_name(uint64_t i) {
    std::ostringstream o;
    o << std::setw(6) << std::setfill('0') << i << ".bin";
    return o.str();
}
inline void atomic_text(const std::filesystem::path &path, const std::string &content) {
    auto temp = path.string() + ".tmp";
    std::ofstream out(temp);
    out << content;
    out.close();
    if (!out)
        throw std::runtime_error("output write failed");
    std::filesystem::rename(temp, path);
}
struct Record {
    uint64_t timestamp = 0, input_tokens = 0;
    uint32_t full_hit = 0;
    std::vector<int64_t> keys;
};
inline void write_record(std::ostream &out, const Record &record) {
    uint32_t count = static_cast<uint32_t>(record.keys.size());
    write(out, record.timestamp);
    write(out, record.input_tokens);
    write(out, count);
    write(out, record.full_hit);
    out.write(reinterpret_cast<const char *>(record.keys.data()), count * sizeof(int64_t));
}
inline void read_record(std::istream &in, Record &record) {
    uint32_t count;
    read(in, record.timestamp);
    read(in, record.input_tokens);
    read(in, count);
    read(in, record.full_hit);
    if (count != record.input_tokens / kBlockTokens || record.full_hit > count)
        throw std::runtime_error("invalid chunk record dimensions");
    record.keys.resize(count);
    in.read(reinterpret_cast<char *>(record.keys.data()), count * sizeof(int64_t));
    if (!in)
        throw std::runtime_error("truncated chunk keys");
}
inline std::array<std::string_view, 6> split_row(const std::string &line) {
    std::array<std::string_view, 6> fields;
    std::string_view view(line);
    size_t start = 0;
    for (size_t i = 0; i < 5; ++i) {
        auto end = view.find('\t', start);
        if (end == std::string_view::npos)
            throw std::runtime_error("missing replay field");
        fields[i] = view.substr(start, end - start);
        start = end + 1;
    }
    fields[5] = view.substr(start);
    return fields;
}
inline std::vector<int64_t> parse_keys(std::string_view text) {
    std::vector<int64_t> keys;
    keys.reserve((text.size() + 1) / 17);
    size_t pos = 0;
    while (pos < text.size()) {
        auto end = text.find(',', pos);
        auto hex = text.substr(pos, end == std::string_view::npos ? end : end - pos);
        if (hex.size() != 16)
            throw std::runtime_error("invalid key width");
        auto bits = number(hex, 16);
        int64_t key;
        std::memcpy(&key, &bits, sizeof(key));
        keys.push_back(key);
        if (end == std::string_view::npos)
            break;
        pos = end + 1;
    }
    return keys;
}
} // namespace finite_eval
