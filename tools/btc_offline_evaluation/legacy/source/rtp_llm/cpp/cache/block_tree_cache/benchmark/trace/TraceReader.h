#pragma once
#include <cstdint>
#include <cstring>
#include <istream>
#include <stdexcept>
#include <vector>
namespace rtp_llm::benchmark {
struct SerialTraceRequest {
    uint64_t             timestamp_ns{0};
    uint64_t             input_tokens{0};
    std::vector<int64_t> keys;
};
class TraceChunkReader {
public:
    TraceChunkReader(std::istream& input, uint32_t tokens_per_key): input_(input), tokens_per_key_(tokens_per_key) {
        uint16_t endian = 1;
        if (*reinterpret_cast<char*>(&endian) != 1 || tokens_per_key == 0) {
            throw std::runtime_error("trace requires little endian and a positive token stride");
        }
        char magic[8];
        input_.read(magic, 8);
        if (!input_ || std::memcmp(magic, "FNLAST01", 8)) {
            throw std::runtime_error("unsupported trace format");
        }
        (void)read<uint64_t>();
        remaining_ = read<uint64_t>();
    }
    bool next(SerialTraceRequest& request) {
        if (!remaining_) {
            if (input_.peek() != std::char_traits<char>::eof()) {
                throw std::runtime_error("trailing trace bytes");
            }
            return false;
        }
        request.timestamp_ns = read<uint64_t>();
        request.input_tokens = read<uint64_t>();
        const auto count     = read<uint32_t>();
        (void)read<uint32_t>();  // Prior theoretical hit is never a BTC input.
        if (!request.input_tokens || count != request.input_tokens / tokens_per_key_ || count > 1048576) {
            throw std::runtime_error("trace token/key dimensions do not match configured canonical stride");
        }
        request.keys.resize(count);
        input_.read(reinterpret_cast<char*>(request.keys.data()), count * sizeof(int64_t));
        if (!input_) {
            throw std::runtime_error("truncated trace keys");
        }
        --remaining_;
        return true;
    }

private:
    template<class T>
    T read() {
        T value;
        input_.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!input_) {
            throw std::runtime_error("truncated trace record");
        }
        return value;
    }
    std::istream& input_;
    uint32_t      tokens_per_key_;
    uint64_t      remaining_{0};
};
}  // namespace rtp_llm::benchmark
