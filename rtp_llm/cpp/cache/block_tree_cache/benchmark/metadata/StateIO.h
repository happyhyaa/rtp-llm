#pragma once
#include <cstdint>
#include <algorithm>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rtp_llm::benchmark {
inline void stateRequire(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
class StateWriter {
public:
    explicit StateWriter(std::ostream& stream): stream_(stream) {}
    void bytes(const char* data, size_t size) {
        stream_.write(data, size);
        stateRequire(bool(stream_), "checkpoint write failed");
        for (size_t i = 0; i < size; ++i)
            hash_ = (hash_ ^ uint8_t(data[i])) * 1099511628211ULL;
    }
    template<class T>
    void scalar(T value) {
        static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
        bytes(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    template<class T>
    void vector(const std::vector<T>& values) {
        scalar(uint64_t(values.size()));
        if (!values.empty())
            bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    }
    void text(const std::string& value) {
        scalar(uint64_t(value.size()));
        bytes(value.data(), value.size());
    }
    void finish() {
        stream_.write(reinterpret_cast<const char*>(&hash_), sizeof(hash_));
        stream_.flush();
        stateRequire(bool(stream_), "checkpoint flush failed");
    }

private:
    std::ostream& stream_;
    uint64_t      hash_{14695981039346656037ULL};
};
class StateReader {
public:
    explicit StateReader(std::istream& stream): stream_(stream) {
        const auto begin = stream.tellg();
        stream.seekg(0, std::ios::end);
        const auto end = stream.tellg();
        stateRequire(begin >= 0 && end - begin >= 8, "truncated checkpoint");
        remaining_ = static_cast<uint64_t>(end - begin) - 8;
        stream.seekg(begin);
        uint64_t hash = 14695981039346656037ULL;
        char     buffer[65536];
        uint64_t left = remaining_;
        while (left) {
            const size_t count = std::min<uint64_t>(left, sizeof(buffer));
            stream.read(buffer, count);
            stateRequire(bool(stream), "truncated checkpoint payload");
            for (size_t i = 0; i < count; ++i)
                hash = (hash ^ uint8_t(buffer[i])) * 1099511628211ULL;
            left -= count;
        }
        uint64_t expected{};
        stream.read(reinterpret_cast<char*>(&expected), sizeof(expected));
        stateRequire(bool(stream) && hash == expected, "checkpoint checksum mismatch");
        stream.clear();
        stream.seekg(begin);
    }
    template<class T>
    T scalar() {
        T value{};
        bytes(reinterpret_cast<char*>(&value), sizeof(value));
        return value;
    }
    void bytes(char* data, size_t count) {
        stateRequire(count <= remaining_, "checkpoint bounds exceeded");
        stream_.read(data, count);
        stateRequire(bool(stream_), "truncated checkpoint");
        remaining_ -= count;
    }
    template<class T>
    std::vector<T> vector() {
        const auto count = scalar<uint64_t>();
        stateRequire(count <= remaining_ / sizeof(T), "invalid checkpoint vector size");
        std::vector<T> values(count);
        if (count)
            bytes(reinterpret_cast<char*>(values.data()), count * sizeof(T));
        return values;
    }
    std::string text() {
        const auto count = scalar<uint64_t>();
        stateRequire(count <= remaining_, "invalid checkpoint text size");
        std::string value(count, '\0');
        if (count)
            bytes(value.data(), count);
        return value;
    }
    void finish() {
        stateRequire(remaining_ == 0, "trailing checkpoint state");
    }

private:
    std::istream& stream_;
    uint64_t      remaining_{0};
};
}  // namespace rtp_llm::benchmark
