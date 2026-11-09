#include <cstring>
#include <sstream>
#include <stdexcept>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/trace/TraceReader.h"
using namespace rtp_llm::benchmark;
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("test assertion failed");
}
template<class T>
void put(std::ostream& s, T v) {
    s.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
int main() {
    std::stringstream s;
    s.write("FNLAST01", 8);
    put(s, uint64_t(0));
    put(s, uint64_t(2));
    put(s, uint64_t(1));
    put(s, uint64_t(2057));
    put(s, uint32_t(2));
    put(s, uint32_t(0));
    put(s, int64_t(-3));
    put(s, int64_t(7));
    put(s, uint64_t(2));
    put(s, uint64_t(17));
    put(s, uint32_t(0));
    put(s, uint32_t(0));
    TraceChunkReader   reader(s, 1024);
    SerialTraceRequest r;
    require(reader.next(r));
    require(r.input_tokens == 2057 && r.keys[0] == -3 && r.keys[1] == 7);
    require(reader.next(r) && r.keys.empty() && r.input_tokens == 17);
    require(!reader.next(r));
    auto truncated = s.str();
    truncated.pop_back();
    std::stringstream bad(truncated);
    bool              failed = false;
    try {
        TraceChunkReader t(bad, 1024);
        while (t.next(r)) {};
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed);
    std::stringstream wrong(s.str());
    failed = false;
    try {
        TraceChunkReader t(wrong, 256);
        t.next(r);
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed);
    std::stringstream extra(s.str() + "x");
    failed = false;
    try {
        TraceChunkReader t(extra, 1024);
        while (t.next(r)) {};
    } catch (const std::runtime_error&) {
        failed = true;
    }
    require(failed);
}
