#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "kv_cache_manager/common/cache/advanced_cache.h"
#include "kv_cache_manager/common/cache/cache.h"

namespace finite_eval {

class BoundedLru {
public:
    explicit BoundedLru(size_t capacity)
        : capacity_(capacity)
        , cache_(kv_cache_manager::NewLRUCache(
              capacity, 0, false, false, 0.0, nullptr, false, kv_cache_manager::kDontChargeCacheMetadata)) {
        if (!capacity)
            throw std::invalid_argument("capacity must be positive");
    }
    bool contains(int64_t key) const { return cache_->Exists(view(key)); }
    bool touch(int64_t key) {
        auto *handle = cache_->Lookup(view(key));
        if (!handle)
            return false;
        cache_->Release(handle);
        return true;
    }
    void upsert(int64_t key) {
        if (touch(key))
            return;
        if (cache_->Insert(view(key), nullptr, &helper_, 1) != kv_cache_manager::EC_OK)
            throw std::runtime_error("upstream LRU insertion failed");
        if (size_ < capacity_)
            ++size_;
        else
            ++evictions_;
    }
    size_t size() const { return size_; }
    uint64_t evictions() const { return evictions_; }
    void validate() const {
        if (cache_->GetOccupancyCount() != size_ || cache_->GetUsage() != size_ || size_ > capacity_)
            throw std::runtime_error("upstream capacity accounting mismatch");
    }

private:
    static std::string_view view(const int64_t &key) { return {reinterpret_cast<const char *>(&key), sizeof(key)}; }
    inline static const kv_cache_manager::Cache::CacheItemHelper helper_{kv_cache_manager::CacheEntryRole::kMisc,
                                                                         nullptr};
    size_t capacity_;
    size_t size_ = 0;
    uint64_t evictions_ = 0;
    std::shared_ptr<kv_cache_manager::Cache> cache_;
};

class FullCache {
public:
    explicit FullCache(size_t capacity) : cache_(capacity) {}
    size_t process(const std::vector<int64_t> &keys) {
        size_t prefix = 0;
        while (prefix < keys.size() && cache_.contains(keys[prefix]))
            ++prefix;
        // Same complete-key, tail-to-head commit contract as LiteHit.
        for (auto it = keys.rbegin(); it != keys.rend(); ++it)
            cache_.upsert(*it);
        return prefix;
    }
    size_t size() const { return cache_.size(); }
    uint64_t evictions() const { return cache_.evictions(); }
    void validate() const { cache_.validate(); }

private:
    BoundedLru cache_;
};

class SwaCache {
public:
    // n==0 denotes no per-request truncation, not infinite pool capacity.
    SwaCache(size_t capacity, size_t n) : cache_(capacity), n_(n) {}
    size_t process_parts(const std::vector<int64_t> &prefix,
                         const std::vector<int64_t> &tail,
                         size_t prefix_limit = SIZE_MAX) {
        size_t hit = std::min(prefix.size(), prefix_limit);
        while (hit > 0 && !cache_.contains(prefix[hit - 1]))
            --hit;
        if (hit && !cache_.touch(prefix[hit - 1]))
            throw std::runtime_error("selected SWA key disappeared");
        const size_t count = n_ == 0 ? tail.size() : std::min(n_, tail.size());
        for (size_t i = 0; i < count; ++i)
            cache_.upsert(tail[tail.size() - 1 - i]);
        return hit;
    }
    size_t process(const std::vector<int64_t> &keys, size_t full_hit) { return process_parts(keys, keys, full_hit); }
    bool contains(int64_t key) const { return cache_.contains(key); }
    size_t size() const { return cache_.size(); }
    uint64_t evictions() const { return cache_.evictions(); }
    void validate() const { cache_.validate(); }

private:
    BoundedLru cache_;
    size_t n_;
};
} // namespace finite_eval
