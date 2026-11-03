#pragma once

#include <charconv>
#include <cstddef>
#include <cstring>
#include <system_error>

enum class ggml_cuda_pressure_policy {
    adaptive,
    latency_first,
};

inline bool ggml_cuda_parse_pressure_policy(const char * env_val, ggml_cuda_pressure_policy & out) {
    out = ggml_cuda_pressure_policy::adaptive;
    if (env_val == nullptr || *env_val == '\0') {
        return true;
    }
    if (std::strcmp(env_val, "adaptive") == 0) {
        out = ggml_cuda_pressure_policy::adaptive;
        return true;
    }
    if (std::strcmp(env_val, "latency-first") == 0 || std::strcmp(env_val, "latency_first") == 0) {
        out = ggml_cuda_pressure_policy::latency_first;
        return true;
    }
    return false;
}

struct ggml_cuda_memory_headroom {
    size_t free_bytes    = 0;
    size_t total_bytes   = 0;
    size_t reserve_bytes = 512ULL * 1024 * 1024;
    bool   valid         = false;
};

inline bool ggml_cuda_graph_under_pressure(const ggml_cuda_memory_headroom & h) {
    return h.valid && h.free_bytes < h.reserve_bytes;
}

// No universal count-based working-set assumption. Zero means no count limit;
// pressure-based admission is a separate policy, not inferred from this value.
inline bool ggml_cuda_graph_cache_limit(const char * value, size_t & limit) {
    limit = 0;
    if (value == nullptr) { return true; }
    if (*value == '\0') { return false; }
    const char * end = value;
    for (; *end; ++end) {
        if (*end < '0' || *end > '9') { return false; }
    }
    size_t parsed = 0;
    const auto result = std::from_chars(value, end, parsed);
    if (result.ec != std::errc() || result.ptr != end) { return false; }
    limit = parsed;
    return true;
}

inline bool ggml_cuda_graph_cache_full(size_t entries, size_t limit) {
    return limit != 0 && entries >= limit;
}

// Caller excludes conflicting submissions/captures. Fence chooses the owning
// device and waits for all uses; failed fencing must not invalidate pointers or
// destroy anything. Clear the active pointer before resource destructors run.
template <typename Context, typename Fence>
bool ggml_cuda_graph_retire(Context & ctx, Fence && fence) {
    if (!fence(ctx.device)) { return false; }
    ctx.cur_graph = nullptr;
    ctx.cuda_graphs.clear();
    return true;
}

// Select a victim without changing the cache. The caller must synchronize the
// owning device before erasing entries: kernels may still use their resources.
// Keep the currently prepared graph alive while reclaiming other entries.
template <typename Cache, typename Graph>
auto ggml_cuda_graph_cache_lru(Cache & cache, const Graph * keep) -> decltype(cache.begin()) {
    auto victim = cache.end();
    for (auto it = cache.begin(); it != cache.end(); ++it) {
        if (it->second.get() == keep) {
            continue;
        }
        if (victim == cache.end() || it->second->last_used < victim->second->last_used) {
            victim = it;
        }
    }
    return victim;
}

// nullptr has no deducible pointee type.
template <typename Cache>
auto ggml_cuda_graph_cache_lru(Cache & cache, decltype(nullptr)) -> decltype(cache.begin()) {
    using Graph = typename Cache::mapped_type::element_type;
    return ggml_cuda_graph_cache_lru(cache, static_cast<const Graph *>(nullptr));
}

// Reclaim up to max_evict LRU entries (excluding keep) under memory pressure.
// Under adaptive policy, reclaims up to max_evict victims to recover headroom.
// Under latency_first policy, caps single-step reclamation to at most 1 victim
// to bound eviction latency while preserving forward progress.
template <typename Cache, typename Graph>
size_t ggml_cuda_graph_pressure_reclaim(
        Cache & cache,
        const Graph * keep,
        ggml_cuda_pressure_policy policy,
        size_t max_evict) {
    if (max_evict == 0) {
        return 0;
    }
    const size_t budget = (policy == ggml_cuda_pressure_policy::latency_first)
        ? 1
        : max_evict;
    size_t evicted = 0;
    while (evicted < budget) {
        auto victim = ggml_cuda_graph_cache_lru(cache, keep);
        if (victim == cache.end()) {
            break;
        }
        cache.erase(victim);
        ++evicted;
    }
    return evicted;
}

template <typename Cache>
size_t ggml_cuda_graph_pressure_reclaim(
        Cache & cache,
        decltype(nullptr),
        ggml_cuda_pressure_policy policy,
        size_t max_evict) {
    using Graph = typename Cache::mapped_type::element_type;
    return ggml_cuda_graph_pressure_reclaim(cache, static_cast<const Graph *>(nullptr), policy, max_evict);
}
