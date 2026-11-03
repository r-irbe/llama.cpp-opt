#pragma once

#include <charconv>
#include <cstddef>
#include <system_error>

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
