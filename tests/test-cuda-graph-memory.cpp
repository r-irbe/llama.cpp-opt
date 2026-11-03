// Model-free tests of the production graph resource owner and LRU policy.
// Mock only the CUDA API; no GPU, driver, or model allocation is needed.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

using cudaGraph_t = void *;
using cudaGraphExec_t = void *;
using cudaGraphNode_t = void *;
struct cudaKernelNodeParams {};
struct ggml_cuda_graph;
static int graphs_freed = 0, execs_freed = 0, tables_freed = 0;
static int selected_device = -1;
static bool fence_complete = false;
static ggml_cuda_graph ** retiring_current = nullptr;
static void check_retirement() {
    if (retiring_current && (selected_device != 2 || !fence_complete || *retiring_current != nullptr)) {
        std::abort();
    }
}
static int cudaGraphDestroy(cudaGraph_t p) { check_retirement(); if (!p) std::abort(); ++graphs_freed; return 0; }
static int cudaGraphExecDestroy(cudaGraphExec_t p) { check_retirement(); if (!p) std::abort(); ++execs_freed; return 0; }
static int cudaFree(void * p) { check_retirement(); if (!p) std::abort(); ++tables_freed; return 0; }
#define CUDA_CHECK(call) do { if ((call) != 0) std::abort(); } while (0)
#define USE_CUDA_GRAPH
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/graph-cache.h"
#include "ggml-cuda/graph-lifecycle.h"

#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond); return 1; \
} } while (0)

int main() {
    {
        ggml_cuda_graph g;
        CHECK(g.dest_ptrs_d == nullptr);
        CHECK(g.polygraph_plan_d == nullptr);
        g.release();
        CHECK(graphs_freed == 0 && execs_freed == 0 && tables_freed == 0);
        g.graph = reinterpret_cast<void *>(1);
        g.instance = reinterpret_cast<void *>(2);
        g.dest_ptrs_d = reinterpret_cast<char **>(3);
        g.polygraph_plan_d = reinterpret_cast<uint32_t *>(5);
        g.polygraph_plan_h[0] = 7;
        g.dest_ptrs_size = 4;
        g.release();
        CHECK(g.graph == nullptr && g.instance == nullptr && g.dest_ptrs_d == nullptr && g.polygraph_plan_d == nullptr);
        CHECK(g.polygraph_plan_h[0] == 0);
        CHECK(g.dest_ptrs_size == 0);
        g.release();
    }
    CHECK(graphs_freed == 1 && execs_freed == 1 && tables_freed == 2);

    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cache;
    const void * hot = reinterpret_cast<void *>(1);
    for (uintptr_t i = 1; i <= 256; ++i) {
        const void * key = reinterpret_cast<void *>(i);
        // Match the production miss path: trim before allocating a new entry.
        while (cache.size() >= 4) {
            auto victim = ggml_cuda_graph_cache_lru(cache, nullptr);
            CHECK(victim != cache.end());
            CHECK(victim->first != hot);
            cache.erase(victim);
        }
        auto entry = std::unique_ptr<ggml_cuda_graph>(new ggml_cuda_graph());
        entry->last_used = 2 * i;
        entry->dest_ptrs_d = reinterpret_cast<char **>(i + 16);
        cache.emplace(key, std::move(entry));
        cache.at(hot)->last_used = 2 * i + 1;
        CHECK(cache.size() <= 4);
    }
    // Memory-pressure trimming keeps the current graph valid until capture
    // fallback has stopped using it, even when it is the oldest entry.
    cache.at(hot)->last_used = 0;
    while (cache.size() > 1) {
        auto victim = ggml_cuda_graph_cache_lru(cache, cache.at(hot).get());
        CHECK(victim != cache.end() && victim->first != hot);
        cache.erase(victim);
    }
    CHECK(ggml_cuda_graph_cache_lru(cache, cache.at(hot).get()) == cache.end());
    CHECK(tables_freed == 257);
    cache.clear();
    CHECK(tables_freed == 258);
    size_t limit = 123;
    CHECK(ggml_cuda_graph_cache_limit(nullptr, limit) && limit == 0);
    CHECK(ggml_cuda_graph_cache_limit("0", limit) && limit == 0);
    CHECK(ggml_cuda_graph_cache_limit("128", limit) && limit == 128);
    CHECK(!ggml_cuda_graph_cache_limit("-1", limit));
    CHECK(!ggml_cuda_graph_cache_limit("64x", limit));
    CHECK(!ggml_cuda_graph_cache_limit("", limit));
    CHECK(!ggml_cuda_graph_cache_limit("184467440737095516160", limit));
    CHECK(ggml_cuda_graph_cache_limit(nullptr, limit));
    for (uintptr_t i = 1; i <= 256; ++i) {
        // The production admission predicate must not trim at an implicit 64.
        CHECK(!ggml_cuda_graph_cache_full(size_t(i - 1), limit));
    }
    CHECK(!ggml_cuda_graph_cache_full(127, 128));
    CHECK(ggml_cuda_graph_cache_full(128, 128));
    size_t hits = 0, evictions = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (uintptr_t i = 1; i <= 128; ++i) {
            const void * key = reinterpret_cast<void *>(i);
            if (cache.find(key) != cache.end()) { ++hits; continue; }
            if (ggml_cuda_graph_cache_full(cache.size(), limit)) {
                cache.erase(ggml_cuda_graph_cache_lru(cache, nullptr));
                ++evictions;
            }
            cache.emplace(key, std::make_unique<ggml_cuda_graph>());
        }
    }
    CHECK(cache.size() == 128 && hits == 128 && evictions == 0);
    cache.clear();

    struct owner {
        int device = 2;
        std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;
        ggml_cuda_graph * cur_graph = nullptr;
    } ctx;
    ctx.cuda_graphs.emplace(hot, std::make_unique<ggml_cuda_graph>());
    ctx.cur_graph = ctx.cuda_graphs.at(hot).get();
    ctx.cur_graph->graph = reinterpret_cast<void *>(1);
    ctx.cur_graph->instance = reinterpret_cast<void *>(2);
    ctx.cur_graph->dest_ptrs_d = reinterpret_cast<char **>(3);
    retiring_current = &ctx.cur_graph;
    const int before_retirement = tables_freed;
    CHECK(!ggml_cuda_graph_retire(ctx, [](int device) { return device != 2; }));
    CHECK(ctx.cur_graph != nullptr && ctx.cuda_graphs.size() == 1);
    CHECK(tables_freed == before_retirement);
    bool fenced = false;
    CHECK(ggml_cuda_graph_retire(ctx, [&](int device) {
        selected_device = device;
        fence_complete = true;
        fenced = device == 2;
        return fenced;
    }));
    CHECK(fenced && ctx.cur_graph == nullptr && ctx.cuda_graphs.empty());
    CHECK(ggml_cuda_graph_retire(ctx, [](int device) { return device == 2; }));
    CHECK(tables_freed == before_retirement + 1);
    retiring_current = nullptr;

    ggml_cuda_graph_lifecycle lifecycle;
    std::mutex context_a, context_b;
    {
        auto submission = lifecycle.submission(context_a);
        // Test from a different thread: recursively locking shared_mutex is invalid.
        CHECK(std::async(std::launch::async, [&] {
            auto retirement = lifecycle.try_retirement();
            return !retirement.owns_lock();
        }).get());
        CHECK(std::async(std::launch::async, [&] {
            auto independent = lifecycle.submission(context_b);
            return independent.owns_lock();
        }).get());
        CHECK(std::async(std::launch::async, [&] {
            if (context_a.try_lock()) { context_a.unlock(); return false; }
            return true;
        }).get());
    }
    CHECK(std::async(std::launch::async, [&] {
        auto retirement = lifecycle.try_retirement();
        return retirement.owns_lock();
    }).get());
    lifecycle.begin_capture();
    bool ran = false;
    CHECK(!lifecycle.try_without_capture([&] { ran = true; }));
    CHECK(!ran);
    lifecycle.end_capture();
    CHECK(lifecycle.try_without_capture([&] { ran = true; }));
    CHECK(ran);
    // While a fence is executing, another thread cannot begin a capture.
    CHECK(lifecycle.without_capture([&] {
        return std::async(std::launch::async, [&] {
            return !lifecycle.try_begin_capture();
        }).get();
    }));
    CHECK(lifecycle.try_begin_capture());
    lifecycle.end_capture();
    // Exercise the blocking paths too, not only their nonblocking probes.
    auto active = std::make_unique<ggml_cuda_graph_lifecycle::submission_guard>(lifecycle.submission(context_a));
    std::promise<void> retirement_started;
    auto retirement_ready = retirement_started.get_future();
    auto retiring = std::async(std::launch::async, [&] {
        retirement_started.set_value();
        auto exclusive = lifecycle.retirement();
        return exclusive.owns_lock();
    });
    retirement_ready.wait();
    const bool retirement_blocked = retiring.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    active.reset();
    CHECK(retiring.get() && retirement_blocked);
    lifecycle.begin_capture();
    std::promise<void> fence_started;
    auto fence_ready = fence_started.get_future();
    auto waiting_fence = std::async(std::launch::async, [&] {
        fence_started.set_value();
        return lifecycle.without_capture([] { return true; });
    });
    fence_ready.wait();
    const bool fence_blocked = waiting_fence.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    lifecycle.end_capture();
    CHECK(waiting_fence.get() && fence_blocked);
    bool threw = false;
    try { lifecycle.without_capture([] { throw std::runtime_error("injected fence exception"); }); }
    catch (const std::runtime_error &) { threw = true; }
    CHECK(threw && lifecycle.try_begin_capture());
    lifecycle.end_capture();

    // Adaptive vs latency-first pressure policy parsing and headroom reclamation.
    ggml_cuda_pressure_policy policy = ggml_cuda_pressure_policy::latency_first;
    CHECK(ggml_cuda_parse_pressure_policy(nullptr, policy) && policy == ggml_cuda_pressure_policy::adaptive);
    CHECK(ggml_cuda_parse_pressure_policy("", policy) && policy == ggml_cuda_pressure_policy::adaptive);
    CHECK(ggml_cuda_parse_pressure_policy("adaptive", policy) && policy == ggml_cuda_pressure_policy::adaptive);
    CHECK(ggml_cuda_parse_pressure_policy("latency-first", policy) && policy == ggml_cuda_pressure_policy::latency_first);
    CHECK(ggml_cuda_parse_pressure_policy("latency_first", policy) && policy == ggml_cuda_pressure_policy::latency_first);
    CHECK(!ggml_cuda_parse_pressure_policy("invalid", policy) && policy == ggml_cuda_pressure_policy::adaptive);

    ggml_cuda_memory_headroom headroom;
    CHECK(!ggml_cuda_graph_under_pressure(headroom));
    headroom.valid = true;
    headroom.total_bytes = 24ULL * 1024 * 1024 * 1024;
    headroom.free_bytes = 1024ULL * 1024 * 1024;
    CHECK(!ggml_cuda_graph_under_pressure(headroom));
    headroom.free_bytes = 256ULL * 1024 * 1024;
    CHECK(ggml_cuda_graph_under_pressure(headroom));

    for (uintptr_t i = 1; i <= 5; ++i) {
        auto g = std::make_unique<ggml_cuda_graph>();
        g->last_used = (i == 1) ? 0 : (100 + i);
        cache.emplace(reinterpret_cast<const void *>(i), std::move(g));
    }
    ggml_cuda_graph * keep_ptr = cache.at(hot).get();
    // latency_first caps single-step reclamation to 1 LRU victim while preserving keep_ptr.
    CHECK(ggml_cuda_graph_pressure_reclaim(cache, keep_ptr, ggml_cuda_pressure_policy::latency_first, 3) == 1);
    CHECK(cache.size() == 4 && cache.find(hot) != cache.end());
    CHECK(cache.find(reinterpret_cast<const void *>(2)) == cache.end());
    // adaptive reclaims up to max_evict LRU victims while preserving keep_ptr even though keep_ptr has last_used == 0.
    CHECK(ggml_cuda_graph_pressure_reclaim(cache, keep_ptr, ggml_cuda_pressure_policy::adaptive, 2) == 2);
    CHECK(cache.size() == 2 && cache.find(hot) != cache.end());
    CHECK(cache.find(reinterpret_cast<const void *>(5)) != cache.end());
    CHECK(ggml_cuda_graph_pressure_reclaim(cache, keep_ptr, ggml_cuda_pressure_policy::adaptive, 8) == 1);
    CHECK(cache.size() == 1 && cache.find(hot) != cache.end());
    CHECK(ggml_cuda_graph_pressure_reclaim(cache, keep_ptr, ggml_cuda_pressure_policy::adaptive, 8) == 0);
    cache.clear();

    std::puts("CUDA graph memory: release, LRU/explicit limits, pressure policy, retirement and submission/capture gates PASS");
    return 0;
}
