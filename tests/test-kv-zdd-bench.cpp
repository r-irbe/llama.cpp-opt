// Hot-path micro-benchmark for the ZDD KV block manager (src/llama-kv-zdd.h).
// Companion to the existing tests/test-kv-zdd.cpp semantic suite. This file
// exists to measure get_node / union / extract throughput before and after the
// SWAR packed-key change (scratch/zdd-gpu-simt-swar-analysis.md §6.4).
// It intentionally does not duplicate the semantic assertions.
#include "llama-kv-zdd.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using llama::kv::zdd::ZddKvManager;
using llama::kv::zdd::ZDD_BASE;

int main() {
    // 1. get_node hot loop: interning-heavy mixed triples (the unique-table path)
    {
        ZddKvManager m(1 << 22);
        std::mt19937 rng(7);
        constexpr int kIters = 300000;
        uint64_t sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) {
            const uint32_t v   = 2 * (rng() % 500000u) + 2;
            const uint32_t lo  = 1u + rng() % ((1u << 20) - 1);
            const uint32_t hi  = 1u + rng() % ((1u << 20) - 1);
            sink += m.get_node(v, lo, hi);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("bench get_node: %d calls, %.1f ns/op, nodes=%zu (sink=%llu)\n",
               kIters, ns / kIters, m.node_count(), (unsigned long long) sink);
    }

    // 2. union-chain build: 4096-block active family via from_block_list
    {
        ZddKvManager m(1 << 22);
        std::vector<uint32_t> blocks;
        blocks.reserve(4096);
        std::mt19937 rng(11);
        for (uint32_t i = 0; i < 4096; ++i) {
            blocks.push_back(2 * i); // ascending, var-order valid
        }
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t root = m.from_block_list(blocks);
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<uint32_t> out;
        m.extract_active_blocks(root, out);
        const auto t2 = std::chrono::steady_clock::now();
        printf("bench build 4096-block family: %.1f us (%zu nodes); extract: %.2f us (%zu blocks)\n",
               std::chrono::duration<double, std::micro>(t1 - t0).count(),
               m.node_count(),
               std::chrono::duration<double, std::micro>(t2 - t1).count(),
               out.size());
    }

    // 3. realistic KV step: build_active_context_blocks at 10M tokens + bandwidth saving
    {
        ZddKvManager m(1 << 18);
        const uint32_t tokens_10m = 10485760;
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t root = m.build_active_context_blocks(tokens_10m);
        const auto t1 = std::chrono::steady_clock::now();
        const float saving = m.calculate_bandwidth_saving(tokens_10m, root);
        printf("bench 10M-ctx active set: build %.2f us, bandwidth saving %.4f%%\n",
               std::chrono::duration<double, std::micro>(t1 - t0).count(), saving * 100.0f);
    }
    return 0;
}
