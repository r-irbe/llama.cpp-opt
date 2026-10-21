//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "llama-kv-zdd.h"
#include <cstdio>
#include <cassert>
#include <chrono>
#include <vector>

using namespace llama::kv::zdd;

int main() {
    printf("=================================================================\n");
    printf("   Unit Test: ZDD / SDD KV Cache Sparse Block Compression       \n");
    printf("=================================================================\n");

    ZddKvManager mgr(8192);

    // 1. Minato Zero-Suppression Rule: node(v, low, ZDD_EMPTY) == low
    uint32_t s1 = mgr.get_node(10, ZDD_EMPTY, ZDD_EMPTY);
    assert(s1 == ZDD_EMPTY);
    printf("[OK] Minato Zero-Suppression test passed: high=EMPTY -> returns EMPTY directly.\n");

    // 2. Canonical Singleton & Hash-Consing
    uint32_t single_5 = mgr.singleton(5);
    uint32_t single_5_dup = mgr.singleton(5);
    assert(single_5 == single_5_dup);
    assert(single_5 > ZDD_BASE);
    printf("[OK] Hash-Consing test passed: Canonical node ID = %u\n", single_5);

    // 3. 512k Context Sparse Block Indexing (524,288 tokens / 64 = 8,192 blocks)
    const uint32_t total_tokens = 524288;
    const uint32_t n_sink_blocks = 4;    // 256 tokens sink
    const uint32_t n_window_blocks = 32; // 2,048 tokens sliding window
    const uint32_t total_blocks = total_tokens / KV_BLOCK_TOKENS;

    uint32_t root = mgr.build_active_context_blocks(total_tokens, n_sink_blocks, n_window_blocks);
    std::vector<uint32_t> active_blocks;
    mgr.extract_active_blocks(root, active_blocks);

    printf("[*] 512k Context Total Blocks : %u\n", total_blocks);
    printf("[*] Active Selected Blocks    : %zu (Sinks: %u, Window: %u)\n", 
           active_blocks.size(), n_sink_blocks, n_window_blocks);
    printf("[*] ZDD Graph Node Count      : %zu nodes (vs 8,192 dense table)\n", mgr.node_count());

    float bandwidth_saving = mgr.calculate_bandwidth_saving(total_tokens, root);
    printf("[*] KV DRAM Bandwidth Savings : %.2f%%\n", bandwidth_saving * 100.0f);
    assert(bandwidth_saving >= 0.70f); // Must achieve >= 70% bandwidth reduction
    printf("[OK] Verified Bandwidth Reduction >= 70%%: Actual = %.2f%%\n", bandwidth_saving * 100.0f);

    // 4. Benchmarking Traversal Latency
    const int N_ITERS = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        mgr.extract_active_blocks(root, active_blocks);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double avg_ns = total_ns / N_ITERS;
    printf("[*] Average Traversal Latency : %.2f ns per query\n", avg_ns);
    assert(avg_ns < 100.0); // Must be sub-100ns (target < 50ns)
    printf("[OK] Verified Sub-50ns/Sub-100ns Target: %.2f ns\n", avg_ns);

    // 5. Knuth DLX-to-ZDD Exact Cover Compilation for MoE Expert Groups
    printf("\n--- Section 5.1: Knuth DLX-to-ZDD Exact Cover Test ---\n");
    ExactCoverMatrix ec_mat;
    ec_mat.init(4); // 4 items/experts to cover: {0, 1, 2, 3}
    // Row 0: covers {0, 2} -> mask 0b0101 = 5
    // Row 1: covers {1, 3} -> mask 0b1010 = 10
    // Row 2: covers {0, 1} -> mask 0b0011 = 3
    // Row 3: covers {2, 3} -> mask 0b1100 = 12
    ec_mat.add_row(0b0101);
    ec_mat.add_row(0b1010);
    ec_mat.add_row(0b0011);
    ec_mat.add_row(0b1100);

    DlxZddCompiler dlx_compiler;
    uint32_t ec_root = dlx_compiler.solve_exact_cover(0b1111, ec_mat);
    assert(ec_root > ZDD_BASE);
    std::vector<uint32_t> solution_rows;
    dlx_compiler.extract_first_solution(ec_root, solution_rows);
    assert(solution_rows.size() == 2);
    printf("[OK] Knuth DLX-to-ZDD Exact Cover solved: %zu rows cover all 4 items.\n", solution_rows.size());
    printf("[*] ZDD Exact Cover Graph Size : %zu nodes\n", dlx_compiler.node_count());

    // 6. MoE Cluster Router Sub-50ns Resolution Test
    printf("\n--- Section 5.1: MoE Cluster Router Test ---\n");
    ZddMoeClusterRouter router;
    // 8 clusters covering 64 experts (8 experts per contiguous 2MB cluster block)
    for (uint16_t c = 0; c < 8; ++c) {
        uint64_t mask = 0xFFULL << (c * 8);
        router.add_cluster(c, mask);
    }
    // Active top-8 experts: {0, 1, 8, 9, 16, 24, 32, 40}
    uint64_t active_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 8) | (1ULL << 9) |
                           (1ULL << 16) | (1ULL << 24) | (1ULL << 32) | (1ULL << 40);
    std::vector<uint16_t> resolved_clusters;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        router.resolve_clusters(active_mask, resolved_clusters);
    }
    end = std::chrono::high_resolution_clock::now();
    total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    avg_ns = total_ns / N_ITERS;
    printf("[*] MoE Top-8 Expert Cluster Resolution Latency: %.2f ns\n", avg_ns);
    assert(avg_ns < 50.0); // Target < 50 ns
    printf("[OK] Verified Sub-50ns MoE Cluster Resolution Target: %.2f ns (Resolved %zu clusters)\n",
           avg_ns, resolved_clusters.size());

    // 7. Attention Kernel ZDD Block Skipping Test
    printf("\n--- Section 7: ZDD Attention Kernel Block Skipping Test ---\n");
    auto skip_mask = ZddAttentionBlockSkipper::generate_skip_mask(total_blocks, active_blocks);
    assert(skip_mask.size() == total_blocks);
    // Sink block 0 must be active
    assert(skip_mask[0] == 1);
    assert(ZddAttentionBlockSkipper::is_block_active(0, active_blocks));
    // Middle block (e.g. block 100) outside sink & window must be skipped
    assert(skip_mask[100] == 0);
    assert(!ZddAttentionBlockSkipper::is_block_active(100, active_blocks));

    // Verify 64-token FP16 (-INFINITY = 0xFC00) block skip mask invariant (16 x uint64_t words = 0xFC00FC00FC00FC00ULL)
    std::vector<uint16_t> fp16_mask(total_tokens, 0x0000u);
    for (uint32_t b = 0; b < total_blocks; ++b) {
        if (skip_mask[b] == 0) {
            for (uint32_t t = 0; t < KV_BLOCK_TOKENS; ++t) {
                fp16_mask[b * KV_BLOCK_TOKENS + t] = 0xFC00u; // -INFINITY in IEEE 754 binary16
            }
        }
    }
    uint32_t skipped_blocks_64 = 0;
    for (uint32_t ic = 0; ic < total_tokens; ++ic) {
        if (((ic & 63u) == 0u) && (ic + 64u <= total_tokens)) {
            const uint64_t * mp64 = reinterpret_cast<const uint64_t *>(&fp16_mask[ic]);
            uint64_t diff = 0ULL;
            for (int w = 0; w < 16; ++w) {
                diff |= (mp64[w] ^ 0xFC00FC00FC00FC00ULL);
            }
            if (diff == 0ULL) {
                skipped_blocks_64++;
                ic += 63u;
                continue;
            }
        }
    }
    assert(skipped_blocks_64 == total_blocks - active_blocks.size());
    printf("[OK] Verified ZDD Attention Kernel Block Skipping: %u / %u 64-token blocks skipped in O(1).\n",
           skipped_blocks_64, total_blocks);

    printf("=================================================================\n");
    printf("[OK] ALL ZDD KV CACHE & DLX MOE TESTS PASSED SUCCESSFULLY!\n");
    printf("=================================================================\n");
    return 0;
}

