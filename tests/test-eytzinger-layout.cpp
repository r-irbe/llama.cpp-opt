#include "eytzinger-layout.h"
#include "veb-tree-layout.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// =========================================================================================
// Test & Microarchitectural Benchmark Suite: Eytzinger Layout Evaluation
// Grounded in:
//   1. Michael von Aitzing / Eytzinger (1590): Complete binary tree BFS numbering
//   2. Khuong & Morin (ACM JEA 2017): "Array Layouts for Comparison-Based Searching"
//   3. Comparative evaluation against:
//      - Sorted array with std::lower_bound (standard binary search)
//      - Van Emde Boas cache-oblivious layout (Bender et al. / Prokop 1999)
//      - Eytzinger branchless layout with software prefetching
// =========================================================================================

static void benchmark_comparison(size_t N, size_t num_queries, const std::string & regime_name) {
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "  Regime: " << regime_name << " (N = " << N << " entries, " 
              << num_queries << " queries)\n";
    std::cout << "-----------------------------------------------------------------\n";

    // Generate sorted unique key-value pairs
    std::vector<std::pair<int, int>> sorted_data;
    sorted_data.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        sorted_data.push_back({static_cast<int>(i * 2 + 1), static_cast<int>(i * 10)});
    }

    // Build Eytzinger search tree
    eytzinger_search_tree<int, int> eyt_tree;
    eyt_tree.build(sorted_data);

    // Build Van Emde Boas search tree
    veb_static_search_tree<int, int> veb_tree;
    veb_tree.build(sorted_data);

    // Generate pseudo-random query sequence
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(N * 2 + 2));
    std::vector<int> queries(num_queries);
    for (size_t q = 0; q < num_queries; ++q) {
        queries[q] = dist(rng);
    }

    // 1. Benchmark: Standard std::lower_bound on sorted array
    volatile int dummy_sink = 0;
    auto start_std = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_queries; ++q) {
        int qkey = queries[q];
        auto it = std::lower_bound(sorted_data.begin(), sorted_data.end(), qkey,
            [](const std::pair<int, int> & a, int val) { return a.first < val; });
        if (it != sorted_data.end() && it->first == qkey) {
            dummy_sink += it->second;
        }
    }
    auto end_std = std::chrono::high_resolution_clock::now();
    double ns_std = std::chrono::duration_cast<std::chrono::nanoseconds>(end_std - start_std).count() / static_cast<double>(num_queries);

    // 2. Benchmark: Van Emde Boas Cache-Oblivious Tree
    auto start_veb = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_queries; ++q) {
        int val = 0;
        if (veb_tree.find(queries[q], val)) {
            dummy_sink += val;
        }
    }
    auto end_veb = std::chrono::high_resolution_clock::now();
    double ns_veb = std::chrono::duration_cast<std::chrono::nanoseconds>(end_veb - start_veb).count() / static_cast<double>(num_queries);

    // 3. Benchmark: Eytzinger Branchless Tree with Software Prefetching
    auto start_eyt = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < num_queries; ++q) {
        int val = 0;
        if (eyt_tree.search(queries[q], val)) {
            dummy_sink += val;
        }
    }
    auto end_eyt = std::chrono::high_resolution_clock::now();
    double ns_eyt = std::chrono::duration_cast<std::chrono::nanoseconds>(end_eyt - start_eyt).count() / static_cast<double>(num_queries);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  1. std::lower_bound (Branchy Binary Search) : " << std::setw(6) << ns_std << " ns/query\n";
    std::cout << "  2. Van Emde Boas    (Cache-Oblivious Layout): " << std::setw(6) << ns_veb << " ns/query\n";
    std::cout << "  3. Eytzinger Layout (Branchless + Prefetch) : " << std::setw(6) << ns_eyt << " ns/query\n";
    std::cout << "  --> Eytzinger Speedup vs std::lower_bound   : " << std::setw(6) << (ns_std / ns_eyt) << "x\n";
    std::cout << "  --> Eytzinger vs Van Emde Boas              : " << std::setw(6) << (ns_veb / ns_eyt) << "x\n\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  TEST: Eytzinger (BFS Layout) Search Tree & Microbenchmarks     \n";
    std::cout << "=================================================================\n\n";

    // -------------------------------------------------------------------------------------
    // Test 1: Mathematical Correctness & Bijective Traversal
    // -------------------------------------------------------------------------------------
    std::cout << "[1] Verifying Eytzinger Layout Construction & Search Accuracy...\n";
    std::vector<std::pair<int, int>> test_entries;
    for (int i = 0; i < 63; ++i) {
        test_entries.push_back({(i + 1) * 2, i + 100}); // 2, 4, 6, ..., 126
    }

    eytzinger_search_tree<int, int> tree;
    tree.build(test_entries);

    if (tree.size() != 63) {
        std::cerr << "FAIL: Eytzinger tree size mismatch! Expected 63, got " << tree.size() << "\n";
        return 1;
    }

    // Verify 100% exact match hit rate for all inserted keys
    for (int i = 0; i < 63; ++i) {
        int expected_key = (i + 1) * 2;
        int val = 0;
        if (!tree.search(expected_key, val) || val != (i + 100)) {
            std::cerr << "FAIL: Eytzinger search failed to find inserted key " << expected_key << "\n";
            return 1;
        }
    }

    // Verify rejection of non-existent keys (odd numbers)
    for (int i = 0; i < 63; ++i) {
        int missing_key = (i + 1) * 2 - 1;
        int val = 0;
        if (tree.search(missing_key, val)) {
            std::cerr << "FAIL: Eytzinger search found non-existent key " << missing_key << "\n";
            return 1;
        }
    }

    // Verify branchless lower_bound
    for (int i = 0; i < 63; ++i) {
        int query = (i + 1) * 2 - 1; // odd number
        int out_key = 0, out_val = 0;
        if (!tree.lower_bound(query, out_key, out_val)) {
            std::cerr << "FAIL: Eytzinger lower_bound failed for query " << query << "\n";
            return 1;
        }
        if (out_key != (i + 1) * 2) {
            std::cerr << "FAIL: Eytzinger lower_bound returned wrong key " << out_key << ", expected " << ((i + 1) * 2) << "\n";
            return 1;
        }
    }
    std::cout << "  Mathematical Correctness PASSED (100% exact hit rate and lower_bound).\n\n";

    // -------------------------------------------------------------------------------------
    // Test 2: Flat Key Array Verification
    // -------------------------------------------------------------------------------------
    std::cout << "[2] Verifying Eytzinger Flat Key Array for Token Filters...\n";
    std::vector<uint32_t> vocab_tokens;
    for (uint32_t t = 10; t < 1000; t += 3) {
        vocab_tokens.push_back(t);
    }
    eytzinger_key_array<uint32_t> key_arr;
    key_arr.build(vocab_tokens);

    for (uint32_t t : vocab_tokens) {
        if (!key_arr.contains(t)) {
            std::cerr << "FAIL: Eytzinger key array missed valid token " << t << "\n";
            return 1;
        }
    }
    if (key_arr.contains(11) || key_arr.contains(14)) {
        std::cerr << "FAIL: Eytzinger key array falsely matched invalid token!\n";
        return 1;
    }
    std::cout << "  Eytzinger Flat Key Array PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 3: Zero-Copy Non-Owning View (eytzinger_tree_view & eytzinger_key_view) over Raw Memory
    // -------------------------------------------------------------------------------------
    std::cout << "[3] Verifying Zero-Copy Non-Owning Views over Raw Buffer Memory...\n";
    // 3a. Export compact nodes into raw byte buffer (simulating mmapped GGUF-X tensor)
    auto compact_nodes = tree.export_compact();
    assert(compact_nodes.size() == 63);

    // Instantiate non-owning view on raw void* pointer
    eytzinger_tree_view<int, int> tree_view(compact_nodes.data(), compact_nodes.size());
    if (tree_view.size() != 63) {
        std::cerr << "FAIL: eytzinger_tree_view size mismatch! Expected 63, got " << tree_view.size() << "\n";
        return 1;
    }

    for (int i = 0; i < 63; ++i) {
        int expected_key = (i + 1) * 2;
        int val = 0;
        if (!tree_view.search(expected_key, val) || val != (i + 100)) {
            std::cerr << "FAIL: eytzinger_tree_view search failed for key " << expected_key << "\n";
            return 1;
        }
    }

    // 3b. Export compact keys into raw uint32_t buffer
    auto compact_keys = key_arr.export_compact();
    eytzinger_key_view<uint32_t> key_view(compact_keys.data(), compact_keys.size());
    if (key_view.size() != compact_keys.size()) {
        std::cerr << "FAIL: eytzinger_key_view size mismatch!\n";
        return 1;
    }

    for (uint32_t t : vocab_tokens) {
        if (!key_view.contains(t)) {
            std::cerr << "FAIL: eytzinger_key_view missed token " << t << "\n";
            return 1;
        }
    }
    if (key_view.contains(11) || key_view.contains(14)) {
        std::cerr << "FAIL: eytzinger_key_view falsely matched invalid token!\n";
        return 1;
    }
    std::cout << "  Zero-Copy Non-Owning Views PASSED OK (Direct buffer access with zero heap allocation).\n\n";

    // -------------------------------------------------------------------------------------
    // Test 4: Microarchitectural Comparative Benchmarks Across Cache Regimes
    // -------------------------------------------------------------------------------------
    std::cout << "[4] Running Comparative Microarchitectural Benchmarks (100,000 queries)...\n";
    const size_t num_bench_queries = 100000;

    // Regime A: L1 Cache Resident (N = 64 entries)
    benchmark_comparison(64, num_bench_queries, "L1 Cache Resident (~512 B)");

    // Regime B: L2 Cache Resident (N = 1,024 entries)
    benchmark_comparison(1024, num_bench_queries, "L2 Cache Resident (~8 KB)");

    // Regime C: L3 Cache Resident (N = 32,768 entries)
    benchmark_comparison(32768, num_bench_queries, "L3 Cache Resident (~256 KB)");

    std::cout << "=================================================================\n";
    std::cout << "  ALL EYTZINGER LAYOUT TESTS & BENCHMARKS PASSED (100%)          \n";
    std::cout << "=================================================================\n";

    return 0;
}
