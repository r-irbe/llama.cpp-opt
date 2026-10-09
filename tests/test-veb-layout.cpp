#include "veb-tree-layout.h"
#include "paged-kv-cache.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// =========================================================================================
// Test Suite: Van Emde Boas (vEB) Cache-Oblivious Memory Organization Architecture
// =========================================================================================

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  TEST: Van Emde Boas (vEB) Cache-Oblivious Memory Organization  \n";
    std::cout << "=================================================================\n\n";

    // -------------------------------------------------------------------------------------
    // Test 1: Van Emde Boas Permutation Mapping & Bijection Invariant
    // -------------------------------------------------------------------------------------
    std::cout << "[1] Testing recursive vEB permutation mapping for complete binary tree...\n";
    const uint32_t test_height = 3; // N = 2^4 - 1 = 15 nodes
    veb_layout_mapper mapper(test_height);

    std::cout << "  Tree height: " << mapper.height() << ", Capacity: " << mapper.capacity() << " nodes\n";
    if (mapper.capacity() != 15) {
        std::cerr << "FAIL: Mapper capacity mismatch!\n";
        return 1;
    }

    // Invariant: Root (Eytzinger index 1) must map to vEB memory index 0
    if (mapper.to_veb(1) != 0) {
        std::cerr << "FAIL: Root does not map to vEB index 0! Got: " << mapper.to_veb(1) << "\n";
        return 1;
    }

    // Invariant: Bijection - each Eytzinger index must map to a unique vEB index in [0, capacity-1]
    std::vector<bool> seen_veb(mapper.capacity(), false);
    for (uint32_t eyt = 1; eyt <= mapper.capacity(); ++eyt) {
        uint32_t veb = mapper.to_veb(eyt);
        if (veb >= mapper.capacity() || seen_veb[veb]) {
            std::cerr << "FAIL: Non-bijective mapping for Eytzinger index " << eyt << " -> " << veb << "\n";
            return 1;
        }
        seen_veb[veb] = true;
    }
    std::cout << "  Bijection verified across all 15 nodes. vEB Mapping PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 2: Cache-Oblivious Van Emde Boas Static Search Tree
    // -------------------------------------------------------------------------------------
    std::cout << "[2] Testing Cache-Oblivious vEB Static Search Tree across 1,000 entries...\n";
    std::vector<std::pair<int32_t, int64_t>> test_data;
    for (int32_t k = 0; k < 1000; ++k) {
        test_data.push_back({k * 3 + 1, static_cast<int64_t>(k * 100)});
    }

    veb_static_search_tree<int32_t, int64_t> tree;
    tree.build(test_data);

    // Verify all keys are found with exact payload values
    for (int32_t k = 0; k < 1000; ++k) {
        int64_t val = 0;
        bool found = tree.find(k * 3 + 1, val);
        if (!found || val != static_cast<int64_t>(k * 100)) {
            std::cerr << "FAIL: Key " << (k * 3 + 1) << " not found or value mismatch!\n";
            return 1;
        }
    }

    // Verify non-existent keys return false
    for (int32_t k = 0; k < 1000; ++k) {
        int64_t val = 0;
        if (tree.find(k * 3, val) || tree.find(k * 3 + 2, val)) {
            std::cerr << "FAIL: Non-existent key found falsely!\n";
            return 1;
        }
    }

    // Benchmark search throughput across 100,000 queries
    auto start_t = std::chrono::high_resolution_clock::now();
    int64_t dummy_sum = 0;
    for (int i = 0; i < 100000; ++i) {
        int64_t val = 0;
        int32_t target = (i % 1000) * 3 + 1;
        if (tree.find(target, val)) {
            dummy_sum += val;
        }
    }
    auto end_t = std::chrono::high_resolution_clock::now();
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count();
    double avg_ns = static_cast<double>(total_ns) / 100000.0;

    std::cout << "  Executed 100,000 vEB search queries in " << (total_ns / 1000000.0) 
              << " ms (average: " << avg_ns << " ns/query)\n";
    std::cout << "  vEB Static Search Tree PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 3: Van Emde Boas Prefix Trie Clusterer
    // -------------------------------------------------------------------------------------
    std::cout << "[3] Testing Van Emde Boas Prefix Trie Clusterer...\n";
    std::vector<veb_clustered_trie_node> sample_trie;
    // Level 0
    sample_trie.push_back({1, 10, 1, 2, 0});
    // Level 1
    sample_trie.push_back({2, 20, 3, 2, 1});
    sample_trie.push_back({3, 30, 5, 2, 1});
    // Level 2
    sample_trie.push_back({4, 40, 0, 0, 2});
    sample_trie.push_back({5, 50, 0, 0, 2});
    sample_trie.push_back({6, 60, 0, 0, 2});
    sample_trie.push_back({7, 70, 0, 0, 2});

    auto serialized = veb_trie_serializer::serialize_recursive(sample_trie, 2);
    if (serialized.size() != sample_trie.size()) {
        std::cerr << "FAIL: Serialized trie size mismatch!\n";
        return 1;
    }

    // Invariant: Top sub-trie (depth <= 1) nodes appear before depth 2 nodes
    for (size_t i = 0; i < 3; ++i) {
        if (serialized[i].depth > 1) {
            std::cerr << "FAIL: Top sub-trie not placed contiguously first!\n";
            return 1;
        }
    }
    std::cout << "  vEB Prefix Trie Clustering PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 4: Dynamic PagedAttention Block Size Autotuning
    // -------------------------------------------------------------------------------------
    std::cout << "[4] Testing Dynamic PagedAttention Block Sizing for Modern Hardware...\n";

    // Scenario A: Short context (< 4096 tokens) on standard CPU -> 16 tokens (minimizes tail fragmentation)
    auto cfg_short = paged_kv_config::auto_tune(512, false);
    if (cfg_short.block_size_tokens != 16) {
        std::cerr << "FAIL: Short context should select block size 16!\n";
        return 1;
    }
    std::cout << "  Scenario A (Context: 512, TMA: No):  Selected block size = " 
              << cfg_short.block_size_tokens << " tokens (Optimal for memory footprint)\n";

    // Scenario B: Long context (>= 4096 tokens) -> 32 tokens (128 KB per layer)
    auto cfg_long = paged_kv_config::auto_tune(8192, false);
    if (cfg_long.block_size_tokens != 32) {
        std::cerr << "FAIL: Long context should select block size 32!\n";
        return 1;
    }
    std::cout << "  Scenario B (Context: 8192, TMA: No): Selected block size = " 
              << cfg_long.block_size_tokens << " tokens (Optimal for TLB & throughput)\n";

    // Scenario C: Hopper / Blackwell TMA hardware flag -> 32 tokens (matches 128B transaction units)
    auto cfg_tma = paged_kv_config::auto_tune(512, true);
    if (cfg_tma.block_size_tokens != 32) {
        std::cerr << "FAIL: TMA hardware should select block size 32!\n";
        return 1;
    }
    std::cout << "  Scenario C (Context: 512, TMA: Yes): Selected block size = " 
              << cfg_tma.block_size_tokens << " tokens (Optimal for TMA 128B warp alignment)\n";

    // Verify cache initialization with tuned configuration
    common_paged_kv_cache tuned_cache(cfg_long);
    if (tuned_cache.free_block_count() != cfg_long.max_blocks) {
        std::cerr << "FAIL: Tuned cache pool initialization failed!\n";
        return 1;
    }
    std::cout << "  Dynamic Block Sizing PASSED OK.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "  ALL VAN EMDE BOAS & DYNAMIC BLOCK SIZE TESTS PASSED (100%)     \n";
    std::cout << "=================================================================\n";
    return 0;
}
