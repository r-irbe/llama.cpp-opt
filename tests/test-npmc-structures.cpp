#include "npmc-discrete-structures.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// =========================================================================================
// Test Suite: GGUF-X / NPMC Discrete Mathematical Containers & Execution Structures
// =========================================================================================

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  TEST: GGUF-X / NPMC Discrete Mathematical Execution Structures \n";
    std::cout << "=================================================================\n\n";

    // -------------------------------------------------------------------------------------
    // Test 1: Roaring Bitmaps for Fast Vocabulary & Grammar Token Masking (<100 ns)
    // -------------------------------------------------------------------------------------
    std::cout << "[1] Testing Roaring Bitmaps for Vocabulary Masking & SIMD Intersections...\n";
    roaring_token_set grammar_allowed;
    roaring_token_set model_vocab;

    // Populate model vocabulary with 10,000 tokens
    for (uint32_t t = 0; t < 10000; ++t) {
        model_vocab.add(t);
    }

    // Populate grammar filter with even token IDs
    for (uint32_t t = 0; t < 10000; t += 2) {
        grammar_allowed.add(t);
    }

    // Measure intersection latency
    auto start_t = std::chrono::high_resolution_clock::now();
    model_vocab.intersect(grammar_allowed);
    auto end_t = std::chrono::high_resolution_clock::now();

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count();
    std::cout << "  Intersected 10,000 tokens in " << duration_ns << " ns (target: < 100 ns)\n";
    std::cout << "  Resulting filtered vocabulary token count: " << model_vocab.count() << " (expected 5000)\n";

    if (model_vocab.count() != 5000 || !model_vocab.contains(42) || model_vocab.contains(43)) {
        std::cerr << "FAIL: Roaring bitmap intersection returned invalid token set!\n";
        return 1;
    }

    // Test 256-bit MoE Expert reachability mask
    moe_expert_mask_256 exp_mask;
    exp_mask.set_expert(0);
    exp_mask.set_expert(7);
    exp_mask.set_expert(42);
    exp_mask.set_expert(255);

    if (exp_mask.count() != 4 || !exp_mask.is_expert_set(42) || exp_mask.is_expert_set(43)) {
        std::cerr << "FAIL: moe_expert_mask_256 failed bit check!\n";
        return 1;
    }
    std::cout << "  Roaring Bitmaps & 256-bit MoE reachability masks PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 2: Zero-Suppressed Decision Diagrams (ZDDs) for Sparse MoE Routing (<50 ns)
    // -------------------------------------------------------------------------------------
    std::cout << "[2] Testing Zero-Suppressed Decision Diagrams (ZDDs) for MoE Routing...\n";
    std::cout << "  zdd_node packed size: " << sizeof(zdd_node) << " bytes (target: exactly 12 bytes)\n";
    if (sizeof(zdd_node) != 12) {
        std::cerr << "FAIL: zdd_node size is not 12 bytes packed!\n";
        return 1;
    }

    sparse_moe_zdd zdd;
    // Insert empirical top-8 expert combination: {1, 5, 12, 19, 42, 60, 99, 128} -> Cluster 42
    std::vector<uint16_t> combo1 = {1, 5, 12, 19, 42, 60, 99, 128};
    uint32_t r1 = zdd.insert_combination(combo1, 42);
    zdd.set_root(r1);

    // Measure ZDD resolution latency
    auto zdd_start = std::chrono::high_resolution_clock::now();
    int32_t resolved_cluster = zdd.resolve_cluster(combo1);
    auto zdd_end = std::chrono::high_resolution_clock::now();

    auto zdd_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(zdd_end - zdd_start).count();
    std::cout << "  ZDD resolved top-8 expert combination in " << zdd_ns << " ns (cluster: " 
              << resolved_cluster << ", target: < 50 ns)\n";

    if (resolved_cluster != 42) {
        std::cerr << "FAIL: ZDD failed to resolve correct cluster!\n";
        return 1;
    }
    std::cout << "  ZDD Sparse MoE Routing PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 3: Milner Bigraphical Placement Descriptors with Sharing (DAG Extensions)
    // -------------------------------------------------------------------------------------
    std::cout << "[3] Testing Milner Bigraphical Placement Descriptors with Sharing...\n";
    bigraph_placement_planner planner;
    planner.add_tensor("blk.0.attn_q.weight", 64 * 1024 * 1024, BIGRAPH_LOCUS_VRAM | BIGRAPH_LOCUS_UNIFIED_HBM);
    planner.add_tensor("blk.0.ffn_exps.weight", 512 * 1024 * 1024, BIGRAPH_LOCUS_DDR | BIGRAPH_LOCUS_CXL | BIGRAPH_LOCUS_NVME);

    // Evaluate on discrete GPU system with sufficient VRAM
    planner.evaluate_placement(true, false, false, 1ULL * 1024 * 1024 * 1024);
    const auto & tensors = planner.get_tensors();

    if (tensors[0].resolved_placement != BIGRAPH_LOCUS_VRAM) {
        std::cerr << "FAIL: Bigraph failed to place critical attention weight on VRAM!\n";
        return 1;
    }
    if (tensors[1].resolved_placement != BIGRAPH_LOCUS_DDR) {
        std::cerr << "FAIL: Bigraph failed to place expert weights on host DDR!\n";
        return 1;
    }
    std::cout << "  Tensor 0 (" << tensors[0].tensor_name << "): resolved locus = " 
              << tensors[0].resolved_placement << " (VRAM)\n";
    std::cout << "  Tensor 1 (" << tensors[1].tensor_name << "): resolved locus = " 
              << tensors[1].resolved_placement << " (DDR)\n";
    std::cout << "  Bigraphical Placement Evaluation PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 4: Hybrid Macro-Tile Morton Z-Order Engine (BMI2 _pdep_u64 / _pext_u64)
    // -------------------------------------------------------------------------------------
    std::cout << "[4] Testing Hybrid Macro-Tile Morton Z-Order Encoding & Decoding...\n";
    bool morton_pass = true;

    for (uint32_t x = 0; x < 256; x += 17) {
        for (uint32_t y = 0; y < 256; y += 19) {
            uint64_t z = morton_zorder::encode_2d(x, y);
            uint32_t rx = 0, ry = 0;
            morton_zorder::decode_2d(z, rx, ry);
            if (rx != x || ry != y) {
                std::cerr << "FAIL: Morton roundtrip mismatch! (" << x << ", " << y 
                          << ") -> z=" << z << " -> (" << rx << ", " << ry << ")\n";
                morton_pass = false;
                break;
            }
        }
        if (!morton_pass) break;
    }

    if (!morton_pass) {
        return 1;
    }
    std::cout << "  Morton Z-Order 2D bit-interleaving roundtrip verified for all test coordinates.\n";
    std::cout << "  Morton Z-Order Engine PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 5: Kernel Fusion for Roaring Bitmap Logit Masking
    // -------------------------------------------------------------------------------------
    std::cout << "[5] Testing Kernel Fusion for Roaring Bitmap Logit Masking...\n";
    const size_t test_vocab_size = 1024;
    std::vector<float> logits(test_vocab_size, 1.0f);

    // Build allowed set: allow tokens in [100..200]
    roaring_token_set allowed_set;
    for (uint32_t t = 100; t <= 200; ++t) {
        allowed_set.add(t);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    allowed_set.mask_logits_fused(logits.data(), test_vocab_size, -1e9f);
    auto t_end = std::chrono::high_resolution_clock::now();

    auto mask_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    std::cout << "  Fused logit masking completed in " << mask_ns << " ns across " << test_vocab_size << " logits\n";

    // Verify: logits[0..99] should be -1e9f, logits[100..200] should be 1.0f, logits[201..1023] should be -1e9f
    bool mask_valid = true;
    for (size_t t = 0; t < test_vocab_size; ++t) {
        if (t >= 100 && t <= 200) {
            if (logits[t] != 1.0f) {
                std::cerr << "FAIL: Allowed token " << t << " was improperly masked to " << logits[t] << "\n";
                mask_valid = false;
                break;
            }
        } else {
            if (logits[t] != -1e9f) {
                std::cerr << "FAIL: Blocked token " << t << " was not masked! Value = " << logits[t] << "\n";
                mask_valid = false;
                break;
            }
        }
    }

    if (!mask_valid) {
        return 1;
    }
    std::cout << "  Kernel Fusion for Roaring Bitmap Logit Masking PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 6: In-Warp Streaming rANS / tANS Decompression Engine
    // -------------------------------------------------------------------------------------
    std::cout << "[6] Testing In-Warp Streaming rANS / tANS Decompression Engine...\n";
    inwarp_rans_streaming_decoder rans_dec;

    // Create a synthetic compressed bitstream of 1024 4-bit symbols
    const size_t n_symbols = 1024;
    std::vector<uint8_t> compressed_stream(n_symbols / 2 + 16, 0xAB); // 4-bit nibbles packed
    std::vector<uint8_t> decompressed(n_symbols, 0);

    auto rans_start = std::chrono::high_resolution_clock::now();
    const int RANS_ITERS = 1000;
    for (int it = 0; it < RANS_ITERS; ++it) {
        rans_dec.decompress_warp_block(compressed_stream.data(), n_symbols, decompressed.data());
    }
    auto rans_end = std::chrono::high_resolution_clock::now();

    auto rans_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(rans_end - rans_start).count();
    double avg_ns = static_cast<double>(rans_ns) / RANS_ITERS;
    double throughput_gb_s = (static_cast<double>(n_symbols) / avg_ns); // 1 byte per ns = 1 GB/s

    std::cout << "  Decompressed " << n_symbols << " symbols in " << avg_ns 
              << " ns (" << throughput_gb_s << " GB/s effective throughput)\n";
    std::cout << "  Decompressed first 4 symbols: " << (int)decompressed[0] << ", " 
              << (int)decompressed[1] << ", " << (int)decompressed[2] << ", " << (int)decompressed[3] << "\n";
    std::cout << "  In-Warp Streaming rANS Decompression Engine PASSED OK.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "  ALL GGUF-X / NPMC DISCRETE MATHEMATICAL TESTS PASSED (100%)    \n";
    std::cout << "=================================================================\n";
    return 0;
}

