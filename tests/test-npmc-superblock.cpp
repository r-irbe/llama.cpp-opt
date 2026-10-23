//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// Unit Test: Zero-Copy NPMC / .leangrph v3 Superblock2MiB & 2-Bitplane Ternary MoE Execution
// Strict 7-bit ASCII compliance (INV-001).

#include "llama-npmc-superblock.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace llama::npmc;

static void test_zero_copy_stream_a_and_b() {
    printf("[1] Testing 2 MiB Superblock Stream A (BitwiseVeb4096 + EliasFanoIndex) & Stream B...\n");

    auto sb = std::make_unique<Superblock2MiB>();
    const uint32_t num_experts = 4096u;
    const uint32_t words_per_expert = 16u; // 16 * 64 = 1024 ternary weights per expert
    const std::vector<uint32_t> active_ids = {0u, 3u, 64u, 65u, 511u, 1024u, 2048u, 4095u};
    const size_t m = active_ids.size();

    std::vector<std::vector<int8_t>> raw_trits(m, std::vector<int8_t>(words_per_expert * 64u, 0));
    std::vector<std::vector<TernaryBitplane64>> packed_weights(m, std::vector<TernaryBitplane64>(words_per_expert));

    for (size_t e = 0; e < m; ++e) {
        for (size_t w = 0; w < words_per_expert; ++w) {
            int8_t block[64];
            for (size_t b = 0; b < 64u; ++b) {
                const size_t idx = w * 64u + b;
                const int v = static_cast<int>((idx + e * 17u) % 5u);
                int8_t t = 0;
                if (v == 1 || v == 2) t = 1;
                else if (v == 3 || v == 4) t = -1;
                block[b] = t;
                raw_trits[e][idx] = t;
            }
            packed_weights[e][w] = TernaryBitplane64::encode64(block, 64u);
            assert(packed_weights[e][w].invariant_ok());
        }
    }

    const bool ok = pack_superblock_2mib(*sb, num_experts, words_per_expert, active_ids, packed_weights);
    assert(ok);

    Superblock2MiBView view = sb->view();
    assert(view.valid());

    // Verify zero-copy pointer identity inside the 2 MiB buffer
    const uint8_t * raw_base = sb->data;
    assert(reinterpret_cast<const uint8_t *>(view.header()) == raw_base);
    assert(reinterpret_cast<const uint8_t *>(view.veb_mask()) == raw_base + view.header()->stream_a_offset);

    EliasFanoView ef = view.elias_fano();
    assert(ef.valid());
    assert(ef.size() == m);

    const uint64_t expected_stride = static_cast<uint64_t>(words_per_expert) * sizeof(TernaryBitplane64);
    for (size_t r = 0; r < m; ++r) {
        assert(ef.get(r) == r * expected_stride);
        const uint32_t eid = active_ids[r];
        assert(view.is_expert_active(eid));
        assert(view.active_expert_rank(eid) == static_cast<int32_t>(r));
        const TernaryBitplane64 * w_ptr = view.expert_bitplanes(eid);
        assert(w_ptr != nullptr);
        assert(reinterpret_cast<const uint8_t *>(w_ptr) ==
               raw_base + view.header()->stream_b_offset + r * expected_stride);
    }

    // Inactive expert checks and boundary values
    assert(!view.is_expert_active(1u));
    assert(!view.is_expert_active(63u));
    assert(!view.is_expert_active(4094u));
    assert(!view.is_expert_active(4096u));
    assert(view.expert_bitplanes(1u) == nullptr);
    assert(view.expert_bitplanes(4096u) == nullptr);

    // Verify BitwiseVeb4096 min/max/next traversal directly on zero-copy Stream A mask
    assert(view.veb_mask()->find_min() == 0);
    assert(view.veb_mask()->find_max() == 4095);
    std::vector<uint32_t> iterated;
    int32_t cur = view.veb_mask()->find_min();
    while (cur >= 0) {
        iterated.push_back(static_cast<uint32_t>(cur));
        cur = view.veb_mask()->find_next(static_cast<uint32_t>(cur) + 1u);
    }
    assert(iterated == active_ids);

    // Verify Stream B 2-bitplane ternary popcount dot product and int8 dot product against reference
    std::vector<int8_t> act_trits(words_per_expert * 64u, 0);
    std::vector<int8_t> act_i8(words_per_expert * 64u, 0);
    std::vector<TernaryBitplane64> act_planes(words_per_expert);
    for (size_t w = 0; w < words_per_expert; ++w) {
        for (size_t b = 0; b < 64u; ++b) {
            const size_t idx = w * 64u + b;
            act_trits[idx] = (idx % 3u == 0u) ? 1 : ((idx % 3u == 1u) ? -1 : 0);
            act_i8[idx] = static_cast<int8_t>((static_cast<int>(idx % 15u) - 7));
        }
        act_planes[w] = TernaryBitplane64::encode64(&act_trits[w * 64u], 64u);
    }

    for (size_t r = 0; r < m; ++r) {
        const TernaryBitplane64 * w_ptr = view.expert_bitplanes(active_ids[r]);
        const int32_t got_pop = TernaryBitplane64::dot_popcount(w_ptr, act_planes.data(), words_per_expert);
        const int32_t got_i8  = TernaryBitplane64::dot_i8(w_ptr, act_i8.data(), words_per_expert);

        int32_t ref_pop = 0;
        int32_t ref_i8  = 0;
        for (size_t idx = 0; idx < words_per_expert * 64u; ++idx) {
            ref_pop += static_cast<int32_t>(raw_trits[r][idx]) * static_cast<int32_t>(act_trits[idx]);
            ref_i8  += static_cast<int32_t>(raw_trits[r][idx]) * static_cast<int32_t>(act_i8[idx]);
        }
        assert(got_pop == ref_pop);
        assert(got_i8 == ref_i8);
        const uint64_t expected_off = static_cast<uint64_t>(view.header()->stream_b_offset) + r * expected_stride;
        assert(view.lookup_expert_offset(active_ids[r]) == expected_off);
        assert(view.gemv_popcnt(active_ids[r], act_planes.data(), words_per_expert) == ref_pop);
    }
    assert(view.lookup_expert_offset(1u) == 0ULL);
    printf("    [OK] Zero-copy Stream A/B pointers, BitwiseVeb4096 + EliasFanoIndex, and popcount dot product verified.\n");
}

static void test_edge_cases_and_validation() {
    printf("[2] Testing empty superblock, corrupt magic rejection, and invariant checks...\n");

    auto sb = std::make_unique<Superblock2MiB>();
    // Empty active expert set
    std::vector<uint32_t> empty_ids;
    std::vector<std::vector<TernaryBitplane64>> empty_weights;
    assert(pack_superblock_2mib(*sb, 4096u, 8u, empty_ids, empty_weights));
    Superblock2MiBView view = sb->view();
    assert(view.valid());
    assert(view.header()->active_experts == 0u);
    assert(view.veb_mask()->find_min() == -1);
    assert(view.expert_bitplanes(0u) == nullptr);

    // Reject overlapping (+1 and -1 simultaneously) bitplane
    std::vector<uint32_t> bad_ids = {7u};
    std::vector<std::vector<TernaryBitplane64>> bad_weights = {{{0x3ULL, 0x1ULL}}};
    assert(!bad_weights[0][0].invariant_ok());
    assert(!pack_superblock_2mib(*sb, 4096u, 1u, bad_ids, bad_weights));

    // Reject unsorted expert IDs
    std::vector<uint32_t> unsorted_ids = {10u, 5u};
    std::vector<std::vector<TernaryBitplane64>> two_weights = {{{1ULL, 0ULL}}, {{0ULL, 1ULL}}};
    assert(!pack_superblock_2mib(*sb, 4096u, 1u, unsorted_ids, two_weights));

    // Reject corrupted magic
    assert(pack_superblock_2mib(*sb, 4096u, 1u, {5u}, {{{1ULL, 0ULL}}}));
    assert(sb->view().valid());
    reinterpret_cast<Superblock2MiBHeader *>(sb->data)->magic ^= 1ULL;
    assert(!sb->view().valid());

    // Reject words_per_expert == 0 and num_experts == 0
    assert(!pack_superblock_2mib(*sb, 4096u, 0u, {5u}, {{}}));
    assert(!pack_superblock_2mib(*sb, 0u, 1u, {}, {}));

    // Re-pack valid superblock and probe structural alignment/bounds checks
    assert(pack_superblock_2mib(*sb, 256u, 2u, {5u, 128u}, {{{1ULL, 0ULL}, {0ULL, 1ULL}}, {{3ULL, 0ULL}, {0ULL, 4ULL}}}));
    assert(sb->view().valid());
    assert(!sb->view().is_expert_active(256u));

    // Reject 64-byte aligned (not 128-byte aligned) base pointer
    alignas(128) static uint8_t raw_buf[SUPERBLOCK_2MIB_BYTES + 128u];
    std::memcpy(raw_buf + 64u, sb->data, SUPERBLOCK_2MIB_BYTES);
    Superblock2MiBView misaligned_view(raw_buf + 64u, SUPERBLOCK_2MIB_BYTES);
    assert(!misaligned_view.valid());

    // Reject truncated stream_a_bytes
    auto * hdr = reinterpret_cast<Superblock2MiBHeader *>(sb->data);
    const uint32_t orig_a_bytes = hdr->stream_a_bytes;
    hdr->stream_a_bytes = sizeof(llama::veb::BitwiseVeb4096) + sizeof(EliasFanoHeader) - 1u;
    assert(!sb->view().valid());
    hdr->stream_a_bytes = orig_a_bytes;
    assert(sb->view().valid());

    // Reject truncated stream_b_bytes
    const uint32_t orig_b_bytes = hdr->stream_b_bytes;
    hdr->stream_b_bytes = orig_b_bytes - 1u;
    assert(!sb->view().valid());
    hdr->stream_b_bytes = orig_b_bytes;
    assert(sb->view().valid());

    // Reject overlapping Stream A and Stream B
    const uint32_t orig_b_off = hdr->stream_b_offset;
    hdr->stream_b_offset = hdr->stream_a_offset;
    assert(!sb->view().valid());
    hdr->stream_b_offset = orig_b_off;
    assert(sb->view().valid());

    // Reject corrupted EliasFanoHeader low_bits >= 64
    auto * ef_hdr = reinterpret_cast<EliasFanoHeader *>(
        sb->data + hdr->stream_a_offset + sizeof(llama::veb::BitwiseVeb4096)
    );
    const uint32_t orig_lb = ef_hdr->low_bits;
    ef_hdr->low_bits = 64u;
    assert(!sb->view().valid());
    ef_hdr->low_bits = orig_lb;
    assert(sb->view().valid());

    printf("    [OK] Edge cases (empty active set, invalid bitplane, unsorted IDs, alignment, truncated streams, EF bounds) verified.\n");
}

static void test_sparse24_and_crown_interval() {
    printf("[3] Testing 2:4 Structured Sparsity Codec & CROWN Affine Interval Bounds...\n");

    // Exhaustively verify all 6 valid (idx0, idx1) pairs in a 4-element quad
    size_t valid_count = 0;
    for (uint8_t nibble = 0; nibble < 16u; ++nibble) {
        if (Sparse24Metadata::is_valid_nibble(nibble)) {
            valid_count++;
            uint8_t i0 = 0, i1 = 0;
            Sparse24Metadata::decode_quad(nibble, i0, i1);
            uint8_t reencoded = 0;
            assert(Sparse24Metadata::encode_quad(i0, i1, reencoded));
            assert(reencoded == nibble);

            const int32_t comp[2] = {11, -7};
            int32_t dense[4] = {0, 0, 0, 0};
            Sparse24Metadata::reconstruct_quad(comp, nibble, dense);
            assert(dense[i0] == 11 && dense[i1] == -7);
            int32_t comp_dot = comp[0] * dense[i0] + comp[1] * dense[i1];
            int32_t full_dot = 0;
            for (int k = 0; k < 4; ++k) full_dot += dense[k] * dense[k];
            assert(comp_dot == full_dot);
        }
    }
    assert(valid_count == 6u);

    // Verify CROWN affine interval bound soundness across all 2^4 = 16 hypercube vertices
    const int64_t weights[4] = {5, -3, 0, 8};
    const CrownInterval box[4] = {{-10, 4}, {-6, 12}, {-100, 100}, {1, 9}};
    const int64_t bias = -15;
    const CrownInterval bounds = crown_affine_interval(weights, box, 4u, bias);
    assert(bounds.valid());

    for (int mask = 0; mask < 16; ++mask) {
        int64_t val = bias;
        for (int j = 0; j < 4; ++j) {
            const int64_t xj = (mask & (1 << j)) ? box[j].upper : box[j].lower;
            val += weights[j] * xj;
        }
        assert(bounds.contains(val));
    }
    printf("    [OK] Sparse24Metadata (6/16 valid nibbles) and CROWN interval soundness verified.\n");
}

static void test_npmc_format_benchmarks() {
    printf("[4] Benchmarking NPMC / .leangrph v3 Superblock2MiB vs Baseline GGUF Layouts...\n");
    using clock_t = std::chrono::steady_clock;

    // 4.1: Active Expert Lookup: BitwiseVeb4096 + EliasFanoView O(1) vs Linear Scan (U=4096, m=32 active)
    auto sb = std::make_unique<Superblock2MiB>();
    const uint32_t num_experts = 4096u;
    const uint32_t words_per_expert = 64u; // 64 * 64 = 4096 trits/expert
    std::vector<uint32_t> active_ids;
    std::vector<std::vector<TernaryBitplane64>> weights(32u, std::vector<TernaryBitplane64>(words_per_expert));
    for (uint32_t i = 0; i < 32u; ++i) {
        active_ids.push_back(i * 127u + 3u);
        for (uint32_t w = 0; w < words_per_expert; ++w) {
            weights[i][w] = {0x5555555555555555ULL, 0x2222222222222222ULL};
        }
    }
    const bool packed_ok = pack_superblock_2mib(*sb, num_experts, words_per_expert, active_ids, weights);
    assert(packed_ok);
    Superblock2MiBView view = sb->view();
    assert(view.valid());

    // Dense table representing baseline linear descriptor scan over 4096 experts
    std::vector<uint32_t> baseline_ids(num_experts);
    for (uint32_t i = 0; i < num_experts; ++i) baseline_ids[i] = i;

    const size_t lookup_iters = 500000u;
    volatile uintptr_t sink_ptr = 0;
    auto t0 = clock_t::now();
    for (size_t it = 0; it < lookup_iters; ++it) {
        const uint32_t q = active_ids[it & 31u];
        const TernaryBitplane64 * p = view.expert_bitplanes(q);
        sink_ptr ^= reinterpret_cast<uintptr_t>(p);
    }
    auto t1 = clock_t::now();
    const double veb_ef_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(lookup_iters);

    volatile size_t sink_idx = 0;
    auto t2 = clock_t::now();
    for (size_t it = 0; it < lookup_iters; ++it) {
        const uint32_t q = active_ids[it & 31u];
        size_t found = 0;
        for (size_t j = 0; j < num_experts; ++j) {
            if (baseline_ids[j] == q) { found = j; break; }
        }
        sink_idx += found;
    }
    auto t3 = clock_t::now();
    const double linear_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / static_cast<double>(lookup_iters);

    // 4.2: TernaryBitplane64::dot_popcount (2-bitplane POPCNT) vs INT8 Scalar/SIMD Dot Product
    std::vector<TernaryBitplane64> act_planes(words_per_expert, {0x3333333333333333ULL, 0x4444444444444444ULL});
    std::vector<int8_t> w_i8(words_per_expert * 64u, 1);
    std::vector<int8_t> a_i8(words_per_expert * 64u, -1);
    const size_t dot_iters = 200000u;
    const double total_trits = static_cast<double>(dot_iters) * static_cast<double>(words_per_expert * 64u);

    volatile int32_t dot_sink = 0;
    const TernaryBitplane64 * w_planes = view.expert_bitplanes_by_rank(0);
    auto t4 = clock_t::now();
    for (size_t it = 0; it < dot_iters; ++it) {
        dot_sink += TernaryBitplane64::dot_popcount(w_planes, act_planes.data(), words_per_expert);
    }
    auto t5 = clock_t::now();
    const double popcnt_sec = std::chrono::duration<double>(t5 - t4).count();
    const double popcnt_gelem_s = (total_trits / 1e9) / popcnt_sec;

    auto t6 = clock_t::now();
    for (size_t it = 0; it < dot_iters; ++it) {
        int32_t acc = 0;
        for (size_t k = 0; k < words_per_expert * 64u; ++k) {
            acc += static_cast<int32_t>(w_i8[k]) * static_cast<int32_t>(a_i8[k]);
        }
        dot_sink += acc;
    }
    auto t7 = clock_t::now();
    const double i8_sec = std::chrono::duration<double>(t7 - t6).count();
    const double i8_gelem_s = (total_trits / 1e9) / i8_sec;

    // 4.3: DRAM Bank Power-of-Two Stride Aliasing (65536B) vs NPMC Padded Stride (65664B = +128B)
    const size_t n_rows = 512u;
    const size_t aliased_stride = 65536u;
    const size_t padded_stride  = 65536u + 128u;
    std::vector<uint8_t> buf_aliased(n_rows * aliased_stride, 1u);
    std::vector<uint8_t> buf_padded(n_rows * padded_stride, 1u);
    const size_t strided_passes = 200u;
    const size_t cols_sampled = 64u; // 64 cache lines per row across all 512 rows (column-major bank conflict pattern)

    volatile uint64_t mem_sink = 0;
    auto t8 = clock_t::now();
    for (size_t p = 0; p < strided_passes; ++p) {
        uint64_t s = 0;
        for (size_t c = 0; c < cols_sampled; ++c) {
            const size_t col_off = (c * 1024u) & (aliased_stride - 1u);
            for (size_t r = 0; r < n_rows; ++r) {
                s += *reinterpret_cast<const uint64_t *>(&buf_aliased[r * aliased_stride + col_off]);
            }
        }
        mem_sink += s;
    }
    auto t9 = clock_t::now();
    const double aliased_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();

    auto t10 = clock_t::now();
    for (size_t p = 0; p < strided_passes; ++p) {
        uint64_t s = 0;
        for (size_t c = 0; c < cols_sampled; ++c) {
            const size_t col_off = (c * 1024u) & (aliased_stride - 1u);
            for (size_t r = 0; r < n_rows; ++r) {
                s += *reinterpret_cast<const uint64_t *>(&buf_padded[r * padded_stride + col_off]);
            }
        }
        mem_sink += s;
    }
    auto t11 = clock_t::now();
    const double padded_ms = std::chrono::duration<double, std::milli>(t11 - t10).count();

    printf("    * Active Expert Lookup (U=4096): BitwiseVeb4096+EliasFano = %.2f ns/op vs Linear = %.2f ns/op (%.1fx speedup)\n",
           veb_ef_ns, linear_ns, linear_ns / (veb_ef_ns > 0.0 ? veb_ef_ns : 1.0));
    printf("    * Ternary Bitplane GEMV (4096 trits): POPCNT = %.2f Gelem/s vs INT8 = %.2f Gelem/s (%.2fx speedup, 4.0x memory compression)\n",
           popcnt_gelem_s, i8_gelem_s, popcnt_gelem_s / (i8_gelem_s > 0.0 ? i8_gelem_s : 1.0));
    printf("    * DRAM Bank-Conflict Strided Gather (512 rows x 64KB): Aliased(65536B) = %.2f ms vs Padded(65664B) = %.2f ms (%.2fx speedup)\n",
           aliased_ms, padded_ms, aliased_ms / (padded_ms > 0.0 ? padded_ms : 1.0));
#ifdef GGML_USE_RUST_MMAP_ENGINE
    // 4.4: Rust-Mmap-Engine & Lean 4 FFI Verification and Latency vs C++20 Inline Mirror
    const int32_t storage_node = rust_mmap_detect_storage_numa_node(".");
    assert(storage_node >= -1);
    assert(rust_mmap_lean_verify_transaction(100ULL, 200ULL, 50000ULL, 1ULL, 0u) == 0u);
    assert(rust_mmap_lean_verify_transaction(100ULL, 200ULL, 950000ULL, 1ULL, 0u) == 1u);
    assert(rust_mmap_lean_verify_transaction(100ULL, 200ULL, 50000ULL, 1ULL, 1u) == 2u);

    volatile uint64_t ffi_off_sink = 0;
    auto t12 = clock_t::now();
    for (size_t it = 0; it < lookup_iters; ++it) {
        const uint32_t q = active_ids[it & 31u];
        uint64_t out_off = 0ULL;
        const int32_t active = rust_mmap_superblock_lookup_expert(sb->data, sizeof(sb->data), q, &out_off);
        assert(active == 1);
        ffi_off_sink ^= out_off;
    }
    auto t13 = clock_t::now();
    const double rust_ffi_lookup_ns = std::chrono::duration<double, std::nano>(t13 - t12).count() / static_cast<double>(lookup_iters);

    const size_t lean_iters = 50000u;
    volatile int32_t lean_gemv_sink = 0;
    auto t14 = clock_t::now();
    for (size_t it = 0; it < lean_iters; ++it) {
        lean_gemv_sink += view.gemv_popcnt(active_ids[it & 31u], act_planes.data(), words_per_expert);
    }
    auto t15 = clock_t::now();
    const double lean_gemv_ns = std::chrono::duration<double, std::nano>(t15 - t14).count() / static_cast<double>(lean_iters);

    printf("    * [E4 Rust/Lean FFI] Storage NUMA node = %d | FFI Expert Lookup = %.2f ns/op (vs %.2f ns C++20 inline) | Lean 4 FFI 64-word GEMV = %.2f ns/op\n",
           storage_node, rust_ffi_lookup_ns, veb_ef_ns, lean_gemv_ns);
    (void)ffi_off_sink; (void)lean_gemv_sink;
#endif
    (void)sink_ptr; (void)sink_idx; (void)dot_sink; (void)mem_sink;
}

int main() {
    printf("=================================================================\n");
    printf("  TEST: Zero-Copy NPMC / .leangrph v3 Superblock2MiB & Ternary   \n");
    printf("=================================================================\n");
    test_zero_copy_stream_a_and_b();
    test_edge_cases_and_validation();
    test_sparse24_and_crown_interval();
    test_npmc_format_benchmarks();
    printf("=================================================================\n");
    printf("[OK] ALL NPMC SUPERBLOCK2MIB TESTS PASSED!\n");
    printf("=================================================================\n");
    return 0;
}
