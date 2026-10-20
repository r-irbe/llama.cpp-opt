//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// Unit Test: Zero-Copy NPMC / .leangrph v3 Superblock2MiB & 2-Bitplane Ternary MoE Execution
// Strict 7-bit ASCII compliance (INV-001).

#include "llama-npmc-superblock.h"

#include <cassert>
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
    }
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

    printf("    [OK] Edge cases (empty active set, invalid bitplane invariant, unsorted IDs, corrupt magic) verified.\n");
}

int main() {
    printf("=================================================================\n");
    printf("  TEST: Zero-Copy NPMC / .leangrph v3 Superblock2MiB & Ternary   \n");
    printf("=================================================================\n");
    test_zero_copy_stream_a_and_b();
    test_edge_cases_and_validation();
    printf("=================================================================\n");
    printf("[OK] ALL NPMC SUPERBLOCK2MIB TESTS PASSED!\n");
    printf("=================================================================\n");
    return 0;
}
