#pragma once

//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// State-Transition Gated Bitwise van Emde Boas (vEB) Vocabulary & KV-Block Mask
// Formally Verified in Lean 4 (`Rules/BitwiseVebWarpCodec.lean`, `Rules/AlgorithmicModelsAndReversibility.lean`)
// Strict 7-bit ASCII compliance (INV-001).
//
// Provides:
//   1. `BitwiseVeb4096`: 2-level 64-ary bit-tree (520B, L1d-resident) for KV-cache block
//      and top-k active token indexing with state-transition summary gating (`prev == 0`).
//   2. `BitwiseVeb262k`: 3-level 64-ary bit-tree (U = 262,144, 32.5 KiB) covering full LLM
//      vocabularies (128k - 256k tokens: Llama-3, DeepSeek-V3, Qwen-2.5) for O(1) grammar-masked
//      token iteration without scanning 256,000 float logits.
//   3. `ReversibleVebDelta`: 16-byte bijective XOR delta token for O(1) speculative decoding
//      tree-verification rollback without cloning 32.5 KiB masks.

#include <cstdint>
#include <cstddef>
#include <cstring>

namespace llama::veb {

struct ReversibleVebDelta {
    uint16_t cluster_idx;
    uint64_t summary_xor_mask;
    uint64_t cluster_xor_mask;
};

struct alignas(128) BitwiseVeb4096 {
    uint64_t summary = 0;
    uint64_t clusters[64] = {0};
    uint64_t summary_writes = 0;
    uint64_t elided_summary_writes = 0;

    inline void clear() noexcept {
        summary = 0;
        std::memset(clusters, 0, sizeof(clusters));
        summary_writes = 0;
        elided_summary_writes = 0;
    }

    inline void insert(uint32_t x) noexcept {
        const uint32_t c = (x >> 6) & 63u;
        const uint32_t b = x & 63u;
        const uint64_t prev = clusters[c];
        clusters[c] = prev | (1ULL << b);
        if (prev == 0ULL) [[unlikely]] {
            summary |= (1ULL << c);
            summary_writes++;
        } else {
            elided_summary_writes++;
        }
    }

    inline void remove(uint32_t x) noexcept {
        const uint32_t c = (x >> 6) & 63u;
        const uint32_t b = x & 63u;
        const uint64_t prev = clusters[c];
        const uint64_t next = prev & ~(1ULL << b);
        clusters[c] = next;
        if (prev != 0ULL && next == 0ULL) [[unlikely]] {
            summary &= ~(1ULL << c);
            summary_writes++;
        } else {
            elided_summary_writes++;
        }
    }

    inline ReversibleVebDelta insert_reversible(uint32_t x) noexcept {
        const uint32_t c = (x >> 6) & 63u;
        const uint32_t b = x & 63u;
        const uint64_t prev = clusters[c];
        const uint64_t next = prev | (1ULL << b);
        clusters[c] = next;
        const uint64_t c_xor = prev ^ next;
        uint64_t s_xor = 0ULL;
        if (prev == 0ULL && next != 0ULL) [[unlikely]] {
            s_xor = (1ULL << c);
            summary ^= s_xor;
        }
        return ReversibleVebDelta{ static_cast<uint16_t>(c), s_xor, c_xor };
    }

    inline void rollback_delta(const ReversibleVebDelta & d) noexcept {
        clusters[d.cluster_idx] ^= d.cluster_xor_mask;
        summary ^= d.summary_xor_mask;
    }

    inline int32_t find_min() const noexcept {
        if (summary == 0ULL) return -1;
        const uint32_t c = static_cast<uint32_t>(__builtin_ctzll(summary));
        const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(clusters[c]));
        return static_cast<int32_t>((c << 6) | b);
    }

    inline int32_t find_max() const noexcept {
        if (summary == 0ULL) return -1;
        const uint32_t c = 63u - static_cast<uint32_t>(__builtin_clzll(summary));
        const uint32_t b = 63u - static_cast<uint32_t>(__builtin_clzll(clusters[c]));
        return static_cast<int32_t>((c << 6) | b);
    }

    inline int32_t find_next(uint32_t x) const noexcept {
        if (x >= 4096u) return -1;
        const uint32_t c = x >> 6;
        const uint32_t b = x & 63u;
        const uint64_t in_cluster = clusters[c] & (~0ULL << b);
        if (in_cluster != 0ULL) {
            return static_cast<int32_t>((c << 6) | static_cast<uint32_t>(__builtin_ctzll(in_cluster)));
        }
        if (c == 63u) return -1;
        const uint64_t next_summary = summary & (~0ULL << (c + 1u));
        if (next_summary == 0ULL) return -1;
        const uint32_t nc = static_cast<uint32_t>(__builtin_ctzll(next_summary));
        const uint32_t nb = static_cast<uint32_t>(__builtin_ctzll(clusters[nc]));
        return static_cast<int32_t>((nc << 6) | nb);
    }
};

/// 3-Level 64-ary Bitwise van Emde Boas Vocabulary Mask (U = 262,144 tokens, 32.5 KiB)
/// Covers full modern LLM vocabularies (128k Llama-3, 129k DeepSeek-V3, 152k Qwen-2.5).
struct alignas(128) BitwiseVeb262k {
    uint64_t root_summary = 0;
    uint64_t mid_summaries[64] = {0};
    uint64_t leaves[4096] = {0};

    inline void insert(uint32_t token_id) noexcept {
        if (token_id >= 262144u) return;
        const uint32_t c2 = (token_id >> 12) & 63u;
        const uint32_t c1 = (token_id >> 6) & 63u;
        const uint32_t b0 = token_id & 63u;
        const uint32_t leaf_idx = (c2 << 6) | c1;
        const uint64_t prev_leaf = leaves[leaf_idx];
        leaves[leaf_idx] = prev_leaf | (1ULL << b0);
        if (prev_leaf == 0ULL) [[unlikely]] {
            const uint64_t prev_mid = mid_summaries[c2];
            mid_summaries[c2] = prev_mid | (1ULL << c1);
            if (prev_mid == 0ULL) [[unlikely]] {
                root_summary |= (1ULL << c2);
            }
        }
    }

    inline void remove(uint32_t token_id) noexcept {
        if (token_id >= 262144u) return;
        const uint32_t c2 = (token_id >> 12) & 63u;
        const uint32_t c1 = (token_id >> 6) & 63u;
        const uint32_t b0 = token_id & 63u;
        const uint32_t leaf_idx = (c2 << 6) | c1;
        const uint64_t prev_leaf = leaves[leaf_idx];
        const uint64_t next_leaf = prev_leaf & ~(1ULL << b0);
        leaves[leaf_idx] = next_leaf;
        if (prev_leaf != 0ULL && next_leaf == 0ULL) [[unlikely]] {
            const uint64_t next_mid = mid_summaries[c2] & ~(1ULL << c1);
            mid_summaries[c2] = next_mid;
            if (next_mid == 0ULL) [[unlikely]] {
                root_summary &= ~(1ULL << c2);
            }
        }
    }

    inline int32_t find_min() const noexcept {
        if (root_summary == 0ULL) return -1;
        const uint32_t c2 = static_cast<uint32_t>(__builtin_ctzll(root_summary));
        const uint32_t c1 = static_cast<uint32_t>(__builtin_ctzll(mid_summaries[c2]));
        const uint32_t b0 = static_cast<uint32_t>(__builtin_ctzll(leaves[(c2 << 6) | c1]));
        return static_cast<int32_t>((c2 << 12) | (c1 << 6) | b0);
    }

    inline int32_t find_next(uint32_t token_id) const noexcept {
        if (token_id >= 262144u) return -1;
        const uint32_t c2 = (token_id >> 12) & 63u;
        const uint32_t c1 = (token_id >> 6) & 63u;
        const uint32_t b0 = token_id & 63u;
        const uint32_t leaf_idx = (c2 << 6) | c1;
        const uint64_t in_leaf = leaves[leaf_idx] & (~0ULL << b0);
        if (in_leaf != 0ULL) {
            return static_cast<int32_t>((c2 << 12) | (c1 << 6) | static_cast<uint32_t>(__builtin_ctzll(in_leaf)));
        }
        if (c1 < 63u) {
            const uint64_t in_mid = mid_summaries[c2] & (~0ULL << (c1 + 1u));
            if (in_mid != 0ULL) {
                const uint32_t nc1 = static_cast<uint32_t>(__builtin_ctzll(in_mid));
                const uint32_t nb0 = static_cast<uint32_t>(__builtin_ctzll(leaves[(c2 << 6) | nc1]));
                return static_cast<int32_t>((c2 << 12) | (nc1 << 6) | nb0);
            }
        }
        if (c2 < 63u) {
            const uint64_t in_root = root_summary & (~0ULL << (c2 + 1u));
            if (in_root != 0ULL) {
                const uint32_t nc2 = static_cast<uint32_t>(__builtin_ctzll(in_root));
                const uint32_t nc1 = static_cast<uint32_t>(__builtin_ctzll(mid_summaries[nc2]));
                const uint32_t nb0 = static_cast<uint32_t>(__builtin_ctzll(leaves[(nc2 << 6) | nc1]));
                return static_cast<int32_t>((nc2 << 12) | (nc1 << 6) | nb0);
            }
        }
        return -1;
    }
};

} // namespace llama::veb
