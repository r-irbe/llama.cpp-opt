#pragma once
//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// Zero-Copy NPMC / .leangrph v3 Superblock2MiB & 2-Bitplane Ternary Compatibility Header
// Strict 7-bit ASCII compliance (INV-001).
//
// Layout of a 2 MiB (2,097,152-byte) Superblock2MiB:
//   - Offset 0..63 (64B): Superblock2MiBHeader (magic, version=3, offsets, expert counts)
//   - Stream A (128B aligned):
//       * BitwiseVeb4096 active expert mask (640B, 128B aligned)
//       * EliasFanoHeader + zero-copy lower_words[] and upper_words[] monotonic byte offsets
//   - Stream B (64B aligned):
//       * Packed 2-bit ternary bitplanes (TernaryBitplane64: {pos_mask, neg_mask}) per active expert,
//         indexed in O(1) via EliasFanoView::get(rank) and evaluated via hardware popcount.

#include "llama-bitwise-veb.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace llama::npmc {

constexpr uint64_t LEANGRPH_V3_MAGIC       = 0x485052474E41454CULL; // "LEANGRPH" in little-endian
constexpr uint32_t NPMC_SUPERBLOCK_MAGIC   = 0x4E504D43u;           // "NPMC" magic header (0x4E504D43)
constexpr uint32_t LEANGRPH_V3_VERSION     = 3u;
constexpr size_t   SUPERBLOCK_2MIB_BYTES   = 2ULL * 1024ULL * 1024ULL;
constexpr uint32_t MAX_SUPERBLOCK_EXPERTS  = 4096u;

struct alignas(64) Superblock2MiBHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t num_experts;
    uint32_t active_experts;
    uint32_t stream_a_offset;
    uint32_t stream_a_bytes;
    uint32_t stream_b_offset;
    uint32_t stream_b_bytes;
    uint32_t words_per_expert;
    uint32_t reserved[5];
};
static_assert(sizeof(Superblock2MiBHeader) == 64, "Superblock2MiBHeader must be 64 bytes");

struct EliasFanoHeader {
    uint32_t count;
    uint32_t low_bits;
    uint64_t low_mask;
    uint32_t lower_words_count;
    uint32_t upper_words_count;
};
static_assert(sizeof(EliasFanoHeader) == 24, "EliasFanoHeader must be 24 bytes");

class EliasFanoView {
public:
    constexpr EliasFanoView() noexcept = default;

    EliasFanoView(const EliasFanoHeader * hdr, const uint64_t * lower, const uint64_t * upper) noexcept
        : hdr_(hdr), lower_(lower), upper_(upper) {}

    inline bool valid() const noexcept {
        if (hdr_ == nullptr) return false;
        if (hdr_->low_bits >= 64u) return false;
        const uint64_t expected_mask = (hdr_->low_bits == 0u) ? 0ULL : ((1ULL << hdr_->low_bits) - 1ULL);
        if (hdr_->low_mask != expected_mask) return false;
        if (hdr_->count == 0u) return true;
        if (lower_ == nullptr || upper_ == nullptr) return false;
        const size_t min_lower_words = (static_cast<size_t>(hdr_->count) * hdr_->low_bits + 63u) / 64u;
        if (hdr_->lower_words_count < min_lower_words) return false;
        if (hdr_->upper_words_count == 0u) return false;
        return true;
    }

    inline uint32_t size() const noexcept {
        return hdr_ ? hdr_->count : 0u;
    }

    inline uint32_t low_bits() const noexcept {
        return hdr_ ? hdr_->low_bits : 0u;
    }

    inline const uint64_t * lower_words_ptr() const noexcept { return lower_; }
    inline const uint64_t * upper_words_ptr() const noexcept { return upper_; }

    inline uint64_t get(size_t rank) const noexcept {
        if (!hdr_ || rank >= hdr_->count) return 0ULL;
        size_t rem = rank;
        size_t unary_pos = 0;
        for (uint32_t w = 0; w < hdr_->upper_words_count; ++w) {
            uint64_t word = upper_[w];
            const size_t pop = static_cast<size_t>(__builtin_popcountll(word));
            if (rem < pop) {
                for (size_t i = 0; i < rem; ++i) {
                    word &= (word - 1ULL);
                }
                unary_pos = (static_cast<size_t>(w) << 6) |
                            static_cast<size_t>(__builtin_ctzll(word));
                break;
            }
            rem -= pop;
        }
        const uint64_t high = static_cast<uint64_t>(unary_pos - rank);
        uint64_t low = 0ULL;
        const uint32_t lb = hdr_->low_bits;
        if (lb > 0u) {
            const size_t bp = rank * static_cast<size_t>(lb);
            const size_t w = bp >> 6;
            const size_t off = bp & 63u;
            low = lower_[w] >> off;
            if (off + lb > 64u && (w + 1u) < hdr_->lower_words_count) {
                low |= (lower_[w + 1u] << (64u - off));
            }
            low &= hdr_->low_mask;
        }
        return (high << lb) | low;
    }

private:
    const EliasFanoHeader * hdr_ = nullptr;
    const uint64_t * lower_ = nullptr;
    const uint64_t * upper_ = nullptr;
};

class EliasFanoIndex {
public:
    EliasFanoIndex() = default;

    explicit EliasFanoIndex(const std::vector<uint64_t> & sorted) {
        n_ = static_cast<uint32_t>(sorted.size());
        if (n_ == 0u) return;
        const uint64_t max_val = sorted.back();
        const uint64_t u = max_val + 1ULL;
        const uint64_t avg_gap = (u / n_) > 0ULL ? (u / n_) : 1ULL;
        low_bits_ = static_cast<uint32_t>(63 - __builtin_clzll(avg_gap));
        low_mask_ = (low_bits_ == 0u) ? 0ULL : ((1ULL << low_bits_) - 1ULL);

        const size_t total_low_bits = static_cast<size_t>(n_) * low_bits_;
        lower_words_.assign((total_low_bits + 63u) / 64u + 1u, 0ULL);

        const size_t max_high = static_cast<size_t>(max_val >> low_bits_);
        upper_words_.assign((static_cast<size_t>(n_) + max_high + 65u) / 64u, 0ULL);

        for (uint32_t r = 0; r < n_; ++r) {
            const uint64_t val = sorted[r];
            if (low_bits_ > 0u) {
                const uint64_t low = val & low_mask_;
                const size_t bp = static_cast<size_t>(r) * low_bits_;
                const size_t w = bp >> 6;
                const size_t off = bp & 63u;
                lower_words_[w] |= (low << off);
                if (off + low_bits_ > 64u) {
                    lower_words_[w + 1u] |= (low >> (64u - off));
                }
            }
            const size_t high = static_cast<size_t>(val >> low_bits_);
            const size_t upos = static_cast<size_t>(r) + high;
            upper_words_[upos >> 6] |= (1ULL << (upos & 63u));
        }
    }

    inline uint32_t size() const noexcept { return n_; }
    inline uint32_t low_bits() const noexcept { return low_bits_; }

    inline size_t serialized_bytes() const noexcept {
        return sizeof(EliasFanoHeader) +
               (lower_words_.size() + upper_words_.size()) * sizeof(uint64_t);
    }

    inline size_t serialize_to(uint8_t * dst, size_t max_bytes) const noexcept {
        const size_t needed = serialized_bytes();
        if (!dst || max_bytes < needed) return 0u;
        EliasFanoHeader hdr{};
        hdr.count = n_;
        hdr.low_bits = low_bits_;
        hdr.low_mask = low_mask_;
        hdr.lower_words_count = static_cast<uint32_t>(lower_words_.size());
        hdr.upper_words_count = static_cast<uint32_t>(upper_words_.size());
        std::memcpy(dst, &hdr, sizeof(hdr));
        uint8_t * cur = dst + sizeof(hdr);
        if (!lower_words_.empty()) {
            const size_t lb_bytes = lower_words_.size() * sizeof(uint64_t);
            std::memcpy(cur, lower_words_.data(), lb_bytes);
            cur += lb_bytes;
        }
        if (!upper_words_.empty()) {
            const size_t ub_bytes = upper_words_.size() * sizeof(uint64_t);
            std::memcpy(cur, upper_words_.data(), ub_bytes);
        }
        return needed;
    }

    static inline EliasFanoView view_from_bytes(const uint8_t * src, size_t max_bytes) noexcept {
        if (!src || max_bytes < sizeof(EliasFanoHeader)) return EliasFanoView{};
        const auto * hdr = reinterpret_cast<const EliasFanoHeader *>(src);
        const size_t words_bytes =
            (static_cast<size_t>(hdr->lower_words_count) + static_cast<size_t>(hdr->upper_words_count)) *
            sizeof(uint64_t);
        if (max_bytes < sizeof(EliasFanoHeader) + words_bytes) return EliasFanoView{};
        const auto * lower = reinterpret_cast<const uint64_t *>(src + sizeof(EliasFanoHeader));
        const auto * upper = lower + hdr->lower_words_count;
        return EliasFanoView(hdr, lower, upper);
    }

    inline uint64_t get(size_t rank) const noexcept {
        EliasFanoHeader hdr{
            n_,
            low_bits_,
            low_mask_,
            static_cast<uint32_t>(lower_words_.size()),
            static_cast<uint32_t>(upper_words_.size()),
        };
        return EliasFanoView(&hdr, lower_words_.data(), upper_words_.data()).get(rank);
    }

private:
    uint32_t n_ = 0;
    uint32_t low_bits_ = 0;
    uint64_t low_mask_ = 0;
    std::vector<uint64_t> lower_words_;
    std::vector<uint64_t> upper_words_;
};

/// 2-bitplane ternary block: 64 trits in {-1, 0, +1} encoded as (pos_mask, neg_mask).
/// Invariant: (pos_mask & neg_mask) == 0ULL.
struct TernaryBitplane64 {
    uint64_t pos_mask;
    uint64_t neg_mask;

    inline bool invariant_ok() const noexcept {
        return (pos_mask & neg_mask) == 0ULL;
    }

    static inline TernaryBitplane64 encode64(const int8_t * trits, size_t count = 64u) noexcept {
        uint64_t pos = 0ULL;
        uint64_t neg = 0ULL;
        const size_t lim = count < 64u ? count : 64u;
        for (size_t i = 0; i < lim; ++i) {
            if (trits[i] > 0) {
                pos |= (1ULL << i);
            } else if (trits[i] < 0) {
                neg |= (1ULL << i);
            }
        }
        return TernaryBitplane64{pos, neg};
    }

    static inline int32_t dot_popcount(
        const TernaryBitplane64 * weights,
        const TernaryBitplane64 * activations,
        size_t num_words
    ) noexcept {
        int32_t acc = 0;
        for (size_t w = 0; w < num_words; ++w) {
            const uint64_t wp = weights[w].pos_mask;
            const uint64_t wn = weights[w].neg_mask;
            const uint64_t ap = activations[w].pos_mask;
            const uint64_t an = activations[w].neg_mask;
            const uint64_t same_sign = (wp & ap) | (wn & an);
            const uint64_t opp_sign  = (wp & an) | (wn & ap);
            acc += __builtin_popcountll(same_sign) - __builtin_popcountll(opp_sign);
        }
        return acc;
    }

    static inline int32_t dot_i8(
        const TernaryBitplane64 * weights,
        const int8_t * activations,
        size_t num_words
    ) noexcept {
        int32_t acc = 0;
        for (size_t w = 0; w < num_words; ++w) {
            uint64_t pos = weights[w].pos_mask;
            uint64_t neg = weights[w].neg_mask;
            const int8_t * act_block = activations + (w << 6);
            while (pos != 0ULL) {
                const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(pos));
                acc += static_cast<int32_t>(act_block[b]);
                pos &= (pos - 1ULL);
            }
            while (neg != 0ULL) {
                const uint32_t b = static_cast<uint32_t>(__builtin_ctzll(neg));
                acc -= static_cast<int32_t>(act_block[b]);
                neg &= (neg - 1ULL);
            }
        }
        return acc;
    }
};
static_assert(sizeof(TernaryBitplane64) == 16, "TernaryBitplane64 must be 16 bytes");

/// Zero-copy reader over a 2 MiB NPMC / .leangrph v3 superblock buffer.
class Superblock2MiBView {
public:
    explicit Superblock2MiBView(const uint8_t * base, size_t size = SUPERBLOCK_2MIB_BYTES) noexcept
        : base_(base), size_(size) {}

    inline bool valid() const noexcept {
        if (!base_ || size_ < SUPERBLOCK_2MIB_BYTES) return false;
        if ((reinterpret_cast<uintptr_t>(base_) & 127u) != 0u) return false;
        const auto * hdr = header();
        const bool valid_magic = (hdr->magic == LEANGRPH_V3_MAGIC) ||
                                 (hdr->magic == static_cast<uint64_t>(NPMC_SUPERBLOCK_MAGIC));
        if (!valid_magic || hdr->version != LEANGRPH_V3_VERSION) return false;
        if (hdr->num_experts == 0u || hdr->num_experts > MAX_SUPERBLOCK_EXPERTS || hdr->active_experts > hdr->num_experts) return false;
        if (hdr->words_per_expert == 0u) return false;
        if (hdr->stream_a_offset < sizeof(Superblock2MiBHeader)) return false;
        if ((hdr->stream_a_offset & 127u) != 0u) return false;
        const size_t stream_a_end = static_cast<size_t>(hdr->stream_a_offset) + hdr->stream_a_bytes;
        if (stream_a_end > size_) return false;
        if (hdr->stream_a_bytes < sizeof(llama::veb::BitwiseVeb4096) + sizeof(EliasFanoHeader)) return false;
        if ((hdr->stream_b_offset & 63u) != 0u) return false;
        if (static_cast<size_t>(hdr->stream_b_offset) < stream_a_end) return false;
        const size_t expected_b_bytes =
            static_cast<size_t>(hdr->active_experts) *
            static_cast<size_t>(hdr->words_per_expert) *
            sizeof(TernaryBitplane64);
        if (hdr->stream_b_bytes < expected_b_bytes) return false;
        if (static_cast<size_t>(hdr->stream_b_offset) + hdr->stream_b_bytes > size_) return false;
        return elias_fano().valid() && elias_fano().size() == hdr->active_experts;
    }

    inline const Superblock2MiBHeader * header() const noexcept {
        return reinterpret_cast<const Superblock2MiBHeader *>(base_);
    }

    inline const llama::veb::BitwiseVeb4096 * veb_mask() const noexcept {
        return reinterpret_cast<const llama::veb::BitwiseVeb4096 *>(base_ + header()->stream_a_offset);
    }

    inline EliasFanoView elias_fano() const noexcept {
        const uint8_t * ef_ptr = base_ + header()->stream_a_offset + sizeof(llama::veb::BitwiseVeb4096);
        const size_t ef_max = header()->stream_a_bytes - sizeof(llama::veb::BitwiseVeb4096);
        return EliasFanoIndex::view_from_bytes(ef_ptr, ef_max);
    }

    inline bool is_expert_active(uint32_t expert_id) const noexcept {
        const auto * hdr = header();
        if (expert_id >= hdr->num_experts || expert_id >= MAX_SUPERBLOCK_EXPERTS) return false;
        const auto * veb = veb_mask();
        const uint32_t c = (expert_id >> 6) & 63u;
        const uint32_t b = expert_id & 63u;
        return (veb->clusters[c] & (1ULL << b)) != 0ULL;
    }

    /// Computes the dense active-expert rank in [0, active_experts) using the BitwiseVeb4096 clusters.
    inline int32_t active_expert_rank(uint32_t expert_id) const noexcept {
        if (!is_expert_active(expert_id)) return -1;
        const auto * veb = veb_mask();
        const uint32_t target_c = (expert_id >> 6) & 63u;
        const uint32_t target_b = expert_id & 63u;
        int32_t rank = 0;
        uint64_t sum_mask = (target_c == 0u) ? 0ULL : (veb->summary & ((1ULL << target_c) - 1ULL));
        while (sum_mask != 0ULL) {
            const uint32_t c = static_cast<uint32_t>(__builtin_ctzll(sum_mask));
            rank += __builtin_popcountll(veb->clusters[c]);
            sum_mask &= (sum_mask - 1ULL);
        }
        const uint64_t lower_bits = (target_b == 0u) ? 0ULL : (veb->clusters[target_c] & ((1ULL << target_b) - 1ULL));
        rank += __builtin_popcountll(lower_bits);
        return rank;
    }

    /// Zero-copy pointer to the rank-th active expert's TernaryBitplane64 weights in Stream B.
    inline const TernaryBitplane64 * expert_bitplanes_by_rank(size_t rank) const noexcept {
        const auto * hdr = header();
        if (rank >= hdr->active_experts) return nullptr;
        const uint64_t byte_offset = elias_fano().get(rank);
        const size_t expert_bytes = static_cast<size_t>(hdr->words_per_expert) * sizeof(TernaryBitplane64);
        if (byte_offset + expert_bytes > hdr->stream_b_bytes) return nullptr;
        return reinterpret_cast<const TernaryBitplane64 *>(base_ + hdr->stream_b_offset + byte_offset);
    }

    /// Zero-copy pointer to expert_id's TernaryBitplane64 weights in Stream B (nullptr if inactive).
    inline const TernaryBitplane64 * expert_bitplanes(uint32_t expert_id) const noexcept {
        const int32_t r = active_expert_rank(expert_id);
        if (r < 0) return nullptr;
        return expert_bitplanes_by_rank(static_cast<size_t>(r));
    }

    /// Resolves the byte offset of expert_id's TernaryBitplane64 weights within the 2 MiB superblock
    /// via BitwiseVeb4096 rank + EliasFanoView::get(rank) in O(1) time (returns 0 if inactive).
    inline uint64_t lookup_expert_offset(uint32_t expert_id) const noexcept {
        const int32_t r = active_expert_rank(expert_id);
        if (r < 0) return 0ULL;
        const auto * hdr = header();
        const uint64_t byte_offset = elias_fano().get(static_cast<size_t>(r));
        const size_t expert_bytes = static_cast<size_t>(hdr->words_per_expert) * sizeof(TernaryBitplane64);
        if (byte_offset + expert_bytes > hdr->stream_b_bytes) return 0ULL;
        return static_cast<uint64_t>(hdr->stream_b_offset) + byte_offset;
    }

    /// Executes 2-bitplane ternary POPCNT GEMV for a single active expert against TernaryBitplane64 activations.
    inline int32_t gemv_popcnt(
        uint32_t expert_id,
        const TernaryBitplane64 * activations,
        size_t num_words = 0u
    ) const noexcept {
        if (!activations) return 0;
        const uint64_t off = lookup_expert_offset(expert_id);
        if (off == 0ULL) return 0;
        const auto * w = reinterpret_cast<const TernaryBitplane64 *>(base_ + off);
        const size_t words = (num_words == 0u || num_words > header()->words_per_expert)
            ? static_cast<size_t>(header()->words_per_expert)
            : num_words;
        return TernaryBitplane64::dot_popcount(w, activations, words);
    }

    /// Executes multi-row 2-bitplane ternary POPCNT GEMV for an active expert into out_rows[0..num_rows).
    inline bool gemv_popcnt(
        uint32_t expert_id,
        const TernaryBitplane64 * activations,
        size_t words_per_row,
        size_t num_rows,
        int32_t * out_rows
    ) const noexcept {
        if (!activations || !out_rows || words_per_row == 0u || num_rows == 0u) return false;
        const uint64_t off = lookup_expert_offset(expert_id);
        if (off == 0ULL) return false;
        const size_t total_words = static_cast<size_t>(header()->words_per_expert);
        if (words_per_row * num_rows > total_words) return false;
        const auto * w = reinterpret_cast<const TernaryBitplane64 *>(base_ + off);
        for (size_t r = 0; r < num_rows; ++r) {
            out_rows[r] = TernaryBitplane64::dot_popcount(w + r * words_per_row, activations, words_per_row);
        }
        return true;
    }

private:
    const uint8_t * base_ = nullptr;
    size_t size_ = 0;
};

struct alignas(128) Superblock2MiB {
    uint8_t data[SUPERBLOCK_2MIB_BYTES];

    inline void clear() noexcept {
        std::memset(data, 0, sizeof(data));
    }

    inline Superblock2MiBView view() const noexcept {
        return Superblock2MiBView(data, sizeof(data));
    }
};
static_assert(sizeof(Superblock2MiB) == SUPERBLOCK_2MIB_BYTES, "Superblock2MiB must be exactly 2 MiB");

/// Packs Stream A (BitwiseVeb4096 + EliasFanoIndex) and Stream B (2-bit ternary bitplanes)
/// into a 2 MiB Superblock2MiB buffer for zero-copy reading.
inline bool pack_superblock_2mib(
    Superblock2MiB & sb,
    uint32_t num_experts,
    uint32_t words_per_expert,
    const std::vector<uint32_t> & active_expert_ids,
    const std::vector<std::vector<TernaryBitplane64>> & expert_weights
) {
    if (num_experts == 0u || num_experts > MAX_SUPERBLOCK_EXPERTS) return false;
    if (words_per_expert == 0u) return false;
    if (active_expert_ids.size() != expert_weights.size()) return false;
    const uint32_t m = static_cast<uint32_t>(active_expert_ids.size());
    for (uint32_t i = 0; i < m; ++i) {
        if (active_expert_ids[i] >= num_experts) return false;
        if (i > 0u && active_expert_ids[i] <= active_expert_ids[i - 1u]) return false;
        if (expert_weights[i].size() != words_per_expert) return false;
        for (const auto & plane : expert_weights[i]) {
            if (!plane.invariant_ok()) return false;
        }
    }

    sb.clear();

    constexpr uint32_t stream_a_off = 128u;
    auto * veb_dst = reinterpret_cast<llama::veb::BitwiseVeb4096 *>(sb.data + stream_a_off);
    veb_dst->clear();
    std::vector<uint64_t> offsets;
    offsets.reserve(m);
    const uint64_t stride_bytes = static_cast<uint64_t>(words_per_expert) * sizeof(TernaryBitplane64);
    for (uint32_t i = 0; i < m; ++i) {
        veb_dst->insert(active_expert_ids[i]);
        offsets.push_back(static_cast<uint64_t>(i) * stride_bytes);
    }

    EliasFanoIndex ef(offsets);
    uint8_t * ef_dst = sb.data + stream_a_off + sizeof(llama::veb::BitwiseVeb4096);
    const size_t ef_max = SUPERBLOCK_2MIB_BYTES - stream_a_off - sizeof(llama::veb::BitwiseVeb4096);
    const size_t ef_written = ef.serialize_to(ef_dst, ef_max);
    if (ef_written == 0u) return false;

    const uint32_t stream_a_bytes = static_cast<uint32_t>(sizeof(llama::veb::BitwiseVeb4096) + ef_written);
    const uint32_t stream_b_off = (stream_a_off + stream_a_bytes + 63u) & ~63u;
    const size_t stream_b_bytes = static_cast<size_t>(m) * static_cast<size_t>(stride_bytes);
    if (static_cast<size_t>(stream_b_off) + stream_b_bytes > SUPERBLOCK_2MIB_BYTES) return false;

    uint8_t * b_dst = sb.data + stream_b_off;
    for (uint32_t i = 0; i < m; ++i) {
        if (stride_bytes > 0u) {
            std::memcpy(b_dst + offsets[i], expert_weights[i].data(), static_cast<size_t>(stride_bytes));
        }
    }

    auto * hdr = reinterpret_cast<Superblock2MiBHeader *>(sb.data);
    hdr->magic = LEANGRPH_V3_MAGIC;
    hdr->version = LEANGRPH_V3_VERSION;
    hdr->flags = 0u;
    hdr->num_experts = num_experts;
    hdr->active_experts = m;
    hdr->stream_a_offset = stream_a_off;
    hdr->stream_a_bytes = stream_a_bytes;
    hdr->stream_b_offset = stream_b_off;
    hdr->stream_b_bytes = static_cast<uint32_t>(stream_b_bytes);
    hdr->words_per_expert = words_per_expert;
    return true;
}

/// NVIDIA Ada Lovelace (sm_89) 2:4 Structured Sparsity 4-Bit Nibble Metadata Codec.
/// Packs two strictly ordered non-zero indices (0 <= idx0 < idx1 < 4) per 4-element quad
/// into a 4-bit nibble: (idx0 & 3) | ((idx1 & 3) << 2), matching mma.sp.sync.aligned.m16n8k32.
struct Sparse24Metadata {
    static inline bool encode_quad(uint8_t idx0, uint8_t idx1, uint8_t & out_nibble) noexcept {
        if (idx0 >= idx1 || idx1 >= 4u) return false;
        out_nibble = static_cast<uint8_t>((idx0 & 0x3u) | ((idx1 & 0x3u) << 2u));
        return true;
    }

    static inline void decode_quad(uint8_t nibble, uint8_t & idx0, uint8_t & idx1) noexcept {
        idx0 = static_cast<uint8_t>(nibble & 0x3u);
        idx1 = static_cast<uint8_t>((nibble >> 2u) & 0x3u);
    }

    static inline bool is_valid_nibble(uint8_t nibble) noexcept {
        uint8_t idx0 = 0u, idx1 = 0u;
        decode_quad(nibble, idx0, idx1);
        return idx0 < idx1 && idx1 < 4u;
    }

    template <typename T>
    static inline void reconstruct_quad(const T compressed[2], uint8_t nibble, T out_dense[4]) noexcept {
        out_dense[0] = T(0);
        out_dense[1] = T(0);
        out_dense[2] = T(0);
        out_dense[3] = T(0);
        uint8_t idx0 = 0u, idx1 = 0u;
        decode_quad(nibble, idx0, idx1);
        if (idx0 < idx1 && idx1 < 4u) {
            out_dense[idx0] = compressed[0];
            out_dense[idx1] = compressed[1];
        }
    }
};

/// Certified CROWN / LiRPA Affine Interval Bound Propagation (Galois interval abstraction).
/// Given an input interval box [lower[j], upper[j]] and affine layer (weights[j], bias),
/// computes sound lower and upper output bounds in O(d) time:
///   L = bias + sum_j (max(w_j, 0) * lower[j] + min(w_j, 0) * upper[j])
///   U = bias + sum_j (max(w_j, 0) * upper[j] + min(w_j, 0) * lower[j])
struct CrownInterval {
    int64_t lower;
    int64_t upper;

    inline bool valid() const noexcept {
        return lower <= upper;
    }

    inline bool contains(int64_t x) const noexcept {
        return lower <= x && x <= upper;
    }
};

inline CrownInterval crown_affine_interval(
    const int64_t * weights,
    const CrownInterval * inputs,
    size_t dim,
    int64_t bias
) noexcept {
    int64_t low = bias;
    int64_t high = bias;
    for (size_t j = 0; j < dim; ++j) {
        const int64_t w = weights[j];
        if (w >= 0) {
            low  += w * inputs[j].lower;
            high += w * inputs[j].upper;
        } else {
            low  += w * inputs[j].upper;
            high += w * inputs[j].lower;
        }
    }
    return CrownInterval{low, high};
}

} // namespace llama::npmc
