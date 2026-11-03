#pragma once

// ============================================================================
// PolyGraph plan-buffer planner — C++ contract twin of the Lean 4 formal spec
//
//   Formal spec of record : rust-mmap-engine/formalization/PolyGraph/Basic.lean
//   Design spec           : docs/specs/polygraph-plan-buffer.md
//   Conformance harness   : tests/test-polygraph-plan.cpp (exhaustive, gamma=16)
//
// Every function below mirrors exactly one Lean definition and cites it.
// The Lean module builds clean (lake build, Lean 4.33) and proves:
//   mask_closure, readable_rows_are_real, kv_cells_disjoint,
//   sanitize_within_limits, real_rows_fit.
// Do not change semantics here without changing the Lean side first.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <algorithm>

#ifdef GGML_USE_CUDA
extern "C" void ggml_cuda_polygraph_set_active_plan(const uint32_t * plan_words) __attribute__((weak));
extern "C" void ggml_cuda_polygraph_clear_active_plan() __attribute__((weak));
#endif

namespace polygraph {

/// 28-byte plan buffer (host-authored, copied to device before replay).
/// Lean: `Plan`
struct alignas(4) Plan {
    uint32_t active_len;   ///< Lean: activeLen — real rows are 0..active_len
    uint32_t pos_base;     ///< Lean: posBase   — KV/RoPE position of row 0
    uint32_t flags;        ///< Lean: flags     — bit0: target-only baseline
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t reserved3;
};
static_assert(sizeof(Plan) == 28, "plan buffer must be exactly 28 bytes");

/// Compile-time verification geometry. Lean: `Limits`
struct Limits {
    uint32_t gamma_max;     ///< Lean: gammaMax
    uint32_t slot_seq;      ///< Lean: slotSeq    — KV sequence of real rows
    uint32_t scratch_seq;   ///< Lean: scratchSeq — KV sequence of padding rows

    /// Lean: Limits.rows — pinned batch rows = gamma_max + 1
    uint32_t rows() const { return gamma_max + 1; }
};

/// Lean: `Plan.valid` — activeLen <= gammaMax
inline bool plan_valid(const Limits & g, const Plan & p) {
    return p.active_len <= g.gamma_max;
}

/// Lean: `Plan.isReal` — row carries a token whose verification output is used.
inline bool is_real(const Plan & p, uint32_t r) {
    return r <= p.active_len;
}

/// Lean: `Plan.rowSeq` — real rows on the slot sequence, padding on scratch.
inline uint32_t row_seq(const Plan & p, const Limits & g, uint32_t r) {
    return r <= p.active_len ? g.slot_seq : g.scratch_seq;
}

/// Lean: `Plan.rowPos` — posBase + row index (padding rows get valid positions).
inline uint32_t row_pos(const Plan & p, uint32_t r) {
    return p.pos_base + r;
}

/// Lean: `Plan.attends` — real rows attend causal real columns; padding rows
/// are unconstrained (scratch outputs discarded).
inline bool attends(const Plan & p, uint32_t r, uint32_t c) {
    return r <= p.active_len ? (c <= r && c <= p.active_len) : true;
}

/// Lean: `Plan.readable` — sampler scores exactly rows r < active_len + 1.
inline bool readable(const Plan & p, uint32_t r) {
    return r < p.active_len + 1;
}

/// Lean: `sanitize` — clamp active_len into [0, gamma_max].
inline Plan sanitize(const Limits & g, const Plan & p) {
    Plan q = p;
    q.active_len = std::min(p.active_len, g.gamma_max);
    return q;
}

/// Select the padding scratch sequence ID for a context with capacity n_seq_max.
/// When n_seq_max > slot_seq + 1, assigns a dedicated disjoint sequence (slot_seq + 1)
/// matching Lean `Limits.scratchSeq != Limits.slotSeq`; falls back to same-seq causal
/// isolation when n_seq_max <= slot_seq + 1 (e.g. single-tenant n_seq_max == 1).
inline uint32_t choose_scratch_seq(uint32_t slot_seq, uint32_t n_seq_max) {
    return (slot_seq + 1 < n_seq_max) ? (slot_seq + 1) : slot_seq;
}

namespace detail {
inline Plan & tls_active_plan() {
    static thread_local Plan plan = {0u, 0u, 0u, 0u, 0u, 0u, 0u};
    return plan;
}
inline bool & tls_plan_is_active() {
    static thread_local bool active = false;
    return active;
}
} // namespace detail

/// Stage an active 28-byte PolyGraph plan on the calling thread (and forward to
/// the CUDA backend staging hook when present) before verification decode.
inline void set_active_plan(const Plan & p) {
    detail::tls_active_plan() = p;
    detail::tls_plan_is_active() = true;
#ifdef GGML_USE_CUDA
    if (ggml_cuda_polygraph_set_active_plan != nullptr) {
        uint32_t words[7];
        std::memcpy(words, &p, sizeof(words));
        ggml_cuda_polygraph_set_active_plan(words);
    }
#endif
}

/// Return a pointer to the currently active thread-local PolyGraph plan, or
/// nullptr when no pinned verification plan is active.
inline const Plan * get_active_plan() {
    return detail::tls_plan_is_active() ? &detail::tls_active_plan() : nullptr;
}

/// Clear the active thread-local PolyGraph plan after verification decode.
inline void clear_active_plan() {
    detail::tls_plan_is_active() = false;
    detail::tls_active_plan() = {0u, 0u, 0u, 0u, 0u, 0u, 0u};
#ifdef GGML_USE_CUDA
    if (ggml_cuda_polygraph_clear_active_plan != nullptr) {
        ggml_cuda_polygraph_clear_active_plan();
    }
#endif
}

} // namespace polygraph
