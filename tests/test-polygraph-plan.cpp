// ============================================================================
// PolyGraph plan-buffer planner — exhaustive conformance harness.
//
// Re-checks the Lean theorems proved in
//   rust-mmap-engine/formalization/PolyGraph/Basic.lean
// against the generated C++ twin in common/spec-verify-planner.h for ALL plans
// at gamma_max = 16 (the exact geometry the counterfactual tuner uses).
//
// Theorem provenance (Lean name -> test section):
//   [1] mask_closure
//   [2] readable_rows_are_real
//   [3] kv_cells_disjoint
//   [4] sanitize_within_limits (+ real_rows_fit)
// ============================================================================

#include "spec-verify-planner.h"

#include <cstdio>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

int main() {
    printf("=================================================================\n");
    printf("  TEST: PolyGraph planner conformance vs Lean spec (gamma=16)\n");
    printf("=================================================================\n");

    // Geometry mirrors Limits in the Lean module; slot != scratch is the
    // configuration invariant kv_cells_disjoint is conditioned on.
    polygraph::Limits g{16, 3, 9};
    TEST_ASSERT(g.slot_seq != g.scratch_seq, "configuration invariant: slot_seq != scratch_seq");
    TEST_ASSERT(g.rows() == 17, "pinned batch rows must be gamma_max + 1");

    long plans_checked = 0;

    {
        printf("\n[1] Lean theorem mask_closure over all plans and row pairs...\n");
        for (uint32_t a = 0; a <= g.gamma_max; ++a) {
            for (uint32_t pb = 0; pb <= 2; ++pb) {
                for (uint32_t fl = 0; fl <= 1; ++fl) {
                    polygraph::Plan p{a, pb, fl, 0, 0, 0, 0};
                    for (uint32_t r = 0; r < g.rows(); ++r) {
                        if (!polygraph::is_real(p, r)) {
                            continue;
                        }
                        for (uint32_t c = 0; c < g.rows(); ++c) {
                            if (polygraph::attends(p, r, c)) {
                                TEST_ASSERT(polygraph::is_real(p, c),
                                    "mask_closure violated: real row attends padding column");
                            }
                        }
                    }
                    ++plans_checked;
                }
            }
        }
        printf("  mask_closure conformance PASSED OK.\n");
    }

    {
        printf("\n[2] Lean theorem readable_rows_are_real over all plans...\n");
        for (uint32_t a = 0; a <= g.gamma_max; ++a) {
            for (uint32_t pb = 0; pb <= 2; ++pb) {
                for (uint32_t fl = 0; fl <= 1; ++fl) {
                    polygraph::Plan p{a, pb, fl, 0, 0, 0, 0};
                    for (uint32_t r = 0; r < g.rows(); ++r) {
                        if (polygraph::readable(p, r)) {
                            TEST_ASSERT(polygraph::is_real(p, r),
                                "readable_rows_are_real violated: sampler scores padding row");
                        }
                    }
                }
            }
        }
        printf("  readable_rows_are_real conformance PASSED OK.\n");
    }

    {
        printf("\n[3] Lean theorem kv_cells_disjoint over all plan/row pairs...\n");
        for (uint32_t a = 0; a <= g.gamma_max; ++a) {
            for (uint32_t pb = 0; pb <= 2; ++pb) {
                for (uint32_t fl = 0; fl <= 1; ++fl) {
                    polygraph::Plan p{a, pb, fl, 0, 0, 0, 0};
                    for (uint32_t r = 0; r < g.rows(); ++r) {
                        for (uint32_t c = r + 1; c < g.rows(); ++c) {
                            const bool same_cell =
                                polygraph::row_seq(p, g, r) == polygraph::row_seq(p, g, c) &&
                                polygraph::row_pos(p, r) == polygraph::row_pos(p, c);
                            TEST_ASSERT(!same_cell,
                                "kv_cells_disjoint violated: distinct rows share a KV cell");
                        }
                    }
                }
            }
        }
        printf("  kv_cells_disjoint conformance PASSED OK.\n");
    }

    {
        printf("\n[4] Lean theorems sanitize_within_limits + real_rows_fit...\n");
        for (uint32_t a = 0; a <= g.gamma_max + 4; ++a) {  // include out-of-range plans
            for (uint32_t pb = 0; pb <= 2; ++pb) {
                for (uint32_t fl = 0; fl <= 1; ++fl) {
                    polygraph::Plan p{a, pb, fl, 0, 0, 0, 0};
                    const polygraph::Plan q = polygraph::sanitize(g, p);
                    TEST_ASSERT(q.active_len <= g.gamma_max,
                        "sanitize_within_limits violated: clamped plan exceeds gamma_max");
                    TEST_ASSERT(polygraph::plan_valid(g, q),
                        "sanitize must produce a plan_valid result");
                    TEST_ASSERT(q.active_len + 1 <= g.rows(),
                        "real_rows_fit violated: real rows exceed pinned batch");
                }
            }
        }
        printf("  sanitize_within_limits + real_rows_fit conformance PASSED OK.\n");
    }

    printf("\nAll test-polygraph-plan assertions PASSED cleanly (%ld plans x geometry).\n",
        plans_checked);
    return 0;
}
