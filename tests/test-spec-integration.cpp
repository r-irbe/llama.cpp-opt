#include "spec-arbiter.h"
#include "spec-counterfactual.h"
#include "spec-systems-tuner.h"
#include "spec-tuner.h"

#include <cstdio>
#include <cassert>
#include <vector>
#include <cmath>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

int main() {
    printf("=================================================================\n");
    printf("  TEST: End-to-End Speculative Arbiter, Systems & TreeGraft Integration\n");
    printf("=================================================================\n");

    // 1. Verify TreeGraft fusion wiring across multiple drafter stages
    printf("\n[1] Verifying TreeGraft multi-branch fusion with realistic token streams...\n");
    {
        std::vector<llama_token> ngram_cand  = {100, 101, 102, 103};
        std::vector<llama_token> neural_cand = {100, 101, 202, 203};

        auto fused = spec_arbiter::fuse_treegraft(ngram_cand, neural_cand, 8);
        TEST_ASSERT(fused.is_tree, "Fused candidates with diverging paths must be marked as tree");
        TEST_ASSERT(fused.primary_source == SPEC_DRAFTER_SOURCE_TREEGRAFT, "Source should be TREEGRAFT");
        TEST_ASSERT(fused.tokens.size() == 6, "Total unique tokens in tree should be 6");
        TEST_ASSERT(fused.tokens[0] == 100 && fused.tokens[1] == 101, "Trunk tokens should match common prefix");
        
        // Check tree topology
        TEST_ASSERT(fused.tree_nodes[0].parent_idx == -1, "Root token parent must be -1");
        TEST_ASSERT(fused.tree_nodes[1].parent_idx == 0, "Second token parent must be 0");
        TEST_ASSERT(fused.tree_nodes[2].parent_idx == 1, "First neural branch parent must be trunk top (1)");
        TEST_ASSERT(fused.tree_nodes[4].parent_idx == 1, "First ngram branch parent must be trunk top (1)");

        printf("  TreeGraft integration check PASSED OK.\n");
    }

    // 2. Verify Systems Tuner ConcurrencyKnee throttling
    printf("\n[2] Verifying ConcurrencyKnee detection and throttling to target-only...\n");
    {
        spec_systems_tuner sys_tuner;
        sys_tuner.enabled = true;
        sys_tuner.c_knee = 8.0;

        // Low concurrency (3 active slots): knee should NOT be saturated
        bool saturated_low = sys_tuner.check_concurrency_knee(3.0);
        TEST_ASSERT(!saturated_low, "Knee must not be saturated at 3 streams (c_knee = 8.0)");
        TEST_ASSERT(!sys_tuner.knee_saturated, "Knee saturated flag should be false");

        // High concurrency (16 active slots): knee MUST be saturated
        bool saturated_high = sys_tuner.check_concurrency_knee(16.0);
        TEST_ASSERT(saturated_high, "Knee must be saturated at 16 streams (c_knee = 8.0)");
        TEST_ASSERT(sys_tuner.knee_saturated, "Knee saturated flag should be true");

        // Verify propose reflects systems parameters
        int threads = 0, numa = 0, prefetch = 0;
        sys_tuner.propose_systems_params(threads, numa, prefetch);
        TEST_ASSERT(threads >= 1, "Threads should be >= 1");
        TEST_ASSERT(numa >= 0, "NUMA domain should be >= 0");
        TEST_ASSERT(prefetch >= 0, "Prefetch depth should be >= 0");

        printf("  ConcurrencyKnee check PASSED OK.\n");
    }

    // 3. Verify Speculative Tuner Feedback Loop with Counterfactual Attribution
    printf("\n[3] Verifying Speculative Tuner integration with Counterfactual Attribution...\n");
    {
        spec_tuner tuner;
        common_params_speculative sparams;
        sparams.n_max = 8;
        sparams.n_min = 1;
        sparams.p_min = 0.5f;

        tuner.init(COMMON_SPECULATIVE_TYPE_DFLASH, sparams, nullptr);
        TEST_ASSERT(tuner.enabled, "Spec tuner should be enabled");
        TEST_ASSERT(tuner.has_dflash_target_only_arm(), "DFlash should register target-only arm");

        // Simulate 10 acceptance feedback cycles
        for (int i = 0; i < 10; ++i) {
            tuner.accept_feedback(3, 6, 45.0); // 3 accepted out of 6 drafted
        }

        // Propose new parameters
        common_params_speculative proposed = sparams;
        tuner.propose(proposed);
        TEST_ASSERT(proposed.n_max >= 0 && proposed.n_max <= 8, "Proposed n_max must be within bounds [0, 8]");

        // Runtime wiring path: observe_concurrency saturates the knee (16
        // streams >= c_knee 8), so the next propose must throttle to the
        // target-only arm (n_max = 0).
        tuner.observe_concurrency(16.0);
        common_params_speculative throttled = sparams;
        tuner.propose(throttled);
        TEST_ASSERT(tuner.systems_tuner.knee_saturated, "16 active streams must saturate the knee at c_knee 8");
        TEST_ASSERT(throttled.n_max == 0, "Saturated knee must throttle speculation to target-only (n_max = 0)");

        // A low stream count clears the knee again.
        tuner.observe_concurrency(2.0);
        TEST_ASSERT(!tuner.systems_tuner.knee_saturated, "2 active streams must not saturate the knee at c_knee 8");

        // c_knee = 0 disables the systems tuner: no saturation ever.
        spec_tuner tuner_no_knee;
        common_params_speculative sparams_no_knee = sparams;
        sparams_no_knee.c_knee = 0.0f;
        tuner_no_knee.init(COMMON_SPECULATIVE_TYPE_DFLASH, sparams_no_knee, nullptr);
        tuner_no_knee.observe_concurrency(64.0);
        TEST_ASSERT(!tuner_no_knee.systems_tuner.knee_saturated, "c_knee 0 must disable ConcurrencyKnee saturation");
        common_params_speculative unthrottled = sparams_no_knee;
        tuner_no_knee.propose(unthrottled);
        TEST_ASSERT(unthrottled.n_max > 0, "Disabled knee must still allow speculative drafting");

        // Cold start must never propose target-only (n_max 0): an unbuilt
        // tuner that proposes 0 would receive no draft feedback and could
        // never warm its arms -- a permanently dead tuner. Exploration
        // sweeps the positive draft arms first, then probes target-only.
        spec_tuner cold;
        common_params_speculative cold_params;
        cold_params.n_max = 8;
        cold.init(COMMON_SPECULATIVE_TYPE_DFLASH, cold_params, nullptr);
        common_params_speculative first = cold_params;
        cold.propose(first);
        TEST_ASSERT(first.n_max > 0, "First cold-start proposal must draft (n_max > 0)");

        // After every positive arm is warmed, target-only becomes reachable
        // (the final undersampled arm of the sweep).
        bool reached_zero = false;
        for (int i = 0; i < 64 && !reached_zero; ++i) {
            common_params_speculative p = cold_params;
            cold.propose(p);
            if (p.n_max == 0) {
                reached_zero = true;
            } else {
                cold.accept_feedback(p.n_max, p.n_max, 40.0);
            }
        }
        TEST_ASSERT(reached_zero, "Target-only arm must remain reachable after the positive arms warm");

        printf("  Speculative Tuner feedback loop check PASSED OK.\n");
    }

    printf("\nAll test-spec-integration assertions PASSED cleanly!\n");
    return 0;
}
