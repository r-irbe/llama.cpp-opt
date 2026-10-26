#include "spec-counterfactual.h"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

int main() {
    printf("=================================================================\n");
    printf("  TEST: Full-Information Counterfactual Prefix Attribution Engine\n");
    printf("  Reference: Not-a-Bandit Theorem 3 (arXiv:2510.20064)           \n");
    printf("=================================================================\n");

    // 1. Initial State & Invariants
    printf("\n[1] Verifying engine initialization and arm distribution...\n");
    {
        spec_counterfactual_engine engine(8, 0.1, 0.15, 1.0);
        TEST_ASSERT(engine.k_max == 8, "k_max should be 8");
        TEST_ASSERT(engine.arms.size() == 9, "arms count should be k_max + 1 (9)");
        TEST_ASSERT(engine.arms[0].k == 0, "Arm 0 must represent target-only baseline (k=0)");
        TEST_ASSERT(engine.arms[8].k == 8, "Arm 8 must represent k=8");

        auto probs = engine.get_action_probabilities();
        TEST_ASSERT(probs.size() == 9, "Action probabilities size mismatch");
        for (double p : probs) {
            TEST_ASSERT(std::abs(p - (1.0 / 9.0)) < 1e-4, "Initial probabilities should be uniform");
        }
        printf("  Initialization check PASSED OK.\n");
    }

    // 2. Counterfactual Reward Calculation (Theorem 3)
    printf("\n[2] Verifying counterfactual prefix attribution rewards across candidate lengths...\n");
    {
        spec_counterfactual_engine engine(8, 0.1, 0.15, 1.0);
        // Suppose proposed length K = 6, but only 3 tokens were accepted (A = 3)
        // Candidate k = 0 (target-only): accepted = 0, drafted = 0 -> reward = 0.0
        double r0 = engine.compute_counterfactual_reward(0, 6, 3);
        TEST_ASSERT(std::abs(r0 - 0.0) < 1e-5, "Target-only baseline reward must be 0.0");

        // Candidate k = 2: k <= A (2 <= 3): accepted = 2, drafted = 2 -> reward = 2*1.0 - 2*0.15 = 1.70
        double r2 = engine.compute_counterfactual_reward(2, 6, 3);
        TEST_ASSERT(std::abs(r2 - 1.70) < 1e-5, "Counterfactual reward for k=2 mismatch");

        // Candidate k = 3: k == A: accepted = 3, drafted = 3 -> reward = 3*1.0 - 3*0.15 = 2.55 (optimal)
        double r3 = engine.compute_counterfactual_reward(3, 6, 3);
        TEST_ASSERT(std::abs(r3 - 2.55) < 1e-5, "Counterfactual reward for k=3 mismatch");

        // Candidate k = 5: k > A: accepted = 3, drafted = 5 -> reward = 3*1.0 - 5*0.15 = 2.25
        double r5 = engine.compute_counterfactual_reward(5, 6, 3);
        TEST_ASSERT(std::abs(r5 - 2.25) < 1e-5, "Counterfactual reward for k=5 mismatch");

        // Candidate k = 6: actual proposed: accepted = 3, drafted = 6 -> reward = 3*1.0 - 6*0.15 = 2.10
        double r6 = engine.compute_counterfactual_reward(6, 6, 3);
        TEST_ASSERT(std::abs(r6 - 2.10) < 1e-5, "Counterfactual reward for k=6 mismatch");

        printf("  Counterfactual reward calculations PASSED OK.\n");
    }

    // 3. Simultaneous Full-Information Hedge Weight Updates
    printf("\n[3] Verifying simultaneous Hedge weight update and convergence...\n");
    {
        spec_counterfactual_engine engine(6, 0.2, 0.2, 1.0);
        // Repeatedly simulate target verification where only 2 tokens are accepted when 5 tokens proposed
        for (int step = 0; step < 20; ++step) {
            engine.update_full_information(5, 2);
        }

        // Under repeated acceptance of 2 tokens, candidate k=2 gives reward 2*1.0 - 2*0.2 = 1.60
        // Candidate k=5 gives 2*1.0 - 5*0.2 = 1.00
        // Candidate k=0 gives 0.00
        int best_k = engine.select_draft_length(true);
        TEST_ASSERT(best_k == 2, "Hedge should converge to optimal prefix length k=2");

        auto probs = engine.get_action_probabilities();
        TEST_ASSERT(probs[2] > probs[0], "Prob(k=2) must exceed Prob(k=0)");
        TEST_ASSERT(probs[2] > probs[5], "Prob(k=2) must exceed Prob(k=5)");
        printf("  Convergence to optimal prefix length k=2 PASSED OK (Prob[2]=%.4f).\n", probs[2]);
    }

    // 4. Low Acceptance Scenario & Shift to Target-Only Arm (k=0)
    printf("\n[4] Verifying shift to target-only baseline arm under zero acceptances...\n");
    {
        spec_counterfactual_engine engine(4, 0.3, 0.25, 1.0);
        // High uncertainty: 0 tokens accepted across repeated proposals
        for (int step = 0; step < 30; ++step) {
            engine.update_full_information(4, 0);
        }

        // For any k > 0, reward is 0 - k * 0.25 < 0. For k = 0, reward is 0.0.
        // Therefore k=0 (target-only baseline) must dominate.
        int best_k = engine.select_draft_length(true);
        TEST_ASSERT(best_k == 0, "Engine must fall back to target-only baseline arm (k=0) when acceptance is zero");
        printf("  Target-only arm fallback check PASSED OK (best_k=%d).\n", best_k);
    }

    printf("\nAll test-spec-counterfactual assertions PASSED cleanly!\n");
    return 0;
}
