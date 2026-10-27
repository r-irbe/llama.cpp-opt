#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

/**
 * Full-Information Counterfactual Prefix Attribution Engine
 * Reference: Not-a-Bandit (Theorem 3, arXiv:2510.20064)
 *
 * Evaluates acceptance across all candidate speculative draft lengths k in [0, n_proposed]
 * simultaneously from a single target forward pass with zero extra target queries.
 *
 * In speculative verification of a draft trajectory of length K, target verification is an
 * autoregressive prefix check. If k tokens are proposed, any candidate prefix of length j <= k
 * can be evaluated simultaneously counterfactually:
 * - If the true acceptance count is A (with 0 <= A <= K), then a candidate policy that proposed
 *   j tokens would have had min(j, A) accepted tokens if j <= A, or A accepted tokens if j > A
 *   (with j - A rejected tokens).
 *
 * Bounded variance K^2 / 4 allows updating Hedge / Exponential Weights across all candidate
 * draft lengths simultaneously, collapsing regret from O(sqrt(N * K * T)) to O(sqrt(T * ln K)).
 */

struct spec_counterfactual_arm {
    int    k          = 0;    // candidate draft length k in [0, K_max]
    double weight     = 1.0;  // Hedge exponential weight
    double cum_reward = 0.0;  // Cumulative counterfactual reward
    double cum_loss   = 0.0;  // Cumulative counterfactual loss (regret)
    int    n_pulls    = 0;    // Times this arm was the selected proposal length
};

class spec_counterfactual_engine {
public:
    int    k_max;             // Maximum candidate draft length (e.g. 8 or 16)
    double eta;               // Hedge learning rate parameter
    double cost_per_draft;    // Cost penalty per drafted token (overhead proxy)
    double reward_per_accept; // Value of an accepted token
    std::vector<spec_counterfactual_arm> arms;

    explicit spec_counterfactual_engine(int max_draft_len = 8,
                                        double learning_rate = 0.1,
                                        double draft_cost = 0.15,
                                        double accept_reward = 1.0)
        : k_max(std::max(1, max_draft_len)),
          eta(learning_rate > 0.0 ? learning_rate : 0.1),
          cost_per_draft(draft_cost),
          reward_per_accept(accept_reward) {
        init_arms();
    }

    void init_arms();

    /**
     * Get probability distribution over candidate draft lengths k in [0, k_max]
     * using current Hedge weights.
     */
    std::vector<double> get_action_probabilities() const;

    /**
     * Select candidate draft length k according to Hedge distribution (or argmax if exploit_only).
     */
    int select_draft_length(bool exploit_only = false, double sample_rand = 0.0) const;

    /**
     * Compute counterfactual reward for candidate length k given:
     * - proposed_length: actual length proposed to target verification (K)
     * - actual_accepted: number of tokens accepted along the prefix (A, where 0 <= A <= proposed_length)
     *
     * Following Theorem 3:
     * If policy had proposed k:
     * - If k <= actual_accepted: all k tokens would be accepted.
     *   effective_accepted = k, effective_drafted = k.
     * - If k > actual_accepted: exactly actual_accepted tokens would be accepted,
     *   and (k - actual_accepted) tokens would be rejected.
     *   effective_accepted = actual_accepted, effective_drafted = k.
     */
    double compute_counterfactual_reward(int candidate_k, int proposed_length, int actual_accepted) const;

    /**
     * Full-Information Update across all k in [0, k_max].
     * Uses unbiased prefix attribution with bounded variance K^2 / 4.
     */
    void update_full_information(int proposed_length, int actual_accepted);

    void reset();

    /**
     * Total number of full-information observations recorded across all arms.
     * Used as the warmup gate before the Hedge distribution drives selection.
     */
    uint64_t total_pulls() const;
};
