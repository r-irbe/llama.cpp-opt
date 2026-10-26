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

    void init_arms() {
        arms.clear();
        arms.resize(k_max + 1);
        for (int k = 0; k <= k_max; ++k) {
            arms[k].k = k;
            arms[k].weight = 1.0;
            arms[k].cum_reward = 0.0;
            arms[k].cum_loss = 0.0;
            arms[k].n_pulls = 0;
        }
    }

    /**
     * Get probability distribution over candidate draft lengths k in [0, k_max]
     * using current Hedge weights.
     */
    std::vector<double> get_action_probabilities() const {
        std::vector<double> probs(arms.size(), 0.0);
        double sum_weights = 0.0;
        for (const auto & arm : arms) {
            sum_weights += arm.weight;
        }
        if (sum_weights <= 0.0) {
            std::fill(probs.begin(), probs.end(), 1.0 / (double)arms.size());
            return probs;
        }
        for (size_t i = 0; i < arms.size(); ++i) {
            probs[i] = arms[i].weight / sum_weights;
        }
        return probs;
    }

    /**
     * Select candidate draft length k according to Hedge distribution (or argmax if exploit_only).
     */
    int select_draft_length(bool exploit_only = false, double sample_rand = 0.0) const {
        if (arms.empty()) {
            return 0;
        }
        if (exploit_only) {
            int best_k = 0;
            double max_w = -1.0;
            for (const auto & arm : arms) {
                if (arm.weight > max_w) {
                    max_w = arm.weight;
                    best_k = arm.k;
                }
            }
            return best_k;
        }

        const auto probs = get_action_probabilities();
        double cum = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (sample_rand <= cum) {
                return arms[i].k;
            }
        }
        return arms.back().k;
    }

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
    double compute_counterfactual_reward(int candidate_k, int proposed_length, int actual_accepted) const {
        int eff_accepted = 0;
        int eff_drafted  = candidate_k;

        if (candidate_k <= actual_accepted) {
            eff_accepted = candidate_k;
        } else {
            eff_accepted = actual_accepted;
        }

        // Net utility: reward for accepted tokens minus overhead for drafted tokens
        // For candidate_k = 0 (target-only baseline), net utility is 0.0 (no draft overhead, 0 speculative accepts)
        double r = (double)eff_accepted * reward_per_accept - (double)eff_drafted * cost_per_draft;
        return r;
    }

    /**
     * Full-Information Update across all k in [0, k_max].
     * Uses unbiased prefix attribution with bounded variance K^2 / 4.
     */
    void update_full_information(int proposed_length, int actual_accepted) {
        if (proposed_length < 0) proposed_length = 0;
        if (actual_accepted < 0) actual_accepted = 0;
        if (actual_accepted > proposed_length) actual_accepted = proposed_length;

        // Record pull count for the actual proposed length
        if (proposed_length >= 0 && proposed_length < (int)arms.size()) {
            arms[proposed_length].n_pulls++;
        }

        // Find max potential counterfactual reward in this step for loss normalization
        double max_step_reward = -1e9;
        std::vector<double> step_rewards(arms.size(), 0.0);
        for (size_t i = 0; i < arms.size(); ++i) {
            step_rewards[i] = compute_counterfactual_reward(arms[i].k, proposed_length, actual_accepted);
            if (step_rewards[i] > max_step_reward) {
                max_step_reward = step_rewards[i];
            }
        }

        // Hedge multiplicative weight update:
        // loss_k = max_step_reward - r_k >= 0
        // weight_k <- weight_k * exp(-eta * loss_k)
        double max_log_w = -1e9;
        for (size_t i = 0; i < arms.size(); ++i) {
            double loss = max_step_reward - step_rewards[i];
            arms[i].cum_reward += step_rewards[i];
            arms[i].cum_loss += loss;

            arms[i].weight *= std::exp(-eta * loss);
            // Protect against zero / underflow
            if (arms[i].weight < 1e-12) {
                arms[i].weight = 1e-12;
            }
            if (arms[i].weight > max_log_w) {
                max_log_w = arms[i].weight;
            }
        }

        // Normalize weights to avoid numerical overflow
        if (max_log_w > 0.0) {
            for (auto & arm : arms) {
                arm.weight /= max_log_w;
            }
        }
    }

    void reset() {
        init_arms();
    }

    /**
     * Total number of full-information observations recorded across all arms.
     * Used as the warmup gate before the Hedge distribution drives selection.
     */
    uint64_t total_pulls() const {
        uint64_t total = 0;
        for (const auto & arm : arms) {
            total += (uint64_t) arm.n_pulls;
        }
        return total;
    }
};
