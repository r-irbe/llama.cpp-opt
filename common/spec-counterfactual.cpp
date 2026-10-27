#include "spec-counterfactual.h"

#include <cmath>

void spec_counterfactual_engine::init_arms() {
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

std::vector<double> spec_counterfactual_engine::get_action_probabilities() const {
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

int spec_counterfactual_engine::select_draft_length(bool exploit_only, double sample_rand) const {
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

double spec_counterfactual_engine::compute_counterfactual_reward(int candidate_k, int proposed_length, int actual_accepted) const {
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

void spec_counterfactual_engine::update_full_information(int proposed_length, int actual_accepted) {
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

void spec_counterfactual_engine::reset() {
    init_arms();
}

uint64_t spec_counterfactual_engine::total_pulls() const {
    uint64_t total = 0;
    for (const auto & arm : arms) {
        total += (uint64_t) arm.n_pulls;
    }
    return total;
}
