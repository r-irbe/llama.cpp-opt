#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>

/**
 * Multi-Coordinate Discounted Thompson Sampling for Hardware Systems Parameters
 *
 * Coordinates:
 * - n_threads_draft: Thread count allocated to drafter CCX / worker pool
 * - companion_numa_domain: NUMA domain affinity for drafter execution
 * - paged_kv_dma_prefetch_depth: Depth of async host paged KV DMA prefetch
 *
 * Also includes ConcurrencyKnee:
 * Dynamically throttles speculative draft length (n_max -> 0, target-only baseline)
 * when host memory bus or queue depth is saturated (S >= C_knee).
 */

struct spec_systems_arm {
    float  value        = 0.0f;
    double mean_reward  = 0.0;
    double variance     = 1.0;
    double pseudo_count = 1.0;
    int    pulls        = 0;
};

struct spec_systems_coord {
    std::string name;
    std::vector<spec_systems_arm> arms;
    int current_idx = 0;
    int best_idx    = 0;

    void init_int_range(int lo, int hi, int step, int default_val) {
        arms.clear();
        for (int v = lo; v <= hi; v += step) {
            spec_systems_arm arm;
            arm.value = (float)v;
            arms.push_back(arm);
        }
        if (arms.empty() || (int)arms.back().value != hi) {
            spec_systems_arm arm;
            arm.value = (float)hi;
            arms.push_back(arm);
        }
        current_idx = find_nearest((float)default_val);
        best_idx    = current_idx;
    }

    int find_nearest(float val) const {
        int best_i = 0;
        float min_diff = 1e9f;
        for (int i = 0; i < (int)arms.size(); ++i) {
            float d = std::fabs(arms[i].value - val);
            if (d < min_diff) {
                min_diff = d;
                best_i = i;
            }
        }
        return best_i;
    }

    int select_discounted_thompson(std::mt19937 & rng) {
        if (arms.empty()) return 0;
        int chosen = best_idx;
        double best_sample = -1e30;

        for (int i = 0; i < (int)arms.size(); ++i) {
            const auto & arm = arms[i];
            double sigma = std::sqrt(std::max(1e-4, arm.variance / std::max(1.0, arm.pseudo_count)));
            std::normal_distribution<double> dist(arm.mean_reward, sigma);
            double sample = dist(rng);
            if (sample > best_sample) {
                best_sample = sample;
                chosen = i;
            }
        }
        current_idx = chosen;
        return chosen;
    }

    void update_discounted(double reward, double gamma = 0.95) {
        if (arms.empty() || current_idx < 0 || current_idx >= (int)arms.size()) return;

        // Apply discount factor gamma to all arms (discounted Thompson sampling for non-stationary environments)
        for (auto & arm : arms) {
            arm.pseudo_count *= gamma;
            if (arm.pseudo_count < 1.0) arm.pseudo_count = 1.0;
        }

        auto & arm = arms[current_idx];
        arm.pulls++;
        arm.pseudo_count += 1.0;
        double delta = reward - arm.mean_reward;
        arm.mean_reward += delta / arm.pseudo_count;
        arm.variance = gamma * arm.variance + (1.0 - gamma) * (delta * (reward - arm.mean_reward));
        if (arm.variance < 1e-4) arm.variance = 1e-4;

        // Update best
        double max_r = -1e30;
        for (int i = 0; i < (int)arms.size(); ++i) {
            if (arms[i].mean_reward > max_r) {
                max_r = arms[i].mean_reward;
                best_idx = i;
            }
        }
    }
};

class spec_systems_tuner {
public:
    bool   enabled = false;
    double discount_gamma = 0.95; // Non-stationary forgetting factor
    double c_knee = 8.0;          // Concurrency knee threshold (active client streams or queue depth)
    bool   knee_saturated = false;

    spec_systems_coord coord_n_threads_draft;
    spec_systems_coord coord_companion_numa;
    spec_systems_coord coord_kv_dma_prefetch_depth;

    spec_systems_tuner() {
        std::random_device rd;
        m_rng.seed(rd());
        init_defaults();
    }

    void init_defaults(int max_threads = 16, int numa_count = 8) {
        // n_threads_draft: [1, 2, 4, 8, max_threads]
        coord_n_threads_draft.name = "n_threads_draft";
        coord_n_threads_draft.init_int_range(1, std::max(1, max_threads), 2, 4);

        // companion_numa_domain: [0, numa_count - 1]
        coord_companion_numa.name = "companion_numa_domain";
        coord_companion_numa.init_int_range(0, std::max(0, numa_count - 1), 1, 0);

        // host paged KV DMA prefetch depth: [0, 1, 2, 4, 8]
        coord_kv_dma_prefetch_depth.name = "kv_dma_prefetch_depth";
        coord_kv_dma_prefetch_depth.init_int_range(0, 8, 2, 2);
    }

    /**
     * Check Concurrency Knee:
     * If active streams / queue depth S >= C_knee, host memory bus is saturated.
     * Speculation must be throttled to target-only (n_max -> 0) to protect target prefill and decoding.
     */
    bool check_concurrency_knee(double active_streams_or_queue_depth) {
        knee_saturated = (active_streams_or_queue_depth >= c_knee);
        return knee_saturated;
    }

    void propose_systems_params(int & out_threads_draft, int & out_numa_domain, int & out_prefetch_depth) {
        out_threads_draft = (int)coord_n_threads_draft.arms[coord_n_threads_draft.select_discounted_thompson(m_rng)].value;
        out_numa_domain   = (int)coord_companion_numa.arms[coord_companion_numa.select_discounted_thompson(m_rng)].value;
        out_prefetch_depth = (int)coord_kv_dma_prefetch_depth.arms[coord_kv_dma_prefetch_depth.select_discounted_thompson(m_rng)].value;
    }

    void update_feedback(double step_tps) {
        coord_n_threads_draft.update_discounted(step_tps, discount_gamma);
        coord_companion_numa.update_discounted(step_tps, discount_gamma);
        coord_kv_dma_prefetch_depth.update_discounted(step_tps, discount_gamma);
    }

private:
    std::mt19937 m_rng;
};
