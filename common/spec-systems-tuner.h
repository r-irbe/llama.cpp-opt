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

    void init_int_range(int lo, int hi, int step, int default_val);

    int find_nearest(float val) const;

    int select_discounted_thompson(std::mt19937 & rng);

    void update_discounted(double reward, double gamma = 0.95);
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

    void init_defaults(int max_threads = 16, int numa_count = 8);

    /**
     * Check Concurrency Knee:
     * If active streams / queue depth S >= C_knee, host memory bus is saturated.
     * Speculation must be throttled to target-only (n_max -> 0) to protect target prefill and decoding.
     */
    bool check_concurrency_knee(double active_streams_or_queue_depth);

    void propose_systems_params(int & out_threads_draft, int & out_numa_domain, int & out_prefetch_depth);

    void update_feedback(double step_tps);

private:
    std::mt19937 m_rng;
};
