#pragma once

#include "common.h"
#include "spec-counterfactual.h"
#include "spec-systems-tuner.h"

struct llama_model;

struct spec_tuner_arm {
    float  value;
    double Q     = 0.0; // mean per-step Tokens-Per-Second (TPS)
    int    N     = 0;
    double M2    = 0.0; // Welford running sum of squared deviations for Thompson sampling
};

struct spec_tuner_coord {
    std::string name;
    std::vector<spec_tuner_arm> arms;
    int current_idx = 0;
    int best_idx    = 0;
    int user_idx    = 0;

    int select_epsilon_greedy(double epsilon) const;
    int select_thompson_sampling() const;

    void update(double reward);

    void reset_scores();

    void build_grid_float(float lo, float hi, int n_points, float user_value);
    void build_grid_int(int lo, int hi, int step, int user_value);
    int  find_nearest_arm(float value) const;
};

struct spec_tuner {
    bool     enabled    = false;

    double   epsilon    = 0.15;  // 15% explore, 85% exploit

    int      configured_n_max = 0;
    int      dflash_min_samples_per_arm = 3;
    int      dflash_recovery_probe_interval = 128;
    double   dflash_quarantine_ratio = 0.90;
    std::vector<bool> dflash_quarantined;
    int      dflash_probe_cursor = 0;
    bool     dflash_last_exploratory = false;
    bool     dflash_last_recovery_probe = false;
    uint64_t n_target_only_selections = 0;
    uint64_t n_dflash_selections = 0;
    uint64_t n_exploratory_selections = 0;
    uint64_t n_quarantines = 0;
    uint64_t n_recovery_probes = 0;

    // task-change detection (per-call)
    // If tuner goes bad for 30 consecutive calls, reset the tuner.
    double   step_ema        = 0.0;
    double   step_ema_alpha  = 0.05;
    double   step_drop_pct   = 0.30;
    int      n_low           = 0;
    int      reset_after     = 30;
    int      cooldown        = 0;
    int      cooldown_max    = 50;
    int      n_resets        = 0;

    int      last_n_drafted  = 0;
    uint64_t n_calls         = 0;
    int      log_every       = 50;

    // per-request tracking
    uint64_t n_requests      = 0;
    int64_t  t_tuner_us      = 0;
    double   ema_tps         = 0.0;
    double   ema_alpha       = 0.3;

    // online draft acceptance-rate and token-entropy gating (DFlash / DSpark / MTP / CCD)
    double   ema_accept_rate        = 0.0;
    double   ema_token_entropy      = 0.0;
    float    entropy_gate_threshold = 3.5f;
    bool     entropy_gated          = false;

    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    std::vector<spec_tuner_coord> coords;

    // Full-information counterfactual prefix attribution engine (Not-a-Bandit Theorem 3)
    spec_counterfactual_engine counterfactual;

    // Multi-coordinate Discounted Thompson Sampling for hardware systems parameters
    spec_systems_tuner systems_tuner;

    // Not-a-Bandit warmup: minimum full-information observations before the Hedge
    // distribution over candidate draft lengths replaces exploration-based selection.
    int counterfactual_warmup_steps = 8;
    uint64_t n_counterfactual_selections = 0;

    void init(common_speculative_type type, const common_params_speculative & user_params, const llama_model * model_tgt);
    void propose(common_params_speculative & params);
    void accept_feedback(int n_accepted, int n_drafted, double step_tps);
    void update_entropy_gate(float token_entropy);
    void end_of_request(double slot_tps, int n_past, common_params_speculative & active_params);
    void enforce_constraints(common_params_speculative & params);
    void print_best() const;
    void reset_exploration();

    void write_best(common_params_speculative & params) const;

    /**
     * Feed the systems tuner's ConcurrencyKnee with the current active-stream /
     * queue-depth sample. When saturated (S >= C_knee), the next propose() throttles
     * speculative draft length to protect target prefill and decoding.
     */
    void observe_concurrency(double active_streams_or_queue_depth);

    bool has_dflash_target_only_arm() const {
        return enabled && (spec_type == COMMON_SPECULATIVE_TYPE_DFLASH || spec_type == COMMON_SPECULATIVE_TYPE_DSPARK || spec_type == COMMON_SPECULATIVE_TYPE_MTP) && configured_n_max > 0;
    }

private:
    int select_dflash_arm(spec_tuner_coord & coord);
    void update_dflash_quarantine();
};
