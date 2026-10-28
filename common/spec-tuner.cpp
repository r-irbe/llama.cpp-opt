#include "spec-tuner.h"

#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <iomanip>
#include <random>

int spec_tuner_coord::find_nearest_arm(float value) const {
    int idx = 0;
    float best_dist = 1e30f;
    for (int i = 0; i < (int)arms.size(); i++) {
        float dist = std::fabs(arms[i].value - value);
        if (dist < best_dist) {
            best_dist = dist;
            idx       = i;
        }
    }
    return idx;
}

int spec_tuner_coord::select_epsilon_greedy(double epsilon) const {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    if (coin(rng) < epsilon) {
        std::uniform_int_distribution<int> dist(0, (int)arms.size() - 1);
        return dist(rng);
    }
    return best_idx;
}

int spec_tuner_coord::select_thompson_sampling() const {
    if (arms.empty()) {
        return 0;
    }
    static thread_local std::mt19937 rng(std::random_device{}());

    int chosen_idx = best_idx;
    double best_sample = -1e30;

    for (int i = 0; i < (int)arms.size(); ++i) {
        const auto & arm = arms[i];
        double sample = arm.Q;
        if (arm.N == 0) {
            std::normal_distribution<double> dist(0.0, 100.0);
            sample = dist(rng);
        } else if (arm.N == 1) {
            const double prior_sigma = std::max(1.0, std::fabs(arm.Q) * 0.25);
            std::normal_distribution<double> dist(arm.Q, prior_sigma);
            sample = dist(rng);
        } else {
            const double var = std::max(1e-4, arm.M2 / (double)(arm.N - 1));
            const double sigma = std::sqrt(var / (double)arm.N);
            std::normal_distribution<double> dist(arm.Q, sigma);
            sample = dist(rng);
        }
        if (i == 0 || sample > best_sample) {
            best_sample = sample;
            chosen_idx  = i;
        }
    }
    return chosen_idx;
}

void spec_tuner_coord::update(double reward) {
    auto & arm = arms[current_idx];
    arm.N += 1;
    const double delta = reward - arm.Q;
    arm.Q += delta / arm.N;
    const double delta2 = reward - arm.Q;
    arm.M2 += delta * delta2;

    double best_Q = -1e30;
    for (int i = 0; i < (int)arms.size(); i++) {
        if (arms[i].N > 0 && arms[i].Q > best_Q) {
            best_Q   = arms[i].Q;
            best_idx = i;
        }
    }
}

void spec_tuner_coord::reset_scores() {
    for (auto & arm : arms) {
        arm.Q  = 0.0;
        arm.N  = 0;
        arm.M2 = 0.0;
    }
    current_idx = user_idx;
    best_idx    = user_idx;
}

void spec_tuner_coord::build_grid_float(float lo, float hi, int n_points, float user_value) {
    arms.clear();
    for (int i = 0; i < n_points; i++) {
        float v = lo + (hi - lo) * i / std::max(1, n_points - 1);
        arms.push_back({v, 0.0, 0});
    }
    bool found = false;
    for (auto & a : arms) {
        if (std::fabs(a.value - user_value) < 1e-6f) { found = true; break; }
    }
    if (!found) {
        arms.push_back({user_value, 0.0, 0});
        std::sort(arms.begin(), arms.end(), [](const spec_tuner_arm & a, const spec_tuner_arm & b) {
            return a.value < b.value;
        });
    }
}

void spec_tuner_coord::build_grid_int(int lo, int hi, int step, int user_value) {
    arms.clear();
    for (int v = lo; v <= hi; v += step) {
        arms.push_back({(float)v, 0.0, 0});
    }
    if (arms.empty() || (int)arms.back().value != hi) {
        arms.push_back({(float)hi, 0.0, 0});
    }
    bool found = false;
    for (auto & a : arms) {
        if ((int)a.value == user_value) { found = true; break; }
    }
    if (!found && user_value >= lo && user_value <= hi) {
        arms.push_back({(float)user_value, 0.0, 0});
        std::sort(arms.begin(), arms.end(), [](const spec_tuner_arm & a, const spec_tuner_arm & b) {
            return a.value < b.value;
        });
    }
}

void spec_tuner::reset_exploration() {
    n_resets++;
    LOG_DBG("Autotune task change detected (n_low=%d) -- resetting MAB (reset #%d)\n", n_low, n_resets);
    for (auto & coord : coords) {
        coord.reset_scores();
    }
    dflash_quarantined.assign(dflash_quarantined.size(), false);
    dflash_probe_cursor = 0;
    dflash_last_exploratory = false;
    dflash_last_recovery_probe = false;
    if (has_dflash_target_only_arm() && !coords.empty() && !coords[0].arms.empty()) {
        coords[0].current_idx = 0;
        coords[0].best_idx = 0;
    }
    n_low = 0;
    cooldown = cooldown_max;
    step_ema = 0.0;
    n_calls = 0;
    ema_accept_rate = 0.0;
    ema_token_entropy = 0.0;
    entropy_gated = false;
}

void spec_tuner::write_best(common_params_speculative & params) const {
    for (const auto & coord : coords) {
        float val = coord.arms[coord.best_idx].value;
        if      (coord.name == "n_max") {
            params.n_max = ((spec_type == COMMON_SPECULATIVE_TYPE_DFLASH || spec_type == COMMON_SPECULATIVE_TYPE_DSPARK || spec_type == COMMON_SPECULATIVE_TYPE_MTP) && (int32_t)val == 0)
                ? configured_n_max
                : (int32_t)val;
        }
        else if (coord.name == "p_min")                params.p_min                = val;
        else if (coord.name == "n_min")                params.n_min                = (int32_t)val;
        else if (coord.name == "ngram_size_n")         params.ngram_size_n         = (uint16_t)val;
        else if (coord.name == "ngram_size_m")         params.ngram_size_m         = (uint16_t)val;
        else if (coord.name == "ngram_min_hits")       params.ngram_min_hits       = (uint16_t)val;
        else if (coord.name == "suffix_min_match_len") params.suffix_min_match_len = (int32_t)val;
        else if (coord.name == "verify_pinned")        params.spec_verify_pinned = ((int32_t)val != 0);
    }
}

void spec_tuner::init(common_speculative_type type, const common_params_speculative & user_params, const llama_model * model_tgt) {
    enabled    = true;
    spec_type  = type;
    configured_n_max = std::max(1, (int) user_params.n_max);
    counterfactual = spec_counterfactual_engine(configured_n_max, 0.1, 0.15, 1.0);
    systems_tuner.init_defaults(user_params.n_threads > 0 ? user_params.n_threads : 16, 8);
    // ConcurrencyKnee threshold is operator-tunable (--spec-c-knee); a value
    // of 0 disables the systems tuner entirely (no knee, no Thompson coords).
    if (user_params.c_knee > 0.0f) {
        systems_tuner.c_knee = (double) user_params.c_knee;
        systems_tuner.enabled = true;
    } else {
        systems_tuner.c_knee = 1e30; // never saturates
        systems_tuner.enabled = false;
    }
    dflash_quarantined.clear();
    dflash_probe_cursor = 0;
    dflash_last_exploratory = false;
    dflash_last_recovery_probe = false;
    n_target_only_selections = 0;
    n_dflash_selections = 0;
    n_exploratory_selections = 0;
    n_quarantines = 0;
    n_recovery_probes = 0;
    n_counterfactual_selections = 0;
    coords.clear();
    n_calls    = 0;
    n_requests = 0;
    ema_tps    = 0.0;
    step_ema   = 0.0;
    n_low      = 0;
    cooldown   = 0;
    n_resets   = 0;
    t_tuner_us = 0;
    last_n_drafted = 0;
    ema_accept_rate = 0.0;
    ema_token_entropy = 0.0;
    entropy_gated = false;

    // all types get n_max
    // For simplicity we will create a fixed grid of possible values
    {
        spec_tuner_coord coord;
        coord.name = "n_max";
        const bool recurrent_target = model_tgt != nullptr && llama_model_has_recurrent(model_tgt);
        int hi = (type == COMMON_SPECULATIVE_TYPE_DFLASH || type == COMMON_SPECULATIVE_TYPE_DSPARK || type == COMMON_SPECULATIVE_TYPE_MTP)
            ? configured_n_max
            : (recurrent_target ? std::max(1, (int) user_params.n_max)
                                 : std::max(16, (int) user_params.n_max));
        const int lo = (type == COMMON_SPECULATIVE_TYPE_DFLASH || type == COMMON_SPECULATIVE_TYPE_DSPARK || type == COMMON_SPECULATIVE_TYPE_MTP) ? 0 : 1;
        coord.build_grid_int(lo, hi, 1, user_params.n_max);
        coords.push_back(std::move(coord));
    }

    if (type == COMMON_SPECULATIVE_TYPE_DFLASH || type == COMMON_SPECULATIVE_TYPE_DSPARK || type == COMMON_SPECULATIVE_TYPE_MTP) {
        dflash_quarantined.assign(coords[0].arms.size(), false);
    }

    // PolyGraph verify-pinning arm: 0 = dynamic verify batch shape,
    // 1 = pinned (stage n_max + 1 rows). The initial arm follows the user's
    // --spec-verify-pinned; with --spec-autotune the tuner explores both arms
    // and learns the per-workload winner from measured step TPS.
    {
        spec_tuner_coord coord;
        coord.name = "verify_pinned";
        coord.build_grid_int(0, 1, 1, user_params.spec_verify_pinned ? 1 : 0);
        coords.push_back(std::move(coord));
    }

    if (type == COMMON_SPECULATIVE_TYPE_DRAFT) {
        {
            spec_tuner_coord coord;
            coord.name = "p_min";
            coord.build_grid_float(0.0f, 0.95f, 11, user_params.p_min);
            coords.push_back(std::move(coord));
        }
        {
            spec_tuner_coord coord;
            coord.name = "n_min";
            coord.build_grid_int(0, 6, 1, user_params.n_min);
            coords.push_back(std::move(coord));
        }
    }

    if (type == COMMON_SPECULATIVE_TYPE_SUFFIX) {
        {
            spec_tuner_coord coord;
            coord.name = "p_min";
            coord.build_grid_float(0.0f, 0.95f, 11, user_params.p_min);
            coords.push_back(std::move(coord));
        }
        {
            spec_tuner_coord coord;
            coord.name = "suffix_min_match_len";
            coord.build_grid_int(1, 12, 1, user_params.suffix_min_match_len);
            coords.push_back(std::move(coord));
        }
    }

    // Ngram can change only n_max/n_min per call
    if (type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD) {
        {
            spec_tuner_coord coord;
            coord.name = "n_min";
            int hi = std::max(0, std::min(4, (int)user_params.n_max - 1));
            coord.build_grid_int(0, hi, 1, user_params.n_min);
            coords.push_back(std::move(coord));
        }
    }

    for (auto & coord : coords) {
        float user_val = 0.0f;
        if      (coord.name == "n_max")                user_val = (float)user_params.n_max;
        else if (coord.name == "p_min")                user_val = user_params.p_min;
        else if (coord.name == "n_min")                user_val = (float)user_params.n_min;
        else if (coord.name == "ngram_size_n")         user_val = (float)user_params.ngram_size_n;
        else if (coord.name == "ngram_size_m")         user_val = (float)user_params.ngram_size_m;
        else if (coord.name == "ngram_min_hits")       user_val = (float)user_params.ngram_min_hits;
        else if (coord.name == "suffix_min_match_len") user_val = (float)user_params.suffix_min_match_len;

        coord.user_idx    = coord.find_nearest_arm(user_val);
        coord.best_idx    = 0;
        coord.current_idx = 0;
    }

    LOG_DBG("Autotune ε-greedy (ε=%.2f) per-draft-call, reward=per-step TPS\n", epsilon);
    for (const auto & coord : coords) {
        std::ostringstream oss;
        oss << "  " << coord.name << ": [";
        for (size_t i = 0; i < coord.arms.size(); i++) {
            if (i > 0) oss << ", ";
            oss << coord.arms[i].value;
        }
        oss << "] (user=" << coord.arms[coord.user_idx].value << ")";
        LOG_DBG("%s\n", oss.str().c_str());
    }
}

int spec_tuner::select_dflash_arm(spec_tuner_coord & coord) {
    dflash_last_exploratory = false;
    dflash_last_recovery_probe = false;

    if (dflash_quarantined.size() != coord.arms.size()) {
        dflash_quarantined.assign(coord.arms.size(), false);
    }

    // Exploration sweep: visit every undersampled arm, but start with the
    // positive draft arms and leave target-only (value 0) for last. A cold
    // start that proposes n_max 0 would disable drafting before any feedback
    // cycle exists, and without drafting no feedback can ever arrive -- the
    // tuner would pin itself to the zero arm forever. Sweeping non-zero arms
    // first guarantees the first proposal drafts (and therefore learns).
    int zero_arm = -1;
    for (int i = 0; i < (int) coord.arms.size(); ++i) {
        if (coord.arms[i].N < dflash_min_samples_per_arm) {
            if (coord.arms[i].value != 0) {
                dflash_last_exploratory = true;
                return i;
            }
            zero_arm = i;
        }
    }
    if (zero_arm >= 0) {
        dflash_last_exploratory = true;
        return zero_arm;
    }

    if (dflash_recovery_probe_interval > 0 && n_calls > 0 &&
            n_calls % dflash_recovery_probe_interval == 0) {
        for (int offset = 0; offset < (int) coord.arms.size(); ++offset) {
            const int i = (dflash_probe_cursor + offset) % (int) coord.arms.size();
            if (dflash_quarantined[i]) {
                dflash_probe_cursor = (i + 1) % (int) coord.arms.size();
                dflash_last_exploratory = true;
                dflash_last_recovery_probe = true;
                n_recovery_probes++;
                return i;
            }
        }
    }

    static thread_local std::mt19937 rng(std::random_device{}());

    // Not-a-Bandit Theorem 3: once the full-information engine is warm, its Hedge
    // distribution over candidate draft lengths drives selection with zero
    // exploration waste; the quarantine mask remains as a safety overlay.
    if ((int) counterfactual.total_pulls() >= counterfactual_warmup_steps) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        const int k = counterfactual.select_draft_length(false, uni(rng));
        const int idx = coord.find_nearest_arm((float) k);
        if (idx >= 0 && idx < (int) coord.arms.size() && !dflash_quarantined[idx]) {
            n_counterfactual_selections++;
            coord.current_idx = idx;
            coord.best_idx = idx;
            return idx;
        }
        // Hedge draw landed on a quarantined arm: fall through to the deterministic
        // best-mean exploit below (quarantine wins over the distribution).
    }

    int best_mean_idx = -1;
    int ts_best_idx   = -1;
    double best_sample = -1e30;

    for (int i = 0; i < (int) coord.arms.size(); ++i) {
        if (dflash_quarantined[i]) {
            continue;
        }
        if (best_mean_idx < 0 || coord.arms[i].Q > coord.arms[best_mean_idx].Q) {
            best_mean_idx = i;
        }
        const auto & arm = coord.arms[i];
        double sample = arm.Q;
        if (arm.N >= 2) {
            const double var = std::max(1e-4, arm.M2 / (double)(arm.N - 1));
            const double sigma = std::sqrt(var / (double)arm.N);
            std::normal_distribution<double> dist(arm.Q, sigma);
            sample = dist(rng);
        }
        if (ts_best_idx < 0 || sample > best_sample) {
            best_sample = sample;
            ts_best_idx = i;
        }
    }
    if (best_mean_idx < 0) {
        best_mean_idx = 0;
    }
    if (ts_best_idx < 0) {
        ts_best_idx = best_mean_idx;
    }
    coord.best_idx = best_mean_idx;
    if (ts_best_idx != best_mean_idx) {
        dflash_last_exploratory = true;
    }
    return ts_best_idx;
}

void spec_tuner::update_dflash_quarantine() {
    if (!has_dflash_target_only_arm() || coords.empty() || coords[0].arms.empty()) {
        return;
    }

    auto & coord = coords[0];
    if (dflash_quarantined.size() != coord.arms.size()) {
        dflash_quarantined.assign(coord.arms.size(), false);
    }

    const int zero_idx = coord.find_nearest_arm(0.0f);
    const auto & zero = coord.arms[zero_idx];
    if (zero.N < 3 || zero.Q <= 0.0) {
        return;
    }

    for (int i = 0; i < (int) coord.arms.size(); ++i) {
        if (i == zero_idx) {
            continue;
        }
        const bool should_quarantine = coord.arms[i].N >= 3 &&
            coord.arms[i].Q < zero.Q * dflash_quarantine_ratio;
        if (should_quarantine && !dflash_quarantined[i]) {
            n_quarantines++;
        }
        dflash_quarantined[i] = should_quarantine;
    }

    int best_idx = -1;
    for (int i = 0; i < (int) coord.arms.size(); ++i) {
        if (dflash_quarantined[i]) {
            continue;
        }
        if (best_idx < 0 || coord.arms[i].Q > coord.arms[best_idx].Q) {
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        coord.best_idx = best_idx;
    }
}

void spec_tuner::propose(common_params_speculative & params) {
    int64_t t_start = ggml_time_us();

    // always select fresh arm for every draft call
    for (auto & coord : coords) {
        if (has_dflash_target_only_arm() && coord.name == "n_max") {
            coord.current_idx = select_dflash_arm(coord);
            if ((int) coord.arms[coord.current_idx].value == 0) {
                n_target_only_selections++;
            } else {
                n_dflash_selections++;
            }
            if (dflash_last_exploratory) {
                n_exploratory_selections++;
            }
        } else {
            bool all_have_variance = !coord.arms.empty();
            for (const auto & arm : coord.arms) {
                if (arm.N < 2) {
                    all_have_variance = false;
                    break;
                }
            }
            coord.current_idx = all_have_variance
                ? coord.select_thompson_sampling()
                : coord.select_epsilon_greedy(epsilon);
        }

        float val = coord.arms[coord.current_idx].value;
        if      (coord.name == "n_max")                params.n_max                = (int32_t)val;
        else if (coord.name == "p_min")                params.p_min                = val;
        else if (coord.name == "n_min")                params.n_min                = (int32_t)val;
        else if (coord.name == "ngram_size_n")         params.ngram_size_n         = (uint16_t)val;
        else if (coord.name == "ngram_size_m")         params.ngram_size_m         = (uint16_t)val;
        else if (coord.name == "ngram_min_hits")       params.ngram_min_hits       = (uint16_t)val;
        else if (coord.name == "suffix_min_match_len") params.suffix_min_match_len = (int32_t)val;
        else if (coord.name == "verify_pinned")        params.spec_verify_pinned = ((int32_t)val != 0);
    }

    // Unified online entropy & acceptance-rate gate across DFlash, DSpark, MTP, and CCD suffix-trie
    if (entropy_gated && (
            spec_type == COMMON_SPECULATIVE_TYPE_DFLASH ||
            spec_type == COMMON_SPECULATIVE_TYPE_DSPARK ||
            spec_type == COMMON_SPECULATIVE_TYPE_MTP    ||
            spec_type == COMMON_SPECULATIVE_TYPE_SUFFIX)) {
        params.n_max = has_dflash_target_only_arm() ? 0 : std::min((int32_t)params.n_max, (int32_t)2);
    }

    // ConcurrencyKnee throttling: when host memory bus or queue depth is saturated,
    // throttle speculative draft length (n_max -> 0) to protect core target processing.
    if (systems_tuner.enabled && systems_tuner.knee_saturated) {
        params.n_max = has_dflash_target_only_arm() ? 0 : std::min((int32_t)params.n_max, (int32_t)1);
    }

    enforce_constraints(params);
    t_tuner_us += (ggml_time_us() - t_start);
}

void spec_tuner::enforce_constraints(common_params_speculative & params) {
    if (params.n_min < 0)    params.n_min = 0;
    const int min_n_max = has_dflash_target_only_arm() ? 0 : 1;
    if (params.n_max < min_n_max) params.n_max = min_n_max;
    if (params.n_min > params.n_max) params.n_min = params.n_max;

    if (params.p_min < 0.0f)  params.p_min = 0.0f;
    if (params.p_min > 0.95f) params.p_min = 0.95f;

    if (params.ngram_size_n < 1)   params.ngram_size_n = 1;
    if (params.ngram_size_m < 1)   params.ngram_size_m = 1;
    if (params.ngram_min_hits < 1) params.ngram_min_hits = 1;
}

void spec_tuner::update_entropy_gate(float token_entropy) {
    if (token_entropy < 0.0f) {
        return;
    }
    if (ema_token_entropy <= 0.0) {
        ema_token_entropy = (double)token_entropy;
    } else {
        ema_token_entropy = 0.2 * (double)token_entropy + 0.8 * ema_token_entropy;
    }
    entropy_gated = (ema_token_entropy > (double)entropy_gate_threshold) &&
                    (n_calls < 4 || ema_accept_rate < 0.25);
}

void spec_tuner::accept_feedback(int n_accepted, int n_drafted, double step_tps) {
    int64_t t_start = ggml_time_us();
    n_calls++;

    if (n_drafted > 0) {
        const double step_accept = (double)std::max(0, n_accepted) / (double)n_drafted;
        if (n_calls <= 1 && ema_accept_rate <= 0.0) {
            ema_accept_rate = step_accept;
        } else {
            ema_accept_rate = 0.15 * step_accept + 0.85 * ema_accept_rate;
        }
    }

    // per-step TPS as reward: captures draft cost, verification cost, and acceptance benefit
    double reward = step_tps;

    for (auto & coord : coords) {
        coord.update(reward);
    }

    // Update full-information counterfactual prefix attribution engine (Theorem 3)
    counterfactual.update_full_information(n_drafted, n_accepted);

    // Update multi-coordinate discounted Thompson systems tuner
    if (systems_tuner.enabled) {
        systems_tuner.update_feedback(reward);
    }

    update_dflash_quarantine();

    if (cooldown > 0) {
        cooldown--;
        if (step_ema <= 0.0) {
            step_ema = step_tps;
        } else {
            step_ema = step_ema_alpha * step_tps + (1.0 - step_ema_alpha) * step_ema;
        }
    } else if (step_ema <= 0.0) {
        step_ema = step_tps;
    } else {
        if (step_tps < step_ema * (1.0 - step_drop_pct)) {
            n_low++;
            if (n_low >= reset_after) {
                reset_exploration();
                t_tuner_us += (ggml_time_us() - t_start);
                return;
            }
        } else {
            n_low = 0;
        }
        step_ema = step_ema_alpha * step_tps + (1.0 - step_ema_alpha) * step_ema;
    }

    if (n_calls <= 5 || (n_calls % log_every == 0)) {
        std::ostringstream oss;
        oss << "Autotune call=" << n_calls
            << " n_drafted=" << n_drafted
            << " n_accepted=" << n_accepted
            << " step_tps=" << std::fixed << std::setprecision(1) << step_tps
            << " ema=" << std::fixed << std::setprecision(1) << step_ema;
        if (has_dflash_target_only_arm()) {
            oss << " configured_n_max=" << configured_n_max;
        }
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            oss << " " << coord.name << "=";
            if (is_int) oss << (int)coord.arms[coord.current_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.current_idx].value;
            if (coord.name == "n_max" && (spec_type == COMMON_SPECULATIVE_TYPE_DFLASH || spec_type == COMMON_SPECULATIVE_TYPE_DSPARK || spec_type == COMMON_SPECULATIVE_TYPE_MTP)) {
                oss << "(target_only=" << ((int) coord.arms[coord.current_idx].value == 0 ? "true" : "false") << ")";
            }
            oss << "→best=";
            if (is_int) oss << (int)coord.arms[coord.best_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value;
            oss << "(Q=" << std::fixed << std::setprecision(1) << coord.arms[coord.best_idx].Q
                << ",N=" << coord.arms[coord.best_idx].N << ")";
        }
        LOG_DBG("%s\n", oss.str().c_str());
    }

    t_tuner_us += (ggml_time_us() - t_start);
}

void spec_tuner::observe_concurrency(double active_streams_or_queue_depth) {
    if (!enabled) {
        return;
    }
    // ConcurrencyKnee (investigation-garden spec 5.3): record the saturation
    // sample; propose() consults knee_saturated to throttle n_max.
    int64_t t_start = ggml_time_us();
    systems_tuner.check_concurrency_knee(active_streams_or_queue_depth);
    t_tuner_us += (ggml_time_us() - t_start);
}

void spec_tuner::end_of_request(double slot_tps, int n_past, common_params_speculative & active_params) {
    int64_t t_start = ggml_time_us();
    n_requests++;

    GGML_UNUSED(n_past);

    if (ema_tps <= 0.0) {
        ema_tps = slot_tps;
    } else {
        ema_tps = ema_alpha * slot_tps + (1.0 - ema_alpha) * ema_tps;
    }

    write_best(active_params);
    enforce_constraints(active_params);

    t_tuner_us += (ggml_time_us() - t_start);
    print_best();
}

void spec_tuner::print_best() const {
    {
        std::ostringstream oss;
        oss << "Autotune req=" << n_requests
            << " calls=" << n_calls
            << " tps=" << std::fixed << std::setprecision(2) << ema_tps;

        if (n_resets > 0) oss << " resets=" << n_resets;
        if (n_low > 0)    oss << " n_low=" << n_low;
        if (has_dflash_target_only_arm()) {
            oss << " configured_n_max=" << configured_n_max
                << " target_only=" << n_target_only_selections
                << " dflash=" << n_dflash_selections
                << " exploratory=" << n_exploratory_selections
                << " quarantines=" << n_quarantines
                << " probes=" << n_recovery_probes;
        }

        oss << " best:";
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            oss << " " << coord.name << "=";
            if (is_int) oss << (int) coord.arms[coord.best_idx].value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value;
            oss << "(Q=" << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].Q
                << ",N=" << coord.arms[coord.best_idx].N << ")";
        }

        if (!coords.empty()) {
            oss << " | n_max arms:";
            for (size_t arm_idx = 0; arm_idx < coords[0].arms.size(); ++arm_idx) {
                const auto & arm = coords[0].arms[arm_idx];
                const bool quarantined = has_dflash_target_only_arm() && arm_idx < dflash_quarantined.size()
                    ? dflash_quarantined[arm_idx]
                    : false;
                oss << " " << (int)arm.value << "(Q=" << std::fixed << std::setprecision(2) << arm.Q
                    << ",N=" << arm.N << ",quarantined=" << (quarantined ? "true" : "false") << ")";
            }
        }

        oss << " tuner=" << std::fixed << std::setprecision(3) << t_tuner_us / 1000.0 << "ms";
        if (has_dflash_target_only_arm()) {
            LOG_INF("%s\n", oss.str().c_str());
        } else {
            LOG_DBG("%s\n", oss.str().c_str());
        }
    }

    {
        std::ostringstream oss;
        oss << "Autotune reuse: --spec-type " << common_speculative_type_to_str(spec_type);
        bool first_kv = true;
        for (const auto & coord : coords) {
            bool is_int = (coord.name != "p_min");
            oss << (first_kv ? ':' : ',') << coord.name << '=';
            first_kv = false;

            int reuse_value = is_int ? (int) coord.arms[coord.best_idx].value : 0;
            if (coord.name == "n_max" && (spec_type == COMMON_SPECULATIVE_TYPE_DFLASH || spec_type == COMMON_SPECULATIVE_TYPE_DSPARK || spec_type == COMMON_SPECULATIVE_TYPE_MTP) && reuse_value == 0) {
                reuse_value = configured_n_max;
            }
            if (is_int) oss << reuse_value;
            else oss << std::fixed << std::setprecision(2) << coord.arms[coord.best_idx].value;
        }
        LOG_INF("%s\n", oss.str().c_str());
    }
}
