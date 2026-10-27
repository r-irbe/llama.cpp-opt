#include "spec-arbiter.h"

#include <algorithm>

spec_fused_candidates spec_arbiter::fuse_treegraft(
        const std::vector<llama_token> & ngram_branch,
        const std::vector<llama_token> & neural_branch,
        int max_total_tokens) {
    spec_fused_candidates fused;

    if (ngram_branch.empty() && neural_branch.empty()) {
        fused.primary_source = SPEC_DRAFTER_SOURCE_TARGET_ONLY;
        return fused;
    }

    if (ngram_branch.empty()) {
        fused.tokens = neural_branch;
        if ((int)fused.tokens.size() > max_total_tokens) {
            fused.tokens.resize(max_total_tokens);
        }
        fused.primary_source = SPEC_DRAFTER_SOURCE_NEURAL;
        fused.is_tree = false;
        for (size_t i = 0; i < fused.tokens.size(); ++i) {
            fused.tree_nodes.push_back({
                fused.tokens[i],
                i == 0 ? -1 : (int32_t)(i - 1),
                (int32_t)i,
                1.0f,
                SPEC_DRAFTER_SOURCE_NEURAL
            });
        }
        return fused;
    }

    if (neural_branch.empty()) {
        fused.tokens = ngram_branch;
        if ((int)fused.tokens.size() > max_total_tokens) {
            fused.tokens.resize(max_total_tokens);
        }
        fused.primary_source = SPEC_DRAFTER_SOURCE_NGRAM;
        fused.is_tree = false;
        for (size_t i = 0; i < fused.tokens.size(); ++i) {
            fused.tree_nodes.push_back({
                fused.tokens[i],
                i == 0 ? -1 : (int32_t)(i - 1),
                (int32_t)i,
                1.0f,
                SPEC_DRAFTER_SOURCE_NGRAM
            });
        }
        return fused;
    }

    // Both branches have candidates: find shared prefix
    fused.primary_source = SPEC_DRAFTER_SOURCE_TREEGRAFT;
    fused.is_tree = true;

    size_t common_len = 0;
    size_t min_len = std::min(ngram_branch.size(), neural_branch.size());
    while (common_len < min_len && ngram_branch[common_len] == neural_branch[common_len]) {
        common_len++;
    }

    // 1. Shared trunk
    for (size_t i = 0; i < common_len; ++i) {
        if ((int)fused.tokens.size() >= max_total_tokens) break;
        fused.tokens.push_back(ngram_branch[i]);
        fused.tree_nodes.push_back({
            ngram_branch[i],
            i == 0 ? -1 : (int32_t)(i - 1),
            (int32_t)i,
            1.0f,
            SPEC_DRAFTER_SOURCE_TREEGRAFT
        });
    }

    int32_t trunk_parent = common_len > 0 ? (int32_t)(common_len - 1) : -1;

    // 2. Neural branch divergence
    int32_t prev_parent = trunk_parent;
    for (size_t i = common_len; i < neural_branch.size(); ++i) {
        if ((int)fused.tokens.size() >= max_total_tokens) break;
        int32_t curr_idx = (int32_t)fused.tokens.size();
        fused.tokens.push_back(neural_branch[i]);
        fused.tree_nodes.push_back({
            neural_branch[i],
            prev_parent,
            (int32_t)i,
            0.9f,
            SPEC_DRAFTER_SOURCE_NEURAL
        });
        prev_parent = curr_idx;
    }

    // 3. Ngram branch divergence
    prev_parent = trunk_parent;
    for (size_t i = common_len; i < ngram_branch.size(); ++i) {
        if ((int)fused.tokens.size() >= max_total_tokens) break;
        int32_t curr_idx = (int32_t)fused.tokens.size();
        fused.tokens.push_back(ngram_branch[i]);
        fused.tree_nodes.push_back({
            ngram_branch[i],
            prev_parent,
            (int32_t)i,
            0.8f,
            SPEC_DRAFTER_SOURCE_NGRAM
        });
        prev_parent = curr_idx;
    }

    return fused;
}

bool spec_arbiter::poll_and_fuse(spec_fused_candidates & out_fused, int max_tokens) {
    drafter_packet pkt_ngram = {};
    drafter_packet pkt_neural = {};
    bool has_ngram = ring_ngram.pop(pkt_ngram);
    bool has_neural = ring_neural.try_consume(pkt_neural);

    if (!has_ngram && !has_neural) {
        return false;
    }

    std::vector<llama_token> ngram_vec;
    if (has_ngram) {
        ngram_vec.assign(pkt_ngram.tokens, pkt_ngram.tokens + pkt_ngram.n_tokens);
    }

    std::vector<llama_token> neural_vec;
    if (has_neural) {
        neural_vec.assign(pkt_neural.tokens, pkt_neural.tokens + pkt_neural.n_tokens);
    }

    out_fused = fuse_treegraft(ngram_vec, neural_vec, max_tokens);
    return true;
}
