#pragma once

//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//

#include "llama.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <functional>

/**
 * Prefix Radix Trie for O(L) Cohort Prompt Prefix Lookup & Dynamic Eviction
 *
 * Indexes active token sequences across server slots.
 * Supports O(L) prefix matching, where L is the token length of the incoming query.
 * Distinguishes cohort-private prefixes from public/untagged prefixes to preserve
 * multi-tenant security boundaries.
 */

struct cohort_trie_match {
    int slot_id = -1;
    int matched_tokens = 0;
    int cohort_reuse_count = 0;
    int64_t t_last_used = 0;
};

class cohort_radix_trie {
public:
    struct trie_node {
        std::unordered_map<llama_token, std::unique_ptr<trie_node>> children;
        std::vector<int> slot_ids; // slots that share prefix up to this node
    };

private:
    std::unique_ptr<trie_node> root;

public:
    cohort_radix_trie() : root(std::make_unique<trie_node>()) {}

    void clear() {
        root = std::make_unique<trie_node>();
    }

    void insert_slot(int slot_id, const std::vector<llama_token>& tokens) {
        if (tokens.empty()) return;
        trie_node* cur = root.get();
        for (llama_token tok : tokens) {
            auto it = cur->children.find(tok);
            if (it == cur->children.end()) {
                auto new_node = std::make_unique<trie_node>();
                it = cur->children.emplace(tok, std::move(new_node)).first;
            }
            cur = it->second.get();
            if (std::find(cur->slot_ids.begin(), cur->slot_ids.end(), slot_id) == cur->slot_ids.end()) {
                cur->slot_ids.push_back(slot_id);
            }
        }
    }

    void remove_slot(int slot_id) {
        remove_slot_recursive(root.get(), slot_id);
    }

    // Searches the trie for the slot with the longest matching prefix for query_tokens
    // that is authorized for task_cohort (either matching cohort or public)
    template <typename SlotAccessor>
    cohort_trie_match find_best_match(
        const std::vector<llama_token>& query_tokens,
        const std::string& task_cohort,
        int min_prefix_tokens,
        SlotAccessor get_slot
    ) const {
        cohort_trie_match best_match;
        if (query_tokens.empty()) return best_match;

        const trie_node* cur = root.get();
        int matched = 0;

        for (llama_token tok : query_tokens) {
            auto it = cur->children.find(tok);
            if (it == cur->children.end()) {
                break;
            }
            cur = it->second.get();
            matched++;

            if (matched >= min_prefix_tokens) {
                for (int slot_id : cur->slot_ids) {
                    const auto* slot = get_slot(slot_id);
                    if (!slot || !slot->available()) continue;

                    // Security check: Private cohort slots can only match same cohort
                    if (!slot->cohort_id.empty() && slot->cohort_id != task_cohort) {
                        continue;
                    }

                    // Score this candidate: prefer longest matched prefix, then highest reuse count, then most recently used
                    bool is_better = false;
                    if (matched > best_match.matched_tokens) {
                        is_better = true;
                    } else if (matched == best_match.matched_tokens) {
                        if (slot->cohort_reuse_count > best_match.cohort_reuse_count) {
                            is_better = true;
                        } else if (slot->cohort_reuse_count == best_match.cohort_reuse_count &&
                                   slot->t_last_used > best_match.t_last_used) {
                            is_better = true;
                        }
                    }

                    if (is_better) {
                        best_match.slot_id = slot_id;
                        best_match.matched_tokens = matched;
                        best_match.cohort_reuse_count = slot->cohort_reuse_count;
                        best_match.t_last_used = slot->t_last_used;
                    }
                }
            }
        }

        return best_match;
    }

    // Searches the trie for ANY slot (including busy processing slots) holding the longest
    // matching prefix in its KV cache (slot->n_past >= matched) that is authorized for
    // task_cohort, enabling cross-slot KV cloning via llama_kv_cache_seq_cp.
    template <typename SlotAccessor>
    cohort_trie_match find_best_donor_match(
        const std::vector<llama_token>& query_tokens,
        const std::string& task_cohort,
        int min_prefix_tokens,
        SlotAccessor get_slot,
        int exclude_slot_id = -1
    ) const {
        cohort_trie_match best_match;
        if (query_tokens.empty()) return best_match;

        const trie_node* cur = root.get();
        int matched = 0;

        for (llama_token tok : query_tokens) {
            auto it = cur->children.find(tok);
            if (it == cur->children.end()) {
                break;
            }
            cur = it->second.get();
            matched++;

            if (matched >= min_prefix_tokens) {
                for (int slot_id : cur->slot_ids) {
                    if (slot_id == exclude_slot_id) continue;
                    const auto* slot = get_slot(slot_id);
                    if (!slot) continue;

                    // Donor slot must have at least `matched` tokens materialized in its KV cache
                    if (slot->n_past < matched) {
                        continue;
                    }

                    // Strict multi-tenant cohort isolation
                    if (slot->cohort_id != task_cohort) {
                        continue;
                    }

                    bool is_better = false;
                    if (matched > best_match.matched_tokens) {
                        is_better = true;
                    } else if (matched == best_match.matched_tokens) {
                        if (slot->cohort_reuse_count > best_match.cohort_reuse_count) {
                            is_better = true;
                        } else if (slot->cohort_reuse_count == best_match.cohort_reuse_count &&
                                   slot->t_last_used > best_match.t_last_used) {
                            is_better = true;
                        }
                    }

                    if (is_better) {
                        best_match.slot_id = slot_id;
                        best_match.matched_tokens = matched;
                        best_match.cohort_reuse_count = slot->cohort_reuse_count;
                        best_match.t_last_used = slot->t_last_used;
                    }
                }
            }
        }

        return best_match;
    }

private:
    void remove_slot_recursive(trie_node* node, int slot_id) {
        if (!node) return;
        auto it = std::remove(node->slot_ids.begin(), node->slot_ids.end(), slot_id);
        node->slot_ids.erase(it, node->slot_ids.end());

        for (auto& child : node->children) {
            remove_slot_recursive(child.second.get(), slot_id);
        }
    }
};
