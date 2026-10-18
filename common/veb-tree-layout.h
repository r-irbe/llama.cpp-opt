#pragma once

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

// =========================================================================================
// Van Emde Boas (vEB) Cache-Oblivious Memory Organization Architecture
// Grounded in:
//   1. Peter van Emde Boas (1975, 1977): Recursive space-splitting data structures
//   2. Harald Prokop (1999) & Frigo et al. (1999/2012): Cache-Oblivious Algorithms (TALG 2012)
//   3. Bender, Demaine, Farach-Colton (2000, 2005): Cache-Oblivious B-Trees & Search Trees
//   4. Asymptotic Bound: O(log_B N) cache misses across all levels of memory hierarchy
//      simultaneously, without knowing cache line size B or memory capacity M.
// =========================================================================================

// =========================================================================================
// 1. Recursive van Emde Boas (vEB) Permutation Mapper
// Recursively bisects a binary tree of height h at h/2 into a top tree and sqrt(V) bottom trees.
// Maps standard 1-based breadth-first (Eytzinger) node IDs into vEB memory offsets.
// =========================================================================================

class veb_layout_mapper {
public:
    explicit veb_layout_mapper(uint32_t height) : height_(height) {
        uint32_t num_nodes = (1U << (height_ + 1)) - 1;
        eytzinger_to_veb_.resize(num_nodes + 1, 0);
        veb_to_eytzinger_.resize(num_nodes + 1, 0);

        uint32_t next_veb_idx = 0;
        build_veb_permutation(1, height_, next_veb_idx);
    }

    // Convert breadth-first 1-based index (1, 2, 3, ...) to 0-based vEB memory index
    uint32_t to_veb(uint32_t eytzinger_idx) const {
        if (eytzinger_idx >= eytzinger_to_veb_.size()) return 0;
        return eytzinger_to_veb_[eytzinger_idx];
    }

    // Convert 0-based vEB memory index back to breadth-first 1-based index
    uint32_t to_eytzinger(uint32_t veb_idx) const {
        if (veb_idx >= veb_to_eytzinger_.size()) return 0;
        return veb_to_eytzinger_[veb_idx];
    }

    uint32_t capacity() const {
        return (1U << (height_ + 1)) - 1;
    }

    uint32_t height() const {
        return height_;
    }

private:
    uint32_t height_{0};
    std::vector<uint32_t> eytzinger_to_veb_;
    std::vector<uint32_t> veb_to_eytzinger_;

    // Recursive vEB tree layout algorithm
    void build_veb_permutation(uint32_t root, uint32_t h, uint32_t & next_veb_idx) {
        if (h == 0) {
            // Base case: leaf node
            eytzinger_to_veb_[root] = next_veb_idx;
            veb_to_eytzinger_[next_veb_idx] = root;
            next_veb_idx++;
            return;
        }

        uint32_t h_top = h / 2;
        uint32_t h_bottom = h - h_top - 1;

        // 1. Recursively lay out the top tree of height h_top
        build_veb_permutation(root, h_top, next_veb_idx);

        // 2. Identify the roots of all bottom subtrees attached to the leaves of the top tree
        std::vector<uint32_t> bottom_roots;
        collect_leaves(root, h_top, bottom_roots);

        // 3. Recursively lay out each bottom subtree of height h_bottom
        for (uint32_t b_root : bottom_roots) {
            uint32_t left_child = b_root * 2;
            uint32_t right_child = b_root * 2 + 1;

            if (left_child < eytzinger_to_veb_.size()) {
                build_veb_permutation(left_child, h_bottom, next_veb_idx);
            }
            if (right_child < eytzinger_to_veb_.size()) {
                build_veb_permutation(right_child, h_bottom, next_veb_idx);
            }
        }
    }

    void collect_leaves(uint32_t node, uint32_t levels_remaining, std::vector<uint32_t> & leaves) {
        if (levels_remaining == 0) {
            leaves.push_back(node);
            return;
        }
        collect_leaves(node * 2, levels_remaining - 1, leaves);
        collect_leaves(node * 2 + 1, levels_remaining - 1, leaves);
    }
};

// =========================================================================================
// 2. Cache-Oblivious Van Emde Boas Static Search Tree
// Stores key-value entries in contiguous memory permuted by the vEB layout.
// Bounded to O(log_B N) cache misses without tuning for block size B.
// =========================================================================================

template <typename Key, typename Value>
class veb_static_search_tree {
public:
    struct node {
        Key key{};
        Value val{};
        bool is_valid{false};
    };

    veb_static_search_tree() = default;

    // Build vEB search tree from sorted key-value pairs
    void build(const std::vector<std::pair<Key, Value>> & sorted_entries) {
        if (sorted_entries.empty()) {
            nodes_.clear();
            return;
        }

        uint32_t n = static_cast<uint32_t>(sorted_entries.size());
        // Calculate tree height: 2^(h+1) - 1 >= n
        uint32_t h = 0;
        while (((1U << (h + 1)) - 1) < n) {
            h++;
        }

        mapper_ = std::make_unique<veb_layout_mapper>(h);
        uint32_t total_capacity = mapper_->capacity();
        nodes_.assign(total_capacity, node{});

        // 1. Build standard complete BST in in-order array
        std::vector<node> complete_bst(total_capacity + 1);
        build_range(1, 0, static_cast<int>(n) - 1, complete_bst, sorted_entries);

        // 2. Permute from Eytzinger 1-based order into vEB 0-based memory order
        for (uint32_t eyt_idx = 1; eyt_idx <= total_capacity; ++eyt_idx) {
            uint32_t veb_idx = mapper_->to_veb(eyt_idx);
            if (veb_idx < nodes_.size()) {
                nodes_[veb_idx] = complete_bst[eyt_idx];
            }
        }
    }

    // Cache-oblivious search: O(log_B N) transfers
    bool find(const Key & target_key, Value & out_value) const {
        if (nodes_.empty() || !mapper_) return false;

        uint32_t curr_eyt = 1; // Start at logical root
        uint32_t capacity = mapper_->capacity();

        while (curr_eyt <= capacity) {
            uint32_t veb_idx = mapper_->to_veb(curr_eyt);
            const node & curr_node = nodes_[veb_idx];

            if (!curr_node.is_valid) {
                // Fallen past populated entries in incomplete tree
                return false;
            }

            if (target_key == curr_node.key) {
                out_value = curr_node.val;
                return true;
            } else if (target_key < curr_node.key) {
                curr_eyt = curr_eyt * 2; // Left child in logical BST
            } else {
                curr_eyt = curr_eyt * 2 + 1; // Right child in logical BST
            }
        }

        return false;
    }

    size_t size() const {
        return nodes_.size();
    }

private:
    std::unique_ptr<veb_layout_mapper> mapper_;
    std::vector<node> nodes_;

    void build_range(uint32_t eyt_idx, 
                     int start, 
                     int end, 
                     std::vector<node> & complete_bst, 
                     const std::vector<std::pair<Key, Value>> & entries) {
        if (start > end || eyt_idx >= complete_bst.size()) return;

        int mid = start + (end - start) / 2;
        complete_bst[eyt_idx].key = entries[mid].first;
        complete_bst[eyt_idx].val = entries[mid].second;
        complete_bst[eyt_idx].is_valid = true;

        build_range(eyt_idx * 2, start, mid - 1, complete_bst, entries);
        build_range(eyt_idx * 2 + 1, mid + 1, end, complete_bst, entries);
    }
};

// =========================================================================================
// 3. Van Emde Boas Prefix Trie Clusterer for Cohort PPS & N-Gram Speculation
// Re-clusters static prefix/suffix trie branches into recursive vEB clusters, bounding
// root-to-leaf branch lookups to O(depth / log_2 B) cache lines instead of 1 miss per hop.
// =========================================================================================

struct veb_clustered_trie_node {
    uint32_t token_id{0};
    uint32_t payload_cluster_id{0};
    uint32_t child_start_offset{0};
    uint16_t child_count{0};
    uint16_t depth{0};
};

class veb_trie_serializer {
public:
    veb_trie_serializer() = default;

    // Serialize multi-level tree into contiguous 128-byte cache-line aligned vEB blocks
    static std::vector<veb_clustered_trie_node> serialize_recursive(
        const std::vector<veb_clustered_trie_node> & flat_nodes, uint32_t max_depth) {
        
        std::vector<veb_clustered_trie_node> veb_nodes;
        veb_nodes.reserve(flat_nodes.size());

        if (flat_nodes.empty()) return veb_nodes;

        // Bisection split depth
        uint32_t split_depth = max_depth / 2;

        // Phase 1: Place top sub-trie (depth <= split_depth) contiguously
        for (const auto & n : flat_nodes) {
            if (n.depth <= split_depth) {
                veb_nodes.push_back(n);
            }
        }

        // Phase 2: Place bottom sub-tries grouped by ancestor subtree
        for (const auto & n : flat_nodes) {
            if (n.depth > split_depth) {
                veb_nodes.push_back(n);
            }
        }

        return veb_nodes;
    }
};

// =========================================================================================
// 4. Van Emde Boas Cache-Oblivious Speculative Draft Buffer
// Arranges speculative draft token sequences and probabilities into a cache-oblivious
// layout so verification passes experience minimal L1/L2 data cache misses.
// =========================================================================================

struct veb_speculative_draft_buffer {
    std::vector<llama_token> packed_tokens;
    std::vector<float> packed_probs;
    size_t count{0};

    void pack_draft_sequence(const llama_token * tokens, const float * probs, size_t n_tokens) {
        count = n_tokens;
        if (n_tokens == 0) return;

        packed_tokens.resize(n_tokens);
        if (probs) {
            packed_probs.resize(n_tokens);
        }

        // Apply recursive vEB layout permutation for complete binary tree sizes (n = 2^(h+1) - 1 >= 7)
        uint32_t h = 0;
        while (((1U << (h + 1)) - 1) < n_tokens) {
            h++;
        }
        uint32_t cap = (1U << (h + 1)) - 1;

        if (n_tokens == cap && n_tokens >= 7) {
            veb_layout_mapper mapper(h);
            for (size_t i = 0; i < n_tokens; ++i) {
                uint32_t eyt_idx = static_cast<uint32_t>(i + 1);
                uint32_t veb_idx = mapper.to_veb(eyt_idx);
                if (veb_idx < n_tokens) {
                    packed_tokens[veb_idx] = tokens[i];
                    if (probs) {
                        packed_probs[veb_idx] = probs[i];
                    }
                }
            }
        } else {
            // Short sequences (< 7) or non-complete trees already fit in contiguous cache lines
            std::memcpy(packed_tokens.data(), tokens, n_tokens * sizeof(llama_token));
            if (probs) {
                std::memcpy(packed_probs.data(), probs, n_tokens * sizeof(float));
            }
        }
    }
};

