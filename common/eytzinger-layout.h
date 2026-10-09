#pragma once

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// =========================================================================================
// Eytzinger (Breadth-First Array) Search Architecture
// Grounded in:
//   1. Michael von Aitzing / Eytzinger (1590): Complete binary tree BFS numbering
//   2. J.W.J. Williams (1964): Binary heap array layout (left = 2k, right = 2k + 1)
//   3. Paul-Virak Khuong & Pat Morin (ACM JEA 2017, arXiv:1509.05053):
//      "Array Layouts for Comparison-Based Searching":
//      - 100% Branchless traversal via arithmetic update: k = 2*k + (x > data[k])
//      - Software prefetching at predictable subtree offsets: _mm_prefetch(&data[k * 16])
//      - Constant-time O(1) bitwise trailing-zero decoding: k >> (__builtin_ctz(~k) + 1)
//   4. Knuth (TAOCP Vol. 3 §6.2): Sequential and tree searching in compact arrays
// =========================================================================================

template <typename Key, typename Value = int>
class eytzinger_search_tree {
public:
    struct node {
        Key key{};
        Value val{};
    };

    eytzinger_search_tree() = default;

    // Linear O(N) construction from sorted entries via complete tree in-order traversal
    void build(const std::vector<std::pair<Key, Value>> & sorted_entries) {
        n_ = sorted_entries.size();
        if (n_ == 0) {
            tree_.clear();
            return;
        }

        // 1-based indexing: index 0 is unused sentinel, nodes reside at [1..n_]
        tree_.resize(n_ + 1);
        size_t sorted_idx = 0;
        build_inorder(1, sorted_idx, sorted_entries);
        assert(sorted_idx == n_);
    }

    // Branchless search with software prefetching (Khuong & Morin 2017)
    bool search(const Key & query, Value & out_val) const {
        if (n_ == 0) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            // Prefetch descendants 4 levels down (2^4 = 16 multiplier)
            _mm_prefetch(reinterpret_cast<const char *>(&tree_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            // Branchless conditional add (compiles to CMOV / ADC)
            k = (k << 1) | (query > tree_[k].key ? 1 : 0);
        }

        // Bitwise trailing-zero decoding: find the last node where search branched left
#if defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        if (idx > 0 && idx <= n_ && tree_[idx].key == query) {
            out_val = tree_[idx].val;
            return true;
        }
        return false;
    }

    // Branchless lower_bound: returns smallest element with key >= query
    bool lower_bound(const Key & query, Key & out_key, Value & out_val) const {
        if (n_ == 0) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&tree_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > tree_[k].key ? 1 : 0);
        }

#if defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        if (idx > 0 && idx <= n_) {
            out_key = tree_[idx].key;
            out_val = tree_[idx].val;
            return true;
        }
        return false;
    }

    bool contains(const Key & query) const {
        Value dummy{};
        return search(query, dummy);
    }

    size_t size() const {
        return n_;
    }

    bool empty() const {
        return n_ == 0;
    }

    const node & get_node(size_t idx) const {
        return tree_[idx];
    }

private:
    size_t n_{0};
    std::vector<node> tree_;

    void build_inorder(size_t k, size_t & sorted_idx, const std::vector<std::pair<Key, Value>> & entries) {
        if (k <= n_) {
            build_inorder(2 * k, sorted_idx, entries);
            tree_[k].key = entries[sorted_idx].first;
            tree_[k].val = entries[sorted_idx].second;
            sorted_idx++;
            build_inorder(2 * k + 1, sorted_idx, entries);
        }
    }
};

// =========================================================================================
// Flat Eytzinger Key Array (for Token Sets, Vocabularies, and Router IDs)
// =========================================================================================

template <typename Key>
class eytzinger_key_array {
public:
    eytzinger_key_array() = default;

    void build(const std::vector<Key> & sorted_keys) {
        n_ = sorted_keys.size();
        if (n_ == 0) {
            tree_.clear();
            return;
        }
        tree_.resize(n_ + 1);
        size_t sorted_idx = 0;
        build_inorder(1, sorted_idx, sorted_keys);
    }

    bool contains(const Key & query) const {
        if (n_ == 0) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&tree_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > tree_[k] ? 1 : 0);
        }

#if defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        return (idx > 0 && idx <= n_ && tree_[idx] == query);
    }

    size_t size() const { return n_; }

private:
    size_t n_{0};
    std::vector<Key> tree_;

    void build_inorder(size_t k, size_t & sorted_idx, const std::vector<Key> & keys) {
        if (k <= n_) {
            build_inorder(2 * k, sorted_idx, keys);
            tree_[k] = keys[sorted_idx++];
            build_inorder(2 * k + 1, sorted_idx, keys);
        }
    }
};
