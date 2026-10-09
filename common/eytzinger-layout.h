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
struct eytzinger_node {
    Key key{};
    Value val{};
};

// =========================================================================================
// 1. Non-Owning Eytzinger Search Tree View (Zero-Copy over Mmapped / Direct I/O Buffers)
// Operates directly on raw memory buffers (const void * ptr or const node * ptr)
// using 1-based indexing: data_ = ptr - 1; k = 1; while (k <= n) ...
// 100% branchless CMOV traversal, software prefetching, and single-cycle TZCNT decoding.
// =========================================================================================

template <typename Key, typename Value = int>
class eytzinger_tree_view {
public:
    using node = eytzinger_node<Key, Value>;

    eytzinger_tree_view() = default;

    eytzinger_tree_view(const void * ptr, size_t n)
        : n_(n), data_(n > 0 && ptr ? reinterpret_cast<const node *>(ptr) - 1 : nullptr) {}

    eytzinger_tree_view(const node * ptr, size_t n)
        : n_(n), data_(n > 0 && ptr ? ptr - 1 : nullptr) {}

    bool search(const Key & query, Value & out_val) const {
        if (n_ == 0 || !data_) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&data_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > data_[k].key ? 1 : 0);
        }

#if defined(__BMI__)
        size_t idx = k >> (_tzcnt_u64(~static_cast<uint64_t>(k)) + 1);
#elif defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#elif defined(_MSC_VER)
        unsigned long trailing_zero = 0;
        _BitScanForward64(&trailing_zero, ~static_cast<unsigned __int64>(k));
        size_t idx = k >> (trailing_zero + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        if (idx > 0 && idx <= n_ && data_[idx].key == query) {
            out_val = data_[idx].val;
            return true;
        }
        return false;
    }

    bool lower_bound(const Key & query, Key & out_key, Value & out_val) const {
        if (n_ == 0 || !data_) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&data_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > data_[k].key ? 1 : 0);
        }

#if defined(__BMI__)
        size_t idx = k >> (_tzcnt_u64(~static_cast<uint64_t>(k)) + 1);
#elif defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#elif defined(_MSC_VER)
        unsigned long trailing_zero = 0;
        _BitScanForward64(&trailing_zero, ~static_cast<unsigned __int64>(k));
        size_t idx = k >> (trailing_zero + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        if (idx > 0 && idx <= n_) {
            out_key = data_[idx].key;
            out_val = data_[idx].val;
            return true;
        }
        return false;
    }

    bool contains(const Key & query) const {
        Value dummy{};
        return search(query, dummy);
    }

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    const node & get_node(size_t idx) const {
        assert(data_ && idx >= 1 && idx <= n_);
        return data_[idx];
    }

    const node * data() const {
        return data_ ? data_ + 1 : nullptr;
    }

private:
    size_t n_{0};
    const node * data_{nullptr};
};

// =========================================================================================
// 2. Non-Owning Eytzinger Flat Key Array View (for Token Sets, Vocabularies, and Router IDs)
// =========================================================================================

template <typename Key>
class eytzinger_key_view {
public:
    eytzinger_key_view() = default;

    eytzinger_key_view(const void * ptr, size_t n)
        : n_(n), data_(n > 0 && ptr ? reinterpret_cast<const Key *>(ptr) - 1 : nullptr) {}

    eytzinger_key_view(const Key * ptr, size_t n)
        : n_(n), data_(n > 0 && ptr ? ptr - 1 : nullptr) {}

    bool contains(const Key & query) const {
        if (n_ == 0 || !data_) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&data_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > data_[k] ? 1 : 0);
        }

#if defined(__BMI__)
        size_t idx = k >> (_tzcnt_u64(~static_cast<uint64_t>(k)) + 1);
#elif defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#elif defined(_MSC_VER)
        unsigned long trailing_zero = 0;
        _BitScanForward64(&trailing_zero, ~static_cast<unsigned __int64>(k));
        size_t idx = k >> (trailing_zero + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        return (idx > 0 && idx <= n_ && data_[idx] == query);
    }

    bool lower_bound(const Key & query, Key & out_key) const {
        if (n_ == 0 || !data_) return false;

        size_t k = 1;
        while (k <= n_) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_prefetch(reinterpret_cast<const char *>(&data_[std::min(k * 16, n_)]), _MM_HINT_T0);
#endif
            k = (k << 1) | (query > data_[k] ? 1 : 0);
        }

#if defined(__BMI__)
        size_t idx = k >> (_tzcnt_u64(~static_cast<uint64_t>(k)) + 1);
#elif defined(__GNUC__) || defined(__clang__)
        size_t idx = k >> (__builtin_ctzll(~static_cast<unsigned long long>(k)) + 1);
#elif defined(_MSC_VER)
        unsigned long trailing_zero = 0;
        _BitScanForward64(&trailing_zero, ~static_cast<unsigned __int64>(k));
        size_t idx = k >> (trailing_zero + 1);
#else
        size_t idx = k;
        while ((idx & 1) && idx > 0) idx >>= 1;
        idx >>= 1;
#endif

        if (idx > 0 && idx <= n_) {
            out_key = data_[idx];
            return true;
        }
        return false;
    }

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    const Key & get_key(size_t idx) const {
        assert(data_ && idx >= 1 && idx <= n_);
        return data_[idx];
    }

    const Key * data() const {
        return data_ ? data_ + 1 : nullptr;
    }

private:
    size_t n_{0};
    const Key * data_{nullptr};
};

// =========================================================================================
// 3. Owning Eytzinger Search Tree
// =========================================================================================

template <typename Key, typename Value = int>
class eytzinger_search_tree {
public:
    using node = eytzinger_node<Key, Value>;

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

    eytzinger_tree_view<Key, Value> view() const {
        return empty() ? eytzinger_tree_view<Key, Value>() : eytzinger_tree_view<Key, Value>(&tree_[1], n_);
    }

    std::vector<node> export_compact() const {
        if (n_ == 0) return {};
        return std::vector<node>(tree_.begin() + 1, tree_.end());
    }

    static std::vector<node> build_compact(const std::vector<std::pair<Key, Value>> & sorted_entries) {
        eytzinger_search_tree<Key, Value> t;
        t.build(sorted_entries);
        return t.export_compact();
    }

    bool search(const Key & query, Value & out_val) const {
        return view().search(query, out_val);
    }

    bool lower_bound(const Key & query, Key & out_key, Value & out_val) const {
        return view().lower_bound(query, out_key, out_val);
    }

    bool contains(const Key & query) const {
        return view().contains(query);
    }

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    const node & get_node(size_t idx) const { return tree_[idx]; }
    const node * data() const { return empty() ? nullptr : &tree_[1]; }

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
// 4. Flat Eytzinger Key Array
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

    eytzinger_key_view<Key> view() const {
        return empty() ? eytzinger_key_view<Key>() : eytzinger_key_view<Key>(&tree_[1], n_);
    }

    std::vector<Key> export_compact() const {
        if (n_ == 0) return {};
        return std::vector<Key>(tree_.begin() + 1, tree_.end());
    }

    static std::vector<Key> build_compact(const std::vector<Key> & sorted_keys) {
        eytzinger_key_array<Key> t;
        t.build(sorted_keys);
        return t.export_compact();
    }

    bool contains(const Key & query) const {
        return view().contains(query);
    }

    bool lower_bound(const Key & query, Key & out_key) const {
        return view().lower_bound(query, out_key);
    }

    size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    const Key * data() const { return empty() ? nullptr : &tree_[1]; }

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
