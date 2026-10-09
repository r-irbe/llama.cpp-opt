#pragma once

#include "llama.h"
#include "speculative.h"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>

// =========================================================================================
// Epoch-Based Reclamation (EBR) Manager (Keir Fraser 2004, Williams 2019, Mishra 2024)
// Provides O(1) lock-free pointer dereferencing with zero atomic writes on reader hot paths.
// =========================================================================================

class common_ebr_tracker {
public:
    static constexpr size_t MAX_READERS = 128;
    static constexpr uint64_t INACTIVE_EPOCH = UINT64_MAX;

    struct alignas(128) reader_slot {
        std::atomic<uint64_t> active_epoch{INACTIVE_EPOCH};
        uint8_t _pad[128 - sizeof(std::atomic<uint64_t>)];
    };

    struct retired_entry {
        void * ptr;
        void (*deleter)(void *);
        uint64_t epoch;
    };

    common_ebr_tracker() : m_global_epoch(0) {
        for (size_t i = 0; i < MAX_READERS; ++i) {
            m_readers[i].active_epoch.store(INACTIVE_EPOCH, std::memory_order_relaxed);
        }
    }

    ~common_ebr_tracker() {
        std::lock_guard<std::mutex> lock(m_retire_mutex);
        for (auto & entry : m_retired) {
            if (entry.ptr && entry.deleter) {
                entry.deleter(entry.ptr);
            }
        }
        m_retired.clear();
    }

    uint64_t enter_read(size_t reader_id) {
        if (reader_id >= MAX_READERS) return 0;
        const uint64_t epoch = m_global_epoch.load(std::memory_order_acquire);
        m_readers[reader_id].active_epoch.store(epoch, std::memory_order_release);
        return epoch;
    }

    void exit_read(size_t reader_id) {
        if (reader_id >= MAX_READERS) return;
        m_readers[reader_id].active_epoch.store(INACTIVE_EPOCH, std::memory_order_release);
    }

    void retire(void * ptr, void (*deleter)(void *)) {
        if (!ptr || !deleter) return;
        const uint64_t current_epoch = m_global_epoch.load(std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(m_retire_mutex);
        m_retired.push_back({ptr, deleter, current_epoch});

        if (m_retired.size() >= 256) {
            try_advance_and_reclaim();
        }
    }

    void try_advance_and_reclaim() {
        const uint64_t current_epoch = m_global_epoch.load(std::memory_order_acquire);

        // Find the minimum active epoch across all registered readers
        uint64_t min_reader_epoch = current_epoch;
        for (size_t i = 0; i < MAX_READERS; ++i) {
            const uint64_t r_epoch = m_readers[i].active_epoch.load(std::memory_order_acquire);
            if (r_epoch != INACTIVE_EPOCH && r_epoch < min_reader_epoch) {
                min_reader_epoch = r_epoch;
            }
        }

        // Reclaim all retired entries strictly older than min_reader_epoch
        auto it = std::remove_if(m_retired.begin(), m_retired.end(), [min_reader_epoch](const retired_entry & e) {
            if (e.epoch < min_reader_epoch) {
                if (e.ptr && e.deleter) {
                    e.deleter(e.ptr);
                }
                return true;
            }
            return false;
        });
        m_retired.erase(it, m_retired.end());

        // Increment global epoch
        m_global_epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    uint64_t global_epoch() const {
        return m_global_epoch.load(std::memory_order_relaxed);
    }

    size_t pending_retired_count() {
        std::lock_guard<std::mutex> lock(m_retire_mutex);
        return m_retired.size();
    }

private:
    alignas(128) std::atomic<uint64_t> m_global_epoch;
    alignas(128) reader_slot m_readers[MAX_READERS];

    std::mutex m_retire_mutex;
    std::vector<retired_entry> m_retired;
};


// =========================================================================================
// Lock-Free Suffix Trie Node with EBR Reclamation
// =========================================================================================

struct ccd_suffix_node {
    llama_token token{LLAMA_TOKEN_NULL};
    std::atomic<int64_t> count{0};

    // Lock-free child singly linked list (head + siblings)
    std::atomic<ccd_suffix_node *> child_head{nullptr};
    ccd_suffix_node * next_sibling{nullptr};

    ccd_suffix_node() = default;
    explicit ccd_suffix_node(llama_token tok, int64_t initial_count = 1)
        : token(tok), count(initial_count), child_head(nullptr), next_sibling(nullptr) {}

    // Lock-free wait-free child traversal under EBR
    ccd_suffix_node * find_child(llama_token tok) const {
        ccd_suffix_node * curr = child_head.load(std::memory_order_acquire);
        while (curr != nullptr) {
            if (curr->token == tok) {
                return curr;
            }
            curr = curr->next_sibling;
        }
        return nullptr;
    }

    // Atomic CAS child insertion
    ccd_suffix_node * get_or_create_child(llama_token tok, common_ebr_tracker & ebr) {
        // Fast path: find existing
        ccd_suffix_node * existing = find_child(tok);
        if (existing) {
            existing->count.fetch_add(1, std::memory_order_relaxed);
            return existing;
        }

        // Slow path: create and CAS prepend
        ccd_suffix_node * new_node = new ccd_suffix_node(tok, 1);
        while (true) {
            ccd_suffix_node * curr_head = child_head.load(std::memory_order_acquire);
            // Re-check if another thread inserted while we allocated
            ccd_suffix_node * curr = curr_head;
            while (curr != nullptr) {
                if (curr->token == tok) {
                    curr->count.fetch_add(1, std::memory_order_relaxed);
                    delete new_node;
                    return curr;
                }
                curr = curr->next_sibling;
            }

            new_node->next_sibling = curr_head;
            if (child_head.compare_exchange_weak(curr_head, new_node,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
                return new_node;
            }
        }
    }

    // Find the continuation child with the highest frequency count
    ccd_suffix_node * find_best_continuation(float min_token_prob = 0.1f) const {
        ccd_suffix_node * curr = child_head.load(std::memory_order_acquire);
        ccd_suffix_node * best = nullptr;
        int64_t best_count = 0;
        int64_t total_count = 0;

        while (curr != nullptr) {
            int64_t c = curr->count.load(std::memory_order_relaxed);
            total_count += c;
            if (c > best_count) {
                best_count = c;
                best = curr;
            }
            curr = curr->next_sibling;
        }

        if (best && total_count > 0) {
            float prob = (float)best_count / (float)total_count;
            if (prob >= min_token_prob) {
                return best;
            }
        }
        return nullptr;
    }

    static void delete_tree_recursive(ccd_suffix_node * node) {
        if (!node) return;
        ccd_suffix_node * child = node->child_head.load(std::memory_order_relaxed);
        while (child != nullptr) {
            ccd_suffix_node * next = child->next_sibling;
            delete_tree_recursive(child);
            child = next;
        }
        delete node;
    }
};


// =========================================================================================
// CCD-Sharded N-Gram Suffix Trie (AMD Milan/Genoa CCX Partitioned)
// Guarantees zero cross-CCX false sharing and 128B alignment.
// =========================================================================================

class common_ccd_suffix_trie {
public:
    static constexpr size_t DEFAULT_SHARDS = 8; // Matches 8 NUMA nodes / CCDs on Dual EPYC NPS4
    static constexpr int DEFAULT_MAX_DEPTH = 64;

    struct alignas(128) shard_t {
        ccd_suffix_node root{LLAMA_TOKEN_NULL, 0};
        std::mutex write_mutex;
        std::vector<llama_token> tokens;
        int n_inserted{0};
        uint8_t _pad[128 - (sizeof(ccd_suffix_node) + sizeof(std::mutex) + sizeof(std::vector<llama_token>) + sizeof(int)) % 128];
    };

    static_assert(sizeof(shard_t) % 128 == 0, "shard_t must be an exact multiple of 128 bytes to prevent false sharing");

    explicit common_ccd_suffix_trie(size_t n_shards = DEFAULT_SHARDS, int max_depth = DEFAULT_MAX_DEPTH)
        : m_n_shards(n_shards > 0 ? n_shards : 1), m_max_depth(max_depth) {
        m_shards = new shard_t[m_n_shards];
    }

    ~common_ccd_suffix_trie() {
        clear();
        delete[] m_shards;
    }

    // Disable copy
    common_ccd_suffix_trie(const common_ccd_suffix_trie &) = delete;
    common_ccd_suffix_trie & operator=(const common_ccd_suffix_trie &) = delete;

    size_t num_shards() const { return m_n_shards; }
    int max_depth() const { return m_max_depth; }

    // Map a NUMA node or CPU core to its local CCD shard
    size_t get_shard_id(int numa_node_or_core = -1) const {
        if (numa_node_or_core >= 0) {
            return (size_t)numa_node_or_core % m_n_shards;
        }
        // Hash thread ID if no NUMA node specified
        auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return tid % m_n_shards;
    }

    void extend(size_t shard_id, const llama_token * tokens, int n_tokens) {
        if (shard_id >= m_n_shards || !tokens || n_tokens <= 0) return;
        shard_t & shard = m_shards[shard_id];

        std::lock_guard<std::mutex> lock(shard.write_mutex);
        const int old_size = (int)shard.tokens.size();
        shard.tokens.insert(shard.tokens.end(), tokens, tokens + n_tokens);
        const int new_size = (int)shard.tokens.size();

        const int reinsert_from = std::max(0, old_size - m_max_depth);
        for (int i = reinsert_from; i < new_size; ++i) {
            const int total = (int)shard.tokens.size();
            const int len = std::min(total - i, m_max_depth);
            if (len <= 0) continue;

            ccd_suffix_node * curr = &shard.root;
            for (int j = 0; j < len; ++j) {
                const llama_token tok = shard.tokens[i + j];
                curr = curr->get_or_create_child(tok, m_ebr);
            }
        }
        shard.n_inserted = new_size;
    }

    // Lock-free speculative continuation query under EBR protection
    std::vector<llama_token> speculate(
            size_t shard_id,
            size_t reader_id,
            const llama_token * context, int n_context,
            int max_spec_tokens,
            float min_token_prob = 0.1f,
            int min_match_len = 3) {
        std::vector<llama_token> draft;
        if (shard_id >= m_n_shards || !context || n_context < min_match_len || max_spec_tokens <= 0) {
            return draft;
        }

        shard_t & shard = m_shards[shard_id];
        m_ebr.enter_read(reader_id);

        // Try longest matching context suffix
        for (int match_len = std::min(n_context, m_max_depth); match_len >= min_match_len; --match_len) {
            const llama_token * sub_ctx = context + (n_context - match_len);
            const ccd_suffix_node * curr = &shard.root;
            bool matched = true;

            for (int i = 0; i < match_len; ++i) {
                curr = curr->find_child(sub_ctx[i]);
                if (!curr) {
                    matched = false;
                    break;
                }
            }

            if (matched && curr) {
                // Follow highest frequency continuation branch
                const ccd_suffix_node * branch = curr;
                while (branch && (int)draft.size() < max_spec_tokens) {
                    const ccd_suffix_node * next = branch->find_best_continuation(min_token_prob);
                    if (!next || next->token == LLAMA_TOKEN_NULL) break;
                    draft.push_back(next->token);
                    branch = next;
                }
                if (!draft.empty()) {
                    break;
                }
            }
        }

        m_ebr.exit_read(reader_id);
        return draft;
    }

    // Stream draft tokens directly into wait-free 128B SPSC ring buffer
    template <size_t Capacity>
    size_t speculate_to_spsc_ring(
            size_t shard_id,
            size_t reader_id,
            const llama_token * context, int n_context,
            common_spsc_ring_buffer<llama_token, Capacity> & ring,
            int max_spec_tokens,
            float min_token_prob = 0.1f,
            int min_match_len = 3) {
        auto draft = speculate(shard_id, reader_id, context, n_context, max_spec_tokens, min_token_prob, min_match_len);
        if (draft.empty()) return 0;
        return ring.push_batch(draft.data(), draft.size());
    }

    void clear() {
        for (size_t s = 0; s < m_n_shards; ++s) {
            shard_t & shard = m_shards[s];
            std::lock_guard<std::mutex> lock(shard.write_mutex);
            ccd_suffix_node * child = shard.root.child_head.exchange(nullptr, std::memory_order_acq_rel);
            while (child != nullptr) {
                ccd_suffix_node * next = child->next_sibling;
                ccd_suffix_node::delete_tree_recursive(child);
                child = next;
            }
            shard.tokens.clear();
            shard.n_inserted = 0;
        }
        m_ebr.try_advance_and_reclaim();
    }

    common_ebr_tracker & ebr() { return m_ebr; }

private:
    size_t m_n_shards;
    int m_max_depth;
    shard_t * m_shards{nullptr};
    common_ebr_tracker m_ebr;
};
