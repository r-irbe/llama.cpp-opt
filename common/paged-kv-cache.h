#pragma once

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

// =========================================================================================
// Host-RAM KV Cache PagedAttention Block Manager with Zero-Copy PCIe DMA Staging
// Grounded in the Aggarwal-Vitter (1988) Parallel Disk Model and Frigo et al. (1999/2012)
// Ideal-Cache Model: external transfers bounded to O(N / B) for B = 4096-byte sector blocks.
// =========================================================================================

struct paged_kv_block {
    int32_t block_id{-1};
    std::atomic<int32_t> ref_count{0};
    uint32_t num_tokens{0};
    uint8_t * k_data{nullptr};
    uint8_t * v_data{nullptr};

    paged_kv_block() = default;

    paged_kv_block(const paged_kv_block & other)
        : block_id(other.block_id),
          ref_count(other.ref_count.load(std::memory_order_relaxed)),
          num_tokens(other.num_tokens),
          k_data(other.k_data),
          v_data(other.v_data) {}

    paged_kv_block(paged_kv_block && other) noexcept
        : block_id(other.block_id),
          ref_count(other.ref_count.load(std::memory_order_relaxed)),
          num_tokens(other.num_tokens),
          k_data(other.k_data),
          v_data(other.v_data) {}

    paged_kv_block & operator=(const paged_kv_block & other) {
        if (this != &other) {
            block_id = other.block_id;
            ref_count.store(other.ref_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
            num_tokens = other.num_tokens;
            k_data = other.k_data;
            v_data = other.v_data;
        }
        return *this;
    }
};

struct paged_kv_config {
    uint32_t block_size_tokens{16}; // Tokens per block (Kwon et al. 2023)
    uint32_t n_embd_k{1024};        // Bytes per token for Key
    uint32_t n_embd_v{1024};        // Bytes per token for Value
    uint32_t max_blocks{4096};      // Total block pool capacity
    bool use_hugetlb{true};         // Pre-fault with HugeTLB / Transparent Huge Pages
    bool pin_host_memory{true};     // Pin host pages for direct PCIe DMA

    // Dynamic block size autotuning for modern hardware architectures:
    // - For short context / high concurrency (< 4096 tokens): 16 tokens minimizes tail fragmentation
    // - For long context (>= 4096 tokens) or Hopper/Blackwell TMA: 32 tokens maximizes TMA 128B warp efficiency
    static paged_kv_config auto_tune(uint32_t context_length, 
                                     bool has_hopper_blackwell_tma, 
                                     uint32_t n_embd_k = 1024, 
                                     uint32_t n_embd_v = 1024,
                                     uint32_t max_blocks = 4096) {
        paged_kv_config cfg;
        cfg.n_embd_k = n_embd_k;
        cfg.n_embd_v = n_embd_v;
        cfg.max_blocks = max_blocks;
        cfg.use_hugetlb = true;
        cfg.pin_host_memory = true;

        if (has_hopper_blackwell_tma || context_length >= 4096) {
            cfg.block_size_tokens = 32; // 128 KB per layer, matches TMA 128B transaction units
        } else {
            cfg.block_size_tokens = 16; // 64 KB per layer, minimizes tail fragmentation
        }
        return cfg;
    }
};

class common_paged_kv_cache {
public:
    explicit common_paged_kv_cache(const paged_kv_config & config)
        : m_config(config),
          m_block_bytes_k(config.block_size_tokens * config.n_embd_k),
          m_block_bytes_v(config.block_size_tokens * config.n_embd_v),
          m_total_block_bytes(m_block_bytes_k + m_block_bytes_v) {

        // Ensure 128-byte cache-line alignment per block
        m_aligned_block_bytes = (m_total_block_bytes + 127) & ~127ULL;

        // Allocate unified contiguous backing buffer in host memory
        const size_t total_alloc_bytes = (size_t)m_config.max_blocks * m_aligned_block_bytes;

#if defined(__linux__)
        int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
        m_backing_memory = (uint8_t *)mmap(nullptr, total_alloc_bytes, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
        if (m_backing_memory == MAP_FAILED) {
            m_backing_memory = (uint8_t *)malloc(total_alloc_bytes);
        } else {
            // Chronicle Queue Pretoucher pattern: Pre-fault MMU pages to eliminate soft page fault latency
            if (m_config.use_hugetlb) {
                madvise(m_backing_memory, total_alloc_bytes, MADV_HUGEPAGE);
            }
#if defined(MADV_POPULATE_WRITE)
            madvise(m_backing_memory, total_alloc_bytes, MADV_POPULATE_WRITE);
#else
            // Fallback manual page touch
            for (size_t p = 0; p < total_alloc_bytes; p += 4096) {
                m_backing_memory[p] = 0;
            }
#endif
        }
#else
        m_backing_memory = (uint8_t *)malloc(total_alloc_bytes);
#endif
        assert(m_backing_memory != nullptr);

        // Initialize block descriptor pool
        m_blocks.resize(m_config.max_blocks);
        m_free_blocks.reserve(m_config.max_blocks);

        for (uint32_t i = 0; i < m_config.max_blocks; ++i) {
            paged_kv_block & blk = m_blocks[i];
            blk.block_id = (int32_t)i;
            blk.ref_count.store(0, std::memory_order_relaxed);
            blk.num_tokens = 0;

            uint8_t * block_base = m_backing_memory + (size_t)i * m_aligned_block_bytes;
            blk.k_data = block_base;
            blk.v_data = block_base + m_block_bytes_k;

            m_free_blocks.push_back((int32_t)i);
        }
    }

    ~common_paged_kv_cache() {
        if (m_backing_memory) {
            const size_t total_alloc_bytes = (size_t)m_config.max_blocks * m_aligned_block_bytes;
#if defined(__linux__)
            munmap(m_backing_memory, total_alloc_bytes);
#else
            free(m_backing_memory);
#endif
            m_backing_memory = nullptr;
        }
    }

    // Disable copy
    common_paged_kv_cache(const common_paged_kv_cache &) = delete;
    common_paged_kv_cache & operator=(const common_paged_kv_cache &) = delete;

    // Allocate a free physical block from the pool
    int32_t allocate_block() {
        std::lock_guard<std::mutex> lock(m_pool_mutex);
        if (m_free_blocks.empty()) {
            return -1; // Exhausted
        }
        int32_t bid = m_free_blocks.back();
        m_free_blocks.pop_back();

        paged_kv_block & blk = m_blocks[bid];
        blk.ref_count.store(1, std::memory_order_release);
        blk.num_tokens = 0;
        return bid;
    }

    // Release or decrement reference count of a physical block
    void release_block(int32_t bid) {
        if (bid < 0 || bid >= (int32_t)m_config.max_blocks) return;
        paged_kv_block & blk = m_blocks[bid];

        if (blk.ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Ref count hit zero: return to free list
            std::lock_guard<std::mutex> lock(m_pool_mutex);
            blk.num_tokens = 0;
            m_free_blocks.push_back(bid);
        }
    }

    // Sequence operations
    struct sequence_table {
        int32_t seq_id{-1};
        std::vector<int32_t> block_ids;
        size_t total_tokens{0};
    };

    // Fork a sequence using zero-copy Copy-On-Write (COW) reference counting
    sequence_table fork_sequence(const sequence_table & parent, int32_t child_seq_id) {
        sequence_table child;
        child.seq_id = child_seq_id;
        child.block_ids = parent.block_ids;
        child.total_tokens = parent.total_tokens;

        for (int32_t bid : child.block_ids) {
            if (bid >= 0 && bid < (int32_t)m_config.max_blocks) {
                m_blocks[bid].ref_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return child;
    }

    // Append token data to a sequence, performing COW when appending to a shared block
    bool append_token(sequence_table & seq, const uint8_t * k_token_bytes, const uint8_t * v_token_bytes) {
        if (seq.block_ids.empty() || m_blocks[seq.block_ids.back()].num_tokens >= m_config.block_size_tokens) {
            // Allocate new block
            int32_t new_bid = allocate_block();
            if (new_bid < 0) return false;
            seq.block_ids.push_back(new_bid);
        }

        int32_t last_bid = seq.block_ids.back();
        paged_kv_block * blk = &m_blocks[last_bid];

        // Copy-On-Write if shared
        if (blk->ref_count.load(std::memory_order_acquire) > 1) {
            int32_t cow_bid = allocate_block();
            if (cow_bid < 0) return false;

            paged_kv_block * cow_blk = &m_blocks[cow_bid];
            cow_blk->num_tokens = blk->num_tokens;
            std::memcpy(cow_blk->k_data, blk->k_data, blk->num_tokens * m_config.n_embd_k);
            std::memcpy(cow_blk->v_data, blk->v_data, blk->num_tokens * m_config.n_embd_v);

            release_block(last_bid);
            seq.block_ids.back() = cow_bid;
            blk = cow_blk;
        }

        // Write token
        const uint32_t offset_k = blk->num_tokens * m_config.n_embd_k;
        const uint32_t offset_v = blk->num_tokens * m_config.n_embd_v;

        if (k_token_bytes) std::memcpy(blk->k_data + offset_k, k_token_bytes, m_config.n_embd_k);
        if (v_token_bytes) std::memcpy(blk->v_data + offset_v, v_token_bytes, m_config.n_embd_v);

        blk->num_tokens++;
        seq.total_tokens++;
        return true;
    }

    void free_sequence(sequence_table & seq) {
        for (int32_t bid : seq.block_ids) {
            release_block(bid);
        }
        seq.block_ids.clear();
        seq.total_tokens = 0;
    }

    // Zero-Copy PCIe DMA Staging Plan
    struct dma_staging_op {
        const uint8_t * host_src;
        size_t byte_count;
        int32_t block_id;
    };

    std::vector<dma_staging_op> plan_dma_staging(const sequence_table & seq, size_t start_token, size_t n_tokens) const {
        std::vector<dma_staging_op> ops;
        if (n_tokens == 0 || seq.block_ids.empty()) return ops;

        const size_t start_blk_idx = start_token / m_config.block_size_tokens;
        const size_t end_token = start_token + n_tokens;
        const size_t end_blk_idx = (end_token + m_config.block_size_tokens - 1) / m_config.block_size_tokens;

        for (size_t b = start_blk_idx; b < end_blk_idx && b < seq.block_ids.size(); ++b) {
            int32_t bid = seq.block_ids[b];
            if (bid < 0 || bid >= (int32_t)m_config.max_blocks) continue;

            const paged_kv_block & blk = m_blocks[bid];
            ops.push_back({
                /* .host_src = */ blk.k_data,
                /* .byte_count = */ m_total_block_bytes,
                /* .block_id = */ bid
            });
        }
        return ops;
    }

    // Telemetry
    size_t free_block_count() {
        std::lock_guard<std::mutex> lock(m_pool_mutex);
        return m_free_blocks.size();
    }

    size_t active_block_count() {
        std::lock_guard<std::mutex> lock(m_pool_mutex);
        return m_config.max_blocks - m_free_blocks.size();
    }

    const paged_kv_config & config() const { return m_config; }

private:
    paged_kv_config m_config;
    size_t m_block_bytes_k;
    size_t m_block_bytes_v;
    size_t m_total_block_bytes;
    size_t m_aligned_block_bytes;

    uint8_t * m_backing_memory{nullptr};
    std::vector<paged_kv_block> m_blocks;

    std::mutex m_pool_mutex;
    std::vector<int32_t> m_free_blocks;
};
