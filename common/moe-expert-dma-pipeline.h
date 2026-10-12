#pragma once

#include "llama-io-uring.h"
#include "ggml-cpu-numa.h"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace common {

/**
 * 2-Tier Asynchronous MoE Expert DMA Streaming Pipeline
 *
 * Implements high-throughput, zero-copy, direct-from-NVMe streaming for deferred
 * Mixture-of-Experts weights (--defer-experts):
 *
 * Tier 1: Local Storage NUMA Node HugeTLB Staging Pool
 *   - Pre-allocated 2MB HugeTLB buffer slots bound to the NVMe storage NUMA domain
 *     (e.g. Node 2 on dual AMD EPYC carrier card, discovered via ggml::cpu::numa::get_storage_numa_node()).
 *   - Pre-touched via MADV_POPULATE_WRITE to eliminate minor soft page faults from the critical path.
 *   - Pre-registered with Linux io_uring (IORING_REGISTER_BUFFERS / IORING_OP_READ_FIXED) to eliminate
 *     page table walks and IOMMU translation latency.
 *
 * Tier 2: Asynchronous Direct DMA Streaming via io_uring + O_DIRECT
 *   - Bypasses Linux page cache double-buffering, memory pressure stalls, and file cache locks.
 *   - Employs kernel SQPOLL thread (IORING_SETUP_SQPOLL) to eliminate syscall overhead.
 *   - When router logits select top-k experts for layer L, the prefetch engine immediately submits
 *     asynchronous read requests for layer L+1 / L+2 experts across PCIe Gen4 x16.
 *   - Overlaps layer L GEMM/GEMV compute with layer L+1 storage DMA, hiding transfer latency.
 *   - 128-byte cache-line separation (alignas(128)) on slot cursors to eliminate false sharing.
 *   - Automatic sector alignment (4096B) and seamless fallback to threaded pread if io_uring is unavailable.
 */

struct moe_expert_slice {
    int    layer = -1;
    int    expert = -1;
    size_t file_offset = 0;
    size_t num_bytes = 0;
};

struct alignas(128) moe_staging_slot {
    uint32_t slot_id = 0;
    void *   base_addr = nullptr;       // 2MB / HugeTLB aligned base address in staging pool
    size_t   capacity = 0;              // Slot capacity in bytes (e.g. 2MB or 4MB)

    // Active slice metadata
    int      layer = -1;
    int      expert = -1;
    size_t   file_offset = 0;
    size_t   bytes = 0;
    size_t   payload_offset = 0;        // Offset within base_addr where requested slice starts
    uint64_t ticket = 0;
    std::atomic<bool> is_ready{false};

    moe_staging_slot() = default;

    moe_staging_slot(const moe_staging_slot & other) {
        slot_id = other.slot_id;
        base_addr = other.base_addr;
        capacity = other.capacity;
        layer = other.layer;
        expert = other.expert;
        file_offset = other.file_offset;
        bytes = other.bytes;
        payload_offset = other.payload_offset;
        ticket = other.ticket;
        is_ready.store(other.is_ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    moe_staging_slot & operator=(const moe_staging_slot & other) {
        if (this != &other) {
            slot_id = other.slot_id;
            base_addr = other.base_addr;
            capacity = other.capacity;
            layer = other.layer;
            expert = other.expert;
            file_offset = other.file_offset;
            bytes = other.bytes;
            payload_offset = other.payload_offset;
            ticket = other.ticket;
            is_ready.store(other.is_ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    moe_staging_slot(moe_staging_slot && other) noexcept {
        slot_id = other.slot_id;
        base_addr = other.base_addr;
        capacity = other.capacity;
        layer = other.layer;
        expert = other.expert;
        file_offset = other.file_offset;
        bytes = other.bytes;
        payload_offset = other.payload_offset;
        ticket = other.ticket;
        is_ready.store(other.is_ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    moe_staging_slot & operator=(moe_staging_slot && other) noexcept {
        if (this != &other) {
            slot_id = other.slot_id;
            base_addr = other.base_addr;
            capacity = other.capacity;
            layer = other.layer;
            expert = other.expert;
            file_offset = other.file_offset;
            bytes = other.bytes;
            payload_offset = other.payload_offset;
            ticket = other.ticket;
            is_ready.store(other.is_ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

struct moe_dma_config {
    size_t slot_size = 2 * 1024 * 1024;    // 2MB HugeTLB page size
    size_t num_slots = 64;                 // 64 slots = 128 MB staging pool
    bool   try_direct_io = true;           // Use O_DIRECT for zero page-cache pollution
    bool   try_sqpoll = true;              // Use kernel SQPOLL thread
    bool   try_register_buffers = true;    // Use IORING_REGISTER_BUFFERS for zero IOMMU walks
    bool   populate_pages = true;          // Use MADV_POPULATE_WRITE for zero soft page faults
    int    numa_node = -1;                 // Storage NUMA node (-1 = auto-discover)
    unsigned queue_depth = 64;             // io_uring ring depth
};

class moe_expert_dma_pipeline {
public:
    using config = moe_dma_config;

    moe_expert_dma_pipeline() = default;
    ~moe_expert_dma_pipeline() {
        shutdown();
    }

    // Movable, non-copyable
    moe_expert_dma_pipeline(moe_expert_dma_pipeline && other) noexcept {
        *this = std::move(other);
    }

    moe_expert_dma_pipeline & operator=(moe_expert_dma_pipeline && other) noexcept {
        if (this != &other) {
            shutdown();
            m_cfg = other.m_cfg;
            m_model_path = std::move(other.m_model_path);
            m_fd = other.m_fd;
            m_owns_fd = other.m_owns_fd;
            m_storage_node = other.m_storage_node;
            m_staging_pool = other.m_staging_pool;
            m_pool_size = other.m_pool_size;
            m_is_huge_pages = other.m_is_huge_pages;
            m_slots = std::move(other.m_slots);
            m_expert_to_slot = std::move(other.m_expert_to_slot);
            m_ticket_to_slot = std::move(other.m_ticket_to_slot);
            m_io_ctx = std::move(other.m_io_ctx);
            m_fallback_mode = other.m_fallback_mode;
            m_slot_cursor.store(other.m_slot_cursor.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_total_prefetches.store(other.m_total_prefetches.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_cache_hits.store(other.m_cache_hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
            m_cache_misses.store(other.m_cache_misses.load(std::memory_order_relaxed), std::memory_order_relaxed);

            other.m_fd = -1;
            other.m_owns_fd = false;
            other.m_staging_pool = nullptr;
            other.m_pool_size = 0;
        }
        return *this;
    }

    moe_expert_dma_pipeline(const moe_expert_dma_pipeline &) = delete;
    moe_expert_dma_pipeline & operator=(const moe_expert_dma_pipeline &) = delete;

    // Initialize pipeline with given model file and configuration
    bool init(const std::string & model_path, int in_fd = -1, const config & cfg = {}) {
        shutdown();
        m_cfg = cfg;
        m_model_path = model_path;

        // Auto-discover storage NUMA node if not specified
        if (m_cfg.numa_node >= 0) {
            m_storage_node = m_cfg.numa_node;
        } else {
            m_storage_node = ggml::cpu::numa::get_storage_numa_node();
            if (m_storage_node < 0 && !model_path.empty()) {
                m_storage_node = ggml::cpu::numa::detect_path_numa_node(model_path.c_str());
            }
            if (m_storage_node < 0) {
                m_storage_node = 0;
            }
        }

        // Allocate 2MB-aligned HugeTLB staging pool
        m_pool_size = m_cfg.num_slots * m_cfg.slot_size;
        m_staging_pool = allocate_staging_pool(m_pool_size, m_storage_node, m_is_huge_pages);
        if (!m_staging_pool) {
            return false;
        }

        // Proactive pretoucher: populate page tables
        if (m_cfg.populate_pages) {
            madvise(m_staging_pool, m_pool_size, MADV_POPULATE_WRITE);
        }

        // Initialize staging slots
        m_slots.resize(m_cfg.num_slots);
        for (size_t i = 0; i < m_cfg.num_slots; ++i) {
            m_slots[i].slot_id = static_cast<uint32_t>(i);
            m_slots[i].base_addr = static_cast<char *>(m_staging_pool) + i * m_cfg.slot_size;
            m_slots[i].capacity = m_cfg.slot_size;
            m_slots[i].layer = -1;
            m_slots[i].expert = -1;
            m_slots[i].file_offset = 0;
            m_slots[i].bytes = 0;
            m_slots[i].payload_offset = 0;
            m_slots[i].ticket = 0;
            m_slots[i].is_ready.store(false, std::memory_order_relaxed);
        }

        // Open model file if fd not provided
        if (in_fd >= 0) {
            m_fd = in_fd;
            m_owns_fd = false;
        } else if (!m_model_path.empty()) {
            int open_flags = O_RDONLY;
            if (m_cfg.try_direct_io && (reinterpret_cast<uintptr_t>(m_staging_pool) % 512 == 0)) {
                open_flags |= O_DIRECT;
            }
            m_fd = open(m_model_path.c_str(), open_flags);
            if (m_fd < 0 && (open_flags & O_DIRECT)) {
                // Fall back without O_DIRECT if filesystem forbids direct I/O
                open_flags &= ~O_DIRECT;
                m_fd = open(m_model_path.c_str(), open_flags);
            }
            if (m_fd < 0) {
                shutdown();
                return false;
            }
            m_owns_fd = true;
        } else {
            shutdown();
            return false;
        }

        // Initialize io_uring subsystem
        llama_io_uring_config io_cfg;
        io_cfg.enable = true;
        io_cfg.try_direct_io = m_cfg.try_direct_io;
        io_cfg.try_sqpoll = m_cfg.try_sqpoll;
        io_cfg.try_register_buffers = m_cfg.try_register_buffers;
        io_cfg.queue_depth = m_cfg.queue_depth;
        io_cfg.populate_pages = false; // already done above

        bool io_ok = m_io_ctx.init(m_fd, m_staging_pool, m_pool_size, io_cfg);
        if (!io_ok) {
            // Graceful fallback to POSIX pread
            m_fallback_mode = true;
        } else {
            m_fallback_mode = false;
        }

        return true;
    }

    void shutdown() {
        if (m_staging_pool) {
            m_io_ctx.cleanup();
            if (m_is_huge_pages) {
                munmap(m_staging_pool, m_pool_size);
            } else {
                free(m_staging_pool);
            }
            m_staging_pool = nullptr;
            m_pool_size = 0;
        }

        if (m_owns_fd && m_fd >= 0) {
            close(m_fd);
        }
        m_fd = -1;
        m_owns_fd = false;
        m_slots.clear();
        m_expert_to_slot.clear();
        m_ticket_to_slot.clear();
        m_fallback_mode = false;
    }

    // Prefetch an individual expert slice asynchronously.
    // Returns a ticket handle (> 0) on success, or 0 on failure.
    uint64_t prefetch_expert(int layer, int expert, size_t file_offset, size_t num_bytes) {
        if (m_fd < 0 || num_bytes == 0) {
            return 0;
        }

        m_total_prefetches.fetch_add(1, std::memory_order_relaxed);
        uint64_t key = make_key(layer, expert);

        std::lock_guard<std::mutex> lock(m_pipe_mtx);

        // Check if slice is already cached in staging slots
        auto it = m_expert_to_slot.find(key);
        if (it != m_expert_to_slot.end()) {
            uint32_t slot_idx = it->second;
            if (slot_idx < m_slots.size()) {
                const auto & slot = m_slots[slot_idx];
                if (slot.layer == layer && slot.expert == expert &&
                    slot.file_offset == file_offset && slot.bytes >= num_bytes) {
                    m_cache_hits.fetch_add(1, std::memory_order_relaxed);
                    return slot.ticket;
                }
            }
        }

        m_cache_misses.fetch_add(1, std::memory_order_relaxed);

        // Allocate next slot circularly
        uint32_t slot_idx = static_cast<uint32_t>(m_slot_cursor.fetch_add(1, std::memory_order_relaxed) % m_slots.size());
        auto & slot = m_slots[slot_idx];

        // Evict old occupant from map if present
        if (slot.layer >= 0 && slot.expert >= 0) {
            uint64_t old_key = make_key(slot.layer, slot.expert);
            m_expert_to_slot.erase(old_key);
        }

        // Calculate 4KB sector alignment for O_DIRECT compliance
        size_t aligned_offset = (file_offset / 4096) * 4096;
        size_t offset_delta = file_offset - aligned_offset;
        size_t aligned_bytes = ((num_bytes + offset_delta + 4095) / 4096) * 4096;

        if (aligned_bytes > slot.capacity) {
            aligned_bytes = slot.capacity;
        }

        slot.layer = layer;
        slot.expert = expert;
        slot.file_offset = file_offset;
        slot.bytes = num_bytes;
        slot.payload_offset = offset_delta;
        slot.is_ready.store(false, std::memory_order_release);

        uint64_t ticket = 0;
        if (!m_fallback_mode && m_io_ctx.is_valid()) {
            ticket = m_io_ctx.submit_read(aligned_offset, slot.base_addr, aligned_bytes);
            if (ticket == 0) {
                // Submit failed, fallback to synchronous read
                ssize_t n = pread(m_fd, slot.base_addr, aligned_bytes, aligned_offset);
                slot.is_ready.store(n >= static_cast<ssize_t>(offset_delta + num_bytes), std::memory_order_release);
                ticket = static_cast<uint64_t>(slot_idx) + 1;
            }
        } else {
            // Direct POSIX pread fallback
            ssize_t n = pread(m_fd, slot.base_addr, aligned_bytes, aligned_offset);
            slot.is_ready.store(n >= static_cast<ssize_t>(offset_delta + num_bytes), std::memory_order_release);
            ticket = static_cast<uint64_t>(slot_idx) + 1;
        }

        slot.ticket = ticket;
        m_expert_to_slot[key] = slot_idx;
        m_ticket_to_slot[ticket] = slot_idx;

        return ticket;
    }

    // Prefetch all routed experts for a given layer in a single batch
    std::vector<uint64_t> prefetch_layer(int layer, const std::vector<int> & expert_ids,
                                        const std::vector<size_t> & file_offsets,
                                        const std::vector<size_t> & num_bytes) {
        std::vector<uint64_t> tickets;
        tickets.reserve(expert_ids.size());

        for (size_t i = 0; i < expert_ids.size(); ++i) {
            size_t off = i < file_offsets.size() ? file_offsets[i] : 0;
            size_t len = i < num_bytes.size() ? num_bytes[i] : 0;
            tickets.push_back(prefetch_expert(layer, expert_ids[i], off, len));
        }

        return tickets;
    }

    // Check if an expert slice is already fully resident and ready
    bool is_expert_ready(int layer, int expert) const {
        uint64_t key = make_key(layer, expert);
        std::lock_guard<std::mutex> lock(m_pipe_mtx);

        auto it = m_expert_to_slot.find(key);
        if (it == m_expert_to_slot.end()) {
            return false;
        }

        uint32_t slot_idx = it->second;
        if (slot_idx >= m_slots.size()) {
            return false;
        }

        const auto & slot = m_slots[slot_idx];
        if (slot.is_ready.load(std::memory_order_acquire)) {
            return true;
        }

        if (!m_fallback_mode && m_io_ctx.is_valid()) {
            // Non-blocking poll
            if (const_cast<llama_io_uring_context &>(m_io_ctx).poll_completion(slot.ticket, false)) {
                const_cast<moe_staging_slot &>(slot).is_ready.store(true, std::memory_order_release);
                return true;
            }
        }

        return false;
    }

    // Wait until an expert slice is fully loaded (blocks if not yet ready)
    bool wait_expert(int layer, int expert, uint64_t timeout_us = 1000000) {
        uint64_t ticket = 0;
        {
            uint64_t key = make_key(layer, expert);
            std::lock_guard<std::mutex> lock(m_pipe_mtx);
            auto it = m_expert_to_slot.find(key);
            if (it == m_expert_to_slot.end()) {
                return false;
            }
            ticket = m_slots[it->second].ticket;
        }
        return wait_ticket(ticket, timeout_us);
    }

    // Wait until a specific ticket has completed
    bool wait_ticket(uint64_t ticket, uint64_t timeout_us = 1000000) {
        if (ticket == 0) return false;

        uint32_t slot_idx = 0;
        {
            std::lock_guard<std::mutex> lock(m_pipe_mtx);
            auto it = m_ticket_to_slot.find(ticket);
            if (it == m_ticket_to_slot.end()) {
                return false;
            }
            slot_idx = it->second;
        }

        if (slot_idx >= m_slots.size()) return false;
        auto & slot = m_slots[slot_idx];

        if (slot.is_ready.load(std::memory_order_acquire)) {
            return true;
        }

        if (m_fallback_mode || !m_io_ctx.is_valid()) {
            return slot.is_ready.load(std::memory_order_acquire);
        }

        bool ok = m_io_ctx.poll_completion(ticket, true);
        if (ok) {
            slot.is_ready.store(true, std::memory_order_release);
        }
        return ok;
    }

    // Wait until all routed experts for a layer are ready
    bool wait_layer(int layer, const std::vector<int> & expert_ids, uint64_t timeout_us = 1000000) {
        bool all_ok = true;
        for (int exp : expert_ids) {
            if (!wait_expert(layer, exp, timeout_us)) {
                all_ok = false;
            }
        }
        return all_ok;
    }

    // Retrieve pointer to the resident expert data in the staging pool
    // Returns nullptr if not loaded or not ready
    const void * get_expert_data(int layer, int expert) const {
        uint64_t key = make_key(layer, expert);
        std::lock_guard<std::mutex> lock(m_pipe_mtx);

        auto it = m_expert_to_slot.find(key);
        if (it == m_expert_to_slot.end()) {
            return nullptr;
        }

        uint32_t slot_idx = it->second;
        if (slot_idx >= m_slots.size()) {
            return nullptr;
        }

        const auto & slot = m_slots[slot_idx];
        if (!slot.is_ready.load(std::memory_order_acquire)) {
            return nullptr;
        }

        return static_cast<const char *>(slot.base_addr) + slot.payload_offset;
    }

    void * get_expert_data(int layer, int expert) {
        return const_cast<void *>(const_cast<const moe_expert_dma_pipeline *>(this)->get_expert_data(layer, expert));
    }

    // Retrieve slot metadata
    const moe_staging_slot * get_slot(uint32_t slot_id) const {
        std::lock_guard<std::mutex> lock(m_pipe_mtx);
        if (slot_id < m_slots.size()) {
            return &m_slots[slot_id];
        }
        return nullptr;
    }

    // Direct read into an external caller-provided buffer (bypassing staging slots)
    uint64_t submit_direct_read(size_t file_offset, void * dest_addr, size_t num_bytes) {
        if (m_fd < 0 || num_bytes == 0 || !dest_addr) {
            return 0;
        }
        if (!m_fallback_mode && m_io_ctx.is_valid()) {
            return m_io_ctx.submit_read(file_offset, dest_addr, num_bytes);
        }
        // Fallback pread
        ssize_t n = pread(m_fd, dest_addr, num_bytes, file_offset);
        return n > 0 ? 1 : 0;
    }

    // Wait for a direct caller buffer read to complete
    bool wait_direct_read(uint64_t ticket) {
        if (ticket == 0) return false;
        if (m_fallback_mode || !m_io_ctx.is_valid()) return true;
        return m_io_ctx.poll_completion(ticket, true);
    }


    // Pipeline observability and performance stats
    llama_io_uring_stats get_io_stats() const {
        return m_io_ctx.get_stats();
    }

    size_t get_num_slots() const { return m_slots.size(); }
    size_t get_slot_size() const { return m_cfg.slot_size; }
    size_t get_pool_size() const { return m_pool_size; }
    int    get_numa_node() const { return m_storage_node; }
    bool   is_direct_io()  const { return m_io_ctx.is_direct_io(); }
    bool   is_sqpoll()     const { return m_io_ctx.is_sqpoll(); }
    bool   is_registered() const { return m_io_ctx.is_registered_buffers(); }
    bool   is_fallback()   const { return m_fallback_mode; }

    uint64_t get_total_prefetches() const { return m_total_prefetches.load(std::memory_order_relaxed); }
    uint64_t get_cache_hits()       const { return m_cache_hits.load(std::memory_order_relaxed); }
    uint64_t get_cache_misses()     const { return m_cache_misses.load(std::memory_order_relaxed); }

private:
    static inline uint64_t make_key(int layer, int expert) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) | static_cast<uint32_t>(expert);
    }

    static void * allocate_staging_pool(size_t size, int numa_node, bool & is_huge) {
        is_huge = false;
        void * ptr = nullptr;

#if defined(__linux__) && defined(MAP_HUGETLB)
        // Try allocating anonymous 2MB HugeTLB pages
        ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (ptr != MAP_FAILED && ptr != nullptr) {
            is_huge = true;
            return ptr;
        }
#endif

        // Fallback: allocate standard 2MB-aligned anonymous memory
        int res = posix_memalign(&ptr, 2 * 1024 * 1024, size);
        if (res == 0 && ptr != nullptr) {
            std::memset(ptr, 0, size);
            is_huge = false;
            return ptr;
        }

        return nullptr;
    }

    config m_cfg;
    std::string m_model_path;
    int m_fd = -1;
    bool m_owns_fd = false;
    int m_storage_node = -1;

    void * m_staging_pool = nullptr;
    size_t m_pool_size = 0;
    bool m_is_huge_pages = false;

    std::vector<moe_staging_slot> m_slots;
    std::unordered_map<uint64_t, uint32_t> m_expert_to_slot;
    std::unordered_map<uint64_t, uint32_t> m_ticket_to_slot;

    llama_io_uring_context m_io_ctx;
    bool m_fallback_mode = false;

    mutable std::mutex m_pipe_mtx;

    alignas(128) std::atomic<uint64_t> m_slot_cursor{0};
    std::atomic<uint64_t> m_total_prefetches{0};
    std::atomic<uint64_t> m_cache_hits{0};
    std::atomic<uint64_t> m_cache_misses{0};
};

} // namespace common
