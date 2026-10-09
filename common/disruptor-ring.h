#pragma once

#include "llama.h"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <thread>
#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

/**
 * Sequenced Circular Publication Ring Buffer (LMAX Disruptor Pattern)
 *
 * Microarchitectural Highlights:
 * 1. 128-byte cache line separation (alignas(128)) between producer and consumer cursors,
 *    eliminating false sharing on AMD Zen 3/Zen 4/EPYC systems with dual-line L2 prefetchers.
 * 2. Shadow cursor caching (m_cached_gating_seq, m_cached_published_seq) minimizing cross-socket
 *    cache coherence invalidation traffic (HangukQuant 2026).
 * 3. Explicit memory ordering: std::memory_order_release when publishing sequence updates and
 *    std::memory_order_acquire when polling cursors, avoiding heavyweight MFENCE / LOCK cycles.
 * 4. Batched publication (claim_batch, publish_batch) for multi-slot prefill dispatch (Ghosh 2023).
 */

inline void disruptor_cpu_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(_MSC_VER)
    _mm_pause();
    #else
    __builtin_ia32_pause();
    #endif
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

template <typename T, size_t Capacity = 1024>
class common_disruptor_ring {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr size_t MASK = Capacity - 1;

    // Cache Line 0 (128-byte aligned): Producer state & shadow gating copy
    alignas(128) std::atomic<int64_t> m_cursor{-1};        // highest published sequence
    std::atomic<int64_t>             m_claimed{-1};       // claimed sequence for batch reservation
    int64_t                          m_cached_gating{-1}; // shadow copy of consumer cursor
    uint64_t                         m_producer_shadow_hits{0};
    uint64_t                         m_producer_remote_polls{0};
    uint8_t                          m_pad0[128 - 2 * sizeof(std::atomic<int64_t>) - sizeof(int64_t) - 2 * sizeof(uint64_t)];

    // Cache Line 1 (128-byte aligned): Consumer state & shadow published copy
    alignas(128) std::atomic<int64_t> m_consumer_cursor{-1}; // sequence processed by consumer
    int64_t                          m_cached_published{-1}; // shadow copy of producer cursor
    uint64_t                         m_consumer_shadow_hits{0};
    uint64_t                         m_consumer_remote_polls{0};
    uint8_t                          m_pad1[128 - sizeof(std::atomic<int64_t>) - sizeof(int64_t) - 2 * sizeof(uint64_t)];

    // Cache Line 2+ (128-byte aligned): Ring Buffer Storage
    alignas(128) T m_buffer[Capacity];

public:
    common_disruptor_ring() {
        clear();
    }

    void clear() {
        m_cursor.store(-1, std::memory_order_relaxed);
        m_claimed.store(-1, std::memory_order_relaxed);
        m_cached_gating = -1;
        m_consumer_cursor.store(-1, std::memory_order_relaxed);
        m_cached_published = -1;
        m_producer_shadow_hits = 0;
        m_producer_remote_polls = 0;
        m_consumer_shadow_hits = 0;
        m_consumer_remote_polls = 0;
    }

    constexpr size_t capacity() const {
        return Capacity;
    }

    // --- Producer APIs ---

    // Reserve a single slot in the ring buffer. Returns sequence number, or -1 if full.
    int64_t claim() {
        const int64_t next_seq = m_claimed.load(std::memory_order_relaxed) + 1;
        const int64_t wrap_point = next_seq - static_cast<int64_t>(Capacity);

        // Check local shadow copy first to prevent cross-socket cache line invalidation
        if (wrap_point > m_cached_gating) {
            m_cached_gating = m_consumer_cursor.load(std::memory_order_acquire);
            m_producer_remote_polls++;
            if (wrap_point > m_cached_gating) {
                return -1; // Ring buffer full
            }
        } else {
            m_producer_shadow_hits++;
        }

        m_claimed.store(next_seq, std::memory_order_relaxed);
        return next_seq;
    }

    // Reserve a batch of n slots in the ring buffer. Returns true on success, false if space is insufficient.
    bool claim_batch(size_t n, int64_t & start_seq, int64_t & end_seq) {
        if (n == 0 || n > Capacity) return false;

        const int64_t current_claimed = m_claimed.load(std::memory_order_relaxed);
        const int64_t proposed_end = current_claimed + static_cast<int64_t>(n);
        const int64_t wrap_point = proposed_end - static_cast<int64_t>(Capacity);

        if (wrap_point > m_cached_gating) {
            m_cached_gating = m_consumer_cursor.load(std::memory_order_acquire);
            m_producer_remote_polls++;
            if (wrap_point > m_cached_gating) {
                return false;
            }
        } else {
            m_producer_shadow_hits++;
        }

        start_seq = current_claimed + 1;
        end_seq = proposed_end;
        m_claimed.store(proposed_end, std::memory_order_relaxed);
        return true;
    }

    // Access element at claimed sequence for in-place writing
    T & operator[](int64_t seq) {
        return m_buffer[static_cast<size_t>(seq) & MASK];
    }

    const T & operator[](int64_t seq) const {
        return m_buffer[static_cast<size_t>(seq) & MASK];
    }

    // Publish single claimed sequence, advancing the visible cursor with release semantics
    void publish(int64_t seq) {
        m_cursor.store(seq, std::memory_order_release);
    }

    // Publish batch of claimed sequences up to end_seq with release semantics
    void publish_batch(int64_t /* start_seq */, int64_t end_seq) {
        m_cursor.store(end_seq, std::memory_order_release);
    }

    // Non-blocking convenience publish
    bool try_publish(const T & item) {
        int64_t seq = claim();
        if (seq < 0) return false;
        m_buffer[static_cast<size_t>(seq) & MASK] = item;
        publish(seq);
        return true;
    }

    // Non-blocking batched convenience publish
    size_t try_publish_batch(const T * items, size_t count) {
        if (!items || count == 0) return 0;
        size_t to_claim = std::min(count, Capacity);
        int64_t start_seq = -1;
        int64_t end_seq = -1;
        if (!claim_batch(to_claim, start_seq, end_seq)) {
            // If full batch doesn't fit, try claiming 1
            int64_t single = claim();
            if (single < 0) return 0;
            m_buffer[static_cast<size_t>(single) & MASK] = items[0];
            publish(single);
            return 1;
        }

        for (size_t i = 0; i < to_claim; ++i) {
            m_buffer[static_cast<size_t>(start_seq + static_cast<int64_t>(i)) & MASK] = items[i];
        }
        publish_batch(start_seq, end_seq);
        return to_claim;
    }

    // --- Consumer APIs ---

    // Poll until expected_seq is published. Uses progressive backoff.
    int64_t wait_for(int64_t expected_seq, int max_spins = 1000) {
        if (m_cached_published >= expected_seq) {
            m_consumer_shadow_hits++;
            return m_cached_published;
        }

        int spins = 0;
        while (true) {
            int64_t current = m_cursor.load(std::memory_order_acquire);
            m_consumer_remote_polls++;
            if (current >= expected_seq) {
                m_cached_published = current;
                return current;
            }
            if (++spins < max_spins) {
                disruptor_cpu_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

    // Non-blocking check for available published sequences
    int64_t get_highest_published() {
        const int64_t next_needed = m_consumer_cursor.load(std::memory_order_relaxed) + 1;
        if (m_cached_published >= next_needed) {
            m_consumer_shadow_hits++;
            return m_cached_published;
        }
        m_cached_published = m_cursor.load(std::memory_order_acquire);
        m_consumer_remote_polls++;
        return m_cached_published;
    }

    // Non-blocking consume of next sequence
    bool try_consume(T & item) {
        const int64_t next_seq = m_consumer_cursor.load(std::memory_order_relaxed) + 1;
        if (next_seq > m_cached_published) {
            m_cached_published = m_cursor.load(std::memory_order_acquire);
            m_consumer_remote_polls++;
            if (next_seq > m_cached_published) {
                return false;
            }
        } else {
            m_consumer_shadow_hits++;
        }

        item = m_buffer[static_cast<size_t>(next_seq) & MASK];
        m_consumer_cursor.store(next_seq, std::memory_order_release);
        return true;
    }

    // Non-blocking batched consume
    size_t try_consume_batch(T * items, size_t max_count) {
        if (!items || max_count == 0) return 0;
        const int64_t next_seq = m_consumer_cursor.load(std::memory_order_relaxed) + 1;
        if (next_seq > m_cached_published) {
            m_cached_published = m_cursor.load(std::memory_order_acquire);
            m_consumer_remote_polls++;
            if (next_seq > m_cached_published) {
                return 0;
            }
        } else {
            m_consumer_shadow_hits++;
        }

        const int64_t available = m_cached_published - next_seq + 1;
        const size_t to_consume = std::min(max_count, static_cast<size_t>(available));

        for (size_t i = 0; i < to_consume; ++i) {
            items[i] = m_buffer[static_cast<size_t>(next_seq + static_cast<int64_t>(i)) & MASK];
        }

        m_consumer_cursor.store(next_seq + static_cast<int64_t>(to_consume) - 1, std::memory_order_release);
        return to_consume;
    }

    // Advance consumer cursor manually after batch processing
    void mark_consumed(int64_t up_to_seq) {
        m_consumer_cursor.store(up_to_seq, std::memory_order_release);
    }

    // Telemetry & Verification Metrics
    uint64_t producer_shadow_hits() const { return m_producer_shadow_hits; }
    uint64_t producer_remote_polls() const { return m_producer_remote_polls; }
    uint64_t consumer_shadow_hits() const { return m_consumer_shadow_hits; }
    uint64_t consumer_remote_polls() const { return m_consumer_remote_polls; }
    
    double producer_shadow_hit_rate() const {
        uint64_t total = m_producer_shadow_hits + m_producer_remote_polls;
        return total > 0 ? (100.0 * m_producer_shadow_hits / total) : 0.0;
    }

    double consumer_shadow_hit_rate() const {
        uint64_t total = m_consumer_shadow_hits + m_consumer_remote_polls;
        return total > 0 ? (100.0 * m_consumer_shadow_hits / total) : 0.0;
    }
};
