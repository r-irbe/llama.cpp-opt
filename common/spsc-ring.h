#pragma once

#include <atomic>
#include <cstdint>
#include <algorithm>

// Cache-line friendly (128-byte padded for Zen 3 / modern dual-line prefetchers), wait-free SPSC ring buffer
template <typename T = int, size_t Capacity = 1024>
class common_spsc_ring_buffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr size_t MASK = Capacity - 1;

    // Cache line 1 (128-byte aligned): Producer state
    alignas(128) std::atomic<uint64_t> m_tail{0};
    uint64_t m_cached_head{0};
    uint8_t  m_pad0[128 - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t)];

    // Cache line 2 (128-byte aligned): Consumer state
    alignas(128) std::atomic<uint64_t> m_head{0};
    uint64_t m_cached_tail{0};
    uint8_t  m_pad1[128 - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t)];

    // Buffer storage (128-byte aligned)
    alignas(128) T m_buffer[Capacity];

public:
    common_spsc_ring_buffer() { clear(); }

    bool push(const T & item) {
        const uint64_t current_tail = m_tail.load(std::memory_order_relaxed);
        if (current_tail - m_cached_head >= Capacity) {
            m_cached_head = m_head.load(std::memory_order_acquire);
            if (current_tail - m_cached_head >= Capacity) return false;
        }
        m_buffer[current_tail & MASK] = item;
        m_tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T & item) {
        const uint64_t current_head = m_head.load(std::memory_order_relaxed);
        if (current_head == m_cached_tail) {
            m_cached_tail = m_tail.load(std::memory_order_acquire);
            if (current_head == m_cached_tail) return false;
        }
        item = m_buffer[current_head & MASK];
        m_head.store(current_head + 1, std::memory_order_release);
        return true;
    }

    size_t push_batch(const T * items, size_t count) {
        if (!items || count == 0) return 0;
        const uint64_t current_tail = m_tail.load(std::memory_order_relaxed);
        uint64_t diff = current_tail - m_cached_head;
        uint64_t free_slots = (diff < Capacity) ? (Capacity - diff) : 0;
        if (free_slots < count) {
            m_cached_head = m_head.load(std::memory_order_acquire);
            diff = current_tail - m_cached_head;
            free_slots = (diff < Capacity) ? (Capacity - diff) : 0;
        }
        const size_t to_push = std::min(count, (size_t)free_slots);
        for (size_t i = 0; i < to_push; ++i) {
            m_buffer[(current_tail + i) & MASK] = items[i];
        }
        if (to_push > 0) {
            m_tail.store(current_tail + to_push, std::memory_order_release);
        }
        return to_push;
    }

    size_t pop_batch(T * items, size_t max_count) {
        if (!items || max_count == 0) return 0;
        const uint64_t current_head = m_head.load(std::memory_order_relaxed);
        uint64_t avail = (m_cached_tail >= current_head) ? (m_cached_tail - current_head) : 0;
        if (avail == 0) {
            m_cached_tail = m_tail.load(std::memory_order_acquire);
            avail = (m_cached_tail >= current_head) ? (m_cached_tail - current_head) : 0;
        }
        const size_t to_pop = std::min(max_count, (size_t)avail);
        for (size_t i = 0; i < to_pop; ++i) {
            items[i] = m_buffer[(current_head + i) & MASK];
        }
        if (to_pop > 0) {
            m_head.store(current_head + to_pop, std::memory_order_release);
        }
        return to_pop;
    }

    size_t size() const {
        const uint64_t head = m_head.load(std::memory_order_acquire);
        const uint64_t tail = m_tail.load(std::memory_order_acquire);
        return (tail >= head) ? (tail - head) : 0;
    }

    constexpr size_t capacity() const {
        return Capacity;
    }

    bool empty() const {
        return m_head.load(std::memory_order_acquire) == m_tail.load(std::memory_order_acquire);
    }

    void clear() {
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
        m_cached_head = 0;
        m_cached_tail = 0;
    }
};
