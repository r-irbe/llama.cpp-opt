#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

/**
 * Bounded lock-free multi-producer / single-consumer ring.
 *
 * Per-cell sequence numbers (D. Vyukov, "Bounded MPMC queue", 1024cores.net).
 * Producers claim a slot with a CAS on the enqueue cursor, write the payload,
 * then publish by storing the cell sequence with release semantics. The single
 * consumer reads the cell sequence with acquire semantics, so it never needs a
 * CAS on the dequeue cursor.
 *
 * - try_push never blocks; it returns false when the ring is full.
 * - try_pop returns false when the next cell in FIFO order is not yet published,
 *   even if later cells are (a claimed-but-unwritten slot stalls the consumer
 *   until that producer finishes its store, which is bounded and short).
 * - T must be default-constructible and move-assignable. After pop the cell is
 *   reset to T{} so heap-owning payloads release memory promptly.
 */
template <typename T>
class common_mpsc_ring {
    static_assert(std::is_default_constructible_v<T>, "T must be default-constructible");
    static_assert(std::is_move_assignable_v<T>,       "T must be move-assignable");

    struct alignas(64) cell {
        std::atomic<size_t> seq;
        T                   data;
    };

public:
    explicit common_mpsc_ring(size_t capacity_pow2)
        : m_mask(capacity_pow2 - 1),
          m_cells(new cell[capacity_pow2]) {
        // capacity must be a power of two >= 2
        if (capacity_pow2 < 2 || (capacity_pow2 & m_mask) != 0) {
            throw std::invalid_argument("common_mpsc_ring: capacity must be a power of two >= 2");
        }
        for (size_t i = 0; i < capacity_pow2; ++i) {
            m_cells[i].seq.store(i, std::memory_order_relaxed);
        }
    }

    common_mpsc_ring(const common_mpsc_ring &) = delete;
    common_mpsc_ring & operator=(const common_mpsc_ring &) = delete;

    size_t capacity() const { return m_mask + 1; }

    // Thread-safe for any number of producers.
    bool try_push(T && value) {
        size_t pos = m_enqueue.load(std::memory_order_relaxed);
        cell * c;
        for (;;) {
            c = &m_cells[pos & m_mask];
            const size_t   seq  = c->seq.load(std::memory_order_acquire);
            const intptr_t diff = (intptr_t) seq - (intptr_t) pos;
            if (diff == 0) {
                if (m_enqueue.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // full
            } else {
                pos = m_enqueue.load(std::memory_order_relaxed);
            }
        }
        c->data = std::move(value);
        c->seq.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Single consumer only.
    bool try_pop(T & out) {
        cell * c = &m_cells[m_dequeue & m_mask];
        const size_t seq = c->seq.load(std::memory_order_acquire);
        if ((intptr_t) seq - (intptr_t) (m_dequeue + 1) != 0) {
            return false; // empty, or next cell not yet published
        }
        out     = std::move(c->data);
        c->data = T{};
        c->seq.store(m_dequeue + m_mask + 1, std::memory_order_release);
        ++m_dequeue;
        return true;
    }

private:
    const size_t             m_mask;
    std::unique_ptr<cell[]>  m_cells;

    alignas(128) std::atomic<size_t> m_enqueue{0}; // shared by producers
    alignas(128) size_t              m_dequeue = 0; // consumer-private
    char m_pad[128 - sizeof(size_t)];
};
