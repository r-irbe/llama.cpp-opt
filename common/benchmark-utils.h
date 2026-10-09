#pragma once

#include <atomic>
#include <cstdint>

/**
 * Microarchitectural Benchmark Reproducibility & Optimization Sink Utilities
 *
 * Prevents compiler Dead Code Elimination (DCE) and variable hoisting across
 * timed measurement intervals in microarchitectural benchmarks.
 *
 * Reference: HangukQuant (2026), Bilokon & Gunduz (2023), Google Benchmark.
 */

namespace benchmark {

template <typename T>
inline void DoNotOptimize(T const & value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(value) : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)value;
#endif
}

template <typename T>
inline void DoNotOptimize(T & value) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+m,r"(value) : : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)value;
#endif
}

inline void ClobberMemory() {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

} // namespace benchmark
