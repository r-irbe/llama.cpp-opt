#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#include <immintrin.h>
#endif

/**
 * Microarchitectural Benchmark Reproducibility & Optimization Sink Utilities
 *
 * Prevents compiler Dead Code Elimination (DCE) and variable hoisting across
 * timed measurement intervals in microarchitectural benchmarks.
 *
 * Provides serialized hardware cycle timestamping using __rdtscp and _mm_lfence
 * with core migration detection (Sites 2021, Nagraj 2024, HangukQuant 2026).
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

// Serialized hardware cycle measurement using __rdtscp and _mm_lfence
// Guarantees zero out-of-order instruction hoisting across start/end boundaries.
inline uint64_t rdtsc_start() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
#else
    return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

inline uint64_t rdtsc_end(uint32_t * aux_core = nullptr) {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t aux = 0;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    if (aux_core) *aux_core = aux;
    return t;
#else
    if (aux_core) *aux_core = 0;
    return (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
#endif
}

inline double calibrate_tsc_freq_ghz() {
    auto start_wall = std::chrono::steady_clock::now();
    uint64_t start_tsc = rdtsc_start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    uint64_t end_tsc = rdtsc_end();
    auto end_wall = std::chrono::steady_clock::now();
    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_wall - start_wall).count();
    if (ns <= 0.0) return 2.5; // fallback
    return (double)(end_tsc - start_tsc) / ns; // cycles per nanosecond = GHz
}

} // namespace benchmark
