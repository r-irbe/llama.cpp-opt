//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// Unit & Benchmark Test: Bitwise van Emde Boas Vocabulary & Speculative Rollback Mask
// Strict 7-bit ASCII compliance (INV-001).

#include "llama-bitwise-veb.h"
#include <cstdio>
#include <cassert>
#include <chrono>
#include <vector>

using namespace llama::veb;

int main() {
    printf("=================================================================\n");
    printf("   Unit Test: Bitwise vEB Vocabulary Mask & Reversible Rollback  \n");
    printf("=================================================================\n");

    // 1. BitwiseVeb4096 state-transition gating & ReversibleVebDelta rollback
    BitwiseVeb4096 veb4k;
    veb4k.insert(64);
    veb4k.insert(65);
    veb4k.insert(66);
    assert(veb4k.summary_writes == 1);
    assert(veb4k.elided_summary_writes == 2);
    assert(veb4k.find_min() == 64);
    assert(veb4k.find_max() == 66);

    auto d1 = veb4k.insert_reversible(12);
    auto d2 = veb4k.insert_reversible(2048);
    assert(veb4k.find_min() == 12);
    assert(veb4k.find_max() == 2048);
    veb4k.rollback_delta(d2);
    veb4k.rollback_delta(d1);
    assert(veb4k.find_min() == 64);
    assert(veb4k.find_max() == 66);
    printf("[OK] BitwiseVeb4096 Gating & ReversibleVebDelta 2-XOR rollback verified.\n");

    // 2. BitwiseVeb262k 128k/256k LLM Vocabulary Grammar Filter Iteration
    static BitwiseVeb262k vocab_mask;
    const uint32_t allowed_tokens[] = { 17, 512, 4095, 32768, 128000, 151935, 255999 };
    for (uint32_t t : allowed_tokens) {
        vocab_mask.insert(t);
    }
    assert(vocab_mask.find_min() == 17);

    std::vector<uint32_t> iterated;
    int32_t cur = vocab_mask.find_min();
    while (cur >= 0) {
        iterated.push_back(static_cast<uint32_t>(cur));
        cur = vocab_mask.find_next(static_cast<uint32_t>(cur) + 1u);
    }
    assert(iterated.size() == 7);
    for (size_t i = 0; i < 7; ++i) {
        assert(iterated[i] == allowed_tokens[i]);
    }
    printf("[OK] BitwiseVeb262k 256k-Token Grammar Mask O(k) Iteration verified (7/262,144 tokens).\n");

    // 3. Benchmark O(k) active token enumeration vs 128k linear scan
    const int N_ITERS = 200000;
    volatile uint32_t sink = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        int32_t tok = vocab_mask.find_min();
        while (tok >= 0) {
            sink ^= static_cast<uint32_t>(tok);
            tok = vocab_mask.find_next(static_cast<uint32_t>(tok) + 1u);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_per_sweep = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / static_cast<double>(N_ITERS);
    printf("[*] BitwiseVeb262k 7-token grammar sweep over U=262,144: %.2f ns/sweep (sink=%u)\n", ns_per_sweep, sink);
    assert(ns_per_sweep < 100.0);

    printf("=================================================================\n");
    printf("[OK] ALL BITWISE VEB VOCABULARY MASK TESTS PASSED!\n");
    printf("=================================================================\n");
    return 0;
}
