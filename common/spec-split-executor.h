#pragma once

#include "llama.h"

#include <vector>
#include <cstdint>

/**
 * Heterogeneous CPU/GPU Split Execution — staged host-side reference harness.
 *
 * Design contract (the end state this unit is a reference for):
 * 1. Drafter embedding / projection layers stay GPU-resident (< 50 MiB working set),
 *    preserving VRAM and Tensor Cores for the target model's prefill and decoding.
 * 2. The H_0 hidden-state slice (batch_size x hidden_dim) streams over PCIe into
 *    pinned host staging memory.
 * 3. Dedicated CPU CCX worker threads execute the drafter FFN backbone with SIMD/VNNI
 *    kernels and return only the final predicted token IDs to the target verification queue.
 *
 * STATUS (truthful): this unit is a deterministic HOST-SIDE REFERENCE of the staged
 * contract above, not a wired GPU/CPU pipeline. stream_h0_to_host() performs a host
 * memory copy from an already-hosted source (no PCIe DMA yet; the pinned allocation and
 * mapped staging are not wired), and execute_cpu_backbone() runs a scalar reference
 * projection (no AVX-512/VNNI kernel yet, no worker threads). It exists to pin down the
 * API contract, staging invariants, and determinism properties the real pipeline must
 * preserve, and is unit-tested in tests/test-spec-integration.cpp (section [4]).
 * Wiring the real heterogeneous pipeline remains open work; do not represent this unit
 * as DMA- or SIMD-capable until then.
 */

struct spec_split_state {
    int32_t batch_size    = 1;
    int32_t hidden_dim    = 0;
    bool    gpu_embedding = true;
    bool    cpu_backbone  = true;

    // Host staging buffer for the H_0 slice (pinned allocation pending real DMA wiring)
    std::vector<float> h0_host_buffer;
    // Predicted candidate tokens produced by the CPU backbone reference path
    std::vector<llama_token> predicted_tokens;
};

class spec_split_executor {
public:
    int32_t hidden_dim;
    bool    enabled;

    explicit spec_split_executor(int32_t d = 2048, bool enable_split = true)
        : hidden_dim(d), enabled(enable_split) {}

    /**
     * Stage 1: copy the H_0 hidden-state slice into the host staging buffer.
     * Reference behavior: plain host-to-host copy of b_size x hidden_dim floats.
     * Returns false on null source, non-positive batch size, or non-positive hidden_dim.
     */
    bool stream_h0_to_host(const float * gpu_h0_src, int32_t b_size, spec_split_state & state);

    /**
     * Stage 2: run the CPU backbone reference projection over the staged H_0 rows and
     * emit n_draft_tokens candidate token IDs. Deterministic: identical staged input
     * yields identical tokens. Returns false when the staging buffer is empty or
     * n_draft_tokens is non-positive.
     */
    bool execute_cpu_backbone(spec_split_state & state, int n_draft_tokens);

    const std::vector<llama_token> & get_predicted_tokens(const spec_split_state & state) const;
};
