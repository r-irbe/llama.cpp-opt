#pragma once

#include "llama.h"
#include "spsc-ring.h"
#include "disruptor-ring.h"

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>

/**
 * Asynchronous Multi-CCX Drafting Arbiter and TreeGraft Fusion Engine.
 *
 * Microarchitectural Highlights:
 * 1. Dedicated CCX worker coordination using 128-byte cacheline-aligned lock-free SPSC rings
 *    (common_spsc_ring_buffer) and sequenced disruptor rings (common_disruptor_ring).
 * 2. TreeGraft multi-branch fusion combining branchless N-gram/Suffix candidate sub-branches
 *    with neural drafter proposals into a single unified verification batch.
 * 3. Hides drafting latency behind target verification by asynchronously pre-computing
 *    candidate proposals on secondary CCXs (e.g. Node 0 / CCX 0 for N-gram, Node 2 / CCX 2 for Neural).
 *
 * STATUS (honest, 2026-10-28): fuse_treegraft and the ring types are unit-tested
 * in tests/test-spec-arbiter.cpp and tests/test-spec-integration.cpp, but the
 * fused output is NOT consumed by the speculative decoding path — the previous
 * `last_fused` store in common/speculative.cpp was dead (no reader) and has been
 * removed; the spec_arbiter instance member was likewise never touched. The
 * dedicated per-CCX worker-thread pool that would asynchronously feed
 * ring_ngram / ring_neural is planned but not connected; nothing currently
 * pushes packets into the rings. Wire (a) an async producer, (b) a fused-output
 * consumer (tree-mask verify pass) before treating TreeGraft fusion as active.
 */

enum spec_drafter_source {
    SPEC_DRAFTER_SOURCE_TARGET_ONLY = 0,
    SPEC_DRAFTER_SOURCE_NGRAM       = 1,
    SPEC_DRAFTER_SOURCE_SUFFIX      = 2,
    SPEC_DRAFTER_SOURCE_NEURAL      = 3, // DFlash / DSpark / MTP
    SPEC_DRAFTER_SOURCE_TREEGRAFT   = 4,
};

struct spec_candidate_token {
    llama_token id;
    int32_t     parent_idx; // -1 if root, otherwise index in candidate batch
    int32_t     depth;
    float       score;
    int32_t     source;     // enum spec_drafter_source
};

struct spec_fused_candidates {
    std::vector<llama_token> tokens;
    std::vector<spec_candidate_token> tree_nodes;
    int32_t primary_source = SPEC_DRAFTER_SOURCE_TARGET_ONLY;
    bool    is_tree = false;
};

class spec_arbiter {
public:
    static constexpr size_t RING_CAPACITY = 256;

    struct drafter_packet {
        int32_t source_type;
        uint32_t n_tokens;
        llama_token tokens[16];
        float confidence[16];
    };

    // SPSC ring for N-gram / Suffix drafter (CCX 0 -> Arbiter)
    common_spsc_ring_buffer<drafter_packet, RING_CAPACITY> ring_ngram;
    
    // Disruptor ring for companion neural drafter (CCX 2 -> Arbiter)
    common_disruptor_ring<drafter_packet, RING_CAPACITY>   ring_neural;

    std::atomic<bool> running{false};

    spec_arbiter() = default;
    ~spec_arbiter() = default;

    /**
     * TreeGraft multi-branch fusion:
     * Merges linear or tree proposals from Suffix/N-gram and Neural drafters.
     * When both sources produce candidates:
     * - Common prefix is consolidated into a single linear trunk.
     * - Diverging candidate paths branch out into TreeGraft candidate sub-branches,
     *   allowing parallel target verification in a single forward pass.
     */
    static spec_fused_candidates fuse_treegraft(
            const std::vector<llama_token> & ngram_branch,
            const std::vector<llama_token> & neural_branch,
            int max_total_tokens = 16);

    /**
     * Poll incoming SPSC and disruptor rings without blocking.
     */
    bool poll_and_fuse(spec_fused_candidates & out_fused, int max_tokens = 16);
};
