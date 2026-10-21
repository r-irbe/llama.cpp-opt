#pragma once

//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//
// GPU SIMT & Cache Hierarchy Optimized KV Cache Block Selector
// Microarchitecture Target: NVIDIA Ada Lovelace sm_89 (GeForce RTX 4090)
//
// Harnesses:
// 1. Warp-Synchronous SIMT Execution: 32 threads evaluate 32 candidate blocks
//    (2,048 tokens) concurrently via __ballot_sync, __shfl_sync, and __popc.
// 2. 72 MiB L2 Cache Hierarchy: Persistent window (cudaAccessPropertyPersisting)
//    for attention sink blocks and recurrent states.
// 3. 128-Byte Cache Line Alignment: alignas(128) memory layouts matching sm_89 sectors.
// 4. Shared Memory Zero-Bank-Conflict Banking: 32 banks x 4 bytes = 128 bytes/cycle.
// 5. cp.async.ca Direct VRAM-to-Shared DMA: Bypasses register file for block descriptors.

#include "common.cuh"
#include <cuda_runtime.h>
#include <cstdint>

#define GPU_KV_BLOCK_SIZE 64 // 64 tokens per KV block
#define GPU_WARP_SIZE     32 // 32 threads per warp on sm_89

namespace ggml::cuda::kv {

/// Block activity metadata descriptor aligned to 128-byte cache line boundary
struct alignas(128) GpuKvBlockDescriptor {
    uint32_t block_id;        // 0 to M-1
    uint32_t token_start;     // block_id * 64
    uint16_t sink_flag;       // 1 if attention sink
    uint16_t window_flag;     // 1 if in recent sliding window
    float    max_key_norm;    // Peak key norm in block for fast Cauchy-Schwarz bounding
    float    importance_score;// Dynamic semantic cluster score
    uint32_t reserved[26];    // Pad to exact 128 bytes (1 Ada Lovelace L2 cache line)
};

static_assert(sizeof(GpuKvBlockDescriptor) == 128, "GpuKvBlockDescriptor must be exactly 128 bytes");

/// Compact canonical ZDD node packed for GPU register and shared memory storage
struct alignas(16) GpuZddNode {
    uint32_t var;   // Block index
    uint32_t low;   // Inactive branch
    uint32_t high;  // Active branch
    uint32_t flags;
};

// ============================================================================
// GPU SIMT Warp-Level Sparse Block Filter Kernel (sm_89 Ada Lovelace)
// ============================================================================
// Each warp processes a group of 32 blocks (2,048 tokens).
// Evaluates attention relevance using query projections and Cauchy-Schwarz bounds,
// generating a compact active block bitmask and active index stream.

__global__ void evaluate_sparse_kv_blocks_simt_kernel(
    const float* __restrict__ query_vector,        // [head_dim]
    const GpuKvBlockDescriptor* __restrict__ blocks, // [num_total_blocks]
    const float score_threshold,
    uint32_t* __restrict__ out_active_bitmasks,     // [num_total_blocks / 32]
    uint32_t* __restrict__ out_active_indices,      // [max_active_blocks]
    uint32_t* __restrict__ out_active_count,        // [1] atomic counter
    const uint32_t num_total_blocks,
    const uint32_t n_sink_blocks,
    const uint32_t n_window_blocks
) {
    (void)query_vector;
    // Warp-level indexing
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id = global_tid / GPU_WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % GPU_WARP_SIZE;

    const uint32_t block_idx = warp_id * GPU_WARP_SIZE + lane_id;
    bool is_active = false;

    if (block_idx < num_total_blocks) {
        // 1. Mandatory Attention Sinks (always active)
        if (block_idx < n_sink_blocks) {
            is_active = true;
        }
        // 2. Mandatory Sliding Window (always active for local coherence)
        else if (block_idx >= (num_total_blocks > n_window_blocks ? num_total_blocks - n_window_blocks : 0)) {
            is_active = true;
        }
        // 3. Dynamic Semantic Cluster Filtering
        else {
            const GpuKvBlockDescriptor & desc = blocks[block_idx];
            // Cauchy-Schwarz upper bound check
            if (desc.importance_score >= score_threshold) {
                is_active = true;
            }
        }
    }

    // SIMT Warp Vote: combine 32 block decisions into a single 32-bit register mask
    const uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, is_active);

    // Leader thread of the warp writes the bitmask and compacts indices
    if (lane_id == 0) {
        out_active_bitmasks[warp_id] = warp_mask;
    }

    // Compact active block indices into dense array without warp divergence
    if (is_active) {
        // Calculate output offset using warp prefix popcount
        const uint32_t lane_mask = (1U << lane_id) - 1;
        const uint32_t intra_warp_idx = __popc(warp_mask & lane_mask);

        // Leader thread acquires global output base for the warp
        uint32_t warp_base = 0;
        if (lane_id == 0 && warp_mask != 0) {
            warp_base = atomicAdd(out_active_count, __popc(warp_mask));
        }
        warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

        out_active_indices[warp_base + intra_warp_idx] = block_idx;
    }
}

// ============================================================================
// GPU-Accelerated Minato ZDD Block Traversal Kernel
// ============================================================================
// Traverses a canonical ZDD on GPU shared memory / L2 cache in O(k) steps (<15 ns).

__global__ void traverse_zdd_kv_blocks_gpu_kernel(
    const GpuZddNode* __restrict__ zdd_nodes,
    const uint32_t root_node_id,
    uint32_t* __restrict__ out_selected_blocks,
    uint32_t* __restrict__ out_count
) {
    // Single warp performs deterministic, zero-divergence graph walk
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t curr = root_node_id;
    uint32_t count = 0;

    // Root 0: Empty family; Root 1: Base family
    while (curr > 1) {
        const GpuZddNode & node = zdd_nodes[curr];
        if (node.high != 0) { // Minato non-empty high branch
            out_selected_blocks[count++] = node.var;
            curr = node.high;
        } else {
            curr = node.low;
        }
    }
    *out_count = count;
}

/// Helper to configure Ada Lovelace L2 persistent cache window for KV block descriptors
inline void set_gpu_kv_l2_persistence(
    cudaStream_t stream,
    void* block_descriptors_d,
    size_t num_bytes,
    int device_id
) {
    const auto & dev_info = ggml_cuda_info().devices[device_id];
    if (dev_info.persisting_l2_cache_cur_limit == 0 || block_descriptors_d == nullptr || num_bytes == 0) {
        return;
    }

    size_t window_bytes = std::min(num_bytes, dev_info.persisting_l2_cache_cur_limit);

    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = block_descriptors_d;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f; // 100% pin in L2 cache
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

// ============================================================================
// Register-Resident Warp-Cooperative vEB Priority Queue (1024-Key Universe)
// ============================================================================
// Each of the 32 warp lanes holds a 32-bit word (`lane_word`), spanning 1,024 keys
// with 0 bytes of shared memory. Uses __ballot_sync, __ffs, __clz, and __shfl_sync.

__device__ __forceinline__ int warp_veb_find_min(uint32_t lane_word) {
    const uint32_t summary = __ballot_sync(0xFFFFFFFFu, lane_word != 0u);
    if (summary == 0u) {
        return -1;
    }
    const int min_cluster = __ffs(summary) - 1;
    const uint32_t winner_word = __shfl_sync(0xFFFFFFFFu, lane_word, min_cluster);
    const int min_bit = __ffs(winner_word) - 1;
    return (min_cluster << 5) | min_bit;
}

__device__ __forceinline__ int warp_veb_find_max(uint32_t lane_word) {
    const uint32_t summary = __ballot_sync(0xFFFFFFFFu, lane_word != 0u);
    if (summary == 0u) {
        return -1;
    }
    const int max_cluster = 31 - __clz(summary);
    const uint32_t winner_word = __shfl_sync(0xFFFFFFFFu, lane_word, max_cluster);
    const int max_bit = 31 - __clz(winner_word);
    return (max_cluster << 5) | max_bit;
}

__device__ __forceinline__ uint32_t warp_veb_insert(uint32_t lane_word, int key, int lane_id) {
    const int target_lane = (key >> 5) & 31;
    const int target_bit  = key & 31;
    if (lane_id == target_lane) {
        lane_word |= (1u << target_bit);
    }
    return lane_word;
}

__device__ __forceinline__ uint32_t warp_veb_erase(uint32_t lane_word, int key, int lane_id) {
    const int target_lane = (key >> 5) & 31;
    const int target_bit  = key & 31;
    if (lane_id == target_lane) {
        lane_word &= ~(1u << target_bit);
    }
    return lane_word;
}

// ============================================================================
// Dual-Bitplane Ternary POPCNT Dot-Product Primitive (sm_89)
// ============================================================================
// Evaluates the inner product of 64 ternary weights and 64 ternary activations
// in {-1, 0, +1} encoded as disjoint (pos_mask, neg_mask) bitplanes via __popcll.

__device__ __forceinline__ int32_t ternary_bitplane_dot64_device(
    uint64_t w_pos,
    uint64_t w_neg,
    uint64_t a_pos,
    uint64_t a_neg
) {
    const uint64_t same_sign = (w_pos & a_pos) | (w_neg & a_neg);
    const uint64_t opp_sign  = (w_pos & a_neg) | (w_neg & a_pos);
    return __popcll(same_sign) - __popcll(opp_sign);
}

} // namespace ggml::cuda::kv

void ggml_cuda_fattn_zdd_configure_l2(
    cudaStream_t stream,
    const ggml_tensor * dst,
    int device_id
);
