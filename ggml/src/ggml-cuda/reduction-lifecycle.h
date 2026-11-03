#pragma once

#include "ggml.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// The caller holds the graph submission guard: membership and primary pointers
// cannot change until validation and the ensuing submission have completed.
// Keep this host-only so the actual registry policy can be tested without CUDA.
template <typename Context>
bool ggml_cuda_reduction_contexts_valid(
        const Context & caller, const ggml_cgraph & graph,
        const std::array<Context *, GGML_CUDA_MAX_DEVICES> * primary,
        const std::vector<Context *> * members, int device_count) {
    for (int n = 0; n < graph.n_nodes; ++n) {
        const auto & node = *graph.nodes[n];
        if (node.op != GGML_OP_REDUCE || node.op_params[3] == 1) {
            continue;
        }
        // Match ggml_is_noop's empty-tensor rule: execution never dispatches
        // an empty REDUCE, so it cannot consume any primary's resources.
        if (std::find(node.ne, node.ne + GGML_MAX_DIMS, int64_t(0)) != node.ne + GGML_MAX_DIMS) {
            continue;
        }
        const int nreduce = node.op_params[1];
        const int nhave = node.op_params[2];
        if (!primary || !members || device_count <= 0 || device_count > GGML_CUDA_MAX_DEVICES ||
                nreduce < 2 || nreduce > device_count || nreduce > GGML_MAX_SRC ||
                nhave < 2 || nhave > nreduce || caller.device < 0 || caller.device >= nreduce ||
                std::find(members->begin(), members->end(), &caller) == members->end() ||
                (*primary)[caller.device] != &caller) {
            return false;
        }
        const auto copy_mask = static_cast<uint32_t>(node.op_params[4]);
        // Device masks are 32-bit; the registry currently supports at most 16.
        static_assert(GGML_CUDA_MAX_DEVICES < 32, "revisit reduction mask bounds");
        if ((copy_mask >> nreduce) != 0 || (copy_mask & (uint32_t(1) << caller.device)) ||
                !node.src[caller.device] || node.data != node.src[caller.device]->data) {
            return false;
        }
        int actual_have = 0;
        for (int d = 0; d < nreduce; ++d) {
            const bool copy = (copy_mask & (uint32_t(1) << d)) != 0;
            if (!node.src[d] && !copy) {
                continue;
            }
            // Check identity against live membership BEFORE dereferencing a
            // primary that might otherwise be a stale non-member pointer.
            auto * peer = (*primary)[d];
            if (!node.src[d] || !peer ||
                    std::find(members->begin(), members->end(), peer) == members->end() ||
                    peer->device != d || peer->model != caller.model) {
                return false;
            }
            actual_have += !copy;
        }
        // nhave == nreduce also ensures all devices touched by NCCL are valid.
        if (actual_have != nhave) {
            return false;
        }
    }
    return true;
}

// Called under exclusive host retirement AND capture exclusion. First drain
// every resource device; only after ALL fences succeed retire any cached graph.
// Resource freeing/unregistration follows in the caller under the same guards.
template <typename Context, typename Fence, typename Retire>
bool ggml_cuda_retire_reduction_group(
        Context & departing, const std::vector<Context *> * members,
        int device_count, Fence && fence, Retire && retire) {
    if (device_count <= 0 || device_count > GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    std::array<bool, GGML_CUDA_MAX_DEVICES> devices {};
    auto record = [&](const Context & peer) {
        if (peer.device < 0 || peer.device >= device_count) {
            return false;
        }
        devices[peer.device] = true;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            bool owns_resources = peer.pools[d] || peer.cublas_handles[d];
            for (auto stream : peer.streams[d]) {
                owns_resources = owns_resources || stream != nullptr;
            }
            if (owns_resources) {
                if (d >= device_count) { return false; }
                devices[d] = true;
            }
        }
        return true;
    };
    if (!record(departing)) { return false; }
    if (members) {
        for (auto * peer : *members) {
            if (!peer || !record(*peer)) { return false; }
        }
    }
    for (int d = 0; d < device_count; ++d) {
        if (devices[d] && !fence(d)) { return false; }
    }
    if (members) {
        for (auto * peer : *members) { retire(*peer); }
    }
    if (!members || std::find(members->begin(), members->end(), &departing) == members->end()) {
        retire(departing);
    }
    return true;
}
