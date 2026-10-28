// Transcription of the Lean-verified evaluate_kv_zdd_sparse_kernel
// (rust-mmap-engine/kernels/verified_kv_zdd_simt.cu) with a CPU reference,
// correctness check, and expansion-throughput micro-benchmark.
//
// Build:  nvcc -O3 -arch=sm_89 -extended-lambda tests/test-zdd-mask-simt.cu -o build/bin/test-zdd-mask-simt
// Motivation: scratch/zdd-gpu-simt-swar-analysis.md §6.1 — replace the
// per-step host-built skip-mask upload with on-GPU ballot expansion.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

// --- verified kernel (verbatim semantics; one KV block per lane) -----------
__global__ void evaluate_kv_zdd_sparse_kernel(
    const float * __restrict__ query_vector,
    const uint32_t * __restrict__ block_tags,
    uint32_t * __restrict__ out_active_bitmasks,
    const uint32_t num_total_blocks,
    const uint32_t n_sink_blocks) {
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id    = global_tid >> 5;
    const uint32_t lane_id    = threadIdx.x & 31U;
    const uint32_t block_idx  = (warp_id << 5) + lane_id;
    const float q_val         = query_vector[lane_id];
    const uint32_t tag        = (block_idx < num_total_blocks) ? block_tags[block_idx] : 0U;
    bool is_active = (block_idx < num_total_blocks) &&
                     ((block_idx < n_sink_blocks) || ((tag != 0U) && (q_val != 0.0f)));
    const uint32_t warp_mask = __ballot_sync(0xFFFFFFFFU, is_active);
    if (lane_id == 0U) {
        out_active_bitmasks[warp_id] = warp_mask;
    }
}

// --- CPU reference ---------------------------------------------------------
static void reference_mask(const std::vector<float> & q,
                           const std::vector<uint32_t> & tags,
                           uint32_t n_blocks, uint32_t n_sink,
                           std::vector<uint32_t> & out) {
    out.assign((n_blocks + 31) / 32, 0U);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const bool active = (b < n_sink) || (tags[b] != 0U && q[b & 31U] != 0.0f);
        if (active) {
            out[b >> 5] |= (1U << (b & 31U));
        }
    }
}

int main() {
    // 10M-context scale: 10,485,760 tokens / 64-token blocks = 163,840 blocks.
    const uint32_t n_blocks = 163840;
    const uint32_t n_sink   = 8;

    std::mt19937 rng(42);
    std::vector<float> q(32);
    for (auto & v : q) { v = (rng() % 4 == 0) ? 0.0f : 1.0f + (rng() % 100) / 100.0f; }
    std::vector<uint32_t> tags(n_blocks);
    for (auto & t : tags) { t = (rng() % 100 < 12) ? uint32_t(rng()) : 0U; } // ~12% tagged

    // device buffers
    float * d_q = nullptr; uint32_t * d_tags = nullptr; uint32_t * d_out = nullptr;
    cudaMalloc(&d_q,    q.size() * 4);
    cudaMalloc(&d_tags, tags.size() * 4);
    cudaMalloc(&d_out,  ((n_blocks + 31) / 32) * 4);
    cudaMemcpy(d_q,    q.data(),    q.size() * 4,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_tags, tags.data(), tags.size() * 4, cudaMemcpyHostToDevice);

    const uint32_t threads = 1024;
    const uint32_t grids   = (n_blocks + threads - 1) / threads;
    const uint32_t warps   = (grids * threads) / 32;

    // correctness
    evaluate_kv_zdd_sparse_kernel<<<grids, threads>>>(d_q, d_tags, d_out, n_blocks, n_sink);
    cudaDeviceSynchronize();
    std::vector<uint32_t> got((n_blocks + 31) / 32);
    cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<uint32_t> ref;
    reference_mask(q, tags, n_blocks, n_sink, ref);
    size_t mismatch = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) { ++mismatch; }
    }
    uint32_t popcount = 0;
    for (auto w : got) { popcount += __builtin_popcount(w); }
    printf("blocks=%u warps=%u active=%u (%.1f%%) mask_words=%zu mismatches=%zu\n",
           n_blocks, warps, popcount, 100.0 * popcount / n_blocks, ref.size(), mismatch);
    if (mismatch != 0) {
        printf("FAIL: GPU mask != CPU reference\n");
        return 1;
    }
    printf("correctness: PASS (ballot mask == CPU reference)\n");

    // throughput: time 1000 expansions
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int iters = 1000;
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) {
        evaluate_kv_zdd_sparse_kernel<<<grids, threads>>>(d_q, d_tags, d_out, n_blocks, n_sink);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    const float us_per = ms * 1000.f / iters;
    const float mblocks_s = n_blocks / us_per; // million blocks per second
    printf("expansion: %.2f us per pass @ %u blocks (%.1f Mblocks/s) over %d iters\n",
           us_per, n_blocks, mblocks_s, iters);
    printf("=> per-decode-step overhead at 10M ctx: %.2f us (vs host build+PCIe upload)\n", us_per);

    cudaFree(d_q); cudaFree(d_tags); cudaFree(d_out);
    return 0;
}
