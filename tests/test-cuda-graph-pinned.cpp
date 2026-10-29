// ============================================================================
// PolyGraph mechanism slice B — CUDA graph telemetry micro-bench.
//
// Proves, at the ggml level and without any model:
//   Phase 1 (pinned shape): ONE graph computes 3x with data-only changes
//     (new fills, same tensors) -> exactly ONE cudaGraphExec_t instance,
//     ONE capture, ZERO re-instantiations (the shape-pinning payoff).
//   Phase 2 (varying shapes): three distinct shapes -> THREE instances
//     (today's per-shape proliferation that PolyGraph eliminates).
//
// Reads the GGML_CUDA_GRAPH_STATS teardown summary by capturing stderr around
// backend destruction. Requires GGML_CUDA with graphs enabled; skips cleanly
// when the telemetry line is absent (non-graph builds, no GPU).
// ============================================================================

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct GraphStats {
    unsigned long long instances = 0;
    unsigned long long captures = 0;
    unsigned long long updates_ok = 0;
    unsigned long long updates_failed = 0;
    unsigned long long reinstantiations = 0;
    unsigned long long disable_events = 0;
    bool present = false;
};

#define BENCH_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

struct BenchGraph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * cf = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * bias = nullptr;
};

// Build [K x Kout] @ [K x N] + bias, all fp32, allocated on the backend.
int bench_build(BenchGraph & g, ggml_backend_t backend, int K, int Kout, int N) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g.ctx = ggml_init(ip);
    if (g.ctx == nullptr) {
        fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    g.a    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, Kout);
    g.b    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, N);
    g.bias = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, Kout, 1);
    struct ggml_tensor * out = ggml_add(g.ctx, ggml_mul_mat(g.ctx, g.a, g.b), g.bias);

    g.cf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.cf, out);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (g.buf == nullptr) {
        fprintf(stderr, "FAIL: backend alloc\n");
        return 1;
    }
    return 0;
}

// Fill all tensors with deterministic fill-dependent data (data-only change).
void bench_fill(const BenchGraph & g, int K, int Kout, int N, int fill) {
    std::vector<float> data(static_cast<size_t>(K) * Kout);
    for (int i = 0; i < K * Kout; ++i) {
        data[i] = 0.01f * ((i + fill) % 17);
    }
    ggml_backend_tensor_set(g.a, data.data(), 0, ggml_nbytes(g.a));

    data.assign(static_cast<size_t>(K) * N, 0.0f);
    for (int i = 0; i < K * N; ++i) {
        data[i] = 0.02f * ((i + fill) % 13);
    }
    ggml_backend_tensor_set(g.b, data.data(), 0, ggml_nbytes(g.b));

    const float bias_val = 0.5f * fill;
    ggml_backend_tensor_set(g.bias, &bias_val, 0, sizeof(float));
}

void bench_destroy(const BenchGraph & g) {
    ggml_backend_buffer_free(g.buf);
    ggml_free(g.ctx);
}

// Capture the GGML_CUDA_GRAPH_STATS teardown summary printed to stderr while
// the backend context is destroyed.
GraphStats free_and_read_stats(ggml_backend_t backend) {
    GraphStats st;
    fflush(stderr);
    const int saved = dup(STDERR_FILENO);
    int fds[2];
    if (saved < 0 || pipe(fds) != 0) {
        if (saved >= 0) close(saved);
        ggml_backend_free(backend);
        return st;
    }
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);

    ggml_backend_free(backend);

    fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);

    std::string captured;
    char chunk[512];
    ssize_t n;
    while ((n = read(fds[0], chunk, sizeof(chunk))) > 0) {
        captured.append(chunk, static_cast<size_t>(n));
    }
    close(fds[0]);

    const size_t pos = captured.find("cuda graph stats:");
    if (pos == std::string::npos) {
        return st;
    }
    st.present = true;
    sscanf(captured.c_str() + pos,
        "cuda graph stats: instances=%llu captures=%llu updates_ok=%llu updates_failed=%llu"
        " reinstantiations=%llu disable_events=%llu",
        &st.instances, &st.captures, &st.updates_ok, &st.updates_failed,
        &st.reinstantiations, &st.disable_events);
    return st;
}

} // namespace

int main() {
    printf("=================================================================\n");
    printf("  TEST: CUDA graph telemetry micro-bench (pinned vs varying shapes)\n");
    printf("=================================================================\n");

    setenv("GGML_CUDA_GRAPH_STATS", "1", 1);

    // NOTE (fork bug, documented in scratch/2026-10-26-spec-tuner-pi-takeover.md):
    // ggml_backend_cuda_init(device, nullptr, nullptr) in this fork leaves the
    // model-keyed CUDA buffer-type registry entry uninitialized, and the first
    // ggml_backend_alloc_ctx_tensors crashes in ggml_backend_buft_get_alignment.
    // The same pattern exists in examples/rpc/rpc-server.cpp. This micro-bench
    // therefore requires a model-handle-free backend that this fork cannot yet
    // construct; build it only for debugging (cmake -DPOLYGRAPH_MICROBENCH=ON),
    // and use llama-bench with GGML_CUDA_GRAPH_STATS=1 for real-workload
    // telemetry meanwhile.
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    if (backend == nullptr) {
        printf("  SKIP: no CUDA backend available\n");
        return 0;
    }

    // ---- Phase 1: pinned shape, three data-only computes ------------------
    {
        BenchGraph g;
        if (bench_build(g, backend, 64, 32, 16) != 0) {
            return 1;
        }
        for (int fill = 0; fill <= 2; ++fill) {
            bench_fill(g, 64, 32, 16, fill);
            if (ggml_backend_graph_compute(backend, g.cf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "FAIL: graph compute %d\n", fill);
                return 1;
            }
        }
        bench_destroy(g);

        const GraphStats st = free_and_read_stats(backend);
        if (!st.present) {
            printf("  SKIP: telemetry summary absent (build without CUDA graphs?)\n");
            return 0;
        }
        BENCH_ASSERT(st.instances == 1, "pinned shape must yield exactly one graph instance");
        BENCH_ASSERT(st.captures == 1, "pinned shape must capture exactly once");
        BENCH_ASSERT(st.reinstantiations == 0, "pinned shape must never re-instantiate");
        BENCH_ASSERT(st.updates_failed == 0, "no exec-update failures expected");
        BENCH_ASSERT(st.disable_events == 0, "graphs must not be disabled");
        printf("  [1] pinned shape: instances=1 captures=1 reinstantiations=0 PASSED OK\n");
    }

    // ---- Phase 2: varying shapes (today's proliferation) ------------------
    {
        const int ks[3] = {64, 96, 128};
        for (int i = 0; i < 3; ++i) {
            BenchGraph g;
            if (bench_build(g, backend, ks[i], 32, 16) != 0) {
                return 1;
            }
            bench_fill(g, ks[i], 32, 16, i);
            if (ggml_backend_graph_compute(backend, g.cf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "FAIL: graph compute shape %d\n", i);
                return 1;
            }
            bench_destroy(g);
        }

        const GraphStats st = free_and_read_stats(backend);
        if (!st.present) {
            printf("  SKIP: telemetry summary absent (build without CUDA graphs?)\n");
            return 0;
        }
        BENCH_ASSERT(st.instances == 3, "three distinct shapes must yield three instances");
        BENCH_ASSERT(st.captures == 3, "three distinct shapes must capture three times");
        printf("  [2] varying shapes: instances=3 captures=3 (proliferation shown) PASSED OK\n");
    }

    printf("\nAll test-cuda-graph-pinned assertions PASSED cleanly!\n");
    return 0;
}
