// Batch-invariance test for MoE mul_mat_id: the output for a token must not depend on how
// many other tokens share the call (1 vs 2 vs 4 tokens, identical routing).
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int check(enum ggml_type wtype, int n_threads, bool fused) {
    const int64_t ne0 = 512, ne1 = 128, n_expert = 8, n_used = 4;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> wf(ne0 * ne1 * n_expert);
    for (auto & x : wf) x = nd(rng);
    std::vector<float> tok(ne0 * n_used);
    for (auto & x : tok) x = nd(rng);
    int32_t routes[n_used] = {5, 1, 6, 2};

    std::vector<uint8_t> wq(ggml_row_size(wtype, ne0) * ne1 * n_expert);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, ne1 * n_expert, ne0, nullptr, nullptr);

    const int counts[3] = {1, 2, 4};
    std::vector<float> ref;
    int bad = 0;
    for (int ci = 0; ci < 3; ++ci) {
        const int nt = counts[ci];
        ggml_init_params ip = {256u * 1024 * 1024, nullptr, false};
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * as  = ggml_new_tensor_3d(ctx, wtype, ne0, ne1, n_expert);
        memcpy(as->data, wq.data(), wq.size());
        ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, n_used, nt);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, nt);
        for (int t = 0; t < nt; ++t) {
            memcpy((char *) b->data + t * b->nb[2], tok.data(), sizeof(float) * ne0 * n_used);
            memcpy((char *) ids->data + t * ids->nb[1], routes, sizeof(routes));
        }
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        (void) fused;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);

        std::vector<float> o((float *) out->data, (float *) out->data + ggml_nelements(out));
        const size_t per_tok = ne1 * n_used;
        if (ci == 0) {
            ref.assign(o.begin(), o.begin() + per_tok);
        } else {
            for (int t = 0; t < nt; ++t) {
                double md = 0; size_t nbad = 0;
                for (size_t i = 0; i < per_tok; ++i) {
                    float d = std::fabs(o[t * per_tok + i] - ref[i]);
                    md = std::fmax(md, d);
                    nbad += d != 0.f;
                }
                if (nbad) {
                    printf("  %-8s thr=%2d n_tok=%d tok=%d: %zu/%zu elems differ, max|d|=%.3g\n",
                           ggml_type_name(wtype), n_threads, nt, t, nbad, per_tok, md);
                    bad++;
                }
            }
        }
        ggml_free(ctx);
    }
    return bad;
}

int main() {
    int bad = 0;
    const enum ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_F16};
    for (auto t : types) {
        for (int thr : {1, 8, 56}) {
            bad += check(t, thr, false);
        }
    }
    printf(bad ? "test-moe-batch-invariance: %d MISMATCHES\n" : "test-moe-batch-invariance: all bitwise identical\n", bad);
    return bad ? 1 : 0;
}
