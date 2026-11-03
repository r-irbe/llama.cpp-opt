// Unit tests for the MXFP8 (E4M3 fn + E8M0 block scale) encode and compute path.
//
// The claim under test is "compute is exact with respect to the storage/gather
// kernel": the dot product must equal the double-precision dot of the values
// that dequantize_row_mxfp8 produces, with only float-accumulation rounding.
//
// Only the public ggml API is used (type traits + row size), so this test does
// not depend on internal block layouts.

#include "ggml.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

static int g_failed = 0;

static void check(bool ok, const char * what, double got = 0.0, double want = 0.0) {
    if (!ok) {
        ++g_failed;
        printf("FAILED  %s (got %.9g, want %.9g)\n", what, got, want);
    } else {
        printf("ok      %s\n", what);
    }
}

static double dot_ref(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        s += double(a[i]) * double(b[i]);
    }
    return s;
}

// Integration: a real MUL_MAT graph with MXFP8 weights on the CPU backend.
// Returns 0 when the gate/expectations match, non-zero on failure.
static int graph_check(bool compute_enabled, const ggml_type_traits_t & mx, const ggml_type_traits_t & q8) {
    struct ggml_init_params params = { /* .mem_size = */ 64u*1024*1024, /* .mem_buffer = */ nullptr, /* .no_alloc = */ false };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    const int n = 512;
    const int rows = 3;
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP8, n, rows);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    struct ggml_tensor * out = ggml_mul_mat(ctx, w, a);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu != nullptr);
    const bool supported = ggml_backend_supports_op(cpu, out);
    ggml_backend_free(cpu);
    if (supported != compute_enabled) {
        printf("FAILED  supports_op(MUL_MAT with MXFP8 weights) == %s (got %s)\n",
               compute_enabled ? "true" : "false", supported ? "true" : "false");
        ggml_free(ctx);
        return 1;
    }
    printf("ok      supports_op(MUL_MAT with MXFP8 weights) == %s\n", compute_enabled ? "true" : "false");
    if (!compute_enabled) {
        ggml_free(ctx);
        return 0;
    }

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> wf(n*rows), af(n);
    for (int b = 0; b < rows; ++b) {
        const float amax = std::ldexp(1.0f, -8 + 8*b);   // three different block scales
        for (int j = 0; j < n; ++j) wf[b*n + j] = u(rng) * amax * 0.5f;
        wf[b*n] = amax;
    }
    for (auto & v : af) v = 2.0f * u(rng);

    std::vector<uint8_t> wq(rows * ggml_row_size(GGML_TYPE_MXFP8, n));
    for (int b = 0; b < rows; ++b) mx.from_float_ref(wf.data() + b*n, wq.data() + b*ggml_row_size(GGML_TYPE_MXFP8, n), n);
    std::memcpy(w->data, wq.data(), wq.size());
    std::memcpy(a->data, af.data(), af.size()*sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 4);

    // reference: decoded weights dotted with the Q8_0 representation of the
    // activations (the CPU backend quantizes src1 to vec_dot_type).
    std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_0, n));
    std::vector<float> ad(n);
    q8.from_float(af.data(), aq.data(), n);
    q8.to_float(aq.data(), ad.data(), n);

    int fails = 0;
    if (st != GGML_STATUS_SUCCESS) {
        printf("FAILED  graph compute status %d\n", (int) st);
        ++fails;
    }
    for (int b = 0; b < rows; ++b) {
        std::vector<float> wd(n);
        mx.to_float(wq.data() + b*ggml_row_size(GGML_TYPE_MXFP8, n), wd.data(), n);
        const double want = dot_ref(wd, ad);
        const float got = ggml_get_f32_1d(out, b);
        const double err = std::fabs(double(got) - want) / (std::fabs(want) + 1e-30);
        if (err >= 1e-5) {
            printf("FAILED  graph row %d: got %.9g want %.9g (rel err %.3g)\n", b, (double) got, want, err);
            ++fails;
        }
    }
    if (fails == 0) {
        printf("ok      graph MUL_MAT with MXFP8 weights matches the decoded reference (3 rows)\n");
    }
    ggml_free(ctx);
    return fails;
}

// Microbenchmark: single-row dot throughput for MXFP8 vs Q8_0 (same activations).
static void bench_dot(const ggml_type_traits_t & mx, const ggml_type_traits_t & q8) {
    const int n = 32*512;          // 16k values per row
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> w(n), a(n);
    for (auto & v : w) v = u(rng);
    for (auto & v : a) v = u(rng);

    std::vector<uint8_t> wx(ggml_row_size(GGML_TYPE_MXFP8, n));
    std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0,  n));
    std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_0,  n));
    mx.from_float_ref(w.data(), wx.data(), n);
    q8.from_float(w.data(), wq.data(), n);
    q8.from_float(a.data(), aq.data(), n);

    const int iters = 4000;
    float acc = 0.0f;
    auto run = [&](const ggml_type_traits_t & tr, const uint8_t * wt, size_t row_bytes, const char * name) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            float s = 0.0f;
            tr.vec_dot(n, &s, 0, wt, 0, aq.data(), 0, 1);
            acc += s;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ns = secs * 1e9 / iters;
        const double gbs = double(row_bytes) * iters / secs / 1e9;
        printf("      %-6s %8.1f ns/dot  %7.2f GB/s (n=%d)\n", name, ns, gbs, n);
    };
    run(mx, wx.data(), wx.size(), "mxfp8");
    // NOTE: the direct Q8_0 vec_dot entry in this fork delegates to iqk_mul_mat
    // and returns garbage when called standalone (see tests/CMakeLists.txt note
    // on test-quantize-fns), so it is not a valid baseline here. The production
    // path is compared below through a real MUL_MAT graph instead.
    printf("      (checksum %.9g)\n", (double) acc);

    // Production-path A/B: identical MUL_MAT graphs with MXFP8 vs Q8_0 weights.
    auto graph_ms = [&](ggml_type wtype, const char * name) {
        struct ggml_init_params gp = { /* .mem_size = */ 512u*1024*1024, /* .mem_buffer = */ nullptr, /* .no_alloc = */ false };
        struct ggml_context * ctx = ggml_init(gp);
        const int gn = 4096, rows = 256;
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, gn, rows);
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gn, 1);
        struct ggml_tensor * o = ggml_mul_mat(ctx, w, a);
        ggml_type_traits_t wt = ggml_internal_get_type_traits(wtype);
        std::vector<float> wf(gn);
        std::mt19937 r2(99);
        std::uniform_real_distribution<float> uu(-1.0f, 1.0f);
        std::vector<uint8_t> rowbuf(ggml_row_size(wtype, gn));
        for (int r = 0; r < rows; ++r) {
            for (auto & v : wf) v = uu(r2);
            wt.from_float_ref(wf.data(), rowbuf.data(), gn);
            std::memcpy((uint8_t *) w->data + r*rowbuf.size(), rowbuf.data(), rowbuf.size());
        }
        for (int i = 0; i < gn; ++i) ggml_set_f32_1d(a, i, uu(r2));
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);
        for (int i = 0; i < 3; ++i) ggml_graph_compute_with_ctx(ctx, gf, 8);
        const int iters = 60;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) ggml_graph_compute_with_ctx(ctx, gf, 8);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double ns_per_val = secs * 1e9 / (double(iters) * gn * rows);
        printf("      graph %-6s %8.1f us/mat  %6.3f ns/value  (value %.6g)\n",
               name, secs*1e6/iters, ns_per_val, (double) ggml_get_f32_1d(o, 0));
        ggml_free(ctx);
        return ns_per_val;
    };
    const double mx_ns = graph_ms(GGML_TYPE_MXFP8, "mxfp8");
    const double q8_ns = graph_ms(GGML_TYPE_Q8_0,  "q8_0");
    printf("      MXFP8/Q8_0 production-path cost ratio: %.2fx\n", mx_ns/q8_ns);
}

int main(int argc, char ** argv) {
    bool compute_enabled = false;
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--compute") == 0) {
            setenv("GGML_MXFP8_COMPUTE", "1", 1);
            compute_enabled = true;
        } else if (std::strcmp(arg, "--bench") == 0) {
            do_bench = true;
        } else {
            printf("error: unknown argument %s\n", arg);
            return 2;
        }
    }

    struct ggml_init_params params = { /* .mem_size = */ 1024, /* .mem_buffer = */ nullptr, /* .no_alloc = */ true };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_quantize_init(GGML_TYPE_MXFP8);
    ggml_quantize_init(GGML_TYPE_Q8_0);

    const ggml_type_traits_t mx = ggml_internal_get_type_traits(GGML_TYPE_MXFP8);
    const ggml_type_traits_t q8 = ggml_internal_get_type_traits(GGML_TYPE_Q8_0);

    check(mx.from_float != nullptr && mx.from_float_ref != nullptr, "mxfp8 has an encoder");
    check(mx.vec_dot != nullptr, "mxfp8 has a dot product");
    check(mx.vec_dot_type == GGML_TYPE_Q8_0, "mxfp8 dot product consumes Q8_0 activations");

    const int n = ggml_blck_size(GGML_TYPE_MXFP8) * 16;   // 512 values
    const size_t row = ggml_row_size(GGML_TYPE_MXFP8, n);

    std::vector<float> x(n);
    std::vector<uint8_t> xq(row);
    std::vector<float> xd(n);

    // 1. Round trip inside the format's contract: one E8M0 scale per 32-value
    //    block covers d*2^-9 .. d*448. E4M3 has a 3-bit mantissa, so a value
    //    needs |err| <= 2^-4*|x| while values far below d obey the absolute
    //    floor of half a subnormal step (2^-10*d); use 2^-9*d to be safe.
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> mag(0.0f, 1.0f);
        std::uniform_int_distribution<int> block_exp(-100, 100);
        std::uniform_int_distribution<int> spread(0, 20);
        std::uniform_int_distribution<int> sign(0, 1);
        std::vector<float> dblk(n/32);
        for (int b = 0; b < n/32; ++b) {
            const float amax = std::ldexp(1.0f, block_exp(rng));
            dblk[b] = amax;                    // floor(log2(2^k)) == k, so d == amax
            for (int j = 0; j < 32; ++j) {
                float v = mag(rng) * std::ldexp(amax, -spread(rng));
                if (sign(rng)) v = -v;
                x[b*32 + j] = v;
            }
        }
        mx.from_float_ref(x.data(), xq.data(), n);
        mx.to_float(xq.data(), xd.data(), n);

        double worst = 0.0;    // worst (err - 2^-4|x|) / (2^-9 d), must stay <= 1
        for (int i = 0; i < n; ++i) {
            const double d = dblk[i/32];
            const double err = std::fabs(double(xd[i]) - double(x[i]));
            const double budget = 0x1p-4 * std::fabs(double(x[i]));
            worst = std::max(worst, (err - budget) / (0x1p-9 * d));
        }
        check(worst <= 1.0, "round-trip within half-ulp + subnormal floor", worst, 1.0);
    }

    // 2. Scale/round-trip stability: with amax exactly on a binade and values
    //    well inside it, re-encoding a decoded block reproduces the block
    //    byte-for-byte (the chosen E8M0 scale cannot move).
    {
        std::mt19937 rng(4321);
        std::uniform_real_distribution<float> mag(0.5f, 1.0f);
        std::uniform_int_distribution<int> block_exp(-60, 60);
        std::uniform_int_distribution<int> spread(0, 5);
        std::uniform_int_distribution<int> sign(0, 1);
        for (int b = 0; b < n/32; ++b) {
            const float amax = std::ldexp(1.0f, block_exp(rng));
            for (int j = 0; j < 32; ++j) {
                float v = mag(rng) * std::ldexp(amax, -spread(rng));
                if (sign(rng)) v = -v;
                x[b*32 + j] = v;
            }
            x[b*32] = amax;                    // pin the block max so the scale is unambiguous
        }
        mx.from_float_ref(x.data(), xq.data(), n);
        mx.to_float(xq.data(), xd.data(), n);
        std::vector<uint8_t> xq2(row);
        mx.from_float_ref(xd.data(), xq2.data(), n);
        check(std::memcmp(xq.data(), xq2.data(), row) == 0, "re-encoding a decoded block is stable");
    }

    // 3. Compute exactness: vec_dot must match the double dot of the *decoded*
    //    weights and *dequantized* activations (no quantization error included).
    {
        std::mt19937 rng(9876);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::uniform_int_distribution<int> scale_exp(-30, 30);
        for (int i = 0; i < n; ++i) {
            x[i] = u(rng) * std::ldexp(1.0f, scale_exp(rng));
        }
        std::vector<float> a(n);
        for (int i = 0; i < n; ++i) {
            a[i] = u(rng) * 4.0f;
        }
        mx.from_float_ref(x.data(), xq.data(), n);
        mx.to_float(xq.data(), xd.data(), n);

        std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_0, n));
        std::vector<float> ad(n);
        q8.from_float(a.data(), aq.data(), n);
        q8.to_float(aq.data(), ad.data(), n);

        float got = 0.0f;
        mx.vec_dot(n, &got, 0, xq.data(), 0, aq.data(), 0, 1);
        const double want = dot_ref(xd, ad);
        const double err = std::fabs(double(got) - want) / (std::fabs(want) + 1e-30);
        check(err < 1e-5, "dot product matches decoded reference (relative)", err, 1e-5);
    }

    // 4. Multi-row / stride form: 4 rows with bx/by strides must equal 4 single
    //    calls. ggml-cpu uses this form when it batches rows.
    {
        const int nr = 4;
        std::mt19937 rng(555);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> xb(nr * n), ab(nr * n);
        for (auto & v : xb) v = u(rng) * std::ldexp(1.0f, std::uniform_int_distribution<int>(-10, 10)(rng));
        for (auto & v : ab) v = u(rng) * 2.0f;

        std::vector<uint8_t> xqb(nr * row), aqb(nr * ggml_row_size(GGML_TYPE_Q8_0, n));
        for (int r = 0; r < nr; ++r) {
            mx.from_float_ref(xb.data() + r*n, xqb.data() + r*row, n);
            q8.from_float(ab.data() + r*n, aqb.data() + r*ggml_row_size(GGML_TYPE_Q8_0, n), n);
        }
        std::vector<float> got(nr, 0.0f);
        mx.vec_dot(n, got.data(), 0, xqb.data(), row, aqb.data(), ggml_row_size(GGML_TYPE_Q8_0, n), nr);
        bool ok = true;
        double worst = 0.0;
        for (int r = 0; r < nr; ++r) {
            float single = 0.0f;
            mx.vec_dot(n, &single, 0, xqb.data() + r*row, 0, aqb.data() + r*ggml_row_size(GGML_TYPE_Q8_0, n), 0, 1);
            const double e = std::fabs(double(got[r]) - double(single));
            worst = std::max(worst, e);
            ok = ok && (e <= 1e-4f * (std::fabs(double(single)) + 1.0));
        }
        check(ok, "strided multi-row dot equals single-row dots", worst, 0.0);
    }

    // 5. NaN codes propagate (0x7F/0xFF are NaN in E4M3 fn; 0xFF scale is NaN).
    {
        std::vector<uint8_t> nan_code(row, 0);
        // block 0: scale 127 (1.0), all codes 0x7F (NaN)
        nan_code[0] = 127;
        std::memset(nan_code.data() + 1, 0x7F, ggml_blck_size(GGML_TYPE_MXFP8));
        std::vector<float> a(n, 1.0f), ad(n, 0.0f);
        std::vector<uint8_t> aq(ggml_row_size(GGML_TYPE_Q8_0, n));
        q8.from_float(a.data(), aq.data(), n);
        float got = 0.0f;
        mx.vec_dot(ggml_blck_size(GGML_TYPE_MXFP8), &got, 0, nan_code.data(), 0, aq.data(), 0, 1);
        check(std::isnan(got), "NaN e4m3 code propagates through the dot product");

        std::vector<uint8_t> nan_scale(row, 0);
        nan_scale[0] = 0xFF;                       // NaN E8M0 scale
        got = 0.0f;
        mx.vec_dot(ggml_blck_size(GGML_TYPE_MXFP8), &got, 0, nan_scale.data(), 0, aq.data(), 0, 1);
        check(std::isnan(got), "NaN e8m0 scale propagates through the dot product");
    }

    // 6. Boundary values, one per block (mixing extremes in one block would
    //    simply saturate/underflow, which is the format's defined behavior):
    //    largest finite (448), smallest subnormal (2^-9), negative zero,
    //    both extreme E8M0 scales, and documented underflow below 2^-126.
    {
        std::vector<float> b(n, 0.0f);
        b[0*32 + 0] = 448.0f;
        b[0*32 + 1] = -448.0f;
        b[0*32 + 2] = -0.0f;                 // zero block would drop the sign, this one keeps it
        b[1*32 + 0] = 0x1p-9f;               // smallest subnormal => d = 2^-10
        b[1*32 + 1] = -0x1p-9f;
        b[2*32 + 0] = 0x1p127f;              // largest scale => e = 254, d = 2^127
        b[3*32 + 0] = 0x1p-126f;             // smallest normal => e = 0, d = 2^-126
        b[4*32 + 0] = 0x1p-130f;             // below the type's range => 0
        b[4*32 + 1] = 464.0f;                // rounds down to 448
        mx.from_float_ref(b.data(), xq.data(), n);
        mx.to_float(xq.data(), xd.data(), n);
        check(xd[0*32 + 0] == 448.0f, "448 encodes to the largest finite code", xd[0*32 + 0], 448.0);
        check(xd[0*32 + 1] == -448.0f, "-448 encodes to the largest finite code", xd[0*32 + 1], -448.0);
        check(xd[0*32 + 2] == 0.0f && std::signbit(xd[0*32 + 2]), "negative zero is preserved", xd[0*32 + 2], 0.0);
        check(xd[1*32 + 0] == 0x1p-9f, "smallest subnormal survives", xd[1*32 + 0], 0x1p-9);
        check(xd[1*32 + 1] == -0x1p-9f, "negative smallest subnormal survives", xd[1*32 + 1], -0x1p-9);
        check(xd[2*32 + 0] == 0x1p127f, "largest scale (2^127) round-trips", xd[2*32 + 0], 0x1p127);
        check(xd[3*32 + 0] == 0x1p-126f, "smallest normal (2^-126) round-trips", xd[3*32 + 0], 0x1p-126);
        check(xd[4*32 + 0] == 0.0f, "2^-130 underflows to zero", xd[4*32 + 0], 0.0);
        check(xd[4*32 + 1] == 448.0f, "464 rounds down to 448", xd[4*32 + 1], 448.0);
    }

    if (do_bench) {
        bench_dot(mx, q8);
    }
    g_failed += graph_check(compute_enabled, mx, q8);

    ggml_free(ctx);

    printf("%s\n", g_failed ? "MXFP8 tests FAILED" : "MXFP8 tests passed");
    return g_failed ? 1 : 0;
}
