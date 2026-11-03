#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = kvalues_iq4nl[vui & 0xF];
    v.y = kvalues_iq4nl[vui >>  4];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x = v.x * d;
    v.y = v.y * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q6_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const dfloat d = x[ib].d;

    const uint8_t h = x[ib].qh[iqs%8] >> 4*(iqs/8);
    v.x = ((x[ib].qs[iqs] & 0xf) | ((h & 0x3) << 4));
    v.y = ((x[ib].qs[iqs] >>  4) | ((h & 0xc) << 2));

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {32.0f, 32.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 32.0f) * d;
    v.y = (v.y - 32.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

// E4M3 (fn) decode, exact in fp32; 0x7F/0xFF are NaN. Used by MXFP8 (gather & compute,
// mainline PR #30817).
static __device__ __forceinline__ float ggml_cuda_e4m3_to_fp32(const uint8_t x) {
    const uint32_t sign = (uint32_t) (x & 0x80) << 24;
    const uint32_t exp  = (x >> 3) & 0xF;
    const uint32_t man  = x & 0x7;
    if ((x & 0x7F) == 0x7F) {
        return __int_as_float(sign | 0x7FC00000u);
    }
    if (exp == 0) {
        return __int_as_float(__float_as_int((float) man * 0x1p-9f) | sign);
    }
    return __int_as_float(sign | ((exp + 120u) << 23) | (man << 20));
}

static __device__ __forceinline__ void dequantize_mxfp8(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_mxfp8 * x = (const block_mxfp8 *) vx;

    const uint8_t e = x[ib].e;
    const float d = e == 0xFF ? __int_as_float(0x7FC00000) : (e == 0 ? 0x1p-127f : __int_as_float((uint32_t) e << 23));

    v.x = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 0]) * d;
    v.y = ggml_cuda_e4m3_to_fp32(x[ib].qs[iqs + 1]) * d;
}
