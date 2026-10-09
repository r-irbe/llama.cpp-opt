//
// Copyright (C) 2025 Iwan Kawrakow / ik_llama.cpp
// MIT license
// SPDX-License-Identifier: MIT
//
// Dual AVX2 / AVX-512 Runtime Dispatcher for IQK GEMM kernels
//

#include "iqk_config.h"
#include "iqk_common.h"
#include "iqk_cpu_ops.h"
#include "iqk_gemm_kquants.h"
#include "iqk_gemm_iquants.h"
#include "iqk_gemm_iqk_quants.h"
#include "iqk_gemm_ktquants.h"
#include "iqk_gemm_legacy_quants.h"
#include "iqk_gemm_1bit.h"
#include "iqk_gemm_floats.h"

// ============================================================================
// Forward declarations for AVX2 variant
// ============================================================================
bool iqk_set_kernels_kquants_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_kquants_q8X_r8_avx2(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);
void iqk_gemm_q8kv_fa_avx2(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_set_kernels_iquants_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_iquants_q80_r8_avx2(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_iqk_quants_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_iqk_quants_q80_r8_avx2(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_ktquants_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_dequantize_ktquants_avx2(int type, int n, const void * vx, size_t bx, void * vy, size_t stride_y, int nrc_x);

bool iqk_set_kernels_legacy_quants_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_legacy_quants_q8_r8_avx2(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);
void iqk_gemm_legacy_fa_avx2(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_set_kernels_1bit_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& funcs, mul_mat_t& func16);
bool iqk_convert_1bit_q80_r8_avx2(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_float_avx2(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels);
void iqk_gemm_default_floats_avx2(int D, int nq, const char * vx, size_t bx, DataInfo& info, int k_step);

// ============================================================================
// Forward declarations for AVX-512 variant
// ============================================================================
#if defined(GGML_USE_IQK_AVX512)
bool iqk_set_kernels_kquants_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_kquants_q8X_r8_avx512(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);
void iqk_gemm_q8kv_fa_avx512(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_set_kernels_iquants_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_iquants_q80_r8_avx512(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_iqk_quants_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_iqk_quants_q80_r8_avx512(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_ktquants_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_dequantize_ktquants_avx512(int type, int n, const void * vx, size_t bx, void * vy, size_t stride_y, int nrc_x);

bool iqk_set_kernels_legacy_quants_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);
bool iqk_convert_legacy_quants_q8_r8_avx512(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);
void iqk_gemm_legacy_fa_avx512(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

bool iqk_set_kernels_1bit_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& funcs, mul_mat_t& func16);
bool iqk_convert_1bit_q80_r8_avx512(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

bool iqk_set_kernels_float_avx512(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels);
void iqk_gemm_default_floats_avx512(int D, int nq, const char * vx, size_t bx, DataInfo& info, int k_step);
#endif

// ============================================================================
// Canonical Exported Dispatchers
// ============================================================================

// 1. K-quants
bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_kquants_avx512(ne00, typeA, typeB, kernels, func16);
    }
#endif
    return iqk_set_kernels_kquants_avx2(ne00, typeA, typeB, kernels, func16);
}

bool iqk_convert_kquants_q8X_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_convert_kquants_q8X_r8_avx512(type, n, vx, bx, vy, nrc_x);
    }
#endif
    return iqk_convert_kquants_q8X_r8_avx2(type, n, vx, bx, vy, nrc_x);
}

void iqk_gemm_q8kv_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        iqk_gemm_q8kv_fa_avx512(D, nq, type_k, k, stride_k, info, k_step);
        return;
    }
#endif
    iqk_gemm_q8kv_fa_avx2(D, nq, type_k, k, stride_k, info, k_step);
}

// 2. I-quants
bool iqk_set_kernels_iquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_iquants_avx512(ne00, typeA, typeB, kernels, func16);
    }
#endif
    return iqk_set_kernels_iquants_avx2(ne00, typeA, typeB, kernels, func16);
}

bool iqk_convert_iquants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_convert_iquants_q80_r8_avx512(type, n, vx, bx, vy, nrc_x);
    }
#endif
    return iqk_convert_iquants_q80_r8_avx2(type, n, vx, bx, vy, nrc_x);
}

// 3. IQK-quants
bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_iqk_quants_avx512(ne00, typeA, typeB, kernels, func16);
    }
#endif
    return iqk_set_kernels_iqk_quants_avx2(ne00, typeA, typeB, kernels, func16);
}

bool iqk_convert_iqk_quants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_convert_iqk_quants_q80_r8_avx512(type, n, vx, bx, vy, nrc_x);
    }
#endif
    return iqk_convert_iqk_quants_q80_r8_avx2(type, n, vx, bx, vy, nrc_x);
}

// 4. KT-quants
bool iqk_set_kernels_ktquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_ktquants_avx512(ne00, typeA, typeB, kernels, func16);
    }
#endif
    return iqk_set_kernels_ktquants_avx2(ne00, typeA, typeB, kernels, func16);
}

bool iqk_dequantize_ktquants(int type, int n, const void * vx, size_t bx, void * vy, size_t stride_y, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_dequantize_ktquants_avx512(type, n, vx, bx, vy, stride_y, nrc_x);
    }
#endif
    return iqk_dequantize_ktquants_avx2(type, n, vx, bx, vy, stride_y, nrc_x);
}

// 5. Legacy quants
bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_legacy_quants_avx512(ne00, typeA, typeB, kernels, func16);
    }
#endif
    return iqk_set_kernels_legacy_quants_avx2(ne00, typeA, typeB, kernels, func16);
}

bool iqk_convert_legacy_quants_q8_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_convert_legacy_quants_q8_r8_avx512(type, n, vx, bx, vy, nrc_x);
    }
#endif
    return iqk_convert_legacy_quants_q8_r8_avx2(type, n, vx, bx, vy, nrc_x);
}

void iqk_gemm_legacy_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        iqk_gemm_legacy_fa_avx512(D, nq, type_k, k, stride_k, info, k_step);
        return;
    }
#endif
    iqk_gemm_legacy_fa_avx2(D, nq, type_k, k, stride_k, info, k_step);
}

// 6. 1-bit
bool iqk_set_kernels_1bit(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& funcs, mul_mat_t& func16) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_1bit_avx512(ne00, typeA, typeB, funcs, func16);
    }
#endif
    return iqk_set_kernels_1bit_avx2(ne00, typeA, typeB, funcs, func16);
}

bool iqk_convert_1bit_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_convert_1bit_q80_r8_avx512(type, n, vx, bx, vy, nrc_x);
    }
#endif
    return iqk_convert_1bit_q80_r8_avx2(type, n, vx, bx, vy, nrc_x);
}

// 7. Floats
bool iqk_set_kernels_float(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        return iqk_set_kernels_float_avx512(ne00, typeA, typeB, kernels);
    }
#endif
    return iqk_set_kernels_float_avx2(ne00, typeA, typeB, kernels);
}

void iqk_gemm_default_floats(int D, int nq, const char * vx, size_t bx, DataInfo& info, int k_step) {
#if defined(GGML_USE_IQK_AVX512)
    if (iqk_cpu_has_avx512_vnni()) {
        iqk_gemm_default_floats_avx512(D, nq, vx, bx, info, k_step);
        return;
    }
#endif
    iqk_gemm_default_floats_avx2(D, nq, vx, bx, info, k_step);
}
