#pragma once

//
// Dual AVX2 / AVX-512 Runtime Dispatch Symbol Remapping
// Renames exported GEMM kernel and converter functions per-variant.
//

#if defined(IQK_AVX512_VARIANT)

#define iqk_set_kernels_kquants         iqk_set_kernels_kquants_avx512
#define iqk_convert_kquants_q8X_r8      iqk_convert_kquants_q8X_r8_avx512
#define iqk_gemm_q8kv_fa                iqk_gemm_q8kv_fa_avx512

#define iqk_set_kernels_iquants         iqk_set_kernels_iquants_avx512
#define iqk_convert_iquants_q80_r8      iqk_convert_iquants_q80_r8_avx512

#define iqk_set_kernels_iqk_quants      iqk_set_kernels_iqk_quants_avx512
#define iqk_convert_iqk_quants_q80_r8   iqk_convert_iqk_quants_q80_r8_avx512

#define iqk_set_kernels_ktquants        iqk_set_kernels_ktquants_avx512
#define iqk_dequantize_ktquants         iqk_dequantize_ktquants_avx512

#define iqk_set_kernels_legacy_quants   iqk_set_kernels_legacy_quants_avx512
#define iqk_convert_legacy_quants_q8_r8 iqk_convert_legacy_quants_q8_r8_avx512
#define iqk_gemm_legacy_fa              iqk_gemm_legacy_fa_avx512

#define iqk_set_kernels_1bit            iqk_set_kernels_1bit_avx512
#define iqk_convert_1bit_q80_r8         iqk_convert_1bit_q80_r8_avx512

#define iqk_set_kernels_float           iqk_set_kernels_float_avx512
#define iqk_gemm_default_floats         iqk_gemm_default_floats_avx512

#elif defined(IQK_AVX2_VARIANT)

#define iqk_set_kernels_kquants         iqk_set_kernels_kquants_avx2
#define iqk_convert_kquants_q8X_r8      iqk_convert_kquants_q8X_r8_avx2
#define iqk_gemm_q8kv_fa                iqk_gemm_q8kv_fa_avx2

#define iqk_set_kernels_iquants         iqk_set_kernels_iquants_avx2
#define iqk_convert_iquants_q80_r8      iqk_convert_iquants_q80_r8_avx2

#define iqk_set_kernels_iqk_quants      iqk_set_kernels_iqk_quants_avx2
#define iqk_convert_iqk_quants_q80_r8   iqk_convert_iqk_quants_q80_r8_avx2

#define iqk_set_kernels_ktquants        iqk_set_kernels_ktquants_avx2
#define iqk_dequantize_ktquants         iqk_dequantize_ktquants_avx2

#define iqk_set_kernels_legacy_quants   iqk_set_kernels_legacy_quants_avx2
#define iqk_convert_legacy_quants_q8_r8 iqk_convert_legacy_quants_q8_r8_avx2
#define iqk_gemm_legacy_fa              iqk_gemm_legacy_fa_avx2

#define iqk_set_kernels_1bit            iqk_set_kernels_1bit_avx2
#define iqk_convert_1bit_q80_r8         iqk_convert_1bit_q80_r8_avx2

#define iqk_set_kernels_float           iqk_set_kernels_float_avx2
#define iqk_gemm_default_floats         iqk_gemm_default_floats_avx2

#endif
