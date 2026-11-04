// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0005f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_1BITS = 0.0200f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0100f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.06f;
constexpr float MAX_DOT_PRODUCT_ERROR_EXTREME_LOWBIT = 0.20f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}
static void generate_bitnet_data(size_t n, float * dst) {
    std::mt19937 rndm(1234);
    for (size_t i = 0; i < n; i++) {
        auto r = rndm();
        dst[i] = r > std::mt19937::max()/2 ? 0.f : r < std::mt19937::max()/4 ? -1.f : 1.f;
    }
}


// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(ggml_type_traits_t & qfns, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns.from_float(test_data, tmp_q.data(), test_size);
    qfns.to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(ggml_type_traits_t & qfns, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    qfns.from_float(test_data, tmp_q.data(), test_size);
    qfns.to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns.from_float_ref(test_data, tmp_q.data(), test_size);
    qfns.to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(
    ggml_type type, ggml_type_traits_t & qfns, size_t test_size, const float * test_data1, const float *test_data2
) {
    if (!qfns.vec_dot) {
        return 0.0f;
    }

    enum ggml_type vdot_type = qfns.vec_dot_type;
    // Map interleaved matrix types back to standard scalar vector dot types
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q1_0_G128:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_MXFP8:
            vdot_type = GGML_TYPE_Q8_0;
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_1:
            vdot_type = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
            vdot_type = GGML_TYPE_Q8_K;
            break;
        default:
            break;
    }

    const size_t bx = ggml_row_size(type, test_size);
    const size_t by = ggml_row_size(vdot_type, test_size);
    std::vector<uint8_t> tmp_q1(bx);
    std::vector<uint8_t> tmp_q2(by);

    auto vdot = ggml_internal_get_type_traits(vdot_type);

    qfns.from_float_ref(test_data1, tmp_q1.data(), test_size);
    vdot.from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns.vec_dot(test_size, &result, 0, tmp_q1.data(), bx, tmp_q2.data(), by, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    const size_t test_size = 32 * 128;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);
    std::vector<float> test_data_bitnet(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());
    generate_bitnet_data(test_data_bitnet.size(), test_data_bitnet.data());

    // Initialize GGML, ensures float conversion tables are initialized
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    int num_failed = 0;
    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        ggml_type_traits_t qfns = ggml_internal_get_type_traits(type);

        // deprecated - skip
        if (qfns.blck_size == 0) {
            continue;
        }

        const char * tname = ggml_type_name(type);
        if (type == GGML_TYPE_Q1_0_G128_R8 || type == GGML_TYPE_Q8_KR8 || type == GGML_TYPE_Q8_0_R8 ||
            type == GGML_TYPE_MXFP4_R8 || type == GGML_TYPE_Q4_0_R8 || type == GGML_TYPE_Q5_0_R4 ||
            type == GGML_TYPE_Q6_0_R4 || type == GGML_TYPE_IQ4_NL_R4 || type == GGML_TYPE_IQ4_KS_R4 ||
            type == GGML_TYPE_IQ5_KS_R4 || strstr(tname, "_r") != nullptr || strstr(tname, "_R") != nullptr) {
            continue;
        }

        auto test_data_quantize = test_data.data();
        auto test_data_vecdot   = test_data2.data();
        const ggml_type ei = (ggml_type)i;
        if (ei == GGML_TYPE_IQ1_BN || ei == GGML_TYPE_IQ2_BN) {
            test_data_quantize = test_data_bitnet.data();
            test_data_vecdot   = test_data_bitnet.data();
            //printf("Skipping %s because test data does not satisfy Bitnet requirements\n", ggml_type_name(ei));
            //continue;
        }

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns.from_float && qfns.to_float) {
            const float total_error = total_quantization_error(qfns, test_size, test_data_quantize);
            float max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR;
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_K:
                case GGML_TYPE_IQ2_KS:
                case GGML_TYPE_PQ2_0:
                case GGML_TYPE_IQ2_KT:
                case GGML_TYPE_IQ2_KL:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_2BITS;
                    break;
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ3_K:
                case GGML_TYPE_IQ3_KS:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_3BITS;
                    break;
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_KT:
                case GGML_TYPE_IQ4_KT:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS;
                    break;
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_IQ1_BN:
                case GGML_TYPE_IQ2_BN:
                case GGML_TYPE_Q1_0:
                case GGML_TYPE_Q1_0_G128:
                case GGML_TYPE_PTQ1_0:
                case GGML_TYPE_IQ1_KT:
                    max_quantization_error = MAX_QUANTIZATION_TOTAL_ERROR_1BITS;
                    break;
                default:
                    break;
            }
            failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, test_size, test_data_quantize);
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(type, qfns, test_size, test_data.data(), test_data_vecdot);
            float max_allowed_error = MAX_DOT_PRODUCT_ERROR;
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_IQ4_KSS:
                case GGML_TYPE_IQ2_KL:
                    max_allowed_error = MAX_DOT_PRODUCT_ERROR_LOWBIT;
                    break;
                case GGML_TYPE_PQ2_0:
                case GGML_TYPE_PTQ1_0:
                case GGML_TYPE_IQ5_KS:
                    max_allowed_error = MAX_DOT_PRODUCT_ERROR_EXTREME_LOWBIT;
                    break;
                default:
                    break;
            }
            failed = !std::isnan(vec_dot_error) && !std::isinf(vec_dot_error) && !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    ggml_free(ctx);

    return num_failed > 0;
}
