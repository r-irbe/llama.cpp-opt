#include "llama.h"
#include "llama-model-loader.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <unistd.h>

static void test_unpadded_dim0_loading() {
    printf("Testing GGUF-X / NPMC unpadded dimension loading...\n");

    const char * fname = "/tmp/test_npmc_unpadded_loader.gguf";

    // Step 1: Create a synthetic GGUF model with padded tensor and npmc.unpadded_dim0 metadata
    struct gguf_context * ctx_gguf = gguf_init_empty();
    assert(ctx_gguf != nullptr);

    gguf_set_val_str(ctx_gguf, "general.architecture", "llama");
    gguf_set_val_u32(ctx_gguf, "llama.block_count", 1);
    gguf_set_val_u32(ctx_gguf, "llama.context_length", 512);
    gguf_set_val_u32(ctx_gguf, "llama.embedding_length", 16384);
    gguf_set_val_u32(ctx_gguf, "llama.feed_forward_length", 16384);
    gguf_set_val_u32(ctx_gguf, "llama.attention.head_count", 16);
    gguf_set_val_u32(ctx_gguf, "llama.attention.head_count_kv", 16);
    gguf_set_val_str(ctx_gguf, "tokenizer.ggml.model", "llama");

    // Register GGUF-X unpadded dimension metadata: logical dim0 = 16384, physical = 16400
    const int64_t logical_dim0  = 16384;
    const int64_t physical_dim0 = 16400;
    const int64_t rows          = 16;
    gguf_set_val_u64(ctx_gguf, "npmc.unpadded_dim0.token_embd.weight", (uint64_t) logical_dim0);

    // Create tensor with physical padded dimensions
    struct ggml_init_params params_mem = {
        /*.mem_size   =*/ 1024 * 1024 * 32,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_tensor = ggml_init(params_mem);
    assert(ctx_tensor != nullptr);

    struct ggml_tensor * dummy_tensor = ggml_new_tensor_2d(ctx_tensor, GGML_TYPE_F32, physical_dim0, rows);
    ggml_set_name(dummy_tensor, "token_embd.weight");
    memset(dummy_tensor->data, 0, ggml_nbytes(dummy_tensor));

    gguf_add_tensor(ctx_gguf, dummy_tensor);

    // Write GGUF model to disk
    gguf_write_to_file(ctx_gguf, fname, false);
    gguf_free(ctx_gguf);
    ggml_free(ctx_tensor);

    printf("  synthetic GGUF written to %s with physical dim0=%" PRId64 ", logical dim0=%" PRId64 "\n",
           fname, physical_dim0, logical_dim0);

    // Step 2: Load model using llama_model_loader
    llama_backend_init();

    {
        llama_model_loader ml(fname,
                              /*ncmoe=*/0,
                              /*use_mmap=*/false,
                              /*check_tensors=*/false,
                              /*repack_tensors=*/false,
                              /*use_thp=*/false,
                              /*merge_qkv=*/false,
                              /*merge_up_gate_exps=*/false,
                              /*defer_experts=*/false,
                              /*param_overrides_p=*/nullptr,
                              /*param_tensor_buft_overrides_p=*/nullptr);

        // Verification 1: Probe unpadded metadata registration
        assert(ml.has_unpadded_dim0("token_embd.weight") == true);
        assert(ml.get_unpadded_dim0("token_embd.weight") == logical_dim0);
        printf("  [PASS] unpadded dim0 registered correctly: %" PRId64 "\n", ml.get_unpadded_dim0("token_embd.weight"));

        // Verification 2: check_tensor_dims accepts logical dimensions
        const struct ggml_tensor * meta_tensor = ml.check_tensor_dims("token_embd.weight", {logical_dim0, rows}, true);
        assert(meta_tensor != nullptr);
        assert(meta_tensor->ne[0] == physical_dim0);
        assert(meta_tensor->nb[1] == physical_dim0 * sizeof(float));
        printf("  [PASS] check_tensor_dims verified logical shape {% " PRId64 ", %" PRId64 "} against physical {% " PRId64 ", %" PRId64 "}\n",
               logical_dim0, rows, meta_tensor->ne[0], meta_tensor->ne[1]);

        // Verification 3: create_tensor sets logical ne[0] while preserving padded row stride nb[1]
        struct ggml_init_params ctx_create_params = {
            /*.mem_size   =*/ 1024 * 1024 * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx_model = ggml_init(ctx_create_params);
        assert(ctx_model != nullptr);

        struct ggml_tensor * loaded_tensor = ml.create_tensor(ctx_model, "token_embd.weight", {logical_dim0, rows});
        assert(loaded_tensor != nullptr);

        assert(loaded_tensor->ne[0] == logical_dim0);
        assert(loaded_tensor->ne[1] == rows);
        assert(loaded_tensor->nb[0] == sizeof(float));
        assert(loaded_tensor->nb[1] == physical_dim0 * sizeof(float));
        assert(loaded_tensor->nb[2] == loaded_tensor->nb[1] * rows);

        printf("  [PASS] loaded_tensor has logical ne0=%" PRId64 " and padded row stride nb1=%zu bytes\n",
               loaded_tensor->ne[0], loaded_tensor->nb[1]);

        // Verification 4: Mismatched shape is correctly rejected
        bool threw_mismatch = false;
        try {
            ml.check_tensor_dims("token_embd.weight", {logical_dim0, rows + 10}, true);
        } catch (const std::runtime_error & e) {
            threw_mismatch = true;
        }
        assert(threw_mismatch);
        printf("  [PASS] mismatched shape correctly rejected\n");

        ggml_free(ctx_model);
    }

    llama_backend_free();
    unlink(fname);
    printf("test_unpadded_dim0_loading: ALL CHECKS PASSED\n\n");
}

int main() {
    test_unpadded_dim0_loading();
    printf("test-model-loader-npmc: OK\n");
    return 0;
}
