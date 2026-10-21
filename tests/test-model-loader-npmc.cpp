#include "llama.h"
#include "llama-model.h"
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

static void test_npmc_superblock_loader_view() {
    printf("Testing GGUF-X / NPMC v3 Superblock2MiBView model loader integration...\n");

    const char * fname = "/tmp/test_npmc_superblock_loader.gguf";

    auto sb = std::make_unique<llama::npmc::Superblock2MiB>();
    const uint32_t num_experts = 256u;
    const uint32_t words_per_expert = 4u;
    const std::vector<uint32_t> active_ids = {2u, 19u, 64u, 255u};
    std::vector<std::vector<llama::npmc::TernaryBitplane64>> weights(
        active_ids.size(),
        std::vector<llama::npmc::TernaryBitplane64>(words_per_expert, {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL})
    );
    assert(llama::npmc::pack_superblock_2mib(*sb, num_experts, words_per_expert, active_ids, weights));
    assert(sb->view().valid());

    struct gguf_context * ctx_gguf = gguf_init_empty();
    assert(ctx_gguf != nullptr);
    gguf_set_val_str(ctx_gguf, "general.architecture", "llama");
    gguf_set_val_u32(ctx_gguf, "npmc.superblock.blk.0.ffn_exps.superblock", llama::npmc::LEANGRPH_V3_VERSION);

    struct ggml_init_params params_mem = {
        /*.mem_size   =*/ 1024 * 1024 * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_tensor = ggml_init(params_mem);
    assert(ctx_tensor != nullptr);

    struct ggml_tensor * sb_tensor = ggml_new_tensor_1d(ctx_tensor, GGML_TYPE_I8, llama::npmc::SUPERBLOCK_2MIB_BYTES);
    ggml_set_name(sb_tensor, "blk.0.ffn_exps.superblock");
    std::memcpy(sb_tensor->data, sb->data, llama::npmc::SUPERBLOCK_2MIB_BYTES);
    gguf_add_tensor(ctx_gguf, sb_tensor);

    // Ensure GGUF data section starts on a 128-byte boundary for zero-copy mmap Superblock2MiBView
    std::string pad_val = "align";
    gguf_set_val_str(ctx_gguf, "npmc.pad", pad_val.c_str());
    while ((gguf_get_meta_size(ctx_gguf) & 127u) != 0u) {
        pad_val.append(32, 'x');
        gguf_set_val_str(ctx_gguf, "npmc.pad", pad_val.c_str());
    }

    gguf_write_to_file(ctx_gguf, fname, false);
    gguf_free(ctx_gguf);
    ggml_free(ctx_tensor);

    llama_backend_init();
    {
        llama_model_loader ml(fname,
                              /*ncmoe=*/0,
                              /*use_mmap=*/true,
                              /*check_tensors=*/false,
                              /*repack_tensors=*/false,
                              /*use_thp=*/false,
                              /*merge_qkv=*/false,
                              /*merge_up_gate_exps=*/false,
                              /*defer_experts=*/false,
                              /*param_overrides_p=*/nullptr,
                              /*param_tensor_buft_overrides_p=*/nullptr);

        assert(ml.has_npmc_superblock("blk.0.ffn_exps.superblock"));
        assert(ml.get_npmc_superblock_version("blk.0.ffn_exps.superblock") == llama::npmc::LEANGRPH_V3_VERSION);

        ml.init_mappings(/*prefetch=*/false);
        llama::npmc::Superblock2MiBView view = ml.get_npmc_superblock_view("blk.0.ffn_exps.superblock");
        assert(view.valid());
        assert(view.header()->num_experts == num_experts);
        assert(view.header()->active_experts == active_ids.size());
        assert(view.veb_mask()->find_min() == 2);
        assert(view.veb_mask()->find_max() == 255);
        for (size_t r = 0; r < active_ids.size(); ++r) {
            assert(view.is_expert_active(active_ids[r]));
            assert(view.active_expert_rank(active_ids[r]) == static_cast<int32_t>(r));
            const auto * bp = view.expert_bitplanes(active_ids[r]);
            assert(bp != nullptr);
            assert(bp[0].pos_mask == 0x5555555555555555ULL);
            assert(bp[0].neg_mask == 0xAAAAAAAAAAAAAAAAULL);
            assert(view.lookup_expert_offset(active_ids[r]) > 0ULL);
        }
        assert(!view.is_expert_active(0u));
        assert(view.lookup_expert_offset(0u) == 0ULL);

        // Verify export_npmc_superblocks and live llama_graph_dispatch_npmc_superblock tensor dispatch
        std::unordered_map<std::string, uint32_t> exported_sb;
        ml.export_npmc_superblocks(exported_sb);
        assert(exported_sb.count("blk.0.ffn_exps.superblock") == 1u);
        assert(exported_sb["blk.0.ffn_exps.superblock"] == llama::npmc::LEANGRPH_V3_VERSION);

        const auto * w_sb = ml.get_weight("blk.0.ffn_exps.superblock");
        assert(w_sb != nullptr && w_sb->tensor != nullptr);
        w_sb->tensor->data = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(view.header()));

        std::vector<llama::npmc::TernaryBitplane64> act_planes(
            words_per_expert,
            {0x5555555555555555ULL, 0x0ULL}
        );
        int32_t dispatched_dot = 0;
        uint64_t dispatched_off = 0ULL;
        assert(llama_graph_dispatch_npmc_superblock(
            w_sb->tensor,
            active_ids[1],
            act_planes.data(),
            words_per_expert,
            &dispatched_dot,
            &dispatched_off
        ));
        assert(dispatched_off == view.lookup_expert_offset(active_ids[1]));
        assert(dispatched_dot == view.gemv_popcnt(active_ids[1], act_planes.data(), words_per_expert));
        assert(!llama_graph_dispatch_npmc_superblock(
            w_sb->tensor,
            0u,
            act_planes.data(),
            words_per_expert,
            &dispatched_dot,
            &dispatched_off
        ));
        printf("  [PASS] Zero-copy Superblock2MiBView mapped and live-dispatched via llama_graph_dispatch_npmc_superblock\n");
    }
    llama_backend_free();
    unlink(fname);
    printf("test_npmc_superblock_loader_view: ALL CHECKS PASSED\n\n");
}

int main() {
    test_unpadded_dim0_loading();
    test_npmc_superblock_loader_view();
    printf("test-model-loader-npmc: OK\n");
    return 0;
}
