#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-kv-zdd.h"
#include "paged-kv-cache.h"
#include "ggml-cuda.h"

#include <cstdlib>
#include <algorithm>
#include <vector>

bool llm_build_context::llama_graph_build_zdd_skip_mask(
        llama_context & lctx,
        const llama_kv_cache & kv,
        int32_t n_kv,
        std::vector<uint8_t> & out_skip_mask) {
    const char * env_skip = std::getenv("LLAMA_KV_ZDD_SKIP");
    const bool zdd_enabled = lctx.cparams.zdd_sparse_kv || (env_skip != nullptr && env_skip[0] != '\0' && env_skip[0] != '0');
    if (!zdd_enabled || n_kv <= 0) {
        out_skip_mask.clear();
        return false;
    }

    const uint32_t total_blocks = (uint32_t)((n_kv + llama::kv::zdd::KV_BLOCK_TOKENS - 1) / llama::kv::zdd::KV_BLOCK_TOKENS);
    if (total_blocks == 0) {
        out_skip_mask.clear();
        return false;
    }

    llama::kv::zdd::ZddKvManager local_mgr(std::max<uint32_t>(8192u, total_blocks));
    llama::kv::zdd::ZddKvManager & mgr = kv.zdd_mgr ? *kv.zdd_mgr : local_mgr;

    const uint32_t n_sink_blocks = 4;
    const uint32_t n_window_blocks = 32;
    const uint32_t root = mgr.build_active_context_blocks((uint32_t)n_kv, n_sink_blocks, n_window_blocks);

    std::vector<uint32_t> active_blocks;
    mgr.extract_active_blocks(root, active_blocks);

    out_skip_mask = llama::kv::zdd::ZddAttentionBlockSkipper::generate_skip_mask(total_blocks, active_blocks);
    if (!active_blocks.empty()) {
        (void)llama::kv::zdd::ZddAttentionBlockSkipper::is_block_active(active_blocks[0], active_blocks);
    }
    return !out_skip_mask.empty();
}

bool llm_build_context::llama_graph_stream_paged_kv(
        llama_context & lctx,
        const llama_kv_cache & kv,
        int32_t n_kv,
        int il) {
    if (!lctx.paged_kv || n_kv <= 0) {
        return false;
    }

    const auto & cfg = lctx.paged_kv->config();
    const uint32_t block_size = cfg.block_size_tokens;
    if (block_size == 0) return false;

    const uint32_t needed_blocks = (uint32_t)((n_kv + block_size - 1) / block_size);
    const size_t block_bytes = (size_t)block_size * cfg.n_embd_k;

    const uint32_t stream_blocks = std::min<uint32_t>(needed_blocks, (uint32_t)lctx.paged_kv->active_block_count());
    if (stream_blocks > 0 && kv.k_l.size() > (size_t)il && kv.k_l[il] != nullptr) {
        void * dev_k = kv.k_l[il]->data;
        void * dev_v = (kv.v_l.size() > (size_t)il && kv.v_l[il]) ? kv.v_l[il]->data : nullptr;
        for (uint32_t b = 0; b < stream_blocks; ++b) {
            uint8_t * host_k = lctx.paged_kv->get_block_k(b);
            uint8_t * host_v = lctx.paged_kv->get_block_v(b);
            if (host_k && dev_k) {
                ggml_backend_cuda_stream_paged_kv_blocks(0, host_k, host_v,
                    (uint8_t *)dev_k + b * block_bytes,
                    dev_v ? (uint8_t *)dev_v + b * block_bytes : nullptr,
                    block_bytes, 1);
            }
        }
    }
    return true;
}

