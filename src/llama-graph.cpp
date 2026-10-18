#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-cparams.h"
#include "llama-kv-zdd.h"

#include <cstdlib>
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
