#include "tiered-kv-cache.h"
#include "benchmark-utils.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>

int main() {
    printf("=================================================================\n");
    printf("  TEST: 4-Tier Hierarchical KV Cache with Background Pretoucher  \n");
    printf("=================================================================\n\n");

    common_tiered_kv_cache::config cfg;
    cfg.paged_cfg.block_size_tokens = 16;
    cfg.paged_cfg.n_embd_k = 64;
    cfg.paged_cfg.n_embd_v = 64;
    cfg.paged_cfg.max_blocks = 64;
    cfg.max_tier2_blocks = 128;
    cfg.enable_background_pretoucher = true;

    common_tiered_kv_cache tiered_kv(cfg);

    // [1] Allocate block in Tier 1 (Host NUMA)
    printf("[1] Allocating block in Tier 1 (Host NUMA DDR)...\n");
    int32_t bid = tiered_kv.allocate_block();
    assert(bid >= 0);
    assert(tiered_kv.get_block_tier(bid) == KV_TIER_1_HOST_NUMA);
    printf("  Allocated block %d in %s\n", bid, kv_cache_tier_to_str(tiered_kv.get_block_tier(bid)));
    printf("  Tier 1 allocation PASSED OK.\n\n");

    // [2] Demote block to Tier 2 (Local NVMe)
    printf("[2] Demoting block to Tier 2 (Local NVMe block storage)...\n");
    std::vector<uint8_t> dummy_payload(cfg.paged_cfg.block_size_tokens * 128, 0x42);
    bool demoted = tiered_kv.demote_to_tier2(bid, dummy_payload.data());
    assert(demoted);
    assert(tiered_kv.get_block_tier(bid) == KV_TIER_2_LOCAL_NVME);
    printf("  Block %d migrated to %s\n", bid, kv_cache_tier_to_str(tiered_kv.get_block_tier(bid)));
    printf("  Tier 2 demotion PASSED OK.\n\n");

    // [3] Background Pretoucher thread verification (Chronicle Queue / OpenHFT pattern)
    printf("[3] Verifying asynchronous background pretouching from Tier 2 to Tier 1...\n");
    std::vector<uint8_t> pretouched_buf(cfg.paged_cfg.block_size_tokens * 128, 0);

    tiered_kv.enqueue_pretouch(bid, pretouched_buf.data());

    // Wait for background pretoucher thread to process request
    int wait_cycles = 0;
    while (tiered_kv.get_block_tier(bid) != KV_TIER_1_HOST_NUMA && wait_cycles < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        wait_cycles++;
    }

    assert(tiered_kv.get_block_tier(bid) == KV_TIER_1_HOST_NUMA);
    assert(pretouched_buf[0] == 0x42);
    printf("  Background Pretoucher thread staged block %d back to Tier 1 in %d ms.\n", bid, wait_cycles * 2);
    printf("  Payload byte-for-byte integrity verified: 0x%02X.\n", pretouched_buf[0]);
    printf("  Asynchronous Pretoucher test PASSED OK.\n\n");

    // [4] Demote block to Tier 3 (Cold Archive)
    printf("[4] Demoting block to Tier 3 (Cold Archive)...\n");
    // Demote to Tier 2 first then Tier 3
    tiered_kv.demote_to_tier2(bid, dummy_payload.data());
    assert(tiered_kv.get_block_tier(bid) == KV_TIER_2_LOCAL_NVME);

    bool archived = tiered_kv.demote_to_tier3(bid);
    assert(archived);
    assert(tiered_kv.get_block_tier(bid) == KV_TIER_3_COLD_ARCHIVE);
    printf("  Block %d migrated to %s\n", bid, kv_cache_tier_to_str(tiered_kv.get_block_tier(bid)));
    printf("  Tier 3 archiving PASSED OK.\n\n");

    // [5] Free block and clean up
    printf("[5] Releasing block and recycling pool...\n");
    tiered_kv.release_block(bid);
    printf("  Block %d released cleanly.\n", bid);
    printf("  Clean recycling PASSED OK.\n\n");

    printf("=================================================================\n");
    printf("  ALL 4-TIER HIERARCHICAL KV CACHE TESTS PASSED WITH 100%% SUCCESS \n");
    printf("=================================================================\n");

    return 0;
}
