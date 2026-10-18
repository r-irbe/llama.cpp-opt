#include "paged-kv-cache.h"
#include "benchmark-utils.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <cstring>

int main() {
    printf("=================================================================\n");
    printf("  TEST: Host-RAM PagedAttention Block Manager & PCIe DMA Staging \n");
    printf("=================================================================\n\n");

    // [1] Serialized hardware timestamping & TSC calibration (from benchmark-utils.h)
    printf("[1] Verifying serialized __rdtscp timestamping and core telemetry...\n");
    double ghz = benchmark::calibrate_tsc_freq_ghz();
    printf("  Calibrated TSC frequency: %.3f GHz\n", ghz);
    assert(ghz > 0.5 && ghz < 10.0);

    uint32_t start_core = 0;
    uint32_t end_core = 0;
    uint64_t t0 = benchmark::rdtsc_start();
    benchmark::DoNotOptimize(t0);
    uint64_t t1 = benchmark::rdtsc_end(&end_core);
    printf("  Serialized cycle delta: %lu cycles (core: %u)\n", t1 - t0, end_core);
    assert(t1 >= t0);
    printf("  Timestamping check PASSED OK.\n\n");

    // [2] Paged KV cache allocation
    printf("[2] Initializing Paged KV block manager with HugeTLB pre-faulting...\n");
    paged_kv_config cfg;
    cfg.block_size_tokens = 16;
    cfg.n_embd_k = 64; // 64 bytes per token for test
    cfg.n_embd_v = 64;
    cfg.max_blocks = 128;
    cfg.use_hugetlb = true;

    common_paged_kv_cache kv(cfg);
    assert(kv.free_block_count() == 128);
    assert(kv.active_block_count() == 0);
    printf("  Initialized %u blocks (128B aligned, 100%% free).\n", cfg.max_blocks);
    printf("  Pool init check PASSED OK.\n\n");

    // [3] Token appending across multiple physical blocks
    printf("[3] Appending 40 tokens (spans across 3 blocks of 16 tokens)...\n");
    common_paged_kv_cache::sequence_table seq1;
    seq1.seq_id = 1;

    std::vector<uint8_t> dummy_k(cfg.n_embd_k, 0xAA);
    std::vector<uint8_t> dummy_v(cfg.n_embd_v, 0xBB);

    for (int i = 0; i < 40; ++i) {
        dummy_k[0] = (uint8_t)(i & 0xFF);
        dummy_v[0] = (uint8_t)((i + 1) & 0xFF);
        bool ok = kv.append_token(seq1, dummy_k.data(), dummy_v.data());
        assert(ok);
    }

    assert(seq1.total_tokens == 40);
    assert(seq1.block_ids.size() == 3); // ceil(40 / 16) = 3
    assert(kv.active_block_count() == 3);
    printf("  Successfully allocated 3 blocks for 40 tokens.\n");
    printf("  Token append check PASSED OK.\n\n");

    // [4] Zero-copy fork (Prefix Sharing)
    printf("[4] Verifying zero-copy Copy-On-Write (COW) sequence forking...\n");
    common_paged_kv_cache::sequence_table seq2 = kv.fork_sequence(seq1, 2);
    assert(seq2.total_tokens == 40);
    assert(seq2.block_ids.size() == 3);
    assert(seq2.block_ids == seq1.block_ids);
    assert(kv.active_block_count() == 3); // Zero new blocks allocated!
    printf("  Forked seq2 from seq1 in O(1) time without block allocation (shared ref count = 2).\n");
    printf("  Zero-copy fork check PASSED OK.\n\n");

    // [5] Copy-On-Write (COW) mutation
    printf("[5] Verifying COW divergence when mutating child sequence...\n");
    std::vector<uint8_t> new_k(cfg.n_embd_k, 0x55);
    std::vector<uint8_t> new_v(cfg.n_embd_v, 0x66);

    // Appending to seq2 should trigger COW on block 2
    int32_t old_shared_bid = seq2.block_ids.back();
    bool cow_ok = kv.append_token(seq2, new_k.data(), new_v.data());
    assert(cow_ok);
    assert(seq2.total_tokens == 41);
    assert(seq1.total_tokens == 40);
    assert(seq2.block_ids.back() != old_shared_bid); // Diverged physical block!
    assert(kv.active_block_count() == 4);            // Allocated 1 COW block for child

    printf("  Seq2 diverged to new physical block %d while Seq1 retained block %d.\n",
           seq2.block_ids.back(), old_shared_bid);
    printf("  COW mutation check PASSED OK.\n\n");

    // [6] DMA Staging Planner
    printf("[6] Verifying Zero-Copy PCIe DMA Staging Planner (O(N/B) transfers)...\n");
    auto dma_ops = kv.plan_dma_staging(seq2, 10, 25); // tokens 10 to 35
    printf("  Planned %zu DMA transfer operations for tokens [10..35]:\n", dma_ops.size());
    for (size_t i = 0; i < dma_ops.size(); ++i) {
        printf("    DMA Op %zu: host_src = %p, bytes = %zu, block_id = %d\n",
               i, (void*)dma_ops[i].host_src, dma_ops[i].byte_count, dma_ops[i].block_id);
    }
    assert(dma_ops.size() == 2); // Spans blocks 0 and 1
    printf("  DMA staging plan check PASSED OK.\n\n");

    // [7] Sequence freeing & recycling
    printf("[7] Freeing sequences and recycling block pool...\n");
    kv.free_sequence(seq1);
    assert(kv.active_block_count() == 3); // Seq2 still holds shared blocks 0, 1, and private block 3

    kv.free_sequence(seq2);
    assert(kv.active_block_count() == 0); // All blocks returned to pool!
    assert(kv.free_block_count() == cfg.max_blocks);
    printf("  All %u blocks cleanly returned to free list (zero memory leaks).\n", cfg.max_blocks);
    printf("  Recycling check PASSED OK.\n\n");

    // -------------------------------------------------------------------------------------
    // Test 8: Direct Zero-Copy DMA Target Resolution (io_uring / PCIe Direct Splicing)
    // -------------------------------------------------------------------------------------
    printf("[8] Verifying Direct Zero-Copy DMA Target Resolution (io_uring Splicing)...\n");
    void * dma_ptr = nullptr;
    size_t dma_bytes = 0;
    bool dma_ok = kv.get_block_dma_target(0, dma_ptr, dma_bytes);
    assert(dma_ok);
    assert(dma_ptr != nullptr);
    assert(dma_bytes == (cfg.block_size_tokens * cfg.n_embd_k + cfg.block_size_tokens * cfg.n_embd_v));
    assert(((uintptr_t)dma_ptr & 127) == 0); // 128-byte cache line alignment
    assert(kv.get_block_k(0) == dma_ptr);
    assert(kv.get_block_v(0) == (uint8_t *)dma_ptr + (cfg.block_size_tokens * cfg.n_embd_k));
    printf("  Block 0 resolved DMA target: %p (bytes: %zu, 128B aligned)\n", dma_ptr, dma_bytes);
    printf("  Direct DMA Splicing target check PASSED OK.\n\n");

    printf("=================================================================\n");
    printf("  ALL HOST-RAM PAGED ATTENTION TESTS PASSED WITH 100%% SUCCESS    \n");
    printf("=================================================================\n");

    return 0;
}
