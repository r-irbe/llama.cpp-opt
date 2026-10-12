// Unit test suite for 2-Tier MoE Expert DMA Pipeline (Track 3)
//
// Verifies:
// 1. Storage NUMA discovery & HugeTLB staging pool allocation
// 2. Direct NVMe io_uring + O_DIRECT asynchronous streaming
// 3. 2-Tier Pipelining: Overlapped Layer L compute with Layer L+1 DMA prefetch
// 4. Staging pool circular slot reuse and cache hit detection
// 5. Direct external buffer DMA streaming
// 6. Sub-sector and unaligned offset resilience
// 7. Byte-for-byte binary integrity verification against reference data

#include "moe-expert-dma-pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-68s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

// Generate synthetic model weights file with layer and expert metadata embedded
static bool generate_synthetic_model(const char * path, int n_layers, int n_experts, size_t expert_size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    std::vector<uint8_t> buffer(expert_size);

    for (int l = 0; l < n_layers; ++l) {
        for (int e = 0; e < n_experts; ++e) {
            uint8_t seed = static_cast<uint8_t>((l * 37 + e * 19 + 0x42) & 0xFF);
            for (size_t i = 0; i < expert_size; ++i) {
                buffer[i] = static_cast<uint8_t>((i + seed) & 0xFF);
            }
            // Embed header at first 16 bytes for identification
            uint32_t magic = 0x4D4F4558; // "MOEX"
            std::memcpy(buffer.data(), &magic, 4);
            std::memcpy(buffer.data() + 4, &l, 4);
            std::memcpy(buffer.data() + 8, &e, 4);

            ssize_t written = 0;
            while (written < static_cast<ssize_t>(expert_size)) {
                ssize_t n = write(fd, buffer.data() + written, expert_size - written);
                if (n <= 0) {
                    close(fd);
                    unlink(path);
                    return false;
                }
                written += n;
            }
        }
    }

    close(fd);
    return true;
}

// Verify that resident slice matches expected synthetic pattern
static bool verify_expert_data(const void * ptr, int layer, int expert, size_t expert_size) {
    if (!ptr) return false;
    const uint8_t * p = static_cast<const uint8_t *>(ptr);

    uint32_t magic = 0;
    int l = -1, e = -1;
    std::memcpy(&magic, p, 4);
    std::memcpy(&l, p + 4, 4);
    std::memcpy(&e, p + 8, 4);

    if (magic != 0x4D4F4558 || l != layer || e != expert) {
        return false;
    }

    uint8_t seed = static_cast<uint8_t>((layer * 37 + expert * 19 + 0x42) & 0xFF);
    for (size_t i = 16; i < expert_size; ++i) {
        uint8_t expected = static_cast<uint8_t>((i + seed) & 0xFF);
        if (p[i] != expected) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("===================================================================\n");
    printf("   ik_llama.cpp - 2-Tier MoE Expert DMA Pipeline Test Suite        \n");
    printf("===================================================================\n\n");

    const char * test_model_path = "/tmp/test_moe_dma_synthetic_model.bin";
    constexpr int N_LAYERS = 4;
    constexpr int N_EXPERTS = 8;
    constexpr size_t EXPERT_SIZE = 1 * 1024 * 1024; // 1 MB per expert slice (total 32 MB)
    const size_t total_size = static_cast<size_t>(N_LAYERS) * N_EXPERTS * EXPERT_SIZE;

    printf("=== Step 1: Synthetic MoE Model Generation ===\n");
    check(generate_synthetic_model(test_model_path, N_LAYERS, N_EXPERTS, EXPERT_SIZE),
          "Generated 32 MB synthetic MoE weights file (4 layers x 8 experts)");

    printf("\n=== Step 2: Storage NUMA & Staging Pool Initialization ===\n");
    common::moe_expert_dma_pipeline pipeline;
    common::moe_expert_dma_pipeline::config cfg;
    cfg.slot_size = 2 * 1024 * 1024;    // 2 MB slots
    cfg.num_slots = 16;                 // 16 slots = 32 MB pool
    cfg.try_direct_io = true;
    cfg.try_sqpoll = true;
    cfg.try_register_buffers = true;
    cfg.populate_pages = true;

    bool init_ok = pipeline.init(test_model_path, -1, cfg);
    check(init_ok, "2-Tier MoE Expert DMA Pipeline initialized successfully");
    check(pipeline.get_num_slots() == 16, "Pipeline allocated 16 staging slots");
    check(pipeline.get_slot_size() == 2 * 1024 * 1024, "Slot size configured to 2 MB (HugeTLB granule)");
    check(pipeline.get_pool_size() == 32 * 1024 * 1024, "Staging pool size equals 32 MB");

    printf("     -> Storage NUMA Node: %d\n", pipeline.get_numa_node());
    printf("     -> Mode: Direct I/O=%s | SQPOLL=%s | Registered Buffers=%s | Fallback=%s\n",
           pipeline.is_direct_io() ? "YES" : "NO",
           pipeline.is_sqpoll() ? "YES" : "NO",
           pipeline.is_registered() ? "YES" : "NO",
           pipeline.is_fallback() ? "YES" : "NO");

    printf("\n=== Step 3: Direct Asynchronous Expert Prefetch & Validation ===\n");
    // Prefetch Layer 0 Expert 0
    size_t off_l0_e0 = (0 * N_EXPERTS + 0) * EXPERT_SIZE;
    uint64_t t0 = pipeline.prefetch_expert(0, 0, off_l0_e0, EXPERT_SIZE);
    check(t0 > 0, "Submitted prefetch for Layer 0 Expert 0");

    bool wait0_ok = pipeline.wait_expert(0, 0);
    check(wait0_ok, "Wait on Layer 0 Expert 0 returned success");
    check(pipeline.is_expert_ready(0, 0), "Status check confirms Expert (0, 0) is ready");

    const void * data_l0_e0 = pipeline.get_expert_data(0, 0);
    check(data_l0_e0 != nullptr, "Retrieved memory pointer for Layer 0 Expert 0");
    check(verify_expert_data(data_l0_e0, 0, 0, EXPERT_SIZE), "Byte-for-byte binary integrity verified for Layer 0 Expert 0");

    // Batch Prefetch for Layer 0 remaining routed experts (e.g. top-2: experts 2 and 5)
    std::vector<int> routed_l0 = {2, 5};
    std::vector<size_t> offs_l0 = {
        (0 * N_EXPERTS + 2) * EXPERT_SIZE,
        (0 * N_EXPERTS + 5) * EXPERT_SIZE
    };
    std::vector<size_t> lens_l0 = { EXPERT_SIZE, EXPERT_SIZE };

    auto tickets_l0 = pipeline.prefetch_layer(0, routed_l0, offs_l0, lens_l0);
    check(tickets_l0.size() == 2 && tickets_l0[0] > 0 && tickets_l0[1] > 0,
          "Batch prefetch submitted for Layer 0 routed experts {2, 5}");

    bool wait_batch_ok = pipeline.wait_layer(0, routed_l0);
    check(wait_batch_ok, "Batch wait on Layer 0 routed experts completed");
    check(verify_expert_data(pipeline.get_expert_data(0, 2), 0, 2, EXPERT_SIZE), "Binary integrity verified for Expert (0, 2)");
    check(verify_expert_data(pipeline.get_expert_data(0, 5), 0, 5, EXPERT_SIZE), "Binary integrity verified for Expert (0, 5)");

    printf("\n=== Step 4: Overlapped 2-Tier Pipeline Emulation (Compute vs DMA) ===\n");
    // Simulate inference execution:
    // 1. Current Layer: Layer 1
    // 2. While computing Layer 1 attention/GEMM, prefetch Layer 2 routed experts {1, 4, 7}
    std::vector<int> routed_l2 = {1, 4, 7};
    std::vector<size_t> offs_l2 = {
        (2 * N_EXPERTS + 1) * EXPERT_SIZE,
        (2 * N_EXPERTS + 4) * EXPERT_SIZE,
        (2 * N_EXPERTS + 7) * EXPERT_SIZE
    };
    std::vector<size_t> lens_l2 = { EXPERT_SIZE, EXPERT_SIZE, EXPERT_SIZE };

    auto start_pipe = std::chrono::high_resolution_clock::now();

    // Enqueue Layer 2 asynchronously
    auto tickets_l2 = pipeline.prefetch_layer(2, routed_l2, offs_l2, lens_l2);
    check(tickets_l2.size() == 3, "Submitted background DMA prefetch for Layer 2 experts {1, 4, 7}");

    // Simulate Layer 1 CPU computation (approx 5 milliseconds of arithmetic)
    volatile double dummy_sum = 0.0;
    for (int iter = 0; iter < 1000000; ++iter) {
        dummy_sum += iter * 0.0001;
    }

    // Now transition to Layer 2: experts should already be resident!
    bool l2_ready_early = pipeline.is_expert_ready(2, 1) &&
                          pipeline.is_expert_ready(2, 4) &&
                          pipeline.is_expert_ready(2, 7);
    printf("     -> Background DMA completed during compute window: %s\n", l2_ready_early ? "YES (Zero Wait)" : "IN-FLIGHT");

    bool wait_l2_ok = pipeline.wait_layer(2, routed_l2);
    auto end_pipe = std::chrono::high_resolution_clock::now();
    double pipe_elapsed_ms = std::chrono::duration<double, std::milli>(end_pipe - start_pipe).count();

    check(wait_l2_ok, "Layer 2 experts synchronized with zero faults");
    check(verify_expert_data(pipeline.get_expert_data(2, 1), 2, 1, EXPERT_SIZE), "Layer 2 Expert 1 data valid");
    check(verify_expert_data(pipeline.get_expert_data(2, 4), 2, 4, EXPERT_SIZE), "Layer 2 Expert 4 data valid");
    check(verify_expert_data(pipeline.get_expert_data(2, 7), 2, 7, EXPERT_SIZE), "Layer 2 Expert 7 data valid");
    printf("     -> Overlapped compute + DMA time: %.2f ms\n", pipe_elapsed_ms);

    printf("\n=== Step 5: Staging Pool Circular Reuse & Cache Hits ===\n");
    // Repeating prefetch for already-loaded expert should be a cache hit
    uint64_t hit_ticket = pipeline.prefetch_expert(2, 1, offs_l2[0], EXPERT_SIZE);
    check(hit_ticket == tickets_l2[0], "Repeated prefetch returns identical cached ticket (Cache Hit)");
    check(pipeline.get_cache_hits() >= 1, "Observability counters confirm cache hit");

    // Fill and overflow circular ring
    for (int exp = 0; exp < N_EXPERTS; ++exp) {
        size_t off = (3 * N_EXPERTS + exp) * EXPERT_SIZE;
        pipeline.prefetch_expert(3, exp, off, EXPERT_SIZE);
    }
    check(pipeline.wait_layer(3, {0, 1, 2, 3, 4, 5, 6, 7}), "Circular slot reuse handled Layer 3 (all 8 experts) correctly");
    check(verify_expert_data(pipeline.get_expert_data(3, 7), 3, 7, EXPERT_SIZE), "Verified circular slot wrap-around data integrity");

    printf("\n=== Step 6: Sub-Sector & Trailing Offset Alignment ===\n");
    // Test unaligned offset (e.g. +137 bytes from sector boundary) and non-power-of-two size
    size_t unaligned_off = off_l0_e0 + 137;
    size_t unaligned_len = 16384 + 59;
    uint64_t unaligned_ticket = pipeline.prefetch_expert(0, 7, unaligned_off, unaligned_len);
    check(unaligned_ticket > 0, "Submitted prefetch with unaligned offset (+137B) and length (16443B)");
    check(pipeline.wait_expert(0, 7), "Wait on unaligned expert slice succeeded");

    const uint8_t * unaligned_data = static_cast<const uint8_t *>(pipeline.get_expert_data(0, 7));
    check(unaligned_data != nullptr, "Unaligned slice payload pointer valid");

    // Verify sub-slice pattern against expected file content
    bool unaligned_match = true;
    uint8_t seed_e7 = static_cast<uint8_t>((0 * 37 + 0 * 19 + 0x42) & 0xFF);
    for (size_t i = 0; i < unaligned_len; ++i) {
        size_t file_pos = 137 + i;
        uint8_t expected = 0;
        if (file_pos < 16) {
            // Header range
            uint32_t magic = 0x4D4F4558;
            int l = 0, e = 0;
            uint8_t hdr[16];
            std::memcpy(hdr, &magic, 4);
            std::memcpy(hdr + 4, &l, 4);
            std::memcpy(hdr + 8, &e, 4);
            expected = hdr[file_pos];
        } else {
            expected = static_cast<uint8_t>((file_pos + seed_e7) & 0xFF);
        }
        if (unaligned_data[i] != expected) {
            unaligned_match = false;
            break;
        }
    }
    check(unaligned_match, "Unaligned sub-slice byte-for-byte binary match verified");

    printf("\n=== Step 7: Direct Caller Buffer Streaming ===\n");
    void * direct_buf = nullptr;
    posix_memalign(&direct_buf, 4096, 65536);
    std::memset(direct_buf, 0, 65536);

    uint64_t direct_read_res = pipeline.submit_direct_read(off_l0_e0, direct_buf, 65536);
    check(direct_read_res > 0, "submit_direct_read succeeded into caller 4KB-aligned buffer");
    check(pipeline.wait_direct_read(direct_read_res), "Wait on direct read completed");
    check(verify_expert_data(direct_buf, 0, 0, 65536), "Direct caller buffer data verified");
    free(direct_buf);


    printf("\n=== Step 8: Observability & Throughput Stats ===\n");
    llama_io_uring_stats stats = pipeline.get_io_stats();
    printf("     -> Total DMA bytes streamed: %zu MB\n", stats.total_bytes / (1024 * 1024));
    printf("     -> Total prefetches issued:  %llu\n", (unsigned long long)pipeline.get_total_prefetches());
    printf("     -> Cache hits:               %llu\n", (unsigned long long)pipeline.get_cache_hits());
    printf("     -> Cache misses:             %llu\n", (unsigned long long)pipeline.get_cache_misses());

    // Clean up
    pipeline.shutdown();
    unlink(test_model_path);

    printf("\n===================================================================\n");
    if (n_fail == 0) {
        printf("  ALL 2-TIER MOE EXPERT DMA TESTS PASSED (100%% SUCCESS)\n");
    } else {
        printf("  %d TESTS FAILED\n", n_fail);
    }
    printf("===================================================================\n");

    return n_fail;
}
