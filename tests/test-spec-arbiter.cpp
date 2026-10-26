#include "spec-arbiter.h"
#include <cstdio>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

int main() {
    printf("=================================================================\n");
    printf("  TEST: Asynchronous Multi-CCX Drafting Arbiter & TreeGraft Fusion\n");
    printf("=================================================================\n");

    // 1. TreeGraft Fusion: N-gram Only
    printf("\n[1] Verifying TreeGraft fusion with N-gram branch only...\n");
    {
        std::vector<llama_token> ngram = {101, 102, 103};
        std::vector<llama_token> neural;
        auto fused = spec_arbiter::fuse_treegraft(ngram, neural, 8);
        TEST_ASSERT(fused.primary_source == SPEC_DRAFTER_SOURCE_NGRAM, "Source should be N-gram");
        TEST_ASSERT(!fused.is_tree, "Single branch should not be tree");
        TEST_ASSERT(fused.tokens.size() == 3, "Token count mismatch");
        TEST_ASSERT(fused.tokens[0] == 101 && fused.tokens[2] == 103, "Token sequence mismatch");
        printf("  N-gram only fusion PASSED OK.\n");
    }

    // 2. TreeGraft Fusion: Neural Only
    printf("\n[2] Verifying TreeGraft fusion with Neural branch only...\n");
    {
        std::vector<llama_token> ngram;
        std::vector<llama_token> neural = {201, 202, 203, 204};
        auto fused = spec_arbiter::fuse_treegraft(ngram, neural, 8);
        TEST_ASSERT(fused.primary_source == SPEC_DRAFTER_SOURCE_NEURAL, "Source should be Neural");
        TEST_ASSERT(!fused.is_tree, "Single branch should not be tree");
        TEST_ASSERT(fused.tokens.size() == 4, "Token count mismatch");
        TEST_ASSERT(fused.tokens[0] == 201 && fused.tokens[3] == 204, "Token sequence mismatch");
        printf("  Neural only fusion PASSED OK.\n");
    }

    // 3. TreeGraft Multi-Branch Fusion (Shared Prefix + Divergent Sub-branches)
    printf("\n[3] Verifying TreeGraft fusion with shared prefix and divergent sub-branches...\n");
    {
        std::vector<llama_token> ngram  = {50, 51, 60, 61};
        std::vector<llama_token> neural = {50, 51, 70, 71};
        auto fused = spec_arbiter::fuse_treegraft(ngram, neural, 16);
        TEST_ASSERT(fused.primary_source == SPEC_DRAFTER_SOURCE_TREEGRAFT, "Source should be TreeGraft");
        TEST_ASSERT(fused.is_tree, "Fused multi-branch must be tree");
        // Shared prefix: {50, 51} (indices 0, 1)
        // Neural branch: {70, 71} (indices 2, 3) attached to trunk parent index 1
        // Ngram branch:  {60, 61} (indices 4, 5) attached to trunk parent index 1
        TEST_ASSERT(fused.tokens.size() == 6, "Total fused tokens should be 6");
        TEST_ASSERT(fused.tokens[0] == 50 && fused.tokens[1] == 51, "Shared trunk mismatch");
        TEST_ASSERT(fused.tree_nodes[0].parent_idx == -1, "Root trunk parent must be -1");
        TEST_ASSERT(fused.tree_nodes[1].parent_idx == 0, "Second trunk token parent must be 0");
        TEST_ASSERT(fused.tree_nodes[2].parent_idx == 1, "First neural divergence parent must be trunk top (1)");
        TEST_ASSERT(fused.tree_nodes[4].parent_idx == 1, "First ngram divergence parent must be trunk top (1)");
        printf("  TreeGraft shared trunk and branch divergence PASSED OK.\n");
    }

    // 4. Lock-Free SPSC and Disruptor Ring Cross-Thread Communication
    printf("\n[4] Verifying cross-thread lock-free ring transfer across drafter workers...\n");
    {
        spec_arbiter arbiter;
        std::atomic<bool> producer_done{false};

        // Worker thread simulating N-gram drafter on CCX 0
        std::thread ngram_worker([&arbiter, &producer_done]() {
            for (int i = 0; i < 50; ++i) {
                spec_arbiter::drafter_packet pkt = {};
                pkt.source_type = SPEC_DRAFTER_SOURCE_NGRAM;
                pkt.n_tokens = 3;
                pkt.tokens[0] = 1000 + i;
                pkt.tokens[1] = 1001 + i;
                pkt.tokens[2] = 1002 + i;
                while (!arbiter.ring_ngram.push(pkt)) {
                    std::this_thread::yield();
                }
            }
            producer_done.store(true);
        });

        // Main thread acting as Arbiter
        int received_packets = 0;
        while (!producer_done.load() || received_packets < 50) {
            spec_arbiter::drafter_packet pkt = {};
            if (arbiter.ring_ngram.pop(pkt)) {
                TEST_ASSERT(pkt.source_type == SPEC_DRAFTER_SOURCE_NGRAM, "Packet source mismatch");
                TEST_ASSERT(pkt.n_tokens == 3, "Token count mismatch");
                TEST_ASSERT(pkt.tokens[0] == 1000 + received_packets, "Token payload mismatch");
                received_packets++;
            } else {
                std::this_thread::yield();
            }
        }
        ngram_worker.join();
        TEST_ASSERT(received_packets == 50, "Should have received all 50 packets");
        printf("  Lock-free SPSC transfer PASSED OK (50 packets verified).\n");
    }

    printf("\nAll test-spec-arbiter assertions PASSED cleanly!\n");
    return 0;
}
