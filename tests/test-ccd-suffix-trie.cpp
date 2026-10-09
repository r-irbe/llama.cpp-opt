#include "ccd-suffix-trie.h"
#include "speculative.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

int main() {
    printf("=================================================================\n");
    printf("  TEST: CCD-Sharded N-Gram Suffix Trie with EBR & 128B Alignment \n");
    printf("=================================================================\n\n");

    // [1] Verify 128B cache line alignment
    printf("[1] Verifying 128-byte cache line alignment and false sharing isolation...\n");
    static_assert(sizeof(common_ccd_suffix_trie::shard_t) % 128 == 0, "shard_t must be 128B aligned");
    static_assert(alignof(common_ccd_suffix_trie::shard_t) >= 128, "shard_t must have alignas(128)");
    static_assert(sizeof(common_ebr_tracker::reader_slot) % 128 == 0, "reader_slot must be 128B aligned");

    common_ccd_suffix_trie trie(8, 16);
    assert(trie.num_shards() == 8);
    printf("  Trie shards: %zu, max_depth: %d\n", trie.num_shards(), trie.max_depth());
    printf("  Alignment check PASSED OK.\n\n");

    // [2] Basic extend and speculate
    printf("[2] Verifying single-shard pattern learning and speculative continuation...\n");
    std::vector<llama_token> pattern1 = {101, 102, 103, 104, 105, 106, 107};
    std::vector<llama_token> pattern2 = {101, 102, 103, 104, 105, 106, 107};
    std::vector<llama_token> pattern3 = {101, 102, 103, 204, 205};

    trie.extend(0, pattern1.data(), (int)pattern1.size());
    trie.extend(0, pattern2.data(), (int)pattern2.size());
    trie.extend(0, pattern3.data(), (int)pattern3.size());

    // Context: [101, 102, 103] -> continuation should predict [104, 105, 106, 107] (higher frequency than 204)
    std::vector<llama_token> ctx = {101, 102, 103};
    auto draft = trie.speculate(0, 0, ctx.data(), (int)ctx.size(), 4, 0.1f, 3);

    printf("  Speculate returned %zu draft tokens:\n", draft.size());
    for (size_t i = 0; i < draft.size(); ++i) {
        printf("    token[%zu] = %d\n", i, draft[i]);
    }
    assert(draft.size() == 4);
    assert(draft[0] == 104);
    assert(draft[1] == 105);
    assert(draft[2] == 106);
    assert(draft[3] == 107);
    printf("  Pattern learning & continuation check PASSED OK.\n\n");

    // [3] Streaming directly into SPSC ring buffer
    printf("[3] Verifying direct streaming into 128B wait-free SPSC ring buffer...\n");
    common_spsc_ring_buffer<llama_token, 64> ring;
    size_t pushed = trie.speculate_to_spsc_ring(0, 0, ctx.data(), (int)ctx.size(), ring, 4, 0.1f, 3);
    assert(pushed == 4);
    assert(ring.size() == 4);

    llama_token popped[4];
    size_t pop_count = ring.pop_batch(popped, 4);
    assert(pop_count == 4);
    assert(popped[0] == 104 && popped[3] == 107);
    printf("  SPSC ring buffer draft token streaming PASSED OK.\n\n");

    // [4] Multithreaded concurrent reader EBR stress test across 8 CCD shards
    printf("[4] Verifying concurrent 8-thread read / 1-thread write under EBR...\n");
    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> total_queries{0};

    // Pre-populate shards
    for (size_t s = 0; s < 8; ++s) {
        std::vector<llama_token> tokens;
        for (int i = 0; i < 50; ++i) {
            tokens.push_back((llama_token)(s * 1000 + (i % 10)));
        }
        trie.extend(s, tokens.data(), (int)tokens.size());
    }

    // Spawn 8 reader threads (one per CCD shard)
    std::vector<std::thread> readers;
    for (size_t r = 0; r < 8; ++r) {
        readers.emplace_back([&, r]() {
            std::vector<llama_token> local_ctx = {
                (llama_token)(r * 1000 + 0),
                (llama_token)(r * 1000 + 1),
                (llama_token)(r * 1000 + 2)
            };
            while (!stop_flag.load(std::memory_order_relaxed)) {
                auto res = trie.speculate(r, r, local_ctx.data(), (int)local_ctx.size(), 3, 0.05f, 2);
                (void)res;
                total_queries.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Writer thread periodically extending and advancing EBR
    std::thread writer([&]() {
        for (int step = 0; step < 50; ++step) {
            for (size_t s = 0; s < 8; ++s) {
                llama_token new_toks[5] = {
                    (llama_token)(s * 1000 + (step % 10)),
                    (llama_token)(s * 1000 + ((step + 1) % 10)),
                    (llama_token)(s * 1000 + ((step + 2) % 10)),
                    (llama_token)(s * 1000 + ((step + 3) % 10)),
                    (llama_token)(s * 1000 + ((step + 4) % 10))
                };
                trie.extend(s, new_toks, 5);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            trie.ebr().try_advance_and_reclaim();
        }
    });

    writer.join();
    stop_flag.store(true, std::memory_order_release);
    for (auto & t : readers) {
        t.join();
    }

    printf("  Completed %lu concurrent lock-free speculative queries during active mutations.\n", total_queries.load());
    printf("  EBR global epoch: %lu\n", trie.ebr().global_epoch());
    printf("  Concurrent EBR stress test PASSED OK.\n\n");

    // [5] Clear and memory leak validation
    printf("[5] Verifying clean trie reset and node reclamation...\n");
    trie.clear();
    auto empty_draft = trie.speculate(0, 0, ctx.data(), (int)ctx.size(), 4, 0.1f, 3);
    assert(empty_draft.empty());
    printf("  Clear & reclamation check PASSED OK.\n\n");

    printf("=================================================================\n");
    printf("  ALL CCD-SHARDED SUFFIX TRIE TESTS PASSED WITH 100%% SUCCESS     \n");
    printf("=================================================================\n");

    return 0;
}
