#include "disruptor-ring.h"

#include <cstdio>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <numeric>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __FILE__, __LINE__, msg); \
            return 1; \
        } \
    } while (0)

struct test_prefill_task {
    int slot_id;
    int n_tokens;
    uint64_t hash;
};

int main() {
    printf("=================================================================\n");
    printf("  TEST: Sequenced Circular Disruptor Ring Buffer (LMAX Pattern)  \n");
    printf("=================================================================\n");

    // 1. Layout & Microarchitectural Alignment Invariants
    printf("\n[1] Verifying 128-byte cache line alignment and false sharing isolation...\n");
    {
        using ring_t = common_disruptor_ring<int, 1024>;
        static_assert(alignof(ring_t) >= 128, "Ring buffer must be at least 128-byte aligned");
        
        ring_t ring;
        uintptr_t addr = reinterpret_cast<uintptr_t>(&ring);
        TEST_ASSERT(addr % 128 == 0, "Ring instance must be 128-byte aligned");
        
        printf("  Ring instance address: 0x%lx (aligned to 128 bytes)\n", (unsigned long)addr);
        printf("  Ring capacity: %zu\n", ring.capacity());
        TEST_ASSERT(ring.capacity() == 1024, "Capacity must be 1024");
        printf("  Alignment check PASSED OK.\n");
    }

    // 2. Basic Push, Claim, Publish, and Consume
    printf("\n[2] Verifying basic claim/publish and consume semantics...\n");
    {
        common_disruptor_ring<int, 8> ring;
        TEST_ASSERT(ring.try_publish(42), "try_publish should succeed on empty ring");
        TEST_ASSERT(ring.try_publish(43), "try_publish should succeed for second item");

        int val = 0;
        TEST_ASSERT(ring.try_consume(val), "try_consume should succeed");
        TEST_ASSERT(val == 42, "First consumed value must be 42");

        TEST_ASSERT(ring.try_consume(val), "try_consume should succeed for second item");
        TEST_ASSERT(val == 43, "Second consumed value must be 43");

        TEST_ASSERT(!ring.try_consume(val), "try_consume should return false when empty");
        printf("  Basic claim/publish/consume check PASSED OK.\n");
    }

    // 3. Batched Publication & Consumption
    printf("\n[3] Verifying batched publication (claim_batch/publish_batch) & batch consume...\n");
    {
        common_disruptor_ring<int, 16> ring;
        int input_batch[6] = {10, 20, 30, 40, 50, 60};
        size_t published = ring.try_publish_batch(input_batch, 6);
        TEST_ASSERT(published == 6, "Published count must be 6");

        int output_batch[10] = {0};
        size_t consumed = ring.try_consume_batch(output_batch, 4);
        TEST_ASSERT(consumed == 4, "Consumed count must be 4");
        for (size_t i = 0; i < 4; ++i) {
            TEST_ASSERT(output_batch[i] == input_batch[i], "Batch data mismatch in first slice");
        }

        consumed = ring.try_consume_batch(output_batch, 10);
        TEST_ASSERT(consumed == 2, "Remaining consumed count must be 2");
        TEST_ASSERT(output_batch[0] == 50, "Second slice value 0 mismatch");
        TEST_ASSERT(output_batch[1] == 60, "Second slice value 1 mismatch");

        printf("  Batched publish/consume PASSED OK.\n");
    }

    // 4. In-Place operator[] Modification
    printf("\n[4] Verifying zero-copy in-place slot modification...\n");
    {
        common_disruptor_ring<test_prefill_task, 16> ring;
        int64_t seq = ring.claim();
        TEST_ASSERT(seq >= 0, "Claim must succeed");
        ring[seq].slot_id = 7;
        ring[seq].n_tokens = 2048;
        ring[seq].hash = 0xDEADBEEFCAFE;
        ring.publish(seq);

        test_prefill_task task{};
        TEST_ASSERT(ring.try_consume(task), "try_consume must succeed");
        TEST_ASSERT(task.slot_id == 7, "slot_id mismatch");
        TEST_ASSERT(task.n_tokens == 2048, "n_tokens mismatch");
        TEST_ASSERT(task.hash == 0xDEADBEEFCAFE, "hash mismatch");
        printf("  Zero-copy in-place slot modification PASSED OK.\n");
    }

    // 5. High-Throughput Concurrent SPSC Streaming (100,000 items)
    printf("\n[5] Verifying high-throughput concurrent SPSC streaming (100,000 items)...\n");
    {
        constexpr size_t N_ITEMS = 100000;
        common_disruptor_ring<uint64_t, 2048> ring;
        std::atomic<bool> producer_done{false};

        std::vector<uint64_t> received;
        received.reserve(N_ITEMS);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Producer Thread
        std::thread producer([&]() {
            for (uint64_t i = 0; i < N_ITEMS; ++i) {
                while (!ring.try_publish(i)) {
                    disruptor_cpu_pause();
                }
            }
            producer_done.store(true, std::memory_order_release);
        });

        // Consumer Thread
        std::thread consumer([&]() {
            uint64_t item = 0;
            while (received.size() < N_ITEMS) {
                if (ring.try_consume(item)) {
                    received.push_back(item);
                } else {
                    if (producer_done.load(std::memory_order_acquire)) {
                        while (ring.try_consume(item)) {
                            received.push_back(item);
                        }
                        break;
                    }
                    disruptor_cpu_pause();
                }
            }
        });

        producer.join();
        consumer.join();

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double throughput_mops = (static_cast<double>(N_ITEMS) / (elapsed_ms / 1000.0)) / 1e6;

        TEST_ASSERT(received.size() == N_ITEMS, "All items must be received");
        for (size_t i = 0; i < N_ITEMS; ++i) {
            TEST_ASSERT(received[i] == i, "FIFO sequence must be strictly preserved");
        }

        printf("  Streamed %zu items in %.2f ms (%.2f M ops/sec)\n", N_ITEMS, elapsed_ms, throughput_mops);
        printf("  Producer shadow hit rate: %.2f%% (hits: %lu, remote: %lu)\n",
               ring.producer_shadow_hit_rate(),
               (unsigned long)ring.producer_shadow_hits(),
               (unsigned long)ring.producer_remote_polls());
        printf("  Consumer shadow hit rate: %.2f%% (hits: %lu, remote: %lu)\n",
               ring.consumer_shadow_hit_rate(),
               (unsigned long)ring.consumer_shadow_hits(),
               (unsigned long)ring.consumer_remote_polls());
        
        TEST_ASSERT(ring.producer_shadow_hit_rate() > 70.0, "Producer shadow hit rate should exceed 70%");
        printf("  High-throughput SPSC streaming PASSED OK.\n");
    }

    // 6. Batched Prefill Dispatch Simulation
    printf("\n[6] Verifying batched multi-slot prefill dispatch simulation...\n");
    {
        constexpr size_t BATCH_SIZE = 16;
        constexpr size_t TOTAL_BATCHES = 1000;
        common_disruptor_ring<test_prefill_task, 512> ring;

        std::atomic<bool> prod_done{false};
        std::atomic<bool> consumer_ok{true};
        std::thread producer([&]() -> void {
            for (size_t b = 0; b < TOTAL_BATCHES; ++b) {
                test_prefill_task batch[BATCH_SIZE];
                for (size_t i = 0; i < BATCH_SIZE; ++i) {
                    batch[i].slot_id = static_cast<int>(i);
                    batch[i].n_tokens = static_cast<int>(128 + i * 16);
                    batch[i].hash = static_cast<uint64_t>(b * BATCH_SIZE + i);
                }
                size_t sent = 0;
                while (sent < BATCH_SIZE) {
                    sent += ring.try_publish_batch(batch + sent, BATCH_SIZE - sent);
                    if (sent < BATCH_SIZE) disruptor_cpu_pause();
                }
            }
            prod_done.store(true);
        });

        std::thread consumer([&]() -> void {
            size_t total_received = 0;
            test_prefill_task batch[BATCH_SIZE * 2];
            while (total_received < TOTAL_BATCHES * BATCH_SIZE) {
                size_t n = ring.try_consume_batch(batch, BATCH_SIZE * 2);
                for (size_t i = 0; i < n; ++i) {
                    if (batch[i].hash != total_received + i) {
                        consumer_ok.store(false);
                    }
                }
                total_received += n;
                if (n == 0) disruptor_cpu_pause();
            }
        });

        producer.join();
        consumer.join();
        TEST_ASSERT(consumer_ok.load(), "Task hash sequence mismatch in consumer");
        printf("  Dispatched %zu batched prefill tasks across slots PASSED OK.\n", TOTAL_BATCHES * BATCH_SIZE);
    }

    printf("\n=================================================================\n");
    printf("  ALL DISRUPTOR RING BUFFER TESTS PASSED WITH 100%% SUCCESS       \n");
    printf("=================================================================\n");
    return 0;
}
