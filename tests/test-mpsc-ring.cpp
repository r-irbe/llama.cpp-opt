#include "mpsc-ring.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "FAIL (line %d): %s\n", __LINE__, msg); return 1; } } while (0)

// heap-owning, move-only-ish payload like server_task (json + vectors)
struct payload {
    int producer = -1;
    int seq      = -1;
    std::unique_ptr<std::string> blob;
};

int main() {
    printf("test-mpsc-ring\n");

    // 1. single-thread semantics: FIFO, full, empty, move-only payload
    {
        common_mpsc_ring<payload> ring(4);
        CHECK(ring.capacity() == 4, "capacity");
        for (int i = 0; i < 4; ++i) {
            payload p{0, i, std::make_unique<std::string>(std::to_string(i))};
            CHECK(ring.try_push(std::move(p)), "push into non-full ring");
        }
        payload extra{0, 99, std::make_unique<std::string>("x")};
        CHECK(!ring.try_push(std::move(extra)), "push into full ring fails");
        CHECK(extra.blob && *extra.blob == "x", "failed push leaves value intact");

        payload out;
        for (int i = 0; i < 4; ++i) {
            CHECK(ring.try_pop(out), "pop");
            CHECK(out.seq == i && out.blob && *out.blob == std::to_string(i), "FIFO order + payload");
        }
        CHECK(!ring.try_pop(out), "pop from empty ring fails");
        printf("  single-thread semantics                         OK\n");
    }

    // 2. invalid capacity rejected
    {
        bool threw = false;
        try { common_mpsc_ring<int> bad(6); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "non power-of-two capacity throws");
        printf("  capacity validation                             OK\n");
    }

    // 3. multi-producer stress: no loss, no duplication, per-producer FIFO
    {
        constexpr int P = 8;
        constexpr int N = 200000;
        common_mpsc_ring<payload> ring(256);
        std::atomic<int> started{0};

        std::vector<std::thread> producers;
        for (int p = 0; p < P; ++p) {
            producers.emplace_back([&, p] {
                started.fetch_add(1);
                while (started.load() < P) {}
                for (int i = 0; i < N; ++i) {
                    payload v{p, i, nullptr};
                    while (!ring.try_push(std::move(v))) { std::this_thread::yield(); }
                }
            });
        }

        std::vector<int> next(P, 0);
        long total = 0;
        bool order_ok = true;
        payload out;
        while (total < (long) P * N) {
            if (ring.try_pop(out)) {
                if (out.seq != next[out.producer]) order_ok = false;
                next[out.producer] = out.seq + 1;
                ++total;
            }
        }
        for (auto & t : producers) t.join();

        CHECK(order_ok, "per-producer FIFO preserved");
        for (int p = 0; p < P; ++p) CHECK(next[p] == N, "all items from every producer received");
        CHECK(!ring.try_pop(out), "ring empty after drain");
        printf("  %d producers x %d items: no loss, per-producer FIFO OK\n", P, N);
    }

    printf("test-mpsc-ring: all tests OK\n");
    return 0;
}
