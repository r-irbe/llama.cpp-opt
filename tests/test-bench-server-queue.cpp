#include "mpsc-ring.h"
#include "benchmark-utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using bench_clock_t = std::chrono::high_resolution_clock;

struct BenchTask {
    int id = -1;
    bench_clock_t::time_point t_posted;
    char payload[64] = {0};
};

class BaselineQueue {
    std::mutex m;
    std::deque<BenchTask> q;
    std::condition_variable cv;
    std::atomic<bool> running{true};
public:
    void post(BenchTask task) {
        std::unique_lock<std::mutex> lock(m);
        q.push_back(std::move(task));
        cv.notify_one();
    }

    bool pop(BenchTask & task) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty() && running.load(std::memory_order_relaxed)) {
            cv.wait(lock);
        }
        if (q.empty()) return false;
        task = std::move(q.front());
        q.pop_front();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(m);
            running.store(false, std::memory_order_release);
        }
        cv.notify_all();
    }
};

class LockfreeMpscQueue {
    static constexpr size_t RING_CAPACITY = 256;
    common_mpsc_ring<BenchTask> ring{RING_CAPACITY};
    std::mutex mutex_overflow;
    std::deque<BenchTask> overflow;
    std::atomic<bool> overflow_active{false};
    std::atomic<uint32_t> signal{0};
    std::atomic<bool> running{true};

    std::deque<BenchTask> local_tasks;
public:
    void post(BenchTask task) {
        if (!overflow_active.load(std::memory_order_relaxed)) {
            if (ring.try_push(std::move(task))) {
                signal.fetch_add(1, std::memory_order_release);
                signal.notify_one();
                return;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_overflow);
        overflow_active.store(true, std::memory_order_relaxed);
        overflow.push_back(std::move(task));
        signal.fetch_add(1, std::memory_order_release);
        signal.notify_one();
    }

    bool pop(BenchTask & task) {
        if (!local_tasks.empty()) {
            task = std::move(local_tasks.front());
            local_tasks.pop_front();
            return true;
        }

        while (running.load(std::memory_order_acquire)) {
            const uint32_t observed = signal.load(std::memory_order_acquire);
            BenchTask item;
            while (ring.try_pop(item)) {
                local_tasks.push_back(std::move(item));
            }
            if (overflow_active.load(std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lock(mutex_overflow);
                while (ring.try_pop(item)) {
                    local_tasks.push_back(std::move(item));
                }
                while (!overflow.empty()) {
                    local_tasks.push_back(std::move(overflow.front()));
                    overflow.pop_front();
                }
                overflow_active.store(false, std::memory_order_release);
            }
            if (!local_tasks.empty()) {
                task = std::move(local_tasks.front());
                local_tasks.pop_front();
                return true;
            }
            if (!running.load(std::memory_order_acquire)) break;
            signal.wait(observed, std::memory_order_acquire);
        }
        return false;
    }

    void stop() {
        running.store(false, std::memory_order_release);
        signal.fetch_add(1, std::memory_order_release);
        signal.notify_all();
    }
};

struct LatencyStats {
    double mean_ns = 0.0;
    double p50_ns  = 0.0;
    double p90_ns  = 0.0;
    double p99_ns  = 0.0;
    double p999_ns = 0.0;
    double max_ns  = 0.0;
    double mops    = 0.0;
};

static LatencyStats compute_stats(std::vector<uint32_t> & samples, double total_seconds) {
    LatencyStats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double sum = 0;
    for (auto v : samples) sum += v;
    s.mean_ns = sum / n;
    s.p50_ns  = samples[n * 50 / 100];
    s.p90_ns  = samples[n * 90 / 100];
    s.p99_ns  = samples[n * 99 / 100];
    s.p999_ns = samples[n * 999 / 1000];
    s.max_ns  = samples.back();
    s.mops    = (n / 1e6) / total_seconds;
    return s;
}

template <typename QueueT>
static void run_bench(const std::string & label, int n_producers, int items_per_producer) {
    QueueT queue;
    const int total_items = n_producers * items_per_producer;

    std::vector<std::vector<uint32_t>> post_latencies(n_producers);
    for (int p = 0; p < n_producers; ++p) {
        post_latencies[p].reserve(items_per_producer);
    }

    std::vector<uint32_t> handoff_latencies;
    handoff_latencies.reserve(total_items);

    std::atomic<bool> start_gate{false};
    std::atomic<int> producers_done{0};

    // Consumer thread
    std::thread consumer([&]() {
        while (!start_gate.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        int consumed = 0;
        BenchTask task;
        while (consumed < total_items) {
            if (queue.pop(task)) {
                auto now = bench_clock_t::now();
                auto handoff_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - task.t_posted).count();
                handoff_latencies.push_back(handoff_ns > 0 ? (uint32_t)handoff_ns : 0);
                benchmark::DoNotOptimize(task);
                consumed++;
            }
        }
    });

    // Producer threads
    std::vector<std::thread> producers;
    producers.reserve(n_producers);

    for (int p = 0; p < n_producers; ++p) {
        producers.emplace_back([&, p]() {
            while (!start_gate.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto & lat = post_latencies[p];
            for (int i = 0; i < items_per_producer; ++i) {
                BenchTask t;
                t.id = p * items_per_producer + i;
                auto t0 = bench_clock_t::now();
                t.t_posted = t0;
                queue.post(std::move(t));
                auto t1 = bench_clock_t::now();
                auto post_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                lat.push_back(post_ns > 0 ? (uint32_t)post_ns : 0);
            }
            producers_done.fetch_add(1, std::memory_order_release);
        });
    }

    auto t_start = bench_clock_t::now();
    start_gate.store(true, std::memory_order_release);

    for (auto & th : producers) {
        th.join();
    }
    consumer.join();
    auto t_end = bench_clock_t::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Aggregate post latencies
    std::vector<uint32_t> all_post_latencies;
    all_post_latencies.reserve(total_items);
    for (int p = 0; p < n_producers; ++p) {
        all_post_latencies.insert(all_post_latencies.end(), post_latencies[p].begin(), post_latencies[p].end());
    }

    auto post_stats = compute_stats(all_post_latencies, total_sec);
    auto handoff_stats = compute_stats(handoff_latencies, total_sec);

    std::cout << std::left << std::setw(18) << label
              << std::right << std::setw(4) << n_producers
              << std::setw(10) << std::fixed << std::setprecision(2) << post_stats.mops
              << std::setw(10) << std::setprecision(1) << post_stats.mean_ns
              << std::setw(10) << std::setprecision(1) << post_stats.p50_ns
              << std::setw(10) << std::setprecision(1) << post_stats.p99_ns
              << std::setw(10) << std::setprecision(1) << post_stats.max_ns
              << " | "
              << std::setw(10) << std::setprecision(1) << handoff_stats.p50_ns
              << std::setw(10) << std::setprecision(1) << handoff_stats.p99_ns
              << std::setw(10) << std::setprecision(1) << handoff_stats.max_ns
              << "\n";
}

int main() {
    std::cout << "==========================================================================================================\n";
    std::cout << "Microarchitectural Benchmark: Server Queue Post & Handoff Latency Distribution\n";
    std::cout << "Dual Socket AMD EPYC 7453 (56 physical cores, 8 NUMA nodes, NPS4)\n";
    std::cout << "==========================================================================================================\n";
    std::cout << std::left << std::setw(18) << "Queue Type"
              << std::right << std::setw(4) << "Th"
              << std::setw(10) << "Mops/s"
              << std::setw(10) << "PostMean"
              << std::setw(10) << "PostP50"
              << std::setw(10) << "PostP99"
              << std::setw(10) << "PostMax"
              << " | "
              << std::setw(10) << "HandP50"
              << std::setw(10) << "HandP99"
              << std::setw(10) << "HandMax"
              << "  (all latencies in ns)\n";
    std::cout << "----------------------------------------------------------------------------------------------------------\n";

    const std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    const int items_per_thread = 50000;

    for (int th : thread_counts) {
        run_bench<BaselineQueue>("Baseline (Mutex)", th, items_per_thread);
        run_bench<LockfreeMpscQueue>("Lockfree (MPSC)", th, items_per_thread);
        std::cout << "----------------------------------------------------------------------------------------------------------\n";
    }

    return 0;
}
