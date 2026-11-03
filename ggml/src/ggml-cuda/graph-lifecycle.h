#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <utility>

// Host-side graph lifetime protocol. Independent contexts may submit together;
// invalidation/registry mutation/teardown exclude all submissions. This does
// not make arbitrary concurrent tensor/model mutation a supported operation.
class ggml_cuda_graph_lifecycle {
    std::shared_mutex lifetime;
    // A retirement waiting for active readers holds admission, so later
    // submissions cannot continually replenish those readers and starve it.
    std::mutex admission;
    std::mutex capture_mutex;
    std::condition_variable capture_finished;
    size_t captures = 0;
    size_t fences_waiting = 0;

public:
    struct submission_guard {
        std::shared_lock<std::shared_mutex> lifetime;
        std::unique_lock<std::mutex> context;
        bool owns_lock() const { return lifetime.owns_lock() && context.owns_lock(); }
    };

    submission_guard submission(std::mutex & context) {
        // Lock order is lifetime -> context -> capture exclusion (when needed).
        std::shared_lock<std::shared_mutex> access;
        {
            std::lock_guard<std::mutex> entering(admission);
            access = std::shared_lock<std::shared_mutex>(lifetime);
        }
        return {std::move(access), std::unique_lock<std::mutex>(context)};
    }
    std::unique_lock<std::shared_mutex> retirement() {
        std::lock_guard<std::mutex> entering(admission);
        return std::unique_lock<std::shared_mutex>(lifetime);
    }
    std::unique_lock<std::shared_mutex> try_retirement() {
        std::unique_lock<std::mutex> entering(admission, std::try_to_lock);
        return entering.owns_lock()
            ? std::unique_lock<std::shared_mutex>(lifetime, std::try_to_lock)
            : std::unique_lock<std::shared_mutex>(lifetime, std::defer_lock);
    }

    void begin_capture() {
        std::unique_lock<std::mutex> lock(capture_mutex);
        capture_finished.wait(lock, [&] { return fences_waiting == 0; });
        ++captures;
    }
    bool try_begin_capture() {
        std::unique_lock<std::mutex> lock(capture_mutex, std::try_to_lock);
        if (!lock.owns_lock() || fences_waiting != 0) { return false; }
        ++captures;
        return true;
    }
    void end_capture() {
        std::lock_guard<std::mutex> lock(capture_mutex);
        if (captures == 0) { std::abort(); } // also enforce the invariant in release builds
        if (--captures == 0) { capture_finished.notify_all(); }
    }

    template <typename Fn>
    decltype(auto) without_capture(Fn && fn) {
        std::unique_lock<std::mutex> lock(capture_mutex);
        ++fences_waiting;
        struct finish_fence {
            ggml_cuda_graph_lifecycle & owner;
            ~finish_fence() {
                --owner.fences_waiting;
                owner.capture_finished.notify_all();
            }
        } finish {*this};
        capture_finished.wait(lock, [&] { return captures == 0; });
        // Retain the lock through the device fence/resource destruction: a new
        // cudaStreamBeginCapture must not race a device-wide synchronization.
        return fn();
    }
    template <typename Fn>
    bool try_without_capture(Fn && fn) {
        std::unique_lock<std::mutex> lock(capture_mutex, std::try_to_lock);
        if (!lock.owns_lock() || captures != 0 || fences_waiting != 0) { return false; }
        fn();
        return true;
    }
};
