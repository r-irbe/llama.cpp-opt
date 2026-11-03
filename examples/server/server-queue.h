#pragma once
#include "server-task.h"
#include "mpsc-ring.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <exception>
#include <optional>
#include <deque>
#include <memory>
#include <list>
#include <mutex>
#include <unordered_set>

struct server_task_multi {
    int id = -1;

    std::set<int> subtasks_remaining;
    std::vector<server_task_result> results;
};


struct server_queue {
    // Entry posted by producers. Front-insertion and pending-task cleanup for
    // cancel tasks are applied by the loop thread when it drains the ring,
    // which preserves the ordering semantics of the previous locked deque.
    struct posted_task {
        server_task task;
        bool        front   = false;
        bool        cleanup = false;
    };

    std::atomic<int>  id{0};
    std::atomic<bool> running{false};

    // Lock-free inbound path (any thread may post, including the loop thread).
    static constexpr size_t RING_CAPACITY = 256;
    std::unique_ptr<common_mpsc_ring<posted_task>> ring =
        std::make_unique<common_mpsc_ring<posted_task>>(RING_CAPACITY);

    // Overflow path when the ring is full. Never blocks the poster on the
    // consumer (the loop thread posts to its own queue). While active, all
    // posts go here so a single producer's tasks are never reordered.
    std::mutex              mutex_overflow;
    std::deque<posted_task> overflow;
    std::atomic<bool>       overflow_active{false};

    // Wake-up signal: bumped after every publish; the loop waits on it (futex).
    std::atomic<uint32_t>   signal{0};

    // Loop-thread-only state (no locking required).
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    // Nodes can retire without moving/destroying nested payload under the lock.
    std::list<server_task_multi> queue_multitasks;

    std::mutex mutex_tasks; // guards queue_multitasks

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(server_task_multi &)> callback_finish_multitask;
    std::function<void(void)>                callback_update_slots;


    // Add a new task to the end of the queue
    int post(server_task task);

    int post(std::vector<server_task>&& tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task&& task);

    // Get the next id for creating anew task
    int get_new_id();

    // Register function to process a new task
    void on_new_task(std::function<void(server_task&&)> callback);

    // Register function to process a multitask when it is finished
    void on_finish_multitask(std::function<void(server_task_multi&)> callback) {
        callback_finish_multitask = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed
    void notify_slot_changed();

    // end the start_loop routine
    void terminate() {
        running.store(false, std::memory_order_release);
        wake();
    }

    // publish one entry (lock-free fast path, overflow fallback)
    void enqueue(posted_task && entry);

    // bump the wake signal after a publish
    void wake() {
        signal.fetch_add(1, std::memory_order_release);
        signal.notify_one();
    }

    // loop thread: move everything published so far into queue_tasks
    // applying front-insertion and cancel cleanup; returns #entries drained
    size_t drain_inbound();

    // remove pending tasks matching id_target (loop thread only)
    void cleanup_pending_task(int id_target);

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     */
    void start_loop();

    //
    // functions to manage multitasks
    //

    // add a multitask by specifying the id of all subtask (subtask is a server_task)
    void add_multitask(int id_multi, std::vector<int>& sub_ids);

    // updatethe remaining subtasks, while appending results to multitask
    void update_multitask(int id_multi, int id_sub, server_task_result& result);

private:
    friend struct server_response;
    // Caller owns the retirement holder and destroys it after releasing BOTH
    // response and task mutexes. This does not cancel/drain child backend work.
    void detach_multitasks(int id_task, const std::unordered_set<int> * id_tasks,
                          std::list<server_task_multi> & retired);
};

enum class server_receive_status { result, closed, retired, timeout };

template<class T> struct server_receive_outcome {
    server_receive_status status;
    std::optional<T> value;
};

// Compatibility failure with a fixed message: never fabricate a successful
// value on closure/retirement. Throwing still needs exception-runtime storage;
// this is not an allocation-independent emergency terminal channel.
struct server_receive_error : std::exception {
    server_receive_status status;
    explicit server_receive_error(server_receive_status status) : status(status) {}
    const char * what() const noexcept override { return "server receive ended without a result"; }
};

struct server_response {
    // The associated queue must outlive this response (as in server_context).
    explicit server_response(server_queue * tasks = nullptr) : task_queue(tasks) {}

    typedef std::function<void(int, int, server_task_result&)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;

    // for keeping track of all tasks waiting for the result
    std::set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::list<server_task_result_ptr> queue_results;

    std::list<server_task_result> queue_results_legacy;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    bool add_waiting_task_id(int id_task);

    // All-or-nothing registration; false after closure, no partial insertion on
    // allocation failure. Callers must register successfully BEFORE posting.
    bool add_waiting_tasks(const std::vector<server_task>& tasks);

    // Create legacy parent tracking only while its delivery registration is live.
    // Uses the same response -> task lock order as legacy result publication.
    bool add_multitask(int id_multi, std::vector<int> & sub_ids);

    // Retire delivery rights and queued ownership, not backend/slot ownership.
    // Destroy detached payloads outside both mutexes; repeated removal is safe.
    // IDs must not be reused while an older producer/cancel can still refer to them.
    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    void remove_waiting_task_ids(const std::unordered_set<int>& id_tasks);

    using receive_clock = std::chrono::steady_clock;
    using receive_deadline = receive_clock::time_point;
    // Match > closed > all selected registrations retired > timeout. A ready
    // match wins at zero/expired deadline; unrelated wakes never renew it.
    server_receive_outcome<server_task_result> receive(
        int id_task, receive_deadline deadline = receive_deadline::max());
    server_receive_outcome<server_task_result_ptr> receive(
        const std::unordered_set<int> & id_tasks, receive_deadline deadline = receive_deadline::max());

    // Compatibility adapters only: production consumers must migrate to explicit
    // outcomes before receive-to-wire stage acceptance.
    // Value receive throws server_receive_error on non-result; pointer polling
    // returns null for any non-result (not just timeout).
    server_task_result recv(int id_task);
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout);

    // Register the function to update multitask
    void on_multitask_update(callback_multitask_t callback) {
        callback_update_multitask = std::move(callback);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result result);

    void send(server_task_result_ptr&& result);

    // Seal new delivery admission/publication and wake current/future receivers.
    // Already queued results can drain. Not backend cancellation/quiescence.
    void close();
    void terminate() { close(); } // Source-compatible nonfatal alias.

private:
    bool running = true; // Protected by mutex_results, including closure.
    server_queue * task_queue;
    void remove_waiting_task_ids_impl(int id_task, const std::unordered_set<int> * id_tasks);
};
