#pragma once
#include "server-task.h"
#include "mpsc-ring.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
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

    std::vector<server_task_multi> queue_multitasks;

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
};

struct server_response {
    typedef std::function<void(int, int, server_task_result&)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;

    bool running = true;
    // for keeping track of all tasks waiting for the result
    std::set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::vector<server_task_result> queue_results_legacy;

    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_tasks(const std::vector<server_task>& tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    void remove_waiting_task_ids(const std::unordered_set<int>& id_tasks);

    // This function blocks the thread until there is a response for this id_task
    server_task_result recv(int id_task);

    // same as recv(), but have timeout in seconds
// if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout);

    // Register the function to update multitask
    void on_multitask_update(callback_multitask_t callback) {
        callback_update_multitask = std::move(callback);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result result);

    void send(server_task_result_ptr&& result);

    // terminate the waiting loop
    void terminate() {
        running = false;
        condition_results.notify_all();
    };
};
