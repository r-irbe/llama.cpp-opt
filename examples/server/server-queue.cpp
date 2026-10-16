#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"

#include "log.h"
#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)


void server_queue::enqueue(posted_task && entry) {
    // Fast path: lock-free ring, unless an overflow episode is in progress
    // (then everything goes to overflow until the loop drains it, so no
    // producer's tasks get reordered).
    if (!overflow_active.load(std::memory_order_acquire)) {
        if (ring->try_push(std::move(entry))) {
            wake();
            return;
        }
        // try_push only moves from `entry` on success, so it is still intact
    }
    {
        std::lock_guard<std::mutex> lock(mutex_overflow);
        overflow_active.store(true, std::memory_order_release);
        overflow.push_back(std::move(entry));
    }
    wake();
}

int server_queue::post(server_task task) {
    if (task.id == -1) {
        task.id = id.fetch_add(1, std::memory_order_relaxed);
        QUE_DBG("new task, id = %d\n", task.id);
    }
    const int task_id = task.id;
    enqueue(posted_task{ std::move(task), false, false });
    return task_id;
}

void server_queue::cleanup_pending_task(int id_target) {
    // loop thread only
    auto rm_func = [id_target](const server_task& task) {
        return task.id == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(), queue_tasks.end(), rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
        queue_tasks_deferred.end());
}

// multi-task version of post()
int server_queue::post(std::vector<server_task>&& tasks, bool front) {
    for (auto& task : tasks) {
        if (task.id == -1) {
            task.id = id.fetch_add(1, std::memory_order_relaxed);
        }
        const bool is_cancel = task.type == SERVER_TASK_TYPE_CANCEL;
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int)tasks.size(), front);
        enqueue(posted_task{ std::move(task), front, is_cancel });
    }
    return 0;
}

size_t server_queue::drain_inbound() {
    size_t n = 0;
    posted_task entry;
    auto apply = [&](posted_task && e) {
        // Same effect as the old post(): cleanup happened at post time on the
        // tasks already queued; everything posted before this entry has been
        // drained already because drain order == publish order per producer.
        if (e.cleanup) {
            cleanup_pending_task(e.task.id_target);
        }
        if (e.front) {
            queue_tasks.push_front(std::move(e.task));
        } else {
            queue_tasks.push_back(std::move(e.task));
        }
        ++n;
    };

    // ring first: anything in overflow was published after the ring filled
    while (ring->try_pop(entry)) {
        apply(std::move(entry));
    }
    if (overflow_active.load(std::memory_order_acquire)) {
        std::deque<posted_task> spill;
        {
            std::lock_guard<std::mutex> lock(mutex_overflow);
            spill.swap(overflow);
            // Ring may have received entries from producers that saw the flag
            // as false before it was set; drain them before the spill so they
            // keep their (earlier) position.
            while (ring->try_pop(entry)) {
                apply(std::move(entry));
            }
            overflow_active.store(false, std::memory_order_release);
        }
        for (auto & e : spill) {
            apply(std::move(e));
        }
    }
    return n;
}

void server_queue::defer(server_task&& task) {
    // loop thread only (called from process_single_task)
    queue_tasks_deferred.push_back(std::move(task));
}

int server_queue::get_new_id() {
    const int new_id = id.fetch_add(1, std::memory_order_relaxed);
    QUE_DBG("new task, id = %d\n", new_id);
    return new_id;
}

void server_queue::notify_slot_changed() {
    // loop thread only (called from update_slots -> release_slots)
    for (auto& task : queue_tasks_deferred) {
        queue_tasks.push_back(std::move(task));
    }
    queue_tasks_deferred.clear();
}

void server_queue::on_new_task(std::function<void(server_task&&)> callback) {
    callback_new_task = std::move(callback);
}


void server_queue::start_loop() {
    running.store(true, std::memory_order_release);

    while (true) {
        LOG_VERBOSE("new task may arrive", {});

        drain_inbound();
        while (!queue_tasks.empty()) {
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            callback_new_task(std::move(task));
            // pick up tasks posted while processing (incl. cancels) in order
            drain_inbound();
        }

        LOG_VERBOSE("update_multitasks", {});

        // check if we have any finished multitasks (extract under lock,
        // invoke callbacks outside it)
        std::vector<server_task_multi> finished;
        {
            std::lock_guard<std::mutex> lock(mutex_tasks);
            auto it = queue_multitasks.begin();
            while (it != queue_multitasks.end()) {
                if (it->subtasks_remaining.empty()) {
                    finished.push_back(std::move(*it));
                    it = queue_multitasks.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (auto & multitask : finished) {
            callback_finish_multitask(multitask);
        }

        // all tasks in the current loop is processed, slots data is now ready
        LOG_VERBOSE("callback_update_slots", {});

        callback_update_slots();

        LOG_VERBOSE("wait for new task", {});
        // Lost-wakeup-free wait: read the signal, re-check for work, then
        // futex-wait on the observed value. Producers publish before bumping.
        while (true) {
            const uint32_t observed = signal.load(std::memory_order_acquire);
            if (drain_inbound() > 0 || !queue_tasks.empty()) {
                break;
            }
            if (!running.load(std::memory_order_acquire)) {
                LOG_VERBOSE("ending start_loop", {});
                return;
            }
            signal.wait(observed, std::memory_order_acquire);
        }
    }
}

void server_queue::add_multitask(int id_multi, std::vector<int>& sub_ids) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    server_task_multi multi;
    multi.id = id_multi;
    std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
    queue_multitasks.push_back(multi);
}


void server_queue::update_multitask(int id_multi, int id_sub, server_task_result& result) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto& multitask : queue_multitasks) {
        if (multitask.id == id_multi) {
            multitask.subtasks_remaining.erase(id_sub);
            multitask.results.push_back(result);
        }
    }
}


void server_response::add_waiting_task_id(int id_task) {
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_tasks(const std::vector<server_task>& tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto& task : tasks) {
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int)waiting_task_ids.size());
        waiting_task_ids.insert(task.id);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    //LOG_VERBOSE("remove waiting for task id", { {"id_task", id_task} });
    QUE_DBG("remove waiting for task id, id = %d\n", id_task);
    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
}


server_task_result server_response::recv(int id_task) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&] {
            return !queue_results_legacy.empty();
            });

        for (int i = 0; i < (int)queue_results_legacy.size(); i++) {
            if (queue_results_legacy[i].id == id_task) {
                assert(queue_results_legacy[i].id_multi == -1);
                server_task_result res = queue_results_legacy[i];
                queue_results_legacy.erase(queue_results_legacy.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

// same as recv(), but have timeout in seconds
// if timeout is reached, nullptr is returned
server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int)queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            SRV_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}
void server_response::remove_waiting_task_ids(const std::unordered_set<int>& id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto& id_task : id_tasks) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int)waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

void server_response::send(server_task_result result) {
    //LOG_VERBOSE("send new result", { {"id_task", result.id} });
    QUE_DBG("send new result, id = %d\n", result.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        // LOG_TEE("waiting task id %i \n", id_task);
        // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
        if (result.id_multi == id_task) {
            //LOG_VERBOSE("callback_update_multitask", { {"id_task", id_task} });
            QUE_DBG("callback_update_multitask, id = %d\n", id_task);
            callback_update_multitask(id_task, result.id, result);
            continue;
        }

        if (result.id == id_task) {
            //LOG_VERBOSE("queue_results_legacy.push_back", { {"id_task", id_task} });
            QUE_DBG("queue_results.push_back, id = %d\n", id_task);
            queue_results_legacy.push_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

// Send a new result to a waiting id_task
void server_response::send(server_task_result_ptr&& result) {
    SRV_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto& id_task : waiting_task_ids) {
        if (result->id == id_task) {
            SRV_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}
