#include "server-task.h"
#include "server-queue.h"
#include "server-common.h"
#include "server-response-calibration.h"

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
        std::list<server_task_multi> finished;
        {
            std::lock_guard<std::mutex> lock(mutex_tasks);
            auto it = queue_multitasks.begin();
            while (it != queue_multitasks.end()) {
                if (it->subtasks_remaining.empty()) {
                    auto completed = it++;
                    finished.splice(finished.end(), queue_multitasks, completed);
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
    server_task_multi multi;
    multi.id = id_multi;
    std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
    // Prepare outside the task mutex; even allocation-failure cleanup stays out.
    std::list<server_task_multi> prepared;
    prepared.push_back(std::move(multi));
    std::lock_guard<std::mutex> lock(mutex_tasks);
    queue_multitasks.splice(queue_multitasks.end(), prepared);
}


void server_queue::update_multitask(int id_multi, int id_sub, server_task_result& result) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto& multitask : queue_multitasks) {
        if (multitask.id == id_multi && multitask.subtasks_remaining.count(id_sub)) {
            // Commit completion membership only after copying succeeds. Duplicate
            // or late child results must not grow completed-parent retention.
            multitask.results.push_back(result);
            multitask.subtasks_remaining.erase(id_sub);
        }
    }
}


static bool selected_task_id(int candidate, int single, const std::unordered_set<int> * ids) {
    return ids ? ids->find(candidate) != ids->end() : candidate == single;
}

void server_queue::detach_multitasks(int id_task, const std::unordered_set<int> * id_tasks,
                                   std::list<server_task_multi> & retired) {
    std::lock_guard<std::mutex> lock(mutex_tasks);
    for (auto it = queue_multitasks.begin(); it != queue_multitasks.end();) {
        auto current = it++;
        if (selected_task_id(current->id, id_task, id_tasks)) {
            retired.splice(retired.end(), queue_multitasks, current);
        }
    }
}

bool server_response::add_multitask(int id_multi, std::vector<int> & sub_ids) {
    GGML_ASSERT(task_queue != nullptr);
    std::unique_lock<std::mutex> lock(mutex_results);
    if (!running || !waiting_task_ids.count(id_multi)) return false;
    task_queue->add_multitask(id_multi, sub_ids);
    return true;
}

bool server_response::add_waiting_task_id(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_results);
    if (!running) return false;
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int)waiting_task_ids.size());
    waiting_task_ids.insert(id_task);
    return true;
}

bool server_response::add_waiting_tasks(const std::vector<server_task>& tasks) {
    // Allocate every new node before changing shared registration. std::less<int>
    // comparisons do not throw; merge transfers compatible allocator nodes.
    std::set<int> prepared;
    for (const auto & task : tasks) prepared.insert(task.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    if (!running) return false;
    waiting_task_ids.merge(prepared);
    return true;
}

void server_response::remove_waiting_task_id(int id_task) {
    remove_waiting_task_ids_impl(id_task, nullptr);
}

void server_response::remove_waiting_task_ids_impl(int id_task, const std::unordered_set<int> * id_tasks) {
    // Construct holders before locking: any setup failure leaves registrations
    // and queues untouched. Splice only compatible default-allocator lists.
    SERVER_CAL_SCOPE(retirement_observation, "detached_holders_destroyed");
    std::list<server_task_result_ptr> retired;
    std::list<server_task_result> retired_legacy;
    std::list<server_task_multi> retired_parents;
    {
        std::unique_lock<std::mutex> lock(mutex_results);
        if (id_tasks) {
            for (int id : *id_tasks) waiting_task_ids.erase(id);
        } else {
            waiting_task_ids.erase(id_task);
        }
        for (auto it = queue_results.begin(); it != queue_results.end();) {
            auto current = it++;
            if (selected_task_id((*current)->id, id_task, id_tasks)) {
                retired.splice(retired.end(), queue_results, current);
            }
        }
        for (auto it = queue_results_legacy.begin(); it != queue_results_legacy.end();) {
            auto current = it++;
            // A child registered independently retains its own delivery rights.
            // Parent-held copies are detached separately, not by matching id_multi.
            if (selected_task_id(current->id, id_task, id_tasks)) {
                retired_legacy.splice(retired_legacy.end(), queue_results_legacy, current);
            }
        }
        if (task_queue) task_queue->detach_multitasks(id_task, id_tasks, retired_parents);
    }
    condition_results.notify_all();
#ifdef SERVER_RESPONSE_CALIBRATION
    for (const auto & item : retired) {
        SERVER_CAL_SNAPSHOT("detached_result", item.get(), item.get());
    }
#endif
    // Including list nodes, legacy nested backing and completed-child copies.
    // This retires delivery ownership, NOT outstanding backend/slot users.
}


namespace {
// Both families share the same classification and mutex protocol. The selected
// node outlives the lock scope; even moved-from backing retires after unlocking.
// Callers retain their ID selection unchanged throughout this synchronous call.
template<class Queue, class Matches, class Live>
server_receive_outcome<typename Queue::value_type> receive_result(
        Queue & queue, std::mutex & mutex, std::condition_variable & changed,
        const bool & running, Matches matches, Live any_live,
        server_response::receive_deadline deadline) {
    using status = server_receive_status;
    Queue consumed;
    {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            auto found = std::find_if(queue.begin(), queue.end(), matches);
            if (found != queue.end()) {
                consumed.splice(consumed.end(), queue, found);
                break;
            }
            if (!running) return {status::closed, std::nullopt};
            if (!any_live()) return {status::retired, std::nullopt};
            if (deadline == server_response::receive_deadline::max()) {
                changed.wait(lock);
            } else {
                if (server_response::receive_clock::now() >= deadline) {
                    return {status::timeout, std::nullopt};
                }
                changed.wait_until(lock, deadline);
            }
            // Includes a protected final scan on timeout, not just notification.
        }
    }
    SERVER_CAL_RECEIVE(consumed.front()); // Stable exclusive ownership, outside lock.
    return {status::result, std::move(consumed.front())};
}
}

server_receive_outcome<server_task_result> server_response::receive(int id_task, receive_deadline deadline) {
    return receive_result(queue_results_legacy, mutex_results, condition_results, running,
        [id_task](const server_task_result & result) { return result.id == id_task; },
        [&] { return waiting_task_ids.count(id_task) != 0; }, deadline);
}

server_receive_outcome<server_task_result_ptr> server_response::receive(
        const std::unordered_set<int> & id_tasks, receive_deadline deadline) {
    return receive_result(queue_results, mutex_results, condition_results, running,
        [&](const server_task_result_ptr & result) { return id_tasks.count(result->id) != 0; },
        [&] {
            return std::any_of(id_tasks.begin(), id_tasks.end(),
                [&](int id) { return waiting_task_ids.count(id) != 0; });
        }, deadline);
}

server_task_result server_response::recv(int id_task) {
    auto outcome = receive(id_task);
    if (outcome.status != server_receive_status::result) throw server_receive_error(outcome.status);
    assert(outcome.value->id_multi == -1);
    return std::move(*outcome.value);
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int>& id_tasks, int timeout) {
    const auto now = receive_clock::now();
    auto deadline = now;
    if (timeout > 0) {
        const auto seconds = std::chrono::seconds(timeout);
        deadline = receive_deadline::max();
        // Bound conversion before adding; do not compute max()-now, which can
        // overflow when an implementation's steady-clock epoch yields now < 0.
        if (seconds < std::chrono::duration_cast<std::chrono::seconds>(receive_clock::duration::max())) {
            const auto delta = std::chrono::duration_cast<receive_clock::duration>(seconds);
            if (now <= receive_deadline::max() - delta) deadline = now + delta;
        }
    }
    auto outcome = receive(id_tasks, deadline);
    return outcome.value ? std::move(*outcome.value) : nullptr;
}

void server_response::close() {
    {
        std::unique_lock<std::mutex> lock(mutex_results);
        running = false;
    }
    condition_results.notify_all();
}
void server_response::remove_waiting_task_ids(const std::unordered_set<int>& id_tasks) {
    remove_waiting_task_ids_impl(-1, &id_tasks);
}

void server_response::send(server_task_result result) {
    //LOG_VERBOSE("send new result", { {"id_task", result.id} });
    QUE_DBG("send new result, id = %d\n", result.id);
    std::unique_lock<std::mutex> lock(mutex_results);
    if (!running) return;
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
    SERVER_CAL_SNAPSHOT("result_before_publication", result.get(), result.get());

    std::unique_lock<std::mutex> lock(mutex_results);
    if (!running) { SERVER_CAL_STAGE("publication_closed", result->id); return; }
    for (const auto& id_task : waiting_task_ids) {
        if (result->id == id_task) {
            SRV_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            SERVER_CAL_STAGE("queue_publication_transfer", id_task);
            condition_results.notify_all();
            return;
        }
    }
}
