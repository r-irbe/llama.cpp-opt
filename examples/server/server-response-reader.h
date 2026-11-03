#pragma once

#include "server-context.h"
#include "server-response-calibration.h"

#include <cstdio>

inline constexpr int HTTP_POLLING_SECONDS = 1;

// Shared production response reader. Host tests exercise this same reader;
// they do not substitute for production parsing, routes or HTTP/SSE acceptance.
// Reader methods are serialized by their owner; this is not a concurrent-reader API.
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_context & ctx_server;
    size_t received_count = 0;
    bool cancelled = false;

    explicit server_response_reader(server_context & server) : ctx_server(server) {}
    ~server_response_reader() {
        try { stop(); }
        catch (...) { report_cancel_failure(); }
    }

    void post_tasks(std::vector<server_task> && tasks) {
        GGML_ASSERT(id_tasks.empty() && !post_attempted && !cancelled);
        id_tasks = server_task::get_list_id(tasks);
        outstanding = id_tasks;
        // Closure leaves outstanding IDs for next() to report as terminal error,
        // but MUST NOT publish work that has no delivery registration.
        if (!ctx_server.queue_results.add_waiting_tasks(tasks)) return;
        SERVER_CAL_ASSOCIATE(tasks);
        post_attempted = true; // Includes partial publication if post throws.
        try { ctx_server.queue_tasks.post(std::move(tasks)); }
        catch (...) {
            try { stop(); }
            catch (...) { report_cancel_failure(); }
            throw;
        }
    }

    bool has_next() const { return !cancelled && !outstanding.empty(); }

    // Null means actual transport stop (or an already-exhausted reader), never
    // queue closure/retirement. Those become explicit ordinary error results.
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (has_next()) {
            if (should_stop()) {
                stop();
                return nullptr;
            }
            const auto deadline = server_response::receive_clock::now()
                + std::chrono::seconds(HTTP_POLLING_SECONDS);
            auto outcome = ctx_server.queue_results.receive(outstanding, deadline);
            if (outcome.status == server_receive_status::timeout) continue;
            if (outcome.status != server_receive_status::result) {
                auto result = make_error(outcome.status == server_receive_status::closed
                    ? "response delivery closed" : "response delivery retired");
                stop();
                return result;
            }
            auto result = std::move(*outcome.value);
            if (result->is_error()) {
                stop();
                return result;
            }
            if (result->is_stop()) {
                outstanding.erase(result->id);
                ++received_count;
            }
            return result;
        }
        return nullptr;
    }

    struct batch_response {
        bool is_terminated = false; // Actual transport stop, not server closure.
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error;
    };

    batch_response wait_for_all(const std::function<bool()> & should_stop) {
        batch_response batch;
        batch.results.resize(id_tasks.size());
        while (has_next()) {
            auto result = next(should_stop);
            if (!result) {
                batch.is_terminated = true;
                return batch;
            }
            if (result->is_error()) {
                batch.error = std::move(result);
                return batch;
            }
            const size_t index = result->get_index();
            if (!result->is_stop() || index >= batch.results.size() || batch.results[index]) {
                batch.error = make_error("invalid aggregate response", ERROR_TYPE_SERVER);
                stop();
                return batch;
            }
            batch.results[index] = std::move(result);
        }
        return batch;
    }

    void stop() {
        // Retire delivery even if cancellation construction/publication fails.
        // Outstanding backend/slot ownership remains with the owner thread.
        ctx_server.queue_results.remove_waiting_task_ids(id_tasks);
        cancelled = true;
        if (!post_attempted || cancellation_posted || outstanding.empty()) return;
        std::vector<server_task> cancels;
        cancels.reserve(outstanding.size());
        for (int id : outstanding) {
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id;
            cancels.push_back(std::move(task));
        }
        // Mark only after the entire post succeeds. A retry after partial failure
        // may duplicate cancels; IDs cannot be reused and owner cancel is idempotent.
        ctx_server.queue_tasks.post(std::move(cancels), true);
        cancellation_posted = true;
    }

private:
    std::unordered_set<int> outstanding;
    bool post_attempted = false;
    bool cancellation_posted = false;

    static server_task_result_ptr make_error(const char * message, error_type type = ERROR_TYPE_UNAVAILABLE) {
        auto result = std::make_unique<server_task_result_error>();
        result->err_msg = message;
        result->err_type = type;
        result->error = true;
        return result;
    }
    static void report_cancel_failure() noexcept {
        // Do not throw from destruction or pretend an allocation-independent
        // control reserve exists. A future owner/control path must close this gap.
        std::fputs("server response reader: cancellation enqueue failed; owner work may remain live\n", stderr);
    }
};
