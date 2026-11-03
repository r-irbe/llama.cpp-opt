#pragma once

#include "server-queue.h"

#include <cstdio>
#include <utility>

// Synchronous legacy control request. Delivery ownership only: returning closed
// or retired does not establish whether a mutating command already executed.
// Callers must inspect status before accessing value and must not blindly replay.
inline server_receive_outcome<server_task_result> server_request_legacy(
        server_queue & tasks, server_response & responses, server_task task) {
    if (task.id == -1) task.id = tasks.get_new_id();
    const int id = task.id; // Stable before ownership moves into the inbound queue.
    if (!responses.add_waiting_task_id(id)) {
        return {server_receive_status::closed, std::nullopt};
    }
    try {
        tasks.post(std::move(task));
        auto outcome = responses.receive(id);
        responses.remove_waiting_task_id(id);
        return outcome;
    } catch (...) {
        // Preserve the original publication/receive exception even if cleanup
        // cannot allocate. This is not an allocation-independent control reserve.
        try { responses.remove_waiting_task_id(id); }
        catch (...) { std::fputs("server control request: delivery retirement failed\n", stderr); }
        throw;
    }
}
