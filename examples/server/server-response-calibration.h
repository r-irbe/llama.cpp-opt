#pragma once

// Compile-only observation seam. Only the calibration fixture defines this;
// production targets evaluate neither arguments nor observation helpers.
#ifdef SERVER_RESPONSE_CALIBRATION
#include "server-task.h"

void server_calibration_snapshot(const char * stage, const void * owner,
        const server_task_result * result = nullptr, const json * value = nullptr,
        const std::string * wire = nullptr, int task = -1) noexcept;
void server_calibration_stage(const char * stage, int task = -1) noexcept;
void server_calibration_set_request(const std::string & identity) noexcept;
void server_calibration_associate(const std::vector<server_task> & tasks) noexcept;
uint64_t server_calibration_current_episode() noexcept;
struct server_calibration_episode_context {
    uint64_t previous;
    explicit server_calibration_episode_context(uint64_t episode) noexcept;
    ~server_calibration_episode_context();
};
void server_calibration_receive(const server_task_result_ptr & result) noexcept;
void server_calibration_receive(const server_task_result & result) noexcept;

struct server_calibration_context {
    const server_task_result * previous_result;
    const std::vector<server_task_result_ptr> * previous_batch;
    explicit server_calibration_context(const server_task_result * result) noexcept;
    explicit server_calibration_context(const std::vector<server_task_result_ptr> * batch) noexcept;
    ~server_calibration_context();
};

// Declare BEFORE the owning local(s), so completion follows their destruction.
// This reports this scope's owned objects only, never backend or HTTP quiescence.
struct server_calibration_scope {
    const char * stage;
    uint64_t generation;
    uint64_t episode;
    explicit server_calibration_scope(const char * stage) noexcept;
    ~server_calibration_scope();
};
#define SERVER_CAL_SNAPSHOT(...) server_calibration_snapshot(__VA_ARGS__)
#define SERVER_CAL_REQUEST(...) server_calibration_set_request(__VA_ARGS__)
#define SERVER_CAL_ASSOCIATE(...) server_calibration_associate(__VA_ARGS__)
#define SERVER_CAL_CAPTURE_EPISODE , calibration_episode = server_calibration_current_episode()
#define SERVER_CAL_RESTORE_EPISODE server_calibration_episode_context episode_context(calibration_episode)
#define SERVER_CAL_STAGE(...) server_calibration_stage(__VA_ARGS__)
#define SERVER_CAL_RECEIVE(...) server_calibration_receive(__VA_ARGS__)
#define SERVER_CAL_CONTEXT(name, value) server_calibration_context name(value)
#define SERVER_CAL_SCOPE(name, stage) server_calibration_scope name(stage)
#else
#define SERVER_CAL_SNAPSHOT(...) ((void) 0)
#define SERVER_CAL_REQUEST(...) ((void) 0)
#define SERVER_CAL_ASSOCIATE(...) ((void) 0)
#define SERVER_CAL_CAPTURE_EPISODE
#define SERVER_CAL_RESTORE_EPISODE ((void) 0)
#define SERVER_CAL_STAGE(...) ((void) 0)
#define SERVER_CAL_RECEIVE(...) ((void) 0)
#define SERVER_CAL_CONTEXT(name, value)
#define SERVER_CAL_SCOPE(name, stage)
#endif
