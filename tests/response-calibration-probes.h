#pragma once
#include "server-response-calibration.h"
#include <cstdio>

// All values are sampled logical/exposed capacity indicators, NOT heap bytes.
// String capacity includes SSO; vector storage includes inline element controls.
struct response_calibration_sizes {
    uint64_t string_size = 0, string_capacity = 0;
    uint64_t storage_size = 0, storage_capacity = 0;
    uint64_t probability_outputs = 0, probability_candidates = 0;
    uint64_t nodes = 0;
    bool incomplete = false;
};
response_calibration_sizes response_calibration_measure(const server_task_result &) noexcept;
response_calibration_sizes response_calibration_measure(const json &) noexcept;
void response_calibration_enable(bool enabled) noexcept;
bool response_calibration_complete() noexcept;
void response_calibration_flush(FILE * protocol);
void response_calibration_self_test();
