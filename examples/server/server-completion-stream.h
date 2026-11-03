#pragma once

#include "server-response-reader.h"
#include "server-common.h"
#include "server-response-calibration.h"

// Production completion callback shared with callback-level regression tests.
// This seam alone does not exercise parsing, middleware, routes or socket I/O.
inline httplib::ContentProviderWithoutLength server_completion_stream_provider(
        json first_result_json, std::shared_ptr<server_response_reader> rd,
        oaicompat_type oaicompat, bool send_done) {
    return [first_result_json = std::move(first_result_json), rd, oaicompat, send_done SERVER_CAL_CAPTURE_EPISODE]
            (size_t, httplib::DataSink & sink) mutable -> bool {
        SERVER_CAL_RESTORE_EPISODE;
        const auto sse = [oaicompat, &sink](const json & res) {
            if (oaicompat == OAICOMPAT_TYPE_ANTHROPIC) {
                return server_sent_anthropic_event(sink, res);
            } else if (oaicompat == OAICOMPAT_TYPE_RESP) {
                return server_sent_oai_resp_event(sink, res);
            } else {
                return server_sent_event(sink, res);
            }
        };
        const auto finish = [&sink, oaicompat, send_done] {
            const bool default_done = oaicompat != OAICOMPAT_TYPE_ANTHROPIC
                && oaicompat != OAICOMPAT_TYPE_NONE && oaicompat != OAICOMPAT_TYPE_RESP;
            if (default_done || send_done) {
                static const std::string ev_done = "data: [DONE]\n\n";
                SERVER_CAL_SNAPSHOT("done_write_begin", &sink, nullptr, nullptr, &ev_done);
                if (!sink.write(ev_done.data(), ev_done.size())) {
                    SERVER_CAL_STAGE("done_write_failed");
                    return false;
                }
                SERVER_CAL_STAGE("done_write_returned_true");
            }
            sink.done();
            // The HTTP writer observes its own framing errors after this return.
            // Returning false here would misclassify completed delivery as cancel.
            return true;
        };

        if (!first_result_json.empty()) {
            if (!sse(first_result_json)) return false;
            SERVER_CAL_SNAPSHOT("first_json_before_clear", &first_result_json, nullptr, &first_result_json);
            first_result_json.clear();
            SERVER_CAL_SNAPSHOT("first_json_after_clear", &first_result_json, nullptr, &first_result_json);
        }
        // The first result may already be final (or serialize to no events).
        if (!rd->has_next()) return finish();

        SERVER_CAL_SCOPE(callback_observation, "callback_result_and_json_destroyed");
        auto result = rd->next([&sink] { return !sink.is_writable(); });
        if (!result) return false; // Transport stop: no normal final framing.

        SERVER_CAL_CONTEXT(result_context, result.get());
        json res_json = result->to_json();
        SERVER_CAL_SNAPSHOT("callback_result_json_ready", result.get(), result.get(), &res_json);
        if (result->is_error()) {
            if (!server_sent_event(sink, json{{"error", std::move(res_json)}})) return false;
            sink.done(); // Error delivered, not successful inference: no DONE.
            return true;
        }
        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial *>(result.get()) != nullptr
            || dynamic_cast<server_task_result_cmpl_final *>(result.get()) != nullptr);
        if (!sse(res_json)) return false;
        if (!rd->has_next()) return finish();
        return true;
    };
}
