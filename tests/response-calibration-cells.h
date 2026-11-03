#pragma once

// Included only by the wire fixture, after its private fault-control declarations.
// Every request attempt, including recovery, has its own bounded episode record.
static void run_response_calibration(httplib::Client & client, int port, FILE * protocol, int repetitions,
        int observed_context, int observed_vocabulary) {
    std::mutex output_mutex;
    std::atomic<unsigned> attempts{0};
    const auto output = [&](json row) {
        row["steady_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        row["thread_role"] = "client_control";
        const auto line = row.dump();
        std::lock_guard<std::mutex> lock(output_mutex);
        CHECK(std::fprintf(protocol, "%s\n", line.c_str()) > 0 && std::fflush(protocol) == 0);
    };
    const auto request = [&](httplib::Client & connection, const std::string & cell,
            const std::string & path, json body, int expected, bool native) {
        const auto id = attempts.fetch_add(1) + 1;
        CHECK(id <= 600);
        output({{"event", "episode_begin"}, {"id", id}, {"cell", cell}});
        auto response = connection.Post(path, {{"X-Response-Calibration", std::to_string(id)}},
            body.dump(), "application/json");
        const bool ok = response && response->status == expected;
        json sample = {{"event", "measurement"}, {"episode", id}, {"cell", cell},
            {"requested_output", body.value("n_predict", body.value("max_tokens", body.value("max_output_tokens", -1)))},
            {"requested_n_probs", body.value("n_probs", 0)}, {"observed_slot_context", observed_context},
            {"observed_vocabulary_entries", observed_vocabulary}, {"response_bytes", response ? response->body.size() : 0},
            {"status", response ? response->status : -1}, {"blocked_write", "unavailable"},
            {"tenant", cell.find("PARENTS-B") == std::string::npos ? "analysis_A" : "analysis_B"},
            {"achieved_parent_concurrency", "unavailable"}, {"tenant_enforcement", false}};
        bool realized_ok = ok;
        const auto candidates = [&](const json & entries) {
            json counts = json::array();
            if (!entries.is_array()) { realized_ok = false; return counts; }
            const int requested = body.value("n_probs", 0);
            for (const auto & token : entries) {
                const json * array = nullptr;
                for (const auto * key : {"probs", "top_probs", "top_logprobs"})
                    if (token.contains(key)) { array = &token.at(key); break; }
                if (!array || !array->is_array()) { realized_ok = false; counts.push_back(nullptr); continue; }
                counts.push_back(array->size());
                if (int(array->size()) != std::min(requested, observed_vocabulary)) realized_ok = false;
            }
            if (requested == 0 ? !entries.empty() : entries.empty()) realized_ok = false;
            return counts;
        };
        if (ok && expected == 200) {
            const bool stream = body.value("stream", false);
            if (stream) {
                check_sse_payloads(response->body);
                realized_ok = response->get_header_value("Content-Type").find("text/event-stream") != std::string::npos;
                const bool done = response->body.find("data: [DONE]\n\n") != std::string::npos;
                if (native) realized_ok = realized_ok && !done;
                else if (path == "/v1/messages") realized_ok = realized_ok && !done
                    && response->body.find("event: message_stop") != std::string::npos;
                else if (path == "/v1/responses") realized_ok = realized_ok && !done
                    && response->body.find("event: response.completed") != std::string::npos;
                else realized_ok = realized_ok && done;
                json entries = json::array();
                int actual = -1;
                bool stopped = false, truncated = false;
                std::istringstream lines(response->body);
                std::string line;
                while (std::getline(lines, line)) {
                    if (line.compare(0, 6, "data: ") || line.substr(6) == "[DONE]") continue;
                    const auto event = json::parse(line.substr(6));
                    if (native) {
                        if (event.contains("tokens_predicted")) actual = event.at("tokens_predicted").get<int>();
                        stopped = stopped || event.value("stop", false);
                        truncated = truncated || event.value("truncated", false);
                        if (event.contains("completion_probabilities"))
                            for (const auto & entry : event.at("completion_probabilities")) entries.push_back(entry);
                    }
                }
                if (native) {
                    sample["actual_output"] = actual;
                    sample["truncated"] = truncated;
                    sample["candidate_counts_per_output"] = candidates(entries);
                    realized_ok = realized_ok && stopped && !truncated && actual == body.value("n_predict", -2);
                } else sample["actual_output"] = "protocol_specific_unavailable";
            } else {
                const auto decoded = json::parse(response->body);
                if (native) {
                    const size_t expected_children = body.at("prompt").is_array() ? body.at("prompt").size() : 1;
                    const auto children = decoded.is_array() ? decoded : json::array({decoded});
                    sample["expected_children"] = expected_children;
                    sample["realized_children"] = children.size();
                    sample["children"] = json::array();
                    realized_ok = children.size() == expected_children;
                    for (size_t index = 0; index < children.size(); ++index) {
                        const auto & one = children.at(index);
                        realized_ok = realized_ok && one.is_object() && !one.contains("error")
                            && !one.value("truncated", false)
                            && one.value("tokens_predicted", -1) == body.value("n_predict", -2);
                        const auto entries = one.contains("completion_probabilities")
                            ? one.at("completion_probabilities") : json::array();
                        sample["children"].push_back({{"index", index},
                            {"actual_output", one.value("tokens_predicted", -1)},
                            {"actual_prompt", one.value("tokens_evaluated", -1)},
                            {"truncated", one.value("truncated", false)},
                            {"probabilities_present", one.contains("completion_probabilities")},
                            {"candidate_counts_per_output", candidates(entries)}});
                    }
                } else {
                    realized_ok = decoded.is_object() && !decoded.contains("error");
                    if (path == "/v1/messages") realized_ok = realized_ok
                        && decoded.value("type", "") == "message" && decoded.at("content").is_array();
                    else if (path == "/v1/responses") realized_ok = realized_ok
                        && decoded.value("object", "") == "response" && decoded.value("status", "") == "completed";
                    else realized_ok = realized_ok && decoded.at("choices").is_array()
                        && decoded.value("object", "") == (path == "/v1/completions" ? "text_completion" : "chat.completion");
                    sample["actual_output"] = "protocol_specific_unavailable";
                }
            }
        }
        if (ok && expected != 200) {
            const auto error = json::parse(response->body);
            realized_ok = error.is_object() && error.contains("error") && error.at("error").is_object();
        }
        output(sample);
        output({{"event", "episode_end"}, {"id", id}, {"ok", realized_ok}});
        CHECK(realized_ok && response_calibration_complete());
    };
    output({{"event", "ready"}, {"trace", "logical_capacity_snapshots"},
        {"provider_copy_lifetimes", "unavailable"}, {"allocator_backing", "unavailable"},
        {"plan", "core_and_protocol_first_stage"}, {"repetitions", repetitions}});
    response_calibration_enable(true);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        json base = {{"prompt", "token3 token4"}, {"n_predict", 2}, {"temperature", 0.0},
            {"seed", 1234}, {"ignore_eos", true}, {"cache_prompt", false}, {"post_sampling_probs", false}};
        for (bool stream : {false, true}) for (int length : {2, 8, 32}) for (int probabilities : {0, 5, 20}) {
            auto body = base;
            body["stream"] = stream; body["n_predict"] = length; body["n_probs"] = probabilities;
            request(client, std::string("N-") + (stream ? "S-" : "P-") + std::to_string(length)
                + "-Q" + std::to_string(probabilities), "/completion", body, 200, true);
        }
        for (int children : {2, 3}) {
            auto body = base;
            body["n_predict"] = 8;
            body["prompt"] = json::array();
            for (int i = 0; i < children; ++i) body["prompt"].push_back("token" + std::to_string(3 + i));
            request(client, "N-P-F" + std::to_string(children), "/completion", body, 200, true);
        }
        {
            httplib::Client peer("127.0.0.1", port);
            peer.set_connection_timeout(2); peer.set_read_timeout(10);
            peer.set_default_headers({{"Authorization", "Bearer fixture-key"}, {"Accept-Encoding", "identity"}});
            auto first = std::async(std::launch::async, [&] {
                request(peer, "N-P-TWO-PARENTS-A", "/completion", base, 200, true);
            });
            request(client, "N-P-TWO-PARENTS-B", "/completion", base, 200, true);
            first.get();
        }
        for (const std::string path : {"/v1/completions", "/v1/chat/completions", "/v1/messages", "/v1/responses"}) {
            json body = {{"model", "fixture-model"}, {"temperature", 0.0}, {"ignore_eos", true}, {"cache_prompt", false}};
            if (path == "/v1/completions") { body["prompt"] = "token3"; body["n_predict"] = 2; }
            else if (path == "/v1/responses") { body["input"] = "token3"; body["max_output_tokens"] = 2; }
            else { body["messages"] = json::array({{{"role", "user"}, {"content", "token3"}}}); body["max_tokens"] = 2; }
            for (bool stream : {false, true}) {
                body["stream"] = stream;
                request(client, path + (stream ? "-S" : "-P"), path, body, 200, false);
            }
        }
        for (const auto status : {GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_ABORTED}) {
            auto body = base; body["prompt"] = json::array({"token3", "token4"});
            failure_status.store(status); countdown.store(2);
            request(client, "FAULT-" + std::to_string(int(status)), "/completion", body,
                status == GGML_STATUS_ALLOC_FAILED ? 503 : 500, true);
            CHECK(countdown.load() == 0 && active_at_failure.load() == 2);
            request(client, "RECOVERY", "/completion", base, 200, true);
        }
    }
    // Slow/no-read, late errors, detached ownership and authenticated tenant
    // enforcement are not fabricated by this first-stage cell selector.
    output({{"event", "measurement"}, {"completed_attempts", attempts.load()},
        {"slow_consumer_cells", "not_yet_instrumented"}, {"retirement_coverage", "partial"},
        {"reservation_enforcement", "unavailable"}, {"production_limits_ready", false}});
}
