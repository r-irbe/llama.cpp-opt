// Real single-instance application, routes/middleware and loopback sockets.
// Synthetic CPU weights; faults exist only in this executable's backend shim.
#include "server-app.h"
#include "server-context.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"
#include "response-calibration-probes.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <sstream>
#include <thread>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::abort(); } } while (0)

struct model_file {
    std::string path;
    model_file() {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/llama-wire-XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        int fd = mkstemp(name.data());
        CHECK(fd >= 0 && close(fd) == 0);
        path = name.data();
        auto * file = gguf_init_empty();
        auto * tensors = ggml_init({1024 * 1024, nullptr, false});
        CHECK(file && tensors);
        gguf_set_val_str(file, "general.architecture", "llama");
        for (const auto & item : std::vector<std::pair<const char *, uint32_t>>{
                {"llama.block_count", 1}, {"llama.context_length", 192},
                {"llama.embedding_length", 32}, {"llama.feed_forward_length", 64},
                {"llama.attention.head_count", 4}, {"llama.attention.head_count_kv", 4}})
            gguf_set_val_u32(file, item.first, item.second);
        gguf_set_val_f32(file, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_str(file, "tokenizer.ggml.model", "llama");
        gguf_set_val_u32(file, "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(file, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(file, "tokenizer.ggml.eos_token_id", 2);
        std::vector<std::string> words {"<unk>", "<s>", "</s>"};
        for (int i = 3; i < 32; ++i) words.push_back("token" + std::to_string(i));
        // Real text tokenization needs the full SPM byte-fallback vocabulary.
        // The queue-only fixture's tiny token-ID vocabulary cannot tokenize text.
        for (int i = 0; i < 256; ++i) {
            char byte[7];
            std::snprintf(byte, sizeof(byte), "<0x%02X>", i);
            words.emplace_back(byte);
        }
        std::vector<const char *> tokens;
        for (const auto & word : words) tokens.push_back(word.c_str());
        gguf_set_arr_str(file, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
        std::vector<int32_t> types(words.size(), LLAMA_TOKEN_TYPE_NORMAL);
        types[0] = LLAMA_TOKEN_TYPE_UNKNOWN;
        types[1] = types[2] = LLAMA_TOKEN_TYPE_CONTROL;
        std::fill(types.begin() + 32, types.end(), LLAMA_TOKEN_TYPE_BYTE);
        gguf_set_arr_data(file, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        auto add = [&](const std::string & name, int64_t n0, int64_t n1 = 0) {
            auto * t = n1 ? ggml_new_tensor_2d(tensors, GGML_TYPE_F32, n0, n1)
                         : ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n0);
            ggml_set_name(t, name.c_str());
            std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), n1 ? 0.01f : 1.0f);
            gguf_add_tensor(file, t);
        };
        add("token_embd.weight", 32, words.size()); add("output_norm.weight", 32); add("output.weight", 32, words.size());
        add("blk.0.attn_norm.weight", 32);
        for (const char * p : {"q", "k", "v", "output"}) add(std::string("blk.0.attn_") + p + ".weight", 32, 32);
        add("blk.0.ffn_norm.weight", 32);
        add("blk.0.ffn_gate.weight", 32, 64); add("blk.0.ffn_up.weight", 32, 64); add("blk.0.ffn_down.weight", 64, 32);
        gguf_write_to_file(file, path.c_str(), false);
        gguf_free(file); ggml_free(tensors);
    }
    ~model_file() { unlink(path.c_str()); }
};

static server_context * observed;
static decltype(ggml_backend_i::graph_compute) original_compute;
static std::atomic<int> countdown{0}, failure_status{GGML_STATUS_FAILED}, failures{0}, active_at_failure{0};
static std::atomic<bool> late_failure{false}, hold_for_disconnect{false}, seal_delivery{false};
static std::mutex gate_mutex;
static std::condition_variable gate_cv;
static bool client_saw_payload = false;

static ggml_status GGML_CALL fault_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    bool inject = false;
    if (late_failure.load() || hold_for_disconnect.load()) {
        bool generated = false;
        for (const auto & slot : observed->slots) generated |= slot.is_processing() && slot.command == SLOT_COMMAND_NONE && slot.n_decoded > 0;
        if (generated) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            CHECK(gate_cv.wait_for(lock, std::chrono::seconds(5), [] { return client_saw_payload; }));
            inject = late_failure.exchange(false);
            hold_for_disconnect.store(false);
        }
    }
    const int remaining = countdown.load();
    if (remaining > 0 && countdown.fetch_sub(1) == 1) inject = true;
    if (inject) {
        int active = 0;
        for (const auto & slot : observed->slots) active += slot.is_processing();
        active_at_failure.store(active);
        ++failures;
        if (seal_delivery.exchange(false)) observed->queue_results.close();
        return static_cast<ggml_status>(failure_status.load());
    }
    return original_compute(backend, graph);
}

static void check_sse_payloads(const std::string & body) {
    std::istringstream lines(body);
    std::string line;
    int payloads = 0;
    while (std::getline(lines, line)) {
        if (line.compare(0, 6, "data: ") != 0 || line.substr(6) == "[DONE]") continue;
        const auto payload = json::parse(line.substr(6));
        CHECK(payload.is_object() && !payload.contains("error"));
        ++payloads;
    }
    CHECK(payloads > 0);
}

#include "response-calibration-cells.h"

int main(int argc, char ** argv_in) {
    if (argc == 2 && std::strcmp(argv_in[1], "--probe-self-test") == 0) {
        response_calibration_self_test();
        return 0;
    }
    const bool calibrate = argc == 2 && std::strcmp(argv_in[1], "--calibrate") == 0;
    CHECK(argc == 1 || calibrate);
    FILE * protocol = nullptr;
    if (calibrate) {
        // Existing server diagnostics use stdout. Keep them on the runner's
        // bounded stderr pipe; the duplicated original stdout is JSONL only.
        const int fd = dup(STDOUT_FILENO);
        CHECK(fd >= 0);
        protocol = fdopen(fd, "w");
        CHECK(protocol && dup2(STDERR_FILENO, STDOUT_FILENO) >= 0);
    }
    // The real entry parses server env after argv. Isolate only this test process
    // from caller model/network overrides; never print or retain their values.
    std::vector<std::string> overrides;
    for (char ** entry = environ; *entry; ++entry) {
        if (std::strncmp(*entry, "LLAMA_ARG_", 10) == 0) {
            const char * end = std::strchr(*entry, '=');
            if (end) overrides.emplace_back(*entry, static_cast<size_t>(end - *entry));
        }
    }
    for (const auto & key : overrides) CHECK(unsetenv(key.c_str()) == 0);
    CHECK(unsetenv("LLAMA_HOTSWAP_ENABLED") == 0);
    model_file model;
    server_context server;
    std::vector<std::string> args {"server-wire", "-m", model.path, "--host", "127.0.0.1", "--port", "0",
        "-ngl", "0", "-c", "192", "-np", "3", "-b", "16", "-ub", "2", "-t", "1", "-tb", "1",
        "--threads-http", "4", "--no-warmup", "--chat-template", "chatml", "--api-key", "fixture-key",
        "--alias", "fixture-model"};
    std::vector<char *> argv;
    for (auto & arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    std::promise<int> bound;
    auto port = bound.get_future();
    std::atomic<int> exit_code{-1};
    int observed_context = 0, observed_vocabulary = 0;
    std::thread app([&] {
        exit_code.store(server_run(static_cast<int>(args.size()), argv.data(), server, [&](int p) {
            observed = &server;
            observed_vocabulary = llama_n_vocab(server.model);
            CHECK(!server.slots.empty());
            observed_context = server.slots.front().n_ctx;
            for (const auto & slot : server.slots) CHECK(slot.n_ctx == observed_context);
            original_compute = server.ctx->backend_cpu->iface.graph_compute;
            server.ctx->backend_cpu->iface.graph_compute = fault_compute;
            bound.set_value(p);
        }));
    });
    CHECK(port.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    const int p = port.get();
    httplib::Client client("127.0.0.1", p);
    client.set_connection_timeout(2); client.set_read_timeout(10);
    client.set_default_headers({{"Authorization", "Bearer fixture-key"}, {"Accept-Encoding", "identity"}});
    if (calibrate) {
        CHECK(observed_context > 0 && observed_vocabulary == 288);
        run_response_calibration(client, p, protocol, 10, observed_context, observed_vocabulary);
        server.queue_tasks.terminate();
        app.join();
        CHECK(exit_code.load() == 0);
        server.ctx->backend_cpu->iface.graph_compute = original_compute;
        observed = nullptr;
        response_calibration_flush(protocol);
        CHECK(response_calibration_complete() && std::fclose(protocol) == 0);
        return 0;
    }
    auto health = client.Get("/health");
    CHECK(health && health->status == 200);
    httplib::Client anonymous("127.0.0.1", p);
    anonymous.set_read_timeout(5);
    auto denied = anonymous.Post("/completion", "{}", "application/json");
    CHECK(denied && denied->status == 401);
    auto invalid = client.Post("/completion", "{", "application/json");
    CHECK(invalid && invalid->status >= 400 && json::parse(invalid->body).contains("error"));

    json request = {{"prompt", "token3 token4"}, {"n_predict", 2}, {"temperature", 0.0},
                    {"ignore_eos", true}, {"cache_prompt", false}};
    auto healthy = client.Post("/completion", request.dump(), "application/json");
    if (healthy && healthy->status != 200) std::fprintf(stderr, "healthy response: %s\n", healthy->body.c_str());
    CHECK(healthy && healthy->status == 200 && json::parse(healthy->body).contains("content"));
    auto fanout = request;
    fanout["prompt"] = json::array({"token3 token4", "token5 token6"});
    auto batch = client.Post("/completion", fanout.dump(), "application/json");
    CHECK(batch && batch->status == 200);
    const auto children = json::parse(batch->body);
    CHECK(children.is_array() && children.size() == 2);
    for (const auto & child : children) CHECK(child.contains("content") && !child.contains("error"));

    for (auto status : {GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_ABORTED}) {
        failure_status.store(status); countdown.store(2);
        const int before = failures.load();
        auto failed = client.Post("/completion", fanout.dump(), "application/json");
        CHECK(failed && failed->status == (status == GGML_STATUS_ALLOC_FAILED ? 503 : 500));
        CHECK(json::parse(failed->body).contains("error") && failures.load() == before + 1);
        CHECK(active_at_failure.load() == 2 && countdown.load() == 0);
        auto recovered = client.Post("/completion", request.dump(), "application/json");
        CHECK(recovered && recovered->status == 200);
    }
    auto streamed = request;
    streamed["stream"] = true;
    auto stream = client.Post("/v1/completions", streamed.dump(), "application/json");
    CHECK(stream && stream->status == 200 && stream->body.find("data: [DONE]\n\n") != std::string::npos);
    CHECK(stream->get_header_value("Content-Type").find("text/event-stream") != std::string::npos);
    auto native_stream = client.Post("/completion", streamed.dump(), "application/json");
    CHECK(native_stream && native_stream->status == 200 && native_stream->body.find("\"stop\":true") != std::string::npos);
    CHECK(native_stream->body.find("[DONE]") == std::string::npos);
    check_sse_payloads(stream->body);
    check_sse_payloads(native_stream->body);
    for (const std::string path : {"/v1/chat/completions", "/v1/messages", "/v1/responses"}) {
        json body = {{"model", "fixture-model"}, {"temperature", 0.0}, {"ignore_eos", true}};
        if (path == "/v1/responses") {
            body["input"] = "token3"; body["max_output_tokens"] = 2;
        } else {
            body["messages"] = json::array({{{"role", "user"}, {"content", "token3"}}});
            body["max_tokens"] = 2;
        }
        auto plain = client.Post(path, body.dump(), "application/json");
        if (plain && plain->status != 200) std::fprintf(stderr, "%s: %s\n", path.c_str(), plain->body.c_str());
        CHECK(plain && plain->status == 200);
        const auto value = json::parse(plain->body);
        CHECK(value.is_object() && !value.contains("error"));
        if (path == "/v1/chat/completions") CHECK(value.at("choices").is_array());
        else if (path == "/v1/messages") CHECK(value.at("type") == "message" && value.at("content").is_array());
        else CHECK(value.at("object") == "response" && value.at("status") == "completed");
        body["stream"] = true;
        auto events = client.Post(path, body.dump(), "application/json");
        CHECK(events && events->status == 200);
        CHECK(events->get_header_value("Content-Type").find("text/event-stream") != std::string::npos);
        check_sse_payloads(events->body);
        if (path == "/v1/chat/completions") CHECK(events->body.find("[DONE]") != std::string::npos);
        else {
            CHECK(events->body.find("[DONE]") == std::string::npos);
            CHECK(events->body.find(path == "/v1/messages" ? "event: message_stop" : "event: response.completed") != std::string::npos);
        }
    }
    failure_status.store(GGML_STATUS_FAILED); countdown.store(1);
    auto first_error = client.Post("/v1/completions", streamed.dump(), "application/json");
    CHECK(first_error && first_error->status == 500 && json::parse(first_error->body).contains("error"));

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        client_saw_payload = false;
    }
    late_failure.store(true);
    streamed["n_predict"] = 8;
    std::string chunks;
    auto late = client.Post("/v1/completions", {}, streamed.dump(), "application/json",
        [&](const char * data, size_t size) {
            chunks.append(data, size);
            { std::lock_guard<std::mutex> lock(gate_mutex); client_saw_payload = true; }
            gate_cv.notify_one();
            return true;
        });
    CHECK(late && late->status == 200 && chunks.find("\"error\"") != std::string::npos);
    CHECK(chunks.find("[DONE]") == std::string::npos && !late_failure.load());

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        client_saw_payload = false;
    }
    hold_for_disconnect.store(true);
    std::future<httplib::Result> peer_result;
    bool disconnected_after_payload = false;
    auto disconnected = client.Post("/v1/completions", {}, streamed.dump(), "application/json",
        [&](const char *, size_t size) {
            CHECK(size > 0);
            peer_result = std::async(std::launch::async, [&] {
                httplib::Client peer("127.0.0.1", p);
                peer.set_read_timeout(10);
                peer.set_default_headers({{"Authorization", "Bearer fixture-key"}});
                return peer.Post("/completion", request.dump(), "application/json");
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            bool registered_peer = false;
            while (!registered_peer && std::chrono::steady_clock::now() < deadline) {
                { std::lock_guard<std::mutex> lock(server.queue_results.mutex_results);
                  registered_peer = server.queue_results.waiting_task_ids.size() >= 2; }
                if (!registered_peer) std::this_thread::yield();
            }
            CHECK(registered_peer);
            { std::lock_guard<std::mutex> lock(gate_mutex); client_saw_payload = true; }
            gate_cv.notify_one();
            disconnected_after_payload = true;
            return false;
        });
    CHECK(!disconnected && disconnected_after_payload);
    auto peer = peer_result.get();
    CHECK(peer && peer->status == 200 && json::parse(peer->body).contains("content"));

    // Seal from the owner during a selected compute failure; real HTTP readers
    // must receive a terminal response. Not a shutdown grace/quiescence test.
    countdown.store(1); seal_delivery.store(true);
    auto closed = client.Post("/completion", request.dump(), "application/json");
    CHECK(closed && closed->status == 503 && json::parse(closed->body).contains("error"));
    server.queue_tasks.terminate();
    app.join();
    CHECK(exit_code.load() == 0);
    server.ctx->backend_cpu->iface.graph_compute = original_compute;
    observed = nullptr;
    std::puts("test-server-wire: production loopback completion/fault/closure paths passed");
}
