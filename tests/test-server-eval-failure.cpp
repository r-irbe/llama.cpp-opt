// Real server context + task/result queues with synthetic CPU weights.
// This is NOT HTTP/SSE wire coverage; no routes are imitated here.
#include <random>
#include "server-context.h"
#include "server-response-reader.h"
#include "server-response-task.h"
#include "server-completion-stream.h"
#include "llama-context.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

// One-shot TEST-ONLY allocation failure. The predicate is armed only around
// empty-payload queue/control operations, never model or JSON payload teardown.
static thread_local bool (*allocation_failure)(size_t) = nullptr;
static thread_local int injected_allocation_failures = 0;
static void * test_allocate(size_t size) {
    if (allocation_failure && allocation_failure(size)) {
        allocation_failure = nullptr; // Recovery/exception cleanup can allocate.
        ++injected_allocation_failures;
        throw std::bad_alloc();
    }
    if (auto * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void * operator new(size_t size) { return test_allocate(size); }
void * operator new[](size_t size) { return test_allocate(size); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }

// Application logging configuration normally supplied by server.cpp/main.
bool server_log_json = false;

// Same tiny architecture as test-eval-failure, with no external model artifact.
struct model_file {
    std::string path;
    model_file() {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/llama-server-eval-XXXXXX";
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
                {"llama.attention.head_count", 4}, {"llama.attention.head_count_kv", 4}}) {
            gguf_set_val_u32(file, item.first, item.second);
        }
        gguf_set_val_f32(file, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        gguf_set_val_str(file, "tokenizer.ggml.model", "llama");
        gguf_set_val_u32(file, "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(file, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(file, "tokenizer.ggml.eos_token_id", 2);
        std::vector<std::string> words {"<unk>", "<s>", "</s>"};
        for (int i = 3; i < 32; ++i) words.push_back("token" + std::to_string(i));
        std::vector<const char *> tokens;
        for (const auto & word : words) tokens.push_back(word.c_str());
        gguf_set_arr_str(file, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
        auto add = [&](const std::string & name, int64_t n0, int64_t n1 = 0) {
            auto * t = n1 ? ggml_new_tensor_2d(tensors, GGML_TYPE_F32, n0, n1)
                         : ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n0);
            ggml_set_name(t, name.c_str());
            std::fill_n(static_cast<float *>(t->data), ggml_nelements(t), n1 ? 0.01f : 1.0f);
            gguf_add_tensor(file, t);
        };
        add("token_embd.weight", 32, 32);
        add("output_norm.weight", 32);
        add("output.weight", 32, 32);
        add("blk.0.attn_norm.weight", 32);
        for (const char * p : {"q", "k", "v", "output"}) add(std::string("blk.0.attn_") + p + ".weight", 32, 32);
        add("blk.0.ffn_norm.weight", 32);
        add("blk.0.ffn_gate.weight", 32, 64);
        add("blk.0.ffn_up.weight", 32, 64);
        add("blk.0.ffn_down.weight", 64, 32);
        gguf_write_to_file(file, path.c_str(), false);
        gguf_free(file);
        ggml_free(tensors);
    }
    ~model_file() { unlink(path.c_str()); }
};

static server_context * observed;
static decltype(ggml_backend_i::graph_compute) original_compute;
static ggml_status failure;
static int calls, successes, active_at_failure;
static bool armed;
static ggml_status GGML_CALL injected_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (armed && ++calls == 2) {
        active_at_failure = 0;
        for (const auto & slot : observed->slots) active_at_failure += slot.is_processing();
        return failure;
    }
    const auto status = original_compute(backend, graph);
    if (armed && status == GGML_STATUS_SUCCESS) ++successes;
    return status;
}
struct injection {
    ggml_backend_t backend;
    explicit injection(server_context & server, ggml_status status) : backend(server.ctx->backend_cpu) {
        observed = &server;
        original_compute = backend->iface.graph_compute;
        failure = status;
        calls = successes = active_at_failure = 0;
        armed = true;
        backend->iface.graph_compute = injected_compute;
    }
    ~injection() { backend->iface.graph_compute = original_compute; observed = nullptr; armed = false; }
};

// Run the same inbound dispatch and real update callbacks deterministically on
// one loop thread; do not race slot reads or iface patching from a client thread.
static void tick(server_context & server) {
    auto & queue = server.queue_tasks;
    queue.drain_inbound();
    while (!queue.queue_tasks.empty()) {
        auto task = std::move(queue.queue_tasks.front());
        queue.queue_tasks.pop_front();
        server.process_single_task(std::move(task));
    }
    server.update_slots();
}
static int post(server_context & server, int slot) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = server.queue_tasks.get_new_id();
    const int id = task.id;
    task.data = {{"id_slot", slot}, {"prompt", json::array({3, 4, 5, 6})},
                 {"n_predict", 1}, {"stream", false}, {"temperature", 0.0}};
    task.tokens = server_tokens(llama_tokens{3, 4, 5, 6}, false);
    server.queue_results.add_waiting_task_id(id); // real pointer-result queue requires registration first
    server.queue_tasks.post(std::move(task));
    return id;
}
static void run(const std::string & path, ggml_status status, bool unregister_first = false) {
    server_context server;
    gpt_params params;
    params.model = path;
    params.n_ctx = 192;
    params.n_parallel = 3;
    params.n_batch = 16;
    params.n_ubatch = 2;
    params.n_threads = params.n_threads_batch = 1;
    params.n_gpu_layers = 0;
    params.no_kv_offload = true;
    params.use_mmap = params.warmup = params.flash_attn = false;
    params.cache_ram_mib = 0;
    params.ctx_checkpoints_n = 0;
    params.chat_template = "chatml";
    params.n_predict = 1;
    CHECK(server.load_model(params));
    server.init();
    CHECK(server.slots.size() == 3);
    llama_token idle_token = 3;
    CHECK(llama_decode(server.ctx, llama_batch_get_one(&idle_token, 1, 0, 2)) == 0);
    server.slots[2].cache_tokens.push_back(idle_token);
    server.slots[2].n_past = 1;
    const int first = post(server, 0), second = post(server, 1);
    const int deferred = post(server, 0);
    if (unregister_first) server.queue_results.remove_waiting_task_id(first);
    injection fault(server, status);
    tick(server);
    CHECK(calls == 2 && successes == 1 && active_at_failure == 2);
    CHECK(server.queue_tasks.queue_tasks_deferred.size() == 1);
    CHECK(server.queue_results.queue_results.size() == (unregister_first ? 1u : 2u));
    for (int id : {first, second}) {
        if (unregister_first && id == first) continue;
        auto result = server.queue_results.recv_with_timeout({id}, 1);
        CHECK(result && result->is_error() && result->id == id);
        auto * error = dynamic_cast<server_task_result_error *>(result.get());
        CHECK(error);
        CHECK(error->err_type == (status == GGML_STATUS_ALLOC_FAILED ? ERROR_TYPE_UNAVAILABLE : ERROR_TYPE_SERVER));
        CHECK(error->to_json().at("code") == (status == GGML_STATUS_ALLOC_FAILED ? 503 : 500));
        server.queue_results.remove_waiting_task_id(id);
    }
    CHECK(server.queue_results.queue_results.empty());
    CHECK(llama_get_kv_cache_used_cells(server.ctx) == 0);
    CHECK(server.system_need_update);
    for (const auto & slot : server.slots) {
        CHECK(slot.n_past == 0 && slot.cache_tokens.empty() && slot.i_batch == -1);
        CHECK(!slot.is_processing());
    }
    armed = false; // same loop thread, no concurrent patch/restore
    server_task_result_ptr recovered;
    for (int i = 0; i < 16 && !recovered; ++i) {
        tick(server);
        recovered = server.queue_results.recv_with_timeout({deferred}, 0);
    }
    CHECK(recovered && !recovered->is_error() && recovered->is_stop());
    CHECK(server.queue_tasks.queue_tasks_deferred.empty());
    CHECK(server.slots[0].n_prompt_tokens_cache == 0);
    CHECK(!server.system_need_update);
    server.queue_results.remove_waiting_task_id(deferred);
    tick(server);
    CHECK(server.slots_idle());
    CHECK(server.queue_results.queue_results.empty());
}
static std::vector<server_task> reader_tasks(server_context & server, int count) {
    std::vector<server_task> tasks;
    for (int i = 0; i < count; ++i) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = server.queue_tasks.get_new_id();
        tasks.push_back(std::move(task));
    }
    return tasks;
}

static server_task_result_ptr reader_final(int id, int index) {
    auto result = std::make_unique<server_task_result_cmpl_final>();
    result->id = id;
    result->index = index;
    return result;
}

// Shared PRODUCTION reader, real response/task queues, no slots/model/HTTP here.
// The callback bound prevents the old retry-on-closed loop from hanging this red.
static void reader_terminal_delivery() {
    for (bool close_queue : {true, false}) {
        server_context server;
        server_response_reader reader(server);
        auto tasks = reader_tasks(server, 1);
        const int id = tasks[0].id;
        reader.post_tasks(std::move(tasks));
        if (close_queue) server.queue_results.close();
        else server.queue_results.remove_waiting_task_id(id);
        int polls = 0;
        auto result = reader.next([&] { return ++polls > 2; });
        CHECK(result && result->is_error());
        CHECK(result->to_json().at("code") == 503);
        CHECK(!reader.has_next());
    }
    {
        server_context server;
        server_response_reader reader(server);
        server.queue_results.close();
        reader.post_tasks(reader_tasks(server, 1));
        CHECK(server.queue_tasks.drain_inbound() == 0); // Never publish unregistered work.
        int polls = 0;
        auto result = reader.next([&] { return ++polls > 2; });
        CHECK(result && result->is_error());
    }
    {
        server_context server;
        server_response_reader reader(server);
        auto tasks = reader_tasks(server, 2);
        const int first = tasks[0].id, second = tasks[1].id;
        reader.post_tasks(std::move(tasks));
        server.queue_results.send(reader_final(first, 0));
        auto result = reader.next([] { return false; });
        CHECK(result && !result->is_error() && reader.has_next());
        // A completed-but-still-registered ID must not keep an impossible
        // outstanding-ID wait alive after the remaining delivery is retired.
        server.queue_results.remove_waiting_task_id(second);
        int polls = 0;
        auto terminal = reader.next([&] { return ++polls > 2; });
        CHECK(terminal && terminal->is_error() && !reader.has_next());
        server.queue_tasks.drain_inbound();
        int cancels = 0;
        for (const auto & task : server.queue_tasks.queue_tasks) {
            if (task.type == SERVER_TASK_TYPE_CANCEL) {
                CHECK(task.id_target == second);
                ++cancels;
            }
        }
        CHECK(cancels == 1);
        reader.stop();
        CHECK(server.queue_tasks.drain_inbound() == 0); // No duplicate cancellation.
    }
    {
        server_context server;
        server_response_reader reader(server);
        auto tasks = reader_tasks(server, 2);
        const int id = tasks[0].id;
        reader.post_tasks(std::move(tasks));
        auto error = std::make_unique<server_task_result_error>();
        error->id = id;
        error->err_type = ERROR_TYPE_SERVER;
        error->err_msg = "reader regression";
        // Deliberately do not set the legacy base 'error' field: the virtual
        // error contract must govern typed result handling.
        server.queue_results.send(std::move(error));
        auto batch = reader.wait_for_all([] { return false; });
        CHECK(!batch.is_terminated && batch.error && batch.error->is_error());
        CHECK(!reader.has_next());
    }
    {
        server_context server;
        server_response_reader reader(server);
        auto tasks = reader_tasks(server, 2);
        const int first = tasks[0].id, second = tasks[1].id;
        reader.post_tasks(std::move(tasks));
        server.queue_results.send(reader_final(second, 1)); // Out-of-order success.
        server.queue_results.send(reader_final(first, 0));
        auto batch = reader.wait_for_all([] { return false; });
        CHECK(!batch.is_terminated && !batch.error && batch.results.size() == 2);
        CHECK(batch.results[0]->id == first && batch.results[1]->id == second);
        CHECK(reader.received_count == 2 && !reader.has_next());
        reader.stop();
        server.queue_tasks.drain_inbound();
        for (const auto & task : server.queue_tasks.queue_tasks) CHECK(task.type != SERVER_TASK_TYPE_CANCEL);
    }
    {
        server_context server;
        server_response_reader reader(server);
        auto tasks = reader_tasks(server, 1);
        const int id = tasks[0].id;
        reader.post_tasks(std::move(tasks));
        const int peer = server.queue_tasks.get_new_id();
        CHECK(server.queue_results.add_waiting_task_id(peer));
        auto batch = reader.wait_for_all([] { return true; }); // Actual transport stop.
        CHECK(batch.is_terminated && !batch.error && !reader.has_next());
        CHECK(!server.queue_results.waiting_task_ids.count(id));
        CHECK(server.queue_results.waiting_task_ids.count(peer));
        server.queue_results.send(reader_final(peer, 0));
        auto survivor = server.queue_results.receive(std::unordered_set<int>{peer});
        CHECK(survivor.status == server_receive_status::result);
        server.queue_results.remove_waiting_task_id(peer);
    }
}

static thread_local server_queue * allocation_queue = nullptr;
static constexpr int failure_batch_size = 16;

static bool fail_overflow_allocation(size_t) {
    // Armed before publication while the ring has one free cell. The selected
    // allocation occurs only after a prefix is published and overflow is active.
    return allocation_queue->overflow_active.load(std::memory_order_acquire);
}
static bool fail_cancel_buffer(size_t size) {
    return size == failure_batch_size * sizeof(server_task);
}
static void fill_to_one_free_ring_cell(server_queue & queue, int already_posted) {
    for (int i = already_posted; i < static_cast<int>(server_queue::RING_CAPACITY) - 1; ++i) {
        server_task filler(SERVER_TASK_TYPE_METRICS);
        queue.post(std::move(filler));
    }
    CHECK(!queue.overflow_active.load() && queue.overflow.empty());
}
static int check_cancelled_targets(server_queue & queue, const std::unordered_set<int> & ids, int expected_fillers) {
    std::unordered_set<int> targets;
    int cancels = 0, fillers = 0;
    for (const auto & task : queue.queue_tasks) {
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            CHECK(ids.count(task.id_target));
            targets.insert(task.id_target);
            ++cancels;
        } else {
            CHECK(task.type == SERVER_TASK_TYPE_METRICS && !ids.count(task.id));
            ++fillers;
        }
    }
    CHECK(targets == ids && fillers == expected_fillers);
    return cancels;
}

static void reader_exception_recovery() {
    // Current task size guarantees this modest batch reaches deque growth on
    // the exercised standard library. The observed partial prefixes below are
    // asserted, not assumed to have a portable fixed length or allocation count.
    CHECK(sizeof(server_queue::posted_task) >= 512);
    {
        server_context server;
        server_response_reader reader(server);
        auto & queue = server.queue_tasks;
        fill_to_one_free_ring_cell(queue, 0);
        auto tasks = reader_tasks(server, failure_batch_size);
        const auto ids = server_task::get_list_id(tasks);
        allocation_queue = &queue;
        injected_allocation_failures = 0;
        allocation_failure = fail_overflow_allocation;
        bool failed = false;
        try { reader.post_tasks(std::move(tasks)); }
        catch (const std::bad_alloc &) { failed = true; }
        allocation_failure = nullptr;
        allocation_queue = nullptr;
        CHECK(failed && injected_allocation_failures == 1);
        CHECK(reader.id_tasks == ids && !reader.has_next());
        for (int id : ids) CHECK(!server.queue_results.waiting_task_ids.count(id));
        const int fillers = server_queue::RING_CAPACITY - 1;
        const auto drained = queue.drain_inbound();
        CHECK(drained > static_cast<size_t>(fillers + failure_batch_size));
        CHECK(drained < static_cast<size_t>(fillers + 2 * failure_batch_size));
        CHECK(check_cancelled_targets(queue, ids, fillers) == failure_batch_size);
        reader.stop();
        CHECK(queue.drain_inbound() == 0);
    }
    for (bool partial_publication : {false, true}) {
        server_context server;
        server_response_reader reader(server);
        auto & queue = server.queue_tasks;
        auto tasks = reader_tasks(server, failure_batch_size);
        const auto ids = server_task::get_list_id(tasks);
        reader.post_tasks(std::move(tasks));
        int fillers = 0;
        if (partial_publication) {
            fill_to_one_free_ring_cell(queue, failure_batch_size);
            fillers = server_queue::RING_CAPACITY - 1 - failure_batch_size;
        }
        allocation_queue = &queue;
        injected_allocation_failures = 0;
        allocation_failure = partial_publication ? fail_overflow_allocation : fail_cancel_buffer;
        bool failed = false;
        try { reader.stop(); }
        catch (const std::bad_alloc &) { failed = true; }
        allocation_failure = nullptr;
        allocation_queue = nullptr;
        CHECK(failed && injected_allocation_failures == 1 && !reader.has_next());
        CHECK(reader.id_tasks == ids);
        for (int id : ids) CHECK(!server.queue_results.waiting_task_ids.count(id));
        reader.stop(); // cancelled flag must NOT prevent cancellation retry.
        const auto drained = queue.drain_inbound();
        const int cancels = check_cancelled_targets(queue, ids, fillers);
        if (partial_publication) {
            CHECK(cancels > failure_batch_size && cancels < 2 * failure_batch_size);
            CHECK(drained == static_cast<size_t>(fillers + failure_batch_size + cancels));
        } else {
            CHECK(cancels == failure_batch_size);
            CHECK(drained == 2 * failure_batch_size);
        }
        reader.stop();
        CHECK(queue.drain_inbound() == 0);
    }
}

static void legacy_request_ownership() {
    using status = server_receive_status;
    for (int mode : {0, 1, 2}) {
        server_context server;
        auto & tasks = server.queue_tasks;
        auto & responses = server.queue_results;
        int handled = 0;
        tasks.on_new_task([&](server_task && task) {
            {
                std::lock_guard<std::mutex> lock(responses.mutex_results);
                CHECK(responses.waiting_task_ids.count(task.id));
            }
            ++handled;
            if (mode == 1) responses.close();
            else if (mode == 2) responses.remove_waiting_task_id(task.id);
            else {
                server_task_result result{};
                result.id = task.id;
                result.data = {{"observed_id", task.id}};
                responses.send(std::move(result));
            }
        });
        tasks.on_update_slots([&] { if (handled) tasks.terminate(); });
        std::thread owner([&] { tasks.start_loop(); });
        server_task task(SERVER_TASK_TYPE_METRICS);
        const int id = task.id = tasks.get_new_id();
        auto reply = server_request_legacy(tasks, responses, std::move(task));
        owner.join();
        CHECK(handled == 1);
        CHECK(reply.status == (mode == 0 ? status::result : mode == 1 ? status::closed : status::retired));
        if (mode == 0) CHECK(reply.value && reply.value->data.at("observed_id") == id);
        else CHECK(!reply.value);
        CHECK(responses.waiting_task_ids.empty());
        CHECK(responses.queue_results_legacy.empty());
    }
    {
        server_context server;
        server.queue_results.close();
        server_task task(SERVER_TASK_TYPE_METRICS);
        auto reply = server_request_legacy(server.queue_tasks, server.queue_results, std::move(task));
        CHECK(reply.status == status::closed && !reply.value);
        CHECK(server.queue_tasks.drain_inbound() == 0);
    }
    {
        server_context server;
        auto & queue = server.queue_tasks;
        fill_to_one_free_ring_cell(queue, 0);
        queue.post(server_task(SERVER_TASK_TYPE_METRICS));
        server_task task(SERVER_TASK_TYPE_METRICS);
        const int id = task.id = queue.get_new_id();
        allocation_queue = &queue;
        injected_allocation_failures = 0;
        allocation_failure = fail_overflow_allocation;
        bool failed = false;
        try { (void) server_request_legacy(queue, server.queue_results, std::move(task)); }
        catch (const std::bad_alloc &) { failed = true; }
        allocation_failure = nullptr;
        allocation_queue = nullptr;
        CHECK(failed && injected_allocation_failures == 1);
        CHECK(!server.queue_results.waiting_task_ids.count(id));
        CHECK(queue.drain_inbound() == server_queue::RING_CAPACITY);
        for (const auto & retained : queue.queue_tasks) CHECK(retained.id != id);
    }
}

struct recording_sink {
    httplib::DataSink sink;
    std::string body;
    int done_calls = 0;
    int writes = 0;
    int fail_write = 0;
    bool fail_done = false;
    bool attempted_done = false;
    recording_sink() {
        sink.write = [&](const char * data, size_t size) {
            ++writes;
            const bool marker = std::string(data, size) == "data: [DONE]\n\n";
            attempted_done |= marker;
            if (writes == fail_write || (fail_done && marker)) return false;
            body.append(data, size);
            return true;
        };
        sink.is_writable = [] { return true; };
        sink.done = [&] { ++done_calls; };
    }
};

// Actual production provider + reader/queues/serializers, not socket/route tests.
static void completion_stream_delivery() {
    for (auto type : {OAICOMPAT_TYPE_COMPLETION, OAICOMPAT_TYPE_CHAT,
                     OAICOMPAT_TYPE_NONE, OAICOMPAT_TYPE_ANTHROPIC, OAICOMPAT_TYPE_RESP}) {
        for (bool send_done : {false, true}) {
            server_context server;
            auto reader = std::make_shared<server_response_reader>(server);
            auto tasks = reader_tasks(server, 1);
            auto final = reader_final(tasks[0].id, 0);
            final->oaicompat = type;
            final->stream = true;
            reader->post_tasks(std::move(tasks));
            server.queue_results.send(std::move(final));
            auto first = reader->next([] { return false; });
            CHECK(first && first->is_stop() && !reader->has_next());
            auto provider = server_completion_stream_provider(first->to_json(), reader, type, send_done);
            recording_sink output;
            const bool ok = provider(0, output.sink);
            const bool expected_done = send_done || type == OAICOMPAT_TYPE_COMPLETION || type == OAICOMPAT_TYPE_CHAT;
            CHECK((output.body.find("data: [DONE]\n\n") != std::string::npos) == expected_done);
            CHECK(ok && output.done_calls == 1);
        }
    }
    for (bool fail_done : {false, true}) {
        server_context server;
        auto reader = std::make_shared<server_response_reader>(server);
        auto tasks = reader_tasks(server, 1);
        auto final = reader_final(tasks[0].id, 0);
        final->oaicompat = OAICOMPAT_TYPE_COMPLETION;
        final->stream = true;
        reader->post_tasks(std::move(tasks));
        server.queue_results.send(std::move(final));
        auto first = reader->next([] { return false; });
        auto provider = server_completion_stream_provider(first->to_json(), reader, OAICOMPAT_TYPE_COMPLETION, false);
        recording_sink output;
        output.fail_write = fail_done ? 0 : 1;
        output.fail_done = fail_done;
        CHECK(!provider(0, output.sink));
        CHECK(output.done_calls == 0);
        CHECK(output.attempted_done == fail_done);
        CHECK(output.body.find("[DONE]") == std::string::npos);
    }
    for (int terminal : {0, 1, 2}) {
        server_context server;
        auto reader = std::make_shared<server_response_reader>(server);
        auto tasks = reader_tasks(server, 1);
        const int id = tasks[0].id;
        reader->post_tasks(std::move(tasks));
        auto partial = std::make_unique<server_task_result_cmpl_partial>();
        partial->id = id;
        partial->oaicompat = OAICOMPAT_TYPE_COMPLETION;
        partial->content = "partial";
        server.queue_results.send(std::move(partial));
        auto first = reader->next([] { return false; });
        CHECK(first && !first->is_stop() && reader->has_next());
        if (terminal == 1) server.queue_results.close();
        else if (terminal == 2) server.queue_results.remove_waiting_task_id(id);
        else {
            auto final = reader_final(id, 0);
            final->oaicompat = OAICOMPAT_TYPE_COMPLETION;
            server.queue_results.send(std::move(final));
        }
        auto provider = server_completion_stream_provider(first->to_json(), reader, OAICOMPAT_TYPE_COMPLETION, false);
        recording_sink output;
        CHECK(provider(0, output.sink));
        CHECK(output.done_calls == 1 && !reader->has_next());
        CHECK((output.body.find("[DONE]") != std::string::npos) == (terminal == 0));
        if (terminal) {
            CHECK(output.body.find("\"error\"") != std::string::npos);
            CHECK(output.body.find("503") != std::string::npos);
        }
    }
    for (int failure : {0, 1, 2}) {
        server_context server;
        auto reader = std::make_shared<server_response_reader>(server);
        auto tasks = reader_tasks(server, 1);
        const int id = tasks[0].id;
        reader->post_tasks(std::move(tasks));
        auto partial = std::make_unique<server_task_result_cmpl_partial>();
        partial->id = id;
        partial->oaicompat = OAICOMPAT_TYPE_COMPLETION;
        partial->content = "partial";
        server.queue_results.send(std::move(partial));
        auto first = reader->next([] { return false; });
        if (failure == 0) {
            auto final = reader_final(id, 0);
            final->oaicompat = OAICOMPAT_TYPE_COMPLETION;
            server.queue_results.send(std::move(final));
        } else if (failure == 1) server.queue_results.close();
        auto provider = server_completion_stream_provider(first->to_json(), reader, OAICOMPAT_TYPE_COMPLETION, false);
        recording_sink output;
        if (failure == 2) output.sink.is_writable = [] { return false; };
        else output.fail_write = 2; // Fail later-final or terminal-error event.
        CHECK(!provider(0, output.sink));
        CHECK(output.done_calls == 0 && !output.attempted_done);
        CHECK(output.body.find("[DONE]") == std::string::npos);
        CHECK(!reader->has_next());
        if (failure == 2) CHECK(!server.queue_results.waiting_task_ids.count(id));
    }
    {
        server_context server;
        auto reader = std::make_shared<server_response_reader>(server);
        auto tasks = reader_tasks(server, 1);
        const int id = tasks[0].id;
        reader->post_tasks(std::move(tasks));
        server.queue_results.send(reader_final(id, 0));
        auto first = reader->next([] { return false; });
        CHECK(first && first->is_stop() && !reader->has_next());
        // Empty serialization is still a completed first result, not a receive.
        auto provider = server_completion_stream_provider(json::array(), reader, OAICOMPAT_TYPE_COMPLETION, false);
        recording_sink output;
        CHECK(provider(0, output.sink) && output.done_calls == 1);
        CHECK(output.body == "data: [DONE]\n\n" && output.writes == 1);
    }
}

static void legacy_parent_error_fold() {
    for (const auto & errors : std::vector<std::vector<bool>>{{false, true}, {true, false}, {true, true}, {false, false}, {}}) {
        server_context server;
        server_task_multi parent;
        parent.id = server.queue_tasks.get_new_id();
        json expected_children = json::array();
        bool expected_error = false;
        for (bool error : errors) {
            server_task_result child{};
            child.id = server.queue_tasks.get_new_id();
            child.id_multi = parent.id;
            child.error = error;
            child.data = {{"id", child.id}, {"failed", error}};
            expected_children.push_back(child.data);
            expected_error |= error;
            parent.results.push_back(std::move(child));
        }
        CHECK(server.queue_results.add_waiting_task_id(parent.id));
        server.on_finish_multitask(parent); // Actual production fold, no substitute.
        auto result = server.queue_results.receive(parent.id, std::chrono::steady_clock::now());
        CHECK(result.status == server_receive_status::result && result.value);
        CHECK(result.value->id == parent.id && result.value->error == expected_error);
        CHECK(result.value->data == json({{"results", expected_children}}));
        for (size_t i = 0; i < errors.size(); ++i) {
            CHECK(parent.results[i].error == errors[i]);
            CHECK(parent.results[i].data == expected_children[i]);
        }
        server.queue_results.remove_waiting_task_id(parent.id);
        CHECK(server.queue_results.queue_results_legacy.empty());
    }
}

int main() {
    legacy_parent_error_fold();
    completion_stream_delivery();
    legacy_request_ownership();
    reader_exception_recovery();
    reader_terminal_delivery();
    model_file file;
    for (auto status : {GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_ABORTED}) run(file.path, status);
    // Queue-level loss of a waiter, not an assertion of actual HTTP disconnect
    // processing: only the remaining connected ID must receive terminal error.
    run(file.path, GGML_STATUS_ABORTED, true);
    std::puts("test-server-eval-failure: real multi-slot context/queues passed; HTTP excluded");
}
