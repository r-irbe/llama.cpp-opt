#include "response-calibration-probes.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <typeinfo>

namespace {
constexpr size_t max_records = 262144;
constexpr uint64_t max_nodes = 32768;
struct record {
    const char * stage = nullptr;
    uint64_t nanos = 0, generation = 0, episode = 0, thread = 0;
    int task = 0;
    response_calibration_sizes result, value, wire;
};
std::array<record, max_records> records;
std::atomic<size_t> next_record{0};
std::atomic<uint64_t> next_generation{1}, next_thread{1};
std::array<std::atomic<uint64_t>, 8192> task_episodes{};
thread_local uint64_t current_episode = 0;
thread_local uint64_t thread_token = 0;
std::atomic<bool> enabled{false}, lost{false};
thread_local const server_task_result * current_result = nullptr;
thread_local const std::vector<server_task_result_ptr> * current_batch = nullptr;

struct visitor {
    response_calibration_sizes out;
    bool visit() noexcept {
        if (out.nodes >= max_nodes) { out.incomplete = true; return false; }
        ++out.nodes;
        return true;
    }
    void add(uint64_t & dst, uint64_t value) noexcept {
        if (value > std::numeric_limits<uint64_t>::max() - dst) {
            out.incomplete = true;
            dst = std::numeric_limits<uint64_t>::max();
        } else dst += value;
    }
    uint64_t bytes(size_t count, size_t width) noexcept {
        if (width && count > std::numeric_limits<uint64_t>::max() / width) {
            out.incomplete = true;
            return std::numeric_limits<uint64_t>::max();
        }
        return count * width;
    }
    void string(const std::string & s) noexcept {
        if (!visit()) return;
        add(out.string_size, s.size()); add(out.string_capacity, s.capacity());
    }
    template<class T> void vector(const T & v) noexcept {
        add(out.storage_size, bytes(v.size(), sizeof(typename T::value_type)));
        add(out.storage_capacity, bytes(v.capacity(), sizeof(typename T::value_type)));
    }
    void value(const json & j, unsigned depth = 0) {
        if (!visit()) return;
        if (depth >= 32) { out.incomplete = true; return; }
        if (j.is_object()) {
            const auto & obj = j.get_ref<const json::object_t &>();
            vector(obj);
            for (const auto & item : obj) {
                if (out.nodes >= max_nodes) { out.incomplete = true; break; }
                string(item.first); value(item.second, depth + 1);
            }
        } else if (j.is_array()) {
            const auto & arr = j.get_ref<const json::array_t &>();
            vector(arr);
            for (const auto & item : arr) {
                if (out.nodes >= max_nodes) { out.incomplete = true; break; }
                value(item, depth + 1);
            }
        } else if (j.is_string()) string(j.get_ref<const json::string_t &>());
        else if (j.is_binary()) vector(j.get_ref<const json::binary_t &>());
    }
    void tool(const common_chat_tool_call & t) noexcept {
        string(t.name); string(t.arguments); string(t.id);
    }
    void message(const common_chat_msg & m) noexcept {
        string(m.role); string(m.content); string(m.reasoning_content);
        string(m.tool_name); string(m.tool_call_id);
        vector(m.content_parts); vector(m.tool_calls);
        for (const auto & p : m.content_parts) {
            if (!visit()) break;
            string(p.type); string(p.text);
        }
        for (const auto & t : m.tool_calls) { if (!visit()) break; tool(t); }
    }
    void result(const server_task_result & r) {
        if (!visit()) return;
        // Containers are constructed even where primitive result flags are not.
        string(r.content); string(r.prompt); string(r.stopping_word);
        string(r.oaicompat_model); string(r.oaicompat_cmpl_id);
        vector(r.tokens); vector(r.response_fields); vector(r.probs_output);
        vector(r.timings.draft_n_by_depth); vector(r.timings.draft_n_accepted_by_depth);
        for (const auto & field : r.response_fields) { if (!visit()) break; string(field); }
        add(out.probability_outputs, r.probs_output.size());
        for (const auto & p : r.probs_output) {
            if (!visit()) break;
            string(p.text_to_send); vector(p.probs);
            add(out.probability_candidates, p.probs.size());
            for (const auto & q : p.probs) { if (!visit()) break; string(q.txt); }
        }
        value(r.data); message(r.oaicompat_msg); vector(r.oaicompat_msg_diffs);
        for (const auto & d : r.oaicompat_msg_diffs) {
            if (!visit()) break;
            string(d.reasoning_content_delta); string(d.content_delta); tool(d.tool_call_delta);
        }
        if (const auto * p = dynamic_cast<const server_task_result_cmpl_partial *>(&r)) {
            string(p->oai_resp_id); string(p->oai_resp_reasoning_id);
            string(p->oai_resp_message_id); string(p->oai_resp_fc_id);
        } else if (const auto * p = dynamic_cast<const server_task_result_cmpl_final *>(&r)) {
            string(p->oai_resp_id); string(p->oai_resp_reasoning_id); string(p->oai_resp_message_id);
        } else if (const auto * p = dynamic_cast<const server_task_result_error *>(&r)) string(p->err_msg);
        else if (const auto * p = dynamic_cast<const server_task_result_embd *>(&r)) {
            vector(p->embedding);
            for (const auto & row : p->embedding) { if (!visit()) break; vector(row); }
        } else if (const auto * p = dynamic_cast<const server_task_result_decision *>(&r)) vector(p->scores);
        else if (typeid(r) != typeid(server_task_result)) out.incomplete = true;
    }
};

void emit(const char * stage, uint64_t generation, const server_task_result * r,
          const json * j, const std::string * wire, int task, bool observe = true,
          uint64_t episode = 0) noexcept {
    if (!enabled.load(std::memory_order_relaxed)) return;
    const size_t index = next_record.fetch_add(1, std::memory_order_relaxed);
    if (index >= records.size()) { lost.store(true); return; }
    auto & dst = records[index];
    dst.stage = stage; dst.generation = generation; dst.task = task;
    if (!thread_token) thread_token = next_thread.fetch_add(1, std::memory_order_relaxed);
    dst.thread = thread_token;
    dst.episode = episode ? episode : current_episode;
    if (task >= 0 && size_t(task) < task_episodes.size()) {
        const auto associated = task_episodes[size_t(task)].load(std::memory_order_acquire);
        if (associated) dst.episode = associated;
    }
    dst.nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (!observe) return; // Under queue locks: scalar record publication only.
    try {
        visitor results;
        if (r) { results.result(*r); dst.task = r->id; }
        else if (current_result) { results.result(*current_result); dst.task = current_result->id; }
        else if (current_batch) for (const auto & item : *current_batch) {
            if (!results.visit()) break;
            if (item) results.result(*item);
        }
        if (dst.task >= 0 && size_t(dst.task) < task_episodes.size()) {
            const auto associated = task_episodes[size_t(dst.task)].load(std::memory_order_acquire);
            if (associated) dst.episode = associated;
        }
        dst.result = results.out;
        if (j) { visitor v; v.value(*j); dst.value = v.out; }
        if (wire) { visitor v; v.string(*wire); dst.wire = v.out; }
    } catch (...) { dst.result.incomplete = dst.value.incomplete = true; }
}
json sizes(const response_calibration_sizes & s) {
    return {{"string_size", s.string_size}, {"string_capacity", s.string_capacity},
        {"storage_size", s.storage_size}, {"storage_capacity", s.storage_capacity},
        {"probability_outputs", s.probability_outputs}, {"probability_candidates", s.probability_candidates},
        {"visited_nodes", s.nodes}, {"incomplete", s.incomplete}};
}
void require(bool ok) { if (!ok) std::abort(); }
}

response_calibration_sizes response_calibration_measure(const server_task_result & r) noexcept {
    visitor v; try { v.result(r); } catch (...) { v.out.incomplete = true; } return v.out;
}
response_calibration_sizes response_calibration_measure(const json & j) noexcept {
    visitor v; try { v.value(j); } catch (...) { v.out.incomplete = true; } return v.out;
}
void response_calibration_enable(bool on) noexcept { enabled.store(on); }
bool response_calibration_complete() noexcept { return !lost.load(); }
void server_calibration_snapshot(const char * stage, const void *, const server_task_result * r,
        const json * value, const std::string * wire, int task) noexcept {
    // Each row is an independent simultaneous snapshot. No raw address or
    // fabricated cross-snapshot representation lifetime is exported.
    emit(stage, 0, r, value, wire, task);
}
void server_calibration_set_request(const std::string & identity) noexcept {
    current_episode = 0;
    if (!enabled.load() || identity.empty() || identity.size() > 3) return;
    uint64_t value = 0;
    for (const char c : identity) {
        if (c < '0' || c > '9') return;
        value = value * 10 + uint64_t(c - '0');
    }
    if (value && value <= 600) current_episode = value;
}
void server_calibration_associate(const std::vector<server_task> & tasks) noexcept {
    if (!enabled.load()) return;
    if (!current_episode || tasks.size() > 3) { lost.store(true); return; }
    for (const auto & task : tasks) {
        if (task.id < 0 || size_t(task.id) >= task_episodes.size()) { lost.store(true); return; }
        task_episodes[size_t(task.id)].store(current_episode, std::memory_order_release);
        emit("episode_child_association", 0, nullptr, nullptr, nullptr, task.id, false);
    }
}
uint64_t server_calibration_current_episode() noexcept { return current_episode; }
server_calibration_episode_context::server_calibration_episode_context(uint64_t episode) noexcept
    : previous(current_episode) { current_episode = episode; }
server_calibration_episode_context::~server_calibration_episode_context() { current_episode = previous; }
void server_calibration_stage(const char * stage, int task) noexcept {
    emit(stage, 0, nullptr, nullptr, nullptr, task, false);
}
void server_calibration_receive(const server_task_result_ptr & r) noexcept {
    emit("dequeue_transfer", 0, r.get(), nullptr, nullptr, r ? r->id : -1);
}
void server_calibration_receive(const server_task_result & r) noexcept {
    emit("legacy_dequeue_transfer", 0, &r, nullptr, nullptr, r.id);
}
server_calibration_context::server_calibration_context(const server_task_result * r) noexcept
    : previous_result(current_result), previous_batch(current_batch) {
    current_result = r; current_batch = nullptr;
}
server_calibration_context::server_calibration_context(const std::vector<server_task_result_ptr> * b) noexcept
    : previous_result(current_result), previous_batch(current_batch) {
    current_result = nullptr; current_batch = b;
}
server_calibration_context::~server_calibration_context() {
    current_result = previous_result; current_batch = previous_batch;
}
server_calibration_scope::server_calibration_scope(const char * s) noexcept
    : stage(s), generation(next_generation.fetch_add(1)), episode(current_episode) {
    emit("owned_scope_begin", generation, nullptr, nullptr, nullptr, -1, false, episode);
}
server_calibration_scope::~server_calibration_scope() {
    // Only the specifically enclosed locals have retired. Never dereference them.
    emit(stage, generation, nullptr, nullptr, nullptr, -1, false, episode);
}
void response_calibration_flush(FILE * protocol) {
    // Call ONLY after application/HTTP threads join. Never format inside a hook.
    response_calibration_enable(false);
    const size_t count = std::min(next_record.load(), records.size());
    for (size_t i = 0; i < count; ++i) {
        const auto & r = records[i];
        json row = {{"event", "measurement"}, {"sequence", i}, {"steady_ns", r.nanos},
            {"stage", r.stage}, {"scope_generation", r.generation}, {"task", r.task},
            {"episode", r.episode ? json(r.episode) : json(nullptr)}, {"thread_token", r.thread},
            {"result", sizes(r.result)}, {"json", sizes(r.value)}, {"wire", sizes(r.wire)}};
        const auto line = row.dump();
        require(std::fprintf(protocol, "%s\n", line.c_str()) > 0);
    }
    require(std::fprintf(protocol, "{\"event\":\"measurement\",\"trace_complete\":%s,\"allocator_backing\":\"unavailable\",\"socket_acceptance\":\"unavailable\",\"provider_copy_lifetimes\":\"unavailable\",\"http_body_destruction\":\"unavailable\"}\n",
        response_calibration_complete() ? "true" : "false") > 0);
    require(std::fflush(protocol) == 0);
}
void response_calibration_self_test() {
    server_task_result r;
    const auto empty = response_calibration_measure(r);
    require(empty.string_size == 0 && empty.storage_size == 0 && !empty.incomplete);
    r.content = "abc"; r.content.reserve(256); r.tokens = {1, 2};
    const auto content = response_calibration_measure(r);
    require(content.string_size == 3 && content.string_capacity >= 256);
    require(content.storage_size == 2 * sizeof(llama_token));
    r.probs_output.resize(1); r.probs_output[0].text_to_send = "ab";
    r.probs_output[0].probs.resize(2);
    r.probs_output[0].probs[0].txt = "x"; r.probs_output[0].probs[1].txt = "yz";
    const auto probs = response_calibration_measure(r);
    require(probs.string_size == 8 && probs.probability_outputs == 1 && probs.probability_candidates == 2);
    const json j = {{"k", json::array({"abc", 7, nullptr})}};
    const auto object = response_calibration_measure(j);
    require(object.string_size == 4 && object.storage_capacity >= object.storage_size && !object.incomplete);
    server_task_result_embd embedding;
    embedding.embedding = {{1.f, 2.f}, {3.f, 4.f, 5.f}};
    const auto e = response_calibration_measure(embedding);
    require(e.storage_size == 2 * sizeof(std::vector<float>) + 5 * sizeof(float));
}
