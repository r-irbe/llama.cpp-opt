// Production response-queue ownership only: no model, HTTP server or GPU work.
#include "server-queue.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <chrono>
#include <climits>
#include <future>
#include <thread>
#include <latch>
#include <type_traits>
#include <utility>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

static_assert(std::has_virtual_destructor<server_task_result>::value,
              "base-owned server results must destroy their derived payloads");

template <class T> constexpr bool value_compatible =
    std::is_default_constructible<T>::value &&
    std::is_copy_constructible<T>::value && std::is_copy_assignable<T>::value &&
    std::is_move_constructible<T>::value && std::is_move_assignable<T>::value;

static_assert(value_compatible<server_task_result>, "preserve legacy value operations");
static_assert(value_compatible<server_task_result_cmpl_partial>, "preserve partial value operations");
static_assert(value_compatible<server_task_result_cmpl_final>, "preserve final value operations");
static_assert(value_compatible<server_task_result_error>, "preserve error value operations");
static_assert(value_compatible<server_task_result_embd>, "preserve embedding value operations");
static_assert(value_compatible<server_task_result_decision>, "preserve decision value operations");

// Test-local allocation failure and backing-deallocation probes. No production
// failpoint, allocator replacement or timing-dependent lock test is introduced.
static thread_local bool deny_allocations = false;
static thread_local int allocations_before_failure = -1;
static thread_local server_response * allocation_response = nullptr;
static thread_local server_queue * allocation_tasks = nullptr;
static thread_local size_t observed_retirement_allocations = 0;

// JSON teardown may allocate scratch itself. Probe that any such allocation is
// OUTSIDE the locks rather than falsely promising allocation-free destruction.
struct retirement_allocation_probe {
    retirement_allocation_probe(server_response & response, server_queue * tasks = nullptr) {
        allocation_response = &response;
        allocation_tasks = tasks;
    }
    ~retirement_allocation_probe() {
        allocation_response = nullptr;
        allocation_tasks = nullptr;
    }
};

struct backing_watch {
    const void * address = nullptr;
    server_response * response = nullptr;
    int * releases = nullptr;
    server_queue * tasks = nullptr;
};
static std::array<backing_watch, 8> backing_watches{};

static void * allocate(size_t size) {
    if (deny_allocations || allocations_before_failure == 0) throw std::bad_alloc();
    if (allocations_before_failure > 0) --allocations_before_failure;
    if (allocation_response) {
        auto * response = allocation_response;
        allocation_response = nullptr; // Do not recursively probe logging/setup.
        response->add_waiting_task_id(999);
        if (allocation_tasks) { std::lock_guard<std::mutex> lock(allocation_tasks->mutex_tasks); }
        ++observed_retirement_allocations;
        allocation_response = response;
    }
    if (auto * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
static void deallocate(void * p) noexcept {
    if (!p) return;
    for (auto & watch : backing_watches) {
        if (watch.address == p) {
            auto saved = watch;
            watch = {};
            ++*saved.releases;
            // This existing-ID insertion reacquires the response mutex without
            // allocating. If reclamation is under that mutex, CTest catches the
            // deadlock; no same-thread try_lock or sleep-based inference.
            saved.response->add_waiting_task_id(999);
            if (saved.tasks) { std::lock_guard<std::mutex> lock(saved.tasks->mutex_tasks); }
        }
    }
    std::free(p);
}
void * operator new(size_t size) { return allocate(size); }
void * operator new[](size_t size) { return allocate(size); }
void operator delete(void * p) noexcept { deallocate(p); }
void operator delete[](void * p) noexcept { deallocate(p); }
void operator delete(void * p, size_t) noexcept { deallocate(p); }
void operator delete[](void * p, size_t) noexcept { deallocate(p); }

struct lifetime_counts {
    int destructors = 0;
    int members = 0;
};

struct member_probe {
    lifetime_counts & counts;
    std::vector<int> backing = std::vector<int>(1024, 7);
    ~member_probe() { ++counts.members; }
};

struct tracked_result : server_task_result {
    lifetime_counts & counts;
    member_probe member;

    explicit tracked_result(int task_id, lifetime_counts & counts)
        : server_task_result{}, counts(counts), member{counts} {
        id = task_id;
        content.assign(1024, 'x');
        tokens = {1, 2, 3};
    }

    // No override here: the pre-repair compile-time gate above is the red.
    ~tracked_result() { ++counts.destructors; }
};

static void check_released(const lifetime_counts & counts, int expected) {
    CHECK(counts.destructors == expected);
    CHECK(counts.members == expected);
}

static void direct_ownership() {
    lifetime_counts counts;
    server_task_result_ptr first = std::make_unique<tracked_result>(1, counts);
    auto * identity = first.get();
    auto second = std::move(first);
    CHECK(!first && second.get() == identity);
    check_released(counts, 0);
    second.reset();
    check_released(counts, 1);
}

static void registered_roundtrip() {
    lifetime_counts counts;
    server_response response;
    response.add_waiting_task_id(10);
    server_task_result_ptr result = std::make_unique<tracked_result>(10, counts);
    auto * identity = result.get();
    response.send(std::move(result));
    CHECK(!result && response.queue_results.size() == 1);
    check_released(counts, 0);
    auto received = response.recv_with_timeout({10}, 0);
    CHECK(received.get() == identity);
    CHECK(dynamic_cast<tracked_result *>(received.get()) != nullptr);
    CHECK(response.queue_results.empty());
    check_released(counts, 0);
    received.reset();
    check_released(counts, 1);
}

static void queued_teardown_and_erase() {
    std::array<lifetime_counts, 12> counts{};
    {
        server_response response;
        for (int i = 0; i < static_cast<int>(counts.size()); ++i) {
            response.add_waiting_task_id(100 + i);
            response.send(std::make_unique<tracked_result>(100 + i, counts[i]));
        }
        auto middle = response.recv_with_timeout({105}, 0);
        CHECK(middle && middle->id == 105);
        CHECK(response.queue_results.size() == counts.size() - 1);
        for (const auto & count : counts) check_released(count, 0);
        size_t i = 0;
        for (const auto & result : response.queue_results) {
            CHECK(result->id == 100 + static_cast<int>(i) + (i >= 5 ? 1 : 0));
            ++i;
        }
        middle.reset();
        check_released(counts[5], 1);
    }
    for (const auto & count : counts) check_released(count, 1);
}

static void rejected_ownership() {
    server_response response;
    lifetime_counts temporary;
    response.send(std::make_unique<tracked_result>(20, temporary));
    CHECK(response.queue_results.empty());
    check_released(temporary, 1);

    lifetime_counts named;
    server_task_result_ptr result = std::make_unique<tracked_result>(21, named);
    response.send(std::move(result));
    // The rvalue-reference parameter is only moved from on accepted publication.
    CHECK(result && result->id == 21);
    CHECK(response.queue_results.empty());
    check_released(named, 0);
    result.reset();
    check_released(named, 1);
}

static server_task_result sample_value(int id) {
    server_task_result result{};
    result.id = id;
    result.content.assign(1024, 'a');
    result.tokens = {3, 5, 7, 11};
    result.response_fields = {"content", "tokens"};
    result.data = {{"tag", "legacy"}, {"values", {1, 2, 3}}};
    result.error = false;
    result.final_result = true;
    result.stream = false;
    result.include_usage = true;
    result.timings.prompt_n = 4;
    result.timings.prompt_ms = 2.5;
    return result;
}

static void check_value(const server_task_result & result, int id) {
    CHECK(result.id == id && result.id_multi == -1);
    CHECK(result.content == std::string(1024, 'a'));
    CHECK(result.tokens == std::vector<llama_token>({3, 5, 7, 11}));
    CHECK(result.data == json({{"tag", "legacy"}, {"values", {1, 2, 3}}}));
    CHECK(result.response_fields == std::vector<std::string>({"content", "tokens"}));
    CHECK(!result.error && result.final_result && !result.stream && result.include_usage);
    CHECK(result.timings.prompt_n == 4 && result.timings.prompt_ms == 2.5);
}

static void value_copy_and_move() {
    auto source = sample_value(30);
    server_task_result copied(source);
    check_value(copied, 30);
    CHECK(copied.tokens.data() != source.tokens.data());
    copied.tokens[0] = 99;
    copied.data["tag"] = "changed";
    CHECK(source.tokens[0] == 3 && source.data["tag"] == "legacy");

    server_task_result copy_assigned{};
    copy_assigned = source;
    check_value(copy_assigned, 30);
    CHECK(copy_assigned.tokens.data() != source.tokens.data());

    auto * backing = source.tokens.data();
    server_task_result moved(std::move(source));
    check_value(moved, 30);
    CHECK(moved.tokens.data() == backing);

    auto assigned = sample_value(31);
    backing = moved.tokens.data();
    assigned = std::move(moved);
    check_value(assigned, 30);
    CHECK(assigned.tokens.data() == backing);
}

static void legacy_roundtrip_and_parent() {
    server_response response;
    response.add_waiting_task_id(40);
    auto source = sample_value(40);
    response.send(source);
    source.tokens[0] = 99;
    auto received = response.recv(40);
    check_value(received, 40);
    CHECK(response.queue_results_legacy.empty());
    response.remove_waiting_task_id(40);

    server_queue tasks;
    std::vector<int> children{51, 52};
    tasks.add_multitask(50, children);
    response.on_multitask_update([&](int parent, int child, server_task_result & result) {
        tasks.update_multitask(parent, child, result);
    });
    response.add_waiting_task_id(50);
    for (int child_id : children) {
        auto child = sample_value(child_id);
        child.id_multi = 50;
        response.send(child);
        child.tokens[0] = 99;
    }
    CHECK(tasks.queue_multitasks.size() == 1);
    const auto & parent = tasks.queue_multitasks.front();
    CHECK(parent.id == 50 && parent.subtasks_remaining.empty());
    CHECK(parent.results.size() == 2);
    CHECK(response.queue_results.empty() && response.queue_results_legacy.empty());
    for (size_t i = 0; i < parent.results.size(); ++i) {
        auto child = parent.results[i];
        CHECK(child.id_multi == 50);
        child.id_multi = -1;
        check_value(child, children[i]);
    }
    auto aggregate = sample_value(50);
    response.send(aggregate);
    auto parent_result = response.recv(50);
    check_value(parent_result, 50);
    CHECK(response.queue_results_legacy.empty());
}

struct reentrant_result : tracked_result {
    server_response & response;
    reentrant_result(int id, lifetime_counts & counts, server_response & response)
        : tracked_result(id, counts), response(response) {}
    ~reentrant_result() override { response.add_waiting_task_id(999); }
};

static void retirement_detaches_both_queues() {
    std::array<lifetime_counts, 6> counts{};
    server_response response;
    response.add_waiting_task_id(999);
    const std::array<int, 6> ids{60, 61, 60, 62, 61, 63};
    for (size_t i = 0; i < ids.size(); ++i) {
        response.add_waiting_task_id(ids[i]);
        response.send(std::make_unique<reentrant_result>(ids[i], counts[i], response));
    }
    for (int id : {60, 61, 62}) response.send(sample_value(id));
    int backing_releases = 0;
    const auto & first = response.queue_results_legacy.front();
    backing_watches[0] = {first.content.data(), &response, &backing_releases};
    backing_watches[1] = {first.tokens.data(), &response, &backing_releases};
    {
        retirement_allocation_probe probe(response);
        response.remove_waiting_task_id(60);
    }
    check_released(counts[0], 1);
    check_released(counts[2], 1);
    CHECK(backing_releases == 2);
    CHECK(!response.waiting_task_ids.count(60));
    std::vector<int> survivors;
    for (const auto & result : response.queue_results) survivors.push_back(result->id);
    CHECK(survivors == std::vector<int>({61, 62, 61, 63}));
    CHECK(response.queue_results_legacy.size() == 2);
    CHECK(response.queue_results_legacy.front().id == 61);
    CHECK(response.queue_results_legacy.back().id == 62);
    const std::unordered_set<int> retire{61, 63};
    {
        retirement_allocation_probe probe(response);
        response.remove_waiting_task_ids(retire);
        response.remove_waiting_task_ids(retire); // Idempotent, including no matches.
        response.remove_waiting_task_id(60);
    }
    for (size_t i : {0u, 1u, 2u, 4u, 5u}) check_released(counts[i], 1);
    check_released(counts[3], 0);
    CHECK(response.queue_results.size() == 1 && response.queue_results.front()->id == 62);
    CHECK(response.queue_results_legacy.size() == 1);
    auto value = response.recv(62);
    check_value(value, 62);
    auto dequeued = response.recv_with_timeout({62}, 0);
    response.remove_waiting_task_id(62);
    check_released(counts[3], 0); // Already-dequeued owner is not reclaimed.
    dequeued.reset();
    check_released(counts[3], 1);
    CHECK(response.queue_results.empty() && response.queue_results_legacy.empty());
    response.remove_waiting_task_ids({});
}

static void publication_and_retirement_ordering() {
    for (bool send_first : {false, true}) {
        lifetime_counts counts;
        server_response response;
        response.add_waiting_task_id(70);
        std::latch turn(1);
        std::thread producer([&] {
            if (!send_first) turn.wait();
            response.send(std::make_unique<tracked_result>(70, counts));
            if (send_first) turn.count_down();
        });
        if (send_first) turn.wait();
        response.remove_waiting_task_id(70);
        if (!send_first) turn.count_down();
        producer.join();
        CHECK(response.queue_results.empty());
        check_released(counts, 1);
        server_task_result_ptr named = std::make_unique<tracked_result>(70, counts);
        response.send(std::move(named));
        CHECK(named && response.queue_results.empty());
        named.reset();
        check_released(counts, 2);
    }
    server_response response;
    response.add_waiting_task_id(71);
    lifetime_counts counts;
    server_task_result_ptr named = std::make_unique<tracked_result>(71, counts);
    bool failed = false;
    deny_allocations = true;
    try { response.send(std::move(named)); } catch (const std::bad_alloc &) { failed = true; }
    deny_allocations = false;
    CHECK(failed && named && response.queue_results.empty());
    CHECK(response.waiting_task_ids.count(71));
    response.send(std::move(named));
    response.remove_waiting_task_id(71);
    check_released(counts, 1);
}

static void abandoned_legacy_parents() {
    server_queue tasks;
    server_response response(&tasks);
    response.add_waiting_task_id(999);
    response.on_multitask_update([&](int parent, int child, server_task_result & result) {
        tasks.update_multitask(parent, child, result);
    });
    std::vector<int> a_children{81, 82};
    std::vector<int> b_children{91, 92};
    response.add_waiting_task_id(80);
    response.add_waiting_task_id(90);
    CHECK(response.add_multitask(80, a_children));
    CHECK(response.add_multitask(90, b_children));
    auto first = sample_value(81);
    first.id_multi = 80;
    bool failed = false;
    deny_allocations = true;
    try { response.send(first); } catch (const std::bad_alloc &) { failed = true; }
    deny_allocations = false;
    CHECK(failed);
    CHECK(tasks.queue_multitasks.front().subtasks_remaining.size() == 2);
    CHECK(tasks.queue_multitasks.front().results.empty());
    // Exercise failure inside the real aggregator, not the by-value send copy.
    deny_allocations = true;
    failed = false;
    try { tasks.update_multitask(80, 81, first); } catch (const std::bad_alloc &) { failed = true; }
    deny_allocations = false;
    CHECK(failed && tasks.queue_multitasks.front().subtasks_remaining.size() == 2);
    CHECK(tasks.queue_multitasks.front().results.empty());
    response.send(first);
    response.send(first); // Duplicate terminal child does not duplicate retention.
    CHECK(tasks.queue_multitasks.front().results.size() == 1);
    int releases = 0;
    const auto & retained = tasks.queue_multitasks.front().results.front();
    backing_watches[0] = {retained.content.data(), &response, &releases, &tasks};
    backing_watches[1] = {retained.tokens.data(), &response, &releases, &tasks};
    {
        retirement_allocation_probe probe(response, &tasks);
        response.remove_waiting_task_id(80);
    }
    CHECK(releases == 2);
    CHECK(tasks.queue_multitasks.size() == 1 && tasks.queue_multitasks.front().id == 90);
    response.send(first); // Parent retired: late child cannot resurrect it.
    deny_allocations = true;
    CHECK(!response.add_multitask(80, a_children)); // Retired before creation.
    deny_allocations = false;
    CHECK(tasks.queue_multitasks.size() == 1);
    // Retire a parent whose child is independently registered: the child's own
    // queued delivery must survive, while the aggregate's copy is reclaimed.
    response.add_waiting_task_id(91);
    auto independent = sample_value(91);
    independent.id_multi = 90;
    response.send(independent);
    CHECK(tasks.queue_multitasks.front().results.size() == 1);
    CHECK(response.queue_results_legacy.size() == 1);
    response.remove_waiting_task_id(90);
    CHECK(tasks.queue_multitasks.empty());
    CHECK(response.queue_results_legacy.front().id == 91);
    CHECK(response.waiting_task_ids.count(91));
    response.remove_waiting_task_id(91);
    CHECK(response.queue_results_legacy.empty());

    // Real owner-loop completion/extraction, not a manually synthesized finish.
    response.add_waiting_task_id(100);
    std::vector<int> final_children{101, 102};
    CHECK(response.add_multitask(100, final_children));
    for (int id : final_children) {
        auto child = sample_value(id);
        child.id_multi = 100;
        response.send(child);
    }
    int finished = 0;
    tasks.on_new_task([](server_task &&) { CHECK(false); });
    tasks.on_finish_multitask([&](server_task_multi & parent) {
        { std::lock_guard<std::mutex> lock(tasks.mutex_tasks); }
        CHECK(parent.id == 100 && parent.results.size() == 2);
        ++finished;
        response.send(sample_value(100));
    });
    tasks.on_update_slots([&] { tasks.terminate(); });
    tasks.start_loop();
    CHECK(finished == 1 && tasks.queue_multitasks.empty());
    auto aggregate = response.recv(100);
    check_value(aggregate, 100);
    response.remove_waiting_task_id(100);
}

static void retirement_does_not_hold_lock_during_destruction() {
    struct blocked_result : tracked_result {
        std::latch & entered;
        std::latch & release;
        blocked_result(lifetime_counts & counts, std::latch & entered, std::latch & release)
            : tracked_result(110, counts), entered(entered), release(release) {}
        ~blocked_result() override { entered.count_down(); release.wait(); }
    };
    lifetime_counts counts;
    server_response response;
    std::latch entered(1), release(1);
    response.add_waiting_task_id(110);
    response.send(std::make_unique<blocked_result>(counts, entered, release));
    std::thread retire([&] { response.remove_waiting_task_id(110); });
    entered.wait();
    // Destructor is deliberately still live, but peers can publish/receive.
    response.add_waiting_task_id(111);
    response.send(sample_value(111));
    auto peer = response.recv(111);
    check_value(peer, 111);
    check_released(counts, 0);
    release.count_down();
    retire.join();
    check_released(counts, 1);
    response.remove_waiting_task_id(111);
}

static void explicit_receive_contract() {
    using clock = std::chrono::steady_clock;
    using status = server_receive_status;
    server_response response;
    response.add_waiting_task_id(120);
    response.add_waiting_task_id(121);
    CHECK(response.receive(120, clock::now()).status == status::timeout);
    CHECK(!response.recv_with_timeout({120}, -1));
    CHECK(!response.recv_with_timeout({120}, INT_MIN));
    CHECK(response.receive(std::unordered_set<int>{120}, clock::now()).status == status::timeout);
    CHECK(response.receive(122, clock::now()).status == status::retired);
    CHECK(response.receive(std::unordered_set<int>{}, clock::now()).status == status::retired);
    response.send(sample_value(120));
    auto value = response.receive(120, clock::time_point::min());
    CHECK(value.status == status::result && value.value);
    check_value(*value.value, 120);
    lifetime_counts counts;
    response.send(std::make_unique<tracked_result>(120, counts));
    auto pointer = response.receive(std::unordered_set<int>{120}, clock::now());
    CHECK(pointer.status == status::result && pointer.value && *pointer.value);
    response.remove_waiting_task_id(120);
    check_released(counts, 0);
    CHECK(response.receive(std::unordered_set<int>{120, 121}, clock::now()).status == status::timeout);
    response.send(std::make_unique<tracked_result>(121, counts));
    auto survivor = response.receive(std::unordered_set<int>{120, 121}, clock::now());
    CHECK(survivor.status == status::result && (*survivor.value)->id == 121);
    response.remove_waiting_task_id(121);
    CHECK(response.receive(std::unordered_set<int>{120, 121}, clock::now()).status == status::retired);
    pointer.value.reset();
    survivor.value.reset();
    check_released(counts, 2);

    response.add_waiting_task_id(123);
    response.send(sample_value(123));
    response.send(std::make_unique<tracked_result>(123, counts));
    response.close();
    response.close();
    CHECK(response.receive(123, clock::now()).status == status::result);
    auto draining = response.receive(std::unordered_set<int>{123}, clock::now());
    CHECK(draining.status == status::result);
    CHECK(response.receive(123).status == status::closed);
    CHECK(response.receive(std::unordered_set<int>{123}, clock::now()).status == status::closed);
    CHECK(response.receive(122, clock::now()).status == status::closed);
    CHECK(!response.add_waiting_task_id(124));
    std::vector<server_task> tasks(2);
    tasks[0].id = 124;
    tasks[1].id = 125;
    CHECK(!response.add_waiting_tasks(tasks));
    CHECK(!response.waiting_task_ids.count(124) && !response.waiting_task_ids.count(125));
    server_task_result_ptr rejected = std::make_unique<tracked_result>(123, counts);
    response.send(std::move(rejected));
    response.send(sample_value(123));
    CHECK(rejected && response.queue_results.empty() && response.queue_results_legacy.empty());
    bool explicit_failure = false;
    try { (void) response.recv(123); }
    catch (const server_receive_error & error) { explicit_failure = error.status == status::closed; }
    CHECK(explicit_failure);
    CHECK(!response.recv_with_timeout({123}, 0)); // Compatibility adapter only.
    CHECK(!response.recv_with_timeout({123}, INT_MAX));
}

static void receive_waiters_observe_terminal_state() {
    using status = server_receive_status;
    for (bool close_queue : {false, true}) {
        server_response response;
        response.add_waiting_task_id(130);
        response.add_waiting_task_id(131);
        response.send(sample_value(131)); // Must not satisfy legacy ID130.
        std::latch started(2);
        auto legacy = std::async(std::launch::async, [&] {
            started.count_down();
            return response.receive(130).status;
        });
        auto pointer = std::async(std::launch::async, [&] {
            started.count_down();
            return response.receive(std::unordered_set<int>{130}).status;
        });
        started.wait();
        if (close_queue) response.close();
        else response.remove_waiting_task_id(130);
        const auto expected = close_queue ? status::closed : status::retired;
        CHECK(legacy.get() == expected && pointer.get() == expected);
        CHECK(response.queue_results_legacy.size() == 1);
        auto peer = response.receive(131);
        CHECK(peer.status == status::result);
        check_value(*peer.value, 131);
        response.remove_waiting_task_id(131);
    }
}

static void asynchronous_matching_publication_succeeds() {
    using clock = std::chrono::steady_clock;
    using status = server_receive_status;
    lifetime_counts counts;
    server_response response;
    CHECK(response.add_waiting_task_id(160));
    CHECK(response.add_waiting_task_id(161));
    response.send(sample_value(161));
    server_task_result_ptr unrelated = std::make_unique<tracked_result>(161, counts);
    auto * unrelated_identity = unrelated.get();
    response.send(std::move(unrelated));
    std::latch launched(2);
    const auto deadline = clock::now() + std::chrono::seconds(2);
    auto legacy = std::async(std::launch::async, [&] {
        launched.count_down();
        return response.receive(160, deadline);
    });
    auto pointer = std::async(std::launch::async, [&] {
        launched.count_down();
        return response.receive(std::unordered_set<int>{160}, deadline);
    });
    launched.wait();
    // Publication follows receiver launch. This latch alone does NOT establish
    // that either receiver has already entered condition_variable::wait.
    response.send(sample_value(160));
    server_task_result_ptr matching = std::make_unique<tracked_result>(160, counts);
    auto * matching_identity = matching.get();
    response.send(std::move(matching));
    auto value = legacy.get();
    auto result = pointer.get();
    CHECK(value.status == status::result && value.value);
    check_value(*value.value, 160);
    CHECK(result.status == status::result && result.value);
    CHECK(result.value->get() == matching_identity);
    CHECK(response.queue_results.size() == 1);
    CHECK(response.queue_results.front().get() == unrelated_identity);
    CHECK(response.queue_results_legacy.size() == 1);
    CHECK(response.queue_results_legacy.front().id == 161);
    check_released(counts, 0);
    response.remove_waiting_task_ids({160, 161});
    check_released(counts, 1); // Unrelated queued owner only.
    result.value.reset();
    check_released(counts, 2); // Returned matching owner was retained by caller.
}

static void unrelated_wakes_do_not_renew_deadline() {
    using clock = std::chrono::steady_clock;
    using status = server_receive_status;
    server_response response;
    response.add_waiting_task_id(140);
    std::latch started(2);
    const auto deadline = clock::now() + std::chrono::milliseconds(60);
    auto pointer = std::async(std::launch::async, [&] {
        started.count_down();
        return response.receive(std::unordered_set<int>{140}, deadline).status;
    });
    auto legacy = std::async(std::launch::async, [&] {
        started.count_down();
        return response.receive(140, deadline).status;
    });
    started.wait();
    bool pointer_done = false, legacy_done = false;
    const auto watchdog = clock::now() + std::chrono::seconds(2);
    // Repeated notifications are deliberately unrelated to matching readiness.
    // The 2s bound is a test watchdog, not a hard runtime return-time guarantee.
    while (clock::now() < watchdog) {
        response.condition_results.notify_all();
        pointer_done = pointer.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready;
        legacy_done = legacy.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready;
        if (pointer_done && legacy_done) break;
    }
    response.close(); // Ensure joinability even on the failing path.
    CHECK(pointer_done && legacy_done);
    CHECK(pointer.get() == status::timeout && legacy.get() == status::timeout);
    CHECK(clock::now() >= deadline);
}

static void registration_failure_is_atomic() {
    server_response response;
    CHECK(response.add_waiting_task_id(150));
    std::vector<server_task> tasks(2);
    tasks[0].id = 151;
    tasks[1].id = 152;
    for (int fail_after : {0, 1}) {
        bool failed = false;
        allocations_before_failure = fail_after;
        try { response.add_waiting_tasks(tasks); }
        catch (const std::bad_alloc &) { failed = true; }
        allocations_before_failure = -1;
        CHECK(failed && response.waiting_task_ids.count(150));
        CHECK(!response.waiting_task_ids.count(151) && !response.waiting_task_ids.count(152));
    }
    CHECK(response.add_waiting_tasks(tasks));
    CHECK(response.waiting_task_ids.size() == 3);
    response.remove_waiting_task_ids({150, 151, 152});
}

int main() {
    registration_failure_is_atomic();
    explicit_receive_contract();
    receive_waiters_observe_terminal_state();
    asynchronous_matching_publication_succeeds();
    unrelated_wakes_do_not_renew_deadline();
    retirement_detaches_both_queues();
    abandoned_legacy_parents();
    retirement_does_not_hold_lock_during_destruction();
    publication_and_retirement_ordering();
    direct_ownership();
    registered_roundtrip();
    queued_teardown_and_erase();
    rejected_ownership();
    value_copy_and_move();
    legacy_roundtrip_and_parent();
    std::printf("unlocked retirement payload allocation observations: %zu\n", observed_retirement_allocations);
    std::puts("server result lifetime/queued retirement: OK (not HTTP/backend-quiescence coverage)");
    return 0;
}
