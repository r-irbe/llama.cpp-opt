// Exercise the real parallel scheduler with CPU storage and mock computation.
// No GPU backend, model, or production fault-injection switch is required.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#ifdef TEST_GGML_OPENMP
#include <omp.h>
#endif

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

static ggml_backend_t backends[3];
static int calls[3];
static std::atomic<int> synchronizations[3];
static std::thread::id threads[3];
static bool used_openmp[3];
static bool expect_native;
static ggml_tensor * backend_one_nodes[2];
static ggml_tensor * backend_two_node;
static int failure_call;
static ggml_status failure_status;
static ggml_status later_failure_status;
static ggml_status other_failure_status;

static ggml_status GGML_CALL compute(ggml_backend_t backend, ggml_cgraph *) {
    for (int i = 0; i < 3; ++i) {
        if (backend != backends[i]) continue;
        // Each backend has one owning scheduler worker. The scheduler joins
        // workers before the caller reads these per-backend records.
        threads[i] = std::this_thread::get_id();
#ifdef TEST_GGML_OPENMP
        used_openmp[i] = used_openmp[i] || omp_in_parallel() != 0;
#endif
        const int call = ++calls[i];
        if (i == 1 && call == 1) return other_failure_status;
        if (i == 0 && call == failure_call) return failure_status;
        if (i == 0 && call == failure_call + 1) return later_failure_status;
        return GGML_STATUS_SUCCESS;
    }
    std::abort();
}

static void GGML_CALL synchronize(ggml_backend_t backend) {
    for (int i = 0; i < 3; ++i) {
        if (backend == backends[i]) ++synchronizations[i];
    }
}

static bool GGML_CALL supports_op(ggml_backend_t backend, const ggml_tensor * node) {
    // CPU storage is compatible across these mocks. Prevent the scheduler's
    // higher-priority upgrade pass from erasing the intended backend ownership.
    if (node == backend_one_nodes[0] || node == backend_one_nodes[1]) return backend == backends[1];
    if (node == backend_two_node) return backend == backends[2];
    // The mock submits no kernels, including the multi-backend REDUCE marker.
    return true;
}

static void run_case(bool openmp_eligible, ggml_status status, int fail_at,
                     ggml_status later_status = GGML_STATUS_SUCCESS,
                     ggml_status other_status = GGML_STATUS_SUCCESS) {
    backend_one_nodes[0] = backend_one_nodes[1] = backend_two_node = nullptr;
    ggml_guid fake_guid = {0x57, 0xb4, 0xee, 0x91};
    ggml_guid_t original_guids[3];
    for (int i = 0; i < 3; ++i) {
        backends[i] = ggml_backend_cpu_init();
        CHECK(backends[i]);
        original_guids[i] = backends[i]->guid;
        backends[i]->iface.graph_compute = compute;
        backends[i]->iface.synchronize = synchronize;
        backends[i]->iface.supports_op = supports_op;
        // Non-CPU identity selects the OpenMP-eligible branch; all storage,
        // allocation, copy and destruction remain real host CPU operations.
        // The last backend remains CPU as required by scheduler construction.
        if (openmp_eligible && i < 2) backends[i]->guid = &fake_guid;
        calls[i] = 0;
        synchronizations[i] = 0;
        used_openmp[i] = false;
        threads[i] = {};
    }
    auto * meta = ggml_init({1024 * 1024, nullptr, true});
    CHECK(meta);
    auto * graph = ggml_new_graph_custom(meta, 64, false);
    auto * a = ggml_scale(meta, ggml_new_tensor_1d(meta, GGML_TYPE_F32, 32), 2.0f);
    auto * b = ggml_scale(meta, ggml_new_tensor_1d(meta, GGML_TYPE_F32, 32), 3.0f);
    ggml_tensor * sources[] = {a, b};
    auto * reduced = ggml_reduce(meta, sources, 2, GGML_OP_ADD);
    auto * c = ggml_scale(meta, reduced, 4.0f);
    // Keep REDUCE alone, followed by another split on the same backend.
    c->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
    auto * d = ggml_scale(meta, c, 5.0f);
    // The highest backend must have real work: a two-thread OpenMP team
    // otherwise hides the missing owner because this backend was idle.
    auto * e = ggml_scale(meta, d, 6.0f);
    backend_one_nodes[0] = b;
    backend_one_nodes[1] = d;
    backend_two_node = e;
    ggml_build_forward_expand(graph, e);
    auto * sched = ggml_backend_sched_new(backends, nullptr, 3, 128, false);
    CHECK(sched);
    ggml_backend_sched_set_split_mode_graph(sched, true, true);
    ggml_backend_sched_set_tensor_backend(sched, a, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, b, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, reduced, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, c, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, d, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, e, backends[2]);
    failure_call = fail_at;
    failure_status = status;
    later_failure_status = later_status;
    other_failure_status = other_status;
    const auto expected = status != GGML_STATUS_SUCCESS ? status : other_status;
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));
    for (auto & n : synchronizations) n = 0;
#ifdef TEST_GGML_OPENMP
    const bool caller_in_parallel = omp_in_parallel() != 0;
#endif
    auto check_submission = [&] {
        // Exact counts also reject partial OpenMP work followed by replay.
        CHECK(calls[0] == 3 && calls[1] == 2 && calls[2] == 1);
        CHECK(threads[0] == std::this_thread::get_id());
        CHECK(threads[1] != threads[0] && threads[2] != threads[0]);
        CHECK(threads[1] != threads[2]);
#ifdef TEST_GGML_OPENMP
        const bool expected_openmp = openmp_eligible && !expect_native;
        CHECK(used_openmp[0] == (expected_openmp || caller_in_parallel));
        CHECK(used_openmp[1] == expected_openmp);
        CHECK(used_openmp[2] == expected_openmp);
#endif
    };
    const auto result = ggml_backend_sched_graph_compute_async(sched, graph);
    if (calls[0] != 3 || calls[1] != 2 || calls[2] != 1) {
        std::fprintf(stderr, "mode=%d splits=%d calls=%d,%d,%d status=%d\n", int(openmp_eligible),
            ggml_backend_sched_get_n_splits(sched), calls[0], calls[1], calls[2], int(result));
    }
    check_submission();
    CHECK(result == expected);
    if (expected != GGML_STATUS_SUCCESS) {
        // Even the otherwise idle third backend is fenced before an async
        // terminal error returns. Worker joins alone do not drain device work.
        CHECK(synchronizations[0] > 0 && synchronizations[1] > 0 && synchronizations[2] > 0);
    }
    // Reusing the worker pool must start a fresh per-evaluation status record.
    failure_call = -1;
    later_failure_status = GGML_STATUS_SUCCESS;
    other_failure_status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < 3; ++i) {
        calls[i] = 0;
        used_openmp[i] = false;
        threads[i] = {};
    }
    CHECK(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    check_submission();
    ggml_backend_sched_free(sched);
    ggml_free(meta);
    for (int i = 0; i < 3; ++i) {
        backends[i]->guid = original_guids[i];
        ggml_backend_free(backends[i]);
    }
}

static void run_suite() {
    for (bool eligible : {false, true}) {
        run_case(eligible, GGML_STATUS_SUCCESS, -1);
        for (const auto status : {GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_ABORTED}) {
            run_case(eligible, status, 1);
            run_case(eligible, status, 2);
            run_case(eligible, status, 3);
        }
        run_case(eligible, GGML_STATUS_FAILED, 1, GGML_STATUS_ALLOC_FAILED);
        run_case(eligible, GGML_STATUS_ABORTED, 1, GGML_STATUS_FAILED);
        // Failure on a non-primary worker and simultaneous distinct errors:
        // select by backend index, not whichever worker finishes last.
        run_case(eligible, GGML_STATUS_SUCCESS, -1, GGML_STATUS_SUCCESS, GGML_STATUS_FAILED);
        run_case(eligible, GGML_STATUS_ABORTED, 1, GGML_STATUS_SUCCESS, GGML_STATUS_ALLOC_FAILED);
        run_case(eligible, GGML_STATUS_ALLOC_FAILED, 3, GGML_STATUS_SUCCESS, GGML_STATUS_FAILED);
    }
}

int main(int argc, char ** argv) {
    const bool nested = argc == 2 && std::strcmp(argv[1], "--nested-serialized") == 0;
    expect_native = argc == 2 && std::strcmp(argv[1], "--expect-native") == 0;
    CHECK(argc == 1 || nested || expect_native);
    if (nested) {
#ifdef TEST_GGML_OPENMP
        expect_native = true;
        omp_set_dynamic(0);
        // This separate process changes no caller/runtime settings in normal tests.
#if _OPENMP >= 200805
        omp_set_max_active_levels(1);
#else
        omp_set_nested(0);
#endif
        bool exercised = false;
        #pragma omp parallel num_threads(2)
        {
            #pragma omp single
            {
                const int outer_width = omp_get_num_threads();
                int inner_width = 0;
                #pragma omp parallel num_threads(3)
                {
                    #pragma omp single
                    inner_width = omp_get_num_threads();
                }
                if (outer_width > 1 && inner_width == 1) {
                    run_suite();
                    exercised = true;
                }
            }
        }
        if (!exercised) {
            std::puts("SKIP: runtime did not provide an active outer/serialized inner team");
            return 77;
        }
#else
        std::puts("SKIP: OpenMP is not enabled");
        return 77;
#endif
    } else {
        run_suite();
    }
    std::puts("test-parallel-split-failure: exact dispatch, failure retention and reuse passed");
}

