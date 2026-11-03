// Host-only tests of production reduction ownership and all-fences-before-retire.
// No CUDA runtime, GPU execution or model allocations.
#include "ggml-cuda/reduction-lifecycle.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

struct context {
    int device = 0;
    const void * model = nullptr;
    void * streams[GGML_CUDA_MAX_DEVICES][2] {};
    void * cublas_handles[GGML_CUDA_MAX_DEVICES] {};
    std::unique_ptr<int> pools[GGML_CUDA_MAX_DEVICES];
    int cached_graphs = 1;
};

struct fixture {
    int key = 0, other_key = 0;
    context a, b, duplicate, copy, foreign;
    std::array<context *, GGML_CUDA_MAX_DEVICES> primary {};
    std::vector<context *> members;
    ggml_tensor dst {}, sources[3] {};
    ggml_tensor * nodes[1] {&dst};
    ggml_cgraph graph {};

    fixture() {
        a.model = b.model = duplicate.model = copy.model = &key;
        b.device = 1;
        copy.device = 2;
        foreign.model = &other_key;
        foreign.device = 1;
        members = {&a, &b, &duplicate};
        primary[0] = &a;
        primary[1] = &b;
        graph.n_nodes = 1;
        graph.nodes = nodes;
        std::fill_n(dst.ne, GGML_MAX_DIMS, int64_t(1));
        dst.op = GGML_OP_REDUCE;
        dst.op_params[0] = GGML_OP_ADD;
        dst.op_params[1] = dst.op_params[2] = 2;
        for (int d = 0; d < 3; ++d) {
            std::fill_n(sources[d].ne, GGML_MAX_DIMS, int64_t(1));
            sources[d].data = &sources[d];
            dst.src[d] = &sources[d];
        }
        dst.data = sources[0].data;
    }

    bool valid(const context & caller) const {
        return ggml_cuda_reduction_contexts_valid(caller, graph, &primary, &members, 4);
    }
};

int main() {
    {
        fixture f;
        CHECK(f.valid(f.a));
        CHECK(!f.valid(f.duplicate)); // same device/group does not make streams interchangeable
        f.primary[1] = nullptr;
        CHECK(!f.valid(f.a));
        f.primary[1] = &f.foreign; // not live membership: do not dereference it as an owner
        CHECK(!f.valid(f.a));
        f.primary[1] = &f.b;
        f.b.model = &f.other_key;
        CHECK(!f.valid(f.a));
        f.b.model = &f.key;
        f.b.device = 2;
        CHECK(!f.valid(f.a));
        f.b.device = 1;
        f.dst.op_params[1] = GGML_CUDA_MAX_DEVICES + 1;
        CHECK(!f.valid(f.a));
        f.dst.op_params[1] = -1;
        CHECK(!f.valid(f.a));
        f.dst.op_params[1] = 2;
        f.dst.op_params[4] = 1; // caller cannot be copy-only
        CHECK(!f.valid(f.a));
        f.dst.op_params[4] = 1 << 4; // outside reduction's device range
        CHECK(!f.valid(f.a));
        f.dst.op_params[4] = 0;
        CHECK(!ggml_cuda_reduction_contexts_valid(f.a, f.graph, &f.primary, &f.members, 1));
        CHECK(!ggml_cuda_reduction_contexts_valid(f.a, f.graph,
                static_cast<const decltype(f.primary) *>(nullptr), &f.members, 4));
        CHECK(!ggml_cuda_reduction_contexts_valid(f.a, f.graph, &f.primary,
                static_cast<const decltype(f.members) *>(nullptr), 4));
    }
    {
        fixture f;
        f.dst.op_params[1] = 3;
        f.dst.op_params[4] = 1 << 2; // copy-only destination must also be live
        CHECK(!f.valid(f.a));
        f.primary[2] = &f.copy;
        f.members.push_back(&f.copy);
        CHECK(f.valid(f.a));
        f.dst.src[2] = nullptr;
        CHECK(!f.valid(f.a));
    }
    {
        fixture f;
        f.primary[0] = nullptr; // departing primary; duplicate remains, no promotion
        f.members.erase(f.members.begin());
        f.dst.data = f.sources[1].data;
        CHECK(!f.valid(f.b));
        CHECK(!f.valid(f.duplicate));
        // Execution skips an empty REDUCE before touching registry ownership.
        // Test every dimension, and do not accidentally skip a later active node.
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            f.dst.ne[d] = 0;
            CHECK(f.valid(f.b));
            CHECK(ggml_cuda_reduction_contexts_valid(f.duplicate, f.graph,
                    static_cast<const decltype(f.primary) *>(nullptr),
                    static_cast<const decltype(f.members) *>(nullptr), 0));
            f.dst.ne[d] = 1;
            CHECK(!f.valid(f.b));
        }
        ggml_tensor empty = f.dst;
        empty.ne[0] = 0;
        ggml_tensor * mixed[] = {&empty, &f.dst};
        f.graph.nodes = mixed;
        f.graph.n_nodes = 2;
        CHECK(!f.valid(f.b));
        f.graph.nodes = f.nodes;
        f.graph.n_nodes = 1;
        // Non-reduction and explicitly disabled workloads keep existing behavior.
        f.dst.op = GGML_OP_NONE;
        CHECK(ggml_cuda_reduction_contexts_valid(f.duplicate, f.graph,
                static_cast<const decltype(f.primary) *>(nullptr),
                static_cast<const decltype(f.members) *>(nullptr), 0));
        f.dst.op = GGML_OP_REDUCE;
        f.dst.op_params[3] = 1;
        CHECK(f.valid(f.duplicate));
    }
    {
        fixture f;
        // Devices 2 and 3 own cross-device stream/handle/pool resources despite
        // no owning group context on either device.
        f.a.streams[2][1] = &f.key;
        f.b.cublas_handles[2] = &f.key;
        f.b.pools[3] = std::make_unique<int>(0);
        std::vector<int> fences;
        int retired = 0;
        CHECK(!ggml_cuda_retire_reduction_group(f.a, &f.members, 4,
            [&](int d) { fences.push_back(d); return d != 1; },
            [&](context &) { ++retired; }));
        CHECK((fences == std::vector<int>{0, 1}));
        CHECK(retired == 0 && f.a.cached_graphs == 1 && f.b.cached_graphs == 1);
        fences.clear();
        CHECK(ggml_cuda_retire_reduction_group(f.a, &f.members, 4,
            [&](int d) { fences.push_back(d); return true; },
            [&](context & peer) {
                CHECK((fences == std::vector<int>{0, 1, 2, 3}));
                peer.cached_graphs = 0;
                ++retired;
            }));
        CHECK(retired == 3 && f.a.cached_graphs == 0 && f.duplicate.cached_graphs == 0);
        CHECK(f.foreign.cached_graphs == 1); // other group unaffected
        fences.clear();
        // No graph cache callback work (e.g. graphs disabled) still drains.
        CHECK(ggml_cuda_retire_reduction_group(f.a, &f.members, 4,
            [&](int d) { fences.push_back(d); return true; }, [](context &) {}));
        CHECK((fences == std::vector<int>{0, 1, 2, 3}));
        fences.clear();
        CHECK(!ggml_cuda_retire_reduction_group(f.a, &f.members, 3,
            [&](int d) { fences.push_back(d); return true; }, [](context &) {}));
        CHECK(fences.empty()); // resource on an unavailable device: no destructive work
    }
    std::puts("test-cuda-reduction-lifecycle: all host ownership checks passed");
}
