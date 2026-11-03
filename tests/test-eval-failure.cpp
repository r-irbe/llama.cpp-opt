// Host-only: a tiny synthetic model and failures injected into the real CPU
// backend/scheduler. No downloaded weights or GPU execution are required.
#include <random>
#include "llama.h"
#include "llama-context.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "../examples/server/server-decode-status.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
    std::exit(1); } } while (0)

static decltype(ggml_backend_i::graph_compute) real_compute;
static decltype(ggml_backend_buffer_type_i::alloc_buffer) real_alloc;
static ggml_status injected_status;
static int fail_call;
static int compute_calls;
static int successful_calls;
static bool fail_alloc;
static int alloc_failures;

struct deferred_copy {
    void * destination;
    std::vector<uint8_t> bytes;
};
static std::vector<deferred_copy> pending_copies;
static llama_context * deferred_context;
static decltype(ggml_backend_i::synchronize) real_synchronize;
static bool pending_at_failure;
static int drained_copies;

static ggml_status GGML_CALL inject_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (++compute_calls == fail_call) {
        if (deferred_context) pending_at_failure = !pending_copies.empty();
        return injected_status;
    }
    const ggml_status status = real_compute(backend, graph);
    successful_calls += status == GGML_STATUS_SUCCESS;
    return status;
}

static ggml_backend_buffer_t GGML_CALL inject_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    if (fail_alloc) {
        ++alloc_failures;
        return nullptr;
    }
    return real_alloc(buft, size);
}

struct tiny_model_file {
    std::string path;

    explicit tiny_model_file(bool t5 = false) {
        char name[] = "/tmp/llama-eval-failure-XXXXXX";
        const int fd = mkstemp(name);
        CHECK(fd >= 0);
        CHECK(close(fd) == 0);
        path = name;

        auto * file = gguf_init_empty();
        auto * tensors = ggml_init({1024 * 1024, nullptr, false});
        CHECK(file && tensors);
        const std::string arch = t5 ? "t5" : "llama";
        gguf_set_val_str(file, "general.architecture", arch.c_str());
        gguf_set_val_u32(file, (arch + ".block_count").c_str(), 1);
        gguf_set_val_u32(file, (arch + ".context_length").c_str(), 64);
        gguf_set_val_u32(file, (arch + ".embedding_length").c_str(), 32);
        gguf_set_val_u32(file, (arch + ".feed_forward_length").c_str(), 64);
        gguf_set_val_u32(file, (arch + ".attention.head_count").c_str(), 4);
        gguf_set_val_u32(file, (arch + ".attention.head_count_kv").c_str(), 4);
        gguf_set_val_f32(file, (arch + ".attention.layer_norm_rms_epsilon").c_str(), 1e-5f);
        if (t5) gguf_set_val_u32(file, "t5.attention.relative_buckets_count", 32);
        gguf_set_val_str(file, "tokenizer.ggml.model", "llama");
        gguf_set_val_u32(file, "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(file, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(file, "tokenizer.ggml.eos_token_id", 2);
        std::vector<std::string> words = {"<unk>", "<s>", "</s>"};
        for (int i = 3; i < 32; ++i) words.push_back("token" + std::to_string(i));
        std::vector<const char *> tokens;
        for (const auto & word : words) tokens.push_back(word.c_str());
        gguf_set_arr_str(file, "tokenizer.ggml.tokens", tokens.data(), tokens.size());

        auto add = [&](const char * tensor_name, int64_t n0, int64_t n1 = 0) {
            ggml_tensor * t = n1 ? ggml_new_tensor_2d(tensors, GGML_TYPE_F32, n0, n1)
                                : ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n0);
            ggml_set_name(t, tensor_name);
            float * data = static_cast<float *>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) data[i] = n1 ? 0.01f : 1.0f;
            gguf_add_tensor(file, t);
        };
        add("token_embd.weight", 32, 32);
        if (t5) {
            add("enc.output_norm.weight", 32);
            add("dec.output_norm.weight", 32);
            for (const std::string prefix : {"enc.blk.0.", "dec.blk.0."}) {
                add((prefix + "attn_norm.weight").c_str(), 32);
                for (const std::string projection : {"q", "k", "v", "o"}) {
                    add((prefix + "attn_" + projection + ".weight").c_str(), 32, 32);
                }
                add((prefix + "attn_rel_b.weight").c_str(), 4, 32);
                add((prefix + "ffn_norm.weight").c_str(), 32);
                add((prefix + "ffn_up.weight").c_str(), 32, 64);
                add((prefix + "ffn_down.weight").c_str(), 64, 32);
            }
            add("dec.blk.0.cross_attn_norm.weight", 32);
            for (const std::string projection : {"q", "k", "v", "o"}) {
                add(("dec.blk.0.cross_attn_" + projection + ".weight").c_str(), 32, 32);
            }
        } else {
        add("output_norm.weight", 32);
        add("output.weight", 32, 32);
        add("blk.0.attn_norm.weight", 32);
        add("blk.0.attn_q.weight", 32, 32);
        add("blk.0.attn_k.weight", 32, 32);
        add("blk.0.attn_v.weight", 32, 32);
        add("blk.0.attn_output.weight", 32, 32);
        add("blk.0.ffn_norm.weight", 32);
        add("blk.0.ffn_gate.weight", 32, 64);
        add("blk.0.ffn_up.weight", 32, 64);
        add("blk.0.ffn_down.weight", 64, 32);
        }
        gguf_write_to_file(file, path.c_str(), false);
        gguf_free(file);
        ggml_free(tensors);
    }

    ~tiny_model_file() { unlink(path.c_str()); }
};

using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static context_ptr make_context(llama_model * model) {
    auto params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = 8;
    params.n_ubatch = 2;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn = false;
    params.defrag_thold = -1.0f;
    context_ptr ctx(llama_init_from_model(model, params), llama_free);
    CHECK(ctx && ctx->backend_cpu);
    // Establish valid previous output: failure must not leave this visible.
    llama_token token = 3;
    CHECK(llama_decode(ctx.get(), llama_batch_get_one(&token, 1, 0, 0)) == 0);
    CHECK(llama_get_logits(ctx.get()) != nullptr);
    return ctx;
}

static void check_no_output(llama_context * ctx) {
    CHECK(ctx->n_outputs == 0);
    CHECK(ctx->n_outputs_embd == 0);
    CHECK(llama_get_logits(ctx) == nullptr);
    CHECK(llama_get_embeddings(ctx) == nullptr);
    CHECK(!ctx->prev && !ctx->prev_mtp);
}

static void test_compute_failure(llama_model * model, ggml_status status, int expected, int fail_at, bool encode) {
    auto ctx = make_context(model);
    real_compute = ctx->backend_cpu->iface.graph_compute;
    ctx->backend_cpu->iface.graph_compute = inject_compute;
    injected_status = status;
    fail_call = fail_at;
    compute_calls = successful_calls = 0;
    llama_token tokens[] = {3, 4, 5, 6};
    auto batch = llama_batch_get_one(tokens, encode ? 2 : 4, 1, 0);
    int8_t logits[] = {1, 1, 1, 1};
    batch.logits = logits; // request outputs from the successful first microbatch
    const int ret = encode ? llama_encode(ctx.get(), batch) : llama_decode(ctx.get(), batch);
    ctx->backend_cpu->iface.graph_compute = real_compute;
    CHECK(ret == expected);
    CHECK(compute_calls == fail_at); // no automatic replay after possible writes
    CHECK(successful_calls == fail_at - 1);
    check_no_output(ctx.get());
}

static void test_alloc_failure(llama_model * model, bool encode) {
    auto ctx = make_context(model);
    // A fresh scheduler has no reserved compute buffer. Fail only its host
    // backing allocator, after model/KV/output allocations have succeeded.
    ctx->reset_scheduler();
    ggml_backend_sched_free(ctx->sched);
    auto * buft = ggml_backend_cpu_buffer_type();
    real_alloc = buft->iface.alloc_buffer;
    buft->iface.alloc_buffer = inject_alloc;
    ggml_backend_t backend = ctx->backend_cpu;
    ctx->sched = ggml_backend_sched_new(&backend, &buft, 1, 8192, false);
    CHECK(ctx->sched);
    fail_alloc = true;
    alloc_failures = 0;
    llama_token tokens[] = {3, 4};
    auto batch = llama_batch_get_one(tokens, 2, 1, 0);
    const int ret = encode ? llama_encode(ctx.get(), batch) : llama_decode(ctx.get(), batch);
    fail_alloc = false;
    buft->iface.alloc_buffer = real_alloc;
    CHECK(alloc_failures > 0);
    CHECK(ret == -2);
    check_no_output(ctx.get());
    // Caller explicitly discards possibly partial state. The same failed
    // allocator plan must be usable again after pressure/fault disappears.
    llama_kv_cache_clear(ctx.get());
    CHECK(llama_decode(ctx.get(), llama_batch_get_one(tokens, 2, 0, 0)) == 0);
    CHECK(llama_get_logits(ctx.get()) != nullptr);
}

static void test_shift_failure(llama_model * model) {
    auto ctx = make_context(model);
    llama_kv_cache_seq_add(ctx.get(), 0, 0, -1, 1);
    CHECK(ctx->kv_self.has_shift);
    real_compute = ctx->backend_cpu->iface.graph_compute;
    ctx->backend_cpu->iface.graph_compute = inject_compute;
    injected_status = GGML_STATUS_ALLOC_FAILED;
    fail_call = 1;
    compute_calls = successful_calls = 0;
    const int ret = llama_kv_cache_update(ctx.get());
    ctx->backend_cpu->iface.graph_compute = real_compute;
    CHECK(ret == -2);
    CHECK(compute_calls == 1);
    CHECK(ctx->kv_self.has_shift); // not acknowledged as applied
    check_no_output(ctx.get());
}

static void test_defrag_failure(llama_model * model) {
    auto ctx = make_context(model);
    llama_token tokens[] = {3, 4};
    CHECK(llama_decode(ctx.get(), llama_batch_get_one(tokens, 2, 1, 0)) == 0);
    llama_kv_cache_seq_rm(ctx.get(), 0, 0, 1);
    llama_kv_cache_defrag(ctx.get());
    CHECK(ctx->kv_self.do_defrag);
    real_compute = ctx->backend_cpu->iface.graph_compute;
    ctx->backend_cpu->iface.graph_compute = inject_compute;
    injected_status = GGML_STATUS_FAILED;
    fail_call = 1;
    compute_calls = successful_calls = 0;
    const int ret = llama_kv_cache_update(ctx.get());
    ctx->backend_cpu->iface.graph_compute = real_compute;
    CHECK(ret == -1);
    CHECK(compute_calls == 1);
    CHECK(ctx->kv_self.do_defrag);
    check_no_output(ctx.get());
}

static void test_slot_warning(llama_model * model, bool after_progress) {
    auto ctx = make_context(model);
    // Leave exactly one microbatch free, or no free cells at all. Other
    // sequence IDs are masked out; this tests slot admission, not model quality.
    for (uint32_t i = after_progress ? 3 : 1; i < ctx->kv_self.size; ++i) {
        ctx->kv_self.cells[i].pos = i;
        ctx->kv_self.cells[i].seq_id.insert(1);
        ++ctx->kv_self.used;
    }
    real_compute = ctx->backend_cpu->iface.graph_compute;
    ctx->backend_cpu->iface.graph_compute = inject_compute;
    fail_call = -1;
    compute_calls = successful_calls = 0;
    llama_token tokens[] = {3, 4, 5, 6};
    const int ret = llama_decode(ctx.get(), llama_batch_get_one(tokens, 4, 1, 0));
    ctx->backend_cpu->iface.graph_compute = real_compute;
    CHECK(ret == (after_progress ? -1 : 1));
    CHECK(compute_calls == (after_progress ? 1 : 0));
    check_no_output(ctx.get());
}

static void test_shared_allocator_failure(bool recover) {
    auto * meta = ggml_init({1024 * 1024, nullptr, true});
    CHECK(meta);
    auto build = [&](int n) {
        auto * graph = ggml_new_graph_custom(meta, 64, false);
        auto * a = ggml_new_tensor_1d(meta, GGML_TYPE_F32, n);
        auto * b = ggml_new_tensor_1d(meta, GGML_TYPE_F32, n);
        ggml_build_forward_expand(graph, ggml_add(meta, a, b));
        return graph;
    };
    auto * buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t bufts[] = {buft, buft};
    auto * alloc = ggml_gallocr_new_n(bufts, 2);
    auto * small = build(32);
    std::vector<int> node_ids(64, 0), leaf_ids(64, 0);
    CHECK(ggml_gallocr_reserve_n(alloc, small, node_ids.data(), leaf_ids.data()));
    auto * large = build(4096);
    real_alloc = buft->iface.alloc_buffer;
    buft->iface.alloc_buffer = inject_alloc;
    fail_alloc = true;
    CHECK(!ggml_gallocr_reserve_n(alloc, large, node_ids.data(), leaf_ids.data()));
    CHECK(!ggml_gallocr_alloc_graph(alloc, large));
    fail_alloc = false;
    buft->iface.alloc_buffer = real_alloc;
    if (recover) {
        CHECK(ggml_gallocr_reserve_n(alloc, large, node_ids.data(), leaf_ids.data()));
        CHECK(ggml_gallocr_alloc_graph(alloc, large));
    }
    ggml_gallocr_free(alloc); // also valid immediately after failed alias growth
    ggml_free(meta);
}

static void GGML_CALL defer_get(ggml_backend_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    deferred_copy copy{data, std::vector<uint8_t>(size)};
    ggml_backend_tensor_get(tensor, copy.bytes.data(), offset, size);
    pending_copies.push_back(std::move(copy));
}

static void GGML_CALL drain_copies(ggml_backend_t backend) {
    if (!pending_copies.empty()) {
        // A backend owns these queued destinations until synchronization. If
        // cleanup invalidates outputs first, catch it before touching storage.
        CHECK(deferred_context && deferred_context->n_outputs > 0);
        for (const auto & copy : pending_copies) {
            std::memcpy(copy.destination, copy.bytes.data(), copy.bytes.size());
            ++drained_copies;
        }
        pending_copies.clear();
    }
    if (real_synchronize) real_synchronize(backend);
}

static void test_deferred_output_failure(llama_model * model) {
    auto ctx = make_context(model);
    auto & iface = ctx->backend_cpu->iface;
    const auto real_get = iface.get_tensor_async;
    real_synchronize = iface.synchronize;
    real_compute = iface.graph_compute;
    iface.get_tensor_async = defer_get;
    iface.synchronize = drain_copies;
    iface.graph_compute = inject_compute;
    deferred_context = ctx.get();
    pending_at_failure = false;
    drained_copies = 0;
    injected_status = GGML_STATUS_FAILED;
    fail_call = 2;
    compute_calls = successful_calls = 0;
    llama_token tokens[] = {3, 4, 5, 6};
    int8_t logits[] = {1, 1, 1, 1};
    auto batch = llama_batch_get_one(tokens, 4, 1, 0);
    batch.logits = logits;
    const int ret = llama_decode(ctx.get(), batch);
    iface.get_tensor_async = real_get;
    iface.synchronize = real_synchronize;
    iface.graph_compute = real_compute;
    deferred_context = nullptr;
    CHECK(ret == -1 && compute_calls == 2 && successful_calls == 1);
    CHECK(pending_at_failure && drained_copies > 0 && pending_copies.empty());
    check_no_output(ctx.get());
}

static void test_encoder_retry() {
    tiny_model_file file(true);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    auto * model = llama_model_load_from_file(file.path.c_str(), mp);
    CHECK(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 16;
    cp.n_ubatch = 2;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn = false;
    cp.defrag_thold = -1.0f;
    context_ptr ctx(llama_init_from_model(model, cp), llama_free);
    CHECK(ctx);
    llama_token tokens[] = {3, 4};
    CHECK(llama_encode(ctx.get(), llama_batch_get_one(tokens, 2, 0, 0)) == 0);
    llama_synchronize(ctx.get());
    CHECK(ctx->embd_enc.size() == 64);
    const auto encoded = ctx->embd_enc;
    const auto sequences = ctx->seq_ids_enc;
    // One free KV cell: two-token admission warns, one-token retry fits.
    for (uint32_t i = 1; i < ctx->kv_self.size; ++i) {
        ctx->kv_self.cells[i].pos = i;
        ctx->kv_self.cells[i].seq_id.insert(1);
        ++ctx->kv_self.used;
    }
    CHECK(llama_decode(ctx.get(), llama_batch_get_one(tokens, 2, 0, 0)) == 1);
    check_no_output(ctx.get());
    CHECK(ctx->embd_enc == encoded);
    CHECK(ctx->seq_ids_enc == sequences);
    CHECK(llama_decode(ctx.get(), llama_batch_get_one(tokens, 1, 0, 0)) == 0);
    CHECK(llama_get_logits(ctx.get()) != nullptr);
    CHECK(ctx->embd_enc == encoded);
    // An encode failure, unlike a decode admission warning, invalidates the
    // attempted encoder's output and must not publish the previous source.
    CHECK(llama_encode(ctx.get(), llama_batch_get_one(tokens, 0, 0, 0)) < 0);
    CHECK(ctx->embd_enc.empty() && ctx->seq_ids_enc.empty());
    ctx.reset();
    llama_free_model(model);
}

int main() {
    test_encoder_retry();
    test_shared_allocator_failure(false);
    test_shared_allocator_failure(true);
    CHECK(server_classify_decode(0) == server_decode_action::success);
    CHECK(server_classify_decode(1) == server_decode_action::retry_kv);
    CHECK(server_classify_decode(-2) == server_decode_action::allocation_failed);
    CHECK(server_classify_decode(-3) == server_decode_action::cancelled);
    CHECK(server_classify_decode(-1) == server_decode_action::failed);
    CHECK(server_classify_decode(2) == server_decode_action::failed);
    tiny_model_file file;
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = false;
    auto * model = llama_model_load_from_file(file.path.c_str(), params);
    CHECK(model);
    for (const auto status : {GGML_STATUS_ALLOC_FAILED, GGML_STATUS_FAILED, GGML_STATUS_ABORTED}) {
        const int expected = status == GGML_STATUS_ABORTED ? -3 : static_cast<int>(status);
        test_compute_failure(model, status, expected, 1, false);
        test_compute_failure(model, status, expected, 2, false);
        // Failure returns before encoder-only output interpretation; this tiny
        // decoder fixture tests the public encode failure boundary, not T5.
        test_compute_failure(model, status, expected, 1, true);
    }
    test_deferred_output_failure(model);
    test_alloc_failure(model, false);
    test_alloc_failure(model, true);
    test_shift_failure(model);
    test_defrag_failure(model);
    test_slot_warning(model, false);
    test_slot_warning(model, true);
    llama_free_model(model);
    std::puts("test-eval-failure: all host fault-injection checks passed");
}
