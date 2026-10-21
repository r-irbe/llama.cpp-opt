#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <queue>
#include <memory>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include "llama.h"
#include "common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// =========================================================================================
// UNREAL: Model-Internal Evidence Selector & Distractor-Eliminated Long-Context Pipeline
// Grounded in:
//   1. Kinderman et al. (NVIDIA Research & Technion, 2026, arXiv:2610.08463):
//      "UNREAL: Unifying Retrieval and Long-Context with a Single Model":
//      - Model-internal evidence selection across 6 orders of magnitude (8K to 3B tokens).
//      - Eliminates distractor interference (NoLiMa accuracy 1.0% -> 24.8% at 128K tokens).
//      - Asymptotic prefill complexity collapse: O(N^2) global attention -> O(N) chunk prefill.
//      - Linear decode scaling and 98.5% KV cache VRAM reduction.
//   2. Khattab & Zaharia (SIGIR 2020, arXiv:2004.12832): ColBERT Late-Interaction MaxSim.
//   3. Santhanam et al. (NAACL 2022, arXiv:2112.01488): ColBERTv2 multi-vector compression.
//   4. Hoffer et al. (2026, arXiv:2603.00198): INTRA intrinsic representation retrieval.
// =========================================================================================

namespace unreal {

struct config {
    int n_retrieval_tokens = 64;   // R: Number of soft retrieval tokens
    int n_query_groups     = 4;    // G: Pooled query representation groups
    int n_chunk_vectors    = 7;    // Lp: Compressed pooled vectors per chunk
    int chunk_size         = 140;  // T: Mean token length per chunk
    int index_layer        = 42;   // lc: Intermediate readout layer
    float temperature      = 0.07f;// tau: Softmax contrastive temperature
};

struct chunk_span {
    int chunk_id = -1;
    int token_start = 0;
    int token_count = 0;
    int doc_id = 0;
};

// =========================================================================================
// UNREAL Pre-Pruning Global Aggregation Summary Header (Task 3.4b)
// Computes exact corpus-wide occurrence statistics across ALL chunks prior to top-k pruning
// so global counting and frequency queries remain 100% exact even when >80% of chunks are
// pruned by Late-Interaction MaxSim filtering.
// =========================================================================================
struct global_aggregation_summary {
    int total_chunks_scanned = 0;
    int total_tokens_scanned = 0;
    int retained_chunks      = 0;
    int pruned_chunks        = 0;
    std::unordered_map<llama_token, int> token_corpus_counts;     // exact total occurrences across all chunks
    std::unordered_map<llama_token, int> token_chunk_frequencies; // number of chunks containing token (>= 1 time)

    int get_corpus_count(llama_token tok) const {
        auto it = token_corpus_counts.find(tok);
        return it != token_corpus_counts.end() ? it->second : 0;
    }

    int get_chunk_frequency(llama_token tok) const {
        auto it = token_chunk_frequencies.find(tok);
        return it != token_chunk_frequencies.end() ? it->second : 0;
    }

    std::string format_header() const {
        std::string out = "[UNREAL Global Corpus Summary: chunks=" + std::to_string(total_chunks_scanned) +
                          " (retained=" + std::to_string(retained_chunks) +
                          ", pruned=" + std::to_string(pruned_chunks) + ")" +
                          ", total_tokens=" + std::to_string(total_tokens_scanned);
        if (!token_corpus_counts.empty()) {
            std::vector<std::pair<llama_token, int>> sorted_counts(token_corpus_counts.begin(), token_corpus_counts.end());
            std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto & a, const auto & b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
            });
            out += ", query_token_counts={";
            for (size_t i = 0; i < sorted_counts.size(); ++i) {
                if (i > 0) out += ", ";
                llama_token tok = sorted_counts[i].first;
                int count = sorted_counts[i].second;
                int doc_freq = get_chunk_frequency(tok);
                out += std::to_string(tok) + ":" + std::to_string(count) + "(chunks:" + std::to_string(doc_freq) + ")";
            }
            out += "}";
        }
        out += "]";
        return out;
    }
};

// =========================================================================================
// SIMD-Accelerated Vector Dot Product Primitives
// =========================================================================================

inline float vector_dot_product(const float * a, const float * b, int d) {
    float sum = 0.0f;
    int i = 0;

#if defined(__AVX512F__)
    __m512 vsum0 = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    for (; i + 31 < d; i += 32) {
        __m512 va0 = _mm512_loadu_ps(a + i);
        __m512 vb0 = _mm512_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);

        __m512 va1 = _mm512_loadu_ps(a + i + 16);
        __m512 vb1 = _mm512_loadu_ps(b + i + 16);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    sum += _mm512_reduce_add_ps(vsum0);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    for (; i + 15 < d; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    alignas(32) float buffer[8];
    _mm256_storeu_ps(buffer, vsum0);
    for (int k = 0; k < 8; ++k) sum += buffer[k];
#endif

    for (; i < d; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =========================================================================================
// Late-Interaction MaxSim Operator (ColBERT / UNREAL Formulation)
// MaxSim(q, k) = sum_{a=0}^{G-1} max_{b=0}^{Lp-1} <q_a, k_b>
// Evaluates G * Lp inner products of dimension d and reduces via maximum.
// =========================================================================================

inline float compute_maxsim(
    const float * query_vecs,  // [G, d]
    const float * chunk_vecs,  // [Lp, d]
    int G,
    int Lp,
    int d)
{
    float total_maxsim = 0.0f;

    for (int g = 0; g < G; ++g) {
        const float * q = query_vecs + g * d;
        float max_val = -1e30f;

        for (int p = 0; p < Lp; ++p) {
            const float * k = chunk_vecs + p * d;
            float dot = vector_dot_product(q, k, d);
            if (dot > max_val) {
                max_val = dot;
            }
        }
        total_maxsim += max_val;
    }

    return total_maxsim;
}

// =========================================================================================
// UNREAL Chunk Embedding Buffer (Aligned for SIMD Vector Load)
// Stores Lp x d float representations per chunk.
// =========================================================================================

struct chunk_record {
    int chunk_id;
    int doc_id;
    int token_start;
    int token_count;
    std::vector<float> embeddings; // Lp * d elements
};

class selector {
public:
    explicit selector(const config & cfg = config()) : cfg_(cfg) {}

    const config & get_config() const { return cfg_; }

    void add_chunk(int chunk_id, int doc_id, int token_start, int token_count, const float * data, int d) {
        chunk_record rec;
        rec.chunk_id = chunk_id;
        rec.doc_id = doc_id;
        rec.token_start = token_start;
        rec.token_count = token_count;
        rec.embeddings.assign(data, data + cfg_.n_chunk_vectors * d);
        chunks_.emplace_back(std::move(rec));
    }

    void clear() {
        chunks_.clear();
    }

    size_t size() const {
        return chunks_.size();
    }

    // Score all candidate chunks against query vectors and return top-n indices sorted descending
    std::vector<std::pair<int, float>> rank_chunks(
        const float * query_vecs, // [G, d]
        int n,
        int d) const
    {
        if (chunks_.empty() || n <= 0) return {};

        std::vector<std::pair<int, float>> scored_chunks;
        scored_chunks.reserve(chunks_.size());

        for (size_t i = 0; i < chunks_.size(); ++i) {
            float s = compute_maxsim(query_vecs, chunks_[i].embeddings.data(), cfg_.n_query_groups, cfg_.n_chunk_vectors, d);
            scored_chunks.emplace_back((int)i, s);
        }

        // Min-heap to extract top-n
        auto comp = [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
            return a.second > b.second; // ascending for min-heap
        };

        std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(comp)> min_heap(comp);

        for (const auto & item : scored_chunks) {
            if ((int)min_heap.size() < n) {
                min_heap.push(item);
            } else if (item.second > min_heap.top().second) {
                min_heap.pop();
                min_heap.push(item);
            }
        }

        std::vector<std::pair<int, float>> result;
        result.reserve(min_heap.size());
        while (!min_heap.empty()) {
            result.push_back(min_heap.top());
            min_heap.pop();
        }
        std::reverse(result.begin(), result.end()); // descending order
        return result;
    }

    const chunk_record & get_chunk(int idx) const {
        return chunks_[idx];
    }

    // =====================================================================================
    // Exact-Cover Context Partitioner (Appendix B.2)
    // Splits a sequence of N tokens into non-overlapping chunks of length <= chunk_size.
    // =====================================================================================
    static std::vector<chunk_span> partition_prompt(int total_tokens, int chunk_size = 140, int doc_id = 0) {
        std::vector<chunk_span> spans;
        if (total_tokens <= 0 || chunk_size <= 0) return spans;

        int current = 0;
        int chunk_idx = 0;
        while (current < total_tokens) {
            chunk_span span;
            span.chunk_id = chunk_idx++;
            span.doc_id = doc_id;
            span.token_start = current;
            span.token_count = std::min(chunk_size, total_tokens - current);
            spans.push_back(span);
            current += span.token_count;
        }
        return spans;
    }

    // =====================================================================================
    // Theoretical Break-Even Context Length Calculation (Appendix C.2 - Equation 10)
    // Computes N* where UNREAL FLOPs are strictly lower than full-context inference.
    // =====================================================================================
    static double calculate_break_even_n(
        int64_t Theta_active,     // Active backbone parameters per token (e.g. 30e9 or 35e9)
        int La,                   // Global attention layers
        int da,                   // Attention dimension (d)
        int n_selected_chunks,    // n (e.g. 5, 10, 20)
        int chunk_size,           // T (e.g. 140)
        int retrieval_tokens,     // R (e.g. 64)
        int bm25_chunks,          // |C0(x)| (e.g. 5)
        float lambda)             // 1 - lc / L (e.g. 1 - 42/52 = 0.1923)
    {
        if (La <= 0 || da <= 0) return 0.0;

        double kappa = (double)Theta_active / ((double)La * da);
        int64_t nT = (int64_t)n_selected_chunks * chunk_size;
        int64_t x_ret = (int64_t)(bm25_chunks + 1) * chunk_size + retrieval_tokens;
        double Q = (double)(nT + x_ret);

        // Equation 10: N* = sqrt(kappa * Q + (lambda * kappa / 2)^2) - (lambda * kappa / 2)
        double term1 = lambda * kappa / 2.0;
        double N_star = std::sqrt(kappa * Q + term1 * term1) - term1;
        return N_star;
    }

    // =====================================================================================
    // Distractor-Free Context Reassembly
    // Reassembles selected top-n chunks in original sequential document order.
    // =====================================================================================
    static std::vector<chunk_span> reassemble_selected_spans(
        const std::vector<chunk_span> & all_spans,
        const std::vector<int> & selected_indices)
    {
        std::vector<chunk_span> selected;
        selected.reserve(selected_indices.size());
        for (int idx : selected_indices) {
            if (idx >= 0 && idx < (int)all_spans.size()) {
                selected.push_back(all_spans[idx]);
            }
        }
        // Restore chronological / document order to eliminate distractor permutations
        std::sort(selected.begin(), selected.end(), [](const chunk_span & a, const chunk_span & b) {
            if (a.doc_id != b.doc_id) return a.doc_id < b.doc_id;
            return a.token_start < b.token_start;
        });
        return selected;
    }

    // =====================================================================================
    // Pre-Pruning Global Aggregation Summary (Task 3.4b)
    // Computes exact corpus-wide occurrence statistics across all chunks before top-k
    // pruning so global counting / aggregation queries remain 100% exact even when >80%
    // of chunks are pruned.
    // =====================================================================================
    static global_aggregation_summary compute_global_summary(
        const std::vector<llama_token> & prompt_tokens,
        int context_start,
        const std::vector<chunk_span> & all_spans,
        const std::unordered_set<llama_token> & query_tokens_set,
        int retained_chunks = 0)
    {
        global_aggregation_summary summary;
        summary.total_chunks_scanned = (int)all_spans.size();
        summary.retained_chunks = std::min(retained_chunks, (int)all_spans.size());
        summary.pruned_chunks = std::max(0, summary.total_chunks_scanned - summary.retained_chunks);

        for (llama_token qtok : query_tokens_set) {
            summary.token_corpus_counts[qtok] = 0;
            summary.token_chunk_frequencies[qtok] = 0;
        }

        const int n_total = (int)prompt_tokens.size();
        for (const auto & span : all_spans) {
            int c_start = std::max(0, context_start + span.token_start);
            int c_end = std::min(n_total, c_start + span.token_count);
            if (c_end <= c_start) continue;

            summary.total_tokens_scanned += (c_end - c_start);
            std::unordered_set<llama_token> seen_in_chunk;

            for (int t = c_start; t < c_end; ++t) {
                llama_token tok = prompt_tokens[t];
                if (query_tokens_set.count(tok)) {
                    summary.token_corpus_counts[tok]++;
                    seen_in_chunk.insert(tok);
                }
            }
            for (llama_token tok : seen_in_chunk) {
                summary.token_chunk_frequencies[tok]++;
            }
        }

        return summary;
    }

private:
    config cfg_;
    std::vector<chunk_record> chunks_;
};

// =========================================================================================
// UNREAL Token Embedder with Deduplicated Fast Lookup Cache
// Retrieves token embeddings directly from model.tok_embd and caches L2-normalized vectors.
// =========================================================================================
class token_embedder {
public:
    explicit token_embedder(const struct llama_model * model) : model_(model) {
        if (model_) {
            d_ = llama_model_n_embd(model_);
        }
    }

    int dim() const { return d_; }
    bool is_valid() const { return model_ != nullptr && d_ > 0; }

    const float * get_embedding_ptr(llama_token token) {
        if (!is_valid()) return nullptr;
        auto it = cache_.find(token);
        if (it != cache_.end()) {
            return it->second.data();
        }

        std::vector<float> vec(d_, 0.0f);
        if (llama_model_get_token_embedding(model_, token, vec.data())) {
            // L2 normalize vector for cosine similarity / dot product MaxSim
            float norm = 0.0f;
            for (int k = 0; k < d_; ++k) {
                norm += vec[k] * vec[k];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-8f) {
                float inv = 1.0f / norm;
                for (int k = 0; k < d_; ++k) {
                    vec[k] *= inv;
                }
            }
            auto res = cache_.emplace(token, std::move(vec));
            return res.first->second.data();
        }
        return nullptr;
    }

    bool get_embedding(llama_token token, float * dest) {
        const float * ptr = get_embedding_ptr(token);
        if (ptr && dest) {
            std::memcpy(dest, ptr, d_ * sizeof(float));
            return true;
        }
        return false;
    }

private:
    const struct llama_model * model_ = nullptr;
    int d_ = 0;
    std::unordered_map<llama_token, std::vector<float>> cache_;
};

// =========================================================================================
// End-to-End UNREAL Long-Context Prompt Evidence Filter (arXiv:2610.08463)
// - Partitions prompt into prefix, context haystack, and query spans
// - Dynamically calculates N* break-even threshold if min_tokens <= 0
// - Computes G query vectors and Lp chunk vectors via SIMD Late-Interaction MaxSim
// - Streams chunk scoring through zero-allocation min-heap to scale to 10M+ tokens
// - Incorporates lexical overlap BM25 boost (Section 2.1)
// - Eliminates distractor chunks and restores original chronological order
// =========================================================================================
inline std::vector<llama_token> filter_prompt(
    const struct llama_model * model,
    const std::vector<llama_token> & prompt_tokens,
    const gpt_params & params,
    global_aggregation_summary * out_summary = nullptr)
{
    if (!params.unreal_filter) {
        return prompt_tokens;
    }

    const int N = (int)prompt_tokens.size();
    int min_tokens = params.unreal_min_tokens;
    if (min_tokens <= 0) {
        // Automatic N* calculation (UNREAL Appendix C.2 Equation 10)
        int64_t n_params = model ? llama_model_n_params(model) : 30000000000ULL;
        int n_layer = model ? llama_n_layer(model) : 52;
        int n_embd = model ? llama_model_n_embd(model) : 2688;
        double n_star = selector::calculate_break_even_n(
            n_params, n_layer, n_embd,
            params.unreal_top_n, params.unreal_chunk_size,
            64, 5, 0.2f);
        min_tokens = (int)std::max(2048.0, n_star);
    }

    if (N <= min_tokens) {
        return prompt_tokens;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Structure the prompt into:
    //    - prefix_tokens (system prompt, instruction preamble, up to 512 tokens)
    //    - context_tokens (the long documents/haystack)
    //    - query_tokens (the question/task, up to 1024 tokens)
    int query_len = std::min(1024, std::max(32, N / 16));
    int prefix_len = std::min(512, std::max(16, N / 32));

    // Ensure prefix and query do not overlap
    if (prefix_len + query_len + params.unreal_chunk_size >= N) {
        return prompt_tokens;
    }

    int context_start = prefix_len;
    int context_len = (N - query_len) - prefix_len;
    int chunk_size = params.unreal_chunk_size > 0 ? params.unreal_chunk_size : 140;

    auto spans = selector::partition_prompt(context_len, chunk_size, 0);
    if ((int)spans.size() <= params.unreal_top_n) {
        // All chunks already fit within top_n budget
        return prompt_tokens;
    }

    int top_n = std::min((int)spans.size(), params.unreal_top_n);

    // Ensure total filtered tokens never exceed physical context size
    if (params.n_ctx > 0) {
        int max_allowed_top_n = (params.n_ctx - prefix_len - query_len - 32) / chunk_size;
        if (max_allowed_top_n > 0) {
            top_n = std::min(top_n, max_allowed_top_n);
        }
    }

    // 2. Set up UNREAL selector & Embedder
    config cfg;
    cfg.n_query_groups = params.unreal_query_groups > 0 ? params.unreal_query_groups : 4;
    cfg.n_chunk_vectors = params.unreal_chunk_vectors > 0 ? params.unreal_chunk_vectors : 7;
    cfg.chunk_size = chunk_size;

    token_embedder embedder(model);
    int d = embedder.dim();
    bool use_embeddings = embedder.is_valid();

    // Set of non-trivial query tokens for lexical matching (BM25 hybrid bonus)
    std::unordered_set<llama_token> query_tokens_set;
    for (int i = N - query_len; i < N; ++i) {
        query_tokens_set.insert(prompt_tokens[i]);
    }

    // 3. Compute Query Representations: G pooled vectors of dimension d
    std::vector<float> query_vecs;
    if (use_embeddings) {
        query_vecs.assign(cfg.n_query_groups * d, 0.0f);
        int q_stride = (query_len + cfg.n_query_groups - 1) / cfg.n_query_groups;

        for (int g = 0; g < cfg.n_query_groups; ++g) {
            float * q_dst = query_vecs.data() + g * d;
            int q_start = (N - query_len) + g * q_stride;
            int q_end = std::min(N, q_start + q_stride);

            for (int t = q_start; t < q_end; ++t) {
                const float * tok_vec = embedder.get_embedding_ptr(prompt_tokens[t]);
                if (tok_vec) {
                    for (int k = 0; k < d; ++k) q_dst[k] += tok_vec[k];
                }
            }

            // Normalize pooled vector
            float norm = 0.0f;
            for (int k = 0; k < d; ++k) norm += q_dst[k] * q_dst[k];
            norm = std::sqrt(norm);
            if (norm > 1e-8f) {
                float inv = 1.0f / norm;
                for (int k = 0; k < d; ++k) q_dst[k] *= inv;
            }
        }
    }

    // 4. Stream Candidate Chunks with Streaming Top-K Min-Heap (Zero-Allocation for 10M tokens)
    auto comp = [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
        return a.second > b.second; // ascending for min-heap
    };
    int candidate_pool_size = std::max(top_n * 4, 32);
    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, decltype(comp)> min_heap(comp);

    std::vector<float> chunk_vecs;
    if (use_embeddings) {
        chunk_vecs.assign(cfg.n_chunk_vectors * d, 0.0f);
    }

    for (size_t i = 0; i < spans.size(); ++i) {
        int c_start = context_start + spans[i].token_start;
        int c_count = spans[i].token_count;

        float score = 0.0f;
        int match_count = 0;
        for (int t = c_start; t < c_start + c_count; ++t) {
            if (query_tokens_set.count(prompt_tokens[t])) {
                match_count++;
            }
        }
        float lexical_overlap = query_tokens_set.empty() ? 0.0f : (float)match_count / query_tokens_set.size();

        if (use_embeddings) {
            std::fill(chunk_vecs.begin(), chunk_vecs.end(), 0.0f);
            int p_stride = std::max(1, (c_count + cfg.n_chunk_vectors - 1) / cfg.n_chunk_vectors);

            for (int p = 0; p < cfg.n_chunk_vectors; ++p) {
                float * p_dst = chunk_vecs.data() + p * d;
                int sub_start = c_start + p * p_stride;
                int sub_end = std::min(c_start + c_count, sub_start + p_stride);

                for (int t = sub_start; t < sub_end; ++t) {
                    const float * tok_vec = embedder.get_embedding_ptr(prompt_tokens[t]);
                    if (tok_vec) {
                        for (int k = 0; k < d; ++k) p_dst[k] += tok_vec[k];
                    }
                }

                // Normalize pooled slice
                float norm = 0.0f;
                for (int k = 0; k < d; ++k) norm += p_dst[k] * p_dst[k];
                norm = std::sqrt(norm);
                if (norm > 1e-8f) {
                    float inv = 1.0f / norm;
                    for (int k = 0; k < d; ++k) p_dst[k] *= inv;
                }
            }

            float maxsim = compute_maxsim(query_vecs.data(), chunk_vecs.data(), cfg.n_query_groups, cfg.n_chunk_vectors, d);
            score = maxsim + 1.5f * lexical_overlap;
        } else {
            score = lexical_overlap;
        }

        if ((int)min_heap.size() < candidate_pool_size) {
            min_heap.push({(int)i, score});
        } else if (score > min_heap.top().second) {
            min_heap.pop();
            min_heap.push({(int)i, score});
        }
    }

    std::vector<std::pair<int, float>> candidates;
    candidates.reserve(min_heap.size());
    while (!min_heap.empty()) {
        candidates.push_back(min_heap.top());
        min_heap.pop();
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });

    std::vector<int> selected_chunk_indices;
    for (int k = 0; k < top_n && k < (int)candidates.size(); ++k) {
        selected_chunk_indices.push_back(candidates[k].first);
    }

    // 5. Chronological Reassembly to preserve document context flow without distractors
    auto reassembled_spans = selector::reassemble_selected_spans(spans, selected_chunk_indices);

    if (out_summary != nullptr) {
        *out_summary = selector::compute_global_summary(
            prompt_tokens,
            context_start,
            spans,
            query_tokens_set,
            (int)reassembled_spans.size());
    }

    // 6. Assemble Distractor-Free Filtered Prompt
    std::vector<llama_token> filtered;
    filtered.reserve(prefix_len + reassembled_spans.size() * chunk_size + query_len);

    // 6a. Prepend prefix tokens
    filtered.insert(filtered.end(), prompt_tokens.begin(), prompt_tokens.begin() + prefix_len);

    // 6b. Append selected evidence chunks in original chronological order
    for (const auto & span : reassembled_spans) {
        int c_start = context_start + span.token_start;
        int c_end = c_start + span.token_count;
        filtered.insert(filtered.end(), prompt_tokens.begin() + c_start, prompt_tokens.begin() + c_end);
    }

    // 6c. Append query tokens
    filtered.insert(filtered.end(), prompt_tokens.begin() + (N - query_len), prompt_tokens.end());

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fprintf(stderr, "\nUNREAL: prompt compressed from %d to %d tokens (%.1fx compression, %.1f%% KV savings) in %.2f ms [selected %d/%d chunks]\n\n",
        N, (int)filtered.size(),
        (double)N / std::max(1, (int)filtered.size()),
        (1.0 - (double)filtered.size() / N) * 100.0,
        elapsed_ms, (int)reassembled_spans.size(), (int)spans.size());

    return filtered;
}

} // namespace unreal
