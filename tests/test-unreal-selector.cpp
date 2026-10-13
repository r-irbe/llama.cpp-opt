#include "unreal-selector.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <random>
#include <chrono>

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion FAILED: " << #cond << " at line " << __LINE__ << std::endl; \
            return 1; \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::fabs((a) - (b)) > (eps)) { \
            std::cerr << "Assertion FAILED: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ") at line " << __LINE__ << std::endl; \
            return 1; \
        } \
    } while (0)

int main() {
    std::cout << "===================================================================\n";
    std::cout << "  ik_llama.cpp - UNREAL Model-Internal Selector Unit Test Suite   \n";
    std::cout << "  Grounded in Kinderman et al. (NVIDIA Research 2026, 2610.08463)  \n";
    std::cout << "===================================================================\n\n";

    // ---------------------------------------------------------------------------------
    // 1. Vector Dot Product Correctness across multiple dimensions
    // ---------------------------------------------------------------------------------
    std::cout << "[Step 1] Verifying SIMD vector dot product accuracy vs scalar reference...\n";
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<int> test_dims = {32, 64, 128, 256, 512, 1024, 2048, 2688};
    for (int d : test_dims) {
        std::vector<float> a(d);
        std::vector<float> b(d);
        float ref_dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
            ref_dot += a[i] * b[i];
        }
        float simd_dot = unreal::vector_dot_product(a.data(), b.data(), d);
        ASSERT_NEAR(simd_dot, ref_dot, 1e-3f);
    }
    std::cout << "  Vector dot products validated across dimensions [32..2688]   OK\n";

    // ---------------------------------------------------------------------------------
    // 2. Late-Interaction MaxSim Operator Correctness
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 2] Verifying Late-Interaction MaxSim operator mathematical invariants...\n";
    int G = 4;
    int Lp = 7;
    int d = 128;
    std::vector<float> query_vecs(G * d);
    std::vector<float> chunk_vecs(Lp * d);

    for (int i = 0; i < G * d; ++i) query_vecs[i] = dist(rng);
    for (int i = 0; i < Lp * d; ++i) chunk_vecs[i] = dist(rng);

    // Compute scalar reference MaxSim
    float ref_maxsim = 0.0f;
    for (int g = 0; g < G; ++g) {
        float max_dot = -1e30f;
        for (int p = 0; p < Lp; ++p) {
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) {
                dot += query_vecs[g * d + k] * chunk_vecs[p * d + k];
            }
            if (dot > max_dot) max_dot = dot;
        }
        ref_maxsim += max_dot;
    }

    float test_maxsim = unreal::compute_maxsim(query_vecs.data(), chunk_vecs.data(), G, Lp, d);
    ASSERT_NEAR(test_maxsim, ref_maxsim, 1e-3f);
    std::cout << "  Late-Interaction MaxSim math matches ground truth exactly    OK\n";

    // ---------------------------------------------------------------------------------
    // 3. Exact-Cover Context Partitioner (Appendix B.2)
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 3] Verifying Exact-Cover context partitioner...\n";
    int total_tokens = 131072; // 128K tokens
    int chunk_size = 140;
    auto spans = unreal::selector::partition_prompt(total_tokens, chunk_size, 0);

    ASSERT_TRUE(!spans.empty());
    int expected_chunks = (total_tokens + chunk_size - 1) / chunk_size;
    ASSERT_TRUE((int)spans.size() == expected_chunks);

    int covered = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        ASSERT_TRUE(spans[i].chunk_id == (int)i);
        ASSERT_TRUE(spans[i].token_start == covered);
        if (i + 1 < spans.size()) {
            ASSERT_TRUE(spans[i].token_count == chunk_size);
        } else {
            ASSERT_TRUE(spans[i].token_count == total_tokens - covered);
        }
        covered += spans[i].token_count;
    }
    ASSERT_TRUE(covered == total_tokens);
    std::cout << "  Exact-cover partitioner generated " << spans.size() << " non-overlapping chunks for 128K tokens OK\n";

    // ---------------------------------------------------------------------------------
    // 4. Gold Evidence Retrieval & Distractor Filtering
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 4] Verifying evidence retrieval and distractor elimination...\n";
    unreal::config cfg;
    cfg.n_query_groups = 4;
    cfg.n_chunk_vectors = 7;
    cfg.chunk_size = 140;
    unreal::selector sel(cfg);

    int num_chunks = 100;
    int embed_dim = 256;
    int gold_chunk_id = 42;

    // Create a distinctive query
    std::vector<float> query(cfg.n_query_groups * embed_dim, 0.0f);
    for (int g = 0; g < cfg.n_query_groups; ++g) {
        query[g * embed_dim + g * 10] = 5.0f; // salient features
    }

    for (int i = 0; i < num_chunks; ++i) {
        std::vector<float> chunk_emb(cfg.n_chunk_vectors * embed_dim);
        if (i == gold_chunk_id) {
            // Gold chunk perfectly matches the query salient features
            for (int p = 0; p < cfg.n_chunk_vectors; ++p) {
                for (int k = 0; k < embed_dim; ++k) chunk_emb[p * embed_dim + k] = 0.01f * dist(rng);
                chunk_emb[p * embed_dim + (p % cfg.n_query_groups) * 10] = 4.5f;
            }
        } else {
            // Distractor random chunk
            for (int k = 0; k < cfg.n_chunk_vectors * embed_dim; ++k) {
                chunk_emb[k] = 0.05f * dist(rng);
            }
        }
        sel.add_chunk(i, 0, i * 140, 140, chunk_emb.data(), embed_dim);
    }

    auto top_ranked = sel.rank_chunks(query.data(), 5, embed_dim);
    ASSERT_TRUE(!top_ranked.empty());
    ASSERT_TRUE(top_ranked[0].first == gold_chunk_id);
    std::cout << "  Gold evidence (Chunk " << gold_chunk_id << ") successfully retrieved at Rank 0 (Score: " 
              << top_ranked[0].second << " vs Distractor Rank 1: " << top_ranked[1].second << ") OK\n";

    // ---------------------------------------------------------------------------------
    // 5. Distractor-Free Context Reassembly
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 5] Verifying distractor-free context reassembly...\n";
    std::vector<int> selected_ids = {88, 12, 42, 5};
    auto reassembled = unreal::selector::reassemble_selected_spans(spans, selected_ids);

    ASSERT_TRUE(reassembled.size() == 4);
    ASSERT_TRUE(reassembled[0].chunk_id == 5);
    ASSERT_TRUE(reassembled[1].chunk_id == 12);
    ASSERT_TRUE(reassembled[2].chunk_id == 42);
    ASSERT_TRUE(reassembled[3].chunk_id == 88);
    std::cout << "  Chronological document order restored across selected evidence spans OK\n";

    // ---------------------------------------------------------------------------------
    // 6. Theoretical Break-Even Context Length (Appendix C.2 - Equation 10)
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 6] Validating theoretical break-even context length (N*)...\n";
    // Parameters for Nemotron-3.5-Lightning-30B-A3B:
    // Active params: ~30B, L=52, da=2688, La=52 (or dense equivalent), lc=42 -> lambda = 1 - 42/52 = 0.1923
    int64_t Theta_nemotron = 30000000000ULL;
    int La_nemotron = 52;
    int da_nemotron = 2688;
    int n_chunks = 10;
    int T_chunk = 140;
    int R_retrieval = 64;
    int bm25_c0 = 5;
    float lambda_nemotron = 1.0f - (42.0f / 52.0f);

    double N_star = unreal::selector::calculate_break_even_n(
        Theta_nemotron, La_nemotron, da_nemotron, n_chunks, T_chunk, R_retrieval, bm25_c0, lambda_nemotron);

    std::cout << "  Calculated break-even context length N* for Nemotron-3.5-Lightning-30B: " 
              << (int)N_star << " tokens\n";
    ASSERT_TRUE(N_star >= 5000.0 && N_star <= 40000.0);
    std::cout << "  N* aligns with theoretical break-even formulation OK\n";

    // ---------------------------------------------------------------------------------
    // 7. Latency and Scalability Benchmark (1000 chunks)
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 7] High-throughput MaxSim latency benchmark (1,000 candidate chunks)...\n";
    unreal::selector large_sel(cfg);
    int big_chunks = 1000;
    std::vector<float> dummy_chunk(cfg.n_chunk_vectors * embed_dim, 0.01f);
    for (int i = 0; i < big_chunks; ++i) {
        large_sel.add_chunk(i, 0, i * 140, 140, dummy_chunk.data(), embed_dim);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto big_top = large_sel.rank_chunks(query.data(), 10, embed_dim);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Scored " << big_chunks << " chunks (28,000 vector dot products) in " 
              << elapsed_ms << " ms (" << (big_chunks / (elapsed_ms / 1000.0)) << " chunks/sec) OK\n";
    ASSERT_TRUE(big_top.size() == 10);

    // ---------------------------------------------------------------------------------
    // 8. End-to-End Prompt Filtering & Distractor Elimination (filter_prompt)
    // ---------------------------------------------------------------------------------
    std::cout << "\n[Step 8] Verifying end-to-end prompt evidence filtering & distractor collapse...\n";
    {
        gpt_params params;
        params.unreal_filter = true;
        params.unreal_min_tokens = 4096;
        params.unreal_chunk_size = 140;
        params.unreal_top_n = 5;

        // Create a 16K long context prompt with distractors and gold evidence needles
        int prompt_len = 16384;
        std::vector<llama_token> long_prompt(prompt_len);
        for (int i = 0; i < prompt_len; ++i) {
            long_prompt[i] = (llama_token)(100 + (i % 500)); // background distractors
        }

        // Distinctive query at the end
        int query_len = 64;
        for (int i = prompt_len - query_len; i < prompt_len; ++i) {
            long_prompt[i] = (llama_token)(9000 + (i % 20)); // rare query tokens
        }

        // Plant gold evidence needle in chunk 25 (around token 3500)
        int needle_pos = 3550;
        for (int i = 0; i < 20; ++i) {
            long_prompt[needle_pos + i] = (llama_token)(9000 + i);
        }

        // Test disabled filter returns original prompt
        params.unreal_filter = false;
        auto unfiltered = unreal::filter_prompt(nullptr, long_prompt, params);
        ASSERT_TRUE(unfiltered.size() == long_prompt.size());

        // Test enabled filter compresses prompt and retains gold evidence
        params.unreal_filter = true;
        auto filtered = unreal::filter_prompt(nullptr, long_prompt, params);
        ASSERT_TRUE(filtered.size() < long_prompt.size());
        ASSERT_TRUE(filtered.size() <= (size_t)(128 + params.unreal_top_n * params.unreal_chunk_size + 256));

        // Verify gold needle tokens are preserved in the filtered output
        bool needle_found = false;
        for (size_t i = 0; i + 10 < filtered.size(); ++i) {
            if (filtered[i] == 9000 && filtered[i + 1] == 9001 && filtered[i + 2] == 9002) {
                needle_found = true;
                break;
            }
        }
        ASSERT_TRUE(needle_found);
        std::cout << "  Filtered prompt compressed from " << long_prompt.size() << " to " 
                  << filtered.size() << " tokens (" << (double)long_prompt.size() / filtered.size() 
                  << "x compression) with gold evidence preserved OK\n";
    }

    std::cout << "\n===================================================================\n";
    std::cout << "  ALL UNREAL SELECTOR TESTS PASSED WITH 100% SUCCESS               \n";
    std::cout << "===================================================================\n";
    return 0;
}
