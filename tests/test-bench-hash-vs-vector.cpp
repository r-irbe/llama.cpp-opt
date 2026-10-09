#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// High-resolution clock helper
using bench_clock_t = std::chrono::high_resolution_clock;
using duration_us_t = std::chrono::duration<double, std::micro>;

// ============================================================================
// Benchmark 1: Repetition Penalty (Hash Map vs Direct Vector Indexing)
// ============================================================================

static void bench_repetition_penalties(int n_vocab, int penalty_last_n, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 1: Repetition Penalties (n_vocab=" << n_vocab 
              << ", penalty_last_n=" << penalty_last_n << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Setup candidate array
    std::vector<llama_token_data> candidates_buf(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        candidates_buf[i] = { (llama_token)i, (float)(i % 100) * 0.1f - 5.0f, 0.0f };
    }
    llama_token_data_array candidates = { candidates_buf.data(), (size_t)n_vocab, false };

    // Setup synthetic last_tokens history with realistic token distributions
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> tok_dist(100, 2000); // 1900 unique active tokens
    std::vector<llama_token> last_tokens(penalty_last_n);
    for (int i = 0; i < penalty_last_n; ++i) {
        last_tokens[i] = (llama_token)tok_dist(rng);
    }

    const float penalty_repeat  = 1.10f;
    const float penalty_freq    = 0.20f;
    const float penalty_present = 0.20f;

    // Baseline: std::unordered_map + full vocabulary scan
    double time_baseline_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            std::unordered_map<llama_token, int> token_count;
            for (size_t i = 0; i < (size_t)penalty_last_n; ++i) {
                token_count[last_tokens[i]]++;
            }

            for (size_t i = 0; i < candidates.size; ++i) {
                const auto token_iter = token_count.find(candidates.data[i].id);
                if (token_iter == token_count.end()) {
                    continue;
                }
                const int count = token_iter->second;
                if (candidates.data[i].logit <= 0) {
                    candidates.data[i].logit *= penalty_repeat;
                } else {
                    candidates.data[i].logit /= penalty_repeat;
                }
                candidates.data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
            }
        }
        auto t1 = bench_clock_t::now();
        time_baseline_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    // Reset candidates
    for (int i = 0; i < n_vocab; ++i) {
        candidates_buf[i] = { (llama_token)i, (float)(i % 100) * 0.1f - 5.0f, 0.0f };
    }

    // Optimized: Flat vector freqs + O(1) direct candidate indexing
    double time_optimized_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            struct token_freq {
                llama_token id;
                int count;
            };
            std::vector<token_freq> freqs;
            freqs.reserve(std::min<size_t>(penalty_last_n, 128));

            for (size_t i = 0; i < (size_t)penalty_last_n; ++i) {
                const llama_token tok = last_tokens[i];
                bool found = false;
                for (auto & tf : freqs) {
                    if (tf.id == tok) {
                        tf.count++;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    freqs.push_back({tok, 1});
                }
            }

            for (const auto & tf : freqs) {
                if (tf.id >= 0 && (size_t)tf.id < candidates.size && candidates.data[tf.id].id == tf.id) {
                    auto & item = candidates.data[tf.id];
                    if (item.logit <= 0) {
                        item.logit *= penalty_repeat;
                    } else {
                        item.logit /= penalty_repeat;
                    }
                    item.logit -= float(tf.count) * penalty_freq + float(tf.count > 0) * penalty_present;
                } else {
                    for (size_t i = 0; i < candidates.size; ++i) {
                        if (candidates.data[i].id == tf.id) {
                            auto & item = candidates.data[i];
                            if (item.logit <= 0) {
                                item.logit *= penalty_repeat;
                            } else {
                                item.logit /= penalty_repeat;
                            }
                            item.logit -= float(tf.count) * penalty_freq + float(tf.count > 0) * penalty_present;
                            break;
                        }
                    }
                }
            }
        }
        auto t1 = bench_clock_t::now();
        time_optimized_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double avg_baseline_us  = time_baseline_us / iterations;
    double avg_optimized_us = time_optimized_us / iterations;
    double speedup = avg_baseline_us / avg_optimized_us;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (unordered_map + 248k probes) : " << avg_baseline_us << " us / step (" 
              << 1e6 / avg_baseline_us << " calls/sec)\n";
    std::cout << "  Optimized (flat vector + direct index) : " << avg_optimized_us << " us / step (" 
              << 1e6 / avg_optimized_us << " calls/sec)\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_baseline_us - avg_optimized_us) << " us saved per token)\n\n";
}

// ============================================================================
// Benchmark 2: Speculative Rejection Sampling Distribution (q_probs)
// ============================================================================

static void bench_q_probs(int proposal_size, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 2: Speculative Proposal Distribution (Q_size=" << proposal_size 
              << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<llama_token> q_ids(proposal_size);
    std::vector<float> q_vals(proposal_size);
    for (int i = 0; i < proposal_size; ++i) {
        q_ids[i] = 1000 + i * 17;
        q_vals[i] = 1.0f / (float)(i + 1);
    }
    const llama_token target_token = q_ids[proposal_size / 2];

    // Baseline: std::unordered_map per draft token
    double time_baseline_us = 0.0;
    float sum_baseline = 0.0f;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            std::unordered_map<llama_token, float> q_probs;
            q_probs.reserve(q_ids.size());
            for (size_t j = 0; j < q_ids.size(); ++j) {
                q_probs[q_ids[j]] += q_vals[j];
            }
            const auto it_tok = q_probs.find(target_token);
            sum_baseline += (it_tok == q_probs.end() ? 0.0f : it_tok->second);
        }
        auto t1 = bench_clock_t::now();
        time_baseline_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    // Optimized: std::vector<token_prob> with linear register scan
    double time_optimized_us = 0.0;
    float sum_optimized = 0.0f;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            struct token_prob {
                llama_token id;
                float prob;
            };
            std::vector<token_prob> q_probs;
            q_probs.reserve(q_ids.size());
            for (size_t j = 0; j < q_ids.size(); ++j) {
                bool found = false;
                for (auto & qp : q_probs) {
                    if (qp.id == q_ids[j]) {
                        qp.prob += q_vals[j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    q_probs.push_back({q_ids[j], q_vals[j]});
                }
            }
            float val = 0.0f;
            for (const auto & qp : q_probs) {
                if (qp.id == target_token) {
                    val = qp.prob;
                    break;
                }
            }
            sum_optimized += val;
        }
        auto t1 = bench_clock_t::now();
        time_optimized_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double avg_baseline_us  = time_baseline_us / iterations;
    double avg_optimized_us = time_optimized_us / iterations;
    double speedup = avg_baseline_us / avg_optimized_us;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Baseline (unordered_map allocation): " << avg_baseline_us << " us / call\n";
    std::cout << "  Optimized (contiguous flat vector) : " << avg_optimized_us << " us / call\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_baseline_us - avg_optimized_us) * 1000.0 << " ns saved per draft token)\n\n";
}

// ============================================================================
// Benchmark 3: Suffix Trie Node Traversals (std::unordered_map vs Flat Vector)
// ============================================================================

struct old_suffix_node {
    int64_t count = 0;
    std::unordered_map<llama_token, std::unique_ptr<old_suffix_node>> children;
};

struct new_suffix_node {
    int64_t count = 0;
    std::vector<std::pair<llama_token, std::unique_ptr<new_suffix_node>>> children;

    new_suffix_node * find_child(llama_token tok) const {
        for (const auto & p : children) {
            if (p.first == tok) return p.second.get();
        }
        return nullptr;
    }

    new_suffix_node * create_child(llama_token tok) {
        auto child = std::make_unique<new_suffix_node>();
        auto * ptr = child.get();
        children.emplace_back(tok, std::move(child));
        return ptr;
    }
};

static void bench_suffix_trie(int n_tokens, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 3: Suffix Trie Insert & Lookup (n_tokens=" << n_tokens 
              << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    // Generate repetitive code-like token stream
    std::vector<llama_token> tokens(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        tokens[i] = (llama_token)((i % 32) + 100);
    }

    // Baseline: old suffix node
    double time_old_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            auto root = std::make_unique<old_suffix_node>();
            for (int s = 0; s < n_tokens - 16; ++s) {
                old_suffix_node * node = root.get();
                for (int i = 0; i < 16; ++i) {
                    const llama_token tok = tokens[s + i];
                    auto it_c = node->children.find(tok);
                    if (it_c == node->children.end()) {
                        auto child = std::make_unique<old_suffix_node>();
                        auto * ptr = child.get();
                        ptr->count = 1;
                        node->children[tok] = std::move(child);
                        node = ptr;
                    } else {
                        node = it_c->second.get();
                        node->count++;
                    }
                }
            }
        }
        auto t1 = bench_clock_t::now();
        time_old_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    // Optimized: new flat vector suffix node
    double time_new_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            auto root = std::make_unique<new_suffix_node>();
            for (int s = 0; s < n_tokens - 16; ++s) {
                new_suffix_node * node = root.get();
                for (int i = 0; i < 16; ++i) {
                    const llama_token tok = tokens[s + i];
                    auto * child = node->find_child(tok);
                    if (!child) {
                        child = node->create_child(tok);
                        child->count = 1;
                    } else {
                        child->count++;
                    }
                    node = child;
                }
            }
        }
        auto t1 = bench_clock_t::now();
        time_new_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double avg_old_us = time_old_us / iterations;
    double avg_new_us = time_new_us / iterations;
    double speedup = avg_old_us / avg_new_us;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (unordered_map node) : " << avg_old_us << " us / pass\n";
    std::cout << "  Optimized (flat vector node)  : " << avg_new_us << " us / pass\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_old_us - avg_new_us) << " us saved per trie pass)\n\n";
}

// ============================================================================
// Benchmark 4: DRY Sampler (Full Vocabulary Scan vs Direct Array Access)
// ============================================================================

static void bench_dry_sampler(int n_vocab, int penalty_last_n, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 4: DRY Sampler Logit Penalty Application (n_vocab=" << n_vocab 
              << ", penalty_last_n=" << penalty_last_n << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<llama_token_data> candidates_buf(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        candidates_buf[i] = { (llama_token)i, (float)(i % 100) * 0.1f - 5.0f, 0.0f };
    }
    llama_token_data_array candidates = { candidates_buf.data(), (size_t)n_vocab, false };

    // Setup synthetic repeat tokens
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> tok_dist(100, 2000);
    std::vector<llama_token> last_tokens(penalty_last_n);
    for (int i = 0; i < penalty_last_n; ++i) {
        last_tokens[i] = (llama_token)tok_dist(rng);
    }

    const float dry_multiplier = 0.8f;
    const float dry_base = 1.75f;
    const int dry_allowed_length = 2;

    // Baseline: std::unordered_map + full vocabulary scan (248k entries)
    double time_baseline_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            std::unordered_map<llama_token, int> dry_max_token_repeat;
            for (int i = 0; i < penalty_last_n - 1; ++i) {
                llama_token tok = last_tokens[i];
                dry_max_token_repeat[tok] = std::max(dry_max_token_repeat[tok], (i % 5) + 2);
            }

            for (size_t i = 0; i < candidates.size; ++i) {
                const auto it_rep = dry_max_token_repeat.find(candidates.data[i].id);
                if (it_rep != dry_max_token_repeat.end()) {
                    int repeat_exp = it_rep->second - dry_allowed_length;
                    float penalty = dry_multiplier * std::pow(dry_base, repeat_exp);
                    candidates.data[i].logit -= penalty;
                }
            }
        }
        auto t1 = bench_clock_t::now();
        time_baseline_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    // Reset candidates
    for (int i = 0; i < n_vocab; ++i) {
        candidates_buf[i] = { (llama_token)i, (float)(i % 100) * 0.1f - 5.0f, 0.0f };
    }

    // Optimized: flat vector + direct indexing candidates[token_id]
    double time_opt_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            std::vector<std::pair<llama_token, int>> dry_max_token_repeat;
            for (int i = 0; i < penalty_last_n - 1; ++i) {
                llama_token tok = last_tokens[i];
                int rep_len = (i % 5) + 2;
                auto it = std::find_if(dry_max_token_repeat.begin(), dry_max_token_repeat.end(),
                    [tok](const std::pair<llama_token, int> & p) { return p.first == tok; });
                if (it == dry_max_token_repeat.end()) {
                    dry_max_token_repeat.push_back({tok, rep_len});
                } else if (it->second < rep_len) {
                    it->second = rep_len;
                }
            }

            for (const auto & entry : dry_max_token_repeat) {
                const llama_token tok_id = entry.first;
                const int max_rep = entry.second;
                if (tok_id >= 0 && (size_t)tok_id < candidates.size && candidates.data[tok_id].id == tok_id) {
                    int repeat_exp = max_rep - dry_allowed_length;
                    float penalty = dry_multiplier * std::pow(dry_base, repeat_exp);
                    candidates.data[tok_id].logit -= penalty;
                }
            }
        }
        auto t1 = bench_clock_t::now();
        time_opt_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double avg_baseline_us = time_baseline_us / iterations;
    double avg_opt_us = time_opt_us / iterations;
    double speedup = avg_baseline_us / avg_opt_us;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (unordered_map + 248k scan) : " << avg_baseline_us << " us / step (" 
              << (1000000.0 / avg_baseline_us) << " calls/sec)\n";
    std::cout << "  Optimized (flat vector + direct index): " << avg_opt_us << " us / step ("
              << (1000000.0 / avg_opt_us) << " calls/sec)\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_baseline_us - avg_opt_us) << " us saved per token!)\n\n";
}

// ============================================================================
// Benchmark 5: Character Trie Prefix Traversal (std::map vs Flat Sorted Vector)
// ============================================================================

struct old_naive_trie {
    bool has_value = false;
    llama_token value = 0;
    std::map<char, old_naive_trie> children;

    void insert(const char * key, size_t len, int32_t val = 0) {
        if (len == 0) {
            has_value = true;
            value = val;
            return;
        }
        children[key[0]].insert(key + 1, len - 1, val);
    }

    const old_naive_trie * traverse(char c) const {
        auto it = children.find(c);
        return it != children.end() ? &it->second : nullptr;
    }
};

struct new_naive_trie {
    bool has_value = false;
    llama_token value = 0;
    std::vector<std::pair<uint8_t, std::unique_ptr<new_naive_trie>>> children;

    void insert(const char * key, size_t len, int32_t val = 0) {
        if (len == 0) {
            has_value = true;
            value = val;
            return;
        }
        const uint8_t uc = static_cast<uint8_t>(key[0]);
        auto it = std::lower_bound(children.begin(), children.end(), uc,
            [](const std::pair<uint8_t, std::unique_ptr<new_naive_trie>> & p, uint8_t v) { return p.first < v; });
        if (it != children.end() && it->first == uc) {
            it->second->insert(key + 1, len - 1, val);
        } else {
            auto node = std::make_unique<new_naive_trie>();
            auto * ptr = node.get();
            children.insert(it, std::make_pair(uc, std::move(node)));
            ptr->insert(key + 1, len - 1, val);
        }
    }

    const new_naive_trie * traverse(char c) const {
        const uint8_t uc = static_cast<uint8_t>(c);
        for (const auto & child : children) {
            if (child.first == uc) return child.second.get();
            if (child.first > uc) return nullptr;
        }
        return nullptr;
    }
};

static void bench_character_trie(int n_words, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 5: Tokenizer Trie Traversal (n_words=" << n_words 
              << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<std::string> vocab_words;
    for (int i = 0; i < n_words; ++i) {
        vocab_words.push_back("token_" + std::to_string(i * 17 % 100000));
    }

    old_naive_trie old_trie;
    for (int i = 0; i < n_words; ++i) {
        old_trie.insert(vocab_words[i].c_str(), vocab_words[i].size(), i);
    }

    new_naive_trie new_trie;
    for (int i = 0; i < n_words; ++i) {
        new_trie.insert(vocab_words[i].c_str(), vocab_words[i].size(), i);
    }

    volatile uintptr_t sink = 0;
    double time_old_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            for (const auto & word : vocab_words) {
                const old_naive_trie * node = &old_trie;
                for (char c : word) {
                    node = node->traverse(c);
                    if (!node) break;
                }
                sink += (uintptr_t)node;
            }
        }
        auto t1 = bench_clock_t::now();
        time_old_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double time_new_us = 0.0;
    {
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            for (const auto & word : vocab_words) {
                const new_naive_trie * node = &new_trie;
                for (char c : word) {
                    node = node->traverse(c);
                    if (!node) break;
                }
                sink += (uintptr_t)node;
            }
        }
        auto t1 = bench_clock_t::now();
        time_new_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }
    (void)sink;

    double avg_old_us = time_old_us / iterations;
    double avg_new_us = time_new_us / iterations;
    double speedup = avg_old_us / avg_new_us;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (std::map trie node)     : " << avg_old_us << " us / pass\n";
    std::cout << "  Optimized (contiguous vector trie): " << avg_new_us << " us / pass\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_old_us - avg_new_us) << " us saved per traversal)\n\n";
}

// ============================================================================
// Benchmark 6: N-Gram Cache Part Lookups & Memory Footprint
// ============================================================================

static void bench_ngram_cache(int n_ngrams, int iterations) {
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "Benchmark 6: N-Gram Cache Distribution Queries (n_ngrams=" << n_ngrams 
              << ", iterations=" << iterations << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    std::vector<std::vector<std::pair<llama_token, int32_t>>> raw_data(n_ngrams);
    std::mt19937 rng(54321);
    std::uniform_int_distribution<int> tok_dist(100, 50000);
    for (int i = 0; i < n_ngrams; ++i) {
        int n_followers = (i % 3) + 1;
        for (int j = 0; j < n_followers; ++j) {
            raw_data[i].push_back({ (llama_token)tok_dist(rng), (int32_t)((j + 1) * 3) });
        }
    }

    double time_old_us = 0.0;
    {
        std::vector<std::unordered_map<llama_token, int32_t>> old_cache(n_ngrams);
        for (int i = 0; i < n_ngrams; ++i) {
            for (const auto & p : raw_data[i]) {
                old_cache[i][p.first] = p.second;
            }
        }

        volatile int64_t sink_old = 0;
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            for (int i = 0; i < n_ngrams; ++i) {
                std::unordered_map<llama_token, int32_t> copy = old_cache[i];
                int max_c = 0;
                llama_token max_t = -1;
                for (const auto & entry : copy) {
                    if (entry.second > max_c) {
                        max_c = entry.second;
                        max_t = entry.first;
                    }
                }
                sink_old += max_t;
            }
        }
        auto t1 = bench_clock_t::now();
        (void)sink_old;
        time_old_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double time_new_us = 0.0;
    {
        std::vector<std::vector<std::pair<llama_token, int32_t>>> new_cache(n_ngrams);
        for (int i = 0; i < n_ngrams; ++i) {
            new_cache[i] = raw_data[i];
        }

        volatile int64_t sink_new = 0;
        auto t0 = bench_clock_t::now();
        for (int it = 0; it < iterations; ++it) {
            for (int i = 0; i < n_ngrams; ++i) {
                const auto & part = new_cache[i];
                int max_c = 0;
                llama_token max_t = -1;
                for (const auto & entry : part) {
                    if (entry.second > max_c) {
                        max_c = entry.second;
                        max_t = entry.first;
                    }
                }
                sink_new += max_t;
            }
        }
        auto t1 = bench_clock_t::now();
        (void)sink_new;
        time_new_us = std::chrono::duration_cast<duration_us_t>(t1 - t0).count();
    }

    double avg_old_us = time_old_us / iterations;
    double avg_new_us = time_new_us / iterations;
    double speedup = avg_old_us / avg_new_us;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Baseline (unordered_map + copy)   : " << avg_old_us << " us / pass\n";
    std::cout << "  Optimized (contiguous vector + ref): " << avg_new_us << " us / pass\n";
    std::cout << "  --> SPEEDUP: " << speedup << "x (" 
              << (avg_old_us - avg_new_us) << " us saved per pass)\n\n";
}

// ============================================================================
// Main Execution
// ============================================================================

int main() {
    std::cout << "\n================================================================================\n";
    std::cout << "MICROARCHITECTURAL BENCHMARK SUITE: VECTOR VS HASH MAP IN HIGH-PERFORMANCE INFERENCE\n";
    std::cout << "================================================================================\n\n";

    // 1. Repetition penalties at different context history depths for Qwen 248k vocab
    bench_repetition_penalties(248320, 64, 200);
    bench_repetition_penalties(248320, 256, 200);
    bench_repetition_penalties(248320, 512, 100);

    // 2. Speculative proposal distributions
    bench_q_probs(10, 5000);
    bench_q_probs(40, 5000);

    // 3. Suffix Trie tree construction and pattern tracking
    bench_suffix_trie(500, 50);

    // 4. DRY sampling logit penalty application
    bench_dry_sampler(248320, 64, 200);
    bench_dry_sampler(248320, 256, 200);

    // 5. Character Trie prefix traversal
    bench_character_trie(1000, 50);

    // 6. N-Gram cache distribution queries
    bench_ngram_cache(5000, 100);

    std::cout << "================================================================================\n";
    std::cout << "ALL BENCHMARKS COMPLETED SUCCESSFULLY.\n";
    std::cout << "================================================================================\n\n";

    return 0;
}
