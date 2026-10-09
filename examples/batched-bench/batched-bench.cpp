#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// mutates the input string
static std::vector<int> parse_list(char * p) {
    std::vector<int> ret;

    char * q = p;

    while (*p) {
        if (*p == ',') {
            *p = '\0';
            ret.push_back(std::atoi(q));
            q = p + 1;
        }

        ++p;
    }

    ret.push_back(std::atoi(q));

    return ret;
}

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    int is_pp_shared = params.is_pp_shared;

    std::vector<int> n_pp = params.n_pp;
    std::vector<int> n_tg = params.n_tg;
    std::vector<int> n_pl = params.n_pl;
    std::vector<int> n_pps = params.n_pps;
    if (n_pps.empty()) {
        n_pps.push_back(params.is_pp_shared ? 1 : 0);
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    // ensure enough sequences are available
    ctx_params.n_seq_max = n_pl.empty() ? 1 : *std::max_element(n_pl.begin(), n_pl.end());

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const int32_t n_kv_max = llama_n_ctx(ctx);

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            llama_synchronize(ctx);
        }

        return true;
    };

    // warm up
    {
        for (int i = 0; i < 16; ++i) {
            common_batch_add(batch, 0, i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    LOG_TEE("\n");
    LOG_TEE("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, is_pp_shared = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.is_pp_shared, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
    LOG_TEE("\n");

    LOG_TEE("|%6s | %6s | %4s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n", "PP",     "TG",     "B",   "PPS",   "N_KV",     "T_PP s",   "S_PP t/s", "T_TG s",   "S_TG t/s", "T s",      "S t/s");
    LOG_TEE("|%6s-|-%6s-|-%4s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "----", "----", "------", "--------", "--------", "--------", "--------", "--------", "--------");

    for (        int i_pp = 0; i_pp < (int) n_pp.size(); ++i_pp) {
        for (    int i_tg = 0; i_tg < (int) n_tg.size(); ++i_tg) {
            for (int i_pl = 0; i_pl < (int) n_pl.size(); ++i_pl) {
                for (int i_pps = 0; i_pps < (int) n_pps.size(); ++i_pps) {
                    const int pp = n_pp[i_pp];
                    const int tg = n_tg[i_tg];
                    const int pl = n_pl[i_pl];
                    const int pps_cfg = n_pps[i_pps];

                    // Determine effective number of prompt clusters M (1 <= M <= pl)
                    // If pps_cfg <= 0, every sequence has an independent prompt (M = pl)
                    const int n_clusters = (pps_cfg <= 0 || pps_cfg >= pl) ? pl : pps_cfg;

                    // Skip redundant subdivision runs when pps_cfg exceeds current pl
                    if (pps_cfg > pl && i_pps > 0 && n_pps[i_pps - 1] >= pl) {
                        continue;
                    }

                    const int cluster_size = (pl + n_clusters - 1) / n_clusters;

                    // Primary sequence IDs for each cluster
                    std::vector<int> prim_seqs;
                    for (int c = 0; c < n_clusters; ++c) {
                        int s = c * cluster_size;
                        if (s < pl) {
                            prim_seqs.push_back(s);
                        }
                    }
                    const int actual_clusters = (int) prim_seqs.size();

                    const int n_ctx_req = actual_clusters * pp + pl * tg;

                    if (n_ctx_req > n_kv_max) {
                        printf("n_ctx_req = %d is greater than n_kv_max = %d for pp = %d, tg = %d, pl = %d, pps = %d\n",
                                n_ctx_req, n_kv_max, pp, tg, pl, actual_clusters);
                        continue;
                    }

                    const auto t_pp_start = ggml_time_us();

                    llama_kv_cache_clear(ctx);

                    for (int s : prim_seqs) {
                        common_batch_clear(batch);
                        for (int i = 0; i < pp; ++i) {
                            common_batch_add(batch, 0, i, { s }, false);
                        }
                        batch.logits[batch.n_tokens - 1] = true;

                        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                            LOG_TEE("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                    }

                    if (actual_clusters < pl) {
                        for (int j = 0; j < pl; ++j) {
                            int c = std::min(j / cluster_size, actual_clusters - 1);
                            int prim = prim_seqs[c];
                            if (j != prim) {
                                llama_kv_cache_seq_cp(ctx, prim, j, -1, -1);
                            }
                        }
                    }

                    const auto t_pp_end = ggml_time_us();

                    const auto t_tg_start = ggml_time_us();

                    for (int i = 0; i < tg; ++i) {
                        common_batch_clear(batch);

                        for (int j = 0; j < pl; ++j) {
                            common_batch_add(batch, 0, pp + i, { j }, true);
                        }

                        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                            LOG_TEE("%s: llama_decode() failed\n", __func__);
                            return 1;
                        }
                    }

                    const auto t_tg_end = ggml_time_us();

                    const int32_t n_kv = n_ctx_req;

                    const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f;
                    const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f;
                    const float t    = t_pp + t_tg;

                    const float speed_pp = (float)(pl * pp) / t_pp;
                    const float speed_tg = (float)(pl * tg) / t_tg;
                    const float speed    = (float)(pl * pp + pl * tg) / t;

                    LOG_TEE("|%6d | %6d | %4d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n",
                            pp, tg, pl, actual_clusters, n_kv, t_pp, speed_pp, t_tg, speed_tg, t, speed);
                }
            }
        }
    }

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
