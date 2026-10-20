# Implementation & Provenance Status at HEAD

This file records the implementation and test verification status of experimental flags, subsystems,
and benchmark provenance gates on branch `feat/numa-split`.

## Status Matrix

| Subsystem / Script | Files | Status | Verification |
| :--- | :--- | :--- | :--- |
| `llama-kv-zdd` | `src/llama-kv-zdd.h`, `src/llama-graph.cpp`, `src/llama-build-context.cpp`, `src/llama.cpp`, `ggml/src/ggml-cuda/fattn-*.cuh` | **Implemented & Tested** | `test-kv-zdd` passes (`ZddKvManager`, `DlxZddCompiler`, `ZddMoeClusterRouter`, `ZddAttentionBlockSkipper` graph mask hook and 6-family CUDA FlashAttention 2D tile skipping gated by `LLAMA_KV_ZDD_SKIP` / `cparams.zdd_sparse_kv`) |
| `llama-cohort-scheduler` | `common/cohort-radix-trie.h`, `examples/server/server-context.cpp` | **Implemented & Tested** | `test-cohort-prefix-sharing` passes (`tests/test-cohort-prefix-sharing.cpp::test_cohort_radix_trie`; radix prefix sharing, weighted LFU/LRU eviction, cohort isolation) |
| `llama-unreal-selector` / `-unreal` | `common/unreal-selector.h`, `examples/main/main.cpp`, `examples/server/server.cpp` | **Implemented & Tested** | `test-unreal-selector` passes (streaming late-interaction MaxSim chunk selection, BM25 lexical boost, prompt chunk partitioning and reassembly) |
| `host-paged-kv` | `common/paged-kv-cache.h`, `src/llama-graph.cpp`, `ggml/src/ggml-cuda.cu` | **Implemented & Tested** | `test-paged-kv-cache` and `test-tiered-kv-cache` pass; wired for Host-to-Device block streaming in attention graph evaluation |
| `scripts/launch_server_disciplined.sh` | `scripts/launch_server_disciplined.sh` | **Implemented & Tested** | Port-safe process cleanup (`SIGTERM` -> `SIGKILL` + `lsof` verification) and binary SHA256 provenance logging (`server_provenance.log`) |
| `scripts/mab_pareto_tuner.py` provenance | `scripts/mab_pareto_tuner.py` | **Implemented & Tested** | SQLite `trials` and `mab_evaluations` tables record `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, `cmd_line`, `mode`, and `VmHWM` memory footprint on every trial insert |
| `-tgb`, `--threads-gen-batch` | `common/common.cpp`, `examples/llama-bench/llama-bench.cpp`, `docs/parameters.md` | **Implemented & Tested** | Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for `llama-cli`, `llama-server`, and `llama-bench` |
| `--numa distribute` / `split` / `expert` / `mirror` | `ggml/src/ggml.c`, `ggml/src/ggml-cpu-numa.cpp`, `src/llama.cpp` | **Implemented & Tested** | `test-numa` passes (`ggml_numa_get_affinity`, `ggml_numa_get_strategy`, per-node tensor placement and worker affinity) |

## ZDD sparse KV (`llama-kv-zdd`)

- Documented in: `README.md` (improvements table), `docs/kv-cache.md` section 3.
- Status: Implemented and tested. The `--zdd-sparse-kv` / `--no-zdd-sparse-kv` CLI flags, `gpt_params`, and `llama_context_params.zdd_sparse_kv` plumbing are wired into `src/llama.cpp`, `src/llama-graph.cpp`, and `src/llama-build-context.cpp`. `ZddAttentionBlockSkipper` generates active-block masks (gated by `cparams.zdd_sparse_kv` or `LLAMA_KV_ZDD_SKIP`), and `tests/test-kv-zdd.cpp` verifies zero-suppression, hash-consing, DLX exact-cover compilation, MoE cluster routing, and attention block skipping.

## Cohort scheduler (`llama-cohort-scheduler`)

- Documented in: `README.md`, `docs/parameters.md`.
- Status: Implemented and tested. `common/cohort-radix-trie.h` and `examples/server/server-context.cpp` implement cohort-aware radix prefix matching and weighted LFU/LRU slot eviction (`cohort_reuse_count` / age scoring), verified by `tests/test-cohort-prefix-sharing.cpp::test_cohort_radix_trie`.

## Early Token Filtering (`llama-unreal-selector`)

- Documented in: `README.md`, `docs/parameters.md`.
- Status: Implemented and tested. `common/unreal-selector.h` provides streaming late-interaction MaxSim chunk selection with BM25 lexical boost and exact-cover partitioning, wired into prompt ingestion in `examples/main/main.cpp` and `examples/server/server.cpp`. Verified by `tests/test-unreal-selector.cpp`.

## Host Paged KV Cache (`host-paged-kv`)

- Documented in: `docs/kv-cache.md`.
- Status: Implemented and tested. `common/paged-kv-cache.h` and `common/tiered-kv-cache.h` provide 128-byte aligned HugeTLB pre-faulted block management. `src/llama-graph.cpp` and `ggml/src/ggml-cuda.cu` (`ggml_backend_cuda_stream_paged_kv_blocks`) stream active KV blocks asynchronously from host memory to device during attention graph evaluation, eliminating the host-KV decode speed penalty at 512k context. Verified by `tests/test-paged-kv-cache.cpp` and `tests/test-tiered-kv-cache.cpp`.

## `-tgb`, `--threads-gen-batch`

- Documented in: `docs/parameters.md`.
- Status: Implemented in `common/common.cpp` and `examples/llama-bench/llama-bench.cpp`. Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for both `llama-cli` and `llama-server`.

## Benchmark provenance (`scripts/mab_pareto_tuner.py` & `scripts/launch_server_disciplined.sh`)

- `scripts/mab_pareto_tuner.py` records `mode` (`live`, `simulated`, `failed`), `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, and `cmd_line` per trial row in SQLite (`trials` and `mab_evaluations` tables, with `ALTER TABLE ... ADD COLUMN` migration fallbacks). `execute_live_trial()` now samples child process `VmHWM` via a background monitor thread (`monitor_hwm` polling `/proc/<pid>/status` every 50ms) for live memory footprint measurement, and refuses to export a report that mixes modes (override: `--allow-mixed-modes`).
- `scripts/launch_server_disciplined.sh` enforces port-safe process cleanup before server launch and logs the server binary SHA256 digest and timestamp to `server_provenance.log`. Tracked in git.

