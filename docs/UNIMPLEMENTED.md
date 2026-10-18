# Implementation & Provenance Status at HEAD

This file records the implementation and test verification status of experimental flags, subsystems,
and benchmark provenance gates on branch `feat/numa-split`.

## Status Matrix

| Subsystem / Script | Files | Status | Verification |
| :--- | :--- | :--- | :--- |
| `llama-kv-zdd` | `src/llama-kv-zdd.h`, `src/llama-graph.cpp`, `src/llama-build-context.cpp`, `src/llama.cpp` | **Implemented & Tested** | `test-kv-zdd` passes (`ZddKvManager`, `DlxZddCompiler`, `ZddMoeClusterRouter`, `ZddAttentionBlockSkipper` graph mask hook gated by `LLAMA_KV_ZDD_SKIP` / `cparams.zdd_sparse_kv`) |
| `llama-cohort-scheduler` | `common/cohort-radix-trie.h`, `examples/server/server-context.cpp` | **Implemented & Tested** | `test-cohort-radix-trie` passes (radix prefix sharing, weighted LFU/LRU eviction, cohort isolation) |
| `scripts/launch_server_disciplined.sh` | `scripts/launch_server_disciplined.sh` | **Implemented & Tested** | Port-safe process cleanup (`SIGTERM` -> `SIGKILL` + `lsof` verification) and binary SHA256 provenance logging (`server_provenance.log`) |
| `scripts/mab_pareto_tuner.py` provenance | `scripts/mab_pareto_tuner.py` | **Implemented & Tested** | SQLite `trials` and `mab_evaluations` tables record `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, `cmd_line`, `mode`, and `VmHWM` memory footprint on every trial insert |
| `-tgb`, `--threads-gen-batch` | `common/common.cpp`, `examples/llama-bench/llama-bench.cpp`, `docs/parameters.md` | **Implemented & Tested** | Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for `llama-cli`, `llama-server`, and `llama-bench` |
| `--numa distribute` / `split` / `expert` / `mirror` | `ggml/src/ggml.c`, `ggml/src/ggml-cpu-numa.cpp`, `src/llama.cpp` | **Implemented & Tested** | `test-numa` passes (`ggml_numa_get_affinity`, `ggml_numa_get_strategy`, per-node tensor placement and worker affinity) |

## ZDD sparse KV (`llama-kv-zdd`)

- Documented in: `README.md` (improvements table), `docs/kv-cache.md` section 3.
- Status: Implemented and tested. The `--zdd-sparse-kv` / `--no-zdd-sparse-kv` CLI flags, `gpt_params`, and `llama_context_params.zdd_sparse_kv` plumbing are wired into `src/llama.cpp`, `src/llama-graph.cpp`, and `src/llama-build-context.cpp`. `ZddAttentionBlockSkipper` generates active-block masks (gated by `cparams.zdd_sparse_kv` or `LLAMA_KV_ZDD_SKIP`), and `tests/test-kv-zdd.cpp` verifies zero-suppression, hash-consing, DLX exact-cover compilation, MoE cluster routing, and attention block skipping.

## Cohort scheduler (`llama-cohort-scheduler`)

- Documented in: `README.md`, `docs/parameters.md`.
- Status: Implemented and tested. `common/cohort-radix-trie.h` and `examples/server/server-context.cpp` implement cohort-aware radix prefix matching and weighted LFU/LRU slot eviction (`cohort_reuse_count` / age scoring), verified by `tests/test-cohort-radix-trie.cpp`.

## `-tgb`, `--threads-gen-batch`

- Documented in: `docs/parameters.md`.
- Status: Implemented in `common/common.cpp` and `examples/llama-bench/llama-bench.cpp`. Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for both `llama-cli` and `llama-server`.

## Benchmark provenance (`scripts/mab_pareto_tuner.py` & `scripts/launch_server_disciplined.sh`)

- `scripts/mab_pareto_tuner.py` records `mode` (`live`, `simulated`, `failed`), `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, and `cmd_line` per trial row in SQLite (`trials` and `mab_evaluations` tables, with `ALTER TABLE ... ADD COLUMN` migration fallbacks), samples child process `VmHWM` for live memory footprint measurement, and refuses to export a report that mixes modes (override: `--allow-mixed-modes`).
- `scripts/launch_server_disciplined.sh` enforces port-safe process cleanup before server launch and logs the server binary SHA256 digest and timestamp to `server_provenance.log`.
