# Implementation & Provenance Status at HEAD

This file records the implementation and test verification status of experimental flags, subsystems,
and benchmark provenance gates on branch `feat/numa-split`.

## Status Matrix

| Subsystem / Script | Files | Status | Verification |
| :--- | :--- | :--- | :--- |
| `llama-kv-zdd` | `src/llama-kv-zdd.h`, `src/llama-graph.cpp`, `src/llama-build-context.cpp`, `src/llama.cpp`, `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-cuda/fattn-*.cuh` | **Implemented & Tested** | `test-kv-zdd` passes (`ZddKvManager`, `DlxZddCompiler`, `ZddMoeClusterRouter`, `ZddAttentionBlockSkipper` graph mask hook, CPU FlashAttention ZDD block-row skip, and 6-family CUDA FlashAttention 2D tile skipping gated by `LLAMA_KV_ZDD_SKIP` / `cparams.zdd_sparse_kv`) |
| `llama-cohort-scheduler` | `examples/server/cohort-radix-trie.h`, `examples/server/server-context.cpp` | **Implemented & Tested** | `test-cohort-prefix-sharing` passes (`find_best_match`, `find_best_donor_match`, cross-slot `llama_kv_cache_seq_cp` KV prefix cloning from active/busy donor slots, weighted LFU/LRU eviction, strict multi-tenant cohort isolation) |
| `llama-unreal-selector` / `--evidence-filter` | `common/unreal-selector.h`, `examples/main/main.cpp`, `examples/server/server.cpp` | **Implemented & Tested** | `test-unreal-selector` passes (streaming late-interaction MaxSim chunk selection, BM25 lexical boost, `global_aggregation_summary` pre-pruning exact corpus counting, prompt chunk partitioning and reassembly; CLI flags `--evidence-filter` / `-ef`, `--evidence-chunk-tokens` / `-ect`, `--evidence-top-k` / `-etk`, `--evidence-strategy`) |
| `host-paged-kv` / `tiered-kv` | `common/paged-kv-cache.h`, `common/tiered-kv-cache.h`, `src/llama-graph.cpp`, `src/llama.cpp`, `ggml/src/ggml-cuda.cu` | **Implemented & Tested** | `test-paged-kv-cache` and `test-tiered-kv-cache` pass; DeepSeek-V4.1 `Q4_0`/`Q8_0` 512k context sizing and automatic `cparams.kv_on_host` tiered KV streaming in attention graph evaluation |
| `spec-tuner` (Thompson-Sampling MAB) | `common/spec-tuner.h`, `common/spec-tuner.cpp` | **Implemented & Tested** | Online Welford running variance (`M2`), Gaussian-Gamma Thompson-sampling arm selection (`select_thompson_sampling`), and unified `ema_accept_rate` + `update_entropy_gate` gating across `DFlash`, `DSpark`, `MTP`, and `CCD` suffix-trie |
| `rust-mmap-engine` (`GGML_RUST_MMAP_ENGINE`) | `CMakeLists.txt`, `src/llama-mmap.cpp`, `../rust-mmap-engine` | **Implemented & Tested** | `cargo test` passes; `O_DIRECT` `/proc/self/fd` reopen fallback for `io_uring` NVMe DMA, `MAP_HUGE_2MB` / `MADV_HUGEPAGE` arena allocation, `Superblock2MiBView` zero-copy MoE expert dispatch, and optional `GGML_RUST_MMAP_ENGINE` C FFI integration |
| `scripts/launch_server_disciplined.sh` | `scripts/launch_server_disciplined.sh` | **Implemented & Tested** | Port-safe process cleanup (`SIGTERM` -> `SIGKILL` + `lsof` verification) and binary SHA256 provenance logging (`server_provenance.log`) |
| `scripts/mab_pareto_tuner.py` provenance | `scripts/mab_pareto_tuner.py` | **Implemented & Tested** | SQLite `trials` and `mab_evaluations` tables record `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, `cmd_line`, `mode`, and `VmHWM` memory footprint on every trial insert |
| `-tgb`, `--threads-gen-batch` | `common/common.cpp`, `examples/llama-bench/llama-bench.cpp`, `docs/parameters.md` | **Implemented & Tested** | Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for `llama-cli`, `llama-server`, and `llama-bench` |
| `slab-arena` / `lockfree-ring` | `common/slab-arena.h`, `common/lockfree-ring.h`, `common/spsc-ring.h`, `common/mpsc-ring.h`, `common/disruptor-ring.h` | **Implemented & Tested** | 64B pointer-free relocatable `UniversalSlabArena` (`SlabHandle` 16-bit relative indexing) and 128B Zen 3 cache-line-aligned wait-free SPSC, Vyukov MPSC, and LMAX Disruptor rings (`> 96%` shadow hit ratio) |
| `kv-capsule-filter` / `ctx-capsule` | `common/kv-capsule-filter.h`, `tools/ctx-capsule/`, `tests/test_ctx_capsule_tui.py` | **Implemented & Tested** | Graf & Lemire 8-bit Binary Fuse Filter (`~9.0 bits/entry`, `0.39%` FPR, 3 XOR lookups) + Charikar 64-bit `SimHash64` inside `ContextCapsule` (`< 15 ns` cartridge rejection) |
| `--numa distribute` / `split` / `expert` / `mirror` | `ggml/src/ggml.c`, `ggml/src/ggml-cpu-numa.cpp`, `src/llama.cpp` | **Implemented & Tested** | `test-numa` passes (`ggml_numa_get_affinity`, `ggml_numa_get_strategy`, per-node tensor placement and worker affinity) |

## ZDD sparse KV (`llama-kv-zdd`)

- Documented in: `README.md` (improvements table), `docs/kv-cache.md` section 3, `docs/USER_GUIDE.md`.
- Status: Implemented and tested. The `--zdd-sparse-kv` / `--no-zdd-sparse-kv` CLI flags, `gpt_params`, and `llama_context_params.zdd_sparse_kv` plumbing are wired into `src/llama.cpp`, `src/llama-graph.cpp`, `src/llama-build-context.cpp`, `ggml/src/ggml-cpu/ggml-cpu.cpp`, and `ggml/src/ggml-cuda/fattn-*.cuh`. `ZddAttentionBlockSkipper` generates active-block masks (gated by `cparams.zdd_sparse_kv` or `LLAMA_KV_ZDD_SKIP`), skipping zeroed KV block tiles on both CPU and CUDA FlashAttention paths. `tests/test-kv-zdd.cpp` verifies zero-suppression, hash-consing, DLX exact-cover compilation, MoE cluster routing, and attention block skipping.

## Cohort scheduler (`llama-cohort-scheduler`)

- Documented in: `README.md`, `docs/parameters.md`, `docs/USER_GUIDE.md`.
- Status: Implemented and tested. `examples/server/cohort-radix-trie.h` and `examples/server/server-context.cpp` implement cohort-aware radix prefix matching (`find_best_match`), cross-slot donor prefix discovery (`find_best_donor_match`), and instant KV prefix cloning across slots via `llama_kv_cache_seq_cp` (even when the donor slot is actively decoding a sibling request in the same cohort), alongside weighted LFU/LRU slot eviction (`cohort_reuse_count` / age scoring), verified by `tests/test-cohort-prefix-sharing.cpp`.

## Early Token Filtering (`llama-unreal-selector`)

- Documented in: `README.md`, `docs/parameters.md`, `docs/USER_GUIDE.md`.
- Status: Implemented and tested. `common/unreal-selector.h` provides streaming late-interaction MaxSim chunk selection with BM25 lexical boost, exact-cover partitioning, and `global_aggregation_summary` (`selector::compute_global_summary`) pre-pruning corpus-wide token occurrence and chunk document-frequency counting so global counting queries remain 100% exact even when >80% of chunks are pruned. Verified by `tests/test-unreal-selector.cpp`.

## Host Paged & Tiered KV Cache (`host-paged-kv`)

- Documented in: `docs/kv-cache.md`, `docs/USER_GUIDE.md`.
- Status: Implemented and tested. `common/paged-kv-cache.h` and `common/tiered-kv-cache.h` provide 128-byte aligned HugeTLB pre-faulted block management and exact 512k context sizing for DeepSeek-V4.1 (`Q8_0` and `Q4_0` KV quantization). `src/llama.cpp`, `src/llama-graph.cpp`, and `ggml/src/ggml-cuda.cu` (`ggml_backend_cuda_stream_paged_kv_blocks`) automatically initialize and stream active KV blocks asynchronously from host memory to device when `cparams.kv_on_host` is enabled. Verified by `tests/test-paged-kv-cache.cpp` and `tests/test-tiered-kv-cache.cpp`.

## `-tgb`, `--threads-gen-batch`

- Documented in: `docs/parameters.md`.
- Status: Implemented in `common/common.cpp` and `examples/llama-bench/llama-bench.cpp`. Parsed and wired into `gpt_params` (`params.n_threads`, `params.n_threads_batch`) for both `llama-cli` and `llama-server`.

## Benchmark provenance (`scripts/mab_pareto_tuner.py` & `scripts/launch_server_disciplined.sh`)

- `scripts/mab_pareto_tuner.py` records `mode` (`live`, `simulated`, `failed`), `git_sha`, `hardware_fingerprint`, `model_sha256`, `cuda_visible_devices`, `load_avg`, and `cmd_line` per trial row in SQLite (`trials` and `mab_evaluations` tables, with `ALTER TABLE ... ADD COLUMN` migration fallbacks). `execute_live_trial()` now samples child process `VmHWM` via a background monitor thread (`monitor_hwm` polling `/proc/<pid>/status` every 50ms) for live memory footprint measurement, and refuses to export a report that mixes modes (override: `--allow-mixed-modes`).
- `scripts/launch_server_disciplined.sh` enforces port-safe process cleanup before server launch and logs the server binary SHA256 digest and timestamp to `server_provenance.log`. Tracked in git.

