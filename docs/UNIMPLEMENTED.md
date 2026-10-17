# Documented but unimplemented at HEAD

This file lists flags, knobs and headline claims that the documentation (mostly `README.md`,
`docs/numa.md`, `docs/parameters.md`, `docs/kv-cache.md`) presents as working, but that the code at
HEAD (`9f1339fc8`, branch `feat/numa-split`) does not implement. Each entry records where the claim
lives, what the code actually does, and what landing it would take.

When a claim here is fixed, delete its entry and its "see `docs/UNIMPLEMENTED.md`" markers.

## `--cohort-prefix-sharing`, `--cohort-min-prefix`, `--cohort-max-idle-sec`, `--no-cohort-prefix-sharing`

- Claimed in: `README.md` (improvements table, formerly also the benchmark table), `docs/parameters.md`
  (server section, formerly "enabled by default").
- Actual: no such CLI flag exists (`common/common.cpp` has no handler; `llama-server --cohort-prefix-sharing`
  fails with `error: unknown argument`). `examples/server/cohort-radix-trie.h` is included only by
  `tests/test-cohort-prefix-sharing.cpp`; `examples/server/server-context.cpp` contains no cohort
  reference. `scripts/bench_concurrent_server.py` sends `cohort_id` / `X-Cohort-ID` /
  `X-Prompt-Cache-Key`, which the server ignores, and `scripts/mab_pareto_tuner.py` no longer samples
  `cohort_prefix_sharing` / `cohort_min_prefix` / `cohort_max_idle_sec` as tunables nor prints a
  "Cohort PPS" column, because there is no flag to pass them to.
- The numbers formerly in `results/cohort_bench_summary.md` and quoted in `README.md`
  (32.2 -> 103.2 tok/s) were withdrawn on 2026-10-18: the only raw run in the workspace
  (`results/cohort_bench_results.json`) has 6 records and 0 successful requests.
- To land it: (1) add `--cohort-*` options to `common/common.cpp` + `common/common.h`; (2) index
  slot prefixes in `server-context.cpp` with `cohort_radix_trie` and route an incoming prompt to a
  matching prefilled slot; (3) clone the matched slot state with `llama_kv_cache_seq_cp` and set
  `cache_n` so the server reports `prompt_n` as the reused prefix; (4) enforce cohort privacy
  (the trie already carries `task_cohort`); (5) re-measure with a live server and record
  `mode='live'` rows.

## ZDD sparse KV ("up to 75% lower memory", "99.5% block read reduction")

- Claimed in: `README.md` (improvements table), `docs/kv-cache.md` section 3.
- Actual: `src/llama-kv-zdd.h` and `tests/test-kv-zdd.cpp` exist, and `src/llama-cparams.h:60`
  declares `zdd_sparse_kv`, but nothing ever reads that field and no CLI flag sets it. The attention
  path (`src/llama-graph.cpp`, FA kernels) has no block-selection branch.
- To land it: set the cparam from a new `--zdd-sparse-kv` flag, plumb the block index through
  `llama_kv_cache`, implement the block-skip in the attention kernels (CPU and CUDA), define the
  threshold/sink/window policy, and measure tokens/s plus KV reads against the dense path.

## `-unreal`, `--unreal-filter`, `--unreal-*`

- Claimed in: `README.md` (improvements table and benchmark table), `docs/parameters.md` (generation section).
- Actual: `common/unreal-selector.h` exists (uncommitted in the worktree at the time of writing) and
  `common/common.cpp` parses the `--unreal-*` options into `gpt_params::unreal_filter`, but no code in
  `src/` or `examples/server/` reads those parameters — enabling them changes nothing.
  `tests/test-unreal-selector.cpp` tests the selector in isolation only.
- To land it: score prompt tokens with the selector inside prompt processing, drop filtered tokens
  before attention, keep the mapping from filtered positions back to the reported `prompt_n`, and
  measure quality (judge agreement) as well as speed.

## `--numa expert` and `--numa mirror`

- Claimed in: `README.md` (improvements table), `docs/numa.md` section 2, `docs/parameters.md` (NUMA section).
- Actual: the 2026-10-17 worktree parser accepts both values and maps them to
  `GGML_NUMA_STRATEGY_EXPERT` / `GGML_NUMA_STRATEGY_MIRROR`, and the parser now prints a warning that
  neither is implemented. `ggml::cpu::numa::is_numa_expert()` has no reader outside
  `tests/test-numa.cpp`; `ggml_cpu_numa_bind_tensor_experts()` has no caller; `set_numa_thread_affinity()`
  in `ggml/src/ggml.c` has no `MIRROR` case, so threads simply stay unpinned. `--numa expert` keeps the
  topology guard (`src/llama.cpp`, `llama_numa_init_ex`) and nothing else.
- To land `expert`: call `ggml_cpu_numa_bind_tensor_experts()` for routed-expert tensors during model
  load, and bind each worker to the node holding the expert it is computing.
- To land `mirror`: implement the `GGML_NUMA_STRATEGY_MIRROR` case in `set_numa_thread_affinity()`
  and allocate per-node replicas of thread-shared state.

## `--numa-replicate-shared`

- Claimed in: `README.md` (improvements table and comparison table), `docs/numa.md` section 2.3, `docs/parameters.md`.
- Actual: the 2026-10-17 worktree parser sets the environment variable `GGML_NUMA_REPLICATE_SHARED`,
  which `ggml::cpu::numa::is_numa_replicate_shared()` reads — but nothing calls that function and no
  replica buffers are ever allocated, so shared tensors are not replicated. The parser now warns
  about this.
- To land it: after load, allocate one replica per NUMA node for the tensors that this function
  selects, point each node's worker at its local replica, and account for the ~10-15% extra memory.

## `-tgb`, `--threads-gen-batch`

- Documented in: `docs/parameters.md` (llama-bench section).
- Actual: implemented in `examples/llama-bench/llama-bench.cpp` only. `llama-server` and `llama-cli`
  reject it (`error: unknown argument`), so it is a *benchmark* knob, not a serving knob.
- No code change needed; the doc line now says so. If it should apply to serving, `-tgb` has to be
  plumbed into `gpt_params` and the graph/batch thread split.

## Benchmark provenance

- `results/` is excluded from git via `.git/info/exclude:15`, so benchmark data does not travel with
  a clone. The shipped `results/pareto_frontier_*.md` reports were produced by the calibrated
  **simulator** (`mode='simulated'` in `results/mab_pareto_tuning.sqlite`) and are now banner-marked
  as such. `scripts/mab_pareto_tuner.py` records `mode` per row and refuses to export a report that
  mixes modes (override: `--allow-mixed-modes`).
- Still missing for reproducible claims: model sha256, `CUDA_VISIBLE_DEVICES` state, load average and
  the exact command in each row's metadata; CPU-only runs must hide the GPU (see
  `docs/development/token_generation_performance_tips.md`).
- `execute_live_trial()` in `scripts/mab_pareto_tuner.py` still records a fixed `mem_gb = 18.0` and,
  when llama-cli's timing lines do not match, derives TTFT from wall time. A `mode='live'` row is
  therefore measured throughput with an *estimated* footprint; sampling `VmHWM` of the child process
  would make that column a measurement too.
