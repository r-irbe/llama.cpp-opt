# Documented but unimplemented at HEAD

This file lists flags, knobs and headline claims that the documentation (mostly `README.md`,
`docs/numa.md`, `docs/parameters.md`, `docs/kv-cache.md`) presents as working, but that the code at
HEAD (`9f1339fc8`, branch `feat/numa-split`) does not implement. Each entry records where the claim
lives, what the code actually does, and what landing it would take.

When a claim here is fixed, delete its entry and its "see `docs/UNIMPLEMENTED.md`" markers.

## ZDD sparse KV ("up to 75% lower memory", "99.5% block read reduction")

- Claimed in: `README.md` (improvements table), `docs/kv-cache.md` section 3.
- Current Status: The `--zdd-sparse-kv` / `--no-zdd-sparse-kv` CLI flags, `gpt_params`, and `llama_context_params.zdd_sparse_kv` plumbing are implemented. `src/llama-kv-zdd.h` and `tests/test-kv-zdd.cpp` unit tests pass.
- Deferred kernel work: Attention kernel block-skipping in `src/llama-graph.cpp` and FlashAttention CUDA/CPU kernels when ZDD block sparsity is active.

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
