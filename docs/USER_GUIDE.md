# `ik_llama.cpp` HPC & Long-Context User Guide

This guide covers the Phase 2 high-performance computing (HPC), 512k+ long-context KV cache, zero-copy MoE mmap, ZDD sparse attention, `UNREAL` evidence selection (`--evidence-filter`), Binary Fuse8 KV context capsules (`ctx-capsule`), Thompson-sampling speculative decoding, multi-tenant cohort prefix sharing, and Profile-Targeted Optimization (`PTO` via Clang 21 + `ld.lld-21` ThinLTO + `-fsplit-machine-functions`) subsystems in `ik_llama.cpp`.

---

## 1. DeepSeek-V4.1 512k--1M Tiered KV Cache & Host-to-Device Paged Streaming

For ultra-long contexts (512k to 1M tokens on DeepSeek-V4.1 and hybrid MLA/DSA architectures), `common/tiered-kv-cache.h` and `common/paged-kv-cache.h` manage quantized KV blocks across GPU VRAM and 128-byte aligned HugeTLB host RAM:

- **Quantized 512k--1M KV Sizing (`Q8_0` & `Q4_0`)**:
  - `calculate_deepseek_v4_512k_kv_bytes(seq_len)` computes the exact `Q8_0` KV footprint (~17.37 GB at 512k tokens across 61 layers).
  - `calculate_deepseek_v4_512k_q4_kv_bytes(seq_len)` computes the `Q4_0` KV footprint (~9.23 GB at 512k tokens, ~11.25 GiB at 1M tokens on 20-layer Flash MLA), enabling full 512k--1M context residency within a single 24 GiB GPU.
- **Automatic Host-to-Device Tiered KV Streaming**:
  - Passing `--kv-on-host` (or when KV tensors spill to host memory at 256k+ context) initializes `ctx->tiered_kv_manager` during context creation and asynchronously streams active KV pages to device staging buffers during graph evaluation (`ggml_backend_cuda_stream_paged_kv_blocks`).

```bash
./build-pto/bin/llama-server \
  -m /path/to/DeepSeek-V4.1.gguf \
  -c 524288 \
  -ctk q4_0 -ctv q4_0 \
  --kv-on-host \
  -fa on
```

---

## 2. Rust Memory-Mapped Engine (`GGML_RUST_MMAP_ENGINE`) & 2 MiB MoE Superblocks

When built with `-DGGML_RUST_MMAP_ENGINE=ON`, `src/llama-mmap.cpp` links against `librust_mmap_engine`:

- **`MAP_HUGE_2MB` & Transparent Hugepage Fallback**: Attempts explicit `MAP_HUGE_2MB` allocation and falls back cleanly to standard pages with `madvise(..., MADV_HUGEPAGE)` to minimize TLB misses on 100 GB+ MoE weights.
- **`io_uring` `O_DIRECT` `/proc/self/fd` Reopen**: Automatically reopens read-only descriptors with `O_DIRECT` via `/proc/self/fd/<fd>` when 4096-byte alignment invariants hold, falling back to page-cached `io_uring` reads for unaligned offsets.
- **`Superblock2MiBView` Zero-Copy MoE Dispatch**: Slices 2 MiB-aligned MoE expert weight buffers directly over active `expert_ids` with hardware `PREFETCHT0` lookahead (`29.58--40.42 ns/op` via `BitwiseVeb4096 + EliasFanoIndex`).

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_RUST_MMAP_ENGINE=ON
./scripts/build-guard.sh cmake --build build --config Release -j $(nproc)
```

---

## 3. ZDD Sparse KV Cache & CPU/CUDA FlashAttention Tile Skipping (`--zdd-sparse-kv`)

Zero-Suppressed Decision Diagram (ZDD) sparse KV pruning (`src/llama-kv-zdd.h`) tracks active attention blocks and skips zero-weight 2D tiles during both CPU and CUDA FlashAttention:

- **CUDA FlashAttention**: Skips inactive 2D KV tiles across all 6 kernel families (`fattn-vec`, `fattn-tile`, `fattn-wmma-f16`, `fattn-mma-f16`, `fattn-mma-f16-cdna`).
- **CPU FlashAttention**: `ggml_compute_forward_flash_attn_ext_f16` inspects the ZDD block mask (`src/llama-graph.cpp`) using 64-bit word comparisons (`0xFC00FC00FC00FC00ULL`) and skips masked 64-token KV blocks in $O(1)$.

```bash
./build-pto/bin/llama-cli \
  -m /path/to/model.gguf \
  --zdd-sparse-kv \
  -fa on \
  -p "Summarize the following repository..."
```

---

## 4. Model-Internal Evidence Selector & Global Aggregation Summary (`--evidence-filter`)

`common/unreal-selector.h` implements Late-Interaction `MaxSim` chunk filtering with BM25 lexical boost, Exact-Cover partitioning, and pre-pruning corpus aggregation:

- **Canonical CLI Flags**: `--evidence-filter` (`-ef`), `--evidence-chunk-tokens` (`-ect`, default `140`), `--evidence-top-k` (`-etk`, default `10`), `--evidence-min-tokens` (default `4096`), `--evidence-strategy` (`adaptive_default`, `code_pinpoint`, `narrative_recall`, `analytic_aggregate`, `temporal_window`, `custom_weights`), `--evidence-symbol-boost`, `--evidence-chunk-expansion`, `--evidence-maxsim-threshold`, and `--evidence-symbol-regex`.
- **Pre-Pruning Global Aggregation Summary (`unreal::global_aggregation_summary`)**: `selector::compute_global_summary` scans all chunks prior to top-$k$ pruning to record exact corpus-wide query-token occurrence counts and chunk document frequencies, ensuring global counting queries remain 100% accurate even when >80% of chunks are pruned.
- **Binary Fuse8 Context Capsules (`common/kv-capsule-filter.h`)**: Spilled KV blocks are indexed by an 8-bit Graf & Lemire Binary Fuse Filter (`~9.0 bits/entry`, `0.39%` FPR, 3 XOR lookups) paired with a 64-bit Charikar `SimHash64` sketch (`< 15 ns` negative-lookup rejection).

```bash
./build-pto/bin/llama-cli \
  -m /path/to/model.gguf \
  --evidence-filter \
  --evidence-strategy code_pinpoint \
  --evidence-top-k 20 \
  --evidence-chunk-tokens 140
```

---

## 5. Thompson-Sampling Speculative Decoding Bandit & Counterfactual Attribution (`--spec-autotune`)

`common/spec-tuner.h`, `common/spec-tuner.cpp`, `common/spec-counterfactual.h`, and `common/spec-arbiter.h` provide:

- **Full-Information Counterfactual Prefix Attribution (*Not-a-Bandit* Theorem 3)**: Evaluates acceptance across all candidate draft lengths $k \in [0, n_{\text{proposed}}]$ from a single target forward pass with zero extra target queries ($O(\sqrt{T \ln K})$ regret).
- **Online Gaussian-Gamma Thompson-Sampling Bandit (`select_thompson_sampling`)**: Maintains Welford running variance (`M2`) and online token-entropy / acceptance-rate gating (`ema_accept_rate`, `update_entropy_gate`) across `DFlash`, `DSpark`, `MTP`, and `CCD` suffix-trie stages.

---

## 6. Multi-Tenant Cohort Prefix Sharing & Cross-Slot KV Cloning (`--cohort-prefix-sharing`)

`examples/server/cohort-radix-trie.h` and `examples/server/server-context.cpp` index active prompt prefixes in an $O(L)$ radix trie with strict multi-tenant cohort isolation:

- **Cross-Slot Donor Cloning (`find_best_donor_match` + `llama_kv_cache_seq_cp`)**: When a concurrent request arrives for a cohort whose primary slot is busy decoding (`slot.is_processing()`), `launch_slot_with_task` locates the active donor slot via `cohort_trie.find_best_donor_match(...)` and clones the shared `[0, matched_tokens)` KV prefix into the newly assigned idle slot via `llama_kv_cache_seq_cp`, skipping redundant prompt prefill while preserving tenant boundaries.

---

## 7. Interactive `ctx-capsule` Hexagonal TUI & 5-Layer Hierarchical TOML Control Plane

`tools/ctx-capsule/ctx_capsule_tui.ts` provides a pure TypeScript Functional Object-Oriented Programming (FOOP) & Hexagonal Architecture control plane and mouse/keyboard interactive TUI modal:

- **5-Layer Hierarchical TOML Configuration (`scripts/configs/hierarchy/`)**:
  1. `00-defaults.toml`: Base engine, KV cache, evidence filter, and PTO defaults.
  2. `hardware/*.toml`: NUMA topology, thread pinning, and GPU VRAM budgets (e.g., `epyc_milan_rtx4090.toml`).
  3. `models/*.toml`: Per-model overrides (`deepseek_v41_flash`, `qwen38_27b`, `qwen38_2_4t`, `glm_53_flash`, `kimi_k3`, `leanstral_119b`).
  4. `templates/*.toml`: Workload templates (`coding_pinpoint`, `deep_formal_audit`, `low_vram_1m`, `max_throughput_np8`).
  5. `instances/*.toml`: Live `llama-server` fleet instances and in-place binary replacement state.
- **Usage**:

```bash
# Render interactive TUI snapshot or launch interactive modal:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --snapshot

# Compile 5-layer hierarchy into scripts/configs/models.toml:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --export-toml

# Trigger PTO trace & binary swap from the control plane:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --pto-optimize
```

---

## 8. Profile-Targeted Optimization (`PTO`: Continuous IRPGO + ThinLTO + Machine Function Splitting)

`scripts/pto_finetune.sh` and `scripts/pto_profile_runner.py` implement a 3-stage compiler and linker optimization pipeline using Clang 21 (`clang-21` / `clang++-21`), `llvm-profdata-21`, `ld.lld-21` ThinLTO, and `-fsplit-machine-functions`:

1. **Stage 1 (`GENERATE` — `build-pto-gen/`)**: Compiles `libggml`, `libllama`, and `llama-*` binaries with `-DGGML_PTO_MODE=GENERATE` (`-fprofile-generate -fprofile-continuous -mllvm -vp-counters-per-site=8`).
2. **Stage 2 (`TRACE & MERGE` — `scripts/pto_profile_runner.py`)**: Executes a 6-stage representative workload sweep (or merges live continuous `%p%c.profraw` daemon counters via `--live-daemon-merge`) into `merged.profdata`.
3. **Stage 3 (`USE + THINLTO + SPLIT` — `build-pto/`)**: Recompiles and links both static ThinLTO monobinaries (`build-pto/bin/{llama-server,llama-cli,llama-bench,llama-perplexity}`) and shared libraries (`build-pto/ggml/src/libggml.so`, `build-pto/src/libllama.so`) with `-DGGML_PTO_MODE=USE -DGGML_THINLTO=ON -DGGML_SPLIT_MACHINE_FUNCTIONS=ON -DGGML_USE_LLD=ON`.

```bash
# Run the full 3-stage PTO pipeline (instrument -> trace -> ThinLTO + function-split build):
./scripts/pto_finetune.sh --stage=all --model=/path/to/model.gguf

# Merge live continuous profile counters from a running llama-server daemon without downtime:
python3 scripts/pto_profile_runner.py \
  --live-daemon-merge \
  --profile-dir /tmp/ik_llama_pgo \
  --output-profdata /tmp/ik_llama_pgo/merged.profdata
```
