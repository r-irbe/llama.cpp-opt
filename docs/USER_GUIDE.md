# `ik_llama.cpp` HPC & Long-Context User Guide

This guide covers the Phase 2 high-performance computing (HPC), 512k+ long-context KV cache, zero-copy MoE mmap, ZDD sparse attention, `UNREAL` evidence selection, Thompson-sampling speculative decoding, and multi-tenant cohort prefix sharing subsystems in `ik_llama.cpp`.

---

## 1. DeepSeek-V4.1 512k Tiered KV Cache & Host-to-Device Paged Streaming

For ultra-long contexts (up to 512k tokens on DeepSeek-V4.1 and hybrid MLA/DSA architectures), `common/tiered-kv-cache.h` and `common/paged-kv-cache.h` manage quantized KV blocks across GPU VRAM and 128-byte aligned HugeTLB host RAM:

- **Quantized 512k KV Sizing (`Q8_0` & `Q4_0`)**:
  - `calculate_deepseek_v4_512k_kv_bytes(seq_len)` computes the exact `Q8_0` KV footprint (~17.37 GB at 512k tokens across 61 layers).
  - `calculate_deepseek_v4_512k_q4_kv_bytes(seq_len)` computes the `Q4_0` KV footprint (~9.23 GB at 512k tokens), enabling full 512k-context residency within tight VRAM budgets.
- **Automatic Host-to-Device Tiered KV Streaming**:
  - Passing `--kv-on-host` sets `cparams.kv_on_host = true`, which automatically initializes `ctx->tiered_kv_manager` during context creation and asynchronously streams active KV pages to device staging buffers during graph evaluation (`ggml_backend_cuda_stream_paged_kv_blocks`).

```bash
./build/bin/llama-server \
  -m /path/to/DeepSeek-V4.1.gguf \
  -c 524288 \
  -ctk q4_0 -ctv q4_0 \
  --kv-on-host \
  -fa
```

---

## 2. Rust Memory-Mapped Engine (`GGML_RUST_MMAP_ENGINE`) & 2 MiB MoE Superblocks

When built with `-DGGML_RUST_MMAP_ENGINE=ON`, `src/llama-mmap.cpp` links against `../rust-mmap-engine`:

- **`MAP_HUGE_2MB` & Transparent Hugepage Fallback**: Attempts explicit `MAP_HUGE_2MB` allocation and gracefully falls back to standard pages with `madvise(..., MADV_HUGEPAGE)` to minimize TLB misses on 100 GB+ MoE weights.
- **`io_uring` `O_DIRECT` `/proc/self/fd` Reopen**: Automatically reopens read-only descriptors with `O_DIRECT` via `/proc/self/fd/<fd>` when 4096-byte alignment invariants hold, falling back to page-cached `io_uring` reads for unaligned offsets.
- **`Superblock2MiBView` Zero-Copy MoE Dispatch**: Slices 2 MiB-aligned MoE expert weight buffers directly over active `expert_ids` with hardware `PREFETCHT0` lookahead.

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_RUST_MMAP_ENGINE=ON
cmake --build build --config Release -j $(nproc)
```

---

## 3. ZDD Sparse KV Cache & CPU/CUDA FlashAttention Tile Skipping (`--zdd-sparse-kv`)

Zero-suppressed Decision Diagram (ZDD) sparse KV pruning (`src/llama-kv-zdd.h`) tracks active attention blocks and skips zero-weight 2D tiles during both CPU and CUDA FlashAttention:

- **CUDA FlashAttention**: Skips inactive 2D KV tiles across all 6 kernel families (`fattn-vec`, `fattn-tile`, `fattn-wmma-f16`, `fattn-mma-f16`, `fattn-mma-f16-cdna`).
- **CPU FlashAttention**: `ggml_compute_forward_flash_attn_ext_f16` inspects the ZDD block mask (`src/llama-graph.cpp`) and skips masked KV block rows before computing dot products.

```bash
./build/bin/llama-cli \
  -m model.gguf \
  --zdd-sparse-kv \
  -fa \
  -p "Summarize the following repository..."
```

---

## 4. `UNREAL` Model-Internal Evidence Selector & Global Aggregation Summary (`--unreal`)

`common/unreal-selector.h` implements Late-Interaction `MaxSim` chunk filtering with BM25 lexical boost and pre-pruning corpus aggregation:

- **Pre-Pruning Global Aggregation Summary (`unreal::global_aggregation_summary`)**: `selector::compute_global_summary` scans all chunks prior to top-$k$ pruning to record exact corpus-wide query-token occurrence counts and chunk document frequencies, ensuring global counting queries remain 100% accurate even when >80% of chunks are pruned.

```bash
./build/bin/llama-cli \
  -m model.gguf \
  --unreal \
  --unreal-top-n 20 \
  --unreal-chunk-size 140
```

---

## 5. Thompson-Sampling Speculative Decoding Bandit (`--spec-autotune`)

`common/spec-tuner.h` and `common/spec-tuner.cpp` provide an online Gaussian-Gamma Thompson-sampling bandit (`select_thompson_sampling`) with Welford running variance (`M2`) and online token-entropy / acceptance-rate gating (`ema_accept_rate`, `update_entropy_gate`):

- Dynamically tunes `n_max`, `n_min`, `p_min`, and `suffix_min_match_len` across `DFlash`, `DSpark`, `MTP`, and `CCD` suffix-trie speculative stages with $O(\sqrt{T})$ cumulative regret.

---

## 6. Multi-Tenant Cohort Prefix Sharing & Cross-Slot KV Cloning (`--cohort-prefix-sharing`)

`examples/server/cohort-radix-trie.h` and `examples/server/server-context.cpp` index active prompt prefixes in an $O(L)$ radix trie with strict multi-tenant cohort isolation:

- **Cross-Slot Donor Cloning (`find_best_donor_match` + `llama_kv_cache_seq_cp`)**: When a concurrent request arrives for a cohort whose primary slot is busy decoding (`!slot->available()`), `launch_slot_with_task` locates the active donor slot via `cohort_trie.find_best_donor_match(...)` and clones the shared `[0, matched_tokens)` KV prefix into the newly assigned idle slot via `llama_kv_cache_seq_cp`, skipping redundant prompt prefill while preserving tenant boundaries.
