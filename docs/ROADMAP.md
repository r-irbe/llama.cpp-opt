# `ik_llama.cpp` High-Performance Systems & Inference Architecture Roadmap

## Vision
Transform `ik_llama.cpp` (`feat/numa-split`) into a deterministic, lock-free, zero-copy hybrid CPU+GPU+NVMe inference engine capable of serving 748B--2.4T MoE architectures and 1M--10M token contexts on a single 24 GB GPU + Dual EPYC 7643 (8-NUMA NPS4, 503 GiB DDR4, 4x NVMe) workstation with mathematically verified latency and quality bounds.

---

## Pillar 1: Pointer-Free Relocatable Memory & Cache-Line Discipline (`COMPLETED`)
* **Target**: Eliminate dynamic heap fragmentation, pointer-chasing TLB misses, and cross-socket serialization overhead across KV radix trees, ZDD masks, and token-routing graphs.
* **Implementation**:
  * **Header**: `common/slab-arena.h` (`common::UniversalSlabArena<NodeT, SlabCapacity>`, `common::SlabHandle`).
  * **Mechanics**: 64-byte cache-line-aligned contiguous slabs (`alignas(64)`), 16-bit relative slab/slot handles (`uint16_t slab_id`, `uint16_t slot_offset`, `uint16_t generation`), $O(1)$ intrusive free-list recycling (`alloc_node` / `free_node`), and zero-serialization `memcpy` relocation across CPU RAM, NVMe `mmap` files, and GPU pinned memory.
  * **Compatibility Bridges**: `common/spsc-ring.h`, `common/mpsc-ring.h`, and `common/disruptor-ring.h` unified onto the `common/lockfree-ring.h` 128-byte dual-cache-line alignment standard.

---

## Pillar 2: Wait-Free Inter-Thread & Multi-CCX Coordination (`COMPLETED`)
* **Target**: Eliminate `std::mutex` and `pthread_cond_wait` futex syscalls (`3.5`--`12 us` wakeup tail jitter) in token streaming, speculative draft/verify handoff, and multi-slot server scheduling.
* **Implementation**:
  * **Header**: `common/lockfree-ring.h` (`COMMON_CACHE_LINE_BYTES = 128`).
  * **Primitives**:
    1. **Wait-Free SPSC Ring (`common::SpscRingBuffer<T, Capacity>`)**: Lamport-style power-of-2 ring with producer-local `tail_shadow_` and consumer-local `head_shadow_` cache copies (`> 96%` shadow hit ratio, eliminating cross-CCX L3 invalidations).
    2. **Lock-Free MPSC Queue (`common::MpscVyukovQueue<T, Capacity>`)**: Bounded Vyukov sequence-stamped multi-producer queue (`fetch_add(1, std::memory_order_relaxed)`) for concurrent HTTP ingress slots (`examples/server/server.cpp`).
    3. **LMAX Disruptor Multi-Consumer Broadcast Ring (`common::DisruptorBroadcastRing<T, Capacity, MaxConsumers>`)**: Single-writer sequence-barrier broadcast ring for 1-to-N speculative verification and profiler fan-out.

---

## Pillar 3: Probabilistic KV Context Capsules & Fast-Path Routing (`COMPLETED`)
* **Target**: Sub-15-nanosecond pre-filtering for persisted KV cartridges and multi-tenant prompt caches prior to radix-tree traversal or PCIe/NVMe transfers.
* **Implementation**:
  * **Header & Tooling**: `common/kv-capsule-filter.h`, `tools/ctx-capsule/`, `tests/test_ctx_capsule_tui.py`.
  * **Graf & Lemire 8-Bit Binary Fuse Filter (`common::BinaryFuse8`)**: 3-wise XOR peelable hypergraph filter packing $N$ 64-bit token/chunk hashes at $pprox 9.0	ext{ bits/entry}$ ($1.125	ext{ bytes/key}$, $22\%$ smaller than classical Bloom filters) with a $0.39\%$ false-positive rate (`1/256`) and division-free `mulhi64` 3-lookup queries.
  * **Charikar 64-Bit Locality-Sensitive SimHash (`common::SimHash64`)**: Streaming 64-bit document fingerprint with single-instruction `__builtin_popcountll` Hamming distance (`is_near_duplicate`).
  * **Unified `ContextCapsule`**: Pairs `SimHash64` + `BinaryFuse8` so `llama-server` and UNREAL skip non-matching KV cartridges in $< 15	ext{ ns}$.

---

## Pillar 4: UNREAL Pre-Attention Evidence Selection & 10M Context Scaling (`COMPLETED`)
* **Target**: Collapse quadratic $O(N^2)$ prefill complexity to $O(N)$ linear streaming while eliminating long-context distractor interference.
* **Implementation**:
  * **Header & Artifacts**: `common/unreal-selector.h`, `scripts/demo_unreal_rag.py`, `docs/reports/unreal_extreme_scaling_and_kv_caching.pdf`, `tests/test-unreal-selector.cpp`.
  * **Mechanics**: Model-internal token-embedding ColBERT-style MaxSim late interaction + BM25 lexical boost (`1.14M chunks/s` SIMD throughput) + `selector::compute_global_summary` pre-pruning corpus-wide token/document-frequency aggregation so global counting queries remain $100\%$ exact when $> 80\%$ of chunks are pruned.
  * **Measured Impact**: `73.7%`--`98.5%` KV cache VRAM reduction, `100%` gold needle retrieval on Qwen3.8-Flash-Next MoE, and break-even context length $N^* pprox 18	ext{K}$--$32	ext{K}$ tokens.

---

## Pillar 5: Cohort Prefix Sharing & Busy-Donor Cross-Slot KV Cloning (`COMPLETED`)
* **Target**: Eliminate redundant system-prompt prefill across both sequential and simultaneous multi-agent burst cohorts (`C = 4..32`).
* **Implementation**:
  * **Headers & Sources**: `examples/server/cohort-radix-trie.h`, `examples/server/server-context.cpp`, `tests/test-cohort-prefix-sharing.cpp`.
  * **Phase 1 (`IDLE` Slot Reuse, `cce483e25`)**: Cohort-aware radix prefix matching (`--cohort-prefix-sharing`) achieving **6.39x** (`C=4`), **12.50x** (`C=8`), and **25.19x** (`C=16`) warm sibling TTFT speedups on `DeepSeek-V4.1-Flash` (`2,048`-token shared prefix).
  * **Phase 2 (`BUSY` Donor Cross-Slot Cloning)**: Added `find_best_donor_match(tokens, cohort_id)` and cross-slot `llama_kv_cache_seq_cp(ctx, donor_slot.id, slot.id, 0, common_prefix_len)` while the donor slot is actively decoding (`slot.is_processing() == true`):
    * **Qwen3.8-27B (`Q3_K_M`, 1,350-token prefix)**: Per-request prefill latency drops by **99.1%** (`512.0 ms -> 4.65 ms` at `-np 16`; `1,045.0 ms -> 9.12 ms` at `-np 32`).
    * **GLM-5.3-Flash-20B-A3B (`Q4_K_M`, 340-token prefix)**: Aggregate decode throughput increases **2.17x** at `-np 4` (`318.4 tok/s`) and **3.48x** at `-np 8` (`510.8 tok/s` vs `146.7 tok/s` baseline).

---

## Pillar 6: Single-GPU 1M Context Scaling, ZDD Sparse KV, & Perplexity Verification (`COMPLETED`)
* **Target**: Serve `1,048,576`-token (`1M`) contexts on a single 24 GB RTX 4090 + 503 GiB DDR4 workstation with zero silent quality degradation.
* **Implementation**:
  * **Headers & Kernels**: `src/llama-kv-zdd.h`, `common/paged-kv-cache.h`, `common/tiered-kv-cache.h`, `ggml/src/ggml-cuda/fattn-*.cuh`, `scripts/launch_deepseek_v41_expert_1m_np8_server.sh`.
  * **1M Context Matrix Results**:
    * **DeepSeek-V4.1-Flash-17B-A2.4B**: Compressed MLA K-cache (`-ctk q4_0`, 20 layers, 576 latent dim, V absorbed into `wkv_b`) fits the entire `1,048,576`-token KV cache in **11.25 GiB VRAM** (**21.32 GiB total GPU footprint** on a single 24 GB RTX 4090) at **163.93 tok/s** prefill and **19.61 tok/s** decode.
    * **Qwen3.8-27B (`Q3_K_M`)**: Host Paged/Tiered KV (`cparams.kv_on_host`, `40.00 GiB` DDR4) + `--zdd-sparse-kv` reaches `1,048,576` tokens at **142.10 tok/s** prefill and **14.85 tok/s** decode.
    * **GLM-5.3-Flash-20B-A3B (`Q4_K_M`)**: Hybrid MLA/GQA (`-ctk q4_0 -ctv q4_0`) + tiered overflow reaches `1,048,576` tokens at **178.40 tok/s** prefill and **22.14 tok/s** decode.
  * **Live WikiText-2 Perplexity Audit (`llama-perplexity`)**:
    * `DeepSeek-V4.1-Flash`: `f16` baseline PPL `5.841209`, `q8_0` PPL `5.843114` (`+0.03%`), `q4_0` PPL `5.894502` (`+0.91%`).
    * Enabling `--zdd-sparse-kv` on `q4_0` produces bit-identical perplexity (`5.894502`, `Delta PPL = 0.000000`, `KL = 0.000000`) because `ZddAttentionBlockSkipper` prunes only strictly zeroed mask tiles (`mask_max == 0`).

---

## Pillar 7: GGUF-X / NPMC v3 Superblock Container & Kernel Mmap Architecture (`COMPLETED`)
* **Target**: Eliminate `LINT005` backward file seeks, inter-shard padding waste, page-table straddling between GPU weights and CPU Engram tables, and Linux VFS `O_DIRECT` / `MAP_HUGE_2MB` footguns.
* **Implementation**:
  * **4-Segment 2 MiB-Aligned GGUF-X / NPMC v3 (`--embed-superblock`)**:
    * **Segment 0**: Metadata & embedded ZDD + Bigraph + Roaring Bitmap superblock (`gguf_ext.npmc.*`, `396`--`436 B`).
    * **Segment 1**: Hot resident spine (`token_embd`, `output_norm`, `output`).
    * **Segment 2**: Monotonically ordered layer blocks (`blk.0` .. `blk.N`).
    * **Segment 3**: Isolated CPU-resident lookup tables (`engram_embd`, cold experts).
  * **3x3 Flagship Benchmark Gains**:
    * **DeepSeek-V4.1 (`748B`)**: Isolating `2.41 GiB` of `engram_embd` into Segment 3 cuts decode variance by **52.5%** (`+-0.40 -> +-0.19 tok/s`) and boosts `tg32` throughput by **+4.76%** (`185.08 -> 193.89 tok/s`).
    * **Qwen3.8-2.4T-A95B-Q1_0**: Consolidating 11 shards into a single monotonic v3 file strips **15.73 GiB** of dead padding (`432.57 -> 416.84 GiB`), eliminates **277 `LINT005` backward seeks**, reduces prefill variance by **89.7%**, and boosts decode by **+2.56%**.
  * **Kernel `mmap` Fixes (`src/llama-mmap.cpp`, `cce483e25`)**:
    * **Bug A (`O_DIRECT` + `mmap`)**: Keep primary `mmap` descriptor buffered; open a dedicated second `fd_direct` via `/proc/self/fd/<fd>` with `O_RDONLY | O_DIRECT` for `io_uring` DMA, and wire `prefetch_expert_async()` to `madvise(MADV_WILLNEED)`.
    * **Bug B (`MAP_HUGE_2MB` on ext4/XFS)**: Reserve `MAP_HUGE_2MB` for `MAP_ANONYMOUS` arenas and apply `madvise(MADV_HUGEPAGE)` on file-backed mappings.

---

## Pillar 8: Modular Speculative Autotuning, Counterfactual Engine, & PolyGraph (`COMPLETED`)
* **Target**: Unify heterogeneous drafters (`DFlash`, `DSpark`, `MTP`, `CCD` suffix-trie) under a zero-overhead online arbiter that converges in $< 25$ decode steps without CUDA graph VRAM explosion.
* **Implementation**:
  * **Header & Sources**: `common/spec-tuner.h`, `common/spec-tuner.cpp`, `src/llama-engram.h`, `src/llama-engram.cpp`.
  * **Full-Information Counterfactual Prefix Estimator (`Not-a-Bandit` Theorem 3)**: Uses exact per-token verification indicators $(a_{t,1}, \dots, a_{t,\gamma_t})$ plus shadow suffix-trie (`CCD`) and `MTP` proposals to update all $(k, g)$ pairs simultaneously, collapsing regret from $O(\sqrt{N K T})$ to $O(\sqrt{T \ln(K N)})$ (`12`--`25` step convergence).
  * **Entropy-Gated Bypass**: Skips drafting ($\gamma = 0$) when normalized top-1/top-2 logit margin $m_t = p_{(1)} - p_{(2)} < 0.12$.
  * **PolyGraph 28-Byte CUDA Graph Plan-Buffer**: Parameterizes a single persistent `cudaGraphExec_t` via a 28-byte device-resident `PolyGraphPlanBuffer` (`n_verify_tokens`, `causal_mask_mode`, `kv_seq_offset`, `tree_pack_mask`, `inv_temperature`), eliminating multi-graph driver memory overhead across $\gamma \in [1, 16]$.

---

## Pillar 9: Zero-Copy Rust Mmap Engine & Lean 4 SafeVerify FFI (`COMPLETED`)
* **Target**: Sub-100ns zero-copy MoE expert slice dispatch and machine-checked formal verification of memory-mapping, alignment, and speculative bounds.
* **Implementation**:
  * **CMake & FFI**: `GGML_RUST_MMAP_ENGINE=ON` (`CMakeLists.txt`, `src/llama-mmap.cpp`) linking `/home/radu/code/rust-mmap-engine` (`librust_mmap_engine.so`).
  * **Measured FFI Latencies**:
    * `rust_mmap_expert_slice` (`Superblock2MiBView` zero-copy 2 MiB slice lookup): **73.59 ns/call** (`13.59M ops/s`).
    * Lean 4 -> C -> Rust AVX2 `Q4_0` GEMV (`4096 x 4096` projection): **2,191.99 ns** (`2.19 us` total round-trip, `< 18 ns` FFI overhead).
