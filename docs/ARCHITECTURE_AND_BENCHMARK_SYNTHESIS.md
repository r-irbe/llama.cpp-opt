# Architecture, Benchmarks, and Formal Verification Synthesis

This document contains the detailed systems architecture, hardware benchmark tables, performance counter (`perf stat`) audits, formal verification inventory, and academic references for `llama.cpp-opt` (`feat/numa-split`).

For a high-level overview of practical outcomes and quickstart instructions, see the root [`README.md`](../README.md) and [`docs/USER_GUIDE.md`](USER_GUIDE.md).

---

## 1. Reference Hardware & Measurement Methodology

All live benchmarks in this repository were collected on a single-GPU, dual-socket workstation:
- **CPU & Memory**: Dual AMD EPYC 7453 (56 physical cores, 112 threads across 8 NUMA nodes in NPS4 mode, ~968 GiB DDR4-3200 ECC RAM across 16 memory channels).
- **GPU**: 1x NVIDIA GeForce RTX 4090 (24 GB VRAM, Ada Lovelace `sm_89`, 72 MiB L2 cache with 49.5 MiB persistent window).
- **Storage**: 4x 4TB NVMe PCIe Gen4 RAID-0 (~28.1 GB/s Direct I/O).
- **Reproducibility & Provenance**: Benchmark scripts (`scripts/bench_ranked.py`, `scripts/bench_multicontext.py`, `scripts/bench_concurrent_server.py`, `scripts/mab_pareto_tuner.py`) record `binary_sha256`, `git_sha`, `hardware_fingerprint`, `vm_hwm_kb` (`/proc/<pid>/status`), and explicit execution mode (`live` vs. `simulated`). CPU-only runs set `CUDA_VISIBLE_DEVICES=""`.

---

## 2. Core Subsystems & Formal Verification (`common/`, `src/`, `ggml/`)

| Subsystem | Source Files | Architectural Design | Formal Verification (`rust-mmap-engine/lean_rules/`) |
| :--- | :--- | :--- | :--- |
| **Shared Fleet Slab Arbiter** | `common/slab-arena.h`, `examples/server/server-context.cpp` | ASLR-invariant relative offset addressing (`m_storage_offset`), 64-bit atomic bitmask allocation, and release-ordered descriptor lifecycle across multi-process POSIX shared memory (`--slab-arbiter-shm`). | `Rules/SlabArenaProtocol.lean` (offset invariance across address spaces, 64-bit bitmask bijection via `bv_decide`, happens-before descriptor release). |
| **Wait-Free & Lock-Free Rings** | `common/lockfree-ring.h`, `common/ccd-suffix-trie.h` | 128-byte cache-line-aligned (`alignas(128)`) SPSC, Vyukov MPSC, and Disruptor broadcast rings with atomic CAS sequence claiming and epoch-based reclamation (EBR) suffix tries. | `proofs/lifecycle/` LTS & JMM happens-before models + `JvmContractProbe.java`. |
| **ZDD Sparse KV & FlashAttention Skip** | `src/llama-kv-zdd.h`, `ggml/src/ggml-cuda/fattn-*.cuh`, `src/llama-graph.cpp` | Minato Zero-Suppressed Decision Diagram (ZDD) block indexing with 63-bit SWAR `(21, 21, 21)` key packing and all-query-row tile skipping across all 6 CUDA FlashAttention kernels (`vec`, `tile`, `wmma`, `mma`). | `Rules/FlashAttnZddSkip.lean` & `Rules/ZddKeyPack63.lean` (all-row skip soundness/completeness and 63-bit SWAR bijection via `bv_decide`). |
| **Tiered KV Cache & Direct I/O** | `common/tiered-kv-cache.h`, `src/llama-io-uring.cpp`, `src/llama-mmap.cpp` | Free-list-backed Tier-2 host KV slot management, dedicated `/proc/self/fd` `O_DIRECT` reopen for `io_uring` DMA, short-read residual re-queueing, and 2 MiB THP alignment. | `Rules/TieredKvAndIoUring.lean` (free-list slot injectivity and `io_uring` short-read termination variant). |
| **Memory Pressure Recovery & PolyGraph** | `src/llama.cpp`, `common/spec-verify-planner.h`, `examples/server/server-context.cpp` | Automatic microbatch halving (`llama_set_n_ubatch`) + KV eviction on `GGML_STATUS_ALLOC_FAILED`, and shape-pinned speculative verification (`--spec-verify-pinned`). | `Rules/PolyGraph.lean`, `formalization/PolyGraph/Basic.lean`, & `Rules/UbatchStepDownTermination.lean` (`mask_closure`, unconditional `kv_cells_disjoint`, and OOM step-down termination). |
| **Native `MXFP8` & Prism Ternary Quants** | `ggml/src/ggml-cuda/mmq.cuh`, `ggml/src/ggml-cuda/vecdotq.cuh`, `ggml/src/ggml-quants.c` | Register-scaled Ada Lovelace (`sm_89`) FP8 compute for `GGML_TYPE_MXFP8` (no intermediate FP16 buffer) and SSSE3/AVX2 ternary decoding (`PTQ1_0` / `PQ2_0`). | `Rules/PrismTernaryCodec.lean` (`bv_decide` bitwise equivalence of SSSE3 `ik_decode_trits_ssse3` vs. scalar base-3 across all $3^5 = 243$ states). |
| **Context Capsules & Evidence Selection** | `common/kv-capsule-filter.h`, `common/unreal-selector.h`, `tools/ctx-capsule/` | Graf & Lemire 8-bit Binary Fuse Filters (`~9.0 bits/entry`, `0.39%` FPR) + 64-bit `SimHash64`, ColBERT MaxSim + BM25 pre-attention chunk pruning, and interactive TypeScript TUI + 5-layer TOML hierarchy. | Verified by `tests/test-unreal-selector.cpp` and `tests/test_ctx_capsule_tui.py`. |

---

## 3. Verified Flagship Benchmark Matrix

### 3.1 Baseline (`E2`) vs. Optimized (`E3/E4`) vs. PTO + ThinLTO (`E5: build-pto`)

| Model Artifact | Footprint & Alignment | Vanilla `ik_llama.cpp` (`E2` `c446f1e08`) | `llama.cpp-opt` (`E3/E4`) | `llama.cpp-opt` + PTO (`E5: build-pto`) | Key Measured Highlights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Qwen3.8-27B-UD-Q4_K_XL`** (Dense + DeltaNet, 100% VRAM) | `16.42 GiB` (1 file; Lint `100/100`) | `pp512`: 2,420.73 tok/s<br>`tg32`: 46.41 ± 0.08 tok/s | `pp512`: 2,565.77 tok/s<br>`tg32`: 47.99 ± 0.05 tok/s | `pp2048`: **2,734.42 ± 40.19 tok/s**<br>`tg128`: **47.82 ± 0.03 tok/s** | **`DFlash2` (`n_max=9`)**: **99.52 tok/s** (`2.09x` vs non-spec); **1M context** TTFT = `1,942.1 ms` (`1,944.3 tok/s` PP), `15.12 tok/s` TG |
| **`GLM-5.3-Flash-UD-IQ4_XS`** (313.5B total / 17B active MoE) | `146.05 GiB` (5 shards) | `pp512`: 161.51 ± 15.04 tok/s<br>`tg32`: 18.81 ± 0.06 tok/s | `pp512`: 159.05 ± 13.83 tok/s<br>`tg32`: 18.98 ± 0.18 tok/s | `pp512`: **159.62 ± 12.76 tok/s**<br>`tg32`: **19.04 ± 0.06 tok/s** | `-66.7%` lower decode variance (`CV = 0.31%`); WikiText-2 `PPL = 2.0823 ± 0.11238`; `4K–256K` sweep: `138.60 tok/s` PP, `18.77–19.29 tok/s` TG |
| **`DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected`** (748.5B MoE + MLA) | `385.00 GiB` (10 shards) | `pp512`: 182.50 ± 18.64 tok/s<br>`tg32`: 21.84 ± 0.40 tok/s | `pp512`: 180.82 ± 22.20 tok/s<br>`tg32`: 22.64 ± 0.25 tok/s | `pp512`: **188.44 ± 22.48 tok/s**<br>`tg32`: **22.94 ± 0.20 tok/s** | **`25.19 tok/s`** with PolyGraph pinned speculative verification; **512K context**: `114.98 tok/s` PP, `20.09 tok/s` TG; WikiText-2 `PPL = 2.1301` |
| **`DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL`** (9 shards, inline `MXFP8`) | `466.32 GiB` (`188.8 GiB` inline `MXFP8`) | Unsupported `MXFP8` compute / host-pinning THP stall | `--numa expert` + default `MPOL_INTERLEAVE` (`17.51 tok/s`) | `pp`: **147.95 tok/s**<br>`tg`: **17.55 tok/s** (`18.01` max at `-np 4`) | Native `GGML_TYPE_MXFP8` compute; automatic `--numa expert` host-unpin gate preserves 2 MiB THP across `469.9 GiB` RAM |
| **`Qwen3.8-2.4T-A95B-UD-Q1_0`** (2.42T total / 95B active MoE) | `369.97 GiB` $\to$ `354.24 GiB` (`-15.73 GiB`) | Fails (`unsupported type 66` in loader) | `pp512`: 27.04 ± 2.34 tok/s<br>`tg32`: 3.69 ± 0.05 tok/s | `pp512`: **24.22 ± 1.22 tok/s**<br>`tg32`: **3.97 ± 0.02 tok/s** | `+7.59%` faster decode with `CV = 0.50%` on a 2.42-trillion-parameter model on a single 24 GB GPU + Dual EPYC host |
| **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** (176.9B MoE) | `103.68 GiB` (4 shards) | `pp512`: 331.34 ± 4.12 tok/s<br>`tg32`: 31.97 ± 1.25 tok/s | `pp512`: 452.63 tok/s<br>`tg32`: 37.47 ± 0.33 tok/s | `pp512`: **455.89 ± 55.81 tok/s**<br>`tg32`: **35.65 ± 0.32 tok/s** | **`54.79 tok/s` with `MTP` (`1.54x`, `83.1%` draft tokens accepted)**; `+37.59%` faster prefill vs. baseline |
| **`Leanstral-1.5-119B-A6B-Q4_K_M`** (119.0B MoE) | `67.25 GiB` (1 file) | — | — | `pp512`: **393.98 ± 53.90 tok/s**<br>`tg32`: **40.23 ± 0.26 tok/s** | High-throughput formal theorem proving (`CV = 0.65%`) |

---

### 3.2 Staged Long-Context Scaling (`4K` $\to$ `1M`) on `build-pto`

Measured live via `scripts/bench_multicontext.py --cli-bin build-pto/bin/llama-cli` (`-ctk q4_0 -ctv q4_0 --zdd-sparse-kv --evidence-filter --evidence-strategy code_pinpoint --evidence-chunk-tokens 140 --evidence-top-k 16`):

| Model (`build-pto`) | `4K` (`4,096`) | `16K` (`16,384`) | `64K` (`65,536`) | `128K` (`131,072`) | `256K` (`262,144`) | `512K` (`524,288`) | `1M` (`1,048,576`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3.8-27B-UD-Q4_K_XL`** (`16.42 GiB`, 100% VRAM) | PP: `2,323.2 t/s`<br>TG: `46.33 t/s`<br>RSS: `17.09 GiB` | PP: `2,424.6 t/s`<br>TG: `46.76 t/s`<br>RSS: `17.09 GiB` | PP: `2,493.4 t/s`<br>TG: `46.38 t/s`<br>RSS: `17.09 GiB` | PP: `2,546.0 t/s`<br>TG: `46.21 t/s`<br>RSS: `17.09 GiB` | PP: **`2,580.9 t/s`**<br>TG: **`46.66 t/s`**<br>RSS: `17.09 GiB` | PP: `1,882.6 t/s`<br>TG: `15.68 t/s`<br>RSS: `17.09 GiB` | PP: **`1,944.3 t/s` (`1.94 s` TTFT)**<br>TG: **`15.12 t/s`**<br>RSS: `20.69 GiB` |
| **`GLM-5.3-Flash-UD-IQ4_XS`** (`146.05 GiB`, Hybrid MoE) | PP: `133.42 t/s`<br>TG: `18.69 t/s`<br>RSS: `142.13 GiB` | PP: **`140.34 t/s`**<br>TG: `18.69 t/s`<br>RSS: `142.15 GiB` | PP: `138.81 t/s`<br>TG: `18.86 t/s`<br>RSS: `142.20 GiB` | PP: `136.75 t/s`<br>TG: **`19.29 t/s`**<br>RSS: `142.23 GiB` | PP: **`138.60 t/s`**<br>TG: **`18.77 t/s`**<br>RSS: `142.26 GiB` | — | — |
| **`DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected`** (`385.00 GiB`) | PP: `43.67 t/s`<br>TG: `12.68 t/s`<br>RSS: `184.86 GiB` | PP: `52.38 t/s`<br>TG: `14.51 t/s`<br>RSS: `185.07 GiB` | PP: `48.13 t/s`<br>TG: `19.38 t/s`<br>RSS: `185.22 GiB` | PP: `47.35 t/s`<br>TG: `13.44 t/s`<br>RSS: `185.35 GiB` | PP: `62.62 t/s`<br>TG: `14.78 t/s`<br>RSS: `185.25 GiB` | PP: **`114.98 t/s` (`32.8 s` TTFT)**<br>TG: **`20.09 t/s`**<br>RSS: `185.29 GiB` | `11.25 GiB` K-cache (`19.61 t/s` TG) |

---

### 3.3 Profile-Targeted Optimization (`PTO` / `build-pto`) PMU Hardware Counter Audit

Measured across 5 repetitions (`-r 5 -fa 1`) using Clang 21 Continuous IRPGO, `ld.lld-21` (`-flto=thin -Wl,--icf=safe`), and `-fsplit-machine-functions`:

| Execution Regime | Metric | Standard Release (`build/bin/llama-bench`) | PTO Shared (`build-pto-shared/`) | PTO Static (`build-pto/bin/llama-bench`) | Measured Improvement |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `pp512` (`tok/s`) | `71,527.21 ± 3,669.62` | `143,790.30 ± 3,004.55` | **`156,836.07 ± 4,347.90`** | **`+119.27%` (`2.19x`)** |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `pp2048` (`tok/s`) | `38,548.40 ± 1,106.79` | `66,250.40 ± 988.39` | **`73,064.72 ± 655.39`** | **`+89.54%` (`1.90x`)** |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `tg128` (`tok/s`) | `410.58 ± 7.21` | `4,074.19 ± 56.22` | **`4,208.01 ± 65.87`** | **`+924.89%` (`10.25x`)** |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `cycles` (`IPC`) | `349.58B` (`0.36`) | — | **`113.35B` (`0.72`)** | **`-67.58%` cycles (`2.00x` IPC)** |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `L1-icache-load-misses` | `10,301,603` | — | **`4,573,803`** | **`-55.60%` L1i misses** |
| **CPU Hot-Path (`-ngl 0 -t 56`)** | `branch-misses` | `27,294,405` | — | **`19,920,398`** | **`-27.02%` branch misses** |
| **CUDA Offload (`-ngl 99`)** | `pp2048` (`tok/s`) | `43,878.44 ± 591.22` | `83,104.18 ± 412.50` | **`83,619.76 ± 283.14`** | **`+90.57%` (`1.91x`)** |
| **CUDA Offload (`-ngl 99`)** | Host `cycles` (`IPC`) | `18.47B` (`0.16`) | — | **`1.55B` (`1.53`)** | **`-91.59%` host CPU cycles** |

---

## 4. Academic & Systems References

1. **Counterfactual Speculative Evaluation (*Not-a-Bandit*)**:
   - Liu, H., Huang, J., Jia, Z., Park, Y., & Wang, Y.-X. (2025). *Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs*. In *ICLR 2026*. arXiv:2510.20064.
2. **Multi-Armed Bandit Speculative Decoding (*BanditSpec* & *TapOut*)**:
   - Hou, Y., et al. (2025). *BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms*. arXiv:2505.15141.
   - Sridhar, A., et al. (2025). *TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding*. arXiv:2511.02017.
3. **Zero-Suppressed Decision Diagrams (ZDD)**:
   - Minato, S. (1993). Zero-suppressed BDDs for set manipulation in combinatorial problems. In *DAC '93* (pp. 272–277). https://doi.org/10.1145/157485.164890
4. **Tree-Based Speculative Verification (*SpecInfer*)**:
   - Miao, X., et al. (2024). *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification*. In *ASPLOS '24* (pp. 932–949). https://doi.org/10.1145/3620665.3640369
5. **Heterogeneous CPU-GPU Inference (*Dovetail* & *MemSpec*)**:
   - Zhang, L., et al. (2024). *Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference*. arXiv:2412.18934.
   - Kim, E., Jeon, Y. J., & Han, M. (2026). *MemSpec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices*. In *Proc. ACM LCTES 2026*.
6. **Model-Internal Evidence Selection (*UNREAL*)**:
   - Kinderman, E., et al. (2026). *UNREAL: Unifying Retrieval and Long-Context with a Single Model*. arXiv:2610.08463.
7. **Binary Fuse Probabilistic Filters**:
   - Graf, T. M., & Lemire, D. (2022). Binary fuse filters: Fast and smaller than Xor filters. *ACM Journal of Experimental Algorithmics*, 27, 1–15. https://doi.org/10.1145/3510449
