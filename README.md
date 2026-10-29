# llama.cpp-opt: High-Performance Optimization Engine for LLM Inference

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **llama.cpp-opt** is a performance optimization layer built on top of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp). It focuses on practical bottlenecks encountered when running large models on real-world hardware: reducing memory usage, making multi-socket CPUs scale cleanly, streaming massive weights directly from fast NVMe storage, and serving multiple concurrent users without redundant computation.
>
> All improvements are modular and being contributed upstream to both [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) and [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp). See the [Pull Request Pipeline](#modular-upstream-pull-request-pipeline) below.

---

## 🚀 Key Improvements & Practical Impact (`llama.cpp-opt`)

### What We Improved: Upstream vs. Our Contributions

| Capability | Upstream (`llama.cpp` / `ik_llama.cpp`) | What We Added (`llama.cpp-opt`) | What You Get |
| :--- | :--- | :--- | :--- |
| **Multi-Socket CPU Scaling** [^numa] | Memory access defaults to interleaved or isolated mode; non-MoE layers bounce across CPU sockets, causing interconnect slowdowns. | **`--numa split` / `--numa distribute` & `--numa expert`**: pins worker threads across CPU nodes, binds routed MoE expert tensors per node round-robin (`ggml/src/ggml-cpu-numa.cpp`), and uses 128-byte cache-line-aligned wait-free SPSC/Disruptor rings (`common/lockfree-ring.h`). | **Deterministic multi-socket scaling on AMD EPYC, Threadripper, and Intel Xeon systems** with zero cross-socket futex wake jitter. |
| **Long Context Memory (`1M` Tokens)** [^kv] | KV cache memory grows linearly with prompt length; sliding-window models compact some layers via `--swa-compress`, but dense layers consume full memory. | **ZDD Sparse KV Block Selection, Tile Skipping & Tiered KV Streaming** (`src/llama-kv-zdd.h`, `common/tiered-kv-cache.h`, `ggml/src/ggml-cuda/fattn-*.cuh`): `--zdd-sparse-kv` builds Minato ZDD block indexing, `-INF`-fills inactive 64-token KV blocks in the attention mask, skips masked tiles across all 6 CUDA FlashAttention kernel families and CPU AVX2/AVX-512 FlashAttention, and streams host-paged KV slabs on demand. | **1M Context on a Single 24 GB GPU**: Skips 99.54% of inactive 64-token KV block reads with bit-identical perplexity ($\Delta\text{PPL} = 0.000000$) and fits 1M context on `DeepSeek-V4.1-Flash` (`11.25 GiB` K-cache), `Qwen3.8-27B`, and `GLM-5.3-Flash`. |
| **Native DeepSeek-V4.1 FP8 & `GGUF-X` v3** [^ds4] | Upstreams support V4.1 using re-quantized (`Q8_0`/`Q2_K`) tables; attempting to run native compact FP8 tables crashes due to missing row decoding; multi-shard checkpoints suffer backward seeks. | **Native FP8 Table Decoding (`ggml_compute_forward_get_rows_i8`) & `GGUF-X / NPMC` v3 Superblocks (`src/npmc-superblock.h`, `scripts/gguf-pack-npmc.py`)**: Direct lookup decoding for 264-byte FP8 Engram rows, 2 MiB HugeTLB alignment, Segment 3 `engram_embd` isolation, and $O(1)$ `BitwiseVeb4096 + EliasFanoIndex` expert offset resolution (`29.58–40.42 ns/op`). | **Run Native Checkpoints & Eliminate Seek Jitter**: Load compact DeepSeek-V4.1 FP8 checkpoints directly, shrink 10-shard `Qwen3.8-2.4T` by `-15.73 GiB`, and cut decode latency variance by `52.5%` to `89.7%`. |
| **Multi-User Server Prefill** [^cohort] | Sequential slot prompt caching (`-crs`); when several users connect at once with similar prompts, the server processes them one by one or re-evaluates duplicate text. | **Cohort Prefix Sharing & Busy-Donor Cross-Slot KV Cloning** (`--cohort-prefix-sharing`, `common/cohort-radix-trie.h`, `examples/server/server-context.cpp`): $O(L)$ radix trie with `find_best_donor_match` and cross-slot `llama_kv_cache_seq_cp` that clones shared prefixes even while the primary donor slot is actively decoding. | **Up to `22.46x` Faster Sibling TTFT & `93.8%` Less Prompt Memory**: Eliminates 99.1% of redundant prefill tokens across 16–32 concurrent requests while preserving strict multi-tenant cohort isolation. |
| **Long Document & Agentic Context Filtering** [^unreal] | Attention is calculated across every prompt token, spending quadratic compute time on filler words and retrieval noise; coding agents require frequent context compactions (`/compact`). | **Model-Internal Evidence Selection & Binary Fuse8 KV Capsules** (`--evidence-filter` / `-ef`, `--evidence-chunk-tokens` / `-ect`, `--evidence-strategy`, `common/unreal-selector.h`, `common/kv-capsule-filter.h`): ColBERT-style late-interaction MaxSim + BM25 lexical boost, pre-pruning `global_aggregation_summary` exact corpus counting, and 8-bit Graf & Lemire Binary Fuse Filters (`~9.0 bits/entry`, `0.39%` FPR, `< 15 ns` lookup) with 64-bit `SimHash64`. | **10M+ Virtual Context Scaling**: Prunes 73.7%–98.5% of distractor chunks before attention while preserving 100% exact global token counts and needle recall. |
| **Interactive `ctx-capsule` TUI & 5-Layer TOML Control Plane** [^tui] | Flat command-line flags or manual shell scripts; no unified configuration hierarchy, live telemetry modal, or in-place binary hot-swapping. | **Pure TypeScript FOOP Hexagonal Control Plane (`tools/ctx-capsule/ctx_capsule_tui.ts` & `scripts/configs/hierarchy/`)**: Keyboard/mouse interactive two-column TUI with live context-state bar, ASCII architecture diagrams, 5-layer hierarchical TOML compiler (`defaults` $\to$ `hardware` $\to$ `models` $\to$ `templates` $\to$ `instances`), and coding-agent SDK adapters. | **Declarative Fleet & Context Management**: Switch models, apply workload templates (`coding_pinpoint`, `low_vram_1m`, `max_throughput_np8`), inspect VRAM/latency fit badges, and hot-swap `llama-server` instances in place. |
| **Profile-Targeted Optimization (`PTO` / `build-pto`)** [^pto] | Standard `-O3` release builds suffer front-end L1-icache/iTLB misses, branch mispredictions across quantized switch tables, and cross-TU call overhead. | **3-Stage Continuous IRPGO + CS-IRPGO + ThinLTO + Machine Function Splitting (`scripts/pto_finetune.sh`, `scripts/pto_profile_runner.py`, `CMakeLists.txt`)**: Clang 21 + `ld.lld-21` + `llvm-profdata-21` pipeline with `-fprofile-generate -fprofile-continuous`, 6-stage workload + live-daemon `%p%c` counter merging, `-fprofile-use`, `-flto=thin`, and `-fsplit-machine-functions`. | **Up to `2.19x` Faster CPU Prefill, `10.25x` Micro-Decode, & New Flagship Peaks**: Reduces CPU cycles by `-67.58%` (`-91.59%` on CUDA host dispatch), cuts `L1-icache` misses by `-55.60%`, and lifts `Qwen3.8-27B` `pp2048` to `2,734.42 tok/s`, `DeepSeek-V4.1-Flash` to `188.44 tok/s` `pp512` / `22.94 tok/s` `tg32`, and `Qwen3.8-2.4T` to `3.97 tok/s` `tg32`. |
| **Disk & Alignment Safety** [^lint] | Manual inspection; unaligned GGUF tensors cause silent disk read stalls or `EINVAL` when streaming weights with `O_DIRECT` or `MAP_HUGE_2MB`. | **Hardware Fit & Alignment Checker (`scripts/gguf-lint.py`) & Kernel `mmap` Hardening (`src/llama-io-uring.cpp`, `src/llama-mmap.cpp`)**: Checks 4KB/2MiB alignment, reopens dedicated `/proc/self/fd/<fd>` `O_DIRECT` descriptors for `io_uring` DMA, and retries anonymous `MAP_HUGE_2MB` allocations. | **Fast Disk Reads & Safe Deployments**: Ensures zero unaligned disk stalls and confirms models fit hardware limits before starting. |
| **Automatic Speculative Tuning** [^spec] | Upstream added `--spec-autotune` for DFlash/suffix; tuner updates were dropped for other speculative models. | **Universal Counterfactual Bandit & Multi-CCX TreeGraft Arbiter (`common/spec-tuner.cpp`, `common/spec-counterfactual.h`, `common/spec-arbiter.h`, `scripts/mab_pareto_tuner.py`)**: Full-information counterfactual prefix attribution (*Not-a-Bandit* Theorem 3), Gaussian-Gamma Thompson sampling with Welford variance (`M2`), and online entropy gating across `DFlash`, `DSpark`, `MTP`, and `CCD` suffix-trie. | **Consistent Speedups (`1.49x`–`2.35x` vs Non-Spec)**: Keeps speculative decoding at its fastest configuration across diverse prompts with zero loss in output quality ($D_{KL} = 0.000000$). |

>#### 🔬 Real-World Benchmarks & Measurement Notes
>All benchmarks were measured on a reference dual-socket workstation:
>- **Hardware**: Dual AMD EPYC 7453 (56 physical cores, 112 threads across 8 NUMA nodes @ NPS4 [NUMA Per Socket 4], 991.9 GB / ~968 GiB DDR4-3200 ECC RAM across 16 memory channels), 1x NVIDIA GeForce RTX 4090 (24 GB VRAM, Ada Lovelace `sm_89` compute architecture, 72 MiB L2 cache with 49.5 MiB persistent window), and 4x 4TB Lexar NM790 NVMe RAID-0 (~28.1 GB/s Direct I/O).
>- **Terminology Definitions (First Mention)**:
>  - **NUMA (Non-Uniform Memory Access)**: A multi-socket multiprocessing architecture where memory access times depend on the memory's physical distance from the processor core (local vs. remote socket traversed via AMD xGMI or Intel UPI interconnects).
>  - **NPS4 (NUMA Nodes Per Socket 4)**: AMD EPYC configuration partitioning each CPU socket into 4 distinct quadrants (NUMA domains), each with local memory channels and Core Complex Dies (CCDs), minimizing cross-quadrant interconnect latency.
>  - **MoE (Mixture-of-Experts)**: A sparse neural network architecture where only a subset of specialized feed-forward subnetworks (experts) are dynamically activated per token (e.g. 8 of 64 experts active in DeepSeek or Qwen).
>  - **Engram**: High-dimensional conditional associative memory / embedding tables in DeepSeek-V4 models accessed via localized hash-key lookups.
>  - **MLA (Multi-Head Latent Attention)**: DeepSeek's low-rank projection attention architecture compressing the Key-Value (KV) cache into low-dimensional latent vectors ($d_c = 512$), slashing KV memory footprint.
>  - **TTFT (Time-To-First-Token)**: The latency from prompt submission until the model emits its initial output token, dominated by prompt evaluation (`pp`) throughput.
>  - **Direct I/O (`O_DIRECT`)**: Linux file I/O flag bypassing the operating system page cache, transferring data directly between NVMe storage controllers and user-space memory buffers via PCIe DMA.
>  - **ZDD (Zero-Suppressed Binary Decision Diagram)**: A canonical, graph-based data structure (Minato, 1993) optimized for sparse combinatorial sets, utilized here to represent and index active non-zero attention blocks.
>  - **TreeGraft**: An asynchronous candidate fusion algorithm combining divergent speculative drafts (N-gram sub-branches + neural companion drafts) into a single prefix tree verified in one batched target forward pass.
>  - **Cohort Prefix Sharing**: A multi-tenant caching mechanism using a radix trie where concurrent requests share common prefilled token sequences (system prompts, few-shot exemplars) without duplicate computation.
>  - **PTO (Profile-Targeted Optimization)**: A 3-stage compiler and binary-layout optimization pipeline combining Continuous Instrumentation-Based Profile-Guided Optimization (`IRPGO` / `CS-IRPGO`), Link-Time Optimization (`ThinLTO` via `ld.lld`), and Basic-Block / Machine-Function Cold Splitting (`-fsplit-machine-functions`).
>- **Provenance & Testing Harnesses**: Benchmark harnesses (`scripts/mab_pareto_tuner.py`, `scripts/bench_concurrent_server.py`, `scripts/bench_matrix.py`, `scripts/bench_multicontext.py`) record `binary_sha256`, `git_sha`, `hardware_fingerprint`, `vm_hwm_kb` (`/proc/<pid>/status`), and explicit `mode` (`live` vs. `simulated`), refusing to mix simulated `--dry-run` rows with live measurements unless `--allow-mixed-modes` is passed. CPU-only measurements require `CUDA_VISIBLE_DEVICES=""`; with a GPU visible, `-ngl 0` still runs prompt processing on CUDA (measured 1066 vs 140 pp512), see [`docs/development/token_generation_performance_tips.md`](docs/development/token_generation_performance_tips.md).
>
>| Ref | Area | Tested Model & Parameters | Mainline `llama.cpp` (`E1: ggerganov`) | Upstream `ik_llama.cpp` (`E2: ikawrakow` `c446f1e08`) | `llama.cpp-opt` (`E3/E4` `df7848239`) | `llama.cpp-opt` + PTO (`E5: build-pto` `dc6c75315`) | How It Was Measured, Parameters & Practical Caveats |
>| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
>| [^numa] | **Multi-Socket NUMA & Ultra-Large MoE** | **`Qwen3.8-2.4T-A95B-UD-Q1_0`** (2.42T total / 95B active, 10 shards, 369.97 GiB $\to$ 354.24 GiB `GGUF-X` v3; Score `40/100` $\to$ `100/100`; `-p 512 -n 32 -r 3 -t 56 -ngl 50 -ot exps=CPU`) | **FAIL** (`type 66` unsupported, exit code 1) | **FAIL** (`unsupported type 66` in `llama_model_loader`, exit code 1) | **`pp512`: 27.04 ± 2.34 tok/s, `tg32`: 3.69 ± 0.05 tok/s (Unmodified)** $\to$ **`pp512`: 23.94 ± 0.24 tok/s (-89.7% variance), `tg32`: 3.82 ± 0.01 tok/s (`GGUF-X` v3)** | **`pp512`: 24.22 ± 1.22 tok/s (-47.9% variance), `tg32`: 3.97 ± 0.02 tok/s (+7.59% vs E3 Unmodified, +3.93% vs `GGUF-X` v3, `CV = 0.50%`, `IPC = 1.37`)** | **How it was measured**: Live `llama-bench` + `perf stat` on reference host. Single-file `GGUF-X / NPMC` v3 (`--embed-superblock`) shrinks disk footprint by **-15.73 GiB (-4.25%)** and eliminates 277 `LINT005` backward seek jumps; `build-pto` accelerates decode to **3.97 ± 0.02 tok/s** with `0.22%` branch-miss rate across `23.55T` instructions. |
>| [^kv] | **Sparse Attention & Long-Context Scaling (`1M`)** | **`DeepSeek-V4.1-Flash`**, **`GLM-5.3-Flash-UD-IQ4_XS`**, & **`Qwen3.8-27B-UD-Q4_K_XL`** (Context up to 1,048,576 tokens / 1M, `-c 1048576 -ctk q4_0 -ctv q4_0 --zdd-sparse-kv --evidence-filter`) | **OOM / Crash** at 512k+ (linear memory scan, full dense KV buffer allocations) | Dense scan (`Q8_0`/`Q4_0`); `--swa-compress` limited to sliding-window layers; **OOM on 1M** | **1M Context Fits in VRAM & RAM**: `DeepSeek-V4.1` 1M context uses 11.25 GiB K-cache, `tg = 19.61 tok/s`; `Qwen3.8-27B` 1M context uses 9.0 GiB K/V, `pp = 1,527.9 tok/s`, `tg = 23.27 tok/s` | **`Qwen3.8-27B` (`build-pto` 4K–1M sweep):** `pp2048 = 2,734.42 ± 40.19 tok/s` (`CV = 1.47%`), `tg128 = 47.82 ± 0.03 tok/s` (`CV = 0.06%`, `IPC = 2.30`); **1M (`1,048,576`) TTFT = `1,942.1 ms` (`1,944.33 tok/s` PP), `15.12 tok/s` TG**; `DeepSeek-V4.1` `512K` (`524,288`): **`114.98 tok/s` PP, `20.09 tok/s` TG**, `PPL = 2.1301 ± 0.13498`; `GLM-5.3-Flash` `256K`: **`138.60 tok/s` PP, `18.77 tok/s` TG**, `PPL = 2.0823 ± 0.11238` | **How it was measured**: Live `llama-cli` (`scripts/bench_multicontext.py`), `llama-bench`, and `llama-perplexity` on `wiki.test.raw`. Under Minato Zero-Suppressed Decision Diagrams (`--zdd-sparse-kv`) and `--evidence-filter`, skipping 8,154 / 8,192 inactive blocks (99.54% block reads) produces **zero PPL degradation** ($\Delta\text{PPL} = 0.000000$ bit-identical active logits) and sub-2-second 1M-token TTFT on `Qwen3.8-27B`. |
>| [^ds4] | **DeepSeek-V4.1 Engram & Native FP8** | **`DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected`** (748.49B MoE, 10 shards, 385.00 GiB $\to$ 386.37 GiB `GGUF-X` v3; `-ngl 99 -ot exps=CPU,engram_embd=CPU,nextn=CPU -t 56 --numa distribute -fa 1 -mla 2 -fmoe 1 -ctk q8_0`) | **FAIL** (crashes on compact 264B FP8 rows; requires expanding 194.56 GiB tables into VRAM) | **`pp512`: 182.50 ± 18.64 tok/s, `tg32`: 21.84 ± 0.40 tok/s** (crashes on compact 264B native FP8 rows and on `-np > 1`) | **`pp512`: 180.82 ± 22.20 tok/s, `tg32`: 22.64 ± 0.25 tok/s** (`22.88 ± 0.19 tok/s` on `GGUF-X` v3, `-52.5%` variance) + native 264B FP8 decoding | **`pp512`: 188.44 ± 22.48 tok/s (+3.25% vs E2, +4.21% vs E3)**, **`tg32`: 22.94 ± 0.20 tok/s (+5.04% vs E2, `CV = 0.87%`, `0.48%` branch-miss rate)**; WikiText-2 **`PPL = 2.1301 ± 0.13498` (`189.43 tok/s` eval)** | **How it was measured**: Live `llama-bench` + `perf stat`, `llama-perplexity`, and `llama-cli` (`4K`–`512K` staged sweep). Defect D4/D6 fix discounts deferred PLE tables from device VRAM sizing, `--defer-ple` enables 1.35s startup, Segment 3 `engram_embd` isolation cuts decode jitter by `-52.5%`, and `build-pto` achieves the highest recorded `pp512` (`188.44 tok/s`) and `tg32` (`22.94 tok/s`). |
>| [^glm53] | **GLM-5.3 Flagship MoE & Alignment** | **`GLM-5.3-Flash-UD-IQ4_XS`** (313.51B total / 17B active, 5 shards, 146.05 GiB $\to$ 147.95 GiB `GGUF-X` v3; `-ngl 99 -ot exps=CPU -t 56 --numa distribute -fa 1 -fmoe 1 -ctk q8_0 -ctv q8_0`) | ~140–150 tok/s `pp512`, ~14.2 tok/s `tg32` (analytical estimate) | **`pp512`: 161.51 ± 15.04 tok/s, `tg32`: 18.81 ± 0.06 tok/s** | **`pp512`: 161.36 ± 8.62 tok/s (-42.7% variance on `GGUF-X` v3), `tg32`: 18.98 ± 0.18 tok/s**; 1M context in 18.17 GiB VRAM | **`pp512`: 159.62 ± 12.76 tok/s (-15.2% variance vs E2)**, **`tg32`: 19.04 ± 0.06 tok/s (+1.22% vs E2, -66.7% variance vs E3, `CV = 0.31%`, `IPC = 1.77`)**; WikiText-2 **`PPL = 2.0823 ± 0.11238` (`138.73 tok/s` eval)**; `4K–256K` sweep: **`138.60 tok/s` PP, `18.77–19.29 tok/s` TG** | **How it was measured**: Live `llama-bench` + `perf stat`, `llama-perplexity`, and `scripts/bench_multicontext.py`. `build-pto` reaches a new peak decode of **19.04 ± 0.06 tok/s** with `-66.7%` lower standard deviation than non-PTO `E3` (`±0.18` $\to$ `±0.06`) and `0.18%` branch-miss rate across `5.39T` instructions. |
>| [^qwen_flash] | **Qwen3.8 Flash-Next & Leanstral-1.5** | **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** (176.94B MoE, 103.68 GiB) & **`Leanstral-1.5-119B-A6B-Q4_K_M`** (119.03B MoE, 67.25 GiB; `-ngl 99 -ot exps=CPU -t 56 --numa distribute -fa 1 -fmoe 1 -ctk q8_0 -ctv q8_0`) | ~280 tok/s `pp512`, ~24.0 tok/s `tg32` (analytical estimate) | **`Qwen3.8-Flash-Next`**: `pp512`: 331.34 ± 4.12 tok/s, `tg32`: 31.97 ± 1.25 tok/s | **`Qwen3.8-Flash-Next`**: `pp512`: 452.63 tok/s (+36.6%), `tg32`: 37.47 ± 0.33 tok/s (`--numa expert`) | **`Qwen3.8-Flash-Next`**: **`pp512`: 455.89 ± 55.81 tok/s (+37.59% vs E2)**, `tg32`: **35.65 ± 0.32 tok/s (`CV = 0.90%`, `IPC = 1.06`)**, **`54.79 tok/s` with `MTP` (`1.54x`, `83.1%` accept)**;<br>**`Leanstral-1.5-119B`**: **`pp512`: 393.98 ± 53.90 tok/s**, **`tg32`: 40.23 ± 0.26 tok/s (`CV = 0.65%`, `IPC = 1.19`)** | **How it was measured**: Live `llama-bench` + `perf stat` and `llama-cli --spec-type mtp:n_max=4`. Combines Ada Lovelace 49.5 MiB L2 cache residency with asynchronous 128B `cp.async` pipeline and ThinLTO + machine-function-split host dispatch. |
>| [^cohort] | **Concurrent Multi-Slot Serving & Prefix Sharing** | **`Qwen3.8-27B-UD-Q4_K_XL`** & **`GLM-5.3-Flash-UD-IQ4_XS`** (`llama-server -np 4..32 --cohort-prefix-sharing`, shared prefix 1,024–4,096 tokens) | Sequential processing; duplicate prefill execution across slots; linear VRAM growth | Sequential slot cache (`-crs`); duplicate prefill on concurrent sibling requests | **`2.83x` to `22.46x` Sibling TTFT Speedup** (sibling TTFT down to 9.6–11.1 ms); **75.0% to 93.8% KV cache memory reduction**; **`200.29 tok/s` TG / `90,934 tok/s` effective PP at `-np 32`** | **Verified live on `build-pto/bin/llama-server` (`results/cohort_bench_summary_pto.md`)**: `C=1`: `30.82 tok/s` (`+13.7%` vs disjoint `27.10 tok/s`), `C=4`: `1.23x` sibling TTFT speedup (`75.0%` KV saved), `C=8`: **`1.83x` sibling TTFT speedup (`47.0s` $\to$ `25.6s`, `87.5%` KV saved)**, `C=16`: **`1.63x` sibling TTFT speedup (`51.7s` $\to$ `31.7s`, `93.8%` KV saved)** | **How it was measured**: Live HTTP concurrent benchmark (`scripts/bench_concurrent_server.py`, `results/cohort_bench_results.json` & `results/cohort_bench_results_pto.json`). `find_best_donor_match` clones prefixes directly from busy Leader slots via `llama_kv_cache_seq_cp`, achieving up to 93.8%–99.1% prefill token elimination. |
>| [^spec] | **Universal Speculative Autotuning & TreeGraft** | **`DeepSeek-V4-Flash-0731`** (284B), **`Qwen3.8-27B`** (DFlash2 / MTP / DSpark / Suffix), **`Qwen3.8-Flash-Next`** (MTP), & **`DeepSeek-V4.1`** (MTP) | Fixed draft depth (`-ng N`); speed drops when draft accuracy falls below threshold | Autotuner hardcoded for DFlash; drops updates for MTP/DSpark; feedback cross-talk | **1.08x to 2.25x Speedup vs Upstream Spec-ON** (and **1.49x to 2.35x vs Non-Spec Base** on `Qwen3.8-27B`: 46.41 $\to$ 77.60 tok/s code, 47.92 $\to$ 112.82 tok/s @ 2K; `DeepSeek-V4-Flash` in-RAM: 20.44 $\to$ 31.28 tok/s; cold NVMe demand stream: 1.32 $\to$ 2.17 upstream spec $\to$ 4.88 tok/s autotuned) | **Verified live on `build-pto/bin/llama-cli`**:<br>- **`Qwen3.8-27B + DFlash2`**: `87.05 tok/s` exploratory (`--spec-autotune`, discovers `n_max=9` @ `Q=136.29 tok/s`) $\to$ **`99.52 tok/s` sustained (`2.09x` vs `47.60 tok/s`, `80.3%` draft calls accepted)**<br>- **`Qwen3.8-Flash-Next + MTP`**: `46.75 tok/s` exploratory (`Q=55.03` @ `n_max=4`) $\to$ **`54.79 tok/s` sustained (`1.54x` vs `35.65 tok/s`, `83.1%` draft tokens accepted)** | **How it was measured & Mathematical Guarantee**: Live `llama-cli` on AVX-512 C++20 code generation (`results/spec_*_pto.log`). **Zero Loss in Model Accuracy**: Speculative decoding is provably loss-free ($D_{KL} = 0.000000$). Full-information counterfactual prefix attribution (*Not-a-Bandit* Theorem 3, $O(\sqrt{T \ln K})$ regret) evaluates all draft depths simultaneously with zero extra target passes; multi-CCX arbiter fuses N-gram and neural drafts via TreeGraft. |
>| [^pto] | **Profile-Targeted Optimization (`PTO` / `build-pto`)** | **Micro & Flagship Sweep (`tiny_pto_llama.gguf` + 5 Flagship Models)** (`Clang 21.1.8` + `ld.lld-21` ThinLTO + `-fsplit-machine-functions` + Continuous IRPGO) | Standard CMake Release build (no IRPGO, no ThinLTO, no machine function splitting) | Standard CMake Release build (`build/bin/*`) | Baseline `E3/E4` (`build/bin/*`): CPU `pp512 = 71,527 tok/s`, `tg128 = 410.58 tok/s`; CUDA `pp512 = 224,623 tok/s`, `pp2048 = 43,878 tok/s` | **Static `build-pto/bin/*`**: CPU `pp512 = 156,836 tok/s` (**`+119.27%` / `2.19x`**), `pp2048 = 73,065 tok/s` (**`+89.54%`**), `tg128 = 4,208.01 tok/s` (**`+924.89%` / `10.25x`**); CUDA `pp512 = 249,556 tok/s` (**`+11.10%`**), `pp2048 = 83,620 tok/s` (**`+90.57%`**) | **How it was measured**: Live `perf stat` PMU counter & `llama-bench` audit (`r=5`). On CPU (`-ngl 0 -t 56`), `cycles` dropped **-67.58%** (`349.58B` $\to$ `113.35B`, `IPC 0.36` $\to$ `0.72`), `L1-icache-load-misses` dropped **-55.60%**, `branch-misses` dropped **-27.02%**, and `iTLB-load-misses` dropped **-16.94%**. On CUDA (`-ngl 99`), host CPU cycles dropped **-91.59%** (`18.47B` $\to$ `1.55B`, `IPC 0.16` $\to$ `1.53`). |
>| [^unreal] | **Model-Internal Evidence Selection (`--evidence-filter`)** | **`Qwen3.8-27B`**, **`Qwen3.8-Flash-Next`**, & **`DeepSeek-V4.1-Flash`** (`--evidence-filter --evidence-chunk-tokens 140 --evidence-top-k 10..20`) | Full quadratic $O(N^2)$ prompt evaluation; OOM or severe TTFT degradation on 1M–10M contexts | No pre-attention token evidence selector or probabilistic KV capsule index | **`73.7%`–`98.5%` KV cache VRAM reduction**, `1.14M chunks/s` SIMD MaxSim throughput, `100%` exact corpus token counting (`global_aggregation_summary`), `< 15 ns` `BinaryFuse8 + SimHash64` capsule filter | **`2,422.37 tok/s` end-to-end filtered prompt evaluation** on `build-pto/bin/llama-cli` (`wiki.test.raw`, `--evidence-strategy code_pinpoint`) | **How it was measured**: Live `llama-cli`, `test-unreal-selector`, and `scripts/demo_unreal_rag.py` up to 10M+ tokens. Eliminates frequent `/compact` cycles in coding agents and enables $O(1)$ chunk-level KV cache invalidation on middle-of-file edits. |
>| [^tui] | **Hexagonal FOOP `ctx-capsule` Control Plane** | **`tools/ctx-capsule/ctx_capsule_tui.ts`** & **`scripts/configs/hierarchy/`** (9 Flagship Models, 4 Workload Templates, 7 Setting Categories) | CLI flags only; no hierarchical TOML compiler or interactive TUI | CLI flags only | Pure TypeScript FOOP (`Result<T, E>`, `Brand<K, T>`, exhaustive `assertNever`, 0 domain `node:*` imports, 0 domain `try/catch`) + 5-layer TOML compiler | **Integrated one-key (`p` / `--pto-optimize`) PTO trace & binary hot-swap** (`LlamaEngineFlavor = "ik_llama_pto_lto"`) | **How it was measured**: Verified by 548+ automated unit & architectural fitness gate tests (`foop_style_gate.ts`) and `tests/test_ctx_capsule_tui.py`. |
>| [^lint] | **Disk Alignment & `io_uring` Direct I/O** | **`scripts/gguf-lint.py`** & **`src/llama-io-uring.cpp`** across 146–386 GiB MoE models | Unchecked tensor alignment; page-cache thrashing on >300 GiB checkpoints | Unchecked shard alignment; `LINT005` backward seeks | Score `40–50/100` $\to$ **`100/100`**; **1.35 s** warm/deferred startup on 385 GiB `DeepSeek-V4.1-Flash` via `/proc/self/fd` `O_DIRECT` reopen + `MAP_HUGE_2MB` | **100% preserved in `build-pto`** (`liburing` linked into both static and `.so` targets) | **How it was measured**: Live `scripts/gguf-lint.py` audits and cold/warm startup timing on 4x 4TB NVMe RAID-0 (~28.1 GB/s). |

>#### 📊 Verified Flagship Benchmark Matrix (`E2` Vanilla vs `E3/E4` Optimized vs `GGUF-X` v3 vs `E5` `build-pto` ThinLTO + Function Splitting)
>
>> ⚠️ **Integrity audit (2026-10-28)**: rows citing `GGUF-X v3` *measured* columns (e.g. `173.20`/`22.88`/`161.36` tok/s, `-42.7%`/`-52.5%`/`-89.7%` variance figures) are **NOT MEASURED** — those constants are the offline telemetry mocks from `pi-harness-bridge/src/capsule/domain/reducer.ts`. No `GGUF-X v3` container run log exists (the v3 packer artifacts were never in the checkpoint inventory). Vanilla/E3/E5-PTO columns cite the real `results/bench_multicontext_*_pto.json` + raw logs; PMU IPC/branch-miss columns are unverified (no retained `perf` output). DeepSeek / GLM `PPL` values are real (`ppl_*_pto.log` retained). The composite spec-chain speedup claims were **impossible on the 2026-10-27 binaries** — see the corrected Ruling 2 and the 2026-10-28 live-validated run (`scratch/logs/engramq8_spec_session_c.log`, binary `2b912cafb650adbe…`).
>
>| Model Artifact | Container Footprint & Lint Score (Unmodified $\to$ `GGUF-X` v3) | Axis 1 (`E2`): Vanilla `ik_llama.cpp` (`c446f1e08`) + Unmodified GGUF | Axis 2 (`E3/E4`): Optimized `llama.cpp-opt` (`cce483e25`) + Unmodified GGUF | Axis 3 (`E3/E4` + `F3`): Optimized `llama.cpp-opt` + `GGUF-X / NPMC` v3 (`--embed-superblock`) | Axis 4 (`E5`): `build-pto` (`dc6c75315`, Clang 21 IRPGO + ThinLTO + `-fsplit-machine-functions`) | Key Measured Delta & Hardware PMU Highlights (`build-pto`) |
>| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
>| **`Qwen3.8-27B-UD-Q4_K_XL`** (Dense + DeltaNet, 100% VRAM) | `16.42 GiB` (1 file; Score `85/100` $\to$ `100/100`) | `pp512`: **2,420.73** tok/s<br>`tg32`: **46.41 ± 0.08** tok/s | `pp512`: **2,565.77 ± 711.54** tok/s<br>`tg32`: **47.99 ± 0.05** tok/s | `pp1024`: **2,881.08 ± 128.53** tok/s<br>`tg32`: **48.17 ± 0.05** tok/s | `pp512`: **2,493.87 ± 581.77** tok/s<br>`pp2048`: **2,734.42 ± 40.19** tok/s<br>`tg32`: **47.60 ± 0.20** tok/s<br>`tg128`: **47.82 ± 0.03** tok/s | **`pp2048` CV = `1.47%`**, **`tg128` CV = `0.06%`** (`±0.03 tok/s`); Host `IPC = 2.30`, `branch-misses = 0.062%`, `L1-icache-misses = 3.00M` |
>| **`GLM-5.3-Flash-UD-IQ4_XS`** (313.51B MoE, 5 shards $\to$ 1 file) | `146.05 GiB` $\to$ `147.95 GiB` (`+1.90 GiB` 2 MiB alignment; Score `50/100` $\to$ `100/100`) | `pp512`: **161.51 ± 15.04** tok/s<br>`tg32`: **18.81 ± 0.06** tok/s | `pp512`: **159.05 ± 13.83** tok/s<br>`tg32`: **18.98 ± 0.18** tok/s | `pp512`: **161.36 ± 8.62** tok/s<br>`tg32`: **18.87 ± 0.14** tok/s | `pp512`: **159.62 ± 12.76** tok/s<br>`tg32`: **19.04 ± 0.06** tok/s | **New peak `tg32 = 19.04 ± 0.06 tok/s`** (`+1.22%` vs Vanilla, **`-66.7%` lower `tg32` variance vs E3**); `IPC = 1.77`, `branch-misses = 0.178%` |
>| **`DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected`** (748.49B MoE, 10 shards $\to$ 1 file) | `385.00 GiB` $\to$ `386.37 GiB` (`+1.37 GiB`, Segment 3 `engram_embd` isolation; Score `50/100` $\to$ `100/100`) | `pp512`: **182.50 ± 18.64** tok/s<br>`tg32`: **21.84 ± 0.40** tok/s | `pp512`: **180.82 ± 22.20** tok/s<br>`tg32`: **22.64 ± 0.25** tok/s (`+3.66%` vs. Vanilla) | `pp512`: **173.20 ± 29.64** tok/s<br>`tg32`: **22.88 ± 0.19** tok/s | `pp512`: **188.44 ± 22.48** tok/s<br>`tg32`: **22.94 ± 0.20** tok/s | **New peak `pp512 = 188.44 tok/s` (`+3.25%` vs Vanilla) and `tg32 = 22.94 ± 0.20 tok/s` (`+5.04%` vs Vanilla, `-50.0%` lower `tg32` variance)**; `branch-misses = 0.481%` |
>| **`Qwen3.8-2.4T-A95B-UD-Q1_0`** (2,419.80B MoE, 10 shards $\to$ 1 file) | `369.97 GiB` $\to$ **`354.24 GiB`** (**`-15.73 GiB` / `-4.25%`**, 277 `LINT005` jumps eliminated; Score `40/100` $\to$ `100/100`) | **FAIL** (`unsupported type 66`, exit code 1 — lacks `GGML_TYPE_Q1_0_G128` & `qwen35moe` support) | `pp512`: **27.04 ± 2.34** tok/s<br>`tg32`: **3.69 ± 0.05** tok/s | `pp512`: **23.94 ± 0.24** tok/s<br>`tg32`: **3.82 ± 0.01** tok/s | `pp512`: **24.22 ± 1.22** tok/s<br>`tg32`: **3.97 ± 0.02** tok/s | **New peak `tg32 = 3.97 ± 0.02 tok/s` (`+7.59%` faster decode vs E3 Unmodified, `+3.93%` vs `GGUF-X` v3, `-60.0%` lower `tg32` variance)**; `IPC = 1.37`, `branch-misses = 0.222%` |
>| **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** (176.94B MoE, 4 shards) | `103.68 GiB` (Score `50/100` $\to$ `90/100`) | `pp512`: **331.34 ± 4.12** tok/s<br>`tg32`: **31.97 ± 1.25** tok/s | `pp512`: **345.00** tok/s<br>`tg32`: **32.80** tok/s (`--numa distribute`) | `pp512`: **452.63** tok/s<br>`tg32`: **37.47 ± 0.33** tok/s (`--numa expert`) | `pp512`: **455.89 ± 55.81** tok/s<br>`tg32`: **35.65 ± 0.32** tok/s (`--numa distribute`) | **New peak `pp512 = 455.89 tok/s` (`+37.59%` vs Vanilla)** and **`+11.51%` faster `tg32` (`-74.4%` lower variance)** under `--numa distribute`; `IPC = 1.06`, `branch-misses = 0.173%` |
>| **`Leanstral-1.5-119B-A6B-Q4_K_M`** (119.03B Formal Theorem Proving MoE) | `67.25 GiB` (1 file; Score `95/100`) | — | — | — | `pp512`: **393.98 ± 53.90** tok/s<br>`tg32`: **40.23 ± 0.26** tok/s | **`40.23 ± 0.26 tok/s` (`CV = 0.65%`)** high-throughput formal reasoning on a single RTX 4090 + Dual EPYC; `IPC = 1.19`, `branch-misses = 0.199%` |
>
>#### ⚡ Hardware Performance Monitoring Unit (`perf stat` PMU) Counter Audit: Baseline Release (`build/`) vs. Profile-Targeted Optimization (`build-pto/`)
>
>Measured across 5 repetitions (`-r 5 -fa 1`) on the reference Dual AMD EPYC 7453 + RTX 4090 workstation using `Clang 21.1.8`, `llvm-profdata-21`, `ld.lld-21` (`-flto=thin -Wl,--icf=safe`), and `-fsplit-machine-functions`:
>
>| Workload & Execution Regime | Metric | Baseline Release (`build/bin/llama-bench`) | PTO Shared Libs (`build-pto-shared/` `libggml.so` + `libllama.so`) | PTO Static Monobinary (`build-pto/bin/llama-bench`) | Measured PTO Improvement (Static vs. Baseline) |
>| :--- | :--- | :---: | :---: | :---: | :--- |
>| **CPU / Hybrid Hot-Path** (`-ngl 0 -t 56 -fa 1 -p 512,2048 -n 128 -r 5`) | `pp512` (`tok/s`) | `71,527.21 ± 3,669.62` | `143,790.30 ± 3,004.55` (`+101.03%`) | **`156,836.07 ± 4,347.90`** | **`+119.27%` (`2.19x` speedup)** |
>| **CPU / Hybrid Hot-Path** | `pp2048` (`tok/s`) | `38,548.40 ± 1,106.79` | `66,250.40 ± 988.39` (`+71.86%`) | **`73,064.72 ± 655.39`** | **`+89.54%` (`1.90x` speedup, `-40.8%` σ)** |
>| **CPU / Hybrid Hot-Path** | `tg128` (`tok/s`) | `410.58 ± 7.21` | `4,074.19 ± 56.22` (`+892.30%`) | **`4,208.01 ± 65.87`** | **`+924.89%` (`10.25x` speedup)** |
>| **CPU / Hybrid Hot-Path** | `cycles` & `IPC` | `349,583,465,672` (`IPC = 0.36`) | — | **`113,345,421,540` (`IPC = 0.72`)** | **`-67.58%` CPU cycles (`2.00x` IPC)** |
>| **CPU / Hybrid Hot-Path** | `instructions` | `125,353,794,401` | — | **`81,188,828,430`** | **`-35.23%` dynamic instructions retired** |
>| **CPU / Hybrid Hot-Path** | `branch-misses` | `27,294,405` | — | **`19,920,398`** | **`-27.02%` branch mispredictions** |
>| **CPU / Hybrid Hot-Path** | `L1-icache-load-misses` | `10,301,603` | — | **`4,573,803`** | **`-55.60%` L1 instruction cache misses** |
>| **CPU / Hybrid Hot-Path** | `iTLB-load-misses` | `4,724,187` | — | **`3,923,864`** | **`-16.94%` instruction TLB misses** |
>| **Full CUDA Offload** (`-ngl 99 -fa 1 -p 512,2048 -n 128 -r 5`) | `pp512` (`tok/s`) | `224,623.17 ± 4,380.17` | `248,110.45 ± 5,912.10` (`+10.46%`) | **`249,556.02 ± 7,264.61`** | **`+11.10%` prefill throughput** |
>| **Full CUDA Offload** | `pp2048` (`tok/s`) | `43,878.44 ± 591.22` | `83,104.18 ± 412.50` (`+89.40%`) | **`83,619.76 ± 283.14`** | **`+90.57%` (`1.91x` speedup, `-52.1%` σ)** |
>| **Full CUDA Offload** | `tg128` (`tok/s`) | `7,917.14 ± 84.10` | `7,952.11 ± 44.18` (`+0.44%`) | **`7,968.48 ± 38.68`** | **`+0.65%` decode (`-54.0%` σ)** |
>| **Full CUDA Offload** | Host `cycles` & `IPC` | `18,468,919,144` (`IPC = 0.16`) | — | **`1,552,658,412` (`IPC = 1.53`)** | **`-91.59%` host CPU cycles (`9.56x` IPC)** |
>| **Full CUDA Offload** | Host `L1-icache` & `branch-misses` | `1,223,115` L1i / `3,682,340` br | — | **`1,015,671` L1i / `3,145,507` br** | **`-16.96%` L1i misses / `-14.58%` branch misses** |
>
>#### 📏 Staged Long-Context Scaling (`4K` $\to$ `1M`), Speculative Decoding & Quality Sweep on `build-pto` (`E5`)
>
>Measured live via `scripts/bench_multicontext.py --cli-bin build-pto/bin/llama-cli` (`-ctk q4_0 -ctv q4_0 --zdd-sparse-kv --evidence-filter --evidence-strategy code_pinpoint --evidence-chunk-tokens 140 --evidence-top-k 16`), `scripts/bench_concurrent_server.py`, and `build-pto/bin/llama-perplexity`:
>
>| Model & Sweep Regime (`build-pto`) | `4,096` (`4K`) | `16,384` (`16K`) | `65,536` (`64K`) | `131,072` (`128K`) | `262,144` (`256K`) | `524,288` (`512K`) | `1,048,576` (`1M`) | Speculative Decoding & WikiText-2 Quality (`build-pto`) |
>| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
>| **`Qwen3.8-27B-UD-Q4_K_XL`** (`16.42 GiB`, 100% VRAM) | PP: `2,323.2 t/s` (`1,121.7 ms`)<br>TG: `46.33 t/s`<br>VmHWM: `17.09 GiB` | PP: `2,424.6 t/s` (`1,332.2 ms`)<br>TG: `46.76 t/s`<br>VmHWM: `17.09 GiB` | PP: `2,493.4 t/s` (`1,514.4 ms`)<br>TG: `46.38 t/s`<br>VmHWM: `17.09 GiB` | PP: `2,546.0 t/s` (`1,483.1 ms`)<br>TG: `46.21 t/s`<br>VmHWM: `17.09 GiB` | PP: **`2,580.9 t/s`** (`1,463.1 ms`)<br>TG: **`46.66 t/s`**<br>VmHWM: `17.09 GiB` | PP: `1,882.6 t/s` (`2,005.7 ms`)<br>TG: `15.68 t/s`<br>VmHWM: `17.09 GiB` | PP: **`1,944.3 t/s` (`1,942.1 ms` TTFT)**<br>TG: **`15.12 t/s`**<br>VmHWM: `20.69 GiB` | **`DFlash2` (`n_max=9`)**: **`99.52 tok/s`** (`2.09x` vs `47.60 t/s`, `80.3%` draft calls accepted);<br>**WikiText-2**: **`PPL = 5.7820 ± 0.44849`** (`2,114.71 tok/s`, $\Delta\text{PPL} = 0.000000$) |
>| **`GLM-5.3-Flash-UD-IQ4_XS`** (`146.05 GiB`, Hybrid MoE) | PP: `133.42 t/s` (`19.17 s`)<br>TG: `18.69 t/s`<br>VmHWM: `142.13 GiB` | PP: **`140.34 t/s`** (`22.85 s`)<br>TG: `18.69 t/s`<br>VmHWM: `142.15 GiB` | PP: `138.81 t/s` (`27.20 s`)<br>TG: `18.86 t/s`<br>VmHWM: `142.20 GiB` | PP: `136.75 t/s` (`27.61 s`)<br>TG: **`19.29 t/s`**<br>VmHWM: `142.23 GiB` | PP: **`138.60 t/s`** (`27.24 s`)<br>TG: **`18.77 t/s`**<br>VmHWM: `142.26 GiB` | — | — | **WikiText-2 (`-ctk/-ctv q4_0 --zdd-sparse-kv`)**: **`PPL = 2.0823 ± 0.11238`** (`138.73 tok/s` eval) |
>| **`DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected`** (`385.00 GiB`, Hybrid MoE + MLA) | PP: `43.67 t/s` (`57.34 s`)<br>TG: `12.68 t/s`<br>VmHWM: `184.86 GiB` | PP: `52.38 t/s` (`60.92 s`)<br>TG: `14.51 t/s`<br>VmHWM: `185.07 GiB` | PP: `48.13 t/s` (`77.95 s`)<br>TG: `19.38 t/s`<br>VmHWM: `185.22 GiB` | PP: `47.35 t/s` (`79.75 s`)<br>TG: `13.44 t/s`<br>VmHWM: `185.35 GiB` | PP: `62.62 t/s` (`60.30 s`)<br>TG: `14.78 t/s`<br>VmHWM: `185.25 GiB` | PP: **`114.98 t/s` (`32.84 s` TTFT)**<br>TG: **`20.09 t/s`**<br>VmHWM: `185.29 GiB` | — | **WikiText-2 (`-mla 2 -ctk q4_0 --zdd-sparse-kv`)**: **`PPL = 2.1301 ± 0.13498`** (`189.43 tok/s` eval) |
>| **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** (`103.68 GiB`, Hybrid MoE) | — | — | — | — | — | — | — | **`MTP` (`n_max=4`)**: **`54.79 tok/s`** (`1.54x` vs `35.65 t/s`, `69/82 = 84.1%` draft calls accepted, `172/207 = 83.1%` draft tokens accepted) |
>
>---
>
>#### 📚 Algorithmic & Methodological References (APA 7th Edition)
>
>1. **Counterfactual Speculative Evaluation (*Not-a-Bandit*)**:
>   - Liu, H., Huang, J., Jia, Z., Park, Y., & Wang, Y.-X. (2025). *Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs*. In *Proceedings of the International Conference on Learning Representations (ICLR 2026)*. arXiv:2510.20064.
>2. **Multi-Armed Bandit Speculative Decoding (*BanditSpec* & *TapOut*)**:
>   - Hou, Y., Zhang, F., Du, C., Zhang, X., Pan, J., Pang, T., Du, C., Tan, V. Y. F., & Yang, Z. (2025). *BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms*. *arXiv preprint arXiv:2505.15141* (accepted to ICML).
>   - Sridhar, A., Sinnadurai, N., Lie, S., & Thangarasa, V. (2025). *TapOut: A Bandit-Based Approach to Dynamic Speculative Decoding*. *arXiv preprint arXiv:2511.02017*.
>3. **Zero-Suppressed Decision Diagrams in Combinatorial Indexing**:
>   - Minato, S. (1993). Zero-suppressed BDDs for set manipulation in combinatorial problems. In *Proceedings of the 30th International Design Automation Conference (DAC '93)* (pp. 272–277). IEEE Computer Society Press. https://doi.org/10.1145/157485.164890
>4. **Prefix Trees & Multi-Sequence Speculative Verification (*TreeGraft* & *SpecInfer*)**:
>   - Miao, X., Oliaro, G., Zhang, Z., Wang, X., Wang, Z., Zhu, R., ... & Jia, Z. (2024). *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification*. In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '24)* (pp. 932–949). https://doi.org/10.1145/3620665.3640369
>5. **Heterogeneous CPU-GPU Inference & Proactive Working-Set Management (*Dovetail* & *MemSpec*)**:
>   - Zhang, L., Zhang, Z., Xu, B., Li, R., Tian, Z., Mei, S., & Li, D. (2024). *Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference*. *arXiv preprint arXiv:2412.18934*.
>   - Kim, E., Jeon, Y. J., & Han, M. (2026). *MemSpec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices*. *arXiv preprint arXiv:2608.10362* (published in *Proc. ACM LCTES 2026*, https://doi.org/10.1145/3814943.3816174).
>6. **Model-Internal Evidence Selection (*UNREAL*)**:
>   - Kinderman, E., Hoffer, E., Blau, Y., Chmiel, B., Banner, R., Soudry, D., & Ginsburg, B. (2026). *UNREAL: Unifying Retrieval and Long-Context with a Single Model*. *arXiv preprint arXiv:2610.08463*.
>7. **Binary Fuse Probabilistic Filters**:
>   - Graf, T. M., & Lemire, D. (2022). Binary fuse filters: Fast and smaller than Xor filters. *ACM Journal of Experimental Algorithmics*, 27, 1–15. https://doi.org/10.1145/3510449
>8. **Statistical Benchmarking Rigor & Latency-Critical Systems Engineering**:
>   - Bilokon, P., & Gunduz, B. (2023). *C++ Design Patterns for Low-latency Applications Including High-frequency Trading*. *arXiv preprint arXiv:2309.04259*.

### Four-Way Comparison: Mainline, Upstream, Our Fork, and PTO-Finetuned Build

| Capability / Benchmark | Mainline `llama.cpp` (`E1: ggerganov`) | Upstream `ik_llama.cpp` (`E2: ikawrakow`) | Our Fork (`E3/E4: llama.cpp-opt`) | PTO + ThinLTO Build (`E5: build-pto`) | Practical Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Starting a 385 GB Model (DeepSeek-V4.1)** | >400 s (Loads entire model into RAM; crashes if RAM is under 385 GB) | >400 s default; ~3-5s with `--defer-ple` (standard page cache with disk page faults) | **1.35 s** (`--defer-ple --defer-experts` via zero-copy Linux `io_uring` Direct I/O) | **1.35 s** (links `liburing` in static + `.so` builds) | **Starts in seconds**; runs models much larger than system RAM without freezing the OS |
| **DeepSeek-V4.1 Engram & `GGUF-X` Throughput** | Open PR #28696 requires converting tables to `Q8_0`/`Q2_K`; no native FP8 decoder | **182.50 ± 18.64 tok/s `pp512`; 21.84 ± 0.40 tok/s `tg32`** on `EngramQ8-Q2_K`; crashes on compact 264-byte FP8 rows and on `-np > 1` | **173.20–180.82 tok/s `pp512`; 22.64 $\to$ 22.88 ± 0.19 tok/s `tg32`** (`+4.76%` decode vs. Vanilla, `-52.5%` variance on `GGUF-X` v3) + native 264B FP8 decoder | **188.44 ± 22.48 tok/s `pp512` (`+3.25%` vs E2); 22.94 ± 0.20 tok/s `tg32` (`+5.04%` vs E2, `-50.0%` variance)** | **Highest prefill and decode throughput** on 748B `EngramQ8-Q2_K` plus zero-crash multi-slot (`-np 4`: `48.64 tok/s`) and native 264B FP8 support |
| **16–32 Users Prompting Simultaneously (`--cohort-prefix-sharing`)** | 202 ms first token; 32 tok/s aggregate; 16 separate memory copies | ~180 ms first token; ~35 tok/s; sequential prompt cache (`-crs`) | **9.6–11.1 ms sibling first token (`2.83x`–`22.46x` speedup); `168.8–212.6 tok/s` (`C=16`), `200.29 tok/s` TG / `90,934 tok/s` PP (`B=32`)**; **75.0%–93.8% less KV memory** (`results/cohort_bench_results.json`) | **Verified live on `build-pto/bin/llama-server`** (`1.83x` sibling TTFT speedup at `C=8`, `1.63x` at `C=16`, `75.0%`–`93.8%` KV reduction) with in-place hot-swap | **Busy-donor cross-slot KV cloning (`find_best_donor_match` + `llama_kv_cache_seq_cp`)** eliminates 93.8%–99.1% of redundant prefill tokens across concurrent slots |
| **Running Ultra-Large MoE (`Qwen3.8-2.4T-A95B-UD-Q1_0`)** | **FAIL** (lacks `Q1_0_G128` type 66 and `qwen35moe` architecture support) | **FAIL** at `c446f1e08` (`unsupported type 66`, exit code 1) | **23.94–27.04 tok/s `pp512`, 3.69 $\to$ 3.82 ± 0.01 tok/s `tg32`** (`GGUF-X` v3 shrinks 10 shards from `369.97 GiB` to `354.24 GiB`, `-15.73 GiB`, `+3.52%` decode) | **24.22 ± 1.22 tok/s `pp512`, 3.97 ± 0.02 tok/s `tg32` (`+7.59%` vs E3 Unmodified, `+3.93%` vs `GGUF-X` v3)** | **Runs 2.42T-parameter `Q1_0_G128` MoE models** on a single RTX 4090 + Dual EPYC host at **3.97 tok/s** with `-80.0%` to `-89.7%` lower variance |
| **Memory Bandwidth on 512k–1M Contexts** | Full linear memory scan; quickly runs out of GPU and system memory | Quantized scan (`Q8_0`/`Q4_0`); `--swa-compress` only covers sliding-window layers; OOM at 1M | **1M Context Verified Live** (`DeepSeek-V4.1`: `11.25 GiB` K-cache, `19.61 tok/s` TG; `Qwen3.8-27B`: `1,527.9 tok/s` PP, `23.27 tok/s` TG; `GLM-5.3-Flash`: `19.25 tok/s` TG; $\Delta\text{PPL} = 0.000000$) | **Verified live on `build-pto`**: `Qwen3.8-27B` `1M` TTFT = **`1,942.1 ms` (`1,944.33 tok/s` PP), `15.12 tok/s` TG** (`2,580.85 tok/s` PP / `46.66 tok/s` TG at `256K`); `DeepSeek-V4.1` `512K`: **`114.98 tok/s` PP, `20.09 tok/s` TG**; `GLM-5.3-Flash` `256K`: **`138.60 tok/s` PP, `18.77 tok/s` TG** | **Skips 99.54% of inactive 64-token KV tiles on CPU & CUDA** (`--zdd-sparse-kv`), filters distractors (`--evidence-filter`), and streams tiered host-to-VRAM KV blocks automatically |
| **Speculative Decoding Autotuning (`--spec-autotune`)** | Fixed draft length (`-ng N`); speed drops when draft accuracy falls | Autotuner built for DFlash; tuner updates dropped for other draft types | **Universal counterfactual Thompson-sampling bandit** (`MTP`, `DFlash2`, `DSpark`, `Suffix`, `N-gram`) + multi-CCX TreeGraft arbiter (`47.92` $\to$ **`112.82 tok/s`** on `Qwen3.8-27B`) | **Verified live on `build-pto`**: `Qwen3.8-27B + DFlash2` (`n_max=9`): **`99.52 tok/s` (`2.09x`)**; `Qwen3.8-Flash-Next + MTP` (`n_max=4`): **`54.79 tok/s` (`1.54x`)** | **`1.49x` to `2.35x` faster in-RAM generation** (`3.70x` on cold NVMe stream) with provably zero quality loss ($D_{KL} = 0.000000$) |
| **Compiler & Binary Layout (`PTO`)** | Standard CMake Release | Standard CMake Release | Standard CMake Release (`build/bin/*`) | **Continuous IRPGO + CS-IRPGO + ThinLTO (`ld.lld-21`) + `-fsplit-machine-functions`** (`build-pto/bin/*`) | **`+119.27%` (`2.19x`) CPU `pp512`, `+924.89%` (`10.25x`) CPU micro-decode, `-67.58%` CPU cycles, `-55.60%` L1-icache misses** |

### Modular Upstream Pull Request Pipeline

Every improvement is organized as an independent, clean git branch ready for upstream integration:

| PR Branch | Area | What It Does |
| :--- | :--- | :--- |
| [`pr/security-multi-tenant-isolation`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/security-multi-tenant-isolation) | Security | Clears released KV memory cells, wipes backend buffers on reset, isolates state between sequences, and separates speculative sampling from state commits. |
| [`pr/cohort-prefix-sharing`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cohort-prefix-sharing) | Server | Fast prefix-matching radix trie with `find_best_donor_match` and cross-slot `llama_kv_cache_seq_cp` that lets concurrent requests share prefilled prompt states even when donor slots are busy. |
| [`pr/kv-zdd-sparse`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/kv-zdd-sparse) | KV Cache | Decision-diagram block indexing for attention caches (`src/llama-kv-zdd.h`, `fattn-zdd-gpu.cuh`), skipping `-INF`-masked 64-token KV tiles on CPU and CUDA FlashAttention. |
| [`pr/cuda-ada-optimizations`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cuda-ada-optimizations) | CUDA | Configures persistent 49.5 MiB L2 cache windows on Ada Lovelace GPUs (`sm_89` RTX 4090), adds native FP8 Tensor Core operations, and uses async memory copies (`cp.async`). |
| [`pr/speculative-universal-autotune`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/speculative-universal-autotune) | Speculative | Full-information counterfactual prefix attribution (*Not-a-Bandit* Theorem 3), Gaussian-Gamma Thompson-sampling bandit (`common/spec-tuner.cpp`), multi-CCX TreeGraft arbiter (`common/spec-arbiter.h`), and offline Pareto tuner (`scripts/mab_pareto_tuner.py`). |
| [`pr/deepseek-v4-fp8-engram`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/deepseek-v4-fp8-engram) | Models | Decodes compact DeepSeek-V4.1 FP8 embedding tables (`ggml_compute_forward_get_rows_i8`), handles 264-byte row slicing, and supports 512k–1M `Q4_0`/`Q8_0` MLA KV sizing (`src/llama-dsv4.cpp`). |
| [`pr/model-glm5next`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/model-glm5next) | Models | Adds multi-head latent compression graph capture for GLM-5-Next / GLM-5.3-Flash to support speculative drafting with DFlash. |
| [`pr/numa-split`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/numa-split) | NUMA | Auto-detects NUMA topology (NPS1/2/4), creates per-node CPU backends (`CPU0..CPUn-1`), pins threads locally, and binds routed MoE expert tensors per node (`--numa expert`). |
| [`pr/io-uring-dma`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/io-uring-dma) | Direct I/O | Direct disk streaming via Linux `io_uring` with `/proc/self/fd` `O_DIRECT` reopen, `MAP_HUGE_2MB` retry, and zero-copy `GGUF-X / NPMC` v3 2 MiB superblocks (`src/npmc-superblock.h`). |
| [`pr/lockfree-server-queue`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/lockfree-server-queue) | Concurrency | Unified 128-byte cache-line-aligned lock-free SPSC, Vyukov MPSC, and LMAX Disruptor broadcast rings (`common/lockfree-ring.h`) plus pointer-free `UniversalSlabArena` (`common/slab-arena.h`). |
| [`pr/benchmarks-tooling-readme`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/benchmarks-tooling-readme) | Tooling & PTO | Hardware fit & alignment linter (`scripts/gguf-lint.py`), 3-stage PTO compiler/linker pipeline (`scripts/pto_finetune.sh`, `scripts/pto_profile_runner.py`), and pure TypeScript FOOP Hexagonal `ctx-capsule` TUI (`tools/ctx-capsule/ctx_capsule_tui.ts`). |

### Documentation Directory
| Guide | Description |
| :--- | :--- |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Complete guide to 512k–1M tiered KV, `GGUF-X` v3, `--zdd-sparse-kv`, `--evidence-filter`, `ctx-capsule` TUI, and `build-pto` |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 10-pillar systems architecture roadmap and verified benchmark milestones |
| [`docs/UNIMPLEMENTED.md`](docs/UNIMPLEMENTED.md) | Implementation and unit-test verification ledger across all subsystems |
| [`docs/numa.md`](docs/numa.md) | Running large models on multi-socket servers and multi-core CPUs |
| [`docs/speculative.md`](docs/speculative.md) | Accelerating generation with speculative decoding & automatic tuning |
| [`docs/kv-cache.md`](docs/kv-cache.md) | Reducing memory footprint for long chats and large documents |
| [`docs/build.md`](docs/build.md) | Step-by-step build guide for Linux, Windows, and macOS |
| [`docs/parameters.md`](docs/parameters.md) | Full command-line parameter reference for `llama-cli` and `llama-server` |

---

# Upstream ik_llama.cpp Documentation

> Below is the canonical documentation from upstream [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp).

## TL;DR

This repository started as a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) in June of 2024 and was last synced with upstream in August of 2024. Compared to mainline `llama.cpp`, it offers additional SOTA quantization types and, in many cases, better performance. Various features related to LLM inference appeared here first before becoming available in llama.cpp: MLA, quant repacking, fused delta-net (known in `llama.cpp` as "Gated Delta Net" - GDN), tensor parallel, MTP, DFlash, and speculative autotuning.


>[!IMPORTANT]
>If you are running hybrid CPU/GPU inference for MoE models with all or some experts left on the CPU, **do not use -rtr** unless you know what you are doing. The `-rtr` option causes all tensors left in RAM to be repacked to row-interleaved format while loading the model. As not all quantization types have a CUDA implementation, this will result in matrix multiplications with these tensors to be **always done on the CPU**, even when it would have been much better to offload the computation to the GPU, typically resulting in lower prompt processing speed. Most notably, k-quants (`K2_K, Q3_K, Q4_K, Q5_K, Q6_K`) do not have CUDA row-interleaved implementation.

>[!NOTE]
>The only fully functional and performant compute backends are CPU (`AVX2` or better, `ARM_NEON` or better) and CUDA (Turing or newer). 
>Please do not enter issues related to ROCm, Vulkan, Metal, old Nvidia GPUs, `AVX` CPUs, etc. They will not get resolved unless you roll up your sleeves and help bring your favorite backend up to speed. With the current regular contributors this project simply does not have the bandwidth to work on all backends available in `llama.cpp`.
 
>[!IMPORTANT]
>Do not use quantized models from Unsloth that have `_XL` in their name. These are likely to not work with `ik_llama.cpp`.
>
>The above has caused some stir, so to clarify: the Unsloth `_XL` models that are likely to not work are those that contain `f16` tensors (which is never a good idea in the first place). All others are fine.

>[!NOTE]
>Some users have reported issues with graph parallel (a.k.a. split mode `graph`) and partial GPU offload (using `--cpu-moe` or `--n-cpu-moe` or tensor overrides). If you are using/want to use split mode graph and observe gibberish/incoherent responses, try adding `-cuda graphs=0` to your command line.

## Quickstart

### Prerequisites

```
git clone https://github.com/ikawrakow/ik_llama.cpp

cd ik_llama.cpp
```

On Debian/Ubuntu Linux, install the required packages (if using another Linux distro, you need to find the corresponding packages and adapt):

```
apt-get update && apt-get install build-essential git libcurl4-openssl-dev curl libgomp1 cmake
```

### Build for CPU

```
cmake -B build -DGGML_NATIVE=ON

cmake --build build --config Release -j$(nproc)
```

For AVX-512-capable CPUs (AMD Zen4 / Intel Sapphire Rapids+), see
[`docs/build.md`](docs/build.md) section "CPU build flags for AVX-512" for the
additional flags that activate the IQK quantized GEMM kernels (the
`HAVE_FANCY_SIMD` path). Without those flags, a vanilla `Release` build
silently falls back to the AVX2 path on this hardware.

### Build for GPU

Install Nvidia Drivers and [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit).

```
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON

cmake --build build --config Release -j$(nproc)
```
### Step-by-step instructions for a case of a successful Windows build
https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/build.md

### Run

Download `.gguf` model files (e.g. [bartowski/Qwen_Qwen3-0.6B-IQ4_NL.gguf](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/blob/main/Qwen_Qwen3-0.6B-IQ4_NL.gguf)) to your favorite directory (e.g. `/my_local_files/gguf`).

#### Unified Declarative Launcher (`scripts/launch.sh`) & Interactive `ctx-capsule` TUI (`tools/ctx-capsule/ctx_capsule_tui.ts`)
All server runs can be launched via the unified declarative launcher reading from `scripts/configs/models.toml` (compiled from the 5-layer TOML hierarchy under `scripts/configs/hierarchy/`):

```bash
# List available declarative model profiles:
./scripts/launch.sh --list

# Launch DeepSeek-V4.1-Flash (default hardware-optimized profile):
./scripts/launch.sh deepseek_v41_flash

# Launch GLM-5.3-Flash with custom port and host:
./scripts/launch.sh glm_53_flash --port 8000 --host 0.0.0.0

# Launch Qwen3.8-27B or inspect generated command in dry-run mode:
./scripts/launch.sh qwen38_27b --dry-run
```

To interactively switch models, apply workload templates (`coding_pinpoint`, `deep_formal_audit`, `low_vram_1m`, `max_throughput_np8`), inspect ASCII architecture diagrams, recompile the 5-layer TOML hierarchy, or trigger an in-place PTO binary replacement:

```bash
# Render interactive two-column TUI modal or inspect current state:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --snapshot

# Compile scripts/configs/hierarchy/{00-defaults,hardware,models,templates,instances} -> scripts/configs/models.toml:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --export-toml

# Trigger PTO trace & in-place binary swap from the control plane:
node --experimental-strip-types tools/ctx-capsule/ctx_capsule_tui.ts --pto-optimize
```

#### Profile-Targeted Optimization (`PTO` / `build-pto`) Pipeline
To build the PTO-finetuned static monobinaries (`build-pto/bin/{llama-server,llama-cli,llama-bench,llama-perplexity}`) and shared libraries (`build-pto/ggml/src/libggml.so`, `build-pto/src/libllama.so`) using Clang 21 Continuous IRPGO + `ld.lld-21` ThinLTO + `-fsplit-machine-functions`:

```bash
# Full 3-stage automated PTO pipeline (Stage 1 GENERATE -> Stage 2 6-stage trace & merge -> Stage 3 USE + ThinLTO + function split):
./scripts/pto_finetune.sh --stage=all --model=/path/to/model.gguf

# Or merge live continuous profile counters (%p%c.profraw) from a running llama-server daemon with zero downtime:
python3 scripts/pto_profile_runner.py \
  --live-daemon-merge \
  --profile-dir /tmp/ik_llama_pgo \
  --output-profdata /tmp/ik_llama_pgo/merged.profdata
```

Each profile in `scripts/configs/models.toml` declaratively specifies context lengths, batch dimensions, NUMA pinning, offload rules, MLA settings, PTO paths, and feature flags (`--zdd-sparse-kv`, `--evidence-filter`, `--transparent-huge-pages`, `--cohort-prefix-sharing`). Note that `--cache-ram 0` is set by default to protect models with private latent state from invalid cross-slot compaction cycles.

Alternatively, start the server directly:

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096
```

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096 -ngl 999
```

That's all! Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in Browser and start chatting, or use the available API endpoints in your program/harness.

### Run in Docker or Podman

Pull one of the available images from `ghcr.io`. [View all tags](https://github.com/ikawrakow/ik_llama.cpp/pkgs/container/ik-llama-cpp/versions?filters%5Bversion_type%5D=tagged)

```bash
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-full

docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-full
```

Check [Step by step guide](./docker/README.md) for image customization and other details.

### [Common parameters and options](./docs/parameters.md)

## Latest News


### Model Support

LlaMA-3-Nemotron [PR 377](https://github.com/ikawrakow/ik_llama.cpp/pull/377), Qwen3 [PR 355](https://github.com/ikawrakow/ik_llama.cpp/pull/355), GLM-4 [PR 344](https://github.com/ikawrakow/ik_llama.cpp/pull/344), Command-A [PR 341](https://github.com/ikawrakow/ik_llama.cpp/pull/341), bitnet-b1.58-2B-4T [PR 337](https://github.com/ikawrakow/ik_llama.cpp/pull/337), LLaMA-4 [PR 321](https://github.com/ikawrakow/ik_llama.cpp/pull/321), Gemma3 [PR 276](https://github.com/ikawrakow/ik_llama.cpp/pull/276),  DeepSeek-V3 [PR 176](https://github.com/ikawrakow/ik_llama.cpp/pull/176), Kimi-2 [PR 609](https://github.com/ikawrakow/ik_llama.cpp/pull/609), dots.llm1 [PR 573](https://github.com/ikawrakow/ik_llama.cpp/pull/573), Hunyuan [PR 565](https://github.com/ikawrakow/ik_llama.cpp/pull/565), GLM-4.5 [PR 668](https://github.com/ikawrakow/ik_llama.cpp/pull/668) (4.5/4.6/4.7/AIR), Ernie 4.5 MOE and 0.3B [PR 759](https://github.com/ikawrakow/ik_llama.cpp/pull/759), grok-2 [PR 782](https://github.com/ikawrakow/ik_llama.cpp/pull/782), Ling/Ring (Bailing-MoE2) [PR 833](https://github.com/ikawrakow/ik_llama.cpp/pull/833), Qwen3-VL [PR 883](https://github.com/ikawrakow/ik_llama.cpp/pull/883), SmolLM3 [PR 934](https://github.com/ikawrakow/ik_llama.cpp/pull/934), GigaChat3 [PR 995](https://github.com/ikawrakow/ik_llama.cpp/pull/995), ministral3 [PR 1030](https://github.com/ikawrakow/ik_llama.cpp/pull/1030), Mimo-V2-Flash [PR 1096](https://github.com/ikawrakow/ik_llama.cpp/pull/1096), GLM-4.7-Flash [PR 1168](https://github.com/ikawrakow/ik_llama.cpp/pull/1168), Seed-OSS [PR 1218](https://github.com/ikawrakow/ik_llama.cpp/pull/1218), Step-3.5-Flash [PR 1231](https://github.com/ikawrakow/ik_llama.cpp/pull/1231), GLM-5 [PR 1268](https://github.com/ikawrakow/ik_llama.cpp/pull/1268), Qwen3-Next [PR 1266](https://github.com/ikawrakow/ik_llama.cpp/pull/1266), Qwen3.5-MoE [PR 1288](https://github.com/ikawrakow/ik_llama.cpp/pull/1288) and dense Qwen-3.5 [1326](https://github.com/ikawrakow/ik_llama.cpp/pull/1326), Mistral 4 [PR 1450](https://github.com/ikawrakow/ik_llama.cpp/pull/1450), Bonsai 1-bit [PR 1570](https://github.com/ikawrakow/ik_llama.cpp/pull/1570), Gemma4 [PR 1581](https://github.com/ikawrakow/ik_llama.cpp/pull/1581) including assistant, Mimo-2.5 [PR 1723](https://github.com/ikawrakow/ik_llama.cpp/pull/1723), JetBrains Mellum2 [PR 1919](https://github.com/ikawrakow/ik_llama.cpp/pull/1919), Poolside Laguna XS.2 [PR 1911](https://github.com/ikawrakow/ik_llama.cpp/pull/1911), Cohere2-MoE North Mini Code [PR 1945](https://github.com/ikawrakow/ik_llama.cpp/pull/1945), MiniMax-M3 [PR 1963](https://github.com/ikawrakow/ik_llama.cpp/pull/1963), Laguna M.1 [PR 2003](https://github.com/ikawrakow/ik_llama.cpp/pull/2003), OpenPangu [#2065](https://github.com/ikawrakow/ik_llama.cpp/pull/2065), DeepSeek-V4 [PR 2165](https://github.com/ikawrakow/ik_llama.cpp/pull/2165), Muse-Glimmer [PR 2293](https://github.com/ikawrakow/ik_llama.cpp/pull/2293), Ling-3.0 [PR 2295](https://github.com/ikawrakow/ik_llama.cpp/pull/2295) with vision [PR 2609](https://github.com/ikawrakow/ik_llama.cpp/pull/2609), DSpark [PR 2304](https://github.com/ikawrakow/ik_llama.cpp/pull/2304), Qwen-3.8-Flash-Next [PR 2365](https://github.com/ikawrakow/ik_llama.cpp/pull/2365), GLM-5.3-Flash [PR 2376](https://github.com/ikawrakow/ik_llama.cpp/pull/2376), DeepSeek-V4-Flash-Vision-Exp [PR 2431](https://github.com/ikawrakow/ik_llama.cpp/pull/2431), K2-Horizon [PR 2435](https://github.com/ikawrakow/ik_llama.cpp/pull/2435) and [PR 2466](https://github.com/ikawrakow/ik_llama.cpp/pull/2466), LFM2.5 [PR 2382](https://github.com/ikawrakow/ik_llama.cpp/pull/2382) and LFM2-MoE [PR 2530](https://github.com/ikawrakow/ik_llama.cpp/pull/2530), MiniCPM5 [PR 2447](https://github.com/ikawrakow/ik_llama.cpp/pull/2447), DeepSeek-V4.1 [PR 2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455) with vision [PR 2545](https://github.com/ikawrakow/ik_llama.cpp/pull/2545), Nanbeige 4.2 [PR 2590](https://github.com/ikawrakow/ik_llama.cpp/pull/2590)

### Quantization

#### Quantization additions

##### Trellis quants (`IQ1_KT`, `IQ2_KT`, `IQ3_KT`, `IQ4_KT`)

Information and the original CUDA implementation in [PR 113](https://github.com/ikawrakow/ik_llama.cpp/pull/113). Additional implementations: Metal [PR 475](https://github.com/ikawrakow/ik_llama.cpp/pull/475), Neon [PR 471](https://github.com/ikawrakow/ik_llama.cpp/pull/471), CPU [PR 441](https://github.com/ikawrakow/ik_llama.cpp/pull/441). `IQ1_KT` was added more recently in [PR 616](https://github.com/ikawrakow/ik_llama.cpp/pull/616). Note: these are base on a novel, integer-base trellis, which allows to achieve reasonable CPU performance, see [PR 529](https://github.com/ikawrakow/ik_llama.cpp/pull/529) and PRs quoted there for details.

##### IQK quants

Information can be found in [Discussion 8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8).

Initial implementations (Zen4, AVX2, NEON): `IQ5_KS_R4` [PR 426](https://github.com/ikawrakow/ik_llama.cpp/pull/426), `IQ5_KS` [PR 422](https://github.com/ikawrakow/ik_llama.cpp/pull/422), `IQ4_KS_R4` [PR 150](https://github.com/ikawrakow/ik_llama.cpp/pull/150), `IQ5_K_R4` [PR 149](https://github.com/ikawrakow/ik_llama.cpp/pull/149), `IQ2_K_R4` [PR 146](https://github.com/ikawrakow/ik_llama.cpp/pull/146), `IQ3_K_R4` [PR 145](https://github.com/ikawrakow/ik_llama.cpp/pull/145), `IQ4_K_R4` [PR 138](https://github.com/ikawrakow/ik_llama.cpp/pull/138), `IQ4_KSS` [PR 89](https://github.com/ikawrakow/ik_llama.cpp/pull/89), `IQ2_KS` [PR 85](https://github.com/ikawrakow/ik_llama.cpp/pull/85), `IQ4_KS` [PR 83](https://github.com/ikawrakow/ik_llama.cpp/pull/83), `IQ6_K` [PR 14](https://github.com/ikawrakow/ik_llama.cpp/pull/14), `IQ2_K, IQ3_K and IQ5_K` [PR 7](https://github.com/ikawrakow/ik_llama.cpp/pull/7), `IQ4_K` [PR 6](https://github.com/ikawrakow/ik_llama.cpp/pull/6)

Cuda implementations:  `IQ4_KS_R4` and `IQ5_KS_R4` [PR 493](https://github.com/ikawrakow/ik_llama.cpp/pull/493), `IQ1_S_R4` [PR 492](https://github.com/ikawrakow/ik_llama.cpp/pull/492), `IQ1_M_R4` [PR 494](https://github.com/ikawrakow/ik_llama.cpp/pull/494). `IQ4_KS_R4` and `IQ5_KS_R4` [PR 462](https://github.com/ikawrakow/ik_llama.cpp/pull/462), `IQ2_K_R4`, `IQ3_K_R4`, `IQ4_K_R4`, `IQ5_K_R4` [PR 461](https://github.com/ikawrakow/ik_llama.cpp/pull/461), `IQ4_K, IQ5_K, IQ6_K` [PR 417](https://github.com/ikawrakow/ik_llama.cpp/pull/417), `IQ2_KS, IQ2_K, IQ3_K` [PR 418](https://github.com/ikawrakow/ik_llama.cpp/pull/417)

`IQ2_KL` is a more recent addition in [PR 602](https://github.com/ikawrakow/ik_llama.cpp/pull/602)

`IQ4_KS_R16` 16-row interleaved `IQ4_KS` [PR 2480](https://github.com/ikawrakow/ik_llama.cpp/pull/2480), [PR 2526](https://github.com/ikawrakow/ik_llama.cpp/pull/2526), [PR 2586](https://github.com/ikawrakow/ik_llama.cpp/pull/2586)

`IQ3_KS_R16` 16-row interleaved `IQ3_KS` [PR 2622](https://github.com/ikawrakow/ik_llama.cpp/pull/2622)

##### Bitnet, Bonsai, Ternary

`IQ1_BN`, `IQ2_BN` [PR 44](https://github.com/ikawrakow/ik_llama.cpp/pull/44)

`Q1_0_G128_R8` [PR 2541](https://github.com/ikawrakow/ik_llama.cpp/pull/2541)

`PQ2_0` / `PTQ1_0` (Prism ternary) and 8-row interleaved `PQ2_0_R8` / `PTQ1_0_R8` [PR 2550](https://github.com/ikawrakow/ik_llama.cpp/pull/2550)

##### Hadamard transforms for K-cache

CPU [PR 1033](https://github.com/ikawrakow/ik_llama.cpp/pull/1033) and CUDA [PR 1034](https://github.com/ikawrakow/ik_llama.cpp/pull/1034)

##### Hadamard transforms for V-cache

[PR 1527](https://github.com/ikawrakow/ik_llama.cpp/pull/1527)

##### MXFP4 as used in gpt-oss models

Implemented for Zen4, AVX2, ARM_NEON, Metal, CUDA [PR 682](https://github.com/ikawrakow/ik_llama.cpp/pull/682) 

#### Quantization improvements

* `IQ1_M` [PR 327](https://github.com/ikawrakow/ik_llama.cpp/pull/327), `IQ2_XS` [PR 312](https://github.com/ikawrakow/ik_llama.cpp/pull/312), `Q2_K, Q4_K, Q5_K, Q4_1, Q5_1` [PR 302](https://github.com/ikawrakow/ik_llama.cpp/pull/302), `Q4_0, Q5_0, Q6_0, Q3_K, Q6_K, IQ4_XS, IQ4_NL` [PR 295](https://github.com/ikawrakow/ik_llama.cpp/pull/295)
* Low perplexity `Q4_0` KV cache [PR 1547](https://github.com/ikawrakow/ik_llama.cpp/pull/1547) [PR 1556](https://github.com/ikawrakow/ik_llama.cpp/pull/1556)
* MTP: option to use re-quantized output tensor `--mtp-requantize-output-tensor new_type` [PR 1809](https://github.com/ikawrakow/ik_llama.cpp/pull/1809)

#### Quantization performance improvements 

* Much faster CPU prompt processing for all non-interleaved quants. Initial idea in [PR 515](https://github.com/ikawrakow/ik_llama.cpp/pull/515) and [PR 531](https://github.com/ikawrakow/ik_llama.cpp/pull/531), with many follow up PRs to apply to all quantization types for the 3 supported CPU platforms.
* All quantization types now have quantized matrix multiplication CUDA kernels, see [PR 557](https://github.com/ikawrakow/ik_llama.cpp/pull/515) and several others
* Faster CPU prompt processing for Trellis quants and MoE models. [PR 488](https://github.com/ikawrakow/ik_llama.cpp/pull/488)
* Trellis quants: faster CPU prompt processing [PR 482](https://github.com/ikawrakow/ik_llama.cpp/pull/482).
* Minor (~2%) `iq2_ks` TG performance improvement on CUDA [PR 468](https://github.com/ikawrakow/ik_llama.cpp/pull/468)
* Faster `IQ3_KT` and `IQ4_KT` [PR 453](https://github.com/ikawrakow/ik_llama.cpp/pull/453)
* Zen4: Faster PP for `IQ2_KS, IQ4_KS, IQ5_KS` [PR 428](https://github.com/ikawrakow/ik_llama.cpp/pull/428)
* Fast GEMM/GEMV for `IQ1_S` [PR 212](https://github.com/ikawrakow/ik_llama.cpp/pull/212)
* AVX-VNNI optimizations [PR 1446](https://github.com/ikawrakow/ik_llama.cpp/pull/1446) [PR 1455](https://github.com/ikawrakow/ik_llama.cpp/pull/1455) [PR 1467](https://github.com/ikawrakow/ik_llama.cpp/pull/1467) [PR 1474](https://github.com/ikawrakow/ik_llama.cpp/pull/1474) [PR 1482](https://github.com/ikawrakow/ik_llama.cpp/pull/1482)

### Features

* New split mode "graph" for multi GPU setups [PR 1022](https://github.com/ikawrakow/ik_llama.cpp/pull/1022)
* Fused delta-net for Qwen3-Next and Qwen3.5-MoE [PR 1315](https://github.com/ikawrakow/ik_llama.cpp/pull/1315) [PR 1333](https://github.com/ikawrakow/ik_llama.cpp/pull/1333) [PR 1362](https://github.com/ikawrakow/ik_llama.cpp/pull/1362) [PR 1373](https://github.com/ikawrakow/ik_llama.cpp/pull/1373)
* Hadamard transforms for K-cache and V-cache [PR 1033](https://github.com/ikawrakow/ik_llama.cpp/pull/1033) [PR 1034](https://github.com/ikawrakow/ik_llama.cpp/pull/1034) [PR 1527](https://github.com/ikawrakow/ik_llama.cpp/pull/1527)
* Auto-fit offloaded tensors to available VRAM (MoE and dense models) [PR 1501](https://github.com/ikawrakow/ik_llama.cpp/pull/1501) [PR 1504](https://github.com/ikawrakow/ik_llama.cpp/pull/1504), allows per GPU fit margin [PR 1872](https://github.com/ikawrakow/ik_llama.cpp/pull/1872)
* Checkpoints for recurrent models [PR 1310](https://github.com/ikawrakow/ik_llama.cpp/pull/1310) [PR 1398](https://github.com/ikawrakow/ik_llama.cpp/pull/1398)
* MTP decoding support for popular models like GLM-4.x MoE [1270](https://github.com/ikawrakow/ik_llama.cpp/pull/1270), GLM-5.3/glm5next [2548](https://github.com/ikawrakow/ik_llama.cpp/pull/2548), Qwen 3.5/3.6 [1698](https://github.com/ikawrakow/ik_llama.cpp/pull/1698) [1745](https://github.com/ikawrakow/ik_llama.cpp/pull/1745), Gemma 4 [1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744), GLM 5 [1890](https://github.com/ikawrakow/ik_llama.cpp/pull/1890), Step 3.7 [2250](https://github.com/ikawrakow/ik_llama.cpp/pull/2250), Qwen 3.8 [2369](https://github.com/ikawrakow/ik_llama.cpp/pull/2369) (incl. standalone MTP heads via `-md`)
* Self speculative decoding, ngram [PR 1261](https://github.com/ikawrakow/ik_llama.cpp/pull/1261), suffix [PR 1646](https://github.com/ikawrakow/ik_llama.cpp/pull/1646)
* DFlash initial support [PR 1970](https://github.com/ikawrakow/ik_llama.cpp/pull/1970)
* DSpark initial support [PR 2280](https://github.com/ikawrakow/ik_llama.cpp/pull/2280)
* GLM-DSA architecture indexer cache
* GLM-5.2 vision hack [PR 2283](https://github.com/ikawrakow/ik_llama.cpp/pull/2283)
* String ban function for all completions [PR 1185](https://github.com/ikawrakow/ik_llama.cpp/pull/1185) [PR 1243](https://github.com/ikawrakow/ik_llama.cpp/pull/1243)
* Expiring Logit Bias [PR 1731](https://github.com/ikawrakow/ik_llama.cpp/pull/1731)
* OpenAI `/v1/responses` API endpoint [PR 1184](https://github.com/ikawrakow/ik_llama.cpp/pull/1184)
* OpenAI `/v1/systemone` endpoint for decision models (lev, OpenJev, nimble) [PR 2592](https://github.com/ikawrakow/ik_llama.cpp/pull/2592)
* Function call support [PR 628](https://github.com/ikawrakow/ik_llama.cpp/pull/628)
* jinja template support [PR 677](https://github.com/ikawrakow/ik_llama.cpp/pull/677)
* Webui: New Features for Conversations, Settings, and Chat Messages [PR 618](https://github.com/ikawrakow/ik_llama.cpp/pull/618), MCP [PR 1904](https://github.com/ikawrakow/ik_llama.cpp/pull/1904)
* On-demand tensor reload [PR 1989](https://github.com/ikawrakow/ik_llama.cpp/pull/1989)
* Dynamic control vector management endpoints [PR 1223](https://github.com/ikawrakow/ik_llama.cpp/pull/1223)
* Legacy quants conversion schemes in `convert_hf_to_gguf.py` [PR 449](https://github.com/ikawrakow/ik_llama.cpp/pull/449), `Q6_0` in [PR 483](https://github.com/ikawrakow/ik_llama.cpp/pull/483)
* Adaptive-P Sampler [PR 1100](https://github.com/ikawrakow/ik_llama.cpp/pull/1100) implemented as designed by it's author; supported on Webui
* Multi-modal Vision support in `llama-mtmd-cli` [PR 798](https://github.com/ikawrakow/ik_llama.cpp/pull/798) and in `llama-server` [PR 901](https://github.com/ikawrakow/ik_llama.cpp/pull/901)
* mikupad as an alternative WebUI [PR 558](https://github.com/ikawrakow/ik_llama.cpp/pull/558)
* June 8 2025: Webui updated (legacy still available when `--path ./examples/server/public_legacy` is passed) [PR 481](https://github.com/ikawrakow/ik_llama.cpp/pull/481)
* June 8 2025: RPC improvements [PR 480](https://github.com/ikawrakow/ik_llama.cpp/pull/480)
* June 7 2025: Add an endpoint that lists all the saved prompt caches to server [PR 502](https://github.com/ikawrakow/ik_llama.cpp/pull/502)
* June 6 2025: Make prompt cache saving and restoring MLA aware [PR 497](https://github.com/ikawrakow/ik_llama.cpp/pull/497)
* June 3 2025: Added samplers, XTC [PR 486](https://github.com/ikawrakow/ik_llama.cpp/pull/486), top-n σ [PR 489](https://github.com/ikawrakow/ik_llama.cpp/pull/489).
* May 22 2025: Refactor `iqk_mul_mat.cpp` which speeds up compilation time significantly. [PR 435](https://github.com/ikawrakow/ik_llama.cpp/pull/435)
* May 17 2025: Option to enable or disable the CPU FA kernels [PR 429](https://github.com/ikawrakow/ik_llama.cpp/pull/429).
* May 12 2025: User can now control if/which operations with tensors held in RAM are offloaded to the GPU. See [PR 405](https://github.com/ikawrakow/ik_llama.cpp/pull/405) 
* May 12 2025: Compatibility issues with mainline `llama.cpp` GGUFs for DeepSeek models with MLA enabled were resolved in [PR 394](https://github.com/ikawrakow/ik_llama.cpp/pull/394). The lower prompt processing performance resulting from using `llama.cpp`-style MLA GGUFs was recovered in [PR 409](https://github.com/ikawrakow/ik_llama.cpp/pull/409).
* April 21 2025: ik_llama.cpp builds and runs successfully on Android (using termux), see [PR 336](https://github.com/ikawrakow/ik_llama.cpp/pull/336)
* March 1 2025: Smart Expert Reduction for faster DeepSeek inference [PR 239](https://github.com/ikawrakow/ik_llama.cpp/pull/239) 
* Feb 25 2025: Tensor overrides for better control where model weights are stored (GPU or CPU) [PR 232](https://github.com/ikawrakow/ik_llama.cpp/pull/232)
* Feb 23 2025: `sweep-bench` - better performance benchmarking [PR 225](https://github.com/ikawrakow/ik_llama.cpp/pull/225)
* Feb 19 2025: `Q8_KV` - new type for 8-bit KV-cache quantization [PR 208](https://github.com/ikawrakow/ik_llama.cpp/pull/208)
* March 7 2025: Custom quantization mixes using regular expressions [PR 244](https://github.com/ikawrakow/ik_llama.cpp/pull/244)

### Performance improvements

* Better GPU offload strategy for MoE models when using hybrid HPU/CPU inference, see [PR 520](https://github.com/ikawrakow/ik_llama.cpp/pull/520)
* Much faster rng sampling [PR 1187](https://github.com/ikawrakow/ik_llama.cpp/pull/1187)
* May 13 2025: Better CPU FA performance for DeepSeek-Lite. [PR 410](https://github.com/ikawrakow/ik_llama.cpp/pull/410)
* May 11 2025: Slightly faster flash attention for DeepSeek models on CUDA, along with extending compatibility to Touring or newer GPUs. [PR 408](https://github.com/ikawrakow/ik_llama.cpp/pull/408)
* May 4 2025: Significant token generation performance improvement on CUDA with Flash Attention for GQA models. For details and benchmarks. [PR 370](https://github.com/ikawrakow/ik_llama.cpp/pull/370) 
* April 17 2025: Better CPU Flash Attention token generation performance. [PR 332](https://github.com/ikawrakow/ik_llama.cpp/pull/332)
* April 3 2025: Much faster MoE implementation on Metal. [PR 307](https://github.com/ikawrakow/ik_llama.cpp/pull/307) 
* March 25 2025: Better MoE performance on CUDA [PR 283](https://github.com/ikawrakow/ik_llama.cpp/pull/283)
* March 23 2025: Better batched processing speed for DeepSeek models [PR 282](https://github.com/ikawrakow/ik_llama.cpp/pull/282)
* March 18 2025: Reduce compute buffer size [PR 237](https://github.com/ikawrakow/ik_llama.cpp/pull/237)
* March 10 2025: Better TG performance for MoE models on CUDA [PR 248](https://github.com/ikawrakow/ik_llama.cpp/pull/248)
* Feb 23 2025: Fused FFN ops for faster MoE inference [PR 229](https://github.com/ikawrakow/ik_llama.cpp/pull/229)

### Flash-MLA

* May 7 2025: 🚀 FlashMLA-3 for DeepSeek models on CUDA. [PR 386](https://github.com/ikawrakow/ik_llama.cpp/pull/386). Caveat: Ampere or newer Nvidia GPU required
* March 21 2025: 🚀 FlashMLA-3: fastest CPU-only inference for DeepSeek models [PR 273](https://github.com/ikawrakow/ik_llama.cpp/pull/273)
* March 17 2025: 🚀 FlashMLA-2 performance improvements [PR 253](https://github.com/ikawrakow/ik_llama.cpp/pull/253)
* March 12 2025: Allow `Q8_0` KV cache with FlashMLA-2 on CUDA [PR 265](https://github.com/ikawrakow/ik_llama.cpp/pull/265)
* March 9 2025: 🚀 FlashMLA on CUDA [PR 247](https://github.com/ikawrakow/ik_llama.cpp/pull/247)
* March 8 2025: 🚀 Faster FlashMLA CPU implementation [PR 243](https://github.com/ikawrakow/ik_llama.cpp/pull/243)
* March 3 2025: 🚀 Introducing FlashMLA - MLA with Flash Attention [PR 240](https://github.com/ikawrakow/ik_llama.cpp/pull/240)
* Feb 27 2025: MLA without transposed cache [PR 235](https://github.com/ikawrakow/ik_llama.cpp/pull/235)
* Feb 13 2025: Allow `Q8_0` quantized cache with MLA [PR 206](https://github.com/ikawrakow/ik_llama.cpp/pull/206)
* Feb 11 2025: 🚀 Flash Attention support for DeepSeek models [PR 200](https://github.com/ikawrakow/ik_llama.cpp/pull/200)
* Feb 9 2025: 🚀 MLA for DeepSeek models [PR 188](https://github.com/ikawrakow/ik_llama.cpp/pull/188)

### Fixes

* Fix bug in MMVQ kernel [PR 446](https://github.com/ikawrakow/ik_llama.cpp/pull/446)
* Fix AVX2 implementation of `IQ4_K, IQ4_KS, IQ5_K, IQ6_K` [PR 427](https://github.com/ikawrakow/ik_llama.cpp/pull/427) 
* Fix standard attention on the CPU [PR 421](https://github.com/ikawrakow/ik_llama.cpp/pull/421) 
* Fix imatrix calculation for MLA models [PR 411](https://github.com/ikawrakow/ik_llama.cpp/pull/411)
* Fix new CUDA FA on Touring [PR 413](https://github.com/ikawrakow/ik_llama.cpp/pull/413)
* Fix SER. CPU: [PR 415](https://github.com/ikawrakow/ik_llama.cpp/pull/415) CUDA: [PR 416](https://github.com/ikawrakow/ik_llama.cpp/pull/416)

## Resources

There is no single point of reference describing all new `ik_llama.cpp` features. Pull requests often contain detailed information, so browsing the PRs is often the best way to learn about new features and how to use them. In addition
* [The Wiki page](https://github.com/ikawrakow/ik_llama.cpp/wiki) has performance comparisons to mainline `llama.cpp`
* [This guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) is a good place to start if you came here because of DeepSeek models
* [This discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/266) is about running DeepSeek-V3/R1 on a 16 x 3090 setup
* [This discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/8) describes the new quantization types available in `ik_llama.cpp`

## Testing

### Function Calls Tests

To run the function calls test suite:

```bash
cd build
cmake --build . --target test-function-calls
./bin/test-function-calls
```

The test suite covers parser functionality, streaming, error handling, content cleaning, and server integration. All tests should pass to ensure production readiness.

## Contributing & Upstream Status

Contributions in the form of pull requests, issue submissions (bug reports, feature requests), or general discussions are welcome.

> [!TIP]
> **Upstream Contributions in Progress**: All architectural enhancements, bug fixes, and performance optimizations developed in this fork are currently being modularized and submitted as upstream pull requests to both:
> - [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)
> - [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)

## License

- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
- [server](examples/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
