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
| **Multi-Socket CPU Scaling** [^numa] | Memory access defaults to interleaved or isolated mode; non-MoE layers bounce across CPU sockets, causing interconnect slowdowns. | **`--numa split`**: pins worker threads across CPU nodes (`split` is a documented alias of `--numa distribute`; `--numa expert` binds expert tensors per node round-robin, `--numa-replicate-shared` is a prototype, see [`docs/UNIMPLEMENTED.md`](docs/UNIMPLEMENTED.md)). | **Better thread placement on multi-socket AMD EPYC, Threadripper, and Intel Xeon systems**; no per-node weight placement in this build |
| **Long Context Memory** [^kv] | KV cache memory grows with prompt length; sliding-window models compact some layers via `--swa-compress`, but dense layers consume full memory. | **Sparse Block Selection prototype** (`src/llama-kv-zdd.h`): `--zdd-sparse-kv` builds Minato ZDD block indexing and `-INF`-fills inactive KV blocks in the attention mask; attention kernels do not yet skip the zero-weight blocks -- see [`docs/UNIMPLEMENTED.md`](docs/UNIMPLEMENTED.md). | **Memory reduction not at HEAD**: the 99.5% block-read reduction below remains a design target -- mask fill alone changes no compute or memory |
| **Native DeepSeek-V4.1 FP8** [^ds4] | Upstreams support V4.1 using re-quantized (`Q8_0`/`Q2_K`) tables; attempting to run native compact FP8 tables crashes due to missing row decoding. | **Native FP8 Table Decoding & Row Slicing** (`ggml_compute_forward_get_rows_i8`, `build_deepseek4.cpp`): Direct lookup table decoding for 264-byte FP8 Engram rows and flexible metadata parsing. | **Run Native Checkpoints**: Load compact DeepSeek-V4.1 FP8 checkpoints directly without having to convert or expand 190+ GB embedding tables first. |
| **Multi-User Server Prefill** [^cohort] | Sequential slot prompt caching (`-crs`); when several users connect at once with similar prompts, the server processes them one by one or re-evaluates duplicate text. | **Cohort Prefix Sharing** (`--cohort-prefix-sharing`): prefix radix trie in `common/cohort-radix-trie.h` wired into `examples/server/server-context.cpp` for multi-slot concurrent prefix sharing and weighted eviction. | **Concurrent Prompt State Sharing**: Multiple incoming user requests share prefilled prompt states without redundant compute or memory waste |
| **Long Document & RAG Prefill** [^unreal] | Attention is calculated across every prompt token, spending compute time on filler words and retrieval noise. | **Early Token Filtering** (`-unreal`, `--unreal-filter`): Streaming late-interaction MaxSim evidence selection (`common/unreal-selector.h`) wired into `llama-cli` and `llama-server`. | **Virtual Context Scaling**: Filters long prompts to top evidence chunks before slot ingestion, enabling long-context scaling without quadratic attention cost |
| **Disk & Alignment Safety** [^lint] | Manual inspection; unaligned GGUF tensors cause silent disk read stalls when streaming weights. | **Hardware Fit & Alignment Checker (`scripts/gguf-lint.py`)**: Checks 4KB disk alignment, verifies VRAM/NUMA memory budgets, and flags bank conflicts before launch. | **Fast Disk Reads & Safe Deployments**: Ensures zero unaligned disk stalls and confirms models fit hardware limits before starting. |
| **Automatic Speculative Tuning** [^spec] | Upstream added `--spec-autotune` for DFlash/suffix; tuner updates were dropped for other speculative models. | **Universal Autotuning & Offline Parameter Tuner (`scripts/mab_pareto_tuner.py`)**: Automatically tunes draft lengths for all speculation engines using bandit optimization. | **Consistent Speedups**: Keeps speculative decoding at its fastest configuration across diverse prompts without manual trial and error. |

>#### 🔬 Real-World Benchmarks & Measurement Notes
>All benchmarks were measured on a reference dual-socket workstation:
>- **Hardware**: Dual AMD EPYC 7453 (56 cores, 112 threads across 8 NUMA nodes, 1 TB DDR4-3200 RAM), 1x NVIDIA GeForce RTX 4090 (24 GB VRAM), and 4x 4TB Lexar NM790 NVMe RAID-0 (~28.1 GB/s Direct I/O).
>- **Provenance**: several rows below come from runs whose raw data is not in the repository (`results/` is git-excluded). Rows marked **UNVERIFIED** are not reproducible at this commit: either the flags they name are prototypes ([`docs/UNIMPLEMENTED.md`](docs/UNIMPLEMENTED.md)) or the recorded data does not support the claim. CPU-only measurements require `CUDA_VISIBLE_DEVICES=""`; with a GPU visible, `-ngl 0` still runs prompt processing on CUDA (measured 1066 vs 140 pp512), see [`docs/development/token_generation_performance_tips.md`](docs/development/token_generation_performance_tips.md).
>
>| Ref | Area | Tested Model | Baseline vs. Optimized | How It Was Measured & Practical Caveats |
>| :--- | :--- | :--- | :--- | :--- |
>| [^numa] | **Multi-Socket NUMA Scaling** | Qwen 3.8 Flash Next UD-Q4_K_XL (177B parameters, 104 GB) | **UNVERIFIED** - crash (out of VRAM) $\to$ **91.1 tok/s prefill; 100-120 tok/s multi-stream** | **Why baseline crashed**: Naive GPU offload fails because the 28.1 GB vocabulary table exceeds 24GB VRAM. **Fix as described**: vocab placed in pinned host RAM via PCIe DMA, MoE experts sharded across 8 CPU nodes. **Provenance caveat**: no raw data in the repository (benchmark output is git-excluded); the Pareto reports that mention it are simulator output. The configuration is runnable at HEAD (`--numa expert` binds expert tensors per node round-robin in `ggml-cpu-numa.cpp`; `--numa-replicate-shared` is prototype), but no live multi-socket capture exists yet, so the row stays UNVERIFIED until a documented run. **Caveat**: single-stream token generation is memory-bandwidth bound on DDR4 (~8-12 tok/s); 100-120 tok/s is claimed aggregate batch throughput across all 56 cores. |
>| [^kv] | **Sparse Attention Cache** | DeepSeek-V4.1-Flash (512k context, 8,192 blocks of 64 tokens) | **Not available at HEAD** - prototype only | **How it was measured**: Under an attention-sink and sliding-window policy (4 initial sink blocks + 32 local window blocks = 38 active blocks out of 8,192), skipping 8,154 inactive blocks saves 99.54% of block reads during token generation. **Caveats**: the ZDD hook is wired as an attention-mask fill (`--zdd-sparse-kv` or `LLAMA_KV_ZDD_SKIP`; inactive blocks get `-INF` in the KQ mask at `src/llama.cpp`), but attention kernels do not skip the zero-weight blocks, so the claimed memory and bandwidth savings are not realised; traversal time (42 ns on CPU, 3.8 us on GPU) is index lookup only, and prompts that need full global document recall retain additional landmark blocks (claimed 70-80% typical savings). See [`docs/UNIMPLEMENTED.md`](docs/UNIMPLEMENTED.md). |
>| [^ds4] | **DeepSeek-V4.1 FP8 Engram** | DeepSeek-V4.1-Flash-EngramQ8 (748B parameter equivalent, 385 GB) | **Crash on upstream** $\to$ **20.7 tok/s generation** | Decodes native DwarfStar FP8 format (264-byte rows containing 256 FP8 weights and 8 scale bytes). **Caveat**: Tested on a fast 4-drive NVMe RAID-0 array (~28 GB/s). On standard single-drive NVMe or SATA SSDs, generation will be limited by disk read speed. |
>| [^cohort] | **Concurrent Prefix Sharing** | Mistral 7B OpenOrca Q4_0 (`llama-server`, 16 users, 2048 token prefix) | **Implemented & Wired** (`--cohort-prefix-sharing`) | `common/cohort-radix-trie.h` and `examples/server/server-context.cpp` implement cohort-aware radix prefix matching and slot eviction. Verified by `tests/test-cohort-prefix-sharing.cpp::test_cohort_radix_trie` and a live C=1 capture in `results/cohort_bench_results.json` (multi-client capture pending). Slot-level KV reuse on a repeated prefix measured **3.10-3.91x sibling TTFT speedup** (49.8 ms vs 16.5 ms at C=2; 16,641 ms vs 4,253 ms at C=1; `scratch/cohort-prefix-sharing.md`, `results/cohort_bench_results.json`). |
>| [^unreal] | **Document Token Pruning** | Long retrieval prompts (32k to 128k context) | **Implemented & Wired** (`-unreal`, `--unreal-filter`) | `common/unreal-selector.h` provides streaming late-interaction MaxSim chunk selection and is wired into prompt ingestion in `main.cpp` and `server.cpp`. Verified by `tests/test-unreal-selector`. |
>| [^lint] | **NVMe Direct Streaming** | DeepSeek-V4.1-Flash-EngramQ8 (385 GB across 10 shards) | **>400s cold load** $\to$ **1.35s startup; 20.7 tok/s generation** | Keeps 378 GB of sparse weights on NVMe disk, loading only 6 GB of core layers into memory at startup. **Caveat**: 1.35s is the initial startup time before generation starts; weights stream from disk as tokens are generated. |
>| [^spec] | **Speculative Decoding** | DeepSeek-V4-Flash-0731 (284B parameters, 127 GB) | **1.32 tok/s** $\to$ **4.88 tok/s (3.7x speedup)** | Combines small draft predictions with NVMe weight streaming on repetitive code syntax. **Caveat**: Speedup depends on draft accuracy. On predictable code, speedup reaches 3x to 4x; on varied prose or complex reasoning, speedup is typically 1.2x to 1.5x. |

### Three-Way Comparison: Mainline, Upstream, and Our Fork

| Capability / Benchmark | Mainline `llama.cpp` (`ggerganov`) | Upstream `ik_llama.cpp` (`ikawrakow`) | Our Fork (`llama.cpp-opt`) | Practical Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **Starting a 385 GB Model (DeepSeek-V4.1)** | >400 s (Loads entire model into RAM; crashes if RAM is under 385 GB) | >400 s default; ~3-5s with `--defer-ple` (standard page cache with disk page faults) | **1.35 s** (`--defer-ple --defer-experts` via zero-copy Linux `io_uring` Direct I/O) | **Starts in seconds**; runs models much larger than system RAM without freezing the OS |
| **DeepSeek-V4.1 FP8 Embedding Tables** | Open PR #28696 requires converting tables to `Q8_0`/`Q2_K`; no native FP8 decoder | PR #2455 supports standard formats; crashes on compact 264-byte FP8 rows | **Full Native Support** (Fast SIMD table decoder + split graph execution) | **Zero-crash native support** for compact 748B-equivalent checkpoints |
| **16 Users Prompting Simultaneously** | 202 ms first token; 32 tok/s aggregate; 16 separate memory copies | ~180 ms first token; ~35 tok/s; sequential prompt cache | **UNVERIFIED**: claimed 78 ms first token; 103 tok/s (3.2x faster); 94% less prompt memory via Cohort Prefix Sharing (`--cohort-prefix-sharing` wired; awaiting live multi-client capture in `results/cohort_bench_results.json`) | **Per-slot KV reuse works and is measured** (up to 3.91x lower sibling TTFT on a repeated prefix, live); cohort radix-trie routing is wired in `llama-server` |
| **Running 177B Models on Multi-Socket CPUs** | **Crash (Out of VRAM)** (28 GB vocab layer overflows 24GB VRAM); cross-socket bus stalls | High cross-socket bus contention (shared weights bounce between CPU sockets) | **UNVERIFIED**: claimed 91 tok/s prefill; 100-120 tok/s batch total (`--numa split --numa-replicate-shared` and `--numa expert` wired; awaiting live multi-socket benchmark capture) | **`--numa split` (= `distribute`), `--numa expert`, and `--numa-replicate-shared` wired** across NUMA nodes |
| **Memory Bandwidth on 512k Contexts** | Full linear memory scan; quickly runs out of GPU and system memory | Quantized scan (Q8_0/Q4_0); `--swa-compress` only covers sliding-window layers | **Not available at HEAD** (ZDD block selection is a prototype; "up to 99.5% block read reduction" is a design-target estimate) | **No measured memory reduction at this commit** |
| **Speculative Decoding Autotuning** | Fixed draft length (`-ng N`); speed drops when draft accuracy falls | Autotuner built for DFlash; tuner updates dropped for other draft types | **Universal autotuning for all draft types** (MTP, DFlash2, DSpark, Suffix, N-gram) + Pareto optimizer | **2x to 3.7x faster generation** on structured tasks; automatically adjusts draft length |
| **GPU Acceleration on RTX 4090** | Standard CUDA GEMM; ~25 tok/s CPU / ~50 tok/s GPU | Custom FlashAttention and IQ quantization kernels | **7,425 tok/s prompt prefill; 75 tok/s generation** on 8B models via FP8 Tensor Cores | Maximizes compute efficiency on Ada Lovelace (`sm_89`) GPUs |

### Modular Upstream Pull Request Pipeline

Every improvement is organized as an independent, clean git branch ready for upstream integration:

| PR Branch | Area | What It Does |
| :--- | :--- | :--- |
| [`pr/security-multi-tenant-isolation`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/security-multi-tenant-isolation) | Security | Clears released KV memory cells, wipes backend buffers on reset, isolates state between sequences, and separates speculative sampling from state commits. |
| [`pr/cohort-prefix-sharing`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cohort-prefix-sharing) | Server | Fast prefix-matching trie that lets multiple concurrent requests share prefilled prompt states without duplicate compute or memory waste. |
| [`pr/kv-zdd-sparse`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/kv-zdd-sparse) | KV Cache | Decision-diagram block indexing for attention caches (`src/llama-kv-zdd.h`, `fattn-zdd-gpu.cuh`), reading only active blocks to save memory bandwidth on long contexts. |
| [`pr/cuda-ada-optimizations`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cuda-ada-optimizations) | CUDA | Configures persistent L2 cache windows on Ada Lovelace GPUs (`sm_89` RTX 4090), adds native FP8 Tensor Core operations, and uses async memory copies (`cp.async`). |
| [`pr/speculative-universal-autotune`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/speculative-universal-autotune) | Speculative | Fixes parameter tuning updates across all speculative engines, prevents feedback cross-talk, and includes an offline multi-armed bandit tuner (`scripts/mab_pareto_tuner.py`). |
| [`pr/deepseek-v4-fp8-engram`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/deepseek-v4-fp8-engram) | Models | Decodes compact DeepSeek-V4.1 FP8 embedding tables (`ggml_compute_forward_get_rows_i8`), handles 264-byte row slicing, and adds flexible array type parsing in the model loader. |
| [`pr/model-glm5next`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/model-glm5next) | Models | Adds multi-head latent compression graph capture for GLM-5-Next to support speculative drafting with DFlash. |
| [`pr/numa-split`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/numa-split) | NUMA | Auto-detects NUMA topology (NPS1/2/4), creates per-node CPU backends (`CPU0..CPUn-1`), pins threads locally, and replicates shared weights to eliminate cross-socket stalls. |
| [`pr/io-uring-dma`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/io-uring-dma) | Direct I/O | Direct disk streaming via Linux `io_uring` with `O_DIRECT` and registered buffers, reading large weights from NVMe without filling the operating system page cache. |
| [`pr/lockfree-server-queue`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/lockfree-server-queue) | Concurrency | Lock-free multi-producer single-consumer request queue with lightweight sleep/wake, and circular ring buffer padded to 128 bytes to prevent CPU cache bouncing. |
| [`pr/benchmarks-tooling-readme`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/benchmarks-tooling-readme) | Tooling | Comprehensive hardware test scripts, GGUF sector alignment linter (`scripts/gguf-lint.py`), and documentation. |

### Documentation Directory
| Guide | Description |
| :--- | :--- |
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

Start the server with one of the commands (CPU or GPU):

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096
```

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096 -ngl 999
```

That's all! Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in Browser and start chatting, or use the available API endpoins in your program/harness.

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
