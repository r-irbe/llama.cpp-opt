# llama.cpp-opt: Run Frontier LLMs on a Single Workstation

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> [!NOTE]
> **llama.cpp-opt** builds on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama.cpp](https://github.com/ggml-org/llama.cpp) to make massive open-weight models practical, fast, and predictable on standard workstation and server hardware.
>
> Instead of requiring an 8-GPU cluster to serve 300B–2.4T Mixture-of-Experts (MoE) models or 1M-token contexts, `llama.cpp-opt` combines a single 24 GB GPU with multi-socket CPU memory and fast NVMe storage. Every subsystem is modular and being contributed upstream; deep architectural specifications, hardware counter audits, and formal Lean 4 proofs live in [`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md).

---

## Why `llama.cpp-opt`? (Practical & Business Value)

| Business & Operational Goal | Typical Bottleneck | What `llama.cpp-opt` Delivers |
| :--- | :--- | :--- |
| **Lower Hardware Costs** | Models like `DeepSeek-V4.1` (385–466 GB) or `Qwen3.8-2.4T` (354 GB) usually require expensive multi-GPU servers or crash with out-of-memory errors. | **Run 300B–2.4T models on 1x 24 GB GPU + system RAM**: Streams cold weights from NVMe in **1.35 seconds** (`--defer-ple --defer-experts`) and supports native compact `MXFP8` and `Q1_0` formats. |
| **Predictable Multi-CPU Speed** | On dual-socket AMD EPYC or Intel Xeon hosts, threads often read weights from the wrong CPU socket, cutting generation speed by up to 40%. | **Consistent multi-socket scaling (`--numa distribute` / `--numa expert`)**: Keeps memory local to each CPU socket and spreads expert weights evenly across nodes (**17.5–25.2 tok/s** on `DeepSeek-V4.1`, **35.7–54.8 tok/s** on `Qwen3.8-Flash-Next`). |
| **1M-Token Context on One GPU** | Long documents and coding sessions quickly exhaust 24 GB of VRAM or slow down as prompt history grows. | **1M context on a single 24 GB GPU**: Skips inactive attention blocks (`--zdd-sparse-kv`), filters irrelevant document chunks (`--evidence-filter`), and tiers cold KV pages to host RAM with zero quality loss ($\Delta\text{PPL} = 0.000000$). |
| **Faster Multi-User & Agent Serving** | When 8–32 users or coding agents send requests sharing a system prompt or repository context, standard servers re-process the shared prompt for each slot. | **Up to 22x faster time-to-first-token & 93.8% less KV memory (`--cohort-prefix-sharing`)**: Clones prefilled prompt caches across concurrent slots—even while the first request is still generating. |
| **Automatic Speed Tuning** | Speculative decoding settings (`-ng`, draft depth) have to be tuned by hand for each model and workload, or they can actually slow generation down. | **1.5x–2.3x faster generation out of the box (`--spec-autotune`)**: Automatically adjusts draft depth online across `DFlash`, `DSpark`, `MTP`, and `N-gram` drafters, and compiles profile-guided binaries (`build-pto`) for up to **2.19x** faster CPU prefill. |
| **Simple Configuration & Resilience** | Launching hybrid CPU+GPU models takes dozens of fragile CLI flags, and sudden memory spikes can crash a running server. | **One-command launcher & automatic OOM recovery**: Switch models and profiles via `scripts/launch.py` or the interactive `ctx-capsule` TUI, with automatic microbatch step-down and shared-memory slab arbitration under load. |

---

## Verified Performance at a Glance

Measured on a single **24 GB NVIDIA RTX 4090** + **Dual AMD EPYC 7453** (56 cores, 8 NUMA nodes, ~968 GiB DDR4) + **4x NVMe RAID-0**. Full methodology, PMU hardware counters, and multi-context tables are in [`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md).

| Model | Size & Type | Context Tested | Prompt Speed (`PP`) | Generation Speed (`TG`) | Key Practical Takeaway |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Qwen3.8-27B-UD-Q4_K_XL`** | `16.4 GiB` Dense + DeltaNet (100% GPU) | `4K` – `1M` | **2,734 tok/s** (`1,944 tok/s` @ `1M`) | **47.8 tok/s** (**99.5 tok/s** w/ `DFlash2`) | **1.94 s time-to-first-token at 1M context**; **2.09x** decode speedup with automatic speculative tuning. |
| **`Qwen3.8-Flash-Next-UD-Q4_K_XL`** | `103.7 GiB` MoE (`177B` total) | `64K` | **455.9 tok/s** (`+37.6%` vs baseline) | **35.7 tok/s** (**54.8 tok/s** w/ `MTP`) | **1.54x** faster generation with `MTP` speculative decoding (`83.1%` draft acceptance). |
| **`Leanstral-1.5-119B-A6B-Q4_K_M`** | `67.3 GiB` MoE (`119B` total) | `64K` | **394.0 tok/s** | **40.2 tok/s** | Fast interactive formal reasoning and code generation on a single GPU + CPU RAM. |
| **`GLM-5.3-Flash-UD-IQ4_XS`** | `146.1 GiB` MoE (`314B` total / `17B` active) | `4K` – `1M` | **159.6 tok/s** | **19.0 tok/s** (`CV = 0.31%`) | Fits **1M context in 18.2 GiB VRAM** with `-66.7%` lower latency jitter. |
| **`DeepSeek-V4.1-Flash-EngramQ8`** | `385.0 GiB` MoE + MLA (`748B` total) | `4K` – `1M` | **188.4 tok/s** (`115.0 tok/s` @ `512K`) | **22.9 tok/s** (**25.2 tok/s** w/ spec) | Starts in **1.35 s** with deferred loading; supports `-np 4` concurrent slots at **512K–1M** context. |
| **`DeepSeek-V4.1-Flash-1017-Q4_K_XL`** | `466.3 GiB` MoE (`188.8 GiB` inline `MXFP8`) | `16K` – `64K` | **148.0 tok/s** | **17.6 tok/s** (`18.0` max @ `-np 4`) | Native `MXFP8` compute + automatic NUMA huge-page preservation across `470 GiB` RAM. |
| **`Qwen3.8-2.4T-A95B-UD-Q1_0`** | `354.2 GiB` MoE (`2.42T` total / `95B` active) | `32K` | **24.2 tok/s** | **3.97 tok/s** (`CV = 0.50%`) | Runs a **2.42-trillion-parameter** model on a single workstation where unpatched loaders fail. |

---

## 5-Minute End-to-End Quickstart (Works on Any Laptop or Single-GPU Desktop)

Want to experience the full `llama.cpp-opt` workflow on a standard machine (16–32 GB RAM, with or without a GPU) in a few minutes? Follow these 5 steps from build to web UI:

### 1. Build (Standard + Profile-Targeted Optimization)

```bash
git clone https://github.com/r-irbe/llama.cpp-opt.git
cd llama.cpp-opt

# Standard build (add -DGGML_CUDA=ON if you have an NVIDIA GPU):
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON
./scripts/build-guard.sh cmake --build build --config Release -j$(nproc)

# (Optional) Build profile-guided (PTO + ThinLTO) binaries into build-pto/ for up to 2.19x faster CPU prefill:
./scripts/pto-build.sh use
```

### 2. Download a Fast MoE or Flint Hybrid GGUF Model

Pick a model that fits your machine's memory budget:
- **Laptop / 8–16 GB RAM (`~2.5 GB`)**: `LFM2-8B-A1B` or `Qwen3-30B-A3B` (compact MoE with only 1B–3B active parameters per token).
- **Single 16–24 GB GPU Workstation (`~16–24 GB`)**: `Recursive-Flint-32B` (`flint` hybrid linear + sliding-window attention, configured in [`scripts/configs/hierarchy/models/recursive_flint.toml`](scripts/configs/hierarchy/models/recursive_flint.toml)) or `Qwen3.8-27B-UD-Q4_K_XL`.

```bash
mkdir -p models

# Option A: Fast compact MoE for laptops & smaller desktops (~5 GB, 1B active params):
./scripts/hf.sh --repo LiquidAI/LFM2-8B-A1B-GGUF --file LFM2-8B-A1B-Q4_K_M.gguf --outdir models

# Option B: Or place Recursive-Flint-32B / any MoE GGUF in models/:
# cp /path/to/Recursive-Flint-32B-Q6_K.gguf models/model.gguf
```

### 3. Convert to `GGUF-X` (`NPMC`) & Score Hardware Alignment

Repack the GGUF into a 2 MiB HugeTLB-aligned `GGUF-X` container (groups MoE expert weights contiguously, removes DRAM bank conflicts, and verifies a **100/100** hardware score):

```bash
# Convert to GGUF-X and automatically validate 100/100 alignment score:
python3 scripts/gguf-pack-npmc.py models/LFM2-8B-A1B-Q4_K_M.gguf \
  -o models/model-ggufx.gguf \
  --embed-superblock \
  --validate

# Inspect hardware fit, KV cache scaling up to 256K, and export an autotuner profile:
python3 scripts/gguf-lint.py models/model-ggufx.gguf --mab-profile results/model_profile.json
```

### 4. Auto-Tune Serving Parameters for Your Hardware

Run the Multi-Armed Bandit Pareto tuner to discover the fastest thread count, batch size, KV quantization, and speculative decoding settings for your exact CPU/GPU/RAM:

```bash
# Fast hardware-calibrated simulation across 4K, 16K, and 64K contexts:
python3 scripts/mab_pareto_tuner.py \
  --mab-profile results/model_profile.json \
  --model-path models/model-ggufx.gguf \
  --context-lengths 4096,16384,65536 \
  --trials 6 \
  --output-md results/my_optimal_config.md

# Or run live binary benchmarks (--live-bench) with ./build/bin/llama-cli:
python3 scripts/mab_pareto_tuner.py \
  --mab-profile results/model_profile.json \
  --model-path models/model-ggufx.gguf \
  --context-lengths 4096,16384 \
  --trials 4 \
  --live-bench \
  --output-md results/my_live_optimal_config.md
```

*(The generated Markdown report gives you a ready-to-copy `llama-server` command tailored to your machine.)*

### 5. Launch via `llama-server` Web UI or Interactive `ctx-capsule` TUI

Start the server with online speculative autotuning and ZDD sparse KV enabled, then open the built-in **llama.cpp WebUI** at **[http://127.0.0.1:8080](http://127.0.0.1:8080)** (verified end-to-end: `100/100 [OPTIMAL]` GGUF-X alignment score, **229.9 tok/s** live MAB-tuned generation, and **387.6 tok/s** speculative `llama-server` generation with `22.4 ms` TTFT):

```bash
# Launch directly with the built-in WebUI + OpenAI-compatible API on port 8080:
./build/bin/llama-server \
  -m models/model-ggufx.gguf \
  -c 32768 -ngl 99 \
  -ctk q8_0 -ctv q8_0 -fa on \
  --zdd-sparse-kv \
  --spec-type ngram-mod:n_max=6,n_min=2 \
  --spec-autotune \
  --host 127.0.0.1 --port 8080
```

Or use the declarative launcher / interactive `ctx-capsule` TUI to manage profiles (`recursive_flint`, `qwen38_27b`, `deepseek-v4.1-1017-ud-q4kxl`, etc.):

```bash
# List all pre-configured model & hardware profiles:
python3 scripts/launch.py --list

# Launch the Recursive-Flint profile (or preview flags with --dry-run):
python3 scripts/launch.py recursive_flint --dry-run

# Open the interactive ctx-capsule TUI (or render a one-shot snapshot with --dry-run):
python3 scripts/launch.py --capsule
python3 scripts/launch.py --capsule --dry-run
```

---

## Documentation & Deep-Dive Guides

All low-level systems architecture, benchmark matrices, and formal verification proofs are organized in `docs/`:

| Document | What It Covers |
| :--- | :--- |
| [`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md) | **Technical Deep Dive**: Full benchmark matrices, `perf stat` hardware counters, Lean 4 formal verification proofs (`PolyGraph`, `SlabArenaProtocol`, `FlashAttnZddSkip`, `PrismTernaryCodec`), and academic references. |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Practical guide to 512k–1M contexts, `--zdd-sparse-kv`, `--evidence-filter`, `ctx-capsule` TUI, and `build-pto`. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | 10-pillar engineering roadmap and completed subsystem milestones. |
| [`docs/numa.md`](docs/numa.md) | Configuring multi-socket CPUs (`--numa distribute`, `--numa expert`, `--numa split`). |
| [`docs/speculative.md`](docs/speculative.md) | Speculative decoding (`DFlash`, `DSpark`, `MTP`, `ngram-mod`) and `--spec-autotune`. |
| [`docs/kv-cache.md`](docs/kv-cache.md) | KV cache quantization, SWA compression (`--swa-compress`), and memory sizing. |
| [`docs/build.md`](docs/build.md) & [`docs/parameters.md`](docs/parameters.md) | Build instructions (Linux, Windows, macOS, AVX-512, CUDA) and full CLI flag reference. |

### Modular Upstream Pull Request Branches

Each capability is maintained in a focused branch for upstream review:
[`pr/numa-split`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/numa-split) · [`pr/kv-zdd-sparse`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/kv-zdd-sparse) · [`pr/cohort-prefix-sharing`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cohort-prefix-sharing) · [`pr/deepseek-v4-fp8-engram`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/deepseek-v4-fp8-engram) · [`pr/model-glm5next`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/model-glm5next) · [`pr/speculative-universal-autotune`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/speculative-universal-autotune) · [`pr/cuda-ada-optimizations`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/cuda-ada-optimizations) · [`pr/io-uring-dma`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/io-uring-dma) · [`pr/lockfree-server-queue`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/lockfree-server-queue) · [`pr/security-multi-tenant-isolation`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/security-multi-tenant-isolation) · [`pr/benchmarks-tooling-readme`](https://github.com/r-irbe/llama.cpp-opt/tree/pr/benchmarks-tooling-readme)

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

Start the server directly:

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096
```

```
./build/bin/llama-server --model /my_local_files/gguf/Qwen_Qwen3-0.6B-IQ4_NL.gguf --ctx-size 4096 -ngl 999
```

That's all! Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in Browser and start chatting, or use the available API endpoints in your program/harness. (For `llama.cpp-opt` declarative model profiles, `ctx-capsule` TUI, and `build-pto` commands, see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).)

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
