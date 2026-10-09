# Comprehensive Benchmarking and Testing Plan: High-Performance Inference Across Heterogeneous Architectures

**Target Repository**: [`ikawrakow/ik_llama.cpp`](../README.md)  
**Target Branch**: `feat/numa-split`  
**Engines Evaluated**:
1. **Target**: `ik_llama.cpp` (`feat/numa-split` with `--numa split`, wait-free 128B SPSC, permanent CUDA graphs, cohort prefix sharing `--cohort-prefix-sharing`, recursive storage NUMA discovery, wide-row column slicing, and THP alignment)
2. **Baseline 1**: Vanilla `ik_llama.cpp` (`master` branch upstream by I. Kawrakow)
3. **Baseline 2**: Upstream mainline `llama.cpp` (`ggerganov/llama.cpp` latest release / master)

**Hardware Platforms**:
1. **Current Workstation**: Dual AMD EPYC 7453 (56 physical cores / 112 threads across 8 NUMA nodes in NPS4 mode, ~1 TB DDR4 RAM), 1x NVIDIA GeForce RTX 4090 (24 GB VRAM, `sm_89`), 4x 4TB Lexar NM790 NVMe RAID-0 on Node 2 PCIe root complex (~28 GB/s), 2x WD_BLACK SN850X NVMe RAID-1 on Node 0.
2. **Apple Silicon Platform**: MacBook Pro with Apple M2 Max (12-core CPU: 8 performance + 4 efficiency cores, 30/38-core GPU, unified LPDDR5 memory with 400 GB/s bandwidth, Metal compute framework).

---

## 1. Executive Summary & Core Objectives

The goal of this benchmarking and testing plan is to provide an empirical, reproducible, and mathematically rigorous evaluation framework comparing `ik_llama.cpp` on the `feat/numa-split` branch against both vanilla upstream `ik_llama.cpp` (`master`) and canonical mainline `llama.cpp` (`ggerganov/llama.cpp`).

Modern multi-hundred-billion parameter Large Language Models (LLMs) and Mixture of Experts (MoE) architectures—ranging from dense models like Llama 3.1/3.3 70B and Qwen 2.5 72B to massive sparse MoE models like DeepSeek-V3/R1 (671B total, 37B active) and hybrid recurrent architectures—present extreme memory bandwidth, interconnect latency, and scheduling bottlenecks.

To eliminate bias and ensure fair evaluation, this plan mandates a **Dual-Track Evaluation Strategy**:
1. **Iso-Parameter Track**: Identical, standardized parameters (threads, batch sizes, micro-batch sizes, GPU offload layers, context lengths, and KV cache quantizations) across all three engines. This isolates pure architectural execution differences, compiler codegen, kernel scheduling, and memory allocation overhead.
2. **Pareto-Optimized Track**: Independently autotuned, maximum-performance parameter searches for *each* engine. Because each engine's architecture may reach peak performance under fundamentally different configurations (e.g. `--numa split` with CCX thread pinning vs. `numactl --interleave=all` with global thread pools, or `-sm graph` vs. `-sm layer`), each engine is allowed to compete at its individual peak.

---

## 2. Hardware Platforms & Microarchitectural Profiles

### 2.1 Platform A: Enterprise Workstation (Dual AMD EPYC 7453 + RTX 4090)

```
                           ┌────────────────────────────────────────────────────────┐
                           │          AMD EPYC 7453 Socket 0 (Nodes 0..3)           │
                           │  28 Physical Cores (56 Threads) across 4 CCX Clusters  │
                           │  8 Memory Channels DDR4-3200 ECC (~192 GB/s Bandwidth) │
                           └───────────────┬────────────────────────┬───────────────┘
                                           │                        │
               ┌───────────────────────────┴────┐        ┌──────────┴────────────────────────┐
               ▼                                ▼        ▼                                   ▼
        Node 0 (SN850X RAID1)            Node 1 (RTX 4090)  Node 2 (4x Lexar NM790 RAID0)  Node 3 (DDR4 Pool)
        PCIe Root Complex                PCIe 4.0 x16       PCIe 4.0 x16 Root Complex      Local DDR4
        OS & System Pool                 24 GB GDDR6X       ~28 GB/s Direct DMA Direct I/O
                                                ▲
                                                │ 4x 16-Lane xGMI-3 Interconnect Links
                                                │ (~64 GB/s Bidirectional, ~1.9x Latency Penalty)
                                                ▼
                           ┌────────────────────────────────────────────────────────┐
                           │          AMD EPYC 7453 Socket 1 (Nodes 4..7)           │
                           │  28 Physical Cores (56 Threads) across 4 CCX Clusters  │
                           │  8 Memory Channels DDR4-3200 ECC (~192 GB/s Bandwidth) │
                           └────────────────────────────────────────────────────────┘
```

#### Detailed Hardware Parameters:
- **Processors**: 2x AMD EPYC 7453 (Zen 3 "Milan" microarchitecture, 7nm, 28 physical cores / 56 threads per socket = 56 physical cores / 112 threads total).
- **Cache Hierarchy**: 32 KiB L1D + 32 KiB L1I per core, 512 KiB private L2 per core, 32 MiB shared L3 cache per 7-core / 8-core CCX cluster (4 CCX clusters per socket = 256 MiB total L3 cache).
- **NUMA Topology**: Configured in **NPS4 mode** (NUMA Per Socket = 4). Sockets are divided into quadrants:
  - **Socket 0**: Node 0 (CPUs 0–6, 56–62), Node 1 (CPUs 7–13, 63–69), Node 2 (CPUs 14–20, 70–76), Node 3 (CPUs 21–27, 77–83).
  - **Socket 1**: Node 4 (CPUs 28–34, 84–90), Node 5 (CPUs 35–41, 91–97), Node 6 (CPUs 42–48, 98–104), Node 7 (CPUs 49–55, 105–111).
- **Host RAM**: ~1 TiB DDR4-3200 ECC registered memory across 16 memory channels (8 channels per socket, ~384 GB/s peak aggregate theoretical DDR bandwidth).
- **Discrete GPU**: 1x NVIDIA GeForce RTX 4090:
  - Architecture: Ada Lovelace (`sm_89`), 16,384 FP32 CUDA cores, 512 Fourth-Generation Tensor Cores.
  - VRAM: 24 GB GDDR6X on a 384-bit memory bus (1,008 GB/s bandwidth).
  - Interconnect: PCIe 4.0 x16 (~31.5 GB/s bidirectional), attached directly to **NUMA Node 1** PCIe root complex.
- **High-Speed Storage Subsystem**:
  - **Storage Pool 1 (Direct DMA Model Storage)**: 4x 4TB Lexar NM790 NVMe M.2 SSDs in Software RAID-0 (`/dev/md0` via Linux mdadm), hosted on a PCIe 4.0 x16 host carrier card wired directly to **NUMA Node 2** PCIe root complex (~28 GB/s aggregate sequential read bandwidth, 4K direct block I/O).
  - **Storage Pool 2 (System Storage)**: 2x WD_BLACK SN850X NVMe SSDs in RAID-1 attached to NUMA Node 0 root complex (~7 GB/s read).
  - **Storage Pool 3 (Cold Archive)**: 1x 22TB Platter HDD on Node 0 SATA controller (~260 MB/s sequential).
- **Memory Pages**: Transparent Huge Pages (THP) enabled with 1 GB HugeTLB pages mounted at `/dev/hugepages`.

---

### 2.2 Platform B: Apple Silicon (MacBook Pro Apple M2 Max)

```
                     ┌────────────────────────────────────────────────────────┐
                     │          Apple M2 Max System-on-Chip (SoC)             │
                     ├───────────────────────────┬────────────────────────────┤
                     │ 12-Core ARM64 CPU         │ 30 / 38-Core Apple GPU     │
                     │ • 8 Performance Cores     │ • Up to 13.5 TFLOPS FP32   │
                     │ • 4 Efficiency Cores      │ • 16-Core Neural Engine    │
                     └─────────────┬─────────────┴─────────────┬──────────────┘
                                   │                           │
                                   ▼                           ▼
                     ┌────────────────────────────────────────────────────────┐
                     │             Unified Memory Architecture (UMA)          │
                     │        64 GB / 96 GB Unified LPDDR5-6400 (512-bit)     │
                     │                 400 GB/s Shared Bandwidth              │
                     │         Zero-Copy Direct GPU Pointers (Shared)         │
                     └────────────────────────────────────────────────────────┘
```

#### Detailed Hardware Parameters:
- **Processor**: Apple M2 Max SoC (5nm process).
- **CPU Cores**: 12 cores total:
  - 8 Performance "Avalanche" cores operating up to 3.67 GHz.
  - 4 Efficiency "Blizzard" cores operating up to 2.42 GHz.
- **GPU Cores**: 30-core or 38-core GPU running Metal 3.1 compute shaders and SIMD-group matrix intrinsics.
- **Unified Memory Subsystem**: 64 GB or 96 GB unified LPDDR5 memory across a 512-bit memory interface delivering **400 GB/s bandwidth**. The entire address space is zero-copy accessible by both CPU threads and GPU shaders (`MTLResourceStorageModeShared`).
- **Software Compute Stack**: macOS Sonoma / Sequoia, Apple Metal framework (`ggml-metal.metal`), Metal Performance Shaders (MPSGraph).

---

## 3. The Three Inference Engines Under Comparison

```
┌───────────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
│ Target Engine: ik_llama.cpp           │ Baseline 1: Vanilla ik_llama.cpp      │ Baseline 2: Upstream llama.cpp        │
│ (`feat/numa-split`)                   │ (`master` upstream by I. Kawrakow)    │ (`ggerganov/llama.cpp` master/latest) │
├───────────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ • `--numa split` (independent nodes)  │ • `--numa distribute` / `numactl`     │ • `--numa distribute` / `numactl`     │
│ • Dynamic socket probing & topology   │ • Flat UMA/interleaved CPU buffers    │ • Standard `ggml-cpu` backends        │
│ • 128B SPSC wait-free ring buffer     │ • Mutex / futex speculative sync      │ • Mutex / condition variable sync     │
│ • Permanent CUDA graph reuse          │ • 4-step CUDA graph tripwire          │ • Standard CUDA graph capture         │
│ • Cohort Prefix Sharing (`--cohort-pps`)│ • Independent slot prefix prefill    │ • Independent slot prefix prefill     │
│ • Storage NUMA root complex discovery │ • Fallback to Node 0 for virtual MD   │ • No virtual block device discovery   │
│ • Wide-row column slicing & THP align │ • Standard IQ quantizations           │ • Standard GGUF quantization formats  │
└───────────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 4. Consequential Open-Source Models Matrix & MoE Offload Architecture

### 4.1 The Complete 19-Model Architecture Matrix

The benchmarking matrix covers 19 consequential open-source architectures, spanning dense models, sparse MoE models, and hybrid recurrent architectures:

| # | Family | Model Architecture | Parameters (Total / Active) | Context Window | Key Quantizations Evaluated | Architectural Highlights | Platform A/B Offload Strategy |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Dense** | **Llama 3.3 / 3.1 70B** | 70.6 B / 70.6 B | 128 k | `UD-Q4_K_XL`, `IQ4_K`, `IQ3_XXS` | GQA (8 KV heads), RoPE $\theta=500\text{k}$, 16k vocab. | A: `-ngl 24` (14GB VRAM) + 8 NUMA nodes; B: `-ngl 999` (UMA) |
| 2 | **Dense** | **Llama 3.1 8B** | 8.0 B / 8.0 B | 128 k | `Q8_0`, `Q4_K_M`, `IQ4_NL` | High single-batch decode baseline, standard MHA/GQA. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |
| 3 | **Dense** | **Qwen 2.5 72B** | 72.7 B / 72.7 B | 128 k | `UD-Q4_K_XL`, `IQ4_K`, `IQ3_XXS` | GQA (8 KV heads), 152k vocabulary, SwiGLU activations. | A: `-ngl 22` + 8 NUMA nodes; B: `-ngl 999` (UMA) |
| 4 | **Dense** | **Qwen 2.5 32B** | 32.5 B / 32.5 B | 128 k | `Q8_0`, `Q4_K_M`, `IQ4_NL` | Intermediate dense scaling anchor, high reasoning density. | A: `-ngl 999` (Fits in 24 GB VRAM); B: `-ngl 999` (UMA) |
| 5 | **Dense** | **Qwen 2.5 7B** | 7.6 B / 7.6 B | 128 k | `Q8_0`, `Q4_K_M` | Ultra-fast interactive latency anchor, edge deployment. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |
| 6 | **Dense** | **Gemma 2 27B** | 27.2 B / 27.2 B | 8 k | `Q8_0`, `Q4_K_M`, `IQ4_NL` | Sliding window local + global attention, logit soft-capping. | A: `-ngl 999` (Fits in 24 GB VRAM); B: `-ngl 999` (UMA) |
| 7 | **Dense** | **Gemma 2 9B** | 9.2 B / 9.2 B | 8 k | `Q8_0`, `Q4_K_M` | Compact dual-attention reasoning model, low KV cache. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |
| 8 | **Dense** | **Mistral Large 2 (123B)**| 123.0 B / 123.0 B | 128 k | `IQ4_K`, `IQ3_XXS`, `IQ2_XXS` | 128k context, Tekken tokenizer (131k vocab), dense flagship. | A: `-ngl 16` + 8 NUMA nodes; B: `-ngl 999` (96GB UMA) |
| 9 | **Dense** | **Mistral NeMo 12B** | 12.2 B / 12.2 B | 128 k | `Q8_0`, `Q4_K_M` | Compact dense reasoning benchmark, 128k native context. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |
| 10 | **Dense** | **Phi-4 14B** | 14.7 B / 14.7 B | 16 k | `Q8_0`, `Q4_K_M`, `IQ4_NL` | Synthetic-trained math/logic engine, high MMLU density. | A: `-ngl 999` (Fits in 24 GB VRAM); B: `-ngl 999` (UMA) |
| 11 | **Sparse MoE** | **DeepSeek-V3 / R1** | 671 B / 37 B | 128 k | `UD-IQ2_XXS`, `IQ1_S`, `UD-Q4_K_XL` | 256 routed experts + 1 shared expert, Multi-Head Latent Attention (MLA). | A: Attention on GPU, 256 experts across 8 NUMA nodes; B: CPU offload |
| 12 | **Sparse MoE** | **DeepSeek-V2.5 / V4** | 236 B / 21 B | 128 k | `IQ3_XXS`, `IQ2_XXS`, `Q4_K_M` | MLA attention, 160 fine-grained experts, shared expert isolation. | A: Attention on GPU, experts across 8 NUMA nodes; B: CPU/GPU hybrid |
| 13 | **Sparse MoE** | **Qwen 2.5-MoE 14B** | 14.3 B / 2.7 B | 32 k | `Q8_0`, `Q4_K_M`, `IQ4_NL` | 64 fine-grained experts, 8 active per token, shared expert. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |
| 14 | **Sparse MoE** | **Mixtral 8x22B** | 141 B / 39 B | 64 k | `IQ4_K`, `IQ3_XXS`, `IQ2_XXS` | Classic Top-2 MoE, 8 routed experts per layer, GQA. | A: `-ngl 18` + 8 NUMA nodes; B: `-ngl 999` (96GB UMA) |
| 15 | **Sparse MoE** | **Mixtral 8x7B** | 46.7 B / 12.9 B | 32 k | `Q8_0`, `Q4_K_M` | Reference standard Top-2 MoE, 8 experts per layer. | A: `-ngl 33` (Full GPU offload); B: `-ngl 999` (UMA) |
| 16 | **Sparse MoE** | **DSpark (Kimi-K3 / GLM-5.2)**| 100 B / 12 B | 64 k | `UD-Q4_K_XL`, `BF16` | Wide routed expert tables with speculative drafting integration. | A: `-ngl 20` + 8 NUMA nodes; B: `-ngl 999` (UMA) |
| 17 | **Recurrent** | **Command R+ (104B)** | 104 B / 104 B | 128 k | `IQ4_K`, `IQ3_XXS` | Multilingual retrieval-augmented flagship, 128k context. | A: `-ngl 18` + 8 NUMA nodes; B: `-ngl 999` (96GB UMA) |
| 18 | **Recurrent** | **OpenPangu-2.0** | 65 B / 65 B | 32 k | `Q8_0`, `Q4_K_M` | Sovereign enterprise dense/recurrent hybrid architecture. | A: `-ngl 24` + 8 NUMA nodes; B: `-ngl 999` (UMA) |
| 19 | **Recurrent** | **Mamba / SSM Hybrids**| 14 B / 14 B | 64 k | `FP16`, `Q8_0`, `Q4_K_M` | Linear state-space recurrent recurrence without quadratic attention. | A: `-ngl 999` (100% on RTX 4090); B: `-ngl 999` (UMA) |

---

### 4.2 Layer-Interleaved Round-Robin MoE NUMA Offload Mechanics

When running massive sparse MoE models like DeepSeek-V3/R1 (671B parameters, 256 routed experts per layer) or Mixtral 8x22B on multi-socket NUMA systems, traditional runtimes suffer severe memory bus throttling and interconnect contention.

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │           RTX 4090 24GB (PCIe on NUMA Node 1)          │
                                  │ • FlashAttention-2 / MLA Attention Kernels             │
                                  │ • Router Gate Weights & Shared Expert Projections      │
                                  │ • Permanent CUDA Graph Execution (-sm graph)           │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │ Top-K Expert Indices & Activations
                                                              ▼
                             ┌─────────────────────────────────────────────────────────────────┐
                             │       Wait-Free 128-Byte SPSC Ring Buffer Token Dispatch        │
                             └────────┬──────────┬──────────┬──────────┬──────────┬──────────┬─┘
                                      │          │          │          │          │          │
         ┌────────────────────────────┼──────────┴──────────┴──────────┴──────────┼──────────┴────────────────────────┐
         ▼                            ▼                                           ▼                                   ▼
┌──────────────────┐        ┌──────────────────┐                        ┌──────────────────┐        ┌──────────────────┐
│  NUMA Node 0     │        │  NUMA Node 1     │                        │  NUMA Node 6     │        │  NUMA Node 7     │
│  Socket 0, CCX 0 │        │  Socket 0, CCX 1 │        ...             │  Socket 1, CCX 2 │        │  Socket 1, CCX 3 │
│  Experts:        │        │  Experts:        │                        │  Experts:        │        │  Experts:        │
│  e ≡ 0 (mod 8)   │        │  e ≡ 1 (mod 8)   │                        │  e ≡ 6 (mod 8)   │        │  e ≡ 7 (mod 8)   │
│  Local DDR4 CH0/1│        │  Local DDR4 CH2/3│                        │  Local DDR4 CH12 │        │  Local DDR4 CH14 │
└──────────────────┘        └──────────────────┘                        └──────────────────┘        └──────────────────┘
```

#### 1. Microarchitectural Saturation of Flat Interleaving
Under standard OS memory interleaving (`numactl --interleave=all`), physical 4 KiB / 2 MiB memory pages are interleaved uniformly across nodes in address order. However, during autoregressive MoE decode:
- The router selects $K$ active experts per token (e.g., $K=8$ for DeepSeek-V3, $K=2$ for Mixtral).
- Under flat interleaving, multiple active expert weights are split across arbitrary nodes, requiring CPU worker threads to perform remote memory loads over the inter-socket xGMI-3 interconnect.
- The 4x 16-lane xGMI-3 link provides ~64 GB/s bidirectional bandwidth with a ~1.9x latency penalty relative to local DDR4 memory channels (~48 GB/s per NUMA node, ~384 GB/s peak aggregate). This interconnect bottleneck throttles inference to $<30\%$ of theoretical memory bandwidth.

#### 2. Deterministic Layer-Interleaved Round-Robin Allocation
In `ik_llama.cpp` on `feat/numa-split`, MoE weights are partitioned using deterministic round-robin mapping:
$$\text{NUMA\_Node}(l, e) = (l + e) \pmod N$$
where $l \in [0, L-1]$ is the transformer layer index, $e \in [0, E-1]$ is the expert index within the layer, and $N$ is the total count of host NUMA domains ($N=8$ on the dual AMD EPYC 7453 workstation in NPS4 mode).

Alternatively, for wide-row MoE structures with tensor override slicing:
$$\text{NUMA\_Node}(l, e) = \left( \left\lfloor \frac{l \cdot E}{N} \right\rfloor + e \right) \pmod N$$
**Key Microarchitectural Benefits**:
- **Zero Cross-Socket DDR Collisions**: When $K$ experts are activated across successive layers, the active working set is distributed evenly across all 16 memory channels, achieving full aggregate DDR4 memory bandwidth (~384 GB/s).
- **Node-Local Execution**: CPU worker threads pinned via `ggml::cpu::numa::bind_current_thread_to_node(node_id, core_id)` execute GEMM/GEMV exclusively against local physical memory allocated via `mbind()`, maintaining a $>99.5\%$ local node hit rate in `numastat`.
- **Wait-Free Activation Dispatch**: Activations and routing decisions are communicated via 128-byte aligned `common_spsc_ring_buffer` structures, completely bypassing kernel `sys_futex` locks.
- **Direct DMA Storage Streaming**: In asynchronous streaming configurations, weights are streamed directly from the 4x Lexar NM790 RAID-0 carrier card on Node 2 PCIe root complex (~28 GB/s Direct DMA) via `io_uring` + `O_DIRECT`, bound to Node 2 memory channels through recursive storage NUMA discovery (`ggml::cpu::numa::detect_dev_numa_node()`).

---

## 5. Dual-Track Benchmark Strategy

### 5.1 Track 1: Iso-Parameter Standardization

The objective of the Iso-Parameter Track is to **isolate engine software efficiency** under completely controlled, identical hardware utilization parameters.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ISO-PARAMETER STANDARDIZATION                        │
├─────────────────────┬──────────────────────────────────────────────────┤
│ Parameter           │ Standardized Benchmark Values                    │
├─────────────────────┼──────────────────────────────────────────────────┤
│ CPU Threads (`-t`)  │ 28 (Single Socket Milan), 56 (Dual Socket Milan) │
│ Batch Size (`-b`)   │ 512 tokens                                       │
│ Micro-Batch (`-ub`) │ 128 tokens, 512 tokens                           │
│ Offload Layers (`-ngl`)│ 0 (Pure CPU), 16 (Partial), Max Fit (Hybrid) │
│ Context Window (`-c`)│ 2,048 tokens, 8,192 tokens                      │
│ KV Cache Type       │ `-ctk f16 -ctv f16` and `-ctk q8_0 -ctv q8_0`    │
│ FlashAttention      │ `-fa 1` enabled across all engines               │
│ Memory Mapping      │ `-mmp 1` (ik_llama.cpp; omitted for mainline llama.cpp)│
│ Baseline NUMA       │ Standard OS interleaving (`numactl --interleave`)|
└─────────────────────┴──────────────────────────────────────────────────┘
```

**Verification Rule**: Under the Iso-Parameter track, no engine-specific proprietary flags (`--numa split`, `-sm graph`, `--cohort-prefix-sharing`) are enabled. This proves baseline parity before applying microarchitectural specializations.

---

### 5.2 Track 2: Pareto-Optimized Tuning

The objective of the Pareto-Optimized Track is to **evaluate each engine at its theoretical hardware ceiling**. Each engine is autotuned independently across its optimal configuration space:

#### 1. Target Engine Tuning Matrix (`ik_llama.cpp feat/numa-split`):
- **Workstation Tuning**:
  - NUMA Strategy: `--numa split` (exposes `CPU0`..`CPU7` as discrete backends).
  - CUDA Graph Strategy: `-sm graph` with permanent CUDA graph reuse.
  - Hybrid Expert Splitting: FlashAttention + router weights pinned to RTX 4090 (`CUDA0` on Node 1 via `-ngl`), routed experts partitioned across NUMA nodes `CPU0`..`CPU7` via tensor override patterns (`-ot`).
  - Thread Pinning: CCX-aligned thread assignment (`-t 7` per active CCX quadrant, or `-t 28` pinned to Socket 0).
  - Memory & Storage Alignment: `-thp 1` (Transparent Huge Pages), direct DMA streaming via `io_uring` + `O_DIRECT` targeting the Lexar NM790 RAID0 carrier card on Node 2 PCIe root complex.
  - Concurrency Serving: Multi-tenant Cohort Prompt Prefix Sharing enabled (`--cohort-prefix-sharing`, `--cohort-min-prefix 64`).
- **Apple M2 Max Tuning**:
  - Memory Offload: `-ngl 999` (full GPU offload within 64GB/96GB unified memory).
  - Micro-Batch Tuning: `-ub 512` to maximize SIMD-group occupancy.
  - FlashAttention: `-fa 1` with Metal threadgroup tiling.

#### 2. Baseline 1 Tuning Matrix (Vanilla `ik_llama.cpp master`):
- **Workstation Tuning**:
  - NUMA Strategy: `--numa distribute` combined with `numactl --interleave=0-7`.
  - Micro-Optimizations: `-rtr 1` (runtime tensor repacking), `-muge 1` (merge up-gate experts), `-mqkv 1` (merge QKV projections), `-fa 1`.
  - Thread Sweep: Sweeping `-t` from 14 to 28 to 56 to 112 threads to identify the inflection point where SMT thrashing degrades memory bandwidth.
- **Apple M2 Max Tuning**:
  - Metal offload `-ngl 999`, `-fa 1`, tuned batching parameters.

#### 3. Baseline 2 Tuning Matrix (Upstream Mainline `llama.cpp`):
- **Workstation Tuning**:
  - NUMA Strategy: `--numa numactl` with `numactl --interleave=all`.
  - Thread Pool Optimization: Sweeping `--threads` ($28, 56, 112$) and `--threads-batch` ($28, 56$).
  - Micro-Batch Optimization: Sweeping `-ub` ($64, 128, 256, 512$).
- **Apple M2 Max Tuning**:
  - Upstream Metal backend `-ngl 999`, `-fa 1`.

---

## 6. Comprehensive Test & Benchmark Dimensions

```
                                  ┌───────────────────────────────────────────────┐
                                  │      5-DIMENSIONAL BENCHMARK FRAMEWORK        │
                                  └──────────────────────┬────────────────────────┘
                                                         │
         ┌───────────────────┬───────────────────────────┼───────────────────────────┬───────────────────┐
         ▼                   ▼                           ▼                           ▼                   ▼
┌──────────────────┐┌──────────────────┐    ┌───────────────────────────┐   ┌──────────────────┐┌──────────────────┐
│  1. Latency &    ││ 2. High-Concurr. │    │    3. Context Scaling     │   │ 4. Hardware &    ││  5. Numerical &  │
│    Throughput    ││    Multi-Client  │    │      (64 to 128k+)        │   │    Profiling     ││    Perplexity    │
│ • TTFT (Prompt)  ││ • 1..32 Clients  │    │ • Short: PP=64, TG=32     │   │ • numastat Hit/M ││ • WikiText-2 PPL │
│ • Gen (tok/s)    ││ • Disjoint PPS=1 │    │ • Std: PP=2048, TG=128    │   │ • xGMI Bus Sat.  ││ • Logits Δ <1e-4 │
│ • ITL Jitter     ││ • Cohort PPS=2.. │    │ • Long: 8k, 32k, 64k      │   │ • Direct DMA I/O ││ • Zero Output    │
│ • p50/p95/p99    ││ • KV VRAM/RAM    │    │ • YaRN / Chunked Prefill  │   │ • 128B SPSC Ping ││   Degradation    │
└──────────────────┘└──────────────────┘    └───────────────────────────┘   └──────────────────┘└──────────────────┘
```

### 6.1 Dimension 1: Latency & Throughput (Single-Stream Execution)

- **Prompt Processing / Time-To-First-Token (TTFT)**:
  - Measures prompt ingestion throughput ($S_{PP}$, in tokens/sec) and TTFT ($T_{PP}$, in milliseconds) across prompt lengths:
    $$PP \in \{64, 512, 2048, 8192, 32768\}$$
  - Evaluates compute-bound matrix multiplication efficiency and FlashAttention-2 tiling.
- **Autoregressive Generation Throughput**:
  - Measures single-token generation speed ($S_{TG}$, in tokens/sec) across generation lengths:
    $$TG \in \{32, 128, 512, 2048\}$$
  - Evaluates memory-bandwidth bound streaming of model weights from DDR4 / GDDR6X / UMA memory.
- **Inter-Token Latency (ITL) & Jitter Percentiles**:
  - Records the exact duration of every individual decode step:
    $$\Delta t_i = t_{i} - t_{i-1}, \quad i \in [1..TG]$$
  - Computes statistical percentiles: $p50$, $p90$, $p95$, $p99$, and maximum latency spike.
  - Detects memory allocation pauses, OS page faults, NUMA remote page thrashing, and kernel scheduler migration delays.

---

### 6.2 Dimension 2: High-Concurrency Multi-Client Scaling & Prefix Sharing

Evaluates server throughput and memory efficiency using `examples/server/server.cpp` under concurrent multi-client load ($C \in \{1, 2, 4, 8, 16, 24, 32\}$ concurrent streams).

#### Workload Scenarios:
1. **Scenario A: Disjoint Multi-Tenant Streams ($PPS = 1$)**:
   - Each client issues a distinct, unrelated system prompt and query.
   - Evaluates continuous batching scheduler performance, slot preemption, and aggregate KV cache contention.
2. **Scenario B: Cohort Prompt Prefix Sharing ($PPS \in \{2, 4, 8, 16, 32\}$)**:
   - Multiple client requests share an identical system prompt prefix ($L_{\text{prefix}} \in \{1024, 4096, 8192\}$ tokens) representing multi-agent swarms, enterprise RAG documents, or few-shot system instructions.
   - Target Engine leverages `--cohort-prefix-sharing` to execute zero-copy sequence cloning (`llama_kv_cache_seq_cp`).
   - Baselines 1 & 2 process prefixes independently for every slot.
   - Metrics Recorded:
     - **Aggregate Serving Throughput**: Total tokens delivered per second across all $C$ streams.
     - **Downstream TTFT Speedup**: TTFT reduction for sibling cohort requests ($C \ge 2$).
     - **KV Cache Memory Footprint**: Active VRAM and Host RAM consumed by KV cache blocks.

---

### 6.3 Dimension 3: Context Window Scaling

Evaluates performance across sequence lengths from conversational contexts to massive document ingestion:
- **Micro-Context ($PP=64, TG=32$)**: Isolates kernel launch latency, CPU-to-GPU command queue overhead, and CUDA graph execution benefits.
- **Standard Context ($PP=2048, TG=128$)**: Standard conversational and coding benchmark.
- **Medium-Long Context ($PP=8192, TG=256$)**: Multi-turn dialogue history and long-context RAG.
- **Ultra-Long Context ($PP=32768, 65536, 131072$)**:
  - Measures quadratic attention scaling vs. FlashAttention-2 memory footprint.
  - Evaluates YaRN RoPE frequency scaling stability.
  - Compares KV cache quantization formats (`f16` vs. `q8_0` vs. `q4_0`).
  - Benchmarks Host RAM KV cache offloading (`-nkvo 1`) when sequence length exceeds discrete GPU VRAM capacity.

---

### 6.4 Dimension 4: Memory & Hardware Counter Profiling

Captures deep microarchitectural counters during benchmark execution:
- **NUMA Memory Allocation & Locality**:
  - Interrogates `/sys/devices/system/node/node*/numastat` and `numastat -m` before and after each run.
  - Monitors `numa_hit`, `numa_miss`, `numa_foreign`, and `local_node` vs. `other_node` allocations.
  - Verifies that `--numa split` achieves $>99.5\%$ local node access, whereas standard interleaving induces $>50\%$ remote interconnect traffic.
- **Inter-Socket Interconnect (xGMI) Saturation**:
  - Profiles remote socket cache misses and memory traffic:
    ```bash
    perf stat -e amd_l3_miss_remote,node-loads,node-load-misses,cycles,instructions
    ```
- **Storage Subsystem Direct DMA Bandwidth**:
  - Evaluates direct I/O read throughput from the 4x Lexar NM790 RAID-0 array on Node 2.
  - Monitors page cache memory allocation via `slabtop` and `/proc/meminfo` to verify zero page-cache pollution under `io_uring` + `O_DIRECT`.
- **SPSC Ring Buffer Speculative Jitter**:
  - Benchmarks inter-socket candidate token streaming latency using `tests/test-numa.cpp`.
  - Compares 128-byte aligned lock-free SPSC circular queue vs. standard 64-byte unaligned queue (quantifying false-sharing L3 invalidations) and OS `sys_futex` wake-up delays.

---

### 6.5 Dimension 5: Generation Quality & Numerical Verification

Optimizations that compromise numerical accuracy are unacceptable. Every benchmarked model configuration must undergo rigorous verification:
- **WikiText-2 Perplexity Verification**:
  - Evaluates perplexity on `scratch/wikitext-2-raw/wiki.test.raw` using `llama-perplexity`:
    $$PPL = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{<i}) \right)$$
  - Tolerance Invariant: Perplexity between `feat/numa-split`, vanilla `ik_llama.cpp`, and upstream `llama.cpp` must match within:
    $$|\Delta PPL| < 0.001$$
- **Logits Divergence & Epsilon Parity**:
  - Dumps raw output logits for fixed prompt evaluation.
  - Computes maximum absolute difference and Kullback-Leibler (KL) divergence:
    $$D_{KL}(P_{\text{target}} \parallel P_{\text{baseline}}) < 10^{-6}$$
  - Confirms wide-row column slicing, permanent CUDA graph execution, and NUMA buffer distribution preserve mathematical equivalence.

---

## 7. Execution Matrix & Standardized Commands

### 7.1 Compilation Invariant

All builds on Linux and macOS must be executed using:
```bash
cmake --build build -j $(nproc)
```
*Never invoke raw `make` or uncoordinated compiler commands.*

---

### 7.2 Workstation Execution Commands (Dual AMD EPYC 7453 + RTX 4090)

#### Track 1: Iso-Parameter Baseline Command (Run on all 3 engines)
```bash
# Standardized across all 3 engines: 56 threads, batch 512, ubatch 512, ngl 16
./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  -t 56 -b 512 -ub 512 \
  -ngl 16 -fa 1 -mmp 1 \
  -r 3 -o json > iso_llama70b_results.json
```

#### Track 2: Pareto-Optimized Target Engine Command (`ik_llama.cpp feat/numa-split`)
```bash
# NUMA split, permanent CUDA graphs, THP enabled, runtime repack, optimal GPU offload
./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  --numa split \
  -sm graph -fa 1 -rtr 1 -thp 1 \
  -t 28 -b 512 -ub 512 \
  -ngl 24 \
  -r 3 -o json > pareto_target_llama70b_results.json
```

#### Track 2: Pareto-Optimized Baseline 1 Command (Vanilla `ik_llama.cpp master`)
```bash
# NUMA distribute, runtime repack, merged up-gate, swept threads
numactl --interleave=0-7 ./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  --numa distribute \
  -fa 1 -rtr 1 -muge 1 -mqkv 1 \
  -t 56 -b 512 -ub 512 \
  -ngl 24 \
  -r 3 -o json > pareto_base1_llama70b_results.json
```

#### Track 2: Pareto-Optimized Baseline 2 Command (Mainline `llama.cpp master`)
```bash
# Upstream llama.cpp with numactl interleave and tuned threads (note: -mmp is omitted for mainline)
numactl --interleave=all ./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  --numa numactl \
  -fa 1 \
  -t 56 -b 512 -ub 512 \
  -ngl 24 \
  -r 3 -o json > pareto_base2_llama70b_results.json
```

---

### 7.3 Apple Silicon Execution Commands (MacBook Pro Apple M2 Max)

#### Track 1: Iso-Parameter Baseline Command (M2 Max)
```bash
# Standardized across all 3 engines on Apple Silicon (8 P-cores pinned)
./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  -t 8 -b 512 -ub 512 \
  -ngl 999 -fa 1 \
  -r 3 -o json > m2max_iso_results.json
```

#### Track 2: Pareto-Optimized Command (M2 Max)
```bash
# Full unified memory offload, Metal threadgroup tuning
./build/bin/llama-bench \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -p 64,512,2048 -n 128 \
  -t 8 -b 1024 -ub 512 \
  -ngl 999 -fa 1 \
  -r 5 -o json > m2max_pareto_results.json
```

---

### 7.4 High-Concurrency Multi-Client Serving Evaluation

#### Step 1: Launch Server with Cohort Prefix Sharing
```bash
# On feat/numa-split:
./build/bin/llama-server \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -c 16384 -b 2048 -ub 512 \
  --numa split -ngl 24 -fa 1 \
  --cohort-prefix-sharing --cohort-min-prefix 64 \
  -np 16 --port 8089 --host 127.0.0.1
```

#### Step 2: Run Concurrent Client Benchmark Suite
```bash
# Execute concurrent load harness sweeping concurrency 1..32
python3 scripts/bench_concurrent_server.py --concurrency 1 2 4 8 16 24 32 --prefix-len 1024 2048 4096
```

---

### 7.5 Numerical Perplexity Verification Command

```bash
./build/bin/llama-perplexity \
  -m models/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
  -f scratch/wikitext-2-raw/wiki.test.raw \
  -c 2048 -b 512 -ub 512 \
  --numa split -ngl 24 -fa 1
```

---

## 8. Data Collection, Statistical Rigor & Low-Latency Systems Invariants

1. **Statistical Rigor & Permutation Testing (Bilokon & Gunduz 2023; `inv-benchmark-claims-require-local-reproduction`)**:
   - Rejects single-run or mean-only throughput averages ($tok/s$).
   - Every benchmark configuration is executed for a minimum of **$N \ge 30$ repetitions** following mandatory warmup passes (`-w 1`) to eliminate cold filesystem cache penalties.
   - Rejects standard paired Student's t-tests and ANOVA on latency distributions: as demonstrated by Bilokon & Gunduz (arXiv:2309.04259, 2023), microarchitectural inference latency is heavy-tailed, asymmetric, and non-Gaussian. Statistical significance must be evaluated via paired permutation tests on specific quantiles of interest (median, $p95$, $p99$) over $\ge 10,000$ resamples:
     $$p = \frac{1}{B} \sum_{b=1}^{B} \mathbb{I}\left( |\hat{\theta}_b^*| \ge |\hat{\theta}_{\text{obs}}| \right)$$
   - Reports bootstrap 99% confidence intervals alongside complete latency tail quantiles ($p50, p90, p99, p99.9, \text{max jitter}$) to capture multi-socket xGMI interconnect and memory channel contention.
2. **Gil Tene Coordinated Omission Avoidance**:
   - In concurrent server load testing ([`scripts/bench_concurrent_server.py`](../scripts/bench_concurrent_server.py)), request arrivals are generated via independent Poisson processes rather than closed-loop synchronous loops. This prevents downstream queuing delays from artificially truncating service-time measurements:
     $$L_{\text{corrected}} = L_{\text{measured}} + (T_{\text{actual}} - T_{\text{expected}})$$
3. **Minor Page Fault Elimination via Pretouching (Chronicle Queue Principle)**:
   - Linux demand paging introduces intermittent $2\text{--}50\,\mu\text{s}$ soft-page-fault latency spikes during autoregressive decode loops.
   - Enforce page pre-faulting via `madvise(MADV_POPULATE_WRITE)` on all HugeTLB model weight arenas and preallocated KV buffers prior to initiating timing loops.
4. **Serialized Nanosecond Hardware Timestamping (x86 & Apple Silicon)**:
   - High-resolution timing loops utilize serialized CPU cycle counters and memory barriers to prevent out-of-order execution from skewing token generation timestamps (Nagraj 2024, Sites 2021, HangukQuant 2026):
     - *x86-64*: Serialized `__rdtscp` paired with `_mm_lfence()`:
       ```cpp
       static inline uint64_t x86_rdtscp_serialized() {
           uint32_t aux;
           uint64_t tsc = __rdtscp(&aux);
           _mm_lfence();
           return tsc;
       }
       ```
     - *ARM64 / Apple Silicon*: Direct reading of the virtual counter register `CNTVCT_EL0` bracketed by Instruction Synchronization Barriers (`isb`) to prevent speculative instruction reordering across measurement boundaries:
       ```cpp
       static inline uint64_t arm64_cntvct_serialized() {
           uint64_t val;
           asm volatile("isb; mrs %0, cntvct_el0; isb" : "=r"(val) : : "memory");
           return val;
       }
       ```
5. **Benchmarking Pitfalls & Dead-Code Elimination (DCE) Guards (HangukQuant 2026)**:
   - Microbenchmarking low-latency C++ inference code presents critical traps exposed by HangukQuant (2026) (*HFT Benchmarking Done Wrong*):
     - *Dead-Code Elimination (DCE)*: Compilers aggressively optimize away decode steps or token transformations when outputs are not consumed downstream. All benchmark timing loops must wrap token generation results and intermediate activation descriptors in explicit compiler memory barriers / `DoNotOptimize` clobbers:
       ```cpp
       template <typename T>
       static inline void do_not_optimize(T const & val) {
           asm volatile("" : : "g"(val) : "memory");
       }
       ```
     - *Payload Realism & Entropy*: Trivial synthetic token inputs fail to trigger realistic KV cache indexing and memory controller access patterns. Tests must evaluate realistic multi-token prompt corpora with non-trivial entropy.
     - *Batching Illusion*: Measuring steady-state throughput under massive batching conceals severe tail latency degradation. Hand-off latency must be evaluated on individual token steps with controlled inter-arrival schedules.
     - *Inter-Core Coherence Variance*: Hand-off latency varies by up to $10\times$ depending on whether threads share a physical core, CCX, or cross a NUMA socket boundary; benchmark reports must disclose exact core affinity.
6. **Memory Locality Verification via `numastat`**:
   - Sample unprivileged `/sys/devices/system/node/node*/numastat` page hits before and after execution to assert memory locality index:
     $$\mathcal{L} = \frac{\Delta \text{numa\_hit}}{\Delta \text{numa\_hit} + \Delta \text{numa\_miss}} \ge 0.992$$
7. **Numerical Integrity & Zero Degradation Checklist**:
   - **Perplexity Parity**: WikiText-2 $|\Delta PPL| < 0.001$ relative to baseline.
   - **Logits Parity**: Maximum relative logit divergence $\epsilon < 10^{-5}$ across first 512 generated tokens.
   - **Kullback-Leibler Divergence**: $D_{KL}(P_{\text{target}} \parallel P_{\text{baseline}}) < 10^{-6}$.
   - **Regression Suite**: Automated test suites ([`tests/test-numa.cpp`](../tests/test-numa.cpp) and [`tests/test-cohort-prefix-sharing.cpp`](../tests/test-cohort-prefix-sharing.cpp)) must pass 100% cleanly.
8. **Persistence Schema**:
   - Benchmark runs emit structured SQL or JSON output, which is ingested directly into an SQLite database (`llama-bench.sqlite`).
9. **Automated Comparative Reporting**:
   - `scripts/compare-llama-bench.py` and `scripts/bench_matrix.py` generate Pareto markdown comparison tables and speedup factors:
     $$\text{Speedup} = \frac{S_{\text{target}}}{S_{\text{baseline}}}$$
10. **Brendan Gregg (2020) USE & RED Methodology & Hardware Performance Counter (PMC) Profiling**:
    - *USE Method (Utilization, Saturation, Errors)*: For each hardware resource (Zen 3/4 CCDs, DDR4/DDR5 channels, PCIe 4.0 carrier lanes, NVMe queues, RTX 4090 Tensor Cores), audit Utilization (% time active), Saturation (SPSC queue backlog, memory bus throttle cycles), and Errors (PCIe replay events, page faults).
    - *RED Method (Rate, Errors, Duration)*: For each inference micro-batch, monitor token throughput Rate ($tok/s$), Error count (numerical logit divergence), and Duration (TTFT and inter-token latency quantiles).
    - *Linux `perf stat` Hardware Event Matrix*: Profile microarchitectural cache and NUMA efficiency across nodes:
      ```bash
      perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,\
      L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,\
      node-loads,node-load-misses \
      ./build/bin/llama-bench -m models/model.gguf -n 128 -b 512
      ```

---

## 9. Privacy Wall & Git Invariants Compliance

In strict compliance with repository security rules:
- **No Private Hostnames or Paths**: All script parameters and documentation reference generic relative model paths (`models/`) or environment variables (`$MODEL_PATH`).
- **Untouchable Host Files**: Host workstation files (`build.sh`, `env.sh`, `run-*.sh`, `probabilities.txt`, `usage.txt`, `aocc_sandbox/`, `AGENTS.md`) remain untouched.
- **Compilation Invariant**: All builds strictly execute `cmake --build build -j $(nproc)`.
