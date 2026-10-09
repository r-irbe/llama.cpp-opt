# Benchmarking Execution Matrix & Reproducibility Guide

**Target Repository**: `ikawrakow/ik_llama.cpp` (`feat/numa-split`)  
**Companion Document**: [`docs/benchmarking-and-testing-plan.md`](../docs/benchmarking-and-testing-plan.md)  
**Execution Script**: [`scripts/bench_matrix.py`](../scripts/bench_matrix.py)

---

## 1. Consequential Open-Source Models: Parameter & Allocation Matrix

The table below details the memory footprints, GPU/CPU partitioning, and recommended execution settings for all target models across our two hardware platforms:
1. **Workstation**: Dual AMD EPYC 7453 (56C/112T, 8 NUMA nodes, ~1 TB DDR4 RAM), 1x NVIDIA RTX 4090 (24 GB VRAM, Node 1), 4x 4TB Lexar NM790 RAID0 (Node 2, ~28 GB/s).
2. **MacBook Pro**: Apple M2 Max (12C CPU, 30/38C GPU, 64/96 GB unified LPDDR5-6400, 400 GB/s).

| Model Identifier | Architecture | Parameters (Total / Active) | Recommended Quant | Footprint (Weights) | Workstation Partitioning (RTX 4090 + EPYC NPS4) | Apple M2 Max Partitioning (Unified UMA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3.3 / 3.1 70B** | Dense GQA | 70.6 B / 70.6 B | `UD-Q4_K_XL` / `IQ4_K` | ~39.2 GiB | `-ngl 24` (14 GiB on Node 1 VRAM), remaining 56 layers split across `CPU0..CPU7` | `-ngl 999` (Full Metal offload into UMA) |
| **Llama 3.1 8B** | Dense GQA | 8.0 B / 8.0 B | `Q8_0` / `Q4_K_M` | ~8.5 GiB / ~4.9 GiB | `-ngl 999` (100% VRAM on RTX 4090) or `-ngl 0` (Pure CPU NUMA baseline) | `-ngl 999` (Full Metal offload) |
| **Qwen 2.5 72B** | Dense SwiGLU | 72.7 B / 72.7 B | `UD-Q4_K_XL` / `IQ4_K` | ~41.5 GiB | `-ngl 22` (14 GiB on Node 1 VRAM), 58 layers split across `CPU0..CPU7` | `-ngl 999` (Full Metal offload) |
| **Qwen 2.5 32B** | Dense SwiGLU | 32.5 B / 32.5 B | `Q4_K_M` / `IQ4_NL` | ~19.8 GiB | `-ngl 999` (Fits entirely in 24 GB VRAM on Node 1) | `-ngl 999` (Full Metal offload) |
| **Qwen 2.5 7B** | Dense SwiGLU | 7.6 B / 7.6 B | `Q8_0` / `Q4_K_M` | ~8.1 GiB / ~4.7 GiB | `-ngl 999` (Pure VRAM) or `-ngl 0` (Pure CPU NUMA) | `-ngl 999` (Full Metal offload) |
| **Gemma 2 27B** | Dense Sliding | 27.2 B / 27.2 B | `Q4_K_M` / `IQ4_NL` | ~16.5 GiB | `-ngl 999` (Fits in 24 GB VRAM) | `-ngl 999` (Full Metal offload) |
| **Gemma 2 9B** | Dense Sliding | 9.2 B / 9.2 B | `Q8_0` / `Q4_K_M` | ~9.8 GiB / ~5.6 GiB | `-ngl 999` (Pure VRAM) | `-ngl 999` (Full Metal offload) |
| **Mistral Large 2 (123B)**| Dense Tekken | 123.0 B / 123.0 B | `IQ3_XXS` / `IQ2_XXS` | ~52 GiB / ~38 GiB | `-ngl 16` (12 GiB on Node 1 VRAM), remaining 72 layers split across `CPU0..CPU7` | `-ngl 999` (fits in 64GB/96GB UMA with IQ2/IQ3) |
| **Mistral NeMo 12B** | Dense Tekken | 12.2 B / 12.2 B | `Q8_0` / `Q4_K_M` | ~13.1 GiB / ~7.5 GiB | `-ngl 999` (Pure VRAM) | `-ngl 999` (Full Metal offload) |
| **Phi-4 14B** | Dense Synth | 14.7 B / 14.7 B | `Q8_0` / `Q4_K_M` | ~15.6 GiB / ~9.1 GiB | `-ngl 999` (Pure VRAM) | `-ngl 999` (Full Metal offload) |
| **DeepSeek-V3 / R1 (671B)**| Sparse MoE MLA | 671 B / 37 B | `UD-IQ2_XXS` / `IQ1_S` | ~189 GiB–245 GiB | Attention + router on Node 1 VRAM (`-ngl 62 -sm layer -nkvo`), 256 routed experts split across 8 NUMA nodes (`CPU0..CPU7`) | Streamed / partitioned or dual APU target |
| **DeepSeek-V2.5 / V4** | Sparse MoE MLA | 236 B / 21 B | `IQ3_XXS` / `IQ2_XXS` | ~88 GiB / ~68 GiB | Attention/MLA on Node 1 VRAM, routed experts split across 8 NUMA nodes | `-ngl 999` with IQ2_XXS in 96GB UMA |
| **Qwen 2.5-MoE (14B/A2.7B)**| Sparse MoE | 14.3 B / 2.7 B | `Q8_0` / `Q4_K_M` | ~15.2 GiB / ~8.9 GiB | `-ngl 999` (Entirely in 24 GB VRAM) | `-ngl 999` (Full Metal offload) |
| **Mixtral 8x22B** | Sparse Top-2 MoE | 141 B / 39 B | `IQ3_XXS` / `IQ2_XXS` | ~58 GiB / ~44 GiB | Shared attention on Node 1 VRAM, 8 experts partitioned across 8 NUMA nodes (1 expert per node!) | `-ngl 999` in 64GB/96GB UMA |
| **Mixtral 8x7B** | Sparse Top-2 MoE | 46.7 B / 12.9 B | `Q4_K_M` / `IQ4_NL` | ~26.4 GiB | Attention on Node 1 VRAM, 8 experts partitioned across 8 NUMA nodes | `-ngl 999` (Fits in 32GB+ UMA) |
| **DSpark (Kimi-K3/GLM-5.2)**| Hybrid MoE Spec | ~50 B–100 B | `UD-Q4_K_XL` / `BF16` | ~35 GiB–65 GiB | Hot router on Node 1 VRAM, draft model on Socket 1 (Node 4), target on Socket 0 (Node 0..3) | `-ngl 999` (Full Metal offload) |
| **Command R+ (104B)** | Dense Hybrid | 104 B / 104 B | `IQ4_K` / `IQ3_XXS` | ~58 GiB / ~44 GiB | `-ngl 18` (12 GiB on Node 1 VRAM), remaining layers on `CPU0..CPU7` | `-ngl 999` with IQ3 in 64GB/96GB UMA |
| **OpenPangu-2.0** | Enterprise Dense | Varied (32B–70B) | `Q4_K_M` / `IQ4_NL` | ~20 GiB–42 GiB | Standard hybrid offload | `-ngl 999` (Full Metal offload) |
| **Mamba / SSM Hybrids**| Recurrent SSM | 8 B–14 B | `FP16` / `Q8_0` | ~16 GiB / ~9 GiB | `-ngl 999` (Pure VRAM) | `-ngl 999` (Full Metal offload) |

---

## 2. Workstation Dual-Socket NUMA Topologies & Thread Pinning Rules

```
                      AMD EPYC 7453 Dual-Socket (8 NUMA Domains)
┌───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│              SOCKET 0 (Nodes 0..3)            │              SOCKET 1 (Nodes 4..7)            │
├───────────────┬───────────────┬───────────────┼───────────────┬───────────────┬───────────────┤
│ Node 0        │ Node 1        │ Node 2        │ Node 4        │ Node 5        │ Node 6        │
│ Cores 0..6    │ Cores 7..13   │ Cores 14..20  │ Cores 28..34  │ Cores 35..41  │ Cores 42..48  │
│ CPUs 0-6,     │ CPUs 7-13,    │ CPUs 14-20,   │ CPUs 28-34,   │ CPUs 35-41,   │ CPUs 42-48,   │
│      56-62    │      63-69    │      70-76    │      84-90    │      91-97    │      98-104   │
│ SN850X RAID1  │ RTX 4090 GPU  │ Lexar RAID0   │ Speculative   │ Free NUMA     │ Free NUMA     │
│ OS & Context  │ PCIe 4.0 x16  │ PCIe 4.0 x16  │ Draft Model   │ MoE Workers   │ MoE Workers   │
└───────────────┴───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
```

### Pinning Directives:
1. **Target Model Primary Compute (Socket 0)**:
   - Primary thread pool pins to Socket 0 (Nodes 0, 1, 2, 3) using `-t 28`.
   - Node 1 handles PCIe DMA synchronization with the RTX 4090 GPU.
   - Node 2 issues asynchronous direct I/O DMA read requests to the 4x Lexar NM790 NVMe RAID-0 carrier card.
2. **Speculative Draft Model Isolation (Socket 1)**:
   - When running speculative drafting (`--speculative`), the draft model pins exclusively to **Socket 1** (Node 4, CPUs 28–34) via `ggml::cpu::numa::get_optimal_draft_node()`.
   - The wait-free Single-Producer Single-Consumer (SPSC) ring buffer links Socket 1 to Socket 0 with strict `alignas(128)` cache padding, eliminating false sharing over xGMI.
3. **MoE Expert Partitioning (`--numa split`)**:
   - For Mixtral 8x7B / 8x22B (8 routed experts): Exactly **1 expert per NUMA node**. Every CCX cluster streams expert weights from its local DDR4 memory channels without traversing xGMI.
   - For DeepSeek-V3 (256 routed experts): Exactly **32 routed experts per NUMA node**.

### 2.4 16-Channel DRAM Bank Assignment & Row-Interleaved Repacking (`-rtr`)
The Dual AMD EPYC 7453 workstation features **16 memory channels** (8 DDR4-3200 channels per physical socket, 2 physical channels per NUMA node / CCX quadrant), yielding ~204.8 GB/s aggregate bandwidth.
- **The Power-of-Two Bank Aliasing Trap**: DDR4 SDRAM organizes physical memory into 16 banks partitioned across 4 bank groups. Memory addresses map to banks via specific lower address bits. When large tensor matrices have row strides matching exact powers of two ($64\text{ KB}$, $256\text{ KB}$, $1\text{ MB}$, $2\text{ MB}$), successive row accesses by parallel OpenMP threads map to the *exact same DRAM bank and bank group*. This forces the memory controller to repeatedly issue row precharge ($t_{\text{RP}} \approx 14\text{ ns}$) and row-to-column delay ($t_{\text{RCD}} \approx 14\text{ ns}$) commands, destroying open-page hit rates and degrading memory throughput by $>50\%$.
- **Row-Interleaved Repacking (`-rtr`) & Odd-Stride Padding**: The `-rtr` option repacks quantized tensor weights into row-interleaved blocks with non-power-of-two strides (inserting 64-byte or 128-byte cache-line padding between consecutive expert row blocks). This rotates successive thread accesses across orthogonal memory channels and bank groups, keeping DDR4 row buffers open and sustaining $>92\%$ of peak bandwidth during parallel MoE expert GEMM execution.

---

## 3. Comparative Test Suite Execution Flow

```
                               ┌────────────────────────────────┐
                               │  Master Benchmark Orchestrator │
                               │    `scripts/bench_matrix.py`   │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
┌───────────────────────────────┐                               ┌───────────────────────────────┐
│     TRACK 1: ISO-PARAMETER    │                               │   TRACK 2: PARETO-OPTIMIZED   │
├───────────────────────────────┤                               ├───────────────────────────────┤
│ Standardized:                 │                               │ Engine 1: Target              │
│ • Threads: 28 and 56          │                               │ • --numa split, -sm graph     │
│ • Batch: 512, UBatch: 512     │                               │ • -fa 1, -thp 1, -rtr 1       │
│ • FlashAttn: 1                │                               │ • Pinning: CCX-aligned        │
│ • NUMA: OS Interleave         │                               │ • Storage: Node 2 NVMe RAID0  │
│                               │                               ├───────────────────────────────┤
│ Runs sequentially across:     │                               │ Engine 2: Vanilla ik_llama    │
│ 1. `feat/numa-split`          │                               │ • --numa distribute           │
│ 2. Vanilla `ik_llama.cpp`     │                               │ • -rtr 1, -muge 1, -mqkv 1    │
│ 3. Upstream mainline `llama`  │                               │ • Thread sweep: 14..112       │
│                               │                               ├───────────────────────────────┤
│ Ingests into SQLite database: │                               │ Engine 3: Mainline llama.cpp  │
│ `results/iso_track.sqlite`    │                               │ • --numa numactl              │
│                               │                               │ • Swept threads & ubatch      │
│ Emits comparison table:       │                               │                               │
│ `results/iso_comparison.md`   │                               │ Emits peak comparison:        │
│                               │                               │ `results/pareto_comparison.md`│
└───────────────────────────────┘                               └───────────────────────────────┘
```

---

## 4. Concurrent Serving & Cohort Prefix Sharing Test Suite

The server benchmark evaluates concurrent multi-tenant throughput and memory savings under `llama-server`.

### Server Launch Configuration (`feat/numa-split`):
```bash
./build/bin/llama-server \
  -m models/Llama-3.3-70B-Instruct-UD-Q4_K_XL.gguf \
  -c 32768 -b 2048 -ub 512 \
  --numa split -ngl 24 -fa 1 -thp 1 \
  --cohort-prefix-sharing --cohort-min-prefix 64 \
  -np 32 --port 8089 --host 127.0.0.1
```

### High-Concurrency Client Scenarios:
1. **Scenario 1: Disjoint Traffic ($PPS = 1$)**:
   - 32 concurrent requests with independent system prompts (concurrency levels: 1, 2, 4, 8, 16, 24, 32).
   - Measures continuous batching efficiency and baseline KV cache memory growth.
2. **Scenario 2: Cohort Prompt Prefix Sharing ($PPS \in \{2, 4, 8, 16, 32\}$)**:
   - 32 concurrent requests sharing a 4,096-token system prompt (agent instructions, tool schemas, or ingested legal filing).
   - Requests provide header `X-Cohort-ID: legal-team-alpha` and `X-Prompt-Cache-Key: contract-corpus-v1`.
   - Sibling slots attach to primary slot via `llama_kv_cache_seq_cp`.
   - Records aggregate throughput (tok/s), Time-To-First-Token (TTFT), and physical KV memory consumption.

---

## 5. Numerical Integrity & Zero Degradation Checklist

Every optimization applied must be certified for zero numerical degradation:

| Check | Tool / Command | Acceptance Invariant |
| :--- | :--- | :--- |
| **WikiText-2 Perplexity** | `./build/bin/llama-perplexity -m <model> -f scratch/wikitext-2-raw/wiki.test.raw` | $|\Delta PPL| < 0.001$ relative to upstream reference |
| **Logits Parity** | `./build/bin/llama-cli -m <model> -p "Test prompt" -n 1 --logits-all` | Max absolute difference $\epsilon < 10^{-5}$; KL divergence $D_{KL} < 10^{-6}$ |
| **Context Extension** | `./build/bin/llama-perplexity -c 32768 --rope-scaling yarn` | Monotonic loss curve without NaN / Inf token logits |
| **NUMA Isolation** | `tests/test-numa` | 100% pass across all 8 NUMA nodes and simulated NPS topologies |
| **Cohort Isolation** | `tests/test-cohort-prefix-sharing` | 100% pass across tenant separation and warm cache affinity |
