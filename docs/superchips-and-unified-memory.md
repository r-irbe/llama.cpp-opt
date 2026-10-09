# Superchips & Unified Memory Inference Architectures: AMD MI300A, NVIDIA Grace Hopper / Blackwell, and Apple Silicon

**Author**: Systems Architecture & High-Performance Inference Group  
**Target Repository**: [`ikawrakow/ik_llama.cpp`](../README.md)  
**Research References**: [`scratch/research_superchips/amd_mi300a_apu_architecture.md`](../scratch/research_superchips/amd_mi300a_apu_architecture.md), [`scratch/research_superchips/nvidia_grace_hopper_blackwell_spark.md`](../scratch/research_superchips/nvidia_grace_hopper_blackwell_spark.md), [`scratch/research_superchips/mlx/`](../scratch/research_superchips/mlx/)

---

## 1. Executive Summary: The Unified Memory Paradigm Shift

Traditional heterogeneous inference systems suffer from the **PCIe Memory Wall**:
- Host CPU RAM is large (up to 1–2 TB) but bandwidth-constrained (150–200 GB/s across DDR4/DDR5 channels).
- Discrete GPU VRAM is ultra-fast (1,000–3,300 GB/s across HBM/GDDR6X) but capacity-constrained (16–80 GB).
- All communication between CPU and GPU must traverse the PCIe bus (Gen4 x16: 31.5 GB/s; Gen5 x16: 63 GB/s).
- For large models (e.g. 70B dense or 671B MoE), weights or KV caches that cannot fit into VRAM must be shuttled across PCIe, degrading throughput by an order of magnitude.

**Unified Memory Superchips** eliminate the PCIe bottleneck by co-packaging high-performance CPU cores and massive GPU compute engines on a single package or high-speed coherent substrate:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SUPERCHIP ARCHITECTURAL COMPARISON                        │
├──────────────────────┬──────────────────────┬───────────────────┬──────────────────────┤
│ Metric               │ AMD Instinct MI300A  │ NVIDIA GH200      │ NVIDIA GB200         │
├──────────────────────┼──────────────────────┼───────────────────┼──────────────────────┤
│ CPU Cores            │ 24 Zen 4 cores       │ 72 Arm Neoverse-V2│ 72 Arm Neoverse-V2   │
│ GPU Compute          │ 228 CDNA 3 CUs       │ Hopper H100 GPU   │ 2x Blackwell GPUs    │
│ Physical Memory      │ 128 GB Unified HBM3  │ 480 GB LPDDR5X +  │ 480 GB LPDDR5X +     │
│                      │                      │ 96/144 GB HBM3/e  │ 384 GB HBM3e         │
│ Memory Bandwidth     │ **5,300 GB/s**       │ 500 GB/s (CPU) +  │ 500 GB/s (CPU) +     │
│                      │ (Unified to all)     │ 4,000 GB/s (GPU)  │ 8,000 GB/s (GPU)     │
│ Interconnect         │ On-Package 4th Gen   │ NVLink-C2C        │ NVLink-C2C           │
│                      │ Infinity Fabric      │ **900 GB/s**      │ **1,800 GB/s**       │
│ Coherence Model      │ Hardware Cache       │ Hardware ATS +    │ Hardware ATS +       │
│                      │ Coherent APU         │ NVLink Coherence  │ NVLink Coherence     │
└──────────────────────┴──────────────────────┴───────────────────┴──────────────────────┘
```

### 1.1 Theoretical Foundation: Bilardi et al. (2016) Decomposable BSP Model (D-BSP)
Heterogeneous unified memory superchips (AMD MI300A, NVIDIA GH200/GB200, Apple Silicon Ultra) are formalized under the **Decomposable Bulk Synchronous Parallel** machine model $\text{D-BSP}(p, \mathbf{g}, \mathbf{l})$ established by Bilardi et al. (2016).
- **Hierarchical Clustering**: The system is modeled as a tree of nested processor clusters across $m$ hierarchical levels, parameterized by:
  - $p$: Total processor cores / compute units.
  - $\mathbf{g} = (g_1, g_2, \dots, g_m)$: Bandwidth gap vector, where $g_i$ denotes the communication reciprocal bandwidth at clustering level $i$.
  - $\mathbf{l} = (l_1, l_2, \dots, l_m)$: Synchronization and latency overhead vector at clustering level $i$.
- **Superchip Decomposition**:
  - *Level 1 (Intra-CCX / Intra-SM)*: Extremely low gap $g_1$ and low synchronization latency $l_1$ (local L1/L2 caches and CUDA shared memory).
  - *Level 2 (Intra-Socket / On-Package HBM)*: High bandwidth ($5.3\text{ TB/s}$ on MI300A), intermediate gap $g_2$.
  - *Level 3 (Cross-Engine Interconnect)*: Communication across NVLink-C2C ($900\text{--}1800\text{ GB/s}$) or 4th Gen Infinity Fabric, characterized by gap $g_3 > g_2$ and latency $l_3 > l_2$.
- **Network-Oblivious Optimality**: Under Bilardi et al.'s framework, static recursive matrix multiplications and tensor reduction algorithms that achieve communication optimality on the abstract $\text{D-BSP}$ model transfer directly to optimal performance across diverse superchip fabrics without hardware-specific manual retuning.

---

## 2. AMD Instinct MI300A APU Architecture

### 2.1 Hardware Microarchitecture
The AMD Instinct MI300A is an Accelerated Processing Unit (APU) that merges CPU and CDNA 3 GPU dies onto a 3D-stacked silicon interposer:
- **Shared Memory Subsystem**: 128 GB of stacked HBM3 memory physically shared between all 24 Zen 4 x86 CPU cores and 228 CDNA 3 GPU compute units.
- **Unified Bandwidth**: Both CPU and GPU access the same memory pool at up to **5.3 TB/s** peak bandwidth.
- **Zero PCIe Overhead**: There is no discrete PCIe link separating CPU host memory from GPU device memory.

### 2.2 Addressing Common Misconceptions & Software Traps
- **Misconception**: "To run on MI300A, the runtime must call `hipHostRegister()` with `hipHostRegisterMapped` or `hipMemcpy()` to migrate buffers."
- **Reality**: Calling `hipHostRegister` or `hipMemcpy` forces unnecessary driver page-locking, invalidates translation lookaside buffers (TLBs), and severely degrades performance.
- **Optimal Software Path for `ik_llama.cpp`**:
  1. Set environment variable `HSA_XNACK=1` to enable hardware-managed page faulting across CPU and GPU.
  2. Map GGUF model files directly into process virtual memory using HugeTLB `mmap`.
  3. Pass host pointers directly into HIP tensor GEMM kernels without staging allocations.
  4. Both prompt scheduling and token generation operate directly in 5.3 TB/s HBM3.

### 2.3 Inter-Die Cache Invalidation Mitigation & Non-Temporal Streaming
- **128-Byte Cache Line Separation**: While Zen 4 CPU cores and CDNA 3 CUs physically share the stacked HBM3 pool, they maintain separate L1/L2/L3 cache hierarchies interconnected via 4th Gen Infinity Fabric. Atomic status cursors, token publication queues, and SPSC ring buffers must be aligned with `alignas(128)` to prevent cross-die cache line invalidation storms.
- **Worst-Case Execution Time (WCET) Scratchpad Bounds (Wehmeyer & Marwedel 2006)**:
  Hardware caches optimize average-case execution throughput, but as proven by Wehmeyer & Marwedel (2006), hardware cache replacement policies introduce non-deterministic conflict misses and state invalidations that severely degrade Worst-Case Execution Time (WCET) and inflate $p99.9$ tail latency. Furthermore, inter-task cache interference in concurrent execution destroys latency predictability. In unified superchip architectures, bounding worst-case latency requires software-managed scratchpads (pinned HugeTLB buffers, CUDA shared memory `__shared__`) with deterministic memory allocation rather than relying on unpredictable hardware cache hierarchies.
- **Non-Temporal Vector Stores**: Bulk memory operations (e.g. pre-zeroing KV cache slots or streaming activation buffers) executed on the Zen 4 CPU must employ non-temporal streaming stores (`_mm512_stream_si512` / `_mm256_stream_si256`). This writes directly to HBM3 without allocating cache lines in the Zen 4 L3 cache, preventing the eviction of hot token embeddings and router weights.

---

## 3. NVIDIA Grace Hopper (GH200), Grace Blackwell (GB200), and DGX Spark

### 3.1 Hardware Microarchitecture
- **NVIDIA GH200**: Integrates a 72-core Arm Neoverse-V2 Grace CPU with a Hopper H100 GPU connected via **NVLink-C2C** (Chip-to-Chip), providing **900 GB/s bidirectional coherent bandwidth** (7x faster than PCIe Gen5).
- **NVIDIA GB200**: Connects a Grace CPU to two Blackwell GPUs via NVLink-C2C, delivering **1,800 GB/s bidirectional coherent bandwidth**.
- **NVIDIA DGX Spark (GB10)**: A compact desktop supercomputer powered by the Grace Blackwell GB10 superchip, featuring a 20-core Arm CPU, 1 PFLOPS FP4 Blackwell GPU, and 128 GB unified LPDDR5X memory.

### 3.2 Coherent Memory Mechanisms
1. **Hardware Address Translation Service (ATS)**: The GPU MMU directly translates virtual addresses using the CPU's hardware page tables, creating a single unified address space.
2. **NVLink-C2C Cache Coherence**: The Blackwell GPU can issue standard load/store instructions to Grace LPDDR5X memory at cache-line granularity without explicit DMA copy operations.
3. **Heterogeneous Memory Management (HMM)**: Transparently migrates hot pages to GPU HBM3e while cold pages remain in CPU LPDDR5X.

### 3.3 Tiered MoE Inference Strategy for `ik_llama.cpp`
- **Attention & Shared FFNs**: Kept permanently in GPU HBM3e (4,000–8,000 GB/s) for ultra-low latency autoregressive decode.
- **MoE Routed Experts**: Reside in Grace LPDDR5X memory (up to 480 GB). During token generation, active experts are streamed across NVLink-C2C at **900–1,800 GB/s** (line-rate streaming that is fast enough to keep GPU Tensor Cores saturated without PCIe stalls).
- **KV Cache Offloading**: For 1M+ context lengths, KV caches spill seamlessly into Grace LPDDR5X without requiring intermediate host staging buffers.

### 3.4 System-Scope Synchronization & Arm Non-Temporal Streaming
- **System-Scope Acquire-Release Semantics**: Fine-grained work signaling between the Arm Grace CPU and the Hopper/Blackwell GPU bypasses OS kernel interrupts and heavyweight `cudaDeviceSynchronize()`. Shared synchronization cursors are managed via system-scoped atomics:
  ```cpp
  // GPU warp producer signals completion across NVLink-C2C
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> cursor(*shared_gpu_cpu_cursor);
  cursor.store(batch_seq_id, cuda::memory_order_release);
  ```
  The Arm Neoverse-V2 CPU thread polls using `std::atomic_ref<uint32_t>::load(std::memory_order_acquire)`, compiled into single-cycle `LDAPR` (Load-Acquire RCpc) instructions.
- **Arm Non-Temporal Stores (`STNP`)**: Bulk prompt ingestion buffers and KV cache spills generated on Grace CPU cores utilize non-temporal pair store instructions (`STNP`), bypassing the 1MB private L2 cache per Neoverse-V2 core and preserving L3 cache capacity for concurrent server scheduling.

---

## 4. Learnings from Apple Silicon & MLX

### 4.1 Apple Silicon Unified Memory Architecture & Milner Bigraphs with Sharing (Sevegnani & Calder 2015)
Apple M-series chips (M2/M3/M4 Max/Ultra) feature unified LPDDR5 memory shared between CPU, GPU, and Neural Engine (up to 192 GB @ 800 GB/s on Ultra).
- **Formal Bigraph Placement DAG**: Following Sevegnani & Calder (2015) (*Bigraphs with Sharing*), physical memory allocation in unified architectures cannot be represented by traditional hierarchical tree structures because memory nodes simultaneously reside inside multiple execution domains. Apple Silicon unified memory is modeled as a place DAG: a single contiguous physical allocation `node_mem` has multiple parent loci (`parent_cpu`, `parent_gpu`, `parent_ane`), enabling zero-copy zero-serialization tensor dispatch across heterogeneous engines.

### 4.2 MLX Architectural Insights for `ik_llama.cpp`
Studying Apple's [`mlx`](../scratch/research_superchips/mlx/) framework yields three fundamental design principles:
1. **Lazy Graph Evaluation with Stream Ordering**: Tensors represent evaluated values or deferred computation nodes. Computations are committed to Metal command buffers only when explicitly synchronized or output is needed, enabling automatic inter-kernel operator fusion.
2. **Zero-Copy Metal Buffers**: MLX allocates unified buffers that are concurrently accessible by CPU and GPU pointers without copying or memory-mapping tricks.
3. **Custom GEMV / GEMM SIMD Kernels**: By tuning warp/simdgroup thread assignments to Apple GPU execution widths (32 threads per SIMDgroup), MLX achieves >85% of peak memory bandwidth on autoregressive decoding.

---

## 5. Architectural Action Plan for `ik_llama.cpp`

1. **Auto-Detection of Superchip Topologies**:
   - Detect MI300A via `hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, dev)`.
   - Detect Grace Hopper / Blackwell via CUDA device attribute `cudaDevAttrDirectManagedMemAccessFromHost` and NVLink-C2C bus IDs.
2. **Unified Buffer Backend**:
   - Introduce `GGML_BACKEND_UNIFIED` that skips host-to-device staging allocations on superchip hardware, directly dispatching kernels against host virtual addresses.
3. **NVLink-C2C Tiered Expert Streaming**:
   - Implement an asynchronous streaming loader for MoE experts across NVLink-C2C, utilizing Hopper/Blackwell Tensor Memory Accelerator (TMA) and asynchronous copy instructions (`cp.async`).
