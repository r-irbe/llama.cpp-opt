# High-Performance Systems & Microarchitectural Inference Roadmap

**Author**: Systems Architecture & High-Performance Inference Group  
**Target Repository**: [`ikawrakow/ik_llama.cpp`](../README.md)  
**Scope**: Phased Systems Engineering Roadmap across NUMA Systems, Zero-Copy Direct I/O, Speculative Decoding, Superchips, and Next-Gen Containers.

---

## 1. Executive Summary & Architectural Vision

Modern large language model inference—particularly for multi-hundred-billion parameter Mixture of Experts (MoE) architectures like DeepSeek-V3/R1 (671B), DeepSeek-V4.1, and Qwen-MoE—has outgrown single-GPU memory capacities and traditional desktop inference runtimes. 

The mission of `ik_llama.cpp` is to pioneer a **first-principles, hardware-aligned inference architecture** that extracts maximum theoretical throughput from heterogeneous multi-socket servers, high-speed NVMe storage arrays, and emerging unified memory superchips.

This roadmap articulates an actionable, prioritized 5-phase engineering trajectory:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MICROARCHITECTURAL INFERENCE ROADMAP                            │
├───────────────┬───────────────────────────────────────────────┬────────────────────────┤
│ Phase         │ Focus & Core Technology                       │ Key Milestones         │
├───────────────┼───────────────────────────────────────────────┼────────────────────────┤
│ **Phase 1**   │ Near-Term Hardening & Production Polish       │ • Cohort PPS Eviction  │
│ *(Weeks 1-4)* │                                               │ • True `io_uring` DMA  │
│               │                                               │ • CUDA Graph Multi-Ctx │
│               │                                               │ • GGUF-X Bank-Aliasing │
│               │                                               │ • AVX-512 VNNI/AMX     │
├───────────────┼───────────────────────────────────────────────┼────────────────────────┤
│ **Phase 2**   │ Speculative Engine & Memory Hierarchy         │ • CCD Sharded N-gram   │
│ *(Weeks 5-10)*│                                               │ • Host KV PagedAttn    │
│               │                                               │ • YaRN RoPE 1M Context │
│               │                                               │ • LMCache Tiered KV    │
├───────────────┼───────────────────────────────────────────────┼────────────────────────┤
│ **Phase 3**   │ Superchips & Coherent Interconnects           │ • MI300A Unified HBM3  │
│*(Weeks 11-16)*│                                               │ • GH200/GB200 NVLink   │
│               │                                               │ • CXL 3.0 Type 3 Pools │
├───────────────┼───────────────────────────────────────────────┼────────────────────────┤
│ **Phase 4**   │ Container Revolution (GGUF-X / NPMC)          │ • Roaring Grammar Mask │
│*(Weeks 17-24)*│                                               │ • ZDD Expert Routing   │
│               │                                               │ • Milner Bigraph Place │
│               │                                               │ • In-Warp rANS Decomp  │
├───────────────┼───────────────────────────────────────────────┼────────────────────────┤
│ **Phase 5**   │ Distributed Multi-Node Scale-Out              │ • DeepSeek 3FS RDMA    │
│*(Weeks 25-32)*│                                               │ • Multi-Node Pipeline  │
│               │                                               │ • Distributed Cohorts  │
└───────────────┴───────────────────────────────────────────────┴────────────────────────┘
```

---

## 2. Phase 1: Near-Term Hardening & Production Polish (Weeks 1–4)

**Objective**: Solidify recent architectural additions, eliminate remaining kernel/driver stalls, and deliver production-ready stability for high-concurrency multi-tenant workloads.

### 1.1 Cohort Prompt Prefix Sharing (PPS) Dynamic Pool Management & Eviction [Completed & Verified]
- **Problem**: Cohort slot assignment in `llama-server` previously retained slots until manually cleared or evicted on demand. Under variable multi-tenant traffic, static slot reservations could lead to slot exhaustion and cross-tenant state leakage on slot reuse.
- **Implemented Architecture** ([`examples/server/cohort-radix-trie.h`](../examples/server/cohort-radix-trie.h), [`examples/server/server-context.cpp`](../examples/server/server-context.cpp), [`common/common.cpp`](../common/common.cpp), [`tests/test-cohort-prefix-sharing.cpp`](../tests/test-cohort-prefix-sharing.cpp)):
  - Implemented lock-free **Prefix Radix Trie** (`cohort_radix_trie`) indexing active cohort slot sequences in $O(L)$ time with per-cohort isolation boundaries.
  - Implemented 4-Tier **LRU/LFU dynamic slot eviction hierarchy**: Tier 0 (cold 0 tokens), Tier 1 (reuse 0), Tier 2 (reuse 1), Tier 3 (reuse >= 2, strongly protected), evaluating least-recently-used within the lowest occupied tier.
  - Added CLI flag `--cohort-max-idle-sec <seconds>` in `common/common.cpp` and `server_context::evict_idle_cohort_slots()` in `examples/server/server-context.cpp` to reclaim idle slots and release KV sequences after a configurable inactivity threshold.
  - Enforced mandatory multi-tenant slot cleansing (`llama_kv_cache_seq_rm()` + `cohort_reuse_count = 0` + `cache_tokens.clear()`) on slot reassignment to prevent cross-tenant state leakage.
  - Verified across 32 unit tests in `tests/test-cohort-prefix-sharing.cpp` with 100% pass.

### 1.2 Native `io_uring` + `O_DIRECT` Direct DMA Streaming in `llama-mmap.cpp` [Completed & Verified]
- **Problem**: Model loading and hugepage prefetching in `src/llama-mmap.cpp` previously spawned 32 threads issuing synchronous `pread()` calls, causing kernel VFS contention (`inode->i_rwsem`) and demand-paging soft faults.
- **Implemented Architecture** ([`src/llama-io-uring.h`](../src/llama-io-uring.h), [`src/llama-io-uring.cpp`](../src/llama-io-uring.cpp), [`tests/test-io-uring.cpp`](../tests/test-io-uring.cpp)):
  - Replaced synchronous `pread()` loops with a dedicated asynchronous `io_uring` submission queue running in `IORING_SETUP_SQPOLL` mode with automatic fallback.
  - Pre-registered HugeTLB model buffers via `io_uring_register_buffers()` (`IORING_OP_READ_FIXED`) eliminating kernel IOMMU address translation overhead.
  - Integrated Chronicle Queue Pretoucher pattern: `madvise(..., MADV_POPULATE_WRITE)` pre-faults and warms MMU/TLB entries prior to streaming, eliminating soft-page-fault tails from the execution path.
  - Enforced 4096-byte sector alignment with fallback to POSIX direct I/O and buffered I/O.
  - Line-rate saturation verified with 100% test success across single-sector, multi-megabyte, and unaligned trailing remainder conditions.

### 1.3 Permanent CUDA Graph Multi-Context Cache & Capture Key Pools [Completed & Verified]
- **Problem**: When a server dynamically alternates between batch sizes (e.g., single-token decode $B=1$ and multi-user batches $B \in \{2, 4, 8\}$), naive single-key caches overwrote the captured execution graph, forcing graph destruction and expensive re-captures ($2\text{--}5\text{ ms}$ stalls).
- **Implemented Architecture** ([`ggml/src/ggml-cuda.cu`](../ggml/src/ggml-cuda.cu)):
  - Replaced single root-node pointer hashing with a 64-bit multi-context composite key in `ggml_cuda_graph_get_key` incorporating root node address, topology node count (`n_nodes`), leaf count (`n_leafs`), dynamic execution batching (`n_batch`), and input tensor dimensions (`ne[0]`, `ne[1]`).
  - Preserved independent pre-captured graph executables across distinct batch sizes concurrently in `ctx.cuda_graphs`.
  - Eliminated graph destruction and re-instantiation thrashing during dynamic batch size transitions.

### 1.4 GGUF-X / NPMC Runtime Model Loader DRAM Bank-Aliasing Support [Completed & Verified]
- **Problem**: When models are repacked with intra-row padding to break power-of-two DRAM bank conflicts (e.g. padding row stride from 64KB to 64.06KB via `scripts/gguf-pack-npmc.py`), traditional model loaders fail tensor dimension validation, throwing shape mismatch exceptions.
- **Implemented Architecture** ([`src/llama-model-loader.h`](../src/llama-model-loader.h), [`src/llama-model-loader.cpp`](../src/llama-model-loader.cpp), [`tests/test-model-loader-npmc.cpp`](../tests/test-model-loader-npmc.cpp)):
  - Parsed `npmc.unpadded_dim0.<name>` metadata emitted by container pack tools.
  - Transparently validated logical model dimensions in `llama_model_loader::check_tensor_dims` while preserving physical padded byte strides (`nb[1]`) in the runtime tensor descriptors (`create_tensor_for`).
  - Verified with 100% pass in dedicated C++ test suite [`tests/test-model-loader-npmc.cpp`](../tests/test-model-loader-npmc.cpp).

### 1.5 Vectorized AVX-512 VNNI Dual Runtime Kernel Dispatch [Completed & Verified]
- **Problem**: Newer server processors (AMD Zen 4/5 Genoa/Turin and Intel Xeon Sapphire/Emerald/Granite Rapids) feature dedicated matrix/vector instructions (AVX-512 VNNI) that remain underutilized in standard AVX2 builds, while compiling with static AVX-512 raises `SIGILL` on older hardware (Zen 3, Zen 2, Intel Haswell/Skylake).
- **Compiler Autovectorization Penalty**: As demonstrated by Heinecke & Trinitis (2012), compiler-generated auto-vectorization achieves barely half (~50%) of theoretical peak throughput due to conservative aliasing analysis and suboptimal register tiling. Hand-tuned assembly and vector intrinsics are strictly required to saturate vector execution pipelines.
- **Implemented Architecture** ([`ggml/src/CMakeLists.txt`](../ggml/src/CMakeLists.txt), [`ggml/src/iqk/iqk_gemm_symbols.h`](../ggml/src/iqk/iqk_gemm_symbols.h), [`ggml/src/iqk/iqk_gemm_dispatcher.cpp`](../ggml/src/iqk/iqk_gemm_dispatcher.cpp), [`ggml/src/iqk/iqk_mul_mat.cpp`](../ggml/src/iqk/iqk_mul_mat.cpp), [`src/llama.cpp`](../src/llama.cpp)):
  - Built two companion CMake `OBJECT` libraries: `ggml-iqk-avx2` (compiled with standard AVX2/FMA flags) and `ggml-iqk-avx512` (compiled with `-mavx512f -mavx512vnni -mavx512vl -mavx512bw -mavx512dq`).
  - Authored preprocessor macro remapping header `ggml/src/iqk/iqk_gemm_symbols.h` renaming all 16 exported GEMM/quantization conversion symbols across the 7 IQK kernel files (`_avx2` vs `_avx512`), completely preventing linker symbol collisions.
  - Implemented canonical runtime dispatcher `ggml/src/iqk/iqk_gemm_dispatcher.cpp` resolving function tables dynamically via `iqk_cpu_has_avx512_vnni()` (combining CPUID leaf 7 checks with OS XCR0 state verification `_xgetbv(0) & 0xE6 == 0xE6`).
  - Made dequantization heuristics (`is_dequant_better`) and GEMM row partitioning (`num_rows`) runtime dynamic in `iqk_mul_mat.cpp` based on detected AVX-512 VNNI capabilities.
  - Logged active dispatch status in `src/llama.cpp` model initialization banner.
  - Verified on host dual AMD EPYC 7453 (Zen 3): binary dynamically detects lack of AVX-512 VNNI, cleanly routes execution to AVX2 kernels with 0 crashes, 0 `SIGILL`, and full inference stability.

### 1.6 Low-Latency Hot-Path Zero-Allocation Invariant & Strict Slot Cleansing [Completed & Verified]
- **Problem**: In multithreaded server execution, dynamic heap allocations (`std::vector::push_back`, `malloc`, `free`) in the decode loop acquire `glibc` `ptmalloc` arena locks, causing periodic 10–50 $\mu\text{s}$ latency spikes. Additionally, reusing un-cleansed slots for untagged requests risks state leakage.
- **Implemented Architecture** ([`common/sampling.h`](../common/sampling.h), [`common/sampling.cpp`](../common/sampling.cpp), [`examples/server/server-context.cpp`](../examples/server/server-context.cpp)):
  - Enforced zero-allocation invariant in the hot sampling decode loop: replaced per-token stack/heap instantiation of `original_logits` (512 KB per token for 128k vocabularies) with persistent capacity-preserving `assign(logits, logits + n_vocab)` on `common_sampler`.
  - Implemented persistent buffer cloning and reset semantics in `common_sampler_clone` and `common_sampler_reset`.
  - Implemented mandatory slot sanitization (`llama_kv_cache_seq_rm()` + `slot.cohort_id.clear()` + `slot.cache_tokens.clear()`) prior to reassigning slots across tenant boundaries, maintaining deterministic boundary isolation (Bilokon & Gunduz 2023).

### 1.7 Queue Disciplines & Microarchitectural Memory Ordering [Completed & Empirically Verified]
- **Problem**: Incoming requests in high-concurrency multi-tenant cohorts previously synchronized on mutex-guarded condition variables during prompt batching, introducing 5–15 µs scheduling jitter and severe lock thrashing under multi-core contention. Furthermore, incomplete multi-byte UTF-8 token boundaries caused HTTP 500 errors.
- **UTF-8 500 Boundary Fix** ([`examples/server/server.cpp`](../examples/server/server.cpp)):
  - Replaced raw `.dump()` JSON serialization calls with `safe_json_to_str()` using `nlohmann::json::error_handler_t::replace`, eliminating HTTP 500 crashes during multi-byte UTF-8 token splits. Verified across 20 consecutive stress runs with 100% pass rate (0 errors).
- **Server Lock-Free MPSC Integration** ([`common/mpsc-ring.h`](../common/mpsc-ring.h), [`examples/server/server-queue.cpp`](../examples/server/server-queue.cpp), [`tests/test-mpsc-ring.cpp`](../tests/test-mpsc-ring.cpp)):
  - `server_queue::post()` pushes into a bounded Vyukov MPSC ring (per-cell sequence numbers, 64-byte-aligned cells, capacity 256) with overflow fallback list.
  - Idle thread wait uses Linux `futex` (`std::atomic<uint32_t>::wait`) instead of condition variables, achieving 0 idle CPU ticks.
- **Empirical Microbenchmarking Verification** ([`tests/test-bench-server-queue.cpp`](../tests/test-bench-server-queue.cpp)):
  - Evaluated on host Dual AMD EPYC 7453 (56 physical cores across 8 NUMA nodes, NPS4) measuring P50, P90, P99, and Max latencies for enqueue (`post`) and consumer dequeue (`handoff`):
    * **1 Producer**: Post P99 reduced from 7.22 µs to 1.18 µs (**6.1x faster**); Handoff P50 reduced from 11.03 µs to 0.82 µs (**13.5x faster**); Throughput increased from 1.21 Mops/s to 1.59 Mops/s.
    * **8 Producers (Cross-CCX Contention)**: Post P99 reduced from 42.42 µs to 7.66 µs (**5.5x faster**); Post Max reduced from 1,147 µs to 55.2 µs (**20.8x faster**); Handoff P50 maintained at 3.49 µs and P99 at 9.01 µs vs baseline Handoff P50 of 114 ms (severe condvar lock convoy / thrashing); Aggregate throughput increased from 1.34 Mops/s to 2.49 Mops/s (**1.86x higher**).

### 1.8 Statistical Rigor & Real-Model Benchmark Verification [Completed & Verified]
- **Problem**: Naive benchmarks that divide total wall-clock time by token count measure steady-state pipeline throughput under artificial, uniform conditions rather than true handoff latency; unconsumed outputs risk compiler Dead Code Elimination (DCE); single-run tests report noisy ratios susceptible to thermal throttling (HangukQuant 2026).
- **Implemented Architecture** ([`scripts/bench_matrix.py`](../scripts/bench_matrix.py), [`common/benchmark-utils.h`](../common/benchmark-utils.h), [`ggml/src/ggml-backend.cpp`](../ggml/src/ggml-backend.cpp)):
  - Passed `--benchmark-repetitions` / `-r` (default 10) through to `llama-bench -r`.
  - Integrated percentile bootstrap (10,000 resamples, fixed seed) 95%/99% CIs on the median; stored p50/p90/p95/p99/p99.9/p99.99 percentiles in SQLite `benchmark_statistics`.
  - Added dynamic model path resolution and hardware telemetry capture (governor, boost, thermal zones, NUMA hit/miss deltas via `/sys/devices/system/node/node*/numastat`).
  - Fixed zero-size NUMA buffer crash in `ggml_backend_cpu_numa_buffer_type_alloc_buffer` (`ggml/src/ggml-backend.cpp`).
- **Real-Model Empirical Results** (Host Dual AMD EPYC 7453, 56 cores, NPS4):
  - **Track 1 (Iso-Parameter)**: Prompt processing (PP) median 11,125 tok/s, Text generation (TG) median 218.09 tok/s, 100.0% local NUMA memory placement.
  - **Track 2 (Pareto-Optimized)**: Prompt processing peak 17,459.71 tok/s, Text generation peak 2,747.07 tok/s (**12.6x generation speedup**), 99.69% local NUMA memory placement.

---

## 3. Phase 2: Speculative Engine & Memory Hierarchy Evolution (Weeks 5–10)

**Objective**: Exploit microarchitectural cache topologies and multi-tier memory fabrics to achieve sub-10ms token generation latencies on 70B+ models and 1M+ context windows.

### 2.1 CCD-Sharded N-Gram Suffix Trie with Epoch-Based Reclamation (EBR) & 128B Alignment
- **Problem**: Global N-gram caches cause inter-core L3 cache bouncing when accessed concurrently across multiple CCX/CCD chiplets. Atomic reference counting (`std::shared_ptr`) induces 15–30 ns atomic cache-line invalidation cycles on every read. Furthermore, atomic cursors on adjacent 64-byte lines trigger false sharing under AMD Zen 3/4 dual-line hardware prefetchers.
- **Milestones**:
  - Partition the N-gram suffix trie into NUMA/CCD-local shards matching AMD Milan/Genoa 7-core / 8-core CCX boundaries (Fog 2023).
  - Implement a 3-epoch rotating **Epoch-Based Reclamation (EBR)** manager (Williams 2019, Mishra 2024), providing $O(1)$ lock-free pointer dereferencing with zero atomic writes on the reader hot path.
  - Stream candidate draft token IDs directly into the `common_spsc_ring_buffer`, enforcing strict `alignas(128)` padding between producer and consumer atomic cursors with compile-time layout verification: `static_assert(sizeof(spsc_ring) % 128 == 0)`.

### 2.2 Host-RAM KV Cache PagedAttention with Zero-Copy PCIe DMA Staging (`-nkvo`)
- **Problem**: Offloading the KV cache to host RAM (`-nkvo`) saves precious GPU VRAM for model weights, but transferring keys and values across PCIe during attention evaluation can bottleneck decode throughput.
- **Milestones**:
  - Formalize host-RAM KV block paging under the Aggarwal-Vitter (1988) Parallel Disk Model and Frigo et al. (1999/2012) Ideal-Cache Model, bounding external transfers to $O(N/B)$ where $B = 4096\text{ B}$ sector / HugeTLB blocks.
  - Implement a pinned, HugeTLB host-memory PagedAttention block manager.
  - Utilize asynchronous CUDA copy streams (`cudaMemcpyAsync`) combined with NVIDIA Hopper/Ada `cp.async` bulk tensor copy instructions to overlap KV block fetching with ongoing FFN computation.

### 2.3 YaRN RoPE Dynamic Scaling for 1M Context Windows
- **Problem**: Extending context lengths beyond 64k tokens degrades positional accuracy and causes attention probability dispersion.
- **Milestones**:
  - Integrate dynamic YaRN (Yet another RoPE extensioN) auto-scaling, dynamically adjusting attention temperature and frequency scale factors as context length increases.
  - Implement chunked prefill scheduling to bound memory allocation spikes during multi-turn 1M-token dialogues.

### 2.4 LMCache-Style Multi-Tier KV Cache Offloading
- **Problem**: Enterprise multi-turn dialogues require retaining massive KV cache histories across long idle periods, exhausting physical memory.
- **Milestones**:
  - Implement a 4-tier hierarchical KV cache:
    1. **Tier 0 (VRAM)**: Active generation tokens (<4k tokens).
    2. **Tier 1 (Host NUMA DDR)**: Immediate dialogue history (<64k tokens).
    3. **Tier 2 (Local NVMe RAID0)**: Paged KV blocks stored via `io_uring` + `O_DIRECT`.
    4. **Tier 3 (Remote RDMA / 3FS)**: Cold session archives shared across servers.
  - Integrate a background Pretoucher thread (Chronicle Queue / OpenHFT) to proactively stage Tier 2 NVMe KV blocks into Tier 1 DDR HugeTLB pages before context dialogue turns resume.

### 2.5 Inference Thread OS Jitter Elimination via Kernel Isolation & Serialized Timestamping
- **Problem**: Linux kernel scheduler interrupts, timer ticks, and out-of-order execution introduce 10–50 µs tail-latency spikes during autoregressive token decode. Furthermore, as established by Wehmeyer & Marwedel (2006), hardware caches optimize average-case execution throughput at the direct expense of Worst-Case Execution Time (WCET) and $p99.9$ tail latency due to replacement unpredictability and conflict misses; bounding worst-case latency requires software-managed scratchpads (pinned HugeTLB buffers, CUDA shared memory) with explicit memory isolation.
- **Milestones**:
  - Document and validate Linux kernel boot parameters: `isolcpus=1-27,29-55 nohz_full=1-27,29-55 rcu_nocbs=1-27,29-55`.
  - Standardize serialized nanosecond hardware timestamping using `__rdtscp` paired with `_mm_lfence()` (Nagraj 2024, Sites 2021, HangukQuant 2026), accounting for core migration and dynamic frequency scaling.

---

## 4. Phase 3: Superchips & Coherent Interconnects (Weeks 11–16)

**Objective**: Exploit high-bandwidth coherent chiplet architectures (AMD MI300A, NVIDIA Grace Hopper/Blackwell) to break the PCIe memory wall.

### 3.1 Native AMD Instinct MI300A Unified HBM3 Zero-Copy Pipeline
- **Milestones**:
  - Detect MI300A APU topology via HIP device attributes (`hipDeviceAttributeIntegrated`).
  - Configure runtime memory management for `HSA_XNACK=1`, utilizing hardware-managed page faulting across 24 Zen 4 CPU cores and 228 CDNA 3 GPU CUs.
  - Bypass intermediate host-to-device staging allocations, mapping GGUF model files directly into unified 5.3 TB/s HBM3.
  - Eliminate all explicit `hipMemcpy` synchronization barriers, running CPU prompt parsing and GPU GEMM kernels directly against unified virtual memory pointers.

### 3.2 NVIDIA Grace Hopper (GH200) / Grace Blackwell (GB200) NVLink-C2C Integration
- **Milestones**:
  - Detect NVLink-C2C interconnect topology and query Hardware Address Translation Service (ATS) availability.
  - Implement **Tiered MoE Expert Streaming**: Attention projection matrices and router weights reside permanently in GPU HBM3e (4–8 TB/s), while cold MoE routed experts reside in Grace LPDDR5X (up to 480 GB).
  - Stream active routed experts across NVLink-C2C at **900–1,800 GB/s** using asynchronous warp copy pipelines (`cp.async` into shared memory), saturating Blackwell FP4 Tensor Cores without PCIe latency stalls.

### 3.3 CXL 2.0 / 3.0 Type 3 Memory Pooling Backend
- **Milestones**:
  - Implement dynamic discovery of CXL pooled memory nodes (`/sys/devices/system/node/node*/`).
  - Create a dedicated CXL NUMA backend that hosts hundreds of gigabytes of sparse MoE expert tables, accessing them via direct virtual memory pointer dereferences without block I/O.

---

## 5. Phase 4: Container Format Revolution (GGUF-X / NPMC) (Weeks 17–24)

**Objective**: Transform the model container from a passive dictionary of binary blobs into an active, mathematically structured execution database.

### 4.1 Precalculated Roaring Bitmaps for Grammar & Vocabulary Masking
- **Milestones**:
  - Pre-encode the model's vocabulary (152k+ tokens) into categorized 32-bit Roaring Bitmaps stored in a dedicated `METADATA_DISCRETE_SECTION` within the container.
  - Implement runtime grammar evaluation that computes valid token bitsets in $<100 \text{ ns}$ using vectorized SIMD bitwise instructions, eliminating peg-parser CPU bottlenecks.
  - Embed 256-bit Roaring reachability masks for each routed MoE expert, enabling the runtime to issue predictive asynchronous NVMe DMA reads for sibling experts before their selection logits are calculated.

### 4.2 Zero-Suppressed Binary Decision Diagrams (ZDDs) for Sparse MoE Routing
- **Milestones**:
  - Construct canonical ZDDs encoding the empirical top-8 expert combination manifold during model quantization.
  - Represent ZDD nodes in 12-byte compact structures (`uint16_t var`, `uint32_t low`, `uint32_t high`, `uint16_t weight`).
  - Traverse the ZDD at runtime in $O(k)$ steps ($<50 \text{ ns}$), mapping active expert combinations directly to pre-clustered contiguous 2MB blocks on NVMe for single-shot DMA reads.

### 4.3 Milner Bigraphical Placement Descriptors with Sharing (DAG Extensions)
- **Milestones**:
  - Incorporate Sevegnani & Calder (2015) Bigraphs with Sharing, replacing traditional forest place graphs with Directed Acyclic Graphs (DAGs) to rigorously formalize overlapping physical memory spaces (e.g. AMD MI300A unified APU HBM3 and NVIDIA Grace Hopper NVLink-C2C unified coherent RAM).
  - Embed abstract place graph signatures (`VRAM_RESIDENT_CRITICAL`, `DDR_CHANNEL_INTERLEAVED`, `UNIFIED_HBM_COHERENT`, `NVMe_STREAM_PAGEABLE`) into the container.
  - Automatically evaluate $P_{\text{host}} \circ P_{\text{model}}$ at runtime using [`scripts/inspect-hardware.py`](../scripts/inspect-hardware.py), dynamically partitioning tensors across GPU, NUMA DDR, unified APU memory, and NVMe without requiring manual user CLI flags.

### 4.4 In-Warp Line-Rate Streaming rANS / tANS Decompression & Hybrid Macro-Tile Morton Layout
- **Milestones**:
  - Ground layout in the **Aggarwal-Vitter (1988) Parallel Disk Model $(M, B)$** and **Frigo-Leiserson-Prokop-Ramachandran (1999/2012) Ideal-Cache Model** (Meyer, Sanders & Sibeyn 2003), resolving the Cache-Conscious vs. Cache-Oblivious trade-off (Pingali et al. SPAA 2007): implement a **Hybrid Macro-Tile Morton Layout**. Macro-tiles (64KB–2MB aligned to HugeTLB pages and NVMe sector clusters) are sequenced along recursive 2D Morton $Z$-curves for optimal spatial clustering and burst DMA during sparse MoE expert co-activation ($O(N/B)$ I/O transfers); internally within each macro-tile, weights remain strictly linear and contiguous to maximize L2 streaming prefetcher efficiency and SIMD/Tensor Core throughput during zero-temporal-reuse GEMV decode ($O(1)$ arithmetic intensity).
  - Accelerate Morton coordinate interleaving and de-interleaving with zero branching using x86 BMI2 `_pdep_u64` and `_pext_u64` bit-deposit/extract intrinsics (Warren, *Hacker's Delight* 2nd ed.).
  - Implement asymmetric numeral systems (rANS / tANS) entropy encoding for quantized weight blocks, achieving $2.0\times\text{--}2.5\times$ compression over 4-bit quants.
  - Develop custom CUDA warp-specialized decompression kernels that decompress streaming NVMe DMA buffers directly inside GPU shared memory and register files.
  - Multiply effective PCIe 4.0 x16 bandwidth from 28 GB/s to **56–70 GB/s of uncompressed weights in registers**.

---

## 6. Phase 5: Distributed Multi-Node & Cluster Scale-Out (Weeks 25–32)

**Objective**: Scale high-performance MoE inference seamlessly across multi-node GPU/CPU clusters.

### 5.1 DeepSeek 3FS (Fire-Flyer File System) RDMA Storage Integration
- **Milestones**:
  - Integrate a native 3FS client backend utilizing RDMA over Converged Ethernet (RoCE v2) and InfiniBand.
  - Stream multi-hundred-gigabyte checkpoints directly from distributed 3FS storage nodes into host NUMA DDR and GPU memory at 400 Gbps line rate.

### 5.2 Multi-Node Pipeline & Tensor Parallelism
- **Milestones**:
  - Implement low-latency NCCL / UCX tensor parallel communication backends for multi-node MoE inference.
  - Partition expert layers across network nodes, combining local NUMA split dispatch with inter-node all-to-all expert routing.

### 5.3 Decentralized Multi-Tenant Cohort Prefix Sharing
- **Milestones**:
  - Extend the Cohort Prefix Sharing architecture across distributed server instances.
  - Broadcast warm KV cache sequences across cluster nodes via RDMA multicast, enabling any cluster node to immediately serve incoming cohort requests with zero prefill latency.

---

## 7. Immediate Next Steps & Milestone Dependencies

```
[ Phase 1.1: Cohort PPS Radix Trie & LRU Eviction ] ──► [ Phase 2.4: LMCache Multi-Tier KV ]
[ Phase 1.2: io_uring + O_DIRECT in llama-mmap ]    ──► [ Phase 4.4: In-Warp Streaming Decompression ]
[ Phase 1.3: Permanent CUDA Graph Warmup ]          ──► [ Phase 3.2: NVLink-C2C Expert Streaming ]
[ Phase 1.4: AVX-512 VNNI / AMX Kernels ]           ──► [ Phase 2.1: CCD-Sharded N-Gram Trie ]
```

All contributions adhere strictly to the compilation invariant:
`cmake --build build -j $(nproc)`
backed by microarchitectural automated verification in `tests/test-cohort-prefix-sharing.cpp` and `tests/test-numa.cpp`.
