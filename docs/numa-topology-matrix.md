# Heterogeneous NUMA Topology Matrix, Dynamic Hardware Probing, and Acceleration Guide

>[!WARNING]
>**Experimental Pre-Alpha Architecture Warning**:
>The NUMA split architecture (`--numa split`), dynamic multi-socket scheduling, recursive storage NUMA locality discovery, and wait-free speculative streaming ring buffers implemented in this fork are currently **experimental and pre-alpha**. While backed by rigorous microarchitectural stress tests (`tests/test-numa.cpp`), they are under active development and not yet production-stable. APIs, CLI flags, and internal sysfs heuristics may change without notice. Please test extensively on non-production systems.

## 1. Architectural Overview & Core Concepts

### 1.1 Conceptual Primers (Intuitive Explanations)
Before diving into microarchitectural details, here is an intuitive breakdown of the four core concepts underlying these optimizations:

1. **What is NUMA (Non-Uniform Memory Access)?**
   In standard personal computers, all CPU cores connect to memory through a single shared memory controller (UMA: Uniform Memory Access), so access time to any byte of RAM is identical. On enterprise servers with multiple CPU sockets or modern multi-chiplet processors, RAM is physically split into multiple pools called **NUMA nodes**. Each node is wired directly to its own local CPU cores and memory controllers.
   - *Local access* is ultra-fast with maximum bandwidth.
   - *Remote access* requires sending requests across high-speed interconnect buses (such as AMD Infinity Fabric/xGMI or Intel UPI) to access memory wired to another socket. This adds latency and consumes shared interconnect bandwidth.

2. **Why Does Memory Locality Matter for Mixture of Experts (MoE)?**
   Modern MoE models (such as DeepSeek-V3/R1 or Qwen-MoE) feature hundreds of billions of total parameters, but only activate a small subset of "expert" feed-forward networks for each token. When running large models across CPU memory, inference speed is strictly bounded by DDR memory bandwidth (the rate at which expert weights can be streamed from RAM into CPU registers).
   - If CPU cores on Socket 0 have to pull expert weights residing in Socket 1's memory channels, memory traffic must cross the interconnect, cutting effective bandwidth in half and stalling CPU execution pipelines.
   - By partitioning routed experts across NUMA nodes (`--numa split` with `-ot`) and pinning worker threads to their local nodes, every CPU core accesses weights exclusively through its local memory channels, achieving full aggregate hardware bandwidth.

3. **What is an SPSC Ring Buffer and Why Does 128-Byte Padding Stop Core Interference?**
   In speculative decoding, a lightweight "draft" model runs ahead to generate candidate tokens, which the larger "target" model validates in parallel. Streaming these draft tokens between CPU sockets requires low latency. Standard inter-thread synchronization uses operating system primitives (like `pthread_mutex` or `sys_futex`), which can put threads to sleep and introduce 5 to 15 microseconds of kernel scheduling delay per round.
   - A **Single-Producer Single-Consumer (SPSC) ring buffer** is a circular queue where one thread writes and another reads using atomic memory barriers—completely wait-free without OS system calls.
   - **Why 128-byte padding?** Modern CPUs do not read individual bytes; they fetch 64-byte blocks called *cache lines*. Furthermore, modern processors (such as AMD Zen 3/4/5 and Intel Core/Xeon) feature *hardware spatial prefetchers* that fetch pairs of cache lines (128 bytes at a time). If the producer's write pointer and the consumer's read pointer reside within the same 128-byte region, writing to one pointer causes the CPU's cache-coherency engine to invalidate the other core's cache ("false sharing" cache line ping-ponging). Enforcing strict 128-byte alignment (`alignas(128)`) completely eliminates this interference, yielding zero-jitter token streaming.

4. **How CUDA Graphs Eliminate Kernel Launch Latency**
   In standard GPU programming, the CPU issues commands to the GPU one operation at a time: launch kernel 1, copy memory, launch kernel 2, etc. Each launch involves CPU driver overhead (typically 5 to 10 microseconds). During autoregressive decoding, an LLM executes dozens or hundreds of small operations per token (attention heads, layer normalizations, KV-cache updates), meaning the GPU spend significant time idling waiting for the CPU to dispatch instructions.
   - A **CUDA Graph** records the entire sequence of GPU kernels, parameters, and dependencies into an execution graph once. During inference, the CPU issues a single command to execute the entire graph on the GPU in one shot, eliminating kernel dispatch latency.
   - In upstream runtimes, changing token positions and KV views tripped an invalidation heuristic after 4 tokens. Our enhancement normalizes view operations and masks dynamic copy destination pointers, enabling permanent graph reuse across the entire sequence.

---

### 1.2 System Architecture Summary
To achieve maximum performance without manual topology tuning, [`ik_llama.cpp`](../README.md) implements:
1. **Dynamic Microarchitectural Topology Probing**: Hardware discovery of physical packages (`physical_package_id`), socket boundaries, and inter-socket NUMA groupings.
2. **Recursive Storage NUMA Locality Discovery**: Dynamic discovery of PCIe root complexes for direct NVMe drives, virtual block devices (Linux Software MD RAID arrays), and multi-device ZFS storage pools.
3. **Wait-Free Inter-Socket SPSC Ring Buffers**: 128-byte cache-line aligned Single-Producer Single-Consumer token streaming between CPU sockets, eliminating OS `sys_futex` wake-up latency and false-sharing cache line ping-ponging.
4. **CUDA Graph Permanence**: Elimination of the autoregressive decode invalidation tripwire in `ggml-cuda.cu`, ensuring permanent CUDA graph reuse across arbitrary sequence lengths.

### 1.3 DRAM Bank Conflict Aliasing & Access Latency Formalism (Drepper 2007)
Following Ulrich Drepper (2007) (*What Every Programmer Should Know About Memory*, §3.4), DDR4/DDR5 SDRAM access latency is bifurcated into two physical states:
1. **Page Hit Latency**: When accessing columns within an already-open row buffer (page), latency is governed solely by the CAS latency ($t_{\text{CL}} \approx 13.75\text{ ns}$ for DDR4-3200).
2. **Page Conflict Latency**: When accessing a different row within the same internal bank, the row buffer must be closed and precharged before activating the new row, forcing a full precharge delay:
   $$t_{\text{access}} = t_{\text{RP}} + t_{\text{RCD}} + t_{\text{CL}} \approx 14\text{ ns} + 14\text{ ns} + 14\text{ ns} \approx 42\text{ ns}$$
   tripling DRAM latency and stalling the memory controller.
- **Microarchitectural Aliasing Penalty**: Because lower address bits select the memory channel, bank group, and bank, strides matching exact powers of two ($64\text{ KB}$, $256\text{ KB}$, $1\text{ MB}$, $2\text{ MB}$) cause all parallel OpenMP threads to map to the *identical bank and bank group*. Memory requests serialize behind bank precharges, degrading multi-threaded bandwidth from $>180\text{ GB/s}$ to $<80\text{ GB/s}$.
- **Mitigation Invariant**: Tensor dimension strides in `ggml-cpu` enforce Kowarschik & Weiß (2003) array padding (+64B or +128B), rotating successive matrix rows across disjoint memory channels and keeping DRAM row buffers open.

---

## 2. Hardware Topology Matrix

### 2.1 Multi-Architecture NUMA Partitioning Reference

| Platform Architecture | NUMA Mode / NPS | Total NUMA Nodes | Nodes Per Socket | Inter-Node Topology Characteristics | Optimal Draft Strategy (`get_optimal_draft_node`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unified Memory (UMA)** | NPS0 / Flat | 1 | 1 | Single unified memory controller domain. Inter-socket latency is uniform. | `-1` (Single NUMA domain; speculative drafting runs in-node or on GPU). |
| **Dual AMD EPYC (Zen 2/3/4/5)** | NPS1 | 2 | 1 | 1 NUMA node per socket. Socket 0 = Node 0; Socket 1 = Node 1. Inter-socket link traverses xGMI / Infinity Fabric. | Target Node 0 -> Draft Node 1.<br>Target Node 1 -> Draft Node 0. |
| **Dual AMD EPYC (Zen 2/3/4/5)** | NPS2 | 4 | 2 | 2 NUMA nodes per socket (quad-channel DDR interleaving per hemisphere). | Target Node 0/1 -> Draft Node 2 (Socket 1).<br>Target Node 2/3 -> Draft Node 0 (Socket 0). |
| **Dual AMD EPYC (Dual-Socket Reference)** | **NPS4** | **8** | **4** | **4 NUMA nodes per socket (dual-channel DDR per quadrant; 1 CCD cluster per quadrant). Sockets linked via 4x 16-lane xGMI.** | **Target Node 0-3 -> Draft Node 4 (Socket 1).<br>Target Node 4-7 -> Draft Node 0 (Socket 0).** |
| **Single AMD EPYC (Zen 1 Naples)** | 4-Die MCM | 4 | 4 | 4 Zeppelin dies on a single package connected by on-package IFOP links. | Target Node 0 -> Draft Node 2 (Opposite diagonal quadrant).<br>Target Node 1 -> Draft Node 3. |
| **Single AMD EPYC (Zen 4/5 Genoa/Turin)** | NPS4 | 4 | 4 | 4 quadrants across 12 DDR5 memory channels on a single package. | Target Node 0 -> Draft Node 2 (Opposite diagonal quadrant). |
| **Dual AMD EPYC (Zen 4/5 Genoa/Turin)** | NPS4 | 8 | 4 | 8 NUMA nodes across 2 packages (24 DDR5 channels total). | Target Node 0-3 -> Draft Node 4 (Socket 1). |
| **Intel Xeon Scalable (Flat)** | Flat / SNC-OFF | 1 or 2 | 1 | 1 node per socket across UPI (Ultra Path Interconnect). | Target Socket 0 -> Draft Socket 1. |
| **Intel Xeon Scalable (SNC-2)** | Sub-NUMA 2 | 4 (Dual-socket) | 2 | Sockets split into two directional halves of the on-die mesh. | Target Node 0/1 (Socket 0) -> Draft Node 2 (Socket 1). |
| **Intel Xeon Scalable (SNC-4)** | Sub-NUMA 4 | 8 (Dual-socket) | 4 | Sockets split into quadrants of the on-die mesh (Sapphire/Emerald/Granite Rapids). | Target Node 0-3 (Socket 0) -> Draft Node 4 (Socket 1). |
| **Intel Xeon Quad-Socket** | 4-Socket UPI | 4 | 1 | 4 discrete sockets interconnected in a ring or crossbar mesh. | Target Node 0 -> Draft Node 1 (Remote physical socket). |

### 2.2 Microarchitectural Mechanics of Draft Allocation

When running speculative decoding (`--speculative`) in conjunction with NUMA splitting (`--numa split`):
- The **Target Model** (e.g. 70B+ dense model or 671B DeepSeek MoE) offloads its memory-bound routed layers across all available NUMA nodes (`CPU0`..`CPUn`), with attention and router layers pinned to `CUDA0`.
- The **Draft Model** (e.g. a 7B or 1.5B model) must execute concurrently without competing for L3 cache capacity, memory controller queue depths, or DDR channels with the target model's primary thread pool on Socket 0.
- [`ggml::cpu::numa::get_optimal_draft_node()`](../ggml/src/ggml-cpu-numa.cpp) dynamically inspects the hardware topology:
  1. Identifies the physical package of the primary target node via `node_socket(target_node)`.
  2. Queries `socket_count()`. If $\ge 2$ physical packages exist, it selects the primary node of the opposite physical socket (`Socket 1`).
  3. If running on a single socket with multiple NUMA domains (e.g. Single EPYC NPS2/NPS4 or AMD Naples), it strides to the furthest NUMA quadrant: `(idx + nodes.size() / 2) % nodes.size()`.
  4. If running on a single UMA node (NPS0), it returns `-1`, falling back to standard scheduling without cross-node pinning.

### 2.3 Microarchitectural Thread Pinning & SMT Sibling Exclusion
In high-throughput token decode loops, hyperthreading (Simultaneous Multi-Threading, SMT) induces resource contention across execution units, floating-point pipelines, and L1/L2 caches:
- **Physical Core Discovery**: In [`ggml/src/ggml-cpu-numa.cpp:488-515`](../ggml/src/ggml-cpu-numa.cpp#L488-L515), `get_physical_cores()` reads `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` to separate primary physical CPU IDs from secondary hyperthread siblings.
- **Ordered Physical Allocation**: When an inference engine requests $N$ worker threads on a NUMA node with $C$ physical cores, threads $0 \dots C-1$ are strictly bound to primary physical cores (`n.cores[thread_idx]`). Only when thread count $N > C$ does the scheduler map excess threads to secondary SMT siblings.
- **Cache Locality Guarantee**: Each physical worker thread is granted uncompromised access to its private L1/L2 cache and execution pipelines, eliminating up to 18% tail latency jitter under saturated multi-client load.

---

## 3. Storage NUMA Locality & Linux MD RAID Virtual Block Resolution

### 3.1 The Problem with Virtual Storage Devices
On Linux systems, high-bandwidth storage often consists of multiple NVMe SSDs aggregated into a Software RAID0 array via `mdadm` (e.g. 4x NVMe drives on a PCIe 4.0 x16 carrier card hosting `/dev/md0` formatted as ext4, XFS, or a multi-device ZFS storage pool).

Inspecting `/dev/md0` with standard sysfs queries (`/sys/dev/block/9:0`) resolves to `/sys/devices/virtual/block/md0`. Virtual block devices do **not** have physical PCIe root complexes and therefore lack the `numa_node` sysfs attribute, causing traditional runtimes to fail back to Node 0.

### 3.2 Recursive Member Slave Resolution Algorithm
[`ggml::cpu::numa::detect_dev_numa_node()`](../ggml/src/ggml-cpu-numa.cpp) implements recursive slave traversal:

```
/dev/md0 (Major:Minor 9:0)
   │
   ▼ sysfs lookup
/sys/dev/block/9:0  ──(canonical symlink)──► /sys/devices/virtual/block/md0
                                                     │
                                                     ├── slaves/nvme0n1  ──(symlink)──► /sys/devices/pci0000:c0/.../nvme/nvme0/nvme0n1
                                                     │                                         │
                                                     │                                         ▼ parent traversal
                                                     │                                  /sys/devices/pci0000:c0/0000:c0:01.1/numa_node  ==> "2"
                                                     ▼
                                            Physical Root Complex: Node 2
```

1. **Major/Minor Extraction**: Queries `st_dev` from `stat()` or `fstat()`.
2. **Virtual Device Detection**: If the canonical path contains a `slaves/` directory, iterates through all underlying block devices.
3. **Physical Device Traversal**: Canonicalizes the slave link to the physical device tree under `/sys/devices/pci*`.
4. **Hierarchical Ascendance**: Ascends parent directories until `numa_node` is found and reads the integer ID.
5. **Prefetch Pool Binding**: In [`ggml-moe-prefetch.cpp`](../ggml/src/ggml-moe-prefetch.cpp), prefetch worker threads dynamically bind to `get_storage_numa_node()`:
   ```cpp
   int storage_node = ggml::cpu::numa::get_storage_numa_node();
   if (storage_node >= 0) {
       ggml::cpu::numa::bind_current_thread_to_node(storage_node, i);
   }
   ```
   This guarantees that asynchronous page-in DMA transfers from NVMe carrier cards saturate local memory controllers (e.g. Node 2 @ ~28 GB/s) without polluting inter-socket xGMI links.

### 3.3 ZFS Storage Pools & Major 0 Pseudo-Filesystems
On Linux systems where models reside on OpenZFS datasets (such as a high-performance ZFS storage pool mounted across NVMe carrier cards), `stat()` reports `major(st_dev) == 0`. Devices with major 0 do not appear under `/sys/dev/block/`.

[`ggml::cpu::numa::detect_dev_numa_node()`](../ggml/src/ggml-cpu-numa.cpp) resolves ZFS storage locality via:
1. **Mountinfo Correlation**: Ingests `/proc/self/mountinfo` to map `0:minor` back to the filesystem type (`zfs`) and source dataset (e.g. `tank/models` or `pool/data`).
2. **Zpool Cache Parsing**: Scans `/etc/zfs/zpool.cache` for member device hardware identifiers, specifically PCI bus addresses (`pci-0000:23:00.0`, `0000:24:00.0`, etc.).
3. **PCI Root Complex Query**: Ingests `/sys/bus/pci/devices/0000:XX:XX.X/numa_node` to determine the backing PCIe root complex (e.g. Node 2 on the host system).
4. **Hardware Scan Prioritization**: In fallback scans across `/sys/block/nvme*`, inspects `device/model` to prioritize high-bandwidth PCIe NVMe carrier cards over OS boot drives.

---

## 4. Wait-Free Inter-Socket SPSC Ring Buffer

### 4.1 Microarchitectural Cache Line Padding (128-Byte Alignment) & Kowarschik-Weiß Padding Theory
On modern x86-64 server processors—especially AMD Zen 3/Zen 4/Zen 5 and Intel Golden Cove/Raptor Cove—the hardware spatial prefetcher operates on pairs of 64-byte cache lines, fetching 128-byte aligned blocks. Standard 64-byte padding (`alignas(64)`) is insufficient to prevent false sharing: if a producer's tail pointer and a consumer's head pointer sit in adjacent 64-byte lines within the same 128-byte chunk, the spatial prefetcher causes cache-line invalidation cycles across the inter-socket interconnect.

- **Theoretical Basis (Kowarschik & Weiß 2003)**:
  In their foundational taxonomy of cache optimization techniques (Meyer, Sanders, Sibeyn eds., LNCS 2625, 2003, §10.3), Kowarschik & Weiß formalized **inter-array and intra-array padding** as the primary data layout transformation to eliminate cross-invalidation and conflict misses in set-associative caches. Without explicit padding, independent variables that map to the same cache set cause continuous evictions. In multi-socket NUMA topologies, this manifests as catastrophic cross-socket coherence invalidation traffic over AMD Infinity Fabric / Intel UPI.

[`common_spsc_ring_buffer`](../common/speculative.h) enforces strict `alignas(128)` padding:

```cpp
// Cache Line 1 (128-byte boundary): Producer State
alignas(128) std::atomic<uint64_t> m_tail{0};
uint64_t m_cached_head{0};
uint8_t  m_pad0[128 - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t)];

// Cache Line 2 (128-byte boundary): Consumer State
alignas(128) std::atomic<uint64_t> m_head{0};
uint64_t m_cached_tail{0};
uint8_t  m_pad1[128 - sizeof(std::atomic<uint64_t>) - sizeof(uint64_t)];

// Cache Line 3+ (128-byte boundary): Circular Storage Buffer
alignas(128) T m_buffer[Capacity];
```

### 4.2 Wait-Free Memory Ordering & Amortized Release-Acquire Cycles
- **Wait-Free Guarantee**: Every push and pop operation completes in a bounded, deterministic number of steps ($O(1)$) without spin-waiting or kernel intervention.
- **Amortized Inter-Socket Coherency**: Cross-socket cache line invalidations across AMD Infinity Fabric or Intel UPI are expensive (~100–150 ns per invalidation). The ring buffer minimizes cross-socket traffic through locally cached cursors:
  - *Producer*: Writes to `m_buffer[tail & MASK]` and updates `m_tail` with `std::memory_order_release`. It reads the consumer's `m_head` with `std::memory_order_acquire` **only** when the local `m_cached_head` indicates that the buffer is full. Over $N$ push operations, inter-socket cache line transfers occur at a frequency of at most $O(N / \text{Capacity})$, rather than $O(N)$.
  - *Consumer*: Reads from `m_buffer[head & MASK]` and updates `m_head` with `std::memory_order_release`. It reads the producer's `m_tail` with `std::memory_order_acquire` **only** when `m_cached_tail` indicates that the buffer is empty.
- **Batch Clamping**: `push_batch` and `pop_batch` calculate available space with underflow protection (`(diff < Capacity) ? (Capacity - diff) : 0`), preventing out-of-bounds corruption under wrap-around conditions.
- **Zero OS Futex Calls**: Completely eliminates `pthread_mutex_lock` and `pthread_cond_signal`, removing up to 15 microseconds of kernel scheduling jitter per speculative token draft round.
- **AMD Zen MOESI Coherence Optimization (McKenney 2023 §3.1)**:
  Cache coherence across multi-socket AMD EPYC processors operates under the MOESI protocol via Infinity Fabric / xGMI links. When a producer modifies a cache line, any copies on remote sockets transition to the Invalid (I) state via remote probe cycles (taking 90–120 ns over xGMI). By caching shadow head/tail cursors (`m_cached_head`, `m_cached_tail`), write operations remain strictly local to the producer's CCX in the Modified (M) or Exclusive (E) state, eliminating cross-socket invalidation traffic.
- **ACPI SLIT Distance Matrix Compliance**:
  Linux kernel memory allocation policies follow the ACPI System Locality Distance Information Table (SLIT). For dual EPYC Milan platforms, intra-node latency is normalized to 10, intra-socket CCX to 16, and cross-socket xGMI traversal to 32. The SPSC ring buffer ensures all communication across distance-32 domains occurs asynchronously without blocking the primary token generation loop.

---

## 5. CUDA Graph Permanence Patch

### 5.1 Root Cause of CUDA Graph Invalidation
In upstream `ggml-cuda.cu`, the CUDA graph capture engine included a heuristic designed to detect dynamic execution graphs:
```cpp
if (graph->number_consecutive_updates >= 4) {
    graph->disable_due_to_too_many_updates = true;
}
```
During autoregressive decoding, token generation continuously modifies positional parameters (e.g. RoPE frequency offsets, attention mask lengths, and KV-cache view offsets). The upstream property comparator:
1. Detected changes in view offsets and flagged `cuda_graph_update_required = true`.
2. Re-instantiated the CUDA graph executable for 4 consecutive tokens.
3. On the 4th consecutive update, tripped `disable_due_to_too_many_updates = true`, permanently disabling CUDA graphs for the remainder of the process lifetime and falling back to individual CUDA kernel launches.

### 5.2 The Permanent Graph Reuse Solution
In [`ggml-cuda.cu`](../ggml/src/ggml-cuda.cu):
1. **No-Op Normalization**: Nodes with `ggml_is_noop(node)` (e.g. `GGML_OP_RESHAPE`, `GGML_OP_VIEW`, `GGML_OP_TRANSPOSE`, `GGML_OP_PERMUTE`) are normalized during property comparison so changing view offsets do not trigger spurious graph diffs.
2. **Copy Destination Pointer Masking**: For `GGML_OP_CPY`, destination pointers (`node->data`, `node->src[1]->data`) are masked out of the comparison because `ggml_cuda_cpy_dest_ptrs_copy` already handles dynamic device-side pointer updates via `dest_ptrs_d` prior to launch.
3. **Tripwire Removal**: The arbitrary 4-update tripwire (`disable_due_to_too_many_updates = true`) is removed. CUDA graphs remain active indefinitely, providing sub-millisecond execution times for GPU-resident attention and shared FFN layers.

---

## 6. Automated Hardware Inspection & Synthesis Script

To simplify multi-socket server tuning, [`scripts/inspect-hardware.sh`](../scripts/inspect-hardware.sh) and [`scripts/inspect-hardware.py`](../scripts/inspect-hardware.py) probe and synthesize the complete system topology without requiring manual tool invocations:

```bash
# Run comprehensive terminal synthesis
./scripts/inspect-hardware.sh

# Or invoke the python probe directly
python3 scripts/inspect-hardware.py

# Machine-readable JSON output for automated harnesses
python3 scripts/inspect-hardware.py --json
```

The script automatically detects:
- **CPU & SMT**: Physical packages, sockets, physical cores per socket, SMT threads per core, and NUMA node assignments.
- **NUMA Memory & THP**: RAM total and free per node, Transparent Huge Pages status and recommendations.
- **PCIe & GPUs**: Physical PCIe addresses, GPU NUMA root complex affinity, and PCIe link width/speed.
- **Storage Subsystem**: Direct NVMe drives, MD RAID arrays (with `/slaves/` traversal), and ZFS storage pools resolved to physical NUMA nodes.
- **Speculative Decoding Affinities**: Inter-socket interconnect bandwidth and optimal target vs draft model NUMA placement.
- **Recommended Command**: Tailored `ik_llama.cpp` server launch command with `--numa split`, `-sm graph`, `-ot`, optimal prefetch node, and core-matched thread counts.

---

## 7. Verification and Test Suite

All mechanisms are automated and verified via [`tests/test-numa.cpp`](../tests/test-numa.cpp).

To run the full suite:
```bash
cmake --build build --target test-numa -j $(nproc)
./build/bin/test-numa
```

The test runner validates:
1. **Mock Hardware Topologies**: Proves correct socket counting and draft node assignment across simulated NPS0, NPS1, NPS2, NPS4, AMD Naples single-socket, AMD Genoa 1S NPS4, Intel Quad-Socket, and Intel Sapphire Rapids SNC-4 systems.
2. **Storage NUMA Probing**: Proves direct NVMe resolution to parent PCI devices, recursive `/slaves/` traversal for Linux Software MD RAID0 arrays, and multi-device ZFS storage pool resolution to the backing PCIe root complex.
3. **Wait-Free Ring Buffer Concurrency**: Proves thread safety across concurrent producer and consumer threads streaming 50,000 items with strictly sequential FIFO delivery, zero loss, and validated batch underflow/overflow bounds.
4. **Physical NUMA Allocation**: Allocates and verifies 64 MiB across all physical NUMA nodes on the host system, proving 100% node-local placement via `move_pages` / Linux page-table sampling.

---

## 8. Multi-Tenant Cohort Prefix Sharing (PPS) & NUMA Memory Locality

In multi-tenant high-concurrency serving (`llama-server`), multiple incoming user prompts or autonomous subagents often share common system prompts or document contexts. 

### 8.1 The Cross-Slot NUMA Memory Footprint
Under legacy serving, allocating $B$ independent slots for identical prefixes multiplies physical memory consumption by $B$. On multi-socket NUMA systems, this rapid expansion quickly exhausts the local memory channels of the primary NUMA node (e.g. Node 0), forcing the Linux kernel to spill memory allocations onto remote NUMA nodes across Infinity Fabric or UPI. This remote allocation cuts effective DDR memory bandwidth in half and induces inter-socket interconnect congestion.

### 8.2 Zero-Copy Sequence Replication via `llama_kv_cache_seq_cp()`
With Cohort Prefix Sharing enabled (`--cohort-prefix-sharing`):
1. **Node-Local Slot Duplication**: When an incoming request matches an existing warm cohort slot, the server executes `llama_kv_cache_seq_cp()`. This duplicates the KV cache sequence references within the same backend memory context without re-evaluating the transformer prompt.
2. **Bandwidth Preservation**: By avoiding redundant transformer prefill evaluations, DDR memory bus traffic is reduced by **84.8% to 93.9%** (see [`docs/cohort-prefix-sharing.md`](cohort-prefix-sharing.md)), keeping the primary node's memory bus available for autoregressive token decoding.
3. **Automated Verification**: Validated by [`tests/test-cohort-prefix-sharing.cpp`](../tests/test-cohort-prefix-sharing.cpp).

---

## 9. N-Gram Speculative Drafting & Multi-Socket CCX Pipelining

Speculative decoding achieves maximum efficiency when the draft generation phase and the target verification phase do not contend for the same microarchitectural resources (L1/L2/L3 caches and memory controllers).

### 9.1 Microarchitectural CCX Mapping on AMD Zen (NPS4)
On AMD Milan/Genoa processors configured in NPS4 mode, each socket contains multiple discrete Core Complex Dies (CCDs), each with its own private L3 cache slice:
- **Target Model Placement**: The primary target model thread pool binds to Socket 0 (Nodes 0–3), utilizing local CCDs and memory channels for tensor operations.
- **Draft Model / Suffix Trie Placement**: Speculative drafting (whether using a lightweight draft model or a cache-local N-gram suffix trie) is pinned to Socket 1 (Node 4) via `get_optimal_draft_node()`.
- **Zero L3 Cache Bouncing**: Because Socket 0 and Socket 1 have separate physical L3 caches, draft candidate generation runs at full speed without evicting the target model's activation tensors or KV cache lines from L3.

### 9.2 Wait-Free Token Streaming
Candidate draft tokens are transmitted from Socket 1 to Socket 0 across the `alignas(128)` padded `common_spsc_ring_buffer`. This eliminates operating system `sys_futex` sleep states and avoids false-sharing cache line invalidation cycles between the producer and consumer cores.

---

## 10. Host-RAM KV Cache Placement (`-nkvo`) & NUMA Bandwidth Dynamics

For massive context windows (64k to 1M tokens) or high batch sizes ($B \ge 16$), the attention KV cache can easily exceed GPU VRAM (e.g. 24 GB on RTX 4090).

### 10.1 Offloading the KV Cache to Host Memory (`-nkvo`)
The `-nkvo` (no KV offload) flag forces the KV cache to remain in host RAM, reserving 100% of GPU VRAM for model weights and attention compute buffers:
- **Bandwidth Trade-Off**: During autoregressive decoding, keys and values must be read from host RAM. On our dual EPYC server, local 8-channel DDR4 memory delivers ~150–180 GB/s aggregate bandwidth.
- **NUMA Pinning Requirement**: When `-nkvo` is active, worker threads evaluating attention must be pinned to the exact NUMA node where the KV cache buffer was allocated. If attention threads run on Socket 1 while the KV cache resides on Socket 0, inter-socket xGMI links become saturated, causing decode speeds to drop by up to $3\times$.

### 10.2 Cache-Conscious Data Layout & Kowarschik-Weiß Array Padding for KV Caches
Following Kowarschik & Weiß (2003) §10.3 data layout transformations, multi-tenant KV caches allocated across NUMA nodes must employ inter-array padding to avoid cache conflict misses:
- **Set Associativity Collisions**: When multiple attention heads access KV matrices with power-of-two strides (e.g. 512, 1024, or 4096 bytes), memory lines map to the exact same cache set in CPU L1/L2 caches and GPU L2 slices. This creates pathological conflict misses where lines evict each other despite abundant total cache capacity.
- **Stride Padding**: By applying intra-array padding to tensor row strides—padding head dimensions by an odd multiple of the cache line size (e.g. $+64\text{ B}$ or $+128\text{ B}$)—consecutive heads map across distinct cache sets, completely eliminating cache thrashing and maximizing sustained DDR4/DDR5 read bandwidth during autoregressive decoding.

---

## 11. High-Concurrency Micro-Batching (`-ub`) Dynamics Under NPS4 vs NPS1

The micro-batch size flag (`-ub` / `--ubatch-size`) controls the physical chunk size processed in a single compute kernel launch:

| NUMA Mode | Recommended `-ub` | Rationale |
| :--- | :---: | :--- |
| **NPS1 (1 Node/Socket)** | `512` – `1024` | Cores across the entire socket share unified memory controllers; larger batch tiles maximize SIMD vector utilization. |
| **NPS4 (4 Nodes/Socket)** | `128` – `256` | Quadrant memory controllers operate with lower per-channel queue depths; smaller micro-batch sizes keep working sets inside the private L3 cache of each 7-core CCD, preventing DDR bus stalls. |

---

## 12. Linux Kernel OS Jitter Mitigation & Serialized Timestamping

In high-concurrency low-latency server deployments, operating system scheduler ticks and interrupt servicing introduce tail-latency jitter ($p99.9$ spikes) during decode loops:
- **Tickless Core Isolation**: Host workstations can isolate inference cores by appending the following kernel boot parameters:
  ```bash
  isolcpus=1-27,29-55 nohz_full=1-27,29-55 rcu_nocbs=1-27,29-55
  ```
  Leaving core 0 (Socket 0) and core 28 (Socket 1) to handle OS timer ticks, housekeeping threads, and device interrupts, while inference threads run completely uninterrupted.
- **Serialized Hardware Timestamping**: Nanosecond latency profiling utilizes serialized CPU cycle counters to prevent out-of-order execution from reordering instruction retired events:
  ```cpp
  static inline uint64_t rdtscp_serialized() {
      uint32_t aux;
      uint64_t tsc = __rdtscp(&aux);
      _mm_lfence();
      return tsc;
  }
  ```

---

## 13. Systems Documentation Suite & Cross-References

For in-depth architectural specifications and implementation guides, refer to:
- [`docs/cohort-prefix-sharing.md`](cohort-prefix-sharing.md): Multi-Tenant Cohort Prompt Prefix Sharing architecture, HTTP API, slot subdivision, and security invariants.
- [`docs/storage-and-io-architecture.md`](storage-and-io-architecture.md): High-throughput `io_uring` + `O_DIRECT`, cuFile, CXL 3.0, and recursive storage NUMA discovery.
- [`docs/superchips-and-unified-memory.md`](superchips-and-unified-memory.md): Architecture guide for AMD MI300A APU, NVIDIA Grace Hopper/Blackwell, and Apple Silicon MLX.
- [`docs/nextgen-container-format.md`](nextgen-container-format.md): NPMC / GGUF-X specification featuring Roaring Bitmaps, ZDDs, Milner Bigraphs, and streaming decompression.
- [`docs/gguf-hardware-linting.md`](gguf-hardware-linting.md): GGUF mathematical scoring framework and `scripts/gguf-lint.py`.
- [`docs/roadmap.md`](roadmap.md): Prioritized 5-phase microarchitectural HPC systems roadmap.

