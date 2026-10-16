# NUMA Architecture and Optimization Guide

This guide describes how to configure, optimize, and run `llama.cpp-opt` on Non-Uniform Memory Access (NUMA) systems, including multi-socket AMD EPYC, Intel Xeon, and ARM Neoverse servers, as well as multi-CCX desktop architectures.

---

## 1. Background: The NUMA Bottleneck in LLM Inference

In modern multi-socket and multi-die architectures, memory bandwidth is divided across separate NUMA nodes (memory controllers). Accessing memory attached to a local socket is fast (~80–110 ns latency), whereas accessing memory across inter-socket links (AMD xGMI or Intel UPI) incurs substantial latency penalties (~180–300 ns) and saturates inter-socket interconnects.

During autoregressive token generation:
- Every active token read must stream gigabytes of weight matrices from system memory into processor caches.
- Uncoordinated multi-threading leads to severe cross-socket bus contention, cache invalidation storms, and dramatic throughput degradation (often 2× to 4× slower than theoretical memory bandwidth).

`llama.cpp-opt` provides first-class NUMA awareness with fine-grained memory affinity, expert partitioning, and shared-tensor replication.

---

## 2. NUMA Operating Modes

`llama.cpp-opt` supports three primary NUMA operating modes:

### 2.1 `--numa split` (Symmetric Workload Partitioning)

`--numa split` partitions thread pools and memory allocations evenly across detected NUMA domains. Each NUMA node executes sub-graph computations on weights resident in its local physical memory channels.

- **Best for**: Dense transformer models (e.g. Llama 3, Qwen 2.5, Mistral) where all parameters are active on every token.
- **Mechanism**: Splits batch processing and tensor operations along parallel dimensions, ensuring each thread accesses exclusively local RAM.

**Usage:**
```bash
./build/bin/llama-cli -m models/llama-3-70b-instruct.Q4_K_M.gguf \
    --numa split \
    -t 64 \
    -p "Explain the principles of non-uniform memory architecture."
```

### 2.2 `--numa expert` (Mixture-of-Experts Partitioning)

`--numa expert` is purpose-built for sparse Mixture-of-Experts (MoE) architectures (such as DeepSeek-V2/V3, Qwen-2.5-MoE, Mixtral 8x7B/8x22B, and GLM-4/5).

- **Best for**: MoE models running on multi-socket or multi-CCX systems.
- **Mechanism**:
  - Assigns different expert groups to dedicated NUMA nodes based on memory locality.
  - When the router activates top-$k$ experts, computation executes locally on the specific NUMA node hosting those expert weights.
  - Eliminates cross-socket streaming of heavy expert weight matrices over UPI / xGMI links.

**Usage:**
```bash
./build/bin/llama-cli -m models/DeepSeek-V3-Q4_K_M.gguf \
    --numa expert \
    -t 64 \
    -p "Analyze asymptotic complexity of parallel prefix sum."
```

### 2.3 `--numa-replicate-shared` (Shared Tensor Duplication)

In MoE models, non-expert layers (such as self-attention QKV projections, attention output projections, RMSNorm weights, and embedding tables) are evaluated on every single token by all worker threads. If these layers reside on a single NUMA node, all other nodes must repeatedly fetch them over cross-socket buses.

`--numa-replicate-shared` duplicates these shared weights into the local physical memory of every NUMA node:
- **Best for**: High-throughput multi-socket MoE serving where memory capacity allows duplicating dense shared layers (~10–15% memory overhead).
- **Mechanism**: Allocates replica buffers per NUMA node during model loading. Each node's thread pool reads its private replica without inter-socket traffic.

**Usage:**
```bash
./build/bin/llama-cli -m models/qwen2.5-moe-72b.Q4_K_M.gguf \
    --numa expert \
    --numa-replicate-shared \
    -t 64
```

---

## 3. Thread Pinning and Core Affinity

To prevent the Linux kernel scheduler from migrating threads between NUMA nodes or CCX domains, explicit thread affinity is strongly recommended.

### 3.1 Using `numactl`

Use `numactl` to pin both thread execution and memory allocations to specific nodes:

```bash
# Inspect system NUMA topology
numactl --hardware

# Run on a single NUMA node (e.g., Node 0 with 32 cores)
numactl --cpunodebind=0 --membind=0 ./build/bin/llama-cli -m model.gguf -t 32

# Run across all nodes with interleaved memory (for standard dense models)
numactl --interleave=all ./build/bin/llama-cli -m model.gguf --numa split -t 64
```

### 3.2 AMD Zen CCX / CCD Tuning (NPS Modes)

On AMD EPYC (Milan, Genoa, Bergamo) and Ryzen processors:
- **NPS Configuration**: In server BIOS, configure NUMA Nodes Per Socket to **NPS4** (or NPS2) to align NUMA domains directly with processor Core Complex Dies (CCDs). This ensures each CCX's 32 MiB L3 cache is fed directly by its closest memory channels.
- **Thread Count (`-t`)**: Set `-t` equal to the number of physical cores in the target NUMA domains (avoid hyperthreads/SMT for memory-bound token generation).

---

## 4. Performance Diagnostics & Monitoring

To verify that NUMA optimizations are working effectively:

### 4.1 Checking NUMA Hit Ratios
Run `numastat` during active generation to inspect memory locality:
```bash
watch -n 1 numastat -c llama-cli
```
A well-optimized `--numa split` or `--numa expert` run should display $>98\%$ `Numa_Hit` and near-zero `Numa_Miss` or `Foreign_Alloc`.

### 4.2 Linux Hardware Performance Counters
Monitor cross-socket interconnect traffic and cache misses:
```bash
perf stat -e node-loads,node-load-misses,cache-misses,cycles,instructions \
    ./build/bin/llama-cli -m model.gguf --numa split -t 64 -n 256
```
A low `node-load-misses` / `node-loads` ratio confirms that data is streaming locally from near memory.
