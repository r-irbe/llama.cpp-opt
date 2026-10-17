# NUMA Architecture and Optimization Guide

This guide describes how to configure, optimize, and run `llama.cpp-opt` on Non-Uniform Memory Access (NUMA) systems, including multi-socket AMD EPYC, Intel Xeon, and ARM Neoverse servers, as well as multi-CCX desktop architectures.

---

## 1. Background: The NUMA Bottleneck in LLM Inference

In modern multi-socket and multi-die architectures, memory bandwidth is divided across separate NUMA nodes (memory controllers). Accessing memory attached to a local socket is fast (~80–110 ns latency), whereas accessing memory across inter-socket links (AMD xGMI or Intel UPI) incurs substantial latency penalties (~180–300 ns) and saturates inter-socket interconnects.

During autoregressive token generation:
- Every active token read must stream gigabytes of weight matrices from system memory into processor caches.
- Uncoordinated multi-threading leads to severe cross-socket bus contention, cache invalidation storms, and dramatic throughput degradation (often 2× to 4× slower than theoretical memory bandwidth).

`llama.cpp-opt` provides NUMA-aware thread pinning plus per-node CPU backends. Expert partitioning and
shared-tensor replication are prototypes that are not wired up at HEAD; see section 2.3 and
[`docs/UNIMPLEMENTED.md`](./UNIMPLEMENTED.md).

---

## 2. NUMA Operating Modes

What the binary at HEAD accepts for `--numa` (`common/common.cpp`, strategy enum in
`ggml/include/ggml.h`, pinning in `ggml/src/ggml.c`):

| `--numa` value | What it does | Status |
| --- | --- | --- |
| `distribute` | Worker threads are pinned round-robin across the detected NUMA nodes (`set_numa_thread_affinity`, `GGML_NUMA_STRATEGY_DISTRIBUTE`). | Implemented |
| `split` | **Documented alias of `distribute`** — same strategy, same behaviour. It does not partition weights or memory. | Implemented alias |
| `isolate` | All worker threads are pinned to the NUMA node the process started on (`GGML_NUMA_STRATEGY_ISOLATE`). | Implemented; see the trap in 2.2 |
| `numactl` | Uses the CPU mask that `numactl` handed to the process (`GGML_NUMA_STRATEGY_NUMACTL`). | Implemented |
| `expert` | Accepted by the parser, warns, and does not shard experts onto nodes. | **Not implemented** — `docs/UNIMPLEMENTED.md` |
| `mirror` | Accepted by the parser, warns, and does not mirror data per node. | **Not implemented** — `docs/UNIMPLEMENTED.md` |

### 2.1 `--numa split` is an alias of `--numa distribute`

`--numa split` selects `GGML_NUMA_STRATEGY_DISTRIBUTE`. The only thing it does is pin worker threads
round-robin over the NUMA nodes; it does **not** place weights or allocations on the node that uses
them, and it does not split batch processing per node. Per-node CPU backends (`CPU0..CPUn`) are
registered whenever two or more NUMA nodes exist (`ggml/src/ggml-backend.cpp`), but that is
independent of this flag.

**Meaning drift, and the decision taken at HEAD.** This flag has meant three things across tree
states: rejected at commit `9f1339fc8` (`error: invalid parameter for argument: --numa`), an alias of
`GGML_NUMA_STRATEGY_EXPERT` on `backup/feat-numa-split-pre-squash`, and an alias of
`GGML_NUMA_STRATEGY_DISTRIBUTE` in the 2026-10-17 worktree (this document previously described it as
symmetric per-node partitioning as well). Decision at HEAD: **`split` means `distribute` and nothing
else**; expert sharding, if it ever lands, will be `--numa expert`. The value is a single spelling
for one strategy, so the tuner no longer samples `split` and `distribute` as separate arms.

**Usage:**
```bash
./build/bin/llama-cli -m models/llama-3-70b-instruct.Q4_K_M.gguf \
    --numa distribute \
    -t 32 \
    -p "Explain the principles of non-uniform memory architecture."
```

### 2.2 `--numa isolate`: match `-t` to the node's core count

`isolate` pins every worker to the node the process started on. On an 8-node dual-EPYC host (7 cores
per node) that makes `-t 32` oversubscribe a single node: measured 3.48 tok/s decode and 29.7 tok/s
pp512, versus 19.2-20.8 tok/s and 140-151 pp512 for `distribute` under the same `-t 32`. Use
`--numa isolate` only together with `-t` equal to that node's core count, or use `--cpu-range`
instead.

### 2.3 Expert sharding and shared-tensor replication are not implemented

`--numa expert`, `--numa mirror` and `--numa-replicate-shared` are prototypes: the helpers exist
(`ggml::cpu::numa::bind_tensor_experts_round_robin`, `bind_tensor_shared_interleaved`,
`GGML_NUMA_STRATEGY_MIRROR`/`EXPERT`) but nothing calls them from the model load path, and the parser
now prints a warning when they are requested. What landing each one would take is written down in
[`docs/UNIMPLEMENTED.md`](./UNIMPLEMENTED.md).

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
A well-pinned `--numa distribute` (or `--numa isolate` with a matching `-t`) run should display
$>98\%$ `Numa_Hit` and near-zero `Numa_Miss` or `Foreign_Alloc`. There is no per-node weight
placement in this build, so weights stay where the allocator put them.

### 4.2 Linux Hardware Performance Counters
Monitor cross-socket interconnect traffic and cache misses:
```bash
perf stat -e node-loads,node-load-misses,cache-misses,cycles,instructions \
    ./build/bin/llama-cli -m model.gguf --numa split -t 64 -n 256
```
A low `node-load-misses` / `node-loads` ratio confirms that data is streaming locally from near memory.
Note that this only measures the thread pinning above; it does not prove per-node placement of weights.
