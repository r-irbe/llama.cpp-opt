# GGUF Model Scoring, Microarchitectural Linting, and Hardware Constraint Optimization

**Author**: Systems Architecture & High-Performance Inference Group  
**Target Repository**: [`ikawrakow/ik_llama.cpp`](../README.md)  
**Implementation Tool**: [`scripts/gguf-lint.py`](../scripts/gguf-lint.py)  
**Reference Document**: [`scratch/gguf_hardware_linter_and_scoring_architecture.md`](../scratch/gguf_hardware_linter_and_scoring_architecture.md)

---

## 1. Executive Summary & Problem Statement

Quantized GGUF models distributed across public repositories are compiled as generic, hardware-agnostic binary files. When executed on high-performance heterogeneous servers—such as multi-socket NUMA systems with Tensor Core GPUs and high-throughput NVMe arrays—unoptimized GGUF layouts suffer from critical microarchitectural mismatches:
1. **Direct I/O Misalignment**: Default 32-byte alignment prevents Linux `io_uring` with `O_DIRECT` from executing zero-copy DMA, forcing all model loading through the Linux page cache.
2. **GPU VRAM Spilling**: Mismatch between quantization bitrates and GPU VRAM capacity forces critical attention layers or shared experts to spill onto CPU DDR, reducing inference speed by up to $10\times$.
3. **MoE Expert Seek Thrashing**: Layer-by-layer interleaving of routed experts across multi-gigabyte files forces discontinuous seeks across storage during dynamic routing.
4. **Quantization Engine Mismatch**: Quantizing GPU-resident attention tensors using lookup-table (LUT) formats (e.g. IQ1_S, IQ2_XXS) prevents native Tensor Core MMA execution and triggers slow ALU bit-unpacking emulation.

[`scripts/gguf-lint.py`](../scripts/gguf-lint.py) evaluates a GGUF artifact against the physical constraints of the host system, assigning a mathematical score $S \in [0, 100]$ and generating actionable remediation commands.

---

## 2. Mathematical Scoring Framework

The hardware-model alignment score $S \in [0, 100]$ evaluates the degree to which a GGUF artifact satisfies the physical constraints of the host:

$$S = 100 - \sum_{i=1}^{M} P_i$$

where $P_i$ represents the penalty associated with violated lint rule $i$.

### 2.1 The Hardware Constraint Vector
The host platform is formalized as a tuple of physical constraints:
$$\mathcal{H} = \left\langle V_{\text{vram}}, N_{\text{numa}}, R_{\text{total}}, BW_{\text{ddr}}, BW_{\text{pcie}}, BW_{\text{nvme}}, \mathcal{A}_{\text{isa}}, \mathcal{C}_{\text{gpu}} \right\rangle$$

- $V_{\text{vram}}$: Usable GPU VRAM capacity after driver reservations.
- $N_{\text{numa}}$: Number of active NUMA nodes (e.g. NPS4: 4 nodes per socket).
- $R_{\text{total}}$: Total host physical RAM across all NUMA nodes.
- $BW_{\text{pcie}}$: PCIe link bandwidth to the primary accelerator and storage carrier card.
- $BW_{\text{nvme}}$: Peak sequential read bandwidth of the model storage subsystem.
- $\mathcal{A}_{\text{isa}}$: Host CPU vector instruction set architecture (AVX2, AVX-512, AMX, ARM NEON).
- $\mathcal{C}_{\text{gpu}}$: GPU Compute Capability (e.g. `sm_89` Ada Lovelace, `sm_90` Hopper, `sm_100` Blackwell).

### 2.2 The GGUF Artifact Model
A GGUF file is represented by its alignment and tensor set:
$$\mathcal{G} = \left\langle \text{Align}, \mathcal{T} = \{t_1, t_2, \dots, t_K\} \right\rangle$$
where each tensor $t_k = (\text{name}_k, \text{dims}_k, \text{type}_k, \text{offset}_k, \text{bytes}_k)$.

Tensors are partitioned into two disjoint subsets:
1. **Critical Path Set ($\mathcal{T}_{\text{crit}}$)**: Embeddings, attention projection matrices ($Q, K, V, O$), attention norms, MoE router weights, shared FFNs, and output heads.
2. **Routed Expert Set ($\mathcal{T}_{\text{moe}}$)**: Sparse routed expert feed-forward weights (`blk.*.ffn_.*_exps.*`).

---

## 3. Lint Rule Catalog

### Rule LINT001: Sector Boundary Alignment (`O_DIRECT` Capability)
- **Condition**: $\text{Align} \not\equiv 0 \pmod{4096} \lor \exists t \in \mathcal{T}, (\text{Offset}_t + \text{HeaderLen}) \not\equiv 0 \pmod{4096}$.
- **Penalty**: $P = 15$.
- **Theoretical Basis**:
  - *Aggarwal & Vitter (1988) Parallel Disk Model (PDM)*: The optimal I/O scanning complexity $\text{Scan}(N) = \Theta(N / (D \cdot B))$ requires transfers to occur in aligned physical block multiples $B$. When offsets violate the 4096-byte hardware sector boundary, the kernel rejects `O_DIRECT` DMA, falling back to buffered copies that destroy the theoretical bound.
  - *Warren (2012) Hacker's Delight*: Power-of-two alignment arithmetic establishes branchless bitwise alignment bounds: $\text{align}(x, 4096) = (x + 4095) \ \& \ \sim 4095$. Any non-zero lower 12 bits (`x & 4095`) trigger alignment violations.
- **Impact**: Without 4096-byte alignment, Linux kernel `open(..., O_DIRECT)` returns `EINVAL`. The engine is forced to fall back to buffered I/O, incurring page cache allocation, CPU copy overhead, and TLB invalidation.
- **Action**: Repack with `--alignment 4096`.

### Rule LINT002: Critical Path VRAM Overflow
- **Condition**: $\sum_{t \in \mathcal{T}_{\text{crit}}} \text{bytes}_t > 0.90 \times V_{\text{vram}}$.
- **Penalty**: $P = 25$.
- **Impact**: When critical attention projections exceed GPU VRAM, attention layers must be offloaded to CPU DDR, bottlenecking every single autoregressive token generation step across the PCIe bus.
- **Action**: Quantize attention matrices using Q4_K_M, Q8_0, or FP8 to ensure critical path fits entirely within VRAM.

### Rule LINT003: NUMA Memory Channel Imbalance
- **Condition**: $\frac{\sum_{t \in \mathcal{T}_{\text{moe}}} \text{bytes}_t}{N_{\text{numa}}} > 0.85 \times \frac{R_{\text{total}}}{N_{\text{numa}}}$.
- **Penalty**: $P = 20$.
- **Impact**: Routed expert allocations exceed per-node memory limits, forcing remote cross-socket NUMA allocations over Infinity Fabric and cutting effective memory bandwidth in half.
- **Action**: Re-quantize routed experts to IQ3_XXS or IQ2_XXS, or enable streaming expert prefetch from NVMe RAID-0.

### Rule LINT004: Quantization-Compute Hardware Mismatch
- **Condition**: $\exists t \in \mathcal{T}_{\text{crit}}, \text{type}_t \in \{\text{IQ1\_S}, \text{IQ2\_XXS}, \text{IQ3\_XXS}\} \land \text{Target}(t) = \text{GPU}$.
- **Penalty**: $P = 10$.
- **Impact**: Ada Lovelace Tensor Cores natively execute FP16, INT8, and INT4 tensor operations. Non-uniform LUT-based IQ quants cannot execute on Tensor Cores, forcing the GPU to execute slow ALU bit-unpacking loops in CUDA warp threads.
- **Action**: Maintain critical GPU tensors in Q4_K, Q8_0, or FP16, reserving IQ quants exclusively for CPU-resident routed experts.

### Rule LINT005: MoE Seek Fragmentation
- **Condition**: $\text{Count}\left( |\text{Offset}_{t_{i+1}} - \text{Offset}_{t_i}| > 64 \text{ MiB} \right) > 0.5 \times |\mathcal{T}_{\text{moe}}|$.
- **Penalty**: $P = 10$.
- **Impact**: Layer-by-layer expert interleaving scatters weights for a single expert across the file. Activating expert 47 across 60 layers requires 60 discontinuous seeks.
- **Action**: Group expert weights contiguously by expert ID or deploy prefetching staging ring buffers.

### Rule LINT006: Cache Line Split Invariant (`alignas(128)`)
- **Condition**: $\exists t \in \mathcal{T}, \text{RowStrideBytes}(t) \not\equiv 0 \pmod{128} \lor \text{Offset}_t \not\equiv 0 \pmod{128}$.
- **Penalty**: $P = 10$.
- **Theoretical Basis**:
  - *Kowarschik & Weiß (2003) Array Padding Theory*: Formulates data layout transformations (intra-array and inter-array padding; Meyer et al. LNCS 2625 §10.3) to prevent cache line splitting and set-associativity conflict misses. Modern dual-line prefetchers on AMD Zen 3/4/5 and Intel Golden Cove/Raptor Cove cores operate in paired 128-byte cache line sectors.
- **Impact**: Misaligned tensor rows or atomic ring buffer cursors spanning a 64-byte or 128-byte boundary trigger split cache-line accesses, generating two distinct bus transactions and doubling memory controller latency.
- **Action**: Pad row strides and memory descriptors to 128-byte multiples with explicit `alignas(128)`.

### Rule LINT007: HugeTLB 2MB / 1GB Boundary Conformity
- **Condition**: $\text{Align} \not\equiv 0 \pmod{2097152} \lor \exists t_{\text{seg}} \in \mathcal{T}_{\text{seg}}, \text{Offset}_{t_{\text{seg}}} \not\equiv 0 \pmod{2097152}$.
- **Penalty**: $P = 15$.
- **Theoretical Basis**:
  - *Rahman (2003) & Rahman-Raman (2000) TLB Penalty Formalization*: Proved that translation lookaside buffer (TLB) miss penalties dwarf L1/L2 data cache misses when address spaces exceed TLB reach (LNCS 2625 Ch. 8 §8.1-8.2). Standard 4KB pages for hundreds of gigabytes of weights generate millions of page table entries, overwhelming L2 STLB capacity (typically 1,536–2,048 entries covering only 6–8 MB). Aligning segment boundaries to 2MB (or 1GB) increases TLB reach by $512\times$ (or $262,144\times$), eliminating TLB thrashing during weight streaming.
- **Impact**: Models exceeding 30B parameters mapped via 4KB standard pages require millions of page table entries. Autoregressive streaming through unaligned weights triggers continuous dTLB misses, L2 TLB walks, and inter-socket TLB shootdowns. Failure to align major segment boundaries to 2MB (or 1GB) prevents the Linux kernel from backing mappings with Transparent Huge Pages (THP) or explicit HugeTLB (`MAP_HUGETLB`), causing up to 25% throughput degradation.
- **Action**: Repack container segments with 2MB alignment (`--alignment 2097152`) and enable `madvise(..., MADV_HUGEPAGE)`.

### Rule LINT008: SIMD Vector Register 64-Byte Alignment
- **Condition**: $\exists t \in \mathcal{T}_{\text{crit}}, \text{Offset}_t \not\equiv 0 \pmod{64} \lor \text{RowStrideBytes}(t) \not\equiv 0 \pmod{64}$.
- **Penalty**: $P = 10$.
- **Theoretical Basis**:
  - *Agner Fog (2023) Vector Microarchitecture*: Aligned vector loads (`_mm512_load_si512`, `vmovdqa64`) execute in a single clock cycle. Unaligned memory operands crossing a 64-byte L1 data cache line boundary generate two separate load micro-ops and a split-load merge penalty in the reservation station, stalling execution units for 1–3 cycles per vector instruction.
- **Impact**: Critical attention projection and GEMV kernels lose 15–25% arithmetic throughput due to pipeline stalls on unaligned vector loads.
- **Action**: Pad tensor allocations and row strides to 64-byte multiples.

### Rule LINT009: DRAM Bank Power-of-Two Stride Aliasing
- **Condition**: $\exists t \in \mathcal{T}, \text{RowStrideBytes}(t) \in \{2^k \mid k \ge 16\} \land \text{Length}(t) \ge 16$.
- **Penalty**: $P = 10$.
- **Theoretical Basis**:
  - *Ulrich Drepper (2007) DRAM Bank Conflict Model*: Memory controllers hash physical address bits to select DRAM channels, ranks, and banks. Power-of-two row strides (e.g. 64 KB, 128 KB, 256 KB) cause consecutive attention heads or batch rows to map to the exact same bank. Accessing different rows within the same bank triggers continuous row buffer conflicts, requiring full precharge ($t_{RP}$) and activate ($t_{RCD}$) cycles and degrading memory bus throughput by $>50\%$.
- **Impact**: Severe DDR4/DDR5 channel throttling during multi-head attention evaluation and batched prefill.
- **Action**: Apply intra-array padding (+64B or +128B) to tensor row strides to distribute consecutive vectors across orthogonal DRAM banks.

---


## 4. Verification and Tool Execution

The linter is fully operational via [`scripts/gguf-lint.py`](../scripts/gguf-lint.py):

```bash
# Evaluate a GGUF model and view human-readable analysis
./scripts/gguf-lint.py /path/to/model.gguf

# Output machine-readable JSON for automated CI/CD pipelines
./scripts/gguf-lint.py /path/to/model.gguf --json
```

The tool:
1. Discovers the host's actual hardware topology dynamically (VRAM, NUMA nodes, RAM, PCIe link).
2. Parses binary GGUF v1/v2/v3 metadata, tensor offsets, and quant types.
3. Calculates penalty deductions and reports the hardware alignment score $S \in [0, 100]$.
4. Synthesizes optimal, copy-paste ready runtime execution flags (`--numa split`, `-sm graph`, `-ot`, etc.).
