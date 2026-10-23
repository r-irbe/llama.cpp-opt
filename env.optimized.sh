#!/usr/bin/env bash
# ==============================================================================
# env.optimized.sh: Hardware & Kernel Environment Tuning
# Target: Dual AMD EPYC 7453 (56 Cores / 112 Threads, 8 NUMA nodes @ NPS4)
#         NVIDIA GeForce RTX 4090 (24GB VRAM, sm_89, PCIe Bus 41:00.0 @ NUMA 1)
#         RAM: ~1 TB DDR4-3200 ECC across 16 channels
# ==============================================================================

# 1. GPU Isolation & Device Mapping
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 2. NUMA & OpenMP / CPU Worker Scheduling
# Dual AMD EPYC 7453 has 56 physical cores across 8 NUMA nodes (7 physical cores / node).
# 56 physical threads prevents SMT2 resource thrashing on Zen 3 CCXs.
export OMP_NUM_THREADS=56
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=ACTIVE
export OMP_STACKSIZE=64M

# 3. CUDA Graph & GPU Memory Infrastructure (CUDA 12/13, Ada Lovelace sm_89)
export GGML_CUDA_ENABLE_GRAPHS=1
export GGML_CUDA_GRAPH_OPT=1
export GGML_CUDA_MAX_NODES=16384
export GGML_CUDA_NO_PINNED=0
export GGML_CUDA_HOST_MALLOC_THP=1
export GGML_OP_OFFLOAD_MIN_BATCH=32

# 4. FlashAttention & Sparse ZDD Acceleration
export LLAMA_KV_ZDD_SKIP=1
export GGML_MOE_URING_PREFETCH=1

# 5. Linux Memory Allocator & Kernel Paging
# Avoid glibc allocator fragmentation across 8 NUMA nodes without unbounded arenas
export MALLOC_ARENA_MAX=8
export MALLOC_TRIM_THRESHOLD_=131072

# 6. Tooling & Harness Invariants
export PI_TELEMETRY=0
export PI_SKIP_VERSION_CHECK=1
export PI_CACHE_OPTIMIZER_TOOL_ORDER=1
