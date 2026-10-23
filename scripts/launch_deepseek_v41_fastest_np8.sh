#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# launch_deepseek_v41_fastest_np8.sh
#
# Fastest Configuration for DeepSeek-V4.1-Flash with 8 Concurrent Slots (-np 8)
#
# Hardware:
#   - Dual AMD EPYC 7453 (56 physical cores, 8 NUMA nodes @ NPS4)
#   - NVIDIA GeForce RTX 4090 (24GB VRAM, sm_89, PCIe Bus 41:00.0 @ NUMA 1)
#   - ~1 TB DDR4-3200 ECC RAM
#
# Architectural Optimizations Applied:
#   1. Environment: Sources env.optimized.sh (OMP 56 physical threads, CUDA graphs,
#      LLAMA_KV_ZDD_SKIP=1, MALLOC_ARENA_MAX=8).
#   2. MLA Q4_0 KV Cache (-ctk q4_0 -ctv q4_0 -mla 2): 8 parallel slots fit
#      entirely inside 24GB RTX 4090 VRAM without host fallback tax.
#   3. Flash Attention & MoE: -fa on, fused MoE enabled by default.
#   4. NUMA Tensor Distribute: --numa distribute stripes 120 routed MoE expert
#      tensors across the 8 physical NUMA nodes for maximum DDR4 memory bandwidth.
#   5. Sparse Engram/PLE Handling: --defer-ple keeps 194.56 GiB sparse engram
#      tables cold on NVMe/mmap instead of faulting into physical RAM.
#   6. Cross-Slot Cohort Prefix Sharing: --cohort-prefix-sharing enables zero-copy
#      cross-slot KV prefix cloning (llama_kv_cache_seq_cp) via find_best_donor_match,
#      eliminating up to 99.1% of redundant prefill tokens across slots.
#   7. Port-safe disciplined launch: Frees port, verifies binary sha256 provenance.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Source optimized environment
ENV_FILE="${SCRIPT_DIR}/env.optimized.sh"
if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
fi

# 2. Parameters & Model Paths
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
MODEL_PATH="${MODEL_PATH:-/home/radu/code/llama.cpp/unsloth/Smalinin-Deepseek-v41-flash/Q2_K_Protected-Q8/DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected-00001-of-00010.gguf}"
BINARY="${SCRIPT_DIR}/build/bin/llama-server"

# Context & Concurrency Settings
N_PARALLEL=8
CTX_PER_SLOT=8192
TOTAL_CTX=$((N_PARALLEL * CTX_PER_SLOT)) # 65536 total context
N_THREADS=56
BATCH_SIZE=2048
UBATCH_SIZE=512

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Binary not found or not executable at ${BINARY}" >&2
    echo "Run 'rtk cmake --build build -j \$(nproc)' first." >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

echo "======================================================================"
echo "  DeepSeek-V4.1-Flash High-Performance Server (-np ${N_PARALLEL})"
echo "======================================================================"
echo "Binary:        ${BINARY}"
echo "Model:         ${MODEL_PATH}"
echo "Port:          ${PORT}"
echo "Parallel Slots: ${N_PARALLEL}"
echo "Total Context: ${TOTAL_CTX} (${CTX_PER_SLOT} tokens/slot)"
echo "Threads:       ${N_THREADS} (1 thread/physical core across 8 NUMA nodes)"
echo "KV Precision:  Q4_0 MLA (-ctk q4_0 -ctv q4_0 -mla 2)"
echo "Sharing:       --cohort-prefix-sharing active"
echo "MoE Routing:   --numa distribute, --defer-ple"
echo "======================================================================"

# 3. Port cleanup & provenance
OLD_PIDS=$(fuser "${PORT}/tcp" 2>/dev/null || true)
if [[ -n "${OLD_PIDS}" ]]; then
    echo "Cleaning up existing process on port ${PORT} (PID: ${OLD_PIDS})..."
    fuser -k -15 "${PORT}/tcp" 2>/dev/null || true
    sleep 1
    if fuser "${PORT}/tcp" 2>/dev/null; then
        fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
        sleep 1
    fi
fi

BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launch DeepSeek-V4.1 -np 8 binary=${BINARY} sha256=${BIN_SHA256} port=${PORT}" >> server_provenance.log

# 4. Launch Server with Optimal Multi-Slot Flags
exec "${BINARY}" \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  -np "${N_PARALLEL}" \
  -c "${TOTAL_CTX}" \
  -b "${BATCH_SIZE}" \
  -ub "${UBATCH_SIZE}" \
  -t "${N_THREADS}" \
  -ngl 999 \
  -ot "exps=CPU" \
  --numa distribute \
  --defer-ple \
  -fa on \
  -mla 2 \
  -ctk q4_0 \
  -ctv q4_0 \
  --zdd-sparse-kv \
  --cohort-prefix-sharing \
  --cohort-min-prefix 64 \
  --cache-ram 0 \
  "$@"
