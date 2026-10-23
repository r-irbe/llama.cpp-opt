#!/usr/bin/env bash
# ==============================================================================
# launch_deepseek_v41_expert_1m_np8_server.sh
# DeepSeek-V4.1-Flash Multi-Tenant Server (1M Context, 8 Parallel Slots, NUMA Expert)
#
# Hardware Targets:
# - Dual AMD EPYC 7453 (56 physical cores, 8 NUMA nodes, NPS4)
# - NVIDIA GeForce RTX 4090 (24 GB VRAM)
# - NVMe Direct I/O storage
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source optimized runtime environment
if [[ -f "${WORKSPACE_ROOT}/env.optimized.sh" ]]; then
    source "${WORKSPACE_ROOT}/env.optimized.sh"
elif [[ -f "${WORKSPACE_ROOT}/env.sh" ]]; then
    source "${WORKSPACE_ROOT}/env.sh"
fi

MODEL_PATH="${MODEL_PATH:-/home/radu/code/llama.cpp/unsloth/Smalinin-Deepseek-v41-flash/Q2_K_Protected-Q8/DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected-00001-of-00010.gguf}"
BINARY="${WORKSPACE_ROOT}/build/bin/llama-server"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Executable not found at ${BINARY}. Run 'cmake --build build -j \$(nproc)' first." >&2
    exit 1
fi

# Clean up stale processes listening on port
if lsof -ti:"${PORT}" > /dev/null 2>&1; then
    echo "Cleaning up stale process listening on port ${PORT}..."
    kill -9 $(lsof -ti:"${PORT}") || true
    sleep 1
fi

# Provenance fingerprint
BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "================================================================================"
echo " Launching DeepSeek-V4.1 Multi-Tenant Server (-np 8, 1M Context)"
echo " Binary SHA256: ${BIN_SHA256}"
echo " Port: ${PORT} | Host: ${HOST}"
echo " Model: ${MODEL_PATH}"
echo " Strategy: NUMA Expert Partitioning + Cohort Prefix Sharing"
echo "================================================================================"

exec "${BINARY}" \
    -m "${MODEL_PATH}" \
    -ngl 99 \
    -ot "exps=CPU,engram_embd=CPU,nextn=CPU" \
    -t 56 \
    -b 512 \
    -ub 512 \
    -c 1048576 \
    -nkvo \
    -ctk q4_0 \
    -ctv q4_0 \
    -mla 2 \
    -fa on \
    --zdd-sparse-kv \
    --defer-ple \
    --numa expert \
    --transparent-huge-pages \
    -np 8 \
    --cohort-prefix-sharing \
    --cohort-min-prefix 64 \
    --cache-ram 0 \
    --port "${PORT}" \
    --host "${HOST}" \
    "$@"
