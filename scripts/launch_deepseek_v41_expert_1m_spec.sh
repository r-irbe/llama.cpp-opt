#!/usr/bin/env bash
# ==============================================================================
# launch_deepseek_v41_expert_1m_spec.sh
# DeepSeek-V4.1-Flash Single-Tenant Inference (1M Context, Speculation & NUMA Expert)
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
BINARY="${WORKSPACE_ROOT}/build/bin/llama-cli"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Executable not found at ${BINARY}. Run 'cmake --build build -j \$(nproc)' first." >&2
    exit 1
fi

# Provenance fingerprint
BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "================================================================================"
echo " Launching DeepSeek-V4.1 Single-Tenant Engine (1M Context + Speculation)"
echo " Binary SHA256: ${BIN_SHA256}"
echo " Model: ${MODEL_PATH}"
echo " Strategy: NUMA Expert Partitioning + Per-Step Speculative Autotuning"
echo " Safe Hardware Rings: DSV4_PER_STEP_MAX_STATE_ROWS = 8 (n_max = 7)"
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
    --spec-type ngram-mod:n_max=7,n_min=2,ngram_size_n=4 \
    --spec-ckpt-mode per-step \
    --spec-autotune \
    "$@"
