#!/usr/bin/env bash
# ==============================================================================
# launch_deepseek_v41_expert_1m_dspark.sh
# DeepSeek-V4.1-Flash with Matched DSpark Companion Draft Speculation
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

# NOTE: prefer the capsule domain models.toml (`LLAMA_MODEL_ROOT` default is
# resolved there). The UD-Q4_K_XL container is a completed 9-shard HF download
# and is now LOADABLE: mainline type 43 (MXFP8, gather-only engram table) was
# ported and validated with a serving smoke test on 2026-10-29 (binary
# 125fdd9b; plan ledger: scratch/2026-10-28-mxfp8-engram-port-and-removals-
# audit.md, "UD-Q4_K_XL mainline packaging" section). Entrainless note: the
# load requires the deepseek41.en-graph KV fallback path (ple.* keys) which
# is part of that port; no sidecar needed.
UD_Q4KXL_ROOT="${LLAMA_MODEL_ROOT:-${HOME}/vm/llm}"
MODEL_PATH="${MODEL_PATH:-${UD_Q4KXL_ROOT}/DeepSeek-V4.1-Flash-1017/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf}"
DRAFT_PATH="${DRAFT_PATH:-${LLAMA_MODEL_ROOT:-${WORKSPACE_ROOT}/models}/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}"
BINARY="${WORKSPACE_ROOT}/build-pto/bin/llama-cli"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Target model not found at ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${DRAFT_PATH}" ]]; then
    echo "ERROR: Draft model not found at ${DRAFT_PATH}" >&2
    exit 1
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Executable not found at ${BINARY}. Run 'cmake --build build -j \$(nproc)' first." >&2
    exit 1
fi

BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "================================================================================"
echo " Launching DeepSeek-V4.1 with Matched DSpark Draft Speculation"
echo " Binary SHA256: ${BIN_SHA256}"
echo " Target Model : ${MODEL_PATH}"
echo " Draft Model  : ${DRAFT_PATH}"
echo " NUMA Strategy: expert + shared layer replication (--numa-replicate-shared)"
echo " Spec Strategy: dspark (n_max=4, n_min=1, per-step)"
echo "================================================================================"

exec "${BINARY}" \
    -m "${MODEL_PATH}" \
    -md "${DRAFT_PATH}" \
    -ngl 99 \
    -ngld 99 \
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
    --numa-replicate-shared \
    --transparent-huge-pages \
    --spec-type dspark:n_max=4,n_min=1 \
    --spec-ckpt-mode per-step \
    "$@"
