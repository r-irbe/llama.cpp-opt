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

# NOTE: prefer the capsule domain models.toml (`LLAMA_MODEL_ROOT` default is
# resolved there). The UD-Q4_K_XL container is a completed 9-shard HF download
# and is now LOADABLE: mainline type 43 (MXFP8, gather-only engram table) was
# ported and validated with a serving smoke test on 2026-10-29 (binary
# 125fdd9b; see the "UD-Q4_K_XL mainline packaging" section of
# scratch/2026-10-28-mxfp8-engram-port-and-removals-audit.md).
UD_Q4KXL_ROOT="${LLAMA_MODEL_ROOT:-${HOME}/vm/llm}"
#MODEL_PATH="${MODEL_PATH:-${UD_Q4KXL_ROOT}/DeepSeek-V4.1-Flash-1017/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf}"
MODEL_PATH="${MODEL_PATH:-${UD_Q4KXL_ROOT}/ista-deepseek-v4.1-1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf}"
DRAFT_PATH="${DRAFT_PATH:-${UD_Q4KXL_ROOT}/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf}"

BINARY="${WORKSPACE_ROOT}/build/bin/llama-server"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}" >&2
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

# Provenance fingerprint
BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "================================================================================"
echo " Launching DeepSeek-V4.1 Single-Tenant Engine (1M Context + Speculation)"
echo " Binary SHA256: ${BIN_SHA256}"
echo " Model: ${MODEL_PATH}"
echo " Draft Model  : ${DRAFT_PATH}"
echo " Strategy: NUMA Expert Partitioning + Per-Step Speculative Autotuning"
echo " Safe Hardware Rings: DSV4_PER_STEP_MAX_STATE_ROWS = 8 (n_max = 7)"
echo "================================================================================"
#rtr  --defer-ple --deder-exports -wgt
exec "${BINARY}" \
    -m "${MODEL_PATH}" --port 8000 --reasoning-format deepseek --jinja -ptcall \
    -ngl 999 --chat-template-file "${CHAT_TEMPLATE_FILE:?set CHAT_TEMPLATE_FILE to the deepseekv41-1017 jinja}" \
    --prefetch-experts \
    -ot "exps=CPU,engram_embd=CPU" \
    -t 16 -tb 24 -cb -np 1 \
    -b 4196 \
    -cuda fusion=2,fa-fabsum=1,fa-offset=0.6931f,mmq-id-size=64 \
    -ub 1024 -dsa -dsatk 1024 -fidx \
    -c 1048576 -nkvo \
    --chat-template-kwargs '{"reasoning_effort":"max"}' \
    --no-evidence-filter -ctk q4_0 -ictk q4_0 -muge -mqkv -ser 4,1 -crs 0.50 -sps 0.20  -mla 2 --cohort-prefix-sharing --cohort-min-prefix 64 --cache-ram -1 --numa expert --numa-replicate-shared -fa on --transparent-huge-pages --temp 0.8 --top-k 0 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.05 --swa-compress -sas --mlock --reasoning on --spec-type ngram-map-k4v:n_max=8,n_min=2  --spec-autotune --zdd-sparse-kv --spec-verify-pinned --timeout 3600
