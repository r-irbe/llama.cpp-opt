#!/usr/bin/env bash
# ==============================================================================
# launch.sh - Declarative Model Launcher Entrypoint for ik_llama.cpp
#
# Usage:
#   ./scripts/launch.sh [model_key] [extra args...]
#   ./scripts/launch.sh --list
#   ./scripts/launch.sh glm_53
#   ./scripts/launch.sh deepseek_v41_flash --port 8000
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source optimized environment
if [[ -f "${WORKSPACE_ROOT}/env.optimized.sh" ]]; then
    source "${WORKSPACE_ROOT}/env.optimized.sh"
elif [[ -f "${WORKSPACE_ROOT}/env.sh" ]]; then
    source "${WORKSPACE_ROOT}/env.sh"
fi

exec python3 "${SCRIPT_DIR}/launch.py" "$@"
