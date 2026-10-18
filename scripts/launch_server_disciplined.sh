#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# launch_server_disciplined.sh: Disciplined Server Launcher
# Ensures port-safe pkill cleanup, computes sha256 binary provenance,
# and starts ./build/bin/llama-server cleanly.
# ==============================================================================

PORT=8080
BINARY="./build/bin/llama-server"
LOG_FILE="llama-server.log"

# Parse port if provided in args
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[i]}" == "--port" || "${args[i]}" == "-p" ]]; then
        next_idx=$((i + 1))
        if [[ $next_idx -lt ${#args[@]} ]]; then
            PORT="${args[next_idx]}"
        fi
    fi
done

echo "=== Disciplined Server Launcher ==="
echo "Target Port: ${PORT}"
echo "Binary: ${BINARY}"

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: Server binary not found or not executable at ${BINARY}" >&2
    exit 1
fi

# 1. Binary Provenance Recording
BIN_SHA256=$(sha256sum "${BINARY}" | awk '{print $1}')
echo "Binary SHA256: ${BIN_SHA256}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launch binary ${BINARY} sha256=${BIN_SHA256} port=${PORT}" >> server_provenance.log

# 2. Port-safe process cleanup
echo "Checking for existing processes on port ${PORT}..."
OLD_PIDS=$(fuser "${PORT}/tcp" 2>/dev/null || true)
if [[ -n "${OLD_PIDS}" ]]; then
    echo "Terminating existing process on port ${PORT} (PIDs: ${OLD_PIDS})..."
    fuser -k -15 "${PORT}/tcp" 2>/dev/null || true
    sleep 1
    # Check if still listening, escalate to SIGKILL if necessary
    REMAINING=$(fuser "${PORT}/tcp" 2>/dev/null || true)
    if [[ -n "${REMAINING}" ]]; then
        echo "Process did not terminate with SIGTERM, escalating to SIGKILL..."
        fuser -k -9 "${PORT}/tcp" 2>/dev/null || true
        sleep 1
    fi
fi

# Ensure port is truly free
if lsof -Pi ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ERROR: Port ${PORT} is still occupied!" >&2
    exit 1
fi
echo "Port ${PORT} is clear."

# 3. Launch Server
echo "Launching server on port ${PORT}..."
"${BINARY}" "${args[@]}"
