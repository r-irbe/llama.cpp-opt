#!/usr/bin/env bash
# verify-spec.sh - verification entry point for the speculative-decoding unit battery.
# Runs the spec unit tests against ./build/bin and fails fast on any failure.
# Used as the G5 gate evidence for commits touching common/spec-* or spec CUDA paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TESTS=(test-spec-counterfactual test-spec-arbiter test-spec-integration)

status=0
for t in "${TESTS[@]}"; do
    bin="${ROOT}/build/bin/${t}"
    if [[ ! -x "${bin}" ]]; then
        echo "VERIFY FAIL: ${t} not built at ${bin} (run scripts/build-guard.sh first)" >&2
        status=1
        continue
    fi
    if "${bin}" > "/tmp/${t}.log" 2>&1; then
        echo "VERIFY OK: ${t}"
    else
        rc=$?
        echo "VERIFY FAIL: ${t} (rc=${rc}) - see /tmp/${t}.log" >&2
        status=1
    fi
done

if [[ ${status} -eq 0 ]]; then
    echo "VERIFY: all spec tests passed"
fi
exit ${status}
