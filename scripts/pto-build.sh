#!/usr/bin/env bash
# scripts/pto-build.sh - One-command PTO (Profile-Targeted Optimization) workflow
#
# Stages (positional or via BUILD_GUARD_PTO=trace|cs|use|train|merge|status|all):
#   status  current build mode/profile provenance (no build)
#   trace   instrumented build (GGML_PTO_MODE=GENERATE) [+ optional trace workload]
#   cs      context-sensitive round-2 instrumented build (CS_GENERATE)
#   merge   llvm-profdata merge of raw .profraw -> merged.profdata
#   use     profile-optimized build (GGML_PTO_MODE=USE; production default)
#   train   trace -> (workload) -> merge
#   all     train -> use
#   flags   print the ctx-capsule/fleet env surface for the current stage
#
# Invariants (AGENTS.md):
#   - every C/CXX/CUDA compilation goes through scripts/build-guard.sh; this
#     script never invokes a compiler or ninja directly;
#   - single build tree at a time (build-guard enforces), stall watchdog armed;
#   - no host-specific paths baked in: profile/build dirs are arguments with
#     generic defaults (/tmp/...), model roots come from --model-root or
#     LLAMA_MODEL_ROOT;
#   - the CMake flag surface is owned by scripts/pto_finetune.sh; this wrapper
#     only sequences stages on top of it.
#
# Examples:
#   scripts/pto-build.sh status
#   scripts/pto-build.sh use
#   scripts/pto-build.sh trace --trace-workload=harness --model-key=deepseek-v4.1-1017-ud-q4kxl --model-root=/path/to/models
#   scripts/pto-build.sh train --trace-workload=harness --model-key=... --model-root=...
#   scripts/pto-build.sh all   --trace-workload=harness --model-key=... --model-root=...
#   BUILD_GUARD_PTO=use scripts/pto-build.sh          # env knob, same as positional

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FINETUNE="${REPO_ROOT}/scripts/pto_finetune.sh"
GUARD="${REPO_ROOT}/scripts/build-guard.sh"

BUILD_DIR="${REPO_ROOT}/build-pto"
PROFILE_DIR="/tmp/ik_llama_pgo"
PROFDATA=""
TRACE_WORKLOAD="none"          # none | harness
MODEL_KEY=""
MODEL_ROOT="${LLAMA_MODEL_ROOT:-}"
TRACE_SCENARIOS="tg_short"
TRACE_REPS="1"
TRACE_PORT="8047"
DRY_RUN=0
JOBS=""

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
    STAGE="${BUILD_GUARD_PTO:-use}"
else
    shift
fi

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --build-dir=<dir>        Build tree (default: <repo>/build-pto)
  --profile-dir=<dir>      Raw .profraw dir (default: /tmp/ik_llama_pgo)
  --profdata=<file>        Merged profile (default: <profile-dir>/merged.profdata)
  --trace-workload=none|harness
                           Run a real workload after `trace` to produce .profraw
                           (harness = scripts/bench_ranked.py scenario battery)
  --model-key=<key>        Model key from scripts/configs/models.toml (trace workload)
  --model-root=<dir>       Model root dir (trace workload; default $LLAMA_MODEL_ROOT)
  --trace-scenarios=<csv>  Harness scenarios for the trace workload (default: tg_short)
  --trace-reps=<n>         Reps per scenario in the trace workload (default: 1)
  --trace-port=<port>      Port for the trace-workload server (default: 8047)
  --jobs=<n>               Optional cap passed to build-guard via BUILD_JOBS
  --dry-run                Print every command instead of executing it
EOF
    exit "${1:-0}"
}

need_val() { [[ $# -ge 2 && -n "$2" ]] || { echo "error: $1 needs a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --build-dir=*)     BUILD_DIR="${1#*=}" ;;
        --profile-dir=*)   PROFILE_DIR="${1#*=}" ;;
        --profdata=*)      PROFDATA="${1#*=}" ;;
        --trace-workload=*) TRACE_WORKLOAD="${1#*=}" ;;
        --model-key=*)     MODEL_KEY="${1#*=}" ;;
        --model-root=*)    MODEL_ROOT="${1#*=}" ;;
        --trace-scenarios=*) TRACE_SCENARIOS="${1#*=}" ;;
        --trace-reps=*)    TRACE_REPS="${1#*=}" ;;
        --trace-port=*)    TRACE_PORT="${1#*=}" ;;
        --jobs=*)          JOBS="${1#*=}" ;;
        --dry-run)         DRY_RUN=1 ;;
        *) echo "error: unknown option '$1'" >&2; usage 2 ;;
    esac
    shift
done

[[ -n "${PROFDATA}" ]] || PROFDATA="${PROFILE_DIR}/merged.profdata"

run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[pto-build][dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

profdata_tool() {
    for t in llvm-profdata-21 llvm-profdata; do
        command -v "$t" >/dev/null 2>&1 && { command -v "$t"; return; }
    done
    echo "error: llvm-profdata not found (clang-21 toolchain expected)" >&2
    exit 1
}

finetune_build() {
    local mode="$1"       # use | generate | cs-generate
    local args=( "--pto=${mode}" "--build-dir=${BUILD_DIR}" "--profile-dir=${PROFILE_DIR}" "--profdata=${PROFDATA}" )
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[pto-build][dry-run] %s %s\n' "${FINETUNE}" "${args[*]}"
        printf '    (run the same command with --dry-run appended to print the resolved cmake/build-guard plan)\n'
        return 0
    fi
    if [[ -n "${JOBS}" ]]; then
        BUILD_JOBS="${JOBS}" "${FINETUNE}" "${args[@]}"
    else
        "${FINETUNE}" "${args[@]}"
    fi
}

stage_status() {
    echo "PTO status"
    echo "  build dir    : ${BUILD_DIR}"
    local cache="${BUILD_DIR}/CMakeCache.txt"
    if [[ -f "${cache}" ]]; then
        grep -E "^GGML_PTO_MODE:|^GGML_PTO_PROFDATA:|^GGML_THINLTO:|^GGML_SPLIT_MACHINE_FUNCTIONS:" "${cache}" \
            | sed 's/^/  /'
    else
        echo "  (no CMake cache yet)"
    fi
    echo "  profile dir  : ${PROFILE_DIR}"
    local n=0
    [[ -d "${PROFILE_DIR}" ]] && n=$(find "${PROFILE_DIR}" -maxdepth 1 -name '*.profraw' | wc -l)
    echo "  raw profiles : ${n}"
    if [[ -f "${PROFDATA}" ]]; then
        echo "  merged       : ${PROFDATA} ($(stat -c '%s bytes, %y' "${PROFDATA}" | cut -d. -f1))"
    else
        echo "  merged       : absent (${PROFDATA})"
    fi
    local bin="${BUILD_DIR}/bin/llama-server"
    if [[ -f "${bin}" ]]; then
        echo "  llama-server : sha256 $(sha256sum "${bin}" | cut -c1-16)  $(stat -c '%s bytes, %y' "${bin}" | cut -d. -f1)"
        "${bin}" --version 2>/dev/null | head -2 | sed 's/^/    /' || true
    fi
}

stage_merge() {
    local tool; tool="$(profdata_tool)"
    local raw=()
    if [[ -d "${PROFILE_DIR}" ]]; then
        mapfile -t raw < <(find "${PROFILE_DIR}" -maxdepth 1 -name '*.profraw' | sort)
    fi
    if [[ ${#raw[@]} -eq 0 ]]; then
        echo "error: no .profraw files in ${PROFILE_DIR}" >&2
        exit 1
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo ">>> (dry-run) merging ${#raw[@]} raw profiles from ${PROFILE_DIR}"
        run mkdir -p "$(dirname "${PROFDATA}")"
        run "${tool}" merge -o "${PROFDATA}" "${raw[@]}"
        return 0
    fi
    # Validate inputs one by one: one truncated profile (e.g. a killed
    # instrumented run) makes `llvm-profdata merge` reject the whole batch.
    local good=() bad=()
    local tmp; tmp="$(mktemp -u "${TMPDIR:-/tmp}/pto_probe.XXXXXX.profdata")"
    for f in "${raw[@]}"; do
        if "${tool}" merge -o "${tmp}" "${f}" >/dev/null 2>&1; then
            good+=("${f}")
        else
            bad+=("${f}")
        fi
        rm -f "${tmp}"
    done
    if [[ ${#bad[@]} -gt 0 ]]; then
        echo ">>> skipping ${#bad[@]} unusable raw profile(s):"
        printf '      %s\n' "${bad[@]#${PROFILE_DIR}/}"
    fi
    if [[ ${#good[@]} -eq 0 ]]; then
        echo "error: no usable .profraw files in ${PROFILE_DIR}" >&2
        exit 1
    fi
    echo ">>> merging ${#good[@]}/${#raw[@]} raw profiles from ${PROFILE_DIR}"
    mkdir -p "$(dirname "${PROFDATA}")"
    "${tool}" merge -o "${PROFDATA}" "${good[@]}"
    echo ">>> wrote ${PROFDATA} ($(stat -c '%s bytes' "${PROFDATA}"))"
}

stage_trace_workload() {
    [[ "${TRACE_WORKLOAD}" == "none" ]] && return 0
    if [[ "${TRACE_WORKLOAD}" != "harness" ]]; then
        echo "error: --trace-workload must be none|harness" >&2
        exit 2
    fi
    [[ -n "${MODEL_KEY}" ]] || { echo "error: --model-key is required for --trace-workload=harness" >&2; exit 2; }
    [[ -n "${MODEL_ROOT}" ]] || { echo "error: --model-root (or \$LLAMA_MODEL_ROOT) is required for --trace-workload=harness" >&2; exit 2; }
    local cfg="${REPO_ROOT}/scripts/configs/models.toml"
    if ! python3 -c "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb'))" "${cfg}" 2>/dev/null; then
        if [[ -f "${REPO_ROOT}/scratch/configs/models_lto.toml" ]]; then
            echo ">>> ${cfg} is not parseable TOML right now; falling back to scratch/configs/models_lto.toml" >&2
            cfg="${REPO_ROOT}/scratch/configs/models_lto.toml"
        else
            echo "error: ${cfg} is not parseable TOML and no scratch fallback exists" >&2
            exit 2
        fi
    fi
    echo ">>> running trace workload (harness) so the instrumented binary writes .profraw"
    run python3 "${REPO_ROOT}/scripts/bench_ranked.py" \
        --config "${cfg}" --models "${MODEL_KEY}" --model-root "${MODEL_ROOT}" \
        --binary "${BUILD_DIR}/bin/llama-server" --port "${TRACE_PORT}" \
        --scenarios "${TRACE_SCENARIOS}" --reps "${TRACE_REPS}" --warmup-rounds 0 \
        --no-pmu --output "${PROFILE_DIR}/trace_workload_$(date -u +%Y%m%dT%H%M%SZ).json"
}

stage_print_flags() {
    local stage="${BUILD_GUARD_PTO:-use}"
    [[ "${STAGE}" != "flags" ]] && stage="${STAGE}"
    cat <<EOF
# ctx-capsule / fleet surface for this PTO workflow (source it)
PTO_STAGE="${stage}"
PTO_BUILD_DIR="${BUILD_DIR}"
PTO_PROFILE_DIR="${PROFILE_DIR}"
PTO_PROFDATA="${PROFDATA}"
# convenient env knob for the same wrapper:
export BUILD_GUARD_PTO=${stage}
EOF
}

case "${STAGE}" in
    status) stage_status ;;
    merge)  stage_merge ;;
    use)    finetune_build use ;;
    trace)  finetune_build generate; stage_trace_workload ;;
    cs)     finetune_build cs-generate; stage_trace_workload ;;
    train)  finetune_build generate; stage_trace_workload; stage_merge ;;
    all)    finetune_build generate; stage_trace_workload; stage_merge; finetune_build use ;;
    flags)  stage_print_flags ;;
    -h|--help|help) usage 0 ;;
    *) echo "error: unknown stage '${STAGE}' (status|trace|cs|merge|use|train|all|flags)" >&2; exit 2 ;;
esac
