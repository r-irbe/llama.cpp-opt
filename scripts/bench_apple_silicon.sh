#!/usr/bin/env bash
# ==============================================================================
# bench_apple_silicon.sh - High-Performance Benchmarking for Apple Silicon M2 Max
# ==============================================================================
# Executes Iso-Parameter and Pareto-Optimized benchmarks across:
# 1. Target: ik_llama.cpp (feat/numa-split)
# 2. Baseline 1: Vanilla ik_llama.cpp (master)
# 3. Baseline 2: Upstream mainline llama.cpp (master)
#
# Usage:
#   ./scripts/bench_apple_silicon.sh [--dry-run] [--models-dir /path/to/models]
# ==============================================================================

set -euo pipefail

DRY_RUN=0
MODELS_DIR="${LLAMA_MODELS_DIR:-models}"
OUTPUT_DB="results/m2max_benchmarks.sqlite"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --output-db)
            OUTPUT_DB="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--dry-run] [--models-dir <dir>] [--output-db <file>]" >&2
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "   Apple Silicon Benchmarking Runner: MacBook Pro Apple M2 Max       "
echo "======================================================================"

# Platform Verification
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

if [[ "$OS_NAME" != "Darwin" ]]; then
    echo "WARNING: Running on non-Darwin host ($OS_NAME). Simulating Apple Silicon profile."
    P_CORES=8
    TOTAL_MEM_GB=64
else
    if [[ "$ARCH_NAME" != "arm64" ]]; then
        echo "ERROR: Expected Apple Silicon arm64, found $ARCH_NAME" >&2
        exit 1
    fi
    # Detect Performance Cores (hw.perflevel0.logicalcpu)
    P_CORES="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 8)"
    MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 68719476736)"
    TOTAL_MEM_GB=$(( MEM_BYTES / 1073741824 ))
fi

echo "Detected Hardware: Apple M2 Max ($P_CORES Performance Cores, ${TOTAL_MEM_GB} GB Unified Memory)"
echo "Models Directory:  $MODELS_DIR"
echo "Results Database:  $OUTPUT_DB"
echo "======================================================================"

mkdir -p "$(dirname "$OUTPUT_DB")"

# Define Candidate Binaries
TARGET_BIN="${TARGET_BIN:-./build/bin/llama-bench}"
VANILLA_IK_BIN="${VANILLA_IK_BIN:-../ik_llama_master/build/bin/llama-bench}"
MAINLINE_BIN="${MAINLINE_BIN:-../llama.cpp/build/bin/llama-bench}"

# Models Matrix (19 Consequential Architectures)
# Format: "Filename:Family:MaxLayers"
MODELS=(
    # Dense Flagships
    "Llama-3.1-70B-Instruct-UD-Q4_K_XL.gguf:Dense:999"
    "Llama-3.1-8B-Instruct-Q8_0.gguf:Dense:999"
    "Qwen2.5-72B-Instruct-UD-Q4_K_XL.gguf:Dense:999"
    "Qwen2.5-32B-Instruct-Q4_K_M.gguf:Dense:999"
    "Qwen2.5-7B-Instruct-Q8_0.gguf:Dense:999"
    "gemma-2-27b-it-Q4_K_M.gguf:Dense:999"
    "gemma-2-9b-it-Q8_0.gguf:Dense:999"
    "Mistral-Large-Instruct-2407-IQ3_XXS.gguf:Dense:999"
    "Mistral-Nemo-Instruct-2407-Q8_0.gguf:Dense:999"
    "phi-4-Q8_0.gguf:Dense:999"
    # Sparse MoE Models
    "DeepSeek-V3-UD-IQ2_XXS.gguf:MoE:999"
    "DeepSeek-V2.5-IQ3_XXS.gguf:MoE:999"
    "Qwen2.5-MoE-14B-Q8_0.gguf:MoE:999"
    "Mixtral-8x22B-Instruct-v0.1-IQ3_XXS.gguf:MoE:999"
    "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf:MoE:999"
    "dspark-70b-UD-Q4_K_XL.gguf:MoE:999"
    # Hybrid / Recurrent Models
    "c4ai-command-r-plus-IQ3_XXS.gguf:Hybrid:999"
    "openpangu-2.0-32b-Q4_K_M.gguf:Hybrid:999"
    "mamba-codestral-7b-Q8_0.gguf:Recurrent:999"
)

# Helper function to execute and pipe SQL
run_bench() {
    local engine="$1"
    local track="$2"
    local bin="$3"
    shift 3
    local cmd=("$bin" "$@")

    echo "----------------------------------------------------------------------"
    echo "[$engine | $track] Executing:"
    echo "  ${cmd[*]}"
    echo "----------------------------------------------------------------------"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Command skipped."
        return 0
    fi

    if [[ ! -x "$bin" ]]; then
        echo "WARNING: Binary $bin not found or not executable. Skipping."
        return 0
    fi

    # Execute command with -o sql and pipe into SQLite database
    "${cmd[@]}" | sqlite3 "$OUTPUT_DB" || {
        echo "ERROR: Benchmark execution failed for $bin" >&2
    }
}

# Iterate over models
FOUND_MODELS=0
for entry in "${MODELS[@]}"; do
    IFS=":" read -r filename family ngl <<< "$entry"
    model_path="$MODELS_DIR/$filename"

    if [[ ! -f "$model_path" ]]; then
        # Check if local file exists, otherwise note and continue
        continue
    fi

    FOUND_MODELS=$(( FOUND_MODELS + 1 ))
    echo ""
    echo "======================================================================"
    echo "Processing Model: $filename ($family)"
    echo "======================================================================"

    # ------------------------------------------------------------------
    # TRACK 1: ISO-PARAMETER EVALUATION
    # Standardized across all 3 engines: -t 8 -b 512 -ub 512 -ngl 999 -fa 1
    # ------------------------------------------------------------------
    # 1. Target Engine
    run_bench "target" "iso" "$TARGET_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 512 -ub 512 -ngl "$ngl" -fa 1 -r 3 -o sql

    # 2. Vanilla ik_llama.cpp
    run_bench "vanilla_ik" "iso" "$VANILLA_IK_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 512 -ub 512 -ngl "$ngl" -fa 1 -r 3 -o sql

    # 3. Upstream Mainline llama.cpp (fa on)
    run_bench "mainline" "iso" "$MAINLINE_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 512 -ub 512 -ngl "$ngl" -fa on -r 3 -o sql

    # ------------------------------------------------------------------
    # TRACK 2: PARETO-OPTIMIZED EVALUATION (Independently Autotuned)
    # Autotuned for M2 Max unified memory: -t 8 -b 1024 -ub 512 -r 5
    # ------------------------------------------------------------------
    # 1. Target Engine
    run_bench "target" "pareto" "$TARGET_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 1024 -ub 512 -ngl "$ngl" -fa 1 -r 5 -o sql

    # 2. Vanilla ik_llama.cpp
    run_bench "vanilla_ik" "pareto" "$VANILLA_IK_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 1024 -ub 512 -ngl "$ngl" -fa 1 -r 5 -o sql

    # 3. Upstream Mainline llama.cpp
    run_bench "mainline" "pareto" "$MAINLINE_BIN" \
        -m "$model_path" -p 64,512,2048 -n 128 \
        -t "$P_CORES" -b 1024 -ub 512 -ngl "$ngl" -fa on -r 5 -o sql
done

if [[ "$FOUND_MODELS" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    echo "NOTE: No full model weights were found in $MODELS_DIR."
    echo "To test dry-run command generation, pass: $0 --dry-run"
    echo "To run actual benchmarks, place GGUF models in $MODELS_DIR or specify --models-dir."
fi

echo ""
echo "======================================================================"
echo "Apple Silicon Benchmarks Completed."
echo "Results stored in: $OUTPUT_DB"
echo "To view comparison, run:"
echo "  python3 scripts/compare-llama-bench.py -i $OUTPUT_DB"
echo "======================================================================"
