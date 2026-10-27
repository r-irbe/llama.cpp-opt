#!/usr/bin/env bash
# scripts/pto_finetune.sh - Profile-Targeted Optimization (PTO) Build & Finetuning Driver
#
# Supports:
#   --pto=generate|cs-generate|use|off
#   --shared-libs                 Build both static ThinLTO-inlined executables AND shared
#                                 libraries (libggml.so, libllama.so, plus optional Rust FFI)
#   --compiler=clang-21|clang-22  Select Clang-21 (default) or Clang-22 (+ -allow-unsupported-compiler)
#   --profile-dir=<dir>           Directory for .profraw traces (default: /tmp/ik_llama_pgo)
#   --profdata=<file>             Merged .profdata path (default: <profile-dir>/merged.profdata)
#   --build-dir=<dir>             Override build directory
#   --targets="<t1> <t2> ..."     Override CMake build targets
#   --dry-run                     Print resolved CMake & build-guard commands without executing
#
# Invariants:
#   - All C++/CUDA compilations execute strictly through scripts/build-guard.sh (AGENTS.md Rule 4).
#   - Single-GPU workstation builds set -DGGML_NCCL=OFF by default so binaries never fail with
#     missing libnccl.so.2 at runtime.
#   - Host C/CXX flags (-flto=thin, -fprofile-*, -fsplit-machine-functions) are scoped via
#     CMake $<COMPILE_LANGUAGE:C,CXX> generator expressions so nvcc device compilation is
#     never broken by host LLVM bitcode emission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_GUARD="${REPO_ROOT}/scripts/build-guard.sh"

PTO_MODE="use"
SHARED_LIBS=0
COMPILER_FLAVOR="clang-21"
PROFILE_DIR="/tmp/ik_llama_pgo"
CS_PROFILE_DIR="/tmp/ik_llama_cs_pgo"
PROFDATA_PATH=""
BUILD_DIR=""
TARGETS="llama-server llama-cli llama-bench llama-perplexity"
DRY_RUN=0
ENABLE_RUST_FFI=0

usage() {
    cat <<'EOF'
Usage: ./scripts/pto_finetune.sh [options]

Options:
  --pto=generate|cs-generate|use|off  Select PTO compilation stage (default: use)
  --shared-libs                       Build shared libraries (libggml.so, libllama.so)
                                      in addition to static LTO executables
  --compiler=clang-21|clang-22        Select LLVM/Clang toolchain (default: clang-21)
  --profile-dir=<path>                Raw .profraw output directory (default: /tmp/ik_llama_pgo)
  --profdata=<path>                   Merged .profdata input file for cs-generate / use
  --build-dir=<path>                  Explicit CMake build directory
  --targets="<t1> <t2> ..."           CMake targets (default: "llama-server llama-cli llama-bench llama-perplexity")
  --rust-ffi                          Also build & link rust-mmap-engine with znver3 fat-LTO
  --dry-run                           Print commands without executing
  -h, --help                          Show this help message
EOF
}

for arg in "$@"; do
    case "$arg" in
        --pto=*)
            PTO_MODE="${arg#*=}"
            ;;
        --shared-libs)
            SHARED_LIBS=1
            ;;
        --compiler=*)
            COMPILER_FLAVOR="${arg#*=}"
            ;;
        --profile-dir=*)
            PROFILE_DIR="${arg#*=}"
            ;;
        --profdata=*)
            PROFDATA_PATH="${arg#*=}"
            ;;
        --build-dir=*)
            BUILD_DIR="${arg#*=}"
            ;;
        --targets=*)
            TARGETS="${arg#*=}"
            ;;
        --rust-ffi)
            ENABLE_RUST_FFI=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$COMPILER_FLAVOR" in
    clang-21|clang21)
        CC_BIN="${CC_BIN:-/usr/bin/clang-21}"
        CXX_BIN="${CXX_BIN:-/usr/bin/clang++-21}"
        LLD_BIN="${LLD_BIN:-/usr/bin/ld.lld-21}"
        PROFDATA_BIN="${PROFDATA_BIN:-/usr/bin/llvm-profdata-21}"
        ALLOW_UNSUPPORTED_CUDA=OFF
        ;;
    clang-22|clang22)
        CC_BIN="${CC_BIN:-$(command -v clang-22 || echo /usr/bin/clang-21)}"
        CXX_BIN="${CXX_BIN:-$(command -v clang++-22 || echo /usr/bin/clang++-21)}"
        LLD_BIN="${LLD_BIN:-$(command -v ld.lld-22 || echo /usr/bin/ld.lld-21)}"
        PROFDATA_BIN="${PROFDATA_BIN:-$(command -v llvm-profdata-22 || echo /usr/bin/llvm-profdata-21)}"
        ALLOW_UNSUPPORTED_CUDA=ON
        ;;
    *)
        echo "Unsupported --compiler=${COMPILER_FLAVOR} (expected clang-21 or clang-22)" >&2
        exit 2
        ;;
esac

if [[ -z "$PROFDATA_PATH" ]]; then
    PROFDATA_PATH="${PROFILE_DIR}/merged.profdata"
fi

case "$PTO_MODE" in
    generate)
        CMAKE_PTO_MODE="GENERATE"
        DEFAULT_BUILD_DIR="build-pto-gen"
        ACTIVE_PROFILE_DIR="$PROFILE_DIR"
        THINLTO_FLAG=OFF
        SPLIT_FUNCS_FLAG=OFF
        ;;
    cs-generate|cs_generate)
        CMAKE_PTO_MODE="CS_GENERATE"
        DEFAULT_BUILD_DIR="build-pto-cs"
        ACTIVE_PROFILE_DIR="$CS_PROFILE_DIR"
        THINLTO_FLAG=OFF
        SPLIT_FUNCS_FLAG=OFF
        ;;
    use)
        CMAKE_PTO_MODE="USE"
        DEFAULT_BUILD_DIR="build-pto"
        ACTIVE_PROFILE_DIR="$PROFILE_DIR"
        THINLTO_FLAG=ON
        SPLIT_FUNCS_FLAG=ON
        ;;
    off|none)
        CMAKE_PTO_MODE="OFF"
        DEFAULT_BUILD_DIR="build-clang"
        ACTIVE_PROFILE_DIR="$PROFILE_DIR"
        THINLTO_FLAG=ON
        SPLIT_FUNCS_FLAG=ON
        ;;
    *)
        echo "Unsupported --pto=${PTO_MODE} (expected generate, cs-generate, use, or off)" >&2
        exit 2
        ;;
esac

if [[ -z "$BUILD_DIR" ]]; then
    BUILD_DIR="$DEFAULT_BUILD_DIR"
fi

run_cmd() {
    echo "  + $*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

echo "================================================================================"
echo "  ik_llama.cpp PTO Finetuning Pipeline"
echo "  Mode              : ${PTO_MODE} (GGML_PTO_MODE=${CMAKE_PTO_MODE})"
echo "  Toolchain         : CC=${CC_BIN} CXX=${CXX_BIN} LLD=${LLD_BIN}"
echo "  Profile Directory : ${ACTIVE_PROFILE_DIR}"
echo "  Merged Profdata   : ${PROFDATA_PATH}"
echo "  Primary Build Dir : ${BUILD_DIR}"
echo "  Shared Libs Build : $([[ "$SHARED_LIBS" -eq 1 ]] && echo "YES (${BUILD_DIR}-shared)" || echo "NO (static LTO executables only)")"
echo "  Targets           : ${TARGETS}"
echo "================================================================================"

if [[ "$ENABLE_RUST_FFI" -eq 1 ]]; then
    RUST_DIR="${REPO_ROOT}/../rust-mmap-engine"
    if [[ -d "$RUST_DIR" ]] || [[ "$DRY_RUN" -eq 1 ]]; then
        echo ">>> [Rust FFI] Building rust-mmap-engine with znver3 + fat LTO..."
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "  + env RUSTFLAGS='-C target-cpu=znver3 -C codegen-units=1 -C lto=fat' cargo build --release --manifest-path ${RUST_DIR}/Cargo.toml"
        else
            env RUSTFLAGS="-C target-cpu=znver3 -C codegen-units=1 -C lto=fat" \
                cargo build --release --manifest-path "${RUST_DIR}/Cargo.toml"
        fi
    fi
fi

COMMON_CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=${CC_BIN}"
    "-DCMAKE_CXX_COMPILER=${CXX_BIN}"
    "-DCMAKE_CUDA_HOST_COMPILER=${CXX_BIN}"
    -DGGML_CUDA=ON
    -DGGML_CUDA_F16=ON
    -DGGML_AVX2=ON
    -DGGML_FMA=ON
    -DGGML_F16C=ON
    -DGGML_IQK_FA_ALL_QUANTS=ON
    -DGGML_CUDA_FA_ALL_QUANTS=ON
    -DGGML_Scheduler=ON
    -DGGML_NCCL=OFF
    "-DGGML_PTO_MODE=${CMAKE_PTO_MODE}"
    "-DGGML_PTO_PROFILE_DIR=${ACTIVE_PROFILE_DIR}"
    "-DGGML_PTO_PROFDATA=${PROFDATA_PATH}"
    "-DGGML_THINLTO=${THINLTO_FLAG}"
    "-DGGML_SPLIT_MACHINE_FUNCTIONS=${SPLIT_FUNCS_FLAG}"
    -DGGML_USE_LLD=ON
    "-DGGML_CUDA_ALLOW_UNSUPPORTED_COMPILER=${ALLOW_UNSUPPORTED_CUDA}"
)

if [[ "$ENABLE_RUST_FFI" -eq 1 ]]; then
    COMMON_CMAKE_ARGS+=(-DGGML_RUST_MMAP_ENGINE=ON)
fi

# 1. Static LTO-inlined executables build (BUILD_SHARED_LIBS=OFF)
echo ">>> [1] Configuring static LTO-inlined executables in '${BUILD_DIR}'..."
run_cmd cmake -B "${BUILD_DIR}" "${COMMON_CMAKE_ARGS[@]}" -DBUILD_SHARED_LIBS=OFF

# shellcheck disable=SC2086
run_cmd "${BUILD_GUARD}" cmake --build "${BUILD_DIR}" --target ${TARGETS} -j "$(nproc)"

# 2. Optional Shared Libraries build (libggml.so + libllama.so)
if [[ "$SHARED_LIBS" -eq 1 ]]; then
    SHARED_BUILD_DIR="${BUILD_DIR}-shared"
    echo ">>> [2] Configuring PTO-optimized shared libraries (libggml.so, libllama.so) in '${SHARED_BUILD_DIR}'..."
    run_cmd cmake -B "${SHARED_BUILD_DIR}" "${COMMON_CMAKE_ARGS[@]}" -DBUILD_SHARED_LIBS=ON
    # shellcheck disable=SC2086
    run_cmd "${BUILD_GUARD}" cmake --build "${SHARED_BUILD_DIR}" --target ggml llama ${TARGETS} -j "$(nproc)"
    run_cmd mkdir -p "${BUILD_DIR}/ggml/src" "${BUILD_DIR}/src"
    run_cmd cp -f "${SHARED_BUILD_DIR}/ggml/src/libggml.so" "${BUILD_DIR}/ggml/src/libggml.so"
    run_cmd cp -f "${SHARED_BUILD_DIR}/src/libllama.so" "${BUILD_DIR}/src/libllama.so"
fi

echo "================================================================================"
echo "  PTO Finetuning Stage '${PTO_MODE}' Complete."
echo "  Binaries : ${BUILD_DIR}/bin/{llama-server,llama-cli,llama-bench,llama-perplexity}"
if [[ "$SHARED_LIBS" -eq 1 ]]; then
    echo "  Libraries: ${BUILD_DIR}/ggml/src/libggml.so, ${BUILD_DIR}/src/libllama.so"
fi
echo "================================================================================"
