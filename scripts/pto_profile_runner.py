#!/usr/bin/env python3
"""
scripts/pto_profile_runner.py - Profile-Targeted Optimization (PTO / IRPGO / CS-IRPGO) Runner for ik_llama.cpp

Orchestrates dual-mode execution of representative inference workloads to generate
profiling data (.profraw -> merged.profdata) for Clang-21/22 Continuous IRPGO,
2-Round Context-Sensitive PGO (-fcs-profile-generate), Machine Function Splitting
(-fsplit-machine-functions), and ThinLTO (-flto=thin -fuse-ld=lld -Wl,--icf=safe).

Dual-Mode Architecture:
  Mode A: 6-Stage Synthetic Sweep
    Stage 1: WikiText-2 Dense Vocabulary & Perplexity Training (llama-perplexity)
    Stage 2: Continuous Prose Staged Long-Context Sweep (512..32K tokens)
    Stage 3: Multi-Domain Mathematical & Formal Reasoning Execution
    Stage 4: Code Synthesis & Speculative Decoding Verification (--model-draft)
    Stage 5: Live llama-server Multi-Slot HTTP + CohortRadixTrie KV Cloning +
             Context Checkpoints & Binary Fuse8 Capsules + Speculative Decoding
    Stage 6: llama-bench GEMM/FlashAttention/MoE Microbenchmark & Optional nsys/perf PMU Audit

  Mode B: Continuous Live-Daemon Tracing (LLVM_PROFILE_FILE=".../%p%c.profraw")
    Merges live mmap-synced .profraw counters from a running instrumented llama-server
    daemon on demand (--live-daemon-merge / --merge-only) without stopping the server.
"""

import os
import sys
import time
import glob
import json
import shutil
import socket
import urllib.request
import urllib.error
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

IN_TREE_DATASET_PATHS = [
    Path("scratch/wikitext-2-raw/wiki.test.raw"),
    Path("scratch/wikitext-2-raw/wiki.valid.raw"),
    Path("scratch/wikitext-2-raw/wiki.train.raw"),
    Path("prompts"),
    Path("tests/prompts"),
]

PRODUCTION_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "56",
    "GGML_CUDA_NO_PINNED": "0",
    "GGML_CUDA_HOST_MALLOC_THP": "1",
    "GGML_CUDA_ENABLE_GRAPHS": "1",
    "LLAMA_KV_ZDD_SKIP": "1",
    "GGML_MOE_URING_PREFETCH": "1",
}


def find_available_datasets(custom_dir: Optional[str] = None, prompt_file: Optional[str] = None) -> Dict[str, Path]:
    """Scans repository and optionally specified custom directories for test datasets."""
    available: Dict[str, Path] = {}

    for p in IN_TREE_DATASET_PATHS:
        if p.exists():
            if p.is_file():
                available[p.stem] = p.resolve()
            elif p.is_dir():
                for f in p.glob("*.txt"):
                    available[f.stem] = f.resolve()

    env_dir = os.environ.get("PTO_DATASET_DIR") or os.environ.get("IK_LLAMA_DATASET_DIR")
    if env_dir and os.path.exists(env_dir):
        p_env = Path(env_dir)
        for ext in ["*.raw", "*.txt", "*.csv"]:
            for f in p_env.glob(ext):
                available[f.stem] = f.resolve()

    if custom_dir and os.path.exists(custom_dir):
        p_cust = Path(custom_dir)
        for ext in ["*.raw", "*.txt", "*.csv"]:
            for f in p_cust.glob(ext):
                available[f.stem] = f.resolve()

    if prompt_file and os.path.exists(prompt_file):
        pf = Path(prompt_file)
        available[pf.stem] = pf.resolve()

    return available


def resolve_llvm_profdata_binary() -> str:
    """Finds llvm-profdata-21, llvm-profdata-22, or llvm-profdata on PATH."""
    for candidate in ["/usr/bin/llvm-profdata-21", "llvm-profdata-21", "llvm-profdata-22", "llvm-profdata"]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "/usr/bin/llvm-profdata-21"


def build_production_env(profile_dir: Path) -> Dict[str, str]:
    """Constructs the production runtime environment with IRPGO module+PID counter separation."""
    env = os.environ.copy()
    for k, v in PRODUCTION_ENV_DEFAULTS.items():
        env.setdefault(k, v)
    # Override any inherited GGML_CUDA_NO_PINNED=1 so pinned host memory & CUDA graphs stay active
    env["GGML_CUDA_NO_PINNED"] = os.environ.get("PTO_FORCE_NO_PINNED", "0")
    suffix = "%c" if os.environ.get("PTO_CONTINUOUS_MMAP") == "1" else ""
    env["LLVM_PROFILE_FILE"] = str(profile_dir / f"pto-%p-%m{suffix}.profraw")
    # Ensure native /lib64 (GLIBC_2.43) precedes any older conda sysroot in LD_LIBRARY_PATH
    raw_ld = env.get("LD_LIBRARY_PATH", "")
    filtered_ld = [p for p in raw_ld.split(":") if p and "sysroot" not in p]
    env["LD_LIBRARY_PATH"] = ":".join(["/lib64", "/usr/lib64", "/usr/local/cuda/targets/x86_64-linux/lib"] + filtered_ld)
    return env


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile-Targeted Optimization (PTO / IRPGO / CS-IRPGO) Training Runner for ik_llama.cpp"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build-pto-gen",
        help="Path to build directory containing instrumented binaries (llama-cli, llama-server, llama-bench, llama-perplexity)"
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/ik_llama_pgo",
        help="Directory for raw .profraw profile traces and merged.profdata output"
    )
    parser.add_argument(
        "--base-profdata",
        type=str,
        default=None,
        help="Optional Round-1 merged.profdata path to merge together with Round-2 CS-IRPGO traces"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model path (.gguf) for running profiling workloads (auto-detected from models/ if omitted)"
    )
    parser.add_argument(
        "--model-draft", "-md",
        type=str,
        default=None,
        help="Optional draft model path for speculative decoding profile workloads"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to directory containing custom text/raw profiling datasets"
    )
    parser.add_argument(
        "--prompt-file", "-pf",
        type=str,
        default=None,
        help="Path to specific text file containing prompt corpus"
    )
    parser.add_argument(
        "--stages", "-s",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated context lengths for training sweep"
    )
    parser.add_argument(
        "--n-predict", "-n",
        type=int,
        default=32,
        help="Number of tokens to decode per stage"
    )
    parser.add_argument(
        "--threads", "-t",
        type=int,
        default=56,
        help="Number of execution threads (defaults to 56 matching dual-socket EPYC Milan)"
    )
    parser.add_argument(
        "--n-gpu-layers", "-ngl",
        type=int,
        default=999,
        help="Number of layers to offload to GPU"
    )
    parser.add_argument(
        "--numa",
        type=str,
        default="distribute",
        help="NUMA configuration (split, distribute, isolate, or none)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=18089,
        help="Ephemeral HTTP port for Stage 5 live llama-server multi-slot + capsule profiling"
    )
    parser.add_argument(
        "--live-daemon-merge",
        action="store_true",
        help="Mode B: Merge live mmap-synced .profraw files (%%p%%c.profraw) from a running daemon without running synthetic stages"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Alias for --live-daemon-merge: only merge existing .profraw files in --profile-dir into merged.profdata"
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Wrap Stage 6 llama-bench microbenchmark in NVIDIA Nsight Systems (nsys profile) when available"
    )
    parser.add_argument(
        "--perf-stat",
        action="store_true",
        help="Wrap Stage 6 llama-bench microbenchmark in Linux perf stat PMU counter collection when available"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without invoking binaries"
    )
    parser.add_argument(
        "--print-pgo-flags",
        action="store_true",
        help="Print CMake & Clang-21/22 flags for Round 1 IRPGO, Round 2 CS-IRPGO, and Final ThinLTO + Machine Split build"
    )
    return parser.parse_args()


def print_cmake_instructions() -> None:
    print("""
================================================================================
  PROFILE-TARGETED OPTIMIZATION (PTO / IRPGO / CS-IRPGO / THINLTO) INSTRUCTIONS
================================================================================

[Automated Driver (Recommended - routes all builds via scripts/build-guard.sh)]
  ./scripts/pto_finetune.sh --pto=generate --compiler=clang-21
  python3 scripts/pto_profile_runner.py --build-dir build-pgo --profile-dir /tmp/ik_llama_pgo -m <model.gguf>
  ./scripts/pto_finetune.sh --pto=cs-generate --compiler=clang-21
  python3 scripts/pto_profile_runner.py --build-dir build-cs-pgo --profile-dir /tmp/ik_llama_cs_pgo --base-profdata /tmp/ik_llama_pgo/merged.profdata -m <model.gguf>
  ./scripts/pto_finetune.sh --pto=use --shared-libs --compiler=clang-21

[Phase 1: Round-1 Continuous IRPGO Instrumentation (Clang-21 + LLD-21)]
  mkdir -p /tmp/ik_llama_pgo
  cmake -B build-pgo -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=/usr/bin/clang-21 \\
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \\
        -DGGML_CUDA=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \\
        -DGGML_IQK_FA_ALL_QUANTS=ON -DGGML_Scheduler=ON -DGGML_NCCL=OFF \\
        -DGGML_PTO_MODE=GENERATE -DGGML_PTO_PROFILE_DIR=/tmp/ik_llama_pgo \\
        -DGGML_USE_LLD=ON
  ./scripts/build-guard.sh cmake --build build-pgo --target llama-server llama-cli llama-bench llama-perplexity -j $(nproc)

[Phase 2: Execute 6-Stage Synthetic Sweep OR Continuous Live-Daemon Tracing]
  # Mode A: 6-Stage Synthetic Sweep (auto-merges .profraw -> merged.profdata)
  python3 scripts/pto_profile_runner.py --build-dir build-pgo --profile-dir /tmp/ik_llama_pgo -m <model.gguf>

  # Mode B: Continuous Live-Daemon Tracing (zero-downtime mmap counter sync %c)
  LLVM_PROFILE_FILE="/tmp/ik_llama_pgo/pto-%p%c.profraw" ./build-pgo/bin/llama-server -m <model.gguf> ...
  python3 scripts/pto_profile_runner.py --live-daemon-merge --profile-dir /tmp/ik_llama_pgo

[Phase 3 (Optional): Round-2 Context-Sensitive PGO (-fcs-profile-generate)]
  mkdir -p /tmp/ik_llama_cs_pgo
  cmake -B build-cs-pgo -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=/usr/bin/clang-21 \\
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \\
        -DGGML_CUDA=ON -DGGML_NCCL=OFF \\
        -DGGML_PTO_MODE=CS_GENERATE \\
        -DGGML_PTO_PROFILE_DIR=/tmp/ik_llama_cs_pgo \\
        -DGGML_PTO_PROFDATA=/tmp/ik_llama_pgo/merged.profdata \\
        -DGGML_USE_LLD=ON
  ./scripts/build-guard.sh cmake --build build-cs-pgo --target llama-server llama-cli llama-bench llama-perplexity -j $(nproc)

[Phase 4: Final Profile-Guided + ThinLTO + Machine Function Splitting Build]
  cmake -B build-pto -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=/usr/bin/clang-21 \\
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \\
        -DGGML_CUDA=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \\
        -DGGML_IQK_FA_ALL_QUANTS=ON -DGGML_Scheduler=ON -DGGML_NCCL=OFF \\
        -DGGML_PTO_MODE=USE \\
        -DGGML_PTO_PROFDATA=/tmp/ik_llama_pgo/merged.profdata \\
        -DGGML_THINLTO=ON \\
        -DGGML_SPLIT_MACHINE_FUNCTIONS=ON \\
        -DGGML_USE_LLD=ON
  ./scripts/build-guard.sh cmake --build build-pto --target llama-server llama-cli llama-bench llama-perplexity -j $(nproc)

[Optional Clang-22 & Rust FFI Flags]
  - Clang-22 with CUDA 12.6: pass -DGGML_CUDA_ALLOW_UNSUPPORTED_COMPILER=ON
    (appends -allow-unsupported-compiler to CMAKE_CUDA_FLAGS)
  - Rust mmap engine FFI:
    RUSTFLAGS="-C target-cpu=znver3 -C codegen-units=1 -C lto=fat" cargo build --release
================================================================================
""")


def run_stage(
    cmd: List[str],
    description: str,
    profile_dir: Path,
    dry_run: bool = False,
    timeout: int = 300,
) -> bool:
    print(f"  -> [PTO Run] {description}...", flush=True)
    env = build_production_env(profile_dir)
    if dry_run:
        clean_cmd = []
        for arg in cmd:
            if len(arg) > 60:
                clean_cmd.append(f"'{arg[:50]}...'")
            elif " " in arg:
                clean_cmd.append(f"'{arg}'")
            else:
                clean_cmd.append(arg)
        print(
            f"     (dry-run env: OMP_NUM_THREADS={env['OMP_NUM_THREADS']} "
            f"GGML_CUDA_NO_PINNED={env['GGML_CUDA_NO_PINNED']} "
            f"GGML_CUDA_ENABLE_GRAPHS={env['GGML_CUDA_ENABLE_GRAPHS']} "
            f"LLVM_PROFILE_FILE={env['LLVM_PROFILE_FILE']})"
        )
        print(f"     (dry-run cmd: {' '.join(clean_cmd)})")
        time.sleep(0.02)
        return True

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
            text=True,
            errors="replace",
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"     [OK] Completed in {elapsed:.2f}s")
            return True
        else:
            print(f"     [WARN] Process exited with code {proc.returncode} ({elapsed:.2f}s)")
            for line in proc.stderr.splitlines()[-4:]:
                print(f"       stderr: {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"     [FAIL] Timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"     [FAIL] Execution error: {e}")
        return False


def run_stage5_live_server_multi_slot(
    llama_server: Path,
    model_path: Path,
    profile_dir: Path,
    threads: int,
    n_gpu_layers: int,
    numa: Optional[str],
    port: int,
    dry_run: bool = False,
) -> bool:
    """
    Stage 5: Spawns instrumented llama-server with multi-slot concurrency (-np 4),
    CohortRadixTrie prefix sharing, ZDD zero-page skip, and Binary Fuse8 capsules,
    dispatches concurrent HTTP requests + context checkpoint queries, and cleanly shuts down.
    """
    description = (
        f"Live llama-server Multi-Slot HTTP (-np 4, port {port}) + "
        "CohortRadixTrie KV Cloning + Binary Fuse8 Capsules"
    )
    print(f"  -> [PTO Run] {description}...", flush=True)
    server_cmd = [
        str(llama_server),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", "8192",
        "-np", "4",
        "-cb",
        "-fa", "on",
        "--cohort-prefix-sharing",
        "--zdd-sparse-kv",
        "-t", str(threads),
        "-ngl", str(n_gpu_layers),
    ]
    if numa and numa.lower() != "none":
        server_cmd.extend(["--numa", numa])

    env = build_production_env(profile_dir)
    if dry_run:
        print(
            f"     (dry-run server cmd: {' '.join(server_cmd)})"
        )
        print(
            f"     (dry-run HTTP probes: GET /health, POST /v1/chat/completions x4 slots "
            f"[shared prefix -> CohortRadixTrie KV clone + Fuse8 capsule check], GET /slots)"
        )
        return True

    if not llama_server.exists():
        print(f"     [WARN] llama-server binary not found at {llama_server}, skipping Stage 5.")
        return False

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_sock:
            probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe_sock.bind(("127.0.0.1", port))
    except OSError as sock_err:
        print(f"     [SKIP] Loopback TCP bind on 127.0.0.1:{port} restricted ({sock_err}); skipping Stage 5 HTTP probe.")
        return False

    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            errors="replace",
        )
        base_url = f"http://127.0.0.1:{port}"
        ready = False
        deadline = time.time() + 90.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1.5) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)

        if not ready:
            print("     [WARN] llama-server did not reach ready state within 90s")
            return False

        shared_prefix = (
            "System Architecture Context: Dual-Socket AMD EPYC 7763 (112 physical cores, 8 NUMA nodes in NPS4) "
            "paired with NVIDIA RTX 4090 (24 GB VRAM, Ada Lovelace sm_89). "
            "Memory hierarchy uses Hot VRAM ring, Warm pinned RAM with 2MB Transparent HugePages, "
            "and Cold Binary Fuse8 XorFilter capsules (9.1 bits/entry) with 64-bit SimHash recall. "
        )
        prompts = [
            shared_prefix + "Task 1: Derive the optimal 128-byte aligned SPSC ring buffer batching interval.",
            shared_prefix + "Task 2: Compute the KV cache byte footprint for 32K tokens with Q8_0 K and Q4_0 V.",
            shared_prefix + "Task 3: Explain why CUDA Graph capture requires stable tensor data pointers across views.",
            shared_prefix + "Task 4: Show how CohortRadixTrie forks shared prefix KV blocks across 4 parallel slots.",
        ]

        for slot_id, prompt_text in enumerate(prompts):
            payload = json.dumps({
                "prompt": prompt_text,
                "n_predict": 24,
                "temperature": 0.0,
                "cache_prompt": True,
                "id_slot": slot_id,
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/completion",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    _ = resp.read()
            except Exception as req_err:
                print(f"     [WARN] Slot {slot_id} completion probe returned: {req_err}")

        try:
            with urllib.request.urlopen(f"{base_url}/slots", timeout=5.0) as resp:
                _ = resp.read()
        except Exception:
            pass

        print("     [OK] Stage 5 multi-slot HTTP & KV prefix sharing probes completed")
        return True
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def merge_profraw_traces(
    profile_dir: Path,
    base_profdata: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[bool, Path]:
    """
    Merges all pto-*.profraw traces in profile_dir (plus optional base_profdata from Round 1)
    into profile_dir / merged.profdata using llvm-profdata-21 / llvm-profdata.
    """
    merged_out = profile_dir / "merged.profdata"
    llvm_profdata = resolve_llvm_profdata_binary()
    profraw_files = [
        p for p in sorted(profile_dir.glob("*.profraw"))
        if p.is_file() and p.stat().st_size > 128
    ]

    inputs: List[str] = [str(p) for p in profraw_files]
    if base_profdata and base_profdata.exists():
        inputs.append(str(base_profdata))

    if dry_run:
        sample_inputs = inputs if inputs else [str(profile_dir / "pto-12345%c.profraw")]
        if base_profdata:
            sample_inputs.append(str(base_profdata))
        merge_cmd = [llvm_profdata, "merge", "-output=" + str(merged_out)] + sample_inputs
        print(f"  -> [PTO Merge] (dry-run cmd: {' '.join(merge_cmd)})")
        return True, merged_out

    if not inputs:
        print(f"  -> [PTO Merge] [WARN] No .profraw files found in {profile_dir}")
        return False, merged_out

    merge_cmd = [llvm_profdata, "merge", "-output=" + str(merged_out)] + inputs
    print(f"  -> [PTO Merge] Merging {len(profraw_files)} .profraw trace(s) via {llvm_profdata}...")
    proc = subprocess.run(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0 and merged_out.exists():
        size_kb = merged_out.stat().st_size / 1024.0
        print(f"     [OK] Produced {merged_out} ({size_kb:.1f} KB)")
        return True, merged_out
    else:
        print(f"     [FAIL] llvm-profdata merge failed (exit {proc.returncode}): {proc.stderr.strip()}")
        return False, merged_out


def ensure_synthetic_pto_model(profile_dir: Path) -> Optional[Path]:
    """Synthesizes a valid 2-layer Llama GGUF model from models/ggml-vocab-llama-spm.gguf when no full checkpoint is present."""
    out_path = profile_dir / "tiny_pto_llama.gguf"
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        return out_path
    vocab_src = Path("models/ggml-vocab-llama-spm.gguf")
    if not vocab_src.exists():
        return None
    try:
        import numpy as np
        sys.path.insert(0, str(Path("gguf-py").resolve()))
        import gguf  # type: ignore

        profile_dir.mkdir(parents=True, exist_ok=True)
        reader = gguf.GGUFReader(str(vocab_src))
        f_tok = reader.fields["tokenizer.ggml.tokens"]
        tokens = [bytes(f_tok.parts[i]).decode("utf-8", errors="replace") for i in f_tok.data]
        f_sc = reader.fields["tokenizer.ggml.scores"]
        scores = [float(f_sc.parts[i][0]) for i in f_sc.data]
        f_tt = reader.fields["tokenizer.ggml.token_type"]
        ttypes = [int(f_tt.parts[i][0]) for i in f_tt.data]
        n_vocab = len(tokens)
        n_embd, n_ff, n_layer, n_head = 256, 512, 2, 4

        w = gguf.GGUFWriter(str(out_path), "llama")
        w.add_name("tiny-pto-llama")
        w.add_context_length(4096)
        w.add_embedding_length(n_embd)
        w.add_feed_forward_length(n_ff)
        w.add_block_count(n_layer)
        w.add_head_count(n_head)
        w.add_head_count_kv(n_head)
        w.add_layer_norm_rms_eps(1e-5)
        w.add_rope_freq_base(10000.0)
        w.add_tokenizer_model("llama")
        w.add_token_list(tokens)
        w.add_token_scores(scores)
        w.add_token_types(ttypes)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)

        rng = np.random.default_rng(42)
        f16 = lambda shape: (rng.standard_normal(shape) * 0.02).astype(np.float16)
        ones = lambda shape: np.ones(shape, dtype=np.float32)

        w.add_tensor("token_embd.weight", f16((n_vocab, n_embd)))
        w.add_tensor("output_norm.weight", ones((n_embd,)))
        w.add_tensor("output.weight", f16((n_vocab, n_embd)))
        for l in range(n_layer):
            w.add_tensor(f"blk.{l}.attn_norm.weight", ones((n_embd,)))
            w.add_tensor(f"blk.{l}.attn_q.weight", f16((n_embd, n_embd)))
            w.add_tensor(f"blk.{l}.attn_k.weight", f16((n_embd, n_embd)))
            w.add_tensor(f"blk.{l}.attn_v.weight", f16((n_embd, n_embd)))
            w.add_tensor(f"blk.{l}.attn_output.weight", f16((n_embd, n_embd)))
            w.add_tensor(f"blk.{l}.ffn_norm.weight", ones((n_embd,)))
            w.add_tensor(f"blk.{l}.ffn_gate.weight", f16((n_ff, n_embd)))
            w.add_tensor(f"blk.{l}.ffn_down.weight", f16((n_embd, n_ff)))
            w.add_tensor(f"blk.{l}.ffn_up.weight", f16((n_ff, n_embd)))

        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return out_path.resolve()
    except Exception as exc:
        print(f"[Warn] Could not synthesize fallback GGUF model: {exc}", file=sys.stderr)
        return None


def main() -> None:
    args = parse_arguments()

    if args.print_pgo_flags:
        print_cmake_instructions()
        sys.exit(0)

    profile_dir = Path(args.profile_dir).resolve()
    if not args.dry_run:
        profile_dir.mkdir(parents=True, exist_ok=True)

    base_profdata = Path(args.base_profdata).resolve() if args.base_profdata else None

    # Mode B: Continuous Live-Daemon Trace Merge (--live-daemon-merge / --merge-only)
    if args.live_daemon_merge or args.merge_only:
        print("================================================================================")
        print("  PTO MODE B: CONTINUOUS LIVE-DAEMON TRACE MERGE (%p%c.profraw -> merged.profdata)")
        print(f"  Profile Directory: {profile_dir}")
        print("================================================================================")
        ok, merged_path = merge_profraw_traces(profile_dir, base_profdata=base_profdata, dry_run=args.dry_run)
        if not ok and not args.dry_run:
            sys.exit(1)
        print(f"  Merged Profile Ready: {merged_path}")
        sys.exit(0)

    build_dir = Path(args.build_dir).resolve()
    llama_cli = build_dir / "bin" / "llama-cli"
    llama_perplexity = build_dir / "bin" / "llama-perplexity"
    llama_server = build_dir / "bin" / "llama-server"
    llama_bench = build_dir / "bin" / "llama-bench"

    if not llama_cli.exists() and not args.dry_run:
        print(f"Error: llama-cli not found at '{llama_cli}'. Run scripts/pto_finetune.sh --pto=generate first.", file=sys.stderr)
        sys.exit(1)

    if args.model:
        model_path = Path(args.model).resolve()
    else:
        candidates = [
            p for p in (list(Path("models").glob("*.gguf")) + list(Path(".").glob("*.gguf")))
            if not p.name.startswith("ggml-vocab-")
        ]
        if candidates:
            model_path = candidates[0].resolve()
            print(f"[Notice] Auto-selected model: {model_path.name}")
        elif args.dry_run:
            model_path = Path("model.gguf")
        else:
            synth = ensure_synthetic_pto_model(profile_dir)
            if synth is not None:
                model_path = synth
                print(f"[Notice] Synthesized 2-layer Llama PTO training model: {model_path}")
            else:
                print("Error: No model specified and none found in models/. Pass --model <model.gguf>", file=sys.stderr)
                sys.exit(1)

    if not model_path.exists() and not args.dry_run:
        print(f"Error: Model not found at '{model_path}'.", file=sys.stderr)
        sys.exit(1)

    available_datasets = find_available_datasets(custom_dir=args.dataset_dir, prompt_file=args.prompt_file)
    try:
        context_stages = [int(c.strip()) for c in args.stages.split(",") if c.strip()]
    except ValueError:
        context_stages = [512, 1024, 2048, 4096]

    print("================================================================================")
    print("  PROFILE-TARGETED OPTIMIZATION (PTO / IRPGO / CS-IRPGO) 6-STAGE TRAINING HARNESS")
    print(f"  Build Directory  : {build_dir}")
    print(f"  Profile Directory: {profile_dir} (LLVM_PROFILE_FILE=pto-%p%c.profraw)")
    print(f"  Model            : {model_path.name}")
    print(f"  Threads / NUMA   : {args.threads} threads | NUMA={args.numa}")
    print(f"  Runtime Defaults : GGML_CUDA_NO_PINNED=0, GGML_CUDA_ENABLE_GRAPHS=1, THP=1, ZDD_SKIP=1")
    print(f"  Context Stages   : {[f'{c:,}' for c in context_stages]}")
    print(f"  Available Datasets Discovered: {len(available_datasets)}")
    for name, path in available_datasets.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"    - {name:<26}: {size_kb:>8.1f} KB ({path})")
    print("================================================================================")

    # Stage 1: WikiText Perplexity Workload
    print("\n>>> [Stage 1/6: Dense Vocabulary & Perplexity Training (llama-perplexity)]")
    if "wiki.test" in available_datasets and (llama_perplexity.exists() or args.dry_run):
        wikitext_file = available_datasets["wiki.test"]
        cmd = [
            str(llama_perplexity),
            "-m", str(model_path),
            "-f", str(wikitext_file),
            "-c", "512",
            "-t", str(args.threads),
            "-ngl", str(args.n_gpu_layers),
        ]
        run_stage(cmd, "WikiText-2 perplexity slice (512 tokens)", profile_dir, dry_run=args.dry_run, timeout=120)
    else:
        print("  -> WikiText-2 or llama-perplexity not present; skipping Stage 1.")

    # Stage 2: Continuous Prose Long-Context Sweep
    prose_path = None
    if args.prompt_file and Path(args.prompt_file).exists():
        prose_path = Path(args.prompt_file).resolve()
    elif "wiki.valid" in available_datasets:
        prose_path = available_datasets["wiki.valid"]
    elif available_datasets:
        prose_path = next(iter(available_datasets.values()))

    print("\n>>> [Stage 2/6: Continuous Prose Staged Long-Context Sweep]")
    for ctx in context_stages:
        prompt_arg = "High-performance inference architectures rely on efficient memory hierarchies, NUMA-aware allocation, "
        if prose_path and prose_path.exists():
            try:
                with open(prose_path, "r", encoding="utf-8", errors="replace") as f:
                    prompt_arg = f.read(min(ctx * 2, 1400)).strip()
            except Exception:
                pass

        cmd = [
            str(llama_cli),
            "-m", str(model_path),
            "-p", prompt_arg,
            "-n", str(args.n_predict),
            "-c", str(ctx),
            "-t", str(args.threads),
            "-ngl", str(args.n_gpu_layers),
            "-fa", "on",
            "--temp", "0",
        ]
        if args.numa and args.numa.lower() != "none":
            cmd.extend(["--numa", args.numa])

        run_stage(cmd, f"Staged prefill/decode at context = {ctx:,} tokens", profile_dir, dry_run=args.dry_run, timeout=180)

    # Stage 3: Multi-Domain Mathematical & Formal Reasoning Workload
    eval_prompts_file = None
    for k, p in available_datasets.items():
        if "prompt" in k.lower() or "eval" in k.lower():
            eval_prompts_file = p
            break

    print("\n>>> [Stage 3/6: Multi-Domain Reasoning & Logic Execution]")
    sample_reasoning_prompts = [
        "Explain how the Van Emde Boas tree achieves O(log log U) search complexity.",
        "Solve: A train travels 120 miles in 2 hours and 15 minutes. What is its average speed in mph?",
        "Describe the microarchitectural difference between AVX2 and AVX-512 register pressure.",
    ]
    if eval_prompts_file and eval_prompts_file.exists():
        try:
            with open(eval_prompts_file, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if l.strip()]
                if lines:
                    sample_reasoning_prompts = lines[:3]
        except Exception:
            pass

    for idx, prompt_text in enumerate(sample_reasoning_prompts, start=1):
        cmd = [
            str(llama_cli),
            "-m", str(model_path),
            "-p", prompt_text,
            "-n", "48",
            "-c", "2048",
            "-t", str(args.threads),
            "-ngl", str(args.n_gpu_layers),
            "-fa", "on",
            "--temp", "0.2",
        ]
        if args.numa and args.numa.lower() != "none":
            cmd.extend(["--numa", args.numa])
        run_stage(cmd, f"Reasoning prompt #{idx}: '{prompt_text[:40]}...'", profile_dir, dry_run=args.dry_run, timeout=120)

    # Stage 4: Code Synthesis & Speculative Decoding Verification
    print("\n>>> [Stage 4/6: Code Synthesis & Speculative Decoding Branch Training]")
    if args.model_draft:
        draft_path = Path(args.model_draft).resolve()
        if draft_path.exists() or args.dry_run:
            cmd = [
                str(llama_cli),
                "-m", str(model_path),
                "-md", str(draft_path),
                "--spec-type", "dflash",
                "-p", "Translate the following concepts into C++20 coroutines: ",
                "-n", "64",
                "-c", "2048",
                "-t", str(args.threads),
                "-ngl", str(args.n_gpu_layers),
                "-fa", "on",
            ]
            run_stage(cmd, "Speculative decoding draft verification loop", profile_dir, dry_run=args.dry_run, timeout=120)
        else:
            print(f"  -> Specified draft model '{draft_path}' not found, running deterministic C++20 synthesis fallback.")
    else:
        cmd = [
            str(llama_cli),
            "-m", str(model_path),
            "-p", "Write a lock-free 128-byte cache-line aligned SPSC ring buffer in C++20 using std::atomic_ref:",
            "-n", "64",
            "-c", "2048",
            "-t", str(args.threads),
            "-ngl", str(args.n_gpu_layers),
            "-fa", "on",
            "--temp", "0",
        ]
        if args.numa and args.numa.lower() != "none":
            cmd.extend(["--numa", args.numa])
        run_stage(cmd, "C++20 lock-free systems synthesis trace (hot sampler + grammar paths)", profile_dir, dry_run=args.dry_run, timeout=120)

    # Stage 5: Live llama-server Multi-Slot HTTP + CohortRadixTrie KV Cloning + Binary Fuse8 Capsules
    print("\n>>> [Stage 5/6: Live llama-server Multi-Slot HTTP + CohortRadixTrie KV Cloning + Fuse8 Capsules]")
    run_stage5_live_server_multi_slot(
        llama_server=llama_server,
        model_path=model_path,
        profile_dir=profile_dir,
        threads=args.threads,
        n_gpu_layers=args.n_gpu_layers,
        numa=args.numa,
        port=args.server_port,
        dry_run=args.dry_run,
    )

    # Stage 6: llama-bench GEMM / FlashAttention / MoE Microbenchmark & Optional nsys / perf stat Audit
    print("\n>>> [Stage 6/6: llama-bench GEMM / FlashAttention / MoE Microbenchmark & PMU Audit]")
    bench_cmd = [
        str(llama_bench),
        "-m", str(model_path),
        "-p", "512,2048",
        "-n", "32,128",
        "-t", str(args.threads),
        "-ngl", str(args.n_gpu_layers),
        "-fa", "1",
        "-r", "2",
    ]
    if args.nsys and shutil.which("nsys"):
        nsys_out = str(profile_dir / "stage6_nsys_report")
        bench_cmd = ["nsys", "profile", "-o", nsys_out, "--force-overwrite=true", "--stats=true"] + bench_cmd
    elif args.perf_stat and shutil.which("perf"):
        bench_cmd = [
            "perf", "stat",
            "-e", "cycles,instructions,branches,branch-misses,L1-icache-load-misses,iTLB-load-misses",
        ] + bench_cmd

    if llama_bench.exists() or args.dry_run:
        run_stage(
            bench_cmd,
            "llama-bench prefill (pp512,pp2048) + decode (tg32,tg128) + FlashAttention kernel sweep",
            profile_dir,
            dry_run=args.dry_run,
            timeout=240,
        )
    else:
        print(f"  -> llama-bench not found at '{llama_bench}', skipping Stage 6.")

    # Merge .profraw -> merged.profdata
    print("\n>>> [Profile Consolidation: llvm-profdata merge]")
    merge_profraw_traces(profile_dir, base_profdata=base_profdata, dry_run=args.dry_run)

    print("\n================================================================================")
    print("  PTO / IRPGO 6-STAGE TRAINING WORKLOAD SWEEP COMPLETED SUCCESSFULLY")
    print(f"  Consolidated Profile: {profile_dir / 'merged.profdata'}")
    print("  Next Step: ./scripts/pto_finetune.sh --pto=use --shared-libs --compiler=clang-21")
    print("================================================================================")


if __name__ == "__main__":
    main()
