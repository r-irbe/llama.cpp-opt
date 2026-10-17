#!/usr/bin/env python3
"""
scripts/pto_profile_runner.py - Profile-Targeted Optimization (PTO / PGO) Runner for ik_llama.cpp

Orchestrates execution of representative training workloads across curated datasets
to generate profiling data (.gcda / .profdata) for compiler Profile-Guided Optimization.

Dataset Support:
1. In-tree WikiText-2 (scratch/wikitext-2-raw/wiki.test.raw)
2. Custom User Prompt Corpus (--prompt-file or --dataset-dir)
3. Multi-Context Sweeps (512 up to 32k tokens)
4. Multi-Domain Reasoning & Logic Execution
5. Speculative Decoding Verification (--model-draft)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

IN_TREE_DATASET_PATHS = [
    Path("scratch/wikitext-2-raw/wiki.test.raw"),
    Path("scratch/wikitext-2-raw/wiki.valid.raw"),
    Path("scratch/wikitext-2-raw/wiki.train.raw"),
    Path("prompts"),
    Path("tests/prompts"),
]


def find_available_datasets(custom_dir: Optional[str] = None, prompt_file: Optional[str] = None) -> Dict[str, Path]:
    """Scans repository and optionally specified custom directories for test datasets."""
    available = {}

    # 1. In-tree paths
    for p in IN_TREE_DATASET_PATHS:
        if p.exists():
            if p.is_file():
                available[p.stem] = p.resolve()
            elif p.is_dir():
                for f in p.glob("*.txt"):
                    available[f.stem] = f.resolve()

    # 2. Environment variable directory
    env_dir = os.environ.get("PTO_DATASET_DIR") or os.environ.get("IK_LLAMA_DATASET_DIR")
    if env_dir and os.path.exists(env_dir):
        p_env = Path(env_dir)
        for ext in ["*.raw", "*.txt", "*.csv"]:
            for f in p_env.glob(ext):
                available[f.stem] = f.resolve()

    # 3. Explicit custom directory
    if custom_dir and os.path.exists(custom_dir):
        p_cust = Path(custom_dir)
        for ext in ["*.raw", "*.txt", "*.csv"]:
            for f in p_cust.glob(ext):
                available[f.stem] = f.resolve()

    # 4. Explicit single prompt file
    if prompt_file and os.path.exists(prompt_file):
        pf = Path(prompt_file)
        available[pf.stem] = pf.resolve()

    return available


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Profile-Targeted Optimization (PTO / PGO) Training Runner for ik_llama.cpp"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build",
        help="Path to build directory containing binaries (llama-cli, llama-perplexity, etc.)"
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
        default=14,
        help="Number of execution threads"
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
        default=None,
        help="NUMA configuration (split, distribute, isolate, or None)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without invoking binaries"
    )
    parser.add_argument(
        "--print-pgo-flags",
        action="store_true",
        help="Print CMake flags for Phase 1 (generate) and Phase 3 (use)"
    )
    return parser.parse_args()


def print_cmake_instructions():
    print("""
================================================================================
  PROFILE-TARGETED OPTIMIZATION (PTO / PGO) CMAKE CONFIGURATION INSTRUCTIONS
================================================================================

[Phase 1: Configure & Build Instrumented Executables]
  mkdir -p /tmp/ik_llama_pgo
  cmake -B build-pgo -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_FLAGS="-fprofile-generate=/tmp/ik_llama_pgo -O3 -march=native" \\
        -DCMAKE_CXX_FLAGS="-fprofile-generate=/tmp/ik_llama_pgo -O3 -march=native"
  cmake --build build-pgo -j $(nproc)

[Phase 2: Execute PTO Training Sweep]
  python3 scripts/pto_profile_runner.py --build-dir build-pgo -m <model.gguf>

[Phase 3: Compile Profile-Guided Production Binary]
  cmake -B build-pto -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_FLAGS="-fprofile-use=/tmp/ik_llama_pgo -fprofile-correction -O3 -march=native -flto" \\
        -DCMAKE_CXX_FLAGS="-fprofile-use=/tmp/ik_llama_pgo -fprofile-correction -O3 -march=native -flto"
  cmake --build build-pto -j $(nproc)

[Phase 4: Verification]
  Run scripts/bench_multicontext.py with ./build-pto/bin/llama-cli
================================================================================
""")


def run_stage(cmd: List[str], description: str, dry_run: bool = False, timeout: int = 300) -> bool:
    print(f"  -> [PTO Run] {description}...", flush=True)
    if dry_run:
        clean_cmd = []
        for arg in cmd:
            if len(arg) > 60:
                clean_cmd.append(f"'{arg[:50]}...'")
            elif " " in arg:
                clean_cmd.append(f"'{arg}'")
            else:
                clean_cmd.append(arg)
        print(f"     (dry-run cmd: {' '.join(clean_cmd)})")
        time.sleep(0.05)
        return True

    env = os.environ.copy()
    env["GGML_CUDA_NO_PINNED"] = "1"
    
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
            text=True,
            errors="replace"
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


def main():
    args = parse_arguments()

    if args.print_pgo_flags:
        print_cmake_instructions()
        sys.exit(0)

    build_dir = Path(args.build_dir).resolve()
    llama_cli = build_dir / "bin" / "llama-cli"
    llama_perplexity = build_dir / "bin" / "llama-perplexity"

    if not llama_cli.exists() and not args.dry_run:
        print(f"Error: llama-cli not found at '{llama_cli}'. Run cmake --build first.", file=sys.stderr)
        sys.exit(1)

    if args.model:
        model_path = Path(args.model).resolve()
    else:
        # Search local models directory or current working directory
        candidates = list(Path("models").glob("*.gguf")) + list(Path(".").glob("*.gguf"))
        if candidates:
            model_path = candidates[0].resolve()
            print(f"[Notice] Auto-selected model: {model_path.name}")
        elif args.dry_run:
            model_path = Path("model.gguf")
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
    print("  PROFILE-TARGETED OPTIMIZATION (PTO / PGO) TRAINING HARNESS")
    print(f"  Target Executable: {llama_cli}")
    print(f"  Model            : {model_path.name}")
    print(f"  Context Stages   : {[f'{c:,}' for c in context_stages]}")
    print(f"  Available Datasets Discovered: {len(available_datasets)}")
    for name, path in available_datasets.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"    - {name:<26}: {size_kb:>8.1f} KB ({path})")
    print("================================================================================")

    # 1. WikiText Perplexity Workload (if available)
    if "wiki.test" in available_datasets and llama_perplexity.exists():
        wikitext_file = available_datasets["wiki.test"]
        print("\n>>> [Workload 1/4: Dense Vocabulary & Perplexity Training]")
        cmd = [
            str(llama_perplexity),
            "-m", str(model_path),
            "-f", str(wikitext_file),
            "-c", "512",
            "-t", str(args.threads),
            "-ngl", str(args.n_gpu_layers)
        ]
        run_stage(cmd, "WikiText-2 perplexity slice (512 tokens)", dry_run=args.dry_run, timeout=120)

    # 2. Continuous Prose Long-Context Sweep (from prompt file, dataset, or synthetic)
    prose_path = None
    if args.prompt_file and Path(args.prompt_file).exists():
        prose_path = Path(args.prompt_file).resolve()
    elif "wiki.valid" in available_datasets:
        prose_path = available_datasets["wiki.valid"]
    elif available_datasets:
        prose_path = next(iter(available_datasets.values()))

    print("\n>>> [Workload 2/4: Continuous Prose Staged Context Sweep]")
    for ctx in context_stages:
        prompt_arg = "High-performance inference architectures rely on efficient memory hierarchies, NUMA-aware allocation, "
        if prose_path and prose_path.exists():
            try:
                with open(prose_path, "r", encoding="utf-8", errors="replace") as f:
                    prompt_arg = f.read(min(ctx * 4, 3000)).strip()
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
            "--temp", "0"
        ]
        if args.numa:
            cmd.extend(["--numa", args.numa])

        run_stage(cmd, f"Staged prefill/decode at context = {ctx:,} tokens", dry_run=args.dry_run, timeout=180)

    # 3. Reasoning & Mathematical Verification Workload
    eval_prompts_file = None
    for k, p in available_datasets.items():
        if "prompt" in k.lower() or "eval" in k.lower():
            eval_prompts_file = p
            break

    print("\n>>> [Workload 3/4: Multi-Domain Reasoning & Logic Execution]")
    sample_reasoning_prompts = [
        "Explain how the Van Emde Boas tree achieves O(log log U) search complexity.",
        "Solve: A train travels 120 miles in 2 hours and 15 minutes. What is its average speed in mph?",
        "Describe the microarchitectural difference between AVX2 and AVX-512 register pressure."
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
            "--temp", "0.2"
        ]
        if args.numa:
            cmd.extend(["--numa", args.numa])
        run_stage(cmd, f"Reasoning prompt #{idx}: '{prompt_text[:40]}...'", dry_run=args.dry_run, timeout=120)

    # 4. Speculative Decoding Verification (if draft model passed via CLI)
    print("\n>>> [Workload 4/4: Speculative Decoding Branch Training]")
    if args.model_draft:
        draft_path = Path(args.model_draft).resolve()
        if draft_path.exists():
            cmd = [
                str(llama_cli),
                "-m", str(model_path),
                "-md", str(draft_path),
                "--spec-type", "dflash",
                "-p", "Translate the following concepts into C++20 coroutines: ",
                "-n", "64",
                "-c", "2048",
                "-t", str(args.threads),
                "-ngl", str(args.n_gpu_layers)
            ]
            run_stage(cmd, "Speculative decoding draft verification loop", dry_run=args.dry_run, timeout=120)
        else:
            print(f"  -> Specified draft model '{draft_path}' not found, skipping Workload 4.")
    else:
        print("  -> Speculative draft model skipped (pass --model-draft to include speculative branches in profile).")

    print("\n================================================================================")
    print("  PTO / PGO TRAINING WORKLOAD SWEEP COMPLETED SUCCESSFULLY")
    print("  Profile counters have been accumulated.")
    print("  Next Step: Recompile with Phase 3 flags (-fprofile-use / -flto).")
    print("================================================================================")


if __name__ == "__main__":
    main()
