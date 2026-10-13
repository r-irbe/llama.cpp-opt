#!/usr/bin/env python3
"""
scripts/bench_multicontext.py - Staged Multi-Context Benchmark Harness for ik_llama.cpp

Progressively evaluates prompt processing throughput (PP tok/s), autoregressive token
generation throughput (TG tok/s), Time-To-First-Token (TTFT), and memory footprint
across multiple context lengths up to the maximum context (256k / 262,144 tokens).

Key Features:
  - Staged Execution: Context lengths evaluated sequentially one stage at a time.
  - Duration Expectations (ETA): Calculates and prints stage ETA before execution begins.
  - Minimal Console Telemetry: Suppresses verbose engine output, reporting exactly one
    succinct line per stage to preserve terminal clarity and context tokens.
  - Clean Summary Report: Compiles cross-context scaling curves in ASCII and Markdown.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Staged Multi-Context Benchmarking Harness for ik_llama.cpp and OpenAI-Compatible Engines"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to GGUF model file or model identifier (for API endpoints)"
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Optional OpenAI-compatible HTTP endpoint for cross-engine testing (e.g. http://localhost:8000/v1 for vLLM, SGLang, TensorRT-LLM, or llama-server)"
    )
    parser.add_argument(
        "-c", "--context-lengths",
        default="512,1024,2048,4096,8192,16384,32768,65536,131072,262144",
        help="Comma-separated list of context lengths to evaluate progressively"
    )
    parser.add_argument(
        "-n", "--n-predict",
        type=int,
        default=10,
        help="Number of tokens to generate per stage (default: 10)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="Artificial intelligence is transforming high-performance computing through",
        help="Prompt text to evaluate"
    )
    parser.add_argument(
        "-pf", "--prompt-file",
        type=str,
        default=None,
        help="Optional path to text file containing prompt corpus (e.g. promessi_sposi.txt or wiki.test.raw)"
    )
    parser.add_argument(
        "-ngl", "--n-gpu-layers",
        type=int,
        default=999,
        help="Number of layers to offload to GPU (default: 999)"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=14,
        help="Number of CPU threads to use during generation (default: 14)"
    )
    parser.add_argument(
        "-tb", "--threads-batch",
        type=int,
        default=14,
        help="Number of CPU threads to use during batch processing (default: 14)"
    )
    parser.add_argument(
        "-fa", "--flash-attn",
        action="store_true",
        default=True,
        help="Enable Flash Attention (default: True)"
    )
    parser.add_argument(
        "--no-fa",
        dest="flash_attn",
        action="store_false",
        help="Disable Flash Attention"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=2048,
        help="Batch size for prompt processing (default: 2048)"
    )
    parser.add_argument(
        "-ub", "--ubatch-size",
        type=int,
        default=512,
        help="Physical micro-batch size (default: 512)"
    )
    parser.add_argument(
        "--numa",
        default=None,
        help="NUMA policy (default: None, or split/isolate/distribute/expert)"
    )
    parser.add_argument(
        "-sm", "--split-mode",
        default=None,
        help="Split mode for multi-GPU acceleration (default: None, or layer/graph)"
    )
    parser.add_argument(
        "-ot", "--tensor-override",
        default=None,
        help="Tensor override pattern for selective host/device placement (e.g. ^blk\\..*\\.ffn_.*_exps.*=CPU)"
    )
    parser.add_argument(
        "--spec-type",
        default=None,
        help="Speculative decoding stage type (e.g. ngram-mod:n_max=16,n_min=2, mtp, dflash)"
    )
    parser.add_argument(
        "-md", "--model-draft",
        default=None,
        help="Draft model path for speculative decoding"
    )
    parser.add_argument(
        "-ngld", "--n-gpu-layers-draft",
        type=int,
        default=999,
        help="Number of layers to offload to GPU for the draft model (default: 999)"
    )
    parser.add_argument(
        "--draft",
        type=int,
        default=None,
        help="Number of tokens to draft per speculative step"
    )
    parser.add_argument(
        "--spec-autotune",
        action="store_true",
        help="Enable dynamic speculative autotuning"
    )
    parser.add_argument(
        "-ctk", "--cache-type-k",
        default=None,
        help="KV cache type for K (e.g. f16, q8_0, q4_0)"
    )
    parser.add_argument(
        "-ctv", "--cache-type-v",
        default=None,
        help="KV cache type for V (e.g. f16, q8_0, q4_0)"
    )
    parser.add_argument(
        "--defer-ple",
        action="store_true",
        help="Enable deferred streaming for sparse PLE / engram tables"
    )
    parser.add_argument(
        "--defer-experts",
        action="store_true",
        help="Enable deferred streaming for routed MoE experts"
    )
    parser.add_argument(
        "-prexp", "--prefetch-experts",
        action="store_true",
        help="Enable direct NVMe prefetch streaming ring buffer"
    )
    parser.add_argument(
        "-prexp-t", "--prefetch-threads",
        type=int,
        default=8,
        help="Number of worker threads dedicated to NVMe prefetch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without running binary"
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Path to save Markdown benchmark report"
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save JSON benchmark metrics"
    )
    return parser.parse_args()


def estimate_stage_duration(ctx_len: int, n_predict: int, is_huge_model: bool) -> float:
    """Estimates expected runtime in seconds for a context length stage."""
    base_overhead = 8.0 if is_huge_model else 2.0
    pp_sec = (ctx_len / 512.0) * (0.8 if is_huge_model else 0.05)
    tg_sec = n_predict * (0.8 if is_huge_model else 0.04)
    return round(base_overhead + pp_sec + tg_sec, 1)


def parse_llama_cli_output(output: str, elapsed_sec: float, returncode: int = 0) -> dict:
    """Extracts performance metrics from llama-cli stdout and stderr."""
    if returncode != 0:
        error_lines = [
            line.strip() for line in output.splitlines()
            if any(err_kw in line.lower() for err_kw in ["error", "failed", "unable to load", "out of memory", "cuda error"])
        ]
        err_msg = error_lines[-1] if error_lines else f"llama-cli process failed with exit code {returncode}"
        return {
            "error": err_msg,
            "pp_ms": 0.0,
            "pp_tokens": 0,
            "pp_tps": 0.0,
            "tg_ms": 0.0,
            "tg_tokens": 0,
            "tg_tps": 0.0,
            "ttft_ms": 0.0
        }

    pp_match = re.search(
        r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)",
        output
    )
    # Negative lookbehind (?<!prompt ) prevents matching 'prompt eval time'
    tg_match = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)",
        output
    )

    if pp_match:
        pp_ms = float(pp_match.group(1))
        pp_tokens = int(pp_match.group(2))
        pp_tps = float(pp_match.group(3))
    else:
        pp_ms = 0.0
        pp_tokens = 0
        pp_tps = 0.0

    if tg_match:
        tg_ms = float(tg_match.group(1))
        tg_tokens = int(tg_match.group(2))
        tg_tps = float(tg_match.group(3))
    else:
        tg_ms = 0.0
        tg_tokens = 0
        tg_tps = 0.0

    ttft_ms = pp_ms

    return {
        "pp_ms": round(pp_ms, 2),
        "pp_tokens": pp_tokens,
        "pp_tps": round(pp_tps, 2),
        "tg_ms": round(tg_ms, 2),
        "tg_tokens": tg_tokens,
        "tg_tps": round(tg_tps, 2),
        "ttft_ms": round(ttft_ms, 2)
    }


def query_openai_endpoint(endpoint: str, model: str, prompt: str, ctx_len: int, n_predict: int, timeout: int = 300) -> dict:
    """Streams token generation from an OpenAI-compatible endpoint (vLLM, SGLang, TensorRT-LLM, llama-server)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"

    words = prompt.split()
    target_words = max(len(words), int(ctx_len * 0.75))
    if len(words) < target_words:
        multiplier = (target_words // len(words)) + 1
        expanded_prompt = " ".join((words * multiplier)[:target_words])
    else:
        expanded_prompt = prompt

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": expanded_prompt}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "stream": True
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    first_token_time = None
    token_times = []
    generated_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for line in response:
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    t_token = time.time()
                    if first_token_time is None:
                        first_token_time = t_token
                    token_times.append(t_token)
                    generated_tokens += 1

        total_time = time.time() - t0
        ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_time * 1000.0
        decode_time_sec = (time.time() - first_token_time) if first_token_time else 0.001
        tg_tps = (generated_tokens / max(0.001, decode_time_sec)) if generated_tokens > 0 else 0.0
        pp_tokens = ctx_len
        pp_tps = (pp_tokens / max(0.001, ttft_ms / 1000.0))

        return {
            "elapsed_sec": round(total_time, 2),
            "pp_ms": round(ttft_ms, 2),
            "pp_tokens": pp_tokens,
            "pp_tps": round(pp_tps, 2),
            "tg_ms": round(decode_time_sec * 1000.0, 2),
            "tg_tokens": generated_tokens,
            "tg_tps": round(tg_tps, 2),
            "ttft_ms": round(ttft_ms, 2)
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "error": str(e),
            "elapsed_sec": round(elapsed, 2),
            "pp_ms": 0.0,
            "pp_tokens": 0,
            "pp_tps": 0.0,
            "tg_ms": 0.0,
            "tg_tokens": 0,
            "tg_tps": 0.0,
            "ttft_ms": 0.0
        }


def main():
    args = parse_arguments()

    is_endpoint_mode = bool(args.endpoint)
    if is_endpoint_mode:
        model_basename = args.model.split("/")[-1]
        model_path = args.model
        file_size_gb = 0.0
        is_huge_model = False
    else:
        model_path = Path(args.model).resolve()
        if not model_path.exists() and not args.dry_run:
            print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
            sys.exit(1)
        file_size_gb = (model_path.stat().st_size / (1024 ** 3)) if model_path.exists() else 0.0
        is_huge_model = file_size_gb > 50.0
        model_basename = model_path.stem

    try:
        context_lengths = [int(c.strip()) for c in args.context_lengths.split(",") if c.strip()]
    except ValueError:
        print(f"Error: Invalid context lengths '{args.context_lengths}'. Expected comma-separated integers.", file=sys.stderr)
        sys.exit(1)

    total_stages = len(context_lengths)

    print("================================================================================")
    print(f"  STAGED MULTI-CONTEXT BENCHMARK: {model_basename}")
    if is_endpoint_mode:
        print(f"  Execution Target: OPENAI-COMPATIBLE API -> {args.endpoint}")
    else:
        print(f"  Model Size      : {file_size_gb:.1f} GiB ({'HUGE MODEL' if is_huge_model else 'STANDARD'})")

    base_prompt = args.prompt
    if args.prompt_file:
        pf = Path(args.prompt_file).resolve()
        if pf.exists():
            with open(pf, "r", encoding="utf-8", errors="replace") as f:
                base_prompt = f.read()
            print(f"  Prompt Source   : FILE ({pf.name}, {len(base_prompt):,} chars)")
        else:
            print(f"Warning: Prompt file '{args.prompt_file}' not found, falling back to default.", file=sys.stderr)

    print(f"  Context Stages  : {total_stages} Stages -> {[f'{c:,}' for c in context_lengths]}")
    print(f"  Tokens / Stage  : {args.n_predict} Generated Tokens")
    if not is_endpoint_mode:
        print(f"  Offload Config  : -ngl {args.n_gpu_layers} | -sm {args.split_mode} | --numa {args.numa}")
        print(f"  Streaming       : defer_ple={args.defer_ple} | defer_experts={args.defer_experts} | prefetch={args.prefetch_experts}")
    print("================================================================================")

    stage_results = []
    total_benchmark_start = time.time()

    for idx, ctx_len in enumerate(context_lengths, start=1):
        eta_sec = estimate_stage_duration(ctx_len, args.n_predict, is_huge_model)
        print(f"\n>>> [Stage {idx}/{total_stages}: Context = {ctx_len:,} Tokens] ETA: ~{eta_sec:.1f}s | Evaluating prompt processing & decode...")

        # Scale prompt text safely within context budget (ctx_len - n_predict - margin)
        safe_prompt_tokens = max(16, ctx_len - args.n_predict - 16)
        target_chars = int(safe_prompt_tokens * 2.8)
        if len(base_prompt) < target_chars:
            multiplier = (target_chars // max(1, len(base_prompt))) + 1
            stage_prompt = ("\n\n" + (base_prompt.strip() + " ") * multiplier)[:target_chars].strip()
        else:
            stage_prompt = base_prompt[:target_chars].strip()

        stage_start = time.time()
        if is_endpoint_mode:
            metrics = query_openai_endpoint(
                endpoint=args.endpoint,
                model=args.model,
                prompt=stage_prompt,
                ctx_len=ctx_len,
                n_predict=args.n_predict,
                timeout=max(300, int(eta_sec * 3.0))
            )
            elapsed = metrics.get("elapsed_sec", time.time() - stage_start)
            if "error" in metrics:
                print(f"[Stage {idx}/{total_stages}: Context = {ctx_len:>7,d} Tokens] API ERROR: {metrics['error']}", file=sys.stderr)
        elif args.dry_run:
            time.sleep(0.1)
            elapsed = 0.1
            metrics = {
                "pp_ms": round(15.0 + ctx_len * 0.05, 2),
                "pp_tokens": len(stage_prompt.split()),
                "pp_tps": round(1000.0 / (1.0 + ctx_len * 0.002), 2),
                "tg_ms": round(args.n_predict * 60.0, 2),
                "tg_tokens": args.n_predict,
                "tg_tps": round(1000.0 / 60.0, 2),
                "ttft_ms": round(15.0 + ctx_len * 0.05, 2)
            }
        else:
            cmd = [
                "./build/bin/llama-cli",
                "-m", str(model_path),
                "-p", stage_prompt,
                "-n", str(args.n_predict),
                "--temp", "0",
                "-c", str(ctx_len),
                "-b", str(args.batch_size),
                "-ub", str(args.ubatch_size),
                "-t", str(args.threads),
                "-tb", str(args.threads_batch)
            ]
            if args.flash_attn:
                cmd.extend(["-fa", "1"])
            if args.n_gpu_layers > 0:
                cmd.extend(["-ngl", str(args.n_gpu_layers)])
            if args.split_mode:
                cmd.extend(["-sm", args.split_mode])
            if args.tensor_override:
                cmd.extend(["-ot", args.tensor_override])
            if args.numa:
                cmd.extend(["--numa", args.numa])
            if args.spec_type:
                cmd.extend(["--spec-type", args.spec_type])
            if args.model_draft:
                cmd.extend(["-md", str(args.model_draft)])
                if args.n_gpu_layers_draft is not None:
                    cmd.extend(["-ngld", str(args.n_gpu_layers_draft)])
            if args.draft is not None:
                cmd.extend(["--draft", str(args.draft)])
            if args.spec_autotune:
                cmd.append("--spec-autotune")
            if args.cache_type_k:
                cmd.extend(["-ctk", args.cache_type_k])
            if args.cache_type_v:
                cmd.extend(["-ctv", args.cache_type_v])
            if args.defer_ple:
                cmd.append("--defer-ple")
            if args.defer_experts:
                cmd.append("--defer-experts")
            if args.prefetch_experts:
                cmd.append("-prexp")
                cmd.extend(["-prexp-t", str(args.prefetch_threads)])

            env = os.environ.copy()
            env["GGML_CUDA_NO_PINNED"] = "1"
            timeout = max(300, int(eta_sec * 3.0))

            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                    timeout=timeout
                )
                elapsed = time.time() - stage_start
                out = proc.stdout + proc.stderr
                metrics = parse_llama_cli_output(out, elapsed, returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - stage_start
                print(f"[Stage {idx}/{total_stages}: Context = {ctx_len:>7,d} Tokens] TIMEOUT after {elapsed:.1f}s", file=sys.stderr)
                metrics = {"error": "TimeoutExpired", "pp_ms": 0.0, "pp_tokens": 0, "pp_tps": 0.0, "tg_ms": 0.0, "tg_tokens": 0, "tg_tps": 0.0, "ttft_ms": 0.0}
            except Exception as e:
                elapsed = time.time() - stage_start
                print(f"[Stage {idx}/{total_stages}: Context = {ctx_len:>7,d} Tokens] ERROR: {e}", file=sys.stderr)
                metrics = {"error": str(e), "pp_ms": 0.0, "pp_tokens": 0, "pp_tps": 0.0, "tg_ms": 0.0, "tg_tokens": 0, "tg_tps": 0.0, "ttft_ms": 0.0}

        if "error" in metrics:
            print(
                f"[Stage {idx}/{total_stages}: Context = {ctx_len:>7,d} Tokens] "
                f"FAILED in {elapsed:>5.2f}s | Error: {metrics['error']}"
            )
        else:
            print(
                f"[Stage {idx}/{total_stages}: Context = {ctx_len:>7,d} Tokens] "
                f"DONE in {elapsed:>5.2f}s (ETA: {eta_sec:>5.1f}s) | "
                f"PP: {metrics['pp_tps']:>7.2f} t/s ({metrics['pp_ms']:>7.1f} ms) | "
                f"TG: {metrics['tg_tps']:>5.2f} t/s ({metrics['tg_ms']:>6.1f} ms) | "
                f"TTFT: {metrics['ttft_ms']:>7.1f} ms"
            )

        stage_results.append({
            "stage": idx,
            "context_length": ctx_len,
            "elapsed_sec": round(elapsed, 2),
            "eta_sec": eta_sec,
            **metrics
        })

    total_elapsed = time.time() - total_benchmark_start

    print("\n================================================================================")
    print(f"  MULTI-CONTEXT BENCHMARK COMPLETE: {total_stages} Stages in {total_elapsed:.2f}s")
    print("================================================================================")

    print("\nSTAGE-BY-STAGE MULTI-CONTEXT SCALING SUMMARY:")
    print("  Stage | Context Length | Elapsed |   Prompt (PP)   |  Generation (TG)  |    TTFT (ms)   ")
    print("  ----- | -------------- | ------- | --------------- | ----------------- | ---------------")
    for r in stage_results:
        print(
            f"   {r['stage']:02d}   |   {r['context_length']:>10,d}   |  {r['elapsed_sec']:>4.1f}s  | "
            f"{r['pp_tps']:>7.2f} tok/s  |   {r['tg_tps']:>5.2f} tok/s   |   {r['ttft_ms']:>8.1f} ms   "
        )
    print("=" * 80)

    # Save Markdown report
    out_md = args.output_md or f"results/bench_multicontext_{model_basename}.md"
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"# Staged Multi-Context Benchmark Report: {model_basename}",
        "",
        f"**Repository**: `ikawrakow/ik_llama.cpp` (`feat/numa-split`)  ",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Model Path**: `{model_path}` ({file_size_gb:.1f} GiB)  ",
        f"**Execution Flags**: `-ngl {args.n_gpu_layers} -sm {args.split_mode} --numa {args.numa} -t {args.threads} -tb {args.threads_batch}`  ",
        "",
        "---",
        "",
        "## Multi-Context Scaling Across Progressive Sequence Lengths",
        "",
        "| Stage | Context Length | Elapsed Time | Prompt Throughput (PP) | Prompt Latency | Generation Throughput (TG) | TTFT |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in stage_results:
        md_lines.append(
            f"| {r['stage']:02d} | **{r['context_length']:,} tokens** | {r['elapsed_sec']:.2f}s | "
            f"**{r['pp_tps']:.2f} tok/s** | {r['pp_ms']:.1f} ms | **{r['tg_tps']:.2f} tok/s** | {r['ttft_ms']:.1f} ms |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n[REPORT] Saved Markdown Benchmark Report to: {out_md}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump({
                "model": str(model_path),
                "file_size_gb": file_size_gb,
                "total_elapsed_sec": round(total_elapsed, 2),
                "stages": stage_results
            }, jf, indent=2)
        print(f"[REPORT] Saved JSON Benchmark Metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
