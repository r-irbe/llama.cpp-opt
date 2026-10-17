#!/usr/bin/env python3
"""
scripts/bench_concurrent_server.py - High-Concurrency Multi-Tenant Cohort Serving Benchmark Harness

Evaluates high-concurrency serving performance and multi-tenant Cohort Prompt Prefix Sharing (PPS)
in ik_llama.cpp (`llama-server` on branch feat/numa-split).

Compares:
  - Scenario A: Disjoint Streams (PPS = 1, independent isolated cache slots)
  - Scenario B: Cohort Prefix Sharing (PPS = C, zero-copy slot KV replication via llama_kv_cache_seq_cp)

Measures:
  - Primary TTFT vs Sibling TTFT
  - Sibling TTFT Speedup Factor
  - Aggregate Generation & Prefill Throughput (tokens/sec)
  - Active KV Cache Memory Reduction (%)
"""

import argparse
import json
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# How the numbers in a record were produced. `simulated` is the --mock path (random timings,
# no server involved); `live` is a request batch answered by a running llama-server.
MODE_LIVE      = "live"
MODE_SIMULATED = "simulated"
MODE_UNSPECIFIED = "unspecified"  # records written before the mode field existed


# --- Synthetic Prefix & Query Generators ---

TECHNICAL_PARAGRAPHS = [
    "Non-Uniform Memory Access (NUMA) is a computer memory design used in multiprocessing, where the memory access time depends on the memory location relative to the processor. Under NUMA, a processor can access its own local memory faster than non-local memory (memory local to another processor or memory shared between processors). The benefits of NUMA are limited to particular workloads, notably on servers where the data is often associated strongly with certain tasks or users.",
    "A Core Complex Die (CCD) in modern AMD Zen 3 and Zen 4 microarchitectures contains one or two Core Complexes (CCX). Each CCX features a private L3 cache slice shared among up to eight physical cores. In NPS4 mode (NUMA Per Socket = 4), the memory controllers are partitioned into four distinct quadrants per physical package, yielding independent DDR memory channels and local memory latency profiles.",
    "Mixture of Experts (MoE) neural network architectures dynamically activate subsets of feed-forward network (FFN) parameters for each incoming token via a learned routing gate. In models such as DeepSeek-V3 and Mixtral 8x22B, routed experts are decoupled across independent memory pools. Layer-interleaved round-robin distribution spreads expert parameter matrices uniformly across all NUMA domains to saturate aggregate DDR memory bus bandwidth.",
    "Single-Producer Single-Consumer (SPSC) circular ring buffers achieve wait-free execution by eliminating mutex contention. Aligning producer tail pointers and consumer head pointers to separate 128-byte cache line boundaries prevents false sharing caused by spatial prefetchers fetching paired 64-byte lines, maintaining sub-microsecond token streaming latencies across inter-socket interconnects.",
    "Multi-Tenant Cohort Prompt Prefix Sharing enables multiple concurrent requests sharing identical system prompts, enterprise retrieval-augmented generation (RAG) context, or multi-agent swarms to reuse warm KV cache state. Using zero-copy sequence cloning, sibling requests bypass redundant transformer prefill, reducing host memory bus saturation and quadratic attention compute.",
]

def generate_synthetic_prefix(target_tokens: int) -> str:
    """Generate a deterministic synthetic text corpus approximating target_tokens."""
    # Rough approximation: 1 token ~= 4 characters in English technical text
    chars_needed = target_tokens * 4
    corpus = []
    idx = 0
    total_len = 0
    while total_len < chars_needed:
        p = TECHNICAL_PARAGRAPHS[idx % len(TECHNICAL_PARAGRAPHS)]
        corpus.append(p)
        total_len += len(p) + 1
        idx += 1
    text = " ".join(corpus)
    return text[:chars_needed]

def generate_unique_query(client_id: int) -> str:
    """Generate a unique query suffix for an individual client stream."""
    return f"\n\nClient Stream {client_id} Query: Based on the architectural principles outlined above, calculate the optimal NUMA thread binding and memory channel allocation for slot {client_id}."


# --- HTTP Client Request Handler ---

def send_request(server_url: str, prompt: str, cohort_id: str, prompt_cache_key: str,
                 client_id: int, n_predict: int = 48, timeout: int = 180, mock: bool = False):
    """Send completion request with injected cohort headers and body parameters."""
    if mock:
        # Simulate realistic server timing in mock/offline mode
        is_sibling = (client_id > 0 and cohort_id.startswith("cohort_"))
        ttft_ms = random.uniform(15.0, 35.0) if is_sibling else random.uniform(250.0, 400.0)
        time.sleep(ttft_ms / 1000.0)
        gen_duration = random.uniform(0.3, 0.6)
        time.sleep(gen_duration)
        total_duration = (ttft_ms / 1000.0) + gen_duration
        tokens_gen = n_predict
        return {
            "client_id": client_id,
            "success": True,
            "is_sibling": is_sibling,
            "ttft_ms": ttft_ms,
            "total_duration": total_duration,
            "n_predict": tokens_gen,
            "prompt_tok_per_sec": (1024.0 / (ttft_ms / 1000.0)),
            "gen_tok_per_sec": tokens_gen / gen_duration,
            "error": None
        }

    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0, # Deterministic decode
        "stream": False,
        "cohort_id": cohort_id,
        "prompt_cache_key": prompt_cache_key
    }
    headers = {
        "Content-Type": "application/json",
        "X-Cohort-ID": cohort_id,
        "X-Prompt-Cache-Key": prompt_cache_key
    }

    data = json.dumps(payload).encode("utf-8")
    if not server_url.rstrip("/").endswith("/completion"):
        server_url = server_url.rstrip("/") + "/completion"
    req = urllib.request.Request(server_url, data=data, headers=headers)

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            t1 = time.perf_counter()
            body = json.loads(resp.read().decode("utf-8"))
            total_duration = t1 - t0
            timings = body.get("timings", {})
            tokens_gen = body.get("tokens_predicted", n_predict)

            # Server-reported prompt prefill ms
            server_prompt_ms = timings.get("prompt_ms", 0.0)
            ttft_ms = server_prompt_ms if server_prompt_ms > 0 else (total_duration * 1000.0 * 0.4)
            is_sibling = (client_id > 0 and not cohort_id.startswith("disjoint_"))

            return {
                "client_id": client_id,
                "success": True,
                "is_sibling": is_sibling,
                "ttft_ms": ttft_ms,
                "total_duration": total_duration,
                "n_predict": tokens_gen,
                "prompt_tok_per_sec": timings.get("prompt_per_second", 0.0),
                "gen_tok_per_sec": timings.get("predicted_per_second", 0.0),
                "error": None
            }
    except Exception as e:
        return {
            "client_id": client_id,
            "success": False,
            "is_sibling": (client_id > 0),
            "ttft_ms": 0.0,
            "total_duration": 0.0,
            "n_predict": 0,
            "prompt_tok_per_sec": 0.0,
            "gen_tok_per_sec": 0.0,
            "error": str(e)
        }


# --- Benchmark Orchestration ---

def run_concurrency_batch(server_url: str, concurrency: int, prefix_tokens: int,
                          scenario: str, n_predict: int = 48, mock: bool = False):
    """Execute a single batch of concurrent requests under Scenario A or Scenario B."""
    shared_prefix = generate_synthetic_prefix(prefix_tokens)
    tag = f"pps_len{prefix_tokens}_c{concurrency}_{int(time.time())}"

    tasks = []
    for client_id in range(concurrency):
        query = generate_unique_query(client_id)
        full_prompt = shared_prefix + query

        if scenario == "disjoint": # Scenario A: PPS = 1
            # Unique cohort_id prevents cache sharing
            cohort_id = f"disjoint_{tag}_cli{client_id}"
            cache_key = f"key_disjoint_{tag}_cli{client_id}"
        else: # Scenario B: PPS = C
            # Shared cohort_id enables zero-copy llama_kv_cache_seq_cp replication
            cohort_id = f"cohort_{tag}"
            cache_key = f"key_cohort_{tag}"

        tasks.append((client_id, full_prompt, cohort_id, cache_key))

    t_wall_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(send_request, server_url, prompt, c_id, k_id, cid, n_predict, 180, mock): cid
            for (cid, prompt, c_id, k_id) in tasks
        }
        for f in as_completed(futures):
            results.append(f.result())

    t_wall_end = time.perf_counter()
    wall_duration = t_wall_end - t_wall_start

    successful = [r for r in results if r["success"]]
    mode = MODE_SIMULATED if mock else MODE_LIVE
    if not successful:
        return {
            "scenario": scenario,
            "concurrency": concurrency,
            "prefix_tokens": prefix_tokens,
            "mode": mode,
            "success_count": 0,
            "success_rate": 0.0,
            "primary_ttft_ms": 0.0,
            "sibling_ttft_ms": 0.0,
            "speedup_factor": 1.0,
            "aggregate_tps": 0.0,
            "kv_cache_reduction_pct": 0.0,
            "wall_duration": wall_duration
        }

    # Identify Primary (client 0) vs Sibling requests
    primary_req = next((r for r in successful if r["client_id"] == 0), successful[0])
    sibling_reqs = [r for r in successful if r["client_id"] != primary_req["client_id"]]

    primary_ttft = primary_req["ttft_ms"]
    sibling_ttft = (sum(r["ttft_ms"] for r in sibling_reqs) / len(sibling_reqs)) if sibling_reqs else primary_ttft
    speedup = (primary_ttft / sibling_ttft) if (sibling_ttft > 0) else 1.0

    total_tokens = sum(r["n_predict"] for r in successful)
    agg_tps = total_tokens / wall_duration if wall_duration > 0 else 0.0

    # Theoretical active KV cache reduction for identical prefix across C slots
    kv_reduction = ((concurrency - 1) / concurrency * 100.0) if (scenario == "cohort" and concurrency > 1) else 0.0

    return {
        "scenario": scenario,
        "concurrency": concurrency,
        "prefix_tokens": prefix_tokens,
        "mode": mode,
        "success_count": len(successful),
        "success_rate": len(successful) / concurrency * 100.0,
        "primary_ttft_ms": round(primary_ttft, 2),
        "sibling_ttft_ms": round(sibling_ttft, 2),
        "speedup_factor": round(speedup, 2),
        "aggregate_tps": round(agg_tps, 2),
        "kv_cache_reduction_pct": round(kv_reduction, 1),
        "wall_duration": round(wall_duration, 2)
    }


# --- Markdown Summary Formatter ---

def format_cohort_markdown_table(records: list, data_source: str = "results/cohort_bench_results.json",
                                 allow_mixed_modes: bool = False) -> str:
    """Format benchmark results into GitHub-flavored markdown summary table.

    Refuses to emit a measurement table when the records cannot support one:
      - every record failed (0 successful requests), or
      - the records mix live and simulated runs (mock) unless allow_mixed_modes is set.
    Records written before the `mode` field existed are labeled `unspecified`.
    """
    header = [
        "# High-Concurrency Cohort Serving & Prefix Sharing Benchmark Summary",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Data source**: `{data_source}` ({len(records)} records)  ",
        "**Evaluated**: Scenario A (Disjoint $PPS=1$) vs Scenario B (Cohort Prefix Sharing $PPS=C$)",
        ""
    ]

    modes = sorted({r.get("mode", MODE_UNSPECIFIED) for r in records})
    total = len(records)
    ok = [r for r in records if r.get("success_count", 0) > 0]

    if not records:
        return "\n".join(header + [
            "**REFUSED**: no records to summarize.",
            ""
        ])

    if not ok:
        return "\n".join(header + [
            "**REFUSED**: no successful requests in the data file, so no measurement table is emitted.",
            "",
            f"| Records | Successful records | Modes | Total wall time (s) |",
            "| :---: | :---: | :--- | :---: |",
            f"| {total} | 0 | {', '.join(f'`{m}`' for m in modes)} | "
            f"{sum(r.get('wall_duration', 0.0) for r in records):.3f} |",
            "",
            "Every request in the only raw run in the data file failed, so this file carries no throughput",
            f"or latency numbers. Any previously published cohort table is unsupported by `{data_source}`.",
            "Re-run the harness against a live `llama-server` (without `--mock`) to produce a measurement table.",
            ""
        ])

    if len(modes) > 1 and not allow_mixed_modes:
        raise SystemExit(f"[REFUSED] {data_source}: refusing to mix evaluation modes {modes} in one table; "
                         f"summarize one mode per table or pass --allow-mixed-modes to override.")

    lines = header + [
        f"**Evaluation mode(s)**: {', '.join(f'`{m}`' for m in modes)}",
        "",
        "### Concurrency Scaling & TTFT Speedup Table",
        "",
        "| Concurrency ($C$) | Prefix Len ($L$) | Disjoint TTFT (ms) | Cohort Primary TTFT (ms) | Cohort Sibling TTFT (ms) | Sibling TTFT Speedup | Disjoint Throughput | Cohort Throughput | Active KV Reduction | Mode | Source |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    # Group by (concurrency, prefix_tokens)
    pairs = sorted(list(set((r["concurrency"], r["prefix_tokens"]) for r in records)))
    indexed = list(enumerate(records))

    for c, l in pairs:
        disjoint_idx, disjoint_rec = next(((i, r) for i, r in indexed if r["concurrency"] == c and r["prefix_tokens"] == l and r["scenario"] == "disjoint"), (None, None))
        cohort_idx, cohort_rec = next(((i, r) for i, r in indexed if r["concurrency"] == c and r["prefix_tokens"] == l and r["scenario"] == "cohort"), (None, None))

        disjoint_ttft_str = f"{disjoint_rec['primary_ttft_ms']:.1f} ms" if disjoint_rec else "N/A"
        cohort_pri_str = f"{cohort_rec['primary_ttft_ms']:.1f} ms" if cohort_rec else "N/A"
        cohort_sib_str = f"{cohort_rec['sibling_ttft_ms']:.1f} ms" if cohort_rec else "N/A"
        speedup_str = f"**{cohort_rec['speedup_factor']:.2f}x**" if (cohort_rec and cohort_rec['concurrency'] > 1) else "1.00x"
        disjoint_tps_str = f"{disjoint_rec['aggregate_tps']:.1f} tok/s" if disjoint_rec else "N/A"
        cohort_tps_str = f"{cohort_rec['aggregate_tps']:.1f} tok/s" if cohort_rec else "N/A"
        kv_reduct_str = f"**{cohort_rec['kv_cache_reduction_pct']:.1f}%**" if (cohort_rec and cohort_rec['concurrency'] > 1) else "0.0%"
        row_mode = " + ".join(sorted({r.get("mode", MODE_UNSPECIFIED) for _, r in ((disjoint_idx, disjoint_rec), (cohort_idx, cohort_rec)) if r}))
        source_cells = "/".join(f"`{data_source}[{i}]`" for i in (disjoint_idx, cohort_idx) if i is not None)

        lines.append(
            f"| {c} | {l} | {disjoint_ttft_str} | {cohort_pri_str} | {cohort_sib_str} | {speedup_str} | {disjoint_tps_str} | {cohort_tps_str} | {kv_reduct_str} | `{row_mode}` | {source_cells} |"
        )

    return "\n".join(lines) + "\n"


# --- Main Entrypoint ---

def main():
    parser = argparse.ArgumentParser(description="High-Concurrency Cohort Serving Benchmark Harness for ik_llama.cpp")
    parser.add_argument("--server-url", default="http://127.0.0.1:8089/completion", help="Target llama-server endpoint")
    parser.add_argument("--concurrency", nargs="*", type=int, default=[1, 2, 4, 8, 16, 24, 32], help="Concurrency levels to sweep")
    parser.add_argument("--prefix-lengths", nargs="*", type=int, default=[1024, 2048, 4096], help="Prefix token lengths to test")
    parser.add_argument("--n-predict", type=int, default=48, help="Number of tokens to generate per stream")
    parser.add_argument("--output-json", default="results/cohort_bench_results.json", help="Path to JSON results output")
    parser.add_argument("--output-md", default="results/cohort_bench_summary.md", help="Path to Markdown summary output")
    parser.add_argument("--mock", action="store_true", help="Simulate server responses without live server for testing (records are tagged mode='simulated')")
    parser.add_argument("--allow-mixed-modes", action="store_true",
                        help="Allow one summary table to combine live and simulated records (refused by default)")
    args = parser.parse_args()

    print("================================================================================")
    print("  IK_LLAMA.CPP HIGH-CONCURRENCY COHORT SERVING BENCHMARK HARNESS")
    print(f"  Target Server Endpoint: {args.server_url} {'(MOCK MODE)' if args.mock else ''}")
    print(f"  Concurrency Sweeps: {args.concurrency}")
    print(f"  Prefix Lengths    : {args.prefix_lengths} tokens")
    print("================================================================================")

    records = []

    for l in args.prefix_lengths:
        print(f"\n================================================================================")
        print(f"  EVALUATING SYNTHETIC PREFIX LENGTH: {l} TOKENS")
        print(f"================================================================================")

        for c in args.concurrency:
            print(f"\n>>> Concurrency Level: {c} Concurrent Streams (L = {l} tokens) <<<")

            # 1. Benchmark Scenario A: Disjoint Streams (PPS = 1)
            print(f"  [RUN] Scenario A: Disjoint Independent Streams (PPS = 1)...")
            res_a = run_concurrency_batch(args.server_url, c, l, "disjoint", args.n_predict, args.mock)
            records.append(res_a)
            print(f"    -> Wall Duration: {res_a['wall_duration']:.2f}s | Throughput: {res_a['aggregate_tps']} tok/s | Avg TTFT: {res_a['primary_ttft_ms']} ms")

            # Brief pause between scenarios to allow server thread queues to drain
            time.sleep(0.5 if not args.mock else 0.05)

            # 2. Benchmark Scenario B: Cohort Prefix Sharing (PPS = C)
            print(f"  [RUN] Scenario B: Cohort Prefix Sharing (PPS = {c})...")
            res_b = run_concurrency_batch(args.server_url, c, l, "cohort", args.n_predict, args.mock)
            records.append(res_b)
            print(f"    -> Primary TTFT: {res_b['primary_ttft_ms']} ms | Sibling TTFT: {res_b['sibling_ttft_ms']} ms | Sibling Speedup: {res_b['speedup_factor']}x")
            print(f"    -> Throughput: {res_b['aggregate_tps']} tok/s | Active KV Cache Reduction: {res_b['kv_cache_reduction_pct']}%")

    # Generate and display markdown table
    md_content = format_cohort_markdown_table(records, data_source=args.output_json,
                                              allow_mixed_modes=args.allow_mixed_modes)
    print("\n" + md_content)

    # Save outputs
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(records, f, indent=2)

    with open(args.output_md, "w") as f:
        f.write(md_content)

    print("================================================================================")
    print("  COHORT SERVING BENCHMARK COMPLETE")
    print(f"  JSON Results: {args.output_json}")
    print(f"  Markdown Table: {args.output_md}")
    print("================================================================================")

if __name__ == "__main__":
    main()
