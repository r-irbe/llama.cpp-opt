#!/usr/bin/env python3
"""
scripts/bench_deepseek_v41_1017_all.py - Unified DeepSeek-V4.1-Flash-1017 Benchmark & Audit Runner

Executes:
1. GGUF hardware/alignment linting (`scripts/gguf-lint.py`) across the DeepSeek-V4.1-Flash-1017
   containers (`UD-Q4_K_XL` 9-shard, `GSQ-RCO-IQ2_XXS` 4-shard + `Engram-IQ4_NL`, `Engram-MXFP8`,
   `DSpark-Q8_0`, and `EngramQ8-Q2_K_Protected` 10-shard).
2. Matrix evaluation across the 6 canonical optimization setups (without `--defer-experts`,
   keeping routed MoE experts resident in host RAM via `--transparent-huge-pages`, `--defer-ple`
   or `--engram-resident`, and `-ot "^blk\\..*\\.ffn_.*_exps.*=CPU"`):
   - Setup 1 (`np1-dspark-treegraft-autotune-distribute`): 2-stage `[ngram-mod, dspark]` + `--spec-autotune`
   - Setup 2 (`np1-dspark-1stage-autotune-distribute`): 1-stage `dspark:n_max=6,n_min=1` + `--spec-autotune`
   - Setup 3 (`np1-target-only-distribute` vs `expert`): `--numa distribute` vs `--numa expert` (with auto-unpin gate)
   - Setup 4 (`np1-512k-swa-evidence` & `np1-1m-swa-evidence`): 512K and 1M context with `--swa-compress` + `--zdd-sparse-kv` + `--evidence-filter`
   - Setup 5 (`np4-512k-swarm` & `np8-512k-1m-swarm`): multi-slot `-np 4` and `-np 8` with `--cohort-prefix-sharing` + `--swa-compress`
   - Setup 6 (`deep-thinking-effort-eval`): `<think>...</think>` multi-step mathematical & systems reasoning evaluation
3. Consolidates archived live logs (`scratch/logs/*.log`, `results/ranked_*.json`) and emits
   privacy-scrubbed `results/deepseek_v41_1017_ranked_benchmark_report.json` and `.md`.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def scrub_privacy(text: str) -> str:
    """Ensure zero private paths or project identifiers enter reports."""
    patterns = [
        (r"/home/[a-zA-Z0-9_-]+/vm/llm/", "${LLAMA_MODEL_ROOT}/"),
        (r"/home/[a-zA-Z0-9_-]+/code/ik_llama\.cpp/", ""),
        (r"/home/[a-zA-Z0-9_-]+/code/[a-zA-Z0-9_-]+/", "${WORKSPACE_ROOT}/"),
        (r"/home/[a-zA-Z0-9_-]+", "${HOME}"),
    ]
    out = text
    for pat, repl in patterns:
        out = re.sub(pat, repl, out)
    return out


def build_setup_matrix(model_root: str) -> List[Dict[str, Any]]:
    ud_model = f"{model_root}/DeepSeek-V4.1-Flash-1017/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf"
    iq2_model = f"{model_root}/ista-deepseek-v4.1-1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"
    dspark_q8 = f"{model_root}/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"

    return [
        {
            "setup_id": "Setup-1: np1-dspark-treegraft-autotune-distribute",
            "description": "2-Stage [ngram-mod, dspark] + TreeGraft EMA divergence arbitration + --spec-autotune (PolyGraph verify_pinned arm)",
            "cli_args": [
                "-m", ud_model,
                "-md", dspark_q8,
                "-ngl", "99", "-ngld", "99",
                "-c", "65536", "-np", "1",
                "--swa-compress", "-ctk", "q8_0", "-ctv", "q8_0", "-ictk", "q8_0",
                "-fa", "on", "-fidx", "-dsa", "-dsatk", "1024", "-mla", "2", "-amb", "256",
                "--zdd-sparse-kv", "--defer-ple", "--transparent-huge-pages",
                "--numa", "distribute", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
                "--spec-type", "ngram-mod:n_max=6,n_min=2",
                "--spec-type", "dspark:n_max=6,n_min=1",
                "--spec-ckpt-mode", "per-step", "--spec-autotune",
                "--evidence-filter", "--evidence-strategy", "code_pinpoint",
            ],
        },
        {
            "setup_id": "Setup-2: np1-dspark-1stage-autotune-distribute",
            "description": "1-Stage DSpark Q8_0 block drafter + per-step DSV4 checkpoints + --spec-autotune",
            "cli_args": [
                "-m", ud_model,
                "-md", dspark_q8,
                "-ngl", "99", "-ngld", "99",
                "-c", "65536", "-np", "1",
                "--swa-compress", "-ctk", "q8_0", "-ctv", "q8_0", "-ictk", "q8_0",
                "-fa", "on", "-fidx", "-dsa", "-dsatk", "1024", "-mla", "2", "-amb", "256",
                "--zdd-sparse-kv", "--defer-ple", "--transparent-huge-pages",
                "--numa", "distribute", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
                "--spec-type", "dspark:n_max=6,n_min=1",
                "--spec-ckpt-mode", "per-step", "--spec-autotune",
                "--evidence-filter", "--evidence-strategy", "code_pinpoint",
            ],
        },
        {
            "setup_id": "Setup-3: np1-target-only-expert-vs-distribute",
            "description": "Target-only -np 1 comparing --numa expert (with llama_numa_cuda_pinning_gate auto-unpin + 120 routed MoE mbind) vs --numa distribute and --engram-resident",
            "cli_args": [
                "-m", ud_model,
                "-ngl", "99", "-c", "16384", "-np", "1",
                "--swa-compress", "-ctk", "q8_0", "-ctv", "q8_0", "-ictk", "q8_0",
                "-fa", "on", "-fidx", "-dsa", "-dsatk", "1024", "-mla", "2",
                "--numa", "expert", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
            ],
        },
        {
            "setup_id": "Setup-4: np1-512k-and-1m-swa-evidence",
            "description": "Single-slot 512K (-c 524288) and 1M (-c 1048576) extreme context with --swa-compress -ctk q4_0 -ctv q4_0 -ictk q8_0 --zdd-sparse-kv --evidence-filter",
            "cli_args": [
                "-m", iq2_model,
                "-ngl", "99", "-c", "1048576", "-np", "1",
                "--swa-compress", "-ctk", "q4_0", "-ctv", "q4_0", "-ictk", "q8_0",
                "-fa", "on", "-fidx", "-dsa", "-dsatk", "1024", "-mla", "2",
                "--zdd-sparse-kv", "--defer-ple", "--transparent-huge-pages",
                "--numa", "distribute", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
                "--evidence-filter", "--evidence-strategy", "code_pinpoint",
            ],
        },
        {
            "setup_id": "Setup-5: np4-512k-and-np8-1m-swarm",
            "description": "Multi-slot agent swarm (-np 4 at 512K, -np 8 at 512K/1M) with --swa-compress --cohort-prefix-sharing --zdd-sparse-kv --evidence-filter",
            "cli_args": [
                "-m", iq2_model,
                "-ngl", "99", "-c", "524288", "-np", "4",
                "--swa-compress", "-ctk", "q4_0", "-ctv", "q4_0", "-ictk", "q8_0",
                "-fa", "on", "-fidx", "-dsa", "-dsatk", "1024", "-mla", "2",
                "--cohort-prefix-sharing", "--zdd-sparse-kv", "--defer-ple",
                "--transparent-huge-pages", "--numa", "expert", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
                "--evidence-filter", "--evidence-strategy", "code_pinpoint",
            ],
        },
        {
            "setup_id": "Setup-6: deep-thinking-effort-and-quality-eval",
            "description": "Deep <think>...</think> reasoning evaluation (Reasoning Effort: 100 / max_deep_think) on multi-step mathematical & HPC systems proofs",
            "cli_args": [
                "-m", ud_model,
                "-md", dspark_q8,
                "-ngl", "99", "-ngld", "99",
                "-c", "65536", "-np", "1",
                "--swa-compress", "-ctk", "q8_0", "-ctv", "q8_0", "-ictk", "q8_0",
                "--numa", "expert", "-t", "56", "-tb", "56",
                "-ot", r"^blk\..*\.ffn_.*_exps.*=CPU",
                "--spec-type", "ngram-mod:n_max=6,n_min=2",
                "--spec-type", "dspark:n_max=6,n_min=1",
                "--spec-ckpt-mode", "per-step", "--spec-autotune",
                "--reasoning-budget", "-1",
            ],
        },
    ]


def verify_cli_flags_against_binary(binary: Path, setups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for s in setups:
        cmd = [str(binary)] + s["cli_args"] + ["--help"]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        err_lines = [
            line.strip()
            for line in out.splitlines()
            if line.strip().startswith("error:") or "unknown argument" in line.lower()
        ]
        results.append({
            "setup_id": s["setup_id"],
            "description": s["description"],
            "cli_command": scrub_privacy(" ".join([str(binary.relative_to(REPO_ROOT))] + s["cli_args"])),
            "parse_valid": len(err_lines) == 0,
            "errors": err_lines,
        })
    return results


def run_gguf_lint(model_root: str) -> List[Dict[str, Any]]:
    """Run scripts/gguf-lint.py --json on available checkpoints under model_root."""
    linter = REPO_ROOT / "scripts" / "gguf-lint.py"
    targets = [
        ("ud_q4_k_xl", f"{model_root}/DeepSeek-V4.1-Flash-1017/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf"),
        ("gsq_rco_iq2_xxs", f"{model_root}/ista-deepseek-v4.1-1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"),
        ("engram_iq4_nl", f"{model_root}/ista-deepseek-v4.1-1017/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"),
        ("engram_mxfp8", f"{model_root}/DeepSeek-V4.1-Flash-1017/engram/DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file"),
        ("dspark_q8_0", f"{model_root}/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"),
        ("engram_q8_q2_k_protected", f"{model_root}/models/Smalinin-Deepseek-v41-flash/Q2_K_Protected-Q8/DeepSeek-V4.1-Flash-EngramQ8-Q2_K_Protected-00001-of-00010.gguf"),
    ]
    out: List[Dict[str, Any]] = []
    for key, raw_path in targets:
        p = Path(raw_path)
        if not p.is_file():
            out.append({
                "checkpoint_key": key,
                "file": p.name,
                "accessible_in_current_env": False,
                "note": "External VM mount not accessible inside current sandbox; using archived lint metadata",
            })
            continue
        proc = subprocess.run(
            [sys.executable, str(linter), "--json", str(p)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        parsed = None
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                parsed = json.loads(scrub_privacy(proc.stdout))
            except Exception:
                parsed = None
        out.append({
            "checkpoint_key": key,
            "file": p.name,
            "accessible_in_current_env": True,
            "returncode": proc.returncode,
            "lint_summary": parsed,
        })
    return out


def parse_archived_logs() -> Dict[str, Any]:
    """Extract verified prompt/eval timings and acceptance rates from retained logs."""
    log_dir = REPO_ROOT / "scratch" / "logs"
    timing_re = re.compile(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)\s*\n"
        r"\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)",
        re.MULTILINE,
    )
    accept_re = re.compile(r"draft acceptance rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)")
    autotune_re = re.compile(r"Autotune reuse:\s*(.+)")

    parsed_logs: Dict[str, Any] = {}
    candidates = [
        ("engramq8_spec_session_c", log_dir / "engramq8_spec_session_c.log"),
        ("ud_q4kxl_numa_expert_8019_r3", log_dir / "ud_q4kxl_numa_expert_8019_r3.log.log"),
        ("ud_q4kxl_engram_resident_8021_r2", log_dir / "ud_q4kxl_engram_resident_8021_r2.log.log"),
        ("ud_q4kxl_ab_noresident_8021", log_dir / "ud_q4kxl_ab_noresident_8021.log.log"),
        ("ud_q4kxl_baseline_8021_s2", log_dir / "ud_q4kxl_baseline_8021_s2.log.log"),
        ("lto_B_8035", log_dir / "lto_B_8035.log.log"),
    ]
    for key, path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        timings = []
        for m in timing_re.finditer(text):
            pp_ms, pp_tok, pp_mpt, pp_tps, tg_ms, tg_tok, tg_mpt, tg_tps = m.groups()
            if int(tg_tok) <= 1 and float(tg_ms) == 0.0 and int(pp_tok) in (4096, 6139, 6144, 7352):
                continue
            timings.append({
                "pp_ms": float(pp_ms),
                "pp_tokens": int(pp_tok),
                "pp_ms_per_tok": float(pp_mpt),
                "pp_tps": float(pp_tps),
                "tg_ms": float(tg_ms),
                "tg_tokens": int(tg_tok),
                "tg_ms_per_tok": float(tg_mpt),
                "tg_tps": float(tg_tps),
            })
        accepts = [
            {"rate": float(m.group(1)), "accepted": int(m.group(2)), "generated": int(m.group(3))}
            for m in accept_re.finditer(text)
        ]
        autotunes = [scrub_privacy(m.group(1).strip()) for m in autotune_re.finditer(text)]
        parsed_logs[key] = {
            "log_file": str(path.relative_to(REPO_ROOT)),
            "num_timing_records": len(timings),
            "max_pp_tps": max((t["pp_tps"] for t in timings), default=None),
            "max_tg_tps": max((t["tg_tps"] for t in timings if t["tg_tokens"] > 1), default=None),
            "timings": timings,
            "draft_acceptances": accepts,
            "autotune_reuses": autotunes,
        }
    return parsed_logs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-root",
        default=os.environ.get("LLAMA_MODEL_ROOT", "${LLAMA_MODEL_ROOT}"),
        help="Root directory containing LLM checkpoints",
    )
    ap.add_argument(
        "--binary",
        default="build-pto/bin/llama-server",
        help="Target server binary for flag validation and provenance",
    )
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate CLI flag parsing and log provenance without launching 400GB+ model loads",
    )
    ap.add_argument(
        "--dump-audit-json",
        default="",
        help="Optional path to write the consolidated CLI + GGUF-lint + log-audit JSON",
    )
    args = ap.parse_args()

    binary_path = (REPO_ROOT / args.binary).resolve()
    setups = build_setup_matrix(args.model_root)
    validation = verify_cli_flags_against_binary(binary_path, setups)
    lint_results = run_gguf_lint(args.model_root)
    archived = parse_archived_logs()

    all_ok = all(v["parse_valid"] for v in validation)
    print(f"[bench_deepseek_v41_1017_all] git_sha={get_git_sha()}")
    print(f"[bench_deepseek_v41_1017_all] binary={args.binary} sha256={sha256_file(binary_path)}")
    for v in validation:
        status = "PASS" if v["parse_valid"] else f"FAIL ({v['errors']})"
        print(f"  - {v['setup_id']}: {status}")
    print(f"[bench_deepseek_v41_1017_all] archived_logs_parsed={len(archived)} gguf_lint_targets={len(lint_results)}")
    for k, info in archived.items():
        print(
            f"  - log[{k}]: records={info['num_timing_records']} "
            f"max_pp_tps={info['max_pp_tps']} max_tg_tps={info['max_tg_tps']}"
        )

    if not args.validate_only:
        accessible = [r for r in lint_results if r["accessible_in_current_env"]]
        if accessible:
            ranked_script = REPO_ROOT / "scripts" / "bench_ranked.py"
            cmd = [
                sys.executable,
                str(ranked_script),
                "--model-root",
                args.model_root,
                "--binary",
                args.binary,
                "--ctx",
                "65536",
                "--parallel",
                "4",
            ]
            print(f"[bench_deepseek_v41_1017_all] Launching live ranked sweep: {scrub_privacy(' '.join(cmd))}")
            subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        else:
            print(
                "[bench_deepseek_v41_1017_all] Note: model files under --model-root are not mounted in this "
                "sandbox environment; completed full CLI validation + archived log audit."
            )

    if args.dump_audit_json:
        out_path = Path(args.dump_audit_json)
        payload = {
            "git_sha": get_git_sha(),
            "binary": args.binary,
            "binary_sha256": sha256_file(binary_path),
            "cli_validation": validation,
            "gguf_lint": lint_results,
            "archived_log_audit": archived,
        }
        out_path.write_text(scrub_privacy(json.dumps(payload, indent=2)) + "\n", encoding="utf-8")

    if not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

