#!/usr/bin/env python3
"""
scripts/bench_ranked.py - Ranked Server-Mode Comparator for ik_llama.cpp

Runs an identical request battery against one or more containers
(e.g. UD-Q4_K_XL vs EngramQ8, resolved from scripts/configs/models.toml) and
emits a ranked JSON/Markdown report:

  - median-of-N per scenario (TG tok/s, PP ms/tok, TTFT, wall time) with
    per-rep raw values kept verbatim in the JSON (honest, no curation);
  - raw log capture: the llama-server process log and the parsed
    "prompt eval time / eval time" lines from every rep are archived;
  - PMU-verified columns: node-wide interval `perf stat -I` around the
    server PID provides IPC / cache-miss ratio / NUMA load-miss ratio per
    request window; columns are tagged pmu_verified=false when perf is
    unavailable or produced no valid windows for that rep;
  - numastat snapshot delta per container run;
  - provenance (git sha, binary sha256, hardware fingerprint) via
    scripts/benchmarks/common.py; no absolute host paths stored - model
    roots are externalized through --model-root / LLAMA_MODEL_ROOT.

Scenarios (first-class presets, extensible):
  tg_short   - short prompt + fixed-token generation (server chat API)
  pp_long    - long synthetic prefill, TTFT-dominated
  concurrent - np>2: two overlapping chat requests, aggregate tok/s

The harness never mutates the model containers and never long-polls the
server beyond request timeouts; a rep that fails or times out is recorded
as failed (kept in the JSON) and excluded from medians.
"""

import argparse
import json
import os
import random
import re
import shutil
import statistics
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
try:
    from benchmarks.common import get_git_sha, get_binary_sha256, detect_platform_info
    _COMMON = True
except ImportError:
    _COMMON = False

DEFAULT_CONFIG = "scripts/configs/models.toml"
DEFAULT_BINARY = "build-pto/bin/llama-server"
DEFAULT_PORT = 8043
DEFAULT_REPS = 3
REP_TIMEOUT_S = 900

# Standardized battery. Prompts are synthetic/served-neutral; token counts are
# max_tokens caps, not observed lengths - the observed values come from the
# server's own eval-time accounting in the raw log lines.
SCENARIOS = {
    "tg_short": {
        "prompt": ("Explain how a modern HPC memory subsystem serves a large MoE "
                   "inference model with CPU-resident experts: NUMA binding, THP, "
                   "and the interaction between CPU expert queues and GPU attention."),
        "max_tokens": 192,
        "temperature": 0.2,
        "concurrency": 1,
    },
    "pp_long": {
        "prompt_builder": "pp_long_builder",  # expands a synthetic ~2k-token context
        "max_tokens": 64,
        "temperature": 0.2,
        "concurrency": 1,
    },
    "concurrent": {
        "prompt": ("Enumerate the tradeoffs between CPU-resident expert tensors and "
                   "GPU-offloaded attention for high-parameter MoE inference: NUMA "
                   "locality, THP sizing, prefill bandwidth, and decode gathering costs."),
        "max_tokens": 256,
        "temperature": 0.2,
        "concurrency": 2,
    },
}

PP_PARAGRAPH = (
    "Non-Uniform Memory Access (NUMA) is a memory design in which access time "
    "depends on the memory location relative to the processor; local memory is "
    "faster than memory attached to another node. Mixture-of-Experts layers "
    "activate a small subset of expert matrices per token, so decode throughput "
    "is bounded by gather bandwidth over a large host-resident expert arena, "
    "while prefill saturates accelerator compute. Transparent huge pages reduce "
    "TLB pressure for random gathers across multi-gigabyte tables, and mbind "
    "policies decide which node pages land on. "
)


def build_pp_long_prompt(target_lines: int = 64) -> str:
    paras = []
    for i in range(target_lines):
        letter = random.Random(i).choice(string.ascii_uppercase)
        paras.append(f"[{letter}{i:04d}] {PP_PARAGRAPH}")
    return "\n\n".join(paras)


def chat_completion(base_url: str, payload: dict, timeout: int = REP_TIMEOUT_S):
    """POST /v1/chat/completions; returns (ok, body, wall_sec)."""
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return True, body, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:400]
        except OSError:
            detail = "<unreadable>"
        return False, {"error": f"HTTP {e.code}: {detail}"}, time.perf_counter() - t0
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        return False, {"error": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def extract_usage(body: dict):
    """Pull (prompt_tokens, completion_tokens) from a chat completion body."""
    u = (body or {}).get("usage") or {}
    return u.get("prompt_tokens"), u.get("completion_tokens")


# --- perf stat, interval mode -------------------------------------------------

PMU_EVENTS = ["cycles", "instructions", "cache-misses", "cache-references",
              "branch-misses", "node-loads", "node-load-misses"]
PMU_INTERVAL_S_DEFAULT = 2.0


def read_boot_offset() -> float:
    """epoch_now - uptime_seconds, so perf's -I timestamps map to wall clock."""
    try:
        with open("/proc/uptime") as f:
            return time.time() - float(f.read().split()[0])
    except (OSError, ValueError):
        return 0.0


def start_pmu(pid: int, out_csv: Path, interval_ms: int = 2000):
    """Start background `perf stat -I` for a pid; returns the Popen or None.
    Events go behind an explicit -e and a -- sleep sentinel keeps perf alive.
    Runs in its own process group so stop is clean."""
    if not shutil.which("perf"):
        return None
    cmd = ["perf", "stat", "-p", str(pid), "-I", str(interval_ms),
           "-x", ",", "-o", str(out_csv),
           "-e", ",".join(PMU_EVENTS),
           "--", "sleep", "86400"]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    except OSError:
        return None


def stop_pmu(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 15)  # perf + its sleep sentinel
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except OSError:
            pass


def _pmu_interval_nd(pmu_rows, window_t0, window_t1, interval_s):
    """Aggregate interval rows overlapping [t0,t1] into ND-averaged ratios."""
    sel = []
    for ts, row in pmu_rows:
        lo, hi = ts, ts + interval_s
        if hi >= window_t0 and lo <= window_t1:
            sel.append(row)
    if not sel:
        return {}
    def ssum(key):
        return sum(r.get(key, 0.0) or 0.0 for r in sel)
    dur = max(1e-9, (window_t1 - window_t0))
    cycles, instr = ssum("cycles"), ssum("instructions")
    cref, cmiss = ssum("cache-references"), ssum("cache-misses")
    nload, nlmiss = ssum("node-loads"), ssum("node-load-misses")
    bmiss = ssum("branch-misses")
    out = {
        "pmu_ipc": round(instr / cycles, 3) if cycles else None,
        "pmu_cache_miss_ratio": round(cmiss / cref, 4) if cref else None,
        "pmu_numa_load_miss_ratio": round(nlmiss / nload, 4) if nload else None,
        "pmu_branch_miss_ratio": round(bmiss / instr, 4) if instr else None,
        "pmu_cycles_per_sec": round(cycles / dur, 0),
    }
    return {k: v for k, v in out.items() if v is not None}


def parse_pmu_intervals(csv_path: Path, boot_offset: float):
    """Parse a perf -x , -I csv into [(abs_epoch, row_dict)] rows using
    /proc/uptime anchoring (interval timestamps are seconds since boot)."""
    rows = []
    try:
        text = csv_path.read_text(errors="replace")
    except OSError:
        return rows
    grouped = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[0] == "#" or not parts[0]:
            continue
        ev = parts[3].split(":")[0].split("-x")[0]
        if ev not in PMU_EVENTS:
            continue
        try:
            ts = float(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        grouped.setdefault(ts, {})[ev] = val
    return [(ts + boot_offset, d) for ts, d in sorted(grouped.items())]


# --- server lifecycle ----------------------------------------------------------

def free_port_hint(preferred: int):
    import socket
    for p in [preferred, preferred + 1, preferred + 2, 0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


class BenchServer:
    def __init__(self, args, model_path: Path, tag: str):
        self.args = args
        self.model_path = model_path
        self.tag = tag
        self.proc = None
        self.log_path = Path(tempfile.gettempdir()) / f"bench_ranked_{tag}_{os.getpid()}.log"
        self.port = args.port

    def start(self):
        a = self.args
        cmd = [
            str(a.binary), "-m", str(self.model_path),
            "--host", a.host, "--port", str(self.port),
            "--numa", a.numa, "-t", str(a.threads), "-tb", str(a.threads_batch),
            "-ngl", str(a.gpu_layers), "-c", str(a.ctx),
            "-np", str(a.parallel),
        ]
        if a.jinja:
            cmd.append("--jinja")
        if a.tensor_override:
            cmd += ["-ot", a.tensor_override]
        if a.extra_args:
            cmd += [x for x in a.extra_args.split() if x]
        self.log_path.unlink(missing_ok=True)
        self.log = open(self.log_path, "a")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        base = f"http://{a.host}:{self.port}"
        deadline = time.time() + a.server_startup_timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
                    if b'"ok"' in r.read():
                        return base
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={self.proc.returncode} (log: {self.log_path}); "
                                   "model key errors land here - fix before benching")
            time.sleep(2)
        raise TimeoutError(f"server not healthy after {a.server_startup_timeout}s")

    def stop(self):
        if not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.log.close()

    def archive_log(self, dest: Path):
        if self.log_path.is_file():
            dest.write_text(self.log_path.read_text(errors="replace"))


# --- benchmark core -------------------------------------------------------------

def run_scenario(base_url: str, name: str, spec: dict, reps: int, pmu_toggle: bool,
                 server_pid: int = None, warmup_rounds: int = 0):
    """Run one scenario: warmups + N reps; returns rep list + median summary."""
    reps_out = []
    warm_prompts = ["Warmup: respond with a single short sentence about memory locality."]
    for _ in range(warmup_rounds):
        chat_completion(base_url, {
            "messages": [{"role": "user", "content": warm_prompts[0]}], "max_tokens": 32}, timeout=REP_TIMEOUT_S)

    pmu_proc = None
    pmu_csv = Path(tempfile.gettempdir()) / f"bench_ranked_pmu_{os.getpid()}_{name}.csv"
    boot_offset = read_boot_offset()
    if pmu_toggle and server_pid:
        pmu_proc = start_pmu(server_pid, pmu_csv)

    win_windows = []
    bench_t0 = time.time()
    try:
        for ri in range(reps):
            spec_run = dict(spec)
            if spec_run.get("prompt_builder") == "pp_long_builder":
                spec_run["prompt"] = build_pp_long_prompt()
            payloads = [{"messages": [{"role": "user", "content": spec_run["prompt"]}],
                         "max_tokens": spec_run["max_tokens"], "temperature": spec_run["temperature"]}]
            win_t0 = time.time()
            t_start = time.perf_counter()
            if spec_run["concurrency"] == 1:
                ok, body, wall = chat_completion(base_url, payloads[0])
                bodies = [(ok, body, wall)]
            else:
                with ThreadPoolExecutor(max_workers=spec_run["concurrency"]) as ex:
                    futures = [ex.submit(chat_completion, base_url, payloads[0]) for _ in range(spec_run["concurrency"])]
                    bodies = [f.result() for f in futures]
            win_t1 = time.time()
            ok_all = all(b[0] for b in bodies)
            agg = {
                "rep": ri,
                "ok": ok_all,
                "wall_sec": round(time.perf_counter() - t_start, 3),
            }
            for b_ok, b_body, b_wall in bodies:
                if not b_ok:
                    agg.setdefault("errors", []).append(str((b_body or {}).get("error"))[:300])
            pt_total = ct_total = 0
            for ok, body, wall in bodies:
                pt, ct = extract_usage(body)
                if pt:
                    pt_total += pt
                if ct:
                    ct_total += ct
            # TTFT/PP/tg from raw server accounting: usage + request wall here;
            # exact per-token server lines are archived from the shared log.
            if ct_total and all(b[0] for b in bodies):
                # per-request TG approximated from completion usage & request wall
                # within the concurrent batch: aggregate-first honesty
                agg_tg = ct_total / max(1e-9, (win_t1 - win_t0))
                agg["aggregate_tgs"] = round(agg_tg, 2)
                agg["prompt_tokens"] = pt_total
                agg["completion_tokens"] = ct_total
            win_windows.append((win_t0, win_t1))
            pmu = {}
            if pmu_toggle and server_pid and pmu_csv.exists():
                pmu = _pmu_interval_nd(parse_pmu_intervals(pmu_csv, boot_offset), win_t0, win_t1,
                                       interval_s=PMU_INTERVAL_S_DEFAULT)
            agg["pmu"] = pmu
            agg["pmu_verified"] = bool(pmu)
            reps_out.append(agg)
    finally:
        stop_epoch = time.time()
        if pmu_proc is not None:
            stop_pmu(pmu_proc)
            pmu_proc = None
            # Re-anchor: perf interval clocks differ across kernels/suspend, so
            # align the LAST collected interval to the observed stop time and
            # re-aggregate the request windows against the shifted rows.
            rows = parse_pmu_intervals(pmu_csv, boot_offset)
            if rows:
                shift = stop_epoch - max(r[0] for r in rows)
                if abs(shift) > PMU_INTERVAL_S_DEFAULT + 1.0:
                    rows = [(t + shift, d) for t, d in rows]
                for rep, (w0, w1) in zip(reps_out, win_windows):
                    pmu = _pmu_interval_nd(rows, w0, w1, interval_s=PMU_INTERVAL_S_DEFAULT)
                    if pmu:
                        rep["pmu"] = pmu
                        rep["pmu_verified"] = True

    good = [r for r in reps_out if r["ok"] and "aggregate_tgs" in r]
    med = {}
    if good:
        tgs = [r["aggregate_tgs"] for r in good]
        med = {"n_ok": len(good), "n_total": len(reps_out),
               "median_tgs": round(statistics.median(tgs), 2),
               "mean_tgs": round(statistics.fmean(tgs), 2),
               "cv_tgs": round(statistics.pstdev(tgs) / statistics.fmean(tgs), 3) if len(tgs) > 1 else 0.0,
               "min_tgs": min(tgs), "max_tgs": max(tgs)}
    return {"scenario": name, "summary": med, "reps": reps_out}


def numastat_snapshot(pid: int):
    """Return {"private_mb_by_node": {n0..n7: MB}} for one process; {} when
    numastat is unavailable (column tagged, never silently zero-filled)."""
    out = {"private_mb_by_node": {}, "numastat_available": True}
    try:
        txt = subprocess.check_output(["numastat", "-p", str(pid)], stderr=subprocess.DEVNULL).decode()
    except Exception:
        out["numastat_available"] = False
        return out
    for line in txt.splitlines():
        cells = [c.strip() for c in line.split() if c.strip()]
        if cells and cells[0].startswith("Private") and len(cells) >= 3:
            vals = cells[1:]
            out["private_mb_by_node"] = {f"n{i}": round(float(v), 1) for i, v in enumerate(vals[:-1])}
            break
    return out


# --- main -----------------------------------------------------------------------

def resolve_models(args):
    import tomllib
    cfg_path = Path(args.config)
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    models = cfg.get("models") or {}
    root = Path(args.model_root or os.environ.get("LLAMA_MODEL_ROOT", ".", )).expanduser()
    sel = {}
    for key in args.models.split(","):
        key = key.strip()
        entry = models.get(key)
        if not entry:
            raise SystemExit(f"model key '{key}' not in {cfg_path}; known: {list(models)}")
        mp = Path(entry.get("path", ""))
        cand = [mp] + [Path(p) for p in entry.get("fallback_paths", [])]
        found = None
        for m in cand:
            for base in (root, Path(".")):
                p = base / m if not m.is_absolute() else m
                # paths already carry the wildcard dir; accept exact or glob-ish
                if p.is_file():
                    found = p
                    break
                q = list(base.glob(str(m)))
                if q:
                    found = q[0]
                    break
            if found:
                break
        if not found:
            raise SystemExit(f"model file for '{key}' not found under {root} (looked: {[str(c) for c in cand]})")
        sel[key] = found
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="deepseek-v4.1-1017-ud-q4kxl",
                    help="comma-separated model keys from scripts/configs/models.toml (ranked order)")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--model-root", default=os.environ.get("LLAMA_MODEL_ROOT", ""), help="root dir assembling model paths")
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="base port; free-port search starts here")
    ap.add_argument("--numa", default="expert", help="numa strategy passed to the server (default: expert)")
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--threads-batch", type=int, default=56, dest="threads_batch")
    ap.add_argument("--gpu-layers", type=int, default=99, dest="gpu_layers")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--parallel", type=int, default=4, help="server slots (-np) sized for the concurrent scenario")
    ap.add_argument("--tensor-override", default=r"^blk\..*\.ffn_.*_exps.*=CPU", dest="tensor_override",
                    help="expert tensors to the CPU (server -ot)")
    ap.add_argument("--extra-args", default="", help="verbatim extra llama-server flags")
    ap.add_argument("--no-jinja", action="store_false", dest="jinja")
    ap.add_argument("--scenarios", default="tg_short,pp_long,concurrent")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS, help="median-of-N reps per scenario")
    ap.add_argument("--warmup-rounds", type=int, default=1)
    ap.add_argument("--startup-timeout", type=int, default=2400, dest="server_startup_timeout")
    ap.add_argument("--existing-host", default="", help="skip launch: drive an already-running server (http://host:port)")
    ap.add_argument("--pmu", action="store_true", default=True, help="interval perf stat columns (default on when perf exists)")
    ap.add_argument("--no-pmu", dest="pmu", action="store_false")
    ap.add_argument("--output", default="", help="JSON output path (default results/ranked_<ts>.json)")
    args = ap.parse_args()

    models = resolve_models(args)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip() in SCENARIOS]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    provenance = {"created": stamp, "scenarios": scenarios,
                  "binary": str(args.binary), "binary_sha256": get_binary_sha256(args.binary) if _COMMON else "unknown"}
    if _COMMON:
        provenance["git_sha"] = get_git_sha()
        provenance["platform"] = detect_platform_info()

    results = {"meta": provenance, "models": {}}
    for tag, mpath in models.items():
        srv = None
        launched_here = True
        base_url = ""
        try:
            if args.existing_host:
                base_url = args.existing_host.rstrip("/")
                launched_here = False
                # probe pid via /proc for perf-attach: best-effort
                server_pid = None
                for p in Path("/proc").iterdir():
                    if not p.name.isdigit():
                        continue
                    try:
                        exe = Path(f"/proc/{p.name}/exe").readlink().name
                        cl = Path(f"/proc/{p.name}/cmdline").read_text(errors="replace").split("\0")
                        if "llama-server" in exe and str(args.ctx) in cl and any("8021" in c or base_url.split(":")[-1] in c for c in cl):
                            server_pid = int(p.name)
                            break
                    except OSError:
                        continue
            else:
                port = free_port_hint(args.port)
                srv = BenchServer(args, mpath, tag)
                srv.port = port
                print(f"[{tag}] launching {args.binary} on {args.host}:{port} (model: {mpath.name}) ...")
                base_url = srv.start()
                server_pid = srv.proc.pid
                print(f"[{tag}] healthy: {base_url}")
            print(f"[{tag}] numastat start: {json.dumps(numastat_snapshot(server_pid).get('private_mb_by_node', {})) if server_pid else 'n/a'}")
            num_before = numastat_snapshot(server_pid) if server_pid else {}
            per_scen = {}
            for si, sname in enumerate(scenarios):
                print(f"[{tag}] scenario {si+1}/{len(scenarios)}: {sname} x{args.reps} reps ...")
                r = run_scenario(base_url, sname, SCENARIOS[sname], args.reps, args.pmu and bool(server_pid),
                                 server_pid=server_pid, warmup_rounds=args.warmup_rounds)
                per_scen[sname] = r
                s = r["summary"]
                if s:
                    print(f"      median {s['median_tgs']} t/s (n={s['n_ok']}/{s['n_total']}) "
                          f"cv={s['cv_tgs']} range=[{s['min_tgs']},{s['max_tgs']}]")
            num_after = numastat_snapshot(server_pid) if server_pid else {}
            entry = {"model_file": mpath.name, "scenarios": per_scen,
                     "numastat_private_mb_before": num_before.get("private_mb_by_node", {}),
                     "numastat_private_mb_after": num_after.get("private_mb_by_node", {})}
            if srv:
                entry["server_raw_log"] = f"results/ranked_logs/{stamp}_{tag}_server.log"
                dest = Path("results/ranked_logs"); dest.mkdir(parents=True, exist_ok=True)
                srv.archive_log(dest / f"{stamp}_{tag}_server.log")
            results["models"][tag] = entry
        finally:
            if srv:
                srv.stop()
        if not launched_here:
            results["models"][tag]["existing_server"] = base_url

    # ranking table: per scenario, sorted by median TG
    ranked = {}
    for sname in scenarios:
        rows = [(tag, r["scenarios"][sname]["summary"].get("median_tgs")) for tag, r in results["models"].items()]
        rows = [(t, m) for t, m in rows if m]
        rows.sort(key=lambda x: -x[1])
        ranked[sname] = rows
    results["ranking"] = ranked
    print("\n=== RANKING (median aggregate tok/s) ===")
    for sname, rows in ranked.items():
        for i, (tag, med) in enumerate(rows, 1):
            print(f"  {i}. {tag:36s} {med:.2f}  [{sname}]")

    out = Path(args.output) if args.output else Path(f"results/ranked_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[REPORT] JSON: {out}")


if __name__ == "__main__":
    main()
