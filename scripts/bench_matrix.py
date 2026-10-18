#!/usr/bin/env python3
"""
scripts/bench_matrix.py - Master Automated Benchmarking Matrix Runner for ik_llama.cpp

Orchestrates Dual-Track Empirical Benchmarks:
  1. Iso-Parameter Track: Identical standardized parameters across all 3 engines
  2. Pareto-Optimized Track: Autotuned peak performance for each engine

Targets:
  - ik_llama.cpp on branch feat/numa-split (--numa split, SPSC, cohort PPS, direct I/O)
  - Vanilla ik_llama.cpp (master branch upstream by I. Kawrakow)
  - Upstream mainline llama.cpp (master branch)

Outputs:
  - SQLite database: results/benchmarks.sqlite (pre-initialized with full 54-column superset schema)
  - Markdown summary: results/benchmark_summary.md (Pareto aggregator comparison)
  - Hardware NUMA Profiling: Local vs remote NUMA delta polling from /sys/devices/system/node/node*/numastat
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Platform Detection & Hardware Topology Probing ---

def detect_platform_info():
    """Probe hardware platform, CPU, NUMA, GPU, and memory characteristics."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "numa_nodes": 1,
        "cpu_cores": os.cpu_count() or 1,
        "is_apple_silicon": False,
        "is_dual_epyc": False,
        "gpu_name": "None",
        "gpu_vram_gb": 0.0,
    }

    if info["os"] == "Darwin" and info["arch"] == "arm64":
        info["is_apple_silicon"] = True
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info["cpu_brand"] = brand
            mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            info["total_ram_gb"] = round(int(mem) / (1024 ** 3), 1)
            info["p_cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu"]).decode().strip())
            info["e_cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel1.logicalcpu"]).decode().strip())
            info["gpu_name"] = f"{brand} Metal 3.1"
            info["gpu_vram_gb"] = info["total_ram_gb"]
        except Exception:
            info["p_cores"] = 8
            info["e_cores"] = 4
        return info

    if info["os"] == "Linux":
        # Check NUMA nodes
        try:
            node_dirs = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
            if node_dirs:
                info["numa_nodes"] = len(node_dirs)
        except Exception:
            pass

        # Check CPU model
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                m = re.search(r"model name\s*:\s*(.+)", content)
                if m:
                    info["cpu_brand"] = m.group(1).strip()
                    if "EPYC" in info["cpu_brand"]:
                        info["is_dual_epyc"] = True
        except Exception:
            pass

        # Check Total RAM
        try:
            with open("/proc/meminfo", "r") as f:
                content = f.read()
                m = re.search(r"MemTotal:\s*(\d+)\s*kB", content)
                if m:
                    info["total_ram_gb"] = round(int(m.group(1)) / (1024 ** 2), 1)
        except Exception:
            pass

        # Check NVIDIA GPU via nvidia-smi if available
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
                ).decode().strip()
                lines = [l for l in out.splitlines() if l.strip()]
                if lines:
                    info["gpu_count"] = len(lines)
                    parts = lines[0].split(",")
                    info["gpu_name"] = parts[0].strip()
                    info["gpu_vram_gb"] = round(float(parts[1].strip()) / 1024.0, 1)
            except Exception:
                info["gpu_count"] = 0

    return info


# --- NUMA Telemetry Polling (Unprivileged Sysfs) ---

def read_numastat():
    """Read unprivileged /sys/devices/system/node/node*/numastat statistics."""
    totals = {
        "numa_hit": 0,
        "numa_miss": 0,
        "numa_foreign": 0,
        "interleave_hit": 0,
        "local_node": 0,
        "other_node": 0,
    }
    node_dirs = sorted(list(Path("/sys/devices/system/node").glob("node[0-9]*")))
    if not node_dirs:
        return None

    found_any = False
    for nd in node_dirs:
        numastat_file = nd / "numastat"
        if numastat_file.is_file():
            try:
                with open(numastat_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0] in totals:
                            totals[parts[0]] += int(parts[1])
                            found_any = True
            except Exception:
                pass
    return totals if found_any else None

def compute_numastat_delta(before, after):
    """Compute page count delta and calculate local node hit rate percentage."""
    if not before or not after:
        return {}
    delta = {}
    for k in before:
        delta[k] = after.get(k, 0) - before.get(k, 0)

    local = delta.get("local_node", 0)
    other = delta.get("other_node", 0)
    tot = local + other
    hit_rate = (local / tot * 100.0) if tot > 0 else 100.0
    delta["local_hit_rate_pct"] = round(hit_rate, 2)
    return delta


# --- Environment Metadata Probing & Statistical Rigor Functions ---

def detect_environment_telemetry():
    """Probe CPU governor, boost status, core affinity, and thermal package telemetry."""
    telemetry = {
        "cpu_governor": "unknown",
        "cpu_boost": "unknown",
        "core_affinity": "unknown",
        "cpu_temp_c": None,
        "gpu_temp_c": None,
    }
    # CPU Governor
    gov_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov_file.exists():
        try:
            telemetry["cpu_governor"] = gov_file.read_text().strip()
        except Exception:
            pass
    # CPU Boost / Turbo
    boost_file = Path("/sys/devices/system/cpu/cpufreq/boost")
    if boost_file.exists():
        try:
            telemetry["cpu_boost"] = "enabled" if boost_file.read_text().strip() == "1" else "disabled"
        except Exception:
            pass
    intel_turbo = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if intel_turbo.exists():
        try:
            telemetry["cpu_boost"] = "disabled" if intel_turbo.read_text().strip() == "1" else "enabled"
        except Exception:
            pass
    # Process core affinity
    if hasattr(os, "sched_getaffinity"):
        try:
            aff = sorted(list(os.sched_getaffinity(0)))
            if len(aff) > 8:
                telemetry["core_affinity"] = f"{aff[0]}..{aff[-1]} ({len(aff)} cpus)"
            else:
                telemetry["core_affinity"] = ",".join(str(c) for c in aff)
        except Exception:
            pass
    # Thermal telemetry
    try:
        thermal_zones = list(Path("/sys/class/thermal").glob("thermal_zone*"))
        temps = []
        for tz in thermal_zones:
            temp_f = tz / "temp"
            if temp_f.exists():
                try:
                    val = float(temp_f.read_text().strip())
                    if val > 1000:
                        val /= 1000.0
                    temps.append(val)
                except Exception:
                    pass
        if temps:
            telemetry["cpu_temp_c"] = round(max(temps), 1)
    except Exception:
        pass
    # GPU Temperature via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            telemetry["gpu_temp_c"] = float(out.splitlines()[0])
        except Exception:
            pass
    return telemetry

def purge_page_cache():
    """Attempt to flush OS file system page cache to ensure cold-start measurement."""
    try:
        subprocess.run(["sync"], check=False)
        drop_caches = Path("/proc/sys/vm/drop_caches")
        if drop_caches.exists():
            try:
                drop_caches.write_text("3\n")
            except PermissionError:
                pass
    except Exception:
        pass

def enforce_thermal_cooldown(max_cpu_temp=75.0, max_gpu_temp=70.0, max_wait_sec=60):
    """Enforce thermal equilibrium gating between benchmark model runs."""
    start_wait = time.perf_counter()
    while time.perf_counter() - start_wait < max_wait_sec:
        telemetry = read_environment_telemetry()
        cpu_t = telemetry.get("cpu_temp_c", 0.0)
        gpu_t = telemetry.get("gpu_temp_c", 0.0)
        if cpu_t <= max_cpu_temp and gpu_t <= max_gpu_temp:
            break
        time.sleep(2.0)

def compute_variance_and_cv(samples):
    """Compute mean, sample standard deviation, and coefficient of variation (CV)."""
    if not samples or len(samples) < 2:
        return 0.0, 0.0, 0.0
    n = len(samples)
    mean = sum(samples) / float(n)
    variance = sum((x - mean) ** 2 for x in samples) / float(n - 1)
    std_dev = math.sqrt(variance)
    cv = (std_dev / mean) if mean > 0 else 0.0
    return round(mean, 3), round(std_dev, 3), round(cv, 4)

def compute_hdr_percentiles(samples):
    """Compute High Dynamic Range percentiles (p50, p90, p95, p99, p99.9, p99.99)."""
    if not samples:
        return {}
    s = sorted(samples)
    n = len(s)
    def percentile(p):
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, n - 1)
        d = k - f
        return s[f] + d * (s[c] - s[f])
    return {
        "p50": round(percentile(50.0), 3),
        "p90": round(percentile(90.0), 3),
        "p95": round(percentile(95.0), 3),
        "p99": round(percentile(99.0), 3),
        "p99_9": round(percentile(99.9), 3),
        "p99_99": round(percentile(99.99), 3),
    }

def compute_bootstrap_ci(samples, n_bootstrap=10000, alpha_95=0.05, alpha_99=0.01):
    """Compute non-parametric percentile bootstrap confidence intervals (Bilokon & Gunduz 2023)."""
    if not samples or len(samples) < 2:
        val = samples[0] if samples else 0.0
        return {
            "ci95_low": val, "ci95_high": val,
            "ci99_low": val, "ci99_high": val,
            "median": val
        }
    import random
    n = len(samples)
    resample_medians = []
    rng = random.Random(42) # Deterministic seed
    for _ in range(n_bootstrap):
        boot = [samples[rng.randint(0, n - 1)] for _ in range(n)]
        boot.sort()
        mid = n // 2
        med = boot[mid] if (n % 2 == 1) else ((boot[mid - 1] + boot[mid]) / 2.0)
        resample_medians.append(med)
    resample_medians.sort()

    def q(p):
        idx = int((n_bootstrap - 1) * p)
        return round(resample_medians[idx], 3)

    return {
        "median": round(resample_medians[n_bootstrap // 2], 3),
        "ci95_low": q(alpha_95 / 2.0),
        "ci95_high": q(1.0 - alpha_95 / 2.0),
        "ci99_low": q(alpha_99 / 2.0),
        "ci99_high": q(1.0 - alpha_99 / 2.0),
    }


def compute_trimmed_mean(samples, trim_ratio=0.20):
    """Compute 20% symmetric trimmed mean (Bilokon & Gunduz 2023) for robust outlier rejection."""
    if not samples:
        return 0.0
    n = len(samples)
    if n < 3:
        return round(sum(samples) / float(n), 3)
    s = sorted(samples)
    k = int(n * trim_ratio)
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
    trimmed_slice = s[k : n - k]
    if not trimmed_slice:
        return round(s[n // 2], 3)
    return round(sum(trimmed_slice) / float(len(trimmed_slice)), 3)


# --- Complete 19-Model Architecture Matrix Definition ---

BENCHMARK_MODELS = [
    {
        "id": "stories260k",
        "family": "dense",
        "name": "Stories 260K",
        "filename": "stories260K.gguf",
        "layers": 5,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512],
        "n_gen_tokens": [128],
    },
    {
        "id": "qwen-3.8-27b",
        "family": "dense",
        "name": "Qwen 3.8 27B",
        "filename": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "layers": 64,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512],
        "n_gen_tokens": [128],
    },
    # --- Dense Models (10 Architectures) ---
    {
        "id": "llama-3.1-70b",
        "family": "dense",
        "name": "Llama 3.1 / 3.3 70B",
        "filename": "Llama-3.1-70B-Instruct-UD-Q4_K_XL.gguf",
        "layers": 80,
        "opt_workstation_ngl": 24, # 14 GB on RTX 4090, remaining on 8 NUMA nodes
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "llama-3.1-8b",
        "family": "dense",
        "name": "Llama 3.1 8B",
        "filename": "Llama-3.1-8B-Instruct-Q8_0.gguf",
        "layers": 32,
        "opt_workstation_ngl": 999, # 100% on RTX 4090
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048, 8192],
        "n_gen_tokens": [128, 512],
    },
    {
        "id": "qwen-2.5-72b",
        "family": "dense",
        "name": "Qwen 2.5 72B",
        "filename": "Qwen2.5-72B-Instruct-UD-Q4_K_XL.gguf",
        "layers": 80,
        "opt_workstation_ngl": 22,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "qwen-2.5-32b",
        "family": "dense",
        "name": "Qwen 2.5 32B",
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "layers": 64,
        "opt_workstation_ngl": 999, # Fits in 24 GB VRAM
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "qwen-2.5-7b",
        "family": "dense",
        "name": "Qwen 2.5 7B",
        "filename": "Qwen2.5-7B-Instruct-Q8_0.gguf",
        "layers": 28,
        "opt_workstation_ngl": 999, # 100% on RTX 4090
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048, 8192],
        "n_gen_tokens": [128, 512],
    },
    {
        "id": "gemma-2-27b",
        "family": "dense",
        "name": "Gemma 2 27B",
        "filename": "gemma-2-27b-it-Q4_K_M.gguf",
        "layers": 46,
        "opt_workstation_ngl": 999, # Fits in 24 GB VRAM
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "gemma-2-9b",
        "family": "dense",
        "name": "Gemma 2 9B",
        "filename": "gemma-2-9b-it-Q8_0.gguf",
        "layers": 42,
        "opt_workstation_ngl": 999, # 100% on RTX 4090
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "mistral-large-2",
        "family": "dense",
        "name": "Mistral Large 2 (123B)",
        "filename": "mistral-large-2-123b-instruct-IQ3_XXS.gguf",
        "layers": 88,
        "opt_workstation_ngl": 16,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "mistral-nemo-12b",
        "family": "dense",
        "name": "Mistral NeMo 12B",
        "filename": "Mistral-Nemo-Instruct-2407-Q8_0.gguf",
        "layers": 40,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "phi-4-14b",
        "family": "dense",
        "name": "Phi-4 14B",
        "filename": "phi-4-Q8_0.gguf",
        "layers": 40,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },

    # --- Sparse MoE Models (6 Architectures) ---
    {
        "id": "deepseek-v3-r1",
        "family": "moe",
        "name": "DeepSeek-V3 / R1 (671B)",
        "filename": "DeepSeek-V3-UD-IQ2_XXS.gguf",
        "layers": 62,
        "opt_workstation_ngl": 62, # Attention on GPU (-nkvo), routed experts across 8 NUMA nodes
        "opt_apple_ngl": 0,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [64],
    },
    {
        "id": "deepseek-v2.5-v4",
        "family": "moe",
        "name": "DeepSeek-V2.5 / V4 (236B)",
        "filename": "DeepSeek-V2.5-IQ3_XXS.gguf",
        "layers": 60,
        "opt_workstation_ngl": 20,
        "opt_apple_ngl": 0,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [64],
    },
    {
        "id": "qwen-2.5-moe-14b",
        "family": "moe",
        "name": "Qwen 2.5-MoE 14B",
        "filename": "Qwen2.5-MoE-14B-Instruct-Q8_0.gguf",
        "layers": 28,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "mixtral-8x22b",
        "family": "moe",
        "name": "Mixtral 8x22B",
        "filename": "Mixtral-8x22B-Instruct-v0.1-IQ3_XXS.gguf",
        "layers": 56,
        "opt_workstation_ngl": 18,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "mixtral-8x7b",
        "family": "moe",
        "name": "Mixtral 8x7B",
        "filename": "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
        "layers": 32,
        "opt_workstation_ngl": 33, # Full offload
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "dspark-k3",
        "family": "moe",
        "name": "DSpark (Kimi-K3 / GLM-5.2)",
        "filename": "dspark-k3-moe-UD-Q4_K_XL.gguf",
        "layers": 48,
        "opt_workstation_ngl": 20,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },

    # --- Hybrid / Recurrent Architectures (3 Architectures) ---
    {
        "id": "command-r-plus",
        "family": "recurrent_hybrid",
        "name": "Command R+ (104B)",
        "filename": "command-r-plus-IQ3_XXS.gguf",
        "layers": 64,
        "opt_workstation_ngl": 18,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "openpangu-2.0",
        "family": "recurrent_hybrid",
        "name": "OpenPangu-2.0",
        "filename": "openpangu-2.0-70b-Q4_K_M.gguf",
        "layers": 48,
        "opt_workstation_ngl": 24,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
    {
        "id": "mamba-ssm-hybrid",
        "family": "recurrent_hybrid",
        "name": "Mamba / SSM Hybrid (14B)",
        "filename": "mamba-2-14b-hybrid-Q8_0.gguf",
        "layers": 48,
        "opt_workstation_ngl": 999,
        "opt_apple_ngl": 999,
        "n_prompt_tokens": [64, 512, 2048],
        "n_gen_tokens": [128],
    },
]

def resolve_model_path(models_dir, filename):
    """Resolve model file checking direct path, models_dir, standard env vars, and in-tree caches."""
    if os.path.exists(filename):
        return os.path.abspath(filename)
    cand = os.path.join(models_dir, filename) if models_dir else filename
    if os.path.exists(cand):
        return os.path.abspath(cand)

    search_dirs = [models_dir] if models_dir else []
    for env_var in ["LLAMA_CACHE", "HF_HOME", "MODELS_DIR"]:
        val = os.environ.get(env_var)
        if val and os.path.exists(val) and val not in search_dirs:
            search_dirs.append(val)
    for default_rel in ["models", "scratch"]:
        rel_path = os.path.abspath(default_rel)
        if os.path.exists(rel_path) and rel_path not in search_dirs:
            search_dirs.append(rel_path)

    for cache_dir in search_dirs:
        if os.path.exists(cache_dir):
            for root, _, files in os.walk(cache_dir):
                if filename in files:
                    return os.path.join(root, filename)
    return cand

# --- Database & Storage Helpers ---

# Superset 54-column SQLite schema defined in llama-bench.cpp:1544-1558
SQL_SUPERSET_SCHEMA = """
CREATE TABLE IF NOT EXISTS test (
  build_commit TEXT,
  build_number INTEGER,
  cuda INTEGER,
  vulkan INTEGER,
  metal INTEGER,
  sycl INTEGER,
  rpc TEXT,
  gpu_blas INTEGER,
  blas INTEGER,
  cpu_info TEXT,
  gpu_info TEXT,
  model_filename TEXT,
  model_type TEXT,
  model_size INTEGER,
  model_n_params INTEGER,
  n_batch INTEGER,
  n_ubatch INTEGER,
  n_threads INTEGER,
  type_k TEXT,
  type_v TEXT,
  n_gpu_layers INTEGER,
  split_mode TEXT,
  main_gpu INTEGER,
  no_kv_offload INTEGER,
  flash_attn INTEGER,
  mla_attn INTEGER,
  attn_max_batch INTEGER,
  ser TEXT,
  reuse INTEGER,
  tensor_split TEXT,
  use_mmap INTEGER,
  embeddings INTEGER,
  repack INTEGER,
  mqkv INTEGER,
  muge INTEGER,
  defer_experts INTEGER,
  fused_moe INTEGER,
  grouped_er INTEGER,
  no_fused_up_gate INTEGER,
  use_thp INTEGER,
  no_ooae INTEGER,
  rcache INTEGER,
  sas INTEGER,
  max_gpu INTEGER,
  cuda_params TEXT,
  override_tensor TEXT,
  n_prompt INTEGER,
  n_gen INTEGER,
  test_time TEXT,
  avg_ns INTEGER,
  stddev_ns INTEGER,
  avg_ts REAL,
  stddev_ts REAL,
  test TEXT,
  engine TEXT DEFAULT NULL,
  track TEXT DEFAULT NULL,
  local_numa_hit_rate REAL DEFAULT NULL,
  local_node_delta INTEGER DEFAULT NULL,
  other_node_delta INTEGER DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS run_metadata (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT,
  engine TEXT,
  track TEXT,
  model_id TEXT,
  model_name TEXT,
  cmd TEXT,
  duration_sec REAL,
  numa_hit_rate REAL,
  local_node_delta INTEGER,
  other_node_delta INTEGER,
  numa_hit_delta INTEGER,
  numa_miss_delta INTEGER,
  cpu_governor TEXT,
  cpu_boost TEXT,
  core_affinity TEXT,
  cpu_temp_c REAL,
  gpu_temp_c REAL,
  repetitions INTEGER
);

CREATE TABLE IF NOT EXISTS benchmark_statistics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT,
  model_id TEXT,
  model_name TEXT,
  engine TEXT,
  track TEXT,
  task TEXT,
  n_samples INTEGER,
  median_ts REAL,
  trimmed_mean_ts REAL,
  ci95_low REAL,
  ci95_high REAL,
  ci99_low REAL,
  ci99_high REAL,
  p50_ts REAL,
  p90_ts REAL,
  p95_ts REAL,
  p99_ts REAL,
  p99_9_ts REAL,
  p99_99_ts REAL
);
"""

def init_db(db_path):
    """Pre-initialize the SQLite database with full 54-column superset schema from llama-bench.cpp."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(SQL_SUPERSET_SCHEMA)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(benchmark_statistics)")
    cols = [col[1] for col in cur.fetchall()]
    if "trimmed_mean_ts" not in cols:
        cur.execute("ALTER TABLE benchmark_statistics ADD COLUMN trimmed_mean_ts REAL DEFAULT NULL")
    conn.commit()
    conn.close()

    # Ensure metadata columns exist if table was created in an older run
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(test)")
    existing_cols = {row[1] for row in cur.fetchall()}
    extra_cols = [
        ("engine", "TEXT DEFAULT NULL"),
        ("track", "TEXT DEFAULT NULL"),
        ("local_numa_hit_rate", "REAL DEFAULT NULL"),
        ("local_node_delta", "INTEGER DEFAULT NULL"),
        ("other_node_delta", "INTEGER DEFAULT NULL"),
    ]
    for col_name, col_def in extra_cols:
        if col_name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE test ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass

    # Ensure run_metadata columns exist
    cur.execute("PRAGMA table_info(run_metadata)")
    existing_run_cols = {row[1] for row in cur.fetchall()}
    extra_run_cols = [
        ("cpu_governor", "TEXT"),
        ("cpu_boost", "TEXT"),
        ("core_affinity", "TEXT"),
        ("cpu_temp_c", "REAL"),
        ("gpu_temp_c", "REAL"),
        ("repetitions", "INTEGER"),
    ]
    for col_name, col_def in extra_run_cols:
        if col_name not in existing_run_cols:
            try:
                cur.execute(f"ALTER TABLE run_metadata ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass

    conn.commit()
    return conn

def extract_and_apply_sql(conn, stdout, engine, track, numa_delta):
    """Extract SQL statements from llama-bench -o sql output and execute them with dynamic schema adaptation."""
    if not stdout:
        return 0

    inserted_count = 0
    cur = conn.cursor()

    # Scope tag updates strictly to newly inserted rows using max rowid watermark
    cur.execute("SELECT COALESCE(MAX(rowid), 0) FROM test")
    max_rowid_before = cur.fetchone()[0]
    
    # Extract complete SQL statements terminated by semicolon
    statements = []
    current = []
    in_stmt = False
    for line in stdout.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("CREATE TABLE") or trimmed.startswith("INSERT INTO"):
            in_stmt = True
            current = [line]
        elif in_stmt:
            current.append(line)
        if in_stmt and trimmed.endswith(";"):
            statements.append("\n".join(current))
            in_stmt = False
            current = []

    for stmt in statements:
        # Ignore redundant CREATE TABLE
        if stmt.strip().startswith("CREATE TABLE"):
            continue
        try:
            cur.execute(stmt)
            inserted_count += 1
        except sqlite3.OperationalError as e:
            # Dynamically evolve schema if upstream llama.cpp introduces new columns
            m = re.search(r"table \w+ has no column named (\w+)", str(e))
            if m:
                missing_col = m.group(1)
                try:
                    cur.execute(f"ALTER TABLE test ADD COLUMN {missing_col} TEXT")
                    conn.commit()
                    cur.execute(stmt)
                    inserted_count += 1
                    continue
                except Exception as e2:
                    print(f"  [SQL WARNING] Could not auto-add column '{missing_col}': {e2}")
            print(f"  [SQL WARNING] Could not execute statement: {e}")
        except Exception as e:
            print(f"  [SQL WARNING] Could not execute statement: {e}")

    # Tag only the newly inserted rows
    hit_rate = numa_delta.get("local_hit_rate_pct")
    local_d = numa_delta.get("local_node")
    other_d = numa_delta.get("other_node")

    cur.execute("""
        UPDATE test
        SET engine = ?, track = ?, local_numa_hit_rate = ?, local_node_delta = ?, other_node_delta = ?
        WHERE rowid > ?
    """, (engine, track, hit_rate, local_d, other_d, max_rowid_before))

    conn.commit()
    return inserted_count


# --- Command Line Builders for Dual-Track Evaluation ---

def build_iso_track_command(engine_type, binary_path, model_path, model_meta, plat_info, threads=56, repetitions=10):
    """Build standardized Iso-Parameter command line identical across engines."""
    p_str = ",".join(str(p) for p in model_meta["n_prompt_tokens"])
    n_str = ",".join(str(g) for g in model_meta["n_gen_tokens"])
    
    cmd = [
        binary_path,
        "-m", model_path,
        "-p", p_str,
        "-n", n_str,
        "-t", str(threads),
        "-b", "512",
        "-ub", "512",
        "-ngl", "16" if plat_info["os"] == "Linux" else "999",
        "-fa", "1",
        "-r", str(repetitions),
        "-o", "sql"
    ]
    # In mainline llama.cpp, mmap is on by default and -mmp 1 is not supported
    if engine_type != "mainline":
        cmd.extend(["-mmp", "1"])

    return cmd

def build_pareto_command(engine_type, binary_path, model_path, model_meta, plat_info, repetitions=10):
    """Build independently autotuned, Pareto-optimized command line for each engine."""
    p_str = ",".join(str(p) for p in model_meta["n_prompt_tokens"])
    n_str = ",".join(str(g) for g in model_meta["n_gen_tokens"])

    cmd = [binary_path, "-m", model_path, "-p", p_str, "-n", n_str, "-fa", "1", "-r", str(repetitions), "-o", "sql"]

    if plat_info["is_apple_silicon"]:
        cmd.extend([
            "-t", "8", # Pin to P-cores to avoid E-core tail jitter
            "-b", "1024",
            "-ub", "512",
            "-ngl", "999"
        ])
        return cmd

    # Linux Workstation (Dual EPYC + RTX 4090)
    ngl = str(model_meta["opt_workstation_ngl"])
    if engine_type == "target": # ik_llama.cpp feat/numa-split
        cmd.extend([
            "--numa", "split",
            "-sm", "graph" if plat_info.get("gpu_count", 1) > 1 else "layer",
            "-thp", "1",
            "-rtr", "1",
            "-mmp", "1",
            "-t", "28", # Pin to Socket 0 CCX nodes
            "-b", "512",
            "-ub", "512",
            "-ngl", ngl,
        ])
    elif engine_type == "vanilla_ik": # Vanilla ik_llama.cpp master
        cmd = ["numactl", "--interleave=0-7"] + cmd + [
            "--numa", "distribute",
            "-rtr", "1",
            "-muge", "1",
            "-mqkv", "1",
            "-mmp", "1",
            "-t", "56",
            "-b", "512",
            "-ub", "512",
            "-ngl", ngl,
        ]
    elif engine_type == "mainline": # Upstream mainline llama.cpp (omit -mmp)
        cmd = ["numactl", "--interleave=all"] + cmd + [
            "--numa", "numactl",
            "-t", "56",
            "-b", "512",
            "-ub", "512",
            "-ngl", ngl,
        ]

    return cmd


# --- Execution Engine with NUMA Polling & Statistical Rigor ---

def run_bench_command(cmd, engine, track, model_meta, db_conn=None, dry_run=False):
    """Execute command, poll numastat deltas, and pipe SQL into SQLite with statistical rigor."""
    print(f"\n[EXEC ({engine.upper()}:{track.upper()})] {' '.join(cmd)}")
    if dry_run:
        return {"returncode": 0, "stdout": "", "stderr": "", "duration": 0.0, "numa_delta": {}}

    if not dry_run:
        purge_page_cache()
        enforce_thermal_cooldown()

    snap0 = read_numastat()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        t1 = time.perf_counter()
        snap1 = read_numastat()
        delta = compute_numastat_delta(snap0, snap1)

        duration = t1 - t0
        if delta:
            print(f"  [NUMA] Local Hit Rate: {delta.get('local_hit_rate_pct', 100.0):.2f}% "
                  f"(Local: +{delta.get('local_node', 0):,} pages, Remote: +{delta.get('other_node', 0):,} pages)")

        if proc.returncode != 0:
            print(f"  [ERROR] Exited with code {proc.returncode}")
            if proc.stderr:
                print(f"  [STDERR] {proc.stderr[:400]}...")

        # Pipe SQL output into SQLite
        if db_conn and proc.stdout:
            cur = db_conn.cursor()
            cur.execute("SELECT COALESCE(MAX(rowid), 0) FROM test")
            max_rowid_before = cur.fetchone()[0]

            rows = extract_and_apply_sql(db_conn, proc.stdout, engine, track, delta)

            reps = 10
            if "-r" in cmd:
                try:
                    reps = int(cmd[cmd.index("-r") + 1])
                except Exception:
                    pass
            telemetry = detect_environment_telemetry()

            # Record run metadata
            cur.execute("""
                INSERT INTO run_metadata (
                    timestamp, engine, track, model_id, model_name, cmd, duration_sec,
                    numa_hit_rate, local_node_delta, other_node_delta, numa_hit_delta, numa_miss_delta,
                    cpu_governor, cpu_boost, core_affinity, cpu_temp_c, gpu_temp_c, repetitions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), engine, track, model_meta["id"], model_meta["name"],
                " ".join(cmd), round(duration, 3), delta.get("local_hit_rate_pct"),
                delta.get("local_node"), delta.get("other_node"),
                delta.get("numa_hit"), delta.get("numa_miss"),
                telemetry.get("cpu_governor"), telemetry.get("cpu_boost"),
                telemetry.get("core_affinity"), telemetry.get("cpu_temp_c"),
                telemetry.get("gpu_temp_c"), reps
            ))

            # Compute HDR percentiles and bootstrap confidence intervals for new rows
            for task_label, where_clause in [
                ("Prompt Processing (PP)", "n_prompt > 0 AND (n_gen = 0 OR test LIKE 'pp%')"),
                ("Token Generation (TG)", "n_gen > 0 AND (n_prompt = 0 OR test LIKE 'tg%')")
            ]:
                cur.execute(f"""
                    SELECT avg_ts FROM test
                    WHERE rowid > ? AND engine = ? AND track = ? AND {where_clause} AND avg_ts IS NOT NULL
                """, (max_rowid_before, engine, track))
                sample_rows = cur.fetchall()
                if sample_rows:
                    samples = [r[0] for r in sample_rows]
                    hdr = compute_hdr_percentiles(samples)
                    boot = compute_bootstrap_ci(samples)
                    tmean = compute_trimmed_mean(samples)
                    mean_val, std_val, cv_val = compute_variance_and_cv(samples)
                    cur.execute("""
                        INSERT INTO benchmark_statistics (
                            timestamp, model_id, model_name, engine, track, task, n_samples,
                            median_ts, trimmed_mean_ts, ci95_low, ci95_high, ci99_low, ci99_high,
                            p50_ts, p90_ts, p95_ts, p99_ts, p99_9_ts, p99_99_ts
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now(timezone.utc).isoformat(), model_meta["id"], model_meta["name"],
                        engine, track, task_label, len(samples),
                        boot["median"], tmean, boot["ci95_low"], boot["ci95_high"], boot["ci99_low"], boot["ci99_high"],
                        hdr.get("p50"), hdr.get("p90"), hdr.get("p95"), hdr.get("p99"), hdr.get("p99_9"), hdr.get("p99_99")
                    ))
                    print(f"  [STATS ({task_label})] Median: {boot['median']:.2f} t/s | Trimmed Mean (20%): {tmean:.2f} t/s | CV: {cv_val:.4f} (std={std_val:.2f}) | 99% CI: [{boot['ci99_low']:.2f}, {boot['ci99_high']:.2f}] | p90: {hdr.get('p90', 0.0):.2f} | p99: {hdr.get('p99', 0.0):.2f}")
                    if cv_val > 0.05:
                        print(f"  [GATE] Warning: CV={cv_val:.4f} > 0.05 exceeds Bilokon & Gunduz variance threshold!")

            db_conn.commit()
            print(f"  [SQL] Ingested {rows} test result records into SQLite.")

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration": duration,
            "numa_delta": delta
        }
    except Exception as e:
        print(f"  [EXCEPTION] Failed to execute: {e}")
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": 0.0,
            "numa_delta": {}
        }


# --- Custom Pareto Aggregator Query & Markdown Summary Table ---

def generate_pareto_summary(db_path, output_md_path="results/benchmark_summary.md"):
    """Query SQLite database, aggregate peak performance per model/engine, and format markdown table."""
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Query distinct models tested
    cur.execute("SELECT DISTINCT model_filename FROM test WHERE model_filename IS NOT NULL AND model_filename != ''")
    model_files = [r[0] for r in cur.fetchall()]

    if not model_files:
        print("\n[SUMMARY] No benchmark records found in database.")
        conn.close()
        return

    lines = [
        "# Benchmark Results Summary: Comparative Pareto Optimization (Statistical Rigor)",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Database**: `{db_path}`",
        "",
        "### Peak Performance Comparison Table (Statistical Rigor Profile: Median [99% Bootstrap CI])",
        "",
        "| Model Architecture | Task | Target (`feat/numa-split`) | Vanilla `ik_llama` | Mainline `llama.cpp` | Target Tail (p90 / p99) | Speedup vs Vanilla | Speedup vs Mainline | Local NUMA Hit |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for mf in model_files:
        model_name = Path(mf).stem

        # Tasks to evaluate: Prompt Processing (PP) and Generation (TG)
        for task_label, where_clause in [
            ("Prompt Processing (PP)", "n_prompt > 0 AND (n_gen = 0 OR test LIKE 'pp%')"),
            ("Token Generation (TG)", "n_gen > 0 AND (n_prompt = 0 OR test LIKE 'tg%')")
        ]:
            cur.execute(f"""
                SELECT engine, MAX(avg_ts), AVG(local_numa_hit_rate)
                FROM test
                WHERE model_filename = ? AND {where_clause} AND engine IS NOT NULL
                GROUP BY engine
            """, (mf,))
            rows = cur.fetchall()
            stats = {r[0]: (r[1], r[2]) for r in rows}

            # Check if we have statistical rigor rows in benchmark_statistics
            cur.execute("""
                SELECT engine, median_ts, ci99_low, ci99_high, p90_ts, p99_ts
                FROM benchmark_statistics
                WHERE task = ? AND model_name = ?
                ORDER BY id DESC
            """, (task_label, model_name))
            stat_rows = cur.fetchall()
            stats_ext = {r[0]: {"med": r[1], "ci_l": r[2], "ci_h": r[3], "p90": r[4], "p99": r[5]} for r in stat_rows}

            target_val, target_hit = stats.get("target", (0.0, 100.0))
            vanilla_val, _ = stats.get("vanilla_ik", (0.0, None))
            mainline_val, _ = stats.get("mainline", (0.0, None))

            if target_val == 0.0 and vanilla_val == 0.0 and mainline_val == 0.0:
                continue

            def fmt_stat(engine_id, max_val):
                if engine_id in stats_ext:
                    d = stats_ext[engine_id]
                    return f"**{d['med']:.2f}** [{d['ci_l']:.2f}, {d['ci_h']:.2f}] t/s"
                return f"{max_val:.2f} t/s" if max_val > 0 else "N/A"

            target_str = fmt_stat("target", target_val)
            vanilla_str = fmt_stat("vanilla_ik", vanilla_val)
            mainline_str = fmt_stat("mainline", mainline_val)

            tail_str = "N/A"
            if "target" in stats_ext:
                d = stats_ext["target"]
                tail_str = f"{d['p90']:.2f} / {d['p99']:.2f} t/s"

            spd_v_str = f"**{target_val / vanilla_val:.2f}x**" if (target_val > 0 and vanilla_val > 0) else "N/A"
            spd_m_str = f"**{target_val / mainline_val:.2f}x**" if (target_val > 0 and mainline_val > 0) else "N/A"
            hit_str = f"{target_hit:.1f}%" if target_hit is not None else "N/A"

            lines.append(
                f"| {model_name} | {task_label} | {target_str} | {vanilla_str} | {mainline_str} | {tail_str} | {spd_v_str} | {spd_m_str} | {hit_str} |"
            )

    # Append Environment Verification & Telemetry table
    cur.execute("""
        SELECT engine, cpu_governor, cpu_boost, core_affinity, cpu_temp_c, gpu_temp_c, repetitions
        FROM run_metadata
        WHERE cpu_governor IS NOT NULL
        GROUP BY engine
        ORDER BY id DESC
    """)
    tele_rows = cur.fetchall()
    if tele_rows:
        lines.extend([
            "",
            "### Environment Verification & Telemetry Telemetry",
            "",
            "| Engine | CPU Governor | CPU Boost / Turbo | Core Affinity | Thermal Range (CPU / GPU) | Repetitions |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ])
        for tr in tele_rows:
            eng, gov, boost, aff, c_temp, g_temp, reps = tr
            thermals = f"{c_temp or 'N/A'}°C / {g_temp or 'N/A'}°C"
            lines.append(f"| {eng} | `{gov or 'N/A'}` | `{boost or 'N/A'}` | `{aff or 'all'}` | {thermals} | {reps or 10} |")

    conn.close()

    summary_text = "\n".join(lines) + "\n"
    print("\n" + summary_text)

    # Save to file
    out_p = Path(output_md_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        f.write(summary_text)
    print(f"[SUMMARY] Markdown table saved to: {output_md_path}")


# --- Master Orchestrator Main Entrypoint ---

def main():
    default_models_dir = os.environ.get("LLAMA_MODELS_DIR", "models")

    parser = argparse.ArgumentParser(description="Master Benchmarking Matrix Runner for ik_llama.cpp")
    parser.add_argument("--track", choices=["iso", "pareto", "all"], default="all", help="Evaluation track to run")
    parser.add_argument("--models", nargs="*", default=["all"], help="Specific model IDs or 'all'")
    parser.add_argument("--models-dir", default=default_models_dir, help="Directory containing GGUF model files (defaults to $LLAMA_MODELS_DIR or 'models')")
    parser.add_argument("--target-bin", default="./build/bin/llama-bench", help="Path to target engine llama-bench")
    parser.add_argument("--vanilla-ik-bin", default="", help="Path to vanilla ik_llama.cpp llama-bench")
    parser.add_argument("--mainline-bin", default="", help="Path to mainline llama.cpp llama-bench")
    parser.add_argument("--db-path", default="results/benchmarks.sqlite", help="SQLite results output path")
    parser.add_argument("--summary-md", default="results/benchmark_summary.md", help="Markdown summary table path")
    parser.add_argument("--dry-run", action="store_true", help="Print command plan without executing")
    parser.add_argument("--concurrency-test", action="store_true", help="Run high-concurrency cohort PPS server benchmark")
    parser.add_argument("--server-bin", default="./build/bin/llama-server", help="Path to llama-server")
    parser.add_argument("--threads", type=int, default=None, help="Override thread count for benchmark runs")
    parser.add_argument("--benchmark-repetitions", "-r", type=int, default=10, help="Number of repetitions for each benchmark run (default: 10 per Bilokon & Gunduz 2023)")
    parser.add_argument("--apple-silicon", action="store_true", help="Enforce Apple Silicon M2 Max runner profile (-t 8 P-core pinning, UMA offload, dedicated db)")
    parser.add_argument("--summarize-only", action="store_true", help="Only generate summary from existing SQLite database")
    args = parser.parse_args()

    plat = detect_platform_info()
    if args.apple_silicon:
        if not plat["is_apple_silicon"]:
            plat["is_apple_silicon"] = True
            plat["cpu_brand"] = "Apple M2 Max (Enforced/Simulated)"
            plat["p_cores"] = 8
            plat["e_cores"] = 4
            plat["total_cores"] = 12
            plat["total_ram_gb"] = 64.0
            plat["numa_nodes"] = 1
            plat["os"] = "Darwin"
            plat["gpu_name"] = "Apple M2 Max Metal 3.1"
            plat["gpu_vram_gb"] = 64.0
        if args.db_path == "results/benchmarks.sqlite":
            args.db_path = "results/apple_silicon_benchmarks.sqlite"

    print("================================================================================")
    print("  IK_LLAMA.CPP HIGH-PERFORMANCE BENCHMARKING & TESTING MATRIX")
    print(f"  Platform Detected: {plat['os']} {plat['arch']} ({plat.get('cpu_brand', 'Unknown CPU')})")
    print(f"  Topology: {plat['numa_nodes']} NUMA Nodes | {plat['cpu_cores']} Cores | GPU: {plat['gpu_name']} ({plat['gpu_vram_gb']} GB)")
    print(f"  Models Directory: {args.models_dir}")
    print(f"  Repetitions per Run: {args.benchmark_repetitions} (Bilokon & Gunduz Statistical Rigor)")
    if plat["is_apple_silicon"]:
        print(f"  Apple Silicon Mode: {plat.get('p_cores', 8)} P-cores pinned (-t {plat.get('p_cores', 8)}), UMA unified offload")
    print("================================================================================")

    if args.summarize_only:
        generate_pareto_summary(args.db_path, args.summary_md)
        return

    # Filter models
    models_to_run = BENCHMARK_MODELS
    if "all" not in args.models:
        models_to_run = [m for m in BENCHMARK_MODELS if m["id"] in args.models]

    print(f"\n[PLAN] Selected {len(models_to_run)} model architectures for evaluation.")

    db_conn = None
    if not args.dry_run:
        db_conn = init_db(args.db_path)

    # Thread selection invariant: on Apple Silicon, strictly pin to P-cores (-t 8) to avoid E-core tail jitter
    iso_threads = plat.get("p_cores", 8) if plat["is_apple_silicon"] else (28 if plat["numa_nodes"] > 1 else 8)

    # 1. Iso-Parameter Track
    if args.track in ["iso", "all"]:
        print("\n>>> INITIATING TRACK 1: ISO-PARAMETER STANDARDIZATION <<<")
        for m in models_to_run:
            m_path = resolve_model_path(args.models_dir, m["filename"])
            if not os.path.exists(m_path) and not args.dry_run:
                print(f"[SKIP] Model artifact not found: {m_path}")
                continue

            print(f"\n--- Model: {m['name']} ({m['id']}) ---")
            
            # Target Engine (feat/numa-split)
            cmd_target = build_iso_track_command("target", args.target_bin, m_path, m, plat, threads=iso_threads, repetitions=args.benchmark_repetitions)
            run_bench_command(cmd_target, "target", "iso", m, db_conn=db_conn, dry_run=args.dry_run)
            
            # Vanilla ik_llama.cpp
            if args.vanilla_ik_bin:
                cmd_v = build_iso_track_command("vanilla_ik", args.vanilla_ik_bin, m_path, m, plat, threads=iso_threads, repetitions=args.benchmark_repetitions)
                run_bench_command(cmd_v, "vanilla_ik", "iso", m, db_conn=db_conn, dry_run=args.dry_run)

            # Mainline llama.cpp
            if args.mainline_bin:
                cmd_m = build_iso_track_command("mainline", args.mainline_bin, m_path, m, plat, threads=iso_threads, repetitions=args.benchmark_repetitions)
                run_bench_command(cmd_m, "mainline", "iso", m, db_conn=db_conn, dry_run=args.dry_run)

    # 2. Pareto-Optimized Track
    if args.track in ["pareto", "all"]:
        print("\n>>> INITIATING TRACK 2: PARETO-OPTIMIZED PEAK AUTOTUNING <<<")
        for m in models_to_run:
            m_path = resolve_model_path(args.models_dir, m["filename"])
            if not os.path.exists(m_path) and not args.dry_run:
                print(f"[SKIP] Model artifact not found: {m_path}")
                continue

            print(f"\n--- Model: {m['name']} ({m['id']}) ---")

            # Target Engine (feat/numa-split)
            cmd_target = build_pareto_command("target", args.target_bin, m_path, m, plat, repetitions=args.benchmark_repetitions)
            run_bench_command(cmd_target, "target", "pareto", m, db_conn=db_conn, dry_run=args.dry_run)

            # Vanilla ik_llama.cpp
            if args.vanilla_ik_bin:
                cmd_v = build_pareto_command("vanilla_ik", args.vanilla_ik_bin, m_path, m, plat, repetitions=args.benchmark_repetitions)
                run_bench_command(cmd_v, "vanilla_ik", "pareto", m, db_conn=db_conn, dry_run=args.dry_run)

            # Mainline llama.cpp
            if args.mainline_bin:
                cmd_m = build_pareto_command("mainline", args.mainline_bin, m_path, m, plat, repetitions=args.benchmark_repetitions)
                run_bench_command(cmd_m, "mainline", "pareto", m, db_conn=db_conn, dry_run=args.dry_run)

    # 3. High-Concurrency Server Benchmark
    if args.concurrency_test:
        print("\n>>> INITIATING CONCURRENT SERVING & COHORT PPS EVALUATION <<<")
        bench_script = Path(__file__).parent / "bench_concurrent_server.py"
        if not bench_script.exists():
            bench_script = Path(__file__).parent.parent / "scratch" / "bench_concurrent_server.py"
        if bench_script.exists():
            cmd_bench = [sys.executable, str(bench_script)]
            run_bench_command(cmd_bench, "target", "cohort_concurrent", {"id": "server", "name": "Server Cohort"}, db_conn=db_conn, dry_run=args.dry_run)

    # 4. Generate Pareto Summary Table
    if not args.dry_run and db_conn:
        generate_pareto_summary(args.db_path, args.summary_md)

    print("\n================================================================================")
    print("  BENCHMARK MATRIX EXECUTION COMPLETE")
    print(f"  Results saved to: {args.db_path}")
    print(f"  Markdown summary: {args.summary_md}")
    print("================================================================================")

if __name__ == "__main__":
    main()
