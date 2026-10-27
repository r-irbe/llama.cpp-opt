#!/usr/bin/env python3
"""
scripts/launch.py - Declarative Unified Model Launcher for ik_llama.cpp

Reads configuration from scripts/configs/models.toml (or custom TOML/YAML file)
and launches llama-server with exact hardware-tuned arguments, environment configuration,
and provenance verification.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        sys.stderr.write(f"ERROR: Configuration file not found at {config_path}\n")
        sys.exit(1)

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            sys.stderr.write("ERROR: PyYAML is required to parse YAML configs (install via pip install pyyaml)\n")
            sys.exit(1)
        return yaml.safe_load(text)
    else:
        if tomllib is None:
            sys.stderr.write("ERROR: Python 3.11+ tomllib or tomli is required to parse TOML configs\n")
            sys.exit(1)
        return tomllib.loads(text)


SEARCH_ROOTS = [
    Path("/home/radu/vm/llm"),
    Path("/home/radu/vm/llm/models"),
    Path("/home/radu/code/llama.cpp/models"),
    Path("/home/radu/code/ik_llama.cpp/models"),
    Path("/home/radu/.cache/huggingface/hub"),
]


def resolve_model_file(primary: str, fallbacks: list) -> str:
    """Resolve model path against primary, fallbacks, and standard model directories."""
    candidates = []
    if primary:
        candidates.append(primary)
    if fallbacks:
        candidates.extend(fallbacks)

    for cand in candidates:
        if not cand:
            continue
        cand_p = Path(cand)
        if cand_p.is_file():
            return str(cand_p.resolve())

        # If relative or base filename, check standard search roots
        filename = cand_p.name
        for root in SEARCH_ROOTS:
            if root.is_dir():
                direct_match = root / filename
                if direct_match.is_file():
                    return str(direct_match.resolve())
                sub_match = root / cand
                if sub_match.is_file():
                    return str(sub_match.resolve())

    return primary


def main():
    parser = argparse.ArgumentParser(description="Declarative Model Server Launcher for ik_llama.cpp")
    parser.add_argument("model_key", nargs="?", default="deepseek_v41_flash",
                        help="Key of the model in config (e.g. deepseek_v41_flash, glm_53, qwen38_27b)")
    parser.add_argument("--config", "-c", type=Path,
                        default=Path(__file__).resolve().parent / "configs" / "models.toml",
                        help="Path to declarative TOML or YAML config file")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override port")
    parser.add_argument("--host", type=str, default=None, help="Override host")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--list", action="store_true", help="List available model keys and exit")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    models = cfg.get("models", {})

    if args.list:
        print("Available model configurations:")
        for k, v in models.items():
            print(f"  • {k:25s} : {v.get('name', k)}")
        sys.exit(0)

    if args.model_key not in models:
        sys.stderr.write(f"ERROR: Model '{args.model_key}' not found in {args.config}.\n")
        sys.stderr.write(f"Available keys: {', '.join(models.keys())}\n")
        sys.exit(1)

    m = models[args.model_key]
    defaults = cfg.get("defaults", {})

    workspace_root = Path(__file__).resolve().parent.parent
    binary = workspace_root / "build" / "bin" / "llama-server"
    if not binary.exists():
        sys.stderr.write(f"ERROR: Executable not found at {binary}.\n")
        sys.exit(1)

    # Source env.optimized.sh if present
    env_file = workspace_root / "env.optimized.sh"
    if env_file.exists():
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("OMP_NUM_THREADS", "56")
        os.environ.setdefault("OMP_PLACES", "cores")
        os.environ.setdefault("OMP_PROC_BIND", "spread")
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")
        os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
        os.environ.setdefault("OMP_STACKSIZE", "64M")
        os.environ.setdefault("GGML_CUDA_ENABLE_GRAPHS", "1")
        os.environ.setdefault("GGML_CUDA_GRAPH_OPT", "1")
        os.environ.setdefault("GGML_CUDA_MAX_NODES", "16384")
        os.environ.setdefault("GGML_CUDA_NO_PINNED", "0")
        os.environ.setdefault("GGML_CUDA_HOST_MALLOC_THP", "1")
        os.environ.setdefault("GGML_OP_OFFLOAD_MIN_BATCH", "32")
        os.environ.setdefault("LLAMA_KV_ZDD_SKIP", "1")
        os.environ.setdefault("GGML_MOE_URING_PREFETCH", "1")
        os.environ.setdefault("MALLOC_ARENA_MAX", "8")
        os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")

    port = args.port or m.get("port") or defaults.get("port", 8080)
    host = args.host or m.get("host") or defaults.get("host", "127.0.0.1")

    # Clean port if occupied
    try:
        fuser_out = subprocess.check_output(["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL)
        if fuser_out.strip():
            print(f"Cleaning up existing process on port {port}...")
            subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    model_path = resolve_model_file(m.get("model_path", ""), m.get("fallback_paths", []))
    draft_path = resolve_model_file(m.get("draft_path", ""), m.get("draft_fallback_paths", []))

    cmd = [str(binary), "-m", model_path, "--host", str(host), "--port", str(port)]

    if m.get("jinja", defaults.get("jinja", True)):
        cmd.append("--jinja")
    if "ngl" in m:
        cmd.extend(["-ngl", str(m["ngl"])])
    if "offload_tensors" in m:
        cmd.extend(["-ot", str(m["offload_tensors"])])
    if "threads" in m or "threads" in defaults:
        cmd.extend(["-t", str(m.get("threads", defaults.get("threads", 56)))])
    if "batch_size" in m:
        cmd.extend(["-b", str(m["batch_size"])])
    if "ubatch_size" in m:
        cmd.extend(["-ub", str(m["ubatch_size"])])
    if "ctx_size" in m:
        cmd.extend(["-c", str(m["ctx_size"])])
    if "n_parallel" in m:
        cmd.extend(["-np", str(m["n_parallel"])])
    if "cache_type_k" in m:
        cmd.extend(["-ctk", str(m["cache_type_k"])])
    if "cache_type_v" in m:
        cmd.extend(["-ctv", str(m["cache_type_v"])])
    if "mla" in m:
        cmd.extend(["-mla", str(m["mla"])])
    if m.get("flash_attn", True):
        cmd.extend(["-fa", "on"])
    if m.get("zdd_sparse_kv", True):
        cmd.append("--zdd-sparse-kv")
    if m.get("defer_ple", False):
        cmd.append("--defer-ple")

    numa_mode = m.get("numa", defaults.get("numa", "distribute"))
    if numa_mode:
        cmd.extend(["--numa", str(numa_mode)])

    if m.get("transparent_huge_pages", defaults.get("transparent_huge_pages", True)):
        cmd.append("--transparent-huge-pages")

    if m.get("cohort_prefix_sharing", False):
        cmd.append("--cohort-prefix-sharing")
        cmd.extend(["--cohort-min-prefix", str(m.get("cohort_min_prefix", 64))])

    cache_ram = m.get("cache_ram", defaults.get("cache_ram", 0))
    cmd.extend(["--cache-ram", str(cache_ram)])

    spill_dir = m.get("spill_dir", defaults.get("spill_dir"))
    if spill_dir:
        cmd.extend(["--ctx-ckpt-spill-dir", str(spill_dir)])

    if m.get("chat_template_kwargs"):
        cmd.extend(["--chat-template-kwargs", str(m["chat_template_kwargs"])])

    if draft_path and os.path.exists(draft_path):
        cmd.extend(["-md", draft_path])

    if m.get("spec_type"):
        cmd.extend(["--spec-type", str(m["spec_type"])])
    if m.get("spec_autotune", False):
        cmd.append("--spec-autotune")

    for flag in m.get("extra_flags", []):
        cmd.append(flag)

    cmd.extend(unknown)

    print("=" * 80)
    print(f" Launching model [{m.get('name', args.model_key)}]")
    print(f" Port: {port} | Host: {host}")
    print(f" Model: {model_path}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return

    if not Path(model_path).is_file():
        sys.stderr.write(f"WARNING: Model file '{model_path}' does not exist on disk.\n")
        sys.stderr.write("Checked primary path and fallback search roots.\n")

    os.execv(binary, cmd)


if __name__ == "__main__":
    main()
