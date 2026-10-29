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


def _get_search_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    env_root = os.environ.get("LLAMA_MODEL_ROOT")
    if env_root:
        roots.append(Path(env_root))
        roots.append(Path(env_root) / "models")
    roots.extend([
        home / "vm" / "llm",
        home / "vm" / "llm" / "models",
        home / "code" / "llama.cpp" / "models",
        home / "code" / "ik_llama.cpp" / "models",
        home / ".cache" / "huggingface" / "hub",
    ])
    return roots


SEARCH_ROOTS = _get_search_roots()


def resolve_model_file(primary: str, fallbacks: list) -> str:
    """Resolve model path against primary, fallbacks, and standard model directories."""
    candidates = []
    if primary:
        candidates.append(primary)
    if fallbacks:
        candidates.extend(fallbacks)

    search_roots = _get_search_roots()
    for cand in candidates:
        if not cand:
            continue
        cand_p = Path(cand)
        if cand_p.is_file():
            return str(cand_p.resolve())

        # If relative or base filename, check standard search roots
        filename = cand_p.name
        stripped = cand[len("models/"):] if cand.startswith("models/") else cand
        for root in search_roots:
            if root.is_dir():
                for rel in (cand, stripped, filename):
                    match = root / rel
                    if match.is_file():
                        return str(match.resolve())

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
    bin_rel = m.get("binary", defaults.get("binary", "build-pto/bin/llama-server"))
    binary = workspace_root / bin_rel
    if not binary.exists():
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
        # NOTE: no LLAMA_KV_ZDD_SKIP force-set here — that env OR-enables the
        # zdd-sparse-kv mask path regardless of --zdd-sparse-kv/--no-zdd-sparse-kv,
        # making the flag's off-state unreachable under this launcher. The
        # per-profile `zdd_sparse_kv` config key drives the real flag instead.
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

    raw_model_path = m.get("model_path") or m.get("path", "")
    raw_draft_path = m.get("draft_path") or m.get("draft_model", "")
    model_path = resolve_model_file(raw_model_path, m.get("fallback_paths", []))
    draft_path = resolve_model_file(raw_draft_path, m.get("draft_fallback_paths", []))

    cmd = [str(binary), "-m", model_path, "--host", str(host), "--port", str(port)]

    if m.get("jinja", defaults.get("jinja", True)):
        cmd.append("--jinja")
    ngl_val = m.get("ngl", m.get("gpu_layers", defaults.get("gpu_layers")))
    if ngl_val is not None:
        cmd.extend(["-ngl", str(ngl_val)])
    ot_val = m.get("offload_tensors", m.get("override_tensor"))
    if ot_val:
        cmd.extend(["-ot", str(ot_val)])
    if "threads" in m or "threads" in defaults:
        cmd.extend(["-t", str(m.get("threads", defaults.get("threads", 56)))])
    tb_val = m.get("threads_batch", defaults.get("threads_batch"))
    if tb_val:
        cmd.extend(["-tb", str(tb_val)])
    if "batch_size" in m:
        cmd.extend(["-b", str(m["batch_size"])])
    if "ubatch_size" in m:
        cmd.extend(["-ub", str(m["ubatch_size"])])
    if "ctx_size" in m:
        cmd.extend(["-c", str(m["ctx_size"])])
    np_val = m.get("n_parallel", m.get("parallel"))
    if np_val is not None:
        cmd.extend(["-np", str(np_val)])
    ctk_val = m.get("cache_type_k", m.get("ctk", defaults.get("ctk")))
    if ctk_val:
        cmd.extend(["-ctk", str(ctk_val)])
    ctv_val = m.get("cache_type_v", m.get("ctv", defaults.get("ctv")))
    if ctv_val:
        cmd.extend(["-ctv", str(ctv_val)])
    ictk_val = m.get("indexer_cache_type_k", m.get("ictk", defaults.get("ictk")))
    if ictk_val:
        cmd.extend(["-ictk", str(ictk_val)])
    if "mla" in m:
        cmd.extend(["-mla", str(m["mla"])])
    fa_val = m.get("flash_attn", defaults.get("flash_attn", True))
    if fa_val:
        cmd.extend(["-fa", "on" if fa_val is True else str(fa_val)])
    if m.get("swa_compress", defaults.get("swa_compress", False)):
        cmd.append("--swa-compress")
    if m.get("fidx", m.get("fused_indexer", False)):
        cmd.append("-fidx")
    if m.get("dsa", m.get("dsa_attention", False)):
        cmd.append("-dsa")
    if "dsatk" in m:
        cmd.extend(["-dsatk", str(m["dsatk"])])
    if m.get("zdd_sparse_kv", defaults.get("zdd_sparse_kv", True)):
        cmd.append("--zdd-sparse-kv")
    if m.get("defer_ple", defaults.get("defer_ple", False)):
        cmd.append("--defer-ple")

    numa_mode = m.get("numa", defaults.get("numa", "distribute"))
    if numa_mode:
        cmd.extend(["--numa", str(numa_mode)])

    if m.get("transparent_huge_pages", defaults.get("transparent_huge_pages", True)):
        cmd.append("--transparent-huge-pages")

    if m.get("cohort_prefix_sharing", defaults.get("cohort_prefix_sharing", False)):
        cmd.append("--cohort-prefix-sharing")
        cmd.extend(["--cohort-min-prefix", str(m.get("cohort_min_prefix", defaults.get("cohort_min_prefix", 64)))])
        if m.get("cohort_max_idle_sec", defaults.get("cohort_max_idle_sec")):
            cmd.extend(["--cohort-max-idle-sec", str(m.get("cohort_max_idle_sec", defaults.get("cohort_max_idle_sec")))])

    cache_ram = m.get("cache_ram", defaults.get("cache_ram", 0))
    cmd.extend(["--cache-ram", str(cache_ram)])

    spill_dir = m.get("spill_dir", defaults.get("spill_dir"))
    if spill_dir:
        cmd.extend(["--ctx-ckpt-spill-dir", str(spill_dir)])

    if m.get("chat_template_kwargs"):
        cmd.extend(["--chat-template-kwargs", str(m["chat_template_kwargs"])])

    if draft_path and (os.path.exists(draft_path) or args.dry_run):
        cmd.extend(["-md", draft_path])
        if "ngld" in m:
            cmd.extend(["-ngld", str(m["ngld"])])

    spec_type_val = m.get("spec_type")
    if spec_type_val:
        if isinstance(spec_type_val, list):
            for st in spec_type_val:
                # NOTE: '--spec-stage' is a disabled legacy alias (throws at
                # parse); the only valid form is repeated '--spec-type'.
                cmd.extend(["--spec-type", str(st)])
        else:
            cmd.extend(["--spec-type", str(spec_type_val)])
    if m.get("spec_ckpt_mode"):
        cmd.extend(["--spec-ckpt-mode", str(m["spec_ckpt_mode"])])
    if m.get("spec_autotune", False):
        cmd.append("--spec-autotune")

    for flag in m.get("extra_flags", []):
        cmd.append(flag)

    # Evidence / reasoning wiring (2026-10-28): the [defaults] keys below are now
    # honored — previously parsed-by-nobody dead config.
    if m.get("evidence_filter", defaults.get("evidence_filter", False)):
        cmd.append("--evidence-filter")
    ev_strategy = m.get("evidence_strategy", defaults.get("evidence_strategy"))
    if ev_strategy:
        cmd.extend(["--evidence-strategy", str(ev_strategy)])
    ev_chunk = m.get("evidence_chunk_tokens", defaults.get("evidence_chunk_tokens"))
    if ev_chunk:
        cmd.extend(["-ect", str(ev_chunk)])
    effort = m.get("thinking_effort", defaults.get("thinking_effort"))
    if effort:
        # same effort -> --reasoning-budget mapping as the capsule TS generator
        budget_by_effort = {"max_deep_think": "-1", "high_analytical": "16384", "balanced": "4096", "fast_direct": "0"}
        budget = budget_by_effort.get(str(effort), "-1")
        cmd.extend(["--reasoning-budget", budget])

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
