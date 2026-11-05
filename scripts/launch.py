#!/usr/bin/env python3
"""
scripts/launch.py - Declarative Unified Model Launcher for ik_llama.cpp
Reads configuration from scripts/configs/models.toml and 5-layer hierarchy files,
supports profiles, presets, ctx-capsule integration, and exact hardware tuning.
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
        return {}
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            return {}
        return yaml.safe_load(text) or {}
    else:
        if tomllib is None:
            return {}
        return tomllib.loads(text) or {}


def load_hierarchical_config(config_path: Path) -> dict:
    cfg = load_config(config_path)
    hierarchy_dir = config_path.parent / "hierarchy"
    if hierarchy_dir.is_dir():
        hw_dir = hierarchy_dir / "hardware"
        if hw_dir.is_dir():
            for f in sorted(hw_dir.glob("*.toml")):
                hw_cfg = load_config(f)
                hw_defs = hw_cfg.get("hardware", {}).get("defaults", {})
                if hw_defs:
                    cfg.setdefault("defaults", {})
                    for hk, hv in hw_defs.items():
                        cfg["defaults"].setdefault(hk, hv)

        defaults_path = hierarchy_dir / "00-defaults.toml"
        if defaults_path.exists():
            defaults_cfg = load_config(defaults_path)
            if "defaults" in defaults_cfg:
                cfg.setdefault("defaults", {}).update(defaults_cfg["defaults"])

        models_dir = hierarchy_dir / "models"
        if models_dir.is_dir():
            cfg.setdefault("models", {})
            for f in sorted(models_dir.glob("*.toml")):
                m_cfg = load_config(f)
                if "models" in m_cfg:
                    for mk, mv in m_cfg["models"].items():
                        rec = mv.get("recommended", {})
                        merged_m = dict(rec)
                        merged_m.update({k: v for k, v in mv.items() if k != "recommended"})
                        merged_m.setdefault("name", merged_m.get("display_name") or merged_m.get("description") or mk)
                        if mk in cfg["models"]:
                            for rk, rv in merged_m.items():
                                cfg["models"][mk].setdefault(rk, rv)
                        else:
                            cfg["models"][mk] = merged_m
                elif f.stem not in cfg["models"]:
                    cfg["models"][f.stem] = m_cfg

        templates_dir = hierarchy_dir / "templates"
        if templates_dir.is_dir():
            cfg.setdefault("profiles", {})
            for f in sorted(templates_dir.glob("*.toml")):
                t_cfg = load_config(f)
                if "templates" in t_cfg:
                    for tk, tv in t_cfg["templates"].items():
                        cfg["profiles"].setdefault(tk, tv)
                elif "profiles" in t_cfg:
                    cfg["profiles"].update(t_cfg["profiles"])
                else:
                    cfg["profiles"].setdefault(f.stem, t_cfg)

        fleet_path = hierarchy_dir / "instances" / "active_fleet.toml"
        if fleet_path.exists():
            fleet_cfg = load_config(fleet_path)
            if "fleet" in fleet_cfg:
                cfg["fleet"] = fleet_cfg["fleet"]

    return cfg


def normalize_val(val):
    if isinstance(val, str):
        low = val.lower()
        if low in ("on", "true", "1"):
            return True
        if low in ("off", "false", "0"):
            return False
    return val


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


def resolve_model_file(primary: str, fallbacks: list) -> str:
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
    parser.add_argument("model_key", nargs="?", default=None,
                        help="Key of the model or profile in config")
    parser.add_argument("--profile", "-P", type=str, default=None,
                        help="Profile/preset key to merge on top of model")
    parser.add_argument("--config", "-c", type=Path,
                        default=Path(__file__).resolve().parent / "configs" / "models.toml",
                        help="Path to declarative TOML config file")
    parser.add_argument("--port", "-p", type=int, default=None, help="Override port")
    parser.add_argument("--host", type=str, default=None, help="Override host")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--list", action="store_true", help="List available model keys and profiles and exit")
    parser.add_argument("--capsule", "--tui", action="store_true", help="Launch interactive ctx-capsule TUI in live mode")
    parser.add_argument("--export-toml", action="store_true", help="Export hierarchical TOML bundle")
    parser.add_argument("--telemetry", action="store_true", help="Show capsule telemetry")

    args, unknown = parser.parse_known_args()

    workspace_root = Path(__file__).resolve().parent.parent
    tui_path = workspace_root / "tools" / "ctx-capsule" / "ctx_capsule_tui.ts"

    # Bare invocation in an interactive terminal opens the live @earendil-works/pi-tui launcher
    if args.capsule or (len(sys.argv) == 1 and sys.stdin.isatty() and sys.stdout.isatty()):
        mode_flag = "--dry-run" if args.dry_run else "--live"
        if args.dry_run and args.capsule:
            os.execvp("node", ["node", "--experimental-strip-types", str(tui_path), "--render-once", "--dry-run"])
        os.execvp("node", ["node", "--experimental-strip-types", str(tui_path), "tui", mode_flag])
    if args.export_toml:
        os.execvp("node", ["node", "--experimental-strip-types", str(tui_path), "--export-toml"])
    if args.telemetry:
        os.execvp("node", ["node", "--experimental-strip-types", str(tui_path), "telemetry"])

    cfg = load_hierarchical_config(args.config)
    models = cfg.get("models", {})
    profiles = cfg.get("profiles", {})

    if args.list:
        print("Available model configurations ([models.*]):")
        for k, v in models.items():
            print(f"  • {k:35s} : {v.get('name', v.get('description', k))}")
        print("\nAvailable profile / preset configurations ([profiles.*]):")
        for k, v in profiles.items():
            print(f"  • {k:35s} : {v.get('display_name', v.get('description', k))}")
        sys.exit(0)

    target_key = args.model_key or cfg.get("defaults", {}).get("active_model", "deepseek_v41_flash")

    m = {}
    profile_cfg = {}
    base_model = "deepseek-v4.1-1017-ud-q4kxl"

    if target_key in profiles:
        profile_cfg = profiles[target_key]
        base_model = profile_cfg.get("base_model") or profile_cfg.get("model_id") or "deepseek-v4.1-1017-ud-q4kxl"
        if base_model in models:
            m = dict(models[base_model])
        else:
            matching = [mk for mk in models.keys() if base_model in mk or mk in base_model]
            if matching:
                m = dict(models[matching[0]])
            elif models:
                m = dict(models[list(models.keys())[0]])
    elif target_key in models:
        m = dict(models[target_key])
        base_model = target_key
    else:
        matching = [mk for mk in models.keys() if target_key in mk or mk in target_key]
        if matching:
            m = dict(models[matching[0]])
            base_model = matching[0]
        elif profiles:
            matching_prof = [pk for pk in profiles.keys() if target_key in pk or pk in target_key]
            if matching_prof:
                profile_cfg = profiles[matching_prof[0]]
                base_model = profile_cfg.get("base_model", "deepseek-v4.1-1017-ud-q4kxl")
                m = dict(models.get(base_model, list(models.values())[0] if models else {}))
        if not m:
            sys.stderr.write(f"ERROR: Model or Profile '{target_key}' not found in {args.config}.\n")
            sys.stderr.write(f"Available keys: {', '.join(list(models.keys()) + list(profiles.keys()))}\n")
            sys.exit(1)

    if args.profile:
        if args.profile in profiles:
            profile_cfg = profiles[args.profile]
        else:
            sys.stderr.write(f"ERROR: Profile '{args.profile}' not found in {args.config}.\n")
            sys.exit(1)

    if profile_cfg:
        overrides = profile_cfg.get("overrides", profile_cfg)
        for k, v in overrides.items():
            if k not in ("display_name", "description", "target_use_case", "built_in", "base_model"):
                m[k] = v

    # Normalize capsule keys
    if "kv_type_k" in m:
        m["ctk"] = m["kv_type_k"]
    if "kv_type_v" in m:
        m["ctv"] = m["kv_type_v"]
    if "parallel_slots" in m:
        m["parallel"] = m["parallel_slots"]
    if "mla_mode" in m:
        m["mla"] = m["mla_mode"]
    if "numa_strategy" in m:
        m["numa"] = m["numa_strategy"]
    if "thp_hugepages" in m:
        m["transparent_huge_pages"] = normalize_val(m["thp_hugepages"])
    if "cohort_sharing" in m:
        m["cohort_prefix_sharing"] = normalize_val(m["cohort_sharing"])
    if "engram_resident" in m:
        m["engram_resident"] = normalize_val(m["engram_resident"])
    if "spec_verify_pinned" in m:
        m["spec_verify_pinned"] = normalize_val(m["spec_verify_pinned"])
    if "defer_experts" in m:
        m["defer_experts"] = normalize_val(m["defer_experts"])

    if m.get("spec_type") == "ngram-mod_plus_dspark":
        m["spec_type"] = ["ngram-mod:n_max=6,n_min=2", "dspark:n_max=6,n_min=1"]

    defaults = cfg.get("defaults", {})
    bin_rel = m.get("binary", defaults.get("binary", "build-pto/bin/llama-server"))
    binary = workspace_root / bin_rel
    if not binary.exists():
        binary = workspace_root / "build" / "bin" / "llama-server"
    if not binary.exists() and not args.dry_run:
        sys.stderr.write(f"ERROR: Executable not found at {binary}.\n")
        sys.exit(1)

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
    os.environ.setdefault("GGML_MOE_URING_PREFETCH", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "8")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")

    port = args.port or m.get("port") or defaults.get("port", 8080)
    host = args.host or m.get("host") or defaults.get("host", "127.0.0.1")

    if not args.dry_run:
        try:
            fuser_out = subprocess.check_output(["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL)
            if fuser_out.strip():
                print(f"Cleaning up existing process on port {port}...")
                subprocess.run(["fuser", "-k", "-9", f"{port}/tcp"], stderr=subprocess.DEVNULL)
        except Exception:
            pass

    raw_model_path = m.get("model_path") or m.get("path", "")
    raw_draft_path = m.get("draft_path") or m.get("draft_model", "")
    raw_engram_path = m.get("engram_path") or m.get("engram", "")
    model_path = resolve_model_file(raw_model_path, m.get("fallback_paths", []))
    draft_path = resolve_model_file(raw_draft_path, m.get("draft_fallback_paths", []))
    engram_path = resolve_model_file(raw_engram_path, m.get("engram_fallback_paths", [])) if raw_engram_path else ""

    cmd = [str(binary), "-m", model_path, "--host", str(host), "--port", str(port)] if binary.name else ["llama-server", "-m", model_path, "--host", str(host), "--port", str(port)]

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
    if normalize_val(m.get("swa_compress", defaults.get("swa_compress", False))):
        cmd.append("--swa-compress")
    if normalize_val(m.get("fidx", m.get("fused_indexer", False))):
        cmd.append("-fidx")
    if normalize_val(m.get("dsa", m.get("dsa_attention", False))):
        cmd.append("-dsa")
    if "dsatk" in m:
        cmd.extend(["-dsatk", str(m["dsatk"])])
    if normalize_val(m.get("zdd_sparse_kv", defaults.get("zdd_sparse_kv", True))):
        cmd.append("--zdd-sparse-kv")
    if normalize_val(m.get("defer_ple", defaults.get("defer_ple", False))):
        cmd.append("--defer-ple")
    if normalize_val(m.get("defer_experts", defaults.get("defer_experts", False))):
        cmd.append("--defer-experts")
    if normalize_val(m.get("engram_resident", defaults.get("engram_resident", False))):
        cmd.append("--engram-resident")
    if engram_path and (os.path.exists(engram_path) or args.dry_run):
        cmd.extend(["--engram", engram_path])

    numa_mode = m.get("numa", defaults.get("numa", "distribute"))
    if numa_mode:
        cmd.extend(["--numa", str(numa_mode)])

    if normalize_val(m.get("transparent_huge_pages", defaults.get("transparent_huge_pages", True))):
        cmd.append("--transparent-huge-pages")

    if normalize_val(m.get("cohort_prefix_sharing", defaults.get("cohort_prefix_sharing", False))):
        cmd.append("--cohort-prefix-sharing")
        cmd.extend(["--cohort-min-prefix", str(m.get("cohort_min_prefix", defaults.get("cohort_min_prefix", 64)))])

    cache_ram = m.get("cache_ram", defaults.get("cache_ram", 0))
    cmd.extend(["--cache-ram", str(cache_ram)])

    spill_dir = m.get("spill_dir", defaults.get("spill_dir"))
    if spill_dir:
        cmd.extend(["--ctx-ckpt-spill-dir", str(spill_dir)])

    if draft_path and (os.path.exists(draft_path) or args.dry_run):
        cmd.extend(["-md", draft_path])
        if "ngld" in m:
            cmd.extend(["-ngld", str(m["ngld"])])

    spec_type_val = m.get("spec_type")
    if spec_type_val:
        if isinstance(spec_type_val, list):
            for st in spec_type_val:
                cmd.extend(["--spec-type", str(st)])
        else:
            cmd.extend(["--spec-type", str(spec_type_val)])
    if m.get("spec_ckpt_mode"):
        cmd.extend(["--spec-ckpt-mode", str(m["spec_ckpt_mode"])])
    if normalize_val(m.get("spec_autotune", False)):
        cmd.append("--spec-autotune")
    if normalize_val(m.get("spec_verify_pinned", defaults.get("spec_verify_pinned", False))):
        cmd.append("--spec-verify-pinned")

    for flag in m.get("extra_flags", []):
        cmd.append(flag)

    if normalize_val(m.get("evidence_filter", defaults.get("evidence_filter", False))):
        cmd.append("--evidence-filter")
    ev_strategy = m.get("evidence_strategy", defaults.get("evidence_strategy"))
    if ev_strategy:
        cmd.extend(["--evidence-strategy", str(ev_strategy)])
    ev_chunk = m.get("evidence_chunk_tokens", defaults.get("evidence_chunk_tokens"))
    if ev_chunk:
        cmd.extend(["-ect", str(ev_chunk)])
    effort = m.get("thinking_effort", defaults.get("thinking_effort"))
    if effort:
        budget_by_effort = {"max_deep_think": "-1", "high_analytical": "16384", "balanced": "4096", "fast_direct": "0"}
        budget = budget_by_effort.get(str(effort), "-1")
        cmd.extend(["--reasoning-budget", budget])

    cmd.extend(unknown)

    print("=" * 80)
    print(f" Launching target [{target_key}] (Model: {m.get('name', base_model)})")
    print(f" Port: {port} | Host: {host}")
    print(f" Model Path: {model_path}")
    print("=" * 80)

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(cmd))
        return

    os.execv(binary, cmd)


if __name__ == "__main__":
    main()
