#!/usr/bin/env python3
"""
tests/test_ctx_capsule_tui.py - Comprehensive Automated Unit & UI Coverage Test Suite
Tests tools/ctx-capsule/ctx_capsule_tui.py (all 5 tabs, 34 settings, geometry invariants,
keyboard navigation, presets, CLI/HTTP builders, multi-agent exports) and common/kv-capsule-filter.h.
"""

import curses
import importlib.util
import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TUI_PATH = REPO_ROOT / "tools" / "ctx-capsule" / "ctx_capsule_tui.py"
KV_CAPSULE_HEADER = REPO_ROOT / "common" / "kv-capsule-filter.h"


def load_tui_module():
    spec = importlib.util.spec_from_file_location("ctx_capsule_tui", TUI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ("W", "F") else 1
    return width


class FakeStdscr:
    """Mock curses window to test run_curses interactive loop with 100% coverage."""

    def __init__(self, key_sequence):
        self.keys = list(key_sequence)
        self.rendered_lines = []
        self.clear_count = 0
        self.refresh_count = 0

    def clear(self):
        self.clear_count += 1
        self.rendered_lines = []

    def addstr(self, y, x, text):
        while len(self.rendered_lines) <= y:
            self.rendered_lines.append("")
        self.rendered_lines[y] = text

    def refresh(self):
        self.refresh_count += 1

    def getch(self):
        if not self.keys:
            return ord("q")
        return self.keys.pop(0)


def test_five_tabs_and_all_34_settings_structure():
    tui = load_tui_module()
    assert hasattr(tui, "TABS")
    assert len(tui.TABS) == 5, f"Expected 5 category tabs, got {len(tui.TABS)}"
    expected_slugs = ["evidence", "kv_cache", "speculative", "hardware", "sampling"]
    assert [t["slug"] for t in tui.TABS] == expected_slugs
    assert len(tui.SETTINGS) == 34

    expected_keys = {
        # Tab 1: Evidence & Context Capsules (7)
        "evidence_filter",
        "evidence_strategy",
        "evidence_chunk_tokens",
        "evidence_top_k",
        "evidence_symbol_boost",
        "evidence_chunk_expansion",
        "capsule_filter",
        # Tab 2: KV Cache, ZDD Sparsity & Tiered Paging (8)
        "cache_type_k",
        "cache_type_v",
        "zdd_sparse_kv",
        "mla_attn",
        "cohort_prefix_sharing",
        "cohort_min_prefix",
        "cache_ram",
        "universal_slab_arena",
        # Tab 3: Speculative Decoding & TreeGraft Arbiter (5)
        "spec_type",
        "spec_autotune",
        "draft_max",
        "draft_min",
        "het_split_drafting",
        # Tab 4: Hardware, NUMA, CUDA Graphs & Direct I/O (7)
        "numa_strategy",
        "numa_replicate_shared",
        "cpu_threads",
        "flash_attn",
        "transparent_huge_pages",
        "defer_io_uring",
        "cuda_l2_persistence",
        # Tab 5: Sampling, Reasoning & Agent Harnesses (7)
        "temp",
        "top_k",
        "top_p",
        "min_p",
        "reasoning_effort",
        "grammar_json",
        "agent_harness",
    }
    actual_keys = {s["key"] for s in tui.SETTINGS}
    assert actual_keys == expected_keys

    for s in tui.SETTINGS:
        assert len(s["values"]) >= 2
        assert 0 <= s["idx"] < len(s["values"])
        assert len(s["ascii"]) >= 3
        assert s["desc"] and s["pros"] and s["cons"] and s["impact"]
        assert "badges" in s and len(s["badges"]) == len(s["values"])


def test_box_drawing_geometry_and_zero_clipping_across_all_tabs_and_values():
    tui = load_tui_module()

    for tab_idx, tab in enumerate(tui.TABS):
        for s_idx, setting in enumerate(tab["settings"]):
            for val_idx, val in enumerate(setting["values"]):
                setting["idx"] = val_idx
                snap = tui.render_ansi_snapshot(s_idx, tab_idx=tab_idx)
                lines = snap.splitlines()
                assert len(lines) >= 18

                expected_w = display_width(lines[0])
                assert expected_w == 136
                assert lines[0].startswith("╔") and lines[0].endswith("╗")
                assert lines[-1].startswith("╚") and lines[-1].endswith("╝")

                for line_no, line in enumerate(lines):
                    w = display_width(line)
                    assert w == expected_w, (
                        f"Line {line_no} width mismatch ({w} != {expected_w}) "
                        f"on tab={tab_idx} setting={setting['key']} val={val}: {line!r}"
                    )
                    assert line[0] in ("╔", "║", "╠", "╚"), f"Unclosed left border on line {line_no}: {line!r}"
                    assert line[-1] in ("╗", "║", "╣", "╝"), f"Unclosed right border on line {line_no}: {line!r}"

                # Active tab pill indicator must be highlighted
                assert f"▶[{tab['short_title']}]◀" in snap

                # Active badge for this specific value must appear in right column header
                expected_badge = tui.get_active_badge(setting)
                assert expected_badge in snap

                # Every setting value in left column must have closing ' >' (never clipped mid-value!)
                val_str = str(val)
                expected_val_display = "all (5 agents)" if val_str.startswith("all (") else val_str
                assert f"< {expected_val_display} >" in snap, (
                    f"Clipped value in left column for {setting['key']}={val_str}"
                )

                # Every ASCII art line must appear intact in the right column
                for art_line in setting["ascii"]:
                    stripped = art_line.strip()
                    if stripped:
                        assert stripped in snap, f"ASCII art line {stripped!r} missing/clipped for {setting['key']}"


def test_keyboard_navigation_all_tabs_settings_and_presets():
    tui = load_tui_module()
    state = tui.TuiState(tab_idx=0, selected_idx=0)

    # Number hotkeys '1'..'5' switch directly to tabs 0..4
    for tab_num in range(1, 6):
        assert state.handle_key(ord(str(tab_num))) is True
        assert state.tab_idx == tab_num - 1

    # TAB and Shift-TAB cycle across all 5 tabs
    state.tab_idx = 0
    for expected_tab in [1, 2, 3, 4, 0]:
        assert state.handle_key(9) is True
        assert state.tab_idx == expected_tab

    btab = getattr(curses, "KEY_BTAB", 353)
    for expected_tab in [4, 3, 2, 1, 0]:
        assert state.handle_key(btab) is True
        assert state.tab_idx == expected_tab

    # Navigate UP/DOWN and cycle LEFT/RIGHT across every setting in every tab
    for t_idx, tab in enumerate(tui.TABS):
        state.tab_idx = t_idx
        state.selected_idx = 0
        n_settings = len(tab["settings"])
        for step in range(n_settings):
            assert state.selected_idx == step
            s = state.active_setting
            orig_idx = s["idx"]
            # KEY_RIGHT advances value index
            state.handle_key(curses.KEY_RIGHT)
            assert s["idx"] == (orig_idx + 1) % len(s["values"])
            # KEY_LEFT restores value index
            state.handle_key(curses.KEY_LEFT)
            assert s["idx"] == orig_idx
            # KEY_DOWN moves to next setting (wrapping around at end)
            state.handle_key(curses.KEY_DOWN)
        assert state.selected_idx == 0
        state.handle_key(curses.KEY_UP)
        assert state.selected_idx == n_settings - 1

    # Preset cycling ('p') and reset ('r')
    state.handle_key(ord("p"))  # low_vram
    assert tui.get_setting_by_key("cache_type_k")["values"][tui.get_setting_by_key("cache_type_k")["idx"]] == "q4_0"
    state.handle_key(ord("p"))  # max_throughput
    assert tui.get_setting_by_key("draft_max")["values"][tui.get_setting_by_key("draft_max")["idx"]] == 16
    state.handle_key(ord("p"))  # deep_audit
    assert (
        tui.get_setting_by_key("evidence_strategy")["values"][tui.get_setting_by_key("evidence_strategy")["idx"]]
        == "analytic_aggregate"
    )
    state.handle_key(ord("r"))  # reset_defaults
    assert tui.get_setting_by_key("cache_type_k")["values"][tui.get_setting_by_key("cache_type_k")["idx"]] == "q8_0"

    # Export key ('e') and Quit key ('q')
    with tempfile.TemporaryDirectory() as tmpdir:
        assert state.handle_key(ord("e"), export_dir=Path(tmpdir)) is True
        assert state.last_exported_path is not None and state.last_exported_path.exists()
    assert state.handle_key(ord("q")) is False


def test_cli_flags_and_http_payload_combinations():
    tui = load_tui_module()
    tui.reset_defaults()

    flags = tui.build_cli_flags()
    for expected_flag in [
        "--evidence-filter",
        "--evidence-strategy code_pinpoint",
        "--evidence-chunk-tokens 140",
        "--evidence-top-k 10",
        "--evidence-symbol-boost 3.5",
        "--evidence-chunk-expansion 1",
        "-ctk q8_0",
        "-ctv q8_0",
        "--zdd-sparse-kv",
        "-mla 3",
        "--cohort-prefix-sharing",
        "--cohort-min-prefix 64",
        "--cache-ram 0",
        "--spec-type ngram-map-k4v:n_max=8,n_min=2",
        "--spec-type mtp:n_max=8,p_min=0.0",
        "--spec-autotune",
        "--numa expert",
        "--numa-replicate-shared",
        "-t 16 -tb 32",
        "-fa on",
        "--transparent-huge-pages",
        "--defer-experts --defer-ple",
        "--temp 0.2",
        "--top-k 40",
        "--top-p 0.95",
        "--min-p 0.05",
        "--reasoning on",
    ]:
        assert expected_flag in flags, f"Missing {expected_flag!r} in {flags!r}"

    # Verify zero legacy --unreal flags
    assert "unreal" not in flags.lower()

    # Test toggling settings off / alternative branches
    tui.get_setting_by_key("evidence_filter")["idx"] = 1  # OFF
    tui.get_setting_by_key("zdd_sparse_kv")["idx"] = 1  # OFF
    tui.get_setting_by_key("cohort_prefix_sharing")["idx"] = 1  # OFF
    tui.get_setting_by_key("spec_type")["idx"] = 3  # dflash
    tui.get_setting_by_key("defer_io_uring")["idx"] = 1  # defer-experts
    tui.get_setting_by_key("reasoning_effort")["idx"] = 4  # off
    alt_flags = tui.build_cli_flags()
    assert "--no-evidence-filter" in alt_flags
    assert "--no-zdd-sparse-kv" in alt_flags
    assert "--no-cohort-prefix-sharing" in alt_flags
    assert "--spec-type dflash:n_max=8" in alt_flags
    assert "--defer-experts" in alt_flags and "--defer-ple" not in alt_flags
    assert "--reasoning off" in alt_flags

    headers, payload = tui.build_http_headers_and_payload()
    assert headers["X-Evidence-Filter"] == "OFF"
    assert payload["evidence_filter"] is False
    assert payload["evidence_capsule"]["enabled"] is False
    assert payload["kv_paging"]["zdd_sparse_kv"] is False
    assert payload["speculative"]["spec_type"] == "dflash"

    # Test DeepSeek-V4.1-Flash-1017 presets and DSpark + ngram-mod flags
    for ds_preset in [
        "ds41-1017-np1-512k-max-speed",
        "ds41-1017-np1-1m-extreme",
        "ds41-1017-np4-512k-swarm",
        "ds41-1017-np8-1m-swarm",
    ]:
        tui.apply_preset(ds_preset)
        ds_flags = tui.build_cli_flags()
        assert "--spec-type ngram-mod:n_max=6,n_min=2" in ds_flags
        # Two-stage chains require a self-spec stage first (ngram-mod); the
        # dflash-family dspark neural stage is a valid second stage and the
        # -md drafter supplies its weights.
        assert "--spec-type dspark:n_max=6,n_min=1" in ds_flags
        assert "--spec-ckpt-mode per-step" in ds_flags
        assert "-mla 2" in ds_flags
        assert "--evidence-top-k 16" in ds_flags
        assert "--defer-ple" in ds_flags and "--defer-experts" not in ds_flags
        if ds_preset == "ds41-1017-np1-512k-max-speed":
            assert "-ctk q8_0" in ds_flags and "-ctv q8_0" in ds_flags
        else:
            assert "-ctk q4_0" in ds_flags and "-ctv q4_0" in ds_flags
        _, ds_payload = tui.build_http_headers_and_payload()
        assert ds_payload["evidence_filter"] is True
        assert ds_payload["evidence_strategy"] == "code_pinpoint"
        assert ds_payload["evidence_top_k"] == 16
        assert ds_payload["evidence_capsule"]["enabled"] is True

    tui.reset_defaults()


def test_export_agent_configs_all_and_individual_harnesses():
    tui = load_tui_module()
    tui.reset_defaults()
    harness_setting = tui.get_setting_by_key("agent_harness")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. All harnesses selected
        out_path = tui.export_agent_configs(Path(tmpdir))
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["project"] == "r-irbe/ctx-capsule"
        assert data["version"] == "2.0.0"
        assert set(data["harnesses"].keys()) == {"pi", "opencode", "claude_code", "codex_cli", "github_copilot"}
        assert set(data["categories"].keys()) == {"evidence", "kv_cache", "speculative", "hardware", "sampling"}

        # 2. Individual harness filtering
        expected_map = {
            "pi-fleet": "pi",
            "opencode": "opencode",
            "claude-code": "claude_code",
            "codex-cli": "codex_cli",
            "github-copilot": "github_copilot",
        }
        for val_name, target_key in expected_map.items():
            harness_setting["idx"] = harness_setting["values"].index(val_name)
            out_path = tui.export_agent_configs(Path(tmpdir))
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert list(data["harnesses"].keys()) == [target_key]
            assert len(data["all_harnesses"]) == 5

    tui.reset_defaults()


def test_run_curses_and_cli_entrypoints(monkeypatch):
    tui = load_tui_module()
    monkeypatch.setattr(curses, "curs_set", lambda _: None)

    keys = [
        curses.KEY_DOWN,
        curses.KEY_RIGHT,
        9,  # Tab to Tab 2
        curses.KEY_DOWN,
        curses.KEY_LEFT,
        ord("3"),  # Jump to Tab 3
        ord("p"),  # Apply preset
        ord("r"),  # Reset defaults
        ord("q"),
    ]
    fake_scr = FakeStdscr(keys)
    tui.run_curses(fake_scr)
    assert fake_scr.clear_count == len(keys)
    assert fake_scr.refresh_count == len(keys)

    # Also test CLI subprocess with --render-once, --tab, --select, --preset, --export-dir
    with tempfile.TemporaryDirectory() as tmpdir:
        for tab_num in range(1, 6):
            cmd = [
                sys.executable,
                str(TUI_PATH),
                "--render-once",
                "--tab",
                str(tab_num),
                "--select",
                "1",
                "--preset",
                "default",
                "--export-dir",
                tmpdir,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            assert res.returncode == 0, f"CLI failed on tab {tab_num}: {res.stderr}"
            assert "r-irbe/ctx-capsule" in res.stdout
            assert (Path(tmpdir) / "ctx-capsule-agents.json").exists()


def test_cpp_kv_capsule_filter_binary_fuse_and_simhash():
    """Compile and execute C++ verification harness testing BinaryFuseFilter & ContextCapsule."""
    assert KV_CAPSULE_HEADER.exists(), f"Missing {KV_CAPSULE_HEADER}"

    cpp_code = r"""
#include "kv-capsule-filter.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

int main() {
    using namespace kv_capsule;

    // 1. Empty filter edge case
    {
        BinaryFuseFilter<uint8_t> empty_filter;
        assert(empty_filter.build({}));
        assert(!empty_filter.contains(42ULL));
    }

    // 2. 8-bit and 16-bit BinaryFuseFilter 100% recall + FPR check
    {
        std::vector<uint64_t> keys;
        for (uint64_t i = 1; i <= 2000; ++i) {
            keys.push_back(i * 0x9e3779b97f4a7c15ULL);
        }
        BinaryFuseFilter<uint8_t> f8(keys);
        BinaryFuseFilter<uint16_t> f16;
        assert(f16.build(keys));

        for (uint64_t k : keys) {
            assert(f8.contains(k));
            assert(f16.contains(k));
        }

        int false_positives_8 = 0;
        int false_positives_16 = 0;
        const int trials = 10000;
        for (int i = 0; i < trials; ++i) {
            uint64_t probe = (uint64_t)(i + 500000) * 0xbf58476d1ce4e5b9ULL;
            if (f8.contains(probe)) false_positives_8++;
            if (f16.contains(probe)) false_positives_16++;
        }
        double fpr8 = (double)false_positives_8 / trials;
        double fpr16 = (double)false_positives_16 / trials;
        assert(fpr8 < 0.02);   // Theoretical ~0.39% (1/256)
        assert(fpr16 < 0.002); // Theoretical ~0.0015% (1/65536)
    }

    // 3. SimHash64 cosine similarity properties
    {
        assert(compute_simhash64(nullptr, 0) == 0ULL);
        std::vector<float> v1(64), v2(64), v_neg(64);
        for (int i = 0; i < 64; ++i) {
            float val = std::sin((float)(i + 1) * 1.37f) + 0.5f * std::cos((float)(i + 1) * 0.73f);
            v1[i] = val;
            v2[i] = val;
            v_neg[i] = -val;
        }
        uint64_t h1 = compute_simhash64(v1.data(), 64);
        uint64_t h2 = compute_simhash64(v2.data(), 64);
        uint64_t h_neg = compute_simhash64(v_neg.data(), 64);
        assert(h1 == h2);
        assert(h_neg == ~h1);
        assert(std::fabs(simhash64_cosine_estimate(h1, h2) - 1.0f) < 1e-5f);
        assert(simhash64_cosine_estimate(h1, h_neg) < -0.99f);
    }

    // 4. ContextCapsule build_from_span and should_wake
    {
        std::vector<int32_t> span_tokens = {101, 205, 309, 412, 518, 620};
        std::vector<float> centroids(4 * 16, 0.5f);
        ContextCapsule cap;
        cap.build_from_span(7, 1400, span_tokens.data(), (int32_t)span_tokens.size(), centroids.data(), 4, 16, 3);
        assert(cap.capsule_id == 7);
        assert(cap.token_start == 1400);
        assert(cap.token_count == 6);

        // Wake via exact token hit
        int32_t q_hit[2] = {9999, 309};
        assert(cap.should_wake(q_hit, 2, ~cap.simhash_64, 1, 0.99f));

        // Wake via bigram hit
        int32_t q_bigram[2] = {205, 309};
        assert(cap.should_wake(q_bigram, 2, ~cap.simhash_64, 10, 0.99f));

        // Wake via SimHash semantic similarity
        int32_t q_miss[2] = {8888, 9999};
        assert(cap.should_wake(q_miss, 2, cap.simhash_64, 10, 0.80f));

        // Stay dormant when neither SimHash nor Fuse matches
        assert(!cap.should_wake(q_miss, 2, ~cap.simhash_64, 2, 0.80f));
    }

    printf("ALL KV CAPSULE FILTER C++ TESTS PASSED\n");
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_file = tmp_path / "test_kv_capsule.cpp"
        bin_file = tmp_path / "test_kv_capsule"
        src_file.write_text(cpp_code, encoding="utf-8")

        compile_cmd = [
            "g++",
            "-O2",
            "-std=c++17",
            f"-I{KV_CAPSULE_HEADER.parent}",
            str(src_file),
            "-o",
            str(bin_file),
        ]
        res_compile = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        assert res_compile.returncode == 0, f"Compilation failed: {res_compile.stderr}"

        res_run = subprocess.run([str(bin_file)], capture_output=True, text=True, check=False)
        assert res_run.returncode == 0, f"Execution failed: {res_run.stderr}"
        assert "ALL KV CAPSULE FILTER C++ TESTS PASSED" in res_run.stdout


def test_pto_profile_runner_and_ctx_capsule_ts_integration():
    """Verify 6-stage PTO profile runner, Mode B live-daemon merge, pto_finetune.sh, and ctx_capsule_tui.ts --pto-optimize."""
    pto_runner = REPO_ROOT / "scripts" / "pto_profile_runner.py"
    pto_finetune = REPO_ROOT / "scripts" / "pto_finetune.sh"
    ts_tui = REPO_ROOT / "tools" / "ctx-capsule" / "ctx_capsule_tui.ts"

    assert pto_runner.exists()
    assert pto_finetune.exists()
    assert ts_tui.exists()

    # 1. Verify --print-pgo-flags outputs Clang-21 IRPGO, CS-IRPGO, ThinLTO, and Machine Function Splitting
    res_flags = subprocess.run(
        [sys.executable, str(pto_runner), "--print-pgo-flags"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_flags.returncode == 0
    assert "GGML_PTO_MODE=GENERATE" in res_flags.stdout
    assert "GGML_PTO_MODE=CS_GENERATE" in res_flags.stdout
    assert "GGML_THINLTO=ON" in res_flags.stdout
    assert "GGML_SPLIT_MACHINE_FUNCTIONS=ON" in res_flags.stdout
    assert "GGML_NCCL=OFF" in res_flags.stdout

    # 2. Verify 6-stage synthetic sweep dry-run uses production env defaults (GGML_CUDA_NO_PINNED=0, -t 56)
    res_dry = subprocess.run(
        [sys.executable, str(pto_runner), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_dry.returncode == 0, f"pto_profile_runner --dry-run failed: {res_dry.stderr}"
    assert "GGML_CUDA_NO_PINNED=0" in res_dry.stdout
    assert "Stage 1/6:" in res_dry.stdout
    assert "Stage 5/6: Live llama-server Multi-Slot HTTP" in res_dry.stdout
    assert "Stage 6/6: llama-bench GEMM / FlashAttention / MoE Microbenchmark" in res_dry.stdout
    assert "llvm-profdata" in res_dry.stdout

    # 3. Verify Mode B (--live-daemon-merge) dry-run
    res_merge = subprocess.run(
        [sys.executable, str(pto_runner), "--live-daemon-merge", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_merge.returncode == 0
    assert "PTO MODE B: CONTINUOUS LIVE-DAEMON TRACE MERGE" in res_merge.stdout

    # 4. Verify scripts/pto_finetune.sh --dry-run routes through scripts/build-guard.sh
    res_sh = subprocess.run(
        [str(pto_finetune), "--pto=use", "--shared-libs", "--compiler=clang-21", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_sh.returncode == 0, f"pto_finetune.sh failed: {res_sh.stderr}"
    assert "scripts/build-guard.sh" in res_sh.stdout
    assert "-DGGML_NCCL=OFF" in res_sh.stdout
    assert "-DGGML_THINLTO=ON" in res_sh.stdout
    assert "-DGGML_SPLIT_MACHINE_FUNCTIONS=ON" in res_sh.stdout
    assert "-DBUILD_SHARED_LIBS=ON" in res_sh.stdout

    # 5. Verify tools/ctx-capsule/ctx_capsule_tui.ts --pto-optimize and --export-toml
    res_pto_only = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(ts_tui),
            "--pto-optimize",
            "cs_irpgo_2stage",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_pto_only.returncode == 0, f"ctx_capsule_tui.ts --pto-optimize failed: {res_pto_only.stderr}"
    assert "STATUS: PTO Fine-Tuned (cs_irpgo_2stage via clang21_lld_znver3)" in res_pto_only.stdout
    assert "ik_llama_pto_lto" in res_pto_only.stdout

    res_ts = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            str(ts_tui),
            "--pto-optimize",
            "cs_irpgo_2stage",
            "--export-toml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_ts.returncode == 0, f"ctx_capsule_tui.ts failed: {res_ts.stderr}"
    assert "[capsule.pto.trace_and_optimize]" in res_ts.stdout
    assert "[PTO Tune & Swap (p)]" in res_ts.stdout
    defaults_toml = (REPO_ROOT / "scripts" / "configs" / "hierarchy" / "00-defaults.toml").read_text(encoding="utf-8")
    assert "[pto]" in defaults_toml
    assert 'pto_mode = "irpgo_thinlto_split"' in defaults_toml
    models_toml = (REPO_ROOT / "scripts" / "configs" / "models.toml").read_text(encoding="utf-8")
    assert "[pto]" in models_toml
    assert 'pto_mode = "cs_irpgo_2stage"' in models_toml

    # Restore default TOML bundle
    subprocess.run(
        ["node", "--experimental-strip-types", str(ts_tui), "--export-toml"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_live_http_server_evidence_handshake_chat_and_completions():
    """Verify live HTTP /v1/chat/completions and /v1/completions evidence_capsule & X-Evidence-* forwarding without top_k collision."""
    import http.client
    import time
    import numpy as np

    server_bin = REPO_ROOT / "build-pto" / "bin" / "llama-server"
    vocab_gguf = REPO_ROOT / "models" / "ggml-vocab-gpt-2.gguf"
    if not server_bin.exists() or not vocab_gguf.exists():
        pytest.skip("llama-server or ggml-vocab-gpt-2.gguf not available")

    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    import gguf

    with tempfile.TemporaryDirectory(prefix="ev_http_test_") as tmpdir:
        model_path = Path(tmpdir) / "tiny_gpt2.gguf"
        reader = gguf.GGUFReader(str(vocab_gguf))
        writer = gguf.GGUFWriter(str(model_path), "llama")
        writer.add_uint32("llama.context_length", 4096)
        writer.add_uint32("llama.embedding_length", 64)
        writer.add_uint32("llama.block_count", 1)
        writer.add_uint32("llama.feed_forward_length", 128)
        writer.add_uint32("llama.attention.head_count", 2)
        writer.add_uint32("llama.attention.head_count_kv", 2)
        writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        writer.add_string(
            "tokenizer.chat_template",
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant: ",
        )

        for key, field in reader.fields.items():
            if key.startswith("tokenizer."):
                val_type = field.types[0]
                if val_type == gguf.GGUFValueType.STRING:
                    s = bytes(field.parts[field.data[0]]).decode("utf-8")
                    writer.add_string(key, s)
                elif val_type == gguf.GGUFValueType.UINT32:
                    writer.add_uint32(key, int(field.parts[field.data[0]][0]))
                elif val_type == gguf.GGUFValueType.ARRAY:
                    sub_type = field.types[1]
                    if sub_type == gguf.GGUFValueType.STRING:
                        arr = [bytes(field.parts[idx]).decode("utf-8", errors="replace") for idx in field.data]
                        writer.add_array(key, arr)
                    elif sub_type == gguf.GGUFValueType.INT32:
                        arr = [int(field.parts[idx][0]) for idx in field.data]
                        writer.add_array(key, arr)
                    elif sub_type == gguf.GGUFValueType.FLOAT32:
                        arr = [float(field.parts[idx][0]) for idx in field.data]
                        writer.add_array(key, arr)

        n_vocab = len(reader.fields["tokenizer.ggml.tokens"].data)
        n_embd = 64
        n_ff = 128
        rng = np.random.default_rng(42)
        writer.add_tensor("token_embd.weight", (rng.standard_normal((n_vocab, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("output_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor("output.weight", (rng.standard_normal((n_vocab, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.attn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor("blk.0.attn_q.weight", (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.attn_k.weight", (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.attn_v.weight", (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.attn_output.weight", (rng.standard_normal((n_embd, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.ffn_norm.weight", np.ones((n_embd,), dtype=np.float32))
        writer.add_tensor("blk.0.ffn_gate.weight", (rng.standard_normal((n_ff, n_embd)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.ffn_down.weight", (rng.standard_normal((n_embd, n_ff)) * 0.02).astype(np.float32))
        writer.add_tensor("blk.0.ffn_up.weight", (rng.standard_normal((n_ff, n_embd)) * 0.02).astype(np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        port = 18599
        srv = subprocess.Popen(
            [
                str(server_bin),
                "-m",
                str(model_path),
                "--port",
                str(port),
                "-c",
                "2048",
                "-np",
                "1",
                "-ngl",
                "0",
                "--jinja",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(40):
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    conn.request("GET", "/health")
                    resp = conn.getresponse()
                    resp.read()
                    conn.close()
                    if resp.status == 200:
                        break
                except Exception:
                    time.sleep(0.25)
            else:
                pytest.fail("llama-server did not become healthy")

            haystack = " ".join(
                [f"Section {i}: alpha beta gamma delta epsilon zeta eta theta." for i in range(60)]
            )
            messages = [
                {"role": "system", "content": "You are a precise code analysis assistant."},
                {"role": "user", "content": haystack + "\nQuestion: What is in Section 42?"},
            ]

            # 1. /v1/chat/completions with JSON body evidence_capsule + top_k=40 (sampling top_k) + evidence_top_k=2
            payload1 = {
                "model": "test",
                "messages": messages,
                "max_tokens": 4,
                "top_k": 40,
                "evidence_top_k": 2,
                "evidence_capsule": {
                    "enabled": True,
                    "strategy": "code_pinpoint",
                    "chunk_tokens": 32,
                    "top_k": 2,
                    "min_tokens": 64,
                    "keep_head": 16,
                    "keep_tail": 32,
                },
            }
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload1),
                headers={"Content-Type": "application/json"},
            )
            resp1 = conn.getresponse()
            hdrs1 = dict(resp1.getheaders())
            body1 = json.loads(resp1.read().decode("utf-8"))
            conn.close()

            assert hdrs1.get("X-Evidence-Filter-Applied") == "true"
            assert int(hdrs1.get("X-Evidence-Tokens-Before", "0")) > int(hdrs1.get("X-Evidence-Tokens-After", "0"))
            assert int(hdrs1.get("X-Evidence-Chunks-Kept", "0")) == 2
            assert hdrs1.get("X-Evidence-Strategy") == "code_pinpoint"
            assert body1["evidence_telemetry"]["active"] is True
            assert body1["evidence_telemetry"]["chunks_kept"] == 2
            assert body1["evidence_telemetry"]["keep_head"] == 16
            assert body1["evidence_telemetry"]["keep_tail"] == 32

            # 2. /v1/chat/completions with X-Evidence-* headers (including X-Evidence-Top-M & X-Evidence-Chunk-Size) and top_k=40
            payload2 = {
                "model": "test",
                "messages": messages,
                "max_tokens": 4,
                "top_k": 40,
            }
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload2),
                headers={
                    "Content-Type": "application/json",
                    "X-Evidence-Filter": "true",
                    "X-Evidence-Strategy": "maxsim_exact_cover",
                    "X-Evidence-Chunk-Size": "32",
                    "X-Evidence-Top-M": "3",
                    "X-Evidence-Min-Tokens": "64",
                    "X-Evidence-Keep-Head": "20",
                    "X-Evidence-Keep-Tail": "28",
                },
            )
            resp2 = conn.getresponse()
            hdrs2 = dict(resp2.getheaders())
            body2 = json.loads(resp2.read().decode("utf-8"))
            conn.close()

            assert hdrs2.get("X-Evidence-Filter-Applied") == "true"
            assert int(hdrs2.get("X-Evidence-Chunks-Kept", "0")) == 3
            assert hdrs2.get("X-Evidence-Strategy") == "maxsim_exact_cover"
            assert body2["evidence_telemetry"]["chunks_kept"] == 3
            assert body2["evidence_telemetry"]["keep_head"] == 20
            assert body2["evidence_telemetry"]["keep_tail"] == 28
        finally:
            srv.terminate()
            srv.wait(timeout=5)



