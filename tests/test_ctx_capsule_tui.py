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
    assert payload["evidence_capsule"]["enabled"] is False
    assert payload["kv_paging"]["zdd_sparse_kv"] is False
    assert payload["speculative"]["spec_type"] == "dflash"
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
