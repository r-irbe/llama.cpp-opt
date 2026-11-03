#!/usr/bin/env python3
"""
tests/test_ctx_capsule_tui.py - Comprehensive Automated Verification Suite
Tests the canonical TypeScript @earendil-works/pi-tui control plane (tools/ctx-capsule/ctx_capsule_tui.ts),
all 7 category tabs, 37 settings, Unicode box-drawing & SettingsList layout geometry,
ranked DeepSeek-V4.1-Flash-1017 presets, 5-layer hierarchical TOML + 8-agent SDK export,
scripts/launch.py integration, C++ common/kv-capsule-filter.h, and live HTTP Evidence Mode.
"""

import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TS_TUI_PATH = REPO_ROOT / "tools" / "ctx-capsule" / "ctx_capsule_tui.ts"
LEGACY_PY_TUI_PATH = REPO_ROOT / "tools" / "ctx-capsule" / "ctx_capsule_tui.py"
KV_CAPSULE_HEADER = REPO_ROOT / "common" / "kv-capsule-filter.h"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\]8;;.*?\x07")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def display_width(text: str) -> int:
    clean = strip_ansi(text)
    width = 0
    for ch in clean:
        if unicodedata.combining(ch):
            continue
        eaw = unicodedata.east_asian_width(ch)
        width += 2 if eaw in ("W", "F") else 1
    return width


def run_ts_tui(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    cmd = ["node", "--experimental-strip-types", str(TS_TUI_PATH), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), check=False)


def test_legacy_python_tui_deleted_and_canonical_ts_tui_present():
    """Verify the duplicate Python curses TUI is removed and canonical @earendil-works/pi-tui entrypoint exists."""
    assert not LEGACY_PY_TUI_PATH.exists(), f"Legacy duplicate {LEGACY_PY_TUI_PATH} must not exist"
    assert TS_TUI_PATH.exists(), f"Missing canonical TypeScript TUI entrypoint {TS_TUI_PATH}"


def test_seven_tabs_and_pi_tui_geometry_invariants():
    """Verify all 7 tabs render via @earendil-works/pi-tui VStack/HStack/SettingsList with exact 136-column visible width."""
    expected_tab_headers = [
        "CATEGORY 1: 1:Models&Tmpl",
        "CATEGORY 2: 2:LlamaFleet",
        "CATEGORY 3: 3:Capsules",
        "CATEGORY 4: 4:KV&ZDD",
        "CATEGORY 5: 5:SpecBandit",
        "CATEGORY 6: 6:HW&NUMA",
        "CATEGORY 7: 7:SDKS&Tel",
    ]
    for tab_num, expected_header in enumerate(expected_tab_headers, start=1):
        res = run_ts_tui("--tab", str(tab_num), "--render-once", "--dry-run")
        assert res.returncode == 0, f"Tab {tab_num} render failed: {res.stderr}"
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        assert len(lines) >= 22
        assert expected_header in strip_ansi(res.stdout)

        # Verify box borders (rounded Unicode or ASCII) and exact 136-column visible width on every line
        first_clean = strip_ansi(lines[0])
        last_clean = strip_ansi(lines[-1])
        assert (first_clean.startswith("╭") and first_clean.endswith("╮")) or (first_clean.startswith("+") and first_clean.endswith("+"))
        assert (last_clean.startswith("╰") and last_clean.endswith("╯")) or (last_clean.startswith("+") and last_clean.endswith("+"))
        for idx, line in enumerate(lines):
            w = display_width(line)
            assert w == 136, f"Tab {tab_num} line {idx} visible width {w} != 136: {strip_ansi(line)!r}"


def test_presets_and_runtime_parameter_tuning_via_ts_control_plane():
    """Verify applying all 4 DeepSeek-V4.1-Flash-1017 presets and tuning parameters via ctx_capsule_tui.ts."""
    ds_presets = [
        "ds41-1017-np1-512k-max-speed",
        "ds41-1017-np1-1m-extreme",
        "ds41-1017-np4-512k-swarm",
        "ds41-1017-np8-1m-swarm",
    ]
    for preset_id in ds_presets:
        res = run_ts_tui("preset", preset_id, "--dry-run")
        assert res.returncode == 0, f"Preset {preset_id} failed: {res.stderr}"
        clean = strip_ansi(res.stdout)
        assert f"[< {preset_id} >]" in clean
        assert "[PTO Tune & Swap (p)]" in clean

    res_tune = run_ts_tui(
        "tune",
        "thinking_effort=high_analytical",
        "kv_type_k=q4_0",
        "--dry-run",
    )
    assert res_tune.returncode == 0, f"Tune failed: {res_tune.stderr}"
    assert "[< q4_0 >]" in strip_ansi(res_tune.stdout)

    # Restore default preset
    run_ts_tui("preset", "coding_pinpoint", "--dry-run")


def test_export_toml_and_eight_agent_sdk_configs():
    """Verify export-toml compiles the 5-layer TOML hierarchy and writes all 8 coding agent SDK configs."""
    res = run_ts_tui("export-toml", "--dry-run")
    assert res.returncode == 0, f"export-toml failed: {res.stderr}"
    sdk_dir = REPO_ROOT / "tools" / "ctx-capsule" / "exports" / "sdks"
    for expected_sdk in [
        "pi-provider.json",
        ".mcp.json",
        "opencode.json",
        "claude-sdk-bridge.json",
        "google-genai-bridge.json",
        "codex-config.toml",
        "copilot-byok.json",
        "weft-capsule-bridge.json",
    ]:
        assert (sdk_dir / expected_sdk).exists(), f"Missing exported SDK config {expected_sdk}"


def test_launch_py_capsule_flag_delegates_to_ts_tui():
    """Verify scripts/launch.py --capsule --dry-run delegates directly to ctx_capsule_tui.ts."""
    launch_script = REPO_ROOT / "scripts" / "launch.py"
    res = subprocess.run(
        [sys.executable, str(launch_script), "--capsule", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert res.returncode == 0, f"launch.py --capsule --dry-run failed: {res.stderr}"
    clean = strip_ansi(res.stdout)
    assert ("CTX-CAPSULE FOOP CONTROL PLANE" in clean or "CTX-CAPSULE CONTROL PLANE" in clean)
    assert "ACTIVE LLAMA.CPP FLEET" in clean



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
    clean_pto_only = strip_ansi(res_pto_only.stdout)
    assert "STATUS: PTO Fine-Tuned (cs_irpgo_2stage via clang21_lld_znver3)" in clean_pto_only
    assert "ik_llama_pto_lto" in clean_pto_only

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
    clean_ts = strip_ansi(res_ts.stdout)
    assert "[capsule.pto.trace_and_optimize]" in clean_ts
    assert "[PTO Tune & Swap (p)]" in clean_ts
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


def test_launch_py_all_models_and_profiles_dry_run_and_idempotent_export():
    import tomllib
    tui_script = REPO_ROOT / "tools" / "ctx-capsule" / "ctx_capsule_tui.ts"
    launch_script = REPO_ROOT / "scripts" / "launch.py"

    # 1. Run --export-toml 3 times consecutively and verify tomllib.loads parses scripts/configs/models.toml with zero duplicate key errors
    for _ in range(3):
        res = subprocess.run(["node", "--experimental-strip-types", str(tui_script), "--export-toml"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert res.returncode == 0, f"Export toml failed: {res.stderr}"

    models_toml = REPO_ROOT / "scripts" / "configs" / "models.toml"
    assert models_toml.exists()
    cfg_text = models_toml.read_text(encoding="utf-8")
    cfg = tomllib.loads(cfg_text)

    assert "models" in cfg
    assert "profiles" in cfg

    # 2. Run python3 scripts/launch.py <key> --dry-run for EVERY model in cfg["models"] and EVERY profile in cfg["profiles"]
    all_keys = list(cfg["models"].keys()) + list(cfg["profiles"].keys())
    for required_prof in ["ds41-1017-np1-512k-max-speed", "ds41-1017-np1-1m-extreme", "ds41-1017-np4-512k-swarm", "ds41-1017-np8-1m-swarm"]:
        if required_prof not in all_keys:
            all_keys.append(required_prof)

    for key in all_keys:
        res = subprocess.run([sys.executable, str(launch_script), key, "--dry-run"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert res.returncode == 0, f"Launch dry-run failed for key '{key}': stdout={res.stdout}, stderr={res.stderr}"

    # 3. Verify --profile combination and --engram-resident / --spec-verify-pinned emission
    combo = subprocess.run(
        [
            sys.executable,
            str(launch_script),
            "deepseek-v4.1-1017-ud-q4kxl",
            "--profile",
            "ds41-1017-np4-512k-swarm",
            "--engram-resident",
            "--spec-verify-pinned",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert combo.returncode == 0, f"Combo dry-run failed: {combo.stderr}"
    assert "-np 4" in combo.stdout
    assert "-c 524288" in combo.stdout
    assert "--swa-compress" in combo.stdout
    assert "--engram-resident" in combo.stdout
    assert "--spec-verify-pinned" in combo.stdout


def test_playwright_tui_devtools_screencast_and_inline_graphics_audit():
    """Verify ctx_capsule_tui.ts --screencast audits all 7 tabs, all 44 options, all 165 variants, Help modal, and Kitty/iTerm2 PNG graphics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cast_path = Path(tmpdir) / "ctx_capsule_audit.cast"
        html_path = Path(tmpdir) / "ctx_capsule_audit.html"
        json_path = Path(tmpdir) / "ctx_capsule_audit.audit.json"

        res = run_ts_tui("--screencast", str(cast_path), "--dry-run")
        assert res.returncode == 0, f"--screencast failed: {res.stderr}"

        report = json.loads(res.stdout)
        assert report["categoriesAudited"] == 7
        assert report["totalSettingsAudited"] == 44
        assert report["totalVariantsCycledAndVerified"] >= 160
        assert report["helpModalVerified"] is True
        assert report["responsive88ColVerified"] is True
        assert report["geometryViolations"] == []
        assert report["explanationViolations"] == []
        assert len(report["inlineGraphicsAudit"]) == 7
        for item in report["inlineGraphicsAudit"]:
            assert item["kittyValidApc"] is True
            assert item["iterm2ValidOsc"] is True
            assert item["pngBase64Length"] > 500

        assert cast_path.exists()
        assert html_path.exists()
        assert json_path.exists()
        cast_lines = cast_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(cast_lines) > 150
        header = json.loads(cast_lines[0])
        assert header["version"] == 2
        assert header["width"] == 136




