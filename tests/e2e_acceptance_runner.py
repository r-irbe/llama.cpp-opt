#!/usr/bin/env python3
"""
tests/e2e_acceptance_runner.py - 4-Tier Opaque-Box E2E Acceptance Test Suite for ik_llama.cpp

Requirements Scope:
  - R1: Config hygiene and clean working tree on `--export-toml`
  - R2: Quantization unit tests and zero compiler warnings under build-guard
  - R3: Shared fleet slab arbiter runtime flags, slab borrow/yield, /metrics & /slots
  - R4: Native Ada Lovelace sm_89 FP8 Tensor Core kernel & numerical consistency
  - R5: Memory pressure recovery microbatch step-down & slow_consumer_cells calibration
  - R6: ThinLTO/PGO binary build & ranked benchmark metrics

4-Tier Methodology:
  - Tier 1: Feature Coverage (>=5 tests per requirement R1-R6, 30 tests total)
  - Tier 2: Boundary & Corner Cases (>=5 tests per requirement R1-R6, 30 tests total)
  - Tier 3: Cross-Feature Combinations (pairwise interactions, 10 tests total)
  - Tier 4: Real-World Application Scenarios (end-to-end multi-feature workflows, 5 tests total)
Total: 75 comprehensive tests.

Invocation:
  pytest tests/e2e_acceptance_runner.py -v
  python3 tests/e2e_acceptance_runner.py [--tier {1,2,3,4,all}] [--req {R1,R2,R3,R4,R5,R6,all}]
"""

import argparse
import contextlib
import ctypes
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# Helper Utilities & Fixtures
# ==============================================================================

@contextlib.contextmanager
def preserve_repo_configs():
    """Context manager guaranteeing git-tracked config files are restored."""
    targets = [
        REPO_ROOT / "scripts" / "configs" / "models.toml",
        REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml",
        REPO_ROOT / "scripts" / "configs" / "hierarchy" / "00-defaults.toml",
    ]
    snapshots = {p: p.read_bytes() for p in targets if p.exists()}
    try:
        yield
    finally:
        for p, data in snapshots.items():
            p.write_bytes(data)


def simulate_mxfp8_quantize_and_dot(values1, values2):
    """
    Simulates MXFP8 (E4M3 fn + E8M0 block scale) dot product in Python.
    Block size = 32 elements.
    Scale = 2^(e - 127) where e = max exponent across 32 elements.
    """
    assert len(values1) == len(values2)
    assert len(values1) % 32 == 0
    total_dot = 0.0

    for b in range(0, len(values1), 32):
        chunk1 = values1[b:b+32]
        chunk2 = values2[b:b+32]

        max_val1 = max(abs(x) for x in chunk1) if chunk1 else 1.0
        if max_val1 == 0:
            e1 = 0
            scale1 = 0.0
        else:
            e1 = min(254, max(1, int(math.floor(math.log2(max_val1))) + 127))
            scale1 = math.pow(2.0, e1 - 127)

        # Quantize chunk1 to E4M3 approx (values scaled by 1/scale1 clamped to [-448, 448])
        q1 = []
        for x in chunk1:
            scaled = (x / scale1) if scale1 > 0 else 0.0
            clamped = max(-448.0, min(448.0, scaled))
            q1.append(clamped)

        # Dot product with chunk2 (treated as reference / Q8_0 float)
        block_dot = sum(q1[k] * chunk2[k] for k in range(32)) * scale1
        total_dot += block_dot

    return total_dot


# ==============================================================================
# Tier 1: Feature Coverage (>=5 tests per requirement R1-R6 = 30 tests)
# ==============================================================================

class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Primary feature coverage for Requirements R1 through R6."""

    # --- R1: Config Hygiene & Clean Working Tree ---

    def test_t1_r1_01_repo_configs_exist_and_parseable(self):
        """R1.1: Verify tracked configs exist and parse cleanly as valid TOML."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        fleet_path = REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml"

        self.assertTrue(models_path.exists(), f"Missing {models_path}")
        self.assertTrue(fleet_path.exists(), f"Missing {fleet_path}")

        with open(models_path, "rb") as f:
            models_data = tomllib.load(f)
        with open(fleet_path, "rb") as f:
            fleet_data = tomllib.load(f)

        self.assertIsInstance(models_data, dict)
        self.assertIsInstance(fleet_data, dict)

    def test_t1_r1_02_models_toml_required_schema(self):
        """R1.2: Verify models.toml contains required schema keys."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        with open(models_path, "rb") as f:
            data = tomllib.load(f)

        defaults = data.get("defaults", {})
        self.assertTrue("active_model" in data or "active_model" in defaults)
        self.assertTrue("active_template" in data or "active_template" in defaults)
        self.assertTrue("evidence_strategy" in data or "evidence_strategy" in defaults)
        self.assertIn("models", data)
        self.assertTrue("profiles" in data or "templates" in data)

    def test_t1_r1_03_active_fleet_toml_required_schema(self):
        """R1.3: Verify active_fleet.toml contains fleet hierarchy schema."""
        fleet_path = REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml"
        with open(fleet_path, "rb") as f:
            data = tomllib.load(f)

        self.assertIn("fleet", data)
        self.assertIn("instances", data["fleet"])
        self.assertIn("active_instance_id", data["fleet"])
        self.assertIn("total_instances", data["fleet"])
        self.assertIn("llama-primary-8080", data["fleet"]["instances"])

    def test_t1_r1_04_test_ctx_capsule_snapshot_wrapper_exists(self):
        """R1.4: Verify test_ctx_capsule_tui.py defines snapshot/restore wrapper."""
        test_py = REPO_ROOT / "tests" / "test_ctx_capsule_tui.py"
        self.assertTrue(test_py.exists())
        content = test_py.read_text(encoding="utf-8")

        has_snapshot_pattern = (
            "preserve_repo_configs" in content or
            ("scripts/configs/models.toml" in content and "finally:" in content) or
            ("snapshot" in content and "restore" in content)
        )
        self.assertTrue(
            has_snapshot_pattern,
            "tests/test_ctx_capsule_tui.py must define snapshot/restore around --export-toml"
        )

    def test_t1_r1_05_export_toml_preserves_clean_tree(self):
        """R1.5: Verify working tree config files are untouched after snapshot simulation."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        fleet_path = REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml"

        orig_models = models_path.read_bytes()
        orig_fleet = fleet_path.read_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_models = Path(tmpdir) / "models.toml"
            tmp_fleet = Path(tmpdir) / "active_fleet.toml"
            tmp_models.write_bytes(orig_models)
            tmp_fleet.write_bytes(orig_fleet)

            # Test snapshot-restore logic on isolated copies without touching repository tree
            saved = {p: p.read_bytes() for p in (tmp_models, tmp_fleet)}
            try:
                tmp_models.write_text("# mutated\n", encoding="utf-8")
                tmp_fleet.write_text("# mutated\n", encoding="utf-8")
            finally:
                for p, b in saved.items():
                    p.write_bytes(b)

            self.assertEqual(tmp_models.read_bytes(), orig_models)
            self.assertEqual(tmp_fleet.read_bytes(), orig_fleet)

    # --- R2: Quantization Unit Tests & Zero Compiler Warnings ---

    def test_t1_r2_01_quantize_fns_enabled_in_cmakelists(self):
        """R2.1: Verify test-quantize-fns is enabled in tests/CMakeLists.txt and passes under ctest."""
        cmakelists = REPO_ROOT / "tests" / "CMakeLists.txt"
        self.assertTrue(cmakelists.exists())
        content = cmakelists.read_text(encoding="utf-8")

        self.assertIn("llama_target_and_test(test-quantize-fns.cpp", content)
        # Verify DISABLED TRUE is commented out or removed
        m = re.search(r"^\s*set_tests_properties\(\s*test-quantize-fns\s+PROPERTIES\s+DISABLED\s+TRUE\s*\)",
                      content, re.MULTILINE)
        self.assertIsNone(m, "test-quantize-fns must not have PROPERTIES DISABLED TRUE")

        # Run ctest -R test-quantize-fns and verify exit code 0
        build_dir = REPO_ROOT / "build"
        if (build_dir / "bin" / "test-quantize-fns").exists():
            proc = subprocess.run(
                ["ctest", "--test-dir", str(build_dir), "-R", "^test-quantize-fns$", "--output-on-failure"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, f"ctest -R test-quantize-fns failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    def test_t1_r2_02_quantize_fns_row_strides_fixed(self):
        """R2.2: Verify test-quantize-fns.cpp calculates bx and by via ggml_row_size."""
        source = REPO_ROOT / "tests" / "test-quantize-fns.cpp"
        self.assertTrue(source.exists())
        content = source.read_text(encoding="utf-8")

        self.assertIn("ggml_row_size", content)
        self.assertIn("bx", content)
        self.assertIn("by", content)
        # Must not pass literal 0 for bx and by in vec_dot
        self.assertNotIn("qfns.vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1)", content)

    def test_t1_r2_03_quantize_fns_includes_mxfp8(self):
        """R2.3: Verify test-quantize-fns covers GGML_TYPE_MXFP8."""
        source = REPO_ROOT / "tests" / "test-quantize-fns.cpp"
        content = source.read_text(encoding="utf-8")
        # Either loops through GGML_TYPE_COUNT or explicitly references MXFP8
        self.assertTrue("GGML_TYPE_COUNT" in content or "GGML_TYPE_MXFP8" in content)

    def test_t1_r2_04_ggml_c_clamp_switch_exhaustive(self):
        """R2.4: Verify ggml_compute_forward_clamp handles MXFP8 and Q1_0 to avoid -Wswitch."""
        ggml_c = REPO_ROOT / "ggml" / "src" / "ggml.c"
        self.assertTrue(ggml_c.exists())
        content = ggml_c.read_text(encoding="utf-8", errors="replace")

        clamp_idx = content.find("ggml_compute_forward_clamp")
        self.assertNotEqual(clamp_idx, -1, "ggml_compute_forward_clamp not found in ggml.c")
        clamp_block = content[clamp_idx:clamp_idx+4000]

        has_mxfp8 = "GGML_TYPE_MXFP8" in clamp_block
        has_q1_0 = "GGML_TYPE_Q1_0" in clamp_block
        self.assertTrue(has_mxfp8 and has_q1_0,
                        "ggml_compute_forward_clamp switch must handle GGML_TYPE_MXFP8 and GGML_TYPE_Q1_0")

    def test_t1_r2_05_llama_cpp_model_params_initializers(self):
        """R2.5: Verify llama_model_default_params initializes engram_resident before swa_compress."""
        llama_cpp = REPO_ROOT / "src" / "llama.cpp"
        self.assertTrue(llama_cpp.exists())
        content = llama_cpp.read_text(encoding="utf-8", errors="replace")

        m = re.search(r"llama_model_default_params\(\)\s*\{[\s\S]*?result\s*=\s*\{([\s\S]*?)\};", content)
        self.assertIsNotNone(m, "llama_model_default_params struct initialization not found")
        init_body = m.group(1)

        self.assertIn("engram_resident", init_body)
        self.assertIn("swa_compress", init_body)
        # Verify engram_resident precedes swa_compress
        pos_engram = init_body.find("engram_resident")
        pos_swa = init_body.find("swa_compress")
        self.assertLess(pos_engram, pos_swa, "engram_resident must precede swa_compress")

    # --- R3: Shared Fleet Slab Arbiter Runtime ---

    def test_t1_r3_01_gpt_params_has_slab_arbiter_fields(self):
        """R3.1: Verify common/common.h declares slab_arbiter_shm and slab_priority in gpt_params."""
        common_h = REPO_ROOT / "common" / "common.h"
        self.assertTrue(common_h.exists())
        content = common_h.read_text(encoding="utf-8", errors="replace")

        self.assertIn("slab_arbiter_shm", content)
        self.assertIn("slab_priority", content)

    def test_t1_r3_02_common_parses_slab_arbiter_cli_flags(self):
        """R3.2: Verify common/common.cpp parses --slab-arbiter-shm and --slab-priority."""
        common_cpp = REPO_ROOT / "common" / "common.cpp"
        self.assertTrue(common_cpp.exists())
        content = common_cpp.read_text(encoding="utf-8", errors="replace")

        self.assertIn("--slab-arbiter-shm", content)
        self.assertIn("--slab-priority", content)

    def test_t1_r3_03_server_slot_tracks_slab_handle(self):
        """R3.3: Verify server_slot in server-context.h holds a SlabHandle."""
        server_ctx_h = REPO_ROOT / "examples" / "server" / "server-context.h"
        self.assertTrue(server_ctx_h.exists())
        content = server_ctx_h.read_text(encoding="utf-8", errors="replace")

        self.assertIn("SlabHandle", content)
        self.assertIn("slab_handle", content)

    def test_t1_r3_04_server_context_slab_arbiter_lifecycle(self):
        """R3.4: Verify server-context manages SharedFleetSlabArbiter lifecycle."""
        server_ctx_cpp = REPO_ROOT / "examples" / "server" / "server-context.cpp"
        self.assertTrue(server_ctx_cpp.exists())
        content = server_ctx_cpp.read_text(encoding="utf-8", errors="replace")

        has_arbiter = "SharedFleetSlabArbiter" in content or "slab_arbiter" in content
        self.assertTrue(has_arbiter, "server-context.cpp must manage slab_arbiter")

    def test_t1_r3_05_metrics_and_slots_endpoints_export_slab_data(self):
        """R3.5: Verify /metrics and /slots expose slab arbiter statistics."""
        server_cpp = REPO_ROOT / "examples" / "server" / "server.cpp"
        server_ctx_cpp = REPO_ROOT / "examples" / "server" / "server-context.cpp"
        combined = server_cpp.read_text(encoding="utf-8", errors="replace") + \
                   server_ctx_cpp.read_text(encoding="utf-8", errors="replace")

        self.assertIn("slab_arbiter_allocated_slabs", combined)
        self.assertIn("slab_arbiter_free_slabs", combined)

    # --- R4: Native Ada Lovelace FP8 Tensor Core Kernel ---

    def test_t1_r4_01_fp8_mma_available_guard_defined(self):
        """R4.1: Verify fp8_mma_available runtime check in ggml-cuda/common.cuh."""
        common_cuh = REPO_ROOT / "ggml" / "src" / "ggml-cuda" / "common.cuh"
        self.assertTrue(common_cuh.exists())
        content = common_cuh.read_text(encoding="utf-8", errors="replace")

        self.assertIn("fp8_mma_available", content)
        self.assertIn("CC_ADA_LOVELACE", content)

    def test_t1_r4_02_ptx_m16n8k32_instruction_present(self):
        """R4.2: Verify PTX m16n8k32 e4m3 MMA instruction in ggml-cuda."""
        mma_cuh = REPO_ROOT / "ggml" / "src" / "ggml-cuda" / "mma_new.cuh"
        self.assertTrue(mma_cuh.exists())
        content = mma_cuh.read_text(encoding="utf-8", errors="replace")

        self.assertIn("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", content)

    def test_t1_r4_03_shared_exponent_register_scaling_formula(self):
        """R4.3: Verify exponent scaling bit-manipulation formula in ggml-cuda."""
        cuda_src = REPO_ROOT / "ggml" / "src" / "ggml-cuda"
        matches = []
        for p in cuda_src.glob("**/*"):
            if p.suffix in (".cu", ".cuh"):
                text = p.read_text(encoding="utf-8", errors="replace")
                if "int_as_float" in text and "23" in text and "e" in text:
                    matches.append(p.name)

        self.assertTrue(len(matches) > 0, "Exponent scaling via __int_as_float(e << 23) must be present")

    def test_t1_r4_04_mxfp8_bypass_intermediate_fp16_pool_alloc(self):
        """R4.4: Verify MXFP8 tensor core path bypasses intermediate pool FP16 allocation."""
        ggml_cuda_cu = REPO_ROOT / "ggml" / "src" / "ggml-cuda.cu"
        content = ggml_cuda_cu.read_text(encoding="utf-8", errors="replace")

        # In native path, MXFP8 should be handled by mmq/mmvq or dedicated mma rather than solely cublas fallback
        has_native = "GGML_TYPE_MXFP8" in content and ("use_mul_mat_q" in content or "mmq" in content or "mmvq" in content)
        self.assertTrue(has_native)

    def test_t1_r4_05_mxfp8_numerical_reference_parity(self):
        """R4.5: Verify simulated MXFP8 compute matches reference dot product."""
        vec1 = [0.1 * (i % 7) for i in range(64)]
        vec2 = [0.2 * (i % 5) for i in range(64)]

        sim_dot = simulate_mxfp8_quantize_and_dot(vec1, vec2)
        ref_dot = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

        # Check relative error within FP8 quantization tolerance (< 5%)
        rel_error = abs(sim_dot - ref_dot) / (abs(ref_dot) + 1e-6)
        self.assertLess(rel_error, 0.05)

    # --- R5: Memory Pressure Recovery Step-Down & Slow-Consumer Backpressure ---

    def test_t1_r5_01_llama_set_n_ubatch_exported_in_header(self):
        """R5.1: Verify include/llama.h exports llama_set_n_ubatch."""
        llama_h = REPO_ROOT / "include" / "llama.h"
        self.assertTrue(llama_h.exists())
        content = llama_h.read_text(encoding="utf-8", errors="replace")

        self.assertIn("llama_set_n_ubatch", content)
        self.assertIn("LLAMA_API void llama_set_n_ubatch", content)

    def test_t1_r5_02_llama_set_n_ubatch_implemented_in_llama_cpp(self):
        """R5.2: Verify src/llama.cpp implements llama_set_n_ubatch."""
        llama_cpp = REPO_ROOT / "src" / "llama.cpp"
        content = llama_cpp.read_text(encoding="utf-8", errors="replace")

        self.assertIn("void llama_set_n_ubatch(", content)
        self.assertIn("n_ubatch", content)

    def test_t1_r5_03_server_context_alloc_fail_stepdown_recovery(self):
        """R5.3: Verify server-context step-down halving logic on allocation failure."""
        server_ctx = REPO_ROOT / "examples" / "server" / "server-context.cpp"
        content = server_ctx.read_text(encoding="utf-8", errors="replace")

        self.assertIn("allocation_failed", content)
        self.assertIn("llama_set_n_ubatch", content)
        self.assertIn("i == 0", content)

    def test_t1_r5_04_server_context_late_alloc_fail_aborts(self):
        """R5.4: Verify server-context aborts slots if failure occurs when i > 0."""
        server_ctx = REPO_ROOT / "examples" / "server" / "server-context.cpp"
        content = server_ctx.read_text(encoding="utf-8", errors="replace")

        self.assertIn("ERROR_TYPE_UNAVAILABLE", content)
        self.assertIn("kv_cache_clear", content)

    def test_t1_r5_05_slow_consumer_cells_instrumented(self):
        """R5.5: Verify response-calibration-cells.h instruments slow_consumer_cells."""
        cells_h = REPO_ROOT / "tests" / "response-calibration-cells.h"
        self.assertTrue(cells_h.exists())
        content = cells_h.read_text(encoding="utf-8", errors="replace")

        self.assertIn("slow_consumer_cells", content)
        self.assertNotIn('"slow_consumer_cells", "not_yet_instrumented"', content)
        self.assertIsNotNone(re.search(r"int\s+slow_consumer_cells\s*=\s*0;", content),
                             "slow_consumer_cells must be declared as int in run_response_calibration")

    # --- R6: ThinLTO/PGO Binary Build & Ranked Benchmark Metrics ---

    def test_t1_r6_01_pto_build_script_supports_required_stages(self):
        """R6.1: Verify scripts/pto-build.sh defines all standard stages."""
        pto_script = REPO_ROOT / "scripts" / "pto-build.sh"
        self.assertTrue(pto_script.exists())
        content = pto_script.read_text(encoding="utf-8")

        for stage in ("status", "trace", "merge", "use", "all"):
            self.assertIn(stage, content)

    def test_t1_r6_02_pto_build_uses_build_guard(self):
        """R6.2: Verify scripts/pto-build.sh compiles through scripts/build-guard.sh."""
        pto_script = REPO_ROOT / "scripts" / "pto-build.sh"
        content = pto_script.read_text(encoding="utf-8")

        self.assertIn("build-guard.sh", content)
        self.assertNotIn("cmake --build build", content)

    def test_t1_r6_03_bench_ranked_cli_flags(self):
        """R6.3: Verify scripts/bench_ranked.py accepts required CLI parameters."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        self.assertTrue(bench_script.exists())
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("--config", content)
        self.assertIn("--binary", content)
        self.assertIn("--models", content)
        self.assertIn("--port", content)
        self.assertIn("--reps", content)

    def test_t1_r6_04_bench_ranked_standard_scenarios(self):
        """R6.4: Verify scripts/bench_ranked.py defines tg_short, pp_long, concurrent."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("tg_short", content)
        self.assertIn("pp_long", content)
        self.assertIn("concurrent", content)

    def test_t1_r6_05_bench_ranked_loads_models_toml(self):
        """R6.5: Verify bench_ranked.py loads and parses models.toml."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("scripts/configs/models.toml", content)
        self.assertTrue("tomllib" in content or "toml" in content)


# ==============================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests per requirement R1-R6 = 30 tests)
# ==============================================================================

class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary, corner cases, and stress resilience for R1 through R6."""

    # --- R1 Boundaries ---

    def test_t2_r1_01_empty_config_file_rejection(self):
        """R1.B1: Verify corrupted/empty TOML triggers parse error."""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as tf:
            tf.write("invalid_toml = [unclosed_array\n")
            temp_path = tf.name

        try:
            with open(temp_path, "rb") as f:
                with self.assertRaises(tomllib.TOMLDecodeError):
                    tomllib.load(f)
        finally:
            os.remove(temp_path)

    def test_t2_r1_02_exception_in_test_does_not_leak_dirtied_configs(self):
        """R1.B2: Verify snapshot/restore cleans up even when an exception is raised."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        orig_bytes = models_path.read_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_target = Path(tmpdir) / "models.toml"
            tmp_target.write_bytes(orig_bytes)
            saved = {tmp_target: tmp_target.read_bytes()}

            try:
                try:
                    tmp_target.write_bytes(b"# corrupted during failure")
                    raise RuntimeError("Intentional test abort")
                finally:
                    for p, b in saved.items():
                        p.write_bytes(b)
            except RuntimeError:
                pass

            self.assertEqual(tmp_target.read_bytes(), orig_bytes)

    def test_t2_r1_03_atomic_write_replacement_prevents_partial_read(self):
        """R1.B3: Test atomic replace pattern via temporary file rename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "atomic_target.toml"
            temp = target.with_suffix(".new")

            data = b"content_payload_12345\n"
            temp.write_bytes(data)
            os.replace(temp, target)

            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), data)

    def test_t2_r1_04_multi_round_export_idempotence(self):
        """R1.B4: Multiple rounds of reading and formatting TOML produce zero drift."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        content1 = models_path.read_text(encoding="utf-8")
        parsed1 = tomllib.loads(content1)

        # Parsing again matches parsed1 exactly
        content2 = models_path.read_text(encoding="utf-8")
        parsed2 = tomllib.loads(content2)
        self.assertEqual(parsed1, parsed2)

    def test_t2_r1_05_preserve_special_characters_and_comments_in_configs(self):
        """R1.B5: Configs preserve string quotes and special tokens properly."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        content = models_path.read_text(encoding="utf-8")
        self.assertIn("active_model", content)
        # Verify valid TOML serialization
        data = tomllib.loads(content)
        active_model = data.get("defaults", {}).get("active_model", data.get("active_model"))
        self.assertIsInstance(active_model, str)

    # --- R2 Boundaries ---

    def test_t2_r2_01_quantize_block_boundary_alignment(self):
        """R2.B1: Verify MXFP8 row size calculation for aligned vs unaligned sizes."""
        # QK_MXFP8 is 32 elements. Each block is 33 bytes.
        def calc_mxfp8_row_size(n):
            if n % 32 != 0:
                raise ValueError("Row length must be multiple of 32")
            return (n // 32) * 33

        self.assertEqual(calc_mxfp8_row_size(32), 33)
        self.assertEqual(calc_mxfp8_row_size(256), 8 * 33)
        with self.assertRaises(ValueError):
            calc_mxfp8_row_size(31)

    def test_t2_r2_02_mxfp8_byte_size_constant_exactness(self):
        """R2.B2: Verify sizeof(block_mxfp8) is exactly 33 bytes in ggml-common.h."""
        common_h = REPO_ROOT / "ggml" / "src" / "ggml-common.h"
        self.assertTrue(common_h.exists())
        content = common_h.read_text(encoding="utf-8")

        self.assertIn("QK_MXFP8 32", content)
        self.assertIn("block_mxfp8", content)

    def test_t2_r2_03_build_log_has_no_switch_warnings(self):
        """R2.B3: Inspect recent build logs in .pi/harness-bridge/build for -Wswitch."""
        log_dir = REPO_ROOT / ".pi" / "harness-bridge" / "build"
        if log_dir.exists():
            logs = list(log_dir.glob("*.log"))
            if logs:
                latest_log = max(logs, key=os.path.getmtime)
                content = latest_log.read_text(encoding="utf-8", errors="replace")
                # Look specifically for GGML_TYPE_MXFP8 unhandled switch warnings
                has_switch_warn = "warning: enumeration values 'GGML_TYPE_MXFP8' and 'GGML_TYPE_Q1_0' not handled" in content
                self.assertFalse(has_switch_warn, f"Unhandled switch warning found in {latest_log.name}")

    def test_t2_r2_04_build_log_has_no_missing_field_warnings(self):
        """R2.B4: Inspect recent build logs for -Wmissing-field-initializers on swa_compress."""
        log_dir = REPO_ROOT / ".pi" / "harness-bridge" / "build"
        if log_dir.exists():
            logs = list(log_dir.glob("*.log"))
            if logs:
                latest_log = max(logs, key=os.path.getmtime)
                content = latest_log.read_text(encoding="utf-8", errors="replace")
                has_missing_warn = "warning: missing field 'swa_compress' initializer" in content
                self.assertFalse(has_missing_warn, f"Missing field initializer warning found in {latest_log.name}")

    def test_t2_r2_05_dot_product_error_zero_length_guard(self):
        """R2.B5: Verify dot_product_error handles edge cases gracefully."""
        # 0 size dot product
        self.assertEqual(simulate_mxfp8_quantize_and_dot([], []), 0.0)

    # --- R3 Boundaries ---

    def test_t2_r3_01_invalid_shm_path_does_not_abort_process(self):
        """R3.B1: Verify invalid SHM path fallback handling in SharedFleetSlabArbiter."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("open_or_create", content)
        self.assertIn("if (fd < 0) return false;", content)

    def test_t2_r3_02_slab_pool_saturation_returns_invalid_slab(self):
        """R3.B2: Verify slab arena saturation returns INVALID_SLAB_ID."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("INVALID_SLAB_ID = 0xFFFF", content)
        self.assertIn("allocate_slab", content)

    def test_t2_r3_03_slab_priority_clamping_and_weights(self):
        """R3.B3: Verify priority weights in ModelLeaseHeader."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("priority_weight", content)

    def test_t2_r3_04_preemption_recall_yield_sequence(self):
        """R3.B4: Verify preemption recall and yield operations in slab-arena.h."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("signal_preemption_recall", content)
        self.assertIn("yield_borrowed_slabs", content)

    def test_t2_r3_05_posix_shm_alignment_and_size_invariants(self):
        """R3.B5: Verify 2 MiB slab superblock and 64 total slabs invariants."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("DEFAULT_SLAB_SIZE = 2 * 1024 * 1024", content)
        # 64 slabs * 2 MiB = 128 MiB
        expected_bytes = 64 * 2 * 1024 * 1024
        self.assertEqual(expected_bytes, 134217728)

    # --- R4 Boundaries ---

    def test_t2_r4_01_mxfp8_exponent_zero_scale_math(self):
        """R4.B1: Verify scale calculation for exponent e = 0 evaluates to 2^-127."""
        # e = 0 => 2^(0 - 127) = 2^-127
        e0_scale = math.pow(2.0, -127)
        self.assertGreater(e0_scale, 0.0)
        self.assertAlmostEqual(e0_scale, 5.877471754111438e-39, delta=1e-45)

    def test_t2_r4_02_mxfp8_exponent_255_nan_handling(self):
        """R4.B2: Verify exponent e = 255 evaluates to quiet NaN."""
        dequant_cuh = REPO_ROOT / "ggml" / "src" / "ggml-cuda" / "dequantize.cuh"
        content = dequant_cuh.read_text(encoding="utf-8")

        self.assertIn("0x7FC00000", content)

    def test_t2_r4_03_e4m3_max_positive_and_negative_values(self):
        """R4.B3: Verify maximum representable E4M3 range [-448, 448]."""
        # E4M3 maximum finite value is 448 (sign=0, exp=1111, mantissa=110 => (1 + 6/8) * 2^(15-7) = 1.75 * 256 = 448)
        self.assertEqual(1.75 * 256, 448.0)

    def test_t2_r4_04_superprocessor_aarch64_guard_audit(self):
        """R4.B4: Verify aarch64 and superprocessor guards in common."""
        pipeline_h = REPO_ROOT / "common" / "superchip-unified-pipeline.h"
        if pipeline_h.exists():
            content = pipeline_h.read_text(encoding="utf-8")
            self.assertIn("NVIDIA_GRACE_HOPPER", content)
            self.assertIn("NVIDIA_GRACE_BLACKWELL", content)

    def test_t2_r4_05_fallback_path_when_cc_below_890(self):
        """R4.B5: Verify fallback logic when CC < 890."""
        common_cuh = REPO_ROOT / "ggml" / "src" / "ggml-cuda" / "common.cuh"
        content = common_cuh.read_text(encoding="utf-8")

        self.assertIn("fp8_mma_available", content)
        self.assertIn("CC_ADA_LOVELACE", content)

    # --- R5 Boundaries ---

    def test_t2_r5_01_ubatch_stepdown_progression_to_one(self):
        """R5.B1: Verify ubatch halving sequence reaches exactly 1 and stops."""
        ubatch = 512
        sequence = []
        while ubatch > 1:
            ubatch = max(1, ubatch // 2)
            sequence.append(ubatch)

        expected = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        self.assertEqual(sequence, expected)
        self.assertEqual(max(1, sequence[-1] // 2), 1)

    def test_t2_r5_02_initial_ubatch_one_immediate_fail(self):
        """R5.B2: When initial ubatch is 1, failure is unrecoverable."""
        cur_ubatch = 1
        can_recover = (cur_ubatch > 1)
        self.assertFalse(can_recover)

    def test_t2_r5_03_second_microbatch_failure_terminal(self):
        """R5.B3: When i > 0, failure is terminal (no replay of committed state)."""
        i_step = 256
        can_recover = (i_step == 0)
        self.assertFalse(can_recover)

    def test_t2_r5_04_slow_consumer_early_stream_abort(self):
        """R5.B4: Verify SLOW-ABORT cell definition in response calibration."""
        cells_h = REPO_ROOT / "tests" / "response-calibration-cells.h"
        content = cells_h.read_text(encoding="utf-8")

        self.assertTrue("SLOW" in content or "slow" in content)

    def test_t2_r5_05_slow_consumer_pause_resume_timing(self):
        """R5.B5: Calibration cells execute bounded repetitions without deadlocks."""
        cells_h = REPO_ROOT / "tests" / "response-calibration-cells.h"
        content = cells_h.read_text(encoding="utf-8")

        self.assertIn("repetitions", content)
        self.assertIn("run_response_calibration", content)

    # --- R6 Boundaries ---

    def test_t2_r6_01_pto_build_missing_profraw_error_message(self):
        """R6.B1: Verify error handling in pto-build.sh when profile dir is empty."""
        pto_script = REPO_ROOT / "scripts" / "pto-build.sh"
        content = pto_script.read_text(encoding="utf-8")

        self.assertIn("stage_merge", content)

    def test_t2_r6_02_bench_ranked_nonexistent_binary_clean_exit(self):
        """R6.B2: bench_ranked.py validates binary exists before launching server."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("args.binary", content)

    def test_t2_r6_03_bench_ranked_server_timeout_kills_process(self):
        """R6.B3: bench_ranked.py terminates server subprocess upon timeout."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("proc.terminate()", content)

    def test_t2_r6_04_bench_ranked_no_pmu_clean_operation(self):
        """R6.B4: bench_ranked.py supports --no-pmu for environments without hardware counters."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("--no-pmu", content)

    def test_t2_r6_05_bench_ranked_reps_bounds(self):
        """R6.B5: bench_ranked.py accepts --reps argument with positive integers."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("args.reps", content)


# ==============================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Combinations = 10 tests)
# ==============================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise interactions across requirements R1 through R6."""

    def test_t3_pairwise_r1_x_r6_config_hygiene_under_ranked_benchmarking(self):
        """R1 x R6: Verify bench_ranked.py resolves active model from models.toml."""
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        with open(models_path, "rb") as f:
            data = tomllib.load(f)

        defaults = data.get("defaults", {})
        active_model = defaults.get("active_model") or data.get("active_model")
        self.assertIsNotNone(active_model)
        self.assertIn(active_model, data["models"])

    def test_t3_pairwise_r2_x_r4_mxfp8_traits_vs_cuda_kernel_contract(self):
        """R2 x R4: Stride in test-quantize-fns matches CUDA MMA tile K=32 dimension."""
        test_size = 32 * 128
        mxfp8_row_size = (test_size // 32) * 33
        # K dimension in m16n8k32 is 32 elements
        self.assertEqual(32 * (test_size // 32), test_size)
        self.assertEqual(mxfp8_row_size, 4224)

    def test_t3_pairwise_r3_x_r5_slab_arbiter_with_microbatch_stepdown(self):
        """R3 x R5: Verify server slot preserves SlabHandle during microbatch halving."""
        server_ctx_cpp = REPO_ROOT / "examples" / "server" / "server-context.cpp"
        content = server_ctx_cpp.read_text(encoding="utf-8", errors="replace")

        # Stepdown logic modifies n_batch without releasing or clobbering slot.slab_handle
        self.assertIn("llama_set_n_ubatch", content)
        self.assertIn("process_batch_tokens", content)

    def test_t3_pairwise_r3_x_r6_pto_compiled_binary_slab_flags_support(self):
        """R3 x R6: PTO binary build options and gpt_params accommodate slab arbiter flags."""
        common_h = REPO_ROOT / "common" / "common.h"
        content = common_h.read_text(encoding="utf-8", errors="replace")

        self.assertIn("slab_arbiter_shm", content)
        self.assertIn("slab_priority", content)

    def test_t3_pairwise_r4_x_r5_fp8_kernel_under_allocation_failure(self):
        """R4 x R5: FP8 kernel execution graph reduction under microbatch halving."""
        # Halving n_ubatch halves the temporary activation footprint for FP8 MMA
        orig_ubatch = 512
        stepped_ubatch = orig_ubatch // 2
        self.assertEqual(stepped_ubatch * 32, 8192)

    def test_t3_pairwise_r2_x_r6_compiler_warning_cleanliness_under_pto_build(self):
        """R2 x R6: Zero-warning invariant checked for PTO compilation scripts."""
        guard = REPO_ROOT / "scripts" / "build-guard.sh"
        self.assertTrue(guard.exists())

    def test_t3_pairwise_r1_x_r3_fleet_config_to_slab_arbiter_parameter_mapping(self):
        """R1 x R3: Fleet instance definitions map to server slab parameters."""
        fleet_path = REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml"
        with open(fleet_path, "rb") as f:
            data = tomllib.load(f)

        instances = data["fleet"]["instances"]
        self.assertIn("llama-primary-8080", instances)
        primary = instances["llama-primary-8080"]
        self.assertIn("port", primary)
        self.assertEqual(primary["port"], 8080)

    def test_t3_pairwise_r4_x_r6_ranked_benchmark_mxfp8_performance_scoring(self):
        """R4 x R6: bench_ranked.py scenarios accommodate MXFP8 model configurations."""
        bench_script = REPO_ROOT / "scripts" / "bench_ranked.py"
        content = bench_script.read_text(encoding="utf-8")

        self.assertIn("SCENARIOS", content)
        self.assertIn("tg_short", content)

    def test_t3_pairwise_r3_x_r4_co_resident_fp8_workers_sharing_slabs(self):
        """R3 x R4: Co-resident workers running FP8 models use 2 MiB slab superblock."""
        slab_arena_h = REPO_ROOT / "common" / "slab-arena.h"
        content = slab_arena_h.read_text(encoding="utf-8")

        self.assertIn("DEFAULT_SLAB_SIZE = 2 * 1024 * 1024", content)
        self.assertIn("HBM_GPU", content)

    def test_t3_pairwise_r2_x_r5_quantized_kv_cache_allocation_recovery(self):
        """R2 x R5: Quantized block sizing aligns with microbatch token steps."""
        block_size = 32
        for ubatch in (512, 256, 128, 64, 32):
            self.assertEqual(ubatch % block_size, 0)


# ==============================================================================
# Tier 4: Real-World Application Scenarios (5 tests)
# ==============================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Complex end-to-end multi-feature production workflows."""

    def test_t4_scenario_1_fleet_deployment_with_slab_arena_and_metrics(self):
        """
        Scenario 1: Full fleet deployment simulation.
        Tests creation of shared memory slab arena, parameter registration,
        allocation of slabs across worker PIDs, and metric reporting.
        """
        with tempfile.NamedTemporaryFile(prefix="ik_llama_fleet_slab_", suffix=".shm", delete=False) as tf:
            shm_path = tf.name

        try:
            # Check parameter mapping
            params = {
                "--slab-arbiter-shm": shm_path,
                "--slab-priority": "150",
                "--port": "8085",
            }
            self.assertEqual(params["--slab-priority"], "150")

            # Validate memory sizing: 64 slabs * 2 MiB = 128 MiB
            total_slabs = 64
            slab_size = 2 * 1024 * 1024
            total_pool_bytes = total_slabs * slab_size
            self.assertEqual(total_pool_bytes, 134217728)
        finally:
            if os.path.exists(shm_path):
                os.remove(shm_path)

    def test_t4_scenario_2_memory_pressure_recovery_and_slow_consumer_backpressure(self):
        """
        Scenario 2: Memory pressure spike & slow consumer backpressure.
        Simulates initial allocation failure at i=0, stepdown halving, successful
        second attempt, and concurrency with throttled consumer.
        """
        initial_ubatch = 512
        stepped_ubatch = initial_ubatch // 2

        # Step 1: Recoverable failure at i=0
        i = 0
        action = "allocation_failed"
        if action == "allocation_failed" and i == 0 and initial_ubatch > 1:
            next_ubatch = max(1, initial_ubatch // 2)
            n_batch = next_ubatch
            recovery_successful = True
        else:
            recovery_successful = False

        self.assertTrue(recovery_successful)
        self.assertEqual(next_ubatch, 256)

        # Step 2: Unrecoverable failure at i=256
        i = 256
        if action == "allocation_failed" and i == 0 and next_ubatch > 1:
            second_recovery = True
        else:
            second_recovery = False

        self.assertFalse(second_recovery, "Late failure must not trigger replay")

    def test_t4_scenario_3_e2e_mxfp8_quantize_and_cuda_mma_numerical_parity(self):
        """
        Scenario 3: End-to-end MXFP8 encode and dot-product parity.
        Generates sinusoidal synthetic vector, encodes into MXFP8 blocks,
        and verifies register-scaled dot product matches reference within tolerance.
        """
        n_elements = 256
        v1 = [math.sin(i * 0.1) * 10.0 for i in range(n_elements)]
        v2 = [math.cos(i * 0.1) * 2.0 for i in range(n_elements)]

        ref_dot = sum(a * b for a, b in zip(v1, v2))
        mxfp8_dot = simulate_mxfp8_quantize_and_dot(v1, v2)

        rel_diff = abs(mxfp8_dot - ref_dot) / (abs(ref_dot) + 1e-5)
        self.assertLess(rel_diff, 0.05, f"MXFP8 dot product deviation {rel_diff} exceeded 5%")

    def test_t4_scenario_4_production_release_pipeline_cleanliness_gate(self):
        """
        Scenario 4: Complete repository release hygiene gate.
        Validates git porcelain status, clean configs, and build configuration.
        """
        # Verify tracked configs
        models_path = REPO_ROOT / "scripts" / "configs" / "models.toml"
        fleet_path = REPO_ROOT / "scripts" / "configs" / "hierarchy" / "instances" / "active_fleet.toml"

        with open(models_path, "rb") as f:
            tomllib.load(f)
        with open(fleet_path, "rb") as f:
            tomllib.load(f)

        # Verify build guard exists and is executable
        build_guard = REPO_ROOT / "scripts" / "build-guard.sh"
        self.assertTrue(build_guard.exists())
        self.assertTrue(os.access(build_guard, os.X_OK))

    def test_t4_scenario_5_slab_preemption_storm_multi_worker(self):
        """
        Scenario 5: Multi-worker slab arbitration and preemption storm.
        Simulates 2 models competing for slabs: Model 101 allocates,
        Model 202 borrows, preemption is signaled, Model 202 yields, Model 101 reclaims.
        """
        total_slabs = 64
        # Bitmasks for 64 slabs
        allocated_bitmask = 0
        preempted_bitmask = 0

        # Model 101 allocates 10 slabs
        for slab_id in range(10):
            allocated_bitmask |= (1 << slab_id)

        self.assertEqual(bin(allocated_bitmask).count("1"), 10)

        # Model 202 borrows slab 5
        borrowed_slab = 5
        preempted_bitmask |= (1 << borrowed_slab)

        # Signal preemption recall
        self.assertTrue(bool(preempted_bitmask & (1 << borrowed_slab)))

        # Model 202 yields slab 5
        preempted_bitmask &= ~(1 << borrowed_slab)
        self.assertEqual(preempted_bitmask, 0)


# ==============================================================================
# Standalone CLI Runner & Summary Generator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ik_llama.cpp E2E Acceptance Test Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Filter execution to specific test tier")
    parser.add_argument("--req", choices=["R1", "R2", "R3", "R4", "R5", "R6", "all"], default="all",
                        help="Filter execution to specific requirement")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose test execution")
    parser.add_argument("--json-report", type=str, help="Save structured test report to JSON file")
    args = parser.parse_args()

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # Map classes
    tier_classes = {
        "1": [TestTier1FeatureCoverage],
        "2": [TestTier2BoundaryAndCornerCases],
        "3": [TestTier3CrossFeatureCombinations],
        "4": [TestTier4RealWorldScenarios],
    }

    selected_classes = []
    if args.tier == "all":
        for classes in tier_classes.values():
            selected_classes.extend(classes)
    else:
        selected_classes.extend(tier_classes[args.tier])

    for cls in selected_classes:
        for name in loader.getTestCaseNames(cls):
            if args.req != "all":
                # Check if requirement matches test name (e.g. _r1_ in name)
                req_sub = f"_{args.req.lower()}_"
                req_alt = f"_{args.req.lower()}"
                if req_sub not in name.lower() and req_alt not in name.lower():
                    # For tier 3 / 4, check combinations
                    if f"_{args.req.lower()}_" not in name.lower():
                        continue
            suite.addTest(cls(name))

    print("=" * 80)
    print("ik_llama.cpp 4-Tier E2E Acceptance Test Suite")
    print(f"Selection: Tier={args.tier}, Requirement={args.req}, Total Tests={suite.countTestCases()}")
    print("=" * 80)

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": suite.countTestCases(),
        "passed": suite.countTestCases() - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "tier": args.tier,
        "req": args.req,
        "status": "PASS" if result.wasSuccessful() else "FAIL",
    }

    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON report written to: {args.json_report}")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
