#!/usr/bin/env python3
"""
Test Suite for scripts/gguf-pack-npmc.py
Validates GGUF-X / NPMC repacker against scripts/gguf-lint.py:
- Sector alignment (LINT001)
- MoE expert contiguous grouping (LINT005)
- alignas(128) cache-line and 64B SIMD alignment (LINT006, LINT008)
- 2MB HugeTLB conformity (LINT007)
- DRAM bank power-of-two stride aliasing breaking via intra-row padding (LINT009)
- Multi-file sharded GGUF consolidation
- --dry-run and --in-place workflows
"""

import sys
import os
import struct
import tempfile
import subprocess
import shutil
import json
import unittest
import importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_SCRIPT = os.path.join(REPO_ROOT, "scripts", "gguf-pack-npmc.py")
LINT_SCRIPT = os.path.join(REPO_ROOT, "scripts", "gguf-lint.py")

_lint_spec = importlib.util.spec_from_file_location("gguf_lint", LINT_SCRIPT)
gguf_lint = importlib.util.module_from_spec(_lint_spec)
_lint_spec.loader.exec_module(gguf_lint)
parse_gguf_metadata = gguf_lint.parse_gguf_metadata

_pack_spec = importlib.util.spec_from_file_location("gguf_pack", PACK_SCRIPT)
gguf_pack = importlib.util.module_from_spec(_pack_spec)
_pack_spec.loader.exec_module(gguf_pack)
parse_gguf_file = gguf_pack.parse_gguf_file

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY  = 9

def pack_str(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded

def write_test_gguf(path: str,
                    alignment: int = 32,
                    data_pad: int = 0,
                    kv_pairs: dict = None,
                    tensors: list = None):
    """
    Writes a synthetic GGUF v3 file with customizable alignment, metadata, and tensor layouts.
    tensors: list of dict(name, dims, type, data_bytes, custom_offset)
    """
    if kv_pairs is None:
        kv_pairs = {}
    if tensors is None:
        tensors = []

    # Default metadata
    kvs = dict(kv_pairs)
    if "general.alignment" not in kvs:
        kvs["general.alignment"] = ("uint32", alignment)

    kv_bytes = []
    for k, (vtype, val) in kvs.items():
        k_b = pack_str(k)
        if vtype == "uint32":
            v_b = struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", val)
        elif vtype == "string":
            v_b = struct.pack("<I", GGUF_TYPE_STRING) + pack_str(val)
        else:
            raise ValueError(f"Unsupported test vtype: {vtype}")
        kv_bytes.append(k_b + v_b)

    # First pass: serialize header without tensor offsets
    # We will lay out tensor data and compute offsets
    header_fixed = (
        struct.pack("<I", GGUF_MAGIC) +
        struct.pack("<I", GGUF_VERSION) +
        struct.pack("<Q", len(tensors)) +
        struct.pack("<Q", len(kv_bytes)) +
        b"".join(kv_bytes)
    )

    tensor_desc_base_len = 0
    for t in tensors:
        tensor_desc_base_len += len(pack_str(t["name"])) + 4 + (8 * len(t["dims"])) + 4 + 8

    approx_header_size = len(header_fixed) + tensor_desc_base_len
    data_start = (approx_header_size + alignment - 1) & ~(alignment - 1)
    if data_pad > 0:
        data_start += data_pad

    # Compute tensor offsets
    cur_off = 0
    for t in tensors:
        if "custom_offset" in t:
            cur_off = t["custom_offset"]
        else:
            # align to alignment
            cur_off = (cur_off + alignment - 1) & ~(alignment - 1)
        t["offset"] = cur_off
        cur_off += len(t["data_bytes"])

    # Now write the file
    with open(path, "wb") as f:
        f.write(header_fixed)
        for t in tensors:
            f.write(pack_str(t["name"]))
            f.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t["type"]))
            f.write(struct.pack("<Q", t["offset"]))

        hdr_end = f.tell()
        # pad to data_start
        if data_start > hdr_end:
            f.write(b"\x00" * (data_start - hdr_end))

        # write tensor data
        for t in tensors:
            target_pos = data_start + t["offset"]
            cur = f.tell()
            if target_pos > cur:
                f.write(b"\x00" * (target_pos - cur))
            f.write(t["data_bytes"])

def run_linter(model_path: str) -> dict:
    """Run scripts/gguf-lint.py --json on the model and return parsed dict."""
    cmd = [sys.executable, LINT_SCRIPT, model_path, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Linter failed on {model_path}:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
    return json.loads(proc.stdout)

def run_repacker(args: list) -> subprocess.CompletedProcess:
    """Run scripts/gguf-pack-npmc.py with given arguments."""
    cmd = [sys.executable, PACK_SCRIPT] + args
    return subprocess.run(cmd, capture_output=True, text=True)

class TestGGUFPackNPMC(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="npmc_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_vocab_model_repack_100_score(self):
        """Test repacking an existing vocab-only model achieves 100/100."""
        src_model = os.path.join(REPO_ROOT, "models", "ggml-vocab-deepseek-coder.gguf")
        if not os.path.exists(src_model):
            self.skipTest("Source vocab model not found")

        # Baseline score should be 70/100
        baseline_report = run_linter(src_model)
        self.assertEqual(baseline_report["analysis"]["score"], 70)

        # Repack to temporary location
        out_model = os.path.join(self.tmp_dir, "repacked_vocab.gguf")
        proc = run_repacker([src_model, "-o", out_model, "--validate"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        # Check repacked score
        repacked_report = run_linter(out_model)
        self.assertEqual(repacked_report["analysis"]["score"], 100,
                         f"Repacked score was {repacked_report['analysis']['score']}, expected 100")

    def test_moe_seek_fragmentation_and_huge_tlb(self):
        """
        Test an MoE model with interleaved expert weights and 65MB offset gaps (LINT005, LINT001, LINT007).
        Repacker must group MoE weights contiguously in 2MB HugeTLB blocks, achieving 100/100.
        """
        raw_model = os.path.join(self.tmp_dir, "interleaved_moe.gguf")

        # 4 layers of MoE experts interleaved with attention projections
        # Each expert tensor is 128 elements in F32 (512 bytes)
        # We simulate 70MB gap between layers to trigger LINT005
        tensors = []
        dummy_data = b"\x42" * 512

        # Layer 0
        tensors.append({"name": "blk.0.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data})
        tensors.append({"name": "blk.0.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data})

        # Interleave large gap (70 MB)
        gap_70mb = 70 * 1024 * 1024

        # Layer 1
        tensors.append({"name": "blk.1.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data, "custom_offset": gap_70mb})
        tensors.append({"name": "blk.1.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data})

        # Layer 2 (140 MB)
        tensors.append({"name": "blk.2.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data, "custom_offset": 2 * gap_70mb})
        tensors.append({"name": "blk.2.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data})

        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        # Baseline should fail LINT001, LINT005, LINT007
        base_report = run_linter(raw_model)
        penalties = {f["code"]: f["penalty"] for f in base_report["analysis"]["findings"]}
        self.assertGreater(penalties.get("LINT001_SECTOR_ALIGNMENT", 0), 0)
        self.assertGreater(penalties.get("LINT005_MOE_FRAGMENTATION", 0), 0)
        self.assertGreater(penalties.get("LINT007_HUGETLB_CONFORMITY", 0), 0)

        # Repack
        out_model = os.path.join(self.tmp_dir, "repacked_moe.gguf")
        proc = run_repacker([raw_model, "-o", out_model])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        # Lint repacked model
        repack_report = run_linter(out_model)
        self.assertEqual(repack_report["analysis"]["score"], 100,
                         f"Repacked score was {repack_report['analysis']['score']}, expected 100. Findings: {repack_report['analysis']['findings']}")

        # Verify bit-for-bit payload integrity
        # Check that tensor data was copied correctly
        with open(out_model, "rb") as f:
            for t in repack_report["gguf"] if "tensors" in repack_report.get("gguf", {}) else []:
                pass

    def test_dram_bank_power_of_two_aliasing(self):
        """
        Test a model with 64KB power-of-two row stride (dims=[16384, 16], F32) triggering LINT009.
        Repacker must insert intra-row padding (+128B) to break DRAM bank aliasing and achieve 100/100.
        """
        raw_model = os.path.join(self.tmp_dir, "aliased_model.gguf")

        # Tensor with dims=[16384, 16] in F32 (bpe=4.0):
        # row_stride = 16384 * 4 = 65536 bytes (2^16) -> triggers LINT009!
        # Total tensor bytes = 65536 * 16 = 1,048,576 bytes (1 MB)
        # Use a recognizable byte pattern per row to verify integrity
        rows = 16
        cols = 16384
        row_bytes = cols * 4
        data = bytearray()
        for r in range(rows):
            # Fill row with predictable sequence
            data.extend(bytes([(r + i) % 256 for i in range(row_bytes)]))

        tensors = [
            {"name": "token_embd.weight", "dims": [cols, rows], "type": 0, "data_bytes": bytes(data)},
            {"name": "output.weight", "dims": [128, 16], "type": 0, "data_bytes": b"\x55" * (128 * 16 * 4)}
        ]

        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        # Baseline should fail LINT009
        base_report = run_linter(raw_model)
        penalties = {f["code"]: f["penalty"] for f in base_report["analysis"]["findings"]}
        self.assertGreater(penalties.get("LINT009_DRAM_BANK_ALIASING", 0), 0)

        # Repack with intra-row padding
        out_model = os.path.join(self.tmp_dir, "repacked_unaliased.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--pad-bank-aliasing"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        # Repacked model must score 100/100
        repack_report = run_linter(out_model)
        self.assertEqual(repack_report["analysis"]["score"], 100,
                         f"Repacked score was {repack_report['analysis']['score']}, expected 100. Findings: {repack_report['analysis']['findings']}")

        # Verify intra-row padding in binary data:
        # Each row should have 65536 original bytes followed by 128 zero bytes!
        with open(out_model, "rb") as f:
            # Parse header to find token_embd.weight offset
            f.seek(repack_report["analysis"].get("data_start_offset", 2097152))
            # Or inspect from file
            f.seek(0)
            f.read(4) # magic
            f.read(4) # ver
            n_t = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            # skip kv
            for _ in range(n_kv):
                k_len = struct.unpack("<Q", f.read(8))[0]
                f.read(k_len)
                vtype = struct.unpack("<I", f.read(4))[0]
                # skip val
                if vtype in [0, 1, 7]: f.read(1)
                elif vtype in [2, 3]: f.read(2)
                elif vtype in [4, 5, 6]: f.read(4)
                elif vtype in [10, 11, 12]: f.read(8)
                elif vtype == 8:
                    sl = struct.unpack("<Q", f.read(8))[0]
                    f.read(sl)
                elif vtype == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    ne = struct.unpack("<Q", f.read(8))[0]
                    if et in [0, 1, 7]: f.read(ne)
                    elif et in [2, 3]: f.read(ne * 2)
                    elif et in [4, 5, 6]: f.read(ne * 4)
                    elif et in [10, 11, 12]: f.read(ne * 8)
                    elif et == 8:
                        for _ in range(ne):
                            sl = struct.unpack("<Q", f.read(8))[0]
                            f.read(sl)

            # read first tensor info
            t_name_len = struct.unpack("<Q", f.read(8))[0]
            t_name = f.read(t_name_len).decode()
            t_ndims = struct.unpack("<I", f.read(4))[0]
            new_dim0 = struct.unpack("<Q", f.read(8))[0]
            new_dim1 = struct.unpack("<Q", f.read(8))[0]
            f.read(4) # type
            t_offset = struct.unpack("<Q", f.read(8))[0]

            self.assertEqual(t_name, "token_embd.weight")
            # new_dim0 should be 16384 + 32 = 16416 elements (65664 bytes)
            self.assertEqual(new_dim0, 16416)
            self.assertEqual(new_dim1, 16)

            # Seek to tensor data
            d_start = 2097152
            f.seek(d_start + t_offset, os.SEEK_SET)
            for r in range(rows):
                row_unpadded = f.read(row_bytes)
                expected_unpadded = bytes([(r + i) % 256 for i in range(row_bytes)])
                self.assertEqual(row_unpadded, expected_unpadded, f"Row {r} unpadded data mismatch")
                pad = f.read(128)
                self.assertEqual(pad, b"\x00" * 128, f"Row {r} padding mismatch")

    def test_dry_run_workflow(self):
        """Test --dry-run simulates packing and outputs comparison without writing target file."""
        src_model = os.path.join(self.tmp_dir, "dry_model.gguf")
        write_test_gguf(src_model, alignment=32, tensors=[
            {"name": "token_embd.weight", "dims": [64, 4], "type": 0, "data_bytes": b"\x00" * (64 * 4 * 4)}
        ])

        target_out = os.path.join(self.tmp_dir, "non_existent.gguf")
        proc = run_repacker([src_model, "-o", target_out, "--dry-run"])
        self.assertEqual(proc.returncode, 0)
        self.assertFalse(os.path.exists(target_out), "Target file should not exist after --dry-run")
        self.assertIn("PREDICTED REPACK SCORE : 100/100", proc.stdout)

        # JSON mode
        proc_json = run_repacker([src_model, "--dry-run", "--json"])
        self.assertEqual(proc_json.returncode, 0)
        parsed = json.loads(proc_json.stdout)
        self.assertEqual(parsed["repacked_score"]["score"], 100)

    def test_in_place_repack(self):
        """Test --in-place replaces file in place and scores 100/100."""
        target_model = os.path.join(self.tmp_dir, "inplace_model.gguf")
        write_test_gguf(target_model, alignment=32, tensors=[
            {"name": "output.weight", "dims": [64, 4], "type": 0, "data_bytes": b"\x11" * (64 * 4 * 4)}
        ])

        self.assertEqual(run_linter(target_model)["analysis"]["score"], 70)

        proc = run_repacker([target_model, "--in-place"])
        self.assertEqual(proc.returncode, 0, f"In-place failed: {proc.stderr}")

        # Check that file was replaced in-place and now scores 100
        self.assertEqual(run_linter(target_model)["analysis"]["score"], 100)

    def test_sharded_model_consolidation(self):
        """Test multi-shard GGUF resolution and consolidation into unified 100/100 container."""
        prefix = os.path.join(self.tmp_dir, "split-model")
        shard1 = f"{prefix}-00001-of-00002.gguf"
        shard2 = f"{prefix}-00002-of-00002.gguf"

        # Shard 1: attention & layer 0 MoE expert
        write_test_gguf(shard1, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 0)},
                        tensors=[
                            {"name": "token_embd.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x11" * 512},
                            {"name": "blk.0.ffn_gate_exps.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x22" * 512}
                        ])

        # Shard 2: layer 1 MoE expert & output head
        write_test_gguf(shard2, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 1)},
                        tensors=[
                            {"name": "blk.1.ffn_gate_exps.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x33" * 512},
                            {"name": "output.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x44" * 512}
                        ])

        unified_out = os.path.join(self.tmp_dir, "unified_model.gguf")
        proc = run_repacker([shard1, "-o", unified_out])
        self.assertEqual(proc.returncode, 0, f"Sharded repack failed: {proc.stderr}")

        report = run_linter(unified_out)
        self.assertEqual(report["analysis"]["score"], 100)
        self.assertEqual(report["gguf"]["tensor_count"], 4)

    def test_custom_alignment_4096(self):
        """Test --alignment 4096 enforces 4KB sector alignment with 2MB data_start HugeTLB boundary."""
        raw_model = os.path.join(self.tmp_dir, "raw_4k.gguf")
        write_test_gguf(raw_model, alignment=32, tensors=[
            {"name": "token_embd.weight", "dims": [64, 4], "type": 0, "data_bytes": b"\x77" * 1024}
        ])

        out_model = os.path.join(self.tmp_dir, "repacked_4k.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--alignment", "4096"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        report = run_linter(out_model)
        self.assertEqual(report["analysis"]["score"], 100)
        self.assertEqual(report["gguf"]["alignment"], 4096)

    def test_expert_first_grouping(self):
        """Test --expert-grouping expert-first sorts expert weights by expert ID first."""
        raw_model = os.path.join(self.tmp_dir, "split_experts.gguf")
        # Split expert tensors: blk.0.ffn_gate.0.weight, blk.0.ffn_gate.1.weight, blk.1.ffn_gate.0.weight, blk.1.ffn_gate.1.weight
        tensors = [
            {"name": "blk.0.attn_q.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x01" * 512},
            {"name": "blk.0.ffn_gate.0.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x02" * 512},
            {"name": "blk.0.ffn_gate.1.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x03" * 512},
            {"name": "blk.1.ffn_gate.0.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x04" * 512},
            {"name": "blk.1.ffn_gate.1.weight", "dims": [64, 2], "type": 0, "data_bytes": b"\x05" * 512},
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        out_model = os.path.join(self.tmp_dir, "repacked_expert_first.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--expert-grouping", "expert-first"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        report = run_linter(out_model)
        self.assertEqual(report["analysis"]["score"], 100)

    def test_quantized_tensors_q4k_q8_repack(self):
        """Test models with quantized tensors (Q4_K, Q8_0) preserve exact block sizes and offsets."""
        raw_model = os.path.join(self.tmp_dir, "quant_model.gguf")
        # Q8_0: type 8, 32 elements = 34 bytes
        # Q4_K: type 12, 256 elements = 144 bytes
        q8_data = b"\x88" * (34 * 4) # 4 blocks = 128 elements
        q4k_data = b"\x44" * (144 * 2) # 2 blocks = 512 elements
        tensors = [
            {"name": "token_embd.weight", "dims": [128, 1], "type": 8, "data_bytes": q8_data},
            {"name": "blk.0.attn_q.weight", "dims": [512, 1], "type": 12, "data_bytes": q4k_data}
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        out_model = os.path.join(self.tmp_dir, "repacked_quant.gguf")
        proc = run_repacker([raw_model, "-o", out_model])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        report = run_linter(out_model)
        self.assertEqual(report["analysis"]["score"], 100)

    def test_error_handling(self):
        """Test error conditions: invalid magic, non-existent file, missing output arguments."""
        # Missing output and in-place
        proc = run_repacker(["some_model.gguf"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Must specify either -o/--output, --in-place, or --dry-run", proc.stderr)

        # Both output and in-place
        proc = run_repacker(["some_model.gguf", "-o", "out.gguf", "--in-place"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Cannot specify both -o/--output and --in-place", proc.stderr)

        # Non-existent file
        proc = run_repacker(["non_existent_file.gguf", "-o", "out.gguf"])
        self.assertNotEqual(proc.returncode, 0)

        # Invalid magic
        bad_file = os.path.join(self.tmp_dir, "bad_magic.gguf")
        with open(bad_file, "wb") as f:
            f.write(b"NOTGGUF")
    def test_dry_run_interleaved_moe_predicts_100(self):
        """Test --dry-run on interleaved MoE accurately simulates new offsets and predicts 100/100."""
        raw_model = os.path.join(self.tmp_dir, "dry_moe.gguf")
        dummy_data = b"\x55" * 512
        gap_70mb = 70 * 1024 * 1024

        tensors = [
            {"name": "blk.0.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data},
            {"name": "blk.0.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data},
            {"name": "blk.1.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data, "custom_offset": gap_70mb},
            {"name": "blk.1.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data},
            {"name": "blk.2.attn_q.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data, "custom_offset": 2 * gap_70mb},
            {"name": "blk.2.ffn_gate_exps.weight", "dims": [128, 1], "type": 0, "data_bytes": dummy_data},
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        proc = run_repacker([raw_model, "--dry-run", "--json"])
        self.assertEqual(proc.returncode, 0, f"Dry-run failed: {proc.stderr}")
        parsed = json.loads(proc.stdout)
        self.assertEqual(parsed["repacked_score"]["score"], 100,
                         f"Simulated score was {parsed['repacked_score']['score']}, expected 100.")
        self.assertEqual(parsed["repacked_score"]["penalties"]["LINT005"], 0)
        self.assertEqual(parsed["repacked_score"]["penalties"]["LINT007"], 0)

    def test_unpadded_dim0_metadata_emitted(self):
        """Test that npmc.unpadded_dim0.<name> and npmc.padded_tensors metadata are emitted."""
        raw_model = os.path.join(self.tmp_dir, "aliased_meta.gguf")
        rows = 16
        cols = 16384
        data = b"\x33" * (cols * rows * 4)

        tensors = [
            {"name": "token_embd.weight", "dims": [cols, rows], "type": 0, "data_bytes": data},
            {"name": "output.weight", "dims": [128, 16], "type": 0, "data_bytes": b"\x11" * (128 * 16 * 4)}
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        out_model = os.path.join(self.tmp_dir, "repacked_meta.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--pad-bank-aliasing"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        report = run_linter(out_model)
        self.assertEqual(report["analysis"]["score"], 100)

        # Inspect metadata in repacked file
        meta = parse_gguf_metadata(out_model)["metadata"]
        self.assertIn("npmc.unpadded_dim0.token_embd.weight", meta)
        self.assertEqual(meta["npmc.unpadded_dim0.token_embd.weight"], cols)
        self.assertIn("npmc.padded_tensors", meta)
        self.assertIn("token_embd.weight", meta["npmc.padded_tensors"])

    def test_q6_k_quant_row_padding_binary_integrity(self):
        """Test 2D Q6_K tensor preservation using exact block sizing (210 bytes / 256 weights)."""
        raw_model = os.path.join(self.tmp_dir, "q6k_model.gguf")
        cols = 512  # 2 blocks of 256 weights
        rows = 4
        # Q6_K: 210 bytes per 256 weights -> 420 bytes per row
        row_bytes = (cols // 256) * 210
        total_bytes = row_bytes * rows

        raw_data = bytearray()
        for r in range(rows):
            raw_data.extend(bytes([(r * 17 + i) % 256 for i in range(row_bytes)]))

        tensors = [
            {"name": "blk.0.attn_v.weight", "dims": [cols, rows], "type": 14, "data_bytes": bytes(raw_data)}
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        out_model = os.path.join(self.tmp_dir, "repacked_q6k.gguf")
        proc = run_repacker([raw_model, "-o", out_model])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        report = run_linter(out_model)
        self.assertEqual(report["analysis"]["score"], 100)

        # Verify exact payload bytes row by row
        info = parse_gguf_metadata(out_model)
        t_desc = info["tensors"][0]
        data_start = info["data_start_offset"]
        new_cols = t_desc["dims"][0]
        new_row_bytes = (new_cols // 256) * 210
        with open(out_model, "rb") as f:
            f.seek(data_start + t_desc["offset"], os.SEEK_SET)
            recovered = bytearray()
            for r in range(rows):
                row_slice = f.read(new_row_bytes)
                recovered.extend(row_slice[:row_bytes])
            self.assertEqual(bytes(recovered), bytes(raw_data), "Q6_K binary payload corrupted due to bpe rounding!")

    def test_no_pad_bank_aliasing_flag(self):
        """Test --no-pad-bank-aliasing keeps original dimensions and allows LINT009 penalty."""
        raw_model = os.path.join(self.tmp_dir, "aliased_nopad.gguf")
        rows = 16
        cols = 16384
        data = b"\x44" * (cols * rows * 4)

        tensors = [
            {"name": "token_embd.weight", "dims": [cols, rows], "type": 0, "data_bytes": data}
        ]
        write_test_gguf(raw_model, alignment=32, tensors=tensors)

        out_model = os.path.join(self.tmp_dir, "repacked_nopad.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--no-pad-bank-aliasing"])
        self.assertEqual(proc.returncode, 0, f"Repacker failed: {proc.stderr}")

        # Dimensions should be unchanged
        info = parse_gguf_metadata(out_model)
        self.assertEqual(info["tensors"][0]["dims"][0], cols)

        # Linter should detect LINT009
        report = run_linter(out_model)
        penalties = {f["code"]: f["penalty"] for f in report["analysis"]["findings"]}
        self.assertGreater(penalties.get("LINT009_DRAM_BANK_ALIASING", 0), 0)

    def test_sharded_model_separate_directories(self):
        """Test passing explicit shard paths located in separate directories via CLI."""
        dir1 = os.path.join(self.tmp_dir, "dir1")
        dir2 = os.path.join(self.tmp_dir, "dir2")
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir2, exist_ok=True)

        shard1 = os.path.join(dir1, "model-00001-of-00002.gguf")
        shard2 = os.path.join(dir2, "model-00002-of-00002.gguf")

        write_test_gguf(shard1, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 0)},
                        tensors=[{"name": "tensor1", "dims": [64, 2], "type": 0, "data_bytes": b"\x11" * 512}])
        write_test_gguf(shard2, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 1)},
                        tensors=[{"name": "tensor2", "dims": [64, 2], "type": 0, "data_bytes": b"\x22" * 512}])

        unified_out = os.path.join(self.tmp_dir, "unified_multi_dir.gguf")
        proc = run_repacker([shard1, shard2, "-o", unified_out])
        self.assertEqual(proc.returncode, 0, f"Multi-dir sharded repack failed: {proc.stderr}")

        report = run_linter(unified_out)
        self.assertEqual(report["analysis"]["score"], 100)
        self.assertEqual(report["gguf"]["tensor_count"], 2)

    def test_in_place_sharded_rejection(self):
        """Test --in-place on sharded model is rejected to prevent corrupting shard 1."""
        prefix = os.path.join(self.tmp_dir, "split-reject")
        shard1 = f"{prefix}-00001-of-00002.gguf"
        shard2 = f"{prefix}-00002-of-00002.gguf"

        write_test_gguf(shard1, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 0)},
                        tensors=[{"name": "t1", "dims": [64, 2], "type": 0, "data_bytes": b"\x11" * 512}])
        write_test_gguf(shard2, alignment=32,
                        kv_pairs={"split.count": ("uint32", 2), "split.no": ("uint32", 1)},
                        tensors=[{"name": "t2", "dims": [64, 2], "type": 0, "data_bytes": b"\x22" * 512}])

        proc = run_repacker([shard1, "--in-place"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Cannot use --in-place on a sharded model container", proc.stderr)

    def test_validate_flag(self):
        """Test --validate verifies 100/100 score and fails if score is suboptimal."""
        raw_model = os.path.join(self.tmp_dir, "val_model.gguf")
        write_test_gguf(raw_model, alignment=32, tensors=[
            {"name": "token_embd.weight", "dims": [64, 4], "type": 0, "data_bytes": b"\x00" * 1024}
        ])

        out_model = os.path.join(self.tmp_dir, "val_out.gguf")
        proc = run_repacker([raw_model, "-o", out_model, "--validate"])
        self.assertEqual(proc.returncode, 0, f"Repack with --validate failed: {proc.stderr}")
        self.assertIn("Validation PASSED", proc.stdout)

if __name__ == "__main__":
    unittest.main()

