#!/usr/bin/env python3
"""
GGUF Hardware Constraint Linter & Performance Scorer
Analyzes GGUF model files against local host hardware constraints (VRAM, NUMA, DDR/PCIe bandwidth, Direct I/O)
and provides actionable optimization recommendations.
"""

import sys
import os
import struct
import argparse
import json
import re

# GGUF Magic and Types
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types & Bitrates (approximate bytes per weight)
GGML_TYPE_NAMES = {
    0: ("F32", 4.0),
    1: ("F16", 2.0),
    2: ("Q4_0", 0.5625),
    3: ("Q4_1", 0.625),
    6: ("Q5_0", 0.6875),
    7: ("Q5_1", 0.75),
    8: ("Q8_0", 1.0625),
    9: ("Q8_1", 1.125),
    10: ("Q2_K", 0.328125),
    11: ("Q3_K", 0.4296875),
    12: ("Q4_K", 0.5625),
    13: ("Q5_K", 0.6875),
    14: ("Q6_K", 0.828125),
    15: ("Q8_K", 1.0625),
    16: ("IQ2_XXS", 0.2578125),
    17: ("IQ2_XS", 0.2890625),
    18: ("IQ3_XXS", 0.3828125),
    19: ("IQ1_S", 0.1953125),
    20: ("IQ4_NL", 0.5625),
    21: ("IQ3_S", 0.4296875),
    22: ("IQ2_S", 0.3125),
    23: ("IQ4_XS", 0.53125),
    24: ("I8", 1.0),
    25: ("I16", 2.0),
    26: ("I32", 4.0),
    27: ("I64", 8.0),
    28: ("F64", 8.0),
    29: ("IQ1_M", 0.21875),
    30: ("BF16", 2.0),
    41: ("Q1_0", 0.1484375),
    66: ("Q1_0", 0.1484375),
    133: ("Q6_0", 0.8125),
    137: ("IQ2_K", 0.296875),
    138: ("IQ3_K", 0.4296875),
    139: ("IQ4_K", 0.5625),
    144: ("IQ4_KS", 0.53125),
    146: ("IQ4_KSS", 0.5),
}

# GGML Quantization Sizes: (block_size, type_size_bytes)
GGML_QUANT_SIZES = {
    0: (1, 4),        # F32
    1: (1, 2),        # F16
    2: (32, 18),      # Q4_0
    3: (32, 20),      # Q4_1
    6: (32, 22),      # Q5_0
    7: (32, 24),      # Q5_1
    8: (32, 34),      # Q8_0
    9: (32, 36),      # Q8_1
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    15: (256, 292),   # Q8_K
    16: (256, 66),    # IQ2_XXS
    17: (256, 74),    # IQ2_XS
    18: (256, 98),    # IQ3_XXS
    19: (256, 50),    # IQ1_S
    20: (32, 18),     # IQ4_NL
    21: (256, 110),   # IQ3_S
    22: (256, 82),    # IQ2_S
    23: (256, 136),   # IQ4_XS
    24: (1, 1),       # I8
    25: (1, 2),       # I16
    26: (1, 4),       # I32
    27: (1, 8),       # I64
    28: (1, 8),       # F64
    29: (256, 56),    # IQ1_M
    30: (1, 2),       # BF16
    41: (128, 18),    # Q1_0_G128
    66: (128, 18),    # Q1_0
    133: (32, 26),    # Q6_0
    137: (256, 76),   # IQ2_K
    138: (256, 110),  # IQ3_K
    139: (256, 144),  # IQ4_K
    144: (256, 136),  # IQ4_KS
    146: (256, 128),  # IQ4_KSS
}

def get_tensor_row_bytes(type_id: int, ne0: int, bpe: float = 2.0) -> int:
    """Computes exact row byte size for a tensor type using integer block arithmetic."""
    if type_id in GGML_QUANT_SIZES:
        blk_size, type_size = GGML_QUANT_SIZES[type_id]
        n_blocks = (ne0 + blk_size - 1) // blk_size
        return n_blocks * type_size
    return int(ne0 * bpe)


def read_str(f):
    len_bytes = f.read(8)
    if len(len_bytes) < 8:
        return ""
    length = struct.unpack("<Q", len_bytes)[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_val(f, val_type):
    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    elif val_type == GGUF_TYPE_STRING:
        return read_str(f)
    elif val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        n_elems = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(min(n_elems, 1024)):
            arr.append(read_val(f, elem_type))
        if n_elems > 1024:
            # Skip remainder
            for _ in range(n_elems - 1024):
                read_val(f, elem_type)
        return arr
    else:
        return None

def parse_gguf_metadata(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError("File too short for GGUF")
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: 0x{magic:08x} (expected 0x{GGUF_MAGIC:08x})")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(metadata_kv_count):
            key = read_str(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = read_val(f, val_type)
            metadata[key] = val

        alignment = metadata.get("general.alignment", 32)

        tensors = []
        for _ in range(tensor_count):
            name = read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack("<Q", f.read(8))[0])
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            tname, bpe = GGML_TYPE_NAMES.get(ttype, (f"TYPE_{ttype}", 2.0))
            num_elements = 1
            for d in dims:
                num_elements *= d
            if len(dims) > 0 and ttype in GGML_QUANT_SIZES:
                row_bytes = get_tensor_row_bytes(ttype, dims[0], bpe)
                num_rows = 1
                for d in dims[1:]:
                    num_rows *= d
                approx_bytes = row_bytes * num_rows
            else:
                approx_bytes = int(num_elements * bpe)

            tensors.append({
                "name": name,
                "dims": dims,
                "type_id": ttype,
                "type_name": tname,
                "offset": offset,
                "elements": num_elements,
                "approx_bytes": approx_bytes
            })

        header_size = f.tell()
        # In GGUF, tensor data starts at first aligned offset after header
        data_start_offset = (header_size + alignment - 1) & ~(alignment - 1)

        # Refine tensor byte sizes from sequential offsets within the file
        if len(tensors) > 0:
            sorted_indices = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
            for idx_pos, orig_idx in enumerate(sorted_indices):
                if idx_pos + 1 < len(sorted_indices):
                    next_orig_idx = sorted_indices[idx_pos + 1]
                    delta = tensors[next_orig_idx]["offset"] - tensors[orig_idx]["offset"]
                    if delta > 0:
                        tensors[orig_idx]["approx_bytes"] = delta
                else:
                    remaining = file_size - data_start_offset - tensors[orig_idx]["offset"]
                    if remaining > 0:
                        tensors[orig_idx]["approx_bytes"] = remaining

    return {
        "file_size": file_size,
        "version": version,
        "tensor_count": tensor_count,
        "alignment": alignment,
        "data_start_offset": data_start_offset,
        "metadata": metadata,
        "tensors": tensors
    }

def resolve_gguf_shards(filepath):
    """Detect whether a GGUF file is part of a multi-file sharded split and return all sibling shard paths."""
    dirname = os.path.dirname(os.path.abspath(filepath))
    basename = os.path.basename(filepath)

    # 1. Regex check for standard llama.cpp / unsloth pattern: <prefix>-00001-of-00010.gguf
    m = re.match(r"^(.*?)-(\d{5})-of-(\d{5})\.gguf$", basename)
    if m:
        prefix, cur_idx, total_shards = m.group(1), int(m.group(2)), int(m.group(3))
        shards = []
        for i in range(1, total_shards + 1):
            s_name = f"{prefix}-{i:05d}-of-{total_shards:05d}.gguf"
            s_path = os.path.join(dirname, s_name)
            if os.path.exists(s_path):
                shards.append(s_path)
        if len(shards) == total_shards:
            return shards

    # 2. Check metadata from target file
    try:
        single_meta = parse_gguf_metadata(filepath)
        split_count = single_meta["metadata"].get("split.count", 1)
        if split_count > 1:
            base_pattern = re.sub(r"-\d{5}(-of-\d{5})?\.gguf$", "", basename)
            candidates = sorted([
                os.path.join(dirname, f) for f in os.listdir(dirname)
                if f.startswith(base_pattern) and f.endswith(".gguf")
            ])
            if len(candidates) >= split_count:
                return candidates[:split_count]
    except Exception:
        pass

    return [filepath]

def load_unified_gguf_model(filepath):
    """Load and unify single-file or multi-file sharded GGUF models."""
    shards = resolve_gguf_shards(filepath)
    if len(shards) == 1:
        info = parse_gguf_metadata(shards[0])
        info["is_sharded"] = False
        info["shard_files"] = [shards[0]]
        info["primary_file"] = shards[0]
        return info

    total_file_size = 0
    all_tensors = []
    merged_metadata = {}
    version = 3
    alignment = 32
    data_start_offset = 0

    shard_stats = []
    for idx, shard_path in enumerate(shards):
        s_info = parse_gguf_metadata(shard_path)
        s_size = s_info["file_size"]
        total_file_size += s_size

        if idx == 0:
            version = s_info["version"]
            alignment = s_info["alignment"]
            data_start_offset = s_info["data_start_offset"]
            merged_metadata.update(s_info["metadata"])
        else:
            for k, v in s_info["metadata"].items():
                if k not in merged_metadata:
                    merged_metadata[k] = v

        for t in s_info["tensors"]:
            t["shard_idx"] = idx + 1
            t["shard_file"] = os.path.basename(shard_path)
            t["shard_data_start"] = s_info["data_start_offset"]
            all_tensors.append(t)

        shard_stats.append({
            "shard_file": os.path.basename(shard_path),
            "size_bytes": s_size,
            "tensor_count": len(s_info["tensors"])
        })

    return {
        "is_sharded": True,
        "shard_count": len(shards),
        "shard_files": shards,
        "shard_stats": shard_stats,
        "primary_file": shards[0],
        "file_size": total_file_size,
        "version": version,
        "tensor_count": len(all_tensors),
        "alignment": alignment,
        "data_start_offset": data_start_offset,
        "metadata": merged_metadata,
        "tensors": all_tensors
    }

def probe_host_hardware():
    """Discover host hardware capabilities or provide dual-socket EPYC defaults."""
    hw = {
        "gpu_vram_bytes": 0,
        "gpu_name": "None",
        "numa_nodes": 1,
        "total_ram_bytes": 0,
        "storage_is_direct_nvme": False
    }

    # Probe RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    hw["total_ram_bytes"] = int(line.split()[1]) * 1024
    except Exception:
        hw["total_ram_bytes"] = 1000 * (1024**3)

    # Probe NUMA nodes
    try:
        node_dirs = [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
        hw["numa_nodes"] = max(1, len(node_dirs))
    except Exception:
        hw["numa_nodes"] = 8

    # Probe GPU
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            line = out.split("\n")[0]
            name, mem_mb = line.split(",")
            hw["gpu_name"] = name.strip()
            hw["gpu_vram_bytes"] = int(mem_mb.strip()) * 1024 * 1024
    except Exception:
        # Default to 24 GB RTX 4090 if nvidia-smi unavailable but system is our host
        hw["gpu_name"] = "NVIDIA GeForce RTX 4090"
        hw["gpu_vram_bytes"] = 24 * (1024**3)

    return hw

def analyze_and_lint_gguf(gguf_info, hw):
    findings = []
    score = 100

    file_size = gguf_info["file_size"]
    alignment = gguf_info["alignment"]
    tensors = gguf_info["tensors"]
    data_start = gguf_info["data_start_offset"]

    # 1. Alignment & Direct I/O (io_uring O_DIRECT) Check
    is_4k_aligned = (alignment % 4096 == 0) and (data_start % 4096 == 0)
    unaligned_offsets = 0
    for t in tensors:
        abs_offset = data_start + t["offset"]
        if abs_offset % 4096 != 0:
            unaligned_offsets += 1

    if not is_4k_aligned or unaligned_offsets > 0:
        penalty = 15
        score -= penalty
        findings.append({
            "code": "LINT001_SECTOR_ALIGNMENT",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "Suboptimal Tensor Alignment for Zero-Copy Direct I/O",
            "detail": f"GGUF alignment is {alignment} bytes ({unaligned_offsets} tensors not 4096-byte sector aligned). "
                      f"Prevents io_uring O_DIRECT zero-copy DMA bypass of Linux page cache. Every read must pass through page cache copy.",
            "recommendation": "Repack GGUF with 4096-byte alignment (`--alignment 4096`) to enable direct NVMe-to-RAM DMA at 28+ GB/s."
        })
    else:
        findings.append({
            "code": "LINT001_SECTOR_ALIGNMENT",
            "severity": "PASS",
            "penalty": 0,
            "title": "4096-Byte Sector Alignment Verified",
            "detail": "All tensor boundaries align to 4096-byte sector boundaries. Compatible with io_uring O_DIRECT.",
            "recommendation": "Enable io_uring O_DIRECT for zero-copy model loading."
        })

    # Categorize Tensors into Critical Path (Attention, Embeddings, Shared FFN, Router) vs Routed MoE Experts
    critical_tensors = []
    moe_expert_tensors = []
    other_tensors = []

    critical_bytes = 0
    moe_bytes = 0

    for t in tensors:
        name = t["name"]
        sz = t["approx_bytes"]
        if re.search(r"\.ffn_.*_exps\.", name) or re.search(r"blk\.\d+\.ffn_(gate|up|down)\.\d+", name) or re.search(r"expert", name, re.IGNORECASE):
            moe_expert_tensors.append(t)
            moe_bytes += sz
        elif any(k in name for k in ["attn_", "attention", "router", "ffn_shared", "embed", "output"]):
            critical_tensors.append(t)
            critical_bytes += sz
        else:
            other_tensors.append(t)

    # 2. GPU VRAM Fit Analysis
    vram_usable = hw["gpu_vram_bytes"] * 0.90 # Leave 10% for KV cache / CUDA context
    if hw["gpu_vram_bytes"] > 0:
        if critical_bytes > vram_usable:
            penalty = 25
            score -= penalty
            findings.append({
                "code": "LINT002_VRAM_CRITICAL_OVERFLOW",
                "severity": "CRITICAL",
                "penalty": penalty,
                "title": "Attention & Critical-Path Weights Exceed GPU VRAM",
                "detail": f"Critical path weights (attention, router, shared FFN, embeddings) require {critical_bytes / (1024**3):.2f} GiB, "
                          f"exceeding 90% usable VRAM ({vram_usable / (1024**3):.2f} GiB of {hw['gpu_vram_bytes'] / (1024**3):.1f} GiB). "
                          f"Forcing attention layers onto CPU degrades latency by 5x-10x.",
                "recommendation": "Quantize attention/router layers to Q4_K, FP8, or Q8_0 so attention stays 100% VRAM resident on GPU."
            })
        else:
            findings.append({
                "code": "LINT002_VRAM_CRITICAL_FIT",
                "severity": "PASS",
                "penalty": 0,
                "title": "Critical Attention Path Fits in GPU VRAM",
                "detail": f"Critical path weights require {critical_bytes / (1024**3):.2f} GiB (fits inside {vram_usable / (1024**3):.2f} GiB usable VRAM).",
                "recommendation": "Offload attention, router, and shared FFN to CUDA0 via `-sm graph -ot \"^.*\\.attn.*=CUDA0,.*\\.router.*=CUDA0,.*\\.ffn_shared.*=CUDA0\"`."
            })

    # 3. MoE Expert Distribution & NUMA Interleaving
    if len(moe_expert_tensors) > 0:
        numa_nodes = hw["numa_nodes"]
        bytes_per_node = moe_bytes / numa_nodes
        ram_per_node = hw["total_ram_bytes"] / numa_nodes

        if bytes_per_node > ram_per_node * 0.85:
            penalty = 20
            score -= penalty
            findings.append({
                "code": "LINT003_NUMA_RAM_EXHAUSTION",
                "severity": "CRITICAL",
                "penalty": penalty,
                "title": "Routed MoE Experts Exceed Per-Node RAM Capacity",
                "detail": f"Routed experts require {moe_bytes / (1024**3):.1f} GiB total ({bytes_per_node / (1024**3):.1f} GiB/node across {numa_nodes} nodes), "
                          f"exceeding safe budget of {ram_per_node / (1024**3):.1f} GiB/node.",
                "recommendation": "Use lower-bit IQ quants (IQ2_XXS / IQ1_S) for routed experts or enable NVMe prefetch streaming."
            })
        else:
            findings.append({
                "code": "LINT003_NUMA_RAM_FIT",
                "severity": "PASS",
                "penalty": 0,
                "title": "Routed MoE Experts Fit Across NUMA Nodes",
                "detail": f"MoE experts ({moe_bytes / (1024**3):.1f} GiB total) divide cleanly into ~{bytes_per_node / (1024**3):.1f} GiB/node across {numa_nodes} nodes.",
                "recommendation": "Launch with `--numa split -ot \"^blk\\..*\\.ffn_.*_exps.*=CPU\"` to interleave experts across all memory channels."
            })

    # 4. Quantization Type Suitability (GPU vs CPU)
    gpu_non_tensor_core_quants = []
    for t in critical_tensors:
        if t["type_name"] in ["IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XXS"]:
            gpu_non_tensor_core_quants.append(t["name"])

    if len(gpu_non_tensor_core_quants) > 0:
        penalty = 10
        score -= penalty
        findings.append({
            "code": "LINT004_QUANT_COMPUTE_MISMATCH",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "Suboptimal Quantization for GPU Tensor Cores",
            "detail": f"Found {len(gpu_non_tensor_core_quants)} critical-path tensors formatted in LUT-based IQ quants ({gpu_non_tensor_core_quants[0]}...). "
                      f"Ada Lovelace Tensor Cores cannot execute LUT quants natively; GPU falls back to ALU bit-unpacking emulation.",
            "recommendation": "Re-quantize attention projections with Q4_K_M or Q8_0 for native Tensor Core acceleration."
        })
    else:
        findings.append({
            "code": "LINT004_QUANT_COMPUTE_MATCH",
            "severity": "PASS",
            "penalty": 0,
            "title": "Quantization Matches Hardware Execution Units",
            "detail": "Critical attention weights use Tensor-Core compatible formats (Q4_K / Q8_0 / FP16).",
            "recommendation": "Enable FP16/INT8 Tensor Core GEMM kernels."
        })

    # 5. MoE Expert Seek Locality
    if len(moe_expert_tensors) > 1:
        # Check if expert tensors are grouped contiguously
        scattered_seeks = 0
        last_offset = 0
        for t in moe_expert_tensors:
            if last_offset > 0 and abs(t["offset"] - last_offset) > (64 * 1024 * 1024):
                scattered_seeks += 1
            last_offset = t["offset"]

        if scattered_seeks > len(moe_expert_tensors) * 0.5:
            penalty = 10
            score -= penalty
            findings.append({
                "code": "LINT005_MOE_FRAGMENTATION",
                "severity": "WARNING",
                "penalty": penalty,
                "title": "Layer-by-Layer Interleaved MoE Weights Cause Seek Thrashing",
                "detail": f"Routed expert tensors are interleaved layer-by-layer across the file ({scattered_seeks} non-contiguous offset jumps). "
                          f"Streaming expert weights from NVMe requires discontinuous seeks across hundreds of gigabytes.",
                "recommendation": "Use contiguous expert grouping or sequential prefetch staging ring buffer."
            })

    # 6. Cache Line Split Invariant (alignas(128))
    misaligned_128_offsets = 0
    misaligned_128_strides = 0
    for t in tensors:
        abs_off = data_start + t["offset"]
        if abs_off % 128 != 0:
            misaligned_128_offsets += 1
        dims = t["dims"]
        if len(dims) >= 2:
            bpe = GGML_TYPE_NAMES.get(t["type_id"], (None, 2.0))[1]
            row_stride = get_tensor_row_bytes(t["type_id"], dims[0], bpe)
            if row_stride % 128 != 0:
                misaligned_128_strides += 1

    if misaligned_128_offsets > 0 or misaligned_128_strides > 0:
        penalty = 10
        score -= penalty
        findings.append({
            "code": "LINT006_CACHE_LINE_SPLIT",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "Cache Line Boundary Split Invariant Violated (alignas(128))",
            "detail": f"{misaligned_128_offsets} tensors have base offsets not aligned to 128 bytes, "
                      f"and {misaligned_128_strides} 2D matrices have row strides not divisible by 128 bytes. "
                      f"On AMD Zen 3/4/5 dual-line spatial prefetchers, unaligned rows trigger split cache-line accesses and bus contention.",
            "recommendation": "Pad row strides and tensor descriptors to 128-byte multiples with explicit alignas(128) intra-tensor padding."
        })
    else:
        findings.append({
            "code": "LINT006_CACHE_LINE_SPLIT",
            "severity": "PASS",
            "penalty": 0,
            "title": "128-Byte Cache Line Boundary Alignment Verified",
            "detail": "All tensor base offsets and row strides align to 128-byte dual-cache-line boundaries. Zero split-cache line prefetch penalties.",
            "recommendation": "Maintain alignas(128) memory layouts."
        })

    # 7. HugeTLB 2MB / 1GB Boundary Conformity
    if data_start % (2 * 1024 * 1024) != 0 or alignment % 4096 != 0:
        penalty = 15
        score -= penalty
        findings.append({
            "code": "LINT007_HUGETLB_CONFORMITY",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "Suboptimal HugeTLB 2MB Boundary Conformity",
            "detail": f"GGUF container data section start (offset {data_start}) is not 2MB aligned ({data_start % (2*1024*1024)} bytes remainder). "
                      f"Prevents the Linux kernel from cleanly backing mapped weight segments with Transparent Huge Pages (THP) or explicit HugeTLB (MAP_HUGETLB), "
                      f"incurring STLB walks and dTLB miss penalties across hundreds of gigabytes.",
            "recommendation": "Repack container segments with 2MB alignment (`--alignment 2097152`) and enable madvise(..., MADV_HUGEPAGE)."
        })
    else:
        findings.append({
            "code": "LINT007_HUGETLB_CONFORMITY",
            "severity": "PASS",
            "penalty": 0,
            "title": "HugeTLB 2MB Boundary Alignment Verified",
            "detail": f"GGUF container data section start (offset {data_start}) is aligned to 2MB boundaries for optimal HugeTLB / THP mapping.",
            "recommendation": "Launch engine with --thp / MAP_HUGETLB."
        })

    # 8. SIMD Vector Register 64-Byte Alignment
    simd_unaligned_crit = 0
    for t in critical_tensors:
        abs_off = data_start + t["offset"]
        dims = t["dims"]
        bpe = GGML_TYPE_NAMES.get(t["type_id"], (None, 2.0))[1]
        row_stride = get_tensor_row_bytes(t["type_id"], dims[0], bpe) if len(dims) >= 2 else (
            get_tensor_row_bytes(t["type_id"], t["elements"], bpe) if "elements" in t else int(dims[0] * bpe)
        )
        if abs_off % 64 != 0 or row_stride % 64 != 0:
            simd_unaligned_crit += 1

    if simd_unaligned_crit > 0:
        penalty = 10
        score -= penalty
        findings.append({
            "code": "LINT008_SIMD_ALIGNMENT",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "Critical Attention Weights Violate 64-Byte SIMD Vector Alignment",
            "detail": f"Found {simd_unaligned_crit} critical attention tensors with base address or row stride not divisible by 64 bytes. "
                      f"Unaligned AVX-512 / AVX2 vector SIMD loads crossing 64-byte boundaries incur split-load merge penalties and pipeline bubbles.",
            "recommendation": "Pad critical tensor row strides and offsets to 64-byte multiples (vmovdqa64 single-cycle load)."
        })
    else:
        findings.append({
            "code": "LINT008_SIMD_ALIGNMENT",
            "severity": "PASS",
            "penalty": 0,
            "title": "64-Byte SIMD Vector Alignment Verified",
            "detail": "All critical attention tensors and row strides align to 64-byte boundaries for single-cycle vector loads.",
            "recommendation": "Compile with AVX-512 / AVX2 aligned load intrinsics."
        })

    # 9. DRAM Bank Power-of-Two Stride Aliasing
    bank_aliasing_tensors = []
    for t in tensors:
        dims = t["dims"]
        if len(dims) >= 2 and dims[1] >= 16:
            bpe = GGML_TYPE_NAMES.get(t["type_id"], (None, 2.0))[1]
            row_stride = get_tensor_row_bytes(t["type_id"], dims[0], bpe)
            # Power of two >= 65536 bytes (64 KB)
            if row_stride >= 65536 and (row_stride & (row_stride - 1)) == 0:
                bank_aliasing_tensors.append((t["name"], row_stride))

    if len(bank_aliasing_tensors) > 0:
        penalty = 10
        score -= penalty
        findings.append({
            "code": "LINT009_DRAM_BANK_ALIASING",
            "severity": "WARNING",
            "penalty": penalty,
            "title": "DRAM Bank Power-of-Two Stride Aliasing Conflict",
            "detail": f"Found {len(bank_aliasing_tensors)} tensors (e.g. {bank_aliasing_tensors[0][0]} with stride {bank_aliasing_tensors[0][1]} bytes) "
                      f"with power-of-two row strides >= 64 KB. Under concurrent multi-channel DDR4/DDR5 streaming, consecutive vector accesses alias to the exact same DRAM bank, "
                      f"triggering continuous t_RP + t_RCD row buffer conflicts and degrading memory bus throughput by >50%.",
            "recommendation": "Apply intra-array stride padding (+64B or +128B) or use runtime flag -rtr to break DRAM bank aliasing."
        })
    else:
        findings.append({
            "code": "LINT009_DRAM_BANK_ALIASING",
            "severity": "PASS",
            "penalty": 0,
            "title": "Zero DRAM Bank Power-of-Two Aliasing Detected",
            "detail": "No tensors exhibit power-of-two stride aliasing that would bottleneck DRAM bank cycle times.",
            "recommendation": "Optimal DRAM bank distribution preserved."
        })

    score = max(0, min(100, score))
    return {
        "score": score,
        "critical_bytes": critical_bytes,
        "moe_bytes": moe_bytes,
        "total_weight_bytes": critical_bytes + moe_bytes,
        "findings": findings
    }

def print_report(filepath, gguf_info, hw, analysis):
    print("=" * 80)
    print("        ik_llama.cpp GGUF Hardware Constraint Linter & Quality Scorer        ")
    print("=" * 80)
    print(f"Model File           : {filepath}")
    if gguf_info.get("is_sharded"):
        print(f"Shards               : {gguf_info['shard_count']} files ({gguf_info['file_size'] / (1024**3):.2f} GiB total across shards)")
    else:
        print(f"File Size            : {gguf_info['file_size'] / (1024**3):.2f} GiB ({gguf_info['file_size'] / 1e9:.2f} GB)")
    print(f"GGUF Version         : v{gguf_info['version']} | Tensors: {gguf_info['tensor_count']} | Alignment: {gguf_info['alignment']} bytes")
    print(f"Host Environment     : {hw['gpu_name']} ({hw['gpu_vram_bytes'] / (1024**3):.1f} GiB VRAM) | {hw['numa_nodes']} NUMA Nodes | {hw['total_ram_bytes'] / (1024**3):.1f} GiB RAM")
    print("-" * 80)

    score = analysis["score"]
    status = "OPTIMAL" if score >= 85 else ("ACCEPTABLE" if score >= 70 else "NEEDS_OPTIMIZATION")
    print(f"HARDWARE ALIGNMENT SCORE: {score}/100 [{status}]")
    print("-" * 80)

    print("DETAILED LINT FINDINGS:")
    for f in analysis["findings"]:
        sev = f["severity"]
        symbol = "[✓]" if sev == "PASS" else ("[X]" if sev == "CRITICAL" else "[!]")
        print(f" {symbol} [{f['code']}] {f['title']} (Penalty: -{f['penalty']})")
        print(f"     • Detail        : {f['detail']}")
        print(f"     • Action        : {f['recommendation']}")
        print()

    print("=" * 80)
    print("RECOMMENDED RUNTIME FLAGS:")
    print("  ./build/bin/llama-server \\")
    print(f"      -m {filepath} \\")
    if hw["numa_nodes"] > 1:
        print("      --numa split \\")
    if hw["gpu_vram_bytes"] > 0:
        print("      -sm graph \\")
        print("      -ot \"^.*\\.attn.*=CUDA0,.*\\.router.*=CUDA0,.*\\.ffn_shared.*=CUDA0,.*\\.nextn\\..*=CPU,^blk\\..*\\.ffn_.*_exps.*=CPU\" \\")
    print("      -t 7 \\")
    print("      -tb 7 \\")
    print("      -thp")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="GGUF Hardware Constraint Linter & Optimizer")
    parser.add_argument("gguf_file", help="Path to GGUF model file")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    try:
        gguf_info = load_unified_gguf_model(args.gguf_file)
        hw = probe_host_hardware()
        analysis = analyze_and_lint_gguf(gguf_info, hw)

        if args.json:
            result = {
                "file": args.gguf_file,
                "gguf": {
                    "version": gguf_info["version"],
                    "tensor_count": gguf_info["tensor_count"],
                    "alignment": gguf_info["alignment"],
                    "file_size": gguf_info["file_size"],
                    "is_sharded": gguf_info.get("is_sharded", False),
                    "shard_count": gguf_info.get("shard_count", 1)
                },
                "hardware": hw,
                "analysis": analysis
            }
            print(json.dumps(result, indent=2))
        else:
            print_report(args.gguf_file, gguf_info, hw, analysis)

    except Exception as e:
        print(f"Error parsing GGUF: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
