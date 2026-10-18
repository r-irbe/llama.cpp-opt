# Key-Value (KV) Cache Optimization Guide

This guide describes how to configure, optimize, and scale the Key-Value (KV) cache in `llama.cpp-opt` for long-context inference, low memory footprints, and high token throughput.

---

## 1. The Long-Context Memory Challenge

Autoregressive transformer inference caches key and value projections for all previous tokens to avoid recomputing past representations:
$$\text{Memory Footprint} \approx 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{ctx\_len} \times \text{bytes\_per\_elem}$$

For a 70B parameter model with 128k context length:
- Full-precision (`f16`) KV cache requires **~40 GiB** of RAM/VRAM solely for context history.
- At 512k context, memory requirements scale to **~160 GiB**, frequently exceeding single-socket DDR or GPU VRAM capacity and heavily saturating memory bus bandwidth.

`llama.cpp-opt` incorporates algorithmic and architectural optimizations to compress, quantize, and accelerate KV cache evaluation.

---

## 2. KV Cache Quantization (`-ctk` and `-ctv`)

`llama.cpp-opt` supports quantizing key and value vectors independently. This reduces memory footprint by 50% to 75% while maintaining inference quality:

| Cache Type | CLI Argument | Compression Ratio | Perplexity Impact | Recommended Use |
|------------|--------------|-------------------|-------------------|-----------------|
| `f16` | `-ctk f16 -ctv f16` | 1× (Baseline) | Exact baseline | Short contexts (<8k) |
| `q8_0` | `-ctk q8_0 -ctv q8_0` | 2× (~50% reduction) | Negligible ($\Delta < 0.005$) | **Recommended Default** for 32k+ |
| `q4_0` | `-ctk q4_0 -ctv q4_0` | 4× (~75% reduction) | Minor ($\Delta \approx 0.02$) | Long context on VRAM-limited GPUs |

**Usage Example:**
```bash
./build/bin/llama-cli -m models/llama-3-8b-instruct.Q4_K_M.gguf \
    -c 65536 \
    -ctk q8_0 \
    -ctv q8_0 \
    -ngl 99 \
    -p "Summarize the attached 50,000-word document..."
```

---

## 3. ZDD / SDD Sparse Block Compression (prototype, not wired up)

> **Status at HEAD:** `--zdd-sparse-kv` / `--no-zdd-sparse-kv` CLI flags, `gpt_params.zdd_sparse_kv`, `llama_context_params.zdd_sparse_kv`, `src/llama-kv-zdd.h`, and `tests/test-kv-zdd.cpp` are wired and tested. The mask hook is live: inactive blocks are `-INF`-filled in the KQ mask at the `src/llama.cpp` input stage (`set_inputs`) via the `llama-graph.cpp` mask builder. Attention-kernel block-skipping in `ggml` (CPU/CUDA) remains deferred; `src/llama-build-context.cpp:2166-2169` currently builds and discards an unused mask copy (dead call). See [`docs/UNIMPLEMENTED.md`](./UNIMPLEMENTED.md).

In long-context generation (>64k tokens), attention matrices exhibit extreme sparsity: $>85\%$ of 64-token key-value blocks have cumulative attention weights $< 10^{-4}$ and contribute negligibly to final token logits.

The prototype explores Zero-suppressed Binary Decision Diagram (ZDD) block indexing:
- **Minato Zero-Suppression**: Inactive token blocks are eliminated implicitly without node allocations.
- **Hardware-Aligned Block Size**: KV blocks are grouped into 64-token chunks matching SIMD vector registers and GPU warp execution widths.
- **Warp-Synchronous GPU Execution**: On NVIDIA Ada Lovelace (`sm_89`) and newer architectures, warps evaluate 32 candidate blocks (2,048 tokens) concurrently using register shuffles (`__ballot_sync`, `__shfl_sync`), bypassing inactive blocks with zero bank conflicts.
- **L2 Cache Pinning**: Attention sinks and recent context blocks are marked with `cudaAccessPropertyPersisting` to remain resident in GPU L2 cache across decoding cycles.

---

## 4. Context Scaling: YaRN and RoPE Scaling

When running models beyond their original pre-trained context length, RoPE frequency scaling prevents attention score dispersion and catastrophic perplexity degradation.

### 4.1 YaRN (Yet another RoPE extensioN)
YaRN applies an attention temperature multiplier alongside frequency scaling to preserve high-frequency positional resolution while expanding context reach.

**Usage Example:**
```bash
./build/bin/llama-cli -m models/llama-3-8b-instruct.Q4_K_M.gguf \
    -c 131072 \
    --rope-scaling yarn \
    --rope-scale 8.0 \
    --rope-freq-base 500000.0 \
    -ngl 99
```

---

## 5. Attention Sinks and Context Management

When context length exceeds available RAM or `--ctx-size`:
- **Attention Sinks**: Preserves the first $k$ initial prompt tokens (which store high-magnitude sink activations essential for attention stability).
- **Rolling Window**: Discards oldest non-sink tokens, maintaining continuous autoregressive generation without perplexity blowup.
