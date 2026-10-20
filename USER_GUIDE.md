# llama.cpp-opt User Guide

Practical reference for building, configuring, and running **`llama.cpp-opt`** across multi-socket NUMA servers, fast NVMe arrays, and NVIDIA Ada Lovelace / Ampere GPUs.

---

## 1. Building `llama.cpp-opt`

### CPU + CUDA Build (Recommended for Hybrid & GPU Inference)

```bash
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### Pure CPU Build (Multi-Socket EPYC / Xeon / Threadripper)

```bash
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

> **Note on CPU-only Benchmarking**: On hosts with a visible NVIDIA GPU, passing `-ngl 0` still dispatches prompt processing (`pp`) to CUDA unless `CUDA_VISIBLE_DEVICES=""` is set in the environment.

---

## 2. Optimization Flags & Subsystems

### 2.1 Multi-Socket NUMA Scaling (`--numa split`, `--numa expert`, `--numa-replicate-shared`)

`llama.cpp-opt` extends `--numa` with explicit thread placement and MoE tensor binding (`common/common.cpp`, `ggml/src/ggml-cpu-numa.cpp`):

| Flag | Behavior |
| :--- | :--- |
| `--numa split` / `--numa distribute` | Pins worker threads round-robin across detected NUMA nodes (`GGML_NUMA_STRATEGY_DISTRIBUTE`). Note: `--numa split` is a space-separated argument (`--numa` followed by `split`), aliased to `distribute`. |
| `--numa expert` | Binds routed MoE expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) round-robin across detected NUMA nodes (`GGML_NUMA_STRATEGY_EXPERT`) and aligns worker threads in contiguous per-node blocks. |
| `--numa mirror` | Pins worker threads round-robin across NUMA nodes (`GGML_NUMA_STRATEGY_MIRROR`); designed to pair with `--numa-replicate-shared`. |
| `--numa-replicate-shared` | Replicates shared MoE expert tensors (`ffn_*_shexp`) onto every NUMA node so local workers never cross socket interconnects for shared layers. |
| `--numa isolate` | Pins all workers to the starting NUMA node. Always set `-t` to the core count of a single node when using `isolate`. |

**Example (Dual-Socket MoE Server)**:
```bash
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  --numa expert \
  --numa-replicate-shared \
  -t 56 -tb 56 \
  -p "Summarize the architecture of non-uniform memory access."
```

---

### 2.2 Native FP8 DeepSeek-V4.1 Row Decoding & Deferred NVMe Residency

Upstream checkpoints for **DeepSeek-V4.1** store compact 264-byte DwarfStar FP8 Engram rows (256 FP8 weights + 8 scale bytes). `llama.cpp-opt` decodes these rows natively in `ggml_compute_forward_get_rows_i8` and aligns raw/extra key and mask streams in `build_deepseek4.cpp` without requiring a 190+ GB `Q8_0`/`Q2_K` expansion.

For checkpoints larger than system RAM (e.g., 385 GB DeepSeek-V4.1-Flash-EngramQ8), combine deferred residency flags with Linux `io_uring` Direct I/O:

| Flag | Behavior |
| :--- | :--- |
| `-dple`, `--defer-ple` | Keeps sparse per-layer token embedding (PLE) and FP8 Engram tables on NVMe storage instead of faulting them into RAM at startup. |
| `-dexp`, `--defer-experts` | Defers MoE expert mmap residency on Linux to reduce cold startup time to ~1.35 seconds. |
| `-thp`, `--transparent-huge-pages` | Maps model weights into 2 MiB HugeTLB pages and automatically activates the zero-copy Linux `io_uring` DMA loader (`src/llama-io-uring.cpp`, `O_DIRECT` + `IORING_SETUP_SQPOLL` + registered buffers) when supported by the kernel. |

**Pre-Flight Disk Alignment Check**:
Before streaming from NVMe with `O_DIRECT`, verify 4 KB sector alignment and hardware memory budgets:
```bash
python3 scripts/gguf-lint.py /path/to/DeepSeek-V4.1-Flash-EngramQ8.gguf
```

**Example (Streaming DeepSeek-V4.1 FP8 from NVMe)**:
```bash
./build/bin/llama-cli \
  -m /path/to/DeepSeek-V4.1-Flash-EngramQ8-00001-of-00010.gguf \
  --defer-ple --defer-experts -thp \
  -ngl 99 -t 28 \
  -p "Explain zero-copy DMA weight streaming."
```

---

### 2.3 Cohort Prefix Sharing (`--cohort-prefix-sharing`)

When running `llama-server` with multiple concurrent slots (`-np N`), standard slot assignment can re-evaluate shared system prompts or RAG prefixes across simultaneous requests. Cohort Prefix Sharing (`common/cohort-radix-trie.h`, `examples/server/server-context.cpp`) maintains a radix prefix trie across active server slots with weighted LFU/LRU eviction.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--cohort-prefix-sharing` | `false` | Enables cohort radix-trie prefix matching and concurrent KV state sharing across server slots. |
| `--no-cohort-prefix-sharing` | - | Explicitly disables cohort prefix sharing. |
| `--cohort-min-prefix N` | `64` | Minimum shared token prefix length required to trigger cohort routing. |
| `--cohort-max-idle-sec N` | `300` | Maximum idle window (seconds) before a cached cohort prefix is eligible for eviction. |

**Example (`llama-server` with 16 Concurrent Slots)**:
```bash
./build/bin/llama-server \
  -m /path/to/model.gguf \
  -c 65536 -np 16 -cb \
  --cohort-prefix-sharing \
  --cohort-min-prefix 64 \
  --port 8080
```

---

### 2.4 Pre-Attention Token Filtering (`-unreal` / `--unreal-filter`)

For long-context RAG and document QA workloads, `-unreal` (`common/unreal-selector.h`) performs streaming late-interaction MaxSim chunk scoring with BM25 lexical boosting and exact-cover partitioning *before* full quadratic self-attention prefill.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-unreal`, `--unreal-filter` | `false` | Enables pre-attention MaxSim + BM25 evidence chunk filtering in `llama-cli` and `llama-server`. |
| `-no-unreal`, `--no-unreal-filter` | - | Disables pre-attention filtering. |
| `--unreal-chunk-size N` | `128` | Token length of each candidate document chunk. |
| `--unreal-top-n N` | `16` | Number of highest-scoring evidence chunks retained for attention prefill. |
| `--unreal-min-tokens N` | `1024` | Minimum prompt token length required before `-unreal` filtering activates. |
| `--unreal-query-groups N` | `8` | Number of query token groups used for MaxSim late-interaction scoring. |
| `--unreal-chunk-vectors N` | `8` | Number of representative vectors extracted per candidate chunk. |

**Example**:
```bash
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -c 32768 \
  -unreal --unreal-chunk-size 128 --unreal-top-n 16 --unreal-min-tokens 1024 \
  -f long_document_prompt.txt
```

---

### 2.5 ZDD Sparse KV Cache & CUDA FlashAttention Tile Skipping (`--zdd-sparse-kv`)

On long contexts (128k-512k tokens), `--zdd-sparse-kv` (`src/llama-kv-zdd.h`) indexes 64-token KV cache blocks in a zero-suppressed decision diagram (ZDD), masks inactive blocks with `-INF` in the KQ mask, and sets `op_params[5] = 1` on `ggml_flash_attn_ext` so all 6 CUDA FlashAttention kernel families (`wmma-f16`, `tile-f16`, `tile-f32`, `vec-f16`, `vec-f32`, `mma-f16`) skip fully masked 2D tiles without reading K/V global memory.

```bash
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -c 131072 -fa \
  --zdd-sparse-kv \
  -f prompt.txt
```

---

### 2.6 NPMC 2 MiB Superblock v3 Zero-Copy Loader (`Q1_0` / Ternary)

`llama.cpp-opt` includes a zero-copy 2 MiB Superblock v3 loader (`src/llama-npmc.h`, `src/llama-npmc.cpp`) supporting:
- Embedded `NPMC_SB3` magic headers inside `GGML_TYPE_Q1_0` tensors (`scripts/pack_npmc_v3.py`).
- Dual-stream ternary coding (Stream A 2-bit sign/zero mask + Stream B Elias-Fano outlier directory).
- Verified 128-byte base alignment, Stream A/B non-overlap invariants, and Hilbert / BP128 / 2:4 Sparse / CROWN / Warp vEB execution primitives.

---

## 3. Verification & Test Suite

Run the focused unit and subsystem tests after building:

```bash
./build/bin/test-numa
./build/bin/test-kv-zdd
./build/bin/test-cohort-prefix-sharing
./build/bin/test-unreal-selector
./build/bin/test-paged-kv-cache
./build/bin/test-tiered-kv-cache
./build/bin/test-npmc
```

## 4. Further Reading

- [NUMA Architecture & Tuning Guide](docs/numa.md)
- [KV Cache & ZDD Sparse Attention Guide](docs/kv-cache.md)
- [Speculative Decoding & Universal Bandit Autotuner](docs/speculative.md)
- [Implementation & Provenance Matrix](docs/UNIMPLEMENTED.md)
- [Full CLI Parameter Reference](docs/parameters.md)
