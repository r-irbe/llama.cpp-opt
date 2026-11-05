# llama.cpp-opt

**Run larger AI models, handle longer documents, and serve more users on the hardware you already own.**

`llama.cpp-opt` is an inference engine built on [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) and [`llama.cpp`](https://github.com/ggerganov/llama.cpp). It removes the memory and speed bottlenecks that normally force individuals and teams to rent cloud APIs or buy multi-GPU servers.

---

## Why Use This Fork? (What Problem Does It Solve?)

Most local AI tools hit a wall as soon as you move past a basic chat prompt: the model is too big for your graphics card, feeding a whole codebase runs out of memory, or running multiple coding agents slows responses to a crawl.

`llama.cpp-opt` solves four practical needs:

1. **Run big models without buying a GPU cluster**: Combine a single consumer GPU (such as a 16 GB or 24 GB card) with your system RAM and fast SSD storage to run 30B to 2.4T parameter models locally.
2. **Analyze entire codebases and long documents on one GPU**: Compress the model's working memory (*KV cache*[^kv-cache]) by up to 24x so you can fit 128K to 1M tokens of context on a single 24 GB card.
3. **Run multi-agent workflows up to 22x faster**: When multiple coding agents or team members work on the same project prompt, the server processes the shared instructions once instead of repeating the work for every request.
4. **Skip manual flag tuning**: Built-in hardware inspection and auto-tuning tools detect your CPU, RAM, and GPU layout and pick the fastest settings automatically.

> **Looking for the deep technical architecture, benchmark tables, or upstream release history?**
> - [**User & Operations Guide (`docs/USER_GUIDE.md`)**](docs/USER_GUIDE.md) -- All CLI flags, tools, and operational recipes.
> - [**Architecture & Benchmark Synthesis (`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`)**](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md) -- Hardware test results, memory designs, and formal verification details.
> - [**Upstream `ik_llama.cpp` Release Notes (`docs/UPSTREAM_IK_LLAMA.md`)**](docs/UPSTREAM_IK_LLAMA.md) -- Quantization formats and upstream pull request history.

---

## Start in 5 Minutes (Laptop or Single-GPU Desktop)

This quickstart gets you from a fresh clone to a live chat WebUI and OpenAI-compatible API (`http://127.0.0.1:8080`) on a single machine.

### 1. Build the Engine

```bash
# With an NVIDIA GPU (remove -DGGML_CUDA=ON for CPU-only laptops)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
./scripts/build-guard.sh cmake --build build --target llama-server llama-cli -j $(nproc)
```

<details>
<summary><b>Optional: Squeeze out 8-15% extra CPU speed with Profile-Guided Optimization (PTO)</b></summary>

If you rely heavily on CPU inference, you can run a one-command guided rebuild that profiles your specific processor and recompiles hot code paths for it:

```bash
./scripts/pto-build.sh use
```
</details>

### 2. Download a Fast Starter Model

Pick the model that matches your available memory:

- **For 8-16 GB RAM/VRAM (Laptops & Entry Desktops)** -- **LFM2-8B-A1B** (~4.8 GB download). This is a *Mixture-of-Experts (MoE)*[^moe] model: it holds 8 billion parameters of knowledge, but only activates 1 billion per word, making it run as fast as a tiny 1B model.
- **For 16-24 GB VRAM (Single Gaming/Workstation GPU)** -- **Recursive-Flint-32B** or **Qwen3.8-27B** (~16.5 GB download) for stronger coding and reasoning.

```bash
mkdir -p models

# Option A: Fast 8B MoE model (~4.8 GB, runs on almost any laptop or GPU)
curl -L -o models/starter-model.gguf \
  https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF/resolve/main/LFM2-8B-A1B-Q4_K_M.gguf

# Option B: 32B Reasoning & Coding model (~16.8 GB, great for 16-24 GB GPUs)
# curl -L -o models/starter-model.gguf \
#   https://huggingface.co/mradermacher/Recursive-Flint-32B-v0.1-i1-GGUF/resolve/main/Recursive-Flint-32B-v0.1.i1-IQ4_XS.gguf
```

### 3. Launch the Built-in WebUI & API

```bash
./build/bin/llama-server \
  -m models/starter-model.gguf \
  -c 16384 -ngl 99 -fa on \
  --host 127.0.0.1 --port 8080
```

- **Open the WebUI**: Visit **`http://127.0.0.1:8080`** in your browser to chat, upload files, and inspect generation speed.
- **Connect Coding Tools (Cursor, Continue, Claude Code, OpenAI SDKs)**: Point your client's base URL to `http://127.0.0.1:8080/v1/chat/completions`.

<details>
<summary><b>Want maximum speed on your machine? Repack to GGUF-X & auto-tune in 2 commands</b></summary>

Before launching, you can align the model file for zero-copy disk loading (**GGUF-X**) and let the auto-tuner measure your hardware to pick the fastest batch, thread, and *speculative decoding*[^speculative] flags:

```bash
# 1. Align the model file for fast disk-to-memory loading and verify hardware fit
python3 scripts/gguf-pack-npmc.py \
  --input models/starter-model.gguf \
  --output models/starter-model-ggufx.gguf
python3 scripts/gguf-lint.py models/starter-model-ggufx.gguf --vram-gb 16

# 2. Auto-tune serving flags for your specific CPU + GPU (runs a quick live benchmark)
python3 scripts/mab_pareto_tuner.py \
  --model models/starter-model-ggufx.gguf \
  --vram-gb 16 \
  --live-bench --max-evals 6 \
  --binary ./build/bin/llama-cli
```

Or let the interactive launcher handle everything and open a live terminal dashboard:

```bash
python3 scripts/launch.py --model models/starter-model-ggufx.gguf --capsule
```
</details>

---

## Adapt to Your Use Case (Customer Journeys)

Choose the scenario below that best matches what you are building or solving. Click any journey to see the recommended setup and ready-to-run command.

<details open>
<summary><b>1. Solo Developer on a Laptop or Single GPU -- Fast Local Coding Without API Bills</b></summary>

- **Who this is for**: Individual developers, researchers, or privacy-conscious users running on a laptop or a single 8-24 GB desktop GPU.
- **Problem solved**: Cloud API costs add up quickly during heavy coding sessions, and standard local runners pause noticeably before typing out long code blocks.
- **How we solve it**: Enables *flash attention* (`-fa on`) to keep memory usage low and *ngram speculative decoding*[^speculative] (`--spec-type ngram_simple`) to guess repeated code patterns (imports, variable names, boilerplate) and verify them in one step, boosting output speed by **1.25x to 1.50x** without needing a second model.

```bash
./build/bin/llama-server \
  -m models/starter-model.gguf \
  -c 32768 -ngl 99 -fa on \
  --spec-type ngram_simple:n_max=16 \
  --host 127.0.0.1 --port 8080
```
</details>

<details>
<summary><b>2. Engineering Teams & Multi-Agent Coding Swarms -- Instant Responses on Shared Prompts</b></summary>

- **Who this is for**: Teams running 4 to 32 concurrent AI agents (code reviewers, test generators, bug finders) or shared internal team servers.
- **Problem solved**: Normally, if 16 agents share the same 8,000-token repository summary or system prompt, the server re-reads those 8,000 tokens 16 separate times, causing a long wait before any agent starts replying.
- **How we solve it**: `--cohort-prefix-sharing` computes the shared prompt prefix once, locks it in memory, and lets all parallel slots (`-np 8`) reuse it immediately. In benchmarks, this speeds up prompt processing up to **22x** (reaching **94,087 tokens/sec** across 16 parallel requests).

```bash
./build/bin/llama-server \
  -m models/starter-model.gguf \
  -c 65536 -np 8 -ngl 99 -fa on \
  --cohort-prefix-sharing \
  --slot-save-path ./cache/slots \
  --host 0.0.0.0 --port 8080
```
</details>

<details>
<summary><b>3. Whole-Codebase & Long-Document Analysis -- Fit 128K to 1M Tokens on One 24 GB GPU</b></summary>

- **Who this is for**: Analysts, legal/financial reviewers, and engineers feeding entire repositories, books, or audit logs into a single prompt.
- **Problem solved**: A 128K to 1M token prompt normally requires tens of gigabytes of GPU memory just to store the conversation history (*KV cache*[^kv-cache]), crashing a 24 GB card long before the model can answer.
- **How we solve it**: Combines quantized working memory (`-ctk q8_0 -ctv q8_0`), sparse memory compression (`--zdd-sparse-kv`), and automatic filtering of low-value filler text (`--evidence-filter`). This cuts working memory by up to **24x** while keeping retrieval accuracy intact.

```bash
./build/bin/llama-server \
  -m models/starter-model.gguf \
  -c 131072 -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --zdd-sparse-kv --evidence-filter \
  --host 127.0.0.1 --port 8080
```
</details>

<details>
<summary><b>4. Private AI Server on a Budget -- Run Massive 300B-2.4T Models on 1 GPU + System RAM</b></summary>

- **Who this is for**: Organizations and labs that need frontier open-weight models (DeepSeek-V3/R1 671B, Kimi-K2 1T, MiniMax-M2.7 229B) on-premises without spending $100,000+ on an 8x H100/A100 server.
- **Problem solved**: When a model does not fit in GPU memory and spills into system RAM on multi-CPU servers, standard inference engines suffer from slow cross-socket memory transfers (*NUMA*[^numa] traffic) and slow disk reads.
- **How we solve it**: Keeps the heavily used attention layers on your single GPU while distributing Mixture-of-Experts weights across CPU memory nodes (`--numa-split`) or streaming cold weights directly from NVMe SSDs (`--npmc`). Our automatic launcher inspects your server topology and configures the split for you.

```bash
# Automatically detect GPUs, CPU sockets/NUMA nodes, and RAM, then launch with live telemetry
python3 scripts/launch.py \
  --model /path/to/large-moe-model.gguf \
  --capsule
```
</details>

---

## Verified Real-World Impact

All optimizations are tested on real models and hardware with automated quality checks to ensure output accuracy matches standard baselines.

<details>
<summary><b>View summary of measured speedups and memory savings</b></summary>

| What You Are Doing | Standard Baseline | With `llama.cpp-opt` | Practical Benefit |
| :--- | :--- | :--- | :--- |
| **35B-122B MoE on 1 GPU + Dual-CPU RAM** | 6.6 - 14.1 tok/s | **9.6 - 29.5 tok/s** | **Up to +109% faster** generation on hybrid CPU+GPU servers |
| **16 Agents Sharing an 8K Prompt** | 4,240 tok/s | **94,087 tok/s** | **22.2x faster** first-token response for agent swarms |
| **Long Context on a Single 24 GB GPU** | ~42K tokens max | **Up to 1,000,000 tokens** | **23.8x more context** on the same graphics card |
| **Single-User Coding & Reasoning** | 1.0x baseline speed | **1.25x - 1.50x speed** | Faster code/text output with zero quality loss (`--spec-type`) |
| **Loading Large Models from NVMe SSD** | OS page-cache thrashing | **Up to 18.8 GB/s direct read** | Predictable latencies when running models larger than RAM |

For full per-model tables across 15 production models (3.8B to 2.4T parameters) and Lean 4 formal verification proofs of our memory and concurrency invariants, see [`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md).
</details>

<details>
<summary><b>Plain-English Glossary of Technical Terms</b></summary>

- **GGUF / GGUF-X**: GGUF is the standard single-file format used to store quantized local AI models. **GGUF-X** is a 100% backward-compatible layout created by `scripts/gguf-pack-npmc.py` that aligns data inside the file to 4 KB disk pages so your SSD and CPU can read weights directly without extra memory copies.
- **KV Cache (Key-Value Cache)**: The model's short-term "working memory" for the current conversation or document. As your prompt grows from 4K to 128K+ tokens, the KV cache grows rapidly and is usually what causes out-of-memory crashes.
- **MoE (Mixture-of-Experts)**: A model design where knowledge is split across many specialized sub-networks ("experts"). Only a few experts wake up for each word, giving you the intelligence of a huge model at the speed of a much smaller one.
- **NUMA (Non-Uniform Memory Access)**: On workstations and servers with multiple CPU sockets or chiplets, each CPU group has its own local RAM. Reading from local RAM is fast; reading across sockets is slow. `llama.cpp-opt` pins threads and model weights to local memory so CPUs do not wait on each other.
- **Speculative Decoding**: A technique that drafts several likely upcoming tokens at once (using pattern matching or a tiny helper model) and verifies them in a single pass of the main model, speeding up text generation with identical mathematical output.
</details>

---

## Where to Go Next

| Guide | What You Will Find Inside |
| :--- | :--- |
| [**User & Operations Guide (`docs/USER_GUIDE.md`)**](docs/USER_GUIDE.md) | Complete reference for all CLI flags, `launch.py`, `gguf-lint.py`, `mab_pareto_tuner.py`, and TUI monitoring (`ctx-capsule`). |
| [**Architecture & Benchmark Synthesis (`docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md`)**](docs/ARCHITECTURE_AND_BENCHMARK_SYNTHESIS.md) | Deep-dive engineering design, 15-model hardware benchmark matrices, quality/perplexity audits, and Lean 4 formal verification proofs. |
| [**Build Guide (`docs/build.md`)**](docs/build.md) & [**Runtime Parameters (`docs/parameters.md`)**](docs/parameters.md) | Platform-specific compilation flags (CUDA, Vulkan, Metal, CPU) and standard `llama.cpp` server options. |
| [**Upstream `ik_llama.cpp` Release Notes (`docs/UPSTREAM_IK_LLAMA.md`)**](docs/UPSTREAM_IK_LLAMA.md) | Full history of upstream `ik_llama.cpp` quantization formats (`IQ1`-`IQ4`, `Q4_K`-`Q8_KV`) and pull requests. |

---

## Contributing & License

Contributions, bug reports, and benchmark reproductions are welcome! Please run `./scripts/build-guard.sh` when compiling changes and check `python3 scripts/gguf-lint.py` when testing model containers. Released under the [MIT License](LICENSE).

[^kv-cache]: **KV Cache**: The working memory a model uses to remember your active prompt and conversation history.
[^moe]: **Mixture-of-Experts (MoE)**: A model architecture that stores large amounts of knowledge but only activates a small fraction of its parameters for each generated token.
[^speculative]: **Speculative Decoding**: Generating multiple candidate tokens ahead of time and verifying them in a single step for faster output with zero loss in quality.
[^numa]: **NUMA (Non-Uniform Memory Access)**: Multi-CPU memory architecture where keeping data in the RAM physically closest to each CPU core avoids slow cross-socket bottlenecks.
