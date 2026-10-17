# Token generation performance troubleshooting

## Verifying that the model is running on the GPU with CUDA
Make sure you compiled llama with the correct env variables according to [this guide](/docs/build.md#cuda), so that llama accepts the `-ngl N` (or `--n-gpu-layers N`) flag. When running llama, you may configure `N` to be very large, and llama will offload the maximum possible number of layers to the GPU, even if it's less than the number you configured. For example:
```shell
./llama-cli -m "path/to/model.gguf" -ngl 200000 -p "Please sir, may I have some "
```

When running llama, before it starts the inference work, it will output diagnostic information that shows whether cuBLAS is offloading work to the GPU. Look for these lines:
```shell
llama_model_load_internal: [cublas] offloading 60 layers to GPU
llama_model_load_internal: [cublas] offloading output layer to GPU
llama_model_load_internal: [cublas] total VRAM used: 17223 MB
... rest of inference
```

If you see these lines, then the GPU is being used.

## Verifying that the CPU is not oversaturated
llama accepts a `-t N` (or `--threads N`) parameter. It's extremely important that this parameter is not too large. If your token generation is extremely slow, try setting this number to 1. If this significantly improves your token generation speed, then your CPU is being oversaturated and you need to explicitly set this parameter to the number of the physical CPU cores on your machine (even if you utilize a GPU). If in doubt, start with 1 and double the amount until you hit a performance bottleneck, then scale the number down.

## Measuring CPU-only performance: `-ngl 0` is not CPU-only when a GPU is visible

On a build with a GPU backend, a run with `-ngl 0` still allocates a GPU compute buffer and runs
prompt processing on the GPU, while token generation stays on the CPU. Measured on an RTX 4090 +
2x EPYC 7453 host with an 8.79 B Q4_K_M model (`llama-bench -r 3 -t 32`):

| configuration | pp512 (tok/s) | tg32 (tok/s) |
| --- | ---: | ---: |
| `CUDA_VISIBLE_DEVICES=""` (true CPU-only) | 140 | 19.2 |
| GPU visible, `-ngl 0` | 1066 | 17.7 |
| GPU visible, `-ngl 20` | 1694 | 28.6 |
| GPU visible, `-ngl 999` | 7285 | 148.5 |

Consequences for reading and writing benchmark data:

- To measure the CPU, hide the GPU: `CUDA_VISIBLE_DEVICES="" ./llama-bench ...`. `llama-bench` warns when
  a GPU is visible together with `-ngl 0`, because the resulting row looks CPU-only but is not.
- In the `llama-bench` output (including its SQLite/CSV/JSON rows), the `cuda` / `vulkan` / `metal` /
  `sycl` / `blas` columns record the **compiled-in backends**, not the backends that executed the run.
  A row with `cuda=1` and `n_gpu_layers=0` is *not* a CPU measurement; only the combination with
  `CUDA_VISIBLE_DEVICES=""` is.
- The same applies to `llama-cli`-driven harnesses such as `scripts/mab_pareto_tuner.py`, which sets
  `CUDA_VISIBLE_DEVICES=""` for live trials configured with zero GPU layers.

# Example of runtime flags effect on inference speed benchmark
These runs were tested on the following machine:
GPU: A6000 (48GB VRAM)
CPU: 7 physical cores
RAM: 32GB

Model: `TheBloke_Wizard-Vicuna-30B-Uncensored-GGML/Wizard-Vicuna-30B-Uncensored.q4_0.gguf` (30B parameters, 4bit quantization, GGML)

Run command: `./llama-cli -m "path/to/model.gguf" -p "An extremely detailed description of the 10 best ethnic dishes will follow, with recipes: " -n 1000 [additional benchmark flags]`

Result:

| command | tokens/second (higher is better) |
| - | - |
| -ngl 2000000 | N/A (less than 0.1) |
| -t 7 | 1.7 |
| -t 1 -ngl 2000000 | 5.5 |
| -t 7 -ngl 2000000 | 8.7 |
| -t 4 -ngl 2000000 | 9.1 |
