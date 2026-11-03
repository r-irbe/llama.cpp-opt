# PTO Build Workflow (one-command UX)

`scripts/pto-build.sh` is the single entry point for Profile-Targeted
Optimization (PTO) builds. It sequences the stages on top of
`scripts/pto_finetune.sh` (which owns the CMake flag surface) and always builds
through `scripts/build-guard.sh`.

## Stages

| Stage | What it does |
|-------|--------------|
| `status` | Prints current `GGML_PTO_MODE`, profile dir, raw-profile count, merged-profile timestamp/size, and the built `llama-server` sha256. No build. |
| `trace` | Instrumented build (`GGML_PTO_MODE=GENERATE`). With `--trace-workload=harness`, immediately runs a real request battery so the instrumented binary writes `.profraw` traces. |
| `cs` | Round-2 context-sensitive instrumented build (`GGML_PTO_MODE=CS_GENERATE`, base = existing merged profile) plus optional trace workload. |
| `merge` | Validates each `.profraw` individually (a truncated raw from a killed run would otherwise poison the whole merge), skips unusable ones with a warning, and writes `merged.profdata` via `llvm-profdata-21`. |
| `use` | Profile-optimized production build (`GGML_PTO_MODE=USE`, ThinLTO + FSMF + LLD/ICF). This is the default stage. |
| `train` | `trace` → workload → `merge` (no optimized rebuild). |
| `all` | `train` → `use` (full cycle). |
| `flags` | Prints the env surface (`PTO_STAGE`, `PTO_BUILD_DIR`, `PTO_PROFILE_DIR`, `PTO_PROFDATA`, `BUILD_GUARD_PTO`) for ctx-capsule / fleet consumers. |

The stage can also come from the environment: `BUILD_GUARD_PTO=use scripts/pto-build.sh`.

## Examples

```sh
scripts/pto-build.sh status
scripts/pto-build.sh use

# Train a fresh profile from a real serving workload, then rebuild optimized:
scripts/pto-build.sh all \
    --trace-workload=harness \
    --model-key=deepseek-v4.1-1017-ud-q4kxl \
    --model-root="$LLAMA_MODEL_ROOT" \
    --trace-scenarios=tg_short,pp_long --trace-reps=2

# Merge only (e.g. after collecting raw profiles manually):
scripts/pto-build.sh merge --profile-dir=/tmp/ik_llama_pgo
```

## Invariants

- Every C/CXX/CUDA compilation goes through `scripts/build-guard.sh`
  (single-flight, stall watchdog, wall-clock cap). The wrapper never calls a
  compiler or `ninja` directly.
- The trace workload drives the *instrumented* binary in the build tree, so the
  profile reflects the exact code being optimized.
- No host-specific paths are baked in: profile/build dirs are options with
  generic defaults, model roots come from `--model-root` or `$LLAMA_MODEL_ROOT`.
- `merge` is resilient by design: unusable raw profiles are reported and
  skipped individually instead of failing the batch.
- `status` is the provenance check to run before trusting any PTO result
  (mode, profile freshness, binary sha).

## Notes

- `-c` on `llama-server` is the *total* context divided across `-np` slots;
  trace-workload prompts must fit one slot (the harness warns when they may
  not).
- A full `train`/`all` cycle recompiles the whole tree twice (instrumented +
  optimized). Schedule it in a quiet window: it is the only part of this
  workflow that is not exercised by the dry-run/merge smoke checks.
