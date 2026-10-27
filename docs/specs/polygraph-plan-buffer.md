# Spec: PolyGraph Plan-Buffer — Shape-Pinned Speculative Verification

Status: DRAFT → implementation in progress (Lean-first track)
Branch: `feat/numa-split`
Owner: radu / spec stream (Pi session 2026-10-27)
Formal spec: `rust-mmap-engine/formalization/PolyGraph/Basic.lean`
Related: ROADMAP Pillar 8 (speculative autotuning), Pillar 9 (Lean 4 SafeVerify FFI)

## 1. Problem (measured, not assumed)

`ggml_cuda_graph_get_key` (ggml/src/ggml-cuda.cu) mixes `n_batch`, node count and
root-tensor extents into the graph cache key, so **every distinct verification
batch shape instantiates its own `cudaGraphExec_t`** in `ctx.cuda_graphs`.

Under `--spec-autotune` the counterfactual tuner selects a different draft
length γ every step (`spec_counterfactual_engine` k_max = 16), and the target
verification batch is rebuilt at exactly `draft.size() + 1` rows
(common/speculative.cpp:3709). Worst case that is **up to ~17 concurrent graph
instances per context** (γ ∈ [0, 16]), each captured (warmup cost), each holding
driver memory, and each still needing `cudaGraphExecUpdate` whenever KV state
moves underneath it. The old `ggml_cuda_polygraph_plan` placeholder (removed in
22ef236d3) claimed to solve this but was never wired to anything.

## 2. Goal

Collapse speculative-decode verification to **one** CUDA graph: pin the
verification batch to `γ_max + 1` rows every step, carry the real γ in a small
plan buffer, and let padding rows ride a scratch KV sequence so the slot's cache
stays clean. Zero re-instantiation after warmup; the wasted-FLOP cost of padding
becomes a measurable tuner arm instead of a claim.

Non-goals (v1): multi-GPU plan replication, draft-model graphs, non-spec decode.

## 3. Design

### 3.1 Pinned verification batch

Row layout for every verification step (`rows = γ_max + 1`):

```
row 0                : root token (last accepted)        seq = slot
rows 1..γ_act        : drafted tokens (real)             seq = slot
rows γ_act+1..γ_max  : padding tokens (any value)        seq = scratch
```

- `llama_batch` supports per-row `seq_id`; real rows never attend across
  sequences, so **padding rows cannot leak into real rows' attention**. This is
  the formal safety property (Lean: `no_padding_flow`, below).
- Scratch-sequence KV cells are rewritten every step at the same positions —
  the scratch seq never grows; it is cleared (or simply overwritten in place)
  per step. The fork already uses companion-sequence patterns for drafters, so
  the pattern has precedent.
- Positions: `rowPos(r) = posBase + r` for all rows (padding positions are
  valid RoPE positions; their outputs are discarded). `(rowSeq, rowPos)` is
  injective across rows ⇒ no two rows share a KV cell (Lean: `kv_cells_disjoint`).

### 3.2 Plan buffer (28 bytes, host-authored)

```c
struct spec_polygraph_plan {        // 28 bytes, packed, alignas(4)
    uint32_t active_len;            // γ_act ∈ [0, γ_max]; real rows = γ_act+1
    uint32_t pos_base;              // KV position of row 0
    uint32_t flags;                 // bit0: target-only baseline (γ_act = 0 forced)
    uint32_t reserved0, reserved1, reserved2, reserved3;
};
```

Host writes it into pinned staging memory; an async H2D copy precedes replay.
The sampler/acceptance logic reads the host copy: logits are scored only for
rows `r ≤ γ_act` (Lean: `readable_rows_are_real`).

### 3.3 Tuner integration (the honest part)

The pinned vs dynamic trade-off is **not decided by this spec** — it becomes two
arms of the existing spec-tuner:

- arm `dynamic` (today's path, γ varies, per-shape graphs),
- arm `pinned` (this design, one graph, γ_act in the plan buffer),

rewarded by measured step TPS as today. Flag `--spec-verify-pinned` (default
off) forces the pinned arm for debugging/benchmarks.

## 4. Formal track (Lean-first, per Pillar 9 pattern)

Module: `rust-mmap-engine/formalization/PolyGraph/Basic.lean` (core Lean 4, no Mathlib).

Certified definitions: `Plan`, `Limits`, `isReal`, `rowSeq`, `rowPos`,
`attends`, `readable`, `sanitize`.

Theorems (proved in Lean; mirrored by exhaustive C++ tests):
1. `mask_closure` — if a real row attends a column, that column is real
   (padding never flows into real outputs).
2. `readable_rows_are_real` — the sampler never scores a padding row.
3. `kv_cells_disjoint` — `(rowSeq r, rowPos r)` injective on rows: no KV cell
   collisions between real and padding rows.
4. `sanitize_within_limits` — clamped plans satisfy `active_len ≤ γ_max`.

Codegen contract: the generated C++ planner
(`common/spec-verify-planner.h`, v1) mirrors each definition 1:1 with a
provenance comment naming its Lean counterpart; `tests/test-polygraph-plan.cpp`
exhaustively re-checks theorems 1–4 for all plans at γ_max = 16 (the Lean proofs
are the specification of record; the C++ test is the conformance harness).

## 5. Mechanism track (runs in parallel)

Slice A (in flight): graph-cache telemetry in ggml-cuda — instance/capture/
update/re-instantiate/disable counters + `GGML_CUDA_GRAPH_STATS=1` summary at
context teardown. This produces the before/after evidence: today's instance
count under autotune vs 1 under pinned mode.

Slice B: micro-benchmark replaying synthetic plans against one pinned graph,
asserting `reinstantiations == 0` and `instances == 1` after warmup.

Slice C (integration): pinned verification-batch construction in the spec
verify path behind `--spec-verify-pinned`, plan-buffer write, sampler row
windowing, scratch-seq lifecycle.

## 6. Risks

| Risk | Mitigation |
|---|---|
| MoE expert routing burns FLOPs on padding rows | This is the cost the A/B tuner arm measures; if net-negative for a workload, the tuner keeps `dynamic`. |
| MLA/RoPE behavior on scratch-seq rows | Padding positions are valid positions; outputs discarded. Verify numerically in Slice C tests. |
| Interaction with `--spec-ckpt-mode per-step` row surgery (`i_batch_dft`) | Slice C must keep checkpoint logic reading only real rows; add a dedicated test. |
| FlashAttention mask construction cost for padded layout | Mask is a pure function of the plan (Lean `plan_deterministic`); built once per step host-side. |
| Large-model step time dwarfs graph overhead (win may be small) | Accepted: telemetry + A/B decide empirically; no performance claims until measured. |

## 7. Verification protocol (Slice C exit criteria)

1. `lake build` clean in `rust-mmap-engine/formalization/`; all 4 theorems prove.
2. `scripts/verify-spec.sh` green + new `test-polygraph-plan` exhaustive conformance.
3. Live A/B on DeepSeek-V4.1-Flash + DSpark drafter (existing launch configs):
   telemetry shows `instances == 1`, `reinstantiations == 0` after warmup under
   pinned mode; step-TPS delta recorded — whichever arm wins becomes the tuned
   default path for that workload.
