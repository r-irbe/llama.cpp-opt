# CUDA graph memory investigation

Status: host retirement/cache-policy revision and adaptive headroom-based
pressure reclamation implemented (`GGML_CUDA_PRESSURE_POLICY=adaptive|latency-first`,
`GGML_CUDA_PRESSURE_RESERVE_MIB`); live-driver long-run serving validation
continues per the broader agreed contract in
[runtime memory pressure recovery](memory-pressure-recovery.md).

## Problem and evidence

A reported `cudaGraphLaunch` OOM occurred after successful generation. The
shape/root-keyed CUDA graph cache currently has no capacity bound. Each cached
entry owns a captured graph, an executable and a device copy-pointer table;
the table is not freed by the current graph destructor. These are verified
retention defects, not proof of the precise allocation behind the reported OOM.

## Contract

- Initial fixed default capacity 64 is NOT accepted: a bounded reproduction
  completed but created/captured 34501 graphs. Determine actual split working
  set and pressure before choosing eviction; keep explicit operator overrides.
  Interim correction: remove the implicit count limit, preserve an explicit
  `GGML_CUDA_GRAPH_CACHE_SIZE` override (zero/unset = no count limit), and record
  accesses/hits/misses/evictions/high-water occupancy. This restores the earlier
  reuse behavior, not a complete adaptive pressure bound. Default pressure-based
  admission/reclamation remains part of the broader recovery controller.
- Eviction and invalidation must select the owning device, stop conflicting
  submissions, complete in-flight work and clear stale graph references.
  Use a shared submission/exclusive retirement gate to protect registry lifetime
  and exclude new submissions during hot-swap/teardown. Serialize same-context
  graph bookkeeping separately while allowing independent contexts to submit.
  Retirement must also wait for active captures before device-wide fencing;
  eviction and preparation fallback must use this same capture/fence discipline.
  Invalidation is synchronous at its own boundary; callers must still exclude
  model-weight mutation across the invalidate/reload operation.
- Initialize the device copy-pointer table (`dest_ptrs_d`) and 28-byte device
  PolyGraph plan buffer (`polygraph_plan_d`) to null. Release both with graph and
  executable resources on eviction and context teardown, on the owning device.
- When `GGML_CUDA_GRAPH_CACHE_SIZE` is unset/zero and `ctx.cuda_graphs.size() > 1`
  on a cache miss, query `cudaMemGetInfo` against the configured reserve
  (`GGML_CUDA_PRESSURE_RESERVE_MIB`, default `512` MiB). When
  `ggml_cuda_graph_under_pressure` triggers, reclaim LRU graphs (excluding
  `ctx.cur_graph`) inside `ggml_cuda_with_graph_fence` via
  `ggml_cuda_graph_pressure_reclaim` according to `GGML_CUDA_PRESSURE_POLICY`
  (`adaptive` reclaims up to half the cache per miss; `latency-first` caps
  single-step reclamation at 1 entry) and record `pressure_evictions` in
  `GGML_CUDA_GRAPH_STATS=1`.
- Explicitly upload newly instantiated or updated executable graphs before
  launch. An allocation failure during preparation (instantiate/update/upload)
  may disable graph execution and use ordinary kernel submission, but only
  after a successful device synchronization. The captured operations have not
  executed at this point; disable copy indirection before direct evaluation.
- Release all cached graphs before the direct fallback. Do not retry capture on
  subsequent calls for that backend context.
- Never blindly replay a graph after `cudaGraphLaunch` fails: CUDA APIs can
  report previous asynchronous failures and KV writes are not transactional.
  Non-allocation errors retain existing checked-error behavior.
- No automatic changes to model placement, context size, attention semantics,
  speculative settings, or host-local launch configuration.

## Verification

A model-free host regression harness mocks CUDA resource calls and exercises
null initialization, idempotent resource release, LRU selection, explicit cache
limits, active-entry protection, and the production lifetime/capture gates.
Retirement checks enforce device selection, successful fencing and active-pointer
clearing before destructors run. Failed fencing leaves entries untouched.
Independent submissions coexist; blocking retirement waits for submissions;
blocking fences wait for captures and exclude new captures. Admission turnstiles
prevent a waiting retirement/fence from being continually replenished by new
submissions/captures. Exception cleanup is tested.

Current registry tracks every backend context, including duplicate devices;
invalidation, registration and teardown use exclusive lifetime ownership.
The gate protects graph lifetime, not arbitrary model/tensor mutation. In
particular, the existing reload code changes tensors before invalidating graphs;
its callers must already establish quiescence over the entire reload operation.
This change does not make concurrent reload/evaluation safe.

The default cache no longer trims solely because occupancy reaches 64. A
128-entry repeated working set is retained in the host policy regression, while
explicit nonzero limits still select LRU victims. Zero/unset means no count
limit, NOT automatic pressure-based bounded memory. Hit/miss, eviction and peak
entry telemetry has been added; actual reuse/footprint tuning remains pending. Build it and the CUDA backend via `scripts/build-guard.sh`.
Then, only within an authorized GPU time budget, run a bounded model reproduction
with the reported placement/settings,
recording readiness, requests, memory samples, failures and timeout separately.
A short reproduction cannot certify an unbounded serving lifetime.

Recorded short-run evidence: four requests completed, 256 generated tokens,
114.01 seconds; peak GPU used 18083 MiB and minimum free 5999 MiB. Teardown
reported 64 retained graphs but 34501 instances/captures. This supports a churn
finding, not a conclusion about the original OOM or absence of leaks. Independent
review also found a hot-swap invalidation ownership gap after table cleanup.
No preparation-OOM fault injection has yet validated the fallback.

Extend host regressions with production guard/fence/retirement helpers: active
submission blocks retirement; independent submissions coexist; active capture
blocks a fence and new captures cannot begin during that fence; same-context
bookkeeping is serialized; failed fences retain resources and current pointers;
successful fences select the owning device before destruction; explicit cache
limits still evict; default admission retains a repeated >64-entry working set.
These checks do not establish real-driver behavior or long-run leak freedom.

### Revision evidence

Guarded configured CUDA backend/server link passed, with host test targets
included: `.pi/harness-bridge/build/build-3689484.log`. The selected checkpoint,
graph-owner/gate, paged-KV and tiered-KV tests passed 4/4. Graph gate/resource
regression additionally passed 100 consecutive host executions. These tests do
not exercise the real driver, concurrent model reload, preparation-OOM fault
injection, adaptive pressure decisions or a long-running serving lifetime.
Active LSP coverage remained inconclusive; a separate clangd probe emitted a
refactoring-tweak failure, not a confirmed clean check. Actual configured
compiler/link and test evidence is separate from those tool limitations.

## Primary reference

[NVIDIA CUDA Runtime graph management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html):
`cudaGraphUpload` uploads without execution; `cudaGraphLaunch` and upload may
report errors from earlier asynchronous launches. This is why fallback is
restricted to pre-launch preparation with successful synchronization.
