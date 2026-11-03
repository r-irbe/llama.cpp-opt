# Response-budget calibration: bounded CPU pilot

## Status and authority

Preparation and an extended CPU-loopback pilot are authorized. Neither has yet
produced calibration evidence. This specification does not enable a production
budget controller, tenant enforcement, transport timers or reserved capacity.
Numeric deployment limits require evidence review and separate operator approval.

The governing ownership/accounting contract is section 9 of
[memory-pressure-recovery.md](memory-pressure-recovery.md). Preserve reservation
readiness: loans must not consume the protected commitment envelope or control/
teardown headroom. Each additional resource domain needs its own attribution,
enforcement and acceptance evidence before it receives a reservation guarantee.
A tenant reservation shortfall alone is not a shared-domain pressure deficit.

## Authorized envelope

- Preparation: test-only instrumentation, correctness checks and guarded builds.
  Cumulative build wall time must not exceed 10 minutes. Track every new guarded
  build attempt, including failures; prior completed task builds are excluded.
- Campaign: at most 600 episodes, 30 minutes wall time and 30 CPU-minutes in the
  monitored workload process tree, whichever limit is reached first.
- At most two parent requests and three children per parent. Only one application
  instance runs at a time; queued children still count toward the parent limit.
- Sampled process-tree RSS stop threshold: 768 MiB. This is an observer-triggered
  abort threshold, not a kernel-enforced hard memory bound. Record sampling cadence
  and possible overshoot; unavailable required containment metrics block the run.
- Combined campaign logs/artifacts: at most 128 MiB. Bound generation and retention
  as well as the final file size. Include measurement overhead in reports.
- No GPU work, external model downloads, deliberate host OOM, affinity/system
  configuration changes or interference with existing servers. Use ephemeral
  loopback ports and disposable synthetic-model files owned by the fixture.
- Read-only OS counters are allowed. Stop on the first applicable cap, crash,
  peer corruption or unexplained continuing live-owned growth after drain.
  Missing attribution essential to a proposed conclusion blocks that conclusion;
  do not replace it with zero or turn a failed episode into a successful sample.

Record preparation and campaign clocks separately. Campaign wall time includes
startup and cooldown. Watchdogs terminate only their own isolated workload tree;
cleanup must not target processes by a broad executable-name match.

## Actual path and isolation

Reuse the single-instance `server_run` seam and synthetic byte-fallback CPU model
from the real wire fixture. Exercise actual text parsing, authentication,
registered routes, queues/context, readers, serializers and loopback sockets.
Do not substitute fabricated JSON bodies or manually ticked queues as calibration
for those production paths. Private callback-level cases may supplement causal
lifetime evidence, but must be labeled separately.

Keep probes and gates test-only, with no request-visible or CLI fault switches.
Use normal standard containers; no custom allocator or pool is justified here.
Owner-mutated slot inspection belongs on the owner thread. Arming from a client
requires synchronized state. Keep bearer values out of traces and public output.
Synthetic tenant labels support offline analysis only; they do not establish
production credential mapping, admission, borrowing or scheduling enforcement.

## Named workload coverage

Prefer a bounded named matrix, not a full Cartesian product. Record requested and
realized lengths and probability entries, protocol support and rejection outcomes.
Use deterministic request schedules and explicit identity encoding.

1. Native plain/SSE response-size and probability-retention curves: proposed
   output lengths 2/8/32 and probability counts 0/5/20, where actually supported.
2. One/two parents with one/two/three children, including staggered completions
   and a growing child while siblings are already final. Preserve default parent
   errors, immutable child outcomes and existing protocol framing.
3. Native, OpenAI text/chat, Anthropic and Responses protocol sentinels, plus
   supported serialization-heavy corners. Verify parameter mapping rather than
   assuming every protocol accepts the same probability option.
4. Fast consumers, bounded throttled reads, pause/resume and bounded no-read then
   close, each with a healthy peer. Delayed client callbacks do not prove blocked
   server writes. Record actual backpressure as unavailable if not established.
5. Existing selected backend errors, first/late SSE error behavior, cancellation,
   client abort with peer success, response closure and healthy recovery.
6. Queue-held, reader-held, aggregate/provider-held and detached ownership, with
   repeated grow/drain/idle phases. An ordering gate demonstrates causal ownership,
   not real network or serializer latency.

The tiny vocabulary/context limit reachable payloads. Do not inflate requests
past validated context to force backpressure. Do not extrapolate synthetic
probability storage to a production vocabulary or certify real-model service.

## Observation contract

Samples identify run/build/config, monotonic time, named cell, parent/child where
known, nonsecret analysis tenant, process/thread role, resource domain and source.
Classify every field as exact, logical/capacity, measured estimate, shared or
unavailable; record freshness and attribution uncertainty.

- Logical sizes and retained capacities are separate. Object size, capacity and
  serialized text size do not establish exact allocator backing.
- Moves transfer ownership; copies overlap. Dequeue, unregister, callback return,
  cancellation posting and victim selection are not deallocation evidence.
- Include construction, slot retention, queue, aggregation, JSON/dump/provider,
  writer and detached destruction where observable. Mark unobserved intervals.
- Observe allocation/destruction scratch only with bounded, reentrancy-safe probes.
  Do not repeat fail-all-allocation experiments during JSON destruction.
- Keep output-ready/preparation, sink acceptance, actual lower-layer acceptance,
  framing, client receipt and retirement milestones distinct. A successful
  `sink.write` boolean is not a measured payload acceptance quantity or receipt.
- Keep useful progress and resource-service evidence separate. Tokens are not a
  universal execution-cost unit. Missing owner acknowledgement or backend
  quiescence cannot be inferred from delivery retirement.
- RSS includes model/client/logging and allocator retention. Do not sum RSS with
  owned counters or attribute it exactly to tenants. TLS/kernel/shared/device
  costs and heterogeneous work remain unvalidated by this plain CPU pilot.

## Acceptance and reporting

Validate probe balances, transfer-versus-copy semantics, bounded trace handling,
watchdog containment and healthy-peer assertions before campaign execution.
Independently review instrumentation and containment before accepting its data.
Record actual episode counts, skips/failures, achieved repetitions, resource peaks,
measurement overhead, observation gaps and every early stop. Do not claim that an
unreached maximum episode count was completed.

Report observed overlap/count/time curves and retained-versus-live distinctions,
not universal reserve percentages or production p99/SLO guarantees. Analyze
reservation scenarios offline without pretending that replay enforces them.
Only after reviewing evidence may the operator choose numerical ceilings,
reservation sizes, lending headroom, service targets and transport deadlines.
