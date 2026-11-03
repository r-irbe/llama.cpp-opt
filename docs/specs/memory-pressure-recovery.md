# Runtime memory pressure recovery

Status: agreed requirements; staged implementation, not a shipped capability.
Related: [CUDA graph memory investigation](cuda-graph-memory.md).

## 1. Objective

Recover from rising accelerator memory demand without silently changing model,
context, attention, sampling or KV correctness. Cover all eligible allocation
classes incrementally. Reclaiming retained storage, moving live data and fixing
leaks are separate operations, with separate evidence.

The existing managed allocator, device-only VMM pool and host-cache scaffolding
do not constitute a complete automatic spill controller. An environment switch
to managed memory alone is not sufficient coverage.

## 2. Policies

Provide a runtime configuration selecting:

- `adaptive`: prioritize completion. Reclaim inactive resources, migrate
  eligible data within host budgets, reduce execution microbatch size when
  correctness permits, and rebuild/disable graphs at safe boundaries. Queue or
  return a controlled resource failure if recovery cannot establish capacity.
- `latency-first`: bound the recovery work permitted for a request. Prefer
  reclamation and admission control to migration thrashing. Spill only within
  an explicit recovery/latency allowance; defer or reject work otherwise.

After correctness and live-driver acceptance gates, adaptive recovery becomes
the default with an explicit opt-out. Until those gates pass, unfinished policy
must not silently become operational default. Adaptive mode may defer admission
within a valid finite request/server deadline; latency-first honors its recovery
allowance. Missing/expired wait budgets produce a typed failure, not an
indefinite queue. This applies before execution, not replay of partial work.

Both policies preserve correctness and report degraded operation. No unbounded
retry loops. Policy cannot overrule unsupported hardware or host resource
limits. Numerical equivalence tolerances must be declared in verification;
execution microbatch changes must not be described as bitwise deterministic.

## 3. Cross-backend contract

Backend-neutral records describe:

- capabilities: device/local memory, managed migration, GPU-accessible host
  memory, supported pool trimming, synchronization and graph invalidation;
- allocation class: weights, experts, KV, scratch/compute, staging, graph
  resources and backend/library overhead;
- ownership: logical live bytes, reserved backing, migratable/reclaimable
  bytes, pinned bytes and in-flight references;
- snapshot: device headroom, host available RAM, cgroup headroom, sample time,
  confidence and unavailable counters;
- recovery result: bytes actually reclaimed/migrated, work/latency cost,
  completion fence, residual risk and typed failure reason.

Missing capability or accounting is explicit, never fabricated as zero. Do not
count model/compute buffers again as pool overhead. Driver-owned graph overhead
may be unobservable: label estimates instead of inventing exact per-graph bytes.

Rollout: common interfaces and deterministic policy tests; NVIDIA/Linux adapter
and tests; AMD/HIP adapter and platform validation. Unsupported backends retain
existing operation and expose that recovery is unavailable. Unified coherent
hardware and discrete devices use separate detected capabilities, not names or
hardcoded capacities.

## 4. Continuous measurement and tuning

Re-evaluate before substantial growth, after allocation failures, at bounded
serving intervals, and after recovery. Device/host snapshots are advisory, not
reservations. Other processes, cgroups, pinned memory, OS pressure and pending
allocation demand can change the usable budget.

Honor explicit reserve/spill limits. Derive defaults and working-set choices
from actual capabilities and measured peaks/costs, not hostnames, model names,
fixed graph cardinality or a universal reserve percentage. Protect a host/cgroup
reserve and allow lower headroom to invalidate a previous admission decision.
Use hysteresis, bounded exploration and cooldown to avoid oscillating placement.
Do not explore unsafe placements to learn a performance reward.

Pressure warnings and alerts need complementary evidence from different source
families where available: device/process accounting, owned allocation/pool
counters, host/cgroup constraints, pressure stalls and allocation outcomes.
Track provenance, freshness, scope and confidence. Multiple wrappers around the
same underlying counter are not independent corroboration; overlapping bytes
must not be summed. Conflicts and missing sources reduce confidence rather than
being hidden by a majority vote. A definitive allocation/I/O failure or hard
limit remains an immediate constraint even without supporting telemetry. When
available evidence cannot establish a safe budget for affected growth, defer or
fail within the finite deadline. No fixed alert quorum has been specified.

Cache tuning must observe split-graph reuse distance and hit/miss/capture
rates. A default 64-entry cache produced 34501 captures in a short reproduction
and is not an accepted working-set policy. A cache capacity override can remain
an explicit operator control, not a substitute for pressure accounting.

## 5. Safety and residency

Managed memory must be capability-gated and allocated through the actual paths
that need migration; default device-only VMM scratch bypasses the current
managed allocator. Existing `cudaMalloc`/device `cuMemCreate` backing is not
converted into managed memory by advice or by changing a flag.

A relocation needs a quiescent boundary: block conflicting submissions, finish
relevant readers/writers, preserve authoritative data and metadata, publish new
residency only after completion, and retire old references safely. Reallocation
can invalidate views, graph arguments and copy-pointer tables. Pointer-stable
managed migration does not remove synchronization requirements.

Graph invalidation, eviction, preparation failure, hot-swap and teardown must
share an ownership discipline: correct device, in-flight completion, no stale
`cur_graph`, no concurrent destruction/submission. Do not blindly replay a
failed graph launch; earlier asynchronous work may already have modified KV.

Graph executable/library storage is not assumed spillable. Reclaim unused
resources or change execution mode at a safe boundary. Failure must propagate
through the scheduler and public decode/request paths, not silently continue
with missing outputs. Audit those callers before adding recoverable errors.

## 6. Staged acceptance

1. Repair confirmed allocation-owner defects and graph-invalidation barriers.
   Test null initialization, repeated cleanup, device ownership and in-flight
   retirement. No claim of leak freedom from these tests alone.
2. Add trustworthy accounting and framework tests: dynamic budget reductions,
   cgroup/host constraints, missing metrics, policy choice, hysteresis, bounded
   retries, non-migratable resources and controlled failure propagation.
3. Validate eligible NVIDIA allocation paths, including the actual scratch
   pool, with forced allocation pressure and output/KV consistency checks.
4. Compare graphs off/on, repeated/varying shapes and warmup/idle/teardown
   plateaus. Record GPU per-process usage, host RSS/PSS, pool live/reserved,
   graph cache events and external GPU use. Separate leaks, high-water pools,
   deliberately retained checkpoints and driver caches.
5. Measure adaptive and latency-first trade-offs under changing external
   pressure and host-budget reduction. A passing request does not demonstrate
   acceptable migration latency or absence of cache thrash.
6. Add AMD/HIP tests and hardware evidence before claiming equivalent support.

Current authorization is host tests/fault injection first; present the exact
configuration and time/resource limits for new live-GPU approval separately.
No new live-GPU pressure run or disk stress is authorized by this interview.

Long GPU stress runs require an explicit time/resource budget. Build all
C++/CUDA changes through the repository build guard. No benchmark claim without
primary logs and binary/configuration provenance.

## 7. NVMe and Linux locking contract

The adaptive hierarchy includes VRAM -> RAM -> configured NVMe storage. The
initial NVMe tier is a disposable, validated per-process cache, not durable or
restart-compatible session state. A read-only file-backed weight can be
reclaimed/refaulted rather than copied to another file. Dirty KV needs explicit
serialization; scratch may be discarded only when safely recomputable. CUDA
managed migration and OS swap are not application NVMe spill.

Require a server-owned configured root and private process directory/files.
Never delete another process's cache on startup. Publish complete files via
same-filesystem atomic rename; retain authoritative RAM state until successful
write/validation/publication. Restore validates format, expected identity,
length and integrity before publishing data. Bound serialization staging and
account for source plus dirty/writeback page cache; a spill label is not proof
of reclaimed backing. Unknown/invalid files trigger safe recomputation, not
unbounded allocation or partial state publication.

Adapt disk admission/rates from current free bytes/inodes, configured reserve,
capacity/write limits and measured I/O cost. Handle ENOSPC/EDQUOT, short writes,
late I/O errors, corruption and cancellation. Observe confidentiality boundaries
with restrictive permissions and process/tenant separation. Plaintext at rest
is possible unless the configured filesystem is externally encrypted; no
application key-management or secure-erasure promise is made.

Linux host admission uses MemAvailable plus applicable cgroup ancestor limits,
pressure/stall and locked/pinned accounting. Do not sum overlapping counters or
assume MemFree + Cached is a reservation. mmap success is not guaranteed future
residency; record exact syscall/errno, allocation type and relevant process
limits. Distinguish anonymous HugeTLB allocation from THP advice.

Locking/pinning is adaptive and bounded by default, with a strict option that
fails explicitly when requested guarantees cannot be maintained. Do not use
global drop_caches, swap/sysctl changes or unbounded mlock as recovery. Reclaim
cold file-backed pages only when no live ownership/locking contract forbids it.

### Implemented prerequisite: operational checkpoint spill

`common/spill-file.h` now backs the server's operational spill path with
private per-process directories, exclusive 0600 temporary files, checked writes,
flush/close checks, bounded readback verification and same-filesystem rename.
The RAM source is retained on failure and released only after publication.
Restore compares trusted metadata and checksum before publishing output.
Startup does not delete other instances' files; shutdown removes owned entries
and attempts to remove only its empty directory. Crash orphans are deliberately
not scanned/deleted automatically. External storage encryption remains required
for an encryption-at-rest guarantee.

`test-checkpoint-spill` covers injected ENOSPC, short writes/EINTR, failed
publication, source/previous-file preservation, isolated directories, private
permissions, symlink rejection, corruption, truncation, identity/length rejection
and unchanged restore output on failure. The linked server builds. This is a
POSIX-tested prerequisite, not a completed adaptive NVMe tier. Existing public
slot-file serialization is separate and has not been hardened by this change.
Disk admission, pressure-aware restore allocation, write-rate policy, orphan
budgeting and adaptive locks/pins remain required follow-up work.

## 8. Evaluation failure prerequisite

Scheduler allocation and computation failures must reach decode/encode and the
server before outputs are consumed. Preserve the public convention: zero is
success, positive one is a KV-slot warning eligible for caller batch reduction,
negative is failure. Translate backend cancellation (positive in ggml_status)
to public cancellation (-3), not a KV retry. Allocation failure remains -2.
After any successful microbatch, a later slot warning is terminal for that
logical batch; no blind replay of already committed recurrent/KV writes.

Failures invalidate published output counts, mappings, draft results and reused
graphs after outstanding writes drain. Successful encoder source state is
input to decoding, not a failed decode's output: preserve it on decode warnings
and errors and on KV-maintenance failure. A failed encode invalidates its source
state instead. This is not transactional KV rollback:
callers must clear/reprocess or restore a validated checkpoint before reusing
possibly partial state. Server errors distinguish compute/allocation failure
from exhausted KV capacity and discard affected prompt-cache state. Existing
checked CUDA aborts and asynchronous completion errors remain separate gaps;
propagating returned statuses does not itself convert those aborts to recovery.

Host regression uses tiny synthetic models, injected CPU backend statuses and
scheduler allocation failures through public decode/encode. It covers
first/later microbatch failures, cancellation mapping, no replay and stale-output
rejection. Deferred host output copies remain queued at the failing microbatch
and must drain while outputs are still live, before invalidation. A synthetic
T5 runs real encoder/decoder graphs: successful encode, early KV warning,
source-state preservation, successful smaller decoder retry, then failed encode
invalidating the previous source. KV-maintenance failures must not clear pending
flags or report success. Production-model encoder fidelity, recurrent/private KV
restoration, driver errors and full multi-slot HTTP integration still need their
own coverage; do not infer them from these fixtures.

Parallel graph scheduling retains each backend worker's first reported failure,
so a later successful split or different error cannot overwrite it. Workers
continue the established barrier/event protocol rather than exit early and
strand their peers. After host workers finish, a failure path synchronizes all
backends before returning the terminal status. Multiple failing backends use
the existing deterministic backend-index selection, not a claimed chronological
global first error. Continuing the protocol does not mean that dependent work
produced valid results; failed evaluations must still be discarded, not replayed.

`test-parallel-split-failure` exercises the real scheduler using CPU storage and
mock compute on three backends, with several splits per evaluation. It checks
first/middle/final failure, cancellation, allocation failure, a later distinct
error, error-path fencing and status reset on pool reuse. The configured OpenMP
build checks both the OpenMP region and persistent-worker route through runtime
assertions, not just eligibility labels. This establishes a host scheduling
contract, not live-device failure handling, driver recovery or transactional KV
restoration. Those and full HTTP/multi-slot lifecycle remain separate gates.

### Multi-slot result delivery prerequisite

A terminal evaluation failure in a shared batch must deliver one terminal error
to each still-connected active request before releasing slots. This includes
backend cancellation: a disconnect can unregister its own waiting ID, but must
not leave a different connected request waiting forever. Use existing unavailable
(503) for allocation failure and server-error (500) for generic failure or shared
backend cancellation. This is not an HTTP status for the disconnected client.
This is containment for genuine terminal backend failure, not the normal request
cancellation policy: disconnecting one request must not manufacture a shared
batch failure for connected peers (see section 9).
Do not replay the partially computed batch. Clear active and idle slot KV/token
metadata and allow deferred work only after normal release/system-reset processing.

The initial `test-server-eval-failure` uses real context initialization, task
inbound dispatch, pointer-result queues and update_slots with synthetic CPU
weights. Inject a second-microbatch failure after one real successful compute;
prove two active independent request IDs, idle-prefix invalidation, terminal
result serialization and subsequent deferred-request success. Keep injection in
the test backend interface, not production flags. RAM prompt-cache/checkpoint
restoration, real HTTP readers/routes, SSE, disconnects and queue shutdown need
separate integration coverage. Do not call context/queue coverage HTTP proof.

### Reduction ownership prerequisite

An active CUDA reduction must validate the opaque execution-group registry before
any device side effect, graph lookup, capture or cached launch. Its caller must
be a registered primary, and every participating/copy destination primary must
still be a registered live context with the matching device. Absent ownership
returns a terminal compute failure; a surviving duplicate is not implicitly
promoted. Disabled reductions and graphs without active reductions retain their
existing behavior. The group key is not universally a model pointer: inference
passes a llama_context, loader backends a model, and generic backends may use null.

Teardown takes exclusive host retirement, excludes captures, drains all devices
owning group streams/pools/handles (including cross-device resources), then retires
every group cache before destroying any peer-referenced resource. A failed drain
must not free resources. Driver errors still follow the existing fail-closed
CUDA_CHECK contract, not recoverable driver-error handling. Raw reduction copy
storage and pool members must be released within that ownership guard rather
than leak or be freed after the guard is gone.

Host mocks test production validation/retirement helpers, missing/copy-only
participants, duplicate ownership, group isolation and fence-before-retire order.
They do not demonstrate live multi-GPU completion or HIP/MUSA compilation.
Reduction buffer growth can separately invalidate captured pointer arguments;
resource versioning at resize remains a distinct follow-up, not solved by
teardown safety or ownership validation.

## 9. Lifecycle, response accounting and OpenMP acceptance

Status: agreed requirements after read-only investigation; not implemented or
validated by the earlier context/queue and scheduler fixtures.

### Cancellation and coverage

Request cancellation must preserve innocent peers. Do not implement it using
unsynchronized HTTP slot inspection, a process-global stop flag, failure of the
shared decode, or replay of partially committed rows. Track distinct milestones:
delivery retirement, owner acknowledgement, last submission, work quiescence,
sequence/checkpoint ownership retirement and slot reuse. Measure these before
selecting a numeric service target; no hard preemption guarantee is established.

A smaller successful decode call alone is insufficient. Submitted arrays remain
immutable; committed server counters, output mappings, sampler advancement and
speculative acceptance must reconcile before another transaction or retirement.
Drain asynchronous owners before reuse. Sequence removal must preserve live
cohort membership and check model-specific/private/companion state contracts.
Do not use the current non-null-model partial-reuse predicate as certification.

Release acceptance covers every configuration the server currently supports,
including applicable private/recurrent caches, cohorts, speculation, multimodal
and encoder modes. Preserve existing unsupported-mode restrictions. Narrow
causal fixtures are intermediate milestones, not permission to certify the
remaining modes or silently remove supported ones.

Full HTTP acceptance uses shared production parsing, applicable middleware,
route registration, templates, real task/context processing, readers and wire
serializers. Test-only token adapters or manual queue ticks are not substitutes.
Cover final-first SSE, ordinary partial/final streams, first/later errors,
fanout, disconnects with surviving peers, recovery and shutdown.

### Result ownership and shutdown

The first implementation slice repairs destruction only. Keep the public base
pointer alias/default deleter and legacy by-value APIs. Add public virtual base
destruction and explicitly default construction, copy and move operations so a
user-declared destructor does not silently turn rvalue base payloads into copies.
Keep member-derived exception specifications and existing initialization behavior.
This changes the internal vtable ABI; rebuild affected server translation units.

A focused host fixture must use the real production response queue: compile-time
virtual-destruction/type-trait gates, sentinel destruction exactly once through
base ownership, registered send/dequeue, queued teardown, rejected temporary versus
rejected named-base-pointer ownership, middle-entry erase/relocation, legacy
value roundtrip and actual parent/child aggregation. Verify nonempty vector backing
transfers on moves rather than relying on move-constructible traits alone. The
pre-repair gate should fail deterministically at compile time; do not execute
undefined base-only deletion to obtain an alleged reliable runtime red.
This slice does not implement unregister purge, closure, cancellation or budgets.

Initial implementation evidence: `test-server-result-lifetime` uses production
`server-queue.cpp` and links real `mtmd` for task-owned media deleters. The
virtual-destructor compile-time gate failed before repair, then the target and
server/context-regression rebuilt successfully through the guard. The selected
server pair passed, and lifetime checks passed 30 consecutive repetitions. Fresh
clangd checking of the fixture reported no errors. This is host ownership evidence,
not sanitizer, cross-platform, HTTP, backend-quiescence or bounded-memory proof.

Repair polymorphic result destruction before relying on payload reclamation.
Retire registration and detach abandoned queued results atomically under the
existing response mutex; destruct outside it. Keep unrelated results and legacy
parent/child aggregation intact. Delivery retirement is not backend quiescence.
Both receive variants need synchronized explicit closure rather than normal
shutdown calling terminate or waiting indefinitely for an absent result. Use
request-specific predicates and a single absolute receive deadline; unrelated or
spurious wakeups must not renew it. Preserve zero-time ready-result polling.
Delivery retirement and global shutdown/owner-sealed draining remain distinct.

Receive-to-wire implementation contract (next stage): add explicit typed
result/closed/retired/timeout outcomes to both pointer and legacy-value receive
families without manufacturing a default successful payload. Under the response
mutex, classify a matching queued result first, then sealed closure, then absence
of all selected registrations, then timeout. A partially retired ID set can still
receive surviving IDs; callers track outstanding IDs so retired missing work
cannot be mistaken for completed fanout. Empty or never-registered selection is
retired on an open queue. A ready result wins even at a zero/expired deadline;
perform the same protected scan after timed wakeup. No strict enqueue-before-
deadline timestamp rule is implied.

Use one caller-supplied absolute steady-clock deadline per typed receive; legacy
relative adapters calculate it once, clamp nonpositive duration to ready polling
and saturate overflow. Infinite receives wait on the same synchronized predicate
without converting the maximum time point into an overflowing OS timeout.
Synchronized idempotent closure seals new registrations, parent creation and
publication; existing queued results may drain first. Closure does not itself
unregister/destroy queued or caller-owned results, cancel children, authorize late
success after shutdown grace, or implement owner-sealed drain membership.

Compatibility adapters preserve valid legacy value and pointer-result ownership
paths; value receive raises an explicit non-result exception rather than returning
an uninitialized/default success. Pointer compatibility polling may return null
for non-result states and is not sufficient for production terminal handling.
Migrate every production consumer to explicit outcomes before accepting this
stage, register IDs before posting on all control routes and retain failure
cleanup. Production JSON/SSE acceptance includes final-first streams and legacy
parent child-error propagation; host-only tests do not meet the stage boundary.

Loopback acceptance now uses `test-server-wire` linked with the actual server
application. The CLI delegates to a shared single-instance `server_run` entry;
its caller-owned context outlives the call. A ready callback runs after model and
route initialization, before HTTP workers start, and receives the bound port.
Port zero uses the HTTP library's ephemeral bind. This is an internal application
seam, not a reentrant/multi-server embedding contract or request-visible fault
switch. Process-global logging/signals remain single-instance.

The wire fixture uses synthetic CPU weights with real byte-fallback text
vocabulary, JSON parsing, authentication middleware, registered native/OpenAI
completion/chat/Anthropic/Responses routes, queues/context, serializers and TCP
loopback. Test-only backend-interface injection is installed on the owner before
HTTP starts; later arming uses atomics, and slot inspection occurs only in the
owner compute wrapper. Bounded client/owner gates establish late-error and client
abort ordering without production fault controls. Failures after a successful
microbatch test fanout terminal statuses and subsequent recovery. Late SSE errors
omit normal DONE; a real client abort with another registered request preserves
peer success. Owner-sealed response delivery yields a terminal HTTP error.

Wire protocol checks parse healthy JSON/SSE data and assert family-specific
shapes/terminators. Caller server environment overrides are removed only inside
the test process. Model fixtures honor TMPDIR; CTest uses the build directory so
unrelated system temporary-file inode exhaustion cannot break model creation.
Guarded builds and 30 repetitions each of wire/context/lifetime fixtures passed.
Hostile model/port environment overrides were also exercised successfully. This
is CPU loopback evidence, not TLS/compression, sanitizers, arbitrary contention,
real-model/GPU pressure, numeric cancellation latency, or owner-sealed shutdown
grace/drain proof. Malformed JSON is checked for explicit non-success, not a newly
promised HTTP status-conformance change. Independent source review found no
issues and accepted the bounded CPU receive-to-wire stage (OK with notes), including
application/CLI lifetime, actual wire paths, selected faults and the legacy parent
fold. The reviewer did not independently execute the reported tests. This closes
that bounded stage only; broader lifecycle policies below remain separate
implementation requirements.

Legacy control requests use one shared helper that assigns/captures an ID before
moving the task, registers before publication, returns a typed legacy-value
receive outcome and retires registration on normal/exception exits. Reject closed
registration without posting. Every route must check non-result before reading
payload fields; closed/retired delivery yields a 503 error, not default 200 or
client-disconnect silence. A mutating operation may already have executed when
its response delivery ends; do not promise rollback or automatically retry it.
Retirement is not command cancellation or owner/backend quiescence.

The production SSE provider must check outstanding work after flushing the first
result, including empty serialized events. A final-first response must terminate
without another receive. Apply the same protocol/explicit send-done policy to
first-final and later-final results; check DONE writes. Do not emit normal DONE
for terminal errors, disconnected readers or failed payload writes. A delivered
error event may complete transport framing without becoming successful inference.
Under vendored `write_content_chunked`, successful provider completion calls
sink.done and returns true; false denotes cancellation, even after done. Failed
writes return false without requesting successful final framing. Sink-level write
acceptance and void done do not prove socket/client receipt; actual framing error
state belongs to the HTTP writer. This callback contract does not implement the
agreed transport timers, byte budgets or owner-sealed shutdown policy.

The shared production reader now lives in `server-response-reader.h` and receives
only outstanding IDs, not already-completed registrations. Closed/retired delivery
maps to ordinary 503 error results, while null/is_terminated denotes actual
transport stop or exhausted-reader use. Successful registration must precede task
publication; no work is posted when the response queue is already sealed. Batch
handling uses the virtual error contract rather than the legacy base boolean.
Cancellation targets remaining IDs only, is idempotent after successful posting,
and does not scan owner-mutated slots or trigger a process-global decode abort.
Owner-queued cancellation is not a numeric promptness or backend-quiescence proof.

Shared-reader host regressions reproduced closed-as-null before migration and now
exercise closed/retired errors, post rejection, completed-ID/remaining-retirement,
virtual error aggregation, out-of-order successful indexing, cancellation targets
and peer delivery on transport stop. Server and focused fixtures build through
the guard, with both tests passing 30 repetitions each. Independent read-only
reader review returned OK with notes, with no demonstrated correctness blocker.
Its exception-path coverage note was addressed with one-shot test-only allocation
failures: partial task publication cleanup preserves original IDs and retires
registrations; cancellation-buffer failure and partial cancellation publication
remain retryable; successful retry makes later stops inert. Tests assert a real
partial prefix, all target IDs and preservation of unrelated queued work rather
than assuming a fixed deque allocation count. The selected failure disables itself
before recovery; no model/JSON teardown failure or total-OOM safety is implied.
The exercised queue/standard-library allocation path and serialized reader/unique
ID assumptions limit the evidence. The focused pair again passed 30 repetitions
each after this test-only remediation, which was not independently rerun/reviewed.
These are not parsing/route/middleware/HTTP/SSE or full-stage acceptance results.
On cancellation allocation/publication failure delivery is retired first, failed
publication can be retried, and destructor failure is explicitly diagnosed rather
than thrown. This does not supply the allocation-independent control reserve or
prove cancellation completion; owner work may remain live on that failure path.

Queued-retirement implementation uses standard node lists for internal response
queues and parent records so matching nodes can be spliced to caller-owned
retirement holders without payload moves, copies or compaction. Construct holders
before locking; failed holder setup must leave registration/queues unchanged.
Unlink delivery rights and matching nodes under the response mutex, acquiring the
task mutex only in the existing response-to-task order for parent records. Destroy
all detached nodes after both locks release, preserving survivor order. No custom
pool or performance gain is implied; node allocation overhead remains part of
future accounting, and platform allocation behavior needs validation.

Queued-retirement host validation rebuilt the server, context regression and
lifetime target through the guard. The focused pair passed and each passed 30
consecutive repetitions. Checks exercise singular/bulk/repeated purge, survivor
order, derived destruction and nested-backing reentry, producer order, already
received ownership, legacy parent retention/late creation/duplicate updates,
allocation-failed publication/aggregation, actual owner-loop finish and peer
progress while a detached destructor remains blocked. This is not closure,
transport, sanitizer, cross-platform, backend-quiescence or bounded-memory proof.
Independent read-only review found no issues and accepted this bounded
queued-delivery retirement slice with notes. It did not rerun execution evidence.
Its residual limits include deterministic producer order rather than arbitrary
contention coverage, sequential retired-before-create testing plus inspected
locking, and no dedicated backing-watch probe around legacy receive. Neither
review nor test results complete the remaining lifecycle requirements.

Important teardown constraint: vendored `json_value::destroy` flattens nonempty
objects/arrays through a heap-backed vector, reserving/growing scratch during
destruction (`vendor/nlohmann/json.hpp`). A fail-all-allocations retirement probe
aborted here under the result destructor; a debugger catch-throw trace confirmed
the path. Node detachment itself does not make payload destruction allocation-free
or safe under total heap exhaustion. The revised fixture probes retirement-time
allocation and backing deallocation outside both locks, with separate fallible
publication/aggregation checks. Future pressure admission/reserves must account for
teardown scratch and define a safe exhaustion strategy; no vendor modification or
OOM-safe destruction claim is included in this slice.

Include abandoned incomplete legacy parent aggregates: their retained completed
children are owned outside the response queues. Preserve response-to-task lock
ordering and never add reverse nesting. Explicit queued detachment must consider
assignment/erase and exception cleanup, not just explicit destructor calls; neither
unregister nor a nominal cancel establishes reclaimed backend/writer capacity.

For mixed nonstream fanout, default to the compatible parent-level error while
keeping committed child outcomes immutable. Add partial child success/failure
outcomes only as an explicit per-request/protocol option with documented framing
and acceptance tests. Never silently return an ordinary successful incomplete
aggregate or change the default wire contract.

The legacy production parent fold now ORs child error flags rather than ANDing
from false, and value-initializes its outgoing base result. A regression through
actual `server_context::on_finish_multitask` and legacy response receive failed on
mixed children before repair, then passed for both mixed orders, all-error,
all-success and empty parents. Checks preserve child records and existing results
payload shape. Guarded server/fixture builds and 30 repetitions of each focused
test passed. This does not demonstrate a parent HTTP response or implement the
separate opt-in partial-outcome protocol. Independent stage review accepted this
fold together with the bounded CPU wire coverage, not a new legacy parent HTTP
protocol or the broader lifecycle program.

Stop admission, allow a configurable grace initially defaulting to 30 seconds,
then publish controlled terminal outcomes and cancel safely. Preserve committed
successful completions and allow bounded draining of already queued output;
active truncated requests must still terminate without normal success. Freeze
success eligibility at grace expiry, then seal active-request drain membership
on owner acknowledgement. Output from already-submitted work may join the drain
until that acknowledgement, subject to transport caps; a late final cannot upgrade
active-at-expiry work to normal success. Preserve previously committed successful
completions and protect all still-live writer/backend ownership independently.
Wake readers before joining HTTP workers. Grace expiration never authorizes destroying live buffers,
backend objects or state referenced by a writer. Record residual drain time
separately from the grace deadline.

### Comprehensive response-memory accounting

Numeric ceilings, reserves and deadline defaults are gated on representative
calibration followed by operator approval, not provisional invented values.
Measurement scope and execution budget must be agreed before starting a campaign.
The selected tenant policy is protected reserved capacity with controlled lending
of unused shares, not ordinary weighted sharing. Specify reservation domains,
borrowing eligibility, commitment headroom, safe reclamation and service guarantees
before claiming isolation; already-committed/live borrowed ownership cannot simply
be revoked or credited as free when the reserving tenant returns.

The selected promise protects reservation readiness: lend only when independent
headroom and the admitted commitment envelope preserve the protected admission
capacity. Apparently idle reservation is not automatically lendable. Control,
terminal-transition and teardown headroom are never lendable. Restricting a loan
to currently uncommitted work is insufficient: its future commitments and live
retirement lag must also fit. A reservation shortfall is distinct from a shared-
domain deficit and does not by itself authorize cross-tenant active retirement.

The end-state contract covers all individually validated domains, including owned
storage, service and relevant platform/shared/device resources. Each domain needs
explicit attribution, enforcement and acceptance evidence before any guarantee
is claimed; missing capability means no guarantee, not zero consumption. CPU
loopback cannot certify the additional domains or hard service/preemption bounds.
Calibrated quantities and enforcement remain open work.

Cross-tenant pressure retirement requires a demonstrated residual shared-domain
deficit after prior remedies. Tenant-local exhaustion alone is not sufficient.
Prefer over-entitlement eligible active whole parents with demonstrable useful
reclamation in the affected domain; preserve committed completions and all live
ownership. No safe useful victim means fail closed and report the residual deficit.

Combine layered bounds with allocation attribution across construction and slot
accumulation, queues, parent/fanout retention, serializers, response body/provider,
write intermediates and detached retirement. Account for allocator backing and
overhead, transport/TLS/socket buffers and relevant kernel/shared domains too.
Separate exact owned allocations, retained capacity, measured attribution,
documented estimates, shared/unattributable values and unavailable metrics.
Do not claim exact RSS or kernel/TLS attribution from payload sizes, double-count
overlapping counters, or silently treat unavailable evidence as zero.

Ownership moves transfer accounting; copies add overlapping charges. Do not
return credits at dequeue or a releaser callback while included storage remains
live. Include producer construction peaks and bounded deferred-destruction
allowances. Bound both bytes and counts, including parent/control metadata.
Use overflow-safe atomic global/parent admission and explicit oversized-result
handling; fanout must not multiply one parent's allowance without limit.

Use adaptive admission targets under configured hard ceilings and reserves,
reassessing host/cgroup constraints with provenance, freshness and hysteresis.
Missing metrics require bounded fallback, never unlimited growth. This telemetry
and controller remain implementation work. No numeric thresholds were measured
by the read-only investigation.

Producers must not wait for client progress. Budget exhaustion requires a bounded
internal terminal transition independent of ordinary exhausted allocations; a
normal allocated error result or unbounded cancellation overflow is insufficient.
Distinguish internally recorded failure, transport write and client receipt.
Never silently drop data and report success. Exact fairness/victim policy,
ceiling defaults and deadline values remain to be specified; do not invent
numeric defaults from this accounting requirement.

Pressure handling preference is: reject new admissions, then fail the growing
parent, then policy-selected victim retirement as the last fallback. Rejecting
new admission does not itself release committed storage. Never block an existing
producer waiting for a client or reuse credit before actual retirement. Victim
selection is restricted to uncommitted active whole parents. Correctively reclaim
abandoned ownership first; connected committed completions are not pressure
victims. Prefer over-entitlement ownership in the affected resource domain;
cross-tenant escalation requires explicit shared-deficit rules. If no safe useful
victim exists, fail closed and report the residual deficit rather than free live
storage. Transport expiry remains a separate basis for aborted delivery, not a
permission to sacrifice committed results as pressure victims. Define remaining
tie-breaks and escalation rules explicitly.

Acceptance includes global/parent bounds, authenticated-tenant quotas and an
explicit admission/progress fairness policy. Resolve identity through explicit
operator-controlled credential-to-stable-tenant mapping. Mapped rotation keys
share entitlement; shared-key clients share their tenant allowance; anonymous
work uses an explicit shared anonymous pool. Cohort names and client-asserted
identity fields are not authenticated principals. Do not expose bearer secrets.

Protect useful progress and resource service separately, with distinct ledgers
and eligible-wait/no-service evidence. Include applicable prefill, decode,
embedding/encoder, media and speculative work without treating useful tokens as
universal execution cost. Define attribution uncertainty, entitlements and
anti-starvation rules before promising numeric guarantees. Per-parent ceilings
alone do not implement scheduling fairness; runtime calibration is outstanding.

Transport enforces configurable no-progress and absolute-duration caps. Only
positive lower-layer acceptance attributable to response payload bytes resets
progress. Provider invocations, heartbeat/framing-only traffic and compressor
input with no payload accepted downstream do not. Socket/TLS acceptance is not
remote application receipt; validate classification for each supported backend.

Run the stall timer while output awaits preparation/delivery, including
serialization/compression and blocked writes. Suspend it during legitimate
inference-only gaps with no pending output; suspension is not a progress event
and must not create a fresh budget. Start the non-resetting absolute cap when the
parent's first output becomes ready, including subsequent inference gaps, retries
and final framing. Provider callbacks, shutdown and drain phases cannot restart
it. Initial admission/inference time remains separately accounted, not covered by
this delivery cap.

Expiration can abort transport but cannot destroy buffers used by a writer or
backend. Numeric defaults require evidence; none were established here.

### OpenMP correctness and engine placement

Engine policy owns CPU budgets and placement, accounting for initialized OpenMP
places/partitions, permitted OS/cgroup CPUs, nesting and native workers. Validate
effective placement; never widen allowed domains or repeatedly fight runtime
binding. Use compatible runtime mechanisms or a supported native fallback.

Scheduler backend ownership requires all owners. Validate actual OpenMP team
size before any submission; a uniform zero-work decline can join then use the
existing persistent fallback. Do not mark incomplete teams successful, replay
submitted work, or abandon matching barriers on failure. Preserve CPU kernel
actual-team adaptation and terminal all-backend drains. Host joins are not GPU
completion fences.

The constrained-team implementation now publishes one actual-width admission
inside `omp single`, retaining its implicit barrier, before the entire dispatch
body. A short team submits no work, joins normally, then uses the existing native
pool. No global OpenMP setters or affinity policy changes were added to production.

The strengthened host fixture gives every backend real compute work, including
one CPU-owned split on the highest backend. Mock operation support prevents the
scheduler's compatible-CPU priority upgrade from erasing intended ownership.
Each evaluation, including successful reuse after errors, checks exact backend
submission counts (3/2/1), distinct owners, execution route and existing failure/
fence semantics. Before repair, the normal full-team test passed while limits one
and two and serialized nesting returned success with omitted owners. After repair,
all four scheduler cases passed, and those plus the three server regressions
passed 30 repetitions each. Nested coverage probes an active outer/serialized
inner team and explicitly skips if the runtime cannot supply that context.
OpenMP-specific tests are gated on the actual production directory definition;
this is not a comprehensive C/C++/package capability-unification implementation.
Independent source review accepted the production guard. A second review found
that thread-limit tests must not expect an OpenMP 3.0 environment capability on
older runtimes: only those two registrations now require detected C++ OpenMP 3.0
or later; the normal and runtime-probed nested cases remain available. All seven
scheduler/server tests again passed 30 repetitions after this CMake correction;
focused portability re-review found no issues and accepted the correction.
The bounded constrained-team dispatch stage is complete. Reviews did not
independently execute the supplied tests. These are CPU-storage/mock-compute dispatch checks, not
numerical, GPU, copy-event ownership, placement or performance proof.

Make copy/event publication ownership and device selection explicit, separately
from reduced-team handling. Repeated concurrent records are a protocol concern,
not evidence of observed driver corruption. Share actual-enabled OpenMP build
capabilities across production, tests and package exports; probe optional C/C++
features and retain supported fallbacks. Places/affinity diagnostics and optional
OMPT are candidates; tasks/dependencies, persistent OpenMP teams and allocator
policies require separate proofs and measurements. No performance gain, loaded
runtime identity or live-GPU correctness is established by source inspection.

## 10. Primary references

- [NVIDIA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
- [CUDA driver VMM API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__VA.html).
- [CUDA graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
- [AMD HIP unified memory](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html).
