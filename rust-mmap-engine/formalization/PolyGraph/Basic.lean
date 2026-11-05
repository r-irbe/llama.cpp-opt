/-!
# PolyGraph Plan-Buffer — formal specification

Certified semantics for shape-pinned speculative verification (PolyGraph).

Design spec: `docs/specs/polygraph-plan-buffer.md`
Codegen contract: `common/spec-verify-planner.h` mirrors every definition here
1:1 (each C++ symbol cites its Lean counterpart). The conformance harness
`tests/test-polygraph-plan.cpp` exhaustively re-checks the theorems below for
all plans at `gammaMax = 16`.

Track legend:
- `Limits`   : compile-time verification geometry (γ_max, sequence ids)
- `Plan`     : the 28-byte host-authored plan buffer semantics
- theorems   : the safety properties the CUDA implementation must preserve
-/

namespace PolyGraph

/-! ## Geometry -/

/-- Compile-time verification geometry. -/
structure Limits where
  /-- Maximum drafted tokens per verification step (γ_max). -/
  gammaMax : Nat
  /-- KV sequence id of the serving slot (real rows). -/
  slotSeq : Nat
  /-- KV sequence id carrying padding rows (rewritten every step). -/
  scratchSeq : Nat
deriving Repr, DecidableEq

/-- Total batch rows is pinned: every verification step launches γ_max + 1 rows. -/
def Limits.rows (g : Limits) : Nat := g.gammaMax + 1

/-- Batch rows are indexed `0 .. rows-1`. -/
abbrev Row (g : Limits) : Type := Fin g.rows

/-! ## The plan buffer -/

/-- The 28-byte plan buffer (host-authored, copied to device before replay). -/
structure Plan where
  /-- Drafted tokens this step; real rows are `0 .. activeLen` (root + γ_act). -/
  activeLen : Nat
  /-- KV/RoPE position of row 0. -/
  posBase : Nat
  /-- bit0: target-only baseline (γ forced to 0); other bits reserved. -/
  flags : Nat
deriving Repr, DecidableEq

/-- A plan is valid for a geometry iff its active length fits the pinned batch. -/
def Plan.valid (p : Plan) (g : Limits) : Prop := p.activeLen ≤ g.gammaMax

/-- A row is real iff it carries a token whose verification output is used. -/
def Plan.isReal (p : Plan) {g : Limits} (r : Row g) : Prop := r.val ≤ p.activeLen

/-- Real rows run on the slot sequence; padding rows on the scratch sequence. -/
def Plan.rowSeq (p : Plan) (g : Limits) (r : Row g) : Nat :=
  if r.val ≤ p.activeLen then g.slotSeq else g.scratchSeq

/-- RoPE/KV position of a row (padding rows get valid positions; outputs discarded). -/
def Plan.rowPos (p : Plan) (g : Limits) (r : Row g) : Nat := p.posBase + r.val

/--
Attention reachability. A real row attends columns `c ≤ r` that are real
(strictly causal within the real prefix). A padding row's attention is
unconstrained: it runs on the scratch sequence and its outputs are discarded.
-/
def Plan.attends (p : Plan) (g : Limits) (r c : Row g) : Prop :=
  if r.val ≤ p.activeLen then (c.val ≤ r.val ∧ c.val ≤ p.activeLen) else True

/-- The sampler scores exactly rows `r` with `r.val < activeLen + 1`. -/
def Plan.readable (p : Plan) (g : Limits) (r : Row g) : Prop := r.val < p.activeLen + 1

/-- Clamp a plan into the valid range for a geometry (tuner-side guard). -/
def sanitize (g : Limits) (p : Plan) : Plan :=
  { p with activeLen := min p.activeLen g.gammaMax }

/-! ## Snake-case aliases and unindexed Nat variants -/

def is_real (p : Plan) (r : Nat) : Prop := r ≤ p.activeLen
def row_seq (p : Plan) (g : Limits) (r : Nat) : Nat :=
  if r ≤ p.activeLen then g.slotSeq else g.scratchSeq
def row_pos (p : Plan) (r : Nat) : Nat := p.posBase + r
def attends (p : Plan) (r c : Nat) : Prop :=
  if r ≤ p.activeLen then (c ≤ r ∧ c ≤ p.activeLen) else True
def readable (p : Plan) (r : Nat) : Prop := r < p.activeLen + 1

/-! ## Theorems (the conformance contract) -/

/-- **Theorem 1 (mask closure).** If a real row attends a column, that column
is real: padding can never flow into real rows' verification outputs. -/
theorem mask_closure {g : Limits} (p : Plan) {r c : Row g}
    (hreal : p.isReal r) (hatt : p.attends g r c) : p.isReal c := by
  unfold Plan.isReal at hreal ⊢
  unfold Plan.attends at hatt
  rw [if_pos hreal] at hatt
  exact hatt.2

theorem mask_closure_nat (p : Plan) (r c : Nat)
    (hreal : is_real p r) (hatt : attends p r c) : is_real p c := by
  unfold is_real at hreal ⊢
  unfold attends at hatt
  rw [if_pos hreal] at hatt
  exact hatt.2

/-- **Theorem 2 (sampler window).** Every scored row is a real row. -/
theorem readable_rows_are_real {g : Limits} (p : Plan) {r : Row g}
    (h : p.readable g r) : p.isReal r := by
  unfold Plan.readable at h
  unfold Plan.isReal
  omega

theorem readable_rows_are_real_nat (p : Plan) (r : Nat)
    (h : readable p r) : is_real p r := by
  unfold readable at h
  unfold is_real
  omega

/-- **Theorem 3 (KV cell disjointness).** Distinct rows never share a KV cell.
Holds unconditionally because `rowPos r = posBase + r` is strictly monotonic:
`r ≠ c` guarantees `rowPos r ≠ rowPos c`, regardless of sequence configuration. -/
theorem kv_cells_disjoint {g : Limits} (p : Plan)
    {r c : Row g} (_hseq : p.rowSeq g r = p.rowSeq g c)
    (hpos : p.rowPos g r = p.rowPos g c) : r = c := by
  unfold Plan.rowPos at hpos
  have hv : r.val = c.val := by omega
  exact Fin.eq_of_val_eq hv

theorem kv_cells_disjoint_contra {g : Limits} (p : Plan) {r c : Row g} (hneq : r ≠ c) :
    ¬ (p.rowSeq g r = p.rowSeq g c ∧ p.rowPos g r = p.rowPos g c) := by
  intro ⟨_, hpos⟩
  unfold Plan.rowPos at hpos
  apply hneq
  have hv : r.val = c.val := by omega
  exact Fin.eq_of_val_eq hv

theorem kv_cells_disjoint_nat (p : Plan) (g : Limits) (r c : Nat) (hneq : r ≠ c) :
    ¬ (row_seq p g r = row_seq p g c ∧ row_pos p r = row_pos p c) := by
  intro ⟨_, hpos⟩
  unfold row_pos at hpos
  omega

theorem kv_cells_disjoint_conditioned {g : Limits} (_hseq : g.slotSeq ≠ g.scratchSeq) (p : Plan)
    {r c : Row g} (hseq' : p.rowSeq g r = p.rowSeq g c)
    (hpos : p.rowPos g r = p.rowPos g c) : r = c :=
  kv_cells_disjoint p hseq' hpos

/-- **Theorem 4 (sanitize preserves validity).** Clamping lands the plan
inside the geometry. -/
theorem sanitize_within_limits (g : Limits) (p : Plan) : (sanitize g p).valid g := by
  unfold sanitize Plan.valid
  exact Nat.min_le_right _ _

/-- Bonus (pinning invariant): a valid plan's real-row count fits the batch. -/
theorem real_rows_fit {g : Limits} {p : Plan} (hv : p.valid g) :
    p.activeLen + 1 ≤ g.rows := by
  unfold Plan.valid at hv
  unfold Limits.rows
  omega

end PolyGraph
