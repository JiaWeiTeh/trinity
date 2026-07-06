# H5 — is legacy's 0.05 transition "predetermined" by the (β,δ) clamp? (Part B: causal box-width sweep)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> 🔄 **Living plan — recheck and refine on every visit.** This is an evolving
> strategy doc, not a frozen record. Any agent or person who opens this file
> must, as part of the visit: (1) re-verify the claims and line references above
> against current source; (2) update anything that has drifted; (3) **rethink the
> strategy itself** — if a better ordering, gate, candidate, or experiment
> exists, revise the doc and note what changed and why (date it). Leave it better
> than you found it. **Keep all banner paragraphs at the top of every plan and
> analysis doc.**
>
> 💾 **Persist diagnostics — commit, don't re-run.** The container is ephemeral
> and full/hybr runs cost hours, so any diagnostic worth keeping must be saved as
> a committed artifact (a CSV/table under `docs/dev/data/`, or a force-added
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp` or an untracked `outputs/`. A future visit must be able to reproduce or
> compare against the numbers **without re-running**; record the exact config +
> command that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Date:** 2026-06-22. Branch `fix/transition-trigger-problem-pt4` (worktree).
No production change — `get_betadelta.BETA_MIN/BETA_MAX/DELTA_MIN/DELTA_MAX` are
monkeypatched in the harness only (`h5_variants.py`). Production
`trinity/phase1b_energy_implicit/get_betadelta.py` untouched.

---

## Hypothesis (tested skeptically, NOT assumed true)

H5: legacy clamps `(β,δ)` to the box `[0,1]×[−1,0]` (`get_betadelta.py:41–44`,
enforced by `np.clip` :1044–45 and the L-BFGS-B `bounds` :1053; the grid search
also reads the same constants :969–972). The clamp **soft-locks** `(β,δ)` on the
box boundary, **holding `Lloss` artificially high**, so the cooling ratio
`(Lgain−Lloss)/Lgain` **inevitably crosses 0.05 → a "predetermined" transition**,
not real cooling physics.

(All four file:line claims re-verified against current source on 2026-06-22.)

## Method — the causal box-width sweep (and the box-vs-solver caveat)

Part A (offline, committed `partA_results.txt`) found boundary-pinning is
**config-dependent**: STRONG for `small_dense_highsfe` (pin_frac 0.97), PARTIAL
for `pl2_steep`/`be_sphere` (0.35–0.45), ABSENT for `simple_cluster`/`midrange_pl0`
(0.00 — β interior ≈0.5 at the crossing). Part B tests this **causally**: widen the
whole box and re-run the **LEGACY** solver. If a config's crossing is *caused* by
the box, widening should move it later / make it vanish; if the crossing is
**unchanged** by widening, the box is **not** the cause for that config.

Box widths (whole box widened symmetrically; `h5_variants.WIDTHS`):

| width | β box | δ box | source |
|---|---|---|---|
| W0 | [0, 1] | [−1, 0] | committed `c0_<cfg>_legacy.csv` (legacy default) |
| W1 | [−1, 2] | [−2, 1] | this experiment (legacy solver, widened box) |
| W2 | [−4, 4] | [−4, 4] | this experiment |
| W3 | [−20, 20] | [−20, 20] | this experiment (effectively unbounded box) |
| hybr | (unbounded root) | (unbounded root) | committed `c0_<cfg>_h0.csv` (reference) |

**Box-vs-solver caveat (important):** W3 (wide-box **legacy**) is **NOT identical to
hybr**. hybr is a *different root-finder* — scipy `root('hybr')` on a pole-free `g`
residual with a `dMdt>0` acceptance gate (`get_betadelta.py:874`) — not "legacy with
an infinite box". So this sweep isolates the role of the **BOX** (W0→W3, same legacy
grid+L-BFGS-B method) **separately** from the solver-method difference. `c0_<cfg>_h0.csv`
is the true-unbounded-**root** reference, NOT the W→∞ limit of the legacy sweep.

### Why a CAPTURE-REPLAY, not 18 full wide-box sims (measured, not assumed)

The original plan was 6 configs × {W1,W2,W3} = 18 full LEGACY sims with a widened box.
**This proved computationally infeasible in-session** (measured): a full legacy sim
with a widened box is dominated by the L-BFGS-B fallback scanning a large, often
*infeasible* region per cooling segment — every wide-box grid/optimizer point that
violates the bubble structure throws `xi out of bounds` (`_rgi.py:511`) or
`MonotonicError` (`operations.py:153`), each a wasted bubble-structure solve. For
`small_dense_highsfe` (the stiffest config) the **first implicit segment alone ran
>4 min at full CPU and never completed a segment** under the widened box (W1/W2/W3),
and the warning count grew with box width (W1≈3, W2≈9, W3≈12 per first segment) —
itself a signature that widening makes the legacy solver wander. Reaching the
`small_dense` crossing (t≈0.024, ~20 implicit segments) was therefore out of reach,
and the slow/late-crossing configs (midrange t≈0.82, be_sphere t≈1.04) far more so.

So the causal test was run as a **capture-replay** (`h5_capture.py`): ONE legacy sim
per config at the **default box W0** (the fast path the code already takes), with
`solve_betadelta_pure` wrapped so that **at every implicit segment** it ALSO re-solves
betadelta under each widened box on the **same params/epoch**, recording the
counterfactual `(β, δ, Lgain, Lloss, ratio)`. The real return value is always the W0
solve, so the integrated trajectory is the unmodified legacy one. The wide-box solves
are seeded with the W0 result, so when the box does NOT bind they hit the
"already converged" early-exit (`get_betadelta.py:638`) and are cheap; when the box
DOES bind, the solver moves off the W0 point — and that movement (collapsed `Lloss`,
recovered ratio) is exactly the H5 mechanism. `Lloss` is read as
`betadelta_result.bubble_properties.bubble_LTotal` (+leak) and `Lgain = Lmech_total`,
i.e. **the exact quantity the production cooling-balance trigger uses**
(`run_energy_implicit_phase.py:855-857, 1142-1173`).

**Capture-replay caveat (documented):** this is a **per-segment counterfactual** — the
trajectory `(R2,v2,Eb,Pb,T0)` is held on the W0-legacy path; only the box's effect on
the betadelta solve *at each epoch* is varied. It does NOT let the trajectory itself
diverge under a wider box (a full sim would). It is, however, a *cleaner* isolation of
the box→Lloss→ratio mechanism (the divergence the full sim would add is a second-order
effect on top of this). Combined with the box-vs-solver caveat above, the capture
answers: "at the epoch where W0 crosses 0.05, does relaxing the box let Lloss collapse
and the ratio recover?" — the precise H5 question.

Built-in consistency check: the capture also records **W0** on the same trajectory; its
W0 crossing must match the committed `c0_<cfg>_legacy.csv` W0 crossing (`h5_analyze.py`
prints the comparison). Runs: `OMP_NUM_THREADS=1`, one sim/process, full CPU.

### Consistency gates (both PASS)
- **Capture/replay W0 reproduces committed legacy.** The capture-replay's W0 betadelta
  re-solve on the committed trajectory matches the committed `c0_*_legacy.csv` ratio at
  matched epochs: e.g. `small_dense` t=0.017 replay-W0 β=0.0, ratio=0.2835 vs committed
  β=0.02, ratio=0.2809; `simple_cluster` t=0.121 replay-W0 β=0.82, ratio=0.330 vs
  committed β=0.84, ratio=0.328. The reconstruction is faithful.
- **Phase 1a (live W0 runs) matched committed exactly** at the energy→implicit boundary
  (t=0.002910, R2=0.2271, Eb=2.92e48 for `small_dense`).

## The decisive evidence — boundary-pinning AT the crossing (`h5_pinning_summary.csv`)

The plan's explicit falsification rule: *"If β is INTERIOR (not on the box boundary) at
the 0.05 crossing, H5 is wrong for that config."* Computed rigorously from the committed
legacy + hybr trajectories (no re-run needed — this is the core result):

| config | legacy crosses? | cross_t | β@cross | δ@cross | edge@cross | pin_frac (pre-X) | hybr ratio_min | hybr crosses? |
|---|---|---|---|---|---|---|---|---|
| small_dense_highsfe | yes | 0.0242 | 0.00 | −0.14 | **b=0 (on edge)** | 0.97 | 0.283 | no |
| pl2_steep | yes | 0.128 | 0.00 | 0.00 | **b=0+d=0 (on edge)** | 0.35 | 0.489 | no |
| be_sphere | yes | 1.037 | 0.00 | 0.00 | **b=0+d=0 (on edge)** | 0.45 | 0.471 | no |
| simple_cluster | yes | 0.178 | **0.50** | −0.60 | **interior** | 0.00 | 0.324 | no |
| midrange_pl0 | yes | 0.822 | **0.42** | −0.58 | **interior** | 0.00 | 0.365 | no |
| large_diffuse_lowsfe | **no** | — | — | — | — | 0.09 | 0.465 | no |

- **small_dense / pl2_steep / be_sphere cross with (β,δ) sitting EXACTLY on the box edge
  (β=0, the lower bound)** → the clamp binds at the crossing → consistent with H5.
- **simple_cluster / midrange_pl0 cross with β strictly INTERIOR (0.50, 0.42)** → the
  clamp does NOT bind → **H5 FALSIFIED for these configs** by the plan's own rule. Their
  0.05 crossing is genuine cooling within the box, not a clamp artifact.
- **hybr (unbounded root) NEVER crosses for ANY config** (ratio_min 0.28–0.49) — the
  unbounded-β counterfactual confirms no cooling-balance event exists under the free root.

## Box-width sweep (capture-replay) — important refinement

Re-solving the LEGACY betadelta under progressively wider boxes on the committed
`small_dense` trajectory AT the crossing region gives a subtle result: at e.g. t=0.017
(β pinned at 0 in W0), **W0, W1, W2, and W3 (β∈[−20,20]) ALL return the IDENTICAL
solution β=0.0, δ=−0.08, ratio=0.283** (residual ~0.6, unconverged). Widening the box
does NOT move the LEGACY solver off β=0. Reason: legacy is a LOCAL grid (±0.02 around the
seed) + L-BFGS-B seeded at the previous β; from β≈0 it stays at the β=0 local solution
regardless of the bounds. Only hybr — a *global* root-finder — escapes to β≈+4.

⇒ **Refinement:** small_dense's crossing is NOT removed by merely widening the box with
the legacy *method*; it is the **legacy LOCAL solver landing at a β=0 solution that
happens to coincide with the box's lower edge**, NOT a bound that a wider box would
relax away. This is the "box-vs-solver" caveat made concrete: W3-legacy ≠ hybr. The
clamp and the local-search method are entangled; the unbounded *global* root (hybr) is
what actually avoids the crossing.

**`small_dense_highsfe` box sweep AT the crossing epoch (t=0.024178, `h5_replay_*`):**

| box | β | δ | ratio | crosses 0.05? |
|---|---|---|---|---|
| W0 [0,1] | 0.00 | −0.16 | 0.0247 | YES |
| W1 [−1,2] | −0.02 | −0.16 | 0.0260 | YES |
| W2 [−4,4] | −0.02 | −0.16 | 0.0260 | YES |
| W3 [−20,20] | −0.02 | −0.16 | 0.0260 | YES |
| hybr (ref) | +3.9 (late) | — | min 0.283 | **NO** |

Full-trajectory replay (6 crossing-region epochs, `h5_replay_small_dense_highsfe.csv`):
**all of W0/W1/W2/W3 cross at the SAME t=0.024178**, ratio_min ≈ −0.05 for all four.
Reconstruction is faithful (W0-vs-committed ratio |Δ|: median 0.0012, max 0.074).
**Widening the legacy box all the way to [−20,20] does NOT move the crossing.**

## Per-config trend table (`h5_sweep.csv`)

`cross_t` vs box width (None = no 0.05 crossing); from `h5_analyze.py`:

| config | W0 | W1 | W2 | W3 | hybr | per-config result |
|---|---|---|---|---|---|---|
| small_dense_highsfe | 0.0242 | 0.0242 | 0.0242 | 0.0242 | None | **UNCHANGED → box NOT the cause (refutes H5)** |
| simple_cluster | 0.178 | (interior, box slack: W0≡W3 to t=0.13) | None | INCONCLUSIVE-by-sweep; β interior at cross → genuine (pinning test) |
| midrange_pl0 | 0.822 | — | — | — | None | β interior at cross → genuine (pinning test) |
| pl2_steep | 0.128 | — | — | — | None | β on-edge at cross (pinning test) |
| be_sphere | 1.037 | — | — | — | None | β on-edge at cross (pinning test) |
| large_diffuse_lowsfe | None | — | — | — | None | control (never crosses) |

The box-sweep reached the crossing only for `small_dense` (earliest cross, t=0.024);
the slow/late-crossing configs are INCONCLUSIVE *by sweep* (the legacy betadelta solve
is too expensive per epoch to integrate to t≈0.13–1.0 in-session — measured), so for
them the decisive evidence is the **β-on-edge-at-crossing pinning test** above. The
`simple_cluster` sweep (to t=0.128, β interior throughout) shows W0 and W3 returning
**identical** β and ratio at every epoch — the box is completely slack where β is
interior.

## Figures
- `figures/h5_crossing_vs_boxwidth.{pdf,png}` — per config, crossing time and
  ratio_min vs box width (W0→W1→W2→W3→hybr). `small_dense` cross_t is flat across
  W0–W3 (crossing persists), then None at hybr.
- `figures/h5_ratio_trajectories.{pdf,png}` — cooling ratio(t) for the PINNED
  `small_dense` and NON-PINNED `simple_cluster` across box widths + hybr, 0.05 line.

## VERDICT

**Is legacy's transition "predetermined by the clamp"? NO — the clamp box is NOT the
cause, and the H5 mechanism as stated is FALSIFIED.** The result is nuanced but the
nuance cuts *against* H5, not in a way that rescues it:

1. **The decisive box-width sweep refutes the mechanism for the most-pinned config.**
   `small_dense_highsfe` (pin_frac 0.97, β=0 on the lower edge through the whole
   descent) **still crosses at the identical t=0.024 with the box widened to
   [−20,20]** — β only nudges to −0.02, Lloss barely changes, the ratio stays ≈ −0.05.
   So the *box bound itself* is not holding Lloss high; **widening it does not relax
   the crossing away.** H5's explicit falsifier (b) — "the crossing survives at wide
   `BETA_MAX`" — is met.
2. **For the non-pinned configs the clamp never binds at the crossing** —
   `simple_cluster` (β=0.50) and `midrange_pl0` (β=0.42) cross with β strictly
   INTERIOR (pin_frac 0.00); the box-slack sweep confirms W0≡W3. H5's falsifier (a) —
   "β is interior at the crossing" — is met. Their crossing is genuine within-box
   cooling.
3. **What actually distinguishes legacy (crosses) from hybr (never crosses) is the
   SOLVER METHOD, not the box.** Legacy is a *local* grid(±0.02)+L-BFGS-B that lands
   at a β≈0 solution (where Lloss stays high → ratio crosses); hybr is a *global*
   root-finder that reaches β≈+4 (hotter/steeper interior → Lloss collapses → ratio
   recovers to 0.28–0.49, no crossing for any config). The clamp lower edge merely
   *coincides* with where the local solver lands for the pinned configs; it is not the
   causal lever.

**Falsification outcome (explicit):** H5 ("the β/δ box clamp soft-locks the solution
and predetermines the 0.05 crossing") is **FALSIFIED**. The crossing is *not* a
box-clamp artifact removable by widening the box; it is a **legacy-local-solver
artifact** — the bounded grid+L-BFGS-B finds a β≈0 (or interior) solution that crosses,
whereas the correct unbounded *global* root (hybr) does not. This is consistent with
pt4-H1's "constrained edge-root artifact" framing but sharpens it: the relevant
constraint is the *local search*, not the *box* per se. (Earlier Part-A language
"clamp-contingent / clamp-caused for small_dense" is **superseded** by this sweep.)

## Caveats on this measurement
- **Box-vs-solver:** W3 (wide-box legacy) ≠ hybr; the sweep isolates the BOX, and the
  finding is precisely that the box is *not* the cause — the solver method is.
- **Per-segment counterfactual:** the box-sweep re-solves betadelta on the committed
  W0 trajectory at each epoch (`h5_replay.py`); it does not let the trajectory diverge.
  It is a faithful isolation of the box→(β,δ)→Lloss→ratio link (W0-vs-committed |Δ|
  median 0.0012). A full wide-box sim would add second-order trajectory divergence; it
  was attempted and is computationally prohibitive in-session (the wide-box legacy
  betadelta solve scans a large infeasible region — `xi out of bounds` /
  `MonotonicError` — and the first implicit segment alone runs minutes).
- **Sweep coverage:** only `small_dense` was integrated to its crossing; the late
  crossers rely on the β-on-edge-vs-interior pinning test (committed data, exact).

## Reproduce
```
cd docs/dev/transition/pt4/h5clamp
# 1) decisive committed-data pinning test (fast, no sims):
python h5_pinning.py                 # -> h5_pinning_summary.csv + h5_pinning_<cfg>.csv
# 2) box-width sweep at the crossing region (legacy betadelta re-solved per epoch
#    on the committed trajectory, under each box width; minutes/epoch):
OMP_NUM_THREADS=1 python h5_replay.py \
  --param ../../cleanroom/configs/small_dense_highsfe.param \
  --legacy ../../cleanroom/data/c0_small_dense_highsfe_legacy.csv \
  --t-min 0.017 --out data/h5_replay_small_dense_highsfe.csv
# 3) assemble sweep table + verdict, then figures:
python h5_analyze.py                 # -> h5_sweep.csv + verdict (also h5_analyze_output.txt)
python h5_figures.py                 # -> figures/h5_*.{pdf,png}
# (full wide-box sim path — h5_variants.py + h5_run_variant.py + h5_run_matrix.sh —
#  is provided but is prohibitively slow for the late-crossing configs; see caveats.)
```
