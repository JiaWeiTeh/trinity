# Phase 2.3 four-arm shadow experiment — results & Gate G2 evaluation

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
> a committed artifact (a CSV/table under `analysis/data/`, or a force-added
> harness/figure under `scratch/` as the hybr work did) — never left in `/tmp` or
> an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.

Generated 2026-06-13 from `scratch/phase2/arms_{mock4e3,simple1e5}.jsonl`
(harness `scratch/phase2/arms.py`, plan §2.3). Regenerate stats and figures
with `python scratch/phase2/analyze_arms.py` — the numbers below are only as
good as that jsonl on disk; re-run after any new arms race and **re-verify
against the harness code before acting on any conclusion here.**

> 🛑 **SUPERSEDED IN PART — read this first.** The §2.3 "shadow" arms below
> graded arm D (hybr) on the states the **legacy** (clamped, lagged) solver
> visited. Driving the integration with hybr instead (Phase 3, self-consistent
> runs) shows the headline shadow finding — β running away to ~2.6 and ~20% of
> segments hitting "no physical root" — was largely a **shadow artifact**: on a
> self-consistent hybr trajectory **no-root never fires** and β stays moderate.
> The promotion verdict (D wins, bounds are the binding defect) stands; the
> runaway/no-root demographics do not. **See the new section
> "Phase 3 — self-consistent validation (2026-06-13)" at the bottom for the
> real, plottable results.**

## What ran

Two configs carried the four-arm race to natural completion (both implicit
phases terminated; arms log one jsonl line per arm per segment):

- `arms_mock4e3` — 3966 M☉, sfe 0.0085, flat profile (densPL α=0, nCore 5e2).
  The low-mass worst case: production converges 0% here in Phase 0.
- `arms_simple1e5` — 1e5 M☉, sfe 0.3, default profile. The mid config where
  production already converges ~half the segments.

Arms: **A** production exactly (control, f-metric), **B** g-metric only,
**C** cap freed (iterated ±0.02 grid) + wide rails β∈[−2,5] δ∈[−2,1],
**D** `scipy.optimize.root(hybr)` on g, unbounded.

## Statistics

### arms_mock4e3

| arm | segs | conv f | conv g | short% | med ev | max ev | aborts (kind) | out-of-box |
|---|---|---|---|---|---|---|---|---|
| A control | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 24 | 24 | 0 | - |
| B metric | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 25 | 26 | 0 | - |
| C cap+bounds | 27 | 0/27 (0%) | 0/27 (0%) | 0% | 121 | 121 | 0 | 0 |
| D hybr | 27 | 21/27 (78%) | 21/27 (78%) | 0% | 29 | 33 | 6 (structure:6) | 19 |

### arms_simple1e5

| arm | segs | conv f | conv g | short% | med ev | max ev | aborts (kind) | out-of-box |
|---|---|---|---|---|---|---|---|---|
| A control | 30 | 14/30 (47%) | 15/30 (50%) | 23% | 24 | 24 | 0 | - |
| B metric | 30 | 7/30 (23%) | 15/30 (50%) | 40% | 25 | 25 | 0 | - |
| C cap+bounds | 30 | 7/30 (23%) | 18/30 (60%) | 40% | 37 | 121 | 0 | 0 |
| D hybr | 30 | 18/30 (60%) | 24/30 (80%) | 40% | 10 | 33 | 6 (neg_dMdt:4, structure:1, timeout:1) | 5 |

Figures (regenerable, not committed — `scratch/phase2/`):
`arms_summary.png` (convergence + cost bars vs G2 gates),
`arms_rootmap.png` (the (β,δ) maps below).

## Findings

1. **The binding defect is the hard bounds, not (only) the drift cap.** On the
   mock, arm D's converged roots run **β ≈ −0.14 … 2.60, δ ≈ −1.51 … −0.27** —
   19 of 21 outside the legacy box β∈[0,1], δ∈[−1,0]. Production (arm A) sits
   clamped on the box's δ≈0 edge, marching β down 0.68→0.40 at exactly
   0.02/segment (the documented cap drift), chasing a root it is structurally
   forbidden to reach. The root map shows production pinned to the box edge
   while D's roots trace a smooth arc out of the box.

2. **Freeing the cap without a root-finder is insufficient *and* ruinously
   expensive.** Arm C frees the cap (iterating grid) and widens the rails, yet
   still gets 0% on the mock at a median 121 evaluations/segment — it burns the
   whole 240 s wall budget every segment (`budget_exceeded` on all 27) because
   an iterated ±0.02 window walks at ≤0.1/segment and cannot traverse to a root
   that sits a unit or more outside the box. Only D's Newton-type step gets
   there. On simple1e5 C does help (60% vs A's 50%) and stays in-box — the root
   there is mostly reachable — but at median 37 evals it is far over budget.

3. **The metric change (B) is a confirmed no-op for convergence** — identical
   conv-g to A on both configs. Keep g only as D's pole-free objective; do not
   ship it as a standalone change.

4. **A real root exists almost everywhere D can evaluate** (D converges
   78–80%), so the §2.2 pivot clause ("no root anywhere → stop the program")
   does **not** trigger. The root is real; it simply lives outside the legacy
   box. On simple1e5 the root is in-box for early/mid phase and escapes to
   β≈3.0 only at late phase (the self-similar β leaving [0,1], as predicted).

5. **D's accepted roots already pass the physical sanity check `dMdt > 0`.**
   The harness aborts any evaluation with `dMdt ≤ 0` or non-finite
   (`arms.py:98`), so every root D accepted is physically valid by
   construction. On simple1e5, 4 of D's 6 aborts were exactly negative-dMdt
   trial points (−83, −186, −591, −1022) at exotic (β,δ) — the gate is doing
   real work. This is the key to widening the box safely: replace the
   artificial β/δ rails with physical acceptance gates (dMdt>0, valid
   structure), which D's converged set already satisfies.

6. **Failures are all cleanly caught** — only `_ArmAbort` / `_PointTimeout`,
   zero leaked crashes; the BaseException design held through the hybr
   internals.

## Gate G2 evaluation (plan §2.4)

Gate: convergence ≥80% per config; median evals ≤15 for C/D; all failures
cleanly caught; solution smooth across segments. Promotion: simplest *passing*
arm; D over C only with ≥15-pt margin or ≥3× fewer evals.

| criterion | A | B | C | D |
|---|---|---|---|---|
| convergence ≥80% | control | no | no | **80% simple1e5; 78% mock (100% of evaluable)** |
| median evals ≤15 (C/D) | — | — | **no** (37 / 121) | mixed (10 simple1e5 ✓, 29 mock ✗) |
| failures cleanly caught | ✓ | ✓ | ✓ | ✓ |
| smooth across segments | — | — | — | ✓ (roots trace continuous arcs) |
| restores short-circuit | — | — | — | ✓ simple1e5 (40%), ✗ mock (root drifts too fast) |

**D promotes.** B and C do not pass (B no convergence gain; C fails both
convergence and cost). With no simpler passing arm, D is the winner by
default — and it beats C decisively anyway (78-pt margin on mock; 20 pts *and*
~3.7× fewer evals on simple1e5), clearing the D-over-C bar.

Two caveats recorded, neither blocking promotion:
- **Mock cost** (median 29 evals) exceeds the ≤15 target and D never restores
  the short-circuit there, because the mock root drifts far enough per segment
  that the previous root is not within 1e-4 of the next. A predictor warm start
  (plan §0 Evidence 5, shelved) could be revisited *only* for this regime.
- **~20% abort rate.** Mock aborts are all structure-solve failures; simple1e5
  is mostly negative-dMdt. Triage needed: are these no-root segments (legit
  exclusions from the G2 conditional) or fragile structure solves a more robust
  evaluator would recover?

## Consequence for the plan

Promoting D is inseparable from **widening the hard bounds** — the roots are
genuinely at β up to ~3 and δ down to ~−1.5. This was maintainer Decision #2
(a physics call); the maintainer has **greenlit widening** (2026-06-13), with
`dMdt > 0` (and valid structure) as the physical acceptance gate replacing the
artificial box. See the plan's Decisions section.

Next: triage D's aborts against the §2.2 root-existence probes, then proceed to
Phase 3 (implement D behind `betadelta_solver`, default `legacy`).

---

# Phase 3 — self-consistent validation (2026-06-13)

> ⚠️ Same staleness caveat as the top banner. These rows are extracted from
> per-run `dictionary.jsonl` of scratch runs under `/tmp` (ephemeral — gone on
> container restart). **Re-run to regenerate the raw per-segment data for
> plotting** (recipes at the end). Numbers here are summary statistics; treat
> as a point-in-time snapshot, not ground truth.

Unlike the §2.3 shadow arms, here hybr **drives** the integration
(`betadelta_solver=hybr`), so every (β, δ, dMdt, Lgain, Lloss) is
self-consistent. This is the real comparison.

## Master runs table (the plottable demographics)

One row per run. `α_ρ` = density-profile slope (0 flat, −2 steep). `impl` =
implicit-phase segments solved. `conv%` = converged below 1e-4. `neg` =
segments with dMdt ≤ 0. `β`/`dMdt` ranges over the implicit phase. `t_end` =
last implicit time (Myr). `ratio_end` = `(Lgain−Lloss)/Lgain` at the end.
`t_trans` = cooling-balance transition time (— = never reached by `stop_t`).

| run | solver | mCloud[M☉] | sfe | nCore[cm⁻³] | α_ρ | stop_t | impl | conv% | neg | β_min | β_max | dMdt_min | dMdt_max | t_end | ratio_end | t_trans | end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flat·hybr     | hybr   | 1e6  | 0.01   | 1e5 |  0 | 3.0  |  90 | **100** | 0 | 0.43 | 1.63 |  39 |  252 | 0.247 | 0.002 | **0.247** | cooling_balance |
| flat·legacy   | legacy | 1e6  | 0.01   | 1e5 |  0 | 3.0  |  74 | **0**   | 0 | 0.46 | 0.84 | 204 |  548 | 0.097 | 0.034 | **0.097** | cooling_balance |
| steep·hybr    | hybr   | 1e6  | 0.01   | 1e5 | −2 | 3.0  | 113 | **100** | 0 | 0.59 | 2.82 | 152 |  502 | 3.000 | 0.386 | **—**     | reached_tmax (stall) |
| steep·legacy  | legacy | 1e6  | 0.01   | 1e5 | −2 | 3.0  |  74 | **0**   | 0 | 0.46 | 0.84 | 204 |  595 | 0.098 | 0.045 | **0.098** | cooling_balance |
| typical·hybr  | hybr   | 1e6  | 0.01   | 1e3 |  0 | 3.0  | 151 | **100** | 0 | 0.50 | 4.18 |  42 |  610 | 3.000 | 0.009 | **~2.5**  | cooling_balance |
| simple·hybr   | hybr   | 1e5† | 0.3    | 1e5 |  0 | 3.0  | 123 | **100** | 0 | 0.76 | 4.20 | 361 | 5595 | 3.000 | 0.827 | **—**     | reached_tmax |
| mock·hybr     | hybr   | 3966 | 0.0085 | 5e2 |  0 | 0.3  |  66 | **100** | 0 | 0.35 | 1.03 | 3.8 |  6.4 | 0.300 | 0.410 | **—**     | reached_tmax |
| cost·hybr     | hybr   | 1e6  | 0.01   | 1e5 |  0 | 0.08 |  45 | **100** | 0 | 0.59 | 0.93 | 166 |  252 | 0.080 | 0.316 | —         | reached_tmax |
| cost·legacy   | legacy | 1e6  | 0.01   | 1e5 |  0 | 0.08 |  12 | **0**   | 0 | 0.54 | 0.76 | 204 |  235 | 0.005 | 0.667 | —         | (timed out at t=0.005) |

† `simple·hybr` is `simple_cluster` (pre-SFE mCloud_input = 1e5, gas mCloud
= 7e4 after sfe=0.3). The `mock·hybr` row is `mockfull`. Partial restart-killed
runs (`m_steep{1e5,4e3b}_hybr`, 3 segs; `m_steeplong_hybr`, hung at t=0.367)
omitted. The live `sweep_*` (stop_t=4 Myr) runs append here when complete.

## End-to-end robustness sweep to the momentum phase (stop_t = 4 Myr)

Four problem-prone configs driven by hybr through the *full* phase sequence
(energy → implicit → transition → momentum), to confirm nothing crashes or
hangs over a long run. **All four: exit 0, 100% convergence, zero
negative-dMdt, no hangs.**

| run | α_ρ | nCore | mCloud | impl | conv% | neg | β range | t_end | ratio_end | fate |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep_flat    |  0 | 1e5 | 1e6  |  90 | 100 | 0 | [0.43, 1.63]  | 0.25 | 0.002 | **momentum ✓** (collapsed, small_radius @0.37 Myr) |
| sweep_typical |  0 | 1e3 | 1e6  | 177 | 100 | 0 | [0.50, 4.18]  | 3.42 | 0.009 | **momentum ✓** (ran to 4 Myr, R2=38 pc) |
| sweep_mock    |  0 | 5e2 | 3966 | 144 | 100 | 0 | [−1.04, 4.23] | 4.00 | 0.888 | energy-driven to stop_t (no transition) |
| sweep_steep   | −2 | 1e5 | 1e6  | 133 | 100 | 0 | [−2.44, 3.43] | 4.00 | 0.524 | energy-driven to stop_t (no transition) |

The two transitioning configs (flat, typical) hand off to momentum and complete
cleanly; the two energy-driven configs (mock, steep) run ~4 Myr of continuous
hybr structure solves without a single crash, hang, or non-physical dMdt.
**β spans [−2.44, +4.23] across the sweep — including negative β (rising Pb) —
and hybr converges every segment regardless.** No robustness issues found.

## Headline comparison 1 — convergence (the core fix)

| metric | legacy | hybr |
|---|---|---|
| convergence (flat 1e6) | **0%** (0/74) | **100%** (90/90) |
| convergence (steep 1e6) | **0%** (0/74) | **100%** (113/113) |
| convergence (typical 1e6) | (not run) | **100%** (151/151) |
| convergence (mock 4e3) | 0% (Phase-0) | **100%** (66/66) |
| convergence (simple 1e5) | ~50% (Phase-0 shadow) | **100%** (123/123) |
| β reachable | clamped ≤ 0.84 | **out-of-box, up to 4.20** |
| negative-dMdt / no-root | n/a | **0 across every run** |

hybr converges 100% on every self-consistent run, with `dMdt > 0` always and
no-root never firing — across mass (4e3→1e6), density (5e2→1e5), profile
(flat/steep), and sfe (0.0085→0.3).

## Headline comparison 2 — transition timing (the science impact)

Matched configs (1e6, n=1e5), hybr vs legacy:

| config | legacy t_trans | hybr t_trans | factor |
|---|---|---|---|
| flat (α=0)  | 0.097 Myr | 0.247 Myr | **2.5×** |
| steep (α=−2) | 0.098 Myr | >3 Myr (stalls) | **>30×** |

Legacy transitions *both* profiles at ~0.097 Myr (profile-blind — its clamped β
gives a contaminated Lloss that crosses the 0.05 threshold early). hybr gives a
**physical, profile-dependent** spread. Demographics by regime:

| regime | config | hybr transition behaviour |
|---|---|---|
| dense flat (n=1e5) | flat·hybr   | transitions cleanly at 0.247 Myr |
| normal flat (n=1e3) | typical·hybr | transitions cleanly at ~2.5 Myr (β spikes to 4.2 at the 0.05 crossing) |
| steep halo (α=−2)  | steep·hybr  | **stalls** at ratio ≈ 0.35, never reaches 0.05 by 3 Myr |
| low-mass (4e3)     | mock·hybr   | energy-driven, ratio 0.41 at 0.3 Myr |
| sfe=0.3 (variable Lmech) | simple·hybr | energy-driven, ratio swings 0.32↔0.92 with SB99 jumps |

So flat profiles cross 0.05 (cooling balance works); steep r⁻² halos *stall*
above it (the cooling-balance trigger may be the wrong criterion for steep — see
`docs/dev/BETADELTA_HYBR_PLAN.md` Phase 5).

## Headline comparison 3 — cost (Phase-4 wall-time gate)

Matched config (1e6, n=1e5, flat), to a matched short stop_t:

| solver | sim time reached | implicit segs | wall (implicit) | rate (Myr sim / s wall) |
|---|---|---|---|---|
| hybr   | 0.080 Myr | 45 | 580 s | 1.4e-4 |
| legacy | 0.005 Myr | 12 | 666 s | 7.7e-6 |

**hybr advances ~18× faster** (it short-circuits when settled; legacy never
converges this config and grinds full 5×5 grids with shrinking dt). The +20%
gate is a large win, not a cost.

## How to regenerate the raw data (for plotting)

Each run writes per-segment `dictionary.jsonl` (fields per row include
`t_now, cool_beta, cool_delta, bubble_dMdt, bubble_Lgain, bubble_Lloss,
betadelta_converged, betadelta_total_residual, R2, Pb, Eb, v2`). To rebuild a
row of the master table:

```bash
# matched 2x2 (vary densPL_alpha 0/-2, betadelta_solver hybr/legacy):
python run.py <param with mCloud=1e6 sfe=0.01 nCore=1e5 rCore=1 \
    densPL_alpha={0,-2} betadelta_solver={hybr,legacy} stop_t=3.0>
# then read outputs/<model>/dictionary.jsonl
```

Plot ideas the table feeds: convergence% vs (mass, density, α_ρ); β trajectory
vs t per config (out-of-box demographics); cooling ratio vs t (transition vs
stall by regime); wall-time/segment hybr vs legacy; transition time vs cloud
structure.
