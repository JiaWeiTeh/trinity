# Phase 2.3 four-arm shadow experiment — results & Gate G2 evaluation

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Generated 2026-06-13 from `scratch/phase2/arms_{mock4e3,simple1e5}.jsonl`
(harness `scratch/phase2/arms.py`, plan §2.3). Regenerate stats and figures
with `python scratch/phase2/analyze_arms.py` — the numbers below are only as
good as that jsonl on disk; re-run after any new arms race and **re-verify
against the harness code before acting on any conclusion here.**

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
