# Plan v2: beta–delta solver repair (drift cap, metric, bounds, hybr)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

v2 (2026-06-12) supersedes v1 of this file. Changes of substance: interim
Phase-0 data from three live baseline runs re-ranked the diagnosis (the
±0.02/segment grid drift cap is the primary defect; the f-metric pole is not
yet operative in any measured segment); the experiment gained a control arm
and a cap-isolating arm; the Phase-0 gate was re-specified; orthogonal safety
fixes (R1 bracket, convergence-flag persistence) were absorbed from an
external draft plan whose code anchors were independently verified against
the tree. Work branch: `bugfix/beta-delta-solver`.

## Evidence

1. **Committed sample run** (`outputs/mockOutput/mockFullrun/`,
   `4e3_sfe001_n5e2_PL0`): 0 of 49 implicit-phase segments converged below
   the 1e-4 threshold; accepted residuals 2.4e-2…3.25; β moved by exactly
   ±0.02 (grid edge) nearly every segment; δ clamped at `DELTA_MAX = 0` for
   ~10 segments; by late phase the two Edot branches disagreed **in sign**
   while the integrator advanced Eb with the β-side value.
2. **Interim Phase-0 baselines** (fresh runs, current code, in progress as of
   this writing — final numbers in the Phase-0 results doc):
   | config | implicit segments | converged < 1e-4 | β at ±0.02 edge | δ at bound |
   |---|---|---|---|---|
   | mock replica (4e3, sfe 0.0085) | 21 | 0% | 100% | 12/21 |
   | `simple_cluster` (1e5, sfe 0.3) | 22 | 32% (rising) | 38% | 0 |
   | homogeneous cloud (1e6, n=1e3) | 80 | 95% | 2.5% | 0 |
   Severity is a config gradient, not mock-only and not universal.
3. **Per-segment mock trace** (the decisive observation): β marches
   0.76 → 0.38 at exactly 0.02/segment for 19 consecutive segments — a
   rate-limited chase after a bad phase-handoff guess. δ rides its 0.0 bound
   for 12 segments **then comes off it unaided** (−0.035) while the energy
   residual collapses from −0.28·Lmech toward 0. The root is *inside* the
   box; it simply outruns the cap (~25 β-units/Myr root drift vs ~20/Myr cap
   at the dt floor).
4. **Offline metric check**: recomputing the proposed g-metric (below) from
   the saved residual components of all three runs gives convergence rates
   *identical* to the legacy f-metric; |Edot_from_beta| sits at
   0.18–0.34 × Lmech in every segment (no pole encountered, no sign change
   yet — E_b still rising in all three runs at time of writing). The pole
   remains a *late-phase* hazard: the committed sample's late sign
   disagreement is consistent with Ėb crossing zero near the E_b peak.

## Diagnosis (ranked by evidence)

1. **Primary — the drift cap.** `_solve_grid` searches only
   ±`GRID_EPSILON` = 0.02 around the previous segment's accepted point
   (`get_betadelta.py:792-795`), so after a phase-handoff mismatch the solver
   needs O(20) segments to walk to the root, silently integrating
   10–30%-of-Lmech energy-imbalance the whole way. No convergence flag is
   persisted; the only trace is per-segment DEBUG logging.
2. **Secondary — hard bounds** (β∈[0,1], δ∈[−1,0], lines 41–44). In the data
   so far they bind only transiently, as the corner the capped chase gets
   clipped against. Whether genuine roots ever lie outside the box (δ>0 at
   WR/SN luminosity jumps) is open — the Phase-2 root-existence scan decides;
   widening bounds is a physics decision, not a solver default.
3. **Hygiene (real defects, not yet operative in measured segments):**
   - f-metric pole: `Edot_residual = (E1−E2)/E1` (line 436) diverges as
     E1→0, and the |E1|≤1e-300 fallback returns *raw* E2 (line 438), mixing
     normalized and unnormalized quantities.
   - Structure-solve failure returns a (100,100) plateau (line 397) that is
     fed to L-BFGS-B as a 1e10 cliff (line 874).
   - `compute_R1_Pb` brentq bracket `[1e-3·R2, R2]` (line 316) with a
     fabricated `R1 = 0.01·R2` fallback on failure (line 324); the same
     bracket is uncaught (crash class) at `energy_phase_ODEs.py:224,363` and
     `run_energy_phase.py:96,327`.
   - P_b docstrings in `get_betadelta.py` say "[cgs]"; `bubble_E2P` returns
     code units (`get_bubbleParams.py:229`).

## Residual metrics (used everywhere below)

- legacy **f** = ((E1−E2)/E1, (T−T0)/T0), total = sum of squares, threshold
  1e-4 — production today.
- new **g** = ((E1−E2)/Lmech_total, (T−T0)/T0) — root-equivalent wherever
  E1≠0, well-defined at the pole, per-segment-constant denominator.
  Recomputable offline from saved `residual_Edot{1,2}_guess`,
  `residual_T{1,2}_guess`, `bubble_Lgain`.
- **Every result below reports both.** Conclusions from one metric only are
  invalid.

## Phase 0 — Baseline measurement (running; no trinity/ changes)

Finish the three baseline runs; final harvest per segment/config:
both metrics at the accepted point; convergence under f and (descriptively)
under g; **drift-cap riding (±GRID_EPSILON edge) counted separately from
hard-bound pinning (BETA_/DELTA_ MIN/MAX)** — different defects; Edot branch
sign agreement; E_b peak location vs phase boundary and whether |E1| shrinks
toward it (pole check); cost baseline: evaluations/segment, per-evaluation
wall time, warm-start short-circuit rate (expected ~0% on the mock — nothing
converges, so every segment pays the full grid).

**Gate G0 (corrected from the draft's "≥90% non-convergence on a worked
example"):** re-scope to Phase 1 + hygiene only if **no** config shows
material non-convergence (>15% of segments) **and** no cap/bound saturation.
The interim table already passes this gate; the draft's version would have
cancelled the program while keeping the one change the data shows to be a
no-op. Record in the results doc (`analysis/`, with staleness banner): on
affected configs, production implicit-phase E_b(t) integrates clamped,
lagged β — a Paper-I-relevant caveat independent of any fix.

## Phase 1 — Orthogonal safety fixes (justified under every outcome)

1.1 **R1 bracket.** Widen `[1e-3·R2, R2]` → `[0, R2]` (for Lmech>0 the
get_r1 equation is >0 at 0 and −R2<0 at R2 — always brackets); guard
`Lmech_total <= 0 → R1 = 0`; delete the fabricated `0.01·R2` fallback; on
genuine failure log state and raise. Implemented as one shared helper
(`get_bubbleParams.solve_R1`) replacing six call sites: `get_betadelta.py`
(compute_R1_Pb), `energy_phase_ODEs.py` ×2, `run_energy_phase.py` ×2 (the
init site used `1e-4·R2`, not 1e-3), and `bubble_luminosity.py`
(get_bubbleproperties_pure — i.e. **every residual evaluation**; a bracket
failure there surfaced as the (100,100) plateau). The draft plan listed only
four of these. `run_transition_phase.py` (505, 747, 832) is covered via the
shared `compute_R1_Pb`; SB99-driven `Lmech_total` is strictly positive at
all tabulated ages, so the `Lmech <= 0` guard is unreachable there in
practice (defensive only).

1.2 **Persistence + dt mitigation.** Persist `betadelta_converged` and
`betadelta_total_residual` (register in `registry.py` + the `dictionary.py`
save list, mirroring `residual_deltaT` end-to-end). Unconverged-streak
counter: on unconverged, shrink the upcoming `dt_segment` by
`ADAPTIVE_FACTOR` down to `DT_SEGMENT_MIN`; WARNING at 3 consecutive.
**Precedence rule the draft lacked:** while the streak counter > 0, suppress
the existing adaptive block's *growth* branch
(`run_energy_implicit_phase.py:877-882`) — a lagged, clamped solver output
makes parameter changes look small, which is exactly the signal that block
uses to grow dt. Without suppression the two writers fight and the
mitigation is undone each segment.

1.3 **Docstrings.** P_b "[cgs]" → code units in `get_betadelta.py`. No logic.

Tests: R1 small-root case (old bracket raises, new converges, matches the
analytic limit R1 = √(K·R2³) to <1%); Lmech=0 → 0 without calling brentq;
mid-range case matches the old bracket **to brentq tolerance** (different
brackets change the iteration path; "machine precision" overclaims). Runner
test with a monkeypatched solver: flag persisted, counter increments/resets,
dt shrinks, growth suppressed while streak active, warning at 3.

Drift budget D1 (branch vs main, three configs): R2/v2/Eb on a common time
grid, max relative deviation < 1e-5 except in segments downstream of a
*logged* old-fallback event (listed individually); Weaver adiabatic
validation unchanged to plotting precision; wall time +5%. Out-of-budget
unattributed drift → STOP, bisect.

## Phase 2 — Four-arm shadow experiment (zero production impact)

The solver is pure (`BubbleParamsView`, no params mutation), so prototype
arms run alongside production in the same process; production snapshot files
must hash-identical to Phase 1's (drift budget D2 = zero).

2.1 **Noise floor first.** Transect probe on representative segments:
g along β- and δ-transects at spacings 1e-5…1e-2; the measured jitter scale
(not the assumed 1e-4) sets hybr's finite-difference `eps` and the honest
tolerance for arms B–D. Floor above ~1e-3 → tighten the dMdt inner tolerance
first (2.1b, own drift check) before any solver comparison.

2.2 **Root-existence scan.** ~10 stratified segments per config (early /
mid / pre-transition), coarse wide grid β∈[−1,2], δ∈[−1,0.5]: does each
residual component change sign; does a common zero plausibly exist; is it
inside the legacy box? **Pivot clause:** if for most segments no root exists
even in the wide box, the A12+structure closure is inconsistent with the
frozen state — STOP the solver program and report (model finding, Paper-I
caveats material), not a code bug.

2.3 **Arms** (per segment: same warm start, dMdt warm-start threading,
structure failure → exception abort — the (100,100) plateau is never fed to
any solver):

- **A — control:** the production path *exactly* (f metric, 1e-4 threshold,
  ±0.02 cap, hard bounds, including the L-BFGS-B fallback gated at
  residual > 5.0). Must reproduce production's accepted (β, δ) — validates
  the harness, nothing else. (The draft gave all arms a harmonized
  tolerance, which contradicts A's control role; harmonization applies to
  B–D only.)
- **B — metric:** grid as A, but g-ranking and g-threshold. Isolates the
  metric. Expected ≈ no-op (the offline check above); kept because ranking
  can differ point-to-point even when accepted-point statistics don't, and
  it is cheap falsification.
- **C — cap:** grid + g, hard bounds kept, but the ±0.02 window *iterates*:
  re-center and rescan from each new best until the optimum is interior or
  10 iterations. Isolates the drift cap — the primary suspect. (The draft
  had no cap-isolating arm; its wide-bounds arm is dropped — bound pinning
  was transient in the data, and the bounds question belongs to 2.2 + D's
  out-of-box log.)
- **D — hybr:** `scipy.optimize.root(method='hybr')` on g, unconstrained,
  `eps` from 2.1, `xtol=1e-8`, `factor=0.1`, `maxfev=30`; out-of-box roots
  accepted and logged against a warn-window (in shadow, never rejected back
  to a bounded fallback); structure failure → `_HybrAbort` → arm reports
  failure for that segment.

All arms log per segment to jsonl: solution, f and g at solution, evaluation
count (warm-start short-circuit = 1), exceptions, wall time. Cost cap: arms
multiply a run's cost ~4×; if a baseline exceeds ~30 min, run B–D on a
stratified 50% subset and record the subsetting.

2.4 **Gate G2 (pre-registered).** On segments where 2.2 confirms a root in
or near the box: convergence ≥80% per config; median evaluations ≤15 for
C/D; all failures cleanly caught; solution smoothness across segments within
the noise floor. Report each arm's short-circuit rate and mean
evaluations/segment against the Phase-0 baseline — an arm that fixes
convergence should *restore* the 1-evaluation short-circuit in settled
stretches and land at or below baseline cost. **Promotion rule: the simplest
passing arm promotes.** D over C only with a ≥15-point convergence margin or
≥3× fewer evaluations at equal convergence.

## Phase 3 — Promotion behind a switch (default unchanged)

Implement the winner inside `get_betadelta.py`: residual-components refactor
+ metric helpers (f stays in outputs for continuity; g drives acceptance);
param key `betadelta_solver` defaulting to `legacy` (byte-identical to
Phase 1, hash-tested on all three configs); winner selectable by name;
unconverged-segment summary WARNING at phase end ships regardless of winner.
Winner-mode unit tests inherited from Phase 2, including the pole regression
test (E1 crosses zero at a synthetic root: f diverges, g converges); a
`stress`-marked integration run on the worst Phase-0 config.

## Phase 4 — Validation and default flip

Winner vs legacy, three configs + one config with a strong WR/SN luminosity
jump:

- Energy-budget closure per phase < 1% of max(E_b) (saved snapshot
  quantities, left-rectangle rule).
- Edot branches agree in sign at ≥95% of accepted roots.
- E_b peak location vs phase boundary: report both modes. The winner moving
  the peak *inside* the phase is expected physics (un-clamped cooling);
  attribute by overlaying Phase-0's rail-riding segments.
- Handoff continuity (E_b, R2, v2 jumps at energy→implicit and
  implicit→transition) no worse than legacy.
- Macro deltas (transition time, terminal momentum, R2 at fixed times):
  every change >5% traceable to formerly-unconverged or formerly-pinned
  segments.
- Weaver adiabatic validation: unchanged to plotting precision.
- Cost: hard gate wall time within +20% of legacy; expected at-or-below via
  the restored short-circuit — if slower despite converging, publish the
  evaluation-count breakdown before flipping.

Pass → flip the default, keep `legacy` selectable one release, write the
final report (Phase-0 table, arm comparison, attribution, "implications for
published tracks" note). Fail on attribution → default stays `legacy`,
findings documented, STOP. End state after one quiet release: delete the
legacy path and the param key — **exactly one solver in the tree**; tags and
history are the archive.

## Decisions that belong to the maintainer, not the code

1. Is the low-mass corner (~4e3 M☉, sfe ~0.01) supported parameter space for
   Paper I? Decides how prominent the "implicit-phase tracks integrate
   lagged β" caveat must be.
2. If 2.2 finds genuine δ>0 (or β outside [0,1]) epochs: widening the hard
   bounds is a physics call.
3. Confirm or adjust the G2 promotion margins (≥80% convergence, 15-point /
   3× margins) before Phase 2 runs.

## Risks

| Risk | Mitigation |
|---|---|
| No in-box root at some epochs | 2.2 detects; bounds widening separate, evidence-backed decision; pivot clause if no root anywhere |
| Residual too noisy for finite differences | 2.1 measures floor, sets eps; worst case C (iterated grid) needs no derivatives |
| Iterated grid (C) oscillates on a noisy landscape | iteration cap 10; oscillation logged → counts as non-converged |
| hybr wanders into ODE-failing regions | abort-and-fallback contract; factor=0.1 keeps steps local |
| f-pole becomes operative near E_b peak | g everywhere in arms; pole regression test pins it |
| dt mitigation fights existing adaptive block | growth-suppression precedence rule (1.2) + runner test |
| Results change vs old runs | intended — old runs carried O(0.1–1)·Lmech imbalance; param switch keeps legacy one release; Phase-4 attribution |
| Shadow arms perturb production | pure functions + D2 hash check (byte-identical snapshots) |
