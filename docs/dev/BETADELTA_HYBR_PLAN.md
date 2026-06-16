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
5. **Offline predictor / consistency-relation test**
   (`scratch/phase0/predictor_test.py`): an analytic warm-start predictor
   (β from the A12 inverse `Ebdot_to_cool_beta` on the balance estimate with
   the previous segment's L_loss; δ = (2/7)(2α − β − 1)) was tested against
   the accepted roots. On converged segments (the 1e6 run, n=76) it loses
   badly to the existing previous-root warm start: median |Δβ| 0.036 and
   |Δδ| 0.067 versus ~0 for the warm start. The consistency relation misses
   solved δ by 0.05–0.14 *on converged segments* — the implicit phase is
   materially non-adiabatic from its start, so "δ slaved to the relation"
   (a proposed 1-D tiered solver) is empirically unsafe and is NOT adopted.
   Open: the predictor's claimed advantage at SB99 luminosity jumps is
   untested (no jump inside the data yet); revisit only with jump-segment
   evidence. Solved α stays 0.38–0.59 in all three (flat-profile) baselines,
   so adiabatic β = 3α − 1 < 0.76 and the [0,1] β-bounds are untested where
   they would bind: self-similar theory puts adiabatic β at 3·(3/(5+α_ρ))−1
   (= 2 for an α_ρ = −2 power-law cloud), **outside BETA_MAX = 1 with no
   cooling involved** — a steep-profile baseline is required to test this.
6. **Steep-profile baseline complete** (`base_cloudPL`: 1e6 M☉, sfe 0.01,
   α_ρ = −2 outside a 1 pc / 1e5 cm⁻³ core; 56 implicit segments,
   t ∈ [0.003, 0.225] Myr, R2 0.27→2.76 pc): **0% convergence**, 73%
   rail-riding, β pinned at BETA_MAX = 1 for 10 consecutive segments right
   after the shell exits the core (α̃ 0.55→0.81; adiabatic 3α̃−1 crosses 1
   and reaches 1.42), then an unwind to β = 0 with δ → −0.96 (touching
   DELTA_MIN once) and residuals up to 8.9; 8 wrong-signed-Ėb segments;
   0% convergence outside the core. Confirms the bounds-exclusion
   prediction empirically: for steep profiles the box, not the cap, is the
   binding defect.

## Diagnosis (ranked by evidence)

1. **Primary — the drift cap.** `_solve_grid` searches only
   ±`GRID_EPSILON` = 0.02 around the previous segment's accepted point
   (`get_betadelta.py:792-795`), so after a phase-handoff mismatch the solver
   needs O(20) segments to walk to the root, silently integrating
   10–30%-of-Lmech energy-imbalance the whole way. No convergence flag is
   persisted; the only trace is per-segment DEBUG logging.
2. **Co-primary for steep profiles — hard bounds** (β∈[0,1], δ∈[−1,0],
   lines 41–44). On flat-profile configs they bind only transiently, as the
   corner the capped chase gets clipped against. But the steep-profile
   baseline (Evidence 6) shows β pinned at BETA_MAX = 1 for 10 consecutive
   segments with the adiabatic root 3α̃−1 reaching 1.42 — the box genuinely
   excludes the root for α_ρ = −2 clouds, with 0% convergence outside the
   core and a catastrophic unwind to β = 0, δ = −0.96 at phase end.
   Widening bounds remains a physics decision, now with direct evidence.
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

**Coverage protocol (applies to every measurement run in Phases 0/2/4):**
a run counts for solver assessment only if its implicit phase terminates
naturally on cooling balance AND the run continues through the end of the
transition phase. `stop_t` is a safety net set ≥ 2× the extrapolated
cooling-balance time, never the terminator: the defects concentrate in the
late implicit phase (simple_cluster: 100% converged mid-phase, 33% in the
final third), and the transition phase consumes the *last implicit-phase
accepted (β, δ)* without re-solving (`run_transition_phase.py` energy
balance; `bubble_luminosity.py` structure ODE) — so a censored implicit
phase both hides the worst regime and hands contaminated values to
transition. `base_cloud1e6` was right-censored at stop_t = 1.0 Myr
(balance ratio 0.41 and falling slowly; linear/exponential extrapolations
reach the 0.05 threshold at ~1.9/~4.2 Myr, with the WR ramp delaying it
further): an extended rerun at stop_t = 6 Myr is in progress, which also
brackets the WR/SN luminosity jumps the later phases need.

Finish the three baseline runs, **plus a fourth steep-profile config**
(densPL_alpha = −2, mid-mass) once a worker frees up — the only baseline
that can test whether the hard β-bound excludes the self-similar root
(Evidence 5); without it the bounds question rests on theory alone.
Final harvest per segment/config:
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
no-op. Record in the results doc (`docs/dev/`, with staleness banner): on
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
Expectation to test: the T-residual is measured at `bubble_xi_Tb = 0.98` of
the bubble *thickness* (`bubble_r_Tb = R1 + ξ(R2−R1)`,
`bubble_luminosity.py`), where the conductive edge gives
d ln T/dξ = −(2/5)/(1−ξ) ≈ −20 — the δ-direction floor is plausibly
edge-amplified by ~20× relative to measuring at ξ = 0.9. If the transect
confirms it, moving ξ_Tb inward is a contingent option **out of this
program's scope** (it re-anchors the meaning of the T0 state variable).
Pre-existing doc bug, flagged not fixed: the registry info strings for
`bubble_xi_Tb`/`bubble_r_Tb` say "xi = r/R2" / "xi_Tb * R2", but the code
uses the thickness fraction.

2.2 **Root-existence scan.** ~10 stratified segments per config (early /
mid / pre-transition), coarse wide grid β∈[−1,2], δ∈[−1,0.5]: does each
residual component change sign; does a common zero plausibly exist; is it
inside the legacy box? **Pivot clause:** if for most segments no root exists
even in the wide box, the A12+structure closure is inconsistent with the
frozen state — STOP the solver program and report (model finding, Paper-I
caveats material), not a code bug.

2.3 **Arms** (per segment: same warm start, dMdt warm-start threading,
structure failure → exception abort — the (100,100) plateau is never fed to
any solver). **Abort contract includes invalid dMdt**: the inner fsolve at
`bubble_luminosity.py` (~line 460) checks neither `ier` nor the sign of its
result, so at the exotic (β, δ) the wide arms explore, a silently
unconverged or negative dMdt can produce garbage residuals *without*
raising. The arm harness must treat dMdt non-finite, ≤ 0, or fsolve
ier ≠ 1 as evaluation failure → abort, same as a structure exception.
(Data note: across all seven Phase-0/D1 runs, 865 accepted segments, the
accepted dMdt was never negative or non-finite — the hazard is at trial
evaluations, which snapshots don't record. The production-side guard —
check ier + positivity, raise instead of fabricating, mirroring the R1
fix — ships with Phase 3.)

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
- **C — cap + bounds:** grid + g with the ±0.02 window *iterating*
  (re-center and rescan from each new best until the optimum is interior or
  10 iterations) AND wide hard rails (β ∈ [−2, 5], δ ∈ [−2, 1]; the old box
  demoted to a logged warn-window). Originally C was cap-only with bounds
  kept, on the flat-profile evidence that pinning was transient — the
  completed steep-profile baseline (Evidence 6) overturned that: β rides
  BETA_MAX = 1 for 10 consecutive segments with the self-similar root
  genuinely outside [0,1], so a grid freed of the cap but not the bounds
  would still fail there. C now isolates "grid freed of both artificial
  constraints" vs D's "root-finder"; C's logs record how often the iterated
  window walks outside the old box (the cap-vs-bounds attribution).
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

2.5 **Results — 2026-06-13** (full detail + stats tables + figures in
`docs/dev/BETADELTA_PHASE2_ARMS.md`; regenerate with
`python scratch/phase2/analyze_arms.py`. **Re-verify these numbers against the
harness `scratch/phase2/arms.py` and the jsonl before acting — this section
drifts like the rest of the doc.**) Two configs ran to completion:
`arms_mock4e3` (0% baseline) and `arms_simple1e5` (~50% baseline).

- **D (hybr) promotes.** Convergence under g: D 78% (mock) / 80% (simple1e5)
  vs A,B 0%/50%, C 0%/60%. On simple1e5 D is also cheapest (median 10 evals,
  under the ≤15 gate); on the mock D costs 29 (over gate) and does not restore
  the short-circuit (root drifts too fast per segment). D clears the D-over-C
  bar comfortably (78-pt margin on mock; 20 pts AND ~3.7× fewer evals on
  simple1e5). B and C do not pass.
- **The hard bounds are the binding defect, confirmed.** D's converged roots
  run β ≈ −0.14…2.60, δ ≈ −1.51…−0.27 (mock: 19/21 outside the legacy box),
  while production sits clamped on the box's δ≈0 edge chasing via the ±0.02
  cap. C proves the cap alone isn't it: freed + widened, an iterated ±0.02 grid
  still gets 0% on the mock at median 121 evals (full 240 s budget every
  segment) — it cannot traverse to a root a unit outside the box; only a
  root-finder reaches it. **B (metric) is a confirmed no-op for convergence.**
- **No §2.2 pivot.** A root exists almost everywhere D evaluates — it just
  lives outside the box. simple1e5's root is in-box until late phase, then
  escapes to β≈3 (self-similar β leaving [0,1], as predicted).
- **Physical acceptance gate already works.** The harness aborts dMdt ≤ 0 /
  non-finite (`arms.py:98`); on simple1e5 four of D's six aborts were exactly
  negative-dMdt trial points (−83…−1022). So every root D *accepts* satisfies
  dMdt > 0 by construction — this is the gate that replaces the artificial box
  once bounds are widened (maintainer Decision #2, now greenlit; see below).
- **Open before Phase 3:** triage D's ~20% aborts against the §2.2 probes
  (no-root segments vs fragile structure solves); decide whether the mock's
  29-eval cost needs a predictor warm start (§0 Evidence 5, shelved).

## Phase 3 — Promotion behind a switch (default unchanged)

Implement the winner inside `get_betadelta.py`: metric helpers (f stays in
outputs for continuity; g drives acceptance); param key `betadelta_solver`
defaulting to `legacy` (byte-identical to Phase 1); winner selectable by name;
unconverged-segment summary WARNING at phase end ships regardless of winner.
Winner-mode unit tests, including the pole regression test (E1 crosses zero at
a synthetic root: f diverges, g converges); a `stress`-marked integration run.

**Progress — 2026-06-13** (work branch `bugfix/beta-delta-solver-pt2`):
- *Commit 1* — `betadelta_solver` param (`legacy` default, validator,
  `default.param` regenerated). `solve_betadelta_pure` is now a dispatcher
  whose `legacy` path is the verbatim former body (`_solve_betadelta_legacy`),
  byte-identical. Tests: `test_betadelta_solver_switch.py`.
- *Commit 2* — `_solve_betadelta_hybr`: the g residual (Lmech denominator),
  `scipy.optimize.root(hybr)` (xtol 1e-8, factor 0.1, maxfev 30, eps 3e-4),
  the `dMdt>0`/valid-structure gate raising `_NoPhysicalRoot`, returning a
  `no_physical_root`-flagged `BetaDeltaResult` when the gate rejects every
  point reached. Tests: `test_betadelta_hybr.py` (convergence incl. roots
  outside the legacy box, the three gate paths, the f-pole case).

**Key finding that revises the no-root design (re-verify before trusting):**
A self-consistent hybr-*driven* run overturns the Phase-2 shadow reading. On
the mock (flat, 4e3) `betadelta_solver=hybr` drove the *entire* implicit phase
to t=0.3: **66/66 converged, dMdt always positive (~4–6), β never above ~1.03,
no-root never fired** — versus legacy **0/35** (cap-riding, "root unreachable")
on the identical config. The Phase-2 β→2.6 / no-root was a **shadow artifact**:
arm D was graded on the *legacy* (lagged, clamped) trajectory's contaminated
states; on its own self-consistent trajectory the root never has to chase, so
β stays moderate and dMdt stays positive. The negative-dMdt hazard is therefore
largely self-inflicted by the legacy lag, not intrinsic physics.

Consequence: **no-root must NOT force a transition** — it would mis-fire,
especially for steep profiles where high β is geometric (adiabatic 3α̃−1≈2),
not cooling. Phase end stays owned by the existing cooling-balance event
(`phase_events.make_cooling_balance_event`, `(Lgain−Lloss)/Lgain < ε`).

**Commit 3 (shipped `94fe38c`) — no-root as a logged safety net, not a phase
trigger:** in `run_energy_implicit_phase`, on `betadelta_result.no_physical_root`,
do NOT transition — emit a WARNING (segment, t, β/δ, held dMdt, Lgain/Lloss),
hold the last physical dMdt for that segment, count it, and report it in the
end-of-phase `betadelta_phase_summary` helper (INFO when every segment
converged and no no-root, WARNING otherwise). Phase end stays owned by the
cooling-balance event. The inner-fsolve dMdt guard once slated here was moved to
Commit 4 and then **dropped** after a line-by-line check — see Commit 4 / A.

**Commit 4 — validation (revised 2026-06-13 after a line-by-line check of the
inner-fsolve guard; re-verify before trusting).**

**A. Inner-fsolve dMdt guard — DROPPED, not shipped.** A trace of a blanket
`raise` on non-positive/non-finite dMdt after the inner fsolve
(`bubble_luminosity.py:461`) showed it does not make sense:
- (i) the energy-phase caller (`run_energy_phase.py:159`) is **unwrapped** (the
  only try is at :319, *after* it), so a raise there propagates to `main.py`'s
  top-level `except` and **aborts the run** — on a shared, solver-independent
  path that today tolerates negative dMdt silently;
- (ii) it is **redundant for hybr** — Commit 2's outer gate already rejects
  dMdt≤0; the inner guard would only change the no-root *reason* string;
- (iii) it does **not fix legacy** — legacy mis-times from the β-clamp, not from
  negative dMdt (its accepted dMdt is positive: 865/865 in Phase 0,
  steep-legacy `dMdt=594`);
- (iv) negative inner dMdt is **not observed** in any tested config.
So it is inert defense bought at an abort risk + a wider byte-identity
footprint; the protection that matters already lives in the hybr outer gate. If
defense-in-depth for legacy is ever wanted, the safe form is a scoped opt-in
(`require_positive_dMdt=False` kwarg; solver callers pass True, the energy phase
keeps False) — never a shared raise. **Consequence:** the "legacy
byte-identical hash" sub-task is **moot** — with no shared-code change the
legacy numeric path is verbatim former code (Commit 1 renamed; 2–3 are pure
additions), byte-identical by construction.

**B. `stress`-marked integration test** — mirror `test_bubble_solver_stress.py`
(`@pytest.mark.stress`, deselected by default via `pyproject.toml`'s
`-m 'not stress'`): run a hybr config end-to-end ×N, assert no crash and 100%
convergence. The real regression guard.

**C. hybr regression test** — pin a short hybr trajectory's accepted (β,δ)
against a recorded golden so future solver changes are caught (replaces the
moot legacy hash; hybr is the path that's new).

**D. (Not code) default flip** — maintainer decision to set `betadelta_solver`
default to `hybr`, keeping `legacy` selectable one release, plus the
"implications for published tracks" note (the macro-delta below).

**Validation already gathered** (self-consistent hybr runs; raw runs were
scratch under `/tmp`, re-run to regenerate):
- **2×2 matrix** (flat α=0 / steep α=−2 × hybr / legacy, 1e6 M☉, n=1e5, to
  3 Myr): no-root never fires on any hybr-driven trajectory; β out-of-box
  (flat→1.63, steep→2.82, simple_cluster→4.20), dMdt always positive, 100% conv
  vs legacy 0%.
- **Macro-delta:** legacy mis-times AND profile-blinds the transition (both
  profiles ~0.097 Myr, clamped β / contaminated Lloss); hybr gives a physical
  spread (dense flat 0.247 Myr, normal flat 2.5 Myr, steep energy-driven past
  3 Myr).
- **Cost gate:** on a config where legacy converges 0%, hybr advances ~18×
  *faster* (it short-circuits; legacy grinds full grids with shrinking dt) — the
  +20% gate is a large win, not a cost.
- **WR/SN robustness:** simple_cluster (sfe 0.3, swinging Lmech) converged 100%
  with β to 4.20.

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

## Phase 5 — Transition-criterion study (DEFERRED to after this program)

Out of scope for the solver repair; queued for *after* the hybr program lands
and the default flips. Recorded here so it is not lost (same staleness caveat
as the rest of this doc — re-verify before acting).

The implicit→momentum transition is the cooling-balance event
`(Lgain−Lloss)/Lgain < ε`, with ε = 0.05 **hardcoded** in
`phase_events.make_cooling_balance_event`. Open questions:

- **NEW (2026-06-13) — the ratio can STALL above ε, so the criterion may be
  the wrong *trigger*, not just mis-tuned.** On the steep self-consistent hybr
  run the ratio fell to ~0.32 by t=0.26 Myr then **plateaued/oscillated at
  ~0.30–0.39 for the rest of the run to 3 Myr** (`Lloss/Lgain ≈ 0.65`,
  both luminosities tracking) — a quasi-steady energy-driven state that **never
  approaches 0.05**. So ε=0.05 is not just too strict for steep profiles, it is
  *unreachable*: such a bubble would never transition on cooling balance — it
  ends on `stop_t`, and its real fate is presumably **blowout (R2 > rCloud) or
  the cluster luminosity dropping** (a WR/SN feature was visible near t≈2.2 Myr:
  β jumped to ~2.8 with an Lgain dip, and hybr converged through it).
  **The stall is steep-specific (resolved 2026-06-13):** both flat runs cross
  0.05 cleanly — dense flat (n=1e5) at 0.247 Myr, *normal-density* flat
  (n=1e3) at **2.5 Myr** (β spiking to ~4 right at the crossing — the high-β
  excursion *is* the flat-profile transition signature). Only the steep r⁻²
  halo sustains the bubble and stalls the ratio. **Implication: cooling balance
  is a fine trigger for flat profiles; steep profiles need a different one
  (blowout `R2 > rCloud`, or cluster-luminosity decline) — the criterion is
  profile-dependent, which a single hardcoded ε cannot express.**
- **NEW (2026-06-14) — the stall is feedback-SUSTAINED, and β goes negative.**
  hybr (not clamped to β∈[0,1] like legacy) shows the stall is not a passive
  plateau: episodic feedback luminosity surges reset it upward. On the steep
  4-Myr run `Lmech_total` (= `bubble_Lgain`) jumps at a wind/WR surge
  (~3.0–3.4 Myr; `Lmech_W` climbs 2.0e8→3.5e8) and again at the SN onset
  (~3.5–3.8 Myr; `Lmech_SN` jumps to >1e8). Each re-energises the bubble: `Eb`
  and `Pb` rise — so **β goes negative** (to −2.4; β = −(t/Pb)dPb/dt, β<0 ⇔ Pb
  rising) — `dMdt` rises in lockstep (~420→~2100), and the cooling ratio **jumps
  back up** (0.44→0.67), *further* from transition. So `(Lgain−Lloss)/Lgain < ε`
  can never fire while the cluster is still in its wind/SN epoch — the criterion
  must be feedback/dynamics-aware (reinforces the force-ratio / blowout
  alternatives below). Full per-segment data + the Lmech_W/SN split:
  `docs/dev/stalling-energy-phase.md`,
  `docs/dev/data/stalling_{steep_1e6_alpha-2,mock_4e3}.csv`. Legacy could never
  show this (β pinned ≥0).
- **Is the energy-ratio criterion physically sound?** It marks "E_b stops
  *growing*", not "the bubble pressure force stops *driving* the shell". The
  momentum phase deletes the `4πR²·Pb` thermal drive
  (`phase2_momentum/run_momentum_phase.py`: Eb≈0, ram pressure only), so the
  dynamically correct transition is where that dropped force becomes
  subdominant — a force/continuity statement, not an energy-accumulation one.
  ε = 0.05 is a convention, not derived.
- **What value / criterion is right, and how do you know from the outputs?**
  (1) v2 / dv2/dt continuity across the seam (the dropped force shows up as a
  kink if you transition too early); (2) dropped-force magnitude `4πR²Pb` vs
  the surviving forces (`pdot_wind+SN`, `F_rad`, `F_HII`) at the candidate
  transition — **decomposition needs care**: the implicit-phase output field
  `F_ram` *is itself* `4πR²Pb` (naming trap), and the shell is driven by
  `max(Pb, P_HII)` (`compute_forces_pure`), so dropping `Pb` only matters when
  `Pb > P_HII`; (3) macro-observable sensitivity sweep over ε
  (insensitive→robust, report the range; sensitive→pin dynamically); (4)
  energy-budget closure across the seam.
- **Principled alternative:** replace the energy ε with a dynamical
  force-ratio trigger (`4πR²Pb / surviving-forces < O(1)`) — continuity-
  preserving by construction, likely more robust than tuning ε.
- **Step 0 (cheap, honest):** lift ε out of `make_cooling_balance_event` into
  a documented param, so "different transition values" is a config knob + a
  sensitivity test instead of a code edit.

**Other physics questions parked here (not solver-repair scope):**
- `bubble_xi_Tb` = 0.98-of-*thickness* T-residual measurement point (§2.1):
  the conductive edge amplifies the δ-direction noise ~20×; moving ξ_Tb inward
  re-anchors the T0 state variable — its own study.
- Registry info-string bug for `bubble_xi_Tb` / `bubble_r_Tb` ("xi = r/R2" vs
  the thickness fraction the code uses) — flagged in §2.1; fix when convenient.

## Phase 6 — Velocity-structure ("Problem 2") investigation (6.0+6.1 DONE — CLOSED 2026-06-14)

Surfaced by the same negative-β runs (2026-06-14). Out of solver-repair scope; a
self-contained phased study. Same staleness caveat — the line refs below were
verified against current source on 2026-06-14, re-verify before acting.
**Phase 6.0 ran; Gate G6 is marginally OPEN on one bounded `dMdt` channel —
cosmetic in 5/6 configs. See the 6.0 result block below.**

**The finding (verified).** The bubble-structure ODE's velocity source term is
`(β+δ)/t` (`bubble_luminosity.py:1150`, `dvdr`). When **β+δ goes strongly
negative** (≲ −0.5) the interior velocity falls through zero — *inflow*, which
the Weaver self-similar (outflow) structure does not admit (WARPFIELD
"Problem 2"). **The acceptance gate does not guard against it:** the inner
velocity residual checks only the inner-edge BC `(v[-1])/(v[0]+1e-4)`
(`bubble_luminosity.py:1085`), `min_T < 3e4` (`:1088`), `nan` (`:1092`), and
monotonic-T (`:1096`); the hybr outer gate checks only structure-success
(`get_betadelta.py:819`) and `dMdt>0` (`:824`). Neither checks interior-v sign.
So such segments are **converged but partially unphysical** in velocity.

Measured (`sweep_steep`, 1e6 M☉ α=−2): 4 of 133 segments, all during the WR
wind surge (β+δ ∈ [−1.11, −0.49]); the negative band is the inner ~2–73 % of the
bubble thickness, `v_min ≈ −0.1…−0.6` pc/Myr vs shell `v2 ≈ 10` (a ~1–6 %
reversal). **Driven by β+δ, not β:** the mock (β to −1.04) keeps (β+δ)_min=+0.25
and has **zero** real inflow segments. Data: `docs/dev/data/stalling_*.csv`
(`v_struct_min`, `v_struct_nneg`, `beta_plus_delta` columns).

**Impact is probably negligible — confirm before treating.** The cooling
luminosity does **not** use v: the three integrals are `chi_e·n²·Λ(T)`
(`bubble_luminosity.py:612`), `dudt(n,T,φ)` (`:659`), and the intermediate
region (`:677+`) — all density/temperature only. v feeds only the coupled ODE
and one interpolated grid point `v_CIEswitch` (`:587,593,600`). In the data
`Lloss`, `dMdt`, `Eb` evolve smoothly and stay converged (~1e-14) straight
through the inflow band. So on current evidence the inflow is **cosmetic**, and
the obvious "clip v≥0" would change essentially nothing.

### Phase 6.0 — Gate: does it EVER contaminate? [DONE 2026-06-14]
Ran six instrumented hybr configs (harness `scratch/phase6/hunt.py`, classifier
`scratch/phase6/analyze_hunt.py`) probing deeper/longer β+δ surges — stronger SN
(sfe 0.01→0.30), denser core, long multi-epoch span, flat control. Per accepted
segment: convergence, `Lloss`/`dMdt`/`Eb` smoothness across the band, and inflow
extent (`v_neg_frac_thick`, `v_min`) vs β+δ. **909 segments, 100% converged.**
Full write-up + plottable data: `docs/dev/stalling-energy-phase.md`
(§ "Phase 6.0 contamination hunt") and `docs/dev/data/hunt_*.csv`.

**Gate G6 result — marginally OPEN, on one bounded channel; cosmetic in 5/6:**
- **No non-convergence anywhere** (the cleanest contamination signal — absent).
- **"Stronger surge → worse inflow" is FALSIFIED:** the deepest dip is in the
  *weakest*-feedback baseline (sfe 0.01: β+δ→−1.11, inflow 74 % of thickness);
  stronger feedback keeps β+δ shallow/positive → no/shallow inflow. The dense
  case's deep band is a short-lived explicit→implicit handoff transient.
- **Energy-budget immune:** v is absent from all three cooling integrals
  (`:612`/`:659`/`:677`), so `Lloss`/`Eb` cannot be corrupted. The only
  v-coupled output is `dMdt`.
- **The dMdt "kink" is the feedback surge, which LEADS the inflow** — h1's
  biggest dMdt jumps (+42 %, +62 %) land *before* β+δ goes negative.
  Deconfounded vs each config's surge ramp: h1 ×0.7, h2/h3 ×0.9 (clean), h4
  handoff-excluded; **only h6** keeps a bounded ×1.9 (10.9 %) dMdt step, and even
  that looks like a *lagged* SN-surge response, not a clean inflow signature.

So the inflow is real, sometimes deep, always converges, and is provably
energy-immune — its sole possible impact is a bounded, ambiguous `dMdt` step in
one config. The screen cannot certify that channel as *exactly* zero, so →
**narrow 6.1** (below). It is **not** the broad contamination the raw first-
difference heuristic suggested before deconfounding.

### Phase 6.1 — Treatments + metric [DONE 2026-06-14 — arm C run, Problem 2 CLOSED]
**Result: the inflow is empirically immaterial → no treatment needed.** Arm C
(reject-and-hold) was run on the four real-inflow configs (harness
`scratch/phase6/hunt.py --hold-inflow`, diff `scratch/phase6/compare_hold.py`):
deleting every inflow segment — a 9.6–42.8 % local kick to `dMdt` — moves the
macro outputs (R2, v2, Eb, terminal momentum) by **≤0.04 %** (h1, the smallest
bubble; the large bubbles ~0, h6 ~1e-9). Propagation is real, not a units/path
artefact (deltas are relative; the held Eb deviates 0.63 % *during* the band then
recovers), so the smallness is physical — the band is brief and `dMdt` is a small
term. Full table + reasoning: `docs/dev/stalling-energy-phase.md` (§ "Phase 6.1
— counterfactual"). Shipped the diagnostic-only `v_neg_frac_thick` snapshot field
(registry + `COOLING_PHASE_KEYS`) as the tripwire; **arm A (accept) stands.**

The treatment arms below were the menu *if* a macro effect had shown up; kept for
the record (none was promoted):
- **A — accept** (status quo): the baseline. **← stands; no effect found.**
- **B — clip v≥0** in the structure output / `v_CIEswitch`: cosmetic unless a
  consumer of v is found; cheapest.
- **C — velocity-sign reject → hold**: treat interior-v<0 as a structure
  failure (a Problem-2 gate, mirroring the dMdt gate and the `min_T` penalty at
  `bubble_luminosity.py:1088`) so the segment flags `no_physical_root` and the
  runner holds the last physical structure (the Commit-3 path). **← the arm run
  as the 6.1 counterfactual; ≤0.04 % macro impact.**
- **D — penalise-in-solver**: add a v<0 penalty in `_get_velocity_residuals`
  alongside the existing penalties — *only* sensible if a positive-v root exists
  nearby (it may not: β+δ<0 is set by the physical Lmech surge).

**Metric (pre-registered, narrowed):** vs arm A on the open config (h6, and h1
for the deepest band) — the **macro deltas** that a changed `dMdt` could move:
`R2`, `v2`, terminal momentum, transition time, and energy-budget closure across
the band. With **no** disturbance to the (already-fine) common case
(byte-identical on non-inflow segments). Promotion: simplest arm that removes any
macro delta without perturbing the common case — *and if arm A's macro deltas
are already negligible (the likely outcome), the result is "document + ship the
diagnostic-only `v_neg_frac_thick` snapshot field, no treatment".*

### Phase 6.2 — Multi-arm experiment [only if G6 opens]
Run arms A–D in parallel (pure structure call, zero production impact — like the
§2.3 shadow arms), per-segment diagnostics to jsonl, compare against the 6.1
metric. Promote the winner behind a param/flag, default = accept.

**The deeper question (not just "which treatment"):** is the inner inflow during
a violent re-pressurisation *physically real* (genuine transient) or a
quasi-steady-structure breakdown? A treatment that suppresses real physics would
be wrong, so 6.0's job is as much to *understand* as to gate.

## Decisions that belong to the maintainer, not the code

1. Is the low-mass corner (~4e3 M☉, sfe ~0.01) supported parameter space for
   Paper I? Decides how prominent the "implicit-phase tracks integrate
   lagged β" caveat must be.
2. If 2.2 or the steep-profile baseline finds genuine δ>0 (or β outside
   [0,1]) epochs: widening the hard bounds is a physics call.
   **RESOLVED 2026-06-13 — widen.** The §2.3 race found genuine roots at
   β up to ~3 and δ down to ~−1.5 (arm D, dMdt>0 by construction); the
   maintainer confirmed these are physical states the model should occupy.
   Phase 3 replaces the artificial β/δ box with physical acceptance gates —
   `dMdt > 0`, finite/valid structure — not wider arbitrary rails. (Re-verify
   the root ranges against `docs/dev/BETADELTA_PHASE2_ARMS.md` and the jsonl
   before encoding any specific bound.)
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
