# Plan: `hybr` root-finding for the beta-delta solver

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

## Problem statement (evidence)

The Phase-1b cooling-parameter solver
(`trinity/phase1b_energy_implicit/get_betadelta.py`) finds (β, δ) by
*minimizing* `Edot_residual² + T_residual²` over a 5×5 grid spanning ±0.02
around the previous segment's values, with an L-BFGS-B fallback gated at
residual > 5.0.

In the committed sample run `outputs/mockOutput/mockFullrun/` (a real
end-to-end simulation, `4e3_sfe001_n5e2_PL0`):

- **0 of 49 implicit-phase segments converged** below the 1e-4 threshold;
  accepted total residuals ranged 2.4e-2 … 3.25 (median 1.2).
- β moved by exactly ±0.02 (the grid edge) nearly every segment — the true
  minimum sits outside the search window and the solver rail-rides after it.
- The L-BFGS-B fallback **never fired** (worst residual 3.25 < gate 5.0).
- δ sat clamped at `DELTA_MAX = 0.0` for ~10 consecutive segments.
- By late phase, `Edot_from_beta` (+7.3e5) and `Edot_from_balance` (−5.4e5)
  disagreed **in sign**; the integrator advances Eb with the β-side value
  (`run_energy_implicit_phase.py` ~line 702), so Eb rose monotonically to the
  phase boundary even though the energy balance said it should be falling.
  The commented-out "kink fix" (`Ed = min(Ed_from_beta, Ed_from_balance)`,
  same file ~line 419) shows this discrepancy has been met before.

Separately, the legacy Edot residual `(E1 − E2)/E1` has a pole where
`Edot_from_beta = 0`. The consistent Ėb must cross zero *inside* the implicit
phase (the phase ends on cooling balance `(Lgain−Lloss)/Lgain < 0.05`,
`phase_general/phase_events.py`, while the PdV term stays positive during
expansion), so a solver that actually converges will visit that pole. Latent
today only because the solver never reaches the root.

## Proposal

Treat the problem as what it is — a 2-D root find F(β, δ) = 0 — and solve it
with `scipy.optimize.root(method='hybr')` (MINPACK Powell hybrid), seeded
from the previous segment's root, with the existing grid as fallback.

### Residual vector (re-normalized)

Root-find on

```
g1(β, δ) = (Edot_from_beta(β) − Edot_from_balance(β, δ)) / Lmech_total
g2(β, δ) = (T_xi(β, δ) − T0) / T0
```

instead of the legacy `f1 = (E1 − E2)/E1`. Root-equivalent to the legacy
form wherever E1 ≠ 0, well-defined at the E1 = 0 pole, and the denominators
are per-segment constants, so minimizing/comparing g cannot be gamed by
inflating the denominator. Both components are dimensionless (E-rates and
`Lmech_total` share au units; T and T0 share K).

Structure worth exploiting: `R1`, `Pb` do not depend on (β, δ) (they come
from R2, Eb — frozen during a segment), so `Edot_from_beta = A − B·β` is
*exactly linear in β* per segment (`cool_beta_to_Ebdot_pure`:
`Pb_dot = −Pb·β/t` enters the numerator linearly). All the nonlinearity
lives in `L_loss(β, δ)` and `T_xi(β, δ)` through the bubble-structure ODE
(`bubble_luminosity.py::_get_bubble_ODE`, coefficients `β + 2.5δ` and
`β + δ`). This is friendly territory for a Powell-hybrid solver.

### Known numerical hazards (drive the design)

1. **Noise floor.** Each residual evaluation runs `fsolve` on dMdt
   (xtol = 1e-4; measured dMdt sensitivity ≤ 0.3%), an LSODA structure solve
   (`_BUBBLE_RTOL = 1e-8`, dMdt-residual path `_RESIDUAL_RTOL = 1e-6`), table
   interpolations, and discrete grid-point switches (`find_nearest`,
   CIE-point insertion). Expect relative noise ~1e-4 in g. `hybr`'s default
   forward-difference step (~1.5e-8) would differentiate noise →
   **must set `options['eps']`** ≳ 1e-3 (final value from the Phase-1
   transect probe).
2. **Failure plateau.** `get_residual_pure` returns (100, 100) when the
   bubble solve fails. Inside `hybr` that is a discontinuous cliff; instead,
   a failed evaluation must abort the `hybr` attempt (raise a private
   exception, catch around `scipy.optimize.root`, fall back to grid).
3. **Bounds.** `hybr` is unconstrained. Policy: run unconstrained, accept
   the root only if it lies inside the configured box
   ([BETA_MIN, BETA_MAX] × [DELTA_MIN, DELTA_MAX]); an out-of-box root is
   recorded (diagnostic) and treated as non-converged → grid fallback. The
   sample data shows δ pinned at 0.0, i.e. the box itself may be wrong at
   some epochs — widening DELTA_MAX is a **separate physics decision**;
   Phase 1 produces the evidence for it.
4. **Cost.** Budget `maxfev ≈ 30` (grid fallback costs up to 24 anyway);
   `factor = 0.1` to keep the first trust-region step local. The existing
   converged-input short-circuit (1 evaluation) is preserved so
   well-converged stretches stay cheap.

---

## Phases

### Phase 0 — Baseline measurement (no solver changes)

Goal: establish how (un)representative the mock run is.

- Small harvest script (in `scratch/`, not shipped) reading per-snapshot
  `residual_betaEdot` / `residual_deltaT` from run outputs (already saved by
  `run_energy_implicit_phase.py`).
- Run 3–4 configs spanning the worked examples in `param/` (at minimum
  `simple_cluster.param` and a config matching the mock run) with the
  current solver. Record per config: % implicit segments converged, residual
  distribution, segment count, wall time.

**Decision gate:** if other configs converge fine, the mock run was
pathological → re-scope to the normalization fix only and stop here.

### Phase 1 — Shadow-mode feasibility (the "is this a good idea" tests)

The solver is pure (no params mutation), so a prototype `hybr` can run
*alongside* the production grid in the same run without affecting it:

- Env-gated shadow hook (e.g. `TRINITY_BETADELTA_SHADOW=1`, following the
  `TRINITY_BUBBLE_STATE_DUMP` pattern): each segment, after the production
  solve, run prototype `_solve_hybr` from the same guess; log both results
  (g- and f-metrics, evaluation counts, root location, in/out of bounds,
  failures) to a sidecar file. Production trajectory unchanged.
- **Transect probe** (~5 segments across the phase): evaluate g along β- and
  δ-transects with spacings 1e-4 … 1e-2; measure the noise floor and look
  for discontinuities from the discrete switches → fixes `eps`, and confirms
  finite differences are usable at all.
- **Root-existence spot check** (~10 segments): wide dense scan (e.g. 50×50
  over [0,1] × [−1, +0.3]) + local polish → does an in-box root exist? Does
  the legacy box exclude it (δ > 0 epochs)? Does hybr's root match the scan?

**Pre-registered success criteria (go/no-go):**

1. hybr converges (g-total < 1e-4, in-bounds) on ≥ 80% of segments where the
   wide scan shows a root exists (baseline: 0% in the sample run).
2. Median ≤ 15 bubble evaluations per non-short-circuited segment.
3. Every hybr failure is cleanly detected (no NaN/garbage accepted).
4. Transects show g smooth at the chosen `eps` (steps between adjacent
   samples ≲ 10× noise floor away from genuine curvature).
5. If the scan shows **no in-box root for a majority of segments**, hybr
   cannot fix the phase → pivot to the bounds/closure-model question
   instead of proceeding.

Deliverable: short `analysis/` report (with the staleness banner) + go/no-go.

### Phase 2 — Implementation (if go)

All changes inside `get_betadelta.py` plus one call-site wire-up; legacy
grid path kept byte-identical and selectable.

1. **Refactor residual internals**: extract
   `get_residual_components_pure(...) -> (E1, E2, T, props)`;
   `get_residual_pure` becomes a thin wrapper (callers untouched).
2. **Single metric helper**: computes legacy f-components (for output
   compatibility: `residual_betaEdot`, `residual_deltaT`,
   `residual_Edot{1,2}_guess` keys keep their meaning) *and* g-components.
   Candidate ranking, convergence tests, and the input short-circuit all
   switch to the g-metric — one consistent metric everywhere. This is a
   deliberate behavioral change (it also affects the grid path) and is the
   normalization fix; threshold stays 1e-4 unless Phase-1 data says
   otherwise.
3. **`_solve_hybr(beta_guess, delta_guess, params, input_props)`**:
   - dMdt warm-start threading via a stateful closure (mirrors
     `_solve_grid`);
   - func returns `[g1, g2]`; bubble failure raises `_HybrAbort` → caught →
     returns None (fallback engages);
   - `scipy.optimize.root(..., method='hybr', options={'eps': <Phase 1>,
     'xtol': 1e-8, 'factor': 0.1, 'maxfev': 30})`;
   - returns (β, δ, props-at-root, g_total, nfev) or None; out-of-box root
     → returned as a non-converged candidate with a logged counter.
4. **Orchestration** in `solve_betadelta_pure`: `method='hybr'` does
   input-short-circuit → hybr → grid fallback → best-of-candidates
   (machinery kept). `method='grid'` preserves today's path for A/B.
   L-BFGS-B left untouched for now (it is unreachable in practice); removal
   is a Phase-4 cleanup decision.
5. **Param key** `betadelta_solver` (`grid` | `hybr`) in
   `trinity/_input/default.param` + spec, read at the call site
   (`run_energy_implicit_phase.py` ~line 603). Default stays `grid` until
   Phase 4 flips it.
6. **Surface non-convergence**: count non-converged segments per phase and
   log one summary WARNING at phase end (today the only trace is per-segment
   DEBUG). Independent of solver choice.

### Phase 3 — Tests

Unit tests extend `test/test_betadelta_solver.py` patterns (monkeypatched
residual; needs a *signed two-component* recorder variant since root-finding
needs signs, not just magnitudes):

- hybr finds the root of a smooth synthetic 2-D system (known root) within
  tolerance and few evaluations; dMdt seeds thread through evaluations.
- Bubble failure mid-hybr → clean grid fallback (recorder observes calls).
- Out-of-bounds root → not accepted as converged; fallback engages; result
  in bounds.
- Converged-input short-circuit: still exactly 1 evaluation.
- **Pole regression test**: synthetic landscape where E1 crosses zero at the
  root — legacy f-metric blows up, g-metric converges. Encodes the
  normalization bug directly.
- Candidate ranking uses the g-metric consistently across input/grid/hybr.
- Back-compat: existing grid tests stay green; `method='grid'` ranking
  changes only via the metric swap (asserted explicitly).
- Integration (marked `stress`): one short real run with
  `betadelta_solver=hybr` asserting (a) converged fraction ≥ threshold from
  Phase 1, (b) total bubble evaluations ≤ 1.25× grid baseline, (c) finite
  trajectories, no new warnings.
- `pytest` green before/after; `pre-commit run --all-files`; `mypy trinity`.

### Phase 4 — Validation & rollout

- Re-run the Phase-0 configs with `betadelta_solver=hybr`. Before/after
  report: convergence fraction, residual distributions, (β, δ), Eb/R2/v2
  trajectories, wall time, and the implicit→transition kink.
- Physical sanity checks: at accepted roots late in the phase, the two Edot
  branches must agree in sign; Eb should now **peak inside** the implicit
  phase (forced by cooling-balance termination + PdV > 0) instead of rising
  monotonically to the boundary.
- Flip `betadelta_solver` default to `hybr`; keep `grid` selectable.
- Separate cleanup decisions (own commits, not bundled): remove the dead
  L-BFGS-B path; revisit `DELTA_MAX = 0` with Phase-1 root-location
  evidence; consider T0 re-anchoring (out of scope here).

## Risks

| Risk | Mitigation |
|---|---|
| No in-box root exists at some epochs (δ pinned at 0 in the data) | Phase-1 wide scan detects; bounds widening is a separate, evidence-backed decision; gate 5 |
| Residual too noisy/discontinuous for finite-difference Jacobian | Phase-1 transect probe sets `eps`; worst case stay on grid (no-go) |
| hybr wanders into ODE-failing regions | abort-and-fallback contract; `factor=0.1` keeps steps local |
| Results change vs. old runs | intended (old runs carried O(1) residuals); A/B via param key; documented in report |
| Cost regression in well-converged stretches | input short-circuit preserved; hybr only runs when input not converged |
| Metric swap changes grid-path ranking | explicit test pinning the change; thresholds re-validated in Phase 1 |
