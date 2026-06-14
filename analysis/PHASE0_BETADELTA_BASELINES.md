# Phase 0 results: beta–delta solver baselines (four configs)

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

Companion to `docs/dev/BETADELTA_HYBR_PLAN.md` (plan v2). Baselines ran the
production solver at worktree commit `1eda451` (code identical to main;
docs-only commits after the merge base). Environment: Python 3.11,
numpy 1.26.4, scipy 1.17.1, astropy 7.2.0. Harvest scripts:
`scratch/phase0/harvest.py`, `scratch/phase0/predictor_test.py` (not
shipped). Convergence criterion throughout: legacy f-metric total
(`residual_betaEdot² + residual_deltaT²`) < 1e-4 at the accepted point.

## Configs

All four in `scratch/phase0/*.param` (uncommitted; contents inline):

| run | mCloud | sfe | profile | nCore | rCore | nISM | stop_t |
|---|---|---|---|---|---|---|---|
| `base_mock4e3` | 3966 | 0.0085 | flat (α=0) | 5e2 | — | 0.1 | 0.3 |
| `base_simple1e5` | 1e5 | 0.3 | flat (α=0) | (default) | — | (default) | 0.3 |
| `base_cloud1e6` | 1e6 | 0.01 | flat (α=0) | 1e3 | — | 10 | 1.0 |
| `base_cloudPL` | 1e6 | 0.01 | α_ρ = −2 | 1e5 | 1 pc | 10 | 1.0 |

`base_mock4e3` replicates the committed sample run (`4e3_sfe001_n5e2_PL0`)
and sits just below the supported low-mass corner (maintainer: mCloud ≳ 1e4,
sfe ~ 0.01). `simple_cluster` and the 1e6 configs are worked examples;
`base_cloudPL` uses the worked PL example with a physically typical 1 pc
core (maintainer guidance) and core density raised to 1e5 cm⁻³ to pass the
`rCloud_max` plausibility validation (rCloud ≈ 23 pc).

## Headline table (implicit phase, per accepted segment)

| run | segs | converged | β at ±0.02 grid edge | hard-bound hits | sign-wrong Ėb segs | residual median/max |
|---|---|---|---|---|---|---|
| `base_mock4e3` | 50 | **0%** | 98% | δ=0: 12 | **16** | 1.2 / 3.3 |
| `base_simple1e5` | 61 | 52.5% | 42% | 0 | **10** | 9.8e-5 / 3.3 |
| `base_cloud1e6_ext` | 121 | 77.7% | 28% | 0 | **10** | 4.0e-5 / 3.1 |
| `base_cloudPL` | 56 | **0%** | 73% | **β=1: 14 (10 consecutive)**, δ=0: 5, δ=−1: 1 | 8 | 0.42 / 8.9 |

(`base_cloud1e6_ext` supersedes the right-censored `base_cloud1e6` row —
91 segs, 93.4% — whose stop_t = 1.0 Myr cut the phase short; the extended
run reached natural cooling-balance termination at t = 2.544 Myr, between
the two extrapolations, and continued through transition and momentum.)

Phase-resolved convergence for `base_simple1e5`: first third 25%, middle
third **100%**, last third 33% — two distinct failure episodes (post-handoff
chase; pre-transition runaway) bracketing a settled regime where the grid
solver is flawless. The same structure appears in `base_cloud1e6_ext`,
time-resolved: 93.3% (t < 1 Myr) → 50% (1–2 Myr) → **0% (t > 2 Myr)**.
**The pre-transition runaway is universal**: every Phase-0 config, across
mass (4e3→1e6) and profile (flat and α_ρ = −2), ends its implicit phase at
0% convergence approaching cooling balance — and then hands its last
(garbage) accepted (β, δ) to the transition phase.

## Findings

1. **Gate G0: PASS — Phases 2–4 proceed.** Material non-convergence on a
   worked example (47.5% of `simple_cluster` segments) and total failure on
   two configs; cap/bound saturation on three of four.
2. **Mechanism, flat profiles: the ±0.02/segment drift cap.** The mock
   chase rail-rides 98% of steps for the entire phase (β 0.76→0.24,
   residuals → 3.3); it nearly converges around segment 20 (residual
   2.1e-2) but never crosses threshold before the root accelerates away.
   δ=0 pinning is transient (12/50 segments) — the corner the capped chase
   gets clipped against, not a root exclusion.
3. **Mechanism, steep profiles: the hard bounds.** `base_cloudPL` pins
   β at BETA_MAX = 1 for 10 consecutive segments right after the shell
   exits the 1 pc core, where the adiabatic self-similar root
   3α̃−1 crosses 1 (reaching 1.42 by phase end, cooling pushing higher).
   0% convergence outside the core; terminal unwind to β = 0, δ = −0.96
   (touching DELTA_MIN). The box excludes the root; the prediction
   β_adia = 3·(3/(5+α_ρ))−1 > 1 for α_ρ < −2/3 is confirmed in production.
4. **The legacy f-metric pole is NOT operative anywhere.**
   |Ėb_from_β| ≥ 0.14·Lmech in every segment of every run; zero
   zero-crossings. Recomputing the g-metric offline leaves every
   convergence rate unchanged, and at the mock's accepted points g grows
   to 4.5 — the failures are real distance from the root, not metric
   artifacts. The metric swap remains justified as hygiene (pole-free,
   denominators per-segment constants), not as a fix.
5. **Production integrates wrong-signed Ėb on affected configs.** In the
   late phase the energy-balance branch goes negative while the β-branch
   (which the integrator uses) stays positive: 16/50 mock segments,
   10/61 simple_cluster segments, 8/56 cloudPL segments. E_b consequently
   rises monotonically to the implicit→transition boundary in all four
   runs instead of peaking inside the phase. **Paper-I caveat material**:
   published implicit-phase E_b(t) tracks for low-mass and steep-profile
   configs integrate clamped, lagged β with sign-wrong Ėb stretches.
6. **Analytic warm-start predictor and tiered 1-D solver: rejected.**
   Tested offline (`predictor_test.py`): on converged segments the
   A12-inverse predictor errs by median |Δβ| 0.036 / |Δδ| 0.067 vs ~0 for
   the previous-root warm start; the consistency relation
   δ = (2/7)(2α̃−β−1) misses solved δ by 0.05–0.14 even where the solver
   converges. Open: behavior at SB99 luminosity jumps (none in the data).
7. **Cost baseline (bounded, not exact** — production does not persist
   per-segment evaluation counts): an unconverged segment pays the full
   5×5 grid (24 evaluations + 1 input check); a converged-input segment
   short-circuits at 1. Convergence rates above therefore bound the
   short-circuit rate: ~0% on mock/cloudPL (every segment pays full
   price), ≤93% on cloud1e6. Exact counts come from the Phase-2 arms.

## Phase 1 drift check (D1) — verdicts

Phase-1 safety fixes (commit `3496b8e`) vs baselines, same configs.
**Energy phase (isolates the R1 bracket change): strict PASS on all four
configs** — max relative deviation 6.6e-9 / 2.7e-9 / 1.4e-8 / 1.9e-8
(cloud1e6 / mock / simple1e5 / cloudPL), pure brentq-tolerance noise;
zero R1 failures anywhere. The implicit phase is dominated everywhere by
the *intentional* dt mitigation (active from segment 0 — the handoff
segment is unconverged on every config), so the strict <1e-5 budget
effectively gates only the energy phase; implicit-phase differences below
are attributable behavior changes, not drift.

- `cloud1e6` (t ≤ 1.0 window): end-state ΔEb 0.17%, ΔR2 0.012%;
  convergence 93.4% → **98.9%**. The mitigation helps cleanly on
  mildly-affected stretches. The *uncapped* extended attempt additionally
  showed β **pinning at BETA_MAX = 1 late in the phase even on this flat
  profile** (t ≈ 1.5+, dt ground to the floor, projected ~4 days) — the
  bound binds wherever the solver tracks well into the cooling-dominated
  regime, not just on steep profiles.
- `cloud1e6` extended, capped (stop_t = 6.0; **complete**, ~67 min):
  traverses all phases with **zero β-pinned segments** — the uncapped
  death-grind did not recur, so that grind was at least partly an
  artifact of the unbounded mitigation itself (floor-dt steps trapping
  the solver in the pinned region). Convergence by era: ~100% through
  t ≈ 1.1, **12.5% for t ≈ 1.2–2.7** — the late-phase failure stands
  regardless of dt policy. Shifts vs the uncensored baseline, from dt
  policy + convergence differences alone: implicit→transition
  2.544 → 2.722 Myr (+7%), momentum onset 3.577 → 3.874 Myr (+8%); at
  t = 6 Myr, R2 54.7 → 58.7 pc (+7%), v2 6.07 → 7.12 km/s (+17%).
  Neither run is ground truth (both leave the late phase largely
  unconverged), so this spread is a **lower bound on solver-induced
  uncertainty in published quantities on the flagship config** — the
  number that motivates Phase 4's attribution gate.
- `mock4e3`: **the baseline's phase boundary was a solver artifact.**
  Phase-1 run: 33.7% converged (baseline 0%), sign-wrong-Ėb segments
  16 → 0, β ends ~0.9 (baseline rail-artifact 0.24) — and cooling balance
  is NOT reached by t = 0.3 Myr, vs the baseline's exit at t = 0.101
  computed from garbage (β, δ). Extrapolated natural end ≈ 1.2 Myr: the
  implicit→transition time moves by ≳3×, corrupting all downstream
  evolution. Cost: 181 segments vs 50 for a third of the (old) phase.
  **Capped-mitigation completion of the same config** (the dt-policy
  trade-off, measured): with the cap the run finishes in ~45 min but the
  whole phase is one long unconverged streak, the mitigation disengages
  ~10 segments in, and the baseline pathology returns (0% converged,
  17 sign-wrong segments, artifact boundary at t = 0.068). Three-way
  verdict: baseline = fast+wrong; uncapped = slow+partially right;
  capped = fast+wrong-but-bounded. **No dt policy is both affordable and
  correct on pathological configs — only the Phase-2/3 solver change
  fixes them.** The cap stays as the shipping default (bounded cost;
  short-streak benefit intact on mildly-affected configs).
- `simple1e5` (partial, stopped deliberately): **adverse interaction
  found.** In its late-phase episode the grid cannot converge at any dt,
  and shrinking dt just gives the broken minimizer more ±0.02 steps per
  unit time: β unwound to 0.0 (hit BETA_MIN; baseline only fell to 0.68),
  ~200 segments to cover what the baseline did in ~50, dt pinned at the
  floor. The mitigation amplifies the parameter-space unwind when no
  catchable root exists.
- `cloudPL` (partial, stopped deliberately): β pins at BETA_MAX = 1
  exactly as in the baseline — no dt policy helps when the root is
  outside the box. Same floor-dt cost blow-up.

**D1 wall-time budget (+5%): catastrophically exceeded on pathological
configs (≳4–10×), inherent to the dt mitigation as first designed.** The
worst field case: the extended Phase-1 heavy-config run ground at the dt
floor in its β=1-pinned stretch at ~1e-4 Myr/segment — a ~4-day projected
completion. Refinement implemented in response
(`BETADELTA_DT_SHRINK_MAX_STREAK = 10`): beyond 10 consecutive
unconverged segments the forced shrink and growth suppression disengage
and standard adaptive stepping resumes, with a WARNING at the transition
— a long streak means the root is unreachable (outside bounds or
outrunning the grid window even at the floor), where floor-dt multiplies
cost without buying correctness. The real remedy for those stretches is
the Phase-2/3 solver change. Both extended Phase-1 runs
(`p1_mock4e3_ext`, `p1_cloud1e6_ext`) run with the cap after a container
restart forced their relaunch — the cap's field test.

## Caveats

- (Resolved) The original `base_cloud1e6` was right-censored at
  stop_t = 1.0 Myr and reported 93.4% convergence; the extended run
  (`base_cloud1e6_ext`, natural phase end at t = 2.544 Myr) shows the true
  full-phase figure is 77.7% with a 0%-converged final stretch. Protocol
  (now in the plan): runs count only if the implicit phase ends naturally
  on cooling balance and the run continues through transition — the
  transition phase consumes the last implicit (β, δ) without re-solving,
  so censored runs both hide the worst regime and mask cross-phase
  contamination (mock hands transition β=0.24/δ=−0.60; cloudPL hands it
  β=0.00/δ=−0.96). The pole stays dead in the extended run too:
  |Ėb_from_β| ≥ 0.16·Lmech with zero sign changes through the full phase.
- Wall-time comparisons across runs are unreliable here: matched log
  markers showed a 1.27× host-speed difference between identical runs at
  different times of day (shared cloud infrastructure). All cost gates in
  later phases should use evaluation counts, not wall time, where possible.

## Phase 2 interim — probe results (steps 2.1/2.2)

Probes ran in-process beside the production solver (transects + wide
coarse scans β∈[−1,1+], δ∈[−1,0.25+]) on `mock4e3` (complete: segments
2, 8, 20, 45 — the capped mock phase has only ~54 segments) and
`simple1e5` (2, 8, 20, 45 done; 90, 160 in progress).

- **Noise floor (2.1): ~1e-7 everywhere.** Finest-spacing transect
  jitter is 5e-9…5e-7 at every probed segment on both configs — four
  orders below the 1e-4 convergence tolerance. The residual landscape is
  smooth; default hybr finite-difference `eps` is safe; **the ξ_Tb = 0.98
  edge-amplification worry is falsified** (δ-direction noise is the
  *smaller* of the two). No dMdt-tolerance tightening needed (2.1b skipped).
- **Root maps (2.2), `simple1e5`:** both residual components change sign
  in-box at all four probed segments — a root exists. The grid minimum
  drifts from mid-box (seg 2–8) to **β ≈ 0.9–1.0 by segments 20–45**,
  i.e. toward the production wall, consistent with the late-phase failure
  being "root crosses the bound" (the bounds hypothesis). Late-segment
  maps (90, 160) pending — they decide it.
- **Root maps (2.2), `mock4e3`:** roots in-box at segments 2 and 8; by
  **segment 20 the T-residual is single-signed across the wide box** (no
  zero contour → no root), and at segment 45 the landscape minimum is
  residual ≈ 1.4 at the box *corner* (β=−1, δ=−1) — no root anywhere
  plausible. Caveat: traversal states on the mock come from the shipping
  solver's unconverged trajectory (0% conv on the capped run), so this
  shows "no root at the states the production code actually visits," not
  "no root along the true trajectory" — the uncapped D1 run (34% conv,
  β tracking to ~0.9) visits very different states. The in-line arms
  (Phase 2.3), which re-solve from each arm's *own* trajectory, are the
  honest test; if D (free hybr) also finds no roots on the mock, the
  pre-registered pivot clause applies to this config (closure
  inconsistency — model finding, not solver bug).
- **Arm-harness smoke (2.3 preview), mock segments 1–2:** hybr (arm D)
  found machine-precision roots (f ≈ 7e-16) at (β=0.230, δ=−0.357) and
  (β=0.143, δ=−0.305) in 25–34 evaluations — far from production's
  accepted (0.78, −0.15) with f ≈ 2–3. The energy-phase handoff guess is
  simply wrong and the ±0.02 window cannot walk to the true root, so
  the mock's early "failures" are a reachability problem, not missing
  roots. Two operational caveats measured: (a) hybr's own status said
  failure (ier 5/2) *at a true root* — acceptance must be by residual
  value, not `sol.success`; (b) the inner dMdt fsolve is multimodal and
  seed-dependent — the same (β, δ) root evaluates cleanly with one
  warm-start seed (dMdt ≈ 8.46) and hangs >20 s with another (≈ 3.56),
  so per-point timeouts are load-bearing in any production hybr path.
