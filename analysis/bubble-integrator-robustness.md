# Bubble-integrator robustness: the flaky `MonotonicError`

> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**
>
> **Audit status (2026-06-08):** reconciled with the code through commit
> `a245c29` (#659); see the Status notes in §I.3 / Step 1 / Step 2 for what
> shipped vs. what's still pending.

Single source of truth for eliminating the intermittent
`trinity._functions.operations.MonotonicError` raised from the bubble
luminosity solver. Same shape as the sibling audits: (Part I) the audit —
*what actually fails and why* — and (Part II) the phased fix and its
stress-test battery — *what to do and how to prove it is gone without
moving a single number on the runs that already pass*.

**This is the one behavior-AFFECTING item in the restructure effort.**
Unlike the A–C structural phases (byte-preserving renames), this changes a
runtime code path. It is therefore sequenced **dead last**, after the
structural churn settles, and held to a stricter bar: *the success path
must stay byte-identical; only the failing path may change.*

> **Status (2026-06): the guard shipped, made shape-aware (Part A).**
> `find_nearest_higher` now tolerates *shallow + localized* numerical
> non-monotonicity and still rejects *deep / sustained-interior* cases
> (`monotonic()` itself is unchanged, so the success path is byte-identical).
> The grid de-refinement (the deeper root-cause angle) is **deferred** pending
> an output-diff and model-author sign-off. **Part I below is the original
> audit (pre-fix "what is").** See **§I.7** for exactly what changed and why
> the grid fix is deferred, and **§I.6** for the diagnostic evidence.

> **Status (2026-06, #659): the detect-and-fail-loud half of Step 2 also
> shipped, and the root-cause model changed.** `odeint` is now wrapped by
> `_odeint_checked` (`bubble_luminosity.py:71`), which reads LSODA's status so a
> failed solve is never consumed; callers raise `BubbleSolverError` (`:63`) or
> return the `_SOLVER_FAIL_RESIDUAL` penalty (`:60`) instead of garbage.
> Critically, #659 found the real source of the *non-determinism* is `odeint`
> returning **uninitialised memory** for the un-integrated tail — *not* BLAS
> thread FP noise — so the §I.3 theory and the Step-1 thread-pinning remedy are
> **superseded** (see the dated corrections in §I.3 and Step 1). The `solve_ivp`
> *recovery* path (Step 2 item 3) is still **not** wired.

## TL;DR

- **Symptom**: `run.py` intermittently aborts with a bare `MonotonicError`
  (exit 1), ~1-in-3 on the quickstart-class param. Same inputs, same numpy
  (1.26.4) — passes on re-run. It is the failure the smoke test exists to
  catch and the fragility called out in `requirements.txt`.
- **Mechanism**: `scipy.integrate.odeint` (LSODA) silently fails on a stiff
  bubble-temperature ODE, returns a **zero-tailed** `T_array`
  (`[30000 … 0. 0. 0.]`), and that non-monotonic tail trips a monotonicity
  guard far downstream. The real failure (a dead integrator) is swallowed;
  the symptom is a cryptic error in an unrelated helper.
- **Root trigger** *(superseded — see the §I.3 correction)*: originally
  attributed to `fsolve`/BLAS FP-thread sensitivity nudging `bubble_dMdt` into
  a regime where `T → 0` and the RHS terms `∝ 1/T`, `1/T^{5/2}`, `dTdr²/T`
  blow up → LSODA gives up. #659 found the dominant cause of the *flake* is
  instead `odeint` handing back uninitialised memory on a failed solve
  (single-threaded + fixed seed still flaked).
- **Fix posture**: (1) make it *deterministic* (cheap), (2) make the
  failure *loud and recoverable* (`full_output` + a real fallback — the
  codebase already carries an unused `solve_ivp` path), (3) guard the RHS
  so `T` cannot cross the physical floor. Prove with an N-run stress
  harness, not a single pass. *(Update: (2)'s detect/fail-loud half shipped in
  #659 via `_odeint_checked`; the `solve_ivp` recovery is still pending. (1)
  was achieved by detection, not thread-pinning — see Step 1.)*
- **Addendum (2026-06)**: a **second, distinct trigger** of the same guard
  was later identified — a small, smooth, *deterministic* startup transient
  at the `T_init=3e4` outer boundary (not the zero tail). It is benign and
  is *caught* (penalty, not crash) in long low-SFE runs. See **I.5**; the
  fix differs by mode, and the **Step-0 `TRINITY_BUBBLE_DIAG` diagnostic**
  (now implemented) tells the two modes apart on real runs.

---

# Part I — Audit (what is)

## I.1 Where it breaks

`trinity/bubble_structure/bubble_luminosity.py`, in
`get_bubbleproperties_pure`:

1. **`bubble_luminosity.py:381`** — `scipy.optimize.fsolve(
   velocity_residuals_wrapper, …, xtol=1e-4, factor=50, epsfcn=1e-4)` solves
   for `bubble_dMdt` from a velocity boundary condition. `fsolve` internally
   does finite-difference Jacobians and BLAS-backed linear algebra → output is
   sensitive to floating-point summation order (which varies with BLAS
   threading). *(But see the §I.3 correction: this FP sensitivity is not the
   dominant cause of the run-to-run flake.)*
2. **`:430`** — `odeint(_get_bubble_ODE, initial_conditions, r_array,
   args=(params, Pb), tfirst=True)`, now wrapped by `_odeint_checked` (`:71`).
   **Originally had no `full_output` / no return check** (the bug): when LSODA
   fails it prints `ODEintWarning: Illegal input detected (internal error)` to
   stderr and returns an array whose unintegrated tail is **uninitialised
   memory** (often zeros). *(Fixed in #659 — `_odeint_checked` reads LSODA's
   status and the caller raises `BubbleSolverError` instead of consuming the
   tail.)*
3. **`:446`** — `n_array = Pb / ((mu_convert/mu_ion)·k_B·T_array)` then divides
   by any zero tail → `RuntimeWarning: divide by zero` (the visible
   breadcrumb). *(Formula updated by the mu audit, #657; was `Pb/(2·k_B·T)`.)*
4. **`:481`** — `operations.find_nearest_higher(T_array, _CIEswitch)`.
   `find_nearest_higher` (`trinity/_functions/operations.py:144`) runs the
   monotonicity guard and **raises `MonotonicError`** because the
   `[…, 0., 0., 0.]` tail is neither increasing nor decreasing. *(The guard is
   now `_is_monotonic_or_tolerable`, not strict `monotonic()` — see §I.7.)*

The exception is raised ~100 lines and one module away from the actual
fault. Nothing logs *"the integrator failed"*; you get a bare
`MonotonicError` with a printed array.

## I.2 Why the RHS is fragile

`_get_bubble_ODE` (`bubble_luminosity.py:1063`, Weaver+77 Eq. 42-43):

```python
if np.abs(T - 0) < 1e-5:
    logger.critical('T is zero in bubble ODE'); sys.exit()   # :1067-1069
ndens = Pb / ((mu_convert/mu_ion) * k_B * T)                 # :1071 (mu audit #657; was 2*k_B*T)
dTdrr = (Pb / (C_thermal * T**(5/2)) * ( … + 2.5*(v - v_term)*dTdr/T - dudt/Pb )
         - 2.5 * dTdr**2 / T - 2 * dTdr / r_arr)
dvdr  = ( … + (v - v_term)*dTdr/T - 2*v/r_arr )
```

Every dominant term has `T` in a denominator (`T`, `T^{5/2}`). As the
integration approaches small `T`, `dTdrr` and `dvdr` stiffen sharply; a
slightly-off `bubble_dMdt` or initial condition can drive `T` down fast
enough that LSODA's step control fails ("illegal input"). The lone guard
(`abs(T) < 1e-5 → sys.exit`) only fires on an *exact* zero inside the RHS,
not on the gradual collapse that actually happens, and `sys.exit` is itself
a hostile failure mode for a library function.

## I.3 Why it is non-deterministic

> **Correction (2026-06, #659) — the theory below is superseded.** The flake is
> **not** primarily BLAS/OpenMP thread FP order. #659 reproduced the OK/crash
> coin-flip **single-threaded with a fixed `PYTHONHASHSEED`** (measured
> 794/786/792 `odeint` calls across identical runs). The real mechanism: when
> LSODA bails (istate ≠ 2), `scipy.integrate.odeint` returns **uninitialised
> memory** for the un-integrated tail, and *those bytes* vary run-to-run (e.g.
> denormals like `1.58e-318`). Consuming that garbage is what made the whole
> bubble solve non-deterministic — `fsolve` could converge on a ~0 garbage
> residual, and `T_array` could gain a random tail that intermittently tripped
> `MonotonicError`. The fix (#659, scipy 1.17.1 / numpy 1.26.4) is to **detect**
> the failed solve (`_odeint_checked`) and never consume the tail; that alone
> makes runs byte-identical. Thread-pinning (the old Step 1) is therefore not
> the cure. The original theory is kept below as the historical record.

Identical `.param`, identical numpy 1.26.4, different outcomes across runs
(observed: passed 2 of 3 identical re-runs in Phase B verification). The
only run-to-run variable is **floating-point summation order in
multithreaded BLAS/OpenMP** feeding `fsolve`/`odeint`. That nudges
`bubble_dMdt` in the last digits, which occasionally tips the stiff ODE
over the edge. This is exactly the class of bug `requirements.txt` pins
numpy `<2` for — and pinning numpy only narrows it, it does not close it.

**Measured baseline flake rate** (this environment, quickstart-class param
`mCloud 1e5 / sfe 0.3 / stop_t 1e-4`):

- Observed in Phase B verification: **1 of 3** identical re-runs failed
  (numpy 1.26.4, default threads). A larger pinned-vs-default sweep is in
  flight to fix the rate and decide whether thread-pinning (Step 1) closes
  it; this line will be updated with the N-run counts. The order of
  magnitude — roughly a third of runs — is what matters for the plan.

## I.4 Existing scaffolding (use it, don't reinvent)

`bubble_luminosity.py` already contains **unused alternative solver
scaffolding** built on `scipy.integrate.solve_ivp`: `_solve_bubble_ode_with_ivp`
(`:935`, no callers) and the solve_ivp-based `_create_adaptive_radius_grid`
(`:809`, disabled — see the NOTE at `:400`). The `odeint` calls themselves are
now all routed through `_odeint_checked` (sites at `:430`, `:546`, `:1008`).
Prior hands clearly started a `solve_ivp` route; the recovery fix should
finish/wire that rather than add a third parallel solver.

## I.5 Addendum (2026-06): a second, distinct trigger — the boundary startup transient

The mechanism in I.1–I.3 (a **dead integrator** leaving a *zero tail* at the
hot/inner end of `T_array`) is **not the only** way the `:481` guard trips. A
separate investigation of a long, low-SFE run — `mCloud 1e5 / sfe 0.01 /
nCore 1e4 / densPL_alpha 0 / PHII` (the `rosette_sweep_denser_PISM1e4` case;
terminated `shell_collapsed`, exit 4), whose log showed several *caught*
`MonotonicError` warnings clustered late in Phase 1b — found a **second,
benign, deterministic** trigger that is the *opposite* of the zero tail:

- **Shape**: the T-profile is fully populated and ends hot
  (`T[-1] ≈ 1e7`). The only non-monotonicity is a **single smooth dip at the
  outer/cold start** — the hardcoded `T_init = 3e4` boundary in
  `_get_bubble_ODE_initial_conditions` (`:1041`; `T_init = 3e4` at `:1043`).
  `T` decreases for the first handful of points, reaches one trough, then
  rises monotonically. It is the
  *other end* of the array from the I.1 zero tail, and the integrator
  reports success (`full_output` `message = 'Integration successful.'`,
  no `ier` error).
- **Magnitude (measured)**: confined to the first ≲0.12 % of the 59 992-point
  grid (≤72 points); worst *cumulative relative drawdown* **1.3e-3** (≈39 K
  out of 30 000 K) at the corner β=1, δ=0. It scales smoothly with
  **(β+δ)** and is **0.000 for most of Phase 1b**, emerging only near the
  energy→momentum transition (drawdown vs t: ≈0 at t≲0.3 → 3.6e-5 at t=0.6 →
  1.3e-3 at t=0.846). Evidence: a faithful reconstruction at the run's logged
  transition state (`R2=4.12, Eb=1.637e6, t=0.846`) with the real cooling
  tables and ODE, swept over the full (β,δ) domain.
- **Severity**: unlike the intermittent crash in I.3, this mode is
  **deterministic** and, in the observed run, **caught** —
  `get_betadelta.get_residual_pure` (`get_betadelta.py:316`; broad
  `except Exception` at `:356` → `return 100.0, 100.0, None` at `:358`) swallows
  the `MonotonicError` and returns the `(100, 100)` sentinel residual. That
  biases the β/δ solver away from the (physically fine) high-(β+δ) region
  and discards the structure; when the *chosen* candidate trips,
  `bubble_props` comes back `None` and `run_energy_implicit_phase.py:621`
  silently carries stale bubble physics forward. The run completes, but the
  trajectory can be perturbed — so the `shell_collapsed` (exit 4) outcome
  **cannot be assumed physical** until this mode is removed.
- **Caveat**: the reconstruction used a formula-restarted `bubble_dMdt`,
  whereas the live run warm-starts it. Whether the real run's events are this
  mode, the I.1 zero-tail mode, or both, is exactly what the Step-0
  diagnostic (below) settles on the actual machine/sweep.

**Caution for any guard-relaxation fix** *(now addressed — see §I.7)*.
`find_nearest_higher` / `find_nearest_lower` determine direction with
`mon_incr = kindof_increasing(array)` (`operations.py:66`) **after** the
monotonic guard. If the guard is loosened to tolerate the boundary dip, that
*strict* `kindof_increasing` returns `False` for a now-tolerated dippy
*increasing* array, flips `mon_incr`, and corrupts the index step → a wrong
CIE/cooling-switch index. The shipped fix did exactly what this section
prescribes: `find_nearest_higher`'s direction test is now
`mon_incr = array[-1] >= array[0]` (`operations.py:165`); `find_nearest_lower`
(`:43`) still uses strict `kindof_increasing` (it keeps the strict guard, so it
is safe).

## I.6 Addendum (2026-06): diagnostic results + the single-point spike

Running the Step-0 `TRINITY_BUBBLE_DIAG` capture on the real
`1e5_sfe001_n1e4_PL0_yesPHII` case classified **every** guard trip as benign:
6 `boundary_transient` (the I.5 cold-edge dip) and 2 `bulk_nonmonotonic`,
with **zero** `dead_integrator` events — the I.1 zero-tail mode did **not**
fire in this run.

The 2 `bulk_nonmonotonic` events are one physical structure (captured twice):
a **single isolated +0.7 % spike at one grid point** — `T` jumps
≈163 160 → 164 320 K then immediately returns to trend (one wrong-direction
step; cumulative drawdown **7.06e-3**), at index ≈19 671 (≈33 % of the array),
`T≈1.6e5`. It is **not** a physical inversion (one point, neighbours on a
smooth rise, hot tail) and **not** at `rCloud` (index 20 474) nor exactly at a
grid stitch (≈20 000).

**Measured fact — the outer grid is pathologically over-refined.**
`_create_legacy_radius_grid` Step 2 packs 20 000 points into a ~1.6e-4 pc
sliver near `r2Prime`, so `dr ≈ 5e-9 pc` (relative spacing ~1.3e-9) — `r` is
constant to six decimals across thousands of consecutive indices. The cleaner
(`MIN_SPACING = 1e-12` relative) sits far below 1.3e-9, so these
**near-duplicate radii survive**. The spike sits in that over-dense band.

**Mechanism (confirmed from the captured step diagnostics).** The captured
LSODA `infodict` is the smoking gun: only **~854 internal steps for the
~60 000 output radii** (≈70× dense-output *interpolation*), with the step size
`|hu|` collapsing to ~`dr` (~5e-9 pc) throughout the over-dense band and
jumping to ~1e-2 pc outside it (`mused ≡ 1`, Adams throughout — no stiff
switch). Requesting thousands of near-duplicate output radii inside a handful
of real steps is exactly what stresses LSODA's dense-output interpolation
(cf. the "intdy-- t illegal" warnings in I.1); one unlucky interpolation
returns a single off sample → the spike ("once, then nothing"). The exact
single sample was *not* reproducible with a guessed `v2` (state-sensitive),
consistent with a rare interpolation hiccup rather than a physical feature.
Note that **both** benign modes (I.5 dip + this spike) live in the over-dense
front band — two faces of the same over-refined-grid fragility, which
strengthens the Step-3 grid-hygiene angle.

**Diagnostic enrichment (this commit).** `_capture_bubble_integration` now
also records `rCloud`/`rCore` and the full LSODA `infodict`
(`info_hu` step sizes, `info_mused`, `info_nst`, …; note `odeint` has no
`ier` — success is read from `message`). `tools/inspect_bubble_diag.py` now
plots grid spacing (`dr`) with the over-dense band shaded, reference lines
(rCloud, R1/R2, stitches, CIE/cool switches), the step-size panel, and prints
a per-event geometric **verdict** — so the interpolation-glitch hypothesis is
checkable from `info_hu` on the next run.

**Implication for the fix.** Every observed trip is benign, so a guard
*tolerance* (cumulative-drawdown ≤ `rtol`, with the spike at 7.06e-3 setting
`rtol` ≳ 1e-2) is safe. The more fundamental cure is **de-refining the grid**
(Step 3): raise the clean threshold / cap the front density so near-duplicate
radii never reach LSODA — this removes *both* faces at the source, but it
changes the success-path sampling and so carries the strict byte-identity
caveat.

---

## I.7 Addendum (2026-06): what shipped (guard) + why grid de-refinement is deferred

**Shipped — the Step-3 guard, made shape-aware (Part A).** `find_nearest_higher`
now uses `_is_monotonic_or_tolerable` instead of the strict `monotonic()`. It
refines I.6's plain cumulative-drawdown idea: a trip is tolerated only when it
is **both shallow** (relative drawdown ≤ `MONOTONIC_RTOL = 1e-2`, covering the
7.06e-3 spike and the I.5 dip) **and localized** (confined to the leading
`BOUNDARY_FRAC = 0.01`, or an isolated run ≤ `MAX_SPIKE_LEN = 2`). A
*sustained interior* inversion (a possible real local `T` maximum) or a
dead-integrator zero/non-finite tail (~100 % drawdown) is therefore still
**rejected** → `get_betadelta` penalises that `(β,δ)` `(100,100)`. This
distinguishes numerical noise from a genuinely bad profile by *shape*, not
magnitude alone. `monotonic()` is left strict, so `find_nearest_lower` and the
`_get_velocity_residuals` check (the `return 1e2` penalty) are byte-identical.
The direction test also moves to `array[-1] >= array[0]` (the all-pairs
`kindof_increasing()` returned the wrong direction once a spike was tolerated).

**Correction to I.6 — "raise the clean threshold" is NOT a valid
de-refinement (measured).** `_clean_radius_grid` compares each point to its
*immediate predecessor*, not the last *kept* point. The over-dense band is
uniformly ~1.3e-9 spaced, so **any** threshold above that deletes the **whole**
band at once: `clean@{5e-9, 1e-8, 1e-7}` all leave **1 point** in the
conduction sliver (total 59 992 → 39 993). That collapses the steep
`T 3e4→1.6e5` conduction zone to a single point → wrong cooling (the exact
failure mode the conduction-zone grid comment at `bubble_luminosity.py:400-407`
warns about). Raising `MIN_SPACING` is off the table.

**Viable de-refinement (deferred).** Reduce the Step-2 point count
(`_create_legacy_radius_grid:796`, `int(2e4)` — the middle of three `int(2e4)`
chunks). Step 2 is largely **redundant** with Step 1: Step 1's reflected
logspace already samples the outer edge at rel-spacing ~`1.6e-7`; Step 2 only
stacks the extra `1.3e-9` near-duplicate layer that is the LSODA stressor.
Measured (build grid, count outer-band density):

| Step-2 N | total pts | min rel-spacing | LSODA stress? | conduction pts |
|---------:|----------:|----------------:|:-------------:|---------------:|
| 20 000 (now) | 59 992 | `1.3e-9` | **yes** | ~20 650 |
| 500 | 40 492 | `5.2e-8` | no | ~650 |
| 200 | 40 192 | `1.3e-7` | no | ~350 |

`N≈500` lifts the spacing out of the stress zone while Step 1 keeps the
conduction zone well above the ≥100-point floor (the `~100+ points` requirement
noted at `bubble_luminosity.py:407`). **But** it changes the
conduction-zone integration resolution for *every* run, so it shifts
`L2Conduction`/downstream outputs by an amount that must be measured
(output-diff) and blessed by the model author — a physics-accuracy choice, not
a guessed constant. **Decision (2026-06, user):** ship the shape-aware guard
alone; treat grid de-refinement as a separate, validated change.

---

# Part II — Phased fix

Each step is independently committable and independently revertable. **Do
not** combine the cheap determinism fix with the physics-guard fix in one
commit — they have different risk profiles.

## Step 0 — Reproduce on demand (prerequisite)

Build a stress harness (a throwaway script, not committed, or a
`-m pytest -k stress` opt-in) that runs the failing scenario **N ≥ 30**
times and counts `MonotonicError`. Establish the baseline rate and, if
possible, a **seed/thread setting that fails reliably** — without a
reproducer there is nothing to verify against.

Capture the failing case with `full_output=True`: record `infodict['ier']`,
`infodict['message']`, the `bubble_dMdt` value, and the
`initial_conditions` that triggered it. This confirms the mechanism and
tells us whether the failure is "LSODA gave up" vs "NaN already in the
initial conditions."

**Step 0 realized (2026-06): the `TRINITY_BUBBLE_DIAG` capture.** A gated,
observational diagnostic is now wired at the `:430` `odeint` site
(`bubble_luminosity._capture_bubble_integration`, called at `:434`). With
`TRINITY_BUBBLE_DIAG=1`, every problematic T-profile (non-finite,
non-monotonic, or sub-floor tail) is saved to
`<path2output>/bubble_diag/event_*.npz` (the `r/v/T/dTdr` arrays plus
`ier`, `message`, `bubble_dMdt`, `initial_conditions`, β, δ, R2, Eb, t) and a
one-line **mode classification** is logged: `dead_integrator` (I.1) |
`boundary_transient` (I.5) | `bulk_nonmonotonic`. The capture only saves files
and logs — it never mutates `T_array` or control flow, so the success path is
byte-identical. *(Originally the diagnostic gated `full_output`; since #659
`_odeint_checked` always passes `full_output=True`, which is byte-identical on
success — it only adds the infodict.)* This is the disambiguator between the I.1
and I.5 modes on real runs — run it before choosing a fix. (Capture is at the
`:430` site that feeds the `:481` guard; the other odeint sites — `:546`
conduction, `:1008` velocity-residual — are not instrumented.)

## Step 1 — Determinism (cheapest; possibly sufficient)

> **Superseded (2026-06, #659).** This step assumed the flake was BLAS thread
> FP noise. It is not: #659 reproduced the crash **single-threaded with a fixed
> `PYTHONHASHSEED`** (an OK/crash coin-flip), proving the non-determinism comes
> from `odeint` consuming an **uninitialised** failure tail, not thread FP
> order. Determinism was won by *detecting* the failed solve (`_odeint_checked`),
> not by thread-pinning. NB: `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` *is* set
> elsewhere — `sweep_runner.py:276` and the SLURM template `sweep_jobs.py` — but
> for an unrelated reason (avoid CPU oversubscription across parallel sweep
> workers), not as this fix, and not in `run.py`/`conftest`.

Pin the BLAS/OpenMP thread counts to 1 (`OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) for the run —
set in `run.py`'s entry path (before numpy import) and in `conftest.py` for
the test suite.

- **If** the diagnostic shows pinned threads → **0 failures**, this removes
  the *nondeterministic* flake at zero physics risk: every run becomes
  reproducible, and CI stops flaking. It does **not** fix the underlying
  stiffness (a genuinely bad input would still fail — deterministically),
  so it is necessary-but-maybe-not-sufficient and Steps 2-3 still land.
- **If** pinning does **not** eliminate failures, the trigger is not just
  thread FP noise; skip straight to Steps 2-3.

This step's value: it makes Steps 2-3 *testable* (deterministic repro) and
de-risks CI immediately.

## Step 2 — Make the failure loud and recoverable

> **Partly shipped (2026-06, #659).** The *detect / fail-loud* half landed:
> `_odeint_checked` (`:71`) runs `odeint` with `full_output=True` and reports
> LSODA's status; on failure `_bubble_luminosity_legacy` raises
> `BubbleSolverError` (`:440`/`:553`) and `_get_velocity_residuals` returns the
> deterministic `_SOLVER_FAIL_RESIDUAL = 1e3` penalty (`:1018`) so `fsolve` is
> steered off the infeasible `dMdt` instead of converging on garbage. **The
> design diverged from items 3–4 below:** there is **no** `solve_ivp` recovery
> and **no** terminal-event integration yet (that scaffolding is still unused —
> §I.4); a failed structure solve raises rather than recovers. The `sys.exit` in
> `_get_bubble_ODE` (item 4) is **also not yet fixed** — still at `:1069`.

In `get_bubbleproperties_pure`, the original plan was to replace the bare
`odeint` (then L172, now `:430`) with a checked call:

1. `psoln, infodict = odeint(..., full_output=True)`.
2. Detect failure: `infodict['ier'] != 2`, or any non-finite / zero-tail in
   `psoln[:, 1]` (T) below a physical floor.
3. On failure, **recover** via the existing `solve_ivp` path using a stiff
   method (`LSODA`/`Radau`/`BDF`) with a **terminal event** that stops the
   integration cleanly before `T` reaches the floor, instead of letting it
   collapse to zero.
4. If recovery also fails, raise a **descriptive** domain error
   (`BubbleIntegrationError` with `bubble_dMdt`, `Pb`, `t_now`, `ier`,
   message) — never a bare `MonotonicError`, and never `sys.exit` from a
   library function (fix the `:1069` `sys.exit` too).

Net effect: the cryptic far-away symptom becomes an actionable,
near-the-cause error, and the common case self-heals.

## Step 3 — Guard the RHS / domain (root-cause hardening)

- Add a **temperature floor** in `_get_bubble_ODE` so the RHS stays finite
  (clamp `T` to a small positive physical value when evaluating the
  stiff terms), turning a blow-up into a bounded derivative the solver can
  step through — rather than `sys.exit`.
- Audit `_create_legacy_radius_grid` (`:771`) so the grid does not over-reach
  past the physical conduction zone (the existing comment at `:400-407` warns
  the adaptive grid under-sampled this zone — keep the constraint in mind).

Step 3 is the deepest and the only one touching physics expressions; do it
last, behind the Step 2 net, and only if Steps 1-2 don't fully close it.

---

# Verification battery (the strict bar)

1. **Stress (the headline test)**: the Step-0 harness, post-fix, must show
   **0 `MonotonicError` over N ≥ 50 runs**, across both `{pinned=1}` and
   `{default}` thread settings. Add it as an opt-in
   `pytest -m stress` test so CI can run it nightly without slowing the
   default suite.
2. **Regression pin**: a fast deterministic test that runs the
   previously-flaky scenario (the smoke param, or a direct
   `get_bubbleproperties_pure` call with the captured triggering state) and
   asserts it completes without `MonotonicError`.
3. **Success-path is byte-identical** (the non-negotiable): pick a seed/run
   that *already passes*, capture `dictionary.jsonl` + `metadata.json`
   before the fix, and assert they are **byte-for-byte unchanged** after.
   The fix may only alter the failing branch. (Step 1 thread-pinning can
   shift last-digit FP — verify whether it changes the success-path
   outputs; if it does, that is a real trade-off to surface to the user,
   not a silent change.)
4. Full existing battery still green (`pytest test/ -q`, ruff, smoke).

# Risk & rollback

- **Risk: medium-high** — the only physics-adjacent change in the effort.
  Mitigated by the strict byte-identity bar on the success path and by
  layering (determinism → detect/recover → RHS guard), each its own
  commit.
- **Rollback**: each step is an isolated commit. Step 1 (env vars) and
  Step 2 (checked call + fallback wiring) are trivially revertable. Step 3
  touches RHS expressions — keep it separate so it can be reverted without
  losing the determinism/observability wins.

# Open decisions for the user

- **The fix is mode-dependent — confirm the mode (Step-0 diagnostic) before
  applying any fix.** Mode (b) (zero-tail / dead integrator, I.1) → Steps
  1-3 (make it loud, recover, RHS floor); do **not** mask it. Mode (a)
  (boundary transient, I.5) → a *tolerance* on the monotonic guard
  (cumulative-drawdown ≤ `rtol`; measured envelope ≈1.3e-3 ⇒ `rtol`≈1e-2
  leaves ~8× margin while a real inversion is ≫ that) **plus** the
  `mon_incr` direction fix from I.5 — failing path only, success path
  byte-identical. The two fixes are compatible (a drawdown-tolerant guard
  still rejects the ~100 %-drawdown zero tail), but the tolerance must not
  be applied blind, or it would silence mode (b).
- **Is Step-1 thread-pinning acceptable as a shipped default?** It makes
  runs reproducible but serializes BLAS (a possible perf hit on
  large/parallel sweeps). Alternative: pin only in tests/CI, leave
  production multi-threaded and rely on Steps 2-3.
- **Floor value & event threshold in Steps 2-3** are physics choices — set
  with the model author, not guessed.
