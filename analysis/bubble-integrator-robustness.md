# Bubble-integrator robustness: the flaky `MonotonicError`

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
- **Root trigger**: `fsolve` for `bubble_dMdt` is FP/thread-sensitive;
  small run-to-run perturbations occasionally land the integration in a
  regime where `T → 0`, and the RHS terms `∝ 1/T`, `1/T^{5/2}`,
  `dTdr²/T` blow up → LSODA gives up.
- **Fix posture**: (1) make it *deterministic* (cheap), (2) make the
  failure *loud and recoverable* (`full_output` + a real fallback — the
  codebase already carries an unused `solve_ivp` path), (3) guard the RHS
  so `T` cannot cross the physical floor. Prove with an N-run stress
  harness, not a single pass.

---

# Part I — Audit (what is)

## I.1 Where it breaks

`trinity/bubble_structure/bubble_luminosity.py`, in
`get_bubbleproperties_pure`:

1. **L141** — `scipy.optimize.fsolve(velocity_residuals_wrapper, …,
   xtol=1e-4, factor=50, epsfcn=1e-4)` solves for `bubble_dMdt` from a
   velocity boundary condition. `fsolve` internally does finite-difference
   Jacobians and BLAS-backed linear algebra → output is sensitive to
   floating-point summation order (which varies with BLAS threading).
2. **L172** — `scipy.integrate.odeint(_get_bubble_ODE, initial_conditions,
   r_array, args=(params, Pb), tfirst=True)`. **No `full_output`, no return
   check.** When LSODA fails it prints
   `ODEintWarning: Illegal input detected (internal error)` to stderr and
   returns an array whose unintegrated tail is **zeros**.
3. **L183** — `n_array = Pb / (2 k_B T_array)` then divides by those zeros
   → `RuntimeWarning: divide by zero` (the visible breadcrumb).
4. **L214** — `operations.find_nearest_higher(T_array, _CIEswitch)`.
   `find_nearest_higher` (`trinity/_functions/operations.py:88`) calls
   `monotonic(array)` and **raises `MonotonicError`** because the
   `[…, 0., 0., 0.]` tail is neither increasing nor decreasing.

The exception is raised ~100 lines and one module away from the actual
fault. Nothing logs *"the integrator failed"*; you get a bare
`MonotonicError` with a printed array.

## I.2 Why the RHS is fragile

`_get_bubble_ODE` (`bubble_luminosity.py:785`, Weaver+77 Eq. 42-43):

```python
if np.abs(T - 0) < 1e-5:
    logger.critical('T is zero in bubble ODE'); sys.exit()
ndens = Pb / (2 * k_B * T)
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

`bubble_luminosity.py` already contains an **unused alternative solver
path** built on `scipy.integrate.solve_ivp` (≈ L664-708) plus `try/except`
wrappers and a second `odeint` call site (L279, L734). Prior hands clearly
hit this and started a `solve_ivp` route. The fix should finish/wire that
rather than add a third parallel solver.

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

## Step 1 — Determinism (cheapest; possibly sufficient)

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

In `get_bubbleproperties_pure`, replace the bare `odeint` (L172) with a
checked call:

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
   library function (fix the L789 `sys.exit` too).

Net effect: the cryptic far-away symptom becomes an actionable,
near-the-cause error, and the common case self-heals.

## Step 3 — Guard the RHS / domain (root-cause hardening)

- Add a **temperature floor** in `_get_bubble_ODE` so the RHS stays finite
  (clamp `T` to a small positive physical value when evaluating the
  stiff terms), turning a blow-up into a bounded derivative the solver can
  step through — rather than `sys.exit`.
- Audit `_create_legacy_radius_grid` (L497) so the grid does not over-reach
  past the physical conduction zone (the existing comment at L160-167 warns
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

- **Is Step-1 thread-pinning acceptable as a shipped default?** It makes
  runs reproducible but serializes BLAS (a possible perf hit on
  large/parallel sweeps). Alternative: pin only in tests/CI, leave
  production multi-threaded and rely on Steps 2-3.
- **Floor value & event threshold in Steps 2-3** are physics choices — set
  with the model author, not guessed.
