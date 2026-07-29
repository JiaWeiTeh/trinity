# Sweep: silent failure and swallowed physics

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
> a committed artifact under `docs/dev/` (a CSV/table in `docs/dev/data/`, or a
> harness/figure in the relevant `docs/dev/<workstream>/` folder) — never left in
> `/tmp`, the local-only `scratch/`, or an untracked `outputs/`. A future visit must be able to reproduce or compare
> against the numbers **without re-running**; record the exact config + command
> that produced each artifact.
>
> 🔗 **Cross-check the sibling docs — keep the workstream self-consistent.** This file is one of
> several living docs for its workstream (its `PLAN.md`, `FINDINGS.md`, `runs/README.md`, `NOTE_PATCHES.md`,
> and any other notes in the same folder). They drift out of sync *with each other* as fast as they drift
> from the code. Any agent or person editing one MUST, as part of the visit, circle back through the
> siblings and reconcile: if a number, status, claim, or line reference here contradicts a sibling — or a
> sibling has gone stale — fix it (or flag it, dated) so no two docs in the workstream disagree. Never
> update one in isolation.

**Status (2026-07-29):** 📘 raw agent report — provenance for `FINDINGS.md`; unreconciled and unverified on its own.

Read-only audit of `/home/user/trinity/trinity/**`, worked against
`docs/dev/code-audit/data/claims_guards.csv` (131 `except` sites, 213 clamp sites). Every claim
below was verified by reading the current source; the CSV's `kind` column was used only as a
worklist, not as evidence. Line numbers are from the tree as of this session.

Two facts frame everything that follows and are established once here:

* **`logging.captureWarnings` is never enabled** (`grep -rn captureWarnings trinity/ run.py` → no
  hits). Anything that surfaces only as a Python `warnings.warn` — `scipy`'s `fsolve`
  non-convergence `RuntimeWarning`, `odeint`'s `ODEintWarning` — goes to raw stderr, is shown
  **once per source location** under the default filter, and never reaches the run's `.log` file,
  `metadata.json`, or `dictionary.jsonl`. For the purposes of "is it recorded?", those count as
  **not recorded**.
* **`main.py` discards every phase runner's return value.** `run_phase_energy(params)`,
  `run_phase_transition(params)`, `run_phase_momentum(params)` are all called for side effects
  (`trinity/main.py:286,306,346`). The `termination_reason` string — the only place a solver
  failure inside 1b/1c/2 is recorded — is dropped on the floor.

---

### SF-001 · `scipy.integrate.odeint` drives the shell structure with `full_output` ignored; a failed solve returns *uninitialised memory* that is then read as a photon-depleted ionisation front
- **file:line** — `trinity/shell_structure/shell_structure.py:165` (ionised region) and
  `trinity/shell_structure/shell_structure.py:324` (neutral region)
  ```python
  sol_ODE = scipy.integrate.odeint(
      get_shellODE.get_shellODE, y0, rShell_arr,
      args=(f_cover, is_ionised, params), mxstep=_SHELL_ODE_MXSTEP
  )
  nShell_arr = sol_ODE[:, 0]
  phiShell_arr = sol_ODE[:, 1]
  ```
- **class** — unchecked-solver-status
- **severity** — **S1 results-wrong**
- **triggering condition** — LSODA exhausts `mxstep` (`_SHELL_ODE_MXSTEP = 50000`) before reaching
  the end of the slice. The module docstring at `shell_structure.py:28-35` states this **already
  happened on a shipped example config** ("odeint's default internal step ceiling (mxstep=500) is
  exhausted in the degenerate code-unit-overflow regime (simple_cluster) … and silently truncates
  the shell integration"). Raising the ceiling to 50 000 made it rarer; nothing detects it. The
  ionised RHS has a genuine finite-radius pole (`dn/dr ∝ +n²`, see `get_shellODE.py:19-32`), so
  step exhaustion past the front is the *expected* behaviour, not an exotic edge.
- **substituted value** — scipy fills the output array in place and returns it regardless of
  `istate < 0`. I verified empirically on the installed scipy 1.17.1 that the un-integrated tail is
  **uninitialised heap memory**, not zeros:
  ```
  [1.00e+000 1.357e+000 2.111e+000 4.750e+000 4.204e+001 6.235e-311 1.070e-296
   6.234e-311 2.976e-311 2.798e-282 3.757e-317 0.000e+000 ... 4.135e+122 1.741e-315]
  ```
  So `nShell_arr` and `phiShell_arr` acquire a garbage tail. scipy emits an `ODEintWarning`
  ("Excess work done on this call") — to stderr, once, uncaptured.
- **downstream fate** — reaches solver state and run output. The garbage phi tail is almost always
  ≤ 1e-9, so `phiCondition = phiShell_arr <= 1e-9` (`shell_structure.py:182`) fires at the
  truncation index; `idx` is set there, `is_phiDepleted = True`, and the ionised loop exits. Every
  downstream shell quantity is then computed from a profile truncated at a *numerical* boundary
  rather than the physical ionisation front: `n_IF` / `R_IF` (lines 224-226), `f_esc_ion`
  (line 229), `n_IF_Str` (lines 242-253), `shellThickness`, `nShell_max`, `tau_kappa_IR`
  (lines 384-395), `f_absorbed*` (398-400). Those flow into `P_HII` → `P_drive` → the momentum /
  energy ODE RHS, into `F_rad` via `shell_tauKappaRatio`, into the dissolution test
  (`shell_nMax < nISM`), and into `dictionary.jsonl` as `shell_*`. Nothing distinguishes them.
  Because the substituted values come from uninitialised memory, **the run is not reproducible**:
  two identical invocations can produce different shell structures.
- **recorded?** — **no**. No `full_output=1`, no `infodict['message']` check, no log line, no flag,
  no output marker.
- **failure scenario** — `param/simple_cluster.param` in the regime the docstring names, or any
  config whose ionised shell is optically thick enough that the recombination pole is reached
  inside one `sliceSize`. The run completes, reports a clean `SimulationEndCode`, and publishes a
  shell density profile, ionisation-front radius, and `P_HII` history built on garbage. Because
  `is_phiDepleted` also gates whether the neutral region is integrated at all
  (`has_neutral`, line 221), a truncation can additionally erase the entire neutral shell.
- **confidence** — **high** (source read; scipy tail behaviour verified experimentally;
  the repo's own `bubble_luminosity.py:67-79` docstring documents the identical hazard as the
  reason the *bubble* path was migrated off `odeint`).

---

### SF-002 · The bubble `dMdt` root-find never checks convergence; a non-converged `fsolve` returns the seed and the whole bubble structure is built on it
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:261-267`
  ```python
  bubble_dMdt = scipy.optimize.fsolve(
          velocity_residuals_wrapper,
          bubble_dMdt,
          xtol=1e-4,
          factor=50,
          epsfcn=1e-4
      )[0]
  ```
- **class** — unchecked-solver-status
- **severity** — **S1 results-wrong**
- **triggering condition** — `fsolve` returns `ier ∈ {2,3,4,5}` (max function evaluations, xtol too
  small, no progress / singular Jacobian). This is *engineered to happen*: whenever the trial
  `dMdt` gives non-finite ICs, a raising RHS, or `sol.success == False`,
  `_get_velocity_residuals` returns the **constant** `_SOLVER_FAIL_RESIDUAL = 1e3`
  (`bubble_luminosity.py:334, 359, 361, 363`). A constant residual has an identically-zero
  finite-difference Jacobian, so MINPACK terminates with `ier = 4` or `5` on the very first
  iteration. The `min_T` and monotonic penalties (lines 371-382) are similarly plateau-shaped.
- **substituted value** — `res['x']`, i.e. the initial guess essentially unchanged: either the
  previous segment's accepted `bubble_dMdt` carried in `params`, or the Weaver Eq. 33 estimate from
  `_get_init_dMdt` on the first call.
- **downstream fate** — straight into solver state and output. `bubble_dMdt` sets `dR2`, `r2Prime`
  and the full ODE initial conditions (`_get_bubble_ODE_initial_conditions`, lines 392-411), hence
  the entire temperature/velocity/density structure, `bubble_LTotal`, `bubble_Tavg`,
  `bubble_T_r_Tb`, `bubble_mass`. `bubble_LTotal` is the `Lcool` in the beta-delta residual and in
  the `Lloss/Lgain` energy→momentum transition trigger; `bubble_T_r_Tb` becomes `T0`, an ODE state
  variable. All of it is written to `dictionary.jsonl`. A non-converged `dMdt` is byte-for-byte
  indistinguishable from a converged one in the output.
- **recorded?** — **no**. `full_output` is not requested, so `ier` is unavailable to the code.
  scipy's fallback is `warnings.warn(msg, RuntimeWarning)` (confirmed in the installed
  `scipy/optimize/_minpack_py.py`), which lands on stderr once per location and never in the run
  record. There is no `bubble_dMdt_converged` key in `registry.py`.
- **failure scenario** — the stiff / low-`Lmech` edge where the structure solve fails for a band of
  trial `dMdt` (the very regime `docs/dev/performance/f1edge_hidens*.param` exists to probe). The
  fsolve sees a flat 1e3 plateau, gives up immediately, and the run integrates a bubble whose mass
  flux was never solved — producing a smooth, plausible, and wrong `bubble_LTotal(t)` for the rest
  of the phase. It fires *every segment* once the trajectory enters that band, which is exactly the
  "systematic, not transient" case.
- **confidence** — **high**

---

### SF-003 · `get_residual_pure` swallows every exception from the bubble solve into a fixed (100, 100) plateau; the caller then re-snapshots the previous segment's bubble physics at the new timestamp
- **file:line** — `trinity/phase1b_energy_implicit/get_betadelta.py:435-439`
  ```python
  try:
      bubble_props = get_bubbleproperties_pure(params_view)
  except Exception as e:
      logger.warning(f"Bubble properties calculation failed: {_describe_exc(e)}")
      return 100.0, 100.0, None
  ```
  consumed at `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:893`
  (`if bubble_props is not None: updateDict(params, bubble_props)`) — with no `else`.
- **class** — swallowed-exception → unrecorded-substitution
- **severity** — **S1 results-wrong**
- **triggering condition** — anything at all raised inside the bubble chain: `BubbleSolverError`
  (failed LSODA, T→0 collapse, negative T), `MonotonicError` from `find_nearest_higher`,
  `ValueError` from `brentq` failing to bracket `r_CIEswitch` (`bubble_luminosity.py:724`),
  `ValueError` from the CIE `interp1d` going out of its tabulated `logT` range
  (`main.py:167`, built with default `bounds_error=True`), `KeyError`, `IndexError`,
  `ZeroDivisionError`, an `assert` failure. The handler cannot tell them apart.
- **substituted value** — `(Edot_residual, T_residual, bubble_props) = (100.0, 100.0, None)`
  → `total_residual = 20000.0`.
- **downstream fate** — two distinct paths, both bad:
  1. **`bubble_props = None` ⇒ stale bubble state is re-published.** `updateDict` is skipped, so
     `bubble_LTotal`, `bubble_dMdt`, `bubble_mass`, `bubble_Tavg`, `bubble_T_r_Tb`, and all four
     profile arrays keep the **previous segment's** values. `params.save_snapshot()` at
     `run_energy_implicit_phase.py:1017` then writes them into `dictionary.jsonl` stamped with the
     **new** `t_now` and `R2`. A consumer reading that file sees a bubble that was solved at this
     time. It was not. Held for as long as the failure persists — up to
     `NO_ROOT_HANDOFF_STREAK = 50` segments on the hybr path, and **indefinitely** on the `legacy`
     path, which never sets `no_physical_root` at all (`get_betadelta.py:170-175`).
  2. **The (100, 100) plateau poisons the search.** In `_solve_grid` and `_solve_lbfgsb` every
     failing point scores the identical 20000, so the optimiser cannot descend; the "best
     candidate" sort at `get_betadelta.py:822` then returns the input guess. `beta`/`delta` are
     frozen and fed to `cool_beta_to_Ebdot_pure` → `Ed` → the ODE. That is frozen *physics*
     driving a live integration.
- **recorded?** — **partially**. Each failure logs at WARNING (`"Bubble properties calculation
  failed: …"`), and `params['betadelta_converged']` (registry default `False`, not
  snapshot-excluded) is written per snapshot, so `converged=False` does reach `dictionary.jsonl`.
  What is **not** recorded is that the bubble columns in that row are *stale rather than
  recomputed* — there is no key that distinguishes "solved and didn't converge" from "not solved,
  values carried forward". `betadelta_total_residual` will read `inf`/20000, which is a hint, but
  the bubble columns themselves look ordinary.
- **failure scenario** — a massive/dense cloud whose bubble drifts out of the CIE cooling table's
  `logT` range. `interp1d` raises `ValueError` every evaluation → every grid point plateaus →
  `beta`, `delta`, and the whole bubble block freeze while `R2`, `v2`, `Eb`, `t` keep advancing.
  With `betadelta_solver='legacy'` there is no streak handoff, so the phase grinds to
  `MAX_SEGMENTS = 5000` publishing 5000 rows of frozen bubble physics against a moving shell.
- **confidence** — **high**

---

### SF-004 · A failed `solve_ivp` in phases 1b / 1c / 2 sets only a local string; `main.py` drops it, the run continues into later phases from the failed state, and can finish with a *clean* exit code
- **file:line** —
  `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1080-1090`,
  `trinity/phase1c_transition/run_transition_phase.py:641-648`,
  `trinity/phase2_momentum/run_momentum_phase.py:723-729`
  ```python
  # run_transition_phase.py:646  (no log at all)
  if not sol.success or len(sol.t) == 0:
      termination_reason = f"solver_failed: {sol.message}"
      break
  ```
- **class** — unchecked-solver-status (returned failure signal ignored by every call site)
- **severity** — **S1 results-wrong**
- **triggering condition** — LSODA returns `success == False` (step size underflow below
  `min_step = 1e-6`, too many steps, RHS non-finite), or `solve_ivp` raises.
- **substituted value** — none numerically; the *control-flow* substitution is the finding. The
  loop breaks with `EndSimulationDirectly` still `False` and `SimulationEndCode` untouched.
- **downstream fate** — `main.py:286/306/346` ignore the returned `ImplicitPhaseResults` /
  `TransitionPhaseResults` / `MomentumPhaseResults`, and gate the next phase only on
  `params['EndSimulationDirectly'].value == False` (`main.py:283, 303, 343`). So a 1b solver
  failure is followed by a full transition phase and a full momentum phase, launched from the last
  partially-integrated state. Whichever later phase ends normally then writes a **clean**
  `SimulationEndCode` (`STOPPING_TIME=1`, `LARGE_RADIUS=2`, `SHELL_DISSOLVED=0` — all in the
  `is_clean()` 0-9 band per `simulation_end.py:109-111`). The run is published as a clean physical
  outcome. Note `SimulationEndCode.ERROR_SOLVER = (22, "error_solver")` exists in the enum and is
  **never assigned anywhere in the package** (`grep -rn ERROR_SOLVER trinity/` → definition only).
- **recorded?** — **no** for the transition and momentum phases (1c logs nothing at the
  `not sol.success` branch; phase 2 logs nothing either — it only surfaces later as
  `logger.info("Momentum phase completed: solver_failed: …")`). Phase 1b does
  `logger.error(...)`, which reaches the `.log` file but not `metadata.json` or
  `dictionary.jsonl`.
- **failure scenario** — a high-mass cloud whose transition-phase `Eb` decay stiffens LSODA past
  `min_step`. 1c breaks silently at t = 4 Myr; phase 2 runs 0.5–15 Myr on top of the truncated
  state, hits `stop_t`, and `metadata.json[termination]` reports
  `{"exit_code": 1, "outcome": "stopping_time"}`. Nothing in the published run says a solver
  failed.
- **confidence** — **high**

---

### SF-005 · The momentum-phase ODE RHS clamps `R2` and `mShell` to `1e-10`, fabricating an enormous outward acceleration during a collapse through zero
- **file:line** — `trinity/phase2_momentum/run_momentum_phase.py:398` and `:415`
  ```python
  R2, v2 = y
  R2 = max(R2, 1e-10)
  ...
  mShell = max(mShell, 1e-10)
  ```
- **class** — clamped-physics
- **severity** — **S1 results-wrong**
- **triggering condition** — the integrator proposes `R2 ≤ 0` (a collapsing shell overshooting the
  origin) or `mShell ≤ 0`. The `min_radius` event
  (`min_r = max(coll_r*1.5, 0.01)`, `phase_events.py:445`) is terminal and should normally catch
  this first, but events are located only *after* a step is accepted, and LSODA evaluates the RHS
  at trial points well past the event root. So the clamped RHS is exercised on the very steps that
  determine whether the event is bracketed at all.
- **substituted value** — `R2 = 1e-10 pc`, `mShell = 1e-10 Msun`.
- **downstream fate** — reaches solver state. At `R2 = 1e-10`,
  `P_ram = Lmech/(2π R2² v_mech)` (`get_bubbleParams.py:308`) is inflated by ~10²⁰ relative to a
  parsec-scale radius, and `F_pressure = 4π R2² (P_drive - P_ext)` uses the *clamped* `R2²` in the
  prefactor too — but `F_grav = G·mShell/R2²·(...)` uses the clamped `R2` as well, so the balance
  is not the physical one at any radius. `vd = (F_pressure - …)/mShell` is then divided by a
  clamped mass. The resulting derivative is fed back to LSODA, which integrates `R2` back outward:
  a **purely numerical bounce** that looks like a physical re-expansion. `R2`, `v2` are the state
  vector and go directly to `dictionary.jsonl`.
- **recorded?** — **no**. No log, no flag, no comment in the source, no test.
- **failure scenario** — a gravitationally-bound recollapse (`isCollapse=True`, `v2 < 0`) in a
  massive cloud with `coll_r` set small. The shell drives through `R2 = 0` between two accepted
  steps; the clamped RHS launches it back out at 10²⁰× the ram pressure; the run then reports a
  re-expansion and a second-generation-like trajectory that is entirely an artefact.
- **confidence** — **medium-high** (the clamp is unambiguous; how often the integrator reaches
  `R2 ≤ 0` before the terminal event fires depends on step control, which I could not exercise
  read-only).

---

### SF-006 · `except Exception: P_ext = 0.0` inside the momentum ODE RHS deletes the inward ionised-gas pressure from the force balance
- **file:line** — `trinity/phase2_momentum/run_momentum_phase.py:425-432`
  ```python
  if FABSi < 1.0:
      try:
          n_r = density_profile.get_density_profile(np.array([rShell]), params)
          ...
          P_ext = (params['mu_convert'].value / params['mu_ion_shell'].value) * n_r * k_B * snapshot.TShell_ion
      except Exception:
          P_ext = 0.0
  ```
- **class** — swallowed-exception → clamped-physics
- **severity** — **S2 latent** (S1 if it fires)
- **triggering condition** — any raise from `get_density_profile`: an unrecognised
  `dens_profile` (`density_profile.py:167` raises `ValueError`), a missing `densBE_f_rho_rhoc`
  or a `KeyError`/`TypeError` inside `be_r2xi`, a `KeyError` on `mu_ion_shell`. Note what does
  **not** raise: a NaN `rShell` propagates as NaN through `tanh` and returns NaN, so the loud
  failure mode is not the one caught. Every condition that *does* raise here is a configuration or
  state property, not a transient — so if it fires at all, it fires on **every RHS evaluation of
  the phase**.
- **substituted value** — `P_ext = 0.0` (the ISM term `PISM * k_B` is still added afterward when
  `rShell >= rCloud`).
- **downstream fate** — reaches solver state directly:
  `F_pressure = 4π R2² (P_drive - P_ext)`; `vd = (F_pressure - mShell_dot·v2 - F_grav + F_rad)/mShell`.
  Dropping the confining pressure makes the shell accelerate outward faster than it should, for the
  whole phase. The resulting `R2(t)`, `v2(t)` are the published trajectory.
- **recorded?** — **no**. Bare `except Exception:` with no `as e`, no log, no flag.
- **failure scenario** — a `densBE` run where the Bonnor-Ebert interpolators are not materialised
  in the momentum phase (they are `runtime_loaded` params). Every RHS call raises, `P_ext ≡ 0`, and
  the momentum-phase expansion is systematically over-driven — a smooth, entirely plausible
  `R2(t)` curve that is wrong by the size of the ionised-gas confinement.
- **confidence** — **medium** (the mechanism is certain; whether any shipped config reaches a
  raising branch I could not determine without running).

---

### SF-007 · The same `except Exception: P_ext = 0.0` in three diagnostic force-budget paths silently zeroes `F_ion_in` / `press_HII_in` in the run output
- **file:line** — `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:509-515`,
  `trinity/phase1c_transition/run_transition_phase.py:301-307`,
  `trinity/phase2_momentum/run_momentum_phase.py:241-247`
- **class** — swallowed-exception → unrecorded-substitution
- **severity** — **S3 misleading**
- **triggering condition** — as SF-006.
- **substituted value** — `P_ext = 0.0`.
- **downstream fate** — output only. These `compute_forces_pure` variants feed
  `params['F_ion_in']` and `params['press_HII_in']`, which are snapshot keys. The energy/implicit
  ODE recomputes `P_ext` independently via `get_press_ion` (`energy_phase_ODEs.py:238, 375`), which
  has **no** try/except and would raise loudly — so the trajectory is unaffected. What is affected
  is the *published force budget*: a paper figure of F_grav / F_ram / F_ion_in / F_rad would show
  a zero inward-ionisation term with no indication it was a swallowed error.
- **recorded?** — **no**.
- **failure scenario** — any run where the diagnostic path raises but the ODE path does not (they
  differ: the diagnostic passes `np.array([rShell])`, the ODE passes it through
  `np.atleast_1d`). The force-budget plot silently loses a term.
- **confidence** — high (that it is output-only), medium (on how often it fires).

---

### SF-008 · Non-CIE cooling cubes are seeded with `NaN` for absent (n, T, φ) triplets; a `NaN` cooling rate is never checked anywhere on the path into the bubble ODE
- **file:line** — `trinity/cooling/non_CIE/read_cloudy.py:226-237` (and `244-254` for heating)
  ```python
  cool_cube = np.empty((len(log_ndens_arr), len(log_temp_arr), len(log_phi_arr)))
  cool_cube[:] = np.nan
  for (ndens_val, temp_val, phi_val, cooling_val) in cool_table:
      ...
      cool_cube[ndens_index, temp_index, phi_index] = cooling_val
  ```
  with the acknowledgement still in the source at line 256-259:
  `# Future TODO: If it fails, i.e., if it returns NaN because the values don't exist in the
  cooling table, we do further operations.`
- **class** — unrecorded-substitution
- **severity** — **S2 latent**
- **triggering condition** — a query whose linear-interpolation stencil touches an unfilled cube
  node. The docstring at `read_cloudy.py:162-165` states plainly that "Some are NaN, because they
  are not available in the cooling table". `RegularGridInterpolator` is constructed with defaults
  (`bounds_error=True`), so *outside* the hull it raises loudly — but *inside* the hull a NaN
  neighbour silently produces NaN.
- **substituted value** — `NaN` cooling / heating / net-cooling rate.
- **downstream fate** — `netcool_interp(...)` → `get_dudt` returns `NaN`
  (`net_coolingcurve.py:154`) → the bubble-structure RHS `dTdrr` is NaN
  (`bubble_luminosity.py:441`) → LSODA either fails (caught, converted to the SF-002 penalty
  plateau) or returns a NaN-tailed profile. `np.any(T_array < 0)` at
  `bubble_luminosity.py:668` is **False for NaN**, so the negative-temperature net does not catch
  it. `L_bubble`/`L_conduction`/`L_intermediate` become NaN, `bubble_LTotal` becomes NaN and is
  written to `dictionary.jsonl`. The energy→momentum trigger
  `(Lgain - Lloss)/Lgain < threshold` (`run_energy_implicit_phase.py:1296`) is **False** for NaN,
  so the phase transition simply never fires — the run continues to `MAX_SEGMENTS`.
  Separately, the direct conduction/intermediate table calls at `bubble_luminosity.py:784-789` and
  `823-828` bypass `get_dudt` and inherit the same NaN with no check.
- **recorded?** — **no** during the run. Partial mitigation at the very end:
  `write_termination_debug_report` builds a NaN/Inf inventory (`simulation_end.py:653-673`) — but
  only for the **last two** snapshots, so a NaN episode in the middle of a run leaves no trace.
- **failure scenario** — a bubble whose (n, T, φ) trajectory crosses a sparse corner of the CLOUDY
  grid (low φ at high n, typical of a deeply-embedded massive cluster). `bubble_LTotal` goes NaN,
  the cooling-balance trigger becomes permanently un-satisfiable, and the run reports
  `max_segments` after publishing thousands of NaN cooling rows.
- **confidence** — **medium-high** (mechanism certain from the source and the author's own TODO;
  whether the bundled `Z1.00` tables have interior holes I did not verify against the data files).

---

### SF-009 · `get_dudt` floors a sub-table temperature up to the cooling file's edge inside the innermost bubble-ODE loop, with no record when it binds
- **file:line** — `trinity/cooling/net_coolingcurve.py:130-131`
  ```python
  if np.log10(T) < nonCIE_Tmin:
      T = 10**nonCIE_Tmin
  ```
- **class** — clamped-physics
- **severity** — **S2 latent**
- **triggering condition** — the bubble-structure integrator sends `T` below the lowest tabulated
  temperature. The comment claims it is "inert on every profiled regime: the bubble ODE never sends
  T below the 3e4 boundary" — but that is exactly the *un-diverged* case; a stiff or diverging
  solve is precisely when `T` dips, and the RHS only raises `BubbleSolverError` once `|T| < 1e-5`
  (`bubble_luminosity.py:418`). The whole interval `(1e-5, 10**nonCIE_Tmin)` is silently floored.
- **substituted value** — `T = 10**nonCIE_Tmin` (the table edge; `nonCIE_Tmin = min(cube.temp)`).
- **downstream fate** — reaches solver state. The floored `T` sets `dudt`, which enters `dTdrr` in
  the bubble ODE (`bubble_luminosity.py:430-443`) and therefore the whole integrated structure;
  the resulting `bubble_LTotal` and `bubble_T_arr` are written to `dictionary.jsonl`. The clamp
  runs thousands of times per run in the hot loop; nothing counts how many.
- **recorded?** — **no**. This is a deliberate, tested behaviour (`test/test_net_coolingcurve.py`
  pins the clamp bit-for-bit) — but the tests pin *that it clamps*, not that it is observable when
  it does.
- **failure scenario** — the low-`Lmech` / high-density edge where the conduction front is thin and
  LSODA undershoots at the cold boundary. The cooling rate is evaluated at the wrong temperature
  every RHS call, biasing the whole structure toward the table-edge cooling rate, and the
  resulting `bubble_LTotal` is smooth, finite, and wrong.
- **confidence** — **medium** (the clamp is certain; whether production trajectories actually dip
  below the table edge is precisely what is unrecorded and therefore unknowable from the run
  record).

---

### SF-010 · `find_nearest_higher` silently clamps an out-of-range index to the array end, collapsing the CIE bubble region to a single point and zeroing its cooling luminosity
- **file:line** — `trinity/_functions/operations.py:179-182`
  ```python
  if idx >= len(array):
      idx = len(array) - 1
  if idx < 0:
      idx = 0
  ```
  with the source's own comment at 175-178: *"Not quite sure what to do with that for now, but
  this part of the code shouldnt need to run anyway."*
- **class** — clamped-physics
- **severity** — **S2 latent**
- **triggering condition** — `find_nearest_higher(T_array, 10**5.5)` when the bubble never gets
  hotter than 10^5.5 K. `T_array` is increasing inward from the 3e4 K boundary; a cool bubble
  (weak wind, heavy cooling, late time) tops out below the CIE switch. Then
  `idx = argmin|T - 10^5.5| = len-1`, `array[idx] - value < 0`, `mon_incr` is True, `idx += 1`,
  and the clamp brings it back to `len-1`.
- **substituted value** — `index_CIE_switch = len(T_array) - 1`.
- **downstream fate** — reaches output and the transition trigger.
  `T_bubble = T_array[index_CIE_switch:]` (`bubble_luminosity.py:737`) is a **one-element** slice,
  so `L_bubble = |trapezoid(1 point)| = 0.0` and `Tavg_bubble = 0.0`. Because
  `index_cooling_switch` clamps the same way, `index_cooling_switch == index_CIE_switch` is likely,
  which also skips the entire conduction-zone branch (line 757) — leaving `L_total = L_intermediate`
  alone. `bubble_LTotal` is then a large underestimate, and it drives `Lloss` in the
  energy→momentum cooling-balance trigger. `Tavg` (line 860-870) is likewise computed from a
  degenerate volume.
- **recorded?** — **no**. No log, no flag; `bubble_L1Bubble = 0.0` appears in the output but reads
  as "no CIE cooling", not "index clamped".
- **failure scenario** — a low-`sfe`, high-density cloud whose bubble interior stays below
  10^5.5 K. `bubble_L1Bubble` is exactly 0 for the whole run, `bubble_LTotal` is too small, the
  `Lloss/Lgain` trigger never fires, and the run stays energy-driven far longer than it physically
  should.
- **confidence** — **medium-high**

---

### SF-011 · `bubble_E2P` floors the shell volume to `1e-13·r2³` when the bubble degenerates, producing a finite but enormous pressure with no record
- **file:line** — `trinity/bubble_structure/get_bubbleParams.py:229-237`
  ```python
  shell_volume = r2**3 - r1**3
  if shell_volume <= 0:
      shell_volume = 1e-13 * r2**3
  Pb = (gamma - 1) * Eb / shell_volume / (4 * np.pi / 3)
  ```
- **class** — clamped-physics
- **severity** — **S2 latent**
- **triggering condition** — `R1 → R2` in float64, i.e. `1 - R1/R2 ≲ 1e-16`. This is the
  catastrophic-cooling degeneracy the comment describes: as `Eb → 0`, `get_r1`'s root
  `r1 = sqrt(Lmech/(v·Eb)·(r2³-r1³))` is pushed to `r2`.
- **substituted value** — `shell_volume = 1e-13 * r2**3`, i.e. `Pb` is up to ~10¹³× the value a
  correctly-resolved thin shell would give.
- **downstream fate** — `Pb` is written to `params['Pb']` in every phase and to
  `dictionary.jsonl`; it enters `P_drive = max(Pb, P_HII)` in the ODE RHS
  (`energy_phase_ODEs.py:258`), `F_ram = Pb·4πR2²`, the PdV term `4πR2²·v2·Pb` in
  `Edot_from_balance`, the leak luminosity, and the shell's inner-edge density `nShell0`
  (`shell_structure.py:124-125`). The comment argues the energy phases catch the collapse via
  `Eb <= 0` — but the floor fires while `Eb` is still *positive*, so one or more steps are
  integrated with the inflated pressure before any collapse check runs.
- **recorded?** — **no**. No log at any level, no flag. The only trace is `R1 ≈ R2` in the
  snapshot, which a reader must notice unaided.
- **failure scenario** — the high-mass PdV-collapse path (`HIMASS_HANDOFF`). One segment lands on
  the underflow, `Pb` jumps ~13 orders, the shell receives an impulsive outward kick, and `Eb`
  crosses zero on the *next* step — so the recorded collapse time and the terminal `R2`/`v2` are
  set by the floor, not by physics.
- **confidence** — **medium-high**

---

### SF-012 · `get_r1` clamps the bubble energy to `1e-30` inside the root function, converting a *negative* (diverged) `Eb` into a well-formed inner radius
- **file:line** — `trinity/bubble_structure/get_bubbleParams.py:405-409`
  ```python
  # set minimum energy to avoid zero
  if Ebubble < 1e-30:
      Ebubble = 1e-30
  equation = np.sqrt( Lmech_total / v_mech_total / Ebubble * (r2**3 - r1**3) ) - r1
  ```
- **class** — clamped-physics
- **severity** — **S2 latent**
- **triggering condition** — `Eb ≤ 1e-30`, **including any negative `Eb`**. `solve_R1`
  (lines 435-443) guards `Lmech_total <= 0`, `R2 <= 0` and non-finiteness, but never checks
  `Eb > 0` — the sign check is delegated to this clamp, which erases it.
- **substituted value** — `Ebubble = 1e-30` inside the bracket search only.
- **downstream fate** — reaches solver state. With `Eb = 1e-30` the root is `R1 ≈ R2`, so `brentq`
  succeeds and returns a plausible radius instead of the `sqrt(<0) → NaN → brentq raises` that a
  negative energy should produce. `compute_R1_Pb` then hands that `R1` to `bubble_E2P`, which with
  the *true* negative `Eb` and a floored `shell_volume` (SF-011) returns a large **negative** `Pb`.
  Negative `Pb` propagates: `nShell0 ∝ Pb` in `shell_structure.py:124` goes negative,
  `n_array = Pb/(...T)` in the bubble structure goes negative,
  `P_drive = max(Pb, P_HII)` picks `P_HII`, `F_ram = Pb·4πR2²` becomes an inward force. All of it
  is written to `dictionary.jsonl`.
- **recorded?** — **no**.
- **failure scenario** — the phase-1b `Eb → 0` collapse. Between the ODE step that takes `Eb`
  negative and the `classify_energy_collapse` check at `run_energy_implicit_phase.py:1148`, this
  clamp is exercised by the reconciliation path and by any in-loop `compute_R1_Pb`. The comment at
  `run_energy_implicit_phase.py:1358-1363` shows the maintainers already found one instance of this
  ("`Pb ~ -1.6e18` as a garbage negative terminal row") and patched it by *skipping the
  reconciliation snapshot* — the underlying clamp is untouched.
- **confidence** — **medium-high**

---

### SF-013 · `cool_beta_to_Ebdot_pure` returns `Edot = 0.0` on a near-zero denominator, freezing the bubble energy; the companion `Ebc` guard is applied inconsistently
- **file:line** — `trinity/phase1b_energy_implicit/get_betadelta.py:255-269`
  ```python
  Ebc = Eb + c_coeff
  c_frac = c_coeff / Ebc if Ebc > 0 else 0.0
  numerator = (... - a_coeff * R1**3 * Eb**2 / Ebc)      # <- Ebc used UNGUARDED here
  denominator = d_coeff * (1 - c_frac)
  if abs(denominator) < 1e-300:
      return 0.0
  ```
- **class** — clamped-physics (+ inconsistent guard)
- **severity** — **S2 latent**
- **triggering condition** — `denominator = (R2³ - R1³)·Eb/(Eb + c)` underflows. Reachable exactly
  where SF-011 fires: `R1 → R2` makes `d_coeff` underflow to 0.0. The `Ebc > 0` ternary is the
  companion defect: when `Ebc <= 0` the *denominator* use is neutralised but line 262 still
  divides by `Ebc` — `ZeroDivisionError`/`inf` when `Ebc == 0`, and an un-neutralised sign flip
  when `Ebc < 0`.
- **substituted value** — `Ed = 0.0`.
- **downstream fate** — straight into the ODE: `get_ODE_implicit_pure` returns
  `[rd, vd, Ed_from_beta, Td_from_delta]` (`run_energy_implicit_phase.py:624`), so `dEb/dt = 0`.
  The bubble energy is **frozen** while `R2` and `v2` keep evolving. Because `Eb` never moves,
  `classify_energy_collapse` never returns `'momentum'` and the energy→momentum handoff never
  fires — the run continues energy-driven on a constant `Eb`.
- **recorded?** — **no**.
- **failure scenario** — the same `R1 → R2` degeneracy as SF-011, in the segment before the
  collapse check. `Eb` pins at its last value for the remaining segments and the reported bubble
  energy history flatlines at a value that has no physical meaning.
- **confidence** — **medium** (the `< 1e-300` threshold is extremely tight; it is reachable
  essentially only through exact float64 underflow of `R2³ - R1³`, which is however precisely the
  regime `bubble_E2P` was patched for).

---

### SF-014 · `scipy.optimize.minimize` (L-BFGS-B) result used without checking `.success`, on an objective that returns a flat `1e10` for every failed evaluation
- **file:line** — `trinity/phase1b_energy_implicit/get_betadelta.py:1116-1145`
  ```python
  def objective(x):
      beta = np.clip(beta, BETA_MIN, BETA_MAX)
      delta = np.clip(delta, DELTA_MIN, DELTA_MAX)
      try:
          Edot_res, T_res, _ = get_residual_pure(beta, delta, params)
          return Edot_res**2 + T_res**2
      except Exception as e:
          logger.warning(f"Residual calculation failed: {e}")
          return 1e10
  ...
      result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', ...)
      return result.x[0], result.x[1], result.nit
  ```
- **class** — unchecked-solver-status + swallowed-exception
- **severity** — **S2 latent** (only reachable with `betadelta_solver='legacy'`; production default
  is `hybr`)
- **triggering condition** — L-BFGS-B fails to make progress (`success == False`), or every
  evaluation raises so the objective is the constant `1e10` and the numerical gradient is exactly
  zero — L-BFGS-B then "converges" at `x0` on iteration 0.
- **substituted value** — `result.x` = the initial guess; and inside the objective, `1e10` for
  every failed point. Note the `np.clip(beta, …)` at 1118-1119 clips the *evaluation point*
  without telling the optimiser, so L-BFGS-B sees a plateau outside the bounds too.
- **downstream fate** — `(beta_lbfgsb, delta_lbfgsb)` become a candidate at
  `get_betadelta.py:786`; if selected, they are the `beta`/`delta` fed to
  `cool_beta_to_Ebdot_pure` → `Ed` → the ODE, and are written to the snapshot.
- **recorded?** — **partially**: each objective failure logs at WARNING; `.success` is never read
  or logged, but the winning point's residual sets `betadelta_converged` in the output.
- **failure scenario** — a `betadelta_solver='legacy'` run in a regime where the bubble solve fails
  across the whole L-BFGS-B search. The optimiser returns the seed, the seed becomes the "best
  candidate", and the phase advances on unchanged cooling parameters.
- **confidence** — high

---

### SF-015 · `scipy.optimize.root(..., method='hybr')` status is never checked; a non-converged root is accepted and drives the ODE
- **file:line** — `trinity/phase1b_energy_implicit/get_betadelta.py:982-1007`
  ```python
  sol = scipy.optimize.root(gvec, [beta_guess, delta_guess], method='hybr', options=HYBR_OPTIONS)
  ...
  b, d = float(sol.x[0]), float(sol.x[1])
  ...
  converged = g_total < RESIDUAL_THRESHOLD
  logger.debug(f"... ier={sol.status}")
  ```
- **class** — unchecked-solver-status
- **severity** — **S2 latent**
- **triggering condition** — `maxfev=30` exhausted (`HYBR_OPTIONS`), or MINPACK stalls. `sol.status`
  is read *only* into a DEBUG log string; `sol.success` is never read.
- **substituted value** — `sol.x` — whatever MINPACK last held.
- **downstream fate** — the accepted `(beta, delta)` go to `params['cool_beta']`/`['cool_delta']`
  and then to `Ed`/`Td` for the segment ODE, and into the snapshot.
- **recorded?** — **partially, and this is the better-designed guard in the file**: the point is
  re-gated on `dMdt > 0` (lines 992-999), `converged = g_total < RESIDUAL_THRESHOLD` is written to
  `params['betadelta_converged']` (a snapshot key), `betadelta_total_residual` is written, and
  `update_unconverged_streak` (`run_energy_implicit_phase.py:332-357`) escalates to WARNING at 3
  consecutive unconverged segments, with an end-of-phase summary at
  `betadelta_phase_summary`. So a systematically unconverged phase *is* visible. What is missing is
  the raw MINPACK status.
- **failure scenario** — a segment where `maxfev=30` is not enough; the accepted point is a
  partially-converged iterate. `betadelta_converged=False` flags it, but the trajectory still
  advances on it.
- **confidence** — high

---

### SF-016 · The phase-1a `solve_ivp` retry silently downgrades tolerances by 10× and its own success flag is never checked
- **file:line** — `trinity/phase1_energy/run_energy_phase.py:310-336`
  ```python
  if not solution.success:
      logger.warning(f'solve_ivp failed: {solution.message}')
      t_segment_end = t_now + SEGMENT_DURATION / 10
      solution = scipy.integrate.solve_ivp(..., method='RK23', rtol=RTOL*10, atol=ATOL*10)
  # (no second success check)
  event_result = check_event_termination(solution, ode_events)
  ...
  R2_new, v2_new, Eb_new = solution.y[:, -1]
  ```
- **class** — unchecked-solver-status
- **severity** — **S2 latent**
- **triggering condition** — the RK45 solve fails, then the RK23 retry also fails.
- **substituted value** — `solution.y[:, -1]` from a failed integration — the last accepted step
  before failure. If the retry fails on its *first* step, `t_new == t_now` and the state is
  unchanged, so the `while` loop re-enters with identical state: a non-terminating loop that
  re-saves snapshots rather than a wrong answer.
- **downstream fate** — `R2`, `v2`, `Eb` become the phase state for the next segment and are
  written to the snapshot. Note the retry also drops `dense_output=True`, which the first call
  requested; and the trajectory is now a mixture of RK45-at-1e-6 and RK23-at-1e-5 segments with no
  marker of which is which.
- **recorded?** — **partially**: `logger.warning('solve_ivp failed: …')` fires for the *first*
  failure only; the retry's outcome is never logged, and neither appears in `dictionary.jsonl`.
- **failure scenario** — a stiff early phase where RK45 fails repeatedly. The published `Eb(t)` for
  those segments carries 10× the tolerance of the rest of the run with nothing in the output to
  say so.
- **confidence** — high

---

### SF-017 · `check_event_termination` returns the first event with a recorded crossing regardless of `terminal`, so the *non-terminal* `velocity_sign` monitor ends the implicit phase
- **file:line** — `trinity/phase_general/phase_events.py:392-405`
  ```python
  for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):
      if len(t_ev) > 0:
          event = events[i]
          return EventResult(triggered=True, ...)
  ```
  with `make_velocity_sign_event` at `phase_events.py:306-316` (`event.terminal = False`,
  *"just records the crossing"*) placed at **index 0** of the implicit-phase list
  (`build_implicit_phase_events`, line 487-491), and the caller at
  `run_energy_implicit_phase.py:1095-1119` unconditionally `break`ing on
  `event_result.triggered`.
- **class** — non-convergence-fallthrough (a returned flag — `terminal` / `is_simulation_ending` —
  that the call site ignores)
- **severity** — **S2 latent**
- **triggering condition** — `v2` crosses zero downward anywhere inside an implicit-phase segment.
  `solve_ivp` populates `t_events[0]` for non-terminal events too, so the monitor is
  indistinguishable from a terminal one at this call site.
- **substituted value** — control flow: the phase terminates with
  `termination_reason = "velocity_sign_change"` and state taken from the event root, discarding the
  rest of the segment.
- **downstream fate** — phase 1b ends at collapse onset instead of continuing; `main.py` proceeds
  to 1c and 2. `apply_event_result` does not set `EndSimulationDirectly` (because
  `is_simulation_ending` is False), so the run continues — but the entire remaining implicit phase
  is skipped. The trajectory is real, just truncated at a point chosen by a *monitoring* event.
- **recorded?** — **partially**: `logger.info("Event 'velocity_sign' triggered …")` and
  `termination_reason` (which `main.py` discards). Nothing in `dictionary.jsonl`.
- **failure scenario** — any recollapsing cloud. The moment `v2` first turns negative, phase 1b
  ends; the physics of the collapse is then integrated by 1c/2 rather than 1b, changing which
  model governs the collapse — invisibly, because the log line reads like an ordinary event.
- **confidence** — **medium-high** (the code path is unambiguous; I did not run a collapsing config
  to confirm scipy populates `t_events` for non-terminal events, though the documented behaviour is
  that it does).

---

### SF-018 · Snapshot serialisation floors densities/temperatures at `1e-300` before `log10`, so a diverged negative value is published as a plausible `-300`
- **file:line** — `trinity/_input/dictionary.py:645, 655, 680, 697`
  ```python
  eps = 1e-300  # used for safe log10()
  ...
  y_arr = np.log10(np.maximum(np.asarray(val), eps))   # bubble_T_arr, bubble_n_arr, shell_n_arr
  y_arr = np.log10(np.maximum(np.abs(v), eps))         # bubble_dTdr_arr, shell_grav_force_m
  ```
- **class** — unrecorded-substitution
- **severity** — **S3 misleading**
- **triggering condition** — any non-positive entry in `bubble_T_arr`, `bubble_n_arr`,
  `shell_n_arr`. Negative temperature *usually* raises first (`bubble_luminosity.py:668`), but
  negative `Pb` (SF-012) makes `bubble_n_arr = Pb/(μ k_B T)` uniformly negative with no check, and
  `shell_n_arr` has no sign guard at all.
- **substituted value** — `log10(1e-300) = -300.0`.
- **downstream fate** — output only; these are the sole recorded form of the profile arrays
  (the raw arrays are not written). A downstream reader — including
  `trinity/_output/cloudy/dlaw.py`, which exports shell densities to CLOUDY — sees `-300` and
  cannot distinguish "genuinely 1e-300" from "was negative". The `np.abs()` variants additionally
  discard the *sign* of `bubble_dTdr_arr` and `shell_grav_force_m` outright.
- **recorded?** — **no**.
- **failure scenario** — a run that passes through SF-012's negative-`Pb` window publishes a bubble
  density profile that is uniformly `-300` for those snapshots, reading as an evacuated cavity.
- **confidence** — high

---

### SF-019 · `phi = max(0.0, phi)` in the shell ODE RHS and `f_esc_ion = max(0.0, phi[-1])` clamp a negative ionising-photon fraction
- **file:line** — `trinity/shell_structure/get_shellODE.py:108-111`
  ```python
  # Clamp phi: negative values are unphysical ...
  phi = max(0.0, phi)   # <-- add this line
  ```
  and `trinity/shell_structure/shell_structure.py:204, 229`
  ```python
  phi0 = max(0.0, phiShell_arr[idx])   # guard against sub-threshold negative phi
  f_esc_ion = max(0.0, phiShell_arr_ion[-1])
  ```
- **class** — clamped-physics
- **severity** — **S3 misleading**
- **triggering condition** — the integrator carries `phi` below zero past the ionisation front.
- **substituted value** — `0.0`.
- **downstream fate** — the RHS clamp is **defensible**: the retained region is truncated at the
  first `phi <= 1e-9` (`shell_structure.py:182`), so within the consumed profile `phi > 1e-9` and
  the clamp cannot bind; it only tames the discarded tail, and LSODA marches forward so the tail
  cannot retroactively affect accepted steps. The **`f_esc_ion` clamp does reach physics**:
  `f_absorbed_ion = 1 - f_esc_ion` and `_Qi_absorbed = (1 - f_esc_ion)·Qi` set `n_IF_Str` →
  `P_HII` → `P_drive`. If the ODE overshot to, say, `phi = -0.3`, clamping to 0 reports "all
  ionising photons absorbed" and hides the magnitude of the integration error.
- **recorded?** — **no**; `f_esc_ion` itself is not a registered snapshot key (only
  `shell_fAbsorbedIon = 1 - f_esc_ion` is), so a reader sees `1.0` and cannot tell whether it came
  from a clean `phi = 0` or a clamped `phi = -0.3`.
- **failure scenario** — a shell where LSODA takes one coarse step across the front. `f_esc_ion`
  clamps, `n_IF_Str` is computed from `Qi_absorbed = Qi`, and `P_HII` is biased high for that
  segment.
- **confidence** — medium

---

### SF-020 · An unrecognised `cooling_boost_mode` silently disables the boost the user asked for
- **file:line** — `trinity/phase1b_energy_implicit/get_betadelta.py:351-357`
  ```python
  Any unrecognised ``mode`` falls back to the resolved loss, so a typo cannot perturb a run.
  ...
  return Lcool + Lleak  # 'none' (default) and any unrecognised token -> resolved loss
  ```
- **class** — unrecorded-substitution
- **severity** — **S3 misleading**
- **triggering condition** — a `.param` sets e.g. `cooling_boost_mode  multipler`. I confirmed
  `cooling_boost_mode` has **no `validator=`** in `registry.py:384` (unlike its siblings
  `cooling_boost_fA`, `betadelta_solver`, `coverFraction`, all of which do), so the typo passes the
  input trust boundary untouched.
- **substituted value** — mode `'none'` semantics: `Lloss = Lcool + Lleak`.
- **downstream fate** — the run is byte-identical to an unboosted run, but the user believes the
  interface-cooling boost was applied. `effective_Lloss_from_params` is the single point feeding
  the beta-delta residual, the energy ODE, and the transition trigger, so *all three* silently
  revert.
- **recorded?** — **partially**: `cooling_boost_mode` is `run_const=True` so the typo string is
  written to `metadata.json` — a careful reader could catch it. No warning is emitted.
- **failure scenario** — a parameter sweep over `cooling_boost_fmix` with a misspelled mode
  produces a set of identical runs presented as a boost study.
- **confidence** — high

---

### SF-021 · Cluster ages outside the cooling-table range are silently snapped to the first/last table
- **file:line** — `trinity/cooling/non_CIE/read_cloudy.py:324-334`
  ```python
  elif age >= max(age_list):
      age_str = format(max(age_list), '.2e')
      ...
  elif age <= min(age_list):
      age_str = format(min(age_list), '.2e')
  ```
- **class** — clamped-physics
- **severity** — **S3 misleading**
- **triggering condition** — `t_now` outside the bundled table ages (1, 2, 3, 4, 5, 10 Myr). The
  default `stop_t = 15` Myr, so **every default run spends its last third past the table's upper
  edge**.
- **substituted value** — the 1e7 yr cooling/heating cubes, frozen.
- **downstream fate** — reaches solver state via `get_dudt` → the bubble ODE. The cooling
  age-dependence is simply switched off for `t > 10` Myr. This is arguably a defensible modelling
  choice, but nothing in the run says it happened.
- **recorded?** — **no** log at any level.
- **failure scenario** — any default-configuration run: the cooling physics for 10–15 Myr is the
  10 Myr table, and a reader of the output cannot tell.
- **confidence** — high

---

### SF-022 · The reader's time interpolation drops non-finite samples without counting them and extrapolates outside the snapshot range
- **file:line** — `trinity/_output/trinity_reader.py:837-844` and `851-856` (and the identical
  array branch at `872-885`)
  ```python
  valid_mask = np.isfinite(y_vals)
  if not np.any(valid_mask):
      interpolated_data[key] = np.nan
      continue
  if np.sum(valid_mask) < 2:
      interpolated_data[key] = y_vals[valid_mask][0] if np.any(valid_mask) else np.nan
      continue
  ...
  interp_func = scipy_interp.interp1d(valid_times, valid_vals, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
  ```
  plus a blanket `except Exception:` fallback to "use closest value" at line 898.
- **class** — unrecorded-substitution
- **severity** — **S3 misleading**
- **triggering condition** — any NaN in the snapshot series being interpolated (routine: SF-008's
  NaN cooling rows, and `reset_keys(COOLING_PHASE_KEYS)` deliberately NaNs the cooling block after
  phase 1c — `main.py:317`).
- **substituted value** — the NaN samples are dropped and the remaining points are interpolated /
  **extrapolated** across the gap; when fewer than 2 remain, a single value is repeated.
- **downstream fate** — analysis and paper-figure output. A NaN gap becomes a smooth line drawn
  straight across it, and `fill_value='extrapolate'` silently continues that line beyond the last
  snapshot. The count of dropped points is never reported.
- **recorded?** — **no**.
- **failure scenario** — a `bubble_LTotal` series with a NaN episode in the middle is plotted as a
  smooth interpolation with no gap; the reader never learns the cooling solve failed there.
- **confidence** — high

---

### SF-023 · Free-parameter floors at `1e-100` in the SPS load can turn a zero column into an astronomically large derived quantity
- **file:line** — `trinity/sps/read_sps.py:35, 214, 215, 233`
  ```python
  EPSILON = 1e-100  # Small number to prevent division by zero
  Mdot_wind = pdot_wind_raw ** 2 / (2 * np.maximum(Lmech_wind_raw, EPSILON))
  velocity_wind = 2 * Lmech_wind_raw / np.maximum(pdot_wind_raw, EPSILON)
  Mdot_SN = 2 * Lmech_SN_raw / np.maximum(velocity_SN_base ** 2, EPSILON)
  ```
- **class** — clamped-physics
- **severity** — **S3 misleading**
- **triggering condition** — a user-supplied SPS file with a zero (or missing-mapped) `Lmech_W` or
  `pdot_W` row while the other is non-zero — e.g. a table whose wind columns start at 0 before
  wind turn-on.
- **substituted value** — the denominator becomes `1e-100`, so `Mdot_wind` or `velocity_wind`
  becomes ~10¹⁰⁰× its scale rather than raising.
- **downstream fate** — `Lmech_total`, `pdot_total`, `v_mech_total` are load-time run constants
  feeding every phase; an inflated wind velocity would set `pRam` and the momentum-phase driving
  pressure for the whole run.
- **recorded?** — **no** for these three (contrast the adjacent `np.maximum(Lmech_SN_raw, 0)` at
  line 208, which **does** log a WARNING first — the right pattern, applied inconsistently three
  lines later).
- **failure scenario** — a custom `sps_path` whose wind columns are zero for the first rows. The
  interpolators are built over a spike of 1e100-scale velocities; every quantity derived from them
  is wrong, and the load reports success.
- **confidence** — medium (requires a user-supplied SPS table; the bundled default has no zero
  rows in the wind columns that I checked).

---

### SF-024 · The phase-boundary reconciliation blocks swallow every exception, so a failure leaves stale derived values as the phase's final published state
- **file:line** — `trinity/phase1_energy/run_energy_phase.py:390-404`,
  `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1366-1396`,
  `trinity/phase1c_transition/run_transition_phase.py:855-861`,
  `trinity/phase2_momentum/run_momentum_phase.py:885-905`
  ```python
  except Exception as e:
      logger.warning(f"Phase-boundary reconciliation failed: {e}")
  ```
- **class** — swallowed-exception
- **severity** — **S3 misleading**
- **triggering condition** — a raise anywhere in the block: `get_current_sps_feedback` outside the
  SPS time range, `solve_R1` non-finite, `shell_structure_pure` failing, `save_snapshot` I/O.
- **substituted value** — none; the *effect* is that `params.save_snapshot()` at the end of the
  block never runs, so the boundary snapshot is missing — and whatever partial mutations the block
  made before raising (e.g. `params['R1'].value = R1_f` set, `params['Pb'].value` not) persist into
  the next phase.
- **downstream fate** — the next phase starts from a half-updated params dict; the boundary row is
  absent from `dictionary.jsonl`, and the comment in the source explains the *reason* the row
  matters ("A bare save_snapshot() would save stale derived values AND block the next phase's
  correct first snapshot via the duplicate guard").
- **recorded?** — **partially**: WARNING in the log; nothing in the output. The momentum-phase
  variant is the best of the four — it appends the exception class and the deepest traceback frame.
- **failure scenario** — a partial mutation followed by a raise leaves `R1` from the new state and
  `Pb` from the old one; the next phase's `P_drive` mixes the two.
- **confidence** — high

---

### SF-025 · `save_snapshot`'s duplicate guard silently discards a snapshot when `(t_now, R2)` repeat
- **file:line** — `trinity/_input/dictionary.py:706-717`
  ```python
  if ("t_now" in last and t_now == last["t_now"]) and ("R2" in last and r2 == last["R2"]):
      logger.debug(f"Duplicate detected in save_snapshot at t = {t_now}. Snapshot not saved.")
      return
  ```
- **class** — unrecorded-substitution
- **severity** — **S3 misleading**
- **triggering condition** — a segment that did not advance `t` or `R2` — which is exactly the
  signature of a stalled or failed solver step, or of a frozen no-physical-root streak whose ODE
  made no progress.
- **substituted value** — the snapshot is dropped entirely; `save_count` is not incremented.
- **downstream fate** — output only. The row that would have shown "nothing moved this segment" is
  the one silently removed, so the published trajectory looks smoother than the run actually was.
  The maintainers already work around this at `run_energy_implicit_phase.py:1019-1026`, checking
  `save_count` deltas to know whether a write happened.
- **recorded?** — **no** (DEBUG only).
- **confidence** — high

---

## Counts

By class:

| class | S1 | S2 | S3 | S4 | total |
|---|---|---|---|---|---|
| unchecked-solver-status | 3 (SF-001, SF-002, SF-004) | 3 (SF-014, SF-015, SF-016) | — | — | 6 |
| swallowed-exception | 1 (SF-003) | 1 (SF-006) | 2 (SF-007, SF-024) | — | 4 |
| clamped-physics | 1 (SF-005) | 5 (SF-009, SF-011, SF-012, SF-013, SF-021*) | 2 (SF-019, SF-023) | — | 8 |
| unrecorded-substitution | — | 2 (SF-008, SF-010) | 4 (SF-018, SF-020, SF-022, SF-025) | — | 6 |
| non-convergence-fallthrough | — | 1 (SF-017) | — | — | 1 |
| dead-handler | — | — | — | 0 | 0 |

\* SF-021 counted at S3 in the entry; placed in the clamped-physics row for class totals.

By severity: **S1 = 5**, **S2 = 11**, **S3 = 9**, **S4 = 0 full entries** (S4-class sites are in the
bulk list below rather than as entries, per the brief).

---

## Guards that are appropriate

Examined and judged correct — genuinely transient, properly recorded, or affecting only
presentation. Grouped so a reader can tell what was cleared from what was skipped.

**Correct by construction — loud failures, no substitution.** These are the model the rest of the
package should follow.
- `bubble_luminosity.py:452-528` `_solve_bubble_structure` — checks `sol.success`, converts a
  raising RHS into an explicit `ok=False` contract, returns an all-NaN `psoln` the caller is
  documented never to consume. Pinned by `test/test_bubble_solver_failures.py`.
- `bubble_luminosity.py:654-671` — `if not _ok: raise BubbleSolverError`; `np.any(T_array < 0)` →
  `logger.critical` + raise. Correct: loud, logged above WARNING.
- `bubble_luminosity.py:414-425` — T→0 in the RHS raises a catchable `BubbleSolverError` rather
  than `sys.exit` (the documented former behaviour).
- `get_bubbleParams.py:435-454` `solve_R1` — explicit non-finite check that raises rather than
  relying on scipy version behaviour, and re-raises the `brentq` failure after logging at ERROR.
  (Its one gap is the missing `Eb > 0` check — SF-012.)
- `sps/read_sps.py:112-115, 174-177`, `sps/sps_columns.py:350-373, 455-468`,
  `cloud_properties/powerLawSphere.py:141-205`, `cloud_properties/bonnorEbertSphere.py:350-384`,
  `phase0_init/get_InitCloudProp.py` mass checks — all validate at the trust boundary and **raise**
  with actionable messages. No substitution.
- `sps/update_feedback.py:156-159` — raises on `t` outside the SPS range rather than extrapolating.
- `cooling/non_CIE/read_cloudy.py:300-305` — unsupported `ZCloud` raises with the supported list.
- `_input/read_param.py:91-101`, `_input/sweep_parser.py:139-143, 225-228` — `except ValueError:
  pass` in a **type-sniffing cascade** (float → Fraction → string). Nothing is being suppressed;
  the fall-through *is* the algorithm.

**Recorded substitutions — the substitution happens but the run says so.**
- `sps/read_sps.py:203-208` — `np.maximum(Lmech_SN_raw, 0)` preceded by an explicit
  `logger.warning`. This is the correct shape for a clamp.
- `_input/fkappa_auto.py:80-92` — `np.clip` to the calibrated hull with a WARNING naming the
  offending coordinates.
- `phase0_init/get_InitPhaseParam.py:115-138` — all three `MIN_*` floors log a WARNING before
  substituting.
- `run_energy_implicit_phase.py:332-373` `update_unconverged_streak` / `betadelta_phase_summary` —
  a genuinely well-built escalation: per-segment detail at DEBUG, a WARNING at 3 consecutive
  unconverged segments, a second WARNING when the dt mitigation disengages, and an end-of-phase
  summary whose log level flips to WARNING when the phase was not clean.
- `run_energy_implicit_phase.py:847-879` no-physical-root handling — first hit WARNs, repeats go
  to DEBUG (deliberate anti-flood), and a 50-segment streak hands off to momentum with a detailed
  WARNING. (The residual gap is SF-003's stale bubble columns, not the logging.)
- `run_energy_phase.py:169-183` and `run_energy_implicit_phase.py:1148-1175` — the energy-collapse
  routing: narrow exception tuple, explicit `SimulationEndCode.ENERGY_COLLAPSED`, WARNING with full
  state. Exactly right.
- `_functions/logging_setup.py:100-104` — `except Exception: return True` on
  `record.getMessage()`; the failure mode is "never suppress", i.e. the guard fails *open* on a
  log filter. Correct.
- `_output/simulation_end.py:653-673` — the NaN/Inf inventory written into
  `metadata.json[termination_debug]`. A real (if last-two-snapshots-only) detection mechanism.

**Intentional physics, documented, and observable in the output** — flagged here so a reader knows
they were examined and deliberately not filed as findings:
- `energy_phase_ODEs.py:211-217` / `run_energy_implicit_phase.py:962-969` — the shell-mass
  never-decrease clamp (`mShell = prev_mShell; mShell_dot = 0`). A modelling choice about swept-up
  mass, applied consistently in the RHS and the diagnostics, and both `shell_mass` and
  `shell_massDot` are snapshot keys, so a reader can see the freeze. Not silent.
- `shell_structure.py:245-253` — `n_IF_Str = min(n_IF_Str, shell_n0)`, the thin-skin pressure-
  equilibrium cap. Both operands are recorded (`n_IF_Str`, `shell_n0`), so the binding is
  detectable.
- `run_transition_phase.py:245` — `Ed = min(Ed_energy_balance, Ed_soundcrossing)`. Documented in
  the module docstring as the transition model, not a guard.
- `get_bubbleParams.py:280-284` — the leak-luminosity guards (`Cf >= 1`, `Pb <= 0`, `c_sound <= 0`
  → `0.0`). Correct: the term self-limits to zero and can never *inject* energy; `Cf=1` must
  reproduce the sealed bubble exactly. Pinned by `test/test_cf_leak.py`.
- `get_shellODE.py:98-100` — `nShell = min(nShell, _NSHELL_MAX=1e120)`, ~55 orders above any
  physical shell density and provably confined to the discarded post-front tail. Pinned by
  `test/test_shell_overflow_guard.py`. Appropriate.
- `bubble_luminosity.py:119-149` `_quiet_lsoda_fortran` — suppresses only Fortran fd-level chatter
  during the solve; `sol.success` is checked independently afterward. The scoping is tight
  (`os.dup2` restored in `finally`) and it cannot hide a solver failure.
- `bubble_luminosity.py:570-617` `_clean_radius_grid` (`np.maximum(avg_magnitude, 1e-30)`) — grid
  hygiene on a construction artefact, DEBUG-logged with the removal count. No physics.
- `bubble_luminosity.py:242-247` — `if np.isnan(bubble_dMdt)` as the "no previous guess" sentinel,
  re-seeding from Weaver Eq. 33. Correct: a NaN seed is refused rather than propagated.

**Diagnostic / presentation-only, correctly scoped (S4 bulk — examined, no entry filed).**
All of the following are gated behind an env var, wrap an observational capture, or only shape a
log/terminal string; none can alter the trajectory or the physics columns of `dictionary.jsonl`:
- `bubble_luminosity.py:1048, 1059, 1080, 1120, 1143` — the `TRINITY_BUBBLE_DIAG` /
  `TRINITY_BUBBLE_STATE_DUMP` capture handlers. Explicitly documented as "purely observational",
  and each failure logs a WARNING.
- `run_energy_implicit_phase.py:434-436`, `run_transition_phase.py:172-174`,
  `run_momentum_phase.py:164-166` — `get_monitor_values`' `except Exception: pass`; a dropped key
  only coarsens the adaptive-`dt` heuristic, which is self-correcting.
- `compute_max_dex_change`'s `except (ValueError, ZeroDivisionError): continue` in all three phases
  — same: adaptive stepping only.
- `run_energy_implicit_phase.py:1435-1448` — the R1 shadow CSV writer (explicitly a sideline file
  that never touches `dictionary.jsonl`).
- `_input/dictionary.py:321, 329, 341, 380, 843, 899, 918, 1063, 1107, 1111`;
  `_output/_metadata_io.py:73, 124`; `_output/show_run.py:124, 271, 289, 291, 311, 490`;
  `_output/terminal_prints.py:156, 158-159, 218`; `_output/trinity_reader.py:385, 406, 452, 544,
  569, 688, 1046, 1314, 1328, 1334`; `_output/simulation_end.py:231, 304, 345, 382, 482, 521-524,
  532, 741`; `_output/cloudy/**` (`dlaw.py:94, 119, 146, 194`, `run_loader.py`,
  `snapshot_to_deck.py`, `trinity_to_cloudy.py`) — serialisation, metadata I/O, CLI formatting and
  export-path validation. The `isfinite` checks in `cloudy/` and `snapshot_to_deck.py` **raise**
  (`DlawError`, `SnapshotInvalid`); the rest degrade a printed string.
- `_input/sweep_runner.py:147, 257, 321, 332, 376, 545, 565`, `_input/sweep_jobs.py:301, 321, 329`,
  `_input/sweep_parser.py:83, 89, 430, 991, 1004` — subprocess orchestration and job-file parsing;
  a swallowed failure marks a sweep entry failed, never a physics value.
- `_functions/unit_conversions.py:462, 494, 557, 586` — the module's `__main__` self-test block
  and one narrow `(ValueError, ZeroDivisionError)` → `raise UnitConversionError` re-raise.
- `_functions/simplify.py` (all 21 CSV rows) — the snapshot array compressor. It is
  presentation-only, and it already *has* the right guard: `_simplify_error` computes a
  reconstruction R² and `dictionary.py:527-531` logs at WARNING when R² < 0.9.
- `_analysis/check_yesno.py:79, 115-116, 122-123, 143-149, 171, 176, 182` — an offline A/B
  comparison helper; its NaNs are comparison sentinels, and `except Exception -> print(...)`
  reports a failed run load to the operator.
- `cloud_properties/validate_gmc.py:594, 679` — `except Exception: continue` inside the
  *suggestion* grid searches that run only after a config has already been rejected. Failing to
  suggest an alternative cannot corrupt anything.
- `cloud_properties/validate_gmc.py:420, 490` — the `GMCValidationResult(valid=False, …=np.nan)`
  returns. `valid=False` is the recorded signal; the NaNs are placeholders on an already-failed
  path.
- `_functions/operations.py:30-65` `find_nearest_lower` — carries the same index clamp as SF-010,
  but is documented as a retained fallback and has **no production caller**
  (`grep -rn find_nearest_lower trinity/` → definition only). Dead in practice; flagged, not filed.

**Not examined line-by-line** (triaged by grep and judged out of the physics path; stated so the
coverage boundary is explicit): `_input/sweep_parser.py` beyond the guard sites,
`_output/cloudy/trinity_to_cloudy.py` CLI argument handling, `_output/show_run.py` rendering,
`_functions/extract_example_snapshots.py`, and the `paper/`, `tools/`, `docs/` trees (out of scope).
