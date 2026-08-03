# Cross-cutting sweep ⑨ — numerical hygiene

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

## Scope and sources

Read-only sweep over `/home/user/trinity/trinity/**` (72 modules), with `run.py`,
`test/test_phase_events.py` and `test/` consulted only to establish how a setting is consumed.
`docs/dev/code-audit/data/claims_guards.csv` (213 clamp sites, 131 `except` sites) was used as a
worklist. `docs/dev/code-audit/slices/` was **not** read.

Environment used for the executable checks: `scipy 1.17.1`, `numpy 1.26.4` (the pinned `<2` line).
Every arithmetic claim below was run standalone (no `trinity` import, no package state touched).

Relevant configured scales, taken from `trinity/_input/registry.py`:
`stop_t = 15` Myr (l.353), `stop_r = 500` pc (l.352), `coll_r = 1` pc (l.355), so
`min_r = max(coll_r*1.5, 0.01) = 1.5` pc (`phase_events.py:445`).

Solver inventory (all `solve_ivp` / `odeint` / `root` / `fsolve` / `brentq` / `minimize` sites in
the package):

| site | solver | state | rtol | atol | max_step | min_step | other |
|---|---|---|---|---|---|---|---|
| `phase1_energy/run_energy_phase.py:299` | `solve_ivp` RK45 | `[R2,v2,Eb]` | 1e-6 | **1e-9** | **—** | — | retry at 10× tol, RK23 |
| `phase1b_energy_implicit/run_energy_implicit_phase.py:1079` | `solve_ivp` LSODA | `[R2,v2,Eb,T0]` | 1e-6 | 1e-8 | **2e-5** | 1e-6 | |
| `phase1c_transition/run_transition_phase.py:640` | `solve_ivp` LSODA | `[R2,v2,Eb]` | 1e-6 | 1e-8 | **2e-4** | 1e-6 | |
| `phase2_momentum/run_momentum_phase.py:722` | `solve_ivp` LSODA | `[R2,v2]` | 1e-6 | 1e-8 | **2e-4** | 1e-6 | |
| `bubble_structure/bubble_luminosity.py:502` | `solve_ivp` LSODA | `[v,T,dTdr]` | 1e-8 | 1e-10 | — | — | `dense_output` |
| `bubble_structure/bubble_luminosity.py:349` | `solve_ivp` LSODA | `[v,T,dTdr]` | 1e-6 | 1e-10 | — | — | `t_eval` 500 pts |
| `bubble_structure/bubble_luminosity.py:261` | `fsolve` (dMdt) | scalar | — | — | — | — | `xtol=1e-4, factor=50, epsfcn=1e-4`; **`ier` discarded** |
| `bubble_structure/bubble_luminosity.py:724` | `brentq` (r at T=10^5.5) | scalar | default | — | — | — | `xtol=1e-8` |
| `bubble_structure/get_bubbleParams.py:445` | `brentq` (R1) | scalar | default | default | — | — | bracket `[0, R2]` |
| `shell_structure/shell_structure.py:165, 324` | `odeint` | `[n,φ,τ]` / `[n,τ]` | **default 1.49e-8** | **default 1.49e-8** | — | — | `mxstep=50000`; **`full_output` unused** |
| `cloud_properties/bonnorEbertSphere.py:254` | `odeint` (Lane-Emden) | `[u,ω]` | **default** | **default** | — | — | **`full_output` unused** |
| `phase1b_energy_implicit/get_betadelta.py:983` | `root` hybr | `(β,δ)` | — | — | — | — | `xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4` |
| `phase1b_energy_implicit/get_betadelta.py:1131` | `minimize` L-BFGS-B | `(β,δ)` | — | — | — | — | `ftol=1e-8, gtol=1e-6, maxiter=15` |

---

## 1. Inert tolerances (`isclose` family)

Six sites exist in the package. Three are inert-or-vacuous, three are correctly scoped.

### NUM-01 — `trinity/_output/trinity_reader.py:721`: `rtol=1e-10` is inert for every TRINITY run

```python
exact_idx = np.where(np.isclose(times, t, rtol=1e-10))[0]
```

`np.isclose` leaves `atol` at its default `1e-8`. The predicate is
`|a-b| <= atol + rtol*|b|`, so the intended `rtol` only becomes the operative term when
`1e-10 * |t| > 1e-8`, i.e. **`t > 100 Myr`**. `stop_t` defaults to 15 Myr and is a hard cap, so
`rtol=1e-10` **never binds on any run this code can produce**; the effective tolerance is a flat
`atol = 1e-8 Myr = 10 years`.

Demonstrated:

```
np.isclose(1e-3, 1e-3+5e-9, rtol=1e-10)  ->  True     # 5e-9 Myr = 5 yr apart, called "exact"
np.isclose(5e-9, 1e-9,      rtol=1e-10)  ->  True     # 5x apart, called "exact"
```

Consequence: `TrinityOutput.get_at_time(t)` takes the `mode='interpolate'/'closest'` branch only
when no snapshot lies within 10 yr of `t`; otherwise it silently returns the **first** snapshot in
that 10-yr window and reports it as exact. Phase 1a runs on 3e-5 Myr (30 yr) segments
(`run_energy_phase.py:54`), so consecutive early snapshots are only ~3× the window apart. Analysis
path only — it does not perturb the trajectory — but it is the archetype the sweep targets, and the
crossover is unambiguous.

### NUM-20 — `trinity/phase0_init/get_InitCloudProp.py:505, 530-531`: verification that cannot fail

```python
idx = np.searchsorted(props.r_arr, props.rCloud)
if idx < len(props.r_arr) and np.isclose(props.r_arr[idx], props.rCloud):     # l.505
...
rCloud_in_array = np.any(np.isclose(props.r_arr, props.rCloud))              # l.530
rCore_in_array  = np.any(np.isclose(props.r_arr, props.rCore))               # l.531
```

`_create_radius_array` (l.412-455) ends with
`r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud])))` — both radii are inserted
**bit-exactly** two calls earlier. `verify_key_radii_in_array` therefore always returns True and its
`logger.warning` branches are unreachable; the docstring's "exactly" is guaranteed by construction,
not tested.

The tolerance itself is not the problem, and is worth recording as arithmetic:
`np.isclose` defaults are `rtol=1e-5, atol=1e-8`, so `rtol` governs above `|b| > 1e-3` pc — true for
both `rCore` (~1 pc) and `rCloud` (~10s pc). At `rCloud = 20` pc the window is `2e-4` pc; the log
grid there has spacing `20 * ((20/1e-3)^(1/999) - 1) ≈ 0.199` pc, ~1000× wider, so the check could
not false-positive on a neighbour either. It is vacuous, not loose.

---

## 2. Solver tolerances

### NUM-14 — phases 1a and 1c integrate the same state vector with different `atol`, method and step cap

`phase1_energy/run_energy_phase.py:58-59` sets `RTOL = 1e-6, ATOL = 1e-9` on `[R2, v2, Eb]` with
RK45 and **no `max_step`**. `phase1c_transition/run_transition_phase.py:132-135` sets
`ODE_RTOL = 1e-6, ODE_ATOL = 1e-8, ODE_MAX_STEP = 2e-4, ODE_MIN_STEP = 1e-6` on the *same*
`[R2, v2, Eb]` with LSODA. Nothing documents why the same state vector gets a 10× different absolute
floor.

Where a scalar `atol` on a heterogeneous state vector actually binds is
`|y_i| < atol/rtol`:

| phase | `atol/rtol` crossover | binds on |
|---|---|---|
| 1a | `1e-9/1e-6 = ` **1e-3** | `v2` inside ±1e-3 pc/Myr |
| 1b, 1c, 2 | `1e-8/1e-6 = ` **1e-2** | `v2` inside ±1e-2 pc/Myr |

- `R2` ≥ `min_r` = 1.5 pc (default `coll_r=1`) ⇒ `atol` **never** governs `R2`.
- `Eb`: `ENERGY_FLOOR` / `ENERGY_HANDOFF_FLOOR` = 1e3 code units ⇒ `rtol*Eb ≥ 1e-3 ≫ atol` ⇒ **inert**.
- `T0` ~ 1e6 K ⇒ **inert**.
- `v2` is the *only* component for which `atol` is ever the operative term, and it is exactly the
  component that crosses zero at stall/turnaround. `atol = 1e-8` pc/Myr ≈ **1e-8 km/s** is ~2 orders
  below any physically meaningful velocity resolution; near turnaround the integrator is being asked
  for nanometre-per-century accuracy, which is a plausible source of the step-size failures the
  `solver_failed` branches exist to catch.

This is **inconsistent** (two sites disagree) plus **wrong-scale on one component**, not inert.

### NUM-15 — `ODE_MAX_STEP` comment is stale by 10× in phases 1c and 2

`run_transition_phase.py:135` and `run_momentum_phase.py:127` both read:

```python
ODE_MAX_STEP = DT_SEGMENT_MIN / 5  # Max step = 2e-5 Myr (ensures >=5 steps per segment)
```

but `DT_SEGMENT_MIN = 1e-3` in both (l.94 / l.87), so the actual value is **2e-4 Myr**, not 2e-5.
Only phase 1b (`DT_SEGMENT_MIN = 1e-4`, l.113) yields the 2e-5 the comment claims. Confirmed
arithmetically. The "≥5 steps per segment" claim is also only true at `DT_SEGMENT_MIN`; at
`DT_SEGMENT_MAX = 5e-2` the cap forces ≥250 steps (1c/2) or ≥2500 (1b).

### NUM-25 — `min_step = 1e-6` leaves LSODA a 20× step window in phase 1b

`ODE_MIN_STEP = 1e-6` combined with `ODE_MAX_STEP = 2e-5` (phase 1b) gives the adaptive controller a
step range spanning only a factor of 20. `min_step` is passed straight to ODEPACK's `HMIN`
(scipy `LSODA.__init__(min_step=0.0)` default); when the error test demands a smaller step LSODA
cannot comply and returns a failure, which the runner converts to
`termination_reason = "solver_failed: …"` and breaks the phase. A stiffness event is therefore
recorded as a solver failure rather than resolved. In phases 1c/2 the window is 200×.

### NUM-07 — `fsolve` convergence flag discarded for the bubble mass-flux `dMdt`

`bubble_structure/bubble_luminosity.py:261-267`:

```python
bubble_dMdt = scipy.optimize.fsolve(
        velocity_residuals_wrapper,
        bubble_dMdt,
        xtol=1e-4,
        factor=50,
        epsfcn=1e-4
    )[0]
```

`full_output` is not requested, so `ier`/`mesg` are unavailable and never checked. On failure MINPACK
returns the last iterate and scipy emits only a `RuntimeWarning`; the code indexes `[0]` and uses it
as the physical evaporative mass-flux. `maxfev` is also unset (MINPACK default `200*(n+1) = 400`
evaluations, each a 500-point LSODA solve). The downstream gate `_usable_dMdt`
(`get_betadelta.py:381-386`) only tests `isfinite(dMdt) and dMdt > 0` — which a stranded iterate
passes; the docstring there already concedes "fsolve stranded on the penalty plateau returning
garbage", so the failure mode is known but not detected at source.

### NUM-16 — `xtol=1e-4` vs `epsfcn=1e-4` are two orders apart in the same `fsolve`

MINPACK's `fdjac1` uses `h = sqrt(max(epsfcn, eps_mach)) * |x|`, so `epsfcn=1e-4` sets the
finite-difference step to **1e-2 · |dMdt|** — 1% of the unknown — while `xtol=1e-4` asks for the root
to 0.01%. The Jacobian probe is 100× coarser than the requested answer. Not fatal for a smooth
scalar residual (Broyden updates recover), but the pairing is internally inconsistent and the
comment at l.94-99 justifies only the loose `xtol`.

### NUM-12 — the shell ionization-front threshold sits **inside** `odeint`'s own error bar

`shell_structure/shell_structure.py:165-168` integrates `[nShell, φ, τ]` with `odeint(..., mxstep=50000)`
and **no `rtol`/`atol`**, so both default to `sqrt(eps) = 1.4901161193847656e-08`. The termination
test two lines later is

```python
phiCondition = phiShell_arr <= 1e-9   # l.182  "small positive threshold"
```

`φ` is the normalised ionizing-flux attenuation, initialised `phi0 = 1` (l.118), so `φ ∈ [0,1]` and
the component's allowed local error is `atol + rtol*|φ| ≥ 1.49e-8` over the whole range — the
absolute term dominates everywhere (crossover at `φ = 1`, the initial value). **The depletion
threshold 1e-9 is 14.9× smaller than the integrator's own absolute error tolerance.** The radius at
which `φ` first drops below 1e-9 — i.e. the location of the ionization front, `R_IF`, which sets
`n_IF`, `f_esc_ion`, the Strömgren balance density and the ionised/neutral split — is therefore
determined by accumulated integration error, not by the ODE. The `max(0.0, φ)` clamps at
`get_shellODE.py:111` and `shell_structure.py:204, 229` exist precisely because `φ` rings negative
around zero at this level.

### NUM-13 — `odeint` success is never checked in the shell solver

Both shell calls (`shell_structure.py:165-168`, `324-327`) omit `full_output=True`. The module
comment at l.28-34 records that the default `mxstep=500` *was* being exhausted, emitting
"Excess work done on this call" and **silently truncating** the shell integration; the fix raised the
ceiling to 50000 but left the detection gap. If 50000 is ever exhausted the same silent truncation
returns, now without even the stderr warning being tied to a failure path.

### NUM-24 — Lane-Emden `odeint` runs on defaults; `atol` governs the entire core

`cloud_properties/bonnorEbertSphere.py:254` integrates `[u, ω]` from `xi_min = 1e-7`, where the
series initial conditions are `u0 ≈ ξ²/6 = 1.7e-15` and `ω0 ≈ ξ/3 = 3.3e-8`. With
`rtol = atol = 1.49e-8`, the crossover `atol/rtol = 1.0` means **`atol` governs for `|u| < 1`**, i.e.
the whole Bonnor-Ebert core (`u < 1` ⇔ `ρ/ρc > e⁻¹ = 0.37`). At the first grid point the allowed
absolute error in `u` is 7 orders larger than `u` itself, and 45% of `ω0`.

Mitigating: the consumed quantity is `ρ/ρc = exp(-u)`, and for `u ≪ 1`, `exp(-u) ≈ 1-u`, so a 1.5e-8
absolute error in `u` becomes a **1.5e-8 relative** error in the density profile — harmless. The
dimensionless mass `m = ξ²·dudξ` (l.264) is the exposed casualty at small `ξ`, but it → 0 there and
the mass integral is dominated by large `ξ`. `full_output` is again unused. Net: not a live defect,
but the tolerance is defaulted rather than chosen and one component is unresolved.

### `brentq` sites — checked, correctly scoped

- `bubble_luminosity.py:724`, `xtol=1e-8` pc on the `T = 10^5.5` crossing radius. This is 4 orders
  looser than `brentq`'s default `2e-12`, but the function being bracketed is a **cubic `interp1d`
  over the 60k-point production grid** (`_create_radius_grid`, l.529). Grid spacing over a
  `[R1, r2Prime]` span of order 1 pc with 6e4 points is ~1e-5 pc, so the interpolant's own error
  dominates `xtol` by ~3 orders. The tolerance does not bind. **Cleared.**
- `get_bubbleParams.py:445`, `brentq(get_r1, 0.0, R2)` on default tolerances with an explicitly
  argued full bracket, an `isfinite` pre-gate (l.439-443) added specifically because
  "scipy < 1.11 brentq silently converges on a NaN-poisoned function", and a re-raise rather than a
  fabricated value. **Cleared** — this is the best-guarded root-find in the package.

---

## 3. Event functions

### NUM-02 — `check_event_termination` returns the first event **by list index**, not by time; a documented non-terminal event terminates phase 1b

`phase_general/phase_events.py:392-405`:

```python
for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):
    if len(t_ev) > 0:
        event = events[i]
        return EventResult(triggered=True, ..., t=float(t_ev[0]), y=y_ev[0].copy(),
                           is_simulation_ending=getattr(event, 'is_simulation_ending', True), ...)
```

Two independent defects compose here.

**(a) Terminality is never consulted.** The module docstring (l.25-26) defines
`velocity_sign` as a *"Monitoring Event (non-terminal; record a crossing only)"*, and
`make_velocity_sign_event` sets `event.terminal = False` with the comment "just records the
crossing" (l.310). But `build_implicit_phase_events` places it at **index 0**
(`phase_events.py:487-491`), and `run_energy_implicit_phase.py:1095-1118` does:

```python
event_result = check_event_termination(sol, ode_events)
if event_result.triggered:
    termination_reason = event_result.reason_code
    R2, v2, Eb, T0 = (event_result.y[...])
    t_now = event_result.t
    ...
    break        # <- ends phase 1b
```

So **the first downward zero-crossing of `v2` anywhere in a segment ends the implicit phase**, rewinds
the state to the crossing (discarding the rest of the integrated segment), and — because
`is_simulation_ending=False` — hands off to 1c → momentum without setting `EndSimulationDirectly`.
A bubble that momentarily stalls and would re-accelerate is force-routed to the momentum phase.

**(b) Index order beats time order.** Because `velocity_sign` is non-terminal, its root is recorded
even when a *terminal* event later stopped the solve, and index 0 wins the loop.

Demonstrated standalone (replicating the exact event list order and the exact 6-line loop; SciPy
1.17.1, `rtol=1e-6, atol=1e-8, max_step=2e-4` as in phase 1b):

```
t_events[0] velocity_sign :  [0.25]
t_events[1] min_radius    :  [1.15138782]

  -> reported event       :  velocity_sign
  -> reported t           :  0.24999999999999795
  -> is_simulation_ending :  False
  -> reported state y     :  [3.125e+00, 2.17e-19]

but the integration actually STOPPED at t = 1.1513878188659943  R2 = 1.4999999999999998
```

The shell reached the collapse radius `min_r = 1.5` pc and the terminal event fired — yet the run
records `velocity_sign_change`, keeps `R2 = 3.125` pc, leaves `EndSimulationDirectly` unset and
`isCollapse` unset, and continues into phase 1c.

**Corroborating symptom:** the collapse detector at `run_energy_implicit_phase.py:1301-1303`
(`if v2 < 0 and R2 < R2_prev: params['isCollapse'].value = True`) sits *after* the event `break`.
It can therefore only ever fire for a run that **entered** phase 1b already contracting — the
entering-collapse case it was written for is unreachable. That is independent evidence that the
event ordering is not the intended behaviour.

`test/test_phase_events.py` covers each factory's `direction`/`terminal`/`is_simulation_ending`
attributes in isolation but has no test of `check_event_termination` with more than one populated
`t_events` entry.

### NUM-03 — `isCollapse` is set by substring match, so reaching `stop_r` is recorded as a collapse

`phase_events.py:626-629`:

```python
if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():
    if 'isCollapse' in params:
        params['isCollapse'].value = True
```

Reason codes are `"small_radius_event"` (`make_min_radius_event`) and `"large_radius_event"`
(`make_max_radius_event`). Both contain `"radius"`:

```
small_radius_event       -> isCollapse set: True
large_radius_event       -> isCollapse set: True     <-- expansion success flagged as collapse
velocity_runaway_event   -> isCollapse set: False
```

`max_radius` is the `R2 > stop_r = 500` pc terminal event — an *expanding* shell, `v2 > 0`. The run
is then written out with `isCollapse = True`. `_output/show_run.py:87-90` renders that as
`collapsed: collapsing`, and `_input/sweep_runner.py:560-561` renders it flatly as
`collapsed: "yes"` in the sweep summary table. A substring test on a human-readable reason code is
the wrong mechanism; `end_code` (`SimulationEndCode.SHELL_COLLAPSED` vs `LARGE_RADIUS`) already
carries the distinction unambiguously.

### NUM-04 — `cooling_balance` is built as a `solve_ivp` event but never attached

`make_cooling_balance_event` (`phase_events.py:319-356`) constructs a proper
`terminal=True, direction=-1` event on `(Lgain - Lloss)/Lgain - threshold`.
`build_implicit_phase_events` returns the factory (l.497, 501), and phase 1b unpacks it —

```python
ode_events, cooling_balance_factory = build_implicit_phase_events(params)   # l.752
```

— and then **never uses `cooling_balance_factory` again** (only occurrence in the file). The
transition is instead detected by an inline segment-boundary test at l.1296:

```python
if 'cooling_balance' in active_triggers and Lgain > 0 and (Lgain - Lloss) / Lgain < threshold:
```

Consequence: the default phase-1b → 1c transition — the primary physical transition of the run — is
located to **one segment**, i.e. up to `DT_SEGMENT_MAX = 5e-2` Myr late, instead of to the event
root. The factory's own docstring (l.325-327) states the reason (`Lgain`/`Lloss` are not available
inside the RHS), so this is a known design compromise; but the dead unpacking makes it look
attached, and the factory's `direction=-1` and threshold are untested against production behaviour.
Phase 1a duplicates the same inline check at `run_energy_phase.py:280-288`.

### NUM-22 — `Eb` crossing zero in phase 1b is likewise only detected at segment endpoints

`make_energy_floor_event` exists and *is* attached in phase 1c
(`build_transition_phase_events`, l.532). Phase 1b, which is where `Eb` actually collapses, has no
energy event: `classify_energy_collapse(Eb)` runs on `sol.y[2,-1]` at
`run_energy_implicit_phase.py:1148`. Within the segment the RHS is evaluated at `Eb < 0` — see
NUM-11 for what `get_r1` does with that — and an excursion that dips negative and recovers inside a
single segment is never detected at all. This is the "event that can be missed by stepping over it"
case, and the machinery to catch it already exists two modules away.

### NUM-05 — `MAX_VELOCITY_EXPANSION` is dead; two of three runaway-event branches are unreachable

`phase_events.py:74` defines `MAX_VELOCITY_EXPANSION = 1000.0` pc/Myr. It has no other occurrence in
`trinity/`. All four phase builders call
`make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse")` (l.450, 490, 534, 572), so
the `"expansion"` and `"both"` branches of `make_velocity_runaway_event` (l.195-206) are exercised
only by `test/test_phase_events.py:36, 45`. **No run can terminate on outward velocity runaway.**

### Event sign conventions — checked, all correct

| factory | expression | `direction` | crossing detected | verdict |
|---|---|---|---|---|
| `make_min_radius_event` (l.120-125) | `R2 - min_r` | -1 | `R2` falling through `min_r` | ✔ |
| `make_max_radius_event` (l.152-157) | `R2 - max_r` | +1 | `R2` rising through `max_r` | ✔ |
| runaway `"collapse"` (l.190-193) | `v2 + v_max` | -1 | `v2` falling through `-v_max` | ✔ |
| runaway `"expansion"` (l.196-199) | `v_max - v2` | -1 | `v2` rising through `+v_max` (expr falls) | ✔ |
| runaway `"both"` (l.202-206) | `v_max - |v2|` | -1 | `|v2|` rising through `v_max` | ✔ |
| `make_cloud_boundary_event` (l.239-244) | `R2 - rCloud` | +1 | `R2` rising through `rCloud` | ✔ |
| `make_energy_floor_event` (l.274-279) | `Eb - floor` | -1 | `Eb` falling through floor | ✔ |
| `make_velocity_sign_event` (l.306-311) | `v2` | -1 | `v2` + → − | ✔ (definition; **consumption** is NUM-02) |
| `make_cooling_balance_event` (l.342-349) | `ratio - threshold` | -1 | ratio falling through threshold | ✔ (never attached, NUM-04) |

**Event roots, not step endpoints, are consumed.** All four runners read `event_result.t` and
`event_result.y`, which come from `sol.t_events[i][0]` / `sol.y_events[i][0]` — i.e. the brentq root
and `sol(te)`, not `sol.t[-1]`. **Cleared.**

**Same-step multi-event resolution among terminal events.** SciPy's `handle_events` sorts roots
ascending and truncates `active_events` at the first terminal one, so when several *terminal* events
fire in one step only the earliest is recorded. In phases 1a, 1c and 2 every event is terminal, so
exactly one entry of `t_events` can be non-empty and NUM-02(b) cannot bite there. **Cleared for
1a/1c/2.**

**`max_step` and event misses.** Phase 1a has no `max_step`, but its segment is only
`SEGMENT_DURATION = 3e-5` Myr (`run_energy_phase.py:54`) — shorter than the 2e-4 cap the other
phases use — and its three events (`R2` vs `rCloud`, `R2` vs `min_r`, `v2` vs `-500`) are all
monotone in the regime concerned. Sign-change detection between step endpoints is adequate.
**Cleared.**

---

## 4. Float equality on physical quantities

### Provably exact — cleared

- `shell_structure/shell_structure.py:285`: `if (phi_dust + phi_hydrogen) == 0.0:`.
  Both terms are `np.sum` of arrays whose every element is `-(positive)` (l.276-282: densities,
  cross-sections, `dr` all positive), so both are `≤ 0` and **cannot cancel**. The sum is exactly 0.0
  only in the degenerate case `dr_ion_arr.size == 0` (a single-point ionised region), where
  `np.sum([]) = 0.0` exactly. The ratio `phi_dust/(phi_dust+phi_hydrogen)` is scale-invariant, so
  even a denormal-magnitude sum divides safely. **Correct as written.**
- `alpha == 0` / `alpha != 0` (`get_InitCloudProp.py:169, 177, 180, 260`;
  `validate_gmc.py:408, 432, 562, 588, 601`; `density_profile.py:138`; `powerLawSphere.py:125, 256`;
  `mass_profile.py:312, 607`; `fkappa_auto.py:109`). `densPL_alpha` is a `.param` literal
  (`0`, `-2`), never computed, so the comparison is exact. The genuinely singular exponent
  `alpha = -3` **is** handled by an epsilon test — `powerLawSphere.py:143`,
  `if abs(3.0 + alpha) < 1e-14` — which is the right treatment. **Cleared.**
- `ZCloud == 1` / `== 0.15` (`read_param.py:426`; `read_cloudy.py:294, 297`; `registry.py:277`).
  `ZCloud` is parsed from a `.param` literal; `float("0.15") == 0.15` and `float("1.5e-1") == 0.15`
  are both exactly true in IEEE-754, and sweep expansion enumerates literals rather than computing
  them. `read_cloudy.py` raises a clear `ValueError` on any other value. **Cleared**, with one note:
  `read_param.py:426` has no `else`, so an unsupported `ZCloud` leaves `path_cooling_CIE` at its
  integer preset index and fails later as a `FileNotFoundError` on the path `"1"` — loud, but far
  from the cause.
- `compute_max_dex_change` (`run_energy_implicit_phase.py:315`, `run_transition_phase.py:151`,
  `run_momentum_phase.py:143`): `if old_val == 0 or new_val == 0: continue` guards `log10(0)`; exact
  zero is the only value that needs skipping and the fallback is to skip the key, not to fabricate a
  number. **Cleared.**
- `fA != 1.0` (`bubble_luminosity.py:436, 846`; `registry.py:137, 141`): `cooling_boost_fA` is a
  `.param` literal defaulting to `'1.0'`; the guard exists to keep the default path byte-identical,
  which is exactly what an exactness test should do here. **Cleared.**
- `math.isclose(shell_r[0], R2, rel_tol=1e-12)` (`snapshot_to_deck.py:155, 160`): `math.isclose`
  defaults `abs_tol=0.0`, so `rel_tol` is the operative term over the whole range — **no inert
  branch**. `R2 > 0` is enforced 40 lines earlier (l.134), so the degenerate `isclose(0,0) → True`
  case is unreachable. `1e-12` is ~4500 ulp, appropriate for a JSON round-trip identity check.
  **Cleared — correctly scoped.**

### No float-keyed dict lookups or `in`-against-float membership tests were found in `trinity/`.

---

## 5. NaN / inf handling

### NUM-06 — `abs(x) > 0` is a NaN-blind guard that converts a NaN residual into a **perfect** residual

`phase1b_energy_implicit/get_betadelta.py:478-481` (and the identical copy at 582-585), and
`:490-493` (copy at 590-593):

```python
if abs(Edot_from_beta) > 1e-300:
    Edot_residual = (Edot_from_beta - Edot_from_balance) / Edot_from_beta
else:
    Edot_residual = Edot_from_balance if abs(Edot_from_balance) > 0 else 0.0
...
if abs(T0) > 1e-300:
    T_residual = (T_bubble - T0) / T0
else:
    T_residual = T_bubble if abs(T_bubble) > 0 else 0.0
```

For any non-NaN `x`, `x if abs(x) > 0 else 0.0` **is** `x` (the only case `abs(x) == 0` gives is
`±0.0`). The conditional therefore has exactly one behavioural effect: it maps NaN to `0.0`.

```
abs(nan) > 1e-300  ->  False       # so a NaN Edot_from_beta takes the else branch
abs(nan) > 0       ->  False       # and a NaN Edot_from_balance becomes 0.0
residual from NaN  ->  0.0
total res          ->  0.0
```

`Edot_from_beta` and `Edot_from_balance` both depend on `Pb` from `compute_R1_Pb` (l.293), which
carries the `Eb ≤ 0` clamps of NUM-11; `T_bubble` comes from the bubble structure solve, whose
failure path fills `psoln` with NaN (`bubble_luminosity.py:483, 512, 515, 520`).

Consumption:

- `_solve_grid` (`get_betadelta.py:1085-1097`) computes `residual = Edot_res**2 + T_res**2`, keeps
  `if residual < best_residual` and **short-circuits the whole scan** on
  `if residual < GRID_EARLY_EXIT_RESIDUAL` (`= RESIDUAL_THRESHOLD/10 = 1e-5`). A point whose energy
  residual is *undefined* scores `0.0 + T_res²` and wins the grid.
- `_solve_lbfgsb` (`get_betadelta.py:1122-1127`) returns the same sum as the objective; `0.0` is the
  global minimum, so L-BFGS-B converges onto the NaN point.

**Mitigation that limits severity to S2:** the *default* solver is `betadelta_solver='hybr'`
(`registry.py:343`), and `_hybr_g_residual` (l.879-905) builds `gE` from the raw
`det.Edot_from_beta - det.Edot_from_balance` rather than from `Edot_residual`, so NaN propagates into
`g_total` and `g_total < RESIDUAL_THRESHOLD` is False — hybr correctly refuses to converge. The
defect is live on `betadelta_solver='legacy'` (a supported, validated value), and on **every**
solver it corrupts the reported `Edot_residual` / `T_residual` diagnostics written to the snapshot
(`_hybr_result`, l.907-928, stores `det.Edot_residual`).

### NUM-09 — the bubble RHS "T collapsed" detector is 9 decades below the physical scale, NaN-blind, and misses negative `T`

`bubble_structure/bubble_luminosity.py:418-423`:

```python
if np.abs(T - 0) < 1e-5:
    ...
    raise BubbleSolverError(f'temperature reached zero in bubble ODE RHS (T={T:.3e})')
```

`T` is in Kelvin. The integration starts at `_T_INIT_BOUNDARY = 3e4` K (l.52) and physical bubble
interiors span 1e4–1e8 K. A threshold of **1e-5 K** is nine orders below the lowest value the
profile is supposed to reach; the guard fires only once `T` is numerically indistinguishable from
zero. Three gaps:

1. `|nan| < 1e-5` is **False**, so a NaN temperature passes straight through into
   `ndens = Pb/(µ k_B T)` and `net_coolingcurve.get_dudt(...)`.
2. `|T| < 1e-5` is False for **negative** `T` of any magnitude, so `T = -1e6` K passes and yields a
   negative number density.
3. Between 1e-5 K and 1e4 K (the `_coolingswitch` floor, l.704) the RHS silently extrapolates the
   cooling tables.

The redundant `np.abs(T - 0)` (rather than `abs(T)`) suggests the intent was a two-sided band; a
positivity test `T <= _some_floor` would cover all three.

### NUM-19 — `np.any(T_array < 0)` cannot see a NaN profile

`bubble_luminosity.py:668`. `nan < 0` is False, so an all-NaN `psoln` passes this "unphysical-solution
net". Upstream `_ok` (l.657) does catch the solver-failure path that produces NaN, so this is
defence-in-depth that does not actually defend; low severity, but it is a NaN-blind detector by the
sweep's definition.

### `np.min` / `np.isnan` ordering in the dMdt residual — checked, correct by luck

`bubble_luminosity.py:370-378` tests `if min_T < _T_INIT_BOUNDARY:` **before**
`if np.isnan(min_T):`. This looks inverted, but `np.min` of an array containing NaN returns NaN and
`nan < 3e4` is False, so control falls through to the NaN branch as intended. **Cleared** — worth a
comment, not a change.

### `nanmax` on possibly-empty arrays — checked

- `_output/simulation_end.py:521-525`: guarded by `if old_arr.size == 0: return` at l.517, and the
  all-NaN result is caught by `if not np.isfinite(max_rel): max_rel = 0.0`. **Cleared.**
- `_functions/simplify.py:468, 520, 580`: reached only after array-size preconditions. **Cleared.**
- `_analysis/check_yesno.py:176, 178`: `np.nanmax` with no empty guard, but this is a standalone
  analysis CLI, not the physics path. Noted, not flagged.

---

## 6. Clamps and floors

### NUM-10 — `r2 += 1e-10` in `bubble_E2P` is a provable no-op

`bubble_structure/get_bubbleParams.py:220-224`:

```python
r1 *= cvt.pc2cm
r2 *= cvt.pc2cm
Eb *= cvt.E_au2cgs
# avoid division by zero
r2 += 1e-10
```

The `1e-10` is added **after** the pc→cm conversion. Verified:

```
r=   0.001 pc -> r_cm=3.085678e+15;  (r_cm + 1e-10) == r_cm  ->  True
r=    0.01 pc -> r_cm=3.085678e+16;  (r_cm + 1e-10) == r_cm  ->  True
r=     1.0 pc -> r_cm=3.085678e+18;  (r_cm + 1e-10) == r_cm  ->  True
r=   500.0 pc -> r_cm=1.542839e+21;  (r_cm + 1e-10) == r_cm  ->  True

crossover: +1e-10 is absorbed by rounding for r_cm >= 1e-10/eps = 4.5e5 cm = 1.46e-13 pc
```

`R2` is floored at 1.5 pc by the `min_radius` event — 13 orders above the crossover. The guard has
**never** changed a value and never will. Its only reachable effect is at exactly `r2 = 0`, where it
produces `shell_volume = 1e-30 cm³` and a correspondingly absurd `Pb` rather than an error. The real
protection is the `shell_volume <= 0` floor eight lines below.

### NUM-11 — `get_r1` maps *any* negative bubble energy to `+1e-30`

`get_bubbleParams.py:405-407`:

```python
# set minimum energy to avoid zero
if Ebubble < 1e-30:
    Ebubble = 1e-30
equation = np.sqrt( Lmech_total / v_mech_total / Ebubble * (r2**3 - r1**3) ) - r1
```

The test is `< 1e-30`, not `abs(...) < 1e-30`, so `Eb = -1e5` (a deeply unphysical, thoroughly dead
bubble) and `Eb = +1e-31` are treated **identically**. In the AU energy unit
(`Msun·pc²/Myr² ≈ 1.9e43 erg`) a live bubble carries `Eb ~ 1e5–1e8`; the sentinel is 35+ orders
below `ENERGY_FLOOR = 1e3`. With `Eb → 1e-30` the equation forces `r1 → R2`, then
`bubble_E2P`'s `shell_volume = r2³ - r1³ → 0` trips its own floor
(`shell_volume = 1e-13 * r2**3`, l.236) and `Pb` comes back inflated by ~1e13/(volume fraction).

`solve_R1` (l.436-443) explicitly gates `R2 <= 0`, `Lmech <= 0` and non-finite inputs, and its
docstring commits to "raises on root-finding failure … instead of fabricating a value" — but
`Eb < 0` slips past all of it and *is* fabricated, one function call deeper. Combined with NUM-22
(the `Eb ≤ 0` test only runs at segment endpoints), an entire phase-1b segment — up to 5e-2 Myr,
up to 2500 steps at `max_step = 2e-5` — can be integrated on this fabricated `R1`/`Pb`, and the
integrator's error control will happily accept those steps because they are smooth.

The `shell_volume = 1e-13 * r2**3` floor itself is **relative**, physically reasoned in-comment, and
documented as bit-identical on every physical bubble — that one is fine.

### NUM-08 — the dMdt penalty landscape changes sign and is non-monotonic in `min_T`

`bubble_luminosity.py:361-382` returns four different penalties to the same scalar `fsolve`:

| condition | returned value | line |
|---|---|---|
| RHS abort / non-finite IC / `not sol.success` | `+_SOLVER_FAIL_RESIDUAL = +1e3` | 333, 359, 361, 363 |
| `min_T < 3e4` | `residual * (3e4/(min_T + 1e-1))**2` — **signed**, magnitude 1.0 to 9e10 | 374 |
| `isnan(min_T)` | **`-1e3`** | 378 |
| `not monotonic(T_array)` | `+1e2` | 382 |

The design intent is stated at l.81-84: *"large and non-zero so fsolve is steered away from the
infeasible dMdt instead of falsely converging on a garbage (~0) residual."* Returning `+1e3` for one
failure mode and `-1e3` for another **manufactures a sign change inside the infeasible region**,
which is precisely the condition a root-finder hunts for.

The `min_T` multiplier is also non-monotonic, with a pole at `min_T = -0.1` K:

```
min_T=  2.99e+04  penalty multiplier = 1.007      <- essentially no penalty
min_T=     1e+04  penalty multiplier = 9
min_T=         1  penalty multiplier = 7.438e+08
min_T=         0  penalty multiplier = 9e+10
min_T=   -0.0999  penalty multiplier = 9e+16      <- pole
min_T=        -1  penalty multiplier = 1.111e+09
min_T=    -1e+04  penalty multiplier = 9
min_T=    -1e+06  penalty multiplier = 0.0009     <- SHRINKS the residual 1000x
```

A catastrophically failed profile (`min_T = -1e6` K) is penalised **less** than a marginal one, and
in fact has its residual damped by 1000×, making it look *better* than a healthy solve. `min_T < 0`
also reaches `min_T + 1e-1 == 0` for `min_T = -0.1` exactly → `ZeroDivisionError`/`inf`.
(NUM-09 explains why negative `min_T` is reachable: the RHS guard only rejects `|T| < 1e-5`.)

### NUM-18 — `EPSILON = 1e-100` absolute floors in the SPS feedback conversion

`sps/read_sps.py:35, 214-215, 233`:

```python
Mdot_wind     = pdot_wind_raw ** 2 / (2 * np.maximum(Lmech_wind_raw, EPSILON))
velocity_wind = 2 * Lmech_wind_raw / np.maximum(pdot_wind_raw,  EPSILON)
Mdot_SN       = 2 * Lmech_SN_raw   / np.maximum(velocity_SN_base ** 2, EPSILON)
```

The floor exists to survive a table row where the wind has switched off exactly. That case is
benign (`0²/2e-100 = 0`). The regime it does **not** survive is `Lmech = 0` with `pdot ≠ 0` — a
table inconsistency or a truncated interpolation — where `Mdot_wind` becomes `pdot² × 5e99`, a
finite, non-NaN, catastrophically wrong mass-loss rate that propagates into every downstream force
budget without tripping any `isfinite` gate. The neighbouring `Lmech_SN_raw = np.maximum(..., 0)`
(l.208) is preceded by an explicit `logger` warning (l.204-207); these three are not. A
zero-out-on-zero (`np.where(L > 0, pdot**2/(2*L), 0.0)`) is the regime-correct form.
The same 1e-100 pattern appears at `phase0_init/get_InitPhaseParam.py:38-40, 115-121, 136-138`.

### NUM-17 — `R2 = max(R2, 1e-10)` inside the momentum RHS

`phase2_momentum/run_momentum_phase.py:398` (and `mShell = max(mShell, 1e-10)` at l.415). Clamping
`R2` to 1e-10 pc makes `F_grav = G·mShell/R2²` roughly **1e20×** its physical value rather than
raising. The clamp is defensible — LSODA probes `R2 < 0` in rejected trial stages and the RHS must
stay finite — but the floor should be the value the physics already commits to,
`MIN_RADIUS_SAFETY = 0.01` pc (`phase_events.py:71`), not an arbitrary 1e-10; below the event
threshold the trajectory is meaningless anyway. `mShell`'s 1e-10 Msun floor is harmless (it only
zeroes `F_grav`) and comes from a per-segment snapshot, not a live state.

### NUM-21 — asymmetric additive guard in the dMdt residual denominator

`bubble_luminosity.py:368`: `residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)`. `v` is pc/Myr and
`v_array[0]` is `O(1–1e2)`, so the perturbation is `≲1e-6` relative — negligible in the normal
regime. But the guard is one-sided: it protects `v_array[0] → 0⁺` and *creates* a pole at
`v_array[0] = -1e-4` pc/Myr. `_get_bubble_ODE_initial_conditions` (l.401-403) computes
`v = cool_alpha·R2/t − dMdt·k_B·T/(4πR2²·µ·Pb)`, a difference of two positive terms, so a small
negative `v_array[0]` is exactly what an over-large trial `dMdt` produces during the `fsolve` probe.
`np.copysign(max(abs(v0), 1e-4), v0)` would be the symmetric form.

### NUM-23 — a single-step non-monotonicity of *any* depth is tolerated

`_functions/operations.py:129-131`:

```python
if end - start == 1:
    # isolated single point: a numerical glitch, never a physical inversion
    # -> tolerate regardless of depth
    continue
```

`MONOTONIC_RTOL = 1e-2` (l.94) bounds multi-step wiggles at 1% relative drawdown, but a single-step
run bypasses the depth test entirely. `_is_monotonic_or_tolerable` gates `find_nearest_higher`, which
is what locates the CIE / conduction / cooling-zone boundaries in the bubble luminosity integral, so
tolerating an unbounded single-step drop admits a profile whose region split can land anywhere. The
`np.all(np.isfinite(L))` precondition (l.110) and the fact that a *tail* collapse produces a long
wrong-direction run (not a single step) make this narrow, but the "any depth" exemption is stated
rather than bounded. Note this is the guard CLAUDE.md flags as numpy-2.x-version-sensitive.

### Small-magnitude clamps checked and cleared

- `bubble_luminosity.py:603` `np.maximum(avg_magnitude, 1e-30)` — a relative-difference denominator
  built from `0.5*(|r[:-1]| + |r[1:]|)`, all positive radii; the floor is unreachable and harmless.
- `bubble_luminosity.py:1007` `np.maximum(np.abs(cmax), 1e-300)` — inside a diagnostic gated on
  `TRINITY_BUBBLE_DIAG`; never on the physics path.
- `get_betadelta.py:266` `if abs(denominator) < 1e-300: return 0.0` — `d_coeff*(1-c_frac)` with
  `d_coeff = R2³ - R1³`; the return-0 fallback is the correct `Ebdot` in the degenerate-volume limit.
- `simplify.py:521-522, 538, 844` — plotting/thinning tolerances; affect which snapshot rows are
  retained in `dictionary.jsonl`, not the trajectory.
- `_input/dictionary.py:620, 645-697` `eps = 1e-300` for `log10` — output-thinning only, but note a
  non-positive value maps to `log10(1e-300) = -300`, a large outlier in the curvature metric that
  drives row selection.
- `_output/simulation_end.py:521` `np.maximum(np.abs(old_arr), 1e-30)` — comparison table only.
- `phase1b_energy_implicit/run_energy_implicit_phase.py:872, 906` `max(Lgain, 1e-300)` — inside
  f-strings for log messages only.

---

## Clearances (checked and found correctly scoped)

1. **`math.isclose(..., rel_tol=1e-12)`** at `_output/cloudy/snapshot_to_deck.py:155, 160` —
   `math.isclose` has `abs_tol=0.0` by default, so the relative tolerance is operative across the
   whole range; `R2 > 0` is pre-enforced. No inert branch.
2. **`brentq(get_r1, 0.0, R2)`** at `get_bubbleParams.py:436-460` — full bracket with an argued
   proof of sign change, an `isfinite` pre-gate added specifically to defeat the scipy<1.11
   NaN-poisoning behaviour, and a re-raise instead of a fabricated value. The best-guarded
   root-find in the package.
3. **`brentq(fT_interp, ..., xtol=1e-8)`** at `bubble_luminosity.py:724` — 4 orders looser than the
   scipy default, but the bracketed function is a cubic interpolant over a ~1e-5 pc grid, so the
   interpolation error dominates by ~3 orders and `xtol` does not bind.
4. **`_BUBBLE_ATOL = 1e-10` with `_BUBBLE_RTOL = 1e-8`** — crossover `atol/rtol = 1e-2`. `T` is
   floored at `3e4` K and `dTdr` is large, so `atol` is inert for both; it binds only on `v`, which
   is exactly the component driven to zero by the boundary condition the residual enforces. Correct
   by design. The deliberate `_RESIDUAL_RTOL = 1e-6` for the locating solve (crossover 1e-4) is
   documented, measured (`≤0.3%` dMdt shift) and justified at l.94-99.
5. **`np.min` before `np.isnan`** at `bubble_luminosity.py:370-378` — reads as inverted but is
   correct: `nan < 3e4` is False, so control reaches the NaN branch.
6. **`(phi_dust + phi_hydrogen) == 0.0`** at `shell_structure.py:285` — provably exact; both summands
   are non-positive so cannot cancel, and exact 0.0 arises only from `np.sum([])`.
7. **`alpha == 0` family, `ZCloud == 1/0.15`, `fA != 1.0`** — all compare `.param` literals that are
   never computed; the one genuinely singular exponent (`alpha = -3`) is handled with an epsilon test
   at `powerLawSphere.py:143`.
8. **Event sign conventions** — all nine factories have `direction` matching the physical transition
   (table in §3). The `"expansion"` runaway's `v_max - v2` with `direction=-1` is correct despite
   looking inverted.
9. **Event roots are consumed, not step endpoints** — all four runners use `sol.t_events[i][0]` /
   `sol.y_events[i][0]`.
10. **Same-step terminal-event resolution in phases 1a, 1c, 2** — every event in those lists is
    terminal, and SciPy truncates at the earliest terminal root, so NUM-02(b) cannot manifest there.
11. **Phase 1a's missing `max_step`** — compensated by a 3e-5 Myr segment, shorter than the 2e-4 cap
    used elsewhere; all three of its events are monotone in the regime concerned.
12. **`t_previousCoolingUpdate` default `1e30`** (`registry.py:455`) — makes the first-pass
    `abs(t_prev - t_now) > COOLING_UPDATE_INTERVAL` test at `run_energy_phase.py:124` always fire,
    so the non-CIE cooling structure is always built. (`DescribedItem.__sub__` at
    `dictionary.py:176` makes the missing `.value` there work; inconsistent with
    `run_energy_implicit_phase.py:783` but not a defect.)
13. **`nanmax` guards** at `simulation_end.py:521-525` — empty-array and all-NaN paths both handled.
14. **`shell_volume = 1e-13 * r2**3`** at `get_bubbleParams.py:236` — a *relative* floor with a
    documented regime and a bit-identical claim on physical bubbles; the right shape of clamp.

---

```json
[
  {"id":"NUM-01","file":"trinity/_output/trinity_reader.py","line":721,"class":"numerical","severity":"S3","claim":"np.isclose(times, t, rtol=1e-10) leaves atol at its 1e-8 default; rtol only binds for t > 100 Myr, which stop_t=15 caps out, so the intended relative tolerance is inert on every run and a flat 1e-8 Myr (10 yr) absolute window governs.","evidence":"Crossover 1e-10*|t| > 1e-8 requires |t| > 100 Myr. registry.py:353 stop_t default '15' Myr. Executed: np.isclose(1e-3, 1e-3+5e-9, rtol=1e-10) -> True; np.isclose(5e-9, 1e-9, rtol=1e-10) -> True.","expected":"An explicit atol matched to the snapshot cadence (phase 1a segments are 3e-5 Myr), e.g. np.isclose(times, t, rtol=0, atol=1e-12), or an exact-index lookup.","failure_scenario":"TrinityOutput.get_at_time(t) reports a snapshot up to 10 yr away as an exact match and silently skips the interpolate/closest branch; in phase 1a (total duration 3e-3 Myr) that is 0.3% of the phase.","repro":"python3 -c \"import numpy as np; print(np.isclose(5e-9,1e-9,rtol=1e-10))\"","confidence":"high"},
  {"id":"NUM-02","file":"trinity/phase_general/phase_events.py","line":392,"class":"state","severity":"S1","claim":"check_event_termination returns the first event by LIST INDEX with a recorded root and never consults event.terminal; because velocity_sign is non-terminal and sits at index 0 of the implicit-phase list, any downward v2 zero-crossing ends phase 1b at the crossing and pre-empts whichever terminal event actually stopped the solve.","evidence":"phase_events.py:392-405 iterates enumerate(zip(sol.t_events, sol.y_events)) and returns on the first non-empty entry. build_implicit_phase_events (l.487-491) puts make_velocity_sign_event() at index 0 with terminal=False (l.310) and the module docstring calls it 'monitoring only' (l.25-26). run_energy_implicit_phase.py:1095-1118 breaks the segment loop on any triggered result. Executed with scipy 1.17.1: t_events[0]=[0.25] (velocity_sign), t_events[1]=[1.15139] (terminal min_radius, which actually stopped the solve at R2=1.5); the loop reports velocity_sign at t=0.25 with is_simulation_ending=False and y=[3.125, 2.17e-19].","expected":"Select the event with the smallest root among those with getattr(event,'terminal',True) true, and treat non-terminal events as recordable-only.","failure_scenario":"A phase-1b bubble that stalls (v2 crosses zero) is truncated at the stall and force-routed 1b -> 1c -> momentum with EndSimulationDirectly and isCollapse unset; if it also reached min_r in that segment the collapse is neither recorded nor terminated and the state is rewound to the stall radius. Corroboration: the collapse detector at run_energy_implicit_phase.py:1301-1303 sits after the break and is therefore unreachable for the entering-collapse case it was written for.","repro":"Standalone scipy reproduction (no trinity import) in the report body: solve_ivp(rhs,(0,1.5),[3.0,1.0],events=[vsign,minrad],rtol=1e-6,atol=1e-8,max_step=2e-4) with vsign.terminal=False,direction=-1 at index 0 and minrad.terminal=True,direction=-1 at index 1.","confidence":"high"},
  {"id":"NUM-03","file":"trinity/phase_general/phase_events.py","line":627,"class":"state","severity":"S2","claim":"isCollapse is set by the substring test \"'radius' in result.reason_code.lower()\", which matches 'large_radius_event' as well as 'small_radius_event', so a shell that expands past stop_r is recorded as collapsing.","evidence":"phase_events.py:626-629; reason codes set at l.128 ('small_radius_event') and l.160 ('large_radius_event'). Executed: 'radius' in 'large_radius_event'.lower() -> True. Rendered by _output/show_run.py:87-90 as collapsed='collapsing' and by _input/sweep_runner.py:560-561 as collapsed='yes'.","expected":"Key on result.end_code (SimulationEndCode.SHELL_COLLAPSED) rather than a substring of a human-readable reason code.","failure_scenario":"Any run terminating on max_radius (stop_r default 500 pc, v2 > 0 at exit) is written out with isCollapse=True and reported as collapsed in the sweep summary table, inverting the recorded fate.","repro":"python3 -c \"print('radius' in 'large_radius_event'.lower())\"","confidence":"high"},
  {"id":"NUM-04","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":752,"class":"deadcode","severity":"S3","claim":"make_cooling_balance_event builds a fully-formed terminal solve_ivp event, build_implicit_phase_events returns the factory and phase 1b unpacks it into cooling_balance_factory, but the name is never used again; the phase-1b -> 1c transition is detected only by an inline test at segment boundaries.","evidence":"phase_events.py:319-356 defines the event (terminal=True, direction=-1); l.497,501 return the factory; run_energy_implicit_phase.py:752 unpacks it and grep shows no further occurrence in the file. The live check is run_energy_implicit_phase.py:1296 (inline ratio test after the ODE returns). Phase 1a duplicates it at run_energy_phase.py:280-288.","expected":"Either attach the per-segment event to solve_ivp (Lgain/Lloss are segment constants, so the closure is valid) or delete the factory and the unused unpacking.","failure_scenario":"The primary physical transition of the run is located to one segment rather than to the event root, i.e. up to DT_SEGMENT_MAX = 5e-2 Myr late; the transition time recorded in the output is quantised to the adaptive segment grid.","repro":"grep -n 'cooling_balance_factory' trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","confidence":"high"},
  {"id":"NUM-05","file":"trinity/phase_general/phase_events.py","line":74,"class":"deadcode","severity":"S4","claim":"MAX_VELOCITY_EXPANSION = 1000.0 pc/Myr has no consumer, and the 'expansion' and 'both' branches of make_velocity_runaway_event are unreachable from production code, so no run can terminate on outward velocity runaway.","evidence":"All four builders call make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction='collapse') at phase_events.py:450, 490, 534, 572. The only other references to the constant or to the two branches are test/test_phase_events.py:36, 45.","expected":"Either attach an expansion-direction runaway event or remove the constant and the unused branches.","failure_scenario":"A numerically unstable outward runaway (v2 -> +1e4 pc/Myr) is not caught by an event and only stops on max_radius or on a solver failure.","repro":"grep -rn 'MAX_VELOCITY_EXPANSION' trinity/","confidence":"high"},
  {"id":"NUM-06","file":"trinity/phase1b_energy_implicit/get_betadelta.py","line":481,"class":"silent-failure","severity":"S2","claim":"The residual fallbacks 'x if abs(x) > 0 else 0.0' are no-ops for every non-NaN x; their only behavioural effect is to convert a NaN residual into exactly 0.0, i.e. a perfect score for the beta-delta root finder.","evidence":"get_betadelta.py:478-481 and 490-493 (duplicated at 582-585 and 590-593). Executed: abs(nan) > 1e-300 -> False; abs(nan) > 0 -> False; so both branches yield 0.0 and Edot_res**2 + T_res**2 = 0.0. Consumed by _solve_grid (l.1085-1097: 'if residual < best_residual' and early exit at GRID_EARLY_EXIT_RESIDUAL = 1e-5) and _solve_lbfgsb (l.1122-1127). NaN sources: Pb via compute_R1_Pb (l.293) under the Eb<=0 clamps, and bubble_T_r_Tb from the NaN-filled psoln at bubble_luminosity.py:483,512,515,520.","expected":"Guard with np.isfinite and return a large positive penalty (or raise) on non-finite components, mirroring _hybr_g_residual's _NoPhysicalRoot gate.","failure_scenario":"On betadelta_solver='legacy' a grid point whose energy residual is undefined scores 0.0 + T_res**2, wins the grid, and short-circuits the scan, so a NaN-poisoned (beta, delta) is accepted as converged. On the default 'hybr' solver g_total propagates NaN and correctly fails to converge, but the reported Edot_residual/T_residual diagnostics written to the snapshot are still 0.0.","repro":"python3 -c \"x=float('nan'); print(x if abs(x)>0 else 0.0)\"","confidence":"high"},
  {"id":"NUM-07","file":"trinity/bubble_structure/bubble_luminosity.py","line":261,"class":"silent-failure","severity":"S2","claim":"The fsolve that determines the bubble evaporative mass flux dMdt discards its convergence flag: full_output is not requested, ier/mesg are never checked, and maxfev is unset.","evidence":"bubble_luminosity.py:261-267 takes scipy.optimize.fsolve(...)[0] directly. On non-convergence MINPACK returns the last iterate with only a RuntimeWarning. The downstream gate _usable_dMdt (get_betadelta.py:381-386) tests only isfinite and > 0, which a stranded iterate passes; its own docstring concedes 'fsolve stranded on the penalty plateau returning garbage'.","expected":"full_output=True, then treat ier != 1 as a solve failure (raise BubbleSolverError or return the ok=False contract the module already has).","failure_scenario":"A non-converged dMdt is used as the physical shell->bubble mass flux, setting the whole bubble structure, its luminosity, and the beta-delta residual for that segment, with no signal in the output.","repro":"sed -n '259,268p' trinity/bubble_structure/bubble_luminosity.py","confidence":"high"},
  {"id":"NUM-08","file":"trinity/bubble_structure/bubble_luminosity.py","line":374,"class":"numerical","severity":"S2","claim":"The dMdt fsolve receives four failure penalties of inconsistent sign and scale (+1e3, -1e3, +1e2, and a signed multiplier spanning 1.0 to 9e10), manufacturing a sign change inside the infeasible region; the min_T multiplier is non-monotonic with a pole at min_T = -0.1 K and DAMPS the residual for deeply negative min_T.","evidence":"bubble_luminosity.py:333/359/361/363 return +_SOLVER_FAIL_RESIDUAL=+1e3; l.374 returns residual*(3e4/(min_T+1e-1))**2; l.378 returns -1e3; l.382 returns +1e2. Executed multipliers: min_T=2.99e4 -> 1.007; 1e4 -> 9; 1 -> 7.4e8; 0 -> 9e10; -0.0999 -> 9e16; -1 -> 1.1e9; -1e4 -> 9; -1e6 -> 0.0009. The design intent stated at l.81-84 is explicitly to avoid a near-zero residual in the infeasible region.","expected":"One-signed penalties of uniform magnitude (e.g. all +1e3), and a monotone, pole-free min_T ramp such as (3e4/max(min_T, 1.0))**2 with the sign of residual preserved.","failure_scenario":"fsolve's Broyden/FD Jacobian sees the -1e3 / +1e3 discontinuity as a bracketed root and converges onto the boundary between two failure modes; separately, a profile with min_T = -1e6 K has its residual shrunk 1000x and scores better than a marginal but usable one. min_T = -0.1 exactly raises ZeroDivisionError / yields inf.","repro":"python3 -c \"T0=3e4;\\nfor m in (1e4,1,-0.0999,-1e4,-1e6): print(m,(T0/(m+1e-1))**2)\"","confidence":"medium"},
  {"id":"NUM-09","file":"trinity/bubble_structure/bubble_luminosity.py","line":418,"class":"regime","severity":"S2","claim":"The bubble-ODE 'temperature collapsed' detector tests |T| < 1e-5 K, nine decades below the 3e4-1e8 K range the profile occupies; it is NaN-blind and does not fire for negative T of any magnitude.","evidence":"bubble_luminosity.py:418-423; the integration starts at _T_INIT_BOUNDARY = 3e4 K (l.52) and the cooling tables floor at _coolingswitch = 1e4 K (l.704). abs(nan) < 1e-5 is False; abs(-1e6) < 1e-5 is False. Downstream the RHS computes ndens = Pb/(mu*k_B*T) (l.427), which is negative for T<0 and NaN-propagating for T=NaN, and calls net_coolingcurve.get_dudt with it.","expected":"A positivity/floor test such as 'if not (T > _coolingswitch)' (which also catches NaN, since nan > x is False), rather than a two-sided band at 1e-5 K.","failure_scenario":"A dying trial solve whose T goes NaN or negative continues to be integrated instead of aborting to the ok=False contract; the cooling-table lookup is called with a negative density and/or a negative temperature.","repro":"python3 -c \"import numpy as np; print(np.abs(np.nan-0)<1e-5, np.abs(-1e6-0)<1e-5)\"","confidence":"medium"},
  {"id":"NUM-10","file":"trinity/bubble_structure/get_bubbleParams.py","line":224,"class":"deadcode","severity":"S4","claim":"The 'avoid division by zero' guard r2 += 1e-10 is applied after the pc->cm conversion, so it is absorbed by floating-point rounding for every radius above 1.46e-13 pc and has never changed a value.","evidence":"get_bubbleParams.py:220-224 converts r2 *= cvt.pc2cm then adds 1e-10. Executed: for r_pc in (1e-3, 0.01, 1.0, 500.0), (r_cm + 1e-10) == r_cm is True in every case. Crossover: 1e-10/eps = 4.5e5 cm = 1.46e-13 pc. R2 is floored at min_r = 1.5 pc by the min_radius event (phase_events.py:445).","expected":"Delete it; the real protection is the shell_volume <= 0 floor at l.235-236, which is a correctly-scaled relative floor.","failure_scenario":"None in practice. Its only reachable effect, at exactly r2 = 0, is to produce shell_volume = 1e-30 cm^3 and a finite absurd Pb instead of an error.","repro":"python3 -c \"c=3.0856775814913673e18; print((1.0*c+1e-10)==1.0*c)\"","confidence":"high"},
  {"id":"NUM-11","file":"trinity/bubble_structure/get_bubbleParams.py","line":406,"class":"sign","severity":"S2","claim":"get_r1 clamps with 'if Ebubble < 1e-30: Ebubble = 1e-30', not on |Ebubble|, so any NEGATIVE bubble energy is silently replaced by a positive 1e-30 sentinel and a collapsed bubble is handed back a finite R1 and Pb.","evidence":"get_bubbleParams.py:405-407. Eb in AU units (Msun*pc^2/Myr^2 ~ 1.9e43 erg) is ~1e5-1e8 for a live bubble and ENERGY_FLOOR is 1e3, so 1e-30 is 33+ orders below the floor. With Eb -> 1e-30 the equation sqrt(L/(v*Eb)*(R2^3-r1^3)) = r1 forces r1 -> R2, which then trips bubble_E2P's shell_volume floor at l.236. solve_R1 (l.436-443) gates R2<=0, Lmech<=0 and non-finite inputs and promises 'raises ... instead of fabricating a value', but has no Eb<0 gate.","expected":"Reject Eb <= 0 in solve_R1 with the same explicit raise used for non-finite inputs, letting the phase's collapse routing (classify_energy_collapse) handle it.","failure_scenario":"Combined with NUM-22 (Eb<=0 only tested at segment endpoints), a whole phase-1b segment (up to 5e-2 Myr, up to 2500 steps at max_step=2e-5) is integrated on a fabricated R1~R2 and an inflated Pb; the integrator accepts those steps because the fabricated RHS is smooth. Eb = -1e5 and Eb = +1e-31 produce identical output.","repro":"sed -n '400,412p' trinity/bubble_structure/get_bubbleParams.py","confidence":"high"},
  {"id":"NUM-12","file":"trinity/shell_structure/shell_structure.py","line":182,"class":"numerical","severity":"S2","claim":"The ionization-front termination test phi <= 1e-9 sits 14.9x BELOW the odeint integration's own default absolute tolerance of 1.4901161193847656e-08, which governs the phi component over its entire [0,1] range.","evidence":"shell_structure.py:165-168 calls odeint with no rtol/atol, so both default to sqrt(eps) = 1.4901161193847656e-08 (verified against scipy 1.17.1). phi is initialised to exactly 1 at l.118, so the allowed local error atol + rtol*|phi| >= 1.49e-8 everywhere; the crossover where rtol would take over is phi = 1, the initial value. 1.49e-8 / 1e-9 = 14.9. The clamps max(0.0, phi) at get_shellODE.py:111 and shell_structure.py:204, 229 exist because phi rings negative at this level.","expected":"Set atol on the phi component (odeint accepts a per-component atol array) to at least 1e-11, or raise the depletion threshold above the integrator's error bar.","failure_scenario":"R_IF, n_IF, f_esc_ion, the Stromgren balance density and the ionised/neutral split of the shell are all keyed on the radius where phi first crosses 1e-9, which is set by accumulated integration error rather than by the ODE; the same input can land the front in a different cell.","repro":"python3 -c \"import numpy as np; print(np.sqrt(np.finfo(float).eps)/1e-9)\"","confidence":"high"},
  {"id":"NUM-13","file":"trinity/shell_structure/shell_structure.py","line":165,"class":"silent-failure","severity":"S2","claim":"Both shell odeint calls omit full_output=True, so an exhausted step budget silently truncates the shell integration with no detectable signal.","evidence":"shell_structure.py:165-168 and 324-327. The module comment at l.28-34 records that this exact failure already happened with the default mxstep=500 ('Excess work done on this call' and 'silently truncates the shell integration'); the fix raised _SHELL_ODE_MXSTEP to 50000 (l.35) but added no success check.","expected":"full_output=True and treat infodict['message'] != 'Integration successful.' as an error.","failure_scenario":"If 50000 steps are exhausted in a stiff regime the integration truncates mid-slice and the truncated nShell/phi/tau arrays feed the mass, force and optical-depth budgets as if complete.","repro":"sed -n '163,170p' trinity/shell_structure/shell_structure.py","confidence":"high"},
  {"id":"NUM-14","file":"trinity/phase1_energy/run_energy_phase.py","line":59,"class":"numerical","severity":"S3","claim":"Phase 1a and phase 1c integrate the same state vector [R2, v2, Eb] with different absolute tolerance (1e-9 vs 1e-8), different method (RK45 vs LSODA) and no vs 2e-4 Myr max_step, with no documented reason; the scalar atol is inert for Eb and T0 over their whole ranges and binds only on v2 near zero, where 1e-8 pc/Myr is ~2 orders below any meaningful velocity resolution.","evidence":"run_energy_phase.py:58-59 (RTOL=1e-6, ATOL=1e-9, no max_step, l.299-310); run_transition_phase.py:132-135 (1e-6/1e-8/2e-4/min_step 1e-6); also run_energy_implicit_phase.py:170-173 and run_momentum_phase.py:124-127. atol binds where |y_i| < atol/rtol: 1e-3 in phase 1a, 1e-2 in 1b/1c/2. R2 >= min_r = 1.5 pc (coll_r default 1, phase_events.py:445) so atol never binds on R2; ENERGY_FLOOR = 1e3 so rtol*Eb >= 1e-3 >> atol; T0 ~ 1e6 K. Only v2 crosses zero.","expected":"One shared tolerance block for the shared state vector, with a per-component atol array whose v2 entry reflects a physically meaningful velocity floor (e.g. 1e-4 pc/Myr) rather than 1e-8.","failure_scenario":"The same physical state integrated across the 1a/1c boundary is held to different accuracy; near stall the solver is asked for ~1e-8 pc/Myr absolute accuracy on v2, which forces the step size down and (with min_step = 1e-6 Myr) can convert a turnaround into a 'solver_failed' phase termination.","repro":"grep -n 'RTOL\\|ATOL\\|ODE_RTOL\\|ODE_ATOL\\|ODE_MAX_STEP\\|ODE_MIN_STEP' trinity/phase1_energy/run_energy_phase.py trinity/phase1b_energy_implicit/run_energy_implicit_phase.py trinity/phase1c_transition/run_transition_phase.py trinity/phase2_momentum/run_momentum_phase.py","confidence":"high"},
  {"id":"NUM-15","file":"trinity/phase1c_transition/run_transition_phase.py","line":135,"class":"citation","severity":"S4","claim":"The ODE_MAX_STEP comment in phases 1c and 2 states 'Max step = 2e-5 Myr' but the expression evaluates to 2e-4 Myr, a 10x error copied from phase 1b.","evidence":"run_transition_phase.py:135 and run_momentum_phase.py:127 both read 'ODE_MAX_STEP = DT_SEGMENT_MIN / 5  # Max step = 2e-5 Myr'. DT_SEGMENT_MIN = 1e-3 at run_transition_phase.py:94 and run_momentum_phase.py:87, so the value is 2e-4. Only phase 1b (DT_SEGMENT_MIN = 1e-4, l.113) actually yields 2e-5.","expected":"Correct the comment to 2e-4 Myr, or state the cap in terms of DT_SEGMENT_MIN so it cannot drift.","failure_scenario":"A future tuning pass reasons from the comment and believes the momentum phase is capped 10x tighter than it is.","repro":"python3 -c \"print(1e-3/5, 1e-4/5)\"","confidence":"high"},
  {"id":"NUM-16","file":"trinity/bubble_structure/bubble_luminosity.py","line":264,"class":"numerical","severity":"S4","claim":"The dMdt fsolve pairs xtol=1e-4 with epsfcn=1e-4, but MINPACK's forward-difference step is sqrt(max(epsfcn, eps_mach))*|x| = 1e-2*|dMdt|, so the Jacobian probe is 100x coarser than the requested solution accuracy.","evidence":"bubble_luminosity.py:264-266. MINPACK fdjac1 uses eps = sqrt(max(epsfcn, epsmch)); h = eps*abs(x). sqrt(1e-4) = 1e-2. The in-code comment at l.94-99 justifies only the loose xtol, not epsfcn.","expected":"Either tighten epsfcn to ~1e-8 (FD step 1e-4, matching xtol) or document why a 1% Jacobian probe is adequate for the residual's noise floor.","failure_scenario":"Slow or oscillatory convergence of the dMdt root, which - combined with the unchecked ier of NUM-07 - is invisible.","repro":"python3 -c \"import math; print(math.sqrt(1e-4))\"","confidence":"medium"},
  {"id":"NUM-17","file":"trinity/phase2_momentum/run_momentum_phase.py","line":398,"class":"numerical","severity":"S3","claim":"The momentum ODE RHS clamps R2 to 1e-10 pc, 10 orders below the min_radius event threshold, so a non-physical trial radius yields a gravitational force ~1e20x too large instead of a controlled failure.","evidence":"run_momentum_phase.py:398 'R2 = max(R2, 1e-10)', consumed at l.417 'F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)'. MIN_RADIUS_SAFETY = 0.01 pc (phase_events.py:71) and min_r = max(coll_r*1.5, 0.01) = 1.5 pc with defaults.","expected":"Clamp to MIN_RADIUS_SAFETY (0.01 pc) - below the event threshold the trajectory is already outside the model's validity, so a large-but-bounded force is preferable to a 1e20 spike.","failure_scenario":"An LSODA trial stage that probes R2 <= 0 produces an F_grav spike that distorts the local error estimate and the step-size controller, even though the step is later rejected.","repro":"sed -n '396,400p' trinity/phase2_momentum/run_momentum_phase.py","confidence":"medium"},
  {"id":"NUM-18","file":"trinity/sps/read_sps.py","line":214,"class":"regime","severity":"S3","claim":"EPSILON = 1e-100 is used as an absolute divide-by-zero floor on Lmech and pdot; it survives the exact-zero case correctly but converts an inconsistent table row (Lmech = 0 with pdot != 0) into a finite, non-NaN mass-loss rate ~1e99 that no isfinite gate catches.","evidence":"read_sps.py:35 defines EPSILON = 1e-100; l.214 Mdot_wind = pdot**2/(2*max(Lmech, EPSILON)); l.215 velocity_wind = 2*Lmech/max(pdot, EPSILON); l.233 Mdot_SN = 2*Lmech_SN/max(v_SN**2, EPSILON). The neighbouring clamp at l.208 (np.maximum(Lmech_SN_raw, 0)) is preceded by an explicit logger warning at l.204-207; these three are silent. Same pattern at phase0_init/get_InitPhaseParam.py:38-40, 115-121, 136-138.","expected":"np.where(L > 0, pdot**2/(2*L), 0.0) - a regime test rather than an absolute floor 100 orders below any physical value.","failure_scenario":"A truncated or mis-mapped SPS column giving pdot > 0 at Lmech = 0 produces Mdot_wind ~ pdot^2 * 5e99, which propagates through every force term as a finite number and passes all downstream isfinite checks.","repro":"python3 -c \"import numpy as np; print(1.0**2/(2*np.maximum(0.0,1e-100)))\"","confidence":"medium"},
  {"id":"NUM-19","file":"trinity/bubble_structure/bubble_luminosity.py","line":668,"class":"silent-failure","severity":"S4","claim":"The 'unphysical-solution net' np.any(T_array < 0) is NaN-blind: nan < 0 is False, so an all-NaN temperature profile passes the check.","evidence":"bubble_luminosity.py:668-671. The NaN profiles are produced at l.483, 512, 515, 520 (np.full(..., np.nan) on the failure paths).","expected":"np.any(~(T_array > 0)) or an explicit np.isfinite check alongside the sign test.","failure_scenario":"None currently reachable - _ok is checked at l.657 before this line - but the guard provides no defence for a future path where sol.success is True yet the dense output evaluates to NaN.","repro":"python3 -c \"import numpy as np; print(np.any(np.array([np.nan,np.nan])<0))\"","confidence":"high"},
  {"id":"NUM-20","file":"trinity/phase0_init/get_InitCloudProp.py","line":530,"class":"deadcode","severity":"S4","claim":"verify_key_radii_in_array's np.isclose checks can never fail, because _create_radius_array inserts rCore and rCloud bit-exactly two calls earlier; the warning branches are unreachable.","evidence":"get_InitCloudProp.py:530-531 tests np.any(np.isclose(r_arr, rCloud/rCore)); _create_radius_array (l.412-455) ends with r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud]))). Same for the l.505 searchsorted+isclose pair. Tolerance arithmetic (not the defect, recorded for completeness): np.isclose defaults rtol=1e-5/atol=1e-8, crossover at |b| > 1e-3 pc, so rtol governs for both radii; at rCloud=20 pc the window is 2e-4 pc against a log-grid spacing of ~0.199 pc, ~1000x wider.","expected":"Either delete the vacuous verification or move it to a test that constructs an array without the insertion.","failure_scenario":"None. The function reports a postcondition guaranteed by construction, so it gives false assurance that the grid contains the key radii for a reason other than the append.","repro":"sed -n '440,455p' trinity/phase0_init/get_InitCloudProp.py","confidence":"high"},
  {"id":"NUM-21","file":"trinity/bubble_structure/bubble_luminosity.py","line":368,"class":"numerical","severity":"S4","claim":"The dMdt residual denominator uses a one-sided additive guard, v_array[0] + 1e-4, which protects v0 -> 0+ but creates a pole at v0 = -1e-4 pc/Myr - exactly the sign an over-large trial dMdt produces.","evidence":"bubble_luminosity.py:368 'residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)'. The initial velocity is v = cool_alpha*R2/t_now - dMdt*k_B*T/(4*pi*R2**2*mu_ion*Pb) (l.401-403), a difference of two positive terms, so v0 < 0 for large trial dMdt. In the normal regime v0 is O(1-1e2) pc/Myr and the guard is a <=1e-6 relative perturbation.","expected":"np.copysign(max(abs(v0), 1e-4), v0), or reject the trial when |v0| is below the guard scale.","failure_scenario":"A trial dMdt landing near v0 = -1e-4 gives a residual with a spurious magnitude and sign, feeding the already-signed penalty landscape of NUM-08.","repro":"sed -n '365,370p' trinity/bubble_structure/bubble_luminosity.py","confidence":"medium"},
  {"id":"NUM-22","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":1148,"class":"state","severity":"S2","claim":"Phase 1b detects the Eb -> 0 collapse only from sol.y[2,-1] at segment endpoints, even though make_energy_floor_event exists and is attached in phase 1c; an Eb excursion that dips negative and recovers within one segment is never detected.","evidence":"run_energy_implicit_phase.py:1148 calls classify_energy_collapse(Eb) on the post-segment value. build_transition_phase_events (phase_events.py:531-532) attaches make_energy_floor_event(energy_floor, y_index=2) for phase 1c; build_implicit_phase_events (l.487-495) attaches no energy event. Segment length runs from DT_SEGMENT_MIN = 1e-4 to DT_SEGMENT_MAX = 5e-2 Myr (l.113-114).","expected":"Attach make_energy_floor_event(0.0 or ENERGY_HANDOFF_FLOOR, y_index=2) to the phase-1b event list so the crossing is located by root-finding.","failure_scenario":"During the segment in which Eb goes negative, the RHS is evaluated at Eb < 0 and get_r1 substitutes +1e-30 (NUM-11), producing a fabricated R1 -> R2 and an inflated Pb for up to 5e-2 Myr of accepted trajectory; a dip-and-recover excursion leaves no trace at all.","repro":"grep -n 'classify_energy_collapse\\|make_energy_floor_event' trinity/phase1b_energy_implicit/run_energy_implicit_phase.py trinity/phase_general/phase_events.py","confidence":"medium"},
  {"id":"NUM-23","file":"trinity/_functions/operations.py","line":129,"class":"numerical","severity":"S3","claim":"_is_monotonic_or_tolerable tolerates a single-step wrong-direction run 'regardless of depth', so an unbounded one-step collapse in the bubble temperature profile passes the monotonicity gate that MONOTONIC_RTOL = 1e-2 otherwise enforces.","evidence":"operations.py:129-131 'if end - start == 1: continue' bypasses the depth test at l.136-138. The gate guards find_nearest_higher (l.145-158), which locates the CIE / conduction / cooling-zone split indices used by the bubble luminosity integral (bubble_luminosity.py:708-709).","expected":"Bound the single-step exemption too, e.g. tolerate a single step only if its relative depth is also <= MONOTONIC_RTOL (or a looser but finite bound).","failure_scenario":"A profile with one catastrophic single-step drop passes as 'monotonic enough' and the region-split indices land arbitrarily, silently reshaping L_bubble / L_conduction / L_intermediate. Narrowed by the np.all(np.isfinite(L)) precondition at l.110 and by the fact that a tail collapse produces a multi-step run.","repro":"sed -n '126,140p' trinity/_functions/operations.py","confidence":"medium"},
  {"id":"NUM-24","file":"trinity/cloud_properties/bonnorEbertSphere.py","line":254,"class":"numerical","severity":"S3","claim":"The Lane-Emden odeint runs on default tolerances (rtol = atol = 1.49e-8), so the crossover atol/rtol = 1.0 means the absolute tolerance governs the entire Bonnor-Ebert core (|u| < 1); at the first grid point the allowed error is 7 orders larger than u itself and 45% of omega. Solver success is also unchecked.","evidence":"bonnorEbertSphere.py:254 calls odeint(lane_emden_ode, [u0, omega0], xi, tfirst=False) with no rtol/atol and no full_output. xi_min = 1e-7 with series ICs u0 ~ xi^2/6 = 1.7e-15 and omega0 ~ xi/3 = 3.3e-8 (get_initial_conditions, l.248). odeint defaults verified as sqrt(eps) = 1.4901161193847656e-08 for both.","expected":"Pass explicit rtol=1e-10, atol=1e-14 (or a per-component atol), and check full_output for integration success.","failure_scenario":"Largely benign for the consumed quantity: rho/rhoc = exp(-u) inherits only ~1.5e-8 RELATIVE error for u << 1. The exposed casualty is the dimensionless mass m = xi^2*dudxi (l.264) at small xi, which is unresolved; and an odeint failure would go undetected.","repro":"python3 -c \"import numpy as np; print(np.sqrt(np.finfo(float).eps), 1e-7**2/6, 1e-7/3)\"","confidence":"medium"},
  {"id":"NUM-25","file":"trinity/phase1b_energy_implicit/run_energy_implicit_phase.py","line":172,"class":"numerical","severity":"S3","claim":"ODE_MIN_STEP = 1e-6 combined with ODE_MAX_STEP = 2e-5 leaves LSODA a step range spanning only a factor of 20 in phase 1b; when the error test demands a smaller step ODEPACK cannot comply and the runner records the stiffness event as a solver failure.","evidence":"run_energy_implicit_phase.py:172-173 and l.1076-1077 (min_step passed only for LSODA). scipy's LSODA.__init__ defaults min_step=0.0 and forwards it as ODEPACK HMIN. The failure is converted to termination_reason = 'solver_failed: ...' at l.1090. Phases 1c/2 have max_step = 2e-4 (NUM-15) so their window is 200x.","expected":"Either drop min_step (default 0.0) and let the error controller work, or document the stiffness regime the floor is intended to escape and assert on it.","failure_scenario":"A genuinely stiff moment - a phase transition or a v2 turnaround where atol = 1e-8 pc/Myr binds (NUM-14) - terminates phase 1b with 'solver_failed' rather than being resolved, and the run is handed on from an arbitrary mid-segment state.","repro":"python3 -c \"print(2e-5/1e-6, 2e-4/1e-6)\"","confidence":"medium"}
]
```
