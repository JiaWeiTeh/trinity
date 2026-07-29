# Sweep: duplicate divergence across parallel code paths

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

Read-only audit of `/home/user/trinity/trinity/**`. Every claim below quotes both sides.

**Note on git evidence.** This repo's history is not usable as a "which side was fixed later"
oracle for most of these files: `trinity/` was imported wholesale in `bf50e44` (2026-06-23,
"plotting scripts for runtime"), and the four phase runners have only 2–3 commits each since.
Where a commit *is* informative I quote it; otherwise I rely on physics and on the code's own
comments, and I say so and lower the confidence.

One commit is directly on-point for this defect class and worth quoting up front — it is a
maintainer *propagating* a fix from one twin to another, and simultaneously documenting a second
asymmetry left un-propagated:

> `2951c0c` — "high-mass: route energy-collapse to momentum + cooling_balance in 1a
> … Phase 1b: on energy-driven collapse (Eb<=0), route to the momentum phase via 1c instead of
> dead-stopping … Phase 1a: add cooling_balance transition check (parity with 1b) so a violently
> cooling cloud can hand off within the fixed ~3000-yr early phase."

---

### DD-001 · A non-terminal *monitoring* event ends the implicit phase, because the shared event checker never looks at `event.terminal`
- **paths** — `trinity/phase_general/phase_events.py:487` vs `trinity/phase_general/phase_events.py:531` / `:570` / `:447`

  implicit (the only builder that injects a non-terminal event, and it is at index 0):
  ```python
  events = [
      make_velocity_sign_event(),
      make_min_radius_event(min_r),
      make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
  ]
  ```
  transition / momentum / energy — terminal events only:
  ```python
  events = [
      make_energy_floor_event(energy_floor, y_index=2),
      make_min_radius_event(min_r),
      make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
  ]
  ```
  and the shared checker, `phase_events.py:392`:
  ```python
  for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):
      if len(t_ev) > 0:
          event = events[i]
          return EventResult(triggered=True, ...)
  ```
  against the event's own declaration, `phase_events.py:310`:
  ```python
  event.terminal = False  # Non-terminal by default - just records the crossing
  ```
- **class** — divergence (implicit's event list violates an invariant the shared checker assumes,
  and which the other three lists satisfy)
- **severity** — S1 results-wrong
- **the difference** — `scipy.integrate.solve_ivp` populates `sol.t_events[i]` for *every* event,
  terminal or not. `check_event_termination` returns on the first event with a non-empty
  `t_events`, ignoring `terminal`. In phases 1a/1c/2 that is harmless (all their events are
  terminal). In phase 1b, index 0 is `velocity_sign`, a deliberately non-terminal monitor. The
  first time `v2` crosses from positive to negative anywhere in a segment, `check_event_termination`
  reports `triggered=True, name='velocity_sign', is_simulation_ending=False`, and
  `run_energy_implicit_phase.py:1096-1119` rewinds the state to the crossing point and `break`s out
  of the implicit phase:
  ```python
  event_result = check_event_termination(sol, ode_events)
  if event_result.triggered:
      ...
      R2 = float(event_result.y[0]); v2 = float(event_result.y[1])
      Eb = float(event_result.y[2]); T0 = float(event_result.y[3])
      t_now = event_result.t
      ...
      break
  ```
  I confirmed the mechanism against the installed scipy:
  ```
  t_events: [array([1.]), array([], dtype=float64)]  status 0
  triggered= True  name= velocity_sign  is_sim_ending= False  reason= velocity_sign_change
  ```
  (`status 0` = the solver ran to the end of the span; nothing terminated it.)
- **which is right** — The other three builders are right; the docstring of
  `make_velocity_sign_event` states the intent explicitly ("Non-terminal by default - just records
  the crossing", `phase_events.py:310`), and the module docstring classifies it under
  "**Monitoring Events** (non-terminal; record a crossing only)" (`phase_events.py:25`). The
  implicit runner has its *own* collapse-onset handler, which is the intended consumer of that
  crossing (`run_energy_implicit_phase.py:1302`):
  ```python
  # Collapse detection: velocity negative AND radius decreasing
  if v2 < 0 and R2 < R2_prev:
      params['isCollapse'].value = True
  ```
  — identical to `run_transition_phase.py:772` and `run_momentum_phase.py:825`. That handler sits
  *after* the event check, so on the first zero-crossing it is unreachable in 1b while it runs
  normally in 1c and 2. Two mechanisms for the same physical signal, only one of which can be right.
- **failure scenario** — Any configuration in which the shell decelerates through `v2 = 0` while
  still in the energy-driven implicit phase: massive/dense GMCs where gravity plus a heavy swept-up
  shell stalls the bubble before cooling balance is reached. The implicit phase terminates at the
  stall with `termination_reason="velocity_sign_change"`, the remainder of that segment's
  integration is discarded, `isCollapse` is never set, and `main.py` hands a still-energy-driven
  bubble (`Eb` well above the 1e3 floor) to phase 1c → 2. The recollapse is then integrated with the
  momentum-phase force budget, i.e. the run reports a momentum-driven recollapse for a bubble that
  was thermally driven. Sweeps over cloud mass will show a spurious behaviour change at whatever
  mass first stalls the shell.
- **confidence** — high (mechanism verified by execution; physics interpretation from the module's
  own docstrings)

---

### DD-002 · `apply_event_result` marks the shell as *collapsing* when it terminates for **large** radius
- **paths** — `trinity/phase_general/phase_events.py:626` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1327` (and `run_transition_phase.py:797`, `run_momentum_phase.py:850`)

  event path:
  ```python
  # Mark collapse if it's a collapse-related event
  if 'radius' in result.reason_code.lower() or 'collapse' in result.reason_code.lower():
      if 'isCollapse' in params:
          params['isCollapse'].value = True
  ```
  where `make_max_radius_event` sets (`phase_events.py:160`):
  ```python
  event.reason_code = "large_radius_event"
  ```
  segment-loop path for the same physical outcome:
  ```python
  stop_r = params['stop_r'].value
  if stop_r is not None and R2 > stop_r:
      termination_reason = "large_radius"
      params['SimulationEndReason'].value = 'Large radius reached'
      params['SimulationEndCode'].value = SimulationEndCode.LARGE_RADIUS.code
      params['EndSimulationDirectly'].value = True
      break
  ```
- **class** — divergence
- **severity** — S3 misleading (state flag / reported outcome)
- **the difference** — `'radius' in 'large_radius_event'` is `True`, so a run that terminates
  because the shell blew *out* past `stop_r` — at large positive `v2` — is flagged
  `isCollapse = True`. The segment-loop check for the identical condition leaves `isCollapse`
  untouched. Which of the two fires is a race decided by whether `R2` crosses `stop_r` inside a
  segment (event) or at a segment boundary (loop check).
- **which is right** — The segment-loop path. `isCollapse`'s registered meaning is
  "Is cloud collapsing?" (`trinity/_input/registry.py:514`) and the only other writers set it on
  `v2 < 0 and R2 < R2_prev`. The comment above the event-path branch states the intent
  ("Mark collapse if it's a collapse-related event") and the substring test simply does not
  implement it; `min_radius`/`small_radius_event` is the collapse-related one.
- **failure scenario** — Any run with `stop_r` set where the shell crosses `stop_r` mid-segment
  (the common case, since `stop_r` is typically far outside `rCloud`). Downstream:
  `trinity/_output/show_run.py:87-91` then renders the collapse status as `"collapsing"`, and
  `trinity/_input/sweep_runner.py:560-561` writes `collapsed: "yes"` into the sweep summary for a
  blowout. Sweep tables misclassify blowouts as recollapses.
- **confidence** — high

---

### DD-003 · The momentum ODE freezes shell mass and `dM/dt` for a whole segment; the energy/implicit/transition ODE recomputes both live
- **paths** — `trinity/phase2_momentum/run_momentum_phase.py:412` vs `trinity/phase1_energy/energy_phase_ODEs.py:204`

  momentum RHS — reads frozen snapshot values:
  ```python
  mShell = snapshot.mShell
  mShell_dot = snapshot.mShell_dot
  mShell = max(mShell, 1e-10)
  ...
  vd = (F_pressure - mShell_dot * v2 - F_grav + F_rad) / mShell
  ```
  energy/implicit/transition RHS (`get_ODE_Edot_pure`, shared by all three) — recomputes at the
  live `R2`, `v2`:
  ```python
  if snapshot.isCollapse:
      mShell = snapshot.shell_mass
      mShell_dot = 0.0
  else:
      mShell_new, mShell_dot = mass_profile.get_mass_profile(
          R2, params_for_feedback, return_mdot=True, rdot=v2
      )
      prev_mShell = snapshot.shell_mass
      if prev_mShell > 0 and mShell_new < prev_mShell:
          mShell = prev_mShell
          mShell_dot = 0.0
      else:
          mShell = mShell_new
  ...
  vd = (4.0 * np.pi * R2**2 * (P_drive - P_ext)
        - mShell_dot * v2 - F_grav + F_rad) / mShell
  ```
- **class** — divergence
- **severity** — S1 results-wrong
- **the difference** — Both RHS functions solve the same momentum equation
  `m v̇ = 4πR²(P_drive − P_ext) − ṁ v − F_grav + F_rad`. In 1a/1b/1c, `m` and `ṁ` are functions of
  the instantaneous `R2` and `v2` inside the RHS; the snapshot is only used as the
  monotonic-mass floor. In phase 2 they are constants for the segment, evaluated once at the
  segment's starting `R2`, `v2`. Phase 2 and phase 1c share the *same* segment schedule
  (`DT_SEGMENT_INIT = 2e-3`, `DT_SEGMENT_MAX = 5e-2` Myr in both files), and phase 2 additionally
  caps `max_step` at `DT_SEGMENT_MIN/5 = 2e-4` Myr — so a single segment can contain ~250 internal
  RHS evaluations, all using a shell mass and accretion rate from up to 5×10⁻² Myr ago.
  `F_grav` in phase 2 is likewise computed from the frozen `mShell`, so the frozen mass enters the
  force budget twice.
- **which is right** — The live evaluation, on physics: `ṁ = 4πR²ρ(R)v` is the swept-up-mass rate
  of a shell *at its current radius*; `mass_profile.compute_mass_accretion_rate`'s own docstring
  says so — "This formula is EXACT for any smooth density profile … NO SOLVER HISTORY NEEDED - just
  instantaneous rho(r) and v(r)" (`mass_profile.py:450` and `:454`). Freezing it makes `−ṁv` a stale
  constant across a segment in which `R2` can grow by ~0.5 pc at 10 pc/Myr, exactly where a
  power-law `ρ(r)` varies fastest. Nothing in the momentum-phase docstrings offers a physical
  reason for the freeze, and the freeze is *not* the `isCollapse` freeze (which phase 2 implements
  separately, and correctly, in `create_momentum_snapshot`/the runner). Git cannot arbitrate: both
  files were imported in `bf50e44` and only touched since by `8843bb7` ("docs(code): fix stale
  comments and incorrect docstrings").
- **failure scenario** — Every run that reaches phase 2 (i.e. essentially all of them), most
  strongly where the density gradient is steep and the shell is still inside the cloud: steep
  `densPL_alpha`, high `nCore`. The momentum-phase trajectory `R2(t)`, `v2(t)` is biased — the shell
  is decelerated by a stale `ṁ` and accelerated by a stale (too small, in an outward-expanding run
  through a declining profile: too *large*) `mShell`. Errors accumulate segment over segment.
  It also breaks continuity across the 1c→2 boundary: the same physical state integrated by 1c and
  by 2 gives different `v̇`.
- **confidence** — medium-high on the divergence being unintended (high that the two paths compute
  different things; medium that the freeze was not a deliberate cost trade — no comment says so)

---

### DD-004 · Phase 1a runs its phase-boundary reconciliation after an `Eb ≤ 0` collapse; phase 1b explicitly skips it, because that recompute writes a garbage negative `Pb`
- **paths** — `trinity/phase1_energy/run_energy_phase.py:368` + `:390` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1365`

  phase 1a — breaks on collapse, then reconciles unconditionally:
  ```python
  if not np.isfinite(Eb) or Eb <= 0:
      params['EndSimulationDirectly'].value = True
      params['SimulationEndReason'].value = (...)
      params['SimulationEndCode'].value = SimulationEndCode.ENERGY_COLLAPSED.code
      ...
      break
  ...
  try:
      feedback_final = get_current_sps_feedback(t_now, params)
      updateDict(params, feedback_final)
      R1_f = get_bubbleParams.solve_R1(R2, Eb, feedback_final.Lmech_total,
                                       feedback_final.v_mech_total)
      Pb_f = get_bubbleParams.bubble_E2P(Eb, R2, R1_f, params['gamma_adia'].value)
      params['R1'].value = R1_f
      params['Pb'].value = Pb_f
  ```
  phase 1b — same reconciliation, gated:
  ```python
  # SKIP on the energy_collapsed exit: there Eb < 0, so compute_R1_Pb would
  # recompute Pb = (gamma-1)*Eb/V as a garbage negative terminal row
  # (Pb ~ -1.6e18). ... See docs/dev/transition/pdv-trigger/
  # PB_COLLAPSE_GUARD_FIX.md.
  if termination_reason != "energy_collapsed":
      try:
          ...
          R1_f, Pb_f = compute_R1_Pb(R2, Eb, feedback_final.Lmech_total, ...)
  ```
- **class** — missing-propagation
- **severity** — S1 results-wrong (terminal snapshot), S3 for the reported stop fate
- **the difference** — `compute_R1_Pb` *is* `solve_R1` + `bubble_E2P` (`get_betadelta.py:327-329`),
  so the two blocks perform the identical computation. With `Eb ≤ 0`, `get_r1` floors the energy to
  `1e-30` (`get_bubbleParams.py:406-407`), driving `R1 → R2`; `bubble_E2P` then hits its
  `shell_volume <= 0` floor (`get_bubbleParams.py:229-236`) and returns
  `Pb = (γ−1)·Eb / (1e-13·r2³) / (4π/3)` — a huge *negative* pressure. Phase 1b guards against
  exactly this; phase 1a does not, and additionally then feeds that negative `Pb` into
  `shell_structure_pure` (via `params['Pb']`), whose inner-edge density
  `nShell0 = μ/(k_B T) · Pb` (`shell_structure.py:124`) goes negative.
- **which is right** — Phase 1b, unambiguously: its comment names the failure mode, the observed
  magnitude (`Pb ~ -1.6e18`) and the write-up that produced it. This is the archetypal signature —
  a fix landed in the phase someone was debugging and not in its twin.
- **failure scenario** — The energy-driven collapse path in phase 1a: massive/dense GMCs where the
  bubble loses `Eb` through PdV work on a heavy shell inside the fixed ~3000-yr early window
  (the code's own comment at `run_energy_phase.py:359-367` describes this regime). Either
  `dictionary.jsonl`'s terminal row carries `Pb ≈ −10¹⁸` and a nonsense shell structure, or
  `shell_structure_pure` raises and the whole reconciliation is swallowed by
  `except Exception` (`run_energy_phase.py:403`) so no terminal snapshot is written at all — while
  `params['Pb']`/`params['R1']` are left holding the garbage for anything that reads them later.
- **confidence** — high

---

### DD-005 · The `EarlyPhaseApproximation` hard override (`vd = -1e8`) is consumed by three phases but reset by only one, on only one of its four exit paths
- **paths** — `trinity/phase1_energy/energy_phase_ODEs.py:269` (consumer, shared) vs `trinity/phase1_energy/run_energy_phase.py:342` (sole reset)

  consumer, inside the RHS shared by 1a, 1b (`get_ODE_implicit_pure`) and 1c (`get_ODE_transition_pure`):
  ```python
  # Early phase approximation
  if snapshot.EarlyPhaseApproximation:
      vd = -1e8
  ```
  sole reset, in phase 1a *after* a successful `solve_ivp` segment:
  ```python
  # Handle early phase approximation switch
  if loop_count == 0 and params['EarlyPhaseApproximation'].value:
      params['EarlyPhaseApproximation'].value = False
      logger.info('Switching to no approximation')
  ```
  and the default (`trinity/_input/registry.py:423`):
  ```python
  ParamSpec(name='EarlyPhaseApproximation', default=True, ...)
  ```
- **class** — missing-propagation (a state-clearing step present in one runner's success path and
  in no twin)
- **severity** — S1 results-wrong when reached; S2 latent otherwise
- **the difference** — The flag is a *global* switch read through `create_ODE_snapshot`
  (`energy_phase_ODEs.py:159`), which phases 1b and 1c also call
  (`run_energy_implicit_phase.py:1051`, `run_transition_phase.py:616`). Phase 1a clears it only if
  it completes a first `solve_ivp` segment. Three earlier `break` paths bypass the reset:
  the bubble-solve failure (`run_energy_phase.py:183`), the `cooling_balance` early exit
  (`:287`), and any triggered event including the phase-ending `cloud_boundary` (`:331`) — plus
  the case where the `while` loop never runs at all. Phases 1b and 1c have no reset of their own.
- **which is right** — The flag must be `False` before 1b/1c integrate. `vd = -1e8` pc/Myr² is not
  a physical acceleration; it is a startup device that overrides the entire assembled force budget
  (`4πR²(P_drive−P_ext) − ṁv − F_grav + F_rad`) on the immediately-preceding lines. No twin
  runner has any comparable device, so a non-1a phase running with it set is a state leak, not a
  model choice.
- **failure scenario** — Compact/high-feedback clouds where `R2` crosses `rCloud` inside the very
  first 3×10⁻⁵ Myr segment (the `cloud_boundary` event fires, `is_simulation_ending=False`, so
  `main.py` proceeds to 1b); or a cloud whose bubble solve degenerates on the first segment; or a
  violently cooling cloud that hits `cooling_balance` on segment 0 (the branch commit `2951c0c`
  added precisely for this regime). In all three, phase 1b's first RHS evaluation returns
  `vd = -1e8`, `v2` crashes through `-500` pc/Myr, and the `velocity_runaway` event ends the run
  with `VELOCITY_RUNAWAY` — a purely numerical stopping fate reported as physics.
- **confidence** — medium-high (code path certain; how often the three `break`s fire on segment 0
  is configuration-dependent and I did not run simulations to measure it)

---

### DD-006 · The shell-dissolution stopping condition exists in phases 1c and 2 and in neither energy phase
- **paths** — `trinity/phase1c_transition/run_transition_phase.py:805` and `trinity/phase2_momentum/run_momentum_phase.py:858` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` (absent) and `trinity/phase1_energy/run_energy_phase.py` (absent)

  1c and 2 (byte-for-byte identical blocks):
  ```python
  # Dissolution check: persistent timer based on shell_nMax < nISM
  shell_nMax = params.get('shell_nMax', None)
  if shell_nMax and hasattr(shell_nMax, 'value'):
      if shell_nMax.value < params['nISM'].value:
          if t_diss_onset == np.inf:
              t_diss_onset = t_now
              ...
          if (t_now - t_diss_onset) >= params['stop_t_diss'].value:
              params['isDissolved'].value = True
              termination_reason = "dissolved"
              ...
  ```
  1b's termination block ends at (`run_energy_implicit_phase.py:1326-1333`):
  ```python
  # Stop radius check (skip if stop_r is None)
  stop_r = params['stop_r'].value
  if stop_r is not None and R2 > stop_r:
      termination_reason = "large_radius"
  ```
  — `t_diss_onset` does not appear anywhere in `phase1_energy/` or `phase1b_energy_implicit/`
  (verified by grep: the only occurrences in `trinity/` are in `phase1c_transition/` and
  `phase2_momentum/`).
- **class** — missing-propagation
- **severity** — S2 latent
- **the difference** — `shell_structure_pure` computes `diss_condition_met` and `shell_nMax` in
  every phase, and `isDissolved` gates `F_rad` to zero in all four force assemblies
  (e.g. `energy_phase_ODEs.py:130-135`, `run_momentum_phase.py:275-280`). But only 1c and 2 ever
  *set* `isDissolved`. In 1a/1b the flag can only arrive already-set from a prior phase — which,
  since 1a and 1b run first, means never.
- **which is right** — Can't be settled by git (all four files were imported together). Physically,
  a shell whose peak density has fallen below the ambient ISM is dissolved regardless of what is
  driving it, and the `stop_t_diss` parameter is registered globally
  (`registry.py:351`, `run_const=True`) rather than per-phase — both point to the check belonging
  in all four. There is a defensible counter-argument for 1a/1b (an energy-driven bubble with a
  hot interior should not have a sub-ISM shell, so the check would be inert), but no comment in
  either file makes it, which under the audit's own rule reclassifies it from intended to
  missing-propagation.
- **failure scenario** — A low-density-ambient run whose shell thins below `nISM` while still
  energy-driven (large `rCloud`, low `nISM`, strong feedback): 1b keeps integrating a shell that
  1c/2 would have declared dissolved, and `F_rad` stays on because `isDissolved` is still `False`.
  The run's stopping fate is decided later and differently than an otherwise identical run that
  crossed the same threshold one phase later.
- **confidence** — medium

---

### DD-007 · The energy phase's event list has neither the `max_radius` (`stop_r`) event nor any in-loop `stop_r` / `coll_r` check its three twins all have
- **paths** — `trinity/phase_general/phase_events.py:447` vs `:487`+`:493`, `:531`+`:537`, `:570`+`:575`

  energy:
  ```python
  events = [
      make_cloud_boundary_event(rCloud),
      make_min_radius_event(min_r),
      make_velocity_runaway_event(MAX_VELOCITY_COLLAPSE, direction="collapse"),
  ]
  ```
  implicit / transition / momentum all append:
  ```python
  # Only add max_radius event if stop_r is set
  if stop_r is not None and stop_r > 0:
      events.append(make_max_radius_event(stop_r))
  ```
  and 1b/1c/2 each additionally carry the in-loop `stop_r`, `coll_r` and `t_now > tmax` checks
  (e.g. `run_energy_implicit_phase.py:1309-1333`); phase 1a's loop has none of them — its only
  loop guard is `while R2 < rCloud and (TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD and continueWeaver`
  (`run_energy_phase.py:138`).
- **class** — intended-difference for `stop_r`/`stop_t`; divergence for `coll_r`
- **severity** — S4 hygiene (as it stands), S2 for the `coll_r`/`isCollapse` half
- **the difference** — Phase 1a is bounded to a fixed 3×10⁻³ Myr window and to `R2 < rCloud`, so
  `stop_t` (which is ≫ 3e-3 Myr in any real config) and `stop_r` (which is > `rCloud` — `main.py:38`
  and `_check_stop_r_rCloud_interaction` treat `stop_r <= rCloud` as a misconfiguration) genuinely
  cannot fire inside it: **intended**, with a stateable physical/structural reason. The `coll_r`
  half is different: 1a *does* carry a `min_radius` event with the same
  `max(coll_r*1.5, 0.01)` threshold as its twins, but has no `isCollapse` detection at all, so
  `snapshot.isCollapse` in its own ODE (`energy_phase_ODEs.py:204`, which freezes shell mass during
  collapse) is permanently `False` in 1a.
- **which is right** — The `stop_r`/`stop_t` omission is right. The `isCollapse` omission is a gap:
  1a reads the flag in its RHS but no code path can ever set it before 1a runs.
- **failure scenario** — 1a's shell-mass freeze-on-collapse branch is dead code, so a shell that
  begins contracting inside the 3000-yr window keeps accreting mass in 1a where 1b/1c/2 would
  freeze it.
- **confidence** — high on the code state; medium on severity (the window is short)

---

### DD-008 · Energy-driven collapse (`Eb ≤ 0`) stops the run in phase 1a and routes to momentum in phase 1b
- **paths** — `trinity/phase1_energy/run_energy_phase.py:368` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:184` + `:1148`

  1a:
  ```python
  if not np.isfinite(Eb) or Eb <= 0:
      params['EndSimulationDirectly'].value = True
      params['SimulationEndReason'].value = (
          "Energy-driven bubble collapsed: Eb fell to <= 0 ...")
      params['SimulationEndCode'].value = SimulationEndCode.ENERGY_COLLAPSED.code
      ...
      break
  ```
  1b:
  ```python
  def classify_energy_collapse(Eb):
      if not np.isfinite(Eb):
          return 'stop'
      if Eb <= 0:
          return 'momentum'
      return None
  ...
  if _collapse == 'momentum':
      Eb = ENERGY_HANDOFF_FLOOR
      params['Eb'].value = Eb
      termination_reason = "energy_to_momentum"
  ```
- **class** — intended-difference (explicitly deferred, not accidental)
- **severity** — S2 latent
- **the difference** — Identical physical condition, opposite fate: 1a ends the simulation,
  1b hands `(R2, v2)` to phase 1c with `Eb` set to the transition floor and lets the run continue
  to momentum.
- **which is right** — 1b's routing is the physics the maintainer settled on. Its justification is
  written down (`run_energy_implicit_phase.py:191-196`): as `Eb → 0` the bubble pressure floors at
  `~P_ram`, so the shell is already momentum-driven and dead-stopping loses the rest of the
  evolution. 1a's difference is *deliberately* deferred, and both sides say so —
  `run_energy_phase.py:165-168`: "(Phase 1b now ROUTES a clean Eb<=0 collapse to the momentum
  phase; routing it from 1a too is deferred …)" — and commit `2951c0c` documents the same split.
  Recorded here as an intended difference on the maintainer's own record, but note that the
  physical argument ("`Pb` floors at `P_ram`") is phase-independent, so the deferral is a schedule
  decision rather than a physics one.
- **failure scenario** — A cloud that collapses inside the fixed ~3000-yr early window ends with
  `ENERGY_COLLAPSED`; an otherwise identical cloud that collapses a few segments later, in 1b,
  runs on to a momentum-driven fate. Two different published outcomes across a knife-edge in
  cloud mass/density.
- **confidence** — high

---

### DD-009 · `bubble_Leak` is recorded in phases 1a and 1b but never refreshed in 1c — whose ODE actively subtracts a freshly computed leak
- **paths** — `trinity/phase1_energy/run_energy_phase.py:254` and `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:813` vs `trinity/phase1c_transition/run_transition_phase.py` (absent)

  1a:
  ```python
  if ode_result.bubble_Leak is not None:
      params['bubble_Leak'].value = ode_result.bubble_Leak
  ```
  1b:
  ```python
  params['bubble_Leak'].value = get_bubbleParams.get_leak_luminosity(
      params['coverFraction'].value, params['R2'].value, params['Pb'].value,
      params['c_sound'].value, params['gamma_adia'].value,
  )
  ```
  1c: no write. But its RHS goes through `get_ODE_Edot_pure`, which does compute and use one
  (`energy_phase_ODEs.py:277-280`):
  ```python
  L_leak = get_bubbleParams.get_leak_luminosity(
      snapshot.coverFraction, R2, press_bubble, snapshot.c_sound, snapshot.gamma_adia
  )
  Ed = (Lmech_total - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak
  ```
  (`grep -o "params\['[A-Za-z0-9_]*'\]\.value *="` over all four runners confirms `bubble_Leak`
  appears only in 1a and 1b.)
- **class** — missing-propagation
- **severity** — S3 misleading
- **the difference** — Every transition-phase snapshot carries the last value phase 1b wrote,
  evaluated at 1b's final `(R2, Pb, c_sound)`, while the energy equation being integrated at that
  snapshot's `t` subtracts a different, live value. Phase 2 also never writes it, but there it is
  genuinely inapplicable (no energy equation) — 1c is the one where the term is live and the
  diagnostic is stale.
- **which is right** — 1a and 1b: the recorded diagnostic should equal the term the RHS subtracts.
  1a's own comment for the equivalent quantity says so — `energy_phase_ODEs.py:404`:
  "Covering-fraction leak diagnostic (same value the RHS subtracts from Edot)".
- **failure scenario** — Any run with `coverFraction < 1`. Post-hoc energy-budget accounting over
  the transition phase (`Lmech − Lcool − PdV − Lleak`) will not close, and the discrepancy will be
  read as a bug in the integrator rather than in the recording.
- **confidence** — high

---

### DD-010 · The cloud density profile is smoothed at `rCloud`; the enclosed-mass profile is not, so `M(r)` and `4πr²ρ(r)` describe different clouds
- **paths** — `trinity/cloud_properties/density_profile.py:128` vs `trinity/cloud_properties/mass_profile.py:312` and `:224`

  `n(r)` — tanh bridge, added specifically to smooth the `ṁ` term:
  ```python
  SMOOTH_FRAC = 0.01
  delta = SMOOTH_FRAC * rCloud
  w_outside = 0.5 * (1.0 + np.tanh((r_arr - rCloud) / delta))
  ...
  n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside
  ```
  with the rationale at `density_profile.py:117-127`: "That step makes `mShell_dot = 4*pi*r^2*rho(r)*v`
  in the phase ODEs jump by ~10^3 across r=rCloud … mass conservation holds to O(SMOOTH_FRAC^2)."

  `M(r)` — hard step, no bridge:
  ```python
  inside_cloud = r_arr <= rCloud
  M_arr[inside_cloud] = (4.0/3.0) * np.pi * r_arr[inside_cloud]**3 * rhoCore
  outside_cloud = r_arr > rCloud
  M_arr[outside_cloud] = mCloud + (4.0/3.0) * np.pi * rhoISM * (
      r_arr[outside_cloud]**3 - rCloud**3
  )
  ```
  while the rate returned alongside it uses the *smoothed* density (`mass_profile.py:224`):
  ```python
  dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr
  ```
- **class** — divergence (same profile, two mutually inconsistent representations)
- **severity** — S2 latent
- **the difference** — `get_mass_profile(..., return_mdot=True)` returns a pair `(M, dM/dt)` in
  which `dM/dt ≠ dM/dr · ṙ` for the returned `M`, across the ±1 %·`rCloud` band. Inside that band
  the discrepancy is order-unity: `ρ_smoothed` is a blend of `ρ_cloud` and `ρ_ISM` (a ~10³
  contrast) while `dM/dr` from the analytic `M` is still the pure cloud or pure ISM value.
  The Bonnor-Ebert branch has the same split (`compute_enclosed_mass_bonnor_ebert` uses the
  unsmoothed Lane-Emden `m(ξ)`), so PL and BE agree with *each other* and both disagree with
  `density_profile`.
- **which is right** — Neither is wrong in isolation; they must agree, and the smoothing is the
  side that moved. `density_profile.py`'s comment shows the bridge was introduced *for the phase
  ODEs' `mShell_dot`* — i.e. it was propagated to the `ρ` used by `ṁ` and not to the `M(r)` that
  `ṁ` is supposed to differentiate. The self-consistent choice is to integrate the smoothed `ρ`
  (the comment's own O(SMOOTH_FRAC²) mass-conservation claim is a claim about exactly that
  integral).
- **failure scenario** — Any run whose shell crosses `rCloud` — the standard blowout, i.e. most
  runs, and specifically the regime `stop_at_rCloud_nSnap` exists to sample. Within
  `0.99·rCloud < R2 < 1.01·rCloud` the shell's mass and its accretion rate are inconsistent, so
  the `−ṁv` deceleration and the `1/mShell` normalisation in `v̇` come from different clouds. It
  also makes `mass_profile.validate_mass_at_rCloud` (which evaluates `M` exactly *at* `rCloud`,
  the midpoint of the bridge) pass on the analytic value while the ODEs see the blended one.
- **confidence** — high on the inconsistency; medium on magnitude (I did not measure the resulting
  trajectory error)

---

### DD-011 · The Bonnor-Ebert numerical-fallback mass integrator writes with subset indices; the power-law branch writes through boolean masks
- **paths** — `trinity/cloud_properties/mass_profile.py:415` vs `trinity/cloud_properties/mass_profile.py:326`

  BE fallback:
  ```python
  for i, (r, rho) in enumerate(zip(r_inside, rho_inside)):
      if i == 0:
          M_arr[i] = 0.0
      else:
          M_arr[i] = scipy.integrate.trapezoid(
              4.0 * np.pi * r_inside[:i+1]**2 * rho_inside[:i+1],
              r_inside[:i+1]
          )
  ```
  power-law:
  ```python
  region1 = r_arr <= rCore
  M_arr[region1] = (4.0/3.0) * np.pi * r_arr[region1]**3 * rhoCore
  region2 = (r_arr > rCore) & (r_arr <= rCloud)
  M_arr[region2] = 4.0 * np.pi * rhoCore * (...)
  ```
- **class** — divergence
- **severity** — S2 latent
- **the difference** — `i` indexes `r_inside = r_arr[inside_cloud]`, but the result is written to
  `M_arr[i]` — the *global* index. These coincide only if every inside-cloud radius occupies a
  prefix of `r_arr`, i.e. only if `r_arr` is sorted ascending. The power-law branch is
  order-independent. The docstring admits the dependency ("Radii (must be sorted!)",
  `mass_profile.py:365`) but `compute_enclosed_mass` — the only caller — does not enforce it, and
  `get_mass_profile` is called from the phase ODEs with arbitrary scalars/arrays. (The docstring
  line is `mass_profile.py:363`, "Radii (must be sorted!)".)
  A second, quieter difference: after the loop the ISM region is written through
  `M_arr[outside_cloud]`, mixing the two conventions inside the same function.
- **which is right** — The masked form. It is the convention `compute_enclosed_mass_powerlaw` and
  the ISM tail of the same BE function both use, and it removes an undocumented precondition on a
  public entry point.
- **failure scenario** — Only when the analytic Lane-Emden path is unavailable, i.e. when
  `densBE_f_m` or `densBE_xi_out` is missing from `params` (`mass_profile.py:384`) — standalone /
  reconstructed-params use, e.g. `cloud_properties/initial_profile.py` or the cloudy exporter, on
  an unsorted radius array. Silently wrong `M(r)` (values scattered into the wrong slots, some
  slots left at 0.0) with no exception.
- **confidence** — high on the code property; low on it being reachable in production runs

---

### DD-012 · The shell ODE applies the covering fraction to `dτ/dr` in the ionised branch only
- **paths** — `trinity/shell_structure/get_shellODE.py:122` vs `trinity/shell_structure/get_shellODE.py:144`

  ionised:
  ```python
  # optical depth
  dtaudr = nShell * sigma_dust * f_cover
  ```
  neutral:
  ```python
  # optical depth
  dtaudr = nShell * sigma_dust
  ```
- **class** — divergence
- **severity** — S2 latent (inert today: `f_cover = 1`)
- **the difference** — `τ` is integrated continuously across the ionisation front — the neutral
  integration is seeded with `tau0_neu = tau0_ion` (`shell_structure.py:309`) — so the two branches
  are two halves of one optical-depth integral, and they apply different attenuation geometry to it.
  With `f_cover < 1` the ionised half would accumulate `f_cover ×` less optical depth per unit
  length than the neutral half, for the same dust column.
- **which is right** — Cannot be determined from the code, and I will not guess: `f_cover` is
  documented as "0 < f_cover <= 1 … f_cover = 1: all remained" (`get_shellODE.py:63-65`) with no
  statement of whether it should scale `τ` at all (a covering fraction is more naturally applied to
  the *emergent* flux than to the local optical-depth gradient). What is certain is that the two
  branches cannot both be right. The module carries `# TODO: add cover fraction cf (f_cover)`
  at `get_shellODE.py:35`, and `shell_structure.py:114-115` hardcodes
  `# TODO: Add f_cover from fragmentation mechanics` / `f_cover = 1` — so this is a half-finished
  feature, and the ionised branch is the half that got written.
- **failure scenario** — None today (`f_cover` is hardcoded to 1 at its single call site, so both
  branches reduce to the same expression). It becomes S1 the moment fragmentation mechanics wire
  `f_cover` up, which the two TODOs advertise as planned.
- **confidence** — high that the branches diverge; low on which is intended

---

### DD-013 · The `nShell` overflow cap exists in the ionised shell branch and not the neutral one
- **paths** — `trinity/shell_structure/get_shellODE.py:100` vs `trinity/shell_structure/get_shellODE.py:131`

  ionised:
  ```python
  # numerical guard: cap nShell so the +nShell**2 pole in the discarded
  # post-front tail cannot overflow float64 (see _NSHELL_MAX above).
  nShell = min(nShell, _NSHELL_MAX)
  ```
  neutral:
  ```python
  # unravel
  nShell, tau = y
  ```
- **class** — intended-difference
- **severity** — S4 hygiene
- **the difference** — Only the ionised branch has the cap.
- **which is right** — Both. The guard exists for a specific pole that only the ionised RHS has:
  the recombination term `+ chi_e * nShell**2 * alpha_B * Li / Qi / c` (`get_shellODE.py:117`) and
  `dphidr`'s `- 4πr²·χ_e·α_B·nShell²/Qi` (`:120`) are quadratic in `nShell`, so `nShell` runs away
  to a finite-radius pole just past the ionisation front. The neutral `dndr`
  (`:140-142`) is strictly *linear* in `nShell`, so it has no pole and cannot overflow the same
  way. The comment at `get_shellODE.py:18-32` states this precisely ("The ionised shell ODE has a
  dn/dr ∝ +nShell**2 recombination term, which is a finite-radius pole"). Recorded as a result:
  this is what a genuinely intended asymmetry looks like — it names the term that differs.
- **failure scenario** — n/a
- **confidence** — high

---

### DD-014 · Three of the four external-ionised-pressure lookups swallow exceptions to `P_ext = 0`; the one inside the energy/implicit/transition RHS does not
- **paths** — `trinity/phase1_energy/energy_phase_ODEs.py:52` vs `trinity/phase2_momentum/run_momentum_phase.py:426` (and `run_energy_implicit_phase.py:508`, `run_transition_phase.py:301`)

  the RHS path (`get_press_ion`, called from `get_ODE_Edot_pure` in 1a/1b/1c):
  ```python
  r = np.atleast_1d(r)
  n_r = density_profile.get_density_profile(r, params)
  P_ion = (params['mu_convert'].value / params['mu_ion_shell'].value) * n_r * params['k_B'].value * params['TShell_ion'].value
  return _scalar(P_ion)
  ```
  the force-assembly / momentum-RHS path, three near-identical copies:
  ```python
  try:
      n_r = density_profile.get_density_profile(np.array([rShell]), params)
      if hasattr(n_r, '__len__') and len(n_r) == 1:
          n_r = n_r[0]
      P_ext = (params['mu_convert'].value / params['mu_ion_shell'].value) * n_r * k_B * TShell_ion
  except Exception:
      P_ext = 0.0
  ```
- **class** — divergence
- **severity** — S3 misleading
- **the difference** — Identical formula, opposite failure behaviour. If
  `get_density_profile` raises (it does raise, on an unknown `dens_profile` —
  `density_profile.py:167`), the RHS propagates and the run fails loudly; the three force
  assemblies silently substitute `P_ext = 0`, i.e. they remove the *inward* confining pressure and
  report an over-accelerating shell. Because `params['F_ion_in']` and `params['press_HII_in']` are
  written from that swallowed value, the snapshot records `0` with no warning of any kind.
- **which is right** — The un-guarded one, on the project's own rule about not adding "error
  handling for impossible cases" and on the fact that `except Exception: P_ext = 0.0` converts a
  configuration error into a silent physics change. The `P_ext` that the ODE integrates and the
  `F_ion_in` that is reported must also be the same number, and today they need not be.
- **failure scenario** — Any condition that makes the density lookup raise (bad `dens_profile`,
  a `None` BE interpolator after `params.reset_keys(COOLING_PHASE_KEYS)` in `main.py:317`,
  a NaN `rShell`). Phases 1b/1c/2 record `F_ion_in = 0` and keep going; phases 1a/1b/1c crash from
  the RHS. Same run, two different fates depending on which of the two duplicated lookups is hit
  first.
- **confidence** — high

---

### DD-015 · The unguarded dict-mutating `cool_beta_to_Ebdot` / `delta2dTdt` sit next to `_pure` twins that carry three divide-by-zero guards each
- **paths** — `trinity/bubble_structure/get_bubbleParams.py:123` vs `trinity/phase1b_energy_implicit/get_betadelta.py:251`

  original (`get_bubbleParams.cool_beta_to_Ebdot`):
  ```python
  a_coeff = 1.5 * pdotdot_total / pdot_total
  c_coeff = 0.75 * pdot_total * R1
  d_coeff = R2**3 - R1**3
  c_frac = c_coeff / (Eb + c_coeff)              # c/(E_b + c)
  ...
  denominator = d_coeff * (1 - c_frac)
  Eb_dot = numerator / denominator
  ```
  pure twin (`get_betadelta.cool_beta_to_Ebdot_pure`), same equation:
  ```python
  a_coeff = 1.5 * pdotdot_total / pdot_total if pdot_total > 0 else 0.0
  c_coeff = 0.75 * pdot_total * R1
  d_coeff = R2**3 - R1**3

  Ebc = Eb + c_coeff
  c_frac = c_coeff / Ebc if Ebc > 0 else 0.0     # c/(E_b + c)
  ...
  if abs(denominator) < 1e-300:
      return 0.0
  return numerator / denominator
  ```
  and likewise `delta2dTdt` (`get_bubbleParams.py:42`) `dTdt = (T/t) * delta` vs
  `delta2dTdt_pure` (`get_betadelta.py:292`) `if t <= 0: return 0.0`.
- **class** — missing-propagation
- **severity** — S4 hygiene (both originals are unreferenced dead code today)
- **the difference** — Three guards — `pdot_total > 0`, `Eb + c_coeff > 0`,
  `|denominator| > 1e-300` — plus `t <= 0`, present only on the `_pure` side. The equations are
  otherwise term-for-term identical (both carry the same "Rahner thesis A12" derivation comment).
- **which is right** — The `_pure` versions: they are the ones the live solver calls
  (`run_energy_implicit_phase.py:992`), and the guards correspond to states the implicit phase
  actually visits (`pdot_total = 0` before the first wind, `Eb → 0` at the collapse boundary that
  DD-004/DD-008 are about).
- **failure scenario** — None currently: `grep` across `trinity/`, `test/` and `tools/` finds no
  caller of `cool_beta_to_Ebdot`, `delta2dTdt`, `dTdt2delta` or `Ebdot_to_cool_beta`. Flagged, not
  proposed for deletion (pre-existing dead code). The risk is that a future reader takes the
  unguarded, better-documented `get_bubbleParams` copy as the reference implementation.
- **confidence** — high

---

### DD-016 · The `cooling_balance` criterion exists twice: a parameterised event factory that is built and discarded, and a hardcoded-fallback inline check that is what actually runs
- **paths** — `trinity/phase_general/phase_events.py:319` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1296`

  factory, threshold fixed at construction (`phase_events.py:497`):
  ```python
  cooling_factory = make_cooling_balance_event(threshold=0.05)
  ```
  ```python
  def factory(Lgain: float, Lloss: float):
      def event(t, y):
          if Lgain <= 0:
              return 1.0  # No event if no gain
          ratio = (Lgain - Lloss) / Lgain
          return ratio - threshold
  ```
  received and never used (`run_energy_implicit_phase.py:752`):
  ```python
  ode_events, cooling_balance_factory = build_implicit_phase_events(params)
  ```
  what actually decides the phase end (`run_energy_implicit_phase.py:1250` + `:1296`):
  ```python
  phase_switch_threshold = params.get('phaseSwitch_LlossLgain', None)
  if phase_switch_threshold and hasattr(phase_switch_threshold, 'value'):
      threshold = phase_switch_threshold.value
  else:
      threshold = 0.05
  ...
  if 'cooling_balance' in active_triggers and Lgain > 0 and (Lgain - Lloss) / Lgain < threshold:
      termination_reason = "cooling_balance"
  ```
- **class** — divergence
- **severity** — S3 misleading (S1 if the factory were ever wired in)
- **the difference** — Two implementations of `(Lgain − Lloss)/Lgain < threshold`. The live one
  reads `phaseSwitch_LlossLgain` from `params`; the dead one hardcodes `0.05` at build time and
  would ignore the user's `.param`. The dead one is also sub-segment (an ODE event) while the live
  one is post-segment.
- **which is right** — The inline check: it honours the registered parameter, and it is the one
  phase 1a was given parity with (`run_energy_phase.py:275-287`, added by commit `2951c0c`) —
  1a copies the *inline* formula including the `phaseSwitch_LlossLgain` read and the
  `_thr if _thr else 0.05` fallback. The factory is a stranded earlier design.
- **failure scenario** — None today (the factory result is discarded). The hazard is a future edit
  that "activates" the event and silently pins every run to `threshold = 0.05` regardless of
  `phaseSwitch_LlossLgain`.
- **confidence** — high

---

### DD-017 · Four phases, four different depths of phase-boundary reconciliation
- **paths** — `trinity/phase1_energy/run_energy_phase.py:390` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1366` vs `trinity/phase1c_transition/run_transition_phase.py:833` vs `trinity/phase2_momentum/run_momentum_phase.py:886`

  1b and 1c (the deepest, and identical to each other) recompute `R1`, `Pb`, shell structure,
  `P_HII`, `F_HII`, and the full force set:
  ```python
  P_HII_f = (params['mu_convert'].value / params['mu_ion_shell'].value) * n_IF_Str_f * params['k_B'].value * params['TShell_ion'].value
  ...
  params['F_HII'].value = FOUR_PI * R2**2 * P_HII_f
  force_f = compute_forces_pure(R2, params['shell_mass'].value, Pb_f, shell_props_f, params)
  params['F_grav'].value = force_f.F_grav
  params['F_ion_in'].value = force_f.F_ion_in
  ...
  params.save_snapshot()
  ```
  1a recomputes `R1`, `Pb`, `shell_mass`, shell structure — and no `P_HII`, no forces:
  ```python
  mShell_f = mass_profile.get_mass_profile(R2, params, return_mdot=False)
  params['shell_mass'].value = mShell_f
  shell_f = shell_structure.shell_structure_pure(params)
  updateDict(params, shell_f)
  params.save_snapshot()
  ```
  2 recomputes `Pb` (= `pRam`), `R1 = R2`, shell structure — and no `shell_mass`, no `P_HII`, no forces:
  ```python
  params['Pb'].value = get_bubbleParams.pRam(
      R2, feedback_final.Lmech_total, feedback_final.v_mech_total)
  params['R1'].value = R2
  shell_props_f = shell_structure_pure(params)
  updateDict(params, shell_props_f)
  params.save_snapshot()
  ```
- **class** — missing-propagation
- **severity** — S3 misleading
- **the difference** — All four blocks carry the same stated purpose — the comment
  "A bare save_snapshot() would save stale derived values AND block the next phase's correct first
  snapshot via the duplicate guard" appears verbatim in 1a and 1b — but they refresh different
  subsets. 1a's and 2's boundary snapshots therefore contain `F_grav`, `F_ram`, `F_rad`,
  `F_ion_in`, `F_HII`, `P_HII`, `P_drive`, `P_ram` evaluated at the *previous* segment's `(t, R2,
  v2)`, tagged with the new ones. In phase 2 that is the run's final row for most stopping fates.
  A further sub-divergence: 1a is the only one that refreshes `shell_mass`, and 2 is the only one
  whose `except` clause reports where the failure happened
  (`run_momentum_phase.py:899-908`) — the other three log a bare message.
- **which is right** — 1b/1c. Their block is the one whose comment matches its behaviour, and
  the duplicate-guard argument in the shared comment applies identically to all four.
- **failure scenario** — Every run. The last row of `dictionary.jsonl` (written by phase 2's
  reconciliation on the normal `reached_tmax` / `large_radius` / `dissolved` exits) mixes
  time-`t_final` state with time-`t_previous` forces. Any figure that plots the force budget to
  the end of the run, or any analysis keyed on the final snapshot, reads the mismatch as physics.
- **confidence** — high

---

### DD-018 · Energy-phase solver tolerance is one decade tighter than its three twins, and two twins carry a copy-pasted, now-wrong `max_step` comment
- **paths** — `trinity/phase1_energy/run_energy_phase.py:58` vs `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:170` / `trinity/phase1c_transition/run_transition_phase.py:132` / `trinity/phase2_momentum/run_momentum_phase.py:124`

  1a:
  ```python
  RTOL = 1e-6  # Relative tolerance for solve_ivp
  ATOL = 1e-9  # Absolute tolerance for solve_ivp
  ```
  1b (with the change recorded in the comment):
  ```python
  ODE_RTOL = 1e-6      # Relative tolerance
  ODE_ATOL = 1e-8      # Absolute tolerance (relaxed from 1e-9)
  ODE_MIN_STEP = 1e-6  # Minimum step size (Myr)
  ODE_MAX_STEP = DT_SEGMENT_MIN / 5  # Max step = 2e-5 Myr (ensures >=5 steps per segment)
  ```
  1c and 2 — same `ODE_MAX_STEP` line, but their `DT_SEGMENT_MIN` is `1e-3`, not `1e-4`:
  ```python
  DT_SEGMENT_MIN = 1e-3  # Myr - minimum segment duration
  ...
  ODE_MAX_STEP = DT_SEGMENT_MIN / 5  # Max step = 2e-5 Myr (ensures >=5 steps per segment)
  ```
- **class** — divergence (tolerance) + S4 stale comment (`max_step`)
- **severity** — S4 hygiene / S3 misleading
- **the difference** — (a) `atol`: 1b's comment shows `1e-9 → 1e-8` was a deliberate relaxation;
  1c and 2 carry the relaxed value, 1a still carries `1e-9`. (b) `ODE_MAX_STEP`: the comment's
  "2e-5 Myr" is correct in 1b and wrong by 10× in 1c and 2, where the expression evaluates to
  `2e-4`. The parenthetical "(ensures >=5 steps per segment)" is still true in all three; only the
  quoted number is stale, which is the fingerprint of a block copied from 1b without re-deriving
  it.
- **which is right** — For `atol`: undecidable from the code. 1a is a fixed-window RK45 phase with
  3×10⁻⁵ Myr segments while the others are LSODA with 10²–10³× longer segments, so a tighter
  absolute tolerance in 1a is defensible — but no comment says so, and the "relaxed from 1e-9"
  note reads as a global decision that reached three files out of four. Marked medium.
  For the comment: 1b's is right; 1c's and 2's are wrong.
- **failure scenario** — `atol` is compared against `Eb`, which is O(10⁴–10⁵) in code units, so the
  absolute tolerance is inert for `Eb`; it bites on `v2` and `R2` near zero, i.e. during collapse.
  A shell decelerating through `v2 ≈ 0` is resolved to 1e-9 in 1a and 1e-8 in 1b/1c/2 — a step-size
  discontinuity across the 1a→1b boundary in exactly the regime DD-001 and DD-005 also concern.
- **confidence** — medium (tolerance intent), high (stale comment)

---

### DD-019 · The GMC validator rejects `nEdge < nISM`; the initialiser silently rewrites `rCore` or `nCore` to fix it
- **paths** — `trinity/cloud_properties/validate_gmc.py:237` vs `trinity/phase0_init/get_InitCloudProp.py:180`

  validator — hard error:
  ```python
  if nEdge < nISM:
      ...
      errors.append(f"Edge density ... below ISM ...")
  ```
  initialiser — silent (well, `logger.warning`) auto-correction of the user's inputs:
  ```python
  if nEdge < nISM and alpha != 0:
      ...
      rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)
      if rCore_min < rCloud:
          ...
          rCore = rCore_try
          rCloud = rCloud_try
          params['rCore'].value = rCore
      ...
      if use_nCore_fix:
          nCore = nCore_min
          params['rCore'].value = rCore
          params['nCore'].value = nCore
  ```
- **class** — divergence
- **severity** — S3 misleading
- **the difference** — Two paths derive `(rCloud, nEdge)` from the same `(mCloud, nCore, alpha,
  rCore)` and reach opposite verdicts on the same condition: the validator reports the
  configuration invalid, the initialiser rewrites `params['rCore']` and/or `params['nCore']` and
  proceeds. The validator's reported `rCloud`/`nEdge` are the *pre*-correction values, so its
  output does not describe the cloud the run will actually integrate.
- **which is right** — They serve different purposes (a pre-flight check vs. a runtime fixer), so
  neither is simply wrong — but they cannot both be the authority on what `nCore`/`rCore` a run
  uses, and today the validator's numbers are not the run's numbers. The initialiser at least logs
  a `WARNING` naming the old and new values.
- **failure scenario** — Steep-`alpha` power-law clouds near the `nEdge ≈ nISM` boundary (the
  common `alpha = -2` isothermal setup with a low `nISM`). A user validates a config, sees
  `rCloud = X`, runs it, and gets a cloud with a different `rCore`, `nCore` and `rCloud` — silently,
  unless they read the log. Every derived quantity (`rCloud`, `nEdge`, the whole density profile)
  shifts.
- **confidence** — medium (the behavioural split is certain; whether it is considered a defect is
  a maintainer call)

---

## Checked and consistent

Twin sets I put side by side and found in genuine agreement (or differing only for a reason the
code states):

1. **Gravity term** — `F_grav = G·mShell/R2²·(mCluster + 0.5·mShell)` is character-identical in all
   four force assemblies: `energy_phase_ODEs.py:220` and `:370`, `run_energy_implicit_phase.py:491`,
   `run_transition_phase.py:284`, `run_momentum_phase.py:222` and `:418`. Same sign, same
   half-shell self-gravity factor, no phase-dependent gating.
2. **Radiation-pressure term (direct + IR-trapped)** — identical five-line block, including the
   `isDissolved → F_rad = 0` gate, in `energy_phase_ODEs.py:130-135`,
   `run_energy_implicit_phase.py:542-547`, `run_transition_phase.py:341-346`,
   `run_momentum_phase.py:275-280` and `:340-345`. Same `(1 + tauKappaRatio·dust_KappaIR)`
   trapping factor everywhere.
3. **Inward ionised-gas pressure `P_ext`** — same formula, same `FABSi < 1.0` switch-off, same
   `rShell >= rCloud → P_ext += PISM·k_B` boundary using `>=` in all four
   (`energy_phase_ODEs.py:237-244` and `:374-380`, `run_energy_implicit_phase.py:506-521`,
   `run_transition_phase.py:299-314`, `run_momentum_phase.py:240-254` and `:425-438`). Only the
   exception handling differs — DD-014.
4. **`P_HII` from the Strömgren balance** — the gate
   `if params['include_PHII'].value and n_IF_Str > 0` and the identical
   `(mu_convert/mu_ion_shell)·n_IF_Str·k_B·TShell_ion` expression appear verbatim in all four
   runners (`run_energy_phase.py:213-217`, `run_energy_implicit_phase.py:980-984`,
   `run_transition_phase.py:563-567`, `run_momentum_phase.py:633-637`).
5. **`P_drive` composition** — `max(Pb, P_HII)` in energy/implicit, `max(Pb, P_HII + P_ram)` in
   transition, `P_HII + P_ram` in momentum. Intended and consistent with the phase physics: the
   thermal term is dropped exactly where `Eb ≡ 0`. I also checked that transition's two
   evaluations agree despite looking different — the ODE computes
   `max(max(P_thermal, P_ram), P_HII + P_ram)` via `get_effective_bubble_pressure`
   (`get_bubbleParams.py:352-364`) while `compute_forces_pure` computes `max(P_thermal,
   P_HII + P_ram)`; these are algebraically equal for `P_HII >= 0`.
6. **`F_ram` bookkeeping** — `Pb·4πR²` in 1a/1b/1c and `P_ram·4πR²` in 2, which is the same
   quantity because phase 2 sets `params['Pb'] = pRam(...)` (`run_momentum_phase.py:585` and `:667`).
7. **Shell-mass monotonicity + collapse-freeze guard** — the "shell mass can NEVER decrease" and
   `isCollapse → mShell_dot = 0` pair is present, with the same comparison
   (`if prev_mShell > 0 and mShell_new < prev_mShell`), in all four runners and in the shared RHS.
   1b/1c/2 additionally repeat it for the post-ODE adaptive-stepping comparison, with the same
   logic. (What differs is *when* it is evaluated in phase 2 — DD-003.)
8. **The adaptive-stepping machinery** — `compute_max_dex_change` and `get_monitor_values` are
   byte-identical triplicates in `run_energy_implicit_phase.py:289/412`,
   `run_transition_phase.py:143/164` and `run_momentum_phase.py:135/156`, including the sign-flip
   →`1.0` rule and the zero/None skips. `ADAPTIVE_MONITOR_KEYS` is the same 34-key list in all
   three.
9. **Velocity-based collapse step control** — same thresholds (`50.0`, `150.0` pc/Myr), same
   `v2 < 0` gate, same two-tier response in 1b/1c/2. Only `DT_SEGMENT_COLLAPSE` differs
   (`5e-5` in 1b vs `5e-4` in 1c/2), and 1b's constant carries the reason inline: "50 years,
   tighter than other phases".
10. **Adaptive threshold** — `ADAPTIVE_THRESHOLD_DEX = 0.05` in 1b vs `0.1` in 1c/2, and 1b alone
    routes `dt` through `next_dt_segment` with the beta-delta non-convergence guard. Intended: 1b
    is the only phase with an inner implicit solve whose non-convergence must throttle the outer
    step, and `next_dt_segment`'s docstring states exactly that.
11. **The `min_radius` / `velocity_runaway` event definitions** — identical construction
    (`max(coll_r * 1.5, 0.01)` pc; `MAX_VELOCITY_COLLAPSE = 500.0` pc/Myr, `direction="collapse"`)
    across all four builders, with matching `terminal=True` and `direction=-1`. I also checked the
    three `make_velocity_runaway_event` direction variants: all three residuals
    (`v2 + v_max`, `v_max - v2`, `v_max - abs(v2)`) are decreasing through the trigger, so
    `direction = -1` is right in all three.
12. **`skipped_past_stop_t` early return** — 1b, 1c and 2 each carry the identical
    `if tmin >= tmax:` guard with the same `STOPPING_TIME` code and reason string. 1a lacks it,
    correctly: it is the first phase, so `t_now` cannot already exceed `stop_t`.
13. **`stop_at_rCloud_nSnap` handling** — identical top-of-loop check plus the identical
    `save_count`-delta-guarded increment in 1b, 1c and 2. 1a's absence is covered by
    `main.py:263-274`, which handles the `nSnap == 0` case at the 1a boundary.
14. **`termination_reason is None → "max_segments"/"unknown"` epilogue** — identical in 1b, 1c
    and 2, including the `logger.warning` escalation for `"unknown"`.
15. **Power-law enclosed-mass formula, four copies** —
    `mass_profile.compute_enclosed_mass_powerlaw:332`,
    `powerLawSphere.compute_rCloud_powerlaw.mass_at_radius:136`,
    `get_InitCloudProp._init_powerlaw_cloud:263` (the forward mass check), and
    `validate_gmc._validate_powerlaw:435` all write
    `4π·ρc·[rc³/3 + (r^(3+α) − rc^(3+α))/((3+α)·rc^α)]` with the same `3+α` handling. No sign,
    exponent or factor drift. (`powerLawSphere` additionally guards `|3+α| < 1e-14`; the others do
    not need to, since they never divide by it in isolation.)
16. **PL vs BE branches of `density_profile.get_density_profile`** — both apply the same
    `w_outside` tanh bridge with the same `SMOOTH_FRAC`, the same
    `n_inside·(1−w) + nISM·w` blend, and the same scalar/array round-tripping. Agreement here is
    what makes DD-010 a `density_profile`-vs-`mass_profile` split rather than a PL-vs-BE split.
17. **`_init_powerlaw_cloud` vs `_init_bonnor_ebert_cloud`** — same `_create_radius_array`, same
    `get_density_profile`/`get_mass_profile` calls, same write-back of `rCloud`/`rCore`/`nEdge`.
    The BE branch's extra Lane-Emden params (`densBE_f_m`, `densBE_xi_out`) and the PL branch's
    `nEdge < nISM` auto-correction are profile-specific by construction (BE's edge density is
    `nCore/Ω` and cannot undershoot the way a power-law tail can).
18. **BE unit round-trip** — `create_BE_sphere`'s `T_eff = mu·MSUN_TO_G·c_s²/(γ·k_B)`
    (`bonnorEbertSphere.py:431`) and `r2xi`'s `c_s = sqrt(γ·k_B·T_eff/(mu·MSUN_TO_G))`
    (`:606`) are exact inverses, and `r2xi`/`xi2r` are mutual inverses through the same
    `a = c_s/sqrt(4πGρc)`. `create_BE_sphere_from_params` and `_init_bonnor_ebert_cloud` write the
    same key set with the same `c_s / 1e5` cm/s→km/s conversion. No unit divergence found.
19. **CIE vs non-CIE cooling in `net_coolingcurve.get_dudt`** — the three branches partition the
    temperature axis without gap or overlap (`log T <= nonCIE_Tcutoff`,
    `log T >= CIE_Tcutoff`, and the strict-inequality interpolation band between), and the
    interpolation branch evaluates each end with the *same* expression its own branch uses
    (`netcool_interp` at `nonCIE_Tcutoff`; `chi_e·n²·Λ` at `10**CIE_Tcutoff`). The `chi_e·n²`
    factor appearing on the CIE side and not the non-CIE side is intended — the non-CIE cloudy
    table is already a volumetric net rate — and the `−1` sign convention is applied identically on
    all three returns. The two cached-cutoff helpers (`_noncie_cutoffs`, `_cie_tcutoff`) are the
    same reduction over the same arrays as the expressions they replaced.
20. **`bubble_luminosity`'s own CIE/non-CIE zone split** — the `_CIEswitch = 10**5.5` constant is
    used consistently for the L1/L2/L3 partition and for the intermediate-region regime mask
    (`bubble_luminosity.py:706, 772, 816`), and the file's comment at `:61` names the three sites
    that must stay in lockstep with the cooling-table-derived `nonCIE_Tcutoff`. Verified they agree.
21. **`_solve_betadelta_hybr` vs `_solve_betadelta_legacy`** — they genuinely differ (hybr uses the
    pole-free `g = (Edot_from_beta − Edot_from_balance)/Lmech_total` residual and a `dMdt > 0`
    acceptance gate that can return `no_physical_root`; legacy uses the `f` residual normalised by
    `Edot_from_beta` and has no gate), but every difference is documented in-place with its reason,
    and both feed the same `get_residual_detailed`. The consequence that `no_physical_root` — and
    therefore the `NO_ROOT_HANDOFF_STREAK` handoff — is unreachable under `betadelta_solver='legacy'`
    is a real behavioural fork, but it follows from the documented design ("byte-identical to the
    pre-switch behaviour") rather than from a fix landing on one side.
22. **`compute_R1_Pb` vs 1a's inline `solve_R1` + `bubble_E2P`** — the same two calls in the same
    order with the same arguments (`get_betadelta.py:327-329` vs `run_energy_phase.py:393-395`).
    No divergence.
23. **`effective_Lloss_from_params` at its three call sites** — the beta-delta residual
    (`get_betadelta.py:577`), the 1b energy→momentum trigger
    (`run_energy_implicit_phase.py:1243/1247`) and the 1a parity check
    (`run_energy_phase.py:279`) all route through the same one-line wrapper with the same
    `(Lcool, leak, Lmech)` argument order — this is the parity commit `2951c0c` landing correctly.
24. **`t_previousCoolingUpdate` cadence** — 1a checks once before its loop with
    `COOLING_UPDATE_INTERVAL = 5e-2` Myr; 1b checks every segment with `5e-3` Myr. Intended: the
    default is `1e30` (`registry.py:455`) so 1a's one-shot always fires, and 1a's total duration
    (3e-3 Myr) is below either interval, so a per-segment check would never fire twice.
    1a's `params['t_previousCoolingUpdate'] - params['t_now']` without `.value` is legal —
    `DescribedItem.__sub__` unwraps both operands (`dictionary.py:176`) — and returns the same
    float as 1b's explicit form.
25. **`shell_structure` ionised vs neutral integration loops** — same slice-and-continue structure,
    same `mShell_arr_cum >= mShell_end` mass-limited termination, same `[:idx]` concatenation plus
    final-value append, same reinitialisation of `(nShell0, tau0, mShell0, rShell_start)` from
    index `idx`, and the same gravity accumulation chained through `grav_ion_m_cum[-1]`. The
    neutral loop's extra `nsteps = 5e3` (vs `1e3`) and its lack of a `phiCondition` are consistent
    with there being no `φ` to deplete there. `f_cover` is the one asymmetry — DD-012.
26. **`initial_profile.build_initial_cloud_profile`** — a reconstruction path that could easily
    have drifted from the initialiser, but does not: it delegates to the same
    `_init_powerlaw_cloud` / `_init_bonnor_ebert_cloud` through a `_MockItem` adapter rather than
    re-deriving anything. Single-sourced by construction.
27. **`F_ISM`** — appears in the `ADAPTIVE_MONITOR_KEYS` list of 1b, 1c and 2 and is computed by
    none of them. Not a divergence: the registry declares it as
    "ISM pressure force (placeholder, never computed — always 0)" (`registry.py:486`), so all four
    phases agree.
