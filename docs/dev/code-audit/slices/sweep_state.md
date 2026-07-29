# Sweep: state mutation, aliasing and ordering dependence

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

Read-only audit of `/home/user/trinity/trinity/**` + `/home/user/trinity/run.py`.
Nothing was edited. Two claims below were verified by *executing* code
(ST-002 via a synthetic `solve_ivp` reproduction; ST-008's initial conditions via a
real `read_param` + `get_y0` on `param/simple_cluster.param`); everything else is
established by reading the source.

**Headline answers**

* **Would a second simulation in the same process be corrupted?** *Yes, but only in
  narrow ways* — see ST-003 (id-keyed cooling cutoff cache), ST-004 (atexit/signal
  handlers accumulate and the first run's output dir is rewritten at exit), ST-015
  (diagnostic counters). There is **no** module-level physics cache that would make a
  second run's trajectory differ from the first under the default configuration.
* **Is the sweep path safe?** *Yes — but only because of process isolation, twice
  over.* `run.py:run_sweep` submits to a `ProcessPoolExecutor`, and each pool task
  (`sweep_runner.run_single_simulation`) then `subprocess.run`s a **fresh
  `python run.py <param>`**. No `params` object, no module global, and no interpreter
  is shared between two configurations. See ST-021.

---

## Findings

### ST-001 · Phase 1a's event exit leaves the loop's local state variables one segment stale, and the reconciliation snapshot is built from them
- **file:line** — `trinity/phase1_energy/run_energy_phase.py:324-331`
  ```python
  event_result = check_event_termination(solution, ode_events)
  if event_result.triggered:
      logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr")
      apply_event_result(params, event_result, event_result.t, event_result.y,
                        state_keys=['R2', 'v2', 'Eb'])
      if event_result.is_simulation_ending:
          return
      break
  ```
  and `trinity/phase1_energy/run_energy_phase.py:390-402`
  ```python
      feedback_final = get_current_sps_feedback(t_now, params)
      updateDict(params, feedback_final)
      R1_f = get_bubbleParams.solve_R1(R2, Eb, feedback_final.Lmech_total,
                                       feedback_final.v_mech_total)
      Pb_f = get_bubbleParams.bubble_E2P(Eb, R2, R1_f, params['gamma_adia'].value)
      params['R1'].value = R1_f
      params['Pb'].value = Pb_f
      mShell_f = mass_profile.get_mass_profile(R2, params, return_mdot=False)
      params['shell_mass'].value = mShell_f
      shell_f = shell_structure.shell_structure_pure(params)
      updateDict(params, shell_f)
      params.save_snapshot()
  ```
- **class** — stale-key (compounded by ordering-dependence)
- **severity** — S1 results-wrong
- **the mechanism** —
  1. `apply_event_result` (`phase_events.py:612-617`) writes the **event-time** state into
     `params['t_now'] / ['R2'] / ['v2'] / ['Eb']`.
  2. It does **not** touch the runner's local Python names `t_now, R2, v2, Eb`. The block
     that would have refreshed them (`run_energy_phase.py:336-349`,
     `R2_new, v2_new, Eb_new = solution.y[:, -1] ... R2 = R2_new`) is *below* the `break`
     and never executes.
  3. Control falls to the reconciliation block, which reads the **locals** for
     `get_current_sps_feedback(t_now, …)`, `solve_R1(R2, Eb, …)`, `bubble_E2P(Eb, R2, …)`
     and `get_mass_profile(R2, …)`, but writes the results back into a `params` whose
     `R2/v2/Eb/t_now` are the *new* event values.
  4. `shell_structure_pure(params)` then reads `params['R2']` (new) together with
     `params['Pb']`, `params['shell_mass']` (both computed from the old `R2`, `Eb`),
     producing a shell profile from mixed state.
  5. `params.save_snapshot()` persists that mixture as the phase-1a → phase-1b handoff
     row, and phase 1b starts from `params` in that mixed condition
     (`run_energy_implicit_phase.py:693-696` reads `R2/v2/Eb/T0` back out of `params`,
     while `params['Pb']`, `params['R1']`, `params['shell_mass']` are stale).
- **when it bites** — Only on the event-exit path of phase 1a, i.e. when
  `cloud_boundary` (`R2 > rCloud`), `min_radius` or `velocity_runaway` fires *during* a
  segment. `cloud_boundary` is the ordinary energy→implicit handoff for any config where
  the shell reaches the cloud edge inside `TFINAL_ENERGY_PHASE = 3e-3` Myr. For
  `param/simple_cluster.param` the measured start state is `rCloud = 1.690 pc`,
  `r0 = 1.27e-3 pc`, `v0 = 3739 pc/Myr` — the shell traverses `rCloud` well inside the
  3000-yr window, so this path is exercised by the project's own quickstart config.
  **Affects a single run in a fresh process.**
- **observable symptom** — one snapshot in `dictionary.jsonl` at the 1a/1b boundary where
  `Pb`, `R1`, `shell_mass`, `rShell`, `shell_nMax`, the `shell_*` absorption fractions and
  the shell profile arrays correspond to a radius/energy that no longer matches the `R2`,
  `Eb`, `t_now` on the same line; phase 1b's first `bubble_Leak` and first `betadelta`
  residual are seeded from that inconsistent `Pb`. Silent — the values are all finite.
- **confidence** — high

---

### ST-002 · A *non-terminal* monitoring event (`velocity_sign`) is reported as a triggered termination, ending the implicit phase and rolling `params` back to the crossing
- **file:line** — `trinity/phase_general/phase_events.py:392-405`
  ```python
  for i, (t_ev, y_ev) in enumerate(zip(sol.t_events, sol.y_events)):
      if len(t_ev) > 0:
          event = events[i]
          return EventResult(triggered=True, ...)
  ```
  with `trinity/phase_general/phase_events.py:310` (`event.terminal = False  # Non-terminal
  by default - just records the crossing`) and `phase_events.py:487-491`, which puts
  `make_velocity_sign_event()` **first** in the implicit-phase event list.
  Consumer: `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095-1119`.
- **class** — ordering-dependence
- **severity** — S1 results-wrong
- **the mechanism** — `solve_ivp` records *every* root of *every* event in
  `t_events`/`y_events`, terminal or not, and keeps integrating past a non-terminal one.
  `check_event_termination` inspects only `len(t_ev) > 0` and never consults
  `event.terminal`. Because `velocity_sign` is index 0, any `v2` crossing from `+` to `−`
  inside a segment makes the function return `triggered=True`. The implicit runner then
  (a) sets `termination_reason = "velocity_sign_change"`, (b) **discards** the segment's
  real endpoint `sol.y[:, -1]` and rewinds `R2/v2/Eb/T0/t_now` to the crossing point,
  (c) calls `apply_event_result` (which writes those rewound values into `params`), and
  (d) `break`s out of the phase entirely. The module's own docstring
  (`phase_events.py:25-26`) states these events "record a crossing only".
  Verified by execution:
  ```
  t_events [array([1.]), array([], dtype=float64)]     # integration ran to t=2.0
  triggered: True  name: velocity_sign  is_sim_ending: False  t: 1.0
  ```
- **when it bites** — Any configuration in which `v2` changes sign during a phase-1b
  segment, i.e. every run that begins to collapse while still energy-driven. Also present
  in `build_transition_phase_events`/`build_momentum_phase_events`? No — `velocity_sign`
  is only in the implicit list, so the blast radius is phase 1b.
  **Affects a single run in a fresh process.**
- **observable symptom** — implicit phase reported as
  `Implicit phase completed: velocity_sign_change` at the first inward excursion; the
  collapse-detection branch below (`if v2 < 0 and R2 < R2_prev: params['isCollapse']=True`,
  line 1302) is never reached for that segment, so `isCollapse` stays `False` while the
  run hands off to the transition phase. Simulated time also *goes backwards* relative to
  the integration the solver actually performed.
- **confidence** — high (mechanism executed and reproduced; the only judgement call is
  whether the "monitoring only" docstring or the runner's `break` is the intent)

---

### ST-003 · `_CIE_TCUTOFF_CACHE` is keyed by `id()` of a NumPy array — unbounded, never invalidated, and vulnerable to address reuse across runs
- **file:line** — `trinity/cooling/net_coolingcurve.py:27-29, 48-55`
  ```python
  _CIE_TCUTOFF_CACHE: dict = {}   # keyed by id(logT_CIE); logT_CIE is built once at
                                  # startup (main.py) and never replaced, so its id
                                  # is stable for the whole run -> no id-reuse hazard.
  def _cie_tcutoff(logT_CIE):
      key = id(logT_CIE)
      cached = _CIE_TCUTOFF_CACHE.get(key)
      if cached is None:
          cached = min(logT_CIE[logT_CIE > 5.5])
          _CIE_TCUTOFF_CACHE[key] = cached
      return cached
  ```
- **class** — module-global
- **severity** — S2 latent
- **the mechanism** — The comment's premise ("never replaced, so its id is stable") holds
  *within one run*: `main.py:169` writes the array once and it stays reachable through
  phases 1a/1b/1c. But (a) `main.py:317` → `params.reset_keys(COOLING_PHASE_KEYS)` sets
  both `cStruc_cooling_CIE_logT` and `cStruc_cooling_CIE_interpolation` to `np.nan`,
  dropping the last references so the array becomes collectable; and (b) nothing ever
  clears `_CIE_TCUTOFF_CACHE`. In a process that runs a second simulation, the fresh
  `np.loadtxt` array can be handed the *same* address CPython just freed, so
  `id(new_array) == id(old_array)` and the stale cutoff of the *previous* run's CIE table
  is returned without recomputation. The dict also grows one entry per array ever seen and
  is never pruned (`id()` keys keep no reference, so entries can never be matched again
  once the object dies — they are pure leak).
- **when it bites** — Never for one run in a fresh process (the cache is written once,
  read with the same live object). Only for a **second in-process run**, and only
  materially when the two runs use different CIE curves (`path_cooling_CIE` selects among
  four bundled tables at `read_param.py:417-429`). The sweep path never hits it (ST-021).
- **observable symptom** — a second in-process run silently using the first run's
  non-CIE→CIE switch temperature, shifting `L_conduction`/`L_bubble` apportionment in
  `bubble_luminosity` and hence the cooling balance; plus monotonic memory growth in any
  long-lived embedding process.
- **confidence** — high on the mechanism, medium on the practical impact (all four bundled
  tables share the same `logT` grid, so the cached value would coincide in most cases)

---

### ST-004 · Every `DescribedDict` instance permanently registers a process-global `atexit` handler and hijacks `SIGINT`/`SIGTERM`
- **file:line** — `trinity/_input/dictionary.py:281-288`
  ```python
  def atexit_handler():
      reason = self._termination_reason or "Normal exit / atexit"
      self._safe_flush(termination_reason=reason)
  atexit.register(atexit_handler)

  # Signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill)
  signal.signal(signal.SIGINT, self._signal_handler)
  signal.signal(signal.SIGTERM, self._signal_handler)
  ```
  Constructed at `trinity/_input/read_param.py:253` and at
  `trinity/_input/dictionary.py:951` (`DescribedDict.load_snapshot` → `params = cls()`).
- **class** — module-global (process-global registration), sweep-unsafe-if-inlined
- **severity** — S2 latent
- **the mechanism** — The handler closes over `self` and is never unregistered. Two
  consequences in a process that builds more than one `DescribedDict`:
  1. At interpreter exit *every* registered handler fires. Each one calls `_safe_flush`,
     which writes `termination_debug` into `metadata.json` **and** rewrites
     `metadata_humanreadable.txt` in *that* dict's `path2output`
     (`dictionary.py:326-342`). A run whose output dir was already finalised gets its
     termination block rewritten by an exit that belongs to a different run.
  2. `signal.signal` is last-writer-wins: after a second `DescribedDict` exists, `Ctrl+C`
     flushes only the newest one and then `sys.exit(128+signum)`
     (`dictionary.py:300`) — the older run's pending snapshots are lost.
     This also silently clobbers `run.py`'s own sweep handlers
     (`run.py:600-605`) if a sweep were ever changed to build `params` in-process, and
     `run.py`'s `finally` block (`run.py:697-702`) would then restore a handler that no
     longer matches the live dict.
  Note `load_snapshot` is a *reader* API: merely reading an old run's snapshots installs an
  exit hook that rewrites that run's metadata.
- **when it bites** — Not for one run in a fresh process. Bites the second in-process run,
  any embedding/notebook workflow, and any test session that constructs several dicts.
  Not the current sweep path (ST-021), but it is exactly what would break first if the
  sweep were refactored to run in-process.
- **observable symptom** — `metadata.json[termination_debug]` / `metadata_humanreadable.txt`
  of an already-finished run being rewritten at a later exit; lost snapshots on `Ctrl+C`
  in a multi-run process.
- **confidence** — high

---

### ST-005 · The transition phase's energy ODE reads `bubble_LTotal` (and `bubble_Tavg` → `c_sound`) as live, but no writer runs during that phase — they are frozen at the phase-1b exit value
- **file:line** — `trinity/phase1_energy/energy_phase_ODEs.py:146`
  (`bubble_LTotal=params['bubble_LTotal'].value`) and `:273-280`
  ```python
  L_bubble = snapshot.bubble_LTotal
  L_leak = get_bubbleParams.get_leak_luminosity(
      snapshot.coverFraction, R2, press_bubble, snapshot.c_sound, snapshot.gamma_adia)
  Ed = (Lmech_total - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak
  ```
  Reached from `trinity/phase1c_transition/run_transition_phase.py:227` via
  `get_ODE_transition_pure` → `get_ODE_Edot_pure`; the snapshot is rebuilt each segment at
  `run_transition_phase.py:616` but re-reads the same unchanged key.
  The only writer of `bubble_LTotal` is `updateDict(params, bubble_props)` in phases 1a/1b
  (`run_energy_phase.py:184`, `run_energy_implicit_phase.py:894`).
- **class** — stale-key
- **severity** — S2 latent
- **the mechanism** — Phase 1c never calls `bubble_luminosity.get_bubbleproperties_pure`,
  so `params['bubble_LTotal']` retains the value written by the last successful phase-1b
  bubble solve. `create_ODE_snapshot` re-reads it every segment (giving the appearance of
  a live quantity) and the transition energy ODE subtracts it from `Lmech_total` while
  `R2`, `Eb`, `t` evolve — potentially for the whole duration of phase 1c. The same holds
  for `c_sound`, derived at `run_transition_phase.py:512-518` from a `bubble_Tavg` that is
  likewise frozen (with a hard `1e6 K` fallback when it is falsy).
- **when it bites** — Every run that enters phase 1c, in a **fresh process**. The error
  grows with the length of the transition phase.
- **observable symptom** — `Ed_energy_balance` in the transition ODE tracks a cooling
  luminosity from an earlier time; the `min(Ed_energy_balance, Ed_soundcrossing)` handoff
  point (and therefore the transition→momentum time) shifts. Nothing in the log flags it.
- **confidence** — high on the mechanism; medium on intent (the docstring at
  `run_transition_phase.py:200-206` argues for continuity with the implicit phase, which
  suggests the *first* segment's reuse is deliberate — freezing it for the whole phase is
  not stated anywhere)

---

### ST-006 · `T0` is written into `params` every segment of phases 1c and 2 from a local that no phase updates — a snapshot column that looks live and is frozen
- **file:line** — `trinity/phase1c_transition/run_transition_phase.py:430` (`T0 = params['T0'].value`)
  and `:491` (`params['T0'].value = T0`); identically
  `trinity/phase2_momentum/run_momentum_phase.py:509` and `:572`.
- **class** — stale-key
- **severity** — S3 misleading
- **the mechanism** — Both phases integrate a state vector without `T0`
  (`[R2, v2, Eb]` and `[R2, v2]`), so the local `T0` read once at phase entry is never
  reassigned. Writing it back at the top of every segment refreshes the *dict entry* but
  not the *value*, so every snapshot of both phases records the phase-1b exit
  temperature under the `T0` key, next to a `t_now` that keeps advancing. The same value
  also lands in `metadata.json[final_state]` via
  `simulation_end.write_simulation_end` → `_build_final_state_block`.
- **when it bites** — Every run that reaches phase 1c, **fresh process**.
- **observable symptom** — a flat `T0` plateau across the entire transition and momentum
  phases in any plot built from `dictionary.jsonl`, and a `final_state.T0` that describes
  a state hundreds of segments earlier.
- **confidence** — high

---

### ST-007 · The bubble structure arrays and bulk bubble scalars survive `reset_keys(COOLING_PHASE_KEYS)` and are re-serialised into every transition/momentum snapshot as if current
- **file:line** — `trinity/_input/dictionary.py:1199-1203` and `:1217-1222`
  ```python
      # Bubble temperature/mass
      # 'bubble_Tavg',
      # 'bubble_T_r_Tb',
      # 'bubble_mass',
      # 'bubble_r_Tb',
  ...
      # Bubble profile arrays
      # 'bubble_v_arr',
      # 'bubble_T_arr',
      # 'bubble_dTdr_arr',
      # 'bubble_r_arr',
      # 'bubble_n_arr',
  ```
  Reset call site: `trinity/main.py:317` (`params.reset_keys(COOLING_PHASE_KEYS)`).
  Re-serialisation: `trinity/_input/dictionary.py:639-701` (`_clean_for_snapshot`'s
  `bubble_*` special cases, which run on every `save_snapshot`).
- **class** — stale-key
- **severity** — S3 misleading
- **the mechanism** — `COOLING_PHASE_KEYS` deliberately blanks the *cooling* scalars
  (`bubble_LTotal`, `bubble_L1Bubble`, …, `bubble_dMdt`) to `np.nan` after phase 1c, but the
  profile arrays and `bubble_mass` / `bubble_Tavg` / `bubble_r_Tb` / `bubble_T_r_Tb` are
  commented out of the list. They therefore keep the last phase-1b solve's contents.
  `_clean_for_snapshot` unconditionally re-simplifies and re-writes
  `log_bubble_T_arr`, `bubble_T_arr_r_arr`, `log_bubble_n_arr`, `bubble_n_arr_r_arr`,
  `log_bubble_dTdr_arr`, `bubble_v_arr`, … into *every* subsequent snapshot. Also:
  `shell_structure_pure` reads `mBubble = params['bubble_mass'].value`
  (`shell_structure.py:103`) and adds it to the cumulative shell gravity in the momentum
  phase, where the hot bubble no longer exists.
- **when it bites** — Every run reaching phase 1c/2, **fresh process**. Also inflates
  `dictionary.jsonl` by `simplify_npoints` points × 5 arrays × every post-1b snapshot.
- **observable symptom** — a bubble radial profile that is byte-identical across every
  transition- and momentum-phase snapshot; `bubble_mass` frozen while `t_now` advances.
- **confidence** — high

---

### ST-008 · `EarlyPhaseApproximation` is consumed by the segment-0 ODE snapshot and only flipped *after* segment 0 has been integrated
- **file:line** — `trinity/phase1_energy/energy_phase_ODEs.py:268-270`
  ```python
  # Early phase approximation
  if snapshot.EarlyPhaseApproximation:
      vd = -1e8
  ```
  Snapshot build: `energy_phase_ODEs.py:159` (`EarlyPhaseApproximation=params['EarlyPhaseApproximation'].value`),
  called at `run_energy_phase.py:292`. Flip: `run_energy_phase.py:342-344`
  ```python
  if loop_count == 0 and params['EarlyPhaseApproximation'].value:
      params['EarlyPhaseApproximation'].value = False
  ```
  Default `True` at `trinity/_input/registry.py:423`.
- **class** — ordering-dependence
- **severity** — S2 latent
- **the mechanism** — The flag is `True` when the first segment's frozen snapshot is
  built, so the whole of segment 0 integrates with a constant, hard-coded
  `dv/dt = -1e8 pc/Myr²` that has no dependence on any physical quantity. The flip to
  `False` happens strictly after `solve_ivp` returns. The velocity removed is therefore
  `1e8 × SEGMENT_DURATION = 1e8 × 3e-5 = 3000 pc/Myr` — a number set by the *segment
  length constant* (`run_energy_phase.py:55`), not by physics. For
  `param/simple_cluster.param` the measured `v0 = 3739.24 pc/Myr`, so segment 0 removes
  ~80 % of the initial velocity. For any configuration whose free-streaming
  `v0 = 2·Lmech_W/pdot_W` is below ~3500 pc/Myr, `v2` is driven **negative** in the first
  30 yr; below ~2500 pc/Myr it crosses `−500 pc/Myr` and trips the terminal
  `velocity_runaway` event (`phase_events.py:450`), which `apply_event_result` marks
  `is_simulation_ending=True` — the whole simulation stops on segment 0.
- **when it bites** — Every run, on segment 0 of phase 1a, in a **fresh process**. The
  *severity* of the outcome is configuration-dependent (`v0` scales with the SPS file's
  wind `2L/ṗ`), which is what makes this a sharp edge rather than a uniform offset.
- **observable symptom** — a first energy-phase step in which `v2` drops discontinuously
  by exactly `3000 pc/Myr`; for low-`v0` clusters, a run that terminates immediately with
  `SimulationEndCode.VELOCITY_RUNAWAY` and no useful output.
- **confidence** — high on the read/flip ordering (verified by reading and by measuring
  `v0` for the shipped config); medium on whether `vd = -1e8` is intended as a
  physical approximation at all

---

### ST-009 · `mCloud` changes meaning mid-load (pre-SFE → post-SFE); the sweep pre-flight validates the pre-SFE mass while the run validates and simulates the post-SFE mass
- **file:line** — `trinity/_input/read_param.py:386-389`
  ```python
  mCloud_input_value = params['mCloud'].value
  mCluster = mCloud_input_value * params['sfe'].value
  mCloud_after_SF = mCloud_input_value - mCluster
  params['mCloud'].value = mCloud_after_SF
  ```
  vs `trinity/_input/sweep_runner.py:120-121`
  ```python
  kwargs = dict(
      mCloud=float(mCloud),          # raw .param value = PRE-SFE
  ```
  vs `trinity/cloud_properties/validate_gmc.py:369` (`mCloud = params["mCloud"].value` =
  POST-SFE) and `trinity/phase0_init/get_InitCloudProp.py:161` (POST-SFE).
- **class** — unit-mismatch-across-writers (same key, two meanings)
- **severity** — S3 misleading
- **the mechanism** — `_validate_sweep_combination` explicitly re-applies the *unit*
  conversions that `read_param` applies (`convert2au('cm**-3')`, `convert2au('m_H')`) so
  the pre-flight matches the run — but it does **not** apply the SFE subtraction, which is
  the one transformation `read_param` performs on `mCloud` itself. The pre-flight therefore
  computes `rCloud` for a cloud `1/(1−sfe)` times more massive than the one the simulation
  builds, i.e. `rCloud` too large by `(1−sfe)^(−1/3)`.
- **when it bites** — Only the sweep pre-flight / `--dry-run` / `--emit-jobs` reporting
  paths in `run.py:483, 537` and `sweep_jobs.emit_jobs`; never the run itself. Measured on
  `param/simple_cluster.param` (`sfe = 0.3`): `mCloud_input = 1e5`, `mCloud = 7e4` —
  a 12 % `rCloud` discrepancy between the pre-flight and the actual run.
- **observable symptom** — `[INVALID GMC]` flagged in `--dry-run` for combinations the run
  accepts (or, near `rCloud_max`, combinations blessed by the pre-flight that then abort
  with "implausible GMC parameters" inside the worker, counted as a `FAILED` row in
  `sweep_report.txt`).
- **confidence** — high

---

### ST-010 · `get_InitCloudProp` silently rewrites `nCore` / `rCore` *after* the pre-run GMC validation has already blessed the originals
- **file:line** — `trinity/phase0_init/get_InitCloudProp.py:229-230`
  ```python
              params['rCore'].value = rCore
              params['nCore'].value = nCore
  ```
  (also `:206`, `:248`, `:277-279`) versus the earlier consumer
  `run.py:210` (`gmc_check = validate_gmc_from_params(params)`), which runs *before*
  `main.start_expansion(params)` at `run.py:231`.
- **class** — ordering-dependence
- **severity** — S3 misleading
- **the mechanism** — The `nEdge < nISM` auto-correction path mutates the two shared keys
  that define the cloud (`nCore` can be raised by an arbitrary factor, `rCore` can be
  halved iteratively up to 50 times, `:250-260`). Every later reader of `nCore`/`rCore`
  (`density_profile.get_density_profile:110-112`, `mass_profile`,
  `get_InitPhaseParam.get_y0:77`) sees the corrected values, while the validation that
  gated the run, the `.param` file, and the sweep folder name all describe the originals.
  Because `nCore` and `rCore` are `run_const=True`, `metadata.json` records the
  *corrected* values, so the file no longer reproduces its own input.
- **when it bites** — Only when `nEdge < nISM` (needs `densPL_alpha != 0`). **Single run
  in a fresh process.** Warnings are logged, so it is not silent in the log — only in the
  outputs.
- **observable symptom** — a run whose `metadata.json[nCore]` differs from the `nCore` in
  the `.param` sitting next to it; a sweep whose folder name (built from the raw values by
  `sweep_parser.generate_run_name`) mislabels the cloud that was actually simulated.
- **confidence** — high

---

### ST-011 · `params['shell_grav_r']` and `params['shell_r_arr']` are the *same* ndarray object whenever the shell has no neutral region
- **file:line** — `trinity/shell_structure/shell_structure.py:263, 273, 413`
  ```python
      grav_ion_r = rShell_arr_ion          # 263 — no copy
  ...
      grav_r = grav_ion_r                  # 273 — no copy
  ...
          shell_r_arr = rShell_arr_ion     # 413 — same object again
  ```
  Both are returned as separate `ShellProperties` fields (`:459`, `:469`) and written to
  two distinct dict keys by `updateDict` (`run_energy_phase.py:208`,
  `run_energy_implicit_phase.py:976`, `run_transition_phase.py:559`,
  `run_momentum_phase.py:629`).
- **class** — aliasing
- **severity** — S4 hygiene (latent — no mutator exists today)
- **the mechanism** — On the `has_neutral == False` branch (photons depleted with no
  neutral region, or all mass swept) the two dataclass fields hold one buffer. The other
  live references are: `params['shell_grav_r'].value`, `params['shell_r_arr'].value`, the
  `ShellProperties` instance held by the caller as `shell_props`, and the `ODESnapshot`
  chain that keeps `shell_props` alive for the whole segment. Any future in-place edit
  (`arr *= cvt.pc2cm`, a unit fix-up, a clip) applied to one key would silently rewrite the
  other. I verified the current consumers do **not** mutate: `_clean_for_snapshot`
  (`dictionary.py:674-701`) only reads and passes to `simplify`, and
  `_functions/simplify._simplify` returns `x_orig[idx], y_orig[idx]` (fancy indexing =
  copy, `simplify.py:490`) and never writes through its inputs.
- **when it bites** — Not today. It is the trap that fires the first time anyone
  normalises units or clips a shell array in place.
- **observable symptom** — would be: `shell_grav_r` and `shell_r_arr` in the same snapshot
  changing together for no reason.
- **confidence** — high (the aliasing is certain; the "no current mutator" conclusion is
  from reading every consumer of both keys)

---

### ST-012 · `bubble_v_arr`, `bubble_T_arr` and `bubble_dTdr_arr` are three views onto one `psoln` buffer on the no-CIE-switch path
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:658-660`
  ```python
      v_array = psoln[:, 0]
      T_array = psoln[:, 1]
      dTdr_array = psoln[:, 2]
  ```
  returned unchanged at `:903-905` and written to three params keys by
  `updateDict(params, bubble_data)`.
- **class** — aliasing
- **severity** — S4 hygiene (latent)
- **the mechanism** — Basic slicing of the `(N, 3)` `psoln` array yields views, not copies.
  When `index_cooling_switch == index_CIE_switch` the `np.insert` block at `:730-734` is
  skipped, so all three survive as views into one contiguous allocation, which then lives
  in `params` under three keys plus in the returned `BubbleProperties`. Live references:
  `params['bubble_v_arr'/'bubble_T_arr'/'bubble_dTdr_arr'].value`, the `BubbleProperties`
  instance (kept as `bubble_props` for the whole segment in phase 1b), and
  `_inflow_frac_thickness`'s arguments. No current consumer writes into them
  (`_inflow_frac_thickness` does `np.asarray(...)` + boolean masking only;
  `_clean_for_snapshot` only reads). Note the *behaviour differs between segments*: when
  the CIE switch is present the three become independent copies, when it is absent they
  alias — so a future in-place edit would corrupt intermittently.
- **when it bites** — Not today.
- **observable symptom** — would be: an in-place edit of one bubble array showing up in
  the other two, only on segments where the profile never crosses `10^5.5 K`.
- **confidence** — high

---

### ST-013 · `bubble_dMdt` is a hidden warm-start seed carried in the state dict, making the bubble solve history-dependent
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:242-246, 261-267`
  ```python
  bubble_dMdt = params['bubble_dMdt'].value
  if np.isnan(bubble_dMdt):
      bubble_dMdt = _get_init_dMdt(params, Pb)
  ...
  bubble_dMdt = scipy.optimize.fsolve(
          velocity_residuals_wrapper, bubble_dMdt,
          xtol=1e-4, factor=50, epsfcn=1e-4)[0]
  ```
- **class** — ordering-dependence
- **severity** — S3 misleading
- **the mechanism** — `params['bubble_dMdt']` is written by `updateDict(params, bubble_data)`
  at the end of each phase-1a/1b segment and read as the `fsolve` seed at the start of the
  next. With `xtol=1e-4` and a residual that has a large rejection plateau
  (`_SOLVER_FAIL_RESIDUAL = 1e3`, the `min_T` penalty at `:371-374`), the converged root is
  a function of the seed, not only of `(R2, Eb, t)`. So "the bubble structure at time t"
  is not reproducible from the state at time t alone — it depends on the whole preceding
  segment sequence, including `dt_segment`, which the adaptive controller changes based on
  monitored-parameter deltas. In phase 1b `BubbleParamsView` deliberately threads a
  per-trial `dMdt_guess` (`get_betadelta.py:126-134`), which makes the residual surface
  seen by the `(beta, delta)` root-finder itself path-dependent.
  Interaction with ST-007: `bubble_dMdt` *is* in `COOLING_PHASE_KEYS` (`dictionary.py:1223`)
  so it is reset to `np.nan` after phase 1c — but the arrays it produced are not.
- **when it bites** — Every run, **fresh process**. It is deliberate (documented as a warm
  start) but it is the reason a "same physical state" replay does not reproduce a run.
- **observable symptom** — full-run trajectories that diverge from a mid-run restart, and
  results that shift when `DT_SEGMENT_*` or `ADAPTIVE_THRESHOLD_DEX` change even though
  neither appears in the physics.
- **confidence** — high

---

### ST-014 · The non-CIE cooling cube `.npy` cache is written non-atomically into the shipped `lib/` tree and shared by all concurrent sweep workers
- **file:line** — `trinity/cooling/non_CIE/read_cloudy.py:172-176, 265`
  ```python
  cube_filename = path2cooling + _stem + '_cube.npy'
  if os.path.exists(cube_filename):
      log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube = np.load(cube_filename, allow_pickle = True)
      return ...
  ...
  np.save(cube_filename, [log_ndens_arr, log_temp_arr, log_phi_arr, cool_cube, heat_cube])
  ```
  `path2cooling` resolves to `<repo>/lib/default/opiate/` by default
  (`registry.py:_resolve_path_cooling_nonCIE:246`).
- **class** — sweep-unsafe
- **severity** — S2 latent
- **the mechanism** — Check-then-write with no lock and no temp-file rename. On a cold
  cache with `--workers N`, all N worker subprocesses see `os.path.exists() == False`,
  all build the cube, and all `np.save` to the same path concurrently: writes interleave,
  and a worker that reaches the `exists()` check while another is mid-write `np.load`s a
  truncated file (`allow_pickle=True`, so the failure mode ranges from a clean exception
  to a garbage array). The cache is also keyed purely on the source filename — editing an
  `opiate_*.dat` never invalidates its `_cube.npy`. And it writes into the package's own
  data directory, so a read-only install (site-packages, a shared HPC module) fails at
  `np.save` rather than degrading.
- **when it bites** — Only when a `_cube.npy` is missing. The shipped `Z1.00` cubes are
  tracked in git (`git ls-files lib/default/opiate` shows `*_cube.npy` for every `Z1.00`
  age), so the default configuration is warm. Reachable via a user-supplied
  `path_cooling_nonCIE` directory, or the `Z0.15` tables (only `age1.00e+06` has a cube
  committed — currently unreachable because `_validate_ZCloud` (`registry.py:99-105`)
  rejects `ZCloud != 1`). **Sweep-only; a single run cannot race itself.**
- **observable symptom** — a subset of sweep runs failing with a pickle/`np.load` error on
  the first sweep after adding cooling tables, and succeeding on a re-run; or a modified
  cooling table having no effect until the `.npy` is deleted by hand.
- **confidence** — high on the mechanism, medium on reachability under the shipped defaults

---

### ST-015 · Gated bubble-diagnostic counters are module globals that are never reset between runs
- **file:line** — `trinity/bubble_structure/bubble_luminosity.py:970, 1094-1095`
  ```python
  _bubble_diag_count = 0
  ...
  _bubble_state_dump_count = 0
  _bubble_state_last_t = 0.0
  ```
  mutated under `global` at `:987, 1074` and `:1101, 1113, 1140`.
- **class** — module-global
- **severity** — S4 hygiene
- **the mechanism** — Both counters are process-lifetime. In a second in-process run,
  `_bubble_diag_count` may already be at `_BUBBLE_DIAG_MAX = 100`, so the second run saves
  zero `bubble_diag/*.npz` events; `_bubble_state_dump_count` likewise exhausts the
  `TRINITY_BUBBLE_STATE_DUMP` budget, and `_bubble_state_last_t` carries the *previous*
  run's `t_now` into the `t_now < _bubble_state_last_t * dt_factor` spacing gate
  (`:1111`), suppressing the second run's early dumps. Files are named
  `event_{count:04d}_t{t:.6e}.npz` and written under the *current* run's `path2output`,
  so numbering is also non-contiguous per run.
- **when it bites** — Only with `TRINITY_BUBBLE_DIAG` / `TRINITY_BUBBLE_STATE_DUMP` set,
  and only for a **second in-process run**. Physics-neutral: neither counter feeds the
  simulation.
- **observable symptom** — missing or misnumbered diagnostic dumps when the harness runs
  several simulations in one interpreter.
- **confidence** — high

---

### ST-016 · The module-level `DEFAULT_SPS_COLUMN_MAP` dict object is stored directly into `params` — one shared object across every run in a process
- **file:line** — `trinity/_input/registry.py:293` (`column_map = sps_columns.DEFAULT_SPS_COLUMN_MAP`)
  and `:319-324` (`params['sps_column_map'] = DescribedItem(column_map, …)`);
  definition at `trinity/sps/sps_columns.py:166-174`.
- **class** — module-global / aliasing
- **severity** — S4 hygiene
- **the mechanism** — Unlike `materialize_runtime`/`apply_active_when`, which explicitly
  `copy.deepcopy(spec.default)` "so mutable defaults like `[]` aren't shared across runs"
  (`registry.py:614-618, 656-661`), the SPS bundle resolver hands the *live* module dict to
  `params`. Anything that mutated `params['sps_column_map'].value` would permanently alter
  the default preset for every subsequent run in the process. I checked every consumer:
  `read_sps._read_sps_user` (`read_sps.py:168`) iterates it read-only, and
  `sps_columns.build_user_column_map` builds a fresh dict for the user path — so there is
  no mutator today. `ColumnSpec` is `@dataclass(frozen=True)`, so the values are safe; only
  the dict itself is shared.
- **when it bites** — Not today; would bite the second in-process run the moment a
  column-map override or normalisation is added.
- **observable symptom** — would be: run 2 loading the SPS file with run 1's column layout.
- **confidence** — high

---

### ST-017 · `params['bubble_Lloss']` stores the *effective* (boost- and leak-inclusive) loss but is read back as the *raw* cooling integral on the no-physical-root fallback path
- **file:line** — writer `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:929-930`
  ```python
  if betadelta_result.L_loss is not None:
      params['bubble_Lloss'].value = betadelta_result.L_loss
  ```
  where `L_loss = effective_Lloss_from_params(params, Lcool, bubble_Leak, Lmech_total)`
  (`get_betadelta.py:473`, `:577`); reader
  `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1244-1247`
  ```python
  else:
      Lloss_param = params.get('bubble_Lloss', None)
      _Lcool = Lloss_param.value if Lloss_param and hasattr(Lloss_param, 'value') else 0.0
      Lloss = effective_Lloss_from_params(params, _Lcool, 0.0, Lgain)
  ```
- **class** — unit-mismatch-across-writers
- **severity** — S2 latent
- **the mechanism** — The variable name `_Cool` and the parameter name `Lcool` in
  `effective_Lloss(mode, fmix, theta_target, Lcool, Lleak, Lmech)` declare the argument to
  be the *resolved* cooling integral. On the `bubble_props is None` branch the already-boosted,
  already-leak-inclusive `bubble_Lloss` is passed in that slot and pushed through the boost a
  second time. With the default `cooling_boost_mode = 'none'` the wrapper short-circuits to
  `Lcool + Lleak = value + 0.0`, so the default path is a no-op. With
  `cooling_boost_mode = 'multiplier'` and `cooling_boost_fmix > 1`, the segment's
  transition-trigger loss becomes `Lleak_prev·f + f²·Lcool_prev` instead of
  `Lleak + f·Lcool`.
- **when it bites** — Requires **both** the opt-in `cooling_boost_mode = multiplier`
  (`fmix != 1`) **and** a segment where `betadelta_result.no_physical_root` is set (the
  documented-rare degenerate branch, `run_energy_implicit_phase.py:847`). Single run,
  fresh process.
- **observable symptom** — the energy→momentum `cooling_balance` trigger firing early on
  segments that already failed to find a physical structure root, i.e. a transition time
  that depends on how many rejection segments happened to precede it.
- **confidence** — high

---

### ST-018 · The duplicate-snapshot guard's memory lives in a buffer that `flush()` empties — the guard is disabled on the first snapshot after every flush
- **file:line** — `trinity/_input/dictionary.py:721-731`
  ```python
  if self.save_count >= 1 and self.previous_snapshot:
      last = self.previous_snapshot.get(str(self.save_count - 1), {})
      try:
          t_now = self["t_now"].value
          r2 = self["R2"].value
          if ("t_now" in last and t_now == last["t_now"]) and ("R2" in last and r2 == last["R2"]):
              ...
              return
  ```
  cleared at `trinity/_input/dictionary.py:867-868`
  ```python
  self.flush_count += 1
  self.previous_snapshot = {}
  ```
- **class** — stale-key
- **severity** — S2 latent
- **the mechanism** — `flush()` (fired every `snapshot_interval = 10` saves, and at every
  phase boundary via `write_termination_report`/`_safe_flush`) resets
  `previous_snapshot = {}` while `save_count` keeps counting. The very next
  `save_snapshot()` finds `self.previous_snapshot` falsy, skips the guard entirely, and
  writes the row even if `(t_now, R2)` are identical to the row just flushed. The three
  runners depend on the guard being effective:
  ```python
  _save_count_before = params.save_count
  params.save_snapshot()
  if (params['stop_at_rCloud_nSnap'].value is not None
          and params.save_count > _save_count_before
          and R2 > params['rCloud'].value):
      params['_snapshots_after_rCloud'].value += 1
  ```
  (`run_energy_implicit_phase.py:1016-1026`, `run_transition_phase.py:592-600`,
  `run_momentum_phase.py:674-682`) — the comment there explicitly says the guard "can
  silently skip the first segment of a phase when its (t_now, R2) match the previous
  phase's reconciliation".
- **when it bites** — Whenever a phase boundary lands on a multiple of 10 snapshots. Which
  phase boundaries do is a function of how many segments each phase happened to take, i.e.
  effectively arbitrary per configuration. **Single run, fresh process.**
- **observable symptom** — a duplicated row in `dictionary.jsonl` at a phase boundary, and
  a `_snapshots_after_rCloud` count that is one higher than intended — so a run configured
  with `stop_at_rCloud_nSnap = N` terminates after `N` or `N+1` past-edge snapshots
  depending on where the flush boundary fell.
- **confidence** — high

---

### ST-019 · Phase 1b's covering-fraction leak is computed from the *previous* segment's `Pb` and `c_sound`
- **file:line** — `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:806-819`
  ```python
  # Covering-fraction energy leak (Eq. leak), consumed by the
  # energy-balance branch of solve_betadelta_pure below. Pb and c_sound
  # are carried from the previous segment (1-step frozen) ...
  params['bubble_Leak'].value = get_bubbleParams.get_leak_luminosity(
      params['coverFraction'].value,
      params['R2'].value,      # current segment
      params['Pb'].value,      # previous segment (written at :939)
      params['c_sound'].value, # previous segment (written at :944)
      params['gamma_adia'].value,
  )
  ```
- **class** — stale-key
- **severity** — S3 misleading
- **the mechanism** — `params['Pb']` and `params['c_sound']` are written at lines 939 and
  944, i.e. ~120 lines *below* this read, so on segment *n* the leak mixes segment-*n*
  `R2` with segment-*(n−1)* `Pb` and `c_sound`. On the very first segment of phase 1b they
  come from phase 1a's reconciliation block — which, on the event-exit path, is itself
  inconsistent (ST-001). The value feeds `solve_betadelta_pure`'s energy-balance residual
  and hence the accepted `(beta, delta)`.
- **when it bites** — Every run with `coverFraction < 1`. `Cf = 1` (the default,
  `registry.py:341`) makes `get_leak_luminosity` return exactly `0.0`
  (`get_bubbleParams.py:282-283`), so the default configuration is unaffected.
  **Single run, fresh process.**
- **observable symptom** — `bubble_Leak` in a snapshot corresponding to a pressure one
  segment old; the leak lags a fast `Pb` change by one adaptive segment (up to
  `DT_SEGMENT_MAX = 5e-2 Myr`).
- **confidence** — high (the comment acknowledges the 1-step freeze; recorded here because
  it is exactly the "silently one timestep stale" pattern the audit targets)

---

### ST-020 · `bubble_E2P` mutates its arguments with `*=` / `+=` — safe only because every current caller passes scalars
- **file:line** — `trinity/bubble_structure/get_bubbleParams.py:219-224`
  ```python
      # Make sure units are in cgs
      r1 *= cvt.pc2cm
      r2 *= cvt.pc2cm
      Eb *= cvt.E_au2cgs
      # avoid division by zero
      r2 += 1e-10
  ```
- **class** — aliasing
- **severity** — S4 hygiene (latent)
- **the mechanism** — For a Python `float` or `np.float64`, `*=` rebinds a local and the
  caller is untouched. For an `np.ndarray`, `*=` is `np.multiply(..., out=...)` and would
  rewrite the caller's buffer in place — converting `params['Eb'].value` from code units
  to erg, permanently. I traced all six call sites and all pass scalars:
  `run_energy_phase.py:100` and `:395` (`params[...].value` floats),
  `bubble_luminosity.py:228` (same), `get_betadelta.py:329` (`compute_R1_Pb`'s float args),
  and `get_bubbleParams.py:358, 374, 376` inside `get_effective_bubble_pressure`, whose
  `Eb`/`R2` originate from `R2, v2, Eb = y` on a `solve_ivp` state vector (numpy scalars,
  immutable).
- **when it bites** — Not today. It is one vectorisation away — the function would then
  silently rescale the state dict on the first call, and every subsequent call would
  rescale again.
- **observable symptom** — would be: `Eb` growing by `E_au2cgs` per bubble evaluation.
- **confidence** — high

---

### ST-021 · The sweep is safe — by process isolation, twice over — and by nothing else
- **file:line** — `run.py:629-636`
  ```python
  future = executor.submit(
      run_single_simulation, params, name, TRINITY_ROOT, base_output_dir)
  ```
  and `trinity/_input/sweep_runner.py:285-292`
  ```python
  result = subprocess.run(
      [sys.executable, str(trinity_root / 'run.py'), str(param_path)],
      cwd=str(trinity_root), capture_output=True, text=True,
      timeout=timeout_hours * 3600, env=sim_env)
  ```
- **class** — sweep-unsafe (assessment; currently *not* unsafe)
- **severity** — S3 misleading (a structural hazard, not a live defect)
- **the mechanism** — Each combination is materialised as its own `.param` file in its own
  `outputs/<run_name>/` directory (`sweep_runner.py:246-255`), then executed by a **fresh
  interpreter**. The pool worker only marshals a plain `dict` of scalars/strings
  (`sweep_parser.generate_combinations*` yields `base_params.copy()` per combination —
  `sweep_parser.py:516, 525, 533, 576` — so combinations do not share dict objects
  either). Consequences:
  * No `DescribedDict`, no `_CIE_TCUTOFF_CACHE`, no `_bubble_diag_count`, no
    `DEFAULT_SPS_COLUMN_MAP`, no atexit/signal registration is shared between two
    configurations.
  * A worker that runs two configurations sequentially spawns two subprocesses; results
    are identical to running them alone.
  * Results are **not** order-dependent, with one exception: the `_cube.npy` disk cache
    (ST-014), which is genuinely shared filesystem state across workers.
  * `resolve_base_output_dir` (`run.py:137-148`) absolutises the base dir and
    `generate_param_file` (`sweep_runner.py:182-183`) overrides `path2output` per run,
    so no two workers write to the same output directory.
  The hazard is that *every* protection here is incidental to `subprocess.run`. All of
  ST-003, ST-004, ST-015 and ST-016 become live the moment someone replaces the subprocess
  with an in-process `main.start_expansion(read_param(...))` call for speed — a natural
  optimisation given each simulation is CPU-light and the subprocess re-imports
  numpy/scipy/astropy/pandas every time.
- **when it bites** — Not today. One refactor away.
- **observable symptom** — would be: sweep results that depend on which worker picked up a
  configuration and in what order.
- **confidence** — high

---

## Appendix A — module-level mutable state inventory

Every module-level binding in `trinity/**` that is a `list`, `dict`, `set`, `ndarray`,
cache, counter or flag. "2nd run?" = would a second simulation in the same interpreter be
corrupted.

| # | Symbol | File:line | Written by | First populated | Reset? | 2nd run? |
|---|--------|-----------|-----------|-----------------|--------|----------|
| 1 | `_CIE_TCUTOFF_CACHE` | `cooling/net_coolingcurve.py:27` | `_cie_tcutoff` | first `get_dudt` CIE branch | **never** | **Yes** — id-reuse can return the previous run's cutoff; unbounded growth (ST-003) |
| 2 | `_bubble_diag_count` | `bubble_structure/bubble_luminosity.py:970` | `_capture_bubble_integration` (`global`) | first problematic profile, gated by `TRINITY_BUBBLE_DIAG` | never | Yes (diagnostics only) — ST-015 |
| 3 | `_bubble_state_dump_count` | `bubble_luminosity.py:1094` | `_dump_bubble_state` (`global`) | first dump, gated by `TRINITY_BUBBLE_STATE_DUMP` | never | Yes (diagnostics only) — ST-015 |
| 4 | `_bubble_state_last_t` | `bubble_luminosity.py:1095` | `_dump_bubble_state` (`global`) | first dump | never | Yes — carries the previous run's `t_now` into the spacing gate (ST-015) |
| 5 | `DEFAULT_SPS_COLUMN_MAP` | `sps/sps_columns.py:166` | *(nothing — read-only today)* | import | n/a | No, but the object is shared into `params` uncopied (ST-016) |
| 6 | `CANONICALS`, `CANONICAL_NAMES`, `UNIT_CONVERSIONS` | `sps/sps_columns.py:65, 90, 114` | nothing | import | n/a | No — read-only lookup tables of frozen dataclasses/floats |
| 7 | `SPECS`, `REGISTRY` | `_input/registry.py:328, 536` | nothing | import | n/a | No — `ParamSpec` is `frozen=True`; mutable `default`s (`[]`, `np.array([])`) are `copy.deepcopy`d at `registry.py:615, 657` before entering `params` |
| 8 | `_LOG_M/_LOG_SFE/_LOG_N`, `_F_FIRE`, `_INTERP` | `_input/fkappa_auto.py:40-72` | nothing | import | n/a | No — `fkappa_fire` only reads; `np.clip(coords, lo, hi)` allocates |
| 9 | `COOLING_PHASE_KEYS` | `_input/dictionary.py:1180` | nothing | import | n/a | No — `reset_keys` iterates only |
| 10 | `SNAPSHOT_PROFILE_ARRAY_KEYS` | `_input/dictionary.py:67` | nothing | import | n/a | No — `frozenset` |
| 11 | `RUN_CONST_KEYS`, `METADATA_EXCLUDE`, `DROPPED_IN_V2`, `RESERVED_TOP_LEVEL_KEYS`, `FINAL_STATE_EXCLUDE_ARRAYS` | `_output/run_constants.py:77-131` | nothing | import (derived from registry) | n/a | No — tuple/frozenset |
| 12 | `ADAPTIVE_MONITOR_KEYS` (×3) | `run_energy_implicit_phase.py:150`, `run_transition_phase.py:112`, `run_momentum_phase.py:104` | nothing | import | n/a | No |
| 13 | `CRITICAL_PARAMS`, `CHANGE_THRESHOLDS` | `_output/simulation_end.py:409, 437` | nothing | import | n/a | No |
| 14 | `PARAM_DOCS` | `_output/trinity_reader.py:140` | nothing | import | n/a | No |
| 15 | `CONV`, `INV_CONV`, `CGS` | `_functions/unit_conversions.py:148, 183, 229` | nothing | import | n/a | No — `@dataclass(frozen=True)` |
| 16 | `_STATE_FIELDS`, `PHASES`, `VALID_DENS_PROFILES`, `__all__`, compiled regexes | `terminal_prints.py:131`, `extract_example_snapshots.py:39`, `cloudy/run_loader.py:36`, various | nothing | import | n/a | No |
| 17 | `_shutdown_requested` | `run.py:261-262` (`global`, assigned inside `run_sweep`) | `run_sweep` / `signal_handler` | at sweep start | reassigned per `run_sweep` call | No — sweep-parent only, freshly assigned each call |
| 18 | per-instance `atexit` + `signal` registrations | `_input/dictionary.py:281-288` | `DescribedDict.__init__` | every `DescribedDict()` | **never unregistered** | **Yes** — ST-004 |
| 19 | `<table>_cube.npy` on disk | `cooling/non_CIE/read_cloudy.py:265` | `create_cubes` | first cold-cache load | never invalidated | Cross-*process*: ST-014 |
| 20 | `cooling_nonCIE._hotpath_cutoffs` (instance attr) | `cooling/net_coolingcurve.py:40-45` | `_noncie_cutoffs` | first `get_dudt` per cube | implicitly — the `cube` class is redefined per `get_coolingStructure` call, so every rebuilt cube starts clean | No — correct as documented |

**Verdict on (1):** a second in-process run *can* be corrupted, by ST-003 (wrong CIE
cutoff via `id()` reuse), ST-004 (the first run's `metadata.json` termination block and
`metadata_humanreadable.txt` rewritten at exit; `Ctrl+C` flushing only the newest dict),
and ST-015 (silently missing diagnostics). Nothing else in the table carries physics
across runs.

---

## Appendix B — state-dict write/read contract for the physics-carrying keys

Phase order in one run (`main.start_expansion` → `run_expansion`):

```
read_param (Step 4-10)           → all keys materialised
  ↓
get_InitCloudProp                → rCloud, rCore, nCore, nEdge, initial_cloud_*_arr, densBE_*
  ↓
read_sps / CIE load (main.py)    → sps_data, sps_f, cStruc_cooling_CIE_*
  ↓
get_InitPhaseParam.get_y0        → (returns) → main.py:234-238 writes t_now, R2, v2, Eb, T0
  ↓
1a  run_energy          (current_phase='energy')
  ↓
1b  run_phase_energy    (current_phase='implicit')
  ↓
1c  run_phase_transition(current_phase='transition')
  ↓
main.py:317  reset_keys(COOLING_PHASE_KEYS)   ← NaNs the cooling/bubble-loss keys
  ↓
2   run_phase_momentum  (current_phase='momentum')
  ↓
flush → write_simulation_end → write_termination_report → atexit _safe_flush
```

Legend for **Flag**: 🔴 = finding above; ⚠️ = notable but no separate finding.

### B.1 Core state vector

| Key | Unit | Writers (file:line) | Key readers | Phase order | Flag |
|---|---|---|---|---|---|
| `t_now` | Myr | `main.py:234`; `run_energy_phase.py:148, 354`; `run_energy_implicit_phase.py:793, 1134`; `run_transition_phase.py:487, 684`; `run_momentum_phase.py:568, 762`; `phase_events.py:612` | 21 sites incl. `bubble_luminosity._get_bubble_ODE:439-446`, `get_betadelta:446`, `read_cloudy.get_coolingStructure:48` | written top+bottom of every segment loop | 🔴 ST-001 (1a event path writes it without updating the local) |
| `R2` | pc | `main.py:235`; `run_energy_phase.py:149, 355`; `run_energy_implicit_phase.py:794, 1135`; `run_transition_phase.py:488, 685`; `run_momentum_phase.py:569, 763`; `phase_events.py:617` | 26 sites | same | 🔴 ST-001 |
| `v2` | pc/Myr | `main.py:236`; `run_energy_phase.py:150, 356`; `run_energy_implicit_phase.py:795, 1136`; `run_transition_phase.py:489, 686`; `run_momentum_phase.py:570, 764`; `phase_events.py:617` | 16 sites | same | 🔴 ST-001, ST-002 |
| `Eb` | Msun·pc²/Myr² | `main.py:237`; `run_energy_phase.py:151, 357`; `run_energy_implicit_phase.py:796, 1137, 1169`; `run_transition_phase.py:490, 687`; `run_momentum_phase.py:511, 571` (forced `0.0`) | 12 sites | 1169 is the `ENERGY_HANDOFF_FLOOR = 1e3` override on the collapse-to-momentum route | ⚠️ meaning shifts: real bubble energy in 1a/1b/1c, a sentinel floor on handoff, identically `0.0` in phase 2 |
| `T0` | K | `main.py:238`; `run_energy_phase.py:152, 187`; `run_energy_implicit_phase.py:797, 1138`; `run_transition_phase.py:491`; `run_momentum_phase.py:572` | 7 sites (`delta2dTdt_pure`, residuals) | last *real* write is `run_energy_implicit_phase.py:1138` | 🔴 ST-006 — 1c/2 writes echo a frozen local |

### B.2 Radii / pressures

| Key | Unit | Writers | Readers | Flag |
|---|---|---|---|---|
| `R1` | pc | `run_energy_phase.py:108, 191, 396`; `run_energy_implicit_phase.py:938, 1372`; `run_transition_phase.py:508, 839`; `run_momentum_phase.py:588, 891` | 1 direct read + every `bubble_E2P`/`cool_beta_to_Ebdot` call | ⚠️ meaning shifts: wind termination shock in 1a–1c; hard-set to `R2` in phase 2 (`run_momentum_phase.py:588`) |
| `Pb` | Msun/pc/Myr² | `run_energy_phase.py:107, 192, 397`; `run_energy_implicit_phase.py:939, 1373`; `run_transition_phase.py:509, 840`; `run_momentum_phase.py:585, 667, 889` | `shell_structure_pure:104, 125`; `run_energy_implicit_phase.py:816` | ⚠️ **meaning shifts**: thermal `bubble_E2P` in 1a/1b, `max(P_thermal, P_ram)` in 1c (via `get_effective_bubble_pressure`), pure `pRam` in phase 2. Consumers (`shell_structure`) read it as "the pressure confining the shell" throughout. 🔴 ST-019 (read one segment stale in 1b) |
| `rShell` | pc | `updateDict(params, shell_props)` ← `ShellProperties.rShell` (4 phases) | `ODESnapshot.rShell` → `get_ODE_Edot_pure:237-244`; `compute_forces_pure` | ⚠️ frozen at the *previous* value when `isDissolved` (`shell_structure.py:429`) |
| `P_HII`, `P_drive`, `P_ram`, `press_HII_in` | Msun/pc/Myr² | each phase writes twice per segment (Strömgren value, then the `force_props` round-trip) | `create_ODE_snapshot:162`; `compute_forces_pure:531` | ⚠️ the second write re-stores the value `compute_forces_pure` just read back out of `params` — self-consistent, but the double write means a `params['P_HII']` read between the two sites sees a different number |
| `rCloud`, `rCore`, `nCore`, `nEdge` | pc, pc⁻³ | `get_InitCloudProp.py:206, 229-230, 248, 277-279, 339-341`; `bonnorEbertSphere.py:563-575` | `density_profile:110-112`; `mass_profile`; `phase_events.build_*`; 21 `rCloud` reads | 🔴 ST-010 — mutated *after* `run.py:210` validated them |

### B.3 Cooling / bubble structure

| Key | Unit | Writers | Readers | Flag |
|---|---|---|---|---|
| `cool_alpha` | – | `run_energy_implicit_phase.py:662, 798`; `run_transition_phase.py:399` | `bubble_luminosity._get_bubble_ODE_initial_conditions:405`, `_get_bubble_ODE:439` | 🔴/⚠️ ST-017-adjacent: **phase 1a's bubble solve reads it before any writer runs**, i.e. on the schema default `0.6` |
| `cool_beta`, `cool_delta` | – | `run_energy_implicit_phase.py:885-886` **only** | `_get_bubble_ODE:441-446`; `cool_beta_to_Ebdot`; `_capture_bubble_integration:1026-1027` | same: phase 1a runs on the defaults `0.8` / `−6/35`; `reset_keys` NaNs them after 1c |
| `bubble_LTotal` | Msun·pc²/Myr³ | `updateDict(params, bubble_props)` — `run_energy_phase.py:184`, `run_energy_implicit_phase.py:894` | `create_ODE_snapshot:146` → `get_ODE_Edot_pure:273` | 🔴 ST-005 — read as live throughout phase 1c with no writer |
| `bubble_Tavg`, `bubble_mass`, `bubble_r_Tb`, `bubble_T_r_Tb` | K, Msun, pc, K | same two `updateDict` sites | `run_energy_implicit_phase.py:943`; `run_transition_phase.py:512`; `shell_structure.py:103` | 🔴 ST-007 — **not** in `COOLING_PHASE_KEYS`; survive into 1c/2 |
| `bubble_r_arr`, `bubble_T_arr`, `bubble_n_arr`, `bubble_dTdr_arr`, `bubble_v_arr` | pc, K, pc⁻³, K/pc, pc/Myr | same two `updateDict` sites | `dictionary._clean_for_snapshot:639-669`; `_inflow_frac_thickness` | 🔴 ST-007, ST-012 |
| `bubble_dMdt` | Msun/Myr | `updateDict(params, bubble_props)` | `bubble_luminosity.py:242` (fsolve seed), `run_energy_implicit_phase.py:860, 905` | 🔴 ST-013 — a solver seed living in the state dict |
| `bubble_Lgain`, `bubble_Lloss` | Msun·pc²/Myr³ | `run_energy_implicit_phase.py:928, 930` | `run_energy_implicit_phase.py:861-872, 1245` | 🔴 ST-017 — `bubble_Lloss` stores an *effective* loss, is read back as a *raw* `Lcool` |
| `bubble_Leak` | Msun·pc²/Myr³ | `run_energy_phase.py:255`; `run_energy_implicit_phase.py:813` | `run_energy_implicit_phase.py:1241`; `get_betadelta:473, 577` | 🔴 ST-019 — computed from a previous-segment `Pb`/`c_sound` |
| `c_sound` | pc/Myr | `run_energy_phase.py:223`; `run_energy_implicit_phase.py:944`; `run_transition_phase.py:518` | `create_ODE_snapshot:164` → `get_leak_luminosity` | ⚠️ derived from the (possibly frozen) `bubble_Tavg`; hard `1e6 K` fallback in 1b/1c when falsy |
| `cStruc_cooling_nonCIE`, `cStruc_heating_nonCIE`, `cStruc_net_nonCIE_interpolation` | – | `run_energy_phase.py:126-128`; `run_energy_implicit_phase.py:785-787` | `net_coolingcurve.get_dudt:99-104`; `bubble_luminosity.py:781-782, 821-822` | ⚠️ rebuilt on a *time* cadence (`COOLING_UPDATE_INTERVAL`), so `get_dudt` always evaluates a cube up to 5e-3 Myr (1b) / 5e-2 Myr (1a) out of date. NaN'd by `reset_keys` after 1c |
| `cStruc_cooling_CIE_logT`, `_logLambda`, `_interpolation` | – | `main.py:169-171` only | `net_coolingcurve._cie_tcutoff`, `CIE.get_Lambda`; `bubble_luminosity.py:742` | 🔴 ST-003 |
| `t_previousCoolingUpdate` | Myr | `run_energy_phase.py:129`; `run_energy_implicit_phase.py:788` | `run_energy_phase.py:124`; `run_energy_implicit_phase.py:783` | ⚠️ `reset_keys` sets it to `np.nan`; `abs(nan − t) > interval` is `False`, so the update predicate is permanently `False` afterwards. Inert only because phase 2 never calls `get_dudt` |

### B.4 Shell

All shell keys (`shell_n0`, `shell_thickness`, `shell_fAbsorbed*`, `shell_fIonisedDust`,
`shell_nMax`, `shell_tauKappaRatio`, `shell_grav_r`, `shell_grav_phi`,
`shell_grav_force_m`, `shell_r_arr`, `shell_n_arr`, `shell_ion_idx`, `n_IF`, `n_IF_ODE`,
`R_IF`, `n_IF_Str`, `isDissolved`, `is_phiDepleted`) are written **only** by
`updateDict(params, shell_props)` from `shell_structure_pure`, at:
`run_energy_phase.py:208, 401`; `run_energy_implicit_phase.py:976, 1375`;
`run_transition_phase.py:559, 842`; `run_momentum_phase.py:629, 893`.

| Key | Flag |
|---|---|
| `shell_grav_r` / `shell_r_arr` | 🔴 ST-011 — **the same ndarray object** when the shell has no neutral region |
| `shell_mass` | ⚠️ written by 9 sites with a monotone ratchet whose "previous" value is read back out of `params` (`run_energy_implicit_phase.py:953, 1184`; `run_transition_phase.py:536, 695`; `run_momentum_phase.py:597, 772`) — the value is therefore path-dependent, and the post-ODE write at `…:1196 / :707 / :786` exists only to feed the adaptive-stepping comparison but is what the *next* segment reads as `prev_mShell` |
| `shell_nMax` | ⚠️ set to `params['nISM']` when dissolved (`shell_structure.py:423`), which is also the dissolution test's threshold — so once dissolved the condition latches |

### B.5 Control flags

| Key | Writers | Readers | Flag |
|---|---|---|---|
| `current_phase` | `main.py:244, 278, 301, 327` | `create_ODE_snapshot:158` → `get_effective_bubble_pressure` (selects the *pressure law*) | ⚠️ the single switch that changes `Pb`'s meaning (B.2) |
| `EarlyPhaseApproximation` | `run_energy_phase.py:343` | `create_ODE_snapshot:159` → `get_ODE_Edot_pure:269` | 🔴 ST-008 — read on segment 0, flipped after segment 0 |
| `isCollapse` | `run_energy_implicit_phase.py:1303`; `run_transition_phase.py:773`; `run_momentum_phase.py:826`; `phase_events.py:629` | shell-mass freeze in all four phases; `create_ODE_snapshot:142` | ⚠️ never cleared; latches for the rest of the run |
| `isDissolved` | `run_transition_phase.py:814`; `run_momentum_phase.py:867` — **and** read back by `shell_structure_pure:130` | `shell_structure_pure` takes the "dissolved" branch | ⚠️ two-way coupling: the flag phases 1c/2 set changes what `shell_structure_pure` computes on the next call, but `ShellProperties.isDissolved` is echoed straight back through `updateDict` |
| `EndSimulationDirectly` | 25 sites across all phases + `phase_events.py:624` | `main.py:283, 303, 343` | ⚠️ 25 writers, 3 readers — the only cross-phase kill switch |
| `SimulationEndCode` / `SimulationEndReason` | 25 sites each | `main.py:202`; `write_simulation_end:184-196`; `format_end_report` | ⚠️ last-writer-wins across phases; the reconciliation blocks can run *after* the code is set |
| `_snapshots_after_rCloud` | `+=` at `run_energy_implicit_phase.py:1026`, `run_transition_phase.py:600`, `run_momentum_phase.py:682` | the loop-top guard in the same three phases | 🔴 ST-018 — the increment is gated on the duplicate guard being effective |

### B.6 Keys with a writer that only *some* paths run

| Key | Written by | Read unconditionally by | Consequence |
|---|---|---|---|
| `cool_alpha`, `cool_beta`, `cool_delta` | phase 1b only | `bubble_luminosity._get_bubble_ODE`, called from **phase 1a** | phase 1a's bubble structure runs on the schema defaults; a re-entrant phase 1a (the stubbed `main.expansion_next`) would inherit 1b's values |
| `bubble_LTotal` | phases 1a/1b only | `create_ODE_snapshot` in **phase 1c** | ST-005 |
| `bubble_Tavg` | phases 1a/1b only | `run_transition_phase.py:512` | ST-005 |
| `T0` | phases 1a/1b only | written back verbatim by 1c/2 | ST-006 |
| `bubble_mass` | phases 1a/1b only | `shell_structure.py:103` in **phase 2** | ST-007 |
| `bubble_dMdt` | `updateDict(params, bubble_props)`, skipped when `no_physical_root` | `bubble_luminosity.py:242` as the fsolve seed | intentional hold, documented at `run_energy_implicit_phase.py:840-846` |
| `P_HII` | set to `0.0` when `include_PHII` is False or `n_IF_Str <= 0` | `compute_forces_pure:531`, `create_ODE_snapshot:162` | fine — always written before read in each segment |
| `Pb`, `c_sound` | bottom of the 1b segment | top of the *next* 1b segment | ST-019 |
