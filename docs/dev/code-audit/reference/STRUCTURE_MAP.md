# TRINITY structure map (Phase 0c)

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

**Status (2026-07-29):** 📘 reference — descriptive structure map for the bugfix/code-audit review.

---

## Scope and method

Purely descriptive. Sources read: `trinity/**` (72 `.py` files), `run.py`, `test/**`
(52 `test_*.py` files + `test/CLAUDE.md`). Nothing under `docs/dev/` was read except the banner
templates in `docs/dev/CLAUDE.md`. No judgements; anything that looked notable is a neutral entry
in §8.

**How the caller/reference columns were produced.** Definitions come from `ast` parsing of every
source file. Reference sites come from an `ast.walk` over every file in `trinity/`, `run.py` and
`test/`, resolving each `Name`/`Attribute` node against that file's import table
(`import x as y`, `from a.b import c`) and its own module namespace. Consequences to keep in mind
when reading the tables:

* Docstrings and comments are **not** counted (they are not AST expression nodes), so a name that
  only appears in a usage example shows as zero references.
* **Methods are resolved by name only** (there is no type inference), so a method column can
  over-count when two classes share a method name (e.g. `filter`, `format`, `close`).
* Module-level free functions and classes are resolved import-aware and are reliable.
* A reference is any read of the name (call, decorator, argument, re-export), not strictly a call.

Counts: **444 definitions** inventoried (module-level functions, classes, and non-dunder methods);
**34** have zero references anywhere in `trinity/`, `run.py` or `test/`.

---

## 1. Entry points and control flow

### 1.1 Process entry

`run.py` is the only entry point (`run.py:778` `if __name__ == '__main__':`).

| step | site | what happens |
|---|---|---|
| argparse | `run.py:780-831` | positional `path2param`; flags `--workers/-w` (`positive_int`, `run.py:122`), `--dry-run/-n`, `--yes/-y`, `--verbose/-v`, and a mutually-exclusive group `--emit-jobs DIR` / `--collect-report DIR` |
| logging | `run.py:835-841` | `logging.basicConfig(DEBUG if --verbose else WARNING)`; deliberately inside `__main__` so spawn-based pool workers do not reconfigure |
| dep advisory | `run.py:843` → `warn_if_unsupported_deps` (`run.py:54`) | warns if numpy≥2 / scipy≥2 / astropy≥8 / matplotlib≥4 / pandas≥3 |
| banner | `run.py:846-847` | `trinity._output.header.display()` (`header.py:17`) |
| `--collect-report` | `run.py:851-854` | `trinity._input.sweep_jobs.collect_report` (`sweep_jobs.py:282`), then `sys.exit(0)` |
| `--emit-jobs` | `run.py:860-875` | requires sweep syntax; `read_sweep_config` (`sweep_parser.py:354`) → `emit_jobs` (`sweep_jobs.py:98`), then `sys.exit(0)` |
| mode select | `run.py:878` | `is_sweep_param_file` (`run.py:81`) scans the `.param` for `tuple(` lines or multi-element `[a, b]` values |
| sweep | `run.py:879` → `run_sweep` (`run.py:238`) | `ProcessPoolExecutor`, one `run_single_simulation` (`sweep_runner.py:212`) per combination |
| single | `run.py:897` → `run_single` (`run.py:155`) | the path traced below |

### 1.2 Single-simulation call sequence

```
run.py:160   read_param.read_param(args.path2param)              -> DescribedDict
run.py:162   header.show_param(params)
run.py:191   logging_setup.setup_logging(... path2output ...)
run.py:210   validate_gmc.validate_gmc_from_params(params)        -> GMCValidationResult
             (invalid -> params.set_termination_reason(...) + sys.exit, run.py:224-228)
run.py:231   main.start_expansion(params)
```

`read_param.read_param` (`read_param.py:43`) is a fixed 10-step pipeline; every step is
numbered in the source:

| step | line | action |
|---|---|---|
| 1 | `read_param.py:106-173` | parse `trinity/_input/default.param` (`# INFO:` / `# UNIT:` comment metadata) |
| 2 | `read_param.py:175-208` | parse the user `.param` |
| 3 | `read_param.py:210-247` | reject unknown keys; `registry.validate_companions` (`registry.py:715`); merge user over defaults |
| 4 | `read_param.py:249-277` | build the `DescribedDict`, applying `cvt.convert2au(unit)` (`unit_conversions.py:315`) to numeric values |
| 5 | `read_param.py:295` | `registry.validate_all` (`registry.py:546`) — runs every spec `validator` |
| 6 | `read_param.py:298-400` | derive `mu_convert/mu_atom/mu_ion/mu_mol/chi_e/mu_ion_shell/chi_e_shell` from `x_He,Z_He,Z_He_shell`; scale `dust_sigma` by `ZCloud`; default `model_name` to the filename; **rebind `mCloud` to the post-SFE mass** and add `mCloud_input`, `mCluster` |
| 7 | `read_param.py:410` + `:417-429` | `registry.resolve_all` (`registry.py:564`) → `_resolve_path2output`, `_resolve_path_cooling_nonCIE`, `_resolve_sps_bundle`, `resolve_fkappa_auto`; then the inline `path_cooling_CIE` integer-preset resolution |
| 8 | `read_param.py:441` | `registry.apply_active_when` (`registry.py:588`) — adds/pops the `densBE_*` / `densPL_alpha` keys |
| 9 | `read_param.py:449-457` | set `exclude_from_snapshot=True` on everything not in the 10-key `time_varying_keys` list |
| 10 | `read_param.py:472` | `registry.materialize_runtime` (`registry.py:624`) — creates every remaining spec as a fresh `DescribedItem` |
| guard | `read_param.py:482-492` | raises `RuntimeError` if any `default.param` key's `DescribedItem` object was *replaced* (identity check) by steps 6/8/10 |

### 1.3 `main.start_expansion` (`main.py:81`)

| order | line | call |
|---|---|---|
| 0 | `main.py:104-114` | logging fallback; `terminal_prints.phase0(startdatetime)` (`terminal_prints.py:35`) |
| A1 | `main.py:122` | `get_InitCloudProp.get_InitCloudProp(params)` (`get_InitCloudProp.py:89`) — sets `rCloud`, `rCore`, `nEdge`, `initial_cloud_{r,n,m}_arr`, and for BE also `densBE_Teff/_sigma/_xi_out/_f_rho_rhoc/_f_m/_xi_arr/_u_arr/_dudxi_arr/_rho_rhoc_arr` |
| A1b | `main.py:128-136` | `_check_stop_r_rCloud_interaction` (`main.py:41`) — advisory only |
| A2 | `main.py:144-153` | `f_mass = mCluster / sps_refmass`; `read_sps.read_sps` (`read_sps.py:38`) → `read_sps.get_interpolation` (`read_sps.py:285`); stores `sps_data`, `sps_f` |
| A3 | `main.py:162-171` | `np.loadtxt(params['path_cooling_CIE'])` → `scipy.interpolate.interp1d(logT, logLambda, kind='linear')`; stores `cStruc_cooling_CIE_logT`, `..._logLambda`, `..._interpolation` |
| B | `main.py:180` | `run_expansion(params)` |
| C | `main.py:191` | `simulation_end.write_simulation_end(params)` (`simulation_end.py:130`) |
| C2 | `main.py:194` | `terminal_prints.format_end_report(params)` (`terminal_prints.py:205`) |
| C3 | `main.py:203` | `params.write_termination_report(reason=...)` (`dictionary.py:355`) |

### 1.4 `main.run_expansion` (`main.py:216`) — the phase ladder

```
main.py:232   get_InitPhaseParam.get_y0(params) -> (t0, r0, v0, E0, T0)
main.py:234-238  params['t_now'|'R2'|'v2'|'Eb'|'T0'] = those
main.py:244   params['current_phase'] = 'energy'
main.py:251   run_energy_phase.run_energy(params)                  # PHASE 1a
main.py:263-272  if stop_at_rCloud_nSnap == 0 and R2 >= rCloud:
                    EndSimulationDirectly = True, code = RCLOUD_BOUNDARY
main.py:278   params['current_phase'] = 'implicit'
main.py:283   if not EndSimulationDirectly:
main.py:286       run_energy_implicit_phase.run_phase_energy(params)   # PHASE 1b
main.py:301   params['current_phase'] = 'transition'
main.py:303   if not EndSimulationDirectly:
main.py:306       run_transition_phase.run_phase_transition(params)    # PHASE 1c
main.py:317   params.reset_keys(COOLING_PHASE_KEYS)                    # -> np.nan
main.py:327   params['current_phase'] = 'momentum'
main.py:339   get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)     # logged diagnostic
main.py:343   if not EndSimulationDirectly:
main.py:346       run_momentum_phase.run_phase_momentum(params)        # PHASE 2
main.py:356   params.flush()
```

**The only gate between phases is the boolean `params['EndSimulationDirectly']`.** There is no
return-value inspection: `run_phase_energy`, `run_phase_transition` and `run_phase_momentum` each
return a results dataclass, and `main.run_expansion` discards all three. A phase that ends for a
non-simulation-ending reason simply falls through to the next `if ... == False` block.
`main.py:366` `expansion_next(...)` is a stub whose body is `return`.

### 1.5 Per-phase loop shape

All four runners share the same segment loop skeleton: *update `params` from local state → get SPS
feedback → compute derived structure → `params.save_snapshot()` → `solve_ivp` over one segment →
`check_event_termination` → extract new state → adaptive `dt_segment` → termination checks*, then a
final "phase-boundary reconciliation snapshot" outside the loop.

| phase | runner | state vector | segment loop | solver |
|---|---|---|---|---|
| 1a energy | `run_energy_phase.run_energy` (`run_energy_phase.py:62`) | `[R2, v2, Eb]` | `while R2 < rCloud and (TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD and continueWeaver` (`run_energy_phase.py:138`); `SEGMENT_DURATION = 3e-5`, `TFINAL_ENERGY_PHASE = 3e-3` Myr | `solve_ivp` RK45 (`run_energy_phase.py:299`), RK23 retry (`:313`) |
| 1b implicit | `run_energy_implicit_phase.run_phase_energy` (`run_energy_implicit_phase.py:631`) | `[R2, v2, Eb, T0]` | `while t_now <= tmax and segment_count < MAX_SEGMENTS (5000)` (`:758`) | `solve_ivp` LSODA (`:1079`) |
| 1c transition | `run_transition_phase.run_phase_transition` (`run_transition_phase.py:367`) | `[R2, v2, Eb]` | `while t_now <= tmax and segment_count < MAX_SEGMENTS (5000)` (`:463`) | `solve_ivp` LSODA (`:640`) |
| 2 momentum | `run_momentum_phase.run_phase_momentum` (`run_momentum_phase.py:461`) | `[R2, v2]` | `while t_now <= tmax and segment_count < MAX_SEGMENTS (10000)` (`:544`) | `solve_ivp` LSODA (`:722`) |

Per-segment physics calls, in order:

* **1a** (`run_energy_phase.py:157-308`): `get_current_sps_feedback` → `bubble_luminosity.get_bubbleproperties_pure` → `mass_profile.get_mass_profile` → `shell_structure_pure` → P_HII from `n_IF_Str` → `operations.get_soundspeed` → `energy_phase_ODEs.create_ODE_snapshot` + `compute_derived_quantities` → `save_snapshot` → inline `cooling_balance` parity check (`:273-287`) → `create_ODE_snapshot` → `solve_ivp`.
* **1b** (`run_energy_implicit_phase.py:783-1079`): periodic `non_CIE.get_coolingStructure` → `get_current_sps_feedback` → `get_bubbleParams.get_leak_luminosity` → **`solve_betadelta_pure`** → `compute_R1_Pb` → `get_soundspeed` → `get_mass_profile` → `shell_structure_pure` → `cool_beta_to_Ebdot_pure` / `delta2dTdt_pure` → `compute_forces_pure` → `save_snapshot` → `create_ODE_snapshot` → `solve_ivp` with `Ed`, `Td` held constant over the segment.
* **1c** (`run_transition_phase.py:496-640`): `get_current_sps_feedback` → `compute_R1_Pb` → `get_soundspeed` → `get_mass_profile` → `shell_structure_pure` → `compute_forces_pure` → `save_snapshot` → `create_ODE_snapshot` → `solve_ivp`. No bubble-structure solve, no beta/delta solve.
* **2** (`run_momentum_phase.py:577-722`): `get_current_sps_feedback` → `Pb = pRam(...)`, `R1 = R2` → `get_mass_profile` → `shell_structure_pure` → `compute_forces_momentum_pure` → `save_snapshot` → `create_momentum_snapshot` → `solve_ivp`.

### 1.6 Termination points, per phase

`SimulationEndCode` (`simulation_end.py:55`) values: `SHELL_DISSOLVED=0`, `STOPPING_TIME=1`,
`LARGE_RADIUS=2`, `RCLOUD_BOUNDARY=3`, `SHELL_COLLAPSED=4`, `ERROR_*=10..23`,
`VELOCITY_RUNAWAY=50`, `ENERGY_COLLAPSED=51`, `UNKNOWN=99`.

**Phase 1a — `run_energy` exit points**

| line | condition | effect |
|---|---|---|
| `run_energy_phase.py:138` | loop guard `R2 >= rCloud` or `t` within `DT_EXIT_THRESHOLD` of `TFINAL_ENERGY_PHASE` | normal fall-through to 1b |
| `:169-183` | `bubble_luminosity.get_bubbleproperties_pure` raises `ValueError/RuntimeError/BubbleSolverError` | `EndSimulationDirectly=True`, `ENERGY_COLLAPSED`, `break` |
| `:276-287` | `'cooling_balance' in transition_trigger` and `(Lgain-Lloss)/Lgain < threshold` | `break` (no end code; hands to 1b) |
| `:324-331` | event fired (`cloud_boundary`, `min_radius`, `velocity_runaway`) | `apply_event_result`; `return` if simulation-ending, else `break` |
| `:368-379` | `not np.isfinite(Eb) or Eb <= 0` | `EndSimulationDirectly=True`, `ENERGY_COLLAPSED`, `break` |
| `:390-404` | — | reconciliation snapshot (recompute `R1`, `Pb`, `shell_mass`, shell structure, then `save_snapshot`) |

**Phase 1b — `run_phase_energy` exit points** (`termination_reason` string in parentheses)

| line | condition | end code |
|---|---|---|
| `run_energy_implicit_phase.py:670-690` | `t_now >= stop_t` on entry (`skipped_past_stop_t`) | `STOPPING_TIME`, returns early |
| `:765-775` | `_snapshots_after_rCloud >= stop_at_rCloud_nSnap` (`stop_at_rCloud`) | `RCLOUD_BOUNDARY` |
| `:865-879` | `no_root_streak >= NO_ROOT_HANDOFF_STREAK (50)` (`no_physical_root_handoff`) | none — hands off to 1c |
| `:1040-1046` | `t_now >= tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:1080-1083` | `solve_ivp` raised (`solver_error: …`) | none |
| `:1085-1090` | `not sol.success` (`solver_failed: …`) | none |
| `:1095-1119` | event fired (`velocity_sign`, `min_radius`, `velocity_runaway`, `max_radius`) | via `apply_event_result` |
| `:1148-1163` | `classify_energy_collapse(Eb) == 'stop'` i.e. non-finite `Eb` (`energy_collapsed`) | `ENERGY_COLLAPSED` |
| `:1164-1175` | `classify_energy_collapse(Eb) == 'momentum'` i.e. finite `Eb<=0` (`energy_to_momentum`) | none — sets `Eb = ENERGY_HANDOFF_FLOOR (1e3)` and routes to 1c |
| `:1288-1294` | `r1_transition_decision` returns `'blowout'` / `'ebpeak'` | none |
| `:1296-1299` | `'cooling_balance' in active_triggers and (Lgain-Lloss)/Lgain < threshold` (`cooling_balance`) | none |
| `:1309-1314` | `t_now > tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:1316-1324` | `isCollapse and R2 < coll_r` (`small_radius`) | `SHELL_COLLAPSED` |
| `:1327-1333` | `stop_r is not None and R2 > stop_r` (`large_radius`) | `LARGE_RADIUS` |
| `:1408-1410` | fell out of the loop | `max_segments` if `segment_count >= 5000`, else `unknown` |
| `:1365-1402` | — | reconciliation snapshot; **skipped** (bare `save_snapshot`) when `termination_reason == "energy_collapsed"` |

**Phase 1c — `run_phase_transition` exit points**

| line | condition | end code |
|---|---|---|
| `run_transition_phase.py:407-424` | `t_now >= stop_t` on entry (`skipped_past_stop_t`) | `STOPPING_TIME` |
| `:468-479` | `stop_at_rCloud_nSnap` reached (`stop_at_rCloud`) | `RCLOUD_BOUNDARY` |
| `:605-611` | `t_now >= tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:641-648` | solver raised / `not sol.success` | none |
| `:653-671` | event fired (`energy_floor`, `min_radius`, `velocity_runaway`, `max_radius`) | via `apply_event_result` |
| `:749-763` | `P_ram/(Pb+P_ram) > 0.9` (`ram_dominated`) | none — to phase 2 |
| `:766-769` | `Eb < ENERGY_FLOOR (1e3)` (`energy_floor`) | none — to phase 2 |
| `:779-784` | `t_now > tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:786-794` | `isCollapse and R2 < coll_r` (`small_radius`) | `SHELL_COLLAPSED` |
| `:797-803` | `stop_r is not None and R2 > stop_r` (`large_radius`) | `LARGE_RADIUS` |
| `:805-826` | `shell_nMax < nISM` sustained for `stop_t_diss` (`dissolved`) | `SHELL_DISSOLVED` |
| `:868-870` | fell out of the loop | `max_segments` / `unknown` |
| `:833-862` | — | reconciliation snapshot (always attempted) |

**Phase 2 — `run_phase_momentum` exit points**

| line | condition | end code |
|---|---|---|
| `run_momentum_phase.py:488-504` | `t_now >= stop_t` on entry (`skipped_past_stop_t`) | `STOPPING_TIME` |
| `:549-560` | `stop_at_rCloud_nSnap` reached (`stop_at_rCloud`) | `RCLOUD_BOUNDARY` |
| `:687-693` | `t_now >= tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:723-730` | solver raised / `not sol.success` | none |
| `:735-750` | event fired (`min_radius`, `velocity_runaway`, `max_radius`) | via `apply_event_result` |
| `:832-837` | `t_now > tmax` (`reached_tmax`) | `STOPPING_TIME` |
| `:839-847` | `isCollapse and R2 < coll_r` (`small_radius`) | `SHELL_COLLAPSED` |
| `:850-856` | `stop_r is not None and R2 > stop_r` (`large_radius`) | `LARGE_RADIUS` |
| `:858-879` | dissolution timer (`dissolved`) | `SHELL_DISSOLVED` |
| `:914-916` | fell out of the loop | `max_segments` / `unknown` |
| `:886-908` | — | reconciliation snapshot |

**Events.** `phase_events.py` builds the terminal-event lists handed to `solve_ivp`:
`build_energy_phase_events` (`:423`) = `[cloud_boundary(rCloud), min_radius(max(1.5*coll_r, 0.01)),
velocity_runaway(-500 pc/Myr)]`; `build_implicit_phase_events` (`:458`) = `[velocity_sign
(non-terminal), min_radius, velocity_runaway]` + `max_radius(stop_r)` if set, plus a
`cooling_balance` factory that is returned but never used by the caller;
`build_transition_phase_events` (`:504`) = `[energy_floor(1e3, y[2]), min_radius,
velocity_runaway]` + `max_radius`; `build_momentum_phase_events` (`:546`) = `[min_radius,
velocity_runaway]` + `max_radius`. `check_event_termination` (`:363`) returns the **first**
event index with a non-empty `t_events` entry. `apply_event_result` (`:588`) writes `t_now`, the
`state_keys`, and — only when `is_simulation_ending` — `SimulationEndReason`, `SimulationEndCode`
and `EndSimulationDirectly`, plus `isCollapse=True` when the reason code contains `radius` or
`collapse`.

### 1.7 Exit-path writers

* `main.py:191` → `write_simulation_end` (`simulation_end.py:130`) merges `termination` +
  `final_state` blocks into `metadata.json`.
* `main.py:203` → `DescribedDict.write_termination_report` (`dictionary.py:355`) →
  `write_termination_debug_report` (`simulation_end.py:558`).
* `DescribedDict.__init__` registers process-wide handlers at `dictionary.py:262-290`: an
  `atexit` handler and `signal.signal(SIGINT/SIGTERM, self._signal_handler)`. Both funnel into
  `_safe_flush` (`dictionary.py:302`), which flushes pending snapshots, writes the termination
  debug report, and writes `metadata_humanreadable.txt` via
  `show_run.format_run_summary` (`show_run.py:371`).

---

## 2. Per-module function inventory

Legend: ⛔ marks a definition with **zero references** anywhere in `trinity/`, `run.py` or `test/`.
See the method-resolution caveat in *Scope and method* above.

#### `run.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 54 | func | `warn_if_unsupported_deps()` | Warn (without failing) when an installed core dependency is newer than | run.py:843 | **none** |
| 81 | func | `is_sweep_param_file(path2file)` | Quick scan to detect if a parameter file contains sweep/tuple syntax. | run.py:878, run.py:861 | **none** |
| 122 | func | `positive_int(value)` | argparse type: a strictly positive integer (for --workers). | run.py:795 | test/test_sweep_workers.py:98, test/test_sweep_workers.py:104 |
| 137 | func | `resolve_base_output_dir(config)` | Absolute base output directory for a sweep. | run.py:458, run.py:868 | **none** |
| 155 | func | `run_single(args)` | Run a single TRINITY simulation. | run.py:897 | **none** |
| 238 | func | `run_sweep(args)` | Run a TRINITY parameter sweep. | run.py:879 | **none** |

#### `trinity/_analysis/check_yesno.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 61 | func | `pair_yes_no(folder)` | Return list of (base_name, yes_path, no_path); missing partner → None. | trinity/_analysis/check_yesno.py:269 | **none** |
| 79 | func | `_get_field(output, name, default=np.nan)` | Load a field as float array, replacing None with default. | trinity/_analysis/check_yesno.py:92, trinity/_analysis/check_yesno.py:93, trinity/_analysis/check_yesno.py:94, trinity/_analysis/check_yesno.py:95 (+2) | **none** |
| 90 | func | `load_run(path)` | — | trinity/_analysis/check_yesno.py:169, trinity/_analysis/check_yesno.py:170 | **none** |
| 111 | func | `compare_trajectories(yes, no)` | Interpolate R2 onto the overlapping time window, return max rel diff. | trinity/_analysis/check_yesno.py:181 | **none** |
| 127 | func | `pressure_dominance(yes)` | In the yesPHII run, when does P_HII actually matter? | trinity/_analysis/check_yesno.py:185 | **none** |
| 159 | func | `diagnose_pair(base, yes_path, no_path, r2_tol, phii_tol)` | — | trinity/_analysis/check_yesno.py:280 | **none** |
| 241 | func | `main()` | — | trinity/_analysis/check_yesno.py:297 | **none** |

#### `trinity/_functions/cluster.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 28 | func | `detect_allocated_cpus()` | Return ``(n_cpus, source)`` for the cores actually available here. | run.py:512, trinity/_functions/cluster.py:60 | test/test_sweep_workers.py:41, test/test_sweep_workers.py:47, test/test_sweep_workers.py:54, test/test_sweep_workers.py:61 (+1) |
| 48 | func | `get_optimal_workers()` | Default worker count when ``--workers`` is not given. | run.py:453 | test/test_sweep_workers.py:79, test/test_sweep_workers.py:85, test/test_sweep_workers.py:91 |

#### `trinity/_functions/extract_example_snapshots.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 47 | func | `_resolve_dict_path(folder) -> Path` | — | trinity/_functions/extract_example_snapshots.py:82 | **none** |
| 56 | func | `_is_terminated(snap_data) -> bool` | — | trinity/_functions/extract_example_snapshots.py:69 | **none** |
| 60 | func | `_pick_phase_index(output, phase) -> Optional[int]` | — | trinity/_functions/extract_example_snapshots.py:97 | **none** |
| 74 | func | `_write_snapshot(out_dir, label, snap_data) -> None` | — | trinity/_functions/extract_example_snapshots.py:94, trinity/_functions/extract_example_snapshots.py:103, trinity/_functions/extract_example_snapshots.py:101 | **none** |
| 81 | func | `extract(folder) -> None` | — | trinity/_functions/extract_example_snapshots.py:111 | **none** |
| 106 | func | `main(argv=None) -> int` | — | trinity/_functions/extract_example_snapshots.py:116 | **none** |

#### `trinity/_functions/logging_setup.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 25 | class | `class LogColors()` | ANSI color codes for colored terminal output. | trinity/_functions/logging_setup.py:55, trinity/_functions/logging_setup.py:56, trinity/_functions/logging_setup.py:57, trinity/_functions/logging_setup.py:58 (+4) | **none** |
| 42 | class | `class ColoredFormatter(logging.Formatter)` | Custom formatter that adds colors to terminal output. | trinity/_functions/logging_setup.py:261 | **none** |
| 62 | method | `ColoredFormatter.format(self, record)` | Format log record with colors. | trinity/_functions/logging_setup.py:74, trinity/_input/sweep_jobs.py:212, trinity/_output/header.py:77 | **none** |
| 79 | class | `class DedupWarningFilter(logging.Filter)` | Collapse identical repeated log messages to a single line. | trinity/_functions/logging_setup.py:269, trinity/_functions/logging_setup.py:300 | test/test_logging_dedup.py:14, test/test_logging_dedup.py:22, test/test_logging_dedup.py:28, test/test_logging_dedup.py:33 |
| 99 | method | `DedupWarningFilter.filter(self, record) -> bool` | — | trinity/_output/cloudy/trinity_to_cloudy.py:214 | test/test_logging_dedup.py:16, test/test_logging_dedup.py:35, test/test_logging_dedup.py:36, test/test_logging_dedup.py:37 (+2) |
| 112 | func | `setup_logging(log_level='INFO', console_output=True, file_output=True, log_file_path=None, log_file_name=None, use_colors=True, format_string=None, suppress_library_debug=True) -> logging.Logger` | Set up TRINITY logging system. | run.py:191, trinity/_functions/logging_setup.py:488, trinity/_functions/logging_setup.py:517, trinity/_functions/logging_setup.py:532 (+1) | test/test_logging_dedup.py:41 |
| 330 | func | `get_module_logger(name) -> logging.Logger` ⛔ | Get a logger for a specific module. | **none** | **none** |
| 367 | func | `set_log_level(level, logger_name=None)` | Change log level after initialization. | trinity/_functions/logging_setup.py:569 | **none** |
| 402 | func | `setup_logging_from_params(params)` ⛔ | Convenience function to set up logging from TRINITY params dictionary. | **none** | **none** |

#### `trinity/_functions/operations.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 19 | func | `find_nearest(array, value)` | finds index idx in array for which array[idx] is closest to value | trinity/_functions/operations.py:48, trinity/_functions/operations.py:167, trinity/bubble_structure/bubble_luminosity.py:884, trinity/bubble_structure/bubble_luminosity.py:878 | **none** |
| 30 | func | `find_nearest_lower(array, value)` ⛔ | This function finds idx in array for which array[idx] satisfies: | **none** | **none** |
| 68 | func | `kindof_increasing(L)` | — | trinity/_functions/operations.py:45, trinity/_functions/operations.py:75 | **none** |
| 71 | func | `kindof_decreasing(L)` | — | trinity/_functions/operations.py:75 | **none** |
| 74 | func | `monotonic(L)` | — | trinity/_functions/operations.py:40, trinity/_functions/operations.py:113, trinity/bubble_structure/bubble_luminosity.py:380 | test/test_operations_monotonic.py:21, test/test_operations_monotonic.py:22, test/test_operations_monotonic.py:27, test/test_operations_monotonic.py:41 (+2) |
| 99 | func | `_is_monotonic_or_tolerable(L, rtol=MONOTONIC_RTOL, boundary_frac=BOUNDARY_FRAC, max_spike_len=MAX_SPIKE_LEN)` | True if L is monotonic, or non-monotonic only as numerical noise: an | trinity/_functions/operations.py:157 | test/test_operations_monotonic.py:33, test/test_operations_monotonic.py:34, test/test_operations_monotonic.py:42, test/test_operations_monotonic.py:54 (+5) |
| 146 | func | `find_nearest_higher(array, value)` | This function finds idx in array for which array[idx] satisfies: | trinity/bubble_structure/bubble_luminosity.py:708, trinity/bubble_structure/bubble_luminosity.py:709 | test/test_operations_monotonic.py:105, test/test_operations_monotonic.py:121, test/test_operations_monotonic.py:127, test/test_operations_monotonic.py:114 |
| 186 | class | `class MonotonicError(Exception)` | — | trinity/_functions/operations.py:42, trinity/_functions/operations.py:159 | test/test_operations_monotonic.py:113 |
| 189 | func | `get_soundspeed(T, params)` | Compute the adiabatic soundspeed | trinity/phase1_energy/run_energy_phase.py:222, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:944, trinity/phase1c_transition/run_transition_phase.py:517 | test/test_mu_audit_drift.py:287, test/test_mu_audit_drift.py:279 |

#### `trinity/_functions/simplify.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 24 | func | `_prev_next_strict(y, greater) -> Tuple[np.ndarray, np.ndarray]` | One-pass monotonic-stack computation of previous/next strictly-greater | trinity/_functions/simplify.py:173, trinity/_functions/simplify.py:205 | **none** |
| 86 | func | `_sparse_table(y, reducer) -> np.ndarray` | Build a sparse table for O(1) range-min or range-max queries on ``y``. | trinity/_functions/simplify.py:174, trinity/_functions/simplify.py:206 | **none** |
| 113 | func | `_rmq(st, lo, hi, reducer) -> np.ndarray` | Vectorised range-min/range-max query over inclusive intervals | trinity/_functions/simplify.py:189, trinity/_functions/simplify.py:193, trinity/_functions/simplify.py:219, trinity/_functions/simplify.py:223 | **none** |
| 123 | func | `_peak_prominences(y, idx) -> np.ndarray` | Compute topological persistence (peak prominence) for local extrema. | trinity/_functions/simplify.py:586 | test/test_simplify.py:475 |
| 246 | func | `_x_uniform_coverage_idx(x, pool_idx, n_chunks=_COVERAGE_CHUNKS) -> np.ndarray` | Pool indices nearest the centres of ``n_chunks`` equal-x-width chunks. | trinity/_functions/simplify.py:698 | test/test_simplify.py:598, test/test_simplify.py:605, test/test_simplify.py:608 |
| 290 | func | `_simplify(x_arr, y_arr, nmin=100, grad_inc=1.0, warn_below_r2=0.9, dedup_tol=_DEDUP_TOL_DEFAULT) -> Tuple[np.ndarray, np.ndarray]` | Heuristic downsampling of a curve y(x) to ``nmin`` points, | trinity/_input/dictionary.py:509 | test/test_simplify.py:80, test/test_simplify.py:88, test/test_simplify.py:94, test/test_simplify.py:101 (+51) |
| 754 | func | `_simplify_error(x_orig, y_orig, x_simp, y_simp) -> dict` | Compute error metrics comparing a simplified curve to the original. | trinity/_input/dictionary.py:524 | test/test_simplify.py:626, test/test_simplify.py:639, test/test_simplify.py:655 |

#### `trinity/_functions/unit_conversions.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 58 | class | `class ConversionConstants()` | Immutable container for unit conversion factors. | trinity/_functions/unit_conversions.py:148 | **none** |
| 156 | class | `class InverseConversionConstants()` | Inverse conversions: Astronomy Units → CGS | trinity/_functions/unit_conversions.py:183 | **none** |
| 193 | class | `class PhysicalConstantsCGS()` | Fundamental physical constants in CGS units. | trinity/_functions/unit_conversions.py:229 | **none** |
| 310 | class | `class UnitConversionError(Exception)` | Raised when unit conversion fails. | trinity/_functions/unit_conversions.py:557, trinity/_functions/unit_conversions.py:441, trinity/_functions/unit_conversions.py:448, trinity/_functions/unit_conversions.py:463 | test/test_unit_conversions.py:77 |
| 315 | func | `convert2au(unit_string) -> float` | Convert a unit string to astronomy units [Msun, pc, Myr]. | trinity/_functions/unit_conversions.py:507, trinity/_functions/unit_conversions.py:523, trinity/_functions/unit_conversions.py:540, trinity/_functions/unit_conversions.py:596 (+5) | test/test_mu_audit_drift.py:45, test/test_mu_audit_drift.py:317, test/test_unit_conversions.py:72, test/test_unit_conversions.py:78 (+1) |

#### `trinity/_input/dictionary.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 78 | class | `class NpEncoder(json.JSONEncoder)` | JSON encoder that converts numpy types to plain Python types. | trinity/_input/dictionary.py:1121, trinity/_input/dictionary.py:564, trinity/_input/dictionary.py:861, trinity/_input/dictionary.py:842 (+3) | **none** |
| 83 | method | `NpEncoder.default(self, obj) -> Any` | — | trinity/_input/dictionary.py:92, trinity/_input/dictionary.py:564, trinity/_input/dictionary.py:1093, trinity/_input/registry.py:657 (+2) | test/test_active_when.py:125, test/test_active_when.py:82, test/test_active_when.py:84, test/test_betadelta_solver_switch.py:45 (+10) |
| 98 | class | `class DescribedItem()` | Container for a value (scalar/array) + light metadata. | trinity/_input/dictionary.py:1293, trinity/_input/dictionary.py:1296, trinity/_input/dictionary.py:1297, trinity/_input/dictionary.py:1300 (+23) | test/test_active_when.py:25, test/test_materialize_runtime.py:28, test/test_materialize_runtime.py:36, test/test_materialize_runtime.py:200 (+56) |
| 133 | method | `DescribedItem.value(self) -> Any` | Return the stored value. | run.py:179, run.py:184, run.py:189, run.py:195 (+889) | test/test_active_when.py:124, test/test_active_when.py:123, test/test_active_when.py:82, test/test_active_when.py:84 (+148) |
| 138 | method | `DescribedItem.value(self, v) -> None` | Set the underlying value (scalar or array). | run.py:179, run.py:184, run.py:189, run.py:195 (+889) | test/test_active_when.py:124, test/test_active_when.py:123, test/test_active_when.py:82, test/test_active_when.py:84 (+148) |
| 169 | method | `DescribedItem._unwrap(x) -> Any` | Extract numeric value if x is a DescribedItem, else return x. | trinity/_input/dictionary.py:174, trinity/_input/dictionary.py:175, trinity/_input/dictionary.py:176, trinity/_input/dictionary.py:177 (+11) | **none** |
| 200 | class | `class DescribedDict(dict)` | A dictionary mapping string keys -> DescribedItem. | trinity/_input/dictionary.py:1022, trinity/_input/dictionary.py:1232, trinity/_input/dictionary.py:1290, trinity/_input/dictionary.py:1332 (+1) | test/test_metadata.py:48, test/test_metadata.py:150, test/test_metadata.py:43, test/test_metadata.py:62 (+5) |
| 262 | method | `DescribedDict._register_crash_handlers(self) -> None` | Register handlers to flush pending snapshots on exit/crash. | trinity/_input/dictionary.py:240 | **none** |
| 292 | method | `DescribedDict._signal_handler(self, signum, frame) -> None` | Handle termination signals by flushing pending snapshots before exit. | trinity/_input/dictionary.py:287, trinity/_input/dictionary.py:288 | **none** |
| 302 | method | `DescribedDict._safe_flush(self, termination_reason='Unknown') -> None` | Flush pending snapshots and write debug report, catching exceptions. | trinity/_input/dictionary.py:299, trinity/_input/dictionary.py:283 | **none** |
| 344 | method | `DescribedDict.set_termination_reason(self, reason) -> None` | Record the reason the process is about to exit. | run.py:224 | **none** |
| 355 | method | `DescribedDict.write_termination_report(self, reason='Unknown') -> None` | Mirror the last-2-snapshot debug block into | trinity/main.py:203 | **none** |
| 387 | method | `DescribedDict.shorten_display(arr, nshow=3)` | Shorten an array for display purposes to avoid clogging output. | trinity/_input/dictionary.py:433 | **none** |
| 449 | method | `DescribedDict.simplify(self, x_arr, y_arr, nmin=None, grad_inc=1.0, keyname='') -> Tuple[np.ndarray, np.ndarray]` | Heuristic downsampling of a curve y(x) to ``nmin`` points, | trinity/_input/dictionary.py:646, trinity/_input/dictionary.py:656, trinity/_input/dictionary.py:665, trinity/_input/dictionary.py:681 (+1) | **none** |
| 543 | method | `DescribedDict._get_output_dir(self) -> Path` | Return output directory from params["path2output"].value. | trinity/_input/dictionary.py:802, trinity/_input/dictionary.py:326, trinity/_input/dictionary.py:336, trinity/_input/dictionary.py:377 | **none** |
| 553 | method | `DescribedDict._to_json_ready_value(self, val) -> Any` | Convert an arbitrary value to something JSON-storable. | trinity/_input/dictionary.py:704, trinity/_input/dictionary.py:572, trinity/_input/dictionary.py:648, trinity/_input/dictionary.py:649 (+9) | **none** |
| 577 | method | `DescribedDict._clean_for_snapshot(self, snap_id) -> Dict[str, Any]` | Build a JSON-ready snapshot dict of the current params. | trinity/_input/dictionary.py:737 | **none** |
| 711 | method | `DescribedDict.save_snapshot(self) -> None` | Save the current state into self.previous_snapshot. | trinity/_input/dictionary.py:1325, trinity/phase1_energy/run_energy_phase.py:262, trinity/phase1_energy/run_energy_phase.py:402, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1017 (+6) | test/test_metadata.py:147, test/test_show_run.py:65, test/test_show_run.py:76 |
| 764 | method | `DescribedDict.flush(self) -> None` | Append pending snapshots to dictionary.jsonl (line-delimited JSON). | trinity/_input/dictionary.py:1328, trinity/_input/dictionary.py:752, trinity/_input/dictionary.py:320, trinity/_input/dictionary.py:375 (+6) | test/test_logging_dedup.py:48, test/test_metadata.py:155, test/test_metadata.py:228, test/test_metadata.py:278 (+11) |
| 874 | method | `DescribedDict.load_snapshots(cls, path2output) -> Dict[str, Dict[str, Any]]` | Load dictionary.jsonl and return all snapshots. | trinity/_input/dictionary.py:944, trinity/_input/dictionary.py:971 | test/test_metadata.py:310 |
| 924 | method | `DescribedDict.load_snapshot(cls, path2output, snap_id) -> 'DescribedDict'` | Load a single snapshot into a DescribedDict. | trinity/_input/dictionary.py:1332, trinity/_input/dictionary.py:976 | **none** |
| 967 | method | `DescribedDict.load_latest_snapshot(cls, path2output) -> 'DescribedDict'` ⛔ | Convenience helper: load the snapshot with the largest integer id. | **none** | **none** |
| 981 | method | `DescribedDict.reset_keys(self, keys, value=np.nan) -> None` | Reset multiple keys to a specified value (default: np.nan). | trinity/main.py:317 | **none** |
| 1022 | func | `save_debug_snapshot(params, output_path=None) -> Path` ⛔ | Save a RAW snapshot of all params for debugging. | **none** | **none** |
| 1128 | func | `load_debug_snapshot(snapshot_path) -> Dict[str, Any]` ⛔ | Load a debug snapshot for use in tests. | **none** | **none** |
| 1232 | func | `updateDict(dictionary, keys_or_dataclass, values=None) -> None` | Bulk update helper supporting two usage patterns: | trinity/phase1_energy/run_energy_phase.py:94, trinity/phase1_energy/run_energy_phase.py:158, trinity/phase1_energy/run_energy_phase.py:184, trinity/phase1_energy/run_energy_phase.py:208 (+15) | **none** |

#### `trinity/_input/errors.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 11 | class | `class ParameterFileError(Exception)` | Raised when a parameter file has formatting or validation errors. | trinity/_input/read_param.py:222, trinity/_input/read_param.py:199, trinity/_input/registry.py:102, trinity/_input/registry.py:111 (+12) | test/test_betadelta_solver_switch.py:34, test/test_cf_leak.py:79, test/test_fA_source_boost.py:69, test/test_resolvers.py:153 (+9) |

#### `trinity/_input/fkappa_auto.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 75 | func | `fkappa_fire(mCloud_input, sfe, nCore) -> float` | Interpolated f_kappa needed for the cooling_balance trigger to fire. | trinity/_input/fkappa_auto.py:117 | test/test_fkappa_auto.py:42, test/test_fkappa_auto.py:32, test/test_fkappa_auto.py:33, test/test_fkappa_auto.py:34 (+6) |
| 97 | func | `resolve_fkappa_auto(value, params)` | Registry resolver for ``cooling_boost_kappa`` (read_param Step 7). | trinity/_input/registry.py:387 | test/test_fkappa_auto.py:73, test/test_fkappa_auto.py:68 |

#### `trinity/_input/param_spec.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 84 | class | `class ParamSpec()` | Declarative spec for one TRINITY parameter. | trinity/_input/registry.py:541, trinity/_input/registry.py:328, trinity/_input/registry.py:329, trinity/_input/registry.py:330 (+199) | test/test_registry.py:199, test/test_registry.py:182, test/test_registry.py:187, test/test_registry.py:195 |

#### `trinity/_input/read_param.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 43 | func | `read_param(path2file)` | Read parameter file and return DescribedDict with all TRINITY parameters. | run.py:160, trinity/_input/read_param.py:517 | test/test_dR2min_magic_number.py:110, test/test_fA_source_boost.py:37, test/test_fA_source_boost.py:57, test/test_fA_source_boost.py:93 (+17) |

#### `trinity/_input/registry.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 78 | func | `_profile_value(params) -> object` | — | trinity/_input/registry.py:84, trinity/_input/registry.py:88 | **none** |
| 83 | func | `_active_densBE(params) -> bool` | — | trinity/_input/registry.py:344, trinity/_input/registry.py:525, trinity/_input/registry.py:526, trinity/_input/registry.py:527 (+6) | test/test_active_when.py:39, test/test_active_when.py:40, test/test_active_when.py:41 |
| 87 | func | `_active_densPL(params) -> bool` | — | trinity/_input/registry.py:345 | test/test_active_when.py:45, test/test_active_when.py:46, test/test_active_when.py:47 |
| 99 | func | `_validate_ZCloud(value, params) -> None` | — | trinity/_input/registry.py:339 | test/test_validators.py:38, test/test_validators.py:84, test/test_validators.py:89 |
| 108 | func | `_validate_dens_profile(value, params) -> None` | — | trinity/_input/registry.py:342 | test/test_validators.py:42, test/test_validators.py:97, test/test_validators.py:102 |
| 117 | func | `_validate_cooling_boost_fA(value, params) -> None` | f_A > 0 required; warn on cross-knob combinations (double-boost). | trinity/_input/registry.py:388 | **none** |
| 151 | func | `_validate_betadelta_solver(value, params) -> None` | Selects the energy-implicit (beta, delta) solver. 'hybr' (default) | trinity/_input/registry.py:343 | test/test_betadelta_solver_switch.py:26, test/test_betadelta_solver_switch.py:27, test/test_betadelta_solver_switch.py:35 |
| 164 | func | `_validate_stop_at_rCloud_nSnap(value, params) -> None` | Validate AND coerce: whole-number floats (e.g. 5.0 from '5') | trinity/_input/registry.py:354 | test/test_validators.py:46, test/test_validators.py:121, test/test_validators.py:126, test/test_validators.py:134 (+4) |
| 189 | func | `_validate_coverFraction(value, params) -> None` | Covering fraction Cf must be a number in (0, 1]. Cf=1 is a sealed | trinity/_input/registry.py:341 | test/test_cf_leak.py:85, test/test_cf_leak.py:80 |
| 204 | func | `_validate_rCloud_max(value, params) -> None` | Maximum plausible cloud radius (rCloud_max) must be a positive number | trinity/_input/registry.py:349 | **none** |
| 229 | func | `_resolve_path2output(value, params) -> str` | Output directory.  Sentinel 'def_dir' resolves to | trinity/_input/registry.py:330 | test/test_resolvers.py:43, test/test_resolvers.py:66, test/test_resolvers.py:73 |
| 241 | func | `_resolve_path_cooling_nonCIE(value, params) -> str` | Non-CIE cooling directory.  Sentinel 'def_dir' resolves to the | trinity/_input/registry.py:393 | test/test_resolvers.py:47, test/test_resolvers.py:82, test/test_resolvers.py:88 |
| 252 | func | `_resolve_sps_bundle(value, params) -> str` | SPS bundle resolver (sps_path + sps_refmass + sps_column_map). | trinity/_input/registry.py:394 | test/test_read_sps.py:66, test/test_resolvers.py:51, test/test_resolvers.py:104, test/test_resolvers.py:160 (+3) |
| 541 | func | `specs_by_category(*categories) -> Iterable[ParamSpec]` ⛔ | — | **none** | **none** |
| 546 | func | `validate_all(params) -> None` | Run every spec's ``validator`` callable against ``params``. | trinity/_input/read_param.py:295 | test/test_validators.py:166 |
| 564 | func | `resolve_all(params) -> None` | Run every spec's ``resolver`` callable against ``params``. | trinity/_input/read_param.py:410 | test/test_resolvers.py:174, test/test_resolvers.py:181 |
| 588 | func | `apply_active_when(params) -> None` | Enforce ``active_when`` presence semantics against ``params``. | trinity/_input/read_param.py:441 | test/test_active_when.py:59, test/test_active_when.py:70, test/test_active_when.py:97, test/test_active_when.py:118 (+4) |
| 624 | func | `materialize_runtime(params) -> None` | Phase-8/9 entry point for ``read_param`` Step 10. | trinity/_input/read_param.py:472 | test/test_materialize_runtime.py:38, test/test_materialize_runtime.py:46, test/test_materialize_runtime.py:56, test/test_materialize_runtime.py:103 (+8) |
| 664 | func | `run_const_keys() -> tuple[str, ...]` | Keys written once to ``metadata.json`` (constant after phase 0). | trinity/_output/run_constants.py:77 | test/test_registry.py:140, test/test_registry.py:317, test/test_registry.py:152 |
| 673 | func | `metadata_exclude_keys() -> frozenset[str]` | Keys explicitly blocked from ``metadata.json`` (paths / loaded | trinity/_output/run_constants.py:83 | test/test_registry.py:146, test/test_registry.py:318, test/test_registry.py:152 |
| 696 | class | `class CompanionRule()` | If the user .param sets ``trigger`` to a value present as a key | trinity/_input/registry.py:704, trinity/_input/registry.py:705 | **none** |
| 715 | func | `validate_companions(user_dict) -> None` | Enforce every ``CompanionRule`` against the raw user .param dict. | trinity/_input/read_param.py:231 | test/test_validators.py:174, test/test_validators.py:188, test/test_validators.py:192, test/test_validators.py:197 (+2) |

#### `trinity/_input/sweep_jobs.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 98 | func | `emit_jobs(config, base_output_dir, jobs_dir, trinity_root, concurrency=None, dry_run=False, sweep_file=None)` | Generate a SLURM job-array bundle for a sweep. | run.py:866 | test/test_sweep_jobs.py:39, test/test_sweep_jobs.py:112, test/test_sweep_jobs.py:122 |
| 244 | func | `_fmt(v)` | Float-with-no-fraction -> int for tidy printing (100000.0 -> 100000); else as-is. | trinity/_input/sweep_jobs.py:276 | **none** |
| 249 | func | `failure_breakdown(failed, manifest_runs)` | Tally failed runs by each *swept* parameter (and by return code) so a regime-shaped | trinity/_input/sweep_jobs.py:378 | **none** |
| 282 | func | `collect_report(jobs_dir)` | Aggregate per-task results into a SweepReport. | run.py:853 | test/test_sweep_jobs.py:143, test/test_sweep_jobs.py:154, test/test_sweep_jobs.py:174, test/test_sweep_jobs.py:163 |

#### `trinity/_input/sweep_parser.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 40 | func | `parse_value(val_str) -> Union[bool, float, str, List[Any]]` | Parse a string value into appropriate Python type. | trinity/_input/sweep_parser.py:906, trinity/_input/sweep_parser.py:330, trinity/_input/sweep_parser.py:443 | **none** |
| 96 | func | `parse_list(list_str) -> List[Any]` | Parse list syntax: [val1, val2, val3] -> [parsed_val1, parsed_val2, ...] | trinity/_input/sweep_parser.py:72 | **none** |
| 154 | func | `parse_tuple_line(line) -> Optional[Tuple[List[str], List[List[Any]]]]` | Parse a tuple definition line. | trinity/_input/sweep_parser.py:1044, trinity/_input/sweep_parser.py:1023, trinity/_input/sweep_parser.py:426 | **none** |
| 244 | class | `class SweepConfig()` | Configuration parsed from a sweep parameter file. | trinity/_input/sweep_parser.py:354, trinity/_input/sweep_parser.py:476, trinity/_input/sweep_parser.py:488, trinity/_input/sweep_parser.py:855 | **none** |
| 252 | method | `SweepConfig.is_tuple_mode(self) -> bool` | Check if this uses tuple syntax (pure tuple or hybrid). | run.py:433, run.py:476, trinity/_input/sweep_parser.py:508, trinity/_input/sweep_parser.py:874 (+1) | **none** |
| 257 | method | `SweepConfig.is_hybrid_mode(self) -> bool` | Check if this is hybrid mode (tuple + sweep params). | run.py:421, run.py:473 | **none** |
| 262 | func | `read_sweep_param(path2file) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]` ⛔ | Read a sweep-enabled parameter file. | **none** | **none** |
| 354 | func | `read_sweep_config(path2file) -> SweepConfig` | Read a sweep-enabled parameter file and return a SweepConfig. | run.py:397, run.py:865 | test/test_sweep_jobs.py:37, test/test_sweep_jobs.py:67, test/test_sweep_jobs.py:110, test/test_sweep_jobs.py:120 |
| 488 | func | `generate_combinations_from_config(config) -> Iterator[Tuple[Dict[str, Any], str]]` | Generate parameter combinations from a SweepConfig. | run.py:534, run.py:582, run.py:471, trinity/_input/sweep_jobs.py:129 | test/test_sweep_jobs.py:68 |
| 543 | func | `generate_combinations(base_params, sweep_params) -> Iterator[Tuple[Dict[str, Any], str]]` | Generate all parameter combinations (Cartesian product). | trinity/_input/sweep_parser.py:540 | **none** |
| 609 | func | `_reject_unsafe_sweep_value(key, value) -> None` | Raise ``ValueError`` if a swept value would be unsafe to embed in a | trinity/_input/sweep_parser.py:670 | **none** |
| 648 | func | `_generic_suffix_token(key, value) -> str` | Build a single ``{key}{value}`` token for an arbitrary swept parameter | trinity/_input/sweep_parser.py:764 | **none** |
| 683 | func | `generate_run_name(params, swept_keys=None) -> str` | Generate output folder name following existing TRINITY convention. | trinity/_input/sweep_parser.py:565, trinity/_input/sweep_parser.py:582, trinity/_input/sweep_parser.py:938, trinity/_input/sweep_parser.py:958 (+5) | **none** |
| 780 | func | `format_scientific(value) -> str` | Format a number in compact scientific notation. | trinity/_input/sweep_parser.py:725, trinity/_input/sweep_parser.py:732, trinity/_input/sweep_parser.py:921 | **none** |
| 832 | func | `count_combinations(sweep_params) -> int` | Count total number of combinations without generating them. | trinity/_input/sweep_parser.py:883, trinity/_input/sweep_parser.py:878 | **none** |
| 855 | func | `count_combinations_from_config(config) -> int` | Count total number of combinations from a SweepConfig. | run.py:403 | **none** |

#### `trinity/_input/sweep_runner.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 36 | class | `class SimulationResult()` | Result of a single simulation run. | run.py:619, run.py:655, run.py:683, trinity/_input/sweep_jobs.py:341 (+8) | **none** |
| 48 | class | `class SweepProgress()` ⛔ | Track progress of parameter sweep. | **none** | **none** |
| 56 | method | `SweepProgress.elapsed(self) -> timedelta` | — | trinity/_input/sweep_runner.py:408, trinity/_input/sweep_runner.py:448, trinity/_input/sweep_runner.py:449, trinity/_input/sweep_runner.py:64 (+3) | **none** |
| 61 | method | `SweepProgress.eta(self) -> Optional[timedelta]` ⛔ | — | **none** | **none** |
| 76 | func | `_validate_sweep_combination(params_dict)` | Validate a single sweep combination's GMC parameters. | run.py:537, run.py:483, trinity/_input/sweep_jobs.py:140 | **none** |
| 155 | func | `generate_param_file(params, run_name, run_output_dir) -> str` | Generate parameter file content for a single simulation. | trinity/_input/sweep_jobs.py:179, trinity/_input/sweep_runner.py:251 | **none** |
| 212 | func | `run_single_simulation(params, run_name, trinity_root, base_output_dir, timeout_hours=24.0) -> SimulationResult` | Execute a single TRINITY simulation. | run.py:630 | **none** |
| 349 | class | `class ProgressBar()` | Progress bar with fallback if tqdm not available. | run.py:585, trinity/_input/sweep_runner.py:637 | **none** |
| 379 | method | `ProgressBar.update(self, name, success)` | Update progress after a simulation completes. | run.py:664, trinity/_input/sweep_runner.py:640, trinity/_input/sweep_runner.py:389, trinity/_output/trinity_reader.py:351 | test/test_cloudy_run_loader.py:298, test/test_log_stopping_fate.py:43, test/test_resolvers.py:98, test/test_sweep_workers.py:115 (+2) |
| 397 | method | `ProgressBar.set_running(self, names)` | Show currently running simulations. | run.py:640 | **none** |
| 405 | method | `ProgressBar._print_progress(self)` | Print progress bar to terminal. | trinity/_input/sweep_runner.py:395 | **none** |
| 439 | method | `ProgressBar.close(self)` | Clean up progress display. | run.py:722, trinity/_input/sweep_runner.py:641, trinity/_input/sweep_runner.py:442, trinity/bubble_structure/bubble_luminosity.py:147 (+2) | test/test_bubble_lsoda_quiet.py:30, test/test_bubble_lsoda_quiet.py:31, test/test_bubble_lsoda_quiet.py:54, test/test_bubble_lsoda_quiet.py:55 |
| 446 | method | `ProgressBar.summary(self) -> str` | Generate summary string. | run.py:749, trinity/_input/sweep_runner.py:642, trinity/_output/cloudy/snapshot_to_deck.py:208, trinity/_output/cloudy/snapshot_to_deck.py:212 | test/test_cloudy_run_loader.py:404, test/test_cloudy_run_loader.py:76, test/test_cloudy_run_loader.py:77, test/test_cloudy_run_loader.py:78 (+1) |
| 471 | class | `class SweepReport()` | Complete report of a parameter sweep. | run.py:732, trinity/_input/sweep_jobs.py:356 | **none** |
| 480 | method | `SweepReport.write_report(self, output_path) -> Path` | Write detailed human-readable report to file. | run.py:742, trinity/_input/sweep_jobs.py:372 | **none** |
| 531 | method | `SweepReport._write_physics_section(self, f) -> None` | Write the per-run physics-outcomes table into an open report file. | trinity/_input/sweep_runner.py:527 | **none** |
| 595 | method | `SweepReport.write_json(self, output_path) -> Path` | Write machine-readable JSON report. | run.py:743, trinity/_input/sweep_jobs.py:368 | **none** |

#### `trinity/_output/_metadata_io.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 40 | class | `class _NpEncoder(json.JSONEncoder)` | JSON encoder that coerces numpy scalars / arrays to plain Python. | trinity/_output/_metadata_io.py:92, trinity/_output/_metadata_io.py:123 | **none** |
| 47 | method | `_NpEncoder.default(self, obj) -> Any` | — | trinity/_input/dictionary.py:92, trinity/_input/dictionary.py:564, trinity/_input/dictionary.py:1093, trinity/_input/registry.py:657 (+2) | test/test_active_when.py:125, test/test_active_when.py:82, test/test_active_when.py:84, test/test_betadelta_solver_switch.py:45 (+10) |
| 59 | func | `read_metadata(run_dir) -> Dict[str, Any]` | Parse ``<run_dir>/metadata.json`` and return the dict. | trinity/_output/_metadata_io.py:117, trinity/_output/simulation_end.py:334 | **none** |
| 78 | func | `write_metadata_atomic(run_dir, payload) -> None` | Write ``payload`` to ``<run_dir>/metadata.json`` atomically. | trinity/_input/dictionary.py:850, trinity/_output/_metadata_io.py:132 | **none** |
| 96 | func | `update_metadata_atomic(run_dir, **block_updates) -> None` | Read ``metadata.json``, merge ``block_updates`` at the top level, | trinity/_output/simulation_end.py:226, trinity/_output/simulation_end.py:740 | **none** |

#### `trinity/_output/cloudy/dlaw.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 45 | class | `class DlawError(ValueError)` | Raised when dlaw construction fails validation. | trinity/_output/cloudy/dlaw.py:95, trinity/_output/cloudy/dlaw.py:99, trinity/_output/cloudy/dlaw.py:101, trinity/_output/cloudy/dlaw.py:105 (+14) | test/test_cloudy_dlaw.py:113, test/test_cloudy_dlaw.py:122, test/test_cloudy_dlaw.py:248, test/test_cloudy_dlaw.py:263 (+10) |
| 49 | func | `build_dlaw_block(shell_r_pc, shell_log_n_pc3, *, ambient_r_pc=None, ambient_log_n_pc3=None, r_in_pc, r_out_pc, min_rows=DEFAULT_MIN_ROWS, dens_profile='densPL', dlaw_open=DEFAULT_DLAW_OPEN, dlaw_row_prefix=DEFAULT_DLAW_ROW_PREFIX, dlaw_close=DEFAULT_DLAW_CLOSE, edge_threshold=DEFAULT_EDGE_THRESHOLD) -> str` | Construct a CLOUDY dlaw block from TRINITY shell + (optional) ambient profiles. | trinity/_output/cloudy/snapshot_to_deck.py:247 | test/test_cloudy_dlaw.py:61, test/test_cloudy_dlaw.py:79, test/test_cloudy_dlaw.py:93, test/test_cloudy_dlaw.py:145 (+18) |
| 205 | func | `_densify_preserving_edges(log_r, log_n, *, target_rows, edge_threshold) -> tuple[np.ndarray, np.ndarray]` | Insert linearly-interpolated points into non-edge spans until | trinity/_output/cloudy/dlaw.py:181 | **none** |

#### `trinity/_output/cloudy/run_loader.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 39 | class | `class RunLoadError(ValueError)` | Raised when a run directory cannot be loaded into a RunBundle. | trinity/_output/cloudy/run_loader.py:85, trinity/_output/cloudy/run_loader.py:91, trinity/_output/cloudy/run_loader.py:96, trinity/_output/cloudy/run_loader.py:89 (+4) | test/test_cloudy_run_loader.py:59, test/test_cloudy_run_loader.py:322, test/test_cloudy_run_loader.py:328, test/test_cloudy_run_loader.py:338 (+4) |
| 44 | class | `class RunBundle()` | Everything trinity_to_cloudy needs about one TRINITY run. | trinity/_output/cloudy/run_loader.py:57, trinity/_output/cloudy/run_loader.py:140, trinity/_output/cloudy/snapshot_to_deck.py:49, trinity/_output/cloudy/trinity_to_cloudy.py:189 (+4) | test/test_cloudy_run_loader.py:58, test/test_cloudy_run_loader.py:70 |
| 57 | func | `load_run(run_dir) -> RunBundle` | Parse a TRINITY run directory and return a RunBundle. | trinity/_output/cloudy/trinity_to_cloudy.py:334 | test/test_cloudy_cli.py:105, test/test_cloudy_cli.py:118, test/test_cloudy_cli.py:136, test/test_cloudy_cli.py:147 (+19) |
| 154 | func | `_parse_summary_txt(text) -> dict[str, Any]` | Parse a legacy ``<model>_summary.txt`` produced by pre-Phase-5 runs. | trinity/_output/cloudy/run_loader.py:114 | test/test_cloudy_run_loader.py:150, test/test_cloudy_run_loader.py:156, test/test_cloudy_run_loader.py:163, test/test_cloudy_run_loader.py:169 (+1) |
| 188 | func | `_parse_simulation_end(text) -> dict[str, Any]` | Pull the outcome category, detail message, exit code, and final-state | trinity/_output/cloudy/run_loader.py:138 | test/test_cloudy_run_loader.py:221, test/test_cloudy_run_loader.py:243, test/test_cloudy_run_loader.py:251, test/test_cloudy_run_loader.py:265 (+1) |
| 289 | func | `_coerce_scalar(s) -> Any` | Parse a summary-file value string into the most specific Python type | trinity/_output/cloudy/run_loader.py:184 | test/test_cloudy_run_loader.py:117, test/test_cloudy_run_loader.py:124, test/test_cloudy_run_loader.py:113, test/test_cloudy_run_loader.py:132 |
| 327 | func | `_looks_like_int(s) -> bool` | — | trinity/_output/cloudy/run_loader.py:307 | **none** |
| 332 | func | `_safe_int(s) -> int \| None` | — | trinity/_output/cloudy/run_loader.py:225 | **none** |

#### `trinity/_output/cloudy/snapshot_to_deck.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 43 | class | `class SnapshotInvalid(ValueError)` | Raised when a snapshot fails validation for CLOUDY conversion. | trinity/_output/cloudy/snapshot_to_deck.py:102, trinity/_output/cloudy/snapshot_to_deck.py:116, trinity/_output/cloudy/snapshot_to_deck.py:121, trinity/_output/cloudy/snapshot_to_deck.py:124 (+18) | test/test_cloudy_run_loader.py:61, test/test_cloudy_snapshot_to_deck.py:181, test/test_cloudy_snapshot_to_deck.py:195, test/test_cloudy_snapshot_to_deck.py:203 (+14) |
| 47 | func | `snapshot_to_values(snap, bundle, *, z_override=None, radius_out_pc=None, age_min_yr=DEFAULT_AGE_MIN_YR, age_max_yr=DEFAULT_AGE_MAX_YR, hard_age_bounds=False, min_rows=DEFAULT_MIN_ROWS, extend_with_ambient=True) -> dict[str, Any]` | Validate a snapshot and compute the values a CLOUDY .in template needs. | trinity/_output/cloudy/trinity_to_cloudy.py:350 | test/test_cloudy_run_loader.py:60, test/test_cloudy_snapshot_to_deck.py:104, test/test_cloudy_snapshot_to_deck.py:153, test/test_cloudy_snapshot_to_deck.py:162 (+29) |

#### `trinity/_output/cloudy/trinity_to_cloudy.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 81 | class | `class UnsubstitutedPlaceholder(ValueError)` | Raised when the rendered deck still contains {{KEY}} placeholders. | trinity/_output/cloudy/trinity_to_cloudy.py:306, trinity/_output/cloudy/trinity_to_cloudy.py:393 | test/test_cloudy_cli.py:55 |
| 89 | func | `_parse_args(argv) -> argparse.Namespace` | — | trinity/_output/cloudy/trinity_to_cloudy.py:331 | test/test_cloudy_cli.py:91, test/test_cloudy_cli.py:108, test/test_cloudy_cli.py:120, test/test_cloudy_cli.py:137 (+9) |
| 184 | class | `class PickedSnapshot()` | — | trinity/_output/cloudy/trinity_to_cloudy.py:189, trinity/_output/cloudy/trinity_to_cloudy.py:269, trinity/_output/cloudy/trinity_to_cloudy.py:192, trinity/_output/cloudy/trinity_to_cloudy.py:197 (+3) | test/test_cloudy_cli.py:107, test/test_cloudy_cli.py:119 |
| 189 | func | `_pick_snapshots(bundle, args) -> list[PickedSnapshot]` | Resolve the picker flags into a list of (index, snapshot) tuples. | trinity/_output/cloudy/trinity_to_cloudy.py:340 | test/test_cloudy_cli.py:138, test/test_cloudy_cli.py:149, test/test_cloudy_cli.py:166, test/test_cloudy_cli.py:176 (+3) |
| 237 | func | `_check_status(bundle, *, force) -> None` | Refuse to convert runs whose termination exit code is not in the clean | trinity/_output/cloudy/trinity_to_cloudy.py:338 | **none** |
| 266 | func | `_build_prefix(args, bundle, pick, age_myr, phase) -> str` | Auto-build a filename-safe prefix:  <model>_<idx>_<phase>_t<age>myr | trinity/_output/cloudy/trinity_to_cloudy.py:360 | test/test_cloudy_cli.py:109, test/test_cloudy_cli.py:123 |
| 284 | func | `_resolve_out_dir(args, bundle) -> Path` | — | trinity/_output/cloudy/trinity_to_cloudy.py:341 | **none** |
| 292 | func | `render_template(template_text, values) -> str` | Substitute {{KEY}} placeholders. Raise UnsubstitutedPlaceholder on any | trinity/_output/cloudy/trinity_to_cloudy.py:368 | test/test_cloudy_cli.py:43, test/test_cloudy_cli.py:49, test/test_cloudy_cli.py:60, test/test_cloudy_cli.py:66 (+1) |
| 312 | func | `_load_template(args) -> str` | — | trinity/_output/cloudy/trinity_to_cloudy.py:343 | **none** |
| 319 | func | `_resolve_linelist(args) -> Path` | — | trinity/_output/cloudy/trinity_to_cloudy.py:344 | **none** |
| 330 | func | `main(argv=None) -> int` | — | trinity/_output/cloudy/trinity_to_cloudy.py:510 | test/test_cloudy_cli.py:203, test/test_cloudy_cli.py:227, test/test_cloudy_cli.py:243, test/test_cloudy_cli.py:290 (+7) |
| 426 | func | `_write_outputs(out_dir, prefix, deck_text, dlaw_block) -> None` | — | trinity/_output/cloudy/trinity_to_cloudy.py:377 | **none** |
| 437 | func | `_copy_linelist(out_dir, src) -> None` | — | trinity/_output/cloudy/trinity_to_cloudy.py:413 | **none** |
| 445 | func | `_write_manifest(out_dir, records) -> None` | — | trinity/_output/cloudy/trinity_to_cloudy.py:416 | **none** |
| 455 | func | `_todo_line(args) -> str \| None` | Closing-summary TODO printed only when the SB99 sentinel is in the deck. | trinity/_output/cloudy/trinity_to_cloudy.py:472 | **none** |
| 466 | func | `_print_summary(bundle, records, args, out_dir) -> None` | — | trinity/_output/cloudy/trinity_to_cloudy.py:418 | **none** |

#### `trinity/_output/header.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 17 | func | `display()` | Display the TRINITY welcome header and initial parameter summary. | run.py:847 | **none** |
| 40 | func | `show_logo()` | Display the TRINITY ASCII art logo. | trinity/_output/header.py:26 | **none** |
| 55 | func | `link(url, label=None)` | Create a clickable hyperlink for terminal output. | trinity/_output/header.py:28, trinity/_output/header.py:30 | **none** |
| 80 | func | `show_param(params)` | Display initial parameter summary. | run.py:162 | **none** |

#### `trinity/_output/run_constants.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 135 | func | `metadata_keys_to_rehydrate(metadata) -> dict` | Return ``metadata`` with the reserved top-level keys removed. | trinity/_input/dictionary.py:914, trinity/_output/cloudy/run_loader.py:107, trinity/_output/trinity_reader.py:458 | **none** |

#### `trinity/_output/show_run.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 64 | func | `_collapse_descriptor(final_state, outcome) -> Optional[str]` | Three-state collapse status from the final snapshot. | trinity/_output/show_run.py:401, trinity/_output/show_run.py:478 | **none** |
| 94 | func | `_status_line(termination, is_successful, collapse_state=None) -> str` | One-line "Status : ✓ SUCCESS  (outcome)" header. | trinity/_output/show_run.py:407, trinity/_output/show_run.py:481 | **none** |
| 118 | func | `_fmt_or_na(value, fmt='.4e', default='n/a') -> str` | Format ``value`` with ``fmt`` or return ``default``. | trinity/_output/show_run.py:154, trinity/_output/show_run.py:173, trinity/_output/show_run.py:180, trinity/_output/show_run.py:185 (+9) | **none** |
| 128 | func | `_cloud_section(md) -> list[str]` | Render the 'Cloud' section from metadata.json's run-constants. | trinity/_output/show_run.py:418 | **none** |
| 160 | func | `_final_state_section(final_state, collapse_state=None) -> list[str]` | Render the 'Final state' section. | trinity/_output/show_run.py:421 | **none** |
| 238 | func | `_resolve_run_status(run_dir) -> dict` | Gather all the bits ``format_run_summary`` and ``main`` need. | trinity/_output/show_run.py:392, trinity/_output/show_run.py:476 | **none** |
| 326 | func | `_termination_debug_section(td) -> list[str]` | Render only the actionable bits of the termination_debug block. | trinity/_output/show_run.py:423 | **none** |
| 371 | func | `format_run_summary(run_dir) -> str` | Build the multi-line pretty-printed summary string. | trinity/_input/dictionary.py:338, trinity/_output/show_run.py:495 | test/test_show_run.py:132, test/test_show_run.py:159, test/test_show_run.py:175, test/test_show_run.py:192 (+4) |
| 436 | func | `main(argv=None) -> int` | — | trinity/_output/show_run.py:500 | test/test_show_run.py:279, test/test_show_run.py:290, test/test_show_run.py:304, test/test_show_run.py:316 (+2) |

#### `trinity/_output/simulation_end.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 55 | class | `class SimulationEndCode(Enum)` | Enumeration of simulation end reasons with exit codes. | trinity/_output/simulation_end.py:192, trinity/_output/simulation_end.py:196, trinity/_output/simulation_end.py:399, trinity/_output/terminal_prints.py:217 (+29) | test/test_energy_collapse_guard.py:78 |
| 100 | method | `SimulationEndCode.code(self) -> int` | Numeric exit code. | trinity/_output/simulation_end.py:240, trinity/_output/simulation_end.py:96, trinity/_output/simulation_end.py:217, trinity/_output/simulation_end.py:125 (+26) | test/test_energy_collapse_guard.py:79, test/test_energy_collapse_guard.py:80, test/test_phase_events.py:148 |
| 105 | method | `SimulationEndCode.outcome(self) -> str` | Short categorical label mirrored into ``metadata.json[termination].outcome``. | trinity/_output/simulation_end.py:97, trinity/_output/simulation_end.py:399, trinity/_output/simulation_end.py:218 | **none** |
| 109 | method | `SimulationEndCode.is_clean(self) -> bool` | True if the run finished with a clean physical/intentional outcome (0-9). | trinity/_output/terminal_prints.py:222 | **none** |
| 113 | method | `SimulationEndCode.is_error(self) -> bool` | True if the run failed with a parameter or numerical error (10-29). | trinity/_output/terminal_prints.py:224 | **none** |
| 117 | method | `SimulationEndCode.is_inspection_required(self) -> bool` ⛔ | True if the run completed but warrants a human look (50-59 or 99). | **none** | **none** |
| 122 | method | `SimulationEndCode.from_code(cls, code) -> 'SimulationEndCode'` | Look up the enum member by numeric code, or UNKNOWN if no match. | trinity/_output/simulation_end.py:196, trinity/_output/simulation_end.py:399, trinity/_output/terminal_prints.py:217 | **none** |
| 130 | func | `write_simulation_end(params, output_dir=None) -> int` | Mirror the end-of-run termination + final-state data into | trinity/main.py:191 | test/test_metadata.py:666, test/test_metadata.py:689, test/test_metadata.py:713, test/test_metadata.py:734 (+5) |
| 243 | func | `_build_final_state_block(params) -> Dict[str, Any]` | Build the ``final_state`` block from the runtime params. | trinity/_output/simulation_end.py:223 | **none** |
| 310 | func | `read_simulation_end(output_dir) -> Optional[Dict[str, Any]]` | Read the termination summary for a run. | trinity/_output/show_run.py:298 | test/test_metadata.py:846, test/test_metadata.py:873, test/test_metadata.py:895, test/test_metadata.py:905 (+1) |
| 448 | func | `_load_last_snapshots(output_dir, n=2) -> List[Dict[str, Any]]` | Load the last N snapshots from dictionary.jsonl. | trinity/_output/simulation_end.py:592 | **none** |
| 488 | func | `_compute_change(old_val, new_val) -> Tuple[str, float, bool]` | Compute change between two values. | trinity/_output/simulation_end.py:629 | **none** |
| 558 | func | `write_termination_debug_report(output_dir, reason='Unknown') -> None` | Mirror last-2-snapshot debug data into ``metadata.json[termination_debug]``. | trinity/_input/dictionary.py:328, trinity/_input/dictionary.py:379 | test/test_conventional_units.py:102, test/test_metadata.py:943, test/test_metadata.py:1025 |
| 679 | func | `_jsonable(val) -> Any` | Coerce a numeric value to a JSON-friendly type. | trinity/_output/simulation_end.py:610, trinity/_output/simulation_end.py:612, trinity/_output/simulation_end.py:640, trinity/_output/simulation_end.py:641 (+1) | **none** |
| 698 | func | `_build_sanity_checks(snap) -> List[Dict[str, Any]]` | Run the small set of last-snapshot physics sanity checks. | trinity/_output/simulation_end.py:673 | **none** |
| 736 | func | `_merge_termination_debug(output_path, block) -> None` | Merge ``termination_debug`` into metadata.json; never raise. | trinity/_output/simulation_end.py:675, trinity/_output/simulation_end.py:601 | **none** |

#### `trinity/_output/terminal_prints.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 24 | func | `_format_banner(message, width=50) -> str` | Format a message as a banner with dashes. | trinity/_output/terminal_prints.py:32, trinity/_output/terminal_prints.py:37, trinity/_output/terminal_prints.py:42, trinity/_output/terminal_prints.py:47 | **none** |
| 30 | func | `bubble()` ⛔ | Log message for bubble structure calculation. | **none** | **none** |
| 35 | func | `phase0(time)` | Log initialization message with timestamp. | trinity/main.py:114 | **none** |
| 40 | func | `phase(string)` | Log phase transition message. | trinity/main.py:247, trinity/main.py:281, trinity/main.py:299, trinity/main.py:325 | **none** |
| 45 | func | `shell()` ⛔ | Log message for shell structure calculation. | **none** | **none** |
| 50 | class | `class cprint()` | A class that deals with printing with colours in terminal. | trinity/_output/header.py:31, trinity/_output/header.py:32, trinity/_output/header.py:33, trinity/_output/header.py:89 (+6) | **none** |
| 106 | func | `log_file_saved(filepath, description='File saved')` ⛔ | Log a file save operation with distinctive formatting. | **none** | **none** |
| 111 | func | `log_warning(message)` ⛔ | Log a warning message with distinctive formatting. | **none** | **none** |
| 116 | func | `log_error(message)` ⛔ | Log an error message with distinctive formatting. | **none** | **none** |
| 143 | func | `_phys(params, key, conv=1.0, fmt='.4e')` | Read params[key].value in display units; tolerant of missing/bad values. | trinity/_output/terminal_prints.py:172 | **none** |
| 163 | func | `format_state(params, label=None, *, oneline=False)` | Format the core bubble state as a log string. | trinity/_output/terminal_prints.py:230, trinity/_output/terminal_prints.py:202, trinity/phase1_energy/run_energy_phase.py:110, trinity/phase1_energy/run_energy_phase.py:407 (+6) | test/test_log_stopping_fate.py:94, test/test_log_stopping_fate.py:101, test/test_log_stopping_fate.py:108, test/test_log_stopping_fate.py:116 (+1) |
| 187 | func | `heartbeat(params, tag, segment, tmin, tmax)` | Emit a throttled one-line progress heartbeat for a long-phase loop. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1179, trinity/phase1c_transition/run_transition_phase.py:690, trinity/phase2_momentum/run_momentum_phase.py:767 | test/test_log_stopping_fate.py:130, test/test_log_stopping_fate.py:132 |
| 205 | func | `format_end_report(params)` | One INFO block: the stopping fate in words, then the final-state block. | trinity/main.py:194 | test/test_log_stopping_fate.py:56, test/test_log_stopping_fate.py:71, test/test_log_stopping_fate.py:80 |

#### `trinity/_output/trinity_reader.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 284 | class | `class Snapshot()` | A single simulation snapshot. | trinity/_output/trinity_reader.py:467, trinity/_output/trinity_reader.py:745, trinity/_output/trinity_reader.py:473, trinity/_output/trinity_reader.py:471 (+4) | **none** |
| 295 | method | `Snapshot.get(self, key, default=None) -> Any` | Get snapshot data with default. | run.py:145, run.py:521, run.py:353, run.py:454 (+207) | test/test_betadelta_hybr_stress.py:41, test/test_betadelta_hybr_stress.py:96, test/test_betadelta_hybr_stress.py:117, test/test_bubble_solver_stress.py:47 (+32) |
| 299 | method | `Snapshot.keys(self) -> List[str]` | Get all available keys. | run.py:475, run.py:479, run.py:709, trinity/_functions/unit_conversions.py:450 (+22) | test/test_cloudy_cli.py:260, test/test_metadata.py:671, test/test_registry.py:135, test/test_resolvers.py:163 |
| 304 | method | `Snapshot.t_now(self) -> float` | Current time. | trinity/_output/cloudy/trinity_to_cloudy.py:199, trinity/_output/cloudy/trinity_to_cloudy.py:200, trinity/_output/cloudy/trinity_to_cloudy.py:162, trinity/_output/trinity_reader.py:316 (+1) | test/test_cloudy_cli.py:142 |
| 309 | method | `Snapshot.phase(self) -> str` | Current phase. | trinity/_output/cloudy/trinity_to_cloudy.py:213, trinity/_output/cloudy/trinity_to_cloudy.py:164, trinity/_output/cloudy/trinity_to_cloudy.py:214, trinity/_output/cloudy/trinity_to_cloudy.py:216 (+7) | test/test_cloudy_cli.py:168, test/test_cloudy_cli.py:177 |
| 323 | class | `class TrinityOutput()` | Reader for TRINITY simulation output files (.jsonl). | trinity/_functions/extract_example_snapshots.py:60, trinity/_functions/extract_example_snapshots.py:83, trinity/_output/cloudy/run_loader.py:54, trinity/_output/cloudy/run_loader.py:121 (+4) | test/test_metadata.py:297, test/test_metadata.py:330, test/test_metadata.py:370, test/test_metadata.py:388 (+10) |
| 358 | method | `TrinityOutput.open(cls, filepath) -> 'TrinityOutput'` | Open a TRINITY output file. | trinity/_functions/extract_example_snapshots.py:83, trinity/_output/cloudy/run_loader.py:121, trinity/_output/show_run.py:263, trinity/_output/trinity_reader.py:1091 (+5) | test/test_bubble_lsoda_quiet.py:20, test/test_bubble_lsoda_quiet.py:42, test/test_metadata.py:297, test/test_metadata.py:330 (+18) |
| 394 | method | `TrinityOutput._load_json_format(cls, filepath) -> List[dict]` | Load old-style .json format (single JSON object with all snapshots). | trinity/_output/trinity_reader.py:378, trinity/_output/trinity_reader.py:386 | **none** |
| 416 | method | `TrinityOutput._load_jsonl_format(cls, filepath) -> List[dict]` | Load new-style .jsonl format (line-delimited JSON). | trinity/_output/trinity_reader.py:380, trinity/_output/trinity_reader.py:384 | **none** |
| 428 | method | `TrinityOutput._rehydrate_metadata(run_dir, snapshots) -> None` | Merge ``<run_dir>/metadata.json`` into every snapshot's data | trinity/_output/trinity_reader.py:412, trinity/_output/trinity_reader.py:424 | **none** |
| 479 | method | `TrinityOutput.termination(self) -> Optional[Dict[str, Any]]` | ``termination`` block from ``metadata.json`` (Phase 2, v3+ | trinity/_output/cloudy/run_loader.py:129, trinity/_output/cloudy/run_loader.py:130, trinity/_output/show_run.py:265, trinity/_output/trinity_reader.py:536 | test/test_metadata.py:781, test/test_metadata.py:813 |
| 495 | method | `TrinityOutput.final_state(self) -> Optional[Dict[str, Any]]` | ``final_state`` block from ``metadata.json``, or ``None`` if | trinity/_output/show_run.py:266 | test/test_metadata.py:788, test/test_metadata.py:814 |
| 510 | method | `TrinityOutput.termination_debug(self) -> Optional[Dict[str, Any]]` | ``termination_debug`` block from ``metadata.json`` (Phase 5, | trinity/_output/show_run.py:267 | test/test_metadata.py:1002 |
| 525 | method | `TrinityOutput.is_successful_run(self) -> Optional[bool]` | Three-valued success flag derived from | trinity/_output/show_run.py:268 | test/test_metadata.py:797, test/test_metadata.py:803, test/test_metadata.py:815 |
| 548 | method | `TrinityOutput.metadata(self) -> Dict[str, Any]` | Parsed ``metadata.json`` for this run (cached, loaded on | trinity/_output/cloudy/snapshot_to_deck.py:120, trinity/_output/cloudy/snapshot_to_deck.py:122, trinity/_output/cloudy/snapshot_to_deck.py:254, trinity/_output/cloudy/trinity_to_cloudy.py:195 (+7) | test/test_cloudy_run_loader.py:73, test/test_cloudy_run_loader.py:74, test/test_metadata.py:627, test/test_metadata.py:571 (+3) |
| 573 | method | `TrinityOutput.initial_cloud_profile(self) -> tuple` | Reconstruct the initial cloud profile ``(r_arr, n_arr, m_arr)`` | trinity/_output/cloudy/snapshot_to_deck.py:230 | test/test_metadata.py:564, test/test_metadata.py:602, test/test_metadata.py:619 |
| 644 | method | `TrinityOutput.model_name(self) -> str` | Model name from first snapshot. | trinity/_output/cloudy/snapshot_to_deck.py:262, trinity/_output/cloudy/trinity_to_cloudy.py:280, trinity/_output/trinity_reader.py:962 | test/test_cloudy_run_loader.py:72, test/test_cloudy_run_loader.py:377, test/test_cloudy_run_loader.py:402 |
| 649 | method | `TrinityOutput.keys(self) -> List[str]` | All available parameter keys. | run.py:475, run.py:479, run.py:709, trinity/_functions/unit_conversions.py:450 (+22) | test/test_cloudy_cli.py:260, test/test_metadata.py:671, test/test_registry.py:135, test/test_resolvers.py:163 |
| 654 | method | `TrinityOutput.phases(self) -> List[str]` | List of unique phases in the output. | trinity/_output/trinity_reader.py:970 | **none** |
| 659 | method | `TrinityOutput.t_min(self) -> float` | Minimum time in output. | trinity/_output/trinity_reader.py:1061, trinity/_output/trinity_reader.py:940, trinity/_output/trinity_reader.py:964 | **none** |
| 664 | method | `TrinityOutput.t_max(self) -> float` | Maximum time in output. | trinity/_output/trinity_reader.py:1061, trinity/_output/trinity_reader.py:942, trinity/_output/trinity_reader.py:964 | **none** |
| 668 | method | `TrinityOutput.get(self, key, as_array=True) -> Union[np.ndarray, List[Any]]` | Get a parameter across all snapshots. | run.py:145, run.py:521, run.py:353, run.py:454 (+207) | test/test_betadelta_hybr_stress.py:41, test/test_betadelta_hybr_stress.py:96, test/test_betadelta_hybr_stress.py:117, test/test_bubble_solver_stress.py:47 (+32) |
| 692 | method | `TrinityOutput.get_at_time(self, t, key=None, mode='interpolate', n_neighbors=5, quiet=False) -> Union[Snapshot, Any]` | Get snapshot at a specific time. | trinity/_output/cloudy/trinity_to_cloudy.py:196, trinity/_output/cloudy/trinity_to_cloudy.py:200, trinity/_output/cloudy/trinity_to_cloudy.py:224 | test/test_cloudy_snapshot_to_deck.py:102 |
| 745 | method | `TrinityOutput._interpolate_snapshot(self, t, n_neighbors=5, quiet=False) -> Snapshot` | Create an interpolated snapshot at time t using neighboring snapshots. | trinity/_output/trinity_reader.py:740 | **none** |
| 913 | method | `TrinityOutput.filter(self, phase=None, t_min=None, t_max=None) -> 'TrinityOutput'` | Filter snapshots by criteria. | trinity/_output/cloudy/trinity_to_cloudy.py:214 | test/test_logging_dedup.py:16, test/test_logging_dedup.py:35, test/test_logging_dedup.py:36, test/test_logging_dedup.py:37 (+2) |
| 949 | method | `TrinityOutput.info(self, verbose=False) -> None` | Print information about the output file. | run.py:199, run.py:200, run.py:201, run.py:202 (+121) | test/test_active_when.py:76, test/test_materialize_runtime.py:131 |
| 980 | method | `TrinityOutput._print_parameters(self) -> None` | Print all parameters with documentation. | trinity/_output/trinity_reader.py:978 | **none** |
| 1035 | method | `TrinityOutput.to_dataframe(self) -> 'pd.DataFrame'` ⛔ | Convert to pandas DataFrame (scalar values only). | **none** | **none** |
| 1068 | func | `read(filepath) -> TrinityOutput` | Open a TRINITY output file (convenience function). | trinity/_output/trinity_reader.py:1095 | **none** |
| 1098 | func | `iter_progress(items, label='Processing') -> Iterator` ⛔ | Iterate over *items* while showing a single-line progress indicator on | **none** | **none** |
| 1133 | func | `find_data_file(base_dir, run_name) -> Optional[Path]` ⛔ | Find the data file for a run, preferring _modified folders and JSONL over JSON. | **none** | **none** |
| 1192 | func | `find_data_path(base_path) -> Path` | Find the data file, preferring JSONL over JSON. | trinity/_output/cloudy/run_loader.py:118, trinity/_output/show_run.py:262, trinity/_output/trinity_reader.py:1327, trinity/_output/trinity_reader.py:1333 (+1) | **none** |
| 1258 | func | `resolve_data_input(data_input, output_dir=None) -> Path` ⛔ | Resolve various data input formats to a data file path. | **none** | **none** |
| 1343 | func | `find_all_simulations(base_dir) -> List[Path]` | Recursively search for all simulation .jsonl files in a directory. | trinity/_analysis/check_yesno.py:65, trinity/_output/trinity_reader.py:1542 | **none** |
| 1388 | func | `parse_simulation_params(folder_name) -> Optional[Dict[str, str]]` | Extract mCloud, sfe, ndens from simulation folder name. | trinity/_output/trinity_reader.py:1437, trinity/_output/trinity_reader.py:1481, trinity/_output/trinity_reader.py:1550 | **none** |
| 1420 | func | `get_unique_ndens(sim_files) -> List[str]` ⛔ | Get list of unique ndens values from simulation files. | **none** | **none** |
| 1443 | func | `organize_simulations_for_grid(sim_files, ndens_filter=None, mCloud_filter=None, sfe_filter=None) -> Dict` ⛔ | Organize simulation files into a grid structure for plotting. | **none** | **none** |
| 1522 | func | `info_simulations(folder_path) -> Dict` ⛔ | Scan a folder and return available simulation parameters. | **none** | **none** |

#### `trinity/bubble_structure/bubble_luminosity.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 120 | func | `_quiet_lsoda_fortran()` | Silence LSODA's Fortran step-underflow chatter ("lsoda-- warning..internal t | trinity/bubble_structure/bubble_luminosity.py:348, trinity/bubble_structure/bubble_luminosity.py:501 | test/test_bubble_lsoda_quiet.py:25, test/test_bubble_lsoda_quiet.py:47, test/test_dR2min_magic_number.py:259 |
| 152 | class | `class BubbleSolverError(Exception)` | Raised when a bubble-structure solve fails or yields an unphysical solution. | trinity/bubble_structure/bubble_luminosity.py:358, trinity/bubble_structure/bubble_luminosity.py:511, trinity/bubble_structure/bubble_luminosity.py:343, trinity/bubble_structure/bubble_luminosity.py:424 (+4) | test/test_bubble_solver_failures.py:26, test/test_bubble_solver_failures.py:56, test/test_bubble_solver_failures.py:76, test/test_residual_resample.py:131 |
| 166 | class | `class BubbleProperties()` | Dataclass containing all bubble properties. | trinity/bubble_structure/bubble_luminosity.py:199, trinity/bubble_structure/bubble_luminosity.py:895, trinity/phase1b_energy_implicit/get_betadelta.py:161, trinity/phase1b_energy_implicit/get_betadelta.py:509 (+5) | test/test_betadelta_hybr.py:25, test/test_betadelta_hybr.py:27, test/test_betadelta_solver.py:31, test/test_betadelta_solver.py:34 |
| 199 | func | `get_bubbleproperties_pure(params) -> BubbleProperties` | Calculate bubble properties and return as a dataclass. | trinity/phase1_energy/run_energy_phase.py:170, trinity/phase1b_energy_implicit/get_betadelta.py:436, trinity/phase1b_energy_implicit/get_betadelta.py:538 | test/test_dR2min_magic_number.py:305, test/test_dR2min_magic_number.py:333, test/test_fA_source_boost.py:145 |
| 297 | func | `_get_init_dMdt(params, Pb) -> float` | Initial guess for dMdt (Equation 33 in Weaver+77). | trinity/bubble_structure/bubble_luminosity.py:246 | **none** |
| 311 | func | `_get_velocity_residuals(dMdt_init, params, Pb, R1) -> float` | Calculate velocity residual for dMdt solver. | trinity/bubble_structure/bubble_luminosity.py:259 | test/test_residual_resample.py:193, test/test_residual_resample.py:176, test/test_residual_resample.py:229 |
| 392 | func | `_get_bubble_ODE_initial_conditions(dMdt, params, Pb, R1)` | Get initial conditions for bubble ODE (Eq 44 in Weaver+77). | trinity/bubble_structure/bubble_luminosity.py:275, trinity/bubble_structure/bubble_luminosity.py:316 | test/test_dR2min_magic_number.py:154, test/test_residual_resample.py:111, test/test_residual_resample.py:224 |
| 414 | func | `_get_bubble_ODE(r_arr, initial_ODEs, params, Pb)` | Bubble structure ODE (Equations 42-43 in Weaver+77). | trinity/bubble_structure/bubble_luminosity.py:342, trinity/bubble_structure/bubble_luminosity.py:495 | test/test_dR2min_magic_number.py:261, test/test_fA_source_boost.py:118, test/test_fA_source_boost.py:99, test/test_residual_resample.py:123 |
| 452 | func | `_solve_bubble_structure(initial_conditions, r_array, params, Pb, rtol=_BUBBLE_RTOL)` | Integrate the bubble-structure ODE and sample it on ``r_array``. | trinity/bubble_structure/bubble_luminosity.py:645 | test/test_bubble_solver_failures.py:30, test/test_bubble_solver_failures.py:41 |
| 531 | func | `_create_radius_grid(R1, r2Prime) -> np.ndarray` | Create the 60k-point radius grid with cleaning. | trinity/bubble_structure/bubble_luminosity.py:638 | **none** |
| 570 | func | `_clean_radius_grid(r_array, min_relative_spacing=MIN_SPACING) -> np.ndarray` | Remove near-duplicate points from a radius grid. | trinity/bubble_structure/bubble_luminosity.py:567 | **none** |
| 625 | func | `_bubble_luminosity(params, R1, Pb, r2Prime, initial_conditions, bubble_r_Tb, bubble_dMdt)` | Compute bubble luminosity on the production radius grid. | trinity/bubble_structure/bubble_luminosity.py:288 | test/test_bubble_solver_failures.py:57, test/test_bubble_solver_failures.py:77 |
| 915 | func | `_get_mass_and_grav(n, r, params)` | Calculate cumulative mass (gravity outputs currently DISABLED). | trinity/bubble_structure/bubble_luminosity.py:891 | **none** |
| 973 | func | `_bubble_diag_enabled()` | True iff the gated bubble-integration diagnostic is requested. | trinity/bubble_structure/bubble_luminosity.py:648 | **none** |
| 978 | func | `_capture_bubble_integration(params, r_array, psoln, infodict, R1, Pb, initial_conditions, bubble_dMdt)` | Save + classify a problematic bubble T-profile (gated diagnostic). | trinity/bubble_structure/bubble_luminosity.py:649 | **none** |
| 1098 | func | `_dump_bubble_state(params, R1, Pb, bubble_dMdt, bubble_r_Tb, r2Prime, initial_conditions, r_array, v_array, T_array, dTdr_array)` | Pickle one bubble-call state for the offline correctness audit (gated). | trinity/bubble_structure/bubble_luminosity.py:680 | **none** |

#### `trinity/bubble_structure/get_bubbleParams.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 27 | func | `delta2dTdt(t, T, delta)` ⛔ | See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf. | **none** | **none** |
| 47 | func | `dTdt2delta(t, T, dTdt)` ⛔ | See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf. | **none** | **none** |
| 69 | func | `cool_beta_to_Ebdot(params)` ⛔ | Convert Weaver cooling parameter beta to dE_b/dt. | **none** | **none** |
| 140 | func | `Ebdot_to_cool_beta(bubble_P, r1, bubble_Edot, my_params)` ⛔ | Inverse of cool_beta_to_Ebdot: convert dE_b/dt to Weaver cooling parameter beta. | **none** | **none** |
| 198 | func | `bubble_E2P(Eb, r2, r1, gamma)` | Convert bubble thermal energy to bubble pressure. | trinity/bubble_structure/bubble_luminosity.py:228, trinity/bubble_structure/get_bubbleParams.py:358, trinity/bubble_structure/get_bubbleParams.py:376, trinity/bubble_structure/get_bubbleParams.py:374 (+3) | test/test_energy_collapse_guard.py:57, test/test_energy_collapse_guard.py:49 |
| 242 | func | `get_leak_luminosity(coverFraction, R2, Pb, c_sound, gamma)` | Geometry-set covering-fraction energy leak (enthalpy flux through the | trinity/phase1_energy/energy_phase_ODEs.py:277, trinity/phase1_energy/energy_phase_ODEs.py:405, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:813 | test/test_cf_leak.py:58, test/test_cf_leak.py:22, test/test_cf_leak.py:27, test/test_cf_leak.py:28 (+3) |
| 286 | func | `pRam(r, Lmech, v_mech)` | Ram pressure from a freely streaming wind: P_ram = L_mech / (2 pi r^2 v_mech). | trinity/bubble_structure/get_bubbleParams.py:351, trinity/bubble_structure/get_bubbleParams.py:359, trinity/main.py:339, trinity/phase1_energy/energy_phase_ODEs.py:254 (+8) | **none** |
| 311 | func | `get_effective_bubble_pressure(current_phase, Eb, R2, R1, gamma, Lmech_total=None, v_mech_total=None, t=None, tSF=None)` | Effective interior pressure felt by the shell. | trinity/phase1_energy/energy_phase_ODEs.py:226, trinity/phase1_energy/energy_phase_ODEs.py:362 | **none** |
| 384 | func | `get_r1(r1, params)` | Root of this equation sets r1 (see Rahners thesis, eq 1.25). | trinity/bubble_structure/get_bubbleParams.py:446 | test/test_r1_bracket.py:52, test/test_r1_bracket.py:31 |
| 414 | func | `solve_R1(R2, Eb, Lmech_total, v_mech_total)` | Solve get_r1 for the inner bubble radius R1 (wind termination shock) [pc]. | trinity/bubble_structure/bubble_luminosity.py:222, trinity/phase1_energy/energy_phase_ODEs.py:223, trinity/phase1_energy/energy_phase_ODEs.py:358, trinity/phase1_energy/run_energy_phase.py:97 (+2) | test/test_energy_collapse_guard.py:68, test/test_energy_collapse_guard.py:69, test/test_energy_collapse_guard.py:70, test/test_r1_bracket.py:33 (+4) |

#### `trinity/cloud_properties/bonnorEbertSphere.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 101 | class | `class LaneEmdenSolution()` | Container for Lane-Emden equation solution. | trinity/cloud_properties/bonnorEbertSphere.py:225, trinity/cloud_properties/bonnorEbertSphere.py:288, trinity/cloud_properties/bonnorEbertSphere.py:305 | **none** |
| 135 | class | `class BESphereResult()` | Result of Bonnor-Ebert sphere creation. | trinity/cloud_properties/bonnorEbertSphere.py:306, trinity/cloud_properties/bonnorEbertSphere.py:501, trinity/cloud_properties/bonnorEbertSphere.py:435 | **none** |
| 179 | func | `lane_emden_ode(y, xi) -> np.ndarray` | Isothermal Lane-Emden equation as first-order ODE system. | trinity/cloud_properties/bonnorEbertSphere.py:255 | **none** |
| 206 | func | `get_initial_conditions(xi0=XI_MIN) -> Tuple[float, float]` | Get accurate initial conditions using series expansion. | trinity/cloud_properties/bonnorEbertSphere.py:248 | **none** |
| 221 | func | `solve_lane_emden(xi_max=XI_MAX, n_points=N_POINTS, xi_min=XI_MIN) -> LaneEmdenSolution` | Solve the isothermal Lane-Emden equation. | trinity/cloud_properties/bonnorEbertSphere.py:540, trinity/cloud_properties/bonnorEbertSphere.py:371, trinity/cloud_properties/validate_gmc.py:654, trinity/phase0_init/get_InitCloudProp.py:321 | test/test_validate_gmc.py:18 |
| 298 | func | `create_BE_sphere(M_cloud, n_core, Omega, mu=1.4, gamma=5.0 / 3.0, validate=True, lane_emden_solution=None) -> BESphereResult` | Create Bonnor-Ebert sphere from user inputs. | trinity/cloud_properties/bonnorEbertSphere.py:543, trinity/cloud_properties/validate_gmc.py:481, trinity/cloud_properties/validate_gmc.py:665, trinity/cloud_properties/validate_gmc.py:674 (+1) | **none** |
| 453 | func | `r_to_xi(r, c_s, rho_core) -> float` | Convert physical radius to dimensionless radius. | trinity/cloud_properties/bonnorEbertSphere.py:617 | **none** |
| 475 | func | `xi_to_r(xi, c_s, rho_core) -> float` | Convert dimensionless radius to physical radius. | trinity/cloud_properties/bonnorEbertSphere.py:654 | **none** |
| 501 | func | `create_BE_sphere_from_params(params) -> BESphereResult` | Create BE sphere from TRINITY params dictionary. | **none** | test/test_mu_audit_drift.py:299 |
| 582 | func | `r2xi(r, params)` | Convert physical radius to dimensionless radius (TRINITY interface). | trinity/cloud_properties/density_profile.py:157 | **none** |
| 622 | func | `xi2r(xi, params)` ⛔ | Convert dimensionless radius to physical radius (TRINITY interface). | **none** | **none** |

#### `trinity/cloud_properties/density_profile.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 34 | func | `_is_scalar(x) -> bool` | Check if input is scalar (not array-like). | trinity/cloud_properties/density_profile.py:105 | **none** |
| 39 | func | `_to_array(x) -> np.ndarray` | Convert input to a 1-D float numpy array. | trinity/cloud_properties/density_profile.py:106 | **none** |
| 44 | func | `_to_output(result, was_scalar)` | Convert result back to scalar if input was scalar. | trinity/cloud_properties/density_profile.py:170 | **none** |
| 55 | func | `get_density_profile(r, params)` | Calculate the number density profile n(r) at given radius/radii. | trinity/cloud_properties/mass_profile.py:113, trinity/phase0_init/get_InitCloudProp.py:285, trinity/phase0_init/get_InitCloudProp.py:360, trinity/phase1_energy/energy_phase_ODEs.py:53 (+4) | **none** |

#### `trinity/cloud_properties/initial_profile.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 44 | class | `class _MockItem()` | Minimal DescribedItem stand-in: just ``.value`` read/write. | trinity/cloud_properties/initial_profile.py:117, trinity/cloud_properties/initial_profile.py:118, trinity/cloud_properties/initial_profile.py:119, trinity/cloud_properties/initial_profile.py:120 (+12) | **none** |
| 58 | func | `build_initial_cloud_profile(*, dens_profile, mCloud, nCore, nISM, rCore, rCloud, densPL_alpha=0.0, mu_convert, densBE_Omega=None, gamma_adia=None, nEdge=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` | Reconstruct ``(r_arr, n_arr, m_arr)`` from run-constant scalars. | trinity/_output/trinity_reader.py:629 | **none** |

#### `trinity/cloud_properties/mass_profile.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 62 | func | `_is_scalar(x) -> bool` | Check if input is scalar (not array-like). | trinity/cloud_properties/mass_profile.py:118, trinity/cloud_properties/mass_profile.py:186, trinity/cloud_properties/mass_profile.py:187 | **none** |
| 67 | func | `_to_array(x) -> np.ndarray` | Convert input to a 1-D float numpy array. | trinity/cloud_properties/mass_profile.py:119, trinity/cloud_properties/mass_profile.py:120, trinity/cloud_properties/mass_profile.py:190, trinity/cloud_properties/mass_profile.py:209 (+2) | **none** |
| 72 | func | `_to_output(result, was_scalar)` | Convert result back to scalar if input was scalar. | trinity/cloud_properties/mass_profile.py:128, trinity/cloud_properties/mass_profile.py:220, trinity/cloud_properties/mass_profile.py:228 | **none** |
| 83 | func | `get_mass_density(r, params) -> ScalarOrArray` | Get mass density rho(r) from number density n(r). | trinity/cloud_properties/mass_profile.py:209, trinity/cloud_properties/mass_profile.py:475 | **none** |
| 131 | func | `get_mass_profile(r, params, return_mdot=False, rdot=None) -> Union[ScalarOrArray, Tuple[ScalarOrArray, ScalarOrArray]]` | Calculate mass profile M(r) and optionally dM/dt. | trinity/cloud_properties/mass_profile.py:525, trinity/phase0_init/get_InitCloudProp.py:286, trinity/phase0_init/get_InitCloudProp.py:361, trinity/phase1_energy/energy_phase_ODEs.py:208 (+10) | **none** |
| 231 | func | `compute_enclosed_mass(r_arr, rho_arr, params) -> np.ndarray` | Compute enclosed mass M(r) = integral[0 to r] 4*pi*r'^2 * rho(r') dr'. | trinity/cloud_properties/mass_profile.py:214 | **none** |
| 267 | func | `compute_enclosed_mass_powerlaw(r_arr, params) -> np.ndarray` | Analytical enclosed mass for power-law profile. | trinity/cloud_properties/mass_profile.py:260 | **none** |
| 347 | func | `compute_enclosed_mass_bonnor_ebert(r_arr, rho_arr, params) -> np.ndarray` | Enclosed mass for Bonnor-Ebert sphere using analytical Lane-Emden formula. | trinity/cloud_properties/mass_profile.py:262 | **none** |
| 437 | func | `compute_mass_accretion_rate(r_arr, rdot_arr, params) -> np.ndarray` ⛔ | Compute mass accretion rate dM/dt = 4*pi*r^2*rho(r)*v(r). | **none** | **none** |
| 488 | func | `validate_mass_at_rCloud(params, tolerance=0.001)` ⛔ | Validate that computed M(rCloud) matches expected mCloud. | **none** | **none** |
| 566 | func | `compute_minimum_rCore(nCore, nISM, rCloud, alpha, margin=1.1)` ⛔ | Compute minimum rCore such that edge density nEdge >= nISM. | **none** | **none** |

#### `trinity/cloud_properties/powerLawSphere.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 51 | func | `compute_rCloud_homogeneous(M_cloud, nCore, mu=1.4)` | Compute cloud radius for homogeneous (α=0) density profile. | trinity/cloud_properties/powerLawSphere.py:126, trinity/cloud_properties/validate_gmc.py:409, trinity/cloud_properties/validate_gmc.py:589, trinity/phase0_init/get_InitCloudProp.py:170 | **none** |
| 77 | func | `compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore=None, rCore_fraction=0.1, mu=1.4)` | Compute cloud radius for power-law density profile. | trinity/cloud_properties/powerLawSphere.py:252, trinity/cloud_properties/validate_gmc.py:416, trinity/cloud_properties/validate_gmc.py:577, trinity/cloud_properties/validate_gmc.py:592 (+4) | **none** |
| 214 | func | `compute_consistent_params(M_cloud, nCore, alpha, rCore_fraction=0.1, mu=1.4, nISM=1.0)` ⛔ | Compute self-consistent cloud parameters from (M_cloud, nCore, alpha). | **none** | **none** |

#### `trinity/cloud_properties/validate_gmc.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 62 | func | `_quiet_loggers(*names, level=logging.INFO)` | Temporarily raise the level of named loggers (restored on exit). | trinity/cloud_properties/validate_gmc.py:577, trinity/cloud_properties/validate_gmc.py:665 | **none** |
| 81 | func | `format_suggestion(s) -> str` | Format a suggestion dict for display, with densities in cm^-3. | run.py:221, trinity/cloud_properties/validate_gmc.py:172 | **none** |
| 120 | class | `class GMCValidationResult()` | Result of GMC parameter validation. | trinity/cloud_properties/validate_gmc.py:457, trinity/cloud_properties/validate_gmc.py:533, trinity/cloud_properties/validate_gmc.py:421, trinity/cloud_properties/validate_gmc.py:491 | **none** |
| 152 | method | `GMCValidationResult.summary(self) -> str` | Format a human-readable summary string. | run.py:749, trinity/_input/sweep_runner.py:642, trinity/_output/cloudy/snapshot_to_deck.py:208, trinity/_output/cloudy/snapshot_to_deck.py:212 | test/test_cloudy_run_loader.py:404, test/test_cloudy_run_loader.py:76, test/test_cloudy_run_loader.py:77, test/test_cloudy_run_loader.py:78 (+1) |
| 181 | func | `check_gmc_constraints(rCloud, nEdge, mCloud, M_computed, nISM=1.0, r_max=R_CLOUD_MAX, mass_tolerance=MASS_TOLERANCE, ndens_to_cgs=None)` | Check the three GMC plausibility constraints on pre-computed values. | trinity/cloud_properties/validate_gmc.py:442, trinity/cloud_properties/validate_gmc.py:510 | **none** |
| 269 | func | `validate_gmc_params(mCloud, nCore, mu, nISM, dens_profile, alpha=None, rCore=None, Omega=None, gamma=5.0 / 3.0, r_max=R_CLOUD_MAX, mass_tolerance=MASS_TOLERANCE, lane_emden_solution=None)` | Validate GMC parameters for physical plausibility. | trinity/_input/sweep_runner.py:146, trinity/cloud_properties/validate_gmc.py:393 | test/test_validate_gmc.py:47, test/test_validate_gmc.py:60, test/test_validate_gmc.py:64 |
| 344 | func | `validate_gmc_from_params(params, r_max=None, mass_tolerance=MASS_TOLERANCE)` | Validate GMC parameters extracted from a TRINITY params dictionary. | run.py:210 | **none** |
| 400 | func | `_validate_powerlaw(mCloud, nCore, mu, nISM, alpha, rCore, r_max, mass_tolerance)` | Validate power-law density profile parameters. | trinity/cloud_properties/validate_gmc.py:327 | **none** |
| 473 | func | `_validate_bonnor_ebert(mCloud, nCore, mu, nISM, Omega, gamma, r_max, mass_tolerance, lane_emden_solution)` | Validate Bonnor-Ebert sphere parameters. | trinity/cloud_properties/validate_gmc.py:332 | **none** |
| 549 | func | `_suggest_powerlaw_alternatives(mCloud, nCore, rCore, alpha, nISM, mu, r_max, mass_tolerance, n_suggestions=3, search_range=0.5)` | Search nearby parameter space for valid power-law GMC configurations. | trinity/cloud_properties/validate_gmc.py:453 | **none** |
| 640 | func | `_suggest_bonnor_ebert_alternatives(mCloud, nCore, mu, nISM, Omega, gamma, r_max, mass_tolerance, lane_emden_solution, n_suggestions=3)` | Search nearby parameter space for valid BE sphere configurations. | trinity/cloud_properties/validate_gmc.py:528 | **none** |

#### `trinity/cooling/CIE/read_coolingcurve.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 25 | func | `get_Lambda(T, cooling_CIE_interpolation, metallicity)` | This function calculates Lambda assuming CIE conditions. | trinity/cooling/net_coolingcurve.py:163, trinity/cooling/net_coolingcurve.py:186 | **none** |

#### `trinity/cooling/net_coolingcurve.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 32 | func | `_noncie_cutoffs(cooling_nonCIE)` | (Tcutoff, Tmin) for the non-CIE temp grid, cached on the cube object. | trinity/cooling/net_coolingcurve.py:121 | test/test_fA_source_boost.py:163, test/test_net_coolingcurve.py:47 |
| 48 | func | `_cie_tcutoff(logT_CIE)` | min(logT_CIE[logT_CIE > 5.5]) for the CIE temp grid, cached by array id. | trinity/cooling/net_coolingcurve.py:122 | **none** |
| 58 | func | `get_dudt(age, ndens, T, phi, params_dict)` | Calculates dudt in cgs, but input and ouput in au.  | trinity/bubble_structure/bubble_luminosity.py:430 | test/test_net_coolingcurve.py:59 |

#### `trinity/cooling/non_CIE/read_cloudy.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 22 | func | `get_coolingStructure(params)` | Time-dependent cooling curve, based on (ndens, temperature, phi) triplets. | trinity/phase1_energy/run_energy_phase.py:125, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:784 | test/test_dR2min_magic_number.py:126, test/test_fA_source_boost.py:47, test/test_net_coolingcurve.py:43, test/test_residual_resample.py:95 |
| 142 | func | `create_cubes(filename, path2cooling)` | This function will take filename and return cooling/heating in the form of cubes. | trinity/cooling/non_CIE/read_cloudy.py:69, trinity/cooling/non_CIE/read_cloudy.py:76, trinity/cooling/non_CIE/read_cloudy.py:78 | **none** |
| 270 | func | `get_filename(age, metallicity, SB99_rotation, path2cooling)` | This function creates the filename appropriate for curent run. | trinity/cooling/non_CIE/read_cloudy.py:63, trinity/cooling/non_CIE/read_cloudy.py:343 | **none** |
| 349 | func | `get_fileage(filename)` | — | trinity/cooling/non_CIE/read_cloudy.py:83, trinity/cooling/non_CIE/read_cloudy.py:84, trinity/cooling/non_CIE/read_cloudy.py:315 | **none** |

#### `trinity/main.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 41 | func | `_check_stop_r_rCloud_interaction(nSnap_rCloud, stop_r, rCloud)` | Decide whether stop_r conflicts with stop_at_rCloud_nSnap. | trinity/main.py:128 | **none** |
| 81 | func | `start_expansion(params)` | This wrapper takes in the parameters and feed them into smaller | run.py:231 | **none** |
| 216 | func | `run_expansion(params)` | Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of | trinity/main.py:180 | **none** |
| 366 | func | `expansion_next(tStart, ODEpar, sps_data_old, sps_f_old, mypath, cloudypath, ii_coll)` ⛔ | — | **none** | **none** |

#### `trinity/phase0_init/get_InitCloudProp.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 51 | class | `class CloudProperties()` | Container for computed cloud properties. | trinity/phase0_init/get_InitCloudProp.py:89, trinity/phase0_init/get_InitCloudProp.py:149, trinity/phase0_init/get_InitCloudProp.py:303, trinity/phase0_init/get_InitCloudProp.py:293 (+3) | **none** |
| 89 | func | `get_InitCloudProp(params) -> CloudProperties` | Initialize cloud properties based on density profile type. | trinity/main.py:122, trinity/phase0_init/get_InitCloudProp.py:582, trinity/phase0_init/get_InitCloudProp.py:610, trinity/phase0_init/get_InitCloudProp.py:639 | **none** |
| 149 | func | `_init_powerlaw_cloud(params) -> CloudProperties` | Initialize power-law density profile cloud. | trinity/cloud_properties/initial_profile.py:133, trinity/phase0_init/get_InitCloudProp.py:128 | **none** |
| 303 | func | `_init_bonnor_ebert_cloud(params) -> CloudProperties` | Initialize Bonnor-Ebert sphere cloud with analytical mass. | trinity/cloud_properties/initial_profile.py:148, trinity/phase0_init/get_InitCloudProp.py:130 | **none** |
| 380 | func | `_validate_params(params) -> None` | Validate input parameters. | trinity/phase0_init/get_InitCloudProp.py:123 | **none** |
| 412 | func | `_create_radius_array(rCloud, rCore, n_inside=1000, n_outside=100) -> np.ndarray` | Create radius array with key radii included exactly. | trinity/phase0_init/get_InitCloudProp.py:282, trinity/phase0_init/get_InitCloudProp.py:357 | **none** |
| 458 | func | `_ensure_be_params_exist(params) -> None` | Ensure BE-specific parameters exist in params dictionary. | trinity/phase0_init/get_InitCloudProp.py:345 | **none** |
| 485 | func | `verify_mass_at_rCloud(props, mCloud) -> float` | Verify that M(rCloud) = mCloud. | trinity/phase0_init/get_InitCloudProp.py:587, trinity/phase0_init/get_InitCloudProp.py:614, trinity/phase0_init/get_InitCloudProp.py:645 | **none** |
| 521 | func | `verify_key_radii_in_array(props) -> bool` | Verify that rCloud and rCore are exactly in the radius array. | trinity/phase0_init/get_InitCloudProp.py:588, trinity/phase0_init/get_InitCloudProp.py:615, trinity/phase0_init/get_InitCloudProp.py:646 | **none** |

#### `trinity/phase0_init/get_InitPhaseParam.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 44 | func | `get_y0(params)` | Obtain initial values for the energy-driven (Weaver) phase by integrating | trinity/main.py:232 | test/test_conventional_units.py:153 |

#### `trinity/phase1_energy/energy_phase_ODEs.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 30 | func | `_scalar(x)` | Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x. | trinity/phase1_energy/energy_phase_ODEs.py:55 | **none** |
| 36 | func | `get_press_ion(r, params)` | Pressure from photoionized part of cloud at radius r. | trinity/phase1_energy/energy_phase_ODEs.py:238, trinity/phase1_energy/energy_phase_ODEs.py:375 | **none** |
| 59 | class | `class ODESnapshot()` | Frozen snapshot of parameters needed for ODE evaluation. | trinity/phase1_energy/energy_phase_ODEs.py:114, trinity/phase1_energy/energy_phase_ODEs.py:137, trinity/phase1_energy/energy_phase_ODEs.py:168, trinity/phase1_energy/energy_phase_ODEs.py:325 (+2) | **none** |
| 114 | func | `create_ODE_snapshot(params, shell_props) -> ODESnapshot` | Create a frozen snapshot of all parameters needed for ODE evaluation. | trinity/phase1_energy/run_energy_phase.py:228, trinity/phase1_energy/run_energy_phase.py:292, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1051, trinity/phase1c_transition/run_transition_phase.py:616 | **none** |
| 168 | func | `get_ODE_Edot_pure(t, y, snapshot, params_for_feedback)` | Pure ODE function for bubble expansion. | trinity/phase1_energy/run_energy_phase.py:297, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:618, trinity/phase1c_transition/run_transition_phase.py:231 | **none** |
| 289 | class | `class ODEResult()` | Result from ODE evaluation, containing values to update params with. | trinity/phase1_energy/energy_phase_ODEs.py:325, trinity/phase1_energy/energy_phase_ODEs.py:409 | **none** |
| 325 | func | `compute_derived_quantities(t, y, snapshot, params_for_feedback) -> ODEResult` | Compute all derived quantities after a successful integration step. | trinity/phase1_energy/run_energy_phase.py:229 | **none** |

#### `trinity/phase1_energy/run_energy_phase.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 62 | func | `run_energy(params)` | Run the energy-driven phase (Phase 1) using adaptive ODE integration. | trinity/main.py:251 | **none** |

#### `trinity/phase1b_energy_implicit/get_betadelta.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 77 | func | `_describe_exc(e) -> str` | Format an exception as 'ClassName: message at file:line' for log warnings. | trinity/phase1b_energy_implicit/get_betadelta.py:438, trinity/phase1b_energy_implicit/get_betadelta.py:540 | **none** |
| 99 | class | `class _MockValue()` | Mimics DescribedItem with a .value attribute. | trinity/phase1b_energy_implicit/get_betadelta.py:130, trinity/phase1b_energy_implicit/get_betadelta.py:131, trinity/phase1b_energy_implicit/get_betadelta.py:134 | **none** |
| 107 | class | `class BubbleParamsView()` | Lightweight view that overrides cool_beta and cool_delta without copying. | trinity/phase1b_energy_implicit/get_betadelta.py:432, trinity/phase1b_energy_implicit/get_betadelta.py:534 | test/test_betadelta_solver.py:261, test/test_betadelta_solver.py:264 |
| 141 | method | `BubbleParamsView.get(self, key, default=None)` | — | run.py:145, run.py:521, run.py:353, run.py:454 (+207) | test/test_betadelta_hybr_stress.py:41, test/test_betadelta_hybr_stress.py:96, test/test_betadelta_hybr_stress.py:117, test/test_bubble_solver_stress.py:47 (+32) |
| 152 | class | `class BetaDeltaResult()` | Container for beta-delta solver results. | trinity/phase1b_energy_implicit/get_betadelta.py:630, trinity/phase1b_energy_implicit/get_betadelta.py:683, trinity/phase1b_energy_implicit/get_betadelta.py:849, trinity/phase1b_energy_implicit/get_betadelta.py:914 (+4) | test/test_betadelta_hybr.py:208, test/test_betadelta_hybr.py:243 |
| 182 | func | `cool_beta_to_Ebdot_pure(beta, Pb, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total) -> float` | Convert Weaver cooling parameter beta to dE_b/dt (pure function version). | trinity/phase1b_energy_implicit/get_betadelta.py:461, trinity/phase1b_energy_implicit/get_betadelta.py:567, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:992 | **none** |
| 272 | func | `delta2dTdt_pure(t, T, delta) -> float` | Convert delta to dT/dt (pure function version). | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:993 | **none** |
| 297 | func | `compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia) -> Tuple[float, float]` | Compute inner radius R1 and bubble pressure Pb. | trinity/phase1b_energy_implicit/get_betadelta.py:454, trinity/phase1b_energy_implicit/get_betadelta.py:564, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:936, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1370 (+3) | test/test_r1_bracket.py:68 |
| 334 | func | `effective_Lloss(mode, fmix, theta_target, Lcool, Lleak, Lmech)` | Effective radiative loss fed CONSISTENTLY to the beta-delta residual, the energy ODE | trinity/phase1b_energy_implicit/get_betadelta.py:371 | test/test_cooling_boost.py:48, test/test_cooling_boost.py:18, test/test_cooling_boost.py:20, test/test_cooling_boost.py:25 (+4) |
| 360 | func | `effective_Lloss_from_params(params, Lcool, Lleak, Lmech)` | Read the ``cooling_boost_*`` knobs off ``params`` and apply :func:`effective_Lloss`. | trinity/phase1_energy/run_energy_phase.py:279, trinity/phase1b_energy_implicit/get_betadelta.py:473, trinity/phase1b_energy_implicit/get_betadelta.py:577, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1243 (+1) | **none** |
| 374 | func | `_usable_dMdt(props) -> Optional[float]` | The solved dMdt from a bubble result, or None when unusable. | trinity/phase1b_energy_implicit/get_betadelta.py:1065, trinity/phase1b_energy_implicit/get_betadelta.py:972, trinity/phase1b_energy_implicit/get_betadelta.py:978, trinity/phase1b_energy_implicit/get_betadelta.py:1084 | test/test_betadelta_solver.py:317, test/test_betadelta_solver.py:318, test/test_betadelta_solver.py:319, test/test_betadelta_solver.py:320 (+1) |
| 393 | func | `get_residual_pure(beta, delta, params, return_bubble_props=False, dMdt_guess=None) -> Tuple[float, float, Optional[BubbleProperties]]` | Calculate residuals for beta and delta without mutating params. | trinity/phase1b_energy_implicit/get_betadelta.py:707, trinity/phase1b_energy_implicit/get_betadelta.py:889, trinity/phase1b_energy_implicit/get_betadelta.py:780, trinity/phase1b_energy_implicit/get_betadelta.py:1079 (+1) | **none** |
| 501 | class | `class ResidualDetails()` | Detailed residual components for diagnostics. | trinity/phase1b_energy_implicit/get_betadelta.py:519, trinity/phase1b_energy_implicit/get_betadelta.py:595, trinity/phase1b_energy_implicit/get_betadelta.py:541 | test/test_betadelta_hybr.py:74 |
| 514 | func | `get_residual_detailed(beta, delta, params, bubble_props=None) -> ResidualDetails` | Calculate residuals with all raw components for diagnostics. | trinity/phase1b_energy_implicit/get_betadelta.py:841, trinity/phase1b_energy_implicit/get_betadelta.py:902 | test/test_betadelta_solver.py:245 |
| 612 | func | `_get_betadelta_solver(params) -> str` | The configured beta-delta solver (production default 'hybr'). | trinity/phase1b_energy_implicit/get_betadelta.py:637 | test/test_betadelta_solver_switch.py:56, test/test_betadelta_solver_switch.py:61, test/test_betadelta_solver_switch.py:66 |
| 625 | func | `solve_betadelta_pure(beta_guess, delta_guess, params, method='grid') -> BetaDeltaResult` | Dispatch to the configured beta-delta solver (``betadelta_solver``). | trinity/phase1b_energy_implicit/get_betadelta.py:1178, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:826 | test/test_betadelta_hybr.py:213, test/test_betadelta_hybr.py:229, test/test_betadelta_hybr.py:248, test/test_betadelta_solver.py:220 (+5) |
| 649 | func | `_rescue_structure_failure(result, beta_guess, delta_guess, params, method)` | Re-seed hybr from the bounded legacy grid when the search wandered out of domain. | trinity/phase1b_energy_implicit/get_betadelta.py:643 | **none** |
| 678 | func | `_solve_betadelta_legacy(beta_guess, delta_guess, params, method='grid') -> BetaDeltaResult` | Solve for optimal beta and delta. | trinity/phase1b_energy_implicit/get_betadelta.py:664, trinity/phase1b_energy_implicit/get_betadelta.py:639 | **none** |
| 869 | class | `class _NoPhysicalRoot(BaseException)` | Raised inside the hybr search when the physical acceptance gate | trinity/phase1b_energy_implicit/get_betadelta.py:962, trinity/phase1b_energy_implicit/get_betadelta.py:986, trinity/phase1b_energy_implicit/get_betadelta.py:894, trinity/phase1b_energy_implicit/get_betadelta.py:899 (+1) | **none** |
| 879 | func | `_hybr_g_residual(beta, delta, params, dMdt_seed)` | The pole-free g residual vector (gE, gT) at (beta, delta), with the | trinity/phase1b_energy_implicit/get_betadelta.py:961, trinity/phase1b_energy_implicit/get_betadelta.py:977, trinity/phase1b_energy_implicit/get_betadelta.py:997 | **none** |
| 909 | func | `_hybr_result(beta, delta, det, g_total, converged, iterations)` | Build a BetaDeltaResult from a hybr-accepted point. The f-metric | trinity/phase1b_energy_implicit/get_betadelta.py:1007, trinity/phase1b_energy_implicit/get_betadelta.py:969 | **none** |
| 932 | func | `_no_root_result(beta_guess, delta_guess, reason)` | The no-physical-root BetaDeltaResult; the runner hands off on the flag. | trinity/phase1b_energy_implicit/get_betadelta.py:963, trinity/phase1b_energy_implicit/get_betadelta.py:987, trinity/phase1b_energy_implicit/get_betadelta.py:999 | **none** |
| 948 | func | `_solve_betadelta_hybr(beta_guess, delta_guess, params, method='grid')` | Unbounded scipy hybr root-finder on the pole-free g residual, gated on | trinity/phase1b_energy_implicit/get_betadelta.py:667, trinity/phase1b_energy_implicit/get_betadelta.py:641 | test/test_betadelta_hybr.py:91 |
| 1010 | func | `_solve_grid(beta_guess, delta_guess, params, input_residual=None, input_props=None) -> Tuple[float, float, Optional[BubbleProperties], float, int]` | Grid search solver using BubbleParamsView (no deepcopy). | trinity/phase1b_energy_implicit/get_betadelta.py:740 | test/test_betadelta_solver.py:130, test/test_betadelta_solver.py:144, test/test_betadelta_solver.py:158, test/test_betadelta_solver.py:171 (+9) |
| 1108 | func | `_solve_lbfgsb(beta_guess, delta_guess, params) -> Tuple[float, float, int]` | L-BFGS-B optimizer solver. | trinity/phase1b_energy_implicit/get_betadelta.py:777 | **none** |
| 1152 | func | `get_beta_delta_wrapper_pure(beta_guess, delta_guess, params) -> Tuple[Tuple[float, float], BetaDeltaResult]` ⛔ | Wrapper that matches the interface of the original get_beta_delta_wrapper. | **none** | **none** |

#### `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 184 | func | `classify_energy_collapse(Eb)` | Routing decision when the energy-driven bubble's Eb stops being viable. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1148 | test/test_energy_collapse_guard.py:91, test/test_energy_collapse_guard.py:92, test/test_energy_collapse_guard.py:97, test/test_energy_collapse_guard.py:98 (+3) |
| 212 | func | `_inflow_frac_thickness(v_arr, r_arr) -> float` | Radial-thickness fraction of the bubble with inward (v<0) flow. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:898 | **none** |
| 233 | func | `evaluate_r1_shadow(R2, rCloud, edot_balance, k_blowout=1.0)` | R1 transition criteria in SHADOW mode (computed/logged, never drives the switch). | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1263 | test/test_r1_shadow.py:20, test/test_r1_shadow.py:24, test/test_r1_shadow.py:28, test/test_r1_shadow.py:29 (+8) |
| 252 | func | `parse_transition_triggers(transition_trigger)` | Parse the `transition_trigger` param into a SET of active criteria. | trinity/phase1_energy/run_energy_phase.py:275, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:728 | test/test_r1_shadow.py:82, test/test_r1_shadow.py:87, test/test_r1_shadow.py:93, test/test_r1_shadow.py:56 (+6) |
| 275 | func | `r1_transition_decision(active_triggers, blowout_fired, ebpeak_fired)` | Which R1 criterion (if any) should DRIVE the energy->momentum transition, | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1288 | test/test_r1_shadow.py:83, test/test_r1_shadow.py:88, test/test_r1_shadow.py:89, test/test_r1_shadow.py:94 (+2) |
| 289 | func | `compute_max_dex_change(params_before, params_after, keys) -> float` | Compute the maximum dex (log10) change across monitored parameters. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1199 | **none** |
| 332 | func | `update_unconverged_streak(streak, converged, t_now, total_residual) -> int` | Consecutive-unconverged counter for the beta-delta solver. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:912 | test/test_betadelta_dt_mitigation.py:24, test/test_betadelta_dt_mitigation.py:25, test/test_betadelta_dt_mitigation.py:36, test/test_betadelta_dt_mitigation.py:42 (+3) |
| 360 | func | `betadelta_phase_summary(solve_count, converged_count, no_root_count) -> tuple` | End-of-phase beta-delta solver summary: ``(clean, message)``. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1427 | test/test_betadelta_dt_mitigation.py:129, test/test_betadelta_dt_mitigation.py:136, test/test_betadelta_dt_mitigation.py:143, test/test_betadelta_dt_mitigation.py:150 |
| 376 | func | `next_dt_segment(dt_segment, max_dex_change, unconverged_streak) -> float` | Adaptive dt_segment update plus the beta-delta non-convergence guard. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1201 | test/test_betadelta_dt_mitigation.py:55, test/test_betadelta_dt_mitigation.py:60, test/test_betadelta_dt_mitigation.py:66, test/test_betadelta_dt_mitigation.py:71 (+5) |
| 412 | func | `get_monitor_values(params) -> dict` | Extract current values of monitored parameters. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1054, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1198 | **none** |
| 444 | class | `class ForceProperties()` | Container for force calculations (pure function output). | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:466, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:549 | **none** |
| 460 | func | `compute_forces_pure(R2, mShell, Pb, shell_props, params) -> ForceProperties` | Compute all force components without mutating params. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:995, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1384 | **none** |
| 569 | class | `class ImplicitPhaseResults()` | Container for implicit phase results. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:631, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1450, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:680 | **none** |
| 586 | func | `get_ODE_implicit_pure(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta) -> np.ndarray` | Pure ODE function for implicit phase. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1067 | **none** |
| 631 | func | `run_phase_energy(params) -> ImplicitPhaseResults` | Run the implicit energy phase using solve_ivp. | trinity/main.py:286 | **none** |

#### `trinity/phase1c_transition/run_transition_phase.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 143 | func | `compute_max_dex_change(params_before, params_after, keys) -> float` | Compute the maximum dex (log10) change across monitored parameters. | trinity/phase1c_transition/run_transition_phase.py:710 | **none** |
| 164 | func | `get_monitor_values(params) -> dict` | Extract current values of monitored parameters. | trinity/phase1c_transition/run_transition_phase.py:619, trinity/phase1c_transition/run_transition_phase.py:709 | **none** |
| 184 | class | `class TransitionPhaseResults()` | Container for transition phase results. | trinity/phase1c_transition/run_transition_phase.py:367, trinity/phase1c_transition/run_transition_phase.py:879, trinity/phase1c_transition/run_transition_phase.py:417 | **none** |
| 198 | func | `get_ODE_transition_pure(t, y, snapshot, params_for_feedback, c_sound) -> np.ndarray` | Pure ODE function for transition phase. | trinity/phase1c_transition/run_transition_phase.py:628 | **none** |
| 255 | class | `class ForceProperties()` | Container for force calculations (pure function output). | trinity/phase1c_transition/run_transition_phase.py:277, trinity/phase1c_transition/run_transition_phase.py:348 | **none** |
| 271 | func | `compute_forces_pure(R2, mShell, Pb, shell_props, params) -> ForceProperties` | Compute all force components without mutating params. | trinity/phase1c_transition/run_transition_phase.py:571, trinity/phase1c_transition/run_transition_phase.py:850 | **none** |
| 367 | func | `run_phase_transition(params) -> TransitionPhaseResults` | Run the transition phase using solve_ivp. | trinity/main.py:306 | **none** |

#### `trinity/phase2_momentum/run_momentum_phase.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 135 | func | `compute_max_dex_change(params_before, params_after, keys) -> float` | Compute the maximum dex (log10) change across monitored parameters. | trinity/phase2_momentum/run_momentum_phase.py:789 | **none** |
| 156 | func | `get_monitor_values(params) -> dict` | Extract current values of monitored parameters. | trinity/phase2_momentum/run_momentum_phase.py:701, trinity/phase2_momentum/run_momentum_phase.py:788 | **none** |
| 176 | class | `class MomentumPhaseResults()` | Container for momentum phase results. | trinity/phase2_momentum/run_momentum_phase.py:461, trinity/phase2_momentum/run_momentum_phase.py:925, trinity/phase2_momentum/run_momentum_phase.py:498 | **none** |
| 190 | class | `class ForceProperties()` | Container for force calculations (pure function output). | trinity/phase2_momentum/run_momentum_phase.py:213, trinity/phase2_momentum/run_momentum_phase.py:282 | **none** |
| 206 | func | `compute_forces_momentum_pure(R2, mShell, Lmech_total, v_mech_total, shell_props, params) -> ForceProperties` | Compute all force components for momentum phase without mutating params. | trinity/phase2_momentum/run_momentum_phase.py:647 | **none** |
| 302 | class | `class MomentumODESnapshot()` | Frozen snapshot of parameters for momentum phase ODE. | trinity/phase2_momentum/run_momentum_phase.py:325, trinity/phase2_momentum/run_momentum_phase.py:347, trinity/phase2_momentum/run_momentum_phase.py:373 | **none** |
| 324 | func | `create_momentum_snapshot(params, shell_props, mShell, mShell_dot) -> MomentumODESnapshot` | Create a frozen snapshot of parameters for ODE integration. | trinity/phase2_momentum/run_momentum_phase.py:698 | **none** |
| 373 | func | `get_ODE_momentum_pure(t, y, snapshot, params) -> np.ndarray` | Pure ODE function for momentum phase. | trinity/phase2_momentum/run_momentum_phase.py:710 | **none** |
| 461 | func | `run_phase_momentum(params) -> MomentumPhaseResults` | Run the momentum-driven phase using solve_ivp. | trinity/main.py:346 | **none** |

#### `trinity/phase_general/phase_events.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 82 | class | `class EventResult()` | Container for event detection results. | trinity/phase_general/phase_events.py:363, trinity/phase_general/phase_events.py:407, trinity/phase_general/phase_events.py:588, trinity/phase_general/phase_events.py:380 (+1) | **none** |
| 99 | func | `make_min_radius_event(min_r, name='min_radius')` | Create event that triggers when R2 falls below min_r. | trinity/phase_general/phase_events.py:449, trinity/phase_general/phase_events.py:489, trinity/phase_general/phase_events.py:533, trinity/phase_general/phase_events.py:571 | test/test_phase_events.py:121, test/test_phase_events.py:24 |
| 134 | func | `make_max_radius_event(max_r, name='max_radius')` | Create event that triggers when R2 exceeds max_r. | trinity/phase_general/phase_events.py:495, trinity/phase_general/phase_events.py:539, trinity/phase_general/phase_events.py:577 | test/test_phase_events.py:25 |
| 166 | func | `make_velocity_runaway_event(v_max=MAX_VELOCITY_COLLAPSE, direction='collapse', name='velocity_runaway')` | Create event that triggers on extreme velocity magnitude. | trinity/phase_general/phase_events.py:450, trinity/phase_general/phase_events.py:490, trinity/phase_general/phase_events.py:534, trinity/phase_general/phase_events.py:572 | test/test_phase_events.py:27, test/test_phase_events.py:36, test/test_phase_events.py:45 |
| 220 | func | `make_cloud_boundary_event(rCloud, name='cloud_boundary')` | Create event that triggers when R2 reaches cloud edge. | trinity/phase_general/phase_events.py:448 | test/test_phase_events.py:122, test/test_phase_events.py:54 |
| 252 | func | `make_energy_floor_event(energy_floor, y_index=2, name='energy_floor')` | Create event that triggers when bubble energy falls below threshold. | trinity/phase_general/phase_events.py:532 | test/test_phase_events.py:63 |
| 287 | func | `make_velocity_sign_event(y_index=1, name='velocity_sign')` | Create event that triggers when velocity changes sign. | trinity/phase_general/phase_events.py:488 | test/test_phase_events.py:72 |
| 319 | func | `make_cooling_balance_event(threshold=0.05, name='cooling_balance')` | Create event factory for cooling balance detection. | trinity/phase_general/phase_events.py:497 | test/test_phase_events.py:18 |
| 363 | func | `check_event_termination(sol, events) -> EventResult` | Check solve_ivp solution for event termination. | trinity/phase1_energy/run_energy_phase.py:324, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1095, trinity/phase1c_transition/run_transition_phase.py:653, trinity/phase2_momentum/run_momentum_phase.py:735 | test/test_phase_events.py:128, test/test_phase_events.py:156 |
| 423 | func | `build_energy_phase_events(params) -> List[Callable]` | Build event list for energy phase. | trinity/phase1_energy/run_energy_phase.py:118 | **none** |
| 458 | func | `build_implicit_phase_events(params) -> Tuple[List[Callable], Callable]` | Build event list for implicit (cooling) phase. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:752 | **none** |
| 504 | func | `build_transition_phase_events(params, energy_floor=1000.0) -> List[Callable]` | Build event list for transition phase. | trinity/phase1c_transition/run_transition_phase.py:457 | **none** |
| 546 | func | `build_momentum_phase_events(params) -> List[Callable]` | Build event list for momentum phase. | trinity/phase2_momentum/run_momentum_phase.py:538 | **none** |
| 588 | func | `apply_event_result(params, result, t, y, state_keys=['R2', 'v2']) -> None` | Apply event result to params dictionary. | trinity/phase1_energy/run_energy_phase.py:327, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1117, trinity/phase1c_transition/run_transition_phase.py:669, trinity/phase2_momentum/run_momentum_phase.py:749 | test/test_phase_events.py:143, test/test_phase_events.py:164 |

#### `trinity/shell_structure/get_shellODE.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 37 | func | `get_shellODE(y, r, f_cover, is_ionised, params)` | A function that returns ODE of the ionised number density (n),  | trinity/shell_structure/shell_structure.py:166, trinity/shell_structure/shell_structure.py:325 | test/test_mu_audit_drift.py:191, test/test_mu_audit_drift.py:224, test/test_shell_overflow_guard.py:37, test/test_shell_overflow_guard.py:55 |

#### `trinity/shell_structure/shell_structure.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 39 | class | `class ShellProperties()` | Dataclass containing all shell structure properties. | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:464, trinity/phase1c_transition/run_transition_phase.py:275, trinity/phase2_momentum/run_momentum_phase.py:211, trinity/phase2_momentum/run_momentum_phase.py:324 (+2) | **none** |
| 85 | func | `shell_structure_pure(params) -> ShellProperties` | Evaluate shell structure and return properties as a dataclass. | trinity/phase1_energy/run_energy_phase.py:207, trinity/phase1_energy/run_energy_phase.py:400, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:975, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1374 (+4) | **none** |

#### `trinity/sps/read_sps.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 38 | func | `read_sps(f_mass, params)` | Read and process SPS stellar feedback data. | trinity/main.py:147 | test/test_read_sps.py:73 |
| 134 | func | `_read_sps_user(filepath, f_mass, params, column_map)` | SPS loader driven by a canonical -> ColumnSpec map. | trinity/sps/read_sps.py:131 | **none** |
| 285 | func | `get_interpolation(sps, ftype='cubic')` | Create cubic interpolation functions for SPS feedback data. | trinity/main.py:148 | **none** |

#### `trinity/sps/sps_columns.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 37 | class | `class ColumnSpec()` | Description of one column in an SPS file. | trinity/sps/sps_columns.py:213, trinity/sps/sps_columns.py:166, trinity/sps/sps_columns.py:167, trinity/sps/sps_columns.py:168 (+10) | **none** |
| 54 | class | `class CanonicalSpec()` | Metadata describing a canonical column. | trinity/sps/sps_columns.py:65, trinity/sps/sps_columns.py:66, trinity/sps/sps_columns.py:67, trinity/sps/sps_columns.py:68 (+10) | **none** |
| 180 | func | `convert_to_canonical_au(arr, canonical, declared_units, log)` | Convert a raw column to canonical AU units. | trinity/sps/read_sps.py:169 | test/test_conventional_units.py:69 |
| 213 | func | `parse_sps_col_value(canonical, raw_value) -> ColumnSpec` | Parse a single sps_col_<canonical> .param value into a ColumnSpec. | trinity/sps/sps_columns.py:274 | **none** |
| 254 | func | `build_user_column_map(params) -> Dict[str, ColumnSpec]` | Walk all sps_col_<canonical> entries in params and assemble the user's | trinity/_input/registry.py:298 | **none** |
| 278 | func | `validate_user_column_map(column_map, sps_path) -> None` | Strict validation for user-mode column maps. Raises ValueError with a | trinity/_input/registry.py:299 | **none** |
| 314 | func | `_format_missing_template(*, sps_path, missing_required, fi_ok, Li_Ln_partial, sn_input_ok, given) -> str` | One-line error: what's expected, what's missing, what was declared. | trinity/sps/sps_columns.py:304 | **none** |
| 334 | func | `validate_t_monotonic(t, filepath) -> None` | Validate that the time array is strictly increasing. | trinity/sps/read_sps.py:186 | **none** |
| 375 | func | `_can_parse_float(s) -> bool` | True iff s parses as a float (covers integers, decimals, scientific | trinity/sps/sps_columns.py:414, trinity/sps/sps_columns.py:438 | **none** |
| 385 | func | `_scan_layout(filepath)` | Scan filepath to determine (data_start, header_names, delimiter). | trinity/sps/sps_columns.py:461 | **none** |
| 445 | func | `load_user_columns(filepath, column_map) -> Dict[str, np.ndarray]` | Load a user-defined SPS file (any layout) and return a dict keyed by | trinity/sps/read_sps.py:159 | **none** |

#### `trinity/sps/update_feedback.py`

| line | kind | signature | one-line description | callers in `trinity/` | callers in `test/` |
|---|---|---|---|---|---|
| 21 | class | `class SPSFeedback()` | Container for SPS stellar feedback parameters at a single time. | trinity/sps/update_feedback.py:98, trinity/sps/update_feedback.py:188 | **none** |
| 98 | func | `get_current_sps_feedback(t, params) -> SPSFeedback` | Get stellar feedback parameters at time t from the SPS interpolators. | trinity/phase1_energy/energy_phase_ODEs.py:195, trinity/phase1_energy/energy_phase_ODEs.py:334, trinity/phase1_energy/run_energy_phase.py:93, trinity/phase1_energy/run_energy_phase.py:157 (+10) | **none** |

---

## 3. The shared state object

### 3.1 What it is

`params` is a `DescribedDict` (`trinity/_input/dictionary.py:200`), a `dict` subclass whose values
must be `DescribedItem` (`dictionary.py:98`) — enforced by `DescribedDict.__setitem__`
(`dictionary.py:242`), which raises `TypeError` on a bare value. Access is always
`params['key'].value`; `DescribedItem` also implements `__float__`, `__format__`, `__add__`,
`__eq__`, `__array__`, … (`dictionary.py:143-194`) so `params['x'] * 2` and `f"{params['x']:.2e}"`
work without `.value`.

Each `DescribedItem` carries `__slots__ = ("_value", "info", "ori_units",
"exclude_from_snapshot")` (`dictionary.py:118`).

Instance-level (per-`DescribedDict`, not module-level) mutable state, set in
`DescribedDict.__init__` (`dictionary.py:215-240`):

| attribute | line | purpose |
|---|---|---|
| `save_count` | `:219` | snapshots saved so far; also the next snapshot id |
| `snapshot_interval` | `:220` | flush cadence, hard-coded `10` |
| `previous_snapshot` | `:221` | pending, unflushed snapshots `{str(id): dict}` |
| `flush_count` | `:222` | number of `flush()` calls; `0` triggers the fresh-run file delete + `metadata.json` write |
| `_excluded_keys` | `:225` | set of keys omitted from snapshots |
| `_termination_reason` | `:232` | atexit reason override, set by `set_termination_reason` (`:344`) |
| `_impl_r2_logged` | `:237` | per-snapshot counter for `simplify()` R² debug lines |

### 3.2 How it is constructed

`read_param.read_param` (`read_param.py:43`) is the only constructor of a populated `params` in the
run path (`run.py:160`). The 10 steps are tabulated in §1.2. The parameter schema itself is the
`SPECS` tuple in `trinity/_input/registry.py:328-534` — **201 `ParamSpec` entries**
(`param_spec.py:84`), each with `name, default, info, category, unit, exclude_from_snapshot,
metadata_exclude, run_const, validator, resolver, active_when, consumed_by`. `REGISTRY`
(`registry.py:536`) is an `OrderedDict` view of the same tuple.

Categories used (`param_spec.py:33-55`): `input_admin`, `input_physical`, `input_profile`,
`input_termination`, `input_sps`, `input_constants`, `input_solver`, `input_cooling`,
`derived_init`, `runtime_control`, `runtime_time`, `runtime_radii`, `runtime_bubble`,
`runtime_bubble_cooling`, `runtime_pressure`, `runtime_force`, `runtime_shell`,
`runtime_feedback`, `runtime_residuals`, `runtime_cloud_profile`, `runtime_loaded`.

Two families of derived facts are projected off the same specs:
`registry.run_const_keys()` (`registry.py:664`) → `run_constants.RUN_CONST_KEYS`
(`run_constants.py:77`), and `registry.metadata_exclude_keys()` (`registry.py:673`) →
`run_constants.METADATA_EXCLUDE` (`run_constants.py:83`).

### 3.3 Two write channels

1. **Direct** — `params['key'].value = …` (or `params['key'] = DescribedItem(…)` in
   `read_param` steps 6/7 and `registry.apply_active_when` / `materialize_runtime`).
2. **Bulk via `updateDict`** (`dictionary.py:1232`) — given a dataclass instance it iterates
   `dataclasses.fields(...)` and does `dictionary[field.name].value = value` **only for fields
   already present in the dict** (silently skipping the rest, `dictionary.py:1264-1269`). The
   three dataclasses routed this way, and the 19 call sites:

   | dataclass | defined | fields written | `updateDict` sites |
   |---|---|---|---|
   | `SPSFeedback` | `sps/update_feedback.py:21` | `t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total` | `run_energy_phase.py:94,158,392`; `run_energy_implicit_phase.py:804,1368`; `run_transition_phase.py:497,835`; `run_momentum_phase.py:578,888` |
   | `BubbleProperties` | `bubble_structure/bubble_luminosity.py:166` | `bubble_LTotal, bubble_T_r_Tb, bubble_Tavg, bubble_mass, bubble_L1Bubble, bubble_L2Conduction, bubble_L3Intermediate, bubble_v_arr, bubble_T_arr, bubble_dTdr_arr, bubble_r_arr, bubble_n_arr, bubble_dMdt, R1, Pb, bubble_r_Tb` | `run_energy_phase.py:184`; `run_energy_implicit_phase.py:894` |
   | `ShellProperties` | `shell_structure/shell_structure.py:39` | `shell_n0, rShell, shell_thickness, shell_fAbsorbedIon, shell_fAbsorbedNeu, shell_fAbsorbedWeightedTotal, shell_fIonisedDust, shell_nMax, shell_tauKappaRatio, shell_grav_r, shell_grav_phi, shell_grav_force_m, isDissolved, is_phiDepleted, diss_condition_met, n_IF, n_IF_ODE, R_IF, n_IF_Str, shell_r_arr, shell_n_arr, shell_ion_idx` | `run_energy_phase.py:208,401`; `run_energy_implicit_phase.py:976,1375`; `run_transition_phase.py:559,842`; `run_momentum_phase.py:629,893` |

   `SPSFeedback.t` and `ShellProperties.diss_condition_met` have **no matching key** in `SPECS`, so
   `updateDict` skips them (the dataclass field is discarded at the boundary).

### 3.4 Indirect read channels (not visible as `params['key']` literals)

These matter when reading the write/read table below, because a key can appear "written but never
read" while still being consumed:

| channel | site | what it reads |
|---|---|---|
| adaptive-step monitoring | `run_energy_implicit_phase.get_monitor_values` (`:412`), `run_transition_phase.get_monitor_values` (`:164`), `run_momentum_phase.get_monitor_values` (`:156`) | `params.get(key)` for each of the 35 names in that module's `ADAPTIVE_MONITOR_KEYS` list (`run_energy_implicit_phase.py:150`, `run_transition_phase.py:112`, `run_momentum_phase.py:104` — the three lists are textually identical) |
| snapshot serialization | `DescribedDict._clean_for_snapshot` (`dictionary.py:577`) | iterates `self.items()`; every non-excluded, non-run-const key is read once per `save_snapshot()` |
| metadata write | `DescribedDict.flush` (`dictionary.py:828-849`) | iterates `RUN_CONST_KEYS` |
| state display | `terminal_prints._phys` (`:143`) via `_STATE_FIELDS` (`:131`) | `t_now, R2, v2, Eb, Pb, T0, R1, shell_mass` |
| end-of-run report | `simulation_end.CRITICAL_PARAMS` (`simulation_end.py:409`) | a list of scalar keys compared between the last two snapshots |

### 3.5 Key write/read table

Columns:

* **in SPECS** — whether a `ParamSpec` with this name exists in `registry.py`.
* **direct writes (`trinity/`)** — `file:line` of every `params['key'].value = …` /
  `params['key'] = …` in `trinity/` or `run.py`.
* **dataclass writes** — the `updateDict` channel from §3.3 (sites listed in that table).
* **reads (`trinity/`)** — `file:line` of every other subscript / `.get()` / `in params` access.
  **Does not include** the indirect channels of §3.4.
* **test refs** — any access from `test/`.

Lists longer than the shown cap are truncated with `(+N)`.

Every entry that resolves through `read_param` Step 4 (the `default.param` merge,
`read_param.py:270`) is written there even when its "direct writes" cell is `—`; that single
generic site is not repeated per key.

**Caveat on this table only.** Unlike §2, the key scan is *textual* (line-regex over
`X['key']` / `X.get('key')` / `'key' in X` with `X` in a fixed receiver-name set), because keys are
strings rather than AST-resolvable names. Consequences: (i) occurrences inside docstrings and
comments **are** counted (e.g. `sps/read_sps.py:42-43`, `_input/dictionary.py:105-106`,
`sps/sps_columns.py:17` are docstring lines); (ii) the module `__main__` demo block in
`_input/dictionary.py:1284-1342` contributes write sites for `t_now`, `R2`, `path2output`,
`small_arr`, `large_arr`; (iii) dynamically-built key names (the `sps_col_*` family) are invisible.
Verify any individual cell against source before relying on it.
| key | in SPECS | direct writes (trinity/) | dataclass writes | reads (trinity/) | test refs |
|---|---|---|---|---|---|
| `C_thermal` | y | — | — | trinity/bubble_structure/bubble_luminosity.py:304, trinity/bubble_structure/bubble_luminosity.py:398, trinity/bubble_structure/bubble_luminosity.py:441 | test/test_fA_source_boost.py:124 |
| `EarlyPhaseApproximation` | y | trinity/phase1_energy/run_energy_phase.py:343 | — | trinity/phase1_energy/energy_phase_ODEs.py:159, trinity/phase1_energy/run_energy_phase.py:342 | — |
| `Eb` | y | trinity/main.py:237, trinity/phase1_energy/run_energy_phase.py:151, trinity/phase1_energy/run_energy_phase.py:357, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:796, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1137, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1169 … (+4) | — | trinity/bubble_structure/bubble_luminosity.py:223, trinity/bubble_structure/bubble_luminosity.py:229, trinity/bubble_structure/bubble_luminosity.py:1029, trinity/bubble_structure/get_bubbleParams.py:118, trinity/bubble_structure/get_bubbleParams.py:174, trinity/main.py:332 … (+7) | — |
| `EndSimulationDirectly` | y | trinity/main.py:266, trinity/phase1_energy/run_energy_phase.py:172, trinity/phase1_energy/run_energy_phase.py:369, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:679, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:774, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1044 … (+19) | — | trinity/main.py:283, trinity/main.py:303, trinity/main.py:343 | test/test_phase_events.py:149 |
| `FB_mColdSNFrac` | y | — | — | trinity/sps/read_sps.py:234, trinity/sps/read_sps.py:238 | — |
| `FB_mColdWindFrac` | y | — | — | trinity/sps/read_sps.py:216, trinity/sps/read_sps.py:219 | — |
| `FB_thermCoeffSN` | y | — | — | trinity/sps/read_sps.py:237 | — |
| `FB_thermCoeffWind` | y | — | — | trinity/sps/read_sps.py:218 | — |
| `FB_vSN` | y | — | — | trinity/sps/read_sps.py:99, trinity/sps/read_sps.py:149, trinity/sps/read_sps.py:228 | — |
| `F_HII` | y | trinity/phase1_energy/run_energy_phase.py:219, trinity/phase1_energy/run_energy_phase.py:237, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:986, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:998, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1383, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1388 … (+6) | — | — | — |
| `F_ISM` | y | — | — | — | — |
| `F_grav` | y | trinity/phase1_energy/run_energy_phase.py:233, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:996, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1386, trinity/phase1c_transition/run_transition_phase.py:572, trinity/phase1c_transition/run_transition_phase.py:852, trinity/phase2_momentum/run_momentum_phase.py:650 | — | — | — |
| `F_ion_in` | y | trinity/phase1_energy/run_energy_phase.py:235, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:997, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1387, trinity/phase1c_transition/run_transition_phase.py:573, trinity/phase1c_transition/run_transition_phase.py:853, trinity/phase2_momentum/run_momentum_phase.py:651 | — | — | — |
| `F_rad` | y | trinity/phase1_energy/run_energy_phase.py:241, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1000, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1390, trinity/phase1c_transition/run_transition_phase.py:576, trinity/phase1c_transition/run_transition_phase.py:856, trinity/phase2_momentum/run_momentum_phase.py:654 | — | — | — |
| `F_ram` | y | trinity/phase1_energy/run_energy_phase.py:239, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:999, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1389, trinity/phase1c_transition/run_transition_phase.py:575, trinity/phase1c_transition/run_transition_phase.py:855, trinity/phase2_momentum/run_momentum_phase.py:653 | — | — | — |
| `F_ram_SN` | y | trinity/phase1_energy/run_energy_phase.py:257, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1009, trinity/phase1c_transition/run_transition_phase.py:585, trinity/phase2_momentum/run_momentum_phase.py:663 | — | — | — |
| `F_ram_wind` | y | trinity/phase1_energy/run_energy_phase.py:256, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1008, trinity/phase1c_transition/run_transition_phase.py:584, trinity/phase2_momentum/run_momentum_phase.py:662 | — | — | — |
| `G` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:150, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:489, trinity/phase1c_transition/run_transition_phase.py:282, trinity/phase2_momentum/run_momentum_phase.py:220, trinity/phase2_momentum/run_momentum_phase.py:348, trinity/shell_structure/shell_structure.py:266 … (+3) | — |
| `Lbol` | y | — | updateDict/SPSFeedback | trinity/phase1_energy/energy_phase_ODEs.py:134, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:546, trinity/phase1c_transition/run_transition_phase.py:345, trinity/phase2_momentum/run_momentum_phase.py:279, trinity/phase2_momentum/run_momentum_phase.py:344 | — |
| `Li` | y | — | updateDict/SPSFeedback | trinity/shell_structure/get_shellODE.py:89, trinity/shell_structure/shell_structure.py:108 | test/test_mu_audit_drift.py:202, test/test_shell_overflow_guard.py:26, test/test_mu_audit_drift.py:179 |
| `Lmech_SN` | y | — | updateDict/SPSFeedback | — | — |
| `Lmech_W` | y | — | updateDict/SPSFeedback | — | — |
| `Lmech_total` | y | — | updateDict/SPSFeedback | trinity/bubble_structure/bubble_luminosity.py:224, trinity/main.py:337, trinity/phase1_energy/energy_phase_ODEs.py:148, trinity/phase1b_energy_implicit/get_betadelta.py:448, trinity/phase1b_energy_implicit/get_betadelta.py:558, trinity/phase1b_energy_implicit/get_betadelta.py:903 … (+2) | test/test_betadelta_hybr.py:68 |
| `Ln` | y | — | updateDict/SPSFeedback | trinity/shell_structure/get_shellODE.py:88, trinity/shell_structure/shell_structure.py:109 | test/test_mu_audit_drift.py:202, test/test_mu_audit_drift.py:231, test/test_shell_overflow_guard.py:26 … (+1) |
| `PISM` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:154, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:499, trinity/phase1c_transition/run_transition_phase.py:292, trinity/phase2_momentum/run_momentum_phase.py:233, trinity/phase2_momentum/run_momentum_phase.py:327 | — |
| `P_HII` | y | trinity/phase1_energy/run_energy_phase.py:217, trinity/phase1_energy/run_energy_phase.py:243, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:984, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1004, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1382, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1391 … (+6) | — | trinity/phase1_energy/energy_phase_ODEs.py:162, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:531, trinity/phase1c_transition/run_transition_phase.py:324, trinity/phase2_momentum/run_momentum_phase.py:264, trinity/phase2_momentum/run_momentum_phase.py:361 | — |
| `P_drive` | y | trinity/phase1_energy/run_energy_phase.py:245, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1005, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1392, trinity/phase1c_transition/run_transition_phase.py:581, trinity/phase1c_transition/run_transition_phase.py:858, trinity/phase2_momentum/run_momentum_phase.py:659 | — | — | — |
| `P_ram` | y | trinity/phase1_energy/run_energy_phase.py:247, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1006, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1393, trinity/phase1c_transition/run_transition_phase.py:582, trinity/phase1c_transition/run_transition_phase.py:859, trinity/phase2_momentum/run_momentum_phase.py:660 | — | — | — |
| `Pb` | y | trinity/phase1_energy/run_energy_phase.py:107, trinity/phase1_energy/run_energy_phase.py:192, trinity/phase1_energy/run_energy_phase.py:397, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:939, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1373, trinity/phase1c_transition/run_transition_phase.py:509 … (+4) | updateDict/BubbleProperties | trinity/bubble_structure/get_bubbleParams.py:112, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:816, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1267, trinity/shell_structure/shell_structure.py:104, trinity/shell_structure/shell_structure.py:125 | — |
| `Qi` | y | — | updateDict/SPSFeedback | trinity/bubble_structure/bubble_luminosity.py:428, trinity/bubble_structure/bubble_luminosity.py:779, trinity/bubble_structure/bubble_luminosity.py:812, trinity/phase1_energy/energy_phase_ODEs.py:147, trinity/shell_structure/get_shellODE.py:90, trinity/shell_structure/shell_structure.py:107 | test/test_mu_audit_drift.py:202, test/test_shell_overflow_guard.py:42, test/test_mu_audit_drift.py:180 … (+1) |
| `R1` | y | trinity/phase1_energy/run_energy_phase.py:108, trinity/phase1_energy/run_energy_phase.py:191, trinity/phase1_energy/run_energy_phase.py:396, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:938, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1372, trinity/phase1c_transition/run_transition_phase.py:508 … (+3) | updateDict/BubbleProperties | trinity/bubble_structure/get_bubbleParams.py:115 | test/test_log_stopping_fate.py:93 |
| `R2` | y | trinity/_input/dictionary.py:14, trinity/_input/dictionary.py:1297, trinity/_input/dictionary.py:1324, trinity/main.py:235, trinity/phase1_energy/run_energy_phase.py:149, trinity/phase1_energy/run_energy_phase.py:355 … (+6) | — | trinity/_input/dictionary.py:13, trinity/_input/dictionary.py:725, trinity/bubble_structure/bubble_luminosity.py:223, trinity/bubble_structure/bubble_luminosity.py:230, trinity/bubble_structure/bubble_luminosity.py:252, trinity/bubble_structure/bubble_luminosity.py:300 … (+22) | test/test_dR2min_magic_number.py:209, test/test_dR2min_magic_number.py:252, test/test_dR2min_magic_number.py:306 … (+2) |
| `R_IF` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1003, trinity/phase1c_transition/run_transition_phase.py:579, trinity/phase2_momentum/run_momentum_phase.py:657 | updateDict/ShellProperties | trinity/phase1_energy/energy_phase_ODEs.py:144 | — |
| `SB99_rotation` | y | — | — | trinity/_input/registry.py:283, trinity/cooling/non_CIE/read_cloudy.py:51 | — |
| `SimulationEndCode` | y | trinity/main.py:270, trinity/phase1_energy/run_energy_phase.py:177, trinity/phase1_energy/run_energy_phase.py:374, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:675, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:773, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1043 … (+19) | — | trinity/_output/simulation_end.py:136, trinity/_output/simulation_end.py:193, trinity/_output/simulation_end.py:194, trinity/_output/terminal_prints.py:215, trinity/phase_general/phase_events.py:622 | test/test_phase_events.py:148, test/test_log_stopping_fate.py:53, test/test_log_stopping_fate.py:68 |
| `SimulationEndReason` | y | trinity/main.py:267, trinity/phase1_energy/run_energy_phase.py:173, trinity/phase1_energy/run_energy_phase.py:370, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:676, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:769, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1042 … (+19) | — | trinity/_output/simulation_end.py:184, trinity/_output/simulation_end.py:185, trinity/_output/terminal_prints.py:220, trinity/main.py:202 | test/test_phase_events.py:147, test/test_log_stopping_fate.py:54, test/test_log_stopping_fate.py:69 |
| `T0` | y | trinity/main.py:238, trinity/phase1_energy/run_energy_phase.py:152, trinity/phase1_energy/run_energy_phase.py:187, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:797, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1138, trinity/phase1c_transition/run_transition_phase.py:491 … (+1) | — | trinity/phase1_energy/run_energy_phase.py:86, trinity/phase1b_energy_implicit/get_betadelta.py:445, trinity/phase1b_energy_implicit/get_betadelta.py:555, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:685, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:696, trinity/phase1c_transition/run_transition_phase.py:430 … (+1) | — |
| `TShell_ion` | y | — | — | trinity/_input/read_param.py:355, trinity/phase1_energy/energy_phase_ODEs.py:54, trinity/phase1_energy/energy_phase_ODEs.py:156, trinity/phase1_energy/run_energy_phase.py:214, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:495, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:981 … (+10) | test/test_mu_audit_drift.py:203 |
| `TShell_neu` | y | — | — | trinity/shell_structure/get_shellODE.py:84, trinity/shell_structure/shell_structure.py:308 | test/test_mu_audit_drift.py:232 |
| `ZCloud` | y | — | — | trinity/_input/read_param.py:366, trinity/_input/read_param.py:367, trinity/_input/read_param.py:417, trinity/_input/read_param.py:426, trinity/_input/registry.py:277, trinity/_input/registry.py:279 … (+4) | — |
| `Z_He` | y | — | — | trinity/_input/read_param.py:309 | test/test_mu_audit_drift.py:62, test/test_mu_audit_drift.py:319 |
| `Z_He_shell` | y | — | — | trinity/_input/read_param.py:330 | test/test_mu_audit_drift.py:323 |
| `_snapshots_after_rCloud` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1026, trinity/phase1c_transition/run_transition_phase.py:600, trinity/phase2_momentum/run_momentum_phase.py:682 | — | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:767, trinity/phase1c_transition/run_transition_phase.py:471, trinity/phase2_momentum/run_momentum_phase.py:552 | — |
| `allowShellDissolution` | y | — | — | trinity/shell_structure/shell_structure.py:443 | — |
| `betadelta_converged` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:910 | — | — | — |
| `betadelta_solver` | y | — | — | trinity/phase1b_energy_implicit/get_betadelta.py:618 | test/test_betadelta_hybr.py:192 |
| `betadelta_total_residual` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:911 | — | — | — |
| `bubble_L1Bubble` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_L2Conduction` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_L3Intermediate` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_LTotal` | y | — | updateDict/BubbleProperties | trinity/phase1_energy/energy_phase_ODEs.py:146 | — |
| `bubble_Leak` | y | trinity/phase1_energy/run_energy_phase.py:255, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:813 | — | trinity/phase1b_energy_implicit/get_betadelta.py:469, trinity/phase1b_energy_implicit/get_betadelta.py:573, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1240 | — |
| `bubble_Lgain` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:928 | — | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:861, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:872, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:906 | — |
| `bubble_Lloss` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:930 | — | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:862, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:872, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:906, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1245 | — |
| `bubble_T_arr` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_T_r_Tb` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_Tavg` | y | — | updateDict/BubbleProperties | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:943, trinity/phase1c_transition/run_transition_phase.py:512 | — |
| `bubble_dMdt` | y | — | updateDict/BubbleProperties | trinity/bubble_structure/bubble_luminosity.py:242, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:860, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:905 | test/test_dR2min_magic_number.py:302, test/test_dR2min_magic_number.py:331, test/test_dR2min_magic_number.py:303 … (+2) |
| `bubble_dMdtGuess` | y | — | — | — | — |
| `bubble_dTdr_arr` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_mass` | y | — | updateDict/BubbleProperties | trinity/shell_structure/shell_structure.py:103 | — |
| `bubble_n_arr` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_r_Tb` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_r_arr` | y | — | updateDict/BubbleProperties | trinity/_input/dictionary.py:644, trinity/_input/dictionary.py:653, trinity/_input/dictionary.py:663 | — |
| `bubble_v_arr` | y | — | updateDict/BubbleProperties | — | — |
| `bubble_xi_Tb` | y | — | — | trinity/bubble_structure/bubble_luminosity.py:250, trinity/phase0_init/get_InitPhaseParam.py:78 | — |
| `cStruc_cooling_CIE_interpolation` | y | trinity/main.py:171 | — | trinity/bubble_structure/bubble_luminosity.py:742, trinity/cooling/net_coolingcurve.py:103 | test/test_dR2min_magic_number.py:122, test/test_fA_source_boost.py:45, test/test_net_coolingcurve.py:40 … (+1) |
| `cStruc_cooling_CIE_logLambda` | y | trinity/main.py:170 | — | — | test/test_dR2min_magic_number.py:120, test/test_fA_source_boost.py:43, test/test_residual_resample.py:88 … (+4) |
| `cStruc_cooling_CIE_logT` | y | trinity/main.py:169 | — | trinity/cooling/net_coolingcurve.py:104 | test/test_dR2min_magic_number.py:119, test/test_fA_source_boost.py:42, test/test_net_coolingcurve.py:38 … (+1) |
| `cStruc_cooling_nonCIE` | y | trinity/phase1_energy/run_energy_phase.py:126, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:785 | — | trinity/bubble_structure/bubble_luminosity.py:781, trinity/bubble_structure/bubble_luminosity.py:821, trinity/cooling/net_coolingcurve.py:99 | test/test_fA_source_boost.py:163, test/test_dR2min_magic_number.py:127, test/test_fA_source_boost.py:48 … (+2) |
| `cStruc_heating_nonCIE` | y | trinity/phase1_energy/run_energy_phase.py:127, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:786 | — | trinity/bubble_structure/bubble_luminosity.py:782, trinity/bubble_structure/bubble_luminosity.py:822 | test/test_dR2min_magic_number.py:128, test/test_fA_source_boost.py:49, test/test_net_coolingcurve.py:45 … (+1) |
| `cStruc_net_nonCIE_interpolation` | y | trinity/phase1_energy/run_energy_phase.py:128, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:787 | — | trinity/cooling/net_coolingcurve.py:101 | test/test_net_coolingcurve.py:67, test/test_dR2min_magic_number.py:129, test/test_fA_source_boost.py:50 … (+2) |
| `c_light` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:134, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:546, trinity/phase1c_transition/run_transition_phase.py:345, trinity/phase2_momentum/run_momentum_phase.py:279, trinity/phase2_momentum/run_momentum_phase.py:344, trinity/shell_structure/get_shellODE.py:87 | test/test_mu_audit_drift.py:199, test/test_mu_audit_drift.py:229 |
| `c_sound` | y | trinity/phase1_energy/run_energy_phase.py:223, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:944, trinity/phase1c_transition/run_transition_phase.py:518 | — | trinity/phase1_energy/energy_phase_ODEs.py:164, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:817 | — |
| `caseB_alpha` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:153, trinity/shell_structure/get_shellODE.py:85, trinity/shell_structure/shell_structure.py:144, trinity/shell_structure/shell_structure.py:248, trinity/shell_structure/shell_structure.py:282 | test/test_mu_audit_drift.py:201, test/test_shell_overflow_guard.py:41 |
| `chi_e` | y | trinity/_input/read_param.py:320 | — | trinity/bubble_structure/bubble_luminosity.py:746, trinity/bubble_structure/bubble_luminosity.py:833, trinity/cooling/net_coolingcurve.py:164, trinity/cooling/net_coolingcurve.py:187 | test/test_mu_audit_drift.py:62, test/test_mu_audit_drift.py:63, test/test_mu_audit_drift.py:113 … (+6) |
| `chi_e_shell` | y | trinity/_input/read_param.py:341 | — | trinity/shell_structure/get_shellODE.py:82, trinity/shell_structure/shell_structure.py:144, trinity/shell_structure/shell_structure.py:248, trinity/shell_structure/shell_structure.py:282 | test/test_mu_audit_drift.py:197, test/test_mu_audit_drift.py:254, test/test_mu_audit_drift.py:325 … (+1) |
| `coll_r` | y | — | — | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1318, trinity/phase1c_transition/run_transition_phase.py:788, trinity/phase2_momentum/run_momentum_phase.py:841, trinity/phase_general/phase_events.py:443, trinity/phase_general/phase_events.py:482, trinity/phase_general/phase_events.py:526 … (+1) | — |
| `cool_alpha` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:662, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:798, trinity/phase1c_transition/run_transition_phase.py:399 | — | trinity/bubble_structure/bubble_luminosity.py:405, trinity/bubble_structure/bubble_luminosity.py:439, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:654, trinity/phase1c_transition/run_transition_phase.py:391 | test/test_fA_source_boost.py:123 |
| `cool_beta` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:885 | — | trinity/bubble_structure/bubble_luminosity.py:442, trinity/bubble_structure/bubble_luminosity.py:446, trinity/bubble_structure/bubble_luminosity.py:1026, trinity/bubble_structure/get_bubbleParams.py:112, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:686, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:715 … (+1) | test/test_fA_source_boost.py:125, test/test_fA_source_boost.py:128 |
| `cool_delta` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:886 | — | trinity/bubble_structure/bubble_luminosity.py:442, trinity/bubble_structure/bubble_luminosity.py:446, trinity/bubble_structure/bubble_luminosity.py:1027, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:687, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:716, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:828 | test/test_fA_source_boost.py:125, test/test_fA_source_boost.py:128 |
| `cooling_boost_fA` | y | — | — | trinity/bubble_structure/bubble_luminosity.py:435, trinity/bubble_structure/bubble_luminosity.py:845 | test/test_fA_source_boost.py:58, test/test_fA_source_boost.py:59, test/test_fA_source_boost.py:98 … (+2) |
| `cooling_boost_fmix` | y | — | — | trinity/phase1b_energy_implicit/get_betadelta.py:369 | — |
| `cooling_boost_kappa` | y | — | — | trinity/_input/registry.py:139, trinity/bubble_structure/bubble_luminosity.py:304, trinity/bubble_structure/bubble_luminosity.py:398, trinity/bubble_structure/bubble_luminosity.py:441 | test/test_fA_source_boost.py:124, test/test_fkappa_auto.py:97, test/test_fkappa_auto.py:98 … (+2) |
| `cooling_boost_mode` | y | — | — | trinity/_input/registry.py:138, trinity/phase1b_energy_implicit/get_betadelta.py:366 | — |
| `cooling_boost_theta` | y | — | — | trinity/phase1b_energy_implicit/get_betadelta.py:370 | — |
| `coverFraction` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:163, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:814 | — |
| `current_phase` | y | trinity/main.py:244, trinity/main.py:278, trinity/main.py:301, trinity/main.py:327 | — | trinity/_input/dictionary.py:531, trinity/phase1_energy/energy_phase_ODEs.py:158 | — |
| `densBE_Omega` | y | trinity/cloud_properties/initial_profile.py:140 | — | trinity/_input/sweep_parser.py:745, trinity/_input/sweep_runner.py:139, trinity/cloud_properties/bonnorEbertSphere.py:535, trinity/cloud_properties/validate_gmc.py:390, trinity/phase0_init/get_InitCloudProp.py:315 | test/test_active_when.py:61 |
| `densBE_Teff` | y | trinity/cloud_properties/bonnorEbertSphere.py:563, trinity/cloud_properties/initial_profile.py:144, trinity/phase0_init/get_InitCloudProp.py:342 | — | trinity/cloud_properties/bonnorEbertSphere.py:599, trinity/cloud_properties/bonnorEbertSphere.py:639 | test/test_mu_audit_drift.py:306 |
| `densBE_dudxi_arr` | y | trinity/cloud_properties/bonnorEbertSphere.py:567 | — | — | — |
| `densBE_f_m` | y | trinity/cloud_properties/bonnorEbertSphere.py:570, trinity/cloud_properties/initial_profile.py:147, trinity/phase0_init/get_InitCloudProp.py:354 | — | trinity/cloud_properties/mass_profile.py:384, trinity/cloud_properties/mass_profile.py:394 | — |
| `densBE_f_rho_rhoc` | y | trinity/cloud_properties/bonnorEbertSphere.py:569, trinity/cloud_properties/initial_profile.py:146, trinity/phase0_init/get_InitCloudProp.py:353 | — | trinity/cloud_properties/density_profile.py:154 | — |
| `densBE_rho_rhoc_arr` | y | trinity/cloud_properties/bonnorEbertSphere.py:568 | — | — | — |
| `densBE_sigma` | y | trinity/cloud_properties/bonnorEbertSphere.py:564, trinity/phase0_init/get_InitCloudProp.py:349 | — | — | test/test_mu_audit_drift.py:301, test/test_mu_audit_drift.py:302 |
| `densBE_u_arr` | y | trinity/cloud_properties/bonnorEbertSphere.py:566 | — | — | — |
| `densBE_xi_arr` | y | trinity/cloud_properties/bonnorEbertSphere.py:565 | — | — | — |
| `densBE_xi_out` | y | trinity/cloud_properties/bonnorEbertSphere.py:571, trinity/cloud_properties/initial_profile.py:145, trinity/phase0_init/get_InitCloudProp.py:352 | — | trinity/cloud_properties/mass_profile.py:384, trinity/cloud_properties/mass_profile.py:395 | test/test_materialize_runtime.py:203 |
| `densPL_alpha` | y | — | — | trinity/_input/fkappa_auto.py:108, trinity/_input/sweep_parser.py:740, trinity/_input/sweep_runner.py:131, trinity/cloud_properties/density_profile.py:136, trinity/cloud_properties/mass_profile.py:303, trinity/cloud_properties/validate_gmc.py:387 … (+1) | test/test_active_when.py:99, test/test_active_when.py:142 |
| `dens_profile` | y | — | — | trinity/_input/fkappa_auto.py:107, trinity/_input/registry.py:79, trinity/_input/sweep_parser.py:737, trinity/_input/sweep_runner.py:99, trinity/_output/header.py:95, trinity/cloud_properties/density_profile.py:135 … (+6) | — |
| `dust_KappaIR` | y | — | — | trinity/phase1_energy/energy_phase_ODEs.py:135, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:547, trinity/phase1c_transition/run_transition_phase.py:346, trinity/phase2_momentum/run_momentum_phase.py:280, trinity/phase2_momentum/run_momentum_phase.py:345 | — |
| `dust_noZ` | y | — | — | trinity/_input/read_param.py:366 | — |
| `dust_sigma` | y | trinity/_input/read_param.py:367, trinity/_input/read_param.py:369 | — | trinity/shell_structure/get_shellODE.py:78, trinity/shell_structure/shell_structure.py:278 | test/test_mu_audit_drift.py:200, test/test_mu_audit_drift.py:230, test/test_shell_overflow_guard.py:43 |
| `gamma_adia` | y | trinity/cloud_properties/initial_profile.py:141 | — | trinity/_functions/operations.py:211, trinity/_input/sweep_runner.py:143, trinity/bubble_structure/bubble_luminosity.py:232, trinity/cloud_properties/bonnorEbertSphere.py:537, trinity/cloud_properties/bonnorEbertSphere.py:602, trinity/cloud_properties/bonnorEbertSphere.py:642 … (+13) | test/test_mu_audit_drift.py:285, test/test_mu_audit_drift.py:305 |
| `include_PHII` | y | — | — | trinity/_input/sweep_parser.py:753, trinity/_input/sweep_parser.py:754, trinity/phase1_energy/energy_phase_ODEs.py:161, trinity/phase1_energy/run_energy_phase.py:213, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:980, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1378 … (+4) | — |
| `initial_cloud_m_arr` | y | trinity/phase0_init/get_InitCloudProp.py:140 | — | trinity/phase0_init/get_InitCloudProp.py:139 | — |
| `initial_cloud_n_arr` | y | trinity/phase0_init/get_InitCloudProp.py:138 | — | trinity/phase0_init/get_InitCloudProp.py:137 | — |
| `initial_cloud_r_arr` | y | trinity/phase0_init/get_InitCloudProp.py:136 | — | trinity/phase0_init/get_InitCloudProp.py:135 | — |
| `isCollapse` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1303, trinity/phase1c_transition/run_transition_phase.py:773, trinity/phase2_momentum/run_momentum_phase.py:826, trinity/phase_general/phase_events.py:629 | — | trinity/phase1_energy/energy_phase_ODEs.py:142, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:954, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1185, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1316, trinity/phase1c_transition/run_transition_phase.py:537, trinity/phase1c_transition/run_transition_phase.py:696 … (+6) | test/test_phase_events.py:150 |
| `isDissolved` | y | trinity/phase1c_transition/run_transition_phase.py:814, trinity/phase2_momentum/run_momentum_phase.py:867 | updateDict/ShellProperties | trinity/shell_structure/shell_structure.py:130 | — |
| `is_phiDepleted` | y | — | updateDict/ShellProperties | — | — |
| `k_B` | y | — | — | trinity/_functions/operations.py:211, trinity/bubble_structure/bubble_luminosity.py:303, trinity/bubble_structure/bubble_luminosity.py:396, trinity/bubble_structure/bubble_luminosity.py:427, trinity/bubble_structure/bubble_luminosity.py:673, trinity/bubble_structure/bubble_luminosity.py:725 … (+16) | test/test_fA_source_boost.py:121, test/test_mu_audit_drift.py:110, test/test_mu_audit_drift.py:198 … (+2) |
| `large_arr` | n | trinity/_input/dictionary.py:1305 | — | trinity/_input/dictionary.py:1317, trinity/_input/dictionary.py:1318 | — |
| `log_colors` | y | — | — | trinity/_functions/logging_setup.py:428, trinity/_functions/logging_setup.py:478 | — |
| `log_console` | y | — | — | run.py:183, run.py:184, trinity/_functions/logging_setup.py:203, trinity/_functions/logging_setup.py:420, trinity/_functions/logging_setup.py:470 | — |
| `log_file` | y | — | — | run.py:188, run.py:189, trinity/_functions/logging_setup.py:204, trinity/_functions/logging_setup.py:424, trinity/_functions/logging_setup.py:474 | — |
| `log_level` | y | — | — | run.py:178, run.py:179, trinity/_functions/logging_setup.py:202, trinity/_functions/logging_setup.py:416, trinity/_functions/logging_setup.py:466 | — |
| `mCloud` | y | trinity/_input/read_param.py:389 | — | trinity/_input/dictionary.py:42, trinity/_input/read_param.py:56, trinity/_input/read_param.py:57, trinity/_input/read_param.py:58, trinity/_input/read_param.py:386, trinity/_input/read_param.py:518 … (+15) | — |
| `mCloud_input` | y | trinity/_input/read_param.py:390 | — | trinity/_input/fkappa_auto.py:118 | — |
| `mCluster` | y | trinity/_input/read_param.py:396 | — | trinity/main.py:144, trinity/phase1_energy/energy_phase_ODEs.py:145, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:490, trinity/phase1c_transition/run_transition_phase.py:283, trinity/phase2_momentum/run_momentum_phase.py:221, trinity/phase2_momentum/run_momentum_phase.py:349 | — |
| `model_name` | y | trinity/_input/read_param.py:373 | — | trinity/_input/read_param.py:372, trinity/_input/registry.py:234, trinity/_output/header.py:90, trinity/_output/simulation_end.py:199, trinity/_output/simulation_end.py:200 | test/test_resolvers.py:182 |
| `mu_atom` | y | trinity/_input/read_param.py:317 | — | trinity/_functions/operations.py:209, trinity/shell_structure/get_shellODE.py:79, trinity/shell_structure/shell_structure.py:307 | test/test_mu_audit_drift.py:193, test/test_mu_audit_drift.py:226, test/test_mu_audit_drift.py:252 … (+1) |
| `mu_convert` | y | trinity/_input/read_param.py:316 | — | trinity/_input/sweep_runner.py:116, trinity/bubble_structure/bubble_luminosity.py:427, trinity/bubble_structure/bubble_luminosity.py:673, trinity/bubble_structure/bubble_luminosity.py:725, trinity/bubble_structure/bubble_luminosity.py:778, trinity/bubble_structure/bubble_luminosity.py:811 … (+31) | test/test_fA_source_boost.py:121, test/test_mu_audit_drift.py:74, test/test_mu_audit_drift.py:77 … (+11) |
| `mu_ion` | y | trinity/_input/read_param.py:318 | — | trinity/_functions/operations.py:207, trinity/bubble_structure/bubble_luminosity.py:302, trinity/bubble_structure/bubble_luminosity.py:397, trinity/bubble_structure/bubble_luminosity.py:427, trinity/bubble_structure/bubble_luminosity.py:673, trinity/bubble_structure/bubble_luminosity.py:725 … (+2) | test/test_fA_source_boost.py:121, test/test_mu_audit_drift.py:77, test/test_mu_audit_drift.py:99 … (+8) |
| `mu_ion_shell` | y | trinity/_input/read_param.py:333 | — | trinity/phase1_energy/energy_phase_ODEs.py:54, trinity/phase1_energy/run_energy_phase.py:214, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:513, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:981, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1379, trinity/phase1c_transition/run_transition_phase.py:306 … (+8) | test/test_mu_audit_drift.py:74, test/test_mu_audit_drift.py:91, test/test_mu_audit_drift.py:194 … (+3) |
| `mu_mol` | y | trinity/_input/read_param.py:319 | — | — | — |
| `nCore` | y | trinity/phase0_init/get_InitCloudProp.py:230 | — | trinity/_input/fkappa_auto.py:121, trinity/_input/sweep_parser.py:722, trinity/_input/sweep_runner.py:104, trinity/_output/header.py:94, trinity/cloud_properties/bonnorEbertSphere.py:534, trinity/cloud_properties/bonnorEbertSphere.py:600 … (+10) | — |
| `nEdge` | y | trinity/cloud_properties/bonnorEbertSphere.py:575, trinity/phase0_init/get_InitCloudProp.py:279, trinity/phase0_init/get_InitCloudProp.py:341 | — | — | — |
| `nISM` | y | — | — | trinity/_input/sweep_runner.py:117, trinity/cloud_properties/density_profile.py:109, trinity/cloud_properties/mass_profile.py:298, trinity/cloud_properties/mass_profile.py:376, trinity/cloud_properties/validate_gmc.py:373, trinity/phase0_init/get_InitCloudProp.py:163 … (+13) | — |
| `n_IF` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1002, trinity/phase1c_transition/run_transition_phase.py:578, trinity/phase2_momentum/run_momentum_phase.py:656 | updateDict/ShellProperties | trinity/phase1_energy/energy_phase_ODEs.py:143 | — |
| `n_IF_ODE` | y | — | updateDict/ShellProperties | — | — |
| `n_IF_Str` | y | — | updateDict/ShellProperties | — | — |
| `ndens` | n | — | — | trinity/_output/trinity_reader.py:1439, trinity/_output/trinity_reader.py:1489, trinity/_output/trinity_reader.py:1554 | — |
| `output_format` | y | — | — | — | — |
| `path2output` | y | trinity/_input/dictionary.py:954, trinity/_input/dictionary.py:1293 | — | run.py:195, run.py:202, trinity/_functions/logging_setup.py:147, trinity/_functions/logging_setup.py:195, trinity/_functions/logging_setup.py:205, trinity/_functions/logging_setup.py:432 … (+20) | test/test_resolvers.py:183 |
| `path_cooling_CIE` | y | trinity/_input/read_param.py:425, trinity/_input/read_param.py:427 | — | trinity/_input/read_param.py:423, trinity/main.py:162 | test/test_dR2min_magic_number.py:118, test/test_fA_source_boost.py:41, test/test_net_coolingcurve.py:37 … (+1) |
| `path_cooling_nonCIE` | y | — | — | trinity/cooling/non_CIE/read_cloudy.py:50 | — |
| `pdot_SN` | y | — | updateDict/SPSFeedback | — | — |
| `pdot_W` | y | — | updateDict/SPSFeedback | — | — |
| `pdot_total` | y | — | updateDict/SPSFeedback | trinity/bubble_structure/get_bubbleParams.py:119, trinity/bubble_structure/get_bubbleParams.py:170, trinity/phase1b_energy_implicit/get_betadelta.py:450, trinity/phase1b_energy_implicit/get_betadelta.py:560 | — |
| `pdotdot_total` | y | — | updateDict/SPSFeedback | trinity/bubble_structure/get_bubbleParams.py:120, trinity/bubble_structure/get_bubbleParams.py:171, trinity/phase1b_energy_implicit/get_betadelta.py:451, trinity/phase1b_energy_implicit/get_betadelta.py:561 | — |
| `phaseSwitch_LlossLgain` | y | — | — | trinity/phase1_energy/run_energy_phase.py:280, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1250 | — |
| `press_HII_in` | y | trinity/phase1_energy/run_energy_phase.py:249, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1007, trinity/phase1c_transition/run_transition_phase.py:583, trinity/phase2_momentum/run_momentum_phase.py:661 | — | — | — |
| `rCloud` | y | trinity/cloud_properties/bonnorEbertSphere.py:574, trinity/phase0_init/get_InitCloudProp.py:277, trinity/phase0_init/get_InitCloudProp.py:339 | — | trinity/cloud_properties/density_profile.py:111, trinity/cloud_properties/mass_profile.py:301, trinity/cloud_properties/mass_profile.py:374, trinity/cloud_properties/mass_profile.py:521, trinity/cloud_properties/powerLawSphere.py:250, trinity/main.py:123 … (+16) | — |
| `rCloud_max` | y | — | — | trinity/_input/sweep_runner.py:127, trinity/cloud_properties/validate_gmc.py:368 | — |
| `rCore` | y | trinity/phase0_init/get_InitCloudProp.py:206, trinity/phase0_init/get_InitCloudProp.py:229, trinity/phase0_init/get_InitCloudProp.py:248, trinity/phase0_init/get_InitCloudProp.py:278, trinity/phase0_init/get_InitCloudProp.py:340 | — | trinity/_input/sweep_runner.py:132, trinity/cloud_properties/density_profile.py:112, trinity/cloud_properties/mass_profile.py:300, trinity/cloud_properties/validate_gmc.py:388, trinity/phase0_init/get_InitCloudProp.py:166, trinity/phase0_init/get_InitCloudProp.py:318 … (+2) | — |
| `rShell` | y | — | updateDict/ShellProperties | trinity/phase1_energy/energy_phase_ODEs.py:140, trinity/shell_structure/shell_structure.py:112 | — |
| `r_max` | n | — | — | — | test/test_validate_gmc.py:63 |
| `residual_Edot1_guess` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:918 | — | — | — |
| `residual_Edot2_guess` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:920 | — | — | — |
| `residual_T1_guess` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:922 | — | — | — |
| `residual_T2_guess` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:924 | — | — | — |
| `residual_betaEdot` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:916 | — | — | — |
| `residual_deltaT` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:915 | — | — | — |
| `sfe` | y | — | — | trinity/_input/fkappa_auto.py:119, trinity/_input/read_param.py:387, trinity/_input/sweep_parser.py:721, trinity/_output/header.py:91, trinity/_output/header.py:92, trinity/_output/trinity_reader.py:1488 … (+1) | — |
| `shell_fAbsorbedIon` | y | — | updateDict/ShellProperties | trinity/phase1_energy/energy_phase_ODEs.py:138 | — |
| `shell_fAbsorbedNeu` | y | — | updateDict/ShellProperties | — | — |
| `shell_fAbsorbedWeightedTotal` | y | — | updateDict/ShellProperties | — | — |
| `shell_fIonisedDust` | y | — | updateDict/ShellProperties | — | — |
| `shell_grav_force_m` | y | — | updateDict/ShellProperties | — | — |
| `shell_grav_phi` | y | — | updateDict/ShellProperties | — | — |
| `shell_grav_r` | y | — | updateDict/ShellProperties | trinity/_input/dictionary.py:679 | — |
| `shell_interpolate_massDot` | y | — | — | — | — |
| `shell_ion_idx` | y | — | updateDict/ShellProperties | — | — |
| `shell_mass` | y | trinity/phase1_energy/run_energy_phase.py:202, trinity/phase1_energy/run_energy_phase.py:251, trinity/phase1_energy/run_energy_phase.py:399, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:969, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1196, trinity/phase1c_transition/run_transition_phase.py:552 … (+3) | — | trinity/phase1_energy/energy_phase_ODEs.py:141, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:953, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1184, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1384, trinity/phase1c_transition/run_transition_phase.py:536, trinity/phase1c_transition/run_transition_phase.py:695 … (+4) | — |
| `shell_massDot` | y | trinity/phase1_energy/run_energy_phase.py:253, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:970, trinity/phase1c_transition/run_transition_phase.py:553, trinity/phase2_momentum/run_momentum_phase.py:623 | — | — | — |
| `shell_n0` | y | — | updateDict/ShellProperties | — | — |
| `shell_nMax` | y | — | updateDict/ShellProperties | trinity/phase1c_transition/run_transition_phase.py:806, trinity/phase2_momentum/run_momentum_phase.py:859 | — |
| `shell_n_arr` | y | — | updateDict/ShellProperties | — | — |
| `shell_r_arr` | y | — | updateDict/ShellProperties | trinity/_input/dictionary.py:695 | — |
| `shell_tauKappaRatio` | y | — | updateDict/ShellProperties | — | — |
| `shell_thickness` | y | — | updateDict/ShellProperties | — | — |
| `simplify_npoints` | y | — | — | trinity/_input/dictionary.py:506 | — |
| `small_arr` | n | trinity/_input/dictionary.py:1300 | — | — | — |
| `sps_col_Lbol` | y | — | — | — | — |
| `sps_col_Li` | y | — | — | — | — |
| `sps_col_Lmech_SN` | y | — | — | — | — |
| `sps_col_Lmech_W` | y | — | — | — | — |
| `sps_col_Lmech_total` | y | — | — | — | — |
| `sps_col_Ln` | y | — | — | — | — |
| `sps_col_Mdot_SN` | y | — | — | — | — |
| `sps_col_Qi` | y | — | — | — | — |
| `sps_col_fi` | y | — | — | — | — |
| `sps_col_pdot_SN` | y | — | — | — | — |
| `sps_col_pdot_W` | y | — | — | — | — |
| `sps_col_t` | y | — | — | — | — |
| `sps_col_v_SN` | y | — | — | — | — |
| `sps_column_map` | y | trinity/_input/registry.py:319 | — | trinity/_input/registry.py:271, trinity/sps/read_sps.py:43, trinity/sps/read_sps.py:129, trinity/sps/sps_columns.py:17 | test/test_read_sps.py:79, test/test_resolvers.py:107, test/test_resolvers.py:163 |
| `sps_data` | y | trinity/main.py:152 | — | — | — |
| `sps_f` | y | trinity/main.py:153 | — | trinity/phase0_init/get_InitPhaseParam.py:88, trinity/sps/read_sps.py:20, trinity/sps/update_feedback.py:106, trinity/sps/update_feedback.py:151 | — |
| `sps_path` | y | — | — | trinity/sps/read_sps.py:42, trinity/sps/read_sps.py:130 | test/test_read_sps.py:77, test/test_read_sps.py:66 |
| `sps_refmass` | y | trinity/_input/registry.py:309 | — | trinity/_input/dictionary.py:105, trinity/_input/dictionary.py:106, trinity/_input/registry.py:268, trinity/_input/registry.py:307, trinity/main.py:144 | test/test_read_sps.py:78, test/test_resolvers.py:106, test/test_resolvers.py:162 |
| `stop_at_rCloud_nSnap` | y | trinity/_input/registry.py:186 | — | trinity/main.py:129, trinity/main.py:263, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:764, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1023, trinity/phase1c_transition/run_transition_phase.py:468, trinity/phase1c_transition/run_transition_phase.py:597 … (+2) | test/test_validators.py:110, test/test_validators.py:127, test/test_validators.py:128 … (+2) |
| `stop_r` | y | — | — | trinity/main.py:130, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1327, trinity/phase1c_transition/run_transition_phase.py:797, trinity/phase2_momentum/run_momentum_phase.py:850, trinity/phase_general/phase_events.py:483, trinity/phase_general/phase_events.py:527 … (+1) | — |
| `stop_t` | y | — | — | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:665, trinity/phase1c_transition/run_transition_phase.py:402, trinity/phase2_momentum/run_momentum_phase.py:483 | — |
| `stop_t_diss` | y | — | — | trinity/phase1c_transition/run_transition_phase.py:813, trinity/phase1c_transition/run_transition_phase.py:820, trinity/phase2_momentum/run_momentum_phase.py:866, trinity/phase2_momentum/run_momentum_phase.py:873 | — |
| `tSF` | y | — | — | trinity/phase0_init/get_InitPhaseParam.py:85, trinity/phase1_energy/energy_phase_ODEs.py:157 | — |
| `t_next` | y | — | — | — | — |
| `t_now` | y | trinity/_input/dictionary.py:1296, trinity/_input/dictionary.py:1323, trinity/main.py:234, trinity/phase1_energy/run_energy_phase.py:148, trinity/phase1_energy/run_energy_phase.py:354, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:793 … (+6) | — | trinity/_input/dictionary.py:724, trinity/_input/dictionary.py:754, trinity/_output/terminal_prints.py:198, trinity/_output/trinity_reader.py:718, trinity/_output/trinity_reader.py:763, trinity/bubble_structure/bubble_luminosity.py:301 … (+24) | test/test_fA_source_boost.py:123, test/test_fA_source_boost.py:125, test/test_fA_source_boost.py:128 … (+4) |
| `t_previousCoolingUpdate` | y | trinity/phase1_energy/run_energy_phase.py:129, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:788 | — | trinity/phase1_energy/run_energy_phase.py:124, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:783 | — |
| `transition_trigger` | y | — | — | trinity/phase1_energy/run_energy_phase.py:275, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:728, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1292 | — |
| `v2` | y | trinity/main.py:236, trinity/phase1_energy/run_energy_phase.py:150, trinity/phase1_energy/run_energy_phase.py:356, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:795, trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:1136, trinity/phase1c_transition/run_transition_phase.py:489 … (+3) | — | trinity/_input/dictionary.py:43, trinity/bubble_structure/get_bubbleParams.py:117, trinity/bubble_structure/get_bubbleParams.py:173, trinity/main.py:258, trinity/phase1_energy/run_energy_phase.py:84, trinity/phase1b_energy_implicit/get_betadelta.py:443 … (+11) | test/test_materialize_runtime.py:39, test/test_phase_events.py:146 |
| `v_mech_total` | y | — | updateDict/SPSFeedback | trinity/bubble_structure/bubble_luminosity.py:224, trinity/main.py:338, trinity/phase1_energy/energy_phase_ODEs.py:149, trinity/phase1b_energy_implicit/get_betadelta.py:449, trinity/phase1b_energy_implicit/get_betadelta.py:559, trinity/phase1c_transition/run_transition_phase.py:328 … (+1) | — |
| `v_neg_frac_thick` | y | trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:898 | — | — | — |
| `x_He` | y | — | — | trinity/_input/read_param.py:308 | test/test_mu_audit_drift.py:62 |

### 3.6 Flagged keys

**(a) Written but never read back through a literal `params['key']` in `trinity/`.** All of these
still reach `dictionary.jsonl` / `metadata.json` through `_clean_for_snapshot`, and many are read
through `ADAPTIVE_MONITOR_KEYS` (marked ✓mon).

| key | write channel | ✓mon |
|---|---|---|
| `F_HII` | direct ×12 | ✓ |
| `F_grav`, `F_ion_in`, `F_rad`, `F_ram` | direct ×6 each | ✓ (`F_grav`,`F_ram`,`F_rad`,`F_ion_in`) |
| `F_ram_SN`, `F_ram_wind` | direct ×4 each | ✓ |
| `P_drive`, `P_ram` | direct ×6 each | — |
| `press_HII_in` | direct ×4 | — |
| `betadelta_converged`, `betadelta_total_residual` | direct ×1 each | — |
| `residual_deltaT`, `residual_betaEdot`, `residual_Edot1_guess`, `residual_Edot2_guess`, `residual_T1_guess`, `residual_T2_guess` | direct ×1 each | — |
| `shell_massDot` | direct ×4 | ✓ |
| `nEdge` | direct ×3 | — |
| `mu_mol` | direct ×1 (`read_param.py:319`) | — |
| `cStruc_cooling_CIE_logLambda` | direct ×1 (`main.py:170`) | — |
| `sps_data` | direct ×1 (`main.py:152`) | — |
| `v_neg_frac_thick` | direct ×1 (`run_energy_implicit_phase.py:898`) | — |
| `densBE_sigma` | direct ×2 | — |
| `densBE_xi_arr`, `densBE_u_arr`, `densBE_dudxi_arr`, `densBE_rho_rhoc_arr` | direct ×1 each | — |
| `Lmech_W`, `Lmech_SN`, `pdot_W`, `pdot_SN` | `updateDict/SPSFeedback` | ✓ (`Lmech_SN`, `pdot_SN`) |
| `bubble_L1Bubble`, `bubble_L2Conduction`, `bubble_L3Intermediate`, `bubble_T_arr`, `bubble_n_arr`, `bubble_dTdr_arr`, `bubble_v_arr`, `bubble_T_r_Tb`, `bubble_r_Tb` | `updateDict/BubbleProperties` | ✓ (`bubble_L1Bubble`, `bubble_L2Conduction`, `bubble_L3Intermediate`, `bubble_r_Tb`) |
| `shell_n0`, `shell_thickness`, `shell_tauKappaRatio`, `shell_fIonisedDust`, `shell_fAbsorbedNeu`, `shell_fAbsorbedWeightedTotal`, `shell_grav_phi`, `shell_grav_force_m`, `shell_ion_idx`, `shell_n_arr`, `n_IF_ODE`, `n_IF_Str`, `is_phiDepleted` | `updateDict/ShellProperties` | ✓ (`shell_n0`, `shell_thickness`, `shell_tauKappaRatio`, `shell_fIonisedDust`) |

Note `bubble_T_arr` / `bubble_n_arr` / `bubble_dTdr_arr` / `bubble_v_arr` / `bubble_r_arr` /
`shell_r_arr` / `shell_n_arr` / `shell_grav_r` / `shell_grav_force_m` are the nine
`SNAPSHOT_PROFILE_ARRAY_KEYS` (`dictionary.py:67`) consumed by the special-case branches of
`_clean_for_snapshot` (`dictionary.py:639-701`).

**(b) Read in `trinity/` but never assigned there** — i.e. their only write is the generic
`read_param` Step 4 merge (`read_param.py:270`), so their value is fixed by `.param` / `default.param`:

`C_thermal`, `FB_mColdSNFrac`, `FB_mColdWindFrac`, `FB_thermCoeffSN`, `FB_thermCoeffWind`,
`FB_vSN`, `G`, `PISM`, `SB99_rotation`, `TShell_ion`, `TShell_neu`, `ZCloud`, `Z_He`,
`Z_He_shell`, `allowShellDissolution`, `betadelta_solver`, `bubble_xi_Tb`, `c_light`,
`caseB_alpha`, `coll_r`, `cooling_boost_fA`, `cooling_boost_fmix`, `cooling_boost_kappa`,
`cooling_boost_mode`, `cooling_boost_theta`, `coverFraction`, `densPL_alpha`, `dens_profile`,
`dust_KappaIR`, `dust_noZ`, `include_PHII`, `k_B`, `log_colors`, `log_console`, `log_file`,
`log_level`, `nISM`, `path_cooling_nonCIE`, `phaseSwitch_LlossLgain`, `rCloud_max`, `sfe`,
`simplify_npoints`, `sps_path`, `stop_r`, `stop_t`, `stop_t_diss`, `tSF`, `transition_trigger`,
`x_He` (49 keys).

**(c) Neither written nor read anywhere in `trinity/` outside the registry declaration itself:**

| key | spec line | note |
|---|---|---|
| `F_ISM` | `registry.py:486` | spec `info` says "placeholder, never computed — always 0"; present in all three `ADAPTIVE_MONITOR_KEYS` lists and in `trinity_reader.PARAM_DOCS` (`:197`) and the reader's Forces group (`:995`) |
| `bubble_dMdtGuess` | `registry.py:509` | appears in `trinity_reader.PARAM_DOCS` (`:247`) |
| `t_next` | `registry.py:434` | appears in `trinity_reader.PARAM_DOCS` (`:149`) and Time group (`:988`) |
| `shell_interpolate_massDot` | `registry.py:477` | appears in `trinity_reader.PARAM_DOCS` (`:274`) |
| `output_format` | `registry.py:331` | spec `info` says "currently inert" |
| `sps_col_t`, `sps_col_Qi`, `sps_col_fi`, `sps_col_Lbol`, `sps_col_Lmech_W`, `sps_col_pdot_W`, `sps_col_Lmech_total`, `sps_col_Lmech_SN`, `sps_col_pdot_SN`, `sps_col_Mdot_SN`, `sps_col_v_SN`, `sps_col_Li`, `sps_col_Ln` | `registry.py:395-407` | read **dynamically**: `sps_columns.build_user_column_map` (`sps_columns.py:254`) builds the key name as `f"sps_col_{canonical}"` inside a loop over `CANONICAL_NAMES` |

**(d) Keys written by more than one module.** Almost every runtime key is; the widest-fan-in ones:

| key | writing modules |
|---|---|
| `R2`, `v2`, `t_now` | `main.py`, `run_energy_phase.py`, `run_energy_implicit_phase.py`, `run_transition_phase.py`, `run_momentum_phase.py`, `phase_events.py` (`apply_event_result`) |
| `Eb` | same six, plus `run_momentum_phase.py:511,571` forcing `0.0` |
| `T0` | `main.py`, `run_energy_phase.py`, `run_energy_implicit_phase.py`, `run_transition_phase.py`, `run_momentum_phase.py`, `phase_events.py` |
| `R1`, `Pb` | `run_energy_phase.py`, `run_energy_implicit_phase.py`, `run_transition_phase.py`, `run_momentum_phase.py`, plus `updateDict/BubbleProperties` |
| `P_HII`, `F_HII` | all four phase runners (each writes `P_HII` twice per segment: once inline from `n_IF_Str`, once from `force_props`) |
| `shell_mass`, `shell_massDot` | all four phase runners (+ `run_energy_phase.py:251-253` from `ODEResult`) |
| `F_grav`, `F_ion_in`, `F_ram`, `F_rad`, `P_drive`, `P_ram`, `press_HII_in` | `run_energy_phase.py` (via `ODEResult`), `run_energy_implicit_phase.py`, `run_transition_phase.py`, `run_momentum_phase.py` |
| `SimulationEndReason`, `SimulationEndCode`, `EndSimulationDirectly` | `main.py`, all four runners, `phase_events.apply_event_result` |
| `isCollapse` | `run_energy_implicit_phase.py:1303`, `run_transition_phase.py:773`, `run_momentum_phase.py:826`, `phase_events.py:629` |
| `rCloud`, `rCore`, `nEdge` | `get_InitCloudProp.py` (both `_init_powerlaw_cloud` and `_init_bonnor_ebert_cloud`) |
| `cool_alpha` | `run_energy_implicit_phase.py:662,798`, `run_transition_phase.py:399` |
| `cool_beta`, `cool_delta` | `run_energy_implicit_phase.py:885-886`; reset to `np.nan` via `COOLING_PHASE_KEYS` at `main.py:317` |

`COOLING_PHASE_KEYS` (`dictionary.py:1180`) is bulk-written to `np.nan` exactly once, at
`main.py:317`, between phase 1c and phase 2, through `DescribedDict.reset_keys`
(`dictionary.py:981`).

---

## 4. Module-level global / mutable state

"Module-level" = created at import and surviving across calls within one process.

### 4.1 Genuinely mutated at run time

| object | defined | mutated by | reset between runs in-process? |
|---|---|---|---|
| `_CIE_TCUTOFF_CACHE: dict` | `cooling/net_coolingcurve.py:27` | `_cie_tcutoff` (`:48-55`) inserts `id(logT_CIE) -> min(logT_CIE[logT_CIE>5.5])` | **No.** Never cleared. Keyed on `id()` of the array created once in `main.py:169`; the source comment (`:27-29`) states the id is stable for the run. Two `params` objects in one process reuse or shadow entries by CPython id reuse. |
| `cooling_nonCIE._hotpath_cutoffs` (attribute) | set in `cooling/net_coolingcurve.py:44` | `_noncie_cutoffs` caches `(max(t[t<=5.5]), min(t))` on the cube instance | Rebuilt automatically because a fresh cube from `get_coolingStructure` has no attribute. The cube object lives in `params['cStruc_cooling_nonCIE']`. |
| `run.py::_shutdown_requested` | `run.py:261-262` (`global` inside `run_sweep`) | signal handler at `run.py:595-598` | Re-created each `run_sweep` call. |
| process-wide `atexit` + `signal` handlers | registered in `DescribedDict._register_crash_handlers` (`dictionary.py:262-290`) | every `DescribedDict()` construction appends one more `atexit` handler and **overwrites** `SIGINT`/`SIGTERM` handlers | **No.** Never unregistered. `read_param` constructs one `DescribedDict` per call (`read_param.py:253`), and `DescribedDict.load_snapshot` (`dictionary.py:951`) constructs more. |
| `_cube.npy` files under `path_cooling_nonCIE` | written by `cooling/non_CIE/read_cloudy.py:265` (`np.save(cube_filename, …)`) | on first `create_cubes` for a given `.dat`; read back at `:174-176` | On-disk cache, persists across processes. |
| `logging` root handlers | `_functions/logging_setup.py:112` `setup_logging` | called from `run.py:191`; clears and re-adds handlers | Per call. |
| `DedupWarningFilter._seen: set` | `_functions/logging_setup.py:97` | `filter()` (`:99-109`) | Per-filter-instance; a new filter is created per `setup_logging` call (`:269`, `:300`). |

### 4.2 Module-level objects built at import (read-only in the run path)

| object | defined | notes |
|---|---|---|
| `SPECS` / `REGISTRY` | `_input/registry.py:328` / `:536` | frozen dataclasses; `apply_active_when` / `materialize_runtime` `copy.deepcopy(spec.default)` so mutable defaults (`[]`, `np.array([])`) are not shared (`registry.py:615`, `:657`) |
| `RUN_CONST_KEYS`, `METADATA_EXCLUDE`, `DROPPED_IN_V2`, `RESERVED_TOP_LEVEL_KEYS`, `FINAL_STATE_EXCLUDE_ARRAYS` | `_output/run_constants.py:77,83,88,113,125` | tuples/frozensets |
| `_F_FIRE` (3×3×7 float array), `_INTERP` (`RegularGridInterpolator`) | `_input/fkappa_auto.py:49`, `:72` | built at import; only read |
| `CANONICALS`, `UNIT_CONVERSIONS`, `DEFAULT_SPS_COLUMN_MAP` | `sps/sps_columns.py:65`, `:114`, `:166` | plain dicts of frozen `ColumnSpec`; `DEFAULT_SPS_COLUMN_MAP` is handed **by reference** into `params['sps_column_map']` at `registry.py:293`+`:319` |
| `ConversionConstants`, `InverseConversionConstants`, `PhysicalConstantsCGS`, `CONV`, `INV_CONV` | `_functions/unit_conversions.py:58`, `:156`, `:193` | `@dataclass(frozen=True)` |
| `COOLING_PHASE_KEYS` (list), `SNAPSHOT_PROFILE_ARRAY_KEYS` (frozenset) | `_input/dictionary.py:1180`, `:67` | |
| `ADAPTIVE_MONITOR_KEYS` (three identical 35-element lists) | `run_energy_implicit_phase.py:150`, `run_transition_phase.py:112`, `run_momentum_phase.py:104` | |
| `_VALID_TRIGGERS` | `run_energy_implicit_phase.py:249` | frozenset |
| `HYBR_OPTIONS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4)` | `phase1b_energy_implicit/get_betadelta.py:74` | passed to `scipy.optimize.root` as `options=` |
| `_STATE_FIELDS` | `_output/terminal_prints.py:131` | |
| `CRITICAL_PARAMS`, `CHANGE_THRESHOLDS` | `_output/simulation_end.py:409`, `:437` | |
| `PARAM_DOCS` | `_output/trinity_reader.py:140` | |
| `_DEP_MAX_MAJOR`, `TRINITY_ROOT` | `run.py:49`, `:41` | |
| `_REPO_ROOT` | `_input/read_param.py:37`, `_input/registry.py:68` | `Path(__file__).resolve().parents[2]` |
| module `logger` objects | 25 modules | `logging.getLogger(__name__)` |
| `_trapezoid` | `bubble_structure/bubble_luminosity.py:37` | `getattr(np, 'trapezoid', None) or np.trapz` — numpy-2 shim |

### 4.3 Environment variables read at run time

| variable | site | effect |
|---|---|---|
| `TRINITY_BUBBLE_DIAG` | `bubble_luminosity._bubble_diag_enabled` (`:973`), used at `:648` | enables `_capture_bubble_integration` (`:978`) writing to `<path2output>/bubble_diag/` |
| `TRINITY_BUBBLE_STATE_DUMP` | `bubble_luminosity.py:679` | enables `_dump_bubble_state` (`:1098`) |
| `SLURM_JOB_ID` | `run.py:454`, `:521` | worker-count source label and advisory message |

---

## 5. External data dependencies

All bundled assets live under `lib/default/`. Paths are anchored to the repo root
(`_REPO_ROOT = Path(__file__).resolve().parents[2]`) — `read_param.py:37` and `registry.py:68` —
not to the CWD.

| dataset | resolved by | read by | format / assumed layout |
|---|---|---|---|
| **CIE cooling curve** — `lib/default/CIE/coolingCIE_{1_Cloudy,2_Cloudy_grains,3_Gnat-Ferland2012,4_Sutherland-Dopita1993}.dat` | `read_param.py:417-429`: integer preset `{1,2,3}` when `ZCloud == 1`; auto-pinned to file 4 when `ZCloud == 0.15`; otherwise `path_cooling_CIE` keeps its raw value | `main.py:165` `np.loadtxt(cooling_path, unpack=True)` | two whitespace columns, header comment `# log(T), log(Lambda)`; unpacked as `logT, logLambda` and wrapped in `interp1d(kind='linear')` (`main.py:167`). Consumed by `cooling/CIE/read_coolingcurve.get_Lambda` (`:25`) as `10**f(log10 T)`, units erg·cm³/s |
| **non-CIE / OPIATE-CLOUDY cubes** — `lib/default/opiate/opiate_cooling_{rot,norot}_Z{1.00,0.15}_age{1.00e+06 … 1.00e+07}.dat` (+ generated `*_cube.npy`) | `registry._resolve_path_cooling_nonCIE` (`:241`), sentinel `def_dir` → `lib/default/opiate/` | `cooling/non_CIE/read_cloudy.get_coolingStructure` (`:22`) → `get_filename` (`:270`) → `create_cubes` (`:142`) | `astropy.io.ascii.read`; required columns **`ndens`, `temp`, `phi`, `cool`, `heat`** (`read_cloudy.py:186-191`; the shipped files also carry `ID`, `nedens`). Cubes are built on the axes `log10(unique ndens) × log10(unique temp) × log10(unique phi)`, each rounded to 3 decimals (`:204-219`), shape e.g. (33, 21, 22) (`:227`). Signs of `heat`/`cool` are flipped positive if the first row is negative (`:193-198`). Missing triplets stay `NaN`. Age selection: exact match, clamp at min/max, else linear interpolation between the two bracketing ages (`:319-344`); `get_fileage` (`:349`) parses the 8 chars after `age` in the filename. |
| **SPS feedback table** — `lib/default/sps/starburst99/1e6cluster_default.csv` | `registry._resolve_sps_bundle` (`:252`), sentinel `def_path`; refuses `ZCloud != 1.0` and `SB99_rotation == 0` (`:277-289`) | `sps/read_sps.read_sps` (`:38`) → `_read_sps_user` (`:134`) → `sps_columns.load_user_columns` (`:445`) | CSV, header `t,Qi,fi,Lbol,Lmech_total,pdot_W,Lmech_W`. Layout assumed by `DEFAULT_SPS_COLUMN_MAP` (`sps_columns.py:166`) as 0-based **positional** indices: 0=`t` [yr, linear], 1=`Qi` [1/s, log], 2=`fi` [dimensionless, log], 3=`Lbol` [erg/s, log], 4=`Lmech_total` [erg/s, log], 5=`pdot_W` [g·cm/s², log], 6=`Lmech_W` [erg/s, log]. Delimiter and header are auto-sniffed by `_scan_layout` (`:385`). `sps_refmass` resolves to `1e6` only for this bundled file (`registry.py:307-317`). |
| **user SPS file** — arbitrary `.txt`/`.csv` | `sps_path` in `.param`; column map from the 13 `sps_col_<canonical>` params via `sps_columns.build_user_column_map` (`:254`) and `validate_user_column_map` (`:278`) | same loader | each `sps_col_*` value parses to a `ColumnSpec(file_column, units, log)` (`parse_sps_col_value`, `:213`); `file_column` is a 0-based int or a header name. `validate_t_monotonic` (`:334`) enforces strictly-increasing `t`. |
| **f_kappa calibration grid** | none — hard-coded array | `_input/fkappa_auto.py:49` `_F_FIRE`, axes `_LOG_M/_LOG_SFE/_LOG_N` (`:40-42`) | 3×3×7 table transcribed from `docs/dev/transition/pdv-trigger/data/fkappa_nH_sweep.csv`; used only when `cooling_boost_kappa == 'auto'` |
| **run outputs** (read side) | — | `_output/trinity_reader.py`, `_output/cloudy/run_loader.py`, `_output/show_run.py`, `_output/_metadata_io.py` | `dictionary.jsonl` (one JSON object per line) + `metadata.json` (`METADATA_VERSION = 4`, `run_constants.py:100`) |

Files a run **writes**: `<path2output>/dictionary.jsonl` (`dictionary.py:804`),
`<path2output>/metadata.json` (`dictionary.py:850` via `_metadata_io.write_metadata_atomic`,
`_metadata_io.py:78`), `<path2output>/trinity.log` (`run.py:196`),
`<path2output>/metadata_humanreadable.txt` (`dictionary.py:339`),
`<path2output>/shadow_R1_1b.csv` (`run_energy_implicit_phase.py:1440-1444`),
plus `debug_snapshot.json` from the (zero-caller) `save_debug_snapshot` (`dictionary.py:1022`).

---

## 6. Numerical solver inventory

Every SciPy/NumPy solver, root-finder, integrator and interpolator call in `trinity/` and `run.py`.

### 6.1 ODE integration

| # | site | routine | method | tolerances / options | events / terminal | solves for |
|---|---|---|---|---|---|---|
| 1 | `phase1_energy/run_energy_phase.py:299` | `scipy.integrate.solve_ivp` | `RK45` | `rtol=RTOL=1e-6` (`:58`), `atol=ATOL=1e-9` (`:59`), `dense_output=True` | `events=build_energy_phase_events(params)` → `cloud_boundary` (terminal, dir +1), `min_radius` (terminal, dir −1), `velocity_runaway` (terminal, dir −1) | `[R2, v2, Eb]` over one `SEGMENT_DURATION=3e-5` Myr segment |
| 2 | `phase1_energy/run_energy_phase.py:313` | `scipy.integrate.solve_ivp` | `RK23` | `rtol=1e-5`, `atol=1e-8` (i.e. `RTOL*10`, `ATOL*10`); no `dense_output` | same event list | retry of #1 after `solution.success == False`, on a segment shortened to `SEGMENT_DURATION/10` |
| 3 | `phase1b_energy_implicit/run_energy_implicit_phase.py:1079` | `scipy.integrate.solve_ivp(**solver_kwargs)` | `LSODA` (`ODE_METHOD`, `:176`) | `rtol=1e-6` (`:170`), `atol=1e-8` (`:171`), `max_step=DT_SEGMENT_MIN/5 = 2e-5` (`:173`), `min_step=1e-6` (`:172`, added only for LSODA at `:1076-1077`) | `events=build_implicit_phase_events(...)` → `velocity_sign` (**non-terminal**), `min_radius`, `velocity_runaway`, and `max_radius` when `stop_r` set | `[R2, v2, Eb, T0]`; `Ed`/`Td` are frozen per segment from `cool_beta_to_Ebdot_pure` / `delta2dTdt_pure` |
| 4 | `phase1c_transition/run_transition_phase.py:640` | `scipy.integrate.solve_ivp(**solver_kwargs)` | `LSODA` (`:136`) | `rtol=1e-6` (`:132`), `atol=1e-8` (`:133`), `max_step=2e-5` (`:135`), `min_step=1e-6` (`:134`) | `build_transition_phase_events(params, energy_floor=ENERGY_FLOOR=1e3)` → `energy_floor` on `y[2]` (terminal, dir −1), `min_radius`, `velocity_runaway`, `max_radius` | `[R2, v2, Eb]` with `Ed = min(Ed_energy_balance, -Eb·c_sound/R2)` (`:245`) |
| 5 | `phase2_momentum/run_momentum_phase.py:722` | `scipy.integrate.solve_ivp(**solver_kwargs)` | `LSODA` (`:128`) | `rtol=1e-6` (`:124`), `atol=1e-8` (`:125`), `max_step=2e-5` (`:127`), `min_step=1e-6` (`:126`) | `build_momentum_phase_events` → `min_radius`, `velocity_runaway`, `max_radius` | `[R2, v2]` |
| 6 | `bubble_structure/bubble_luminosity.py:349` | `scipy.integrate.solve_ivp` | `LSODA` | `rtol=_RESIDUAL_RTOL=1e-6` (`:100`), `atol=_BUBBLE_ATOL=1e-10` (`:92`), `t_eval=np.linspace(r2Prime, R1, _RESIDUAL_NPTS=500)` (`:108`) | none | inner integration for the dMdt velocity residual `_get_velocity_residuals` (`:311`); wrapped in `_quiet_lsoda_fortran()` (`:119`) |
| 7 | `bubble_structure/bubble_luminosity.py:502` | `scipy.integrate.solve_ivp` | `LSODA` | `rtol=_BUBBLE_RTOL=1e-8` (`:91`), `atol=1e-10`, `dense_output=True` | none | the production bubble-structure solve `_solve_bubble_structure` (`:452`); `t_span=(r_array[0], r_array[-1])`, sampled via `sol.sol(r_array)` (`:519`) |
| 8 | `shell_structure/shell_structure.py:165` | `scipy.integrate.odeint` | LSODA (odeint) | `mxstep=_SHELL_ODE_MXSTEP=50000` (`:35`); default `rtol/atol` (~1.49e-8) | none | ionised shell `[nShell, phi, tau]` on `np.arange(rShell_start, rShell_stop, rShell_step)`; RHS `get_shellODE` (`get_shellODE.py:37`) with `is_ionised=True` |
| 9 | `shell_structure/shell_structure.py:324` | `scipy.integrate.odeint` | LSODA (odeint) | `mxstep=50000`; default tolerances | none | neutral shell `[nShell, tau]`, RHS `get_shellODE` with `is_ionised=False` |
| 10 | `cloud_properties/bonnorEbertSphere.py:254` | `scipy.integrate.odeint` | LSODA (odeint) | defaults, `tfirst=False` | none | isothermal Lane-Emden `[u, du/dxi]` on `np.logspace(log10(1e-7), log10(20), 5000)` (`XI_MIN/XI_MAX/N_POINTS`) |

### 6.2 Root finding / optimisation

| # | site | routine | method | tolerances / options | solves for |
|---|---|---|---|---|---|
| 11 | `bubble_structure/bubble_luminosity.py:261` | `scipy.optimize.fsolve` | hybrd | `xtol=1e-4`, `factor=50`, `epsfcn=1e-4` | `bubble_dMdt` (shell→bubble conduction mass flux) from `_get_velocity_residuals`; seeded from `params['bubble_dMdt']` or `_get_init_dMdt` (`:297`, Weaver+77 Eq. 33). Failures return the deterministic penalty `_SOLVER_FAIL_RESIDUAL = 1e3` (`:84`); a non-monotonic `T` returns `1e2` (`:382`); `NaN` min-T returns `-1e3` (`:378`); `min_T < 3e4` scales the residual by `(3e4/(min_T+0.1))**2` (`:374`) |
| 12 | `bubble_structure/bubble_luminosity.py:724` | `scipy.optimize.brentq` | Brent | `xtol=1e-8`, bracket `[min(r_interp), max(r_interp)]` | `r_CIEswitch`, the radius where the cubic `interp1d` of `T - 10**5.5` crosses zero |
| 13 | `bubble_structure/get_bubbleParams.py:445` | `scipy.optimize.brentq` | Brent | default `xtol=2e-12`, `rtol≈8.9e-16`; bracket `[0.0, R2]` | `R1` (wind termination shock) as the root of `get_r1` (`:384`). Short-circuits to `0.0` when `Lmech_total <= 0` or `not (R2 > 0)`; raises `ValueError` on non-finite `Eb/Lmech/v_mech` with physical `R2` (`:439-443`) |
| 14 | `phase1b_energy_implicit/get_betadelta.py:983` | `scipy.optimize.root` | `'hybr'` | `options=HYBR_OPTIONS = dict(xtol=1e-8, factor=0.1, maxfev=30, eps=3e-4)` (`:74`) | `(beta, delta)` from the pole-free residual `gvec = (gE, gT)` (`_hybr_g_residual`, `:879`). Acceptance gate: `_NoPhysicalRoot` (a `BaseException`, `:869`) raised when the structure solve fails or `dMdt <= 0`; convergence is `gE**2+gT**2 < RESIDUAL_THRESHOLD = 1e-4` (`:47`) |
| 15 | `phase1b_energy_implicit/get_betadelta.py:1131` | `scipy.optimize.minimize` | `'L-BFGS-B'` | `bounds=[(0.0,1.0), (-1.0,0.0)]` (`BETA_MIN/MAX`, `DELTA_MIN/MAX`, `:41-44`), `options={'maxiter': MAX_ITERATIONS=15, 'ftol': 1e-8, 'gtol': 1e-6}` | `(beta, delta)` minimising `Edot_res**2 + T_res**2`; only reached on the `legacy` solver when the grid residual exceeds `LBFGSB_FALLBACK_THRESHOLD = 5.0` (`:53`, gate at `:771`) |
| — | `phase1b_energy_implicit/get_betadelta.py:1010` `_solve_grid` | hand-rolled | 5×5 `np.linspace` grid | `GRID_SIZE=5`, `GRID_EPSILON=0.02` (`:56-57`), early exit below `GRID_EARLY_EXIT_RESIDUAL = RESIDUAL_THRESHOLD/10` (`:68`), center-out scan order (`:1055-1059`) | `(beta, delta)`; not a SciPy call but the production `legacy` search |

### 6.3 Interpolators

| # | site | routine | kind / method | axes |
|---|---|---|---|---|
| 16 | `main.py:167` | `scipy.interpolate.interp1d` | `'linear'` | `logT → logLambda` (CIE cooling), stored in `cStruc_cooling_CIE_interpolation` |
| 17–26 | `sps/read_sps.py:341,342,343,344,347,348,349,352,353,354` | `scipy.interpolate.interp1d` ×10 | `kind=ftype`, default `'cubic'` (`:285`) | `t_Myr →` `Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total`; no `bounds_error`/`fill_value` given, so out-of-range raises. `get_current_sps_feedback` (`update_feedback.py:98`) pre-checks the range at `:153-159` |
| 27 | `cooling/non_CIE/read_cloudy.py:98` | `scipy.interpolate.RegularGridInterpolator` | `method='linear'` | `(log ndens, log temp, log phi) → log10(cool_cube)` |
| 28 | `cooling/non_CIE/read_cloudy.py:100` | `scipy.interpolate.RegularGridInterpolator` | `method='linear'` | same axes → `log10(heat_cube)` |
| 29 | `cooling/non_CIE/read_cloudy.py:136` | `scipy.interpolate.RegularGridInterpolator` | default (`'linear'`) | same axes → `netcooling = cool_cube - heat_cube` (**not** logged); stored in `cStruc_net_nonCIE_interpolation` |
| 30 | `_input/fkappa_auto.py:72` | `RegularGridInterpolator` | `method='linear'` | `(log10 mCloud_input, log10 sfe, log10 nCore) → log10 f_kappa_fire`; coordinates clamped to the hull in `fkappa_fire` (`:81-94`) |
| 31 | `cloud_properties/bonnorEbertSphere.py:270` | `scipy.interpolate.interp1d` | `'cubic'`, `bounds_error=False`, `fill_value=(1.0, rho_rhoc[-1])` | `xi → rho/rho_c` |
| 32 | `cloud_properties/bonnorEbertSphere.py:275` | `scipy.interpolate.interp1d` | `'cubic'`, `bounds_error=False`, `fill_value=(0.0, m[-1])` | `xi → m = xi²·du/dxi` |
| 33 | `cloud_properties/bonnorEbertSphere.py:283` | `scipy.interpolate.interp1d` | `'cubic'`, `bounds_error=False`, `fill_value=(xi[-1], xi[0])` | inverse `rho/rho_c → xi` (built on `np.unique` of the reversed array) |
| 34 | `bubble_structure/bubble_luminosity.py:718` | `interp1d` | `'linear'` | `r → dTdr` on the leading `index_CIE_switch + 20` points |
| 35 | `bubble_structure/bubble_luminosity.py:720` | `interp1d` | `'cubic'` | `r → T - 10**5.5` (the function handed to brentq #12) |
| 36 | `bubble_structure/bubble_luminosity.py:721` | `interp1d` | `'linear'` | `r → v` |
| 37 | `bubble_structure/bubble_luminosity.py:803` | `interp1d` | `'linear'` | 2-point `r → T` across the intermediate (1e4 K) region |
| 38 | `_output/trinity_reader.py:851` / `:879` | `scipy.interpolate.interp1d` | see source | reader-side profile resampling |
| 39 | `cooling/net_coolingcurve.py:194` | `np.interp` | linear | blends `dudt_nonCIE` ↔ `dudt_CIE` across `[nonCIE_Tcutoff, CIE_Tcutoff]` in `log10 T` |
| 40 | `phase0_init/get_InitCloudProp.py:509` | `np.interp` | linear | `M(rCloud)` from `(r_arr, M_arr)` in `verify_mass_at_rCloud` (`:485`) |
| 41 | `_output/cloudy/dlaw.py:261` | `np.interp` | linear | inner density fill in `_densify_preserving_edges` (`:205`) |
| 42 | `_functions/simplify.py:739`, `:833`, `:880` | `np.interp` | linear | R² reconstruction inside `_simplify` (`:290`) / `_simplify_error` (`:754`) |
| 43 | `_analysis/check_yesno.py:119-120` | `np.interp` | linear | R2(t) resampling onto a common grid |

### 6.4 Quadrature

| # | site | routine | integrand |
|---|---|---|---|
| 44 | `bubble_structure/bubble_luminosity.py:748` | `_trapezoid` (`np.trapezoid`/`np.trapz`) | `chi_e·n²·Lambda·4πr²` over the CIE bubble zone → `L_bubble` |
| 45 | `bubble_structure/bubble_luminosity.py:750` | `_trapezoid` | `r²·T` over the bubble zone → `Tavg_bubble` |
| 46 | `bubble_structure/bubble_luminosity.py:795` | `_trapezoid` | `dudt·4πr²` over the conduction band, sampled at `_CONDUCTION_NPTS = 2000` (`:116`) points from the dense-output solution → `L_conduction` |
| 47 | `bubble_structure/bubble_luminosity.py:797` | `_trapezoid` | `r²·T` over the conduction band |
| 48 | `bubble_structure/bubble_luminosity.py:835` | `_trapezoid` | intermediate-region cooling, per CIE / non-CIE mask, on 1000 points (`:809`) |
| 49 | `bubble_structure/bubble_luminosity.py:837` | `_trapezoid` | `r²·T` over the intermediate region |
| 50 | `bubble_structure/bubble_luminosity.py:936` | `scipy.integrate.cumulative_trapezoid` | `4π·rho·r²` with `initial=0` → cumulative bubble mass in `_get_mass_and_grav` (`:915`); the two gravity outputs are returned as `None` (`:947-948`) |
| 51 | `shell_structure/shell_structure.py:266` | `scipy.integrate.simpson` | `r·rho` over the ionised shell → `grav_ion_phi` |
| 52 | `shell_structure/shell_structure.py:370` | `scipy.integrate.simpson` | `r·rho` over the neutral shell → `grav_neu_phi` |
| 53 | `cloud_properties/mass_profile.py:419` | `scipy.integrate.trapezoid` | `4π·r²·rho` — numerical fallback in `compute_enclosed_mass_bonnor_ebert` (`:347`) when the analytic Lane-Emden route is unavailable; called in a Python `for` loop over each radius (`:415-421`) |

### 6.5 Non-SciPy numerical helpers on the hot path

| site | what |
|---|---|
| `_functions/operations.py:146` `find_nearest_higher` | directional grid search used to split the bubble profile into CIE / conduction / intermediate regions (`bubble_luminosity.py:708-709`); raises `MonotonicError` (`:186`) unless `_is_monotonic_or_tolerable` (`:99`) accepts the profile (rtol 1e-2, boundary 1 %, spike ≤ 2) |
| `_functions/operations.py:19` `find_nearest` | argmin of `|array - value|`; used at `bubble_luminosity.py:878`, `:884` |
| `_functions/operations.py:189` `get_soundspeed` | `sqrt(gamma·k_B·T/mu)` with `mu = mu_ion` (T > 1e4 K) else `mu_atom` |
| `bubble_luminosity.py:531` `_create_radius_grid` | three stitched `np.logspace` blocks of 2e4 points each → ~60k-point decreasing grid, then `_clean_radius_grid` (`:570`, `MIN_SPACING = 1e-12`) |
| `_functions/simplify.py:290` `_simplify` | curve downsampling to `simplify_npoints` (default 100) applied to the nine profile arrays at snapshot time |

---

## 7. Test-to-source mapping

`test/` holds **52 `test_*.py` files** plus `test/CLAUDE.md` and `test/data/`.

### 7.1 Forward map

| test file | source modules it imports / drives |
|---|---|
| `test_active_when.py` | `_input/registry.py`, `_input/dictionary.py` |
| `test_bench_theta_cum.py` | none in `trinity/` — loads `docs/dev/transition/pdv-trigger/data` scripts |
| `test_betadelta_dt_mitigation.py` | `phase1b_energy_implicit/run_energy_implicit_phase.py`, `_input/dictionary.py`, `_input/registry.py` |
| `test_betadelta_hybr.py` | `phase1b_energy_implicit/get_betadelta.py`, `bubble_structure/bubble_luminosity.py` |
| `test_betadelta_hybr_stress.py` | full pipeline via `subprocess` on `run.py` (marked `stress`) |
| `test_betadelta_solver.py` | `phase1b_energy_implicit/get_betadelta.py`, `bubble_structure/bubble_luminosity.py` |
| `test_betadelta_solver_switch.py` | `phase1b_energy_implicit/get_betadelta.py`, `_input/registry.py`, `_input/errors.py` |
| `test_bubble_lsoda_quiet.py` | `bubble_structure/bubble_luminosity.py` (`_quiet_lsoda_fortran`) |
| `test_bubble_solver_failures.py` | `bubble_structure/bubble_luminosity.py` |
| `test_bubble_solver_stress.py` | full pipeline via `subprocess` on `run.py` |
| `test_cf_leak.py` | `bubble_structure/get_bubbleParams.py`, `_input/registry.py`, `_functions/unit_conversions.py` |
| `test_cloudy_cli.py` | `_output/cloudy/trinity_to_cloudy.py`, `_output/cloudy/run_loader.py` |
| `test_cloudy_dlaw.py` | `_output/cloudy/dlaw.py`, `_functions/unit_conversions.py` |
| `test_cloudy_run_loader.py` | `_output/cloudy/run_loader.py`, `_output/cloudy/dlaw.py`, `_output/cloudy/snapshot_to_deck.py` |
| `test_cloudy_snapshot_to_deck.py` | `_output/cloudy/snapshot_to_deck.py`, `_output/cloudy/dlaw.py`, `_output/cloudy/run_loader.py` |
| `test_conventional_units.py` | `_functions/unit_conversions.py`, `_output/run_constants.py`, `_output/simulation_end.py`, `sps/sps_columns.py`, `phase0_init/get_InitPhaseParam.py` |
| `test_cooling_boost.py` | `phase1b_energy_implicit/get_betadelta.py` (`effective_Lloss`) |
| `test_dR2min_magic_number.py` | `bubble_structure/bubble_luminosity.py`, `cooling/non_CIE/read_cloudy.py`, `_input/read_param.py` |
| `test_docs_dev_conventions.py` | none in `trinity/` — checks `docs/dev/` file conventions |
| `test_energy_collapse_guard.py` | `bubble_structure/get_bubbleParams.py`, `_output/simulation_end.py`, `phase1b_energy_implicit/run_energy_implicit_phase.py` |
| `test_energy_collapse_snapshot.py` | full pipeline via `subprocess` on `run.py` |
| `test_engine_purity.py` | AST scan of `trinity/` imports (no runtime import) |
| `test_fA_source_boost.py` | `bubble_structure/bubble_luminosity.py`, `cooling/non_CIE/read_cloudy.py`, `cooling/net_coolingcurve.py`, `_input/read_param.py`, `_input/registry.py` |
| `test_fkappa_auto.py` | `_input/fkappa_auto.py`, `_input/read_param.py` |
| `test_gen_default_param.py` | `_input/registry.py` |
| `test_log_stopping_fate.py` | `_output/terminal_prints.py`, `_functions/unit_conversions.py` |
| `test_logging_dedup.py` | `_functions/logging_setup.py` |
| `test_materialize_runtime.py` | `_input/registry.py`, `_input/dictionary.py` |
| `test_metadata.py` | `_input/dictionary.py`, `_output/run_constants.py`, `_output/trinity_reader.py`, `_output/simulation_end.py` |
| `test_mu_audit_drift.py` | `_functions/unit_conversions.py`, `_input/read_param.py`, `shell_structure/get_shellODE.py`, `_functions/operations.py`, `cloud_properties/bonnorEbertSphere.py` |
| `test_net_coolingcurve.py` | `cooling/net_coolingcurve.py`, `cooling/non_CIE/read_cloudy.py`, `_input/read_param.py`, `_functions/unit_conversions.py` |
| `test_operations_monotonic.py` | `_functions/operations.py` |
| `test_phase_boundary.py` | full pipeline via `subprocess` on `run.py` |
| `test_phase_events.py` | `phase_general/phase_events.py` |
| `test_phase_helper_sync.py` | AST comparison of `run_energy_implicit_phase.py` / `run_transition_phase.py` / `run_momentum_phase.py` (no import) |
| `test_r1_bracket.py` | `bubble_structure/get_bubbleParams.py`, `phase1b_energy_implicit/get_betadelta.py` |
| `test_r1_shadow.py` | `phase1b_energy_implicit/run_energy_implicit_phase.py` |
| `test_read_sps.py` | `sps/read_sps.py`, `sps/sps_columns.py`, `_input/registry.py`, `_functions/unit_conversions.py` |
| `test_registry.py` | `_input/registry.py`, `_input/param_spec.py`, `_input/read_param.py`, `_input/dictionary.py`, `_output/run_constants.py` |
| `test_residual_resample.py` | `bubble_structure/bubble_luminosity.py`, `_functions/operations.py`, `cooling/non_CIE/read_cloudy.py`, `_input/read_param.py` |
| `test_resolvers.py` | `_input/registry.py`, `_input/errors.py`, `sps/sps_columns.py` |
| `test_rosette_cf_harness.py` | none in `trinity/` — checks `docs/dev/rosette-cf/harness/` |
| `test_run_smoke.py` | full pipeline via `subprocess` on `run.py` |
| `test_shell_overflow_guard.py` | `shell_structure/get_shellODE.py`, `_input/read_param.py` |
| `test_show_run.py` | `_output/show_run.py`, `_input/dictionary.py`, `_output/run_constants.py`, `_output/simulation_end.py`, `_functions/unit_conversions.py` |
| `test_simplify.py` | `_functions/simplify.py` |
| `test_sweep_jobs.py` | `_input/sweep_jobs.py`, `_input/sweep_parser.py` |
| `test_sweep_workers.py` | `run.py`, `_functions/cluster.py` |
| `test_theta5_harvest.py` | none in `trinity/` — checks `docs/dev/transition/pdv-trigger/runs/` scripts |
| `test_unit_conversions.py` | `_functions/unit_conversions.py` |
| `test_validate_gmc.py` | `cloud_properties/validate_gmc.py`, `cloud_properties/bonnorEbertSphere.py`, `_functions/unit_conversions.py` |
| `test_validators.py` | `_input/registry.py`, `_input/read_param.py`, `_input/errors.py` |

### 7.2 Inverse map — source modules **no test imports**

31 of the 73 source files (72 in `trinity/` + `run.py`) are never imported by any test. Splitting
by whether the end-to-end `subprocess` tests (`test_run_smoke.py`, `test_phase_boundary.py`,
`test_energy_collapse_snapshot.py`, `test_bubble_solver_stress.py`,
`test_betadelta_hybr_stress.py`) execute them as part of a full `run.py` run:

**Executed by the subprocess end-to-end tests, but never imported directly by any test:**

`trinity/main.py`, `trinity/phase0_init/get_InitCloudProp.py`,
`trinity/phase1_energy/energy_phase_ODEs.py`, `trinity/phase1_energy/run_energy_phase.py`,
`trinity/phase1c_transition/run_transition_phase.py`,
`trinity/phase2_momentum/run_momentum_phase.py`, `trinity/shell_structure/shell_structure.py`,
`trinity/sps/update_feedback.py`, `trinity/cloud_properties/mass_profile.py`,
`trinity/cloud_properties/density_profile.py`, `trinity/cloud_properties/powerLawSphere.py`,
`trinity/cooling/CIE/read_coolingcurve.py`, `trinity/_output/_metadata_io.py`,
`trinity/_output/header.py`, `trinity/_input/sweep_runner.py` (sweep path only), and the
package `__init__.py` files (`trinity/__init__.py`, `_analysis/`, `_input/`, `_output/`,
`cloud_properties/`, `cooling/CIE/`, `cooling/non_CIE/`, `phase0_init/`, `phase1_energy/`,
`phase1b_energy_implicit/`, `phase1c_transition/`, `phase2_momentum/`, `shell_structure/`).

**Not reachable from `run.py` at all and not imported by any test** (standalone tools/analysis):

`trinity/_analysis/check_yesno.py`, `trinity/_functions/extract_example_snapshots.py`.

**Reachable but only through a lazy import in the reader:**
`trinity/cloud_properties/initial_profile.py` — its sole consumer is
`TrinityOutput.initial_cloud_profile` (`trinity_reader.py:618-629`).

For reference, the modules **not reachable from `run.py`** by any import edge (including
function-local imports) are: `_analysis/__init__.py`, `_analysis/check_yesno.py`,
`_functions/__init__.py`, `_functions/extract_example_snapshots.py`, `_output/cloudy/__init__.py`,
`_output/cloudy/dlaw.py`, `_output/cloudy/run_loader.py`, `_output/cloudy/snapshot_to_deck.py`,
`_output/cloudy/trinity_to_cloudy.py`, `bubble_structure/__init__.py`, `cooling/CIE/__init__.py`,
`cooling/non_CIE/__init__.py`, `phase_general/__init__.py`. (The five `cloudy/` files *are*
covered by four dedicated test files.)

---

## 8. OBSERVATIONS (unjudged)

Neutral factual statements only. No recommendation is implied by inclusion here.

### 8.1 Zero-reference definitions

The following 34 definitions have no reference anywhere in `trinity/`, `run.py`, or `test/`
(docstring/example mentions excluded; each was re-checked with `grep`):

| definition | site |
|---|---|
| `get_module_logger` | `_functions/logging_setup.py:330` |
| `setup_logging_from_params` | `_functions/logging_setup.py:402` |
| `find_nearest_lower` | `_functions/operations.py:30` (the source comment at `:79-83` says the sibling `find_nearest_higher` guard is deliberately retained as a fallback) |
| `DescribedDict.load_latest_snapshot` | `_input/dictionary.py:967` |
| `save_debug_snapshot` | `_input/dictionary.py:1022` |
| `load_debug_snapshot` | `_input/dictionary.py:1128` |
| `specs_by_category` | `_input/registry.py:541` |
| `read_sweep_param` | `_input/sweep_parser.py:262` |
| `SweepProgress` (class) and `SweepProgress.eta` | `_input/sweep_runner.py:48`, `:61` |
| `SimulationEndCode.is_inspection_required` | `_output/simulation_end.py:117` (a test asserts the code band numerically, `test_energy_collapse_guard.py:75-81`, without calling the method) |
| `terminal_prints.bubble` | `_output/terminal_prints.py:30` |
| `terminal_prints.shell` | `_output/terminal_prints.py:45` |
| `log_file_saved`, `log_warning`, `log_error` | `_output/terminal_prints.py:106`, `:111`, `:116` |
| `TrinityOutput.to_dataframe` | `_output/trinity_reader.py:1035` |
| `iter_progress` | `_output/trinity_reader.py:1098` |
| `find_data_file` | `_output/trinity_reader.py:1133` |
| `resolve_data_input` | `_output/trinity_reader.py:1258` |
| `get_unique_ndens` | `_output/trinity_reader.py:1420` |
| `organize_simulations_for_grid` | `_output/trinity_reader.py:1443` |
| `info_simulations` | `_output/trinity_reader.py:1522` |
| `delta2dTdt` | `bubble_structure/get_bubbleParams.py:27` (the pure twin `delta2dTdt_pure`, `get_betadelta.py:272`, is the one used) |
| `dTdt2delta` | `bubble_structure/get_bubbleParams.py:47` |
| `cool_beta_to_Ebdot` | `bubble_structure/get_bubbleParams.py:69` (the pure twin `cool_beta_to_Ebdot_pure`, `get_betadelta.py:182`, is the one used) |
| `Ebdot_to_cool_beta` | `bubble_structure/get_bubbleParams.py:140` |
| `xi2r` | `cloud_properties/bonnorEbertSphere.py:622` (the sibling `r2xi`, `:582`, is referenced) |
| `compute_mass_accretion_rate` | `cloud_properties/mass_profile.py:437` |
| `validate_mass_at_rCloud` | `cloud_properties/mass_profile.py:488` |
| `compute_minimum_rCore` | `cloud_properties/mass_profile.py:566` |
| `compute_consistent_params` | `cloud_properties/powerLawSphere.py:214` |
| `expansion_next` | `main.py:366` (body is a bare `return`) |
| `get_beta_delta_wrapper_pure` | `phase1b_energy_implicit/get_betadelta.py:1152` |

### 8.2 Structural observations

1. `main.run_expansion` (`main.py:216`) calls four phase runners and discards each return value;
   the only inter-phase signal is `params['EndSimulationDirectly']`
   (`main.py:283`, `:303`, `:343`).
2. `build_implicit_phase_events` (`phase_events.py:458`) returns a 2-tuple
   `(events, cooling_balance_factory)`. The caller unpacks both
   (`run_energy_implicit_phase.py:752`) but `cooling_balance_factory` is never used again in that
   file; the cooling-balance test is the inline ratio comparison at `:1296`.
   `make_cooling_balance_event` (`phase_events.py:319`) is referenced twice: that factory
   construction (`phase_events.py:497`) and `test/test_phase_events.py:18`.
3. `compute_max_dex_change` and `get_monitor_values` exist as three textually independent copies
   (`run_energy_implicit_phase.py:289,412`; `run_transition_phase.py:143,164`;
   `run_momentum_phase.py:135,156`), as do the three `ForceProperties` dataclasses
   (`run_energy_implicit_phase.py:444`, `run_transition_phase.py:255`, `run_momentum_phase.py:190`)
   and the three `ADAPTIVE_MONITOR_KEYS` lists. `test/test_phase_helper_sync.py` asserts the AST of
   the `compute_max_dex_change` bodies matches across the three.
4. `compute_forces_pure` in `run_energy_implicit_phase.py:460` and `run_transition_phase.py:271`
   have the same name and near-identical bodies; they differ in the `P_drive` expression
   (`max(Pb, P_HII)` vs `max(Pb, P_HII + P_ram)`) and in whether `P_ram` is computed
   (`run_transition_phase.py:329`) or hard-set to `0.0` (`run_energy_implicit_phase.py:559`).
5. `_get_mass_and_grav` (`bubble_luminosity.py:915`) returns `grav_phi = None` and
   `grav_force_m = None`; the gravity block is commented out at `:939-946`. The sole caller
   (`:891`) unpacks `m_cumulative, _, _`.
6. `params['P_HII']` is assigned twice per segment in every phase runner: once inline from
   `shell_props.n_IF_Str` (e.g. `run_energy_implicit_phase.py:984`) and once from
   `force_props.P_HII` (`:1004`); `compute_forces_pure` itself reads `params['P_HII']`
   (`run_energy_implicit_phase.py:531`, `run_transition_phase.py:324`,
   `run_momentum_phase.py:264`), so the second write is the value the first write produced.
7. `run_energy_phase.run_energy` imports `parse_transition_triggers` and
   `effective_Lloss_from_params` inside the loop body (`run_energy_phase.py:273-274`), executed
   once per segment.
8. `read_param` Step 4 (`read_param.py:270`) constructs `DescribedItem` without
   `exclude_from_snapshot`; Step 9 (`:455-457`) then sets `exclude_from_snapshot = True` on every
   key not in the 10-name `time_varying_keys` list, and Step 10 (`materialize_runtime`,
   `registry.py:624`) adds runtime keys afterwards with the flag taken from the spec, bypassing
   Step 9's sweep. The ordering is documented in `registry.py:636-641`.
9. `DescribedDict.save_snapshot` (`dictionary.py:711`) skips the write when the previous pending
   snapshot has identical `t_now` **and** `R2` (`:721-731`). Three of the four phase runners
   compensate by comparing `params.save_count` before/after to decide whether to increment
   `_snapshots_after_rCloud` (`run_energy_implicit_phase.py:1016,1023-1026`;
   `run_transition_phase.py:592,597-600`; `run_momentum_phase.py:674,679-682`).
   `run_energy_phase.py` has no such counter increment.
10. `DescribedDict.__init__` registers an `atexit` handler and replaces the `SIGINT`/`SIGTERM`
    handlers on every construction (`dictionary.py:284-288`), including for every
    `DescribedDict.load_snapshot` (`dictionary.py:951`) and `load_snapshots` consumer.
11. `run.py::run_sweep` installs its own `SIGINT`/`SIGTERM` handlers at `run.py:600-607` and
    restores them at `:697-702`; the parent sweep process also constructs no `DescribedDict`, so
    the two handler installations are in different processes.
12. `SPSFeedback.t` and `ShellProperties.diss_condition_met` are dataclass fields with no
    corresponding `ParamSpec`, so `updateDict` (`dictionary.py:1268`) silently drops them.
    `diss_condition_met` is computed at `shell_structure.py:446` and returned at `:464`; the
    dissolution decision in phases 1c/2 instead re-derives `shell_nMax < nISM` inline
    (`run_transition_phase.py:806-808`, `run_momentum_phase.py:859-861`).
13. `phase_events.make_velocity_runaway_event` (`:166`) sets `event.direction = -1` for **all
    three** `direction` branches, including `"expansion"` (`:199`) and `"both"` (`:205`). Only
    `direction="collapse"` is constructed in the four builders.
14. `MAX_VELOCITY_EXPANSION = 1000.0` (`phase_events.py:74`) is defined and never referenced.
15. `main.py:263-272` handles `stop_at_rCloud_nSnap == 0` after phase 1a; `nSnap_rCloud > 0` is
    handled inside phases 1b/1c/2 via `_snapshots_after_rCloud`, which is incremented only in
    those three runners.
16. `_get_velocity_residuals` (`bubble_luminosity.py:311`) returns four distinct sentinel/penalty
    magnitudes on rejection: `_SOLVER_FAIL_RESIDUAL = 1e3` (`:334`, `:359`, `:361`, `:363`),
    `residual * (3e4/(min_T+0.1))**2` (`:374`), `-1e3` (`:378`), and `1e2` (`:382`).
17. `net_coolingcurve.get_dudt` (`:58`) mutates its `ndens` and `phi` arguments in place with
    `/=` (`:82-83`) before use.
18. `read_cloudy.get_coolingStructure` reads `params['t_now']` as `params['t_now'] * 1e6`
    (`:48`) — the `DescribedItem.__mul__` operator overload, not `.value`.
19. `create_cubes` (`read_cloudy.py:142`) writes a `<stem>_cube.npy` next to the source `.dat`
    (`:265`) and prefers it on subsequent calls (`:174`). The bundled `lib/default/opiate/`
    directory already contains 11 such `.npy` files.
20. `registry._resolve_sps_bundle` (`:252`) hands the module-level
    `sps_columns.DEFAULT_SPS_COLUMN_MAP` dict object directly into
    `params['sps_column_map']` (`:293`, `:319`) without copying.
21. `TrinityOutput.PARAM_DOCS` (`trinity_reader.py:140`) documents four keys —
    `t_next`, `F_ISM`, `bubble_dMdtGuess`, `shell_interpolate_massDot` — that are never written
    in `trinity/` (see §3.6c).
22. `run_energy_implicit_phase.py` writes a sideline CSV `shadow_R1_1b.csv` to
    `params['path2output']` at `:1435-1448`, accumulating one `shadow_rows` dict per segment in
    memory (`:1264-1272`) for the whole phase.
23. `phase1b` computes `Ed` and `Td` once per segment (`run_energy_implicit_phase.py:992-993`) and
    passes them as constants into `get_ODE_implicit_pure` (`:586`), so `dEb/dt` and `dT0/dt` are
    piecewise-constant across each segment while `dR2/dt` and `dv2/dt` are evaluated at every RHS
    call.
24. `get_ODE_momentum_pure` (`run_momentum_phase.py:373`) calls `get_current_sps_feedback(t,
    params)` on every RHS evaluation (`:407`) while the rest of its inputs come from the frozen
    `MomentumODESnapshot`.
25. `shell_structure_pure` (`shell_structure.py:85`) contains two unbounded `while` loops
    (`:157`, `:316`) whose exit depends on `is_allMassSwept` / `is_phiDepleted` becoming True.
26. `run_energy_phase.py:342-344` clears `EarlyPhaseApproximation` only when `loop_count == 0`;
    while it is True, `get_ODE_Edot_pure` overrides the computed acceleration with
    `vd = -1e8` (`energy_phase_ODEs.py:269-270`).
27. `params['mCloud']` is rebound at `read_param.py:389` from the pre-SFE input mass to the
    post-SFE residual mass; `mCloud_input` and `mCluster` hold the other two quantities
    (`:390-400`). The behaviour is documented in the comment at `:376-385`.
28. `simulation_end.write_simulation_end` (`:130`) is called inside a `try/except Exception` in
    `main.py:190-198`, which sets `exit_code = 99` on failure; `main.start_expansion` then
    `return 0` regardless (`main.py:211`).
29. `trinity/_output/cloudy/` (5 modules, 4 test files) and `trinity/_analysis/check_yesno.py`
    and `trinity/_functions/extract_example_snapshots.py` are not imported anywhere in the
    `run.py` execution path.
30. `test/test_phase_helper_sync.py:23-25` names the three phase-runner copies and asserts their
    `compute_max_dex_change` bodies are AST-identical; there is no equivalent assertion for
    `get_monitor_values`, `ForceProperties`, or `ADAPTIVE_MONITOR_KEYS`.
