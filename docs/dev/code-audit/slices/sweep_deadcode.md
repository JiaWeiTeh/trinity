# Sweep: dead code and unused contracts

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

Scope: `trinity/**` (72 modules, 26 359 lines, 519 definitions), with `test/`, `run.py`, `tools/`,
`paper/` and `docs/source/` consulted only to establish whether a caller exists. Read-only; nothing
was edited.

**Dynamic-dispatch ruling, done once for the whole sweep.** Every `getattr`/`setattr` site in the
package was enumerated (`grep -rn "getattr\|globals()\|locals()\|__dict__\|eval(\|exec(\|importlib\|setattr" trinity/`,
27 hits). All of them take a *literal* attribute name (`getattr(item, "value", …)`,
`getattr(event, 'name', …)`, `getattr(np, 'trapezoid', None)`, `getattr(logging, level.upper())`).
There is **no** `globals()[…]`, no `eval`/`exec`, no `importlib`, and no string-keyed callable
registry anywhere in `trinity/` (checked for `Callable] = {`, `DISPATCH`, `_REGISTRY =`, `HANDLERS`,
and for dict literals mapping strings to bare identifiers — no matches). As a positive control, every
name claimed dead below was additionally grepped **as a string literal** across
`trinity/ test/ tools/ run.py paper/` — all returned 0 occurrences. Individual entries therefore say
"no string-literal form" rather than repeating this apparatus.

The one genuinely dynamic lookup in the package is the `sps_col_*` prefix family; it is resolved in
DC-101 (they are **not** orphans).

---

### DC-001 · CIE cooling accepts a `metallicity` argument, documents it as the library selector, and never reads it
- **file:line** — trinity/cooling/CIE/read_coolingcurve.py:25
- **class** — ignored-param
- **severity** — S3 misleading
- **evidence** — signature `def get_Lambda(T, cooling_CIE_interpolation, metallicity):`; the whole
  body is four statements (lines 60-64): `T = np.log10(T)`, `Lambda = 10**(cooling_CIE_interpolation(T))`,
  `return Lambda`. AST unused-argument scan reports
  `trinity/cooling/CIE/read_coolingcurve.py:25 get_Lambda(T, cooling_CIE_interpolation, metallicity) UNUSED=['metallicity']`.
  The docstring nonetheless states: `metallicity : float / Cloud metallicity (selects/validates against the CIE library).`
  Both production call sites pass a real value —
  `trinity/cooling/net_coolingcurve.py:163  Lambda_CIE = CIE.get_Lambda(T, CIE_interp, params_dict['ZCloud'].value)`
  and `:186  Lambda = CIE.get_Lambda(10**CIE_Tcutoff, CIE_interp, params_dict['ZCloud'].value)`.
  The module even carries `# TODO: add for non-solar metallicity` at line 20, contradicting its own
  docstring 5 lines later.
- **ruled out** — Not dynamic: `metallicity` is a positional parameter, never re-read via `locals()`
  or `**kwargs` (the function takes no `**kwargs`). Not test-only: both callers are production.
  Not external API: `grep -rn "get_Lambda" docs/source/ README.md` → 0.
- **consequence** — Table *selection* by metallicity does happen, but upstream in
  `read_param.py:417-429`, not here. Inside the cooling call the argument is inert, so a reader
  auditing the CIE path sees a metallicity-aware interface that is a no-op, and the docstring's
  "selects/validates" is false. Combined with DC-002 the practical situation is: TRINITY only ever
  runs at Z=1, and this signature suggests otherwise.
- **confidence** — high

### DC-002 · The entire `ZCloud = 0.15` code path is unreachable — the validator rejects any ZCloud != 1
- **file:line** — trinity/_input/read_param.py:426 (and trinity/cooling/non_CIE/read_cloudy.py:297, :300)
- **class** — unreachable
- **severity** — S3 misleading
- **evidence** — `trinity/_input/registry.py:99`:
  ```
  def _validate_ZCloud(value, params) -> None:
      from trinity._input.errors import ParameterFileError
      if value != 1:
          raise ParameterFileError(f"Metallicity Z={value} not implemented. Currently only Z=1 (solar) is supported.")
  ```
  wired as `ParamSpec(name='ZCloud', …, validator=_validate_ZCloud)`. Ordering in `read_param`:
  `grep -n "Step 5\|validate_all\|Step 7\|resolve_all" trinity/_input/read_param.py` →
  `289: # Step 5: Validate critical parameters`, `295: validate_all(params)`,
  `403: # Step 7: Resolve sentinel`, `410: resolve_all(params)`. The ZCloud-keyed CIE-table block
  sits at 412-429, i.e. **after** validation. Therefore:
  * `read_param.py:426  elif params['ZCloud'].value == 0.15:` — unreachable;
  * `read_cloudy.py:297  elif float(metallicity) == 0.15:` — unreachable;
  * `read_cloudy.py:300-305  else: raise ValueError(f"ZCloud={metallicity} is unsupported … (available: 1.0, 0.15). … or set ZCloud to a supported value.")`
    — unreachable, and its message advertises 0.15 as available.
  The user-visible docstring in `read_coolingcurve.py:44-45` likewise says
  `4: Sutherland and Dopita 1993, for [Fe/H] = -1. Auto-pinned when ZCloud == 0.15 regardless of path_cooling_CIE.`
- **ruled out** — Not dynamic: `ZCloud` is validated through the static SPECS tuple
  (`validate_all` iterates `SPECS`, `registry.py:545+`), no monkeypatching of `_validate_ZCloud` in
  production. Not test-only: `test/test_validators.py` exercises `_validate_ZCloud` and confirms the
  `!= 1` raise. No config file in `param/` sets ZCloud to 0.15
  (`grep -rn "ZCloud" param/` → only `1`).
- **consequence** — A user reading `docs/source/parameters.rst:561` ("Per-age files are selected at
  runtime from `SB99_rotation` + `ZCloud`"), the CIE library table, or the "available: 1.0, 0.15"
  error message believes 0.15 solar is a supported configuration. It is not — the run dies at Step 5
  with a different message. Three separate implementations of a feature that cannot execute.
- **confidence** — high

### DC-003 · The beta-delta solver's `method` argument is documented as a solver selector and is never read
- **file:line** — trinity/phase1b_energy_implicit/get_betadelta.py:682 (declared), :696-699 (documented)
- **class** — ignored-param
- **severity** — S3 misleading
- **evidence** — `_solve_betadelta_legacy(beta_guess, delta_guess, params, method: str = 'grid')`
  docstring:
  ```
  method : str, optional
      Solver method: 'grid' (default, fast grid search) or 'lbfgsb' (optimizer).
      When method='grid', automatically falls back to 'lbfgsb' if grid search fails …
  ```
  Body scan of lines 678-866 for the bare token: `sed -n '678,866p' … | grep -n "\bmethod\b"` returns
  only `5: method: str = 'grid',` (the signature), `18-20:` (the docstring) and
  `149: # Determine convergence and method description` (a comment; the code there uses
  `best_method` / `method_desc`, different names). AST scan confirms:
  `get_betadelta.py:678 _solve_betadelta_legacy(…, method) UNUSED=['method']`. The control flow is
  unconditional: grid always runs (line 740), L-BFGS-B runs iff
  `not grid_converged and grid_residual > LBFGSB_FALLBACK_THRESHOLD` (line 771). `method='lbfgsb'`
  does nothing. The value is threaded one level up from
  `solve_betadelta_pure(…, method: str = 'grid')` (line 629) into both branches
  (`:639`, `:641`) and `_solve_betadelta_hybr` also ignores it — that one at least says so
  (`:955-956  "``method`` is accepted for signature parity with the legacy solver and ignored"`).
  Every production call site omits it: `run_energy_implicit_phase.py:826-830` passes three positional
  args only. The only callers that pass `method=` are `test/test_betadelta_solver_switch.py:84,108`.
- **ruled out** — Not dynamic (plain positional/keyword parameter, no `**kwargs` on either function).
  Not test-only in the sense that matters: the *parameter* is unread in both implementations, so the
  tests that pass `method="grid"` are asserting on a no-op.
- **consequence** — A maintainer trying to force the L-BFGS-B optimizer (e.g. to reproduce an old
  result, or to isolate a grid-search bug) sets `method='lbfgsb'`, gets the grid path, and draws the
  wrong conclusion. The docstring is the only documentation of this knob and it is false.
- **confidence** — high

### DC-004 · `build_implicit_phase_events` returns a cooling-balance event factory that the implicit phase never uses
- **file:line** — trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:752; factory at trinity/phase_general/phase_events.py:319
- **class** — zero-caller
- **severity** — S3 misleading
- **evidence** — `grep -rn "cooling_balance_factory\|cooling_factory" trinity/ -r --include="*.py"` →
  ```
  trinity/phase_general/phase_events.py:479:    cooling_balance_factory : callable
  trinity/phase_general/phase_events.py:497:    cooling_factory = make_cooling_balance_event(threshold=0.05)
  trinity/phase_general/phase_events.py:501:    return events, cooling_factory
  trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:751:    # Returns (events_list, cooling_balance_factory)
  trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:752:    ode_events, cooling_balance_factory = build_implicit_phase_events(params)
  ```
  The name is bound at 752 and never referenced again anywhere in the 1460-line module. The real
  cooling-balance switch is done inline at `run_energy_implicit_phase.py:1250-1254`:
  ```
  phase_switch_threshold = params.get('phaseSwitch_LlossLgain', None)
  if phase_switch_threshold and hasattr(phase_switch_threshold, 'value'):
      threshold = phase_switch_threshold.value
  else:
      threshold = 0.05
  ```
  `make_cooling_balance_event` therefore has exactly one non-test reference (line 497, inside the
  builder whose second return value is discarded); its only other callers are
  `test/test_phase_events.py`. Note the factory hardcodes `threshold=0.05` and ignores the
  `phaseSwitch_LlossLgain` parameter entirely.
- **ruled out** — Not dynamic: it is a tuple-unpacked local, not stored in `params`, not passed to
  `solve_ivp` (`grep -n "events=" run_energy_implicit_phase.py` shows only `ode_events`). Not
  test-only in production terms — the tests keep a corpse warm.
- **consequence** — The module docstring of `phase_events.py:22` advertises
  "cooling_balance: L_cool ~ L_gain (implicit -> transition)" as a *Phase-Ending Event*. It is not an
  event; nothing is armed. Anyone changing `phaseSwitch_LlossLgain` and then editing
  `make_cooling_balance_event`'s hardcoded `0.05` to match will see no effect. Two implementations
  of the transition trigger, one of them inert.
- **confidence** — high

### DC-005 · `get_effective_bubble_pressure`'s `'momentum'` branch cannot be reached
- **file:line** — trinity/bubble_structure/get_bubbleParams.py:349-351
- **class** — unreachable
- **severity** — S3 misleading
- **evidence** — the function has exactly two call sites:
  `grep -rn "get_effective_bubble_pressure" trinity/` →
  `energy_phase_ODEs.py:226` (inside `get_ODE_Edot_pure`) and
  `energy_phase_ODEs.py:362` (inside `compute_derived_quantities`); both pass
  `current_phase=snapshot.current_phase`. `snapshot.current_phase` is populated once, in
  `create_ODE_snapshot` (`energy_phase_ODEs.py:158  current_phase=params['current_phase'].value`).
  `params['current_phase']` is assigned in exactly one file:
  `grep -rn "current_phase" trinity/ | grep "\.value ="` → `main.py:244 'energy'`, `:278 'implicit'`,
  `:301 'transition'`, `:327 'momentum'`. The momentum phase never calls into `energy_phase_ODEs`;
  it uses its own `MomentumODESnapshot` / `get_ODE_momentum_pure` / `compute_forces_momentum_pure`
  and calls `get_bubbleParams.pRam` directly (`run_momentum_phase.py:225,421,585,666,889`). Hence
  `current_phase == 'momentum'` never reaches line 349.
- **ruled out** — Not dynamic: `current_phase` is a frozen dataclass field on `ODESnapshot`
  (`energy_phase_ODEs.py:59`), never mutated. Not test-only: no test calls
  `get_effective_bubble_pressure` at all (`grep -rn get_effective_bubble_pressure test/` → 0).
- **consequence** — The docstring says "This function MUST be called in both the ODE and in
  compute_derived_quantities to guarantee consistency between the integrator and diagnostics" and
  lists a momentum behaviour. A future editor changing the momentum-phase pressure model will edit
  this branch and see nothing change, because phase 2 duplicates the `pRam` call at five other
  sites.
- **confidence** — high

### DC-006 · Two `current_phase == 'transition'` branches inside `compute_derived_quantities` are unreachable
- **file:line** — trinity/phase1_energy/energy_phase_ODEs.py:389-391 and :399-402
- **class** — unreachable
- **severity** — S3 misleading
- **evidence** — `compute_derived_quantities` has one caller:
  `grep -rn "compute_derived_quantities" trinity/ test/ tools/` →
  `run_energy_phase.py:229  ode_result = energy_phase_ODEs.compute_derived_quantities(`, plus an
  **unused import** at `run_energy_implicit_phase.py:79` (ruff: `F401 trinity.phase1_energy.energy_phase_ODEs.compute_derived_quantities imported but unused`)
  and a docstring mention in `get_bubbleParams.py:320`. `run_energy_phase.run_energy` is invoked
  only from `main.py:251`, immediately after `main.py:244 params['current_phase'].value = 'energy'`;
  neither `run_energy_phase.py` nor `run_energy_implicit_phase.py` ever assigns `current_phase`
  (`grep -n "current_phase" trinity/phase1_energy/run_energy_phase.py trinity/phase1b_energy_implicit/run_energy_implicit_phase.py` → no output).
  Consequently `snapshot.current_phase` is always `'energy'` here, so `P_drive = max(Pb, P_HII + P_ram)`
  (391) and `P_ram_val = P_ram` (400) never execute; the diagnostic always reports `P_ram = 0.0`.
  Contrast: the *sibling* branch in `get_ODE_Edot_pure` (line 253) **is** live, because
  `run_transition_phase.py:231` calls it with a `'transition'` snapshot — so this is a real
  asymmetry, not a blanket dead-branch.
- **ruled out** — Not dynamic (dataclass field). Not test-only (no test constructs an
  `ODESnapshot(current_phase='transition')` and calls `compute_derived_quantities`;
  `grep -rn "compute_derived_quantities" test/` → 0).
- **consequence** — The comment at line 398 (`# P_ram: only relevant in transition; 0 in energy/implicit`)
  describes an intent that this function cannot fulfil; anyone who later routes the transition phase
  through `compute_derived_quantities` for diagnostics parity would assume it already works.
- **confidence** — high

### DC-007 · `output_format` is a documented parameter with no consumer
- **file:line** — trinity/_input/default.param:31; spec at trinity/_input/registry.py:331
- **class** — orphan-schema-key
- **severity** — S3 misleading
- **evidence** — a per-key consumer scan over all 80 keys in `default.param` against every `.py` in
  `trinity/` (excluding `registry.py`, which only declares them) leaves `output_format` with a single
  hit, and that hit is a comment: `trinity/_input/param_spec.py:35  "input_admin",  # model_name, path2output, output_format, log_*, simplify_npoints`.
  `grep -rn "output_format" trinity/ run.py tools/ test/ --include="*.py"` →
  `registry.py:331`, `param_spec.py:35`, `test/test_metadata.py:100` (a fixture dict). No reader.
  The spec `info` is honest — `'Output-format selector; currently inert — snapshots are always written as JSONL.'` —
  but `docs/source/parameters.rst:96-98` presents it as a live knob:
  `* - ``output_format`` / - ``JSON`` / - Output format. Currently only JSON (JSONL) is supported.`
- **ruled out** — Not read by prefix or dynamically: the only prefix-driven key family is
  `sps_col_*` (DC-101); `output_format` matches no prefix loop. Not test-only-meaningful:
  `test_metadata.py:100` puts it in an expected-metadata dict, i.e. it is asserted to *be written*,
  never to *do* anything.
- **consequence** — Setting `output_format` in a `.param` is accepted by `read_param` (it is in the
  schema, so it is not rejected) and silently does nothing. It is written into `metadata.json`
  (`run_const=True`), so downstream consumers see a format field that does not describe a choice.
- **confidence** — high

### DC-008 · `mu_mol` is derived at load and then read by nothing
- **file:line** — trinity/_input/read_param.py:319; spec at trinity/_input/registry.py:365
- **class** — orphan-schema-key
- **severity** — S3 misleading
- **evidence** — `grep -rn "mu_mol" trinity/ --include="*.py"` →
  ```
  trinity/_input/read_param.py:313:    _mu_mol = _muH / (Fraction(1, 2) + _xHe)     # molecular mean mass/particle
  trinity/_input/read_param.py:319:    params['mu_mol'].value     = float(_mu_mol) * _mH_au
  trinity/_input/registry.py:365:    ParamSpec(name='mu_mol', …)
  trinity/_functions/unit_conversions.py:374:        # Used for mu parameters (mu_atom, mu_ion, mu_ion_shell, mu_mol, mu_convert)
  ```
  Line 374 is a comment. There is no `params['mu_mol']` **read** anywhere. Its three siblings are all
  consumed: `mu_atom` (shell_structure.py, get_shellODE.py, operations.py, trinity_reader.py),
  `mu_ion` (bubble_luminosity.py ×12), `mu_convert` (mass_profile.py ×17, get_InitCloudProp.py ×11,
  bonnorEbertSphere.py ×8, shell_structure.py ×8).
- **ruled out** — Not dynamic: no `params[f"mu_{…}"]` construction exists
  (`grep -rn "params\[f" trinity/` → only `sps_columns.py:266  key = f"sps_col_{canonical}"`).
  Not test-only-relevant: it is not written to `metadata.json` per-snapshot
  (`exclude_from_snapshot=True`) but *is* `run_const=True`, so it appears in metadata as a value no
  code ever uses.
- **consequence** — `docs/source/parameters.rst:611-613` documents `mu_mol` in a parameter table.
  A user setting it gets it silently overwritten at load (Step 6) *and* it would have had no effect
  even if it survived. Molecular gas mean mass is simply not part of the physics; the schema claims
  otherwise. This is the sharpest example in the schema of a documented knob that does nothing.
- **confidence** — high

### DC-009 · Weaver β↔Ė_b conversion exists twice; the dict-based pair is dead and lacks the live version's guards
- **file:line** — trinity/bubble_structure/get_bubbleParams.py:69 (`cool_beta_to_Ebdot`) and :140 (`Ebdot_to_cool_beta`); live twin at trinity/phase1b_energy_implicit/get_betadelta.py:182 (`cool_beta_to_Ebdot_pure`)
- **class** — superseded-duplicate
- **severity** — S3 misleading
- **evidence** — `grep -rn "cool_beta_to_Ebdot" --include="*.py" .` →
  ```
  trinity/bubble_structure/get_bubbleParams.py:69:def cool_beta_to_Ebdot(params):
  trinity/bubble_structure/get_bubbleParams.py:143:    Inverse of cool_beta_to_Ebdot: convert dE_b/dt …      <- docstring
  trinity/bubble_structure/get_bubbleParams.py:150:    See cool_beta_to_Ebdot for the equation↔code variable map.   <- docstring
  trinity/phase1b_energy_implicit/get_betadelta.py:182:def cool_beta_to_Ebdot_pure(
  trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:992:  Ed = cool_beta_to_Ebdot_pure(beta, Pb, t_now, R1, R2, v2, Eb, feedback.pdot_total, feedback.pdotdot_total)
  trinity/phase1b_energy_implicit/get_betadelta.py:461/:567: Edot_from_beta = cool_beta_to_Ebdot_pure(
  ```
  So the *only* references to `cool_beta_to_Ebdot` outside its own `def` line are two docstring
  mentions inside the dead `Ebdot_to_cool_beta`. `Ebdot_to_cool_beta` itself has exactly one
  occurrence in the whole repo (its `def`).
  **Numerical agreement:** the three functions implement the same Rahner A12 relation and agree
  exactly on physical inputs. `Ebdot_to_cool_beta` is an exact algebraic inverse of
  `cool_beta_to_Ebdot` (solve the `numerator/denominator` form for `Pb_dot`; the terms match
  term-for-term). The live `_pure` version differs *only* in three degeneracy guards the dead one
  lacks: `a_coeff = … if pdot_total > 0 else 0.0` (:251), `c_frac = c_coeff / Ebc if Ebc > 0 else 0.0`
  (:256), and `if abs(denominator) < 1e-300: return 0.0` (:266-267). The dead twin divides by zero in
  each of those regimes.
- **ruled out** — No string-literal form. `get_bubbleParams` is imported as a module in 6 places
  (`import trinity.bubble_structure.get_bubbleParams as get_bubbleParams`), and every attribute
  access on it was enumerated: `solve_R1`, `bubble_E2P`, `pRam`, `get_leak_luminosity`,
  `get_effective_bubble_pressure`, `get_r1` — never `cool_beta_to_Ebdot` or `Ebdot_to_cool_beta`.
  No test references either name.
- **consequence** — Someone fixing the cooling-parameter conversion (a core Weaver relation) has a
  50 % chance of editing the wrong copy, and the wrong copy is the one whose docstring is the most
  thorough (a full equation↔code variable map). The dead copies' unguarded division is also a
  ready-made trap if anyone revives them.
- **confidence** — high

### DC-010 · `delta ↔ dT/dt` conversion exists twice; the live copy has a `t <= 0` guard the dead one lacks, and the inverse direction has no live copy at all
- **file:line** — trinity/bubble_structure/get_bubbleParams.py:27 (`delta2dTdt`) and :47 (`dTdt2delta`); live twin at trinity/phase1b_energy_implicit/get_betadelta.py:272 (`delta2dTdt_pure`)
- **class** — superseded-duplicate
- **severity** — S4 hygiene
- **evidence** — `grep -rn "\bdelta2dTdt\b" --include="*.py" .` → only
  `get_bubbleParams.py:27` (the `def`). `grep -rn "\bdTdt2delta\b"` → only `get_bubbleParams.py:47`.
  The live path: `run_energy_implicit_phase.py:84  delta2dTdt_pure,` (import) and `:993  Td = delta2dTdt_pure(t_now, T0, delta)`.
  **Numerical agreement:** identical formula `(T/t)*delta`; the live one adds
  `if t <= 0: return 0.0` (`get_betadelta.py:292-293`) where the dead one would raise
  `ZeroDivisionError` / return `inf`. `dTdt2delta` (`delta = (t/T)*dTdt`) has no `_pure` counterpart
  and no caller at all — the inverse direction is simply unused.
- **ruled out** — No string-literal form; module-attribute accesses on `get_bubbleParams` fully
  enumerated (see DC-009); no test references.
- **consequence** — none for a user; for an editor, two more decoy definitions of a documented
  thesis equation (Rahner A5) sitting in the file that *looks* like the physics home for them.
- **confidence** — high

### DC-011 · `find_nearest_lower` is a zero-caller twin of `find_nearest_higher` whose monotonicity guard has since diverged — and it is explicitly protected from deletion
- **file:line** — trinity/_functions/operations.py:30
- **class** — superseded-duplicate
- **severity** — S3 misleading
- **evidence** — `grep -rn "\bfind_nearest_lower\b" trinity/ test/ tools/ run.py paper/ --include="*.py"` →
  ```
  trinity/_functions/operations.py:30:def find_nearest_lower(array, value):
  trinity/_functions/operations.py:82:# find_nearest_lower above) as a robustness fallback for any grid-based code
  trinity/_functions/operations.py:156:    # check whether array is monotonic. DEBUG log, not print (see find_nearest_lower).
  ```
  — the `def` plus two comment mentions. `find_nearest_higher` is live:
  `bubble_luminosity.py:708-709  index_CIE_switch = operations.find_nearest_higher(T_array, _CIEswitch)` /
  `index_cooling_switch = …`.
  **Divergence:** the live `find_nearest_higher` gates on the tolerant
  `_is_monotonic_or_tolerable(array)` (:157) which forgives single-point LSODA spikes and shallow
  boundary dips, and picks direction from endpoints (`mon_incr = array[-1] >= array[0]`, :163). The
  dead `find_nearest_lower` still uses the strict `monotonic(array)` (:40) and the all-pairs
  `kindof_increasing(array)` (:45) — i.e. it would raise `MonotonicError` on exactly the profiles the
  live one was hardened to accept. Two behaviours, one live.
  The protection comment at :79-83 reads
  `# RETAINED FALLBACK: the bubble-luminosity solver is moving to a solve_ivp event-based regime split that does not call find_nearest_higher, so this guard may become unused by production. … do not remove it as "dead code".`
  That comment is itself stale: `find_nearest_higher` **is** still called (lines 708-709 above).
- **ruled out** — No string-literal form. `operations` is imported as a module in
  `bubble_luminosity.py`; every `operations.*` attribute access in the package is
  `find_nearest`, `find_nearest_higher`, `monotonic`, `get_soundspeed` — never `find_nearest_lower`.
  No test imports it (`grep -rn find_nearest_lower test/` → 0).
- **consequence** — The "do not remove" comment guarantees this stays. A future editor who *does*
  revive it for a grid path will get spurious `MonotonicError`s on real bubble temperature profiles,
  because the tolerance fix was only applied to the twin. The comment's premise about
  `find_nearest_higher` being on its way out is contradicted by the current source.
- **confidence** — high

### DC-012 · `_suggest_powerlaw_alternatives` documents a `search_range` search width that it never reads
- **file:line** — trinity/cloud_properties/validate_gmc.py:549-558
- **class** — ignored-param
- **severity** — S3 misleading
- **evidence** — AST scan:
  `trinity/cloud_properties/validate_gmc.py:549 _suggest_powerlaw_alternatives(mCloud, nCore, rCore, alpha, nISM, mu, r_max, mass_tolerance, n_suggestions, search_range) UNUSED=['search_range']`.
  Docstring at :556-557: `Varies mCloud, nCore, rCore by ±search_range and returns the closest valid combinations…`.
  The body instead uses three hardcoded factor arrays (:559-566):
  `mCloud_factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])`, likewise `nCore_factors`, and an
  11-element `rCore_factors`. `search_range=0.5` appears nowhere else.
- **ruled out** — Not dynamic (plain kwarg, no `**kwargs`). Not test-only: the function is called
  from `_validate_powerlaw` (`validate_gmc.py:400+`) on the production pre-flight path; no caller
  passes `search_range` explicitly, so the default is dead too.
- **consequence** — The GMC-suggestion search width looks tunable and is not. A user getting poor
  suggestions cannot widen the search; the parameter that says it does that is inert.
- **confidence** — high

### DC-013 · `expansion_next` is an empty stub with a WARPFIELD-era signature and zero callers
- **file:line** — trinity/main.py:366-368
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — full body:
  ```
  def expansion_next(tStart, ODEpar, sps_data_old, sps_f_old, mypath, cloudypath, ii_coll):

      return
  ```
  `grep -rn "\bexpansion_next\b" --include="*.py" --include="*.rst" --include="*.md" .` (excluding
  `docs/dev/`, `old_doNotRead/`, `outputs/`, `scratch/`, `tbd/`) returns exactly one line: the `def`
  itself. AST scan flags all seven parameters unused. None of `ODEpar`, `sps_data_old`, `sps_f_old`,
  `mypath`, `cloudypath`, `ii_coll` is a name used anywhere else in `trinity/`
  (`grep -rn "ODEpar\|cloudypath\|ii_coll" trinity/` → 0 outside this line).
- **ruled out** — No string-literal form. `main` is imported as `from trinity import main` in
  `run.py:165`; the only attribute accessed is `main.start_expansion`
  (`grep -n "main\." run.py`). No test references it.
- **consequence** — It sits directly below `run_expansion` in the top-level orchestration module and
  is preceded by `main.py:208-209  # ########### STEP 2: In case of recollapse, prepare next expansion ###… # TODO: add loop …`,
  so it reads as the recollapse continuation hook. It does nothing and its parameter names refer to
  data structures the codebase no longer has.
- **confidence** — high

### DC-014 · `get_beta_delta_wrapper_pure` is a "backward compatibility" wrapper for a function that no longer exists
- **file:line** — trinity/phase1b_energy_implicit/get_betadelta.py:1148-1179
- **class** — stale-artifact
- **severity** — S4 hygiene
- **evidence** — section header at :1148-1150 is `# Wrapper for Backward Compatibility`; the
  docstring says `Wrapper that matches the interface of the original get_beta_delta_wrapper. This allows drop-in replacement while using pure functions internally.`
  `grep -rn "get_beta_delta_wrapper\b" --include="*.py" .` (excluding `docs/dev/`) → **0 matches**:
  the "original" is gone. `grep -rn "get_beta_delta_wrapper_pure"` → only `get_betadelta.py:1152`
  (the `def`).
- **ruled out** — No string-literal form. `get_betadelta` is imported by name
  (`from trinity.phase1b_energy_implicit.get_betadelta import solve_betadelta_pure, …` at
  `run_energy_implicit_phase.py:82-88`) — the import list is explicit and does not include it. Tests
  import `GBD.solve_betadelta_pure` / `_solve_grid` / `get_residual_pure`, never the wrapper.
- **consequence** — none for a user. For an editor it advertises a compatibility contract with a
  vanished caller, which is exactly the kind of thing that gets preserved out of caution.
- **confidence** — high

### DC-015 · `read_sweep_param` is superseded by `read_sweep_config` and has no callers
- **file:line** — trinity/_input/sweep_parser.py:262
- **class** — superseded-duplicate
- **severity** — S4 hygiene
- **evidence** — `grep -rn "\bread_sweep_param\b" trinity/ test/ tools/ run.py paper/ --include="*.py"` →
  one line, the `def`. The live entry point is `read_sweep_config` (`sweep_parser.py:354`), used at
  `run.py:248` (import) and `run.py:397  config = read_sweep_config(str(param_path))`, plus
  `test/test_sweep_jobs.py:19,37,67,110,120`. `read_sweep_param` returns the bare
  `(base_params, sweep_params)` tuple; `read_sweep_config` returns the richer `SweepConfig` that also
  carries tuple-mode data (`sweep_parser.py:244-260`). The same supersession pattern is *not* present
  for `generate_combinations` (called by `generate_combinations_from_config` at :540) or
  `count_combinations` (called at :878, :883) — both of those are live internals.
- **ruled out** — No string-literal form. `sweep_parser` is imported with an explicit name list in
  `run.py:247-251` (`read_sweep_config`, `generate_combinations_from_config`,
  `count_combinations_from_config`) and in `sweep_jobs.py:32`. No test imports it.
- **consequence** — Sweep parsing has two entry points; only one understands `tuple(...)` syntax.
  Anyone extending sweep parsing via the obvious-looking `read_sweep_param` extends a dead path.
- **confidence** — high

### DC-016 · `_get_bubble_ODE_initial_conditions` takes `R1` and never reads it — in the innermost bubble-solver loop
- **file:line** — trinity/bubble_structure/bubble_luminosity.py:392
- **class** — ignored-param
- **severity** — S4 hygiene
- **evidence** — AST scan:
  `trinity/bubble_structure/bubble_luminosity.py:392 _get_bubble_ODE_initial_conditions(dMdt, params, Pb, R1) UNUSED=['R1']`.
  The 20-line body (lines 393-412) reads `k_B`, `mu_ion`, `cooling_boost_kappa`, `C_thermal`,
  `params['R2']`, `params['cool_alpha']`, `params['t_now']`, `dMdt` and `Pb` only. Both production
  call sites pass it: `bubble_luminosity.py:275` and `:316`. Tests pass it too
  (`test/test_dR2min_magic_number.py:154`, `test/test_residual_resample.py:111,224`) and
  `tools/bubble_conduction_convergence.py:95`.
- **ruled out** — Not dynamic (positional parameter, no `**kwargs`). Not compensated elsewhere:
  `R1` is used in the same file for `Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, …)` at :228, i.e.
  its influence enters *through* `Pb`, which is passed separately.
- **consequence** — A reader auditing the Weaver Eq-44 initial conditions sees `R1` in the signature
  and assumes the wind termination shock enters the ICs directly. It does not — only via `Pb`.
  Given the project's rule-5 emphasis on this exact code path, a spurious argument here is a real
  audit hazard.
- **confidence** — high

### DC-017 · `get_dudt` accepts and documents an `age` argument it no longer uses; the commented-out call that used it is still in place
- **file:line** — trinity/cooling/net_coolingcurve.py:58
- **class** — ignored-param
- **severity** — S4 hygiene
- **evidence** — AST scan: `trinity/cooling/net_coolingcurve.py:58 get_dudt(age, ndens, T, phi, params_dict) UNUSED=['age']`.
  Docstring at :64 documents `age [Myr]: TYPE`. The sole production caller passes a live value:
  `bubble_luminosity.py:430  dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)`.
  The reason is still in the file as dead comments: `:109  # cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)`
  and `:94-96  # In order to improve speed, here we use dictionary. This means that the age will not be as accurate, …`.
  That commented signature is also stale: the live function is
  `read_cloudy.py:22  def get_coolingStructure(params)` — it takes `params`, not `age`.
- **ruled out** — Not dynamic. Not test-only: `test/test_net_coolingcurve.py:59` calls
  `ncc.get_dudt(0.1, nd_au, T, ph_au, params)` — passing `0.1` for an argument that is discarded.
- **consequence** — Minor per-call waste; more importantly the innermost cooling call looks
  time-dependent and is not, and the surrounding commented block documents a call signature that no
  longer exists.
- **confidence** — high

### DC-018 · `build_dlaw_block` accepts `dens_profile`, never reads it, and the caller does a metadata lookup to supply it
- **file:line** — trinity/_output/cloudy/dlaw.py:58
- **class** — ignored-param
- **severity** — S4 hygiene
- **evidence** — AST scan:
  `trinity/_output/cloudy/dlaw.py:49 build_dlaw_block(…, dens_profile, …) UNUSED=['dens_profile']`.
  The docstring is honest (`:81-83  TRINITY profile shape; reserved for future PCHIP-on-densBE support. Currently unused; densification is linear-in-(log r, log n).`).
  The production caller nonetheless computes it:
  `trinity/_output/cloudy/snapshot_to_deck.py:253  dens_profile=str(bundle.metadata.get("dens_profile", "densPL")),`.
- **ruled out** — Not dynamic (keyword-only parameter, no `**kwargs`). It is part of a public
  re-export (`trinity/_output/cloudy/__init__.py:16,32  build_dlaw_block`), so the *function* is
  external API — but the *argument* is inert regardless of caller.
- **consequence** — none functionally; hygiene only. Flagged because it is the only ignored-param in
  the package that is also on a documented public API surface, so it will outlive the others.
- **confidence** — high

### DC-019 · Crash-time debug snapshot facility (`save_debug_snapshot` / `load_debug_snapshot` / `DEBUG_SNAPSHOT_FILE`) has no callers
- **file:line** — trinity/_input/dictionary.py:1020 (constant), :1022, :1128
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — `grep -rn "save_debug_snapshot\|load_debug_snapshot" trinity/ test/ tools/ run.py paper/ --include="*.py"` →
  ```
  trinity/_input/dictionary.py:1022:def save_debug_snapshot(...)
  trinity/_input/dictionary.py:1047:    from trinity._input.dictionary import save_debug_snapshot     <- inside its own docstring (Usage section)
  trinity/_input/dictionary.py:1048:    save_debug_snapshot(params)                                   <- docstring
  trinity/_input/dictionary.py:1051:    save_debug_snapshot(params, "/tmp/debug")                     <- docstring
  trinity/_input/dictionary.py:1128:def load_debug_snapshot(...)
  trinity/_input/dictionary.py:1145/1148:                                                                <- docstring ("Usage in tests")
  ```
  Verified those line ranges are inside triple-quoted docstrings (`sed -n '1018,1060p'`,
  `sed -n '1124,1155p'`). `load_debug_snapshot`'s docstring is headed **"Usage in tests"** — and no
  test uses it (`grep -rn "debug_snapshot" test/` → 0).
- **ruled out** — No string-literal form. `dictionary` is imported with explicit name lists
  (`from trinity._input.dictionary import updateDict` etc.); no wildcard imports exist in the package
  (`grep -rn "import \*" trinity/` → 0). The live crash path is separate and *is* wired:
  `dictionary.py:262 _register_crash_handlers` → `:292 _signal_handler` → `:302 _safe_flush` →
  `:327-328 write_termination_debug_report`.
- **consequence** — none for a user. For a debugging session it is a facility that looks available
  ("call periodically or before risky operations") but is not exercised by anything, so its
  serialisation branches are untested against the current `params` shape.
- **confidence** — high

### DC-020 · `DescribedDict.load_snapshot` / `load_latest_snapshot` are a dead pair kept alive only by a `__main__` demo
- **file:line** — trinity/_input/dictionary.py:924, :967
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — `grep -rn "\bload_latest_snapshot\b" trinity/ test/ tools/ run.py paper/` → one
  line, the `def`. `grep -rn "\bload_snapshot\b" …` →
  ```
  trinity/_input/dictionary.py:41:params = DescribedDict.load_snapshot(path2output, snap_id)   <- module docstring
  trinity/_input/dictionary.py:924:    def load_snapshot(
  trinity/_input/dictionary.py:976:        return cls.load_snapshot(path2output, last_id)          <- called only by load_latest_snapshot (itself dead)
  trinity/_input/dictionary.py:1332:    loaded = DescribedDict.load_snapshot("./_example_output", 1)  <- inside `if __name__ == "__main__":`
  ```
  Contrast the plural `load_snapshots` (`:874`), which **is** live: `test/test_metadata.py` uses it,
  and `load_snapshot` itself calls it. So the cluster is: `load_snapshots` (live) ←
  `load_snapshot` (dead) ← `load_latest_snapshot` (dead).
- **ruled out** — No string-literal form; no `getattr(DescribedDict, …)` anywhere. Not external API:
  `grep -rn "load_snapshot" docs/source/` → 0; `trinity/__init__.py` documents
  `TrinityOutput.open` as the read path, not this.
- **consequence** — none for a user; a maintainer touching the snapshot schema has two extra
  reconstruction paths to keep consistent, only one of which is ever executed.
- **confidence** — high

### DC-021 · Two whole modules inside the package have no importer anywhere: `_analysis/check_yesno.py` and `_functions/extract_example_snapshots.py`
- **file:line** — trinity/_analysis/check_yesno.py:1 (297 lines); trinity/_functions/extract_example_snapshots.py:1 (116 lines)
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — a module-level import scan across `trinity/`, `test/`, `tools/`, `run.py`, `paper/`
  reports both as never referenced. Direct confirmation:
  `grep -rn "check_yesno\|extract_example_snapshots\|_analysis" trinity/ test/ tools/ run.py paper/ docs/source pyproject.toml --include="*.py" --include="*.toml" --include="*.rst"`
  returns **nothing** outside the two files themselves (the only two hits are in
  `test/test_bench_theta_cum.py` for an unrelated `make_bench5_analysis` name).
  `trinity/_analysis/__init__.py` is empty (0 lines), so nothing is re-exported.
  Both are `python -m` CLIs (`check_yesno.py:34-36` documents
  `python -m trinity._analysis.check_yesno -f …`; `extract_example_snapshots.py:23` likewise), each
  ending in `if __name__ == '__main__': main()`.
- **ruled out** — Not dynamic (no `importlib` in the package). Not test-only (zero test references).
  Not console-script entry points: `pyproject.toml` has no `[project.scripts]` section
  (`grep -n "scripts\|console" pyproject.toml` → only `[tool.setuptools.packages.find]` /
  `include = ["trinity*"]` / black's `include`). `check_yesno.py:48-49` even carries a
  `sys.path.insert(0, str(_HERE.parent.parent))` bootstrap, which is only needed when the file is run
  as a loose script.
- **consequence** — 413 lines of analysis tooling shipped inside the importable package with no
  test coverage and no discoverability. `CLAUDE.md` designates `tools/` as the home for "small CLI
  utilities"; these two are the same kind of thing living in the library namespace, so they get
  installed with the package and will silently rot against `trinity_reader` / snapshot-schema
  changes. `check_yesno.py` in particular is a one-off investigation script (its docstring is a
  hypothesis write-up about `_yesPHII`/`_noPHII` runs).
- **confidence** — high (that nothing imports them); medium (that they are unwanted — they are
  runnable CLIs, so "dead" is a judgement call)

### DC-022 · Formatted logging helpers `log_file_saved` / `log_warning` / `log_error` have no callers
- **file:line** — trinity/_output/terminal_prints.py:106, :111, :116
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — `grep -rn "\blog_file_saved\b" / "\blog_warning\b" / "\blog_error\b"` across
  `trinity/ test/ tools/ run.py paper/` each return exactly one line — the `def`. They are the only
  consumers of `cprint.SAVE` and the intended consumers of `cprint.WARN` / `cprint.FAIL` outside
  `header.py`. The rest of the package logs with the plain module logger
  (`logger.warning(f"…")`, ~200 sites).
- **ruled out** — No string-literal form. `terminal_prints` is imported as a module
  (`import trinity._output.terminal_prints as terminal_prints` in `main.py:27`); every attribute
  accessed on it in the package is `phase0`, `phase`, `format_end_report`, `cprint` — never the three
  log helpers.
- **consequence** — none — hygiene only. Worth noting alongside DC-023: `cprint.LINK`
  (`terminal_prints.py:87`) is likewise never read; `header.py:28,30` hardcodes `'\033[32m'` instead.
- **confidence** — high

### DC-023 · Miscellaneous zero-caller public helpers
- **file:line** — see list below
- **class** — zero-caller
- **severity** — S4 hygiene
- **evidence** — each verified with `grep -rn "\b<name>\b" trinity/ test/ tools/ run.py paper/ --include="*.py"`
  returning only the `def`/`class` line (and, where noted, comment-only mentions):
  | name | file:line | notes |
  |---|---|---|
  | `iter_progress` | `trinity/_output/trinity_reader.py:1098` | stderr progress-bar generator; the sibling alias `load_output = read` (:1095) **is** live (used by `_analysis/check_yesno.py`, `tools/reduce_sweep.py`, 5 `paper/` figure scripts) |
  | `SweepProgress` (+ its `elapsed`/`eta` methods) | `trinity/_input/sweep_runner.py:48` | progress/ETA dataclass; `run.py` prints its own progress |
  | `specs_by_category` | `trinity/_input/registry.py:541` | the only registry projection helper with no consumer; `run_const_keys` / `metadata_exclude_keys` / `validate_all` / `resolve_all` / `apply_active_when` are all live |
  | `xi2r` | `trinity/cloud_properties/bonnorEbertSphere.py:622` | the *TRINITY-interface* ξ→r converter; its mirror `r2xi` (:582) **is** live (`density_profile.py:28  from … import r2xi as be_r2xi`) |
  | `compute_minimum_rCore` | `trinity/cloud_properties/mass_profile.py:566` | analytic rCore floor; `validate_gmc._suggest_powerlaw_alternatives` (:549) does a brute-force factor grid instead |
  | `compute_mass_accretion_rate` | `trinity/cloud_properties/mass_profile.py:437` | see DC-024 |
  | `SimulationEndCode.is_inspection_required` | `trinity/_output/simulation_end.py:117` | siblings `is_clean` (used at `terminal_prints.py:222`) and `is_error` (`:224`) are live; the 50-59/99 band has no reader |
  | `create_BE_sphere_from_params` | `trinity/cloud_properties/bonnorEbertSphere.py:501` | **test-only** — sole references are `test/test_mu_audit_drift.py:296,299`; production uses `create_BE_sphere` (`get_InitCloudProp.py:324`, `validate_gmc.py:481,674`) |
  | `MAX_VELOCITY_EXPANSION` | `trinity/phase_general/phase_events.py:74` | dead constant; all four `build_*_phase_events` pass `MAX_VELOCITY_COLLAPSE, direction="collapse"`, so the `direction == "expansion"` (:195-200) and `else: # both` (:201-206) branches of `make_velocity_runaway_event` are production-unreachable and exercised only by `test/test_phase_events.py:36,45` |
  | `cprint.LINK` | `trinity/_output/terminal_prints.py:87` | never read |
- **ruled out** — All ten names were grepped in string-literal form
  (`grep -rn "['\"]<name>['\"]"`) across `trinity/ test/ tools/ run.py paper/` → 0 occurrences each,
  so none is reachable through a name-keyed lookup. None appears in `docs/source/` or `README.md`.
  `trinity/__init__.py` exports nothing (only `__version__`/`__author__`) and every package
  `__init__.py` except `trinity/_output/cloudy/__init__.py` is empty, so none of these is an
  `__all__` re-export. `trinity/_output/cloudy/__init__.py`'s `__all__` lists only
  `DlawError, RunBundle, RunLoadError, SnapshotInvalid, build_dlaw_block, load_run, snapshot_to_values` —
  none of the above.
- **consequence** — none individually — hygiene only. Collectively they are the reason a name-based
  grep in this codebase returns definitions that look like the live implementation
  (`xi2r` next to a live `r2xi`; `is_inspection_required` next to a live `is_clean`;
  `compute_minimum_rCore` in the module a reader would search for it in).
- **confidence** — high

### DC-024 · `compute_mass_accretion_rate` duplicates a formula that `get_mass_profile` computes inline
- **file:line** — trinity/cloud_properties/mass_profile.py:437 (dead) vs :225 (live inline)
- **class** — superseded-duplicate
- **severity** — S4 hygiene
- **evidence** — dead body (:479-483):
  ```
  rho_arr = _to_array(get_mass_density(r_arr, params))
  dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr
  ```
  live inline (`get_mass_profile`, :223-225):
  ```
  # dM/dt = dM/dr * dr/dt = 4*pi*r^2 * rho(r) * v(r)
  dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr
  ```
  Identical expression, identical `rho_arr` provenance (`:208  rho_arr = _to_array(get_mass_density(r_arr, params))`).
  **They agree numerically and bit-for-bit** — same operations in the same order. `grep -rn "compute_mass_accretion_rate"`
  across `trinity/ test/ tools/ run.py paper/` → the `def` line only.
- **ruled out** — No string-literal form. `mass_profile` is imported by explicit name lists; the dead
  function is in none of them.
- **consequence** — none numerically. The dead copy carries a much richer docstring
  ("This formula is EXACT for any smooth density profile… NO SOLVER HISTORY NEEDED"), so it is the
  copy a reader will find and trust, while the one that actually runs is three lines with a comment.
- **confidence** — high

### DC-025 · `lbfgsb_result` is assigned and never tested
- **file:line** — trinity/phase1b_energy_implicit/get_betadelta.py:768 (init), :790 (assignment)
- **class** — unreachable
- **severity** — S4 hygiene
- **evidence** — ruff: `trinity/phase1b_energy_implicit/get_betadelta.py:790:17: F841 Local variable lbfgsb_result is assigned to but never used`.
  `grep -n "lbfgsb_result" trinity/phase1b_energy_implicit/get_betadelta.py` →
  `768: lbfgsb_result = None`, `790: lbfgsb_result = (beta_lbfgsb, delta_lbfgsb, total_res_lbfgsb, iter_lbfgsb)` —
  two writes, zero reads. Its sibling `grid_result` (:738, :749) *is* read at `:769`
  (`grid_residual = grid_result[2] if grid_result else float('inf')`), so the asymmetry is a
  half-finished symmetry, not a deliberate design.
- **ruled out** — Not dynamic; it is a plain local, not stored in `params` or a closure. Not
  test-only (locals are invisible to tests).
- **consequence** — none — hygiene only. Included because it sits in the solver the project's
  working rules single out as the highest-risk path, and because the parallel `grid_result` *is*
  load-bearing, which makes the dead one look load-bearing too.
- **confidence** — high

### DC-026 · Validator/resolver/handler signatures declare a `params` (or `frame`/`bundle`/`snap_id`) argument that the body ignores
- **file:line** — trinity/_input/registry.py:99, :108, :151, :189, :204, :241; trinity/_input/dictionary.py:292, :577; trinity/_output/cloudy/trinity_to_cloudy.py:466
- **class** — ignored-param
- **severity** — S4 hygiene
- **evidence** — AST unused-argument scan:
  ```
  registry.py:99  _validate_ZCloud(value, params)                UNUSED=['params']
  registry.py:108 _validate_dens_profile(value, params)          UNUSED=['params']
  registry.py:151 _validate_betadelta_solver(value, params)      UNUSED=['params']
  registry.py:189 _validate_coverFraction(value, params)         UNUSED=['params']
  registry.py:204 _validate_rCloud_max(value, params)            UNUSED=['params']
  registry.py:241 _resolve_path_cooling_nonCIE(value, params)    UNUSED=['params']
  dictionary.py:292 _signal_handler(self, signum, frame)         UNUSED=['frame']
  dictionary.py:577 _clean_for_snapshot(self, snap_id)           UNUSED=['snap_id']
  trinity_to_cloudy.py:466 _print_summary(bundle, records, args, out_dir)  UNUSED=['bundle']
  ```
- **ruled out** — The six `registry.py` cases are **not** findings against the code: `param_spec.py:115`
  declares the callback type as `Callable[[Any, dict], None]` and `validate_all` calls every
  validator uniformly (`registry.py:545+`), so `params` is a required interface slot. The two
  validators that *do* use it (`_validate_cooling_boost_fA:138-139`,
  `_validate_stop_at_rCloud_nSnap:186`) prove the slot is real. `_signal_handler(frame)` is the POSIX
  `signal.signal` contract. These are listed for completeness and explicitly **not** claimed as dead.
  The two that are genuine: `_clean_for_snapshot(snap_id)` — the caller
  (`dictionary.py:737  clean_dict = self._clean_for_snapshot(snap_id=snap_id)`) computes and passes a
  value the method never reads — and `_print_summary(bundle)`.
- **consequence** — none — hygiene only. Recorded so a future sweep does not re-flag the six
  interface-contract slots as bugs.
- **confidence** — high

### DC-027 · Unused imports and dead locals across the package (ruff F401/F841, 44 findings)
- **file:line** — see below; full list reproducible with `ruff check --select F401,F811,F841 --output-format concise trinity/`
- **class** — stale-artifact
- **severity** — S4 hygiene
- **evidence** — 44 findings (35 auto-fixable). The ones that carry information beyond noise:
  * `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:78,79` — `ODEResult` and
    `compute_derived_quantities` imported from `energy_phase_ODEs` and never used. This is the
    residue that makes DC-006's unreachability non-obvious: the implicit phase *looks* like a caller.
  * `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:87` — `BetaDeltaResult` imported,
    unused.
  * `trinity/sps/update_feedback.py:13` — `from trinity._input.dictionary import updateDict`, unused
    (every other phase module that imports `updateDict` calls it).
  * `trinity/cloud_properties/mass_profile.py:33,34` — `compute_rCloud_homogeneous`,
    `compute_rCloud_powerlaw` imported under the comment
    `# Import utility for computing rCloud from physical parameters`, both unused.
  * `trinity/main.py:29` — `DescribedItem`, `DescribedDict` imported, unused (only
    `COOLING_PHASE_KEYS` from that line is live).
  * `trinity/{phase1b,phase1c,phase2,shell_structure,sps}/…` — `trinity._functions.unit_conversions`
    imported as `cvt` and unused in 5 modules.
  * dead locals: `run_energy_implicit_phase.py:498 nISM`, `run_transition_phase.py:291 nISM`,
    `run_momentum_phase.py:232 nISM` — the same dead `params['nISM'].value` read repeated in all
    three copies of the force-computation helper, i.e. a copy-paste fingerprint;
    `run_energy_implicit_phase.py:699 n_estimate` (a pre-allocation size that pre-allocates nothing);
    `shell_structure.py:104 pBubble` (`params['Pb'].value`, never used);
    `shell_structure.py:311 tau_max = 100` (set in the neutral-shell branch, never read);
    `mass_profile.py:119 r_arr`, `:187 rdot_was_scalar`.
- **ruled out** — ruff's F401 already excludes `__init__.py` re-export patterns and `__all__`
  membership; none of these files has an `__all__` containing the flagged names (only
  `read_param.py`, `run_loader.py`, `snapshot_to_deck.py`, `cloudy/__init__.py` define `__all__`, and
  none lists them). The project's pre-commit ruff set is deliberately narrow (`F821/F811/F823/E9`
  per `CLAUDE.md`), which is why these persist — that is a policy choice, not an oversight, so this
  entry is informational.
- **consequence** — none — hygiene only, with one exception: the two unused imports at
  `run_energy_implicit_phase.py:78-79` actively mislead about which phase drives
  `compute_derived_quantities` (DC-006).
- **confidence** — high

### DC-028 · Six library modules carry `if __name__ == "__main__":` self-test / demo blocks that no test runs
- **file:line** — trinity/_input/sweep_parser.py:890-1048; trinity/_input/dictionary.py:1284-1342; trinity/_input/read_param.py:511; trinity/_functions/unit_conversions.py:484; trinity/_functions/logging_setup.py:503; trinity/phase0_init/get_InitCloudProp.py:549
- **class** — stale-artifact
- **severity** — S4 hygiene
- **evidence** — `grep -rn "__name__ == " trinity/ --include="*.py"` returns 12 hits. Six are real
  CLI entry points (`show_run.py:499`, `trinity_to_cloudy.py:509`, `check_yesno.py:296`,
  `extract_example_snapshots.py:115`, `sweep_runner.py:631`, plus `trinity_to_cloudy.py:42`'s
  `sys.path` bootstrap). The other six are hand-rolled test harnesses that print `PASS`/`FAIL`:
  `sweep_parser.py:891  # Quick test of parsing functions` … `:1048  print("\nAll tests complete!")`;
  `dictionary.py:1283  # Quick test example` … `:1341  print("All tests passed!")`;
  `read_param.py:510  # Quick test (commented out)` (the header says commented out; the block is not);
  `unit_conversions.py:481  # Test suite (run with: python unit_conversions.py)`;
  `logging_setup.py:500  # Usage Examples and Testing`;
  `get_InitCloudProp.py:546  # Test / Example usage` with its own local `class MockParam`.
  None is collected by pytest (`pytest` only collects `test/test_*.py` per `CLAUDE.md`), and ruff
  flags dead code inside one of them (`unit_conversions.py:557 F841`), i.e. nobody has run it
  recently.
- **ruled out** — Not reachable on import (guarded by `__name__`), so not a false positive from a
  dynamic path. Not test-only-but-useful: pytest never executes them; several assert-by-print, so
  even manual execution reports failures as text rather than a non-zero exit.
- **consequence** — 300+ lines of unrun assertions that read like coverage. `CLAUDE.md` is explicit
  that "tests go in the `pytest` suite, not ad-hoc self-checks", so these directly contradict the
  stated convention. `dictionary.py`'s block writes to `./_example_output` and calls the dead
  `load_snapshot` (DC-020) — running it is the only thing keeping that method exercised.
- **confidence** — high

### DC-029 · Stale comments and docstrings that name signatures, counts, or call-graph facts that are no longer true
- **file:line** — see list
- **class** — stale-artifact
- **severity** — S4 hygiene
- **evidence** —
  * `trinity/_functions/operations.py:79-83` — "the bubble-luminosity solver is moving to a solve_ivp
    event-based regime split that does not call find_nearest_higher". It still does:
    `bubble_luminosity.py:708-709`.
  * `trinity/cooling/net_coolingcurve.py:109` — `# cooling_nonCIE, heating_nonCIE = non_CIE.get_coolingStructure(age)`;
    the live signature is `def get_coolingStructure(params)` (`read_cloudy.py:22`) and it returns
    three values, not two.
  * `trinity/cooling/net_coolingcurve.py:133-135, 139, 160, 169` — five commented-out `print`s
    referencing `cpr.WARN` / `cpr.END`; `net_coolingcurve.py` does not import `cpr` (only
    `header.py` does, at `:13`), so these would `NameError` if uncommented.
  * `trinity/_input/param_spec.py:16` — "The registry is fully populated (200 specs)"; actual count
    is 201 (`python -c "from trinity._input.registry import SPECS; print(len(SPECS))"` → `201`).
  * `trinity/_input/param_spec.py:121` — "Three specs carry resolvers today (path2output,
    path_cooling_nonCIE, sps_path)"; actual is four —
    `[s.name for s in SPECS if s.resolver]` → `['path2output', 'cooling_boost_kappa', 'path_cooling_nonCIE', 'sps_path']`.
  * `trinity/_output/run_constants.py:58-60` — "Legacy text-parse readers stay (with
    `DeprecationWarning`) for one cycle, then are removed in Phase 6." They are still present
    (`run_loader.py:154 _parse_summary_txt`, `:188 _parse_simulation_end`) and are **test-only**:
    `grep -rn "_parse_summary_txt\|_parse_simulation_end" trinity/` shows them called only from
    `run_loader.py`'s own fallback chain, whose sole exercisers are
    `test/test_cloudy_run_loader.py` (8 references each).
- **ruled out** — Each claim was re-checked against current source rather than taken from any
  write-up; `docs/dev/` was not read, per scope.
- **consequence** — none — hygiene only, but these are precisely the statements a future editor
  would use to decide *not* to touch something (the `operations.py` one is load-bearing for DC-011's
  "do not remove" instruction).
- **confidence** — high

### DC-030 · The `deprecated` ParamSpec category and `deprecated_note` field are a declared contract with zero members
- **file:line** — trinity/_input/param_spec.py:61, :140, :147-150
- **class** — orphan-schema-key
- **severity** — S4 hygiene
- **evidence** — `[s.name for s in SPECS if s.category=='deprecated']` → `[]`;
  `[s.name for s in SPECS if s.deprecated_note]` → `[]`. The category comment says so itself:
  `"deprecated",  # back-compat retired specs (currently none; kept for future use)`.
  Consumers exist and are live for the empty set: `tools/gen_default_param.py:71,77,89-90`,
  `tools/_param_text.py:21,57`, `test/test_registry.py:181-182,272,285,294`,
  `test/test_materialize_runtime.py:73`, `test/test_gen_default_param.py:103,119`.
- **ruled out** — Not a false positive from dynamic use: the membership check is a literal string
  comparison in five files, all enumerated above. It is deliberately, documentedly empty and the
  `__post_init__` guard is tested (`test_registry.py:181`), so this is a *live-but-unpopulated*
  contract, not a broken one.
- **consequence** — none — hygiene only. Listed because it is the clearest example in the package of
  "declared but never wired up" that is nonetheless intentional, and re-flagging it in a future sweep
  would be wasted effort.
- **confidence** — high

---

## Resolved non-findings (checked, and explicitly *not* dead)

### DC-101 · The 13 `sps_col_*` schema keys ARE read — dynamically, by prefix
- **file:line** — trinity/sps/sps_columns.py:265-274
- The naive per-key consumer scan reports 10 of the 13 `sps_col_*` keys as having no consumer
  (`sps_col_t`, `sps_col_fi`, `sps_col_Lbol`, `sps_col_Lmech_W`, `sps_col_pdot_W`, `sps_col_pdot_SN`,
  `sps_col_Mdot_SN`, `sps_col_v_SN`, `sps_col_Li`, `sps_col_Ln`). **This is a false positive.**
  `build_user_column_map` constructs the key at runtime:
  ```
  for canonical in CANONICAL_NAMES:
      key = f"sps_col_{canonical}"
      if key not in params: continue
      raw_value = params[key].value
      if raw_value == 'def_unset' or raw_value is None: continue
      column_map[canonical] = parse_sps_col_value(canonical, raw_value)
  ```
  with `CANONICAL_NAMES = tuple(CANONICALS.keys())` (:90) enumerating exactly the 13 canonicals. The
  registry mirrors this with `consumed_by='sps_path'` on all 13 specs plus `sps_refmass`
  (`[s.name for s in SPECS if s.consumed_by]` → the 14 expected names), and `sps_path`'s resolver
  `_resolve_sps_bundle` (`registry.py:252`) owns the bundle. All 13 keys are live. This is the only
  prefix-driven key family in the schema.

### DC-102 · `allowShellDissolution` IS honoured
- **file:line** — trinity/shell_structure/shell_structure.py:443-446
- The single consumer is `allow_dissolution = params.get('allowShellDissolution', True)` — which
  returns a `DescribedItem` (always truthy). Lines 444-445 unwrap it
  (`if hasattr(allow_dissolution, 'value'): allow_dissolution = allow_dissolution.value`) before
  `diss_condition_met = bool(allow_dissolution and nShell_max < nISM)`. Not a bug.

### DC-103 · `DescribedItem`'s 20 arithmetic/comparison dunders are live via operator dispatch
- **file:line** — trinity/_input/dictionary.py:143-193
- `__float__`, `__int__`, `__truediv__`, `__lt__`, `__array__` etc. all show "zero callers" to a
  name-based grep. They are invoked implicitly by Python — e.g.
  `main.py:144  f_mass = params['mCluster'] / params['sps_refmass']` (→`__truediv__`),
  `header.py:91  np.log10(params['mCloud']/(1-params['sfe']))` (→`__truediv__`, `__rsub__`,
  `__array__`), `operations.py:207-211` (arithmetic on `params['mu_ion']`, `params['k_B']`). Not dead.

### DC-104 · `_solve_lbfgsb` is reachable
- **file:line** — trinity/phase1b_energy_implicit/get_betadelta.py:1108
- Despite DC-003 (the `method` argument being inert), `_solve_lbfgsb` *is* called at :777 whenever
  `not grid_converged and grid_residual > LBFGSB_FALLBACK_THRESHOLD` (:771). Live under
  `betadelta_solver='legacy'`.

### DC-105 · `get_dudt`'s trailing `else: raise Exception(...)` is reachable via NaN
- **file:line** — trinity/cooling/net_coolingcurve.py:200-201
- The clamp at :130-131 (`if np.log10(T) < nonCIE_Tmin: T = 10**nonCIE_Tmin`) plus the three
  temperature branches cover `[Tmin, ∞)` exhaustively, so the `else` looks unreachable — but a NaN
  `T` makes every comparison `False` and falls through to the raise. Not dead.

### DC-106 · `compute_forces_pure` in phases 1b and 1c are *not* duplicates
- **file:line** — trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:460 vs trinity/phase1c_transition/run_transition_phase.py:271
- Same name, different physics: 1c adds `P_ram = get_bubbleParams.pRam(R2, …)` and uses
  `P_drive = max(Pb, P_HII + P_ram)` / `P_ram=P_ram`, where 1b uses `P_drive = max(Pb, P_HII)` /
  `P_ram=0.0`. Both live. `test/test_phase_helper_sync.py:10-11` explicitly excludes them from its
  sync check. Not a superseded pair — but a same-name/different-behaviour collision worth knowing
  about when grepping.

---

## Suspected, unproven

* **`compute_max_dex_change` is triplicated byte-for-byte across the three phase runners.**
  `run_energy_implicit_phase.py:289`, `run_transition_phase.py:143`, `run_momentum_phase.py:135`.
  An AST comparison with docstrings stripped gives `1b == 1c: True`, `1b == 2: True` — three
  identical live copies with no single source of truth. `get_monitor_values` (1b:412, 1c:164, 2:156)
  and `ForceProperties` (1b:444, 1c:255, 2:190) are the same shape. `test/test_phase_helper_sync.py`
  exists to pin them together, so the maintainers know. Not classified as dead code (all three run),
  but it is the mechanism that produced the three identical dead `nISM` locals in DC-027, and any
  future edit must be applied three times.
* **`trinity/_output/cloudy/run_loader.py`'s legacy text-parse fallbacks** (`_parse_summary_txt:154`,
  `_parse_simulation_end:188`, and `simulation_end.read_simulation_end`'s text branch) are
  production-unreachable *for runs produced by the current writer* (v4 metadata only), but remain
  reachable for pre-Phase-5 output directories a user may still have on disk. Whether any such
  directory still exists is not determinable from the source, so this is not claimed as dead —
  only that `run_constants.py:58-60` says they were to be removed "in Phase 6" and they were not.
* **`trinity/_output/show_run.py:298` legacy branch** — same situation, gated on
  `read_simulation_end` returning non-`None`, which requires a `simulationEnd.txt` on disk.
* **`SB99_rotation = 0`** — `default.param:135` documents that `0` (norot) "requires the user to
  supply matching cooling tables and an sps_path pointing at a norot SPS file; the default SPS
  fallback rejects SB99_rotation=0". `read_cloudy.get_filename:289-292` implements the `norot`
  branch. Unlike DC-002 there is *no* validator forbidding it, so the branch is reachable in
  principle — but only with user-supplied tables that do not ship. Not claimed dead; flagged because
  it is the same shape of half-supported option as ZCloud=0.15 and would be worth a runtime check.

---

## Summary

### By class

| class | count |
|---|---|
| zero-caller | 8 (DC-004, 013, 019, 020, 021, 022, 023, 014*) |
| ignored-param | 6 (DC-001, 003, 012, 016, 017, 018, 026*) |
| orphan-schema-key | 3 (DC-007, 008, 030) |
| unreachable | 4 (DC-002, 005, 006, 025) |
| superseded-duplicate | 5 (DC-009, 010, 011, 015, 024) |
| stale-artifact | 4 (DC-014, 027, 028, 029) |
| **total main findings** | **30** |

\* DC-014 counted under stale-artifact, DC-026 under ignored-param; the table assigns each finding to
its primary class only.

Plus 6 resolved non-findings (DC-101…106) and 4 suspected/unproven items.

### By severity

| severity | count | findings |
|---|---|---|
| S1 results-wrong | 0 | — |
| S2 latent | 0 | — |
| S3 misleading | 12 | DC-001, 002, 003, 004, 005, 006, 007, 008, 009, 011, 012, and DC-029's `operations.py` claim |
| S4 hygiene | 18 | DC-010, 013, 014, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 025, 026, 027, 028, 030 |

No S1 or S2 finding: nothing in this sweep changes a number that a run reports, and no dead branch
is one step away from executing. The damage is uniformly *misdirection* — documented knobs that do
nothing, and dead twins that are better documented than the live implementation.

### Counts by verification method

| | |
|---|---|
| definitions indexed (AST) | 519 across 72 modules |
| schema keys audited against consumers | 80 |
| dynamic-lookup sites enumerated and classified | 27 (all literal-name `getattr`; 1 genuine prefix family, DC-101) |
| string-literal counter-checks run on claimed-dead names | 25, all returning 0 |
| ruff F401/F811/F841 findings folded in | 44 |
