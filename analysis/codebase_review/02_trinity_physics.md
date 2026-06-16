> ⚠️ **This document may be out of date — verify before trusting it.** It is a
> point-in-time analysis/audit, not a maintained spec; the code moves faster
> than these notes (paths, line numbers, and "what shipped" status drift).
> **Any agent or person reading this: treat it as unverified. Flag that it may
> be stale and re-check each claim, snippet, and line reference against the
> current source before relying on it.**

# 02 — trinity/ physics modules

**Scope.** Static, read-only audit of the physics modules a fresh cloner exercises:
`trinity/bubble_structure/`, `trinity/cloud_properties/`, `trinity/cooling/`,
`trinity/shell_structure/`, `trinity/sps/`, and phase modules `phase0_init/`,
`phase1_energy/`, `phase1b_energy_implicit/`, `phase1c_transition/`,
`phase2_momentum/`, `phase_general/`. Focus was units consistency (cross-checked
against `trinity/_functions/unit_conversions.py`), docstrings vs actual formulas,
dead/commented physics, and anything that would surprise a stranger who clones and
runs `python run.py param/simple_cluster.param`. No source was modified. The good
news up front: the default cooling/SPS data needed for a quickstart run **does**
ship (`lib/default/CIE/*.dat`, `lib/default/opiate/opiate_cooling_rot_*.dat`,
`lib/default/sps/...`), and `SB99_rotation 1` in `default.param` matches the only
flavour (`rot`) of opiate tables shipped, so the out-of-the-box path resolves.
No hardcoded personal absolute paths, machine names, or secrets were found in scope
(only `@author: Jia Wei Teh` headers and Rahner-thesis URLs, which are legitimate
attribution/citation). Most findings are stale docstrings and leftover dead code,
not run-breakers.

---

## 🔴 High

### [🔴] `read_cloudy.get_coolingStructure` only resolves metallicity Z=1.0 or Z=0.15 — any other `ZCloud` raises `UnboundLocalError`
- **Where:** `trinity/cooling/non_CIE/read_cloudy.py:290-295` (inside `get_filename`)
- **Issue:** `Z_str` is assigned only in an `if float(metallicity)==1.0 / elif ==0.15` ladder with **no `else`**:
  ```python
  if float(metallicity) == 1.0:
      Z_str = '1.00'
  elif float(metallicity) == 0.15:
      Z_str = '0.15'
  # (no else) -> Z_str referenced below, undefined for any other value
  ```
  `Z_str` is then used unconditionally in the filename. The shipped opiate tables only cover Z 1.00 and 0.15, so a user who sets `ZCloud` to anything else (e.g. 0.5, 2.0 — physically reasonable) crashes with an obscure `UnboundLocalError: local variable 'Z_str' referenced before assignment` from deep inside the cooling loader, with no actionable message.
- **Impact (git-puller):** A plausible parameter choice (non-1.0/0.15 metallicity) hard-crashes mid-run with a cryptic error rather than a clear "unsupported metallicity, available: 1.0, 0.15". The CIE side (`read_coolingcurve.get_Lambda`) silently ignores `metallicity` entirely (TODO at line 20), so the two cooling channels disagree on what metallicities are supported.
- **Fix:** Add an `else: raise ValueError(f"ZCloud={metallicity} unsupported; bundled non-CIE tables cover Z in {{1.0, 0.15}}. Supply matching opiate tables via path_cooling_nonCIE.")`. Ideally validate `ZCloud` against the available tables at param-load time so the failure is reported before the simulation starts.

---

## 🟠 Medium

### [🟠] `get_dudt` docstring states wrong output units ("M_sun/pc/yr3", garbled bracket)
- **Where:** `trinity/cooling/net_coolingcurve.py:34-37`
- **Issue:** The Returns block reads:
  > `dudt is [M_sun/pc/yr3] (erg/cm3/s), because cooling is in units of (erg cm3/s) [M_sun*pc5/s3]`

  The function actually multiplies by `cvt.dudt_cgs2au`, whose definition in `unit_conversions.py:124-125` is "Energy density rate: erg/cm³/s → **Msun/pc/Myr³**". So the time unit should be `Myr³`, not `yr3`; and the `[M_sun*pc5/s3]` bracket for the cooling function is wrong/garbled (the Lambda AU unit is `Msun·pc⁵/Myr³`, see `unit_conversions.py:127-128`). `s3` vs `Myr³` is exactly the unit-drift bug class CLAUDE.md flags.
- **Impact (git-puller):** Anyone reasoning about cooling units from the docstring gets `yr` vs `Myr` wrong (a factor of 1e6 per power) and a nonsensical CGS↔AU bracket. The *code* is correct; only the doc misleads.
- **Fix:** Change to `dudt is [Msun/pc/Myr³] internally (computed in cgs erg/cm³/s, converted via cvt.dudt_cgs2au); cooling Lambda is erg·cm³/s in cgs → Msun·pc⁵/Myr³ in AU.`

### [🟠] `get_shellODE` docstring claims "This routine assumes cgs" but the ODE is in code/pc units
- **Where:** `trinity/shell_structure/get_shellODE.py:32` (docstring) vs `38-58` and body
- **Issue:** Line 32 states `This routine assumes cgs`, yet the same docstring documents `nShell [1/pc3]`, `r [pc]`, and returns `dndr [1/pc4]`, `dphidr [1/pc]`, `dtaudr [1/pc]` — all code (AU) units, and the body pulls `k_B`, `caseB_alpha`, `c_light` etc. from `params` in AU. The "cgs" claim is a stale leftover.
- **Impact (git-puller):** A reader trying to verify the shell ODE will mis-derive every term by the cm↔pc conversion if they take "assumes cgs" at face value.
- **Fix:** Replace "This routine assumes cgs" with "All quantities are in code/astronomy units [Msun, pc, Myr]" to match the parameter annotations.

### [🟠] Self-referential "Key difference from <thisfile>" module docstrings (stale copy-paste from a deleted sibling)
- **Where:** `trinity/phase1_energy/energy_phase_ODEs.py:11` and `trinity/shell_structure/shell_structure.py:10`
- **Issue:** Both module docstrings say:
  - `energy_phase_ODEs.py`: "Key difference from energy_phase_ODEs.py: - get_ODE_Edot_pure() returns only derivatives, never writes to params"
  - `shell_structure.py`: "Key difference from shell_structure.py: - shell_structure_pure() returns a ShellProperties dataclass"

  Each file claims to differ from **itself**. These are clearly carried over from a "pure vs mutating" split where the pure variant replaced the original; the contrast target no longer exists (`get_bubbleProperties` / mutating shell module are gone). `bubble_luminosity.py:7-11` has the analogous "Key difference" phrasing but at least avoids naming itself.
- **Impact (git-puller):** Confusing — a newcomer searches for the "other" `energy_phase_ODEs.py`/`shell_structure.py` that the docstring implies exists and finds nothing.
- **Fix:** Drop the "Key difference from <samefile>" line, or reword to "This module uses pure (non-mutating) functions so it is safe under adaptive solvers' rejected trial steps."

### [🟠] `read_sps.get_interpolation` example uses wrong velocity formula vs the code's actual definition
- **Where:** `trinity/sps/read_sps.py:333` (docstring example) vs `trinity/sps/update_feedback.py:181`
- **Issue:** The `get_interpolation` docstring example shows:
  ```
  >>> v_mech_total = 2.0 * Lmech_W / sps_f['fpdot_W'](t)  # Correct formula!
  ```
  but the production code computes `v_mech_total = 2. * Lmech_total / pdot_total` (`update_feedback.py:181`), and that file's own Notes (lines 140-143) correctly say "uses total quantities". So the `read_sps` example (wind-only) contradicts the runtime definition (total). Note the *energy-phase init* `get_y0` deliberately uses wind-only `2*Lmech_W/pdot_W` (`get_InitPhaseParam.py:134`) — so both formulas exist for different purposes, which makes the mislabeled "Correct formula!" comment actively misleading about which is which.
- **Impact (git-puller):** A user copying the documented "correct" snippet to reproduce `v_mech_total` gets the wind-only value, not the total used by the ODEs/ram-pressure.
- **Fix:** In the `get_interpolation` example use `v_mech_total = 2.0 * Lmech_total / sps_f['fpdot_total'](t)` to match `update_feedback.get_current_sps_feedback`.

### [🟠] `read_cloudy.get_coolingStructure` docstring `Parameters` lists `age [yr]`, but the function takes `params` and reads `t_now` (Myr) — and uses a fragile operator-overload to convert
- **Where:** `trinity/cooling/non_CIE/read_cloudy.py:22-44`
- **Issue:** The signature is `get_coolingStructure(params)`, but the docstring `Parameters` section documents `age : float [yr]` (a parameter that does not exist). Line 44 then does `age = params['t_now'] * 1e6` — note **no `.value`**: it relies on `DescribedItem.__mul__` (`dictionary.py:178`) to unwrap, returning a float. It happens to work, but every other call site in the package writes `params['t_now'].value`; this lone reliance on the operator overload is brittle (e.g. it would break if `t_now` were ever wrapped differently) and inconsistent with the rest of the codebase.
- **Impact (git-puller):** Docstring describes a removed `age` parameter; the `params['t_now'] * 1e6` line reads like a bug (missing `.value`) on first inspection even though it currently works.
- **Fix:** Rewrite the `Parameters` block to document `params` (and note `age` is derived internally as `t_now * 1e6` yr); change line 44 to `age = params['t_now'].value * 1e6` for consistency and robustness.

### [🟠] Dead/unused alternative code paths kept in production physics files (not referenced anywhere in the package or `run.py`)
- **Where:**
  - `trinity/phase1_energy/run_energy_phase.py:343` `run_energy_continuous(params)` — "Alternative implementation"; `git grep` finds it only in `analysis/*.md`, never imported/called.
  - `trinity/bubble_structure/bubble_luminosity.py:875` `_create_adaptive_radius_grid(...)` — docstring says "CURRENTLY UNUSED (kept for reference)".
  - `trinity/bubble_structure/bubble_luminosity.py:1002` `_solve_bubble_ode_with_ivp(...)` — docstring says "Alternative solve_ivp wrapper (CURRENTLY UNUSED)".
  - `trinity/bubble_structure/get_bubbleParams.py:232` `bubble_P2E(...)` — inverse of `bubble_E2P`; only referenced in `analysis/*.md`, not in package code. It also takes **astropy Quantity** inputs while its inverse `bubble_E2P` takes raw cgs floats (`get_bubbleParams.py:198-230`), an internal API-style inconsistency.
- **Issue:** These are self-labeled dead/alternative functions left in the most performance- and correctness-critical physics module. They aren't broken, but they pad the file a stranger must read to understand the live path and (in `bubble_P2E`'s case) advertise a units convention (astropy Quantity) that no other function in the module uses.
- **Impact (git-puller):** A reader auditing the bubble solver wastes time on `_create_adaptive_radius_grid` / `_solve_bubble_ode_with_ivp` / `run_energy_continuous` before realizing the legacy grid + `_solve_bubble_structure` + segment loop are the only live paths.
- **Fix:** Per CLAUDE.md rule 3, don't delete pre-existing dead code as part of an unrelated change — but these should be flagged for removal (or moved to a `scratch/`-style location). If `bubble_P2E` is kept, align it to the same raw-cgs-float convention as `bubble_E2P` or clearly document why it diverges.

### [🟠] `run_energy_implicit_phase` docstring says "Grid search first (4x4 grid by default)" but the grid is 5x5
- **Where:** `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:34` (module docstring) vs `trinity/phase1b_energy_implicit/get_betadelta.py:56`
- **Issue:** Module docstring: "1. Grid search first (4x4 grid by default)". The actual solver constant is `GRID_SIZE = 5  # 5x5 grid` (`get_betadelta.py:56`), and `_solve_grid` builds `np.linspace(..., GRID_SIZE)` in both axes → a 5×5 grid. (A second copy of the "4x4" claim is not present, but the comment at `get_betadelta.py:946` correctly says "5x5 grid".)
- **Impact (git-puller):** Minor but contradicts code; someone tuning solver cost will mis-estimate evaluations per segment (25, not 16).
- **Fix:** Change "4x4 grid by default" to "5x5 grid by default (GRID_SIZE=5)".

---

## 🟡 Low

### [🟡] `bonnorEbertSphere` debug/log strings label code-unit densities as `cm⁻³`
- **Where:** `trinity/cloud_properties/bonnorEbertSphere.py:365` and `433` (also `BESphereResult` docstring `n_out [1/pc³]` at line 142-160)
- **Issue:** `create_BE_sphere` logs `n_core={n_core:.2e} cm⁻³` and `n_out={n_out:.2e} cm⁻³`, but per the function's own contract `n_core`/`n_out` are in code units `[1/pc³]` (the docstring says so explicitly at lines 318 and 425). `n_out = n_core / Omega` stays in `1/pc³`. So the logged "cm⁻³" tag is wrong (off by `ndens_au2cgs ≈ 2.94e55`). Mirror issue in the `__main__` test prints of `get_InitCloudProp.py:585,612,641` ("nEdge = ... cm^-3" printed from a `1/pc³` value), and `CloudProperties` docstring (`get_InitCloudProp.py:63,66`) labels `nEdge`/`n_arr` as `[cm^-3]` though they hold `1/pc³`.
- **Impact (git-puller):** Only affects log/print/docstring readability, not computed results. Someone copying a logged "cm⁻³" density back into a `.param` (where densities really are cm⁻³) would be off by ~55 orders of magnitude. `validate_gmc.format_suggestion` (`validate_gmc.py:94,102`) does it right (multiplies by `ndens_au2cgs` before printing cm^-3), which highlights the inconsistency.
- **Fix:** Either print `cvt.ndens_au2cgs * n` with a `cm⁻³` label, or change the label to `1/pc³ (code units)`. Fix the `CloudProperties` docstring unit tags to `[1/pc³]`.

### [🟡] Leftover commented-out physics / debug `print`s / `sys.exit()` scaffolding in cooling and cloudy modules
- **Where:**
  - `trinity/cooling/net_coolingcurve.py:43-48,71-75,97-99,103-117,131-141,148-149,155` — extensive commented-out unit checks, `print('ndens', ndens)`, depreciated interpolation blocks.
  - `trinity/phase0_init/get_InitCloudyDens.py:44-46` — `# print('Checking initial density') / # print(n) / # sys.exit()` commented debug halt; plus `# TODO: shouldn't this be +dx_small then?` (line 36) and `# TODO: make this sound better...` (line 59).
  - `trinity/cooling/non_CIE/read_cloudy.py:116-126` — commented astropy-unit attribute assignments.
- **Issue:** Dead commented code and debug scaffolding. The `sys.exit()` lines are especially jarring left in (even commented) in a module a fresh user runs.
- **Impact (git-puller):** Cosmetic; clutters the read of the cooling/cloudy path. Harmless to execution.
- **Fix:** Remove the commented debug `print`/`sys.exit` and depreciated interpolation blocks; either resolve or keep the TODOs but they should not be load-bearing.

### [🟡] Scattered `TODO`s referencing unfinished behavior in user-facing physics
- **Where:**
  - `trinity/cooling/net_coolingcurve.py:78-83` — TODO that the low-T cooling floor "in the future has to depend on the file" plus a long uncertainty note ("Not sure why though, as the temperature should be around 1e7, not 1e4").
  - `trinity/cooling/CIE/read_coolingcurve.py:20,23` — `TODO: add for non-solar metallicity`, `TODO: add file saving`. The `metallicity` argument to `get_Lambda` is accepted but **never used**, so CIE cooling is always solar regardless of `ZCloud` — contradicting the non-CIE path which does branch on Z (and ties into the 🔴 finding).
  - `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:106` — `# TODO: very fine grid in this phase...`
  - `trinity/shell_structure/get_shellODE.py:19`, `trinity/shell_structure/shell_structure.py:105` — `TODO: add cover fraction cf (f_cover)` / `f_cover = 1` hardcoded.
- **Issue:** Uncertainty notes and unfinished features in code a stranger runs. The CIE `metallicity`-ignored case is a silent physics shortcut: a user setting `ZCloud` expecting it to affect CIE cooling won't get that, with no warning.
- **Impact (git-puller):** Mostly informational, but the silently-ignored CIE metallicity could produce results a user misattributes to their `ZCloud` setting.
- **Fix:** At minimum, log a one-time warning in `get_Lambda` that CIE cooling ignores metallicity (uses the selected `path_cooling_CIE` table as-is), so the shortcut is visible.

### [🟡] Commented-out "fix kink" alternative ODE in implicit phase references a non-shipped `paper/` artifact
- **Where:** `trinity/phase1b_energy_implicit/run_energy_implicit_phase.py:532-535`
- **Issue:** Dead code block left after the `return`:
  ```python
  # try to fix kink (see paper/rCloud_bump)
  # Ed_from_balance = dydt_energy[2]
  # Ed = min(Ed_from_beta, Ed_from_balance)
  ```
  It points at `paper/rCloud_bump`, a local research artifact a cloner won't recognize, and is unreachable (after `return`).
- **Impact (git-puller):** Cosmetic; a curious reader chases a `paper/rCloud_bump` reference for context that isn't part of the public physics.
- **Fix:** Remove the dead block, or convert the `paper/rCloud_bump` pointer to a short inline explanation of the kink it addresses.

### [🟡] `WEAVER_TEMP_COEFFICIENT` formula comment omits the time unit, where Weaver+77 Eq. 37 is dimensional (cgs time)
- **Where:** `trinity/phase0_init/get_InitPhaseParam.py:30-32,169-176`
- **Issue:** The comment gives `T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4` and the code feeds `dt_phase0` (in **Myr**) into `t^(-6/35)`. Weaver+77 Eq. 37 is a dimensional fit whose `t` is in seconds (or yr); the comment never states which, and `L`/`n` are explicitly converted to cgs (`cvt.L_au2cgs`, `cvt.ndens_au2cgs`) on adjacent lines while `t` is left in Myr. This is at minimum an undocumented unit choice in the known units-bug class; whether it is also a *numeric* error depends on the original fit's time unit (I could not confirm the intended unit from in-repo sources, so flagging as Low pending verification, not asserting a bug).
- **Impact (git-puller):** A reader verifying the initial bubble temperature against Weaver+77 cannot tell from the comment whether `t` should be Myr, yr, or s; the surrounding cgs conversions make the bare-Myr `t` look suspicious.
- **Fix:** State the intended time unit for `t` in the comment and confirm `dt_phase0`'s unit matches the fit (convert with `cvt.Myr2s` if the fit's `t` is seconds). Add a unit assertion/test pinning `T0` for a known input.

---

## Counts: 1 high / 7 medium / 5 low
