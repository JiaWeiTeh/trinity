# TRINITY `/src/` Codebase Analysis Report

**Date:** 2026-03-13
**Scope:** All 104 Python files (~54,000 lines) in `/src/`
**Methodology:** Full source read of every module, cross-referencing between files

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Bugs](#2-critical-bugs)
3. [Architectural Problems](#3-architectural-problems)
4. [Code Duplication / DRY Violations](#4-code-duplication--dry-violations)
5. [Dead Code and Commented-Out Blocks](#5-dead-code-and-commented-out-blocks)
6. [Magic Numbers and Hardcoded Constants](#6-magic-numbers-and-hardcoded-constants)
7. [Exception Handling Issues](#7-exception-handling-issues)
8. [Naming Inconsistencies](#8-naming-inconsistencies)
9. [Missing Validation and Error Handling](#9-missing-validation-and-error-handling)
10. [Non-Conventional Patterns](#10-non-conventional-patterns)
11. [Performance Concerns](#11-performance-concerns)
12. [Unresolved TODOs and Technical Debt](#12-unresolved-todos-and-technical-debt)
13. [Module-by-Module Summary Table](#13-module-by-module-summary-table)
14. [Prioritized Recommendations](#14-prioritized-recommendations)

---

## 1. Executive Summary

TRINITY is a well-structured scientific simulation code with clear physical motivation and good modular separation of simulation phases. The codebase has undergone significant modernization (January 2026 rewrite of parameter handling, JSONL output, dataclass returns for ODE purity). However, the analysis reveals **3 critical bugs**, **significant code duplication** across phase runners, **~400 lines of dead commented-out code** in `main.py` alone, and **numerous magic numbers** without documentation. Legacy modules (`*_old.py`, `__cloudy__.py`, `get_InitNewCluster.py`) contain runtime errors that will crash if called.

**Statistics:**
- 104 Python source files, ~54,000 lines
- 3 critical bugs (potential incorrect physics or runtime crashes)
- ~15 medium-severity issues (logic errors, bare exceptions, missing validation)
- ~30+ low-severity issues (style, naming, dead code)
- 8 files exceeding 1,000 lines (largest: `velocity_radius.py` at 2,348 lines)

---

## 2. Critical Bugs

### BUG-1: Wind velocity formula contradicts documented fix — `update_feedback.py:180`

**Severity:** CRITICAL (affects physics output)

```python
# Lines 176-180 in src/sb99/update_feedback.py
# Comment says:
#   OLD BUG: Used pdot_total (wind + SN) instead of pdot_W
#   This caused 10-80% error depending on SN contribution!
# But the actual code STILL uses pdot_total:
v_mech_total = (2. * Lmech_total / pdot_total)[()]  # ← NOT FIXED!
```

The docstring (lines 133-135) explicitly states the fix should be `v_mech_total = 2 * Lmech_total / pdot_W`, but the implementation on line 180 still divides by `pdot_total`. This is either:
- (a) A real bug where the fix was documented but never applied, or
- (b) An intentional design decision where the comment is stale/wrong.

Either way, the comment and code are contradictory and must be resolved. If `pdot_total` is correct (because `Lmech_total` includes both wind+SN), then the comment is misleading. If `pdot_W` is correct (because `v_mech` represents the wind terminal velocity), then the formula is wrong.

**Impact:** Affects all pressure calculations, ODE dynamics, and every simulation phase. Could produce 10-80% error in wind velocity as the comment itself states.

---

### BUG-2: Runtime NameError in `cloudy/__cloudy__.py:134`

**Severity:** CRITICAL (crash on call)

```python
# Line 18: import is commented out
# from src.warpfield.functions import nameparser

# But line 134 uses it:
nameparser.dir_up()       # ← NameError: name 'nameparser' is not defined
get_set_save_prefix()     # ← NameError: name 'get_set_save_prefix' is not defined
```

The module's import was commented out during refactoring but the code still references `nameparser`. Any call path reaching this code will crash.

**Impact:** Currently mitigated because Cloudy integration is marked as TODO/future work in `main.py:136`. However, if anyone attempts to use Cloudy functionality, it will crash immediately.

---

### BUG-3: Undefined variable in `get_InitNewCluster.py:33`

**Severity:** CRITICAL (crash on call)

```python
# Line 20: import is commented out
# warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# Line 33 uses it:
if warpfield_params.mult_SF >= 1:  # ← NameError
```

Same pattern as BUG-2: commented-out import with live references.

**Impact:** Currently mitigated because recollapse/multi-cluster functionality is not yet implemented (`main.py:175` TODO). Will crash if the recollapse loop is ever activated.

---

## 3. Architectural Problems

### 3.1 Massive `main.py` with ~350 lines of commented-out legacy code

`src/main.py` is 719 lines, of which **lines 354-716 (~360 lines)** are entirely commented-out legacy code from WARPFIELD. This dead code:
- Bloats the file and makes navigation difficult
- References functions that no longer exist (`ODE_tot_aux`, `getSB99_data`, `bubble_interp`, etc.)
- Creates confusion about what is active vs. legacy

**Recommendation:** Remove all commented-out code. If historical reference is needed, that's what git history is for.

### 3.2 Duplicate import on line 26 of `main.py`

```python
from src.phase0_init import (get_InitCloudProp, get_InitPhaseParam)  # line 25
from src.phase0_init import get_InitPhaseParam                       # line 26 (duplicate!)
```

### 3.3 God-dictionary anti-pattern

The `params` DescribedDict serves as a single mutable state container for the entire simulation (~150+ keys by the time runtime parameters are added in `read_param.py`). While this was likely a pragmatic choice, it creates:
- Tight coupling between all modules (every module reads/writes the same dict)
- Difficulty reasoning about what state each function actually needs
- Risk of stale values when params are updated in one phase but not another

The `ODESnapshot` dataclass in `energy_phase_ODEs_modified.py` is a good step toward decoupling, but it's only used in the energy phase. Other phases still read directly from params during ODE evaluation.

### 3.4 `_calc/` files are monolithic

Eight files in `_calc/` exceed 1,000 lines, with `velocity_radius.py` at 2,348 lines. Each file combines:
- Data loading and filtering
- Physics computation
- Plotting
- CLI argument handling

These should be decomposed into computation modules (reusable) and plotting/CLI scripts (consumers).

### 3.5 `create_dictionary.py` appears orphaned

`src/_input/create_dictionary.py` defines runtime parameters that overlap with what `read_param.py` (lines 392-516) already initializes. It's unclear if `create_dictionary.py` is ever called in the current codebase, and its parameters use different naming conventions (e.g., `isDissolution` vs `isDissolved`, `v0` vs `v2`, phase labels `'1a'` vs `'energy'`).

---

## 4. Code Duplication / DRY Violations

### 4.1 Phase runner helper functions (duplicated 4x)

The following helper functions are **copy-pasted identically** across all four phase runners:
- `compute_max_dex_change()` — ~15 lines each
- `get_monitor_values()` — ~20 lines each

**Files affected:**
- `run_energy_implicit_phase_modified.py` (lines 162-202)
- `run_transition_phase_modified.py` (lines 139-172)
- `run_momentum_phase_modified.py` (lines 131-164)

**Recommendation:** Extract to `src/phase_general/phase_helpers.py`.

### 4.2 Shell mass update logic (duplicated 4x)

Each phase runner contains nearly identical shell mass update code:
```python
if snapshot.isCollapse:
    mShell = snapshot.shell_mass
    mShell_dot = 0.0
else:
    mShell_new, mShell_dot = mass_profile.get_mass_profile(...)
    if prev_mShell > 0 and mShell_new < prev_mShell:
        mShell = prev_mShell
        mShell_dot = 0.0
    else:
        mShell = mShell_new
```

This block appears in:
- `energy_phase_ODEs_modified.py` lines 174-188 AND 316-329
- `run_energy_implicit_phase_modified.py` lines 623-641 AND 768-781
- `run_transition_phase_modified.py` lines 503-533
- `run_momentum_phase_modified.py` lines 560-577

### 4.3 Dissolution condition check (duplicated 3x)

The dissolution timer logic is repeated across:
- `run_energy_implicit_phase_modified.py` (~lines 856-870)
- `run_transition_phase_modified.py` (~lines 735-753)
- `run_momentum_phase_modified.py` (~lines 782-803)

### 4.4 `_is_scalar()` / `_to_array()` / `_to_output()` helpers

Duplicated between `density_profile.py` (lines 62-76) and `mass_profile.py` (lines 62-76). Should be in a shared utility.

### 4.5 `get_residual_pure()` vs `get_residual_detailed()` in `get_betadelta_modified.py`

These two functions (lines 274-373 and 435-452) share ~80% identical code. Only the return value differs (summary tuple vs detailed dict).

### 4.6 Value parsing logic duplicated in `read_param.py` and `sweep_parser.py`

Both files implement `parse_value()` with identical boolean/number/fraction/string precedence. The sweep_parser version additionally handles list syntax.

---

## 5. Dead Code and Commented-Out Blocks

| File | Lines | Description |
|------|-------|-------------|
| `main.py` | 354-716 | ~360 lines of commented-out WARPFIELD legacy code |
| `main.py:355-357` | 3 | Empty `expansion_next()` function (placeholder) |
| `get_bubbleParams.py` | 70-155 | 85 lines of commented-out legacy code |
| `get_shellParams.py` | 27-58 | 32 lines of commented-out magnetic field code |
| `read_coolingcurve_old.py` | 63-85 | Commented-out cooling curves |
| `read_cloudy_old.py` | 93-119 | Commented-out deprecated code |
| `bubble_luminosity_modified.py` | 536-660 | `_create_adaptive_radius_grid()` marked disabled |
| `get_betadelta_modified.py` | 536-733 | `_solve_bubble_ode_with_ivp()` unused, kept "for future experimentation" |
| `pressure_blend.py` | entire file | Module header says "DEPRECATED" — no longer called |
| `input_warnings.py` | mostly empty | Function body is ~5 lines of checks; rest is TODO comments and commented-out dict |
| `run_energy_phase_modified.py:349-408` | 60 | `run_energy_continuous()` alternative implementation (unused) |
| `read_param_legacy.py` | entire file | Legacy reader, unclear if used |

**Total dead/commented code: ~1,000+ lines**

---

## 6. Magic Numbers and Hardcoded Constants

### 6.1 Phase runner tuning parameters without justification

```python
# run_energy_phase_modified.py
TFINAL_ENERGY_PHASE = 3e-3    # Why 3000 years?
SEGMENT_DURATION = 3e-5       # Why 30 years?
DT_EXIT_THRESHOLD = 1e-4      # Why this threshold?
COOLING_UPDATE_INTERVAL = 5e-2 # Why 50,000 years?

# run_energy_implicit_phase_modified.py
DT_SEGMENT_MIN = 5e-4         # Different from energy phase — why?
DT_SEGMENT_MAX = 5e-2
VELOCITY_THRESHOLD_COLLAPSE = -300.0  # No reference

# run_transition_phase_modified.py
DT_SEGMENT_MIN = 5e-4         # Same as implicit but different from energy
DT_SEGMENT_COLLAPSE = 5e-4    # Defined but NEVER USED
RAM_DOMINANCE_THRESHOLD = 0.9  # Hardcoded inside loop, not as constant

# run_momentum_phase_modified.py
DT_SEGMENT_INIT = 2e-3        # Larger than other phases — why?
```

### 6.2 Physics thresholds without references

```python
# bubble_luminosity_modified.py
_coolingswitch = 1e4           # K — temperature threshold, no reference
_CIEswitch = 10**5.5           # K — no justification
_xtra = 20                     # Extra grid points — arbitrary
_highres = 1e2                 # High resolution points — arbitrary
MIN_SPACING = 1e-12            # Undocumented threshold

# net_coolingcurve.py
5.5                            # log10(T) cutoff between CIE/non-CIE regimes
1e4                            # Temperature cutoff — no parameterization

# get_shellODE.py
tau > 500                      # Underflow prevention threshold

# phase_events.py
MAX_VELOCITY_COLLAPSE = 500.0  # pc/Myr — no reference
MAX_VELOCITY_EXPANSION = 1000.0
MIN_RADIUS_SAFETY = 0.01      # pc

# get_bubbleParams.py
1e-10                          # Added to r2 to avoid division by zero (hidden)
1e-30                          # Energy floor — arbitrary
1.646                          # dMdt_factor — from Weaver+77 but no equation reference
dt_switchon = 1e-3             # Hardcoded in function body
```

### 6.3 Numerical derivative step size

```python
# update_feedback.py:183
dt = 1e-9  # Myr — no justification for this specific step size
```

This extremely small step size (1e-9 Myr ~ 1 second) for a central difference derivative of interpolated functions could amplify interpolation noise.

---

## 7. Exception Handling Issues

### 7.1 Bare/overly broad `except` clauses

| File | Line(s) | Issue |
|------|---------|-------|
| `mass_profile.py` | 543 | `except:` — bare except swallows all errors silently |
| `powerLawSphere.py` | 543 | `except:` — bare except |
| `get_betadelta_modified.py` | 314, 407, 689 | `except Exception as e:` returning placeholder values |
| `run_energy_implicit_phase_modified.py` | 309, 314, 708 | `except Exception as e:` swallowing errors |
| `run_transition_phase_modified.py` | 304, 580 | `except Exception:` silently fails |
| `run_momentum_phase_modified.py` | 412, 652 | `except Exception as e:` too broad |
| `simulation_end.py` | 313, 405 | Bare except |

### 7.2 `print()` used for error reporting instead of logging/exceptions

```python
# operations.py:36-38
if any(array < 0):
    print(array)                          # Should be logger.warning()

if not monotonic(array):
    print(f"array has to be monotonic!")   # Should raise or use logger

# powerLawSphere.py:289, 307-309
print(rCore)                              # Debug print left in production
print('cloud radius is [pc]', rCloud)     # Same
```

### 7.3 `sys.exit()` for error handling

```python
# input_warnings.py:54
sys.exit('The parameter \'%s\' accepts only 0 or 1 as input' % pars)
# Should raise ValueError instead

# read_coolingcurve_old.py:123
sys.exit()  # No error message at all!
```

---

## 8. Naming Inconsistencies

### 8.1 camelCase vs snake_case mixing

The codebase mixes Python naming conventions inconsistently:

| Pattern | Examples | Convention |
|---------|----------|------------|
| camelCase files | `bonnorEbertSphere.py`, `powerLawSphere.py`, `get_bubbleParams.py` | Non-standard Python |
| camelCase params | `mCloud`, `rCloud`, `nCore`, `TShell_ion` | Physics notation (acceptable) |
| snake_case params | `shell_mass`, `bubble_LTotal`, `current_phase` | Python standard |
| Mixed in same dict | `mCluster` vs `shell_mass` vs `F_grav` vs `bubble_LTotal` | Inconsistent |

### 8.2 Inconsistent phase identifiers

```python
# create_dictionary.py:31
main_dict['current_phase'] = DescribedItem('1a', ...)  # Uses '1a', '1b', '2', '3'

# read_param.py:393
params['current_phase'] = DescribedItem('', ...)       # Uses 'energy', 'implicit', 'transition', 'momentum'
```

### 8.3 Inconsistent boolean naming

```python
# create_dictionary.py
'isDissolution'    # Noun form

# read_param.py
'isDissolved'      # Past tense (different key!)
'isCollapse'       # Present tense
'is_fullyIonised'  # snake_case with British spelling
```

### 8.4 Filename convention inconsistency

```python
# Some files use "get_" prefix:
get_InitCloudProp.py, get_InitPhaseParam.py, get_bubbleParams.py, get_shellODE.py

# Some use descriptive names:
density_profile.py, mass_profile.py, bubble_luminosity_modified.py

# Some use "_modified" suffix:
shell_structure_modified.py, run_energy_phase_modified.py

# Some use dunder convention:
__cloudy__.py  # Non-standard; dunders are for Python internals
```

---

## 9. Missing Validation and Error Handling

### 9.1 No NaN/Inf validation at ODE boundaries

None of the phase runners validate that the state vector `[R2, v2, Eb]` is finite before passing to `solve_ivp`. If a previous segment produced NaN (e.g., from division by zero in shell mass), the next solve_ivp call will silently propagate it.

**Recommendation:** Add before each solve_ivp call:
```python
if not np.all(np.isfinite(y0)):
    raise RuntimeError(f"Non-finite state vector: {y0}")
```

### 9.2 No validation of SB99 interpolation output

`update_feedback.py` calls SB99 interpolation functions but never checks if the returned values are physical (positive luminosity, non-negative momentum rate, etc.).

### 9.3 Division by zero in `header.py:96`

```python
print(f"mCloud  = {params['mCloud']/(1-params['sfe']):.4E} Msun (before SF)")
```

If `sfe = 1.0`, this produces a `ZeroDivisionError`. No guard exists.

### 9.4 `brentq` bracket failures unhandled

In `energy_phase_ODEs_modified.py:193` and `bubble_luminosity_modified.py:101`, `scipy.optimize.brentq` is called without try/except. If the function doesn't change sign in the bracket `[1e-3*R2, R2]`, it raises `ValueError` and crashes the simulation.

### 9.5 Potential infinite loop in shell structure

`shell_structure_modified.py:147-198` iterates the ionized region without a maximum iteration guard. While it should terminate when flux drops to zero, numerical issues could keep it going indefinitely.

---

## 10. Non-Conventional Patterns

### 10.1 `from __future__ import annotations` after docstring

```python
# dictionary.py:3
from __future__ import annotations
"""
Created on Wed Jul 26 15:21:52 2023
...
```

The `from __future__` import should be the very first statement in the file (before any docstring or comments). While Python tolerates this placement, it's non-standard and some linters will flag it.

### 10.2 Arithmetic operators on DescribedItem without returning DescribedItem

`DescribedItem.__add__` returns a raw value, not a `DescribedItem`. This means:
```python
result = params['mCloud'] + params['mCluster']  # Returns float, not DescribedItem
params['total_mass'].value = result              # Must explicitly assign to .value
```

This is arguably acceptable but can lead to subtle bugs when users expect arithmetic to preserve the container type.

### 10.3 `__eq__` on DescribedItem compares values

```python
params['phase'] == 'energy'  # Works! Compares .value to string
```

But this also means DescribedItems can't be reliably used as dictionary keys or in sets, since `__hash__` is not defined (Python defaults to `id()` when `__eq__` is overridden but `__hash__` is not).

### 10.4 `humanfriendly` dependency in `clock.py`

The `Timer` class in `clock.py` depends on the `humanfriendly` package just for formatting time spans. This is an unnecessary dependency for a scientific code — `datetime.timedelta` formatting would suffice.

### 10.5 Module-level singleton in `clock.py`

```python
_timer = Timer()  # Module-level mutable singleton
```

This encourages global mutable state. The `Timer` class should be used via explicit instantiation, not a module singleton.

### 10.6 `input_warnings.py` checks `{0, 1}` instead of `{True, False}`

```python
trueFalseValues = [0, 1]  # Checks for 0/1 instead of True/False
```

This is from an older era of the codebase where booleans were represented as integers. Now that `read_param.py` properly parses `True`/`False` strings to Python bools, these checks will always fail for actual boolean values.

---

## 11. Performance Concerns

### 11.1 `unit_conversions.py` backward-compatibility layer

Lines 229-270 define ~40 module-level aliases duplicating the values already in `CONV` and `INV_CONV` dataclasses. These are imported throughout the codebase via `import src._functions.unit_conversions as cvt` and accessed as `cvt.L_au2cgs`. While this works, it means every constant exists three times in memory (dataclass field, module alias, and the imported module's namespace).

### 11.2 `convert2au()` called at import time

Every parameter's unit conversion goes through the full regex-based `convert2au()` parser. For the ~60 parameters in default.param, this is fine. But for parameter sweeps with thousands of combinations, caching the conversion factors would help.

### 11.3 JSON serialization of large arrays

`DescribedDict.save_snapshot()` serializes numpy arrays to JSON lists inline. For large arrays (e.g., `bubble_r_arr` with hundreds of elements), this produces large JSONL lines. Binary formats (e.g., `.npz` sidecars) would be more efficient for array-heavy snapshots.

### 11.4 Repeated shell/bubble structure in phase loops

Each phase runner recomputes the full shell structure and bubble luminosity every segment. These are expensive operations involving ODE integration. Memoization or adaptive recalculation intervals would reduce cost.

---

## 12. Unresolved TODOs and Technical Debt

### High-Priority TODOs

| File | Line | TODO |
|------|------|------|
| `main.py` | 64 | "put this in read_param, and make it depend on param file" (logging config) |
| `main.py` | 136 | "add CLOUDY support in the future" |
| `main.py` | 175 | "add loop so that simulation starts over for recollapse" |
| `shell_structure_modified.py` | 104 | "Add f_cover from fragmentation mechanics" (hardcoded to 1) |
| `get_shellODE.py` | 19 | "add cover fraction cf (f_cover)" |
| `net_coolingcurve.py` | 78-79 | Known limitation in temperature cutoff handling |

### Medium-Priority TODOs

| File | Line | TODO |
|------|------|------|
| `create_dictionary.py` | 21 | "in future for multiple collapses, add if/else" |
| `create_dictionary.py` | 59-60 | "make sure areas are properly initiated" with "np.nan may cause problems" |
| `create_dictionary.py` | 67 | "i think this is not used? also there seem to be shell termination within shell_structure" |
| `read_param_legacy.py` | 139-141 | "remove _summary, provide only yaml" |
| `input_warnings.py` | 29-33 | 5 validation rules listed but never implemented |
| `bubble_luminosity_modified.py` | 104 | "Add f_cover from fragmentation mechanics" |
| `run_energy_phase_modified.py` | 266 | "I think this is not used anymore" (EarlyPhaseApproximation) |

### Stale/Legacy Files

| File | Status |
|------|--------|
| `read_param_legacy.py` | Legacy reader with incomplete refactoring |
| `read_coolingcurve_old.py` | Old cooling curve reader with unused imports |
| `read_cloudy_old.py` | Depends on environment variable `WARPFIELD3_SETTING_MODULE` |
| `__cloudy__.py` | Broken imports, references dead code |
| `get_InitNewCluster.py` | Broken imports, references dead variable |
| `get_InitCloudyDens.py` | Trivial; could be inlined |
| `pressure_blend.py` | Self-documented as DEPRECATED |

---

## 13. Module-by-Module Summary Table

| Module | Files | Lines | Severity | Key Issues |
|--------|-------|-------|----------|------------|
| `_functions/` | 5 | ~750 | Low | `print()` in operations.py, humanfriendly dep, module singleton |
| `_input/` | 7 | ~2,800 | Medium | Duplicate parse_value, orphaned create_dictionary.py, empty input_warnings |
| `_output/` | 5 | ~2,300 | Low | Large trinity_reader.py, minor bare excepts in simulation_end |
| `_calc/` | 11 | ~16,000 | Low | Monolithic files (8 exceed 1000 lines), but generally well-written |
| `_plots/` | 28 | ~8,000 | Low | Style is consistent, no critical issues |
| `phase0_init/` | 4 | ~1,100 | Medium | BUG-3 in get_InitNewCluster, broken __cloudy__ integration |
| `phase1_energy/` | 2 | ~810 | Low | Clean design with pure ODE functions |
| `phase1b_energy_implicit/` | 2 | ~1,675 | Medium | DRY violations, broad exceptions, 80% duplicate in residual functions |
| `phase1c_transition/` | 2 | ~870 | Medium | DRY violations, unused constant, dissolution logic |
| `phase2_momentum/` | 2 | ~900 | Medium | DRY violations, inconsistent log levels |
| `phase_general/` | 2 | ~710 | Low | Well-designed event system; pressure_blend.py deprecated |
| `bubble_structure/` | 2 | ~1,240 | Medium | Magic numbers, assert crashes, unhandled brentq |
| `cloud_properties/` | 4 | ~2,030 | Medium | Bare excepts in powerLawSphere, helper duplication |
| `shell_structure/` | 3 | ~640 | Medium | Dead code, potential infinite loop, missing f_cover |
| `cooling/` | 5 | ~960 | Medium | Magic numbers, precision issues in read_cloudy |
| `sb99/` | 2 | ~660 | **CRITICAL** | BUG-1 in update_feedback.py |
| `cloudy/` | 1 | ~520 | **CRITICAL** | BUG-2 — broken imports |

---

## 14. Prioritized Recommendations

### P0 — Fix Immediately

1. **Resolve BUG-1** (`update_feedback.py:180`): Determine whether `v_mech_total` should use `pdot_total` or `pdot_W`, and make the code match the documentation.

2. **Guard against BUG-2 and BUG-3**: Either fix the broken imports in `__cloudy__.py` and `get_InitNewCluster.py`, or add explicit `raise NotImplementedError("Cloudy/recollapse support not yet implemented")` at their entry points so they fail clearly rather than with cryptic NameErrors.

3. **Add NaN/Inf guards** before every `solve_ivp` call across all phase runners.

### P1 — Fix Soon

4. **Extract shared phase helpers** (`compute_max_dex_change`, `get_monitor_values`, shell mass update logic, dissolution check) into `src/phase_general/phase_helpers.py`. This eliminates ~200 lines of duplicated code.

5. **Consolidate `get_residual_pure()` and `get_residual_detailed()`** in `get_betadelta_modified.py` into a single function with an optional `detailed=False` parameter.

6. **Remove all dead commented-out code** (main.py lines 354-716, get_bubbleParams.py lines 70-155, get_shellParams.py lines 27-58). Total: ~500+ lines.

7. **Replace `print()` with `logging`** in `operations.py`, `powerLawSphere.py`, and `header.py`.

8. **Replace bare `except:` and `except Exception:` clauses** with specific exception types across the 10+ files listed in Section 7.1.

### P2 — Improve When Convenient

9. **Document all magic numbers** with named constants and literature references (especially in bubble_luminosity_modified.py and phase runners).

10. **Delete deprecated files** or mark them clearly: `pressure_blend.py`, `read_param_legacy.py`, `read_coolingcurve_old.py`, `read_cloudy_old.py`, `get_InitCloudyDens.py`.

11. **Fix `input_warnings.py`** — either implement the validation checks listed in its TODOs, or remove the file and move validation to `read_param.py` where it already partially exists.

12. **Standardize naming conventions** — adopt a project-wide style guide. Suggestion: camelCase for physics quantities (`mCloud`, `rCloud`, `nCore`), snake_case for everything else (`shell_mass`, `current_phase`).

13. **Add `__hash__ = None`** to `DescribedItem` to make the broken hash behavior explicit (since `__eq__` is overridden).

14. **Refactor `_calc/` monolithic files** — separate computation from plotting.

15. **Handle `brentq` bracket failures** gracefully in `energy_phase_ODEs_modified.py` and `bubble_luminosity_modified.py`.

### P3 — Nice to Have

16. Replace `humanfriendly` dependency in `clock.py` with stdlib `datetime` formatting.
17. Add type hints throughout physics modules.
18. Consider binary snapshot format for array-heavy data.
19. Add a comprehensive test suite (currently no tests directory visible).
20. Fix `from __future__ import annotations` placement in `dictionary.py`.
