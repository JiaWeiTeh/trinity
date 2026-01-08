# COMPREHENSIVE ANALYSIS: net_coolingcurve.py

**File:** `src/cooling/net_coolingcurve.py`
**Lines of Code:** 167
**Analysis Date:** 2026-01-07
**Overall Severity:** MEDIUM - Complex logic with several issues

---

## PURPOSE

This file provides the master cooling rate function that combines both CIE (Collisional Ionization Equilibrium) and non-CIE cooling regimes. It intelligently switches between:
- **Non-CIE:** For T ≤ 10^5.5 K (depends on n, T, φ)
- **CIE:** For T ≥ 10^5.5 K (depends only on T)
- **Interpolation:** For temperatures in between

**Key Function:**
- `get_dudt(age, ndens, T, phi, params_dict)` - Returns net cooling rate du/dt [M_sun/pc/yr³]

---

## WHAT IT DOES

### Function: `get_dudt(age, ndens, T, phi, params_dict)`

**Inputs:**
- `age` [Myr]: Current age (UNUSED in function body!)
- `ndens` [1/pc³]: Number density
- `T` [K]: Temperature
- `phi` [1/pc²/Myr]: Photon flux
- `params_dict`: Dictionary with cooling structures

**Process:**
1. Convert units from AU (pc-based) to CGS (cm-based)
2. Read cached cooling structures from params_dict
3. Determine temperature regime (non-CIE, CIE, or transition)
4. Calculate cooling rate for that regime
5. Return -1 × dudt (negative because cooling removes energy)

**Output:**
- `dudt` [M_sun/pc/yr³] = [erg/cm³/s]: Net cooling rate (negative)

---

## CRITICAL ISSUES IDENTIFIED

### CRITICAL ISSUE #1: Temperature Clamping Without Warning

**Severity:** HIGH
**Lines:** 85-86
**Impact:** Silent physics errors

**The Problem:**
```python
if T < 1e4:
    T = 1e4
```

**Why this is CRITICAL:**
- **Silently modifies input** without telling user
- **No logging, no warning** that temperature was changed
- Can mask underlying problems (why is T < 10^4 K?)
- Breaks physics: if shell actually has T = 8000 K, code pretends it's 10,000 K
- **Different cooling rate** than reality

**Comment says (Lines 78-83):**
> "The reason im adding this is because the temperature seem to run at some very low value (~1e3.91) and the lowest available value of the cooling file that we use is only until 3.99. Not sure why though, as the temperature should be around 1e7, not 1e4."

**This is a HUGE RED FLAG!** The comment admits:
1. Temperature is wrong (should be 10^7, actually 10^3.91)
2. Don't know why
3. "Fixed" by clamping instead of finding root cause

**Correct Approach:**
```python
if T < 1e4:
    logger.error(f"Temperature {T:.2e} K below cooling table minimum (10^4 K). "
                 f"This suggests a physics error. Check shell temperature calculation.")
    raise ValueError(f"Temperature {T} too low for cooling tables")
```

Or if clamping is truly needed:
```python
if T < T_min:
    logger.warning(f"Temperature {T:.2e} K clamped to minimum {T_min:.2e} K. "
                   f"Cooling rate may be inaccurate.")
    T = T_min
```

---

### CRITICAL ISSUE #2: Deprecated Parameters in Docstring

**Severity:** MEDIUM (misleading documentation)
**Lines:** 22-37

**The Problem:**
Docstring says inputs are `age, ndens, T, phi`, but:
- `age` parameter is **NEVER USED** in function body!
- Originally age was used (see commented code line 72)
- Now age comes from `params_dict['t_now']` in `read_cloudy.py`

**Impact:**
- Confusing for users (what should age be?)
- Function signature doesn't match actual behavior
- Misleading documentation

**Fix:**
Remove `age` parameter entirely:
```python
def get_dudt(ndens, T, phi, params_dict):
    """..."""
```

Or keep but clarify:
```python
def get_dudt(age, ndens, T, phi, params_dict):
    """
    age: DEPRECATED - No longer used, kept for backward compatibility.
          Age is read from params_dict['t_now'].value instead.
    """
```

---

### HIGH ISSUE #3: Inconsistent Unit Conversions

**Severity:** HIGH
**Lines:** 45-46

**The Problem:**
```python
ndens /= cvt.ndens_cgs2au  # pc^-3 to cm^-3
phi /= cvt.phi_cgs2au      # 1/pc^2/yr to 1/cm^2/s
```

**Issues:**
1. **Variable names are misleading:** `cgs2au` suggests converting FROM cgs TO au, but code does opposite (au → cgs)
2. **No validation:** What if input is already in CGS? Double conversion!
3. **Modifies inputs in-place:** Changes ndens and phi (side effect)

**Correct Approach:**
```python
# Convert from AU to CGS (clearer naming!)
ndens_cgs = ndens / cvt.ndens_cgs2au  # [pc^-3] → [cm^-3]
phi_cgs = phi / cvt.phi_cgs2au        # [pc^-2 yr^-1] → [cm^-2 s^-1]

# Use ndens_cgs and phi_cgs from here on
```

---

### HIGH ISSUE #4: Complex Nested If/Elif Logic

**Severity:** MEDIUM (maintainability)
**Lines:** 102-163

**The Problem:**
Four deeply nested branches based on temperature:
1. T ≤ nonCIE_Tcutoff and T ≥ min(cooling_nonCIE.temp) → non-CIE
2. T ≥ CIE_Tcutoff → CIE
3. nonCIE_Tcutoff < T < CIE_Tcutoff → Interpolation
4. else → Error

**Issues:**
- Hard to follow logic
- Duplicated code (dudt conversion appears 3 times)
- Magic numbers (5.5 = log10(10^5.5 K))
- Tight coupling to data structure

**Better Structure:**
```python
def get_dudt(ndens, T, phi, params_dict):
    # Clamp/validate temperature
    T_clamped = validate_temperature(T, T_min, T_max)

    # Determine regime
    regime = determine_cooling_regime(T_clamped, T_nonCIE_max, T_CIE_min)

    # Calculate based on regime
    if regime == 'non_CIE':
        dudt_cgs = calculate_nonCIE_cooling(ndens_cgs, T_clamped, phi_cgs, params_dict)
    elif regime == 'CIE':
        dudt_cgs = calculate_CIE_cooling(ndens_cgs, T_clamped, params_dict)
    elif regime == 'transition':
        dudt_cgs = interpolate_cooling(ndens_cgs, T_clamped, phi_cgs, params_dict)
    else:
        raise ValueError(f"Temperature {T} K outside valid range")

    # Convert back to AU
    return -1 * dudt_cgs * cvt.dudt_cgs2au
```

---

### MEDIUM ISSUE #5: Commented-Out Code

**Severity:** LOW (code smell)
**Lines:** 48-49, 63, 71-74, 107-110, 136-139

**Examples:**
```python
# Line 48-49:
# ndens = ndens * (1/u.cm**3)
# phi = phi * (1/u.cm**2/u.s)

# Lines 107-110:
# netcooling grid (depreciated)
# netcooling = cooling_nonCIE.datacube - heating_nonCIE.datacube
# create interpolation function (depreciated)
# f_dudt = scipy.interpolate.RegularGridInterpolator(...)
```

**Impact:**
- Clutters code
- Confusing (is this needed? Should I uncomment?)
- Shows unclear development history

**Fix:** Remove all commented code, use version control instead

---

### MEDIUM ISSUE #6: Typo "depreciated" → "deprecated"

**Severity:** TRIVIAL
**Lines:** 71, 107, 136

```python
# depreciated  # ← Wrong spelling!
```

Should be "deprecated" (meaning outdated), not "depreciated" (accounting term for value decrease).

---

### MEDIUM ISSUE #7: Using print() Instead of logging

**Severity:** LOW
**Lines:** 48-49 (commented out)

Commented-out print statements suggest debugging was done with print. Should use logging module.

---

### MEDIUM ISSUE #8: Magic Numbers Without Constants

**Severity:** MEDIUM
**Lines:** 85, 93, 95

**The Problem:**
```python
if T < 1e4:         # Magic number!
    T = 1e4

nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp <= 5.5])  # Magic 5.5!
CIE_Tcutoff = min(logT_CIE[logT_CIE > 5.5])                           # Magic 5.5!
```

**Better:**
```python
# Physical constants at module level
T_MIN_COOLING = 1e4  # K - Minimum temperature for cooling tables
LOG_T_TRANSITION = 5.5  # log10(K) - Transition between non-CIE and CIE

# In function:
if T < T_MIN_COOLING:
    T = T_MIN_COOLING

nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp <= LOG_T_TRANSITION])
CIE_Tcutoff = min(logT_CIE[logT_CIE > LOG_T_TRANSITION])
```

---

### MEDIUM ISSUE #9: Inconsistent Return Pattern

**Severity:** LOW
**Lines:** 120, 127, 158

All three branches return the same transformation:
```python
return -1 * dudt * cvt.dudt_cgs2au
```

This should be done ONCE at the end, not three times.

**Better:**
```python
# Calculate dudt_cgs based on regime
if regime == 'non_CIE':
    dudt_cgs = ...
elif regime == 'CIE':
    dudt_cgs = ...
else:
    dudt_cgs = ...

# Convert and return ONCE
return -1 * dudt_cgs * cvt.dudt_cgs2au
```

---

### LOW ISSUE #10: Unclear Variable Naming

**Severity:** LOW
**Lines:** 62, 64, 66, 67

**The Problem:**
```python
cooling_nonCIE = params_dict['cStruc_cooling_nonCIE'].value
netcool_interp = params_dict['cStruc_net_nonCIE_interpolation'].value
CIE_interp = params_dict['cStruc_cooling_CIE_interpolation'].value
logT_CIE = params_dict['cStruc_cooling_CIE_logT'].value
```

**Issues:**
- Abbreviations: `cStruc` = cooling structure?
- Inconsistent: `cooling_nonCIE` vs `CIE_interp`
- No type hints or documentation

**Better:**
```python
# Non-CIE cooling structure (contains datacube and interpolator)
nonCIE_cooling_structure = params_dict['cooling_structure_nonCIE'].value
nonCIE_interpolator = params_dict['interpolator_net_nonCIE'].value

# CIE cooling interpolator and temperature array
CIE_interpolator = params_dict['interpolator_cooling_CIE'].value
CIE_log_temperatures = params_dict['log_temperatures_CIE'].value
```

---

## CORRECTNESS CHECK

### Physics: ⚠️ MOSTLY CORRECT with caveats

**Correct:**
- ✓ Switches between non-CIE and CIE regimes
- ✓ Interpolates in transition region
- ✓ Formula for CIE cooling: dudt = n² × Lambda(T)
- ✓ Uses pre-computed cooling structures

**Questionable:**
- ⚠️ Temperature clamping (Line 85-86): Masks underlying problem
- ⚠️ Transition at 10^5.5 K: Is this physical? (probably from cooling table limits)

### Mathematics: ✓ CORRECT

- ✓ Linear interpolation between non-CIE and CIE at transition
- ✓ Unit conversions appear correct (assuming cvt.* are correct)
- ✓ Log-space interpolation for non-CIE

---

## PERFORMANCE ANALYSIS

**Current Performance:**
- Fast (uses pre-computed interpolators from params_dict)
- No expensive calculations per call
- Caching strategy (cooling structure updated every 5000 years) is good

**No bottlenecks identified** in this file.

---

## CODE QUALITY SUMMARY

| Issue | Severity | Line | Impact |
|-------|----------|------|--------|
| Temperature clamping without warning | HIGH | 85-86 | Silent physics errors |
| Unused 'age' parameter | MEDIUM | 22 | Confusing interface |
| In-place unit conversion | MEDIUM | 45-46 | Side effects |
| Complex nested logic | MEDIUM | 102-163 | Hard to maintain |
| Commented-out code | LOW | Multiple | Clutter |
| Magic numbers | MEDIUM | 85, 93, 95 | Unclear constants |
| Typo "depreciated" | TRIVIAL | Multiple | Professionalism |
| Duplicated return logic | LOW | 120, 127, 158 | DRY violation |

---

## RECOMMENDATIONS

### Critical (Must Fix):
1. **Remove silent temperature clamping** or add loud warnings
2. **Investigate root cause** of T < 10^4 K (comment line 78-83)
3. **Remove or document unused 'age' parameter**

### High Priority:
4. **Refactor if/elif logic** into separate helper functions
5. **Add proper input validation**
6. **Use logging instead of silent modifications**

### Medium Priority:
7. **Remove all commented-out code**
8. **Replace magic numbers with named constants**
9. **Consolidate duplicated return statements**
10. **Fix "depreciated" → "deprecated" typo**

### Low Priority:
11. **Improve variable naming**
12. **Add type hints**
13. **Add unit tests**

---

## REFACTORED VERSION

See `REFACTORED_net_coolingcurve.py` for improved version with:
- No silent temperature clamping (raises error or warns loudly)
- Refactored regime selection logic
- Helper functions for each cooling regime
- Named constants instead of magic numbers
- Proper logging
- Cleaner structure
- Removed dead code

---

## TESTING RECOMMENDATIONS

1. **Test all three regimes:**
   ```python
   # Non-CIE regime (T < 10^5.5 K)
   dudt_low = get_dudt(1e2, 1e5, 1e8, params)

   # CIE regime (T > 10^5.5 K)
   dudt_high = get_dudt(1e2, 1e7, 1e8, params)

   # Transition regime
   dudt_mid = get_dudt(1e2, 10**(5.45), 1e8, params)
   ```

2. **Test temperature clamping:**
   ```python
   # What happens at T = 1e3?
   dudt_clamp = get_dudt(1e2, 1e3, 1e8, params)
   # Should this raise error or warn?
   ```

3. **Test edge cases:**
   ```python
   # T exactly at cutoffs
   dudt_cutoff_low = get_dudt(1e2, 10**nonCIE_Tcutoff, 1e8, params)
   dudt_cutoff_high = get_dudt(1e2, 10**CIE_Tcutoff, 1e8, params)
   ```

4. **Test continuity at transition:**
   ```python
   # Should be smooth transition
   T_arr = np.linspace(10**5.4, 10**5.6, 50)
   dudt_arr = [get_dudt(1e2, T, 1e8, params) for T in T_arr]

   # Check no discontinuities
   assert np.all(np.abs(np.diff(dudt_arr)) < threshold)
   ```

---

## SUMMARY

**Overall Severity:** MEDIUM

This file has **one critical issue** (silent temperature clamping) and several medium-severity issues (unused parameter, complex logic, magic numbers).

**Main Risks:**
1. **Silent physics errors** from temperature clamping
2. **Confusing interface** (unused age parameter)
3. **Maintainability issues** (complex nested logic)

**Priority Actions:**
1. Fix temperature clamping (add warnings or errors)
2. Investigate why T < 10^4 K occurs
3. Refactor if/elif logic for clarity

**Estimated Fix Time:** 2-3 hours for critical issues, 1 day for full refactoring

---

## RELATED FILES

- `src/cooling/CIE/read_coolingcurve.py` - Called for CIE regime
- `src/cooling/non_CIE/read_cloudy.py` - Provides non-CIE cooling structures
- `src/phase1b_energy_implicit/run_energy_implicit_phase.py` - Calls this function

---

## CONCLUSION

This file **works** but has a **critical hidden issue**: silent temperature clamping that masks underlying physics problems. The comment admitting "not sure why" temperature is wrong is a red flag.

**Recommendation:** Before fixing code style, **investigate root cause** of low temperatures. Once that's resolved, refactor for clarity and robustness.
