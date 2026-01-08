# COMPREHENSIVE ANALYSIS: read_coolingcurve.py

**File:** `src/cooling/CIE/read_coolingcurve.py`
**Lines of Code:** 76
**Analysis Date:** 2026-01-07
**Overall Severity:** LOW - Simple file, mostly functional

---

## PURPOSE

This file provides a simple function to calculate the cooling function Lambda(T) assuming Collisional Ionization Equilibrium (CIE) conditions. In CIE, Lambda depends only on temperature T, unlike non-CIE where it depends on (n, T, φ).

**Key Function:**
- `get_Lambda(T, cooling_CIE_interpolation, metallicity)` - Returns Lambda [erg/s·cm³] for given temperature

---

## WHAT IT DOES

### Function: `get_Lambda(T, cooling_CIE_interpolation, metallicity)`

**Inputs:**
- `T` [K]: Temperature (float or array)
- `cooling_CIE_interpolation`: scipy.interpolate function (log(T) → log(Lambda))
- `metallicity`: Metallicity (solar = 1.0)

**Process:**
1. Converts T to log10(T)
2. Interpolates log(Lambda) from cooling curve
3. Converts back to linear: Lambda = 10^(interp(log(T)))

**Output:**
- `Lambda` [erg/s·cm³]: Cooling function

**Usage:**
```python
# Assuming cooling_CIE_interpolation is pre-loaded
T = 1e6  # K
Lambda = get_Lambda(T, cooling_CIE_interpolation, metallicity=1.0)
```

---

## ISSUES IDENTIFIED

### ISSUE #1: Commented-Out Code (Lines 60-68)

**Severity:** LOW (code smell)
**Lines:** 60-68

**The Problem:**
```python
# if metallicity != 1:
#     sys.exit('Need to implement non-solar metallicity.')
# # get path to library
# # See example_pl.param for more information.
# path2cooling = warpfield_params.path_cooling_CIE
# # unpack from file
# logT, logLambda = np.loadtxt(path2cooling, unpack = True)
# # create interpolation
# cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind = 'linear')
```

**Why it's a problem:**
- Dead code clutters the file
- Confusing for readers (is this needed?)
- Unclear what the intention was
- The comment suggests non-solar metallicity is TODO, but parameter is unused

**Impact:** Maintainability (low severity)

---

### ISSUE #2: No Input Validation

**Severity:** MEDIUM
**Lines:** 25-74

**The Problem:**
```python
def get_Lambda(T, cooling_CIE_interpolation, metallicity):
    # No checks if T is valid
    T = np.log10(T)  # What if T <= 0?
    Lambda = 10**(cooling_CIE_interpolation(T))  # What if T outside interpolation range?
    return Lambda
```

**What could go wrong:**
1. **T ≤ 0:** `np.log10(T)` raises warning/error
2. **T outside interpolation range:** Extrapolation fails (scipy.interpolate.interp1d doesn't extrapolate by default)
3. **T = NaN:** Returns NaN without warning

**Impact:** Can cause silent failures or cryptic errors

---

### ISSUE #3: Metallicity Parameter Unused

**Severity:** LOW
**Lines:** 25, 60-61

**The Problem:**
```python
def get_Lambda(T, cooling_CIE_interpolation, metallicity):
    # ...
    # metallicity is NEVER USED!
    # if metallicity != 1:
    #     sys.exit('Need to implement non-solar metallicity.')
```

**Why it's a problem:**
- Function signature implies metallicity affects output
- Actually, it does nothing
- Misleading for users

**Impact:** Confusing interface, potential future bugs

---

### ISSUE #4: No Bounds Checking on Interpolation

**Severity:** MEDIUM
**Lines:** 56-58, 72

**The Problem:**
```python
# Comment says:
# Might be a problem here because this does not support extrapolation. If
# this happens, implement a function that does that.

# But then does nothing about it!
Lambda = 10**(cooling_CIE_interpolation(T))
```

**What happens:**
- If T outside interpolation range, scipy raises `ValueError: A value in x_new is outside the interpolation range`
- No graceful handling
- No warning to user

**Impact:** Runtime crashes for edge cases

---

### ISSUE #5: Unused Imports

**Severity:** LOW
**Lines:** 14-16

**The Problem:**
```python
import sys  # Never used
import astropy.units as u  # Never used
```

**Impact:** Minor code smell

---

### ISSUE #6: Typo in Docstring

**Severity:** TRIVIAL
**Lines:** 34-36

**The Problem:**
```python
# Available libraries (specified in .param file) include:
#     1: CLOUDY cooling curve for HII region, solar metallicity.
#     2: CLOUDY cooling curve for HII region, solar metallicity.
#         Includes the evaporative (sublimation) cooling of icy interstellar
#         grains (occurs e.g., when heated by cosmic-ray particle)
```

Options 1 and 2 have identical first lines (both say "solar metallicity"). This looks like copy-paste error.

---

## CORRECTNESS CHECK

**Physics:** ✓ Correct
- CIE assumption: Lambda = Lambda(T) only ✓
- Log-space interpolation ✓
- Conversion back to linear ✓

**Mathematics:** ✓ Correct
- Log10 interpolation is standard practice ✓
- 10^(interp(log(T))) = correct ✓

**Overall:** The actual calculation is correct, but lacks robustness.

---

## PERFORMANCE ANALYSIS

**Current Performance:**
- Very fast (single interpolation call)
- No significant bottlenecks

**Potential Issues:**
- None for this simple function

---

## CODE QUALITY ISSUES

1. **Dead code** (commented out sections)
2. **Unused imports** (sys, astropy.units)
3. **Unused parameter** (metallicity)
4. **No error handling** (bounds checking)
5. **No input validation** (T > 0, T not NaN)
6. **Unclear docstring** (duplicate option descriptions)

---

## RECOMMENDATIONS

### Short Term (Critical):
1. **Add input validation:**
   ```python
   if np.any(T <= 0):
       raise ValueError(f"Temperature must be positive, got {T}")
   ```

2. **Add bounds checking:**
   ```python
   T_min, T_max = 1e4, 1e9  # Get from interpolation
   if np.any(T < T_min) or np.any(T > T_max):
       logger.warning(f"Temperature {T} outside interpolation range [{T_min}, {T_max}]")
   ```

### Medium Term:
3. **Remove dead code** (lines 60-68)
4. **Remove unused imports** (sys, astropy.units)
5. **Either use metallicity or remove it** from function signature

### Long Term:
6. **Implement non-solar metallicity** support (if needed)
7. **Add proper logging** instead of comments
8. **Add unit tests**

---

## REFACTORED VERSION

See `REFACTORED_read_coolingcurve.py` for improved version with:
- Input validation
- Bounds checking with graceful fallback
- Proper logging
- Dead code removed
- Cleaner docstring
- Optional extrapolation support

---

## TESTING RECOMMENDATIONS

1. **Test with valid inputs:**
   ```python
   T = np.logspace(4, 9, 100)  # 10^4 to 10^9 K
   Lambda = get_Lambda(T, interp, metallicity=1.0)
   assert np.all(Lambda > 0)
   ```

2. **Test edge cases:**
   ```python
   # T at boundaries
   Lambda_min = get_Lambda(T_min, interp, metallicity=1.0)
   Lambda_max = get_Lambda(T_max, interp, metallicity=1.0)

   # T outside range (should warn or handle gracefully)
   Lambda_low = get_Lambda(1e2, interp, metallicity=1.0)  # Too low
   Lambda_high = get_Lambda(1e12, interp, metallicity=1.0)  # Too high
   ```

3. **Test error handling:**
   ```python
   # Negative temperature
   with pytest.raises(ValueError):
       get_Lambda(-100, interp, metallicity=1.0)

   # Zero temperature
   with pytest.raises(ValueError):
       get_Lambda(0, interp, metallicity=1.0)
   ```

---

## SUMMARY

**Severity:** LOW

This file is relatively clean and functional, but lacks robustness:
- ✓ Physics is correct
- ✓ No major bugs
- ✗ No input validation
- ✗ No bounds checking
- ✗ Dead code present
- ✗ Unused imports/parameters

**Main Risk:** Silent failures when T is outside interpolation range or invalid.

**Fix Priority:** Medium (add validation/bounds checking)

**Estimated Fix Time:** 30 minutes

---

## RELATED FILES

- `net_coolingcurve.py` - Calls this function for CIE regime
- `read_cloudy.py` - Provides non-CIE cooling (different approach)

---

## CONCLUSION

Overall, this is a simple, mostly correct file with minor issues. The main improvements needed are:
1. Input validation (T > 0, not NaN)
2. Bounds checking (warn if outside interpolation range)
3. Remove dead code
4. Clarify metallicity handling

None of these are critical bugs, but they would improve robustness and maintainability.
