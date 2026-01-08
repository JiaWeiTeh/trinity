# COMPREHENSIVE ANALYSIS: read_SB99.py

**File:** `src/sb99/read_SB99.py`
**Lines of Code:** 230
**Analysis Date:** 2026-01-07
**Overall Severity:** MEDIUM - Simple structure but has error handling and code quality issues

---

## PURPOSE

This file reads Starburst99 stellar evolution model output files and processes them for use in the TRINITY simulation. Starburst99 provides time-dependent feedback parameters (ionizing photons, luminosities, mechanical energy, momentum) for stellar clusters.

**Key Functions:**
1. `read_SB99(f_mass, params)` - Main function: loads SB99 file, scales by cluster mass
2. `get_filename(params)` - Constructs filename from simulation parameters
3. `get_interpolation(SB99, ftype='cubic')` - Creates scipy interpolators for feedback evolution

---

## WHAT IT DOES

### Function 1: `read_SB99(f_mass, params)`

**Purpose:** Load and process Starburst99 data files.

**Process:**
1. Construct filename from params (Line 57)
2. Load SB99 file with np.loadtxt (Line 61)
3. Extract columns:
   - t [yr] → convert to [Myr] (Line 65)
   - Qi (ionizing photon rate), fi (ionizing fraction)
   - Lbol (bolometric luminosity), Lmech (mechanical luminosity)
   - pdot_W (wind momentum rate), Lmech_W (wind luminosity)
4. Calculate derived quantities (Lines 82-86):
   - Li = Lbol × fi (ionizing luminosity)
   - Ln = Lbol × (1-fi) (non-ionizing luminosity)
   - Lmech_SN = Lmech - Lmech_W (supernova luminosity)
5. Apply feedback scaling factors (Lines 92-122):
   - Winds: Adjust for cold mass fraction, thermal efficiency
   - SNe: Adjust for cold mass fraction, thermal efficiency
6. Insert t=0 values for interpolation (Lines 131-138)
7. Return [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN]

**Output:**
Arrays of stellar feedback parameters vs time

### Function 2: `get_filename(params)`

**Purpose:** Construct SB99 filename from simulation parameters.

**Format:** `[mass]cluster_[rotation]_[metallicity]_[blackholeCutoffMass].txt`
**Example:** `1e6cluster_rot_Z0014_BH120.txt`

**Process:**
1. Extract params: SB99_mass, SB99_rotation, ZCloud, SB99_BHCUT
2. Format mass as scientific notation (Lines 154-156)
3. Map rotation boolean → 'rot'/'norot'
4. Map metallicity → 'Z0014'/'Z0002'
5. Map BH cutoff → 'BH120'/'BH40'
6. Concatenate filename string

### Function 3: `get_interpolation(SB99, ftype='cubic')`

**Purpose:** Create scipy.interpolate.interp1d functions for all feedback quantities.

**Process:**
1. Unpack SB99 arrays
2. Create cubic interpolators for each quantity
3. Return dictionary of interpolation functions

---

## CRITICAL ISSUES IDENTIFIED

### CRITICAL ISSUE #1: Silent Unit Conversions

**Severity:** HIGH
**Lines:** 65, 68, 71, 73, 75, 77
**Impact:** Unclear units, potential for errors

**The Problem:**
```python
# Line 65: No validation that input is in years!
t = SB99_file[:,0] /1e6  # Assumes [yr] → [Myr]

# Lines 68-77: Scaling with f_mass and unit conversions
Qi = 10**SB99_file[:,1] * f_mass / cvt.s2Myr
Lbol = 10**SB99_file[:,3] * f_mass * cvt.L_cgs2au
# ... etc
```

**Issues:**
1. **No unit checking** - Assumes SB99_file has specific units
2. **Magic conversion factors** - cvt.s2Myr, cvt.L_cgs2au not documented here
3. **Mixed unit systems** - Converts from log10 to linear, CGS to AU, without validation
4. **No dimensional analysis** - Could multiply wrong units silently

**Better Approach:**
```python
import astropy.units as u

# Explicit unit handling
t_yr = SB99_file[:,0] * u.yr
t_Myr = t_yr.to(u.Myr).value

# Document expected units
Qi_log = SB99_file[:,1]  # log10(photons/s) for 1e6 Msun cluster
Qi = (10**Qi_log * u.s**-1) * f_mass  # Scale by actual mass
Qi_au = (Qi / u.Myr).to(u.yr**-1).value  # Convert to AU units
```

---

### CRITICAL ISSUE #2: Broken Exception Handling

**Severity:** HIGH
**Lines:** 180-182
**Impact:** Misleading error messages, undefined variable

**The Problem:**
```python
try:
    # ... build filename (Lines 152-178)
    filename = SBmass_str + 'cluster_' + rot_str + '_' + z_str + '_' + BH_str + '.txt'
    return filename
except Exception as e:
    print(e)
    # BUG: filename might not be defined here if error occurs before Line 178!
    raise Exception(f"Starburst99 file {filename} not found. ...")
```

**What's wrong:**
1. **Undefined variable:** If error occurs before Line 178, `filename` doesn't exist yet
2. **Misleading message:** Says "file not found" but error might be something else (KeyError, TypeError, etc.)
3. **Over-broad try/except:** Catches ALL exceptions, not just file-related ones

**Example Failure:**
```python
# If params['SB99_mass'] raises KeyError:
try:
    SBmass_str = format_e(params['SB99_mass'])  # KeyError!
except Exception as e:
    print(e)  # "KeyError: 'SB99_mass'"
    raise Exception(f"Starburst99 file {filename} not found...")  # NameError: filename not defined!
```

**Correct Approach:**
```python
try:
    SBmass_str = format_e(params['SB99_mass'])
    rot_str = 'rot' if params['SB99_rotation'] else 'norot'
    z_str = 'Z0014' if params['ZCloud'] == 1.0 else 'Z0002'
    BH_str = f"BH{params['SB99_BHCUT']}"
    filename = f"{SBmass_str}cluster_{rot_str}_{z_str}_{BH_str}.txt"
    return filename
except KeyError as e:
    raise ValueError(f"Missing required parameter: {e}")
except Exception as e:
    raise RuntimeError(f"Error constructing SB99 filename: {e}")
```

---

### HIGH ISSUE #3: Hardcoded Metallicity Mapping

**Severity:** MEDIUM
**Lines:** 164-169, 171-176
**Impact:** Only supports 2 metallicities, brittle code

**The Problem:**
```python
if params['ZCloud'] == 1.0:
    # solar
    z_str = 'Z0014'
elif params['ZCloud'] == 0.15:
    # 0.15 solar
    z_str = 'Z0002'
# What if ZCloud = 0.5? → z_str undefined → error!
```

**Issues:**
1. **Only 2 values supported:** Z = 1.0 or 0.15 solar
2. **No error for other values:** Falls through without setting z_str
3. **Magic numbers:** 0.15, 1.0 not defined as constants
4. **TODO comment (Line 18) never implemented:** "Implement interpolation function for in-between metallicities"

**Impact:**
If user sets ZCloud = 0.5 (intermediate metallicity):
- z_str never gets defined
- Line 178: `filename = ... + z_str + ...` → NameError!

**Better Approach:**
```python
# Define supported metallicities
Z_SOLAR = 1.0
Z_LOW = 0.15
Z_TOLERANCE = 0.01

if abs(params['ZCloud'] - Z_SOLAR) < Z_TOLERANCE:
    z_str = 'Z0014'
elif abs(params['ZCloud'] - Z_LOW) < Z_TOLERANCE:
    z_str = 'Z0002'
else:
    raise ValueError(
        f"Metallicity ZCloud = {params['ZCloud']} not supported. "
        f"Available: {Z_SOLAR} (solar), {Z_LOW} (0.15 solar)"
    )
```

---

### HIGH ISSUE #4: Physics Calculation Without Validation

**Severity:** MEDIUM
**Lines:** 93-94, 113
**Impact:** Division by zero potential, NaN propagation

**The Problem:**
```python
# Lines 93-94: Convert momentum and energy to mass loss rate and velocity
Mdot_W = pdot_W ** 2 / (2 * Lmech_W)  # What if Lmech_W = 0?
velocity_W = 2 * Lmech_W / pdot_W      # What if pdot_W = 0?

# Line 113:
Mdot_SN = 2 * Lmech_SN / velocity_SN**2  # What if velocity_SN = 0?
```

**When can this happen:**
- Early times: Lmech_W ≈ 0 (no winds yet)
- Late times: Lmech_SN ≈ 0 (no SNe)
- Bad data: NaN or inf in SB99 file

**Result:**
- Division by zero → inf
- 0/0 → NaN
- NaN propagates through all subsequent calculations

**Fix:**
```python
# Add small epsilon to prevent division by zero
EPSILON = 1e-50  # Very small positive number

Mdot_W = pdot_W ** 2 / (2 * (Lmech_W + EPSILON))
velocity_W = 2 * Lmech_W / (pdot_W + EPSILON)

# Or check and handle explicitly:
if Lmech_W > 0 and pdot_W > 0:
    Mdot_W = pdot_W ** 2 / (2 * Lmech_W)
    velocity_W = 2 * Lmech_W / pdot_W
else:
    logger.warning(f"Zero wind luminosity or momentum at t={t}")
    Mdot_W = 0.0
    velocity_W = 0.0
```

---

### MEDIUM ISSUE #5: Unused Imports

**Severity:** LOW
**Lines:** 14

```python
import sys  # Never used in file!
```

**Impact:** Minor code smell

---

### MEDIUM ISSUE #6: Inconsistent Variable Naming

**Severity:** LOW
**Throughout file**

**Examples:**
- `pdot_W` (underscore) vs `Lmech_W` (underscore) vs `Lbol` (no underscore)
- `Mdot_W` (capital M) vs `pdot_W` (lowercase p)
- `velocity_W` (full word) vs `v2` (abbreviation in other files)

**Better:** Consistent naming convention

---

### MEDIUM ISSUE #7: No Input Validation on f_mass

**Severity:** MEDIUM
**Lines:** 21, 68-77

**The Problem:**
```python
def read_SB99(f_mass, params):
    # No validation of f_mass!
    # What if f_mass = 0? → All feedback = 0
    # What if f_mass < 0? → Negative luminosities?
    # What if f_mass = NaN? → Everything becomes NaN

    Qi = 10**SB99_file[:,1] * f_mass / cvt.s2Myr
```

**Fix:**
```python
def read_SB99(f_mass, params):
    # Validate f_mass
    if f_mass <= 0:
        raise ValueError(f"f_mass must be positive, got {f_mass}")
    if np.isnan(f_mass) or np.isinf(f_mass):
        raise ValueError(f"f_mass must be finite, got {f_mass}")
```

---

### LOW ISSUE #8: Magic Number in format_e Function

**Severity:** LOW
**Lines:** 154-156

**The Problem:**
```python
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'e' + a.split('E')[1].strip('+').strip('0')
```

**Issues:**
1. **Nested function** - Should be module-level or use standard library
2. **Complex string manipulation** - Hard to understand
3. **No docstring** - What does this do?

**Better:**
```python
def format_scientific_notation(value: float) -> str:
    """
    Format number in scientific notation without trailing zeros.

    Example: 1000000.0 → '1e6', not '1.000000e+06'
    """
    # Python's format can do this directly
    return f"{value:.0e}".replace('+', '')
```

---

### LOW ISSUE #9: Commented-Out Code

**Severity:** LOW
**Lines:** 60, 64, 67

**Examples:**
```python
# Line 60:
# SB99_file = np.loadtxt(warpfield_params.path_sps + filename)

# Lines 64, 67:
# u.Myr
# / u.s
```

**Fix:** Remove commented code, use version control

---

### LOW ISSUE #10: No Docstring for Return Values

**Severity:** LOW
**Lines:** 21-51

**The Problem:**
Docstring lists return values (Lines 27-50) but doesn't match actual return statement (Line 141).

**Docstring says:**
```
Returns
-------
Here are the parameters directly from Starburst99 runs:
    t: time [yr]; however, saved as [Myr] in the output of this function.
    Qi: ...
    ...
```

**Actual return:**
```python
return [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN]
```

**Missing from docstring:**
- pdot (total momentum rate)
- Lmech (total mechanical luminosity)
- pdot_SN (SN momentum rate)

**Missing from return:**
- fi (ionizing fraction)
- Lmech_W (wind mechanical luminosity)

---

## CORRECTNESS CHECK

### Physics: ✓ MOSTLY CORRECT

**Correct:**
- ✓ Velocity from energy and momentum: v = 2L/p, M = p²/(2L)
- ✓ Feedback scaling for cold mass fraction
- ✓ Feedback scaling for thermal efficiency
- ✓ Splitting wind and SN contributions

**Questionable:**
- ⚠️ No validation (div by zero possible)
- ⚠️ Silent unit conversions

### Mathematics: ✓ CORRECT

- ✓ Log to linear conversion: 10**x
- ✓ Array operations with numpy
- ✓ Cubic interpolation appropriate

### Implementation: ⚠️ ISSUES

- ✗ Broken exception handling (undefined filename)
- ✗ No input validation (f_mass, params)
- ✗ Hardcoded metallicity mapping (brittle)
- ⚠️ Silent unit conversions

---

## PERFORMANCE ANALYSIS

**Current Performance:**
- Fast: Single file load with np.loadtxt (~10ms typical)
- Array operations are vectorized (good)
- No significant bottlenecks

**No performance issues identified**

---

## CODE QUALITY SUMMARY

| Issue | Severity | Lines | Impact |
|-------|----------|-------|--------|
| Silent unit conversions | HIGH | 65-77 | Unclear units, error-prone |
| Broken exception handling | HIGH | 180-182 | Misleading errors, undefined variable |
| Hardcoded metallicity | MEDIUM | 164-176 | Only 2 values supported |
| No div-by-zero check | MEDIUM | 93-94, 113 | NaN propagation |
| No f_mass validation | MEDIUM | 21 | Silent failures |
| Unused imports | LOW | 14 | Code smell |
| Commented-out code | LOW | 60, 64, 67 | Clutter |
| Inconsistent naming | LOW | Throughout | Readability |
| Complex format_e | LOW | 154-156 | Hard to understand |
| Incomplete docstring | LOW | 21-51 | Misleading docs |

---

## RECOMMENDATIONS

### Critical (Must Fix):
1. **Fix exception handling** (Lines 180-182)
   - Don't reference filename if not yet defined
   - Catch specific exceptions (KeyError, etc.)
2. **Add input validation** for f_mass
3. **Fix hardcoded metallicity** (handle unsupported values)

### High Priority:
4. **Add div-by-zero protection** (Lines 93-94, 113)
5. **Document unit conversions** explicitly
6. **Add logging** instead of silent operations

### Medium Priority:
7. **Remove unused imports** (sys)
8. **Remove commented-out code**
9. **Simplify format_e function**
10. **Fix docstring** to match actual return

### Low Priority:
11. **Consistent naming convention**
12. **Add type hints**
13. **Add unit tests**

---

## REFACTORED VERSION

See `REFACTORED_read_SB99.py` for improved version with:
- Proper exception handling
- Input validation (f_mass > 0, params complete)
- Explicit unit handling with astropy.units
- Division-by-zero protection
- Support for more metallicities (with clear errors)
- Proper logging
- Type hints
- Comprehensive docstrings
- No dead code

---

## TESTING RECOMMENDATIONS

1. **Test valid inputs:**
   ```python
   f_mass = 1.0  # Standard cluster
   SB99 = read_SB99(f_mass, params)
   assert len(SB99) == 8
   assert all(len(arr) > 0 for arr in SB99)
   ```

2. **Test edge cases:**
   ```python
   # Very small mass
   SB99_small = read_SB99(0.001, params)

   # Very large mass
   SB99_large = read_SB99(100.0, params)
   ```

3. **Test error handling:**
   ```python
   # Negative mass
   with pytest.raises(ValueError):
       read_SB99(-1.0, params)

   # Zero mass
   with pytest.raises(ValueError):
       read_SB99(0.0, params)

   # Missing param
   params_bad = params.copy()
   del params_bad['SB99_mass']
   with pytest.raises(ValueError):
       read_SB99(1.0, params_bad)
   ```

4. **Test metallicity handling:**
   ```python
   # Supported metallicities
   params_solar = params.copy()
   params_solar['ZCloud'] = 1.0
   SB99_solar = read_SB99(1.0, params_solar)

   # Unsupported metallicity
   params_bad = params.copy()
   params_bad['ZCloud'] = 0.5
   with pytest.raises(ValueError):
       get_filename(params_bad)
   ```

---

## SUMMARY

**Overall Severity:** MEDIUM

This file is **mostly functional** but has **critical bugs in error handling** and **lacks robustness**:

1. **Broken exception handling** (undefined variable in error message)
2. **No input validation** (f_mass, params)
3. **Hardcoded metallicity** (only 2 values, silent failure for others)
4. **No div-by-zero protection** (NaN propagation possible)
5. **Silent unit conversions** (unclear units, error-prone)

**Main Risks:**
- Misleading error messages (broken exception)
- Silent failures (bad f_mass, unsupported Z)
- NaN propagation (div by zero)

**Priority Actions:**
1. Fix exception handling (Line 180-182)
2. Add input validation for f_mass
3. Handle unsupported metallicities explicitly
4. Add div-by-zero checks

**Estimated Fix Time:** 2-3 hours for critical issues

---

## RELATED FILES

- `getSB99_data.py` - More complex version from WARPFIELD
- `update_feedback.py` - Calls this to get current feedback
- `run_energy_implicit_phase.py` - Uses SB99 feedback

---

## CONCLUSION

This file **works for happy path** but **fails poorly** on edge cases. The broken exception handling (Line 180-182) is a **critical bug** that needs immediate fixing. Adding validation and better error handling would make it much more robust.

**Recommendation:** Fix critical bugs immediately, then refactor for clearer unit handling and better documentation.
