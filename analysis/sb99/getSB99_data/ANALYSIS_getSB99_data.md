# COMPREHENSIVE ANALYSIS: getSB99_data.py

**File:** `src/sb99/getSB99_data.py`
**Lines of Code:** 645
**Analysis Date:** 2026-01-07
**Overall Severity:** CRITICAL - Contains syntax errors, missing imports, deprecated code

---

## PURPOSE

This file appears to be imported from WARPFIELD (another HII region simulation code) and provides advanced SB99 loading functionality including:
- Metallicity interpolation
- Multiple stellar population handling
- Time-shifted cluster combinations
- More comprehensive error handling than read_SB99.py

**Key Functions:**
1. `getSB99_main()` - Main entry point
2. `load_stellar_tracks()` - Load SB99 with metallicity handling
3. `getSB99_data()` - Core file reader
4. `getSB99_data_interp()` - Interpolate between metallicities
5. `make_interpfunc()` - Create interpolators
6. `sum_SB99()` - Combine multiple stellar populations
7. Various helper functions

---

## CRITICAL ISSUES - FILE IS BROKEN!

### CRITICAL ISSUE #1: SYNTAX ERROR

**Severity:** CRITICAL (CODE WILL NOT RUN)
**Lines:** 99-102
**Impact:** Python will not parse this file

**The Problem:**
```python
# Line 99-102:
elif (Zism < 1.0 and Zism > 0.15):
    [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data_interp(Zism,
                                                                                                   SB99_file_Z0002,
                                                                                                   0.15,
elif (Zism == 1.0 or (Zism <= 0.15 and Zism >= 0.14)):
```

**Line 102:** Starts with `elif` but previous `getSB99_data_interp()` call on Lines 99-101 is **INCOMPLETE**!

Missing:
- Closing parenthesis
- Fourth argument (SB99_file_Z0014)
- Fifth argument (1.0, the Z value for file 2)

**Correct syntax should be:**
```python
elif (Zism < 1.0 and Zism > 0.15):
    [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = getSB99_data_interp(
        Zism,
        SB99_file_Z0002,
        0.15,
        SB99_file_Z0014,  # ← MISSING!
        1.0,              # ← MISSING!
        f_mass=f_mass,
        test_plot=test_plot,
        log_t=log_t,
        tmax=tmax
    )
elif (Zism == 1.0 or (Zism <= 0.15 and Zism >= 0.14)):
```

**Result:** This file **CANNOT RUN** as-is. Python parser will fail.

---

### CRITICAL ISSUE #2: Missing Imports

**Severity:** CRITICAL
**Lines:** 5-11
**Impact:** ImportError on module load

**The Problem:**
```python
import auxiliary_functions as aux      # ← Does not exist in TRINITY!
import warp_nameparser                 # ← Does not exist in TRINITY!
import init as i                       # ← Does not exist in TRINITY!
```

These modules are from WARPFIELD codebase, not TRINITY. File cannot run without them.

**Used throughout file:**
- `aux.printl()` - Line 171
- `aux.find_nearest_lower()` - Lines 539, 596
- `warp_nameparser.get_SB99_filename()` - Lines 82-83
- `i.force_SB99file` - Line 55
- `i.SB99_mass` - Lines 82, 83, 561
- `i.f_Mcold_W`, `i.thermcoeff_clwind`, etc. - Lines 201-209

**Impact:** File is **NOT USABLE** in TRINITY without porting these dependencies.

---

### CRITICAL ISSUE #3: Deprecated Function Still Present

**Severity:** MEDIUM
**Lines:** 579-629
**Impact:** Dead code, confusing

**The Problem:**
```python
def sum_SB99_old(SB99_data1, SB99_data2, dtSF):
    """
    # depricated
    ...
```

**AND:**
```python
def time_shift_old(SB99_data, t):
    """
    add t to time vector in SB99_data
    ...
```

**Issues:**
1. Misspelled "depricated" → "deprecated"
2. Why keep deprecated code if there's a new version?
3. 50 lines of dead code (Lines 579-643)

---

### HIGH ISSUE #4: Inconsistent Parameter Naming

**Severity:** MEDIUM
**Lines:** 237, 244

**The Problem:**
```python
# Line 237: Function signature
def getSB99_data_interp(Zism, file1, Zfile1, file2, Zfile2, f_mass = 1.0):
    """
    interpolate metallicities from SB99 data
    :param Zism: metallicity you want (between metallicity 1 and metallicity 2)
    :param file1: path to file for tracks with metallicity 1
    :param Zfile1: metallicity 1
    :param file2: path to file for tracks with metallicity 2
    :param Zfile: metallicity 2  # ← BUG: Says "Zfile" not "Zfile2"!
    ...
```

**Docstring has typo:** Parameter is `Zfile2` but doc says `:param Zfile:`

---

### HIGH ISSUE #5: Dangerous Typo in Array Indexing

**Severity:** HIGH
**Lines:** 261

**The Problem:**
```python
# Line 260-261:
tend1 = t1[-1]
tend2 = t2[-2]  # ← BUG: Why [-2] not [-1]?
```

**This is INCONSISTENT and DANGEROUS:**
- `tend1` uses last element `[-1]`
- `tend2` uses second-to-last element `[-2]`

**Why?** Probably a typo or workaround for a bug elsewhere.

**Result:** Time arrays get cut to wrong length.

**Should be:**
```python
tend1 = t1[-1]
tend2 = t2[-1]  # Consistent!
tend = np.min([tend1, tend2])
```

---

### HIGH ISSUE #6: Print Statements Instead of Logging

**Severity:** MEDIUM
**Lines:** 93, 94, 109, 110, 119, 171, 287, 364

**Examples:**
```python
# Line 93-94:
print(("SB99file: " + SB99file))
print(("SB99cloudy_file +.txt: " + i.SB99cloudy_file))

# Line 287:
print("FATAL: files do not have the same time vectors")

# Line 364:
print("FATAL: files do not have the same time vectors")
```

**Issues:**
1. **Using print() not logging** - Can't control verbosity
2. **Inconsistent formatting** - Some with double parens `print(("..."))`
3. **"FATAL" messages** but doesn't always exit

---

### MEDIUM ISSUE #7: Commented-Out Code Everywhere

**Severity:** LOW (code smell)
**Lines:** 1, 173-175, 230-232, 401-425

**Examples:**
```python
# Line 1:
#from pyexcel_ods import get_data

# Lines 173-175:
#data_dict = get_data(file)
#data_list = data_dict['Sheet1']
#data = np.array(data_list)

# Lines 230-232:
# print('checkSB99')
# for ii in [t,Qi,Li,Ln,Lbol,Lmech,pdot,pdot_SN]:
#     print(np.sum(ii))

# Lines 401-425: Entire test script commented out!
```

**Impact:** 50+ lines of dead code cluttering file

---

### MEDIUM ISSUE #8: Duplicated Logic

**Severity:** MEDIUM
**Lines:** 249-258 duplicates 267-281, 275-281 duplicates 619-625

**Example:**
```python
# Lines 267-281: Cutting arrays to same length
Qi1 = Qi1[t1 <= tend]
Li1 = Li1[t1 <= tend]
Ln1 = Ln1[t1 <= tend]
# ... 7 lines total

# This pattern repeats 3 times in different functions!
```

**Should be helper function:**
```python
def trim_to_time(data_tuple, t_arr, t_max):
    """Trim all arrays to same time range."""
    mask = t_arr <= t_max
    return tuple(arr[mask] for arr in data_tuple)
```

---

### MEDIUM ISSUE #9: Complex Nested Conditionals

**Severity:** MEDIUM
**Lines:** 84-122

**The Problem:**
```python
if force_file != 0:  # case: specific file is forced
    # ... 10 lines
elif (Zism < 1.0 and Zism > 0.15):
    # ... interpolation case (BROKEN SYNTAX!)
elif (Zism == 1.0 or (Zism <= 0.15 and Zism >= 0.14)):
    # ... exact match case
else:
    # ... out of range case
```

**Issues:**
1. Complex nested conditions hard to follow
2. Magic numbers (1.0, 0.15, 0.14) not defined as constants
3. Different code paths do similar things (could be refactored)

---

### LOW ISSUE #10: Inconsistent Function Naming

**Severity:** LOW
**Throughout file**

**Examples:**
- `getSB99_main()` - camelCase
- `load_stellar_tracks()` - snake_case
- `getSB99_data()` - camelCase
- `getSB99_data_interp()` - camelCase with underscore
- `make_interpfunc()` - snake_case
- `getpdotLmech()` - camelCase no underscore
- `getMdotv()` - camelCase no underscore

**Python convention:** snake_case for functions

---

## ADDITIONAL ISSUES

### ISSUE #11: Deprecated collections.Mapping

**Severity:** LOW
**Lines:** 445, 460

```python
if isinstance(SB99_data, collections.Mapping):
```

**Python 3.3+:** `collections.Mapping` is deprecated, use `collections.abc.Mapping`

**Fix:**
```python
from collections.abc import Mapping

if isinstance(SB99_data, Mapping):
```

---

### ISSUE #12: Unsafe Path Handling

**Severity:** MEDIUM
**Lines:** 177-183

```python
if os.path.isfile(file):
    data = np.loadtxt(file)
elif os.path.isfile(pathlib.Path(__file__).parent / file):
    data = np.loadtxt(pathlib.Path(__file__).parent / file)
else:
    sys.exit("Specified SB99 file does not exist:", file)
```

**Issues:**
1. `sys.exit()` with tuple (should be string)
2. Should use pathlib throughout

**Correct:**
```python
file_path = Path(file)
if not file_path.is_file():
    file_path = Path(__file__).parent / file

if file_path.is_file():
    data = np.loadtxt(file_path)
else:
    raise FileNotFoundError(f"SB99 file not found: {file}")
```

---

## CORRECTNESS CHECK

### Physics: ⚠️ CANNOT VERIFY (Syntax errors)

Due to syntax errors and missing imports, **cannot run to verify physics**.

**Assuming syntax fixed:**
- ✓ Same physical formulas as read_SB99.py
- ✓ Metallicity interpolation approach seems sound
- ⚠️ Array indexing bug (Line 261) could affect results

### Mathematics: ⚠️ SUSPECT

- ⚠️ Inconsistent array indexing `t2[-2]` vs `t1[-1]`
- ✓ Linear interpolation for metallicity (appropriate)

### Implementation: ✗ BROKEN

- ✗ Syntax error (Line 99-102) - **WILL NOT RUN**
- ✗ Missing imports - **WILL NOT RUN**
- ✗ Array indexing bug (Line 261)
- ⚠️ Print instead of logging
- ⚠️ Deprecated code present

---

## RELATIONSHIP TO read_SB99.py

**getSB99_data.py** appears to be the **ORIGINAL** from WARPFIELD that **read_SB99.py** was simplified from:

| Feature | getSB99_data.py | read_SB99.py |
|---------|-----------------|--------------|
| Metallicity interpolation | ✓ Yes (Lines 237-300) | ✗ No (hardcoded 2 values) |
| Multiple populations | ✓ Yes (sum_SB99) | ✗ No |
| Time shifting | ✓ Yes | ✗ No |
| Dependencies | WARPFIELD modules | TRINITY only |
| Working state | ✗ Broken (syntax error) | ✓ Works |
| Complexity | 645 lines | 230 lines |

**Conclusion:** read_SB99.py is a **simplified, working version** for TRINITY. getSB99_data.py has **more features** but is **broken and not adapted** for TRINITY.

---

## USAGE IN TRINITY

**Question:** Is this file actually used in TRINITY?

Let me check imports:
```python
# This file imports from WARPFIELD:
import auxiliary_functions as aux
import warp_nameparser
import init as i
```

**These don't exist in TRINITY!** So either:
1. This file is **NOT USED** (legacy/reference code)
2. Or there are missing dependency files

**Recommendation:** If not used, **DELETE** this file or move to archive. If used, **FIX IMMEDIATELY** (syntax errors + missing imports).

---

## CODE QUALITY SUMMARY

| Issue | Severity | Lines | Impact |
|-------|----------|-------|--------|
| Syntax error | CRITICAL | 99-102 | File won't run |
| Missing imports | CRITICAL | 5-11 | ImportError |
| Array indexing bug | HIGH | 261 | Wrong results |
| Print not logging | MEDIUM | Multiple | No verbosity control |
| Deprecated code | MEDIUM | 579-643 | 50 lines dead code |
| Commented code | LOW | Multiple | 50+ lines clutter |
| Inconsistent naming | LOW | Throughout | Readability |
| collections.Mapping | LOW | 445, 460 | Deprecated API |

---

## RECOMMENDATIONS

### IMMEDIATE (Critical):

**Option A: If file is NOT USED in TRINITY:**
1. **DELETE** or move to `/archive/` folder
2. Document that read_SB99.py is the active version

**Option B: If file IS USED:**
1. **FIX SYNTAX ERROR** (Lines 99-102)
2. **Port missing imports** from WARPFIELD
3. **Fix array indexing bug** (Line 261: `t2[-2]` → `t2[-1]`)

### High Priority (If keeping file):
4. Replace print() with logging
5. Remove all commented-out code (50+ lines)
6. Remove deprecated functions (sum_SB99_old, time_shift_old)
7. Fix collections.Mapping deprecation

### Medium Priority:
8. Extract duplicated array-trimming logic
9. Simplify nested conditionals
10. Use consistent naming (snake_case)
11. Add type hints
12. Proper pathlib usage

---

## REFACTORED VERSION

**NOT PROVIDED** because:
1. File has critical syntax errors
2. Missing dependencies from WARPFIELD
3. Unclear if file is actually used in TRINITY

**Instead, see:**
- `REFACTORED_read_SB99.py` - Cleaned up, working version
- If advanced features needed (metallicity interpolation, multi-population), those can be added to read_SB99.py

---

## TESTING RECOMMENDATIONS

**CANNOT TEST** until syntax errors and missing imports are fixed.

Once fixed, same test suite as read_SB99.py plus:
1. Test metallicity interpolation (Z = 0.5 between 0.15 and 1.0)
2. Test multiple population summing
3. Test time shifting

---

## SUMMARY

**Overall Severity:** CRITICAL

This file is **COMPLETELY BROKEN** and **CANNOT RUN**:

1. **Syntax error** (Line 99-102) - Python won't parse
2. **Missing imports** - Requires WARPFIELD modules
3. **Array indexing bug** (Line 261)
4. **Dead code** (50+ lines deprecated/commented)

**Status:** Either **UNUSED** (legacy reference) or **NEEDS IMMEDIATE FIX**

**Recommendation:**
- If **NOT USED:** Delete or archive
- If **USED:** Fix syntax, port dependencies, test thoroughly

**DO NOT USE** this file as-is. Use read_SB99.py instead.

---

## CONCLUSION

This file appears to be **copied from WARPFIELD** but **not properly adapted** for TRINITY:
- Still has WARPFIELD imports
- Has syntax error (incomplete function call)
- Contains deprecated code
- Much more complex than needed

**Action:** Determine if file is actually used. If not, remove. If yes, fix immediately or replace with read_SB99.py functionality.

The fact that this file has a **syntax error** and **missing imports** strongly suggests it's **NOT ACTUALLY USED** in TRINITY (or it would have been caught immediately).

**Priority:** INVESTIGATE and either FIX or DELETE.
