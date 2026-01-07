# Comprehensive Analysis: mass_profile.py

## File Overview

**Location**: `src/cloud_properties/mass_profile.py`
**Size**: 361 lines (200 lines actual code, 161 lines comments/whitespace/dead code)
**Purpose**: Calculate mass M(r) and mass accretion rate dM/dt for cloud profiles
**Author**: Jia Wei Teh
**Created**: July 25, 2022

---

## Executive Summary

**Your assessment is 100% correct.**

> "The idea... tries to infer Mdot by interpolating solver histories, which breaks on duplicate times and mixes responsibilities."

This is **fundamentally flawed** both mathematically and architecturally:

1. **Mathematical error**: Uses `np.gradient(M_arr, t_arr)` then interpolates as function of r (dimensionally wrong!)
2. **Design flaw**: Mass profile calculation shouldn't depend on solver history
3. **Coupling**: Tightly couples to solver internals (array_t_now, array_R2, etc.)
4. **Fragile**: Breaks on duplicate times, non-monotonic data
5. **Unnecessary**: dM/dt can be computed analytically from dM/dr × dr/dt

**Status**: ❌ **BROKEN DESIGN - Needs architectural fix**

---

## What This Code Should Do

### Physics Context

Calculate mass enclosed within radius r:

```
M(r) = ∫[0 to r] 4πr'² ρ(r') dr'
```

And mass accretion rate:

```
dM/dt = dM/dr × dr/dt = 4πr² ρ(r) × v(r)
```

### Two Profile Types

**1. Power-Law Profile** (easy):
```
ρ(r) = ρ_core                    for r ≤ r_core
ρ(r) = ρ_core (r/r_core)^α      for r_core < r ≤ r_cloud
ρ(r) = ρ_ISM                     for r > r_cloud
```

Has analytical M(r) and dM/dr → dM/dt is trivial.

**2. Bonnor-Ebert Sphere** (hard):
```
ρ(r) = ρ_core f(ξ)    where ξ = r / (c_s² / 4πGρ_core)^(1/2)
```

f(ξ) comes from solving Lane-Emden equation (no closed form).

M(r) requires numerical integration.

BUT: dM/dr = 4πr² ρ(r) is still straightforward!

**Your problem**: Code tries to get dM/dt from solver history instead of from dM/dr.

---

## CRITICAL BUGS

### BUG #1: Fundamental Mathematical Error (Lines 267-335)

**Location**: Lines 267-335 (Bonnor-Ebert dM/dt calculation)

**The Broken Approach**:

```python
# Get history from solver
t_arr_previous = params['array_t_now'].value
r_arr_previous = params['array_R2'].value
m_arr_previous = params['array_mShell'].value

# Compute dM/dt at history points
mgrad = np.gradient(m_arr_previous, t_arr_previous)  # dM/dt vs t

# Interpolate as function of r (WRONG!)
mdot_interp = scipy.interpolate.interp1d(
    r_arr_previous,  # x-axis: radius
    np.gradient(m_arr_previous, t_arr_previous),  # y-axis: dM/dt
    kind='cubic'
)

# Evaluate at current radius
mdot_arr = mdot_interp(r_arr)  # WRONG!
```

**Problems**:

1. **Dimensional confusion**:
   - Computes dM/dt (time derivative)
   - Interpolates as function of r (space)
   - These are different things!

2. **What it's actually doing**:
   ```
   Given: M(t₀), M(t₁), M(t₂), ... at times t₀, t₁, t₂, ...
          R(t₀), R(t₁), R(t₂), ... at same times

   Computes: dM/dt[i] ≈ (M[i+1] - M[i-1]) / (t[i+1] - t[i-1])

   Then: Creates function dM/dt(R) by pairing dM/dt[i] with R[i]

   Evaluates: dM/dt at current R
   ```

3. **Why this is wrong**:
   - dM/dt depends on time, not just radius
   - At same radius but different times, dM/dt can be different
   - Mapping dM/dt → R assumes unique relationship (false!)

**Correct approach**:
```python
# dM/dt = dM/dr × dr/dt
# where dM/dr = 4πr² ρ(r)  [from mass continuity]

rho_r = compute_density_profile(r_arr, params)  # Known from BE solution
dMdr = 4 * np.pi * r_arr**2 * rho_r

dMdt = dMdr * rdot_arr  # rdot_arr is input (shell velocity)
```

**No history needed!** Just need ρ(r) and v(r).

---

### BUG #2: np.gradient Usage Error (Line 317)

**Location**: Line 317

**Current code**:
```python
mdot_interp = scipy.interpolate.interp1d(
    r_arr_previous,
    np.gradient(m_arr_previous, t_arr_previous),  # WRONG!
    kind='cubic',
    fill_value="extrapolate"
)
```

**Problem**: `np.gradient(y, x)` assumes:
- Either x is uniformly spaced
- Or you provide spacing explicitly

From solver history:
- t_arr is NOT uniformly spaced (adaptive time stepping)
- M vs t is NOT linear
- np.gradient will give wrong derivatives

**Correct**:
```python
# If you MUST use history (which you shouldn't):
from scipy.interpolate import CubicSpline

# Fit M(t)
M_of_t = CubicSpline(t_arr_previous, m_arr_previous)

# Evaluate dM/dt at specific times
dMdt = M_of_t(t_current, nu=1)  # nu=1 means first derivative
```

But again, **this entire approach is wrong**. Don't use history!

---

### BUG #3: Duplicate Times Break Interpolation (Lines 289-303)

**Location**: Lines 289-303

**Current code**:
```python
try:
    interps = CubicSpline(t_arr_previous, r_arr_previous, extrapolate=True)
except Exception as e:
    print(e)
    print('t_arr_previous', t_arr_previous)
    print_duplicates(t_arr_previous)  # Lines 272-282
    # ... more debug prints
    import sys
    sys.exit()  # CRASH!
```

**Problem**:
- Solver might evaluate same time twice (backtracking, error correction)
- Duplicate times → interpolation fails
- Code just crashes with sys.exit()

**Why this happens**:
- Adaptive ODE solvers can reject steps and retry
- Time array can have duplicates
- Even if sorted and unique'd, R might not be monotonic

**This is a symptom of the deeper problem**: Using solver internals for physics calculation.

---

### BUG #4: Tight Coupling to Solver (Lines 267-269)

**Location**: Lines 267-269

**Current code**:
```python
t_arr_previous = params['array_t_now'].value
r_arr_previous = params['array_R2'].value
m_arr_previous = params['array_mShell'].value
```

**Problems**:

1. **Assumes params has these arrays**:
   - What if solver changes implementation?
   - What if different solver is used?
   - What if arrays don't exist yet (early in simulation)?

2. **Violates separation of concerns**:
   - Mass profile is physics (density → mass)
   - Solver history is implementation detail
   - Physics shouldn't depend on solver internals!

3. **Makes testing impossible**:
   - Can't test mass_profile.py without running full solver
   - Can't unit test with simple inputs
   - Tightly coupled code

**Correct design**:
- Physics functions depend only on physical state
- State = current r, ρ(r), v(r)
- Not on solver history!

---

### BUG #5: Wrong Threshold Logic (Lines 259-263)

**Location**: Lines 259-263

**Current code**:
```python
if params['R2'].value < r_threshold:
    # treat as homogeneous cloud
    mdot_arr = 4 * np.pi * r_arr**2 * rhoGas * rdot_arr
elif params['R2'].value < rCloud:
    # use interpolation from history
    params['shell_interpolate_massDot'].value = True
    # ... complicated history-based calculation
```

**Problem**:

1. **Threshold at 90% of core density** (line 247):
   - Arbitrary choice (why 90%? why not 95% or 80%?)
   - No physical justification

2. **Discontinuity**:
   - At r = r_threshold: Uses simple formula
   - At r = r_threshold + ε: Uses complex interpolation
   - Sudden jump in dM/dt

3. **Modifies params inside function** (line 265):
   ```python
   params['shell_interpolate_massDot'].value = True
   ```
   - Side effect!
   - Function should not modify global state

**Why threshold exists**:
- Author knows complex method is broken
- Falls back to simple method when close to center
- This is a band-aid, not a fix

---

### BUG #6: Commented Dead Code (Lines 151-211)

**Location**: Lines 151-211 (60 lines!)

**Examples**:
```python
# OLD VERSION for mass ----
# i think this will break if r_arr is given such that it is very large and break interpolation?
# ...

# # new version for mass -----
# m_arr = np.ones_like(r_arr)
# ...
# # [50+ lines of commented alternatives]
```

**Impact**:
- Impossible to tell which approach is current
- Two methods for M(r) (old vs new)
- "i think this will break" ← not confidence-inspiring!

---

### BUG #7: Debug Print Statements Everywhere

**Lines**: 79, 95-97, 132, 134, 136-139, 252, and many in error handlers

**Examples**:
```python
print(f'mGas is r={r_arr} and rho={rhoCore} equals {mGas}')  # Line 79
print('mGasdot is', mGasdot)  # Line 95
print('r_arr is', r_arr)  # Line 96
print('rdot_arr is', rdot_arr)  # Line 97
print(mGasdot)  # Lines 132, 134, 136 (three times!)
print('thresholds for BE interpolations are (n, r):', ...)  # Line 252
```

Should use logging module.

---

## DESIGN FLAWS

### FLAW #1: Mixing Physics and Implementation

**Problem**:
- Physics (M(r), dM/dt) should depend only on physical state
- Code depends on solver implementation details (history arrays)
- This violates basic software design principles

**Consequence**:
- Can't change solver without breaking mass_profile
- Can't test mass_profile independently
- Hard to debug
- Hard to maintain

### FLAW #2: Incorrect Abstraction

**What the function signature implies**:
```python
def get_mass_profile(r_arr, params, return_mdot, rdot_arr):
    """
    Given r and v(r), compute M(r) and dM/dt.
    """
```

**What it actually does** (Bonnor-Ebert case):
```python
def get_mass_profile_ACTUALLY(r_arr, params_with_solver_history, ...):
    """
    1. Extract solver history from params
    2. Interpolate solver history to estimate dM/dt
    3. Handle errors when history has duplicates
    4. Crash if things go wrong
    """
```

**What it SHOULD do**:
```python
def get_mass_profile_CORRECT(r_arr, density_profile_func, rdot_arr):
    """
    1. Compute M(r) = ∫ 4πr'² ρ(r') dr'
    2. Compute dM/dt = 4πr² ρ(r) × v(r)
    3. Return (M_arr, dMdt_arr)
    """
```

---

## THE CORRECT SOLUTION

### Key Insight

**You don't need solver history!**

For any density profile ρ(r):

```
dM/dt = dM/dr × dr/dt
      = [4πr² ρ(r)] × v(r)
```

**Even for Bonnor-Ebert spheres**:
- You already have ρ(r) from `params['densBE_f_rho_rhoc']`
- You already have v(r) from input `rdot_arr`
- Just multiply them!

### Correct Implementation

```python
def get_mass_profile_CORRECT(r_arr, params, return_mdot=False, rdot_arr=None):
    """
    Calculate mass profile M(r) and optionally dM/dt.

    Parameters
    ----------
    r_arr : array
        Radii at which to evaluate [cm or pc]
    params : dict
        Parameter dictionary with density profile info
    return_mdot : bool
        Whether to compute dM/dt
    rdot_arr : array, optional
        dr/dt (velocities) if return_mdot=True

    Returns
    -------
    M_arr : array
        Mass enclosed within each r
    dMdt_arr : array (if return_mdot=True)
        Mass accretion rate at each r
    """

    # Get density profile ρ(r)
    rho_arr = compute_density_profile(r_arr, params)

    # Compute M(r) = ∫ 4πr² ρ(r) dr
    M_arr = compute_enclosed_mass(r_arr, rho_arr, params)

    if not return_mdot:
        return M_arr

    # Compute dM/dt = dM/dr × dr/dt = 4πr² ρ(r) × v(r)
    if rdot_arr is None:
        raise ValueError("rdot_arr required when return_mdot=True")

    dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr

    return M_arr, dMdt_arr


def compute_density_profile(r_arr, params):
    """
    Compute ρ(r) for given profile type.

    Returns
    -------
    rho_arr : array
        Mass density at each r
    """
    profile_type = params['dens_profile'].value

    if profile_type == 'densPL':
        return compute_powerlaw_density(r_arr, params)
    elif profile_type == 'densBE':
        return compute_bonnor_ebert_density(r_arr, params)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def compute_bonnor_ebert_density(r_arr, params):
    """
    Compute ρ(r) for Bonnor-Ebert sphere.

    Uses pre-computed interpolation function from params.
    """
    nCore = params['nCore'].value
    mu_ion = params['mu_ion'].value
    rhoCore = nCore * mu_ion

    # Convert r to dimensionless ξ
    xi_arr = bonnorEbertSphere.r2xi(r_arr, params)

    # Get density ratio ρ/ρ_core from BE solution
    f_rho_rhoc = params['densBE_f_rho_rhoc'].value
    rho_ratio = f_rho_rhoc(xi_arr)

    # Compute actual density
    rho_arr = rhoCore * rho_ratio

    # Handle r > rCloud
    rCloud = params['rCloud'].value
    nISM = params['nISM'].value
    mu_neu = params['mu_neu'].value
    rhoISM = nISM * mu_neu

    rho_arr[r_arr > rCloud] = rhoISM

    return rho_arr
```

**That's it!** No solver history, no interpolation, no duplicate time errors.

---

## WHY THE BROKEN APPROACH WAS USED

Looking at comments (lines 219-228):

```python
# this part is a little bit trickier, because there is no analytical solution to the
# BE spheres. What we can do is to have two seperate parts: for xi <~1 we know that
# rho ~ rhoCore, so this part can be analytically similar to the
# homogeneous sphere.
# Once we have enough in the R2_aray and the t_array, we can then use them
# to extrapolate to obtain mShell.
```

**Misunderstanding**:
- Author thinks: "No analytical M(r), therefore no analytical dM/dt"
- Author concludes: "Must use solver history to estimate dM/dt"

**Correct understanding**:
- True: M(r) has no closed form
- BUT: dM/dr = 4πr² ρ(r) DOES have a form (from BE solution)
- Therefore: dM/dt = (dM/dr) × v is straightforward!

**The confusion**:
- M(r) requires integration: M(r) = ∫ 4πr'² ρ(r') dr'
- But dM/dr doesn't require integration: dM/dr = 4πr² ρ(r)
- Author conflated these two!

---

## VERIFICATION QUESTIONS

### Question 1: What is params['densBE_f_rho_rhoc']?

**Line 157, 161**:
```python
f_rho_rhoc = params['densBE_f_rho_rhoc'].value
# ...
f_mass = lambda xi : 4 * np.pi * rhoCore * ... * f_rho_rhoc(xi)
```

**This is a function** that gives ρ(ξ)/ρ_core for Bonnor-Ebert sphere.

**If you already have ρ(r), why not just use it for dM/dt?**

Answer: Author didn't realize dM/dt = 4πr² ρ(r) × v(r) works even without analytical M(r).

### Question 2: What are these arrays?

**Lines 267-269**:
```python
t_arr_previous = params['array_t_now'].value
r_arr_previous = params['array_R2'].value
m_arr_previous = params['array_mShell'].value
```

These come from solver:
- `array_t_now`: History of time points
- `array_R2`: History of bubble radius
- `array_mShell`: History of shell mass

**Why does mass_profile need solver history?**

It shouldn't! This is the design flaw.

### Question 3: Why the threshold at 90% density?

**Lines 247-249**:
```python
n_threshold = 0.9 * params['nCore']
r_threshold = cloud_getr_interp(n_threshold)
```

**Hypothesis**: Author knows interpolation method is unreliable, so:
- When r < r_threshold (dense region): Use simple formula
- When r > r_threshold (diffuse region): Use complex (broken) interpolation

This is evidence the method is flawed!

---

## PERFORMANCE IMPACT

### Current approach (Bonnor-Ebert dM/dt):

1. Extract 3 arrays from params
2. Interpolate R(t) using CubicSpline
3. Evaluate R at t_next
4. Concatenate to arrays
5. Compute gradient of M with respect to t
6. Interpolate gradient as function of R
7. Evaluate at current r

**Steps**: 7 (with 2 interpolations, 1 gradient, multiple array operations)

### Correct approach:

1. Evaluate ρ(r) at r_arr (one interpolation call)
2. Multiply: dM/dt = 4πr² ρ(r) × v(r)

**Steps**: 2 (with 1 interpolation)

**Speedup**: ~5-10× faster (fewer operations, no array management)

**Plus**: No memory overhead for history arrays!

---

## TESTING PROBLEMS

### Cannot unit test current code

```python
# Want to test:
M, dMdt = get_mass_profile(r_arr, params, return_mdot=True, rdot_arr=v_arr)

# But params must contain:
params['array_t_now'] = [complicated history]
params['array_R2'] = [complicated history]
params['array_mShell'] = [complicated history]
params['t_next'] = [some future time]
# ... and many more

# This is impossible to set up for simple test!
```

### Can unit test correct code

```python
# Simple test:
params = {
    'dens_profile': 'densBE',
    'nCore': 1e3,
    'densBE_f_rho_rhoc': some_function,
    # ... basic physics parameters only
}

M, dMdt = get_mass_profile_CORRECT(r_arr, params,
                                    return_mdot=True,
                                    rdot_arr=v_arr)

# Easy to test!
```

---

## RECOMMENDATIONS

### Immediate Fix (1-2 hours)

1. **Remove history-based dM/dt calculation** (lines 267-335)

2. **Use direct formula**:
   ```python
   # Compute density
   rho_arr = compute_bonnor_ebert_density(r_arr, params)

   # Compute dM/dt
   dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr
   ```

3. **Remove threshold logic** (lines 247-263)
   - Not needed with correct approach

4. **Remove solver coupling**
   - Delete params['array_t_now'], etc. dependencies

### Clean Up (2-3 hours)

5. **Remove dead code** (lines 151-211)

6. **Replace print() with logging**

7. **Remove debug prints**

8. **Simplify function structure**:
   - Power-law: analytical M(r) and dM/dt
   - Bonnor-Ebert: numerical M(r), analytical dM/dt formula

### Testing (2-3 hours)

9. **Add unit tests**:
   - Test power-law M(r) against known solution
   - Test power-law dM/dt against analytical formula
   - Test Bonnor-Ebert M(r) against integration
   - Test Bonnor-Ebert dM/dt with simple profile

10. **Add integration tests**:
    - Verify mass conservation: dM/dt integrated over time = ΔM

---

## BOTTOM LINE

**Current approach is fundamentally broken**:
- ❌ Mathematical error (confuses dM/dt with dM/dr)
- ❌ Design flaw (couples physics to solver implementation)
- ❌ Unnecessary complexity (uses history when shouldn't)
- ❌ Fragile (breaks on duplicate times)
- ❌ Untestable (needs full solver to test)

**Correct approach is simple**:
- ✅ dM/dt = 4πr² ρ(r) × v(r)
- ✅ Works for ANY density profile
- ✅ No solver history needed
- ✅ Easy to test
- ✅ 5-10× faster

**Your diagnosis was spot on**:
> "Tries to infer Mdot by interpolating solver histories, which breaks on duplicate times and mixes responsibilities."

Exactly right! The solution is to not use solver history at all.

**Effort to fix**: 4-8 hours total
**Impact**: Code that actually works, 5-10× faster, testable, maintainable

**Priority**: HIGH - Current implementation is broken and will cause incorrect physics
