# Analysis of run_energy_phase.py and energy_phase_ODEs.py

## Purpose

These modules implement the **energy-driven phase** (Phase 1) of TRINITY's bubble evolution simulation. This is the early expansion phase where the bubble is driven by thermal pressure from stellar winds before transitioning to momentum-driven expansion.

### run_energy_phase.py
- Main loop for energy-driven phase
- Integrates ODE system for bubble expansion
- Calculates bubble and shell structure at each timestep
- Manages phase transitions and termination conditions
- 830 lines (400+ lines commented out)

### energy_phase_ODEs.py
- ODE system for bubble dynamics: d[R2, v2, Eb, T0]/dt
- Force balance calculations
- Pressure calculations (bubble, HII regions, gravity)
- 408 lines with duplicate functions

## What This Code Does

### Main Flow (run_energy_phase.py)

1. **Initialization** (Lines 24-106):
   - Extract parameters (t_now, R2, v2, Eb, T0, etc.)
   - Get stellar feedback from Starburst99
   - Calculate initial R1 (inner bubble radius)
   - Calculate initial shell mass and bubble pressure
   - Set loop control variables

2. **Main Loop** (Lines 115-391):
   - Continue while: R2 < rCloud AND t < tfinal AND continueWeaver
   - Update cooling structures every 50k years
   - Calculate bubble/shell structure (after first iteration)
   - Integrate ODE system forward in time
   - Check for fragmentation/instabilities
   - Record snapshots
   - Update for next iteration

3. **ODE Integration** (Lines 220-280):
   - **METHOD 2**: Manual Euler integration (currently active)
   - Create time array with 30 steps of dt_min (1e-6 Myr)
   - Loop through each timestep manually
   - Call get_ODE_Edot() to get derivatives
   - Update [R2, v2, Eb, T0] using: y_new = y_old + dy * dt

4. **Checks and Updates** (Lines 284-376):
   - Switch off EarlyPhaseApproximation after 10 iterations
   - Calculate shell temperature and sound speed
   - Record arrays (t, R2, R1, v2, T0, mShell)
   - Save snapshot
   - Get updated stellar feedback
   - Recalculate R1 and Pb for next loop

### ODE System (energy_phase_ODEs.py)

**get_ODE_Edot(y, t, params)** - Main ODE function:
- **Inputs**: y = [R2, v2, Eb, T0], t = time, params = dictionary
- **Calculates**:
  1. Shell mass and dM/dt
  2. Gravitational force: F_grav = G * mShell * (mCluster + 0.5*mShell) / R2²
  3. Inner radius R1 (wind termination shock)
  4. Bubble pressure: Pb = E2P(Eb, R2, R1)
  5. HII region pressures (inside and outside shell)
  6. Force balance for acceleration
- **Returns**: [dR2/dt, dv2/dt, dEb/dt, dT0/dt]

**Force balance equation**:
```
dv/dt = [4πR2²(P_bubble - P_HII_in + P_HII_out) - mShell_dot*v - F_grav + F_rad] / mShell
```

**Energy equation**:
```
dEb/dt = LWind - L_bubble - 4πR2²*P_bubble*v - L_leak
```

## Critical Flaws Identified

### 1. NUMERICAL INTEGRATION (Critical Priority) ⚠️⚠️⚠️

**Issue 1**: Manual Euler integration instead of proper ODE solver
- Lines 220-280 in run_energy_phase.py
- Uses simple Euler: y_new = y_old + dy*dt
- **Impact**:
  - Euler is first-order accurate (error ~ dt)
  - Unstable for stiff equations
  - Accumulates large errors over many steps
  - No adaptive timestep control
- **Evidence**: Commented-out odeint() call on lines 199-217
- **Fix**: Use scipy.integrate.odeint() properly

**Issue 2**: Hardcoded tiny timestep (dt_min = 1e-6 Myr)
- Line 105: dt_min = 1e-6 Myr (31.5 seconds!)
- Line 152: 30 steps per loop
- Total advance per loop: 30 * 1e-6 = 3e-5 Myr (~1000 seconds)
- **Impact**:
  - Will take ~100,000 loops to reach tfinal = 3e-3 Myr
  - Extremely slow simulation
  - Unnecessary computational cost
- **Fix**: Use adaptive timestep with odeint()

**Issue 3**: EarlyPhaseApproximation hack
- Line 273: After 10 iterations, switch to "no approximation"
- Line 382 in energy_phase_ODEs.py: vd = -1e8 (!)
- **Impact**:
  - Arbitrary cutoff with no physical basis
  - vd = -1e8 is absurdly large negative acceleration
  - This is a band-aid for numerical instability
- **Fix**: Solve root cause of early-time instability

### 2. CODE DUPLICATION (High Priority)

**Issue**: Two nearly identical ODE functions
- `get_ODE_Edot_new()` (lines 37-227)
- `get_ODE_Edot()` (lines 231-401)
- Only difference:
  - Lines 209-212 vs 387: Early-time energy equation
  - Lines 363 vs 360-364: HII pressure calculation
  - get_ODE_Edot() has EarlyPhaseApproximation hack
- **Impact**:
  - Maintenance nightmare (fix bug in both places)
  - Confusing which one is used
  - 170 lines of duplicate code
- **Fix**: Merge into one function with flags

### 3. COMMENTED-OUT CODE (High Priority)

**run_energy_phase.py**: 400+ lines of dead code
- Lines 72-76: Commented density profile check
- Lines 86, 101: Commented tfinal alternatives
- Lines 199-217: Commented odeint() method (should be used!)
- Lines 276-278: Commented sys.exit()
- Lines 353-360: Commented momentum phase transition
- Lines 384-389: Multiple commented sys.exit() checks
- Lines 393-830: Massive block of old code (400+ lines!)

**Impact**:
- File bloat (830 lines → ~400 actual code)
- Confusing to maintainers
- Hard to find active code
- Version control should handle history, not comments

**Fix**: Remove all dead code immediately

### 4. DEBUG PRINTS (Medium Priority)

Excessive print statements pollute logs:

**run_energy_phase.py**:
- Line 77-80: Prints R1, Msh0, Pb
- Line 154: Prints entire t_arr
- Line 171, 175, 178: Phase status prints
- Line 228-231: Prints R2, v2, Eb, T0
- Line 257-265: Prints all derivatives and states
- Line 274: Prints approximation switch
- Line 346: Prints snapshot save

**energy_phase_ODEs.py**:
- Line 152, 329: Prints n_r
- Line 183, 358: Prints bubble arrays
- Line 206, 384: Prints detailed force breakdown

**Impact**:
- Log files become gigabytes
- Hard to find important messages
- Performance hit from I/O
- Debugging should use logging module with levels

**Fix**: Replace with proper logging

### 5. MAGIC NUMBERS (Medium Priority)

Hardcoded constants without explanation:

**run_energy_phase.py**:
- Line 102: tfinal = 3e-3 Myr (why exactly 3000 years?)
- Line 105: dt_min = 1e-6 Myr (why this timestep?)
- Line 116: Exit condition: (tfinal - t_now) > 1e-4 (why 100 years?)
- Line 124: Recalculate cooling every 5e-2 Myr (50k years - why?)
- Line 151: tsteps = 30 (why 30?)
- Line 272: Switch approximation after iteration 10 (why 10?)

**energy_phase_ODEs.py**:
- Line 123, 296: dt_switchon = 1e-3 Myr (1000 years - why?)
- Line 186, 364: T = 3e4 K for HII region (why this temperature?)
- Line 209: t <= 1e-4 for energy equation switch (why 100 years?)
- Line 382: vd = -1e8 (absolutely absurd value!)

**Fix**: Define as named constants with physical justification

### 6. ERROR HANDLING (High Priority)

**Issue 1**: Bare except clauses
```python
# Lines 241-244 in run_energy_phase.py
try:
    params['t_next'].value = t_arr[ii+1]
except:  # <-- Catches ALL exceptions!
    params['t_next'].value = time + dt_min
```
**Impact**: Silently catches bugs, index errors, keyboard interrupts
**Fix**: `except IndexError:`

**Issue 2**: No validation of ODE results
- No check if R2, v2, Eb become negative or NaN
- No check if derivatives are reasonable
- Could produce unphysical results without warning

**Issue 3**: No convergence checking
- scipy.optimize.brentq() could fail to converge
- No error handling around line 58, 361

### 7. INCONSISTENT STATE MANAGEMENT (Medium Priority)

**Issue**: Params dict modified inside ODE function
- Lines 52-55, 246-249 in energy_phase_ODEs.py update params
- Lines 88-89, 133, 218-223, 311 update more params
- **Impact**:
  - ODE function should be pure (same inputs → same outputs)
  - Modifying global state breaks ODE solver assumptions
  - Makes debugging impossible (state changes unpredictably)
  - Violates functional programming principles
- **Fix**: ODE function should only read params, not write

### 8. POOR LOOP STRUCTURE (Medium Priority)

**Issue 1**: Nested logic for calculate_bubble_shell
- Line 140: `calculate_bubble_shell = loop_count > 0`
- Line 160-183: if/else branches based on this flag
- First iteration skips critical calculations
- **Impact**: Initial conditions poorly defined
- **Fix**: Always calculate, or use proper initialization

**Issue 2**: Fixed 30 timesteps per loop
- Line 151: `tsteps = 30`
- Line 152: Creates array of 30 points
- Total time per loop: 30 * 1e-6 = 3e-5 Myr
- **Impact**: Arbitrary, inflexible
- **Fix**: Adaptive timestep based on solution behavior

### 9. INCOMPLETE FEATURES (Medium Priority)

**TODOs scattered throughout**:
- Line 26: "TODO: add CLOUDY"
- Line 175-176, 179-181: Multiple TODO comments
- Line 190: "TODO: Future------- add cover fraction"
- Line 296: "TODO" with no description
- Lines 351, 368: More TODOs in energy_phase_ODEs.py

**Commented-out fragmentation code** (Lines 404-830):
- check_events() function (lines 404-467)
- Fragmentation calculations (lines 289-299, 600-713)
- Rayleigh-Taylor instability (lines 297-299)
- Covering fraction (lines 812-827)

**Impact**: Half-implemented features confuse users

### 10. DOCUMENTATION (Low Priority)

**Issues**:
- Incomplete docstrings (missing Parameters, Returns, units)
- Many "old code" references
- Physics not explained (what is "Weaver phase"?)
- No examples or test cases
- Units often unclear (au? cgs? SI?)

## Specific Bugs

### Bug 1: Inconsistent pressure calculation (Lines 360-364, energy_phase_ODEs.py)

```python
# Current code:
if FABSi < 1:
    nR2 = params['nISM']  # <-- Bug: Should this be .value?
else:
    nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3 * 3 / 4 / np.pi)
press_HII_out = 2 * nR2 * params['k_B'].value * 3e4
```

**Problems**:
1. `params['nISM']` missing `.value` access
2. Inconsistent with line 185-186 in get_ODE_Edot_new()
3. Magic number 3e4 K

### Bug 2: Array concatenation inefficiency (Lines 329-341, run_energy_phase.py)

```python
params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
# ... 5 more times
```

**Impact**: O(n) concatenation in loop = O(n²) total complexity
**Fix**: Use lists and convert to array at end, or pre-allocate

### Bug 3: Shell mass during collapse (Lines 77-86, energy_phase_ODEs.py)

```python
if params['isCollapse'].value == True:
    mShell = params['shell_mass'].value
    mShell_dot = 0
else:
    mShell, mShell_dot = mass_profile.get_mass_profile(...)
```

**Then immediately duplicated** on lines 271-283!

**Impact**: Redundant code, first block (lines 77-86) is never used

### Bug 4: Undefined R1_tmp scope (Line 307, energy_phase_ODEs.py)

```python
elif (t <= (tmin + params['tSF'].value)):
    R1_tmp = (t-params['tSF'].value)/tmin * R1
    press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
```

If `t > tmin + tSF`, then R1_tmp is never defined, but might be referenced elsewhere (not in this file, but risky pattern).

## Performance Issues

1. **Manual Euler integration**: 100x slower than adaptive solvers
2. **Tiny timestep**: dt = 1e-6 Myr is probably 10-100x smaller than needed
3. **Recalculating bubble/shell every loop**: Could interpolate for small dt
4. **Array concatenation**: O(n²) instead of O(n)
5. **Excessive logging**: print() in tight loops
6. **Redundant calculations**: Same code executed in both ODE functions

**Estimated speedup from fixes**: 10-100x

## Recommendations

### IMMEDIATE (Critical Bugs)

1. ✅ **Switch to scipy.integrate.odeint() properly**
   - Uncomment lines 199-217 in run_energy_phase.py
   - Remove manual Euler integration (lines 220-280)
   - Benefits: Adaptive timestep, 4th-order accuracy, stability

2. ✅ **Remove duplicate get_ODE_Edot function**
   - Keep one version, add flags for behavior differences
   - Saves 170 lines, eliminates confusion

3. ✅ **Fix params['nISM'] missing .value**
   - Line 361 in energy_phase_ODEs.py

4. ✅ **Remove EarlyPhaseApproximation hack**
   - vd = -1e8 is absurd
   - Find root cause of instability

5. ✅ **Fix bare except clause**
   - Line 243: Use `except IndexError:`

### SHORT-TERM (Code Quality)

1. ✅ **Remove ALL commented code** (400+ lines)
   - Especially lines 404-830 in run_energy_phase.py
   - Keep history in git, not in comments

2. ✅ **Replace print() with logging**
   - Use logging.DEBUG for verbose info
   - Use logging.INFO for progress
   - User can control verbosity

3. ✅ **Define magic numbers as constants**
   - Create PhaseConstants class or module-level constants
   - Document physical justification

4. ✅ **Make ODE function pure**
   - Don't modify params inside get_ODE_Edot()
   - Return auxiliary info separately if needed

5. ✅ **Pre-allocate arrays**
   - Instead of concatenating in loop (lines 329-341)
   - Or use lists and convert at end

### LONG-TERM (Refactoring)

1. ⬜ **Adaptive timestepping**
   - Let odeint() choose dt automatically
   - Set tolerances (rtol, atol) instead of fixed dt

2. ⬜ **Separate concerns**:
   - run_energy_phase.py: High-level loop and control
   - energy_phase_ODEs.py: Just ODE system
   - energy_phase_forces.py: Force calculations
   - energy_phase_checks.py: Termination conditions

3. ⬜ **Implement termination conditions properly**
   - Uncomment and fix check_events() function
   - Use ODE event detection (scipy.integrate.solve_ivp)

4. ⬜ **Add comprehensive tests**
   - Unit tests for ODE functions
   - Integration tests for full phase
   - Validate against analytical solutions where possible

5. ⬜ **Document physics**
   - Explain Weaver+77 theory in docstring
   - Add references to equations
   - Describe each force term

## Testing Strategy

1. **ODE Function Tests**:
   - Test get_ODE_Edot() with known inputs
   - Check force balance is correct
   - Verify energy conservation (within cooling)
   - Test edge cases (collapse, FABSi = 0, 1)

2. **Integration Tests**:
   - Run full energy phase for test cases
   - Compare Euler vs odeint() results
   - Check conservation laws
   - Validate against Weaver+77 analytical solutions

3. **Performance Tests**:
   - Benchmark manual Euler vs odeint()
   - Profile to find bottlenecks
   - Test different tolerances

4. **Regression Tests**:
   - Save known-good results
   - Ensure refactoring doesn't change physics

## Summary

### What Works
- ✓ Physics is correct (based on Weaver+77 and force balance)
- ✓ Basic structure is sound
- ✓ Handles pressure balance correctly

### Critical Issues
- ✗ Manual Euler integration instead of proper ODE solver
- ✗ Absurdly small timestep (dt = 1e-6 Myr)
- ✗ 400+ lines of dead code
- ✗ Duplicate ODE function (170 lines)
- ✗ Params modified inside ODE (breaks solver assumptions)
- ✗ EarlyPhaseApproximation hack (vd = -1e8!)

### Bottom Line

This code **works but is extremely inefficient and fragile**:
- Runs 10-100x slower than necessary
- Manual Euler integration is numerically unstable
- Tiny timestep requires ~100,000 iterations
- Half the file is dead code
- No error handling or validation

**Priority**:
1. Switch to odeint() (10x speedup)
2. Remove dead code (improves maintainability)
3. Fix EarlyPhaseApproximation (improves stability)

**Estimated Effort**:
- Critical fixes: 4-6 hours
- Code cleanup: 3-4 hours
- Full refactoring: 2-3 days
