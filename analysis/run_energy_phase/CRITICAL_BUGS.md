# Critical Bugs in run_energy_phase.py and energy_phase_ODEs.py

## Bug 1: Manual Euler Integration ⚠️⚠️⚠️ CRITICAL

**Location**: run_energy_phase.py, lines 220-280

**Current Code**:
```python
# METHOD 2 own equations, this solves problem with dictionary
r_arr = []
v_arr = []
Eb_arr = []

for ii, time in enumerate(t_arr):
    y = [R2, v2, Eb, T0]

    try:
        params['t_next'].value = t_arr[ii+1]
    except:
        params['t_next'].value = time + dt_min

    rd, vd, Ed, Td =  energy_phase_ODEs.get_ODE_Edot(y, time, params)

    if ii != (len(t_arr) - 1):
        R2 += rd * dt_min   # <-- Manual Euler!
        v2 += vd * dt_min
        Eb += Ed * dt_min
        T0 += Td * dt_min
```

**Problems**:
1. **Euler method**: Only first-order accurate (error ~ dt)
2. **Unstable**: Blows up for stiff equations
3. **No adaptive timestep**: Fixed dt regardless of solution behavior
4. **Accumulating errors**: Over 100,000 iterations, errors compound
5. **100x slower**: Than using scipy's adaptive solvers

**Evidence**: Lines 199-217 show the CORRECT method was already written but commented out!

**Fix**:
```python
# Uncomment and use the proper ODE solver!
y0 = [R2, v2, Eb, T0]

# Call ODE solver with adaptive timestep
psoln = scipy.integrate.odeint(
    energy_phase_ODEs.get_ODE_Edot,
    y0,
    t_arr,
    args=(params,),
    rtol=1e-6,  # Relative tolerance
    atol=1e-8   # Absolute tolerance
)

# Extract results
r_arr = psoln[:, 0]
v_arr = psoln[:, 1]
Eb_arr = psoln[:, 2]
T0_arr = psoln[:, 3]  # Note: T0 doesn't actually evolve (derivative = 0)

# Update final values
R2 = r_arr[-1]
v2 = v_arr[-1]
Eb = Eb_arr[-1]
T0 = T0_arr[-1]
```

**Impact**:
- Current: ~100,000 iterations needed, each with Euler's poor accuracy
- Fixed: 10-100x faster, 4th-order accurate, stable


## Bug 2: Absurd EarlyPhaseApproximation Hack ⚠️⚠️⚠️ CRITICAL

**Location**: energy_phase_ODEs.py, lines 381-382

**Current Code**:
```python
if params['EarlyPhaseApproximation'].value == True:
    vd = -1e8  # <-- This is INSANE!
```

**And**: run_energy_phase.py, line 273
```python
if ii == 10:
    params['EarlyPhaseApproximation'].value = False
    print('\n\n\n\n\n\n\nswitch to no approximation\n\n\n\n\n\n')
```

**Problems**:
1. **vd = -1e8 pc/Myr²** is acceleration of **-1e8 pc/Myr²**
   - That's about **-3e14 m/s²** or **-3e13 g**
   - The bubble would collapse to zero radius in microseconds!
2. **Arbitrary cutoff**: After exactly 10 iterations, switch off
   - No physical basis
   - What if timestep changes?
3. **Band-aid**: This is covering up early-time numerical instability
4. **Breaks physics**: Acceleration should come from force balance, not hardcoded

**Root Cause**: Early-time instability when R1 is being "switched on" gradually

**Proper Fix**:
```python
# Remove the hack entirely
# Instead, fix the root cause in the R1 switchon logic

# Option 1: Start R1 at correct value immediately (no switchon needed)
if params['current_phase'].value in ['momentum']:
    press_bubble = get_bubbleParams.pRam(R2, LWind, vWind)
else:
    # Just use correct R1 from the start
    press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

# Option 2: If switchon is really needed, do it smoothly over physical time
if params['current_phase'].value not in ['momentum']:
    dt_switchon = 1e-3  # Myr
    if (t - params['tSF'].value) <= dt_switchon:
        # Smooth ramp from 0 to 1
        frac = (t - params['tSF'].value) / dt_switchon
        R1_eff = frac * R1
        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_eff, params['gamma_adia'].value)
    else:
        # After switchon period, use full R1
        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

# Then calculate forces normally - NO hardcoded vd!
```


## Bug 3: Duplicate ODE Functions ⚠️ HIGH

**Location**: energy_phase_ODEs.py, lines 37-227 and 231-401

**Problem**: Two nearly identical functions with subtle differences:

**get_ODE_Edot_new()** (lines 37-227):
- Used: Nowhere (dead code?)
- Line 186: `nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3)`
- Lines 209-212: Different energy equation for early times

**get_ODE_Edot()** (lines 231-401):
- Used: In run_energy_phase.py line 249
- Lines 360-364: More complex nR2 calculation with FABSi check
- Line 387: Simple energy equation
- Lines 381-382: Has EarlyPhaseApproximation hack

**Impact**:
- Fix a bug, must fix in two places
- 170 lines of wasted code
- Confusion about which to use

**Fix**: Merge into one function with flags
```python
def get_ODE_Edot(y, t, params, use_early_energy=False):
    """
    ODE system for bubble expansion.

    Parameters
    ----------
    y : list
        [R2, v2, Eb, T0]
    t : float
        Time [Myr]
    params : DescribedDict
        Parameters
    use_early_energy : bool
        If True, use simplified energy equation for t < 1e-4 Myr
    """
    R2, v2, Eb, T0 = y
    # ... main calculations ...

    # Energy equation
    if use_early_energy and t <= 1e-4:
        Ed = (LWind - L_bubble) - 6/11 * LWind
    else:
        Ed = (LWind - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak

    # HII pressure outside (don't hardcode nR2, calculate properly)
    if FABSi < 1.0:
        # Ionization front hasn't broken out
        nR2 = params['nISM'].value
    else:
        # Stromgren sphere approximation
        nR2 = np.sqrt(Qi / params['caseB_alpha'].value / R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * params['k_B'].value * params['TShell_ion'].value

    # NO hardcoded vd!
    vd = (4*np.pi*R2**2 * (press_bubble - press_HII_in + press_HII_out) -
          mShell_dot*v2 - F_grav + F_rad) / mShell

    return [rd, vd, Ed, 0]
```


## Bug 4: params['nISM'] Missing .value ⚠️ HIGH

**Location**: energy_phase_ODEs.py, line 361

**Current Code**:
```python
if FABSi < 1:
    nR2 = params['nISM']  # <-- BUG: Missing .value
else:
    nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3 * 3 / 4 / np.pi)
```

**Problem**:
- `params['nISM']` returns a DescribedItem object, not a float
- Will cause TypeError when used in arithmetic
- Inconsistent with all other parameter accesses (which use .value)

**Fix**:
```python
if FABSi < 1:
    nR2 = params['nISM'].value  # <-- Add .value
else:
    nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3 * 3 / 4 / np.pi)
```


## Bug 5: Bare except Clause ⚠️ MEDIUM

**Location**: run_energy_phase.py, lines 241-244

**Current Code**:
```python
try:
    params['t_next'].value = t_arr[ii+1]
except:  # <-- Catches EVERYTHING!
    params['t_next'].value = time + dt_min
```

**Problems**:
- Catches ALL exceptions: IndexError, KeyError, KeyboardInterrupt, SystemExit, etc.
- User presses Ctrl+C → silently caught, simulation continues
- Bug in params dict → silently caught, uses fallback
- Makes debugging impossible

**Fix**:
```python
try:
    params['t_next'].value = t_arr[ii+1]
except IndexError:  # Only catch the expected error
    params['t_next'].value = time + dt_min
```


## Bug 6: Redundant Shell Mass Calculation ⚠️ LOW

**Location**: energy_phase_ODEs.py, lines 77-86 and 271-283

**Current Code**: Same block appears twice!

```python
# Lines 77-86
if params['isCollapse'].value == True:
    mShell = params['shell_mass'].value
    mShell_dot = 0
else:
    mShell, mShell_dot = mass_profile.get_mass_profile(...)

# Lines 271-283 - EXACT DUPLICATE!
if params['isCollapse'].value == True:
    mShell = params['shell_mass'].value
    mShell_dot = 0
else:
    mShell, mShell_dot = mass_profile.get_mass_profile(...)
```

**Impact**:
- First block (lines 77-86) is completely pointless
- Results are immediately overwritten by second block
- Confusing duplication

**Fix**: Remove lines 77-86 entirely


## Bug 7: Inefficient Array Concatenation ⚠️ MEDIUM

**Location**: run_energy_phase.py, lines 329-341

**Current Code**:
```python
params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
params['array_R1'].value = np.concatenate([params['array_R1'].value, [params['R1'].value]])
params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [Msh0]])
```

**Problem**:
- np.concatenate() creates a NEW array each time
- O(n) operation in loop → O(n²) total complexity
- For 100,000 iterations, this is extremely slow

**Fix**: Use list append, convert to array at end
```python
# At initialization
array_t_now = []
array_R2 = []
# ... etc

# In loop
array_t_now.append(t_now)
array_R2.append(R2)
# ... etc

# At end
params['array_t_now'].value = np.array(array_t_now)
params['array_R2'].value = np.array(array_R2)
# ... etc
```

**Or pre-allocate**:
```python
# At initialization
max_iterations = 100000
params['array_t_now'].value = np.zeros(max_iterations)
params['array_R2'].value = np.zeros(max_iterations)
idx = 0

# In loop
params['array_t_now'].value[idx] = t_now
params['array_R2'].value[idx] = R2
idx += 1

# At end, trim to actual size
params['array_t_now'].value = params['array_t_now'].value[:idx]
```


## Bug 8: Modifying params Inside ODE ⚠️ HIGH

**Location**: energy_phase_ODEs.py, lines 52-55, 88-89, 133, 218-223, 311, etc.

**Current Code**:
```python
def get_ODE_Edot(y, t, params):
    R2, v2, Eb, T0 = y

    # Modifying global state inside ODE function!
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb

    # ... more modifications ...
    params['shell_mass'].value = mShell
    params['Pb'].value = press_bubble
    params['F_grav'].value = F_grav
    # ... etc
```

**Problems**:
1. **ODE function should be pure**: Same inputs → same outputs
2. **Breaks solver assumptions**: scipy.integrate.odeint() may evaluate function multiple times per step, rewind, etc.
3. **Unpredictable state**: params dict changes during solver iterations
4. **Debugging impossible**: Can't reproduce results
5. **Violates functional programming**: Side effects everywhere

**Fix**: ODE function should ONLY read params, never write
```python
def get_ODE_Edot(y, t, params):
    """
    ODE system - PURE FUNCTION, no side effects.

    Only reads from params, never writes.
    """
    R2, v2, Eb, T0 = y

    # Read parameters (OK)
    FABSi = params["shell_fAbsorbedIon"].value
    mCluster = params["mCluster"].value
    # ... etc

    # Do calculations
    # ...

    # Return derivatives only
    return [rd, vd, Ed, 0]

# Update params AFTER ODE solve, not during
y0 = [R2, v2, Eb, T0]
psoln = scipy.integrate.odeint(get_ODE_Edot, y0, t_arr, args=(params,))

# Now update params with final results
params['R2'].value = psoln[-1, 0]
params['v2'].value = psoln[-1, 1]
params['Eb'].value = psoln[-1, 2]
params['t_now'].value = t_arr[-1]
```
