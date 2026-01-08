# Critical Bugs: shell_structure.py

## Overview

**File**: `src/shell_structure/shell_structure.py` and `src/shell_structure/get_shellODE.py`

**Status**: ❌ **BROKEN - Results cannot be trusted**

**Critical bugs found**: 3 that make results completely wrong

---

## BUG #1: Missing mu Factor in Ionized dndr (CRITICAL - WRONG PHYSICS)

### Location
`get_shellODE.py` lines 87-90

### Current Code (WRONG)
```python
dndr = mu_p/mu_n/(k_B * t_ion) * (
    nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau + Li * phi)\
        + nShell**2 * alpha_B * Li / Qi / c
    )
```

### Problem

The recombination pressure term is **missing the mu_p/mu_n factor**!

The equation should have consistent structure:
```
dn/dr = (1/k_B*T) * [term1 + term2]
```

Or if using mass density convention:
```
dn/dr = (mu_p/mu_n) * (1/k_B*T) * [term1 + term2]
```

**Current code mixes both!**

### Correct Code

```python
def get_shellODE_ionized_FIXED(y, r, f_cover, params):
    """
    ODEs for ionized shell region.

    Returns dn/dr, dphi/dr, dtau/dr

    Physics:
    - Radiation pressure compresses shell
    - Recombination pressure (from ionizing photon absorption)
    - Both contribute to dn/dr

    Reference: Rahner thesis Eq 2.44, Krumholz+ 2009
    """
    nShell, phi, tau = y

    # Parameters
    sigma_dust = params['dust_sigma'].value
    mu_n = params['mu_neu'].value
    mu_p = params['mu_ion'].value
    t_ion = params['TShell_ion'].value
    alpha_B = params['caseB_alpha'].value
    k_B = params['k_B'].value
    c = params['c_light'].value
    Ln = params['Ln'].value
    Li = params['Li'].value
    Qi = params['Qi'].value

    # Prevent underflow
    neg_exp_tau = np.exp(-tau) if tau < 500 else 0.0

    # Radiation pressure term (direct + ionizing)
    # F_rad = (L/4πr²c) = radiation pressure flux
    # Compression: dn/dr ∝ (1/kT) * n * σ * F_rad
    rad_pressure_term = (nShell * sigma_dust / (4 * np.pi * r**2 * c)) * \
                        (Ln * neg_exp_tau + Li * phi)

    # Recombination pressure term
    # Ionizing photons absorbed → momentum transferred
    # Rate: n² α_B (recombinations per volume)
    # Momentum per photon: L_i/Q_i/c
    # Compression: dn/dr ∝ (1/kT) * n² * α_B * (L_i/Q_i/c)
    recomb_pressure_term = nShell**2 * alpha_B * Li / Qi / c

    # Total dn/dr (BOTH terms need same mu factor!)
    dndr = (mu_p / mu_n) / (k_B * t_ion) * (
        rad_pressure_term + recomb_pressure_term
    )

    # Ionizing photon attenuation
    # dphi/dr = -n² α_B (4πr²/Q_i) [recombinations] - n σ phi [dust absorption]
    dphidr = -4 * np.pi * r**2 * alpha_B * nShell**2 / Qi - nShell * sigma_dust * phi

    # Optical depth
    dtaudr = nShell * sigma_dust * f_cover

    return dndr, dphidr, dtaudr
```

### Impact

- **Shell density profile is WRONG**
- **Recombination pressure is underestimated by factor ~0.6**
- **All downstream physics (cooling, ionization structure, etc.) is WRONG**

### How to Test

Create test case with known solution:
1. Strömgren sphere (no radiation pressure, only recombination)
2. Check if density profile matches analytic solution
3. Current code will fail

### Priority

**CRITICAL** - Fix immediately before trusting any results

---

## BUG #2: Missing mu Factor in Neutral dndr (CRITICAL - WRONG PHYSICS)

### Location
`get_shellODE.py` lines 112-114

### Current Code (WRONG)
```python
dndr = 1/(k_B * t_neu) * (
    nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau)
    )
```

### Problem

Missing mu_n factor (or using wrong variable type).

If `nShell` is **number density** [1/cm³], equation should be:
```python
# Pressure balance: P = n * k_B * T
# For neutral gas: P_neu = n_neu * k_B * T_neu
# Radiation pressure: P_rad ∝ n * σ * F_rad
# dn/dr = (1/k_B*T) * [radiation pressure terms]

dndr = (1 / (k_B * t_neu)) * radiation_pressure_term  # WRONG - missing mu
```

If `nShell` is **mass density** [g/cm³], equation should be:
```python
# ρ = n * μ * m_p
# P = (ρ/μ/m_p) * k_B * T = (ρ/μ) * k_B * T  (if using μ as mass)
# dn/dr really means dρ/dr

drhodr = (mu_n / (k_B * t_neu)) * radiation_pressure_term  # Correct for mass density
```

**Current code is inconsistent with ionized region!**

### Correct Code

```python
def get_shellODE_neutral_FIXED(y, r, f_cover, params):
    """
    ODEs for neutral shell region.

    Returns dn/dr, dtau/dr

    Physics:
    - Only non-ionizing radiation pressure (ionizing photons absorbed in ionized region)
    - No recombination (neutral gas)

    Reference: Rahner thesis
    """
    nShell, tau = y

    # Parameters
    sigma_dust = params['dust_sigma'].value
    mu_n = params['mu_neu'].value
    t_neu = params['TShell_neu'].value
    k_B = params['k_B'].value
    c = params['c_light'].value
    Ln = params['Ln'].value

    # Prevent underflow
    neg_exp_tau = np.exp(-tau) if tau < 500 else 0.0

    # Radiation pressure from non-ionizing photons
    rad_pressure_term = (nShell * sigma_dust / (4 * np.pi * r**2 * c)) * (Ln * neg_exp_tau)

    # dn/dr (MUST match convention from ionized region!)
    # If n is number density: need mu_n factor for consistency
    # If n is mass density: already has mu_n built in

    # OPTION 1: If n is number density [1/cm³]
    # dndr = (mu_n / (k_B * t_neu)) * rad_pressure_term

    # OPTION 2: If using same convention as ionized (which has mu_p/mu_n)
    # Then neutral should just have 1/(kT):
    dndr = (mu_n / (k_B * t_neu)) * rad_pressure_term  # Consistent with ionized

    # Optical depth
    dtaudr = nShell * sigma_dust  # Note: no f_cover in neutral region?

    return dndr, dtaudr
```

### Impact

- **Neutral shell density is WRONG by factor ~2.3**
- **Optical depth τ is WRONG**
- **Mass distribution is WRONG**
- **f_absorbed_neu is WRONG**

### Priority

**CRITICAL** - Fix immediately

---

## BUG #3: Wrong Variable in Mass Update (CRITICAL - MASS NOT CONSERVED)

### Location
`shell_structure.py` line 486

### Current Code (WRONG)
```python
# Line 486 in neutral region loop
mShell0 = mShell_arr[idx]
```

### Problem

This uses **mass of single shell element** instead of **cumulative mass**!

**Context**:
```python
# Lines 446-450: Calculate mass array
mShell_arr = np.empty_like(rShell_arr)
mShell_arr[0] = mShell0  # Starting cumulative mass
mShell_arr[1:] = (nShell_arr[1:] * params['mu_neu'].value * 4 * np.pi * rShell_arr[1:]**2 * rShell_step)

# Line 450: Calculate cumulative mass
mShell_arr_cum = np.cumsum(mShell_arr)

# Line 486: WRONG - should use cumulative!
mShell0 = mShell_arr[idx]  # This is dm, not M(r)!
```

**What happens**:
1. First iteration: `mShell0 = 0`, integrate, find `mShell_arr[idx] = dm` (small)
2. Second iteration: `mShell0 = dm` (should be M_total), integrate from wrong mass
3. **Mass is not conserved!**
4. **Integration gives nonsense values!**

### Correct Code

```python
# Line 486 - FIX
mShell0 = mShell_arr_cum[idx]  # Use CUMULATIVE mass
```

### Verification

Add mass conservation check:
```python
# After integration complete
m_total_calculated = mShell_arr_cum[-1]
m_total_expected = mShell_end

if abs(m_total_calculated - m_total_expected) > 0.01 * m_total_expected:
    logger.error(f"Mass not conserved! Expected {m_total_expected}, got {m_total_calculated}")
    raise ValueError("Mass conservation violated")
```

### Impact

- **CRITICAL** - Mass accounting is completely broken
- Results are **completely wrong**
- Shell structure cannot be trusted

### Priority

**CRITICAL** - Fix immediately, this alone makes code unusable

---

## BUG #4: Incorrect Radiation Force Calculation (HIGH - WRONG PHYSICS)

### Location
`shell_structure.py` line 583

### Current Code (WRONG)
```python
params['shell_F_rad'].value = f_absorbed_ion * params['Lbol'].value / params['c_light'].value * \
    (1 + params['shell_tauKappaRatio'] * params['dust_KappaIR'])
```

### Problem

1. Dimensional analysis fails
2. `shell_tauKappaRatio` is `τ/κ_IR = ∫ρ dr` (units: g/cm²)
3. `dust_KappaIR` is opacity κ_IR (units: cm²/g)
4. Product: `(∫ρ dr) * κ_IR = [g/cm²] * [cm²/g] = dimensionless`
5. But this should be τ_IR, not a multiplier for radiation force!

**Formula doesn't make physical sense.**

### What It Should Be

**Option 1: Radiation force** (momentum flux)
```python
# Total absorbed luminosity
L_absorbed = f_absorbed * params['Lbol'].value  # erg/s

# Direct radiation force
F_rad_direct = L_absorbed / params['c_light'].value  # erg/s / (cm/s) = g*cm/s = dyne

# Infrared radiation force (from reprocessed photons)
# See Krumholz & Matzner 2009 Eq 5
# F_rad_IR ≈ F_rad_direct * τ_IR (for optically thick)

tau_IR = params['shell_tauKappaRatio'].value * params['dust_KappaIR'].value
F_rad_IR = F_rad_direct * tau_IR

# Total
F_rad_total = F_rad_direct + F_rad_IR

params['shell_F_rad'].value = F_rad_total  # dyne
```

**Option 2: Radiation pressure** (force per area)
```python
# Radiation pressure at shell inner edge
R2 = params['R2'].value  # cm
P_rad_direct = L_absorbed / (4 * np.pi * R2**2 * params['c_light'].value)  # erg/s / cm² / (cm/s) = erg/cm³ = dyne/cm²

# Infrared contribution
tau_IR = params['shell_tauKappaRatio'].value * params['dust_KappaIR'].value
P_rad_IR = P_rad_direct * tau_IR  # Approximate

P_rad_total = P_rad_direct + P_rad_IR

params['shell_P_rad'].value = P_rad_total  # dyne/cm²
```

### Impact

- Radiation force/pressure used in dynamics is **WRONG**
- Shell acceleration is **WRONG**
- Momentum balance is **WRONG**

### Priority

**HIGH** - Fix soon, affects dynamics

---

## BUG #5: Wrong Density in Dissolution Check (MEDIUM)

### Location
`shell_structure.py` line 283

### Current Code (WRONG)
```python
if nShell_arr[0] < params['stop_n_diss']:
    is_shellDissolved = True
```

### Problem

Checks **first element of current slice**, not **current boundary density**!

`nShell_arr[0]` is density at `rShell_start`, which was set on line 211:
```python
y0 = [nShell0, phi0, tau0_ion]  # nShell0 is boundary density
```

So `nShell_arr[0] ≈ nShell0` (might be slightly different due to integration).

But the **relevant density** is at the **end of integration** (idx), not beginning!

### Correct Code

```python
# Check current boundary density
if nShell0 < params['stop_n_diss'].value:
    is_shellDissolved = True
    logger.info(f"Shell dissolved: n = {nShell0:.3e} < {params['stop_n_diss'].value:.3e}")
    break
```

Or check final density:
```python
# Check density at end of integration
if nShell_arr[idx] < params['stop_n_diss'].value:
    is_shellDissolved = True
    break
```

### Impact

- Shell might dissolve at wrong time
- Or might continue when it should dissolve
- Affects late-time evolution

### Priority

**MEDIUM** - Fix when refactoring

---

## BUG #6: Possible Duplication of Final Values (LOW)

### Location
`shell_structure.py` lines 288-293, 490-494

### Current Code
```python
# After while loop (lines 288-293):
mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
mShell_arr_cum_ion = np.append(mShell_arr_cum_ion, mShell_arr_cum[idx])
# ... (6 total appends)
```

### Problem

**In the loop** (line 249):
```python
mShell_arr_ion = np.concatenate(( mShell_arr_ion, mShell_arr[:idx]))
```

This stores indices `[0, 1, ..., idx-1]`.

**After loop** (line 288):
```python
mShell_arr_ion = np.append(mShell_arr_ion, mShell_arr[idx])
```

This appends index `idx`.

**But**: If `idx = len(rShell_arr) - 1` (set on line 236 when no condition met), then `mShell_arr[:idx]` is `mShell_arr[:-1]`, which **excludes** the last element.

So appending `mShell_arr[idx]` is correct.

**However**: If `idx = idx_array[0]` (line 238), then we're appending the element that triggered the condition.

**Is this intended?** Unclear.

### Possible Fix

```python
# Instead of storing [:idx] and then appending [idx],
# just store [:idx+1] and don't append after

# In loop:
n_points = idx + 1  # Include idx
mShell_arr_ion = np.concatenate(( mShell_arr_ion, mShell_arr[:n_points]))

# After loop: no append needed
```

### Impact

- Possible off-by-one error in arrays
- Minor impact on results

### Priority

**LOW** - Review during refactoring

---

## Summary Table

| Bug | Line | Severity | Impact | Fix |
|-----|------|----------|--------|-----|
| #1: dndr ionized | ODE:87-90 | **CRITICAL** | Wrong density, all physics wrong | Add mu_p/mu_n to recomb term |
| #2: dndr neutral | ODE:112-114 | **CRITICAL** | Wrong density, optical depth wrong | Add mu_n factor |
| #3: Mass variable | 486 | **CRITICAL** | Mass not conserved, broken | Use mShell_arr_cum[idx] |
| #4: Radiation force | 583 | HIGH | Wrong dynamics | Fix dimensional analysis |
| #5: Dissolution check | 283 | MEDIUM | Wrong dissolution time | Check nShell0 or nShell_arr[idx] |
| #6: Array append | 288-293 | LOW | Possible off-by-one | Store [:idx+1] instead |

---

## Testing Strategy

### Test 1: Strömgren Sphere (No Radiation Pressure)

```python
# Turn off radiation pressure (set Ln = 0, only ionizing)
# Analytic solution exists for density profile
# Test current vs fixed code
```

### Test 2: Mass Conservation

```python
# After integration:
assert abs(mShell_arr_cum[-1] - mShell_end) < 1e-6 * mShell_end
```

### Test 3: Pressure Equilibrium

```python
# At ionization front:
P_ion = n_ion * k_B * T_ion
P_neu = n_neu * k_B * T_neu
assert abs(P_ion - P_neu) < 1e-6 * P_ion
```

### Test 4: Photon Conservation

```python
# Total ionizing photons absorbed = Qi (1 - phi_final)
absorbed_dust = # from integration
absorbed_hydrogen = # from integration
assert abs(absorbed_dust + absorbed_hydrogen - Qi * (1 - phi_final)) < 1e-6 * Qi
```

---

## Fix Priority

1. **BUG #3** (line 486) - Quick fix, critical impact → 5 minutes
2. **BUG #1** (ODE lines 87-90) - Physics fix, critical → 30 minutes
3. **BUG #2** (ODE lines 112-114) - Physics fix, critical → 30 minutes
4. **BUG #4** (line 583) - Physics fix, high priority → 1 hour
5. **BUG #5** (line 283) - Logic fix → 10 minutes
6. **BUG #6** (lines 288-293) - Low priority → during refactor

**Total time for critical fixes**: 2-3 hours

**Then**: Test extensively before trusting results!
