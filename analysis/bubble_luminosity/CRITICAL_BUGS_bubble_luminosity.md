# Critical Bugs in bubble_luminosity.py

## Bug 1: Inconsistent .value access (Line 103) ⚠️ HIGH PRIORITY

**Current code:**
```python
params['bubble_r_Tb'].value = params['R1'] + xi_Tb * (params['R2'] - params['R1'])
```

**Problem:** Mixing DescribedItem objects with .value access causes TypeError

**Fix:**
```python
params['bubble_r_Tb'].value = params['R1'].value + xi_Tb * (params['R2'].value - params['R1'].value)
```

## Bug 2: Broken cumsum usage (Line 505) ⚠️ HIGH PRIORITY

**Current code:**
```python
m_new = 4 * np.pi * scipy.integrate.simps(rho_new * r_new**2, x = r_new)
m_cumulative = np.cumsum(m_new)  # BUG: m_new is a scalar!
```

**Problem:** `simps()` returns a scalar, not an array. `cumsum()` on a scalar returns array of repeated values.

**Fix:**
```python
# Calculate cumulative mass properly
m_cumulative = np.zeros_like(r_new)
for i in range(len(r_new)):
    m_cumulative[i] = 4 * np.pi * scipy.integrate.simps(
        rho_new[:i+1] * r_new[:i+1]**2,
        x=r_new[:i+1]
    )
```

## Bug 3: Typos and encoding errors

**Lines with typos:**
- Line 58: "Pbure" → "Pressure"
- Line 501: `rho_new = rho_new #.to(u.g/u.cm**3)å` (å character)
- Line 513: "gettemåå"
- Line 807: "Temeprature" → "Temperature"

**Fix:** Search and replace all typos

## Bug 4: Division by zero workarounds mask real issues

**Line 630:**
```python
residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)  # Hides zero velocity bug
```

**Line 640:**
```python
residual *= (3e4/(min_T+1e-1))**2  # Hides zero temperature bug
```

**Problem:** Adding small numbers masks the root cause of zero values

**Fix:** Proper error handling:
```python
if abs(v_array[0]) < 1e-10:
    raise ValueError(f"Initial velocity is essentially zero: {v_array[0]}")
residual = (v_array[-1] - 0) / v_array[0]
```

## Bug 5: Inconsistent .value access (Line 726)

**Current code:**
```python
dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2']**2) )
```

**Should be:**
```python
dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2) )
```

## Bug 6: Complex one-liner hard to debug (Lines 554-555)

**Current code:**
```python
dMdt_init = 12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * params['R2']**3 / params['t_now']\
    * params['mu_neu'] / params['k_B'] * (params['t_now'] * params['C_thermal'] / params['R2']**2)**(2/7) * params['Pb']**(5/7)
```

**Problem:** Operator precedence unclear, hard to debug, inconsistent .value access

**Fix:**
```python
# Break down Weaver+77 Eq. 33 for clarity
time_factor = params['t_now'].value
geometry_factor = 4 * np.pi * params['R2'].value**3 / time_factor
material_factor = params['mu_neu'].value / params['k_B'].value
thermal_factor = (time_factor * params['C_thermal'].value / params['R2'].value**2)**(2/7)
pressure_factor = params['Pb'].value**(5/7)

dMdt_init = (12/75) * dMdt_factor**(5/2) * geometry_factor * \
            material_factor * thermal_factor * pressure_factor
```
