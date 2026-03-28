# Task: Implement standalone Strömgren P_HII as independent feedback channel

Branch name: PHII-fixing/Standalone-Phii

## Physics summary

TRINITY currently computes P_HII from the shell inner-edge density, which is
anchored to Pb via pressure equilibrium (Rahner+2017). This makes P_HII ≈ Pb
by construction — it is not an independent feedback channel.

We fix this by computing a "standalone Strömgren" P_HII: the pressure that
photoionization alone would provide if there were no wind bubble. This is a
**counterfactual pressure floor** computed from the ambient cloud density
profile. The actual shell gas is denser (swept up and compressed by the
bubble), but P_HII_St represents the pressure scale that photoionization can
independently sustain.

**Key design decision: shell structure and P_drive are decoupled.**
- Shell structure stays Pb-anchored (nShell0 ∝ Pb). This correctly
  describes the actual shell density, absorption fractions, and gravity.
- P_drive uses the independently computed P_HII_St. This enters only
  the momentum equation (shell acceleration), not the shell internal structure.

### Key formulae

**Strömgren radius** in the ambient cloud density profile n_cloud(r),
accounting for the transparent bubble cavity:

    Qi = alpha_B * integral_{R2}^{R_St} [ n_cloud^2(r) * 4*pi*r^2 dr ]

The lower limit is R2 (the current bubble outer radius), NOT 0.
Reason: the hot bubble interior is transparent to ionizing photons —
zero recombinations occur from 0 to R2. The full Qi budget arrives at R2
and ionizes ambient-density gas from R2 outward. Note however that the
gas between R2 and R_shell is swept-up shell material at high density,
NOT ambient cloud gas. This integral is a counterfactual: it asks what
pressure photoionization would provide if the gas were undisturbed.

Solve for R_St given Qi(t) and R2(t) at each timestep.

**Standalone HII pressure** at the ionization front:

    P_HII_St = 2 * n_cloud(R_St) * k_B * T_ion

No correction factor needed — the formula is standard. The cavity
geometry is fully captured by the lower integration limit.

**Driving pressure** by phase:
- Energy / implicit:  P_drive = max(Pb, P_HII_St)
- Transition:         P_drive = max(Pb, P_HII_St + P_ram)
- Momentum:           P_drive = P_HII_St + P_ram

**Shell inner-edge boundary condition: UNCHANGED.**
    nShell0 = (mu_ion / mu_atom) / (k_B * T_ion) * Pb
The shell density profile remains anchored to the actual confining
pressure (Pb in energy phase, P_ram in momentum phase via the existing hack).

## READ FIRST (mandatory)

Before writing ANY code, read and understand these files:

1. `src/cloud_properties/density_profile.py`
   - Find `get_density_profile(r, params)` — understand its signature,
     what params keys it needs, and what units it returns.
   - **Units: returns number density in code units [1/pc³].**
     All params values are auto-converted from CGS to code units
     [Msun, pc, Myr] by `read_param.py`. Everything inside main()
     is in code units.
   - This is how you evaluate n_cloud(r) at any radius.

2. `src/shell_structure/shell_structure_modified.py`
   - Find the line: `nShell0 = (mu_ion / mu_atom) / (k_B * T_ion) * Pb`
   - This boundary condition is NOT modified. Read to understand the
     existing flow so you don't accidentally break it.

3. `src/sb99/update_feedback.py`
   - Find how `Qi` (ionizing photon rate) is accessed from feedback.
   - Flow: SB99 interpolation → `feedback.Qi` (attribute on SB99Feedback
     dataclass) → copied to `params['Qi'].value` via `updateDict(params, feedback)`.
   - In phase runners, access as `params['Qi'].value`.

4. `src/_functions/unit_conversions.py`
   - Check conversion constants (e.g. `ndens_cgs2au`, `k_B_cgs2au`).
   - **Note:** `caseB_alpha` is NOT in this file. It is a param declared
     in `param/default.param` (unit: `cm**3 * s**-1`) and auto-converted
     to code units by `read_param.py`. Access as `params['caseB_alpha'].value`.
   - Similarly, `k_B` is a param: `params['k_B'].value` (code units:
     `Msun*pc²/Myr²/K`).
   - All calculations should stay in code units; only plotting scripts
     convert to CGS.

5. Phase runner files (read the main loop structure):
   - `src/phase1_energy/run_energy_phase_modified.py`
   - `src/phase1b_energy_implicit/run_energy_implicit_phase_modified.py`
   - `src/phase1c_transition/run_transition_phase_modified.py`
   - `src/phase2_momentum/run_momentum_phase_modified.py`
   - In each: find where Qi, Pb, P_drive are computed/set per timestep.

6. `src/bubble_structure/get_bubbleParams.py`
   - Find `pRam()` function signature:
     `pRam(r, Lwind, v_mech_total)` → returns `Lwind / (2*pi*r²*v_mech_total)`.

7. `src/phase1_energy/energy_phase_ODEs_modified.py`
   - Find `get_ODE_Edot_pure()` and `compute_derived_quantities()`.
   - These also compute P_drive — they need the same update.
   - Find the `ODESnapshot` dataclass and `create_ODE_snapshot()`.
   - Note: P_drive already branches by phase (energy vs transition)
     inside the ODE function. The existing logic is:
     ```python
     P_HII = 2.0 * snapshot.n_IF * snapshot.k_B * snapshot.TShell_ion
     if snapshot.current_phase == 'transition':
         P_drive = max(press_bubble, P_HII + P_b_ram)
     else:
         P_drive = max(press_bubble, P_HII)
     ```

## Implementation plan

### Step 1: Create a utility function for R_St and P_HII_St

Create a new file `src/cloud_properties/stromgren.py`:

```python
def compute_P_HII_Stromgren(Qi, R2, params):
    """
    Compute standalone Strömgren HII pressure from ambient cloud profile,
    accounting for the transparent bubble cavity.

    Solves:  Qi = alpha_B * integral_{R2}^{R_St} [ n_cloud^2(r) 4 pi r^2 dr ]
    Returns: P_HII_St = 2 * n_cloud(R_St) * k_B * T_ion

    The lower integration limit is R2 (bubble outer radius), not 0:
    the hot bubble interior is transparent to ionizing photons, so the
    full Qi budget arrives at R2 undiminished.

    Parameters
    ----------
    Qi : float
        Ionising photon rate [1/Myr] (code units).
    R2 : float
        Current bubble outer radius [pc]. Used as lower integration limit.
    params : dict-like
        Must contain keys needed by density_profile.get_density_profile(),
        plus 'caseB_alpha', 'k_B', 'TShell_ion', 'rCloud'.
        All values in code units [Msun, pc, Myr].

    Returns
    -------
    P_HII_St : float
        Standalone Strömgren HII pressure [Msun/Myr²/pc] (code units).
    R_St : float
        Strömgren radius [pc] (measured from centre, not from R2).
    n_St : float
        Ambient cloud density at R_St [1/pc³] (code units).

    Notes
    -----
    If Qi <= 0, returns (0, 0, 0).
    If R_St > rCloud, clamps to rCloud and uses n_cloud(rCloud).
    """
```

Implementation notes:
- Use `scipy.integrate.quad` for the integral of `n_cloud^2(r) * 4*pi*r^2`
  from **R2** to r, where n_cloud(r) comes from `get_density_profile(r, params)`.
- Use `scipy.optimize.brentq` to find R_St where the cumulative integral
  (from R2 to R_St) equals Qi / alpha_B.
- Set upper bracket for brentq to rCloud (or some multiple).
  If the integral from R2 to rCloud is less than Qi/alpha_B, the HII region
  extends beyond the cloud — clamp R_St = rCloud.
- Handle edge cases: Qi = 0, R2 very small (use max(R2, small_floor)),
  R_St very close to R2.

### Step 2: Modify each phase runner

In each phase runner's main loop, AFTER getting feedback values (Qi, Lmech, etc.)
and BEFORE the shell structure call, compute P_HII_St:

```python
# Compute standalone Strömgren HII pressure (cavity-aware: integral from R2)
P_HII_St, R_St, n_St = compute_P_HII_Stromgren(Qi, R2, params)
params['P_HII_St'].value = P_HII_St
params['R_St'].value = R_St
params['n_St'].value = n_St
```

Then, AFTER the shell structure call, set P_drive using P_HII_St:

**Energy / implicit phase:**
```python
# Shell structure already computed with Pb-anchored BC (unchanged).
# P_drive uses the independent Strömgren pressure as a floor.
P_drive = max(Pb, P_HII_St)
params['P_drive'].value = P_drive
```

**Transition phase:**
```python
P_drive = max(Pb, P_HII_St + P_ram)
params['P_drive'].value = P_drive
```

**Momentum phase:**
```python
P_drive = P_HII_St + P_ram
params['P_drive'].value = P_drive
```

**Note on R2 access:** R2 is available differently in each phase:
- Energy phase: from ODE state variable
- Implicit/transition: check how R2 is tracked in the loop
- Momentum phase: from params or loop variable
Read each phase runner to find the correct R2 variable.

### Step 3: Shell structure — DO NOT CHANGE

The shell boundary condition stays as-is:

```python
nShell0 = (mu_ion / mu_atom) / (k_B * T_ion) * Pb
```

Rationale: the shell gas is physically compressed by the bubble (or ram
pressure). The Pb-anchored BC correctly describes the actual shell density
structure. P_HII_St is a counterfactual pressure floor representing what
photoionization alone would provide in the undisturbed cloud — it enters
the driving pressure (momentum equation) but does NOT set the shell
internal structure.

The momentum-phase hack `params['Pb'] = pRam(...)` also STAYS — it gives
nShell0 a physically meaningful value from ram pressure confinement when
Eb = 0.

### Step 4: Modify energy_phase_ODEs_modified.py

In `get_ODE_Edot_pure()` and `compute_derived_quantities()`:

- Add `P_HII_St` to the `ODESnapshot` dataclass (frozen at segment start).
- In `create_ODE_snapshot()`: read `P_HII_St` from `params['P_HII_St'].value`.
- In the ODE function, change the P_drive calculation:

  ```python
  # OLD: P_drive = max(press_bubble, P_HII)
  #   where P_HII = 2 * n_IF * k_B * T_ion (shell-ODE density, ≈ Pb)
  # NEW: P_drive = max(press_bubble, P_HII_St)
  #   where P_HII_St is the standalone Strömgren pressure (independent of Pb)
  P_drive = max(press_bubble, snapshot.P_HII_St)
  ```

- Keep computing P_HII from n_IF for diagnostics (F_HII, F_ion_out), but
  it no longer enters P_drive.
- The existing phase branching (energy vs transition) for P_ram should
  be preserved — only replace the P_HII term with P_HII_St.

Similarly update `compute_derived_quantities()` to use `snapshot.P_HII_St`
for P_drive while keeping n_IF-based P_HII for diagnostic output fields.

### Step 5: Add new params fields

Ensure these fields exist in the params dictionary initialisation
(in `src/_input/read_param.py`, where other runtime params like `P_HII`,
`P_drive`, `P_ram` are initialised):

- `P_HII_St` — standalone Strömgren pressure (for driving + diagnostics)
- `R_St` — Strömgren radius (for diagnostics/output)
- `n_St` — density at R_St (for diagnostics/output)

No `Pb_raw` needed — Pb is never overwritten now.

## DO NOT change

- The shell ODE integration (Eqs. 49-51) and its boundary condition
  (nShell0 ∝ Pb). The shell internal structure stays Pb-anchored.
- The momentum-phase `params['Pb'] = pRam(...)` hack — it stays, giving
  the shell a physically meaningful inner-edge density when Eb = 0.
- The `get_density_profile()` function itself.
- The feedback interpolation (SB99 tables).
- The plotting scripts (they read whatever fields are in the output).
- Mass profile, gravity, radiation pressure calculations.
- The R1 (termination shock) calculation.
- The n_IF / n_IF_ODE / n_IF_Str fields in shell_structure — keep computing
  them for diagnostics, but they no longer enter P_drive.

## Verification checklist

- [ ] `compute_P_HII_Stromgren()` returns sensible values:
      for n=1e3 uniform cloud, Qi=1e49 s⁻¹, R_St should be ~few pc.
- [ ] In energy phase with strong bubble: Pb > P_HII_St, so P_drive = Pb
      and behaviour is unchanged from current code.
- [ ] In momentum phase: P_drive = P_HII_St + P_ram, NOT 2*P_ram.
- [ ] Shell nShell0 still uses Pb (NOT P_drive). Confirm unchanged.
- [ ] Momentum-phase `params['Pb'] = pRam(...)` hack still present.
- [ ] New fields (P_HII_St, R_St, n_St) appear in output snapshots.
- [ ] Code runs without errors on existing test cases.
- [ ] Run the pressureZeta diagnostic plot to verify P_HII_St diverges
      from P_HII_ODE when n_cloud(R2)/n_IF_ODE > 1.

## Edge cases

- **Qi = 0** (before massive stars form): P_HII_St = 0, P_drive = Pb. OK.
- **R_St > rCloud**: clamp to rCloud. P_HII_St = 2 n_cloud(rCloud) k_B T_ion
  (low, since cloud edge is at ambient density).
- **R_St barely exceeds R2**: Strömgren sphere marginally larger than bubble.
  P_HII_St ≈ 2 n_cloud(R2) k_B T_ion, i.e. local ambient pressure. Correct.
- **Very steep profiles (α = -2)**: n_cloud²(r) * r² ~ r^{-2} — integral
  is logarithmic. `scipy.integrate.quad` handles this fine.
- **Bonnor-Ebert profiles**: `get_density_profile` should handle natively.
  Test with a BE run.
- **P_HII_St < P_ram in momentum phase**: P_drive is dominated by P_ram,
  which is correct — weak HII region adds a small correction.
- **R2 very small (early times)**: integral from R2 ≈ 0 approaches the
  classical (no-cavity) Strömgren sphere. Use max(R2, small_floor) to avoid
  numerical issues with power-law profiles near r = 0.
