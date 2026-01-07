#!/usr/bin/env python3
"""
Quick fixes for run_energy_phase.py and energy_phase_ODEs.py

Apply these patches to fix critical bugs without full rewrite.
Copy-paste the relevant sections to replace buggy code.
"""

# =============================================================================
# FIX 1: Replace manual Euler with proper ODE solver
# =============================================================================
# In run_energy_phase.py, REPLACE lines 220-280 with:

def run_energy_with_proper_ode(params):
    """Example of how to use odeint() properly."""

    # ... [initialization code stays the same] ...

    while R2 < rCloud and (tfinal - t_now) > 1e-4 and continueWeaver:

        # Update cooling structures
        if np.abs(params['t_previousCoolingUpdate'] - params['t_now']) > 5e-2:
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
            params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            params['cStruc_heating_nonCIE'].value = heating_nonCIE
            params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            params['t_previousCoolingUpdate'].value = params['t_now'].value

        # Calculate bubble and shell structure
        calculate_bubble_shell = loop_count > 0

        if calculate_bubble_shell:
            _ = bubble_luminosity.get_bubbleproperties(params)
            T0 = params['bubble_T_r_Tb'].value
            params['T0'].value = T0
            Tavg = params['bubble_Tavg'].value
            shell_structure.shell_structure(params)
        else:
            Tavg = T0

        c_sound = operations.get_soundspeed(Tavg, params)
        params['c_sound'].value = c_sound

        # =============================================================================
        # PROPER ODE INTEGRATION
        # =============================================================================

        # Create time array with reasonable timestep
        # Let solver choose actual timestep adaptively
        tsteps = 30
        t_arr = np.linspace(t_now, t_now + (dt_min * tsteps), tsteps + 1)[1:]

        # Initial conditions
        y0 = [R2, v2, Eb, T0]

        # Call ODE solver with adaptive timestep
        psoln = scipy.integrate.odeint(
            energy_phase_ODEs.get_ODE_Edot,
            y0,
            t_arr,
            args=(params,),
            rtol=1e-6,  # Relative tolerance
            atol=1e-8,  # Absolute tolerance
            full_output=False
        )

        # Extract solutions
        r_arr = psoln[:, 0]
        v_arr = psoln[:, 1]
        Eb_arr = psoln[:, 2]
        # T0_arr = psoln[:, 3]  # Not used (derivative = 0)

        # Update for next iteration
        t_now = t_arr[-1]
        R2 = r_arr[-1]
        v2 = v_arr[-1]
        Eb = Eb_arr[-1]
        # T0 stays the same or is updated from bubble_T_r_Tb

        # =============================================================================
        # Rest of loop stays the same
        # =============================================================================

        # Record arrays (use lists for efficiency)
        if loop_count == 0:
            # Initialize lists
            array_t_now = [t_now]
            array_R2 = [R2]
            array_R1 = [params['R1'].value]
            array_v2 = [v2]
            array_T0 = [T0]
            array_mShell = []
        else:
            # Append to lists
            array_t_now.append(t_now)
            array_R2.append(R2)
            array_R1.append(params['R1'].value)
            array_v2.append(v2)
            array_T0.append(T0)

        # Get shell mass
        mShell_arr = mass_profile.get_mass_profile(r_arr, params, return_mdot=False)
        Msh0 = mShell_arr[-1]
        array_mShell.append(Msh0)

        # Save snapshot
        params.save_snapshot()

        # Update feedback
        [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] = get_currentSB99feedback(t_now, params)

        # Recalculate R1 and Pb
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1,
                                   1e-3 * R2, R2,
                                   args=([LWind, Eb, vWind, R2]))
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

        updateDict(params,
                  ['R1', 'R2', 'v2', 'Eb', 't_now', 'Pb', 'shell_mass'],
                  [R1, R2, v2, Eb, t_now, Pb, Msh0])

        loop_count += 1

    # Convert lists to arrays at end
    params['array_t_now'].value = np.array(array_t_now)
    params['array_R2'].value = np.array(array_R2)
    params['array_R1'].value = np.array(array_R1)
    params['array_v2'].value = np.array(array_v2)
    params['array_T0'].value = np.array(array_T0)
    params['array_mShell'].value = np.array(array_mShell)

    return


# =============================================================================
# FIX 2: Merge duplicate ODE functions and remove hack
# =============================================================================
# In energy_phase_ODEs.py, REPLACE both get_ODE_Edot functions with this one:

def get_ODE_Edot(y, t, params):
    """
    ODE system for bubble expansion during energy-driven phase.

    Parameters
    ----------
    y : list of float
        State vector [R2, v2, Eb, T0]
        R2 : Shell outer radius [pc]
        v2 : Shell velocity [pc/Myr]
        Eb : Bubble energy [Msun*pc^2/Myr^2]
        T0 : Temperature at bubble/shell interface [K]
    t : float
        Time [Myr]
    params : DescribedDict
        Dictionary with all simulation parameters

    Returns
    -------
    derivs : list of float
        Time derivatives [dR2/dt, dv2/dt, dEb/dt, dT0/dt]

    Notes
    -----
    This is a PURE function - it only READS from params, never writes.
    Force balance: F_net = F_pressure - F_gravity + F_radiation - F_drag
    Energy: dEb/dt = L_wind - L_cooling - PdV - L_leak
    """

    # Unpack state
    R2, v2, Eb, T0 = y
    R2 = float(R2)
    v2 = float(v2)
    Eb = float(Eb)

    # Pull frequently-used parameters
    FABSi = params["shell_fAbsorbedIon"].value
    F_rad = params["shell_F_rad"].value
    mCluster = params["mCluster"].value
    L_bubble = params["bubble_LTotal"].value
    gamma = params["gamma_adia"].value
    tSF = params["tSF"].value
    G = params["G"].value
    Qi = params["Qi"].value
    LWind = params["LWind"].value
    vWind = params["vWind"].value
    k_B = params["k_B"].value
    TShell_ion = params["TShell_ion"].value

    # Calculate shell mass and time derivative
    if params['isCollapse'].value:
        # During collapse, shell mass stays constant
        mShell = params['shell_mass'].value
        mShell_dot = 0.0
    else:
        # During expansion, shell sweeps up mass
        mShell, mShell_dot = mass_profile.get_mass_profile(
            R2, params,
            return_mdot=True,
            rdot_arr=v2
        )
        mShell = _scalar(mShell)
        mShell_dot = _scalar(mShell_dot)

    # Gravitational force (cluster + shell self-gravity)
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # Calculate inner bubble radius (wind termination shock)
    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-10,  # Small lower bound to avoid R1=0
            R2,
            args=([LWind, Eb, vWind, R2])
        )
    except ValueError:
        # If brentq fails, use small fraction of R2
        logger.warning(f"R1 calculation failed at t={t:.3e}, using R1=0.001*R2")
        R1 = 0.001 * R2

    # Bubble pressure - smooth switchon over dt_switchon after star formation
    if params['current_phase'].value in ['momentum']:
        # Momentum phase: pressure from ram pressure
        press_bubble = get_bubbleParams.pRam(R2, LWind, vWind)
    else:
        # Energy phase: pressure from bubble energy
        dt_switchon = 1e-3  # Myr - gradual switchon time
        time_since_SF = t - tSF

        if time_since_SF <= dt_switchon:
            # Gradually ramp up R1 from 0 to full value
            frac = np.clip(time_since_SF / dt_switchon, 0.0, 1.0)
            R1_eff = frac * R1
        else:
            # After switchon period, use full R1
            R1_eff = R1

        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_eff, gamma)

    # HII region pressure INSIDE shell (inward force)
    if FABSi < 1.0:
        # Ionization front hasn't broken out of shell
        # Pressure from photoionized gas outside shell
        try:
            r_shell = params['rShell'].value
            n_r = density_profile.get_density_profile(np.array([r_shell]), params)
            press_HII_in = 2.0 * n_r[0] * k_B * TShell_ion
        except:
            press_HII_in = 0.0
    else:
        press_HII_in = 0.0

    # Add ISM pressure if beyond cloud radius
    if params['rShell'].value >= params['rCloud'].value:
        press_HII_in += params['PISM'].value * k_B

    # HII region pressure OUTSIDE shell (outward force)
    if FABSi < 1.0:
        # Ionization confined - use ISM density
        nR2 = params['nISM'].value
    else:
        # Ionization broken out - use Stromgren sphere approximation
        nR2 = np.sqrt(Qi / params['caseB_alpha'].value / R2**3 * 3.0 / 4.0 / np.pi)

    press_HII_out = 2.0 * nR2 * k_B * TShell_ion

    # Leaking luminosity (future: add covering fraction after fragmentation)
    L_leak = 0.0

    # =============================================================================
    # Time derivatives
    # =============================================================================

    # dR2/dt = v2 (velocity definition)
    rd = v2

    # dv2/dt = F_net / mShell (Newton's 2nd law)
    F_pressure = 4.0 * np.pi * R2**2 * (press_bubble - press_HII_in + press_HII_out)
    F_drag = mShell_dot * v2

    vd = (F_pressure - F_drag - F_grav + F_rad) / mShell

    # dEb/dt = L_wind - L_cool - PdV - L_leak (energy balance)
    PdV_work = 4.0 * np.pi * R2**2 * press_bubble * v2
    Ed = LWind - L_bubble - PdV_work - L_leak

    # dT0/dt = 0 (T0 is updated from bubble structure, not evolved by ODE)
    Td = 0.0

    return [rd, vd, Ed, Td]


# =============================================================================
# FIX 3: Add logging instead of print statements
# =============================================================================
# At top of run_energy_phase.py, add:

import logging
logger = logging.getLogger(__name__)

# Then replace all print() statements with logger calls:

# Line 77-80: REPLACE
# print(f'Inner discontinuity: {R1} pc')
# print(f'Initial bubble mass: {Msh0}')
# print(f'Initial bubble pressure: {Pb}')
# WITH:
logger.info(f'Energy phase initialization:')
logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
logger.info(f'  Initial shell mass: {Msh0:.6e} Msun')
logger.info(f'  Initial bubble pressure: {Pb:.6e} Msun/pc/Myr^2')

# Line 154: REPLACE
# print('t_arr is this', t_arr)
# WITH:
logger.debug(f'Time array: {t_arr[0]:.6e} to {t_arr[-1]:.6e} Myr ({len(t_arr)} steps)')

# Line 171, 175, 178: REPLACE
# print('\n\nFinish bubble\n\n')
# print('\n\nShell structure calculated.\n\n')
# print('bubble and shell not calculated.')
# WITH:
logger.debug('Bubble structure calculated')
logger.debug('Shell structure calculated')
logger.debug('Skipping bubble/shell calculation (first iteration)')

# Line 228-231: REPLACE all those prints WITH:
logger.debug(f'Initial conditions: R2={R2:.6e}, v2={v2:.6e}, Eb={Eb:.6e}, T0={T0:.2e}')

# Line 257-265: REPLACE WITH:
logger.debug(f'Derivatives: rd={rd:.6e}, vd={vd:.6e}, Ed={Ed:.6e}, Td={Td:.6e}')
logger.debug(f'Updated: R2={R2:.6e}, v2={v2:.6e}, Eb={Eb:.6e}, T0={T0:.2e}')

# Line 274: REPLACE
# print('\n\n\n\n\n\n\nswitch to no approximation\n\n\n\n\n\n')
# WITH:
logger.info('Switched off EarlyPhaseApproximation after 10 iterations')

# Line 346: REPLACE
# print('saving snapshot')
# WITH:
logger.debug(f'Saving snapshot at t={t_now:.6e} Myr')


# =============================================================================
# FIX 4: Define magic numbers as constants
# =============================================================================
# At top of run_energy_phase.py, after imports:

# Energy phase constants
TFINAL_ENERGY_PHASE = 3e-3  # Myr - max duration of energy phase (~3000 years)
DT_MIN = 1e-6  # Myr - minimum timestep (31.5 seconds)
DT_EXIT_THRESHOLD = 1e-4  # Myr - exit when this close to tfinal (~100 years)
COOLING_UPDATE_INTERVAL = 5e-2  # Myr - recalculate cooling every 50k years
TIMESTEPS_PER_LOOP = 30  # Number of timesteps per main loop iteration
EARLY_APPROX_ITERATIONS = 10  # Switch off approximation after this many iterations

# Then replace hardcoded values with these constants throughout


# =============================================================================
# FIX 5: Fix bare except clause
# =============================================================================
# Line 241-244: REPLACE
# try:
#     params['t_next'].value = t_arr[ii+1]
# except:
#     params['t_next'].value = time + dt_min
# WITH:
try:
    params['t_next'].value = t_arr[ii+1]
except IndexError:
    # Last element - next time is one more timestep
    params['t_next'].value = time + dt_min


# =============================================================================
# FIX 6: Remove redundant shell mass block
# =============================================================================
# In energy_phase_ODEs.py, DELETE lines 77-86 entirely
# (The duplicate calculation on lines 271-283 is sufficient)


# =============================================================================
# FIX 7: Fix inefficient array concatenation
# =============================================================================
# In run_energy_phase.py, before main loop:

# Initialize as lists, not arrays
array_t_now = []
array_R2 = []
array_R1 = []
array_v2 = []
array_T0 = []
array_mShell = []

# In loop (lines 329-341), REPLACE with:
array_t_now.append(t_now)
array_R2.append(R2)
array_R1.append(params['R1'].value)
array_v2.append(v2)
array_T0.append(T0)
array_mShell.append(Msh0)

# After loop exits, convert to arrays:
params['array_t_now'].value = np.array(array_t_now)
params['array_R2'].value = np.array(array_R2)
params['array_R1'].value = np.array(array_R1)
params['array_v2'].value = np.array(array_v2)
params['array_T0'].value = np.array(array_T0)
params['array_mShell'].value = np.array(array_mShell)


# =============================================================================
# FIX 8: Remove ALL commented code
# =============================================================================
# In run_energy_phase.py:

# DELETE lines 72-76 (commented density profile check)
# DELETE lines 86, 101 (commented tfinal alternatives)
# DELETE lines 199-217 (use this code instead!)
# DELETE lines 276-278 (commented sys.exit)
# DELETE lines 353-360 (commented momentum transition)
# DELETE lines 384-389 (commented checks)
# DELETE lines 393-830 (ENTIRE block of old code - 437 lines!)

# In energy_phase_ODEs.py:

# DELETE lines 99-112 (commented R1 and P_bub calculations)
# DELETE lines 378-380 (commented mShell_dot hack)

# This should reduce run_energy_phase.py from 830 lines to ~400 lines
