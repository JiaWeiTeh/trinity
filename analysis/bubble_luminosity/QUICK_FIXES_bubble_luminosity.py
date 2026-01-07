#!/usr/bin/env python3
"""
Quick fixes for bubble_luminosity.py

Apply these patches to fix critical bugs without full rewrite.
Copy-paste the relevant sections to replace buggy code.
"""

# =============================================================================
# FIX 1: Line 103 - Consistent .value access
# =============================================================================
# BEFORE:
# params['bubble_r_Tb'].value = params['R1'] + xi_Tb * (params['R2'] - params['R1'])

# AFTER:
params['bubble_r_Tb'].value = params['R1'].value + xi_Tb * (params['R2'].value - params['R1'].value)


# =============================================================================
# FIX 2: Lines 493-511 - Fix cumulative mass calculation
# =============================================================================
# BEFORE:
# def get_mass_and_grav(n, r):
#     r_new = r[::-1]
#     rho_new = n[::-1] * params['mu_ion'].value
#     rho_new = rho_new #.to(u.g/u.cm**3)å  # <-- Remove encoding error
#     m_new = 4 * np.pi * scipy.integrate.simps(rho_new * r_new**2, x = r_new)
#     m_cumulative = np.cumsum(m_new)  # <-- BUG: m_new is scalar!
#     ...

# AFTER:
def get_mass_and_grav(n, r):
    """
    Calculate cumulative mass and gravitational potential in bubble.

    Parameters
    ----------
    n : ndarray
        Number density [1/pc³], monotonically decreasing
    r : ndarray
        Radius [pc], monotonically decreasing

    Returns
    -------
    m_cumulative : ndarray
        Cumulative mass from center [Msun]
    grav_phi : float
        Gravitational potential [pc²/Myr²]
    grav_force_m : ndarray
        Gravitational force per unit mass [pc/Myr²]
    """
    # Flip arrays to be monotonically increasing
    r_new = r[::-1]
    rho_new = n[::-1] * params['mu_ion'].value  # Mass density [Msun/pc³]

    # Calculate cumulative mass properly
    m_cumulative = np.zeros_like(r_new)
    for i in range(len(r_new)):
        m_cumulative[i] = 4 * np.pi * scipy.integrate.simps(
            rho_new[:i+1] * r_new[:i+1]**2,
            x=r_new[:i+1]
        )

    # Gravitational potential [pc²/Myr²]
    grav_phi = -4 * np.pi * params['G'].value * scipy.integrate.simps(
        r_new * rho_new, x=r_new
    )

    # Gravitational force per unit mass [pc/Myr²]
    # Add small number to avoid division by zero at r=0
    grav_force_m = params['G'].value * m_cumulative / (r_new**2 + 1e-10)

    return m_cumulative, grav_phi, grav_force_m


# =============================================================================
# FIX 3: Lines 541-558 - Cleaner dMdt initialization
# =============================================================================
# BEFORE:
# dMdt_init = 12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * params['R2']**3 / params['t_now']\
#     * params['mu_neu'] / params['k_B'] * (params['t_now'] * params['C_thermal'] / params['R2']**2)**(2/7) * params['Pb']**(5/7)

# AFTER:
def get_init_dMdt(params):
    """
    Initial guess for dMdt using Weaver+77 Equation 33.

    dMdt is the mass flux from shell back into hot region via thermal conduction.

    Parameters
    ----------
    params : DescribedDict
        Parameters with keys: R2, t_now, mu_neu, k_B, C_thermal, Pb

    Returns
    -------
    dMdt_init : float
        Initial guess for mass flux [Msun/Myr]
    """
    # Weaver+77 empirical factor
    dMdt_factor = 1.646

    # Break down equation for clarity
    time_factor = params['t_now'].value
    geometry_factor = 4 * np.pi * params['R2'].value**3 / time_factor
    material_factor = params['mu_neu'].value / params['k_B'].value
    thermal_factor = (
        time_factor * params['C_thermal'].value / params['R2'].value**2
    )**(2/7)
    pressure_factor = params['Pb'].value**(5/7)

    dMdt_init = (12/75) * dMdt_factor**(5/2) * geometry_factor * \
                material_factor * thermal_factor * pressure_factor

    return dMdt_init


# =============================================================================
# FIX 4: Line 726 - Consistent .value access
# =============================================================================
# BEFORE:
# dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2']**2) )

# AFTER:
dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * dMdt_params_au['R2'].value**2))


# =============================================================================
# FIX 5: Remove all debug print statements
# =============================================================================
# Replace these lines with proper logging:

import logging
logger = logging.getLogger(__name__)

# Line 40: print('entering get_bubbleproperties')
logger.debug('Entering get_bubbleproperties')

# Line 81: print(f"The initial guess for dMdt is {params['bubble_dMdt'].value}.")
logger.debug(f"Initial dMdt guess: {params['bubble_dMdt'].value:.3e} Msun/Myr")

# Lines 236-242: Remove or replace with logger.debug()
logger.debug(f"Bubble structure: T_range=[{T_array.min():.2e}, {T_array.max():.2e}] K")

# Line 639: print('Rejected. minimum temperature:', min_T)
logger.debug(f'Rejected dMdt: min_T={min_T:.2e} < 3e4 K')

# Line 644: print('Rejected. minimum temperature:', min_T)
logger.debug(f'Rejected dMdt: NaN temperature')

# Line 648: print('temperature not monotonic')
logger.debug('Rejected dMdt: non-monotonic temperature')

# Line 661: print('record, and min temp is', min_T)
logger.debug(f'Accepted dMdt: min_T={min_T:.2e} K')

# Line 794: print('T is zero')
logger.error('Temperature is zero in ODE')

# Line 807: print('Getting np.nan Temeprature')
logger.error(f'NaN temperature in ODE at t={dMdt_params_au["t_now"].value:.3e}')


# =============================================================================
# FIX 6: Define magic numbers as constants (add at top of file)
# =============================================================================
"""
Add these constants near the top of the file after imports:
"""

# Physical temperature thresholds [K]
T_INIT = 3e4  # Initial temperature for ODE boundary condition
T_COOLING_SWITCH = 1e4  # Below this, no cooling is considered
T_CIE_SWITCH = 10**5.5  # Above this, use CIE cooling; below use non-CIE

# Numerical parameters
R1_SEARCH_FACTOR = 1e-3  # Fraction of R2 for lower bound in R1 search
ODE_RESOLUTION = int(2e4)  # Number of points in ODE integration
ODE_RESOLUTION_CONDUCTION_MIN = 100  # Minimum resolution for conduction zone
INTERPOLATION_EXTRA_POINTS = 20  # Extra points around CIE switch

# Solver tolerances
SOLVER_XTOL = 1e-4
SOLVER_EPSFCN = 1e-4
SOLVER_FACTOR = 50

# Physical constraints
MIN_TEMPERATURE = 3e4  # Minimum allowed temperature [K]
T_ZERO_THRESHOLD = 1e-5  # Threshold for considering temperature as zero

# Weaver+77 empirical factor
WEAVER_DMDT_FACTOR = 1.646

# Then replace all hardcoded values with these constants throughout the file


# =============================================================================
# FIX 7: Better residual calculation (Lines 628-641)
# =============================================================================
# BEFORE:
# residual = (v_array[-1] -  0) / (v_array[0] + 1e-4)
# min_T = np.min(T_array)
# if min_T < 3e4:
#     print('Rejected. minimum temperature:', min_T)
#     residual *= (3e4/(min_T+1e-1))**2
#     return residual

# AFTER:
def get_velocity_residuals(dMdt_init, dMdt_params_au):
    """Calculate residual for dMdt solver."""

    # Handle array input from solvers
    if hasattr(dMdt_init, '__iter__'):
        dMdt_init = dMdt_init[0]

    # ... [ODE integration code] ...

    # Calculate residual
    v_final = v_array[-1]
    v_initial = v_array[0]

    # Check for zero initial velocity
    if abs(v_initial) < 1e-10:
        logger.warning(f'Initial velocity near zero: {v_initial:.3e}')
        return 1e3  # Large penalty

    residual = v_final / v_initial

    # Check physical constraints
    min_T = np.min(T_array)

    if np.isnan(min_T):
        logger.debug(f'Rejected dMdt={dMdt_init:.3e}: NaN temperature')
        return 1e3

    if min_T < MIN_TEMPERATURE:
        logger.debug(f'Rejected dMdt={dMdt_init:.3e}: min_T={min_T:.2e} < {MIN_TEMPERATURE}')
        penalty = (MIN_TEMPERATURE / (min_T + 0.1))**2
        return residual * penalty

    if not operations.monotonic(T_array):
        logger.debug(f'Rejected dMdt={dMdt_init:.3e}: non-monotonic temperature')
        return 1e2

    logger.debug(f'Accepted dMdt={dMdt_init:.3e}: residual={residual:.3e}, min_T={min_T:.2e} K')
    return residual


# =============================================================================
# FIX 8: Typo corrections
# =============================================================================
# Line 58: "# The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]"
# Change to: "# The bubble Pressure [cgs - g/cm/s2, or dyn/cm2]"

# Line 244: "# calculate densit"
# Change to: "# calculate density"

# Line 501: Remove encoding error "å"
# rho_new = rho_new  # [Msun/pc³]

# Line 513: "# gettemåå"
# Change to: "# Calculate mass and gravity"

# Line 807: "print('Getting np.nan Temeprature')"
# Change to: "logger.error('Getting NaN Temperature')"
