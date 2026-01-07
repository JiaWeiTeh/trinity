#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED: energy_phase_ODEs.py

Key change: get_ODE_Edot_pure() is a PURE FUNCTION
- Only READS from params
- NEVER writes to params
- Same inputs -> same outputs (deterministic)
- Safe for scipy.integrate.odeint() to call multiple times

This solves your dictionary corruption issue!
"""

import sys
import numpy as np
import scipy.optimize
import logging

import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.sb99.update_feedback import get_currentSB99feedback

logger = logging.getLogger(__name__)

# Constants
FOUR_PI = 4.0 * np.pi


def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


def get_ODE_Edot_pure(y, t, params):
    """
    PURE ODE function for bubble expansion during energy-driven phase.

    This function is PURE - it only READS from params, never WRITES.
    This makes it safe for scipy.integrate.odeint() to call multiple times
    without corrupting your dictionary structure.

    Parameters
    ----------
    y : array-like
        State vector [R2, v2, Eb, T0]
        R2 : Shell outer radius [pc]
        v2 : Shell velocity [pc/Myr]
        Eb : Bubble energy [Msun*pc^2/Myr^2]
        T0 : Temperature at bubble/shell interface [K] (not evolved, carried along)
    t : float
        Time [Myr]
    params : DescribedDict
        Dictionary with simulation parameters (READ ONLY!)

    Returns
    -------
    derivs : list of float
        Time derivatives [dR2/dt, dv2/dt, dEb/dt, dT0/dt]

    Notes
    -----
    PURE FUNCTION:
    - No side effects
    - Does NOT modify params
    - Deterministic (same inputs -> same outputs)
    - Safe for scipy to call any number of times

    Physics:
    - Force balance: F_net = F_pressure - F_gravity + F_radiation - F_drag
    - Energy: dEb/dt = L_wind - L_cooling - PdV - L_leak
    """

    # Unpack state
    R2, v2, Eb, T0 = y
    R2 = float(R2)
    v2 = float(v2)
    Eb = float(Eb)

    # =========================================================================
    # READ parameters (OK to read, just don't write!)
    # =========================================================================

    # Frequently used parameters
    FABSi = params["shell_fAbsorbedIon"].value
    F_rad = params["shell_F_rad"].value
    mCluster = params["mCluster"].value
    L_bubble = params["bubble_LTotal"].value
    gamma = params["gamma_adia"].value
    tSF = params["tSF"].value
    G = params["G"].value
    Qi = params["Qi"].value
    k_B = params["k_B"].value
    TShell_ion = params["TShell_ion"].value

    # Get current stellar feedback at time t
    # Note: This reads from SB99 interpolation, doesn't modify params
    [Qi_t, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] = get_currentSB99feedback(t, params)

    # =========================================================================
    # Calculate shell mass and dM/dt
    # =========================================================================

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

    # =========================================================================
    # Calculate gravitational force
    # =========================================================================

    # F_grav = G * M_shell / R^2 * (M_cluster + 0.5 * M_shell)
    # Factor of 0.5 accounts for shell self-gravity
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # =========================================================================
    # Calculate inner bubble radius R1 (wind termination shock)
    # =========================================================================

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

    # =========================================================================
    # Calculate bubble pressure with smooth switchon
    # =========================================================================

    if params['current_phase'].value in ['momentum']:
        # Momentum phase: ram pressure dominates
        press_bubble = get_bubbleParams.pRam(R2, LWind, vWind)
    else:
        # Energy phase: thermal pressure from bubble energy
        dt_switchon = 1e-3  # Myr - gradual switchon time after star formation
        time_since_SF = t - tSF

        if time_since_SF <= dt_switchon and time_since_SF >= 0:
            # Gradually ramp up R1 from 0 to full value
            # This prevents numerical instability at very early times
            frac = time_since_SF / dt_switchon
            R1_eff = frac * R1
        else:
            # After switchon period, use full R1
            R1_eff = R1 if time_since_SF > dt_switchon else 0.0

        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_eff, gamma)

    # =========================================================================
    # Calculate HII region pressure INSIDE shell (inward force)
    # =========================================================================

    if FABSi < 1.0:
        # Ionization front hasn't broken out of shell
        # Pressure from photoionized gas outside shell pushes inward
        try:
            r_shell = params['rShell'].value
            n_r = density_profile.get_density_profile(np.array([r_shell]), params)
            press_HII_in = 2.0 * n_r[0] * k_B * TShell_ion
        except Exception as e:
            logger.debug(f"Error calculating press_HII_in: {e}")
            press_HII_in = 0.0
    else:
        press_HII_in = 0.0

    # Add ISM pressure if bubble extends beyond cloud radius
    if params['rShell'].value >= params['rCloud'].value:
        press_HII_in += params['PISM'].value * k_B

    # =========================================================================
    # Calculate HII region pressure OUTSIDE shell (outward force)
    # =========================================================================

    if FABSi < 1.0:
        # Ionization confined within shell - use ISM density
        nR2 = params['nISM'].value  # Fixed: added .value
    else:
        # Ionization broken out - use Stromgren sphere approximation
        # n = (3 * Q_i / (4π * α_B * R^3))^(1/2)
        nR2 = np.sqrt(Qi_t / params['caseB_alpha'].value / R2**3 * 3.0 / 4.0 / np.pi)

    # Pressure from ionized gas: P = 2 n k_B T (factor of 2 for full ionization)
    press_HII_out = 2.0 * nR2 * k_B * TShell_ion

    # =========================================================================
    # Leaking luminosity (future: add covering fraction after fragmentation)
    # =========================================================================

    L_leak = 0.0

    # =========================================================================
    # Calculate time derivatives
    # =========================================================================

    # dR2/dt = v2 (definition of velocity)
    rd = v2

    # dv2/dt = F_net / M_shell (Newton's 2nd law)
    # F_net = Pressure forces - Drag - Gravity + Radiation
    F_pressure = FOUR_PI * R2**2 * (press_bubble - press_HII_in + press_HII_out)
    F_drag = mShell_dot * v2  # Momentum loss to swept-up mass

    vd = (F_pressure - F_drag - F_grav + F_rad) / mShell

    # dEb/dt = L_in - L_out - PdV (energy balance)
    # L_in = wind luminosity
    # L_out = cooling + PdV work + leaking
    PdV_work = FOUR_PI * R2**2 * press_bubble * v2
    Ed = LWind - L_bubble - PdV_work - L_leak

    # dT0/dt = 0 (T0 is updated from bubble structure calculation, not ODE)
    Td = 0.0

    # =========================================================================
    # Return derivatives ONLY - NO side effects!
    # =========================================================================

    return [rd, vd, Ed, Td]


# =============================================================================
# OPTIONAL: Helper function to update params after ODE solve
# =============================================================================

def update_params_after_ode(params, t_final, R2_final, v2_final, Eb_final, T0_final):
    """
    Update params dictionary after ODE integration completes.

    Call this AFTER scipy.integrate.odeint() finishes.
    This is where we update the dictionary - not during ODE evaluation.

    Parameters
    ----------
    params : DescribedDict
        Parameters dictionary to update
    t_final, R2_final, v2_final, Eb_final, T0_final : float
        Final values from ODE integration

    Returns
    -------
    None
        Modifies params in place
    """

    # Update primary state variables
    params['t_now'].value = t_final
    params['R2'].value = R2_final
    params['v2'].value = v2_final
    params['Eb'].value = Eb_final
    params['T0'].value = T0_final

    # Calculate auxiliary quantities at final time
    # (These depend on R2, v2, etc., so we recalculate)

    # Shell mass
    if params['isCollapse'].value:
        mShell = params['shell_mass'].value
        mShell_dot = 0.0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(
            R2_final, params,
            return_mdot=True,
            rdot_arr=v2_final
        )
        mShell = _scalar(mShell)
        mShell_dot = _scalar(mShell_dot)

    params['shell_mass'].value = mShell
    params['shell_massDot'].value = mShell_dot

    # Get stellar feedback at final time
    [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] = get_currentSB99feedback(
        t_final, params
    )

    # Inner radius R1
    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-3 * R2_final,
            R2_final,
            args=([LWind, Eb_final, vWind, R2_final])
        )
    except ValueError:
        R1 = 0.001 * R2_final

    # Bubble pressure
    Pb = get_bubbleParams.bubble_E2P(
        Eb_final, R2_final, R1,
        params['gamma_adia'].value
    )

    params['R1'].value = R1
    params['Pb'].value = Pb

    # Calculate forces (for diagnostics/output)
    mCluster = params['mCluster'].value
    G = params['G'].value

    F_grav = G * mShell / (R2_final**2) * (mCluster + 0.5 * mShell)
    params['F_grav'].value = F_grav

    # Could calculate other forces here if needed...
    # params['F_ion_in'].value = ...
    # params['F_ion_out'].value = ...
    # params['F_ram'].value = ...

    logger.debug(f"Updated params: t={t_final:.6e}, R2={R2_final:.6e}, v2={v2_final:.6e}")


# =============================================================================
# Keep old function for backwards compatibility (but mark as deprecated)
# =============================================================================

def get_ODE_Edot(y, t, params):
    """
    DEPRECATED: Use get_ODE_Edot_pure() instead.

    This version modifies params (unsafe for scipy solvers).
    Kept for backwards compatibility only.
    """
    logger.warning("Using deprecated get_ODE_Edot() - switch to get_ODE_Edot_pure()")

    # Just call the pure version
    derivs = get_ODE_Edot_pure(y, t, params)

    # For backwards compatibility, also update params
    # (This is what causes the dictionary corruption issue!)
    R2, v2, Eb, T0 = y
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb

    return derivs
