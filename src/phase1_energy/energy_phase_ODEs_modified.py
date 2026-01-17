#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified ODE functions for TRINITY energy phase.

This module provides ODE functions for bubble expansion that are compatible
with scipy.integrate.solve_ivp, while preserving all physics from the original code.

Key design:
- ODE reads from params dict for physics calculations
- ODE does NOT mutate params during integration (no writing)
- After successful segment, call update_params_after_segment() to store ALL values
- Short segments ensure params values are approximately constant during integration

State vector: y = [R2, v2, Eb] (3 variables)
Note: T0 is NOT integrated - it's calculated externally via bubble_luminosity.

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.optimize
import scipy.interpolate
import logging
from typing import Tuple

# Import for physics calculations
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile

logger = logging.getLogger(__name__)

# Constants
FOUR_PI = 4.0 * np.pi


# =============================================================================
# R1 Cache for efficient brentq caching
# =============================================================================

class R1Cache:
    """
    Cache for inner radius R1 to avoid expensive brentq inside ODE.

    R1 changes slowly, so we:
    1. Compute R1 at segment start
    2. Use cached value during segment
    3. Optionally build interpolator for smoother variation
    """

    def __init__(self):
        self.t_values = []
        self.R1_values = []
        self._interp = None

    def update(self, t: float, R2: float, Eb: float, Lmech_total: float, v_mech_total: float) -> float:
        """
        Compute and cache R1 at given time.

        Parameters
        ----------
        t : float
            Current time [Myr]
        R2 : float
            Outer bubble radius [pc]
        Eb : float
            Bubble energy [internal units]
        Lmech_total : float
            Wind luminosity [internal units]
        v_mech_total : float
            Wind velocity [internal units]

        Returns
        -------
        R1 : float
            Inner bubble radius [pc]
        """
        try:
            R1 = scipy.optimize.brentq(
                get_bubbleParams.get_r1,
                1e-6 * R2, R2 * 0.999,
                args=([Lmech_total, Eb, v_mech_total, R2])
            )
        except ValueError:
            # Fallback if brentq fails (e.g., no root in interval)
            R1 = 0.01 * R2
            logger.warning(f"R1 brentq failed at t={t:.3e}, using R1=0.01*R2")

        self.t_values.append(t)
        self.R1_values.append(R1)

        # Build interpolator if we have enough points
        if len(self.t_values) >= 2:
            self._interp = scipy.interpolate.interp1d(
                self.t_values, self.R1_values,
                kind='linear', fill_value='extrapolate'
            )

        return R1

    def get(self, t: float) -> float:
        """
        Get R1 at time t via interpolation (or return last cached value).

        Parameters
        ----------
        t : float
            Time [Myr]

        Returns
        -------
        R1 : float
            Inner bubble radius [pc]
        """
        if self._interp is None:
            return self.R1_values[-1] if self.R1_values else 0.0
        return float(self._interp(t))

    def clear(self):
        """Clear the cache."""
        self.t_values = []
        self.R1_values = []
        self._interp = None


# =============================================================================
# Helper Functions
# =============================================================================

def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x


def _get_mass_from_profile(R2: float, v2: float, params) -> Tuple[float, float]:
    """
    Get shell mass and mass accretion rate using existing mass_profile module.

    This wraps the existing get_mass_profile() function which is already
    side-effect-free.

    Parameters
    ----------
    R2 : float
        Shell radius [pc]
    v2 : float
        Shell velocity [pc/Myr]
    params : dict
        Parameter dictionary with .value attributes

    Returns
    -------
    mShell : float
        Enclosed mass at R2 [Msun]
    mShell_dot : float
        Mass accretion rate at R2 [Msun/Myr]
    """
    # Handle collapse mode - freeze shell mass
    if params['isCollapse'].value == True:
        return params['shell_mass'].value, 0.0

    # Call existing mass profile function
    result = mass_profile.get_mass_profile(R2, params, return_mdot=True, rdot=v2)

    # Handle both tuple and single return
    if isinstance(result, tuple):
        mShell, mShell_dot = result
    else:
        mShell = result
        mShell_dot = 0.0

    # Ensure scalar outputs
    mShell = _scalar(mShell)
    mShell_dot = _scalar(mShell_dot)

    return float(mShell), float(mShell_dot)


def _get_mShell_dot_with_activation(mShell_dot_raw: float, R2: float, params) -> float:
    """
    Gradually activate mass accretion drag term as shell forms.

    This fixes the v²/r singularity at small R2 where the thin-shell
    approximation breaks down.

    Parameters
    ----------
    mShell_dot_raw : float
        Raw mass accretion rate [Msun/Myr]
    R2 : float
        Shell radius [pc]
    params : dict
        Parameter dictionary with .value attributes

    Returns
    -------
    mShell_dot : float
        Activated mass accretion rate [Msun/Myr]
    """
    # Activation radius: shell is well-formed above this
    rCore = params['rCore'].value
    R2_activate = rCore * 0.1  # 10% of core radius

    if R2 < R2_activate:
        # Linear ramp from 0 to 1 as R2 grows from 0 to R2_activate
        activation = R2 / R2_activate
        return mShell_dot_raw * activation
    else:
        return mShell_dot_raw


def get_press_ion(r, params):
    """
    Calculate pressure from photoionized part of cloud at radius r.

    This is the original get_press_ion() function from energy_phase_ODEs.py.

    Parameters
    ----------
    r : float or array
        Radius [pc]
    params : dict
        Parameter dictionary with .value attributes

    Returns
    -------
    P_ion : float
        Pressure of ionized gas [internal units]
    """
    # n_r: total number density of particles (H+, He++, electrons)
    try:
        r = np.array([r])
    except:
        pass

    n_r = density_profile.get_density_profile(r, params)

    P_ion = 2 * n_r * params['k_B'].value * params['TShell_ion'].value

    # Ensure scalar output
    if hasattr(P_ion, '__len__'):
        if len(P_ion) == 1:
            P_ion = P_ion[0]

    return P_ion


# =============================================================================
# Pure ODE Function
# =============================================================================

def get_ODE_Edot_pure(t: float, y: np.ndarray, params, R1_cached: float) -> np.ndarray:
    """
    ODE function for bubble expansion - reads params but does NOT mutate.

    This function is compatible with scipy.integrate.solve_ivp.
    It reads from the params dict but does NOT write to it during integration.
    All params mutations happen in update_params_after_segment() after success.

    State vector: y = [R2, v2, Eb]
    Note: T0 is NOT in state - it's updated between integration segments.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    params : dict
        Parameter dictionary with .value attributes (READ ONLY during ODE)
    R1_cached : float
        Pre-computed inner bubble radius [pc]

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt, dEb/dt]
    """
    R2, v2, Eb = y
    R2 = float(R2)
    v2 = float(v2)
    Eb = float(Eb)

    # Ensure positive values
    R2 = max(R2, 1e-10)
    Eb = max(Eb, 1e-10)

    # --- Pull frequently-used parameters once ---
    FABSi = params['shell_fAbsorbedIon'].value
    F_rad = params['shell_F_rad'].value
    mCluster = params['mCluster'].value
    L_bubble = params['bubble_LTotal'].value
    gamma = params['gamma_adia'].value
    tSF = params['tSF'].value
    G = params['G'].value
    Qi = params['Qi'].value
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value
    k_B = params['k_B'].value

    # --- Calculate shell mass and mass derivative ---
    mShell, mShell_dot_raw = _get_mass_from_profile(R2, v2, params)

    # Apply gradual activation to fix v²/r singularity
    mShell_dot = _get_mShell_dot_with_activation(mShell_dot_raw, R2, params)

    # Ensure positive shell mass
    mShell = max(mShell, 1e-10)

    # --- Gravity force (self + cluster) ---
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # --- Bubble pressure ---
    if params['current_phase'].value in ['momentum']:
        press_bubble = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
    else:
        # Pressure switch-on at early times
        dt_switchon = 1e-3  # Myr
        tmin = dt_switchon

        if (t > (tmin + tSF)):
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_cached, gamma)
        elif (t <= (tmin + tSF)):
            R1_tmp = (t - tSF) / tmin * R1_cached
            R1_tmp = max(R1_tmp, 1e-10)
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, gamma)

    # --- Calculate press_HII_in using density_profile (INSIDE ODE) ---
    # Calc inward pressure from photoionized gas outside the shell
    # (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        rShell = params['rShell'].value
        press_HII_in = get_press_ion(rShell, params)
    else:
        press_HII_in = 0.0

    # Add ambient pressure if shell is beyond cloud
    rCloud = params['rCloud'].value
    if params['rShell'].value >= rCloud:
        press_HII_in += params['PISM'].value * k_B

    # --- Calculate press_HII_out (INSIDE ODE) ---
    # Method 2: all HII region approximation
    if FABSi < 1:
        nR2 = params['nISM'].value
    else:
        caseB_alpha = params['caseB_alpha'].value
        nR2 = np.sqrt(Qi / caseB_alpha / R2**3 * 3 / 4 / np.pi)

    press_HII_out = 2 * nR2 * k_B * 3e4

    # --- Time derivatives ---
    rd = v2  # dR/dt = velocity

    # dv/dt = (F_pressure - momentum_drag - F_grav + F_rad) / mShell
    vd = (FOUR_PI * R2**2 * (press_bubble - press_HII_in + press_HII_out)
          - mShell_dot * v2 - F_grav + F_rad) / mShell

    # EarlyPhaseApproximation check
    if params['EarlyPhaseApproximation'].value == True:
        vd = -1e8

    # dE/dt = Lmech_total - L_bubble - P*dV/dt (energy balance)
    # Note: L_leak = 0 for now (no fragmentation)
    L_leak = 0.0
    Ed = (Lmech_total - L_bubble) - (FOUR_PI * R2**2 * press_bubble) * v2 - L_leak

    return np.array([rd, vd, Ed])


# =============================================================================
# Update Params After Successful Segment
# =============================================================================

def update_params_after_segment(t: float, R2: float, v2: float, Eb: float,
                                 params, R1_cached: float):
    """
    Update params dict with ALL computed values after successful segment.

    This matches what the original ODE did on every call, but we only
    do it after successful integration to avoid stale/duplicate values
    during solver backtracking.

    TRINITY uses the params dict as a timestep snapshot - all computed
    values must be stored here for the next segment.

    Parameters
    ----------
    t : float
        Current time [Myr]
    R2 : float
        Shell radius [pc]
    v2 : float
        Shell velocity [pc/Myr]
    Eb : float
        Bubble energy [internal units]
    params : dict
        Parameter dictionary with .value attributes
    R1_cached : float
        Cached inner bubble radius [pc]
    """
    # --- State variables ---
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb

    # --- Shell mass ---
    if params['isCollapse'].value == True:
        mShell = params['shell_mass'].value
        mShell_dot = 0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(
            R2, params, return_mdot=True, rdot=v2
        )
        mShell = _scalar(mShell)
        mShell_dot = _scalar(mShell_dot)

    params['shell_mass'].value = mShell
    params['shell_massDot'].value = mShell_dot

    # --- Bubble pressure and R1 ---
    gamma = params['gamma_adia'].value
    press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_cached, gamma)
    params['Pb'].value = press_bubble
    params['R1'].value = R1_cached

    # --- HII pressures (calculated using density_profile) ---
    FABSi = params['shell_fAbsorbedIon'].value
    k_B = params['k_B'].value

    if FABSi < 1.0:
        press_HII_in = get_press_ion(params['rShell'].value, params)
    else:
        press_HII_in = 0.0

    if params['rShell'].value >= params['rCloud'].value:
        press_HII_in += params['PISM'].value * k_B

    Qi = params['Qi'].value
    if FABSi < 1:
        nR2 = params['nISM'].value
    else:
        nR2 = np.sqrt(Qi / params['caseB_alpha'].value / R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * k_B * 3e4

    # --- Gravity ---
    G = params['G'].value
    mCluster = params['mCluster'].value
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)
    F_rad = params['shell_F_rad'].value

    # --- Store forces for diagnostics ---
    params['F_grav'].value = F_grav
    params['F_ion_in'].value = press_HII_in * FOUR_PI * R2**2
    params['F_ion_out'].value = press_HII_out * FOUR_PI * R2**2
    params['F_ram'].value = press_bubble * FOUR_PI * R2**2
    params['F_rad'].value = F_rad


# =============================================================================
# Event Functions for solve_ivp (optional)
# =============================================================================

def radius_exceeds_cloud_event(t: float, y: np.ndarray, params) -> float:
    """
    Event function: triggers when shell reaches cloud boundary.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    params : dict
        Parameter dictionary

    Returns
    -------
    value : float
        Positive when R2 < rCloud, negative when R2 > rCloud
    """
    R2, v2, Eb = y
    rCloud = params['rCloud'].value
    return rCloud - R2

radius_exceeds_cloud_event.terminal = False
radius_exceeds_cloud_event.direction = -1


# =============================================================================
# Backward Compatibility Wrapper
# =============================================================================

def get_ODE_Edot_wrapper(y, t, params):
    """
    Wrapper that provides interface compatible with old code (odeint signature).

    DEPRECATED: Use get_ODE_Edot_pure with solve_ivp instead.

    Parameters
    ----------
    y : list
        State vector [R2, v2, Eb, T0]
    t : float
        Time [Myr]
    params : dict
        Parameter dictionary

    Returns
    -------
    derivs : list
        Derivatives [rd, vd, Ed, Td]
    """
    import warnings
    warnings.warn(
        "get_ODE_Edot_wrapper is deprecated. Use get_ODE_Edot_pure with solve_ivp.",
        DeprecationWarning,
        stacklevel=2
    )

    R2, v2, Eb, T0 = y

    # Get R1 for this state
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value

    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-6 * R2, R2 * 0.999,
            args=([Lmech_total, Eb, v_mech_total, R2])
        )
    except:
        R1 = 0.01 * R2

    # Call the ODE function
    y_arr = np.array([R2, v2, Eb])
    dydt = get_ODE_Edot_pure(t, y_arr, params, R1)

    # Return with T0 derivative = 0 (T0 is external)
    return [dydt[0], dydt[1], dydt[2], 0]
