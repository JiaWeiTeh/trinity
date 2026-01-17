#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified momentum phase runner for TRINITY.

This module implements the momentum-driven phase (Phase 2) using
scipy.integrate.solve_ivp.

Key features:
- Bubble energy Eb = 0 (energy-driven terms negligible)
- Only rd, vd are evolved; Ed = Td = 0
- Uses ODE function that reads params but does NOT mutate during integration
- update_params_after_segment() called after each successful segment

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import logging
from typing import Dict
from dataclasses import dataclass

from src.phase_general import phase_ODEs
import src._functions.unit_conversions as cvt
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
from src.sb99.update_feedback import get_currentSB99feedback

# Import ODE helpers from energy phase
from src.phase1_energy.energy_phase_ODEs_modified import (
    R1Cache,
    _get_mass_from_profile,
    _get_mShell_dot_with_activation,
    _scalar,
    get_press_ion,
    update_params_after_segment,
)
import src.bubble_structure.get_bubbleParams as get_bubbleParams

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DT_SEGMENT = 1e-3  # Myr - larger segments OK in momentum phase
MAX_SEGMENTS = 10000
FOUR_PI = 4.0 * np.pi


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class MomentumPhaseResults:
    """Container for momentum phase results."""
    t: np.ndarray
    R2: np.ndarray
    v2: np.ndarray
    termination_reason: str
    final_time: float


# =============================================================================
# Pure ODE for Momentum Phase
# =============================================================================

def get_ODE_momentum_pure(t: float, y: np.ndarray, params) -> np.ndarray:
    """
    ODE function for momentum phase.

    In momentum phase, Eb = 0 so bubble pressure is ram pressure only.
    Reads params but does NOT mutate during integration.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2]
    params : dict
        Parameter dictionary (READ ONLY during ODE)

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt]
    """
    R2, v2 = y
    R2 = max(R2, 1e-10)

    # Get parameters
    G = params['G'].value
    mCluster = params['mCluster'].value
    LWind = params['LWind'].value
    vWind = params['vWind'].value
    k_B = params['k_B'].value
    FABSi = params['shell_fAbsorbedIon'].value
    F_rad = params['shell_F_rad'].value
    Qi = params['Qi'].value

    # Calculate shell mass
    mShell, mShell_dot_raw = _get_mass_from_profile(R2, v2, params)

    # Apply activation
    mShell_dot = _get_mShell_dot_with_activation(mShell_dot_raw, R2, params)

    mShell = max(mShell, 1e-10)

    # Gravity
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # Ram pressure (momentum phase - no thermal pressure)
    press_ram = get_bubbleParams.pRam(R2, LWind, vWind)

    # HII pressures (calculated using density_profile)
    if FABSi < 1.0:
        rShell = params['rShell'].value
        press_HII_in = get_press_ion(rShell, params)
    else:
        press_HII_in = 0.0

    # Add ambient pressure if shell is beyond cloud
    rCloud = params['rCloud'].value
    if params['rShell'].value >= rCloud:
        press_HII_in += params['PISM'].value * k_B

    # Calculate press_HII_out
    if FABSi < 1:
        nR2 = params['nISM'].value
    else:
        caseB_alpha = params['caseB_alpha'].value
        nR2 = np.sqrt(Qi / caseB_alpha / R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * k_B * 3e4

    # Net pressure force
    F_pressure = FOUR_PI * R2**2 * (press_ram - press_HII_in + press_HII_out)

    # Derivatives
    rd = v2
    vd = (F_pressure - mShell_dot * v2 - F_grav + F_rad) / mShell

    return np.array([rd, vd])


def update_params_momentum(t: float, R2: float, v2: float, params):
    """
    Update params dict after successful momentum phase segment.

    Similar to update_params_after_segment but for momentum phase
    where Eb = 0.

    Parameters
    ----------
    t : float
        Current time [Myr]
    R2 : float
        Shell radius [pc]
    v2 : float
        Shell velocity [pc/Myr]
    params : dict
        Parameter dictionary with .value attributes
    """
    # --- State variables ---
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = 0.0

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

    # --- Get parameters ---
    G = params['G'].value
    mCluster = params['mCluster'].value
    LWind = params['LWind'].value
    vWind = params['vWind'].value
    k_B = params['k_B'].value
    FABSi = params['shell_fAbsorbedIon'].value
    F_rad = params['shell_F_rad'].value
    Qi = params['Qi'].value

    # --- Ram pressure ---
    press_ram = get_bubbleParams.pRam(R2, LWind, vWind)
    params['Pb'].value = press_ram

    # --- HII pressures ---
    if FABSi < 1.0:
        press_HII_in = get_press_ion(params['rShell'].value, params)
    else:
        press_HII_in = 0.0

    if params['rShell'].value >= params['rCloud'].value:
        press_HII_in += params['PISM'].value * k_B

    if FABSi < 1:
        nR2 = params['nISM'].value
    else:
        nR2 = np.sqrt(Qi / params['caseB_alpha'].value / R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * k_B * 3e4

    # --- Gravity ---
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # --- Store forces for diagnostics ---
    params['F_grav'].value = F_grav
    params['F_ion_in'].value = press_HII_in * FOUR_PI * R2**2
    params['F_ion_out'].value = press_HII_out * FOUR_PI * R2**2
    params['F_ram'].value = press_ram * FOUR_PI * R2**2
    params['F_rad'].value = F_rad


# =============================================================================
# Main Function
# =============================================================================

def run_phase_momentum(params) -> MomentumPhaseResults:
    """
    Run the momentum-driven phase using solve_ivp.

    In this phase, thermal pressure is negligible (Eb ≈ 0).
    Expansion is driven by ram pressure of the wind.

    Parameters
    ----------
    params : ParameterDict
        Parameter dictionary

    Returns
    -------
    results : MomentumPhaseResults
        Results container
    """
    # =============================================================================
    # Initialization
    # =============================================================================

    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # Initialize state (Eb = 0 in momentum phase)
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = 0.0
    T0 = params['T0'].value

    params['Eb'].value = 0.0

    # Pre-allocate results
    t_results = [tmin]
    R2_results = [R2]
    v2_results = [v2]

    t_now = tmin
    segment_count = 0
    termination_reason = None

    # =============================================================================
    # Main loop
    # =============================================================================

    while t_now < tmax and segment_count < MAX_SEGMENTS:
        segment_count += 1

        # ---------------------------------------------------------------------
        # Update params
        # ---------------------------------------------------------------------
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = 0.0
        params['T0'].value = T0

        # ---------------------------------------------------------------------
        # Get feedback and shell structure
        # ---------------------------------------------------------------------
        feedback = get_currentSB99feedback(t_now, params)
        LWind = params['LWind'].value
        vWind = params['vWind'].value

        shell_structure.shell_structure(params)

        # Set R1 = R2 (no inner shock in momentum phase)
        params['R1'].value = R2

        # ---------------------------------------------------------------------
        # Integrate segment
        # ---------------------------------------------------------------------
        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_momentum_pure(t, y, params),
                t_span=t_span,
                y0=y0,
                method='LSODA',
                rtol=1e-6,
                atol=1e-9,
            )
        except Exception as e:
            logger.error(f"solve_ivp failed at t={t_now:.6e}: {e}")
            termination_reason = f"solver_error: {e}"
            break

        if not sol.success or len(sol.t) == 0:
            termination_reason = f"solver_failed: {sol.message}"
            break

        # ---------------------------------------------------------------------
        # Extract final state
        # ---------------------------------------------------------------------
        R2 = float(sol.y[0, -1])
        v2 = float(sol.y[1, -1])
        t_now = float(sol.t[-1])

        # Store results
        t_results.append(t_now)
        R2_results.append(R2)
        v2_results.append(v2)

        # ---------------------------------------------------------------------
        # Update params after successful segment
        # ---------------------------------------------------------------------
        update_params_momentum(t_now, R2, v2, params)

        # ---------------------------------------------------------------------
        # Update history arrays
        # ---------------------------------------------------------------------
        params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
        params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
        params['array_R1'].value = np.concatenate([params['array_R1'].value, [R2]])  # R1 = R2
        params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
        params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])

        mShell = params['shell_mass'].value
        params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])

        params.save_snapshot()

        # ---------------------------------------------------------------------
        # Check termination conditions
        # ---------------------------------------------------------------------
        # Check collapse
        if v2 < 0:
            params['isCollapse'].value = True

        if t_now > tmax:
            termination_reason = "reached_tmax"
            params['SimulationEndReason'].value = 'Stopping time reached'
            params['EndSimulationDirectly'].value = True
            break

        if params.get('isCollapse', {}).value == True and R2 < params['coll_r'].value:
            termination_reason = "small_radius"
            params['SimulationEndReason'].value = 'Small radius reached'
            params['EndSimulationDirectly'].value = True
            break

        if R2 > params['stop_r'].value:
            termination_reason = "large_radius"
            params['SimulationEndReason'].value = 'Large radius reached'
            params['EndSimulationDirectly'].value = True
            break

        # Dissolution check
        if params['shell_nMax'].value < params['stop_n_diss'].value:
            params['isDissolved'].value = True
            termination_reason = "dissolved"
            params['SimulationEndReason'].value = 'Shell dissolved'
            params['EndSimulationDirectly'].value = True
            break

        # Cloud boundary check
        if params.get('expansionBeyondCloud', True) == False:
            if R2 > params['rCloud'].value:
                termination_reason = "cloud_boundary"
                params['SimulationEndReason'].value = 'Bubble radius larger than cloud'
                params['EndSimulationDirectly'].value = True
                break

    # =============================================================================
    # Build results
    # =============================================================================

    if termination_reason is None:
        termination_reason = "max_segments" if segment_count >= MAX_SEGMENTS else "unknown"

    logger.info(f"Momentum phase completed: {termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr, Final R2: {R2:.6e} pc")

    return MomentumPhaseResults(
        t=np.array(t_results),
        R2=np.array(R2_results),
        v2=np.array(v2_results),
        termination_reason=termination_reason,
        final_time=t_now,
    )
