#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified transition phase runner for TRINITY.

This module implements the transition phase between energy-driven and
momentum-driven expansion, using scipy.integrate.solve_ivp.

Key features:
- Energy decays on sound-crossing timescale: dE/dt = -Eb / t_sound
- Uses ODE function that reads params but does NOT mutate during integration
- update_params_after_segment() called after each successful segment
- No T0 evolution (dT/dt = 0)

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import logging
from typing import Dict, Tuple
from dataclasses import dataclass

import src.phase_general.phase_ODEs as phase_ODEs
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
import src._functions.unit_conversions as cvt
from src.sb99.update_feedback import get_currentSB99feedback

# Import ODE functions and helpers
from src.phase1_energy.energy_phase_ODEs_modified import (
    get_ODE_Edot_pure,
    update_params_after_segment,
    R1Cache,
    _get_mass_from_profile,
    _scalar,
)
import src.bubble_structure.get_bubbleParams as get_bubbleParams

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DT_SEGMENT = 1e-4  # Myr
MAX_SEGMENTS = 5000
ENERGY_FLOOR = 1e3  # Minimum energy before transition to momentum phase


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class TransitionPhaseResults:
    """Container for transition phase results."""
    t: np.ndarray
    R2: np.ndarray
    v2: np.ndarray
    Eb: np.ndarray
    termination_reason: str
    final_time: float


# =============================================================================
# Pure ODE for Transition Phase
# =============================================================================

def get_ODE_transition_pure(t: float, y: np.ndarray, params, R1_cached: float,
                            c_sound: float) -> np.ndarray:
    """
    ODE function for transition phase.

    Energy decays on sound-crossing timescale.
    Reads params but does NOT mutate during integration.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    params : dict
        Parameter dictionary (READ ONLY during ODE)
    R1_cached : float
        Cached inner bubble radius [pc]
    c_sound : float
        Sound speed [pc/Myr]

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt, dEb/dt]
    """
    R2, v2, Eb = y

    # Get rd, vd from energy ODE
    dydt_energy = get_ODE_Edot_pure(t, y, params, R1_cached)
    rd = dydt_energy[0]  # = v2
    vd = dydt_energy[1]  # acceleration

    # Energy decay: dE/dt = -Eb / t_sound_crossing
    # t_sound_crossing = R2 / c_sound
    if c_sound > 0 and R2 > 0:
        t_sound_crossing = R2 / c_sound
        Ed = -Eb / t_sound_crossing
    else:
        Ed = 0.0

    return np.array([rd, vd, Ed])


# =============================================================================
# Main Function
# =============================================================================

def run_phase_transition(params) -> TransitionPhaseResults:
    """
    Run the transition phase using solve_ivp.

    This phase bridges energy-driven and momentum-driven expansion.
    Energy decays on the sound-crossing timescale until it reaches
    a floor value, then momentum phase begins.

    Parameters
    ----------
    params : ParameterDict
        Parameter dictionary

    Returns
    -------
    results : TransitionPhaseResults
        Results container
    """
    # =============================================================================
    # Initialization
    # =============================================================================

    # Set initial v2 from alpha
    params['v2'].value = params['cool_alpha'].value * params['R2'].value / params['t_now'].value

    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # Initialize state (no T0 in state vector for transition)
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value

    # Pre-allocate results
    t_results = [tmin]
    R2_results = [R2]
    v2_results = [v2]
    Eb_results = [Eb]

    # R1 cache
    r1_cache = R1Cache()

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
        params['Eb'].value = Eb
        params['T0'].value = T0

        # ---------------------------------------------------------------------
        # Get feedback and shell structure
        # ---------------------------------------------------------------------
        feedback = get_currentSB99feedback(t_now, params)
        L_mech_total = params['L_mech_total'].value
        v_mech_total = params['v_mech_total'].value

        shell_structure.shell_structure(params)

        # Get sound speed
        c_sound = params['c_sound'].value

        # ---------------------------------------------------------------------
        # Get R1
        # ---------------------------------------------------------------------
        try:
            R1 = scipy.optimize.brentq(
                get_bubbleParams.get_r1,
                1e-3 * R2, R2,
                args=([L_mech_total, Eb, v_mech_total, R2])
            )
        except:
            R1 = 0.01 * R2

        r1_cache.update(t_now, R2, Eb, L_mech_total, v_mech_total)
        params['R1'].value = R1

        # ---------------------------------------------------------------------
        # Integrate segment
        # ---------------------------------------------------------------------
        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2, Eb])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_transition_pure(t, y, params, R1, c_sound),
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
        Eb = float(sol.y[2, -1])
        t_now = float(sol.t[-1])

        # Store results
        t_results.append(t_now)
        R2_results.append(R2)
        v2_results.append(v2)
        Eb_results.append(Eb)

        # ---------------------------------------------------------------------
        # Update params after successful segment
        # ---------------------------------------------------------------------
        update_params_after_segment(t_now, R2, v2, Eb, params, R1)

        # ---------------------------------------------------------------------
        # Update history arrays
        # ---------------------------------------------------------------------
        params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
        params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
        params['array_R1'].value = np.concatenate([params['array_R1'].value, [R1]])
        params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
        params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])

        mShell = params['shell_mass'].value
        params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])

        params.save_snapshot()

        # ---------------------------------------------------------------------
        # Check termination: energy floor reached
        # ---------------------------------------------------------------------
        if Eb < ENERGY_FLOOR:
            termination_reason = "energy_floor"
            logger.info(f"Energy dropped below floor ({ENERGY_FLOOR}), transitioning to momentum")
            break

        # Check collapse
        if v2 < 0 and R2 < params['R2'].value:
            params['isCollapse'].value = True

        # Other termination conditions
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

    # =============================================================================
    # Build results
    # =============================================================================

    if termination_reason is None:
        termination_reason = "max_segments" if segment_count >= MAX_SEGMENTS else "unknown"

    logger.info(f"Transition phase completed: {termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr, Final Eb: {Eb:.6e}")

    return TransitionPhaseResults(
        t=np.array(t_results),
        R2=np.array(R2_results),
        v2=np.array(v2_results),
        Eb=np.array(Eb_results),
        termination_reason=termination_reason,
        final_time=t_now,
    )
