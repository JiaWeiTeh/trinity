#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified energy implicit phase runner for TRINITY.

This module continues the energy phase with real-time beta/delta calculations,
using scipy.integrate.solve_ivp instead of manual Euler stepping.

Key improvements:
1. Uses pure ODE functions (no dictionary mutations during integration)
2. scipy.integrate.solve_ivp(LSODA) for adaptive integration
3. Segment-based integration with beta/delta updates between segments
4. Pre-allocated result arrays

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import src.phase_general.phase_ODEs as phase_ODEs
import src.cloud_properties.mass_profile as mass_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.phase1b_energy_implicit.get_betadelta as get_betadelta
import src._functions.unit_conversions as cvt
import src.cooling.non_CIE.read_cloudy as non_CIE
import src.shell_structure.shell_structure as shell_structure
import src._functions.operations as operations
from src.sb99.update_feedback import get_currentSB99feedback

# Import pure ODE functions
from src.phase1_energy.energy_phase_ODEs_modified import (
    StaticODEParams,
    get_ODE_Edot_pure,
    extract_static_params,
    R1Cache,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

COOLING_UPDATE_INTERVAL = 5e-3  # Myr - recalculate cooling
DT_SEGMENT = 1e-4  # Myr - segment duration for beta/delta updates
MAX_SEGMENTS = 5000


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class ImplicitPhaseResults:
    """Container for implicit phase results."""
    t: np.ndarray
    R2: np.ndarray
    v2: np.ndarray
    Eb: np.ndarray
    T0: np.ndarray
    beta: np.ndarray
    delta: np.ndarray
    termination_reason: str
    final_time: float


# =============================================================================
# Pure ODE for Implicit Phase
# =============================================================================

def get_ODE_implicit_pure(t: float, y: np.ndarray, static: StaticODEParams,
                          Ed_from_beta: float, Td_from_delta: float) -> np.ndarray:
    """
    Pure ODE function for implicit phase.

    Uses rd, vd from standard energy ODE but Ed, Td from beta/delta.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb, T0]
    static : StaticODEParams
        Immutable parameters
    Ed_from_beta : float
        Energy derivative from beta calculation
    Td_from_delta : float
        Temperature derivative from delta calculation

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt, dEb/dt, dT0/dt]
    """
    R2, v2, Eb, T0 = y

    # Get rd, vd from energy ODE (y without T0)
    y_energy = np.array([R2, v2, Eb])
    dydt_energy = get_ODE_Edot_pure(t, y_energy, static)

    rd = dydt_energy[0]  # = v2
    vd = dydt_energy[1]  # acceleration from pressure balance

    # Use Ed and Td from beta/delta calculations (computed outside ODE)
    return np.array([rd, vd, Ed_from_beta, Td_from_delta])


# =============================================================================
# Event Functions
# =============================================================================

def cooling_balance_event(t: float, y: np.ndarray, Lgain: float, Lloss: float) -> float:
    """Event: Lcool approaches Lgain (energy loss dominates)."""
    if Lgain <= 0:
        return 1.0  # No event if no gain
    ratio = (Lgain - Lloss) / Lgain
    return ratio - 0.05  # Trigger when ratio < 5%


def velocity_sign_event(t: float, y: np.ndarray) -> float:
    """Event: velocity changes sign (collapse onset)."""
    R2, v2, Eb, T0 = y
    return v2


velocity_sign_event.direction = -1


# =============================================================================
# Main Function
# =============================================================================

def run_phase_energy(params) -> ImplicitPhaseResults:
    """
    Run the implicit energy phase using solve_ivp.

    This phase solves for real-time beta and delta values (cooling parameters)
    that were approximated in the initial energy phase.

    Parameters
    ----------
    params : ParameterDict
        Parameter dictionary

    Returns
    -------
    results : ImplicitPhaseResults
        Results container with arrays and termination info
    """
    # =============================================================================
    # Initialization
    # =============================================================================

    # Set initial v2 from alpha
    params['v2'].value = params['cool_alpha'].value * params['R2'].value / params['t_now'].value

    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # Initialize state
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value

    # Pre-allocate results (estimate based on time range)
    n_estimate = min(int(200 * np.log10(tmax/tmin)), MAX_SEGMENTS)
    t_results = []
    R2_results = []
    v2_results = []
    Eb_results = []
    T0_results = []
    beta_results = []
    delta_results = []

    # Store initial values
    t_results.append(tmin)
    R2_results.append(R2)
    v2_results.append(v2)
    Eb_results.append(Eb)
    T0_results.append(T0)
    beta_results.append(params['cool_beta'].value)
    delta_results.append(params['cool_delta'].value)

    # R1 cache
    r1_cache = R1Cache()

    t_now = tmin
    segment_count = 0
    termination_reason = None

    # =============================================================================
    # Main loop (segment-based)
    # =============================================================================

    while t_now < tmax and segment_count < MAX_SEGMENTS:
        segment_count += 1

        # ---------------------------------------------------------------------
        # Update cooling structure periodically
        # ---------------------------------------------------------------------
        if abs(params['t_previousCoolingUpdate'].value - t_now) > COOLING_UPDATE_INTERVAL:
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
            params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            params['cStruc_heating_nonCIE'].value = heating_nonCIE
            params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            params['t_previousCoolingUpdate'].value = t_now

        # ---------------------------------------------------------------------
        # Update params with current state
        # ---------------------------------------------------------------------
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = Eb
        params['T0'].value = T0
        params['cool_alpha'].value = t_now / R2 * v2

        # ---------------------------------------------------------------------
        # Get feedback and shell structure
        # ---------------------------------------------------------------------
        feedback = get_currentSB99feedback(t_now, params)
        LWind = params['LWind'].value
        vWind = params['vWind'].value

        shell_structure.shell_structure(params)

        # ---------------------------------------------------------------------
        # Calculate beta and delta
        # ---------------------------------------------------------------------
        (beta, delta), result_params = get_betadelta.get_beta_delta_wrapper(
            params['cool_beta'].value,
            params['cool_delta'].value,
            params
        )

        result_params['cool_beta'].value = beta
        result_params['cool_delta'].value = delta
        result_params['c_sound'].value = operations.get_soundspeed(
            result_params['bubble_Tavg'].value, result_params
        )

        # ---------------------------------------------------------------------
        # Get R1 and Pb
        # ---------------------------------------------------------------------
        try:
            R1 = scipy.optimize.brentq(
                get_bubbleParams.get_r1,
                1e-3 * R2, R2,
                args=([LWind, Eb, vWind, R2])
            )
        except ValueError:
            R1 = 0.01 * R2

        r1_cache.update(t_now, R2, Eb, LWind, vWind)

        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        result_params['R1'].value = R1
        result_params['Pb'].value = Pb

        # ---------------------------------------------------------------------
        # Convert beta/delta to Ed, Td
        # ---------------------------------------------------------------------
        Ed = get_bubbleParams.beta2Edot(result_params)
        Td = get_bubbleParams.delta2dTdt(t_now, T0, delta)

        # ---------------------------------------------------------------------
        # Build static params and integrate segment
        # ---------------------------------------------------------------------
        static = extract_static_params(result_params, R1_cached=R1)

        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2, Eb, T0])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_implicit_pure(t, y, static, Ed, Td),
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
        T0 = float(sol.y[3, -1])
        t_now = float(sol.t[-1])

        # Store results
        t_results.append(t_now)
        R2_results.append(R2)
        v2_results.append(v2)
        Eb_results.append(Eb)
        T0_results.append(T0)
        beta_results.append(beta)
        delta_results.append(delta)

        # ---------------------------------------------------------------------
        # Update history arrays
        # ---------------------------------------------------------------------
        params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
        params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
        params['array_R1'].value = np.concatenate([params['array_R1'].value, [R1]])
        params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
        params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])

        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params, return_mdot=True, rdot=v2)
        if hasattr(mShell, '__len__') and len(mShell) == 1:
            mShell = mShell[0]
        params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])

        # Save snapshot
        result_params.save_snapshot()

        # ---------------------------------------------------------------------
        # Check termination conditions
        # ---------------------------------------------------------------------
        Lgain = result_params.get('bubble_Lgain', {})
        Lloss = result_params.get('bubble_Lloss', {})
        if hasattr(Lgain, 'value'):
            Lgain = Lgain.value
        if hasattr(Lloss, 'value'):
            Lloss = Lloss.value

        if Lgain > 0 and (Lgain - Lloss) / Lgain < 0.05:
            termination_reason = "cooling_balance"
            break

        if v2 < 0 and R2 < params['R2'].value:
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

    # =============================================================================
    # Build results
    # =============================================================================

    if termination_reason is None:
        termination_reason = "max_segments" if segment_count >= MAX_SEGMENTS else "unknown"

    logger.info(f"Implicit phase completed: {termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr, Segments: {segment_count}")

    return ImplicitPhaseResults(
        t=np.array(t_results),
        R2=np.array(R2_results),
        v2=np.array(v2_results),
        Eb=np.array(Eb_results),
        T0=np.array(T0_results),
        beta=np.array(beta_results),
        delta=np.array(delta_results),
        termination_reason=termination_reason,
        final_time=t_now,
    )
