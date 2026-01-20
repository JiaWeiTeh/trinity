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
import src._functions.unit_conversions as cvt
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src.sb99.update_feedback import get_currentSB99feedback
from src._input.dictionary import updateDict

# Import pure/modified functions
from src.phase1_energy.energy_phase_ODEs_modified import (
    ODESnapshot,
    get_ODE_Edot_pure,
    create_ODE_snapshot,
    ODEResult,
    compute_derived_quantities,
)
from src.phase1b_energy_implicit.get_betadelta_modified import (
    solve_betadelta_pure,
    beta2Edot_pure,
    delta2dTdt_pure,
    compute_R1_Pb,
    BetaDeltaResult,
)
from src.shell_structure.shell_structure_modified import (
    shell_structure_pure,
    ShellProperties,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

COOLING_UPDATE_INTERVAL = 5e-3  # Myr - recalculate cooling
DT_SEGMENT = 1e-3  # Myr - segment duration for beta/delta updates
MAX_SEGMENTS = 5000
FOUR_PI = 4.0 * np.pi


# =============================================================================
# Force Properties Dataclass
# =============================================================================

@dataclass
class ForceProperties:
    """Container for force calculations (pure function output)."""
    F_grav: float       # Gravitational force
    F_ion_in: float     # Inward ionization pressure force
    F_ion_out: float    # Outward ionization pressure force
    F_ram: float        # Ram pressure force (from bubble pressure)
    F_rad: float        # Radiation pressure force


def compute_forces_pure(
    R2: float,
    mShell: float,
    Pb: float,
    shell_props: ShellProperties,
    params,
) -> ForceProperties:
    """
    Compute all force components without mutating params.

    Parameters
    ----------
    R2 : float
        Shell outer radius
    mShell : float
        Shell mass
    Pb : float
        Bubble pressure
    shell_props : ShellProperties
        Shell structure properties
    params : dict-like
        Parameter dictionary (read-only)

    Returns
    -------
    ForceProperties
        Dataclass with all force values
    """
    # Gravitational force: F = G * mShell / R2^2 * (mCluster + mShell/2)
    G = params['G'].value
    mCluster = params['mCluster'].value
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # Ionization pressure forces
    k_B = params['k_B'].value
    TShell_ion = params['TShell_ion'].value
    rCloud = params['rCloud'].value
    rShell = shell_props.rShell
    nISM = params['nISM'].value
    PISM = params.get('PISM', None)
    if PISM is not None and hasattr(PISM, 'value'):
        PISM = PISM.value
    else:
        PISM = 0.0

    # Inward pressure from photoionized gas outside shell
    FABSi = shell_props.shell_fAbsorbedIon
    if FABSi < 1.0:
        # Get density profile at shell radius for ionized region
        from src.cloud_properties import density_profile
        try:
            n_r = density_profile.get_density_profile(np.array([rShell]), params)
            if hasattr(n_r, '__len__') and len(n_r) == 1:
                n_r = n_r[0]
            press_HII_in = n_r * k_B * TShell_ion
        except Exception:
            press_HII_in = 0.0
    else:
        press_HII_in = 0.0

    # Add ISM pressure if shell extends beyond cloud
    if rShell >= rCloud:
        press_HII_in += PISM * k_B

    # Outward ionization pressure
    if FABSi < 1.0:
        nR2 = nISM
    else:
        # Strömgren sphere density estimate
        Qi = params['Qi'].value
        caseB_alpha = params['caseB_alpha'].value
        nR2 = np.sqrt(Qi / caseB_alpha / (R2**3) * 3 / FOUR_PI)
    press_HII_out = 2.0 * nR2 * k_B * 3e4  # 3e4 K is typical HII region temperature

    F_ion_in = press_HII_in * FOUR_PI * R2**2
    F_ion_out = press_HII_out * FOUR_PI * R2**2

    # Ram pressure force (from bubble pressure)
    F_ram = Pb * FOUR_PI * R2**2

    # Radiation pressure force (from shell structure)
    F_rad = shell_props.shell_F_rad

    return ForceProperties(
        F_grav=F_grav,
        F_ion_in=F_ion_in,
        F_ion_out=F_ion_out,
        F_ram=F_ram,
        F_rad=F_rad,
    )


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

def get_ODE_implicit_pure(t: float, y: np.ndarray, snapshot: ODESnapshot,
                          params_for_feedback,
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
    snapshot : ODESnapshot
        Frozen snapshot of parameters
    params_for_feedback : DescribedDict
        Original params dict for feedback interpolation
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
    y_energy = [R2, v2, Eb]
    dydt_energy = get_ODE_Edot_pure(t, y_energy, snapshot, params_for_feedback)

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

    t_now = tmin
    segment_count = 0
    termination_reason = None

    # Track previous R2 for collapse detection
    R2_prev = R2

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
        updateDict(params, feedback)

        # Extract feedback values we'll need later

        # Calculate shell structure using pure function
        shell_props = shell_structure_pure(params)
        updateDict(params, shell_props)

        # ---------------------------------------------------------------------
        # Calculate beta and delta using pure function
        # ---------------------------------------------------------------------
        betadelta_result = solve_betadelta_pure(
            params['cool_beta'].value,
            params['cool_delta'].value,
            params
        )

        beta = betadelta_result.beta
        delta = betadelta_result.delta

        # Update params with new beta/delta
        params['cool_beta'].value = beta
        params['cool_delta'].value = delta

        # Get bubble properties from result
        bubble_props = betadelta_result.bubble_properties

        # Update params with all bubble properties from the pure function result
        # This is critical: without this, bubble_mass, bubble arrays, etc. remain stale
        if bubble_props is not None:
            updateDict(params, bubble_props)

        # Save residual diagnostics to dictionary (after ODE, not during)
        params['residual_deltaT'].value = betadelta_result.T_residual
        params['residual_betaEdot'].value = betadelta_result.Edot_residual
        if betadelta_result.Edot_from_beta is not None:
            params['residual_Edot1_guess'].value = betadelta_result.Edot_from_beta
        if betadelta_result.Edot_from_balance is not None:
            params['residual_Edot2_guess'].value = betadelta_result.Edot_from_balance
        if betadelta_result.T_bubble is not None:
            params['residual_T1_guess'].value = betadelta_result.T_bubble
        if betadelta_result.T0 is not None:
            params['residual_T2_guess'].value = betadelta_result.T0

        # ---------------------------------------------------------------------
        # Get R1 and Pb
        # ---------------------------------------------------------------------
        gamma_adia = params['gamma_adia'].value
        R1, Pb = compute_R1_Pb(R2, Eb, feedback.Lmech_total, feedback.v_mech_total, gamma_adia)

        params['R1'].value = R1
        params['Pb'].value = Pb

        # Get sound speed from bubble average temperature
        # (bubble_Tavg is now in params via updateDict above)
        bubble_Tavg = params['bubble_Tavg'].value if params['bubble_Tavg'].value else 1e6
        params['c_sound'].value = operations.get_soundspeed(bubble_Tavg, params)

        # ---------------------------------------------------------------------
        # Convert beta/delta to Ed, Td using pure functions
        # ---------------------------------------------------------------------

        Ed = beta2Edot_pure(beta, Pb, t_now, R1, R2, v2, Eb, feedback.pdot_total, feedback.pdotdot_total)
        Td = delta2dTdt_pure(t_now, T0, delta)

        # ---------------------------------------------------------------------
        # Build snapshot and integrate segment
        # ---------------------------------------------------------------------
        snapshot = create_ODE_snapshot(params)

        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2, Eb, T0])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_implicit_pure(t, y, snapshot, params, Ed, Td),
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

        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params, return_mdot=True, rdot=v2)
        params['shell_mass'].value = mShell
        params['shell_massDot'].value = mShell_dot

        # ---------------------------------------------------------------------
        # Compute and store forces (pure function, no mutation during calc)
        # ---------------------------------------------------------------------
        force_props = compute_forces_pure(R2, mShell, Pb, shell_props, params)
        params['F_grav'].value = force_props.F_grav
        params['F_ion_in'].value = force_props.F_ion_in
        params['F_ion_out'].value = force_props.F_ion_out
        params['F_ram'].value = force_props.F_ram
        params['F_rad'].value = force_props.F_rad

        # Save snapshot
        params.save_snapshot()

        # ---------------------------------------------------------------------
        # Check termination conditions
        # ---------------------------------------------------------------------

        # Get Lgain and Lloss from bubble properties or params
        if bubble_props is not None:
            Lloss = bubble_props.bubble_LTotal
            # Add leak if available
            bubble_Leak = params.get('bubble_Leak', None)
            if bubble_Leak is not None and hasattr(bubble_Leak, 'value'):
                Lloss += bubble_Leak.value
        else:
            Lloss_param = params.get('bubble_Lloss', None)
            Lloss = Lloss_param.value if Lloss_param and hasattr(Lloss_param, 'value') else 0.0

        Lgain = feedback.Lmech_total  # From feedback calculation above

        # Store for reference
        params['bubble_Lgain'].value = Lgain
        params['bubble_Lloss'].value = Lloss

        # Save snapshot
        params.save_snapshot()
        
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
        

        # Get threshold from params (default 0.05)
        phase_switch_threshold = params.get('phaseSwitch_LlossLgain', None)
        if phase_switch_threshold and hasattr(phase_switch_threshold, 'value'):
            threshold = phase_switch_threshold.value
        else:
            threshold = 0.05

        if Lgain > 0 and (Lgain - Lloss) / Lgain < threshold:
            termination_reason = "cooling_balance"
            logger.info(f"Cooling balance reached: Lloss/Lgain ratio below {threshold}")
            break

        # Collapse detection: velocity negative AND radius decreasing
        if v2 < 0 and R2 < R2_prev:
            params['isCollapse'].value = True

        # Update R2_prev for next iteration
        R2_prev = R2

        if t_now > tmax:
            termination_reason = "reached_tmax"
            params['SimulationEndReason'].value = 'Stopping time reached'
            params['EndSimulationDirectly'].value = True
            break

        is_collapse = params.get('isCollapse', None)
        if is_collapse and hasattr(is_collapse, 'value') and is_collapse.value:
            coll_r = params['coll_r'].value
            if R2 < coll_r:
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
