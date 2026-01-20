#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified transition phase runner for TRINITY.

This module implements the transition phase between energy-driven and
momentum-driven expansion, using scipy.integrate.solve_ivp.

Key features:
- Energy decays on sound-crossing timescale: dE/dt = -Eb / t_sound
- Uses pure ODE functions (no dictionary mutations during integration)
- scipy.integrate.solve_ivp(LSODA) for adaptive integration
- Segment-based integration with parameter updates between segments
- Uses shell_structure_pure() and updateDict() pattern

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import logging
from typing import Tuple
from dataclasses import dataclass

import src.phase_general.phase_ODEs as phase_ODEs
import src.cloud_properties.mass_profile as mass_profile
import src._functions.unit_conversions as cvt
import src._functions.operations as operations
from src.sb99.update_feedback import get_currentSB99feedback
from src._input.dictionary import updateDict

# Import pure/modified functions
from src.phase1_energy.energy_phase_ODEs_modified import (
    ODESnapshot,
    get_ODE_Edot_pure,
    create_ODE_snapshot,
)
from src.phase1b_energy_implicit.get_betadelta_modified import (
    compute_R1_Pb,
)
from src.shell_structure.shell_structure_modified import (
    shell_structure_pure,
    ShellProperties,
)
import src.bubble_structure.get_bubbleParams as get_bubbleParams

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DT_SEGMENT = 5e-3  # Myr - segment duration, coarser grid towards end
MAX_SEGMENTS = 5000
ENERGY_FLOOR = 1e3  # Minimum energy before transition to momentum phase
FOUR_PI = 4.0 * np.pi


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

def get_ODE_transition_pure(t: float, y: np.ndarray, snapshot: ODESnapshot,
                            params_for_feedback, c_sound: float) -> np.ndarray:
    """
    Pure ODE function for transition phase.

    Energy decays on sound-crossing timescale.
    Reads snapshot but does NOT mutate during integration.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2, Eb]
    snapshot : ODESnapshot
        Frozen snapshot of parameters
    params_for_feedback : DescribedDict
        Original params dict for feedback interpolation
    c_sound : float
        Sound speed [pc/Myr]

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt, dEb/dt]
    """
    R2, v2, Eb = y

    # Get rd, vd from energy ODE
    dydt_energy = get_ODE_Edot_pure(t, y, snapshot, params_for_feedback)
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
# Force Computation (same as in energy_implicit)
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
    """
    # Gravitational force
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
        Qi = params['Qi'].value
        caseB_alpha = params['caseB_alpha'].value
        nR2 = np.sqrt(Qi / caseB_alpha / (R2**3) * 3 / FOUR_PI)
    press_HII_out = 2.0 * nR2 * k_B * 3e4

    F_ion_in = press_HII_in * FOUR_PI * R2**2
    F_ion_out = press_HII_out * FOUR_PI * R2**2

    # Ram pressure force
    F_ram = Pb * FOUR_PI * R2**2

    # Radiation pressure force
    F_rad = shell_props.shell_F_rad

    return ForceProperties(
        F_grav=F_grav,
        F_ion_in=F_ion_in,
        F_ion_out=F_ion_out,
        F_ram=F_ram,
        F_rad=F_rad,
    )


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
        # Update params with current state
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
        updateDict(params, feedback)

        # Calculate shell structure using pure function
        shell_props = shell_structure_pure(params)
        updateDict(params, shell_props)

        # Get sound speed from bubble average temperature
        bubble_Tavg = params.get('bubble_Tavg', None)
        if bubble_Tavg and hasattr(bubble_Tavg, 'value') and bubble_Tavg.value:
            T_for_sound = bubble_Tavg.value
        else:
            T_for_sound = 1e6
        c_sound = operations.get_soundspeed(T_for_sound, params)
        params['c_sound'].value = c_sound

        # ---------------------------------------------------------------------
        # Get R1 and Pb
        # ---------------------------------------------------------------------
        Lmech_total = feedback.Lmech_total
        v_mech_total = feedback.v_mech_total
        gamma_adia = params['gamma_adia'].value

        R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)
        params['R1'].value = R1
        params['Pb'].value = Pb

        # ---------------------------------------------------------------------
        # Compute shell mass and forces BEFORE ODE - all values consistent at t_now
        # ---------------------------------------------------------------------
        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params, return_mdot=True, rdot=v2)
        params['shell_mass'].value = mShell
        params['shell_massDot'].value = mShell_dot

        force_props = compute_forces_pure(R2, mShell, Pb, shell_props, params)
        params['F_grav'].value = force_props.F_grav
        params['F_ion_in'].value = force_props.F_ion_in
        params['F_ion_out'].value = force_props.F_ion_out
        params['F_ram'].value = force_props.F_ram
        params['F_rad'].value = force_props.F_rad

        # ---------------------------------------------------------------------
        # Save snapshot BEFORE ODE - all values are consistent at t_now
        # ---------------------------------------------------------------------
        # At this point: t_now, R2, v2, Eb, feedback, shell_props, R1, Pb,
        # forces, mShell are all computed for the SAME t_now
        params.save_snapshot()

        # ---------------------------------------------------------------------
        # Build snapshot and integrate segment
        # ---------------------------------------------------------------------
        snapshot = create_ODE_snapshot(params)

        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2, Eb])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_transition_pure(t, y, snapshot, params, c_sound),
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
        # Check termination conditions
        # ---------------------------------------------------------------------

        # Energy floor reached -> transition to momentum phase
        if Eb < ENERGY_FLOOR:
            termination_reason = "energy_floor"
            logger.info(f"Energy dropped below floor ({ENERGY_FLOOR}), transitioning to momentum")
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

        # Dissolution check
        shell_nMax = params.get('shell_nMax', None)
        if shell_nMax and hasattr(shell_nMax, 'value'):
            if shell_nMax.value < params['stop_n_diss'].value:
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

    logger.info(f"Transition phase completed: {termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr, Final Eb: {Eb:.6e}, Segments: {segment_count}")

    return TransitionPhaseResults(
        t=np.array(t_results),
        R2=np.array(R2_results),
        v2=np.array(v2_results),
        Eb=np.array(Eb_results),
        termination_reason=termination_reason,
        final_time=t_now,
    )
