#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified Momentum Phase Runner for TRINITY
==========================================

This module implements the momentum-driven phase using scipy.integrate.solve_ivp.
In this phase, thermal pressure is negligible and expansion is driven purely by
ram pressure from stellar winds and supernovae.

Overview
--------
The momentum phase is the final expansion phase where:
- Bubble thermal energy Eb ≈ 0 (thermal pressure negligible)
- Only radius and velocity (R2, v2) are evolved
- Expansion driven by ram pressure only

Key Features
------------
1. **Eb = 0**: Energy-driven terms negligible
2. **Only rd, vd evolved**: Ed = Td = 0 (no energy/temperature evolution)
3. **Pure ODE functions**: No dictionary mutations during integration
4. **scipy.integrate.solve_ivp(LSODA)**: Adaptive integration for accuracy
5. **Segment-based integration**: Parameter updates between segments
6. **Consistent snapshots**: All values saved at consistent timestamps

Snapshot Consistency (January 2026)
-----------------------------------
Snapshots are saved BEFORE ODE integration to ensure all values correspond
to the same timestamp (t_now). The snapshot includes:
- t_now, R2, v2 (current state)
- feedback properties (Lmech, pdot, v_mech, etc.)
- shell_props (shell structure)
- Pb (ram pressure only, since Eb = 0)
- forces (F_grav, F_ram, F_ion, F_rad)
- mShell, mShell_dot (shell mass and accretion rate)

Main Function
-------------
run_phase_momentum(params) -> MomentumPhaseResults

Returns
-------
MomentumPhaseResults : dataclass
    Contains t, R2, v2 arrays and termination info

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import logging
from dataclasses import dataclass

from src.phase_general import phase_ODEs
import src._functions.unit_conversions as cvt
import src.cloud_properties.mass_profile as mass_profile
from src.sb99.update_feedback import get_currentSB99feedback
from src._input.dictionary import updateDict

# Import pure/modified functions
from src.shell_structure.shell_structure_modified import (
    shell_structure_pure,
    ShellProperties,
)
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.cloud_properties import density_profile

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DT_SEGMENT = 5e-2  # Myr - segment duration (larger OK in momentum phase)
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
# Force Computation
# =============================================================================

@dataclass
class ForceProperties:
    """Container for force calculations (pure function output)."""
    F_grav: float       # Gravitational force
    F_ion_in: float     # Inward ionization pressure force
    F_ion_out: float    # Outward ionization pressure force
    F_ram: float        # Ram pressure force
    F_rad: float        # Radiation pressure force


def compute_forces_momentum_pure(
    R2: float,
    mShell: float,
    Lmech_total: float,
    v_mech_total: float,
    shell_props: ShellProperties,
    params,
) -> ForceProperties:
    """
    Compute all force components for momentum phase without mutating params.

    In momentum phase, pressure is ram pressure only (no thermal pressure).
    """
    # Gravitational force
    G = params['G'].value
    mCluster = params['mCluster'].value
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # Ram pressure (momentum phase - no thermal pressure)
    press_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)

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
    F_ram = press_ram * FOUR_PI * R2**2

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
# ODE Snapshot for Momentum Phase
# =============================================================================

@dataclass
class MomentumODESnapshot:
    """Frozen snapshot of parameters for momentum phase ODE."""
    G: float
    mCluster: float
    Lmech_total: float
    v_mech_total: float
    k_B: float
    Qi: float
    caseB_alpha: float
    nISM: float
    PISM: float
    rCloud: float
    rShell: float
    FABSi: float
    F_rad: float
    mShell: float
    mShell_dot: float
    isCollapse: bool


def create_momentum_snapshot(params, shell_props: ShellProperties,
                              mShell: float, mShell_dot: float) -> MomentumODESnapshot:
    """Create a frozen snapshot of parameters for ODE integration."""
    PISM = params.get('PISM', None)
    if PISM is not None and hasattr(PISM, 'value'):
        PISM = PISM.value
    else:
        PISM = 0.0

    is_collapse = params.get('isCollapse', None)
    if is_collapse and hasattr(is_collapse, 'value'):
        is_collapse = is_collapse.value
    else:
        is_collapse = False

    return MomentumODESnapshot(
        G=params['G'].value,
        mCluster=params['mCluster'].value,
        Lmech_total=params['Lmech_total'].value,
        v_mech_total=params['v_mech_total'].value,
        k_B=params['k_B'].value,
        Qi=params['Qi'].value,
        caseB_alpha=params['caseB_alpha'].value,
        nISM=params['nISM'].value,
        PISM=PISM,
        rCloud=params['rCloud'].value,
        rShell=shell_props.rShell,
        FABSi=shell_props.shell_fAbsorbedIon,
        F_rad=shell_props.shell_F_rad,
        mShell=mShell,
        mShell_dot=mShell_dot,
        isCollapse=is_collapse,
    )


# =============================================================================
# Pure ODE for Momentum Phase
# =============================================================================

def get_ODE_momentum_pure(t: float, y: np.ndarray, snapshot: MomentumODESnapshot,
                          params) -> np.ndarray:
    """
    Pure ODE function for momentum phase.

    In momentum phase, Eb = 0 so bubble pressure is ram pressure only.
    Reads snapshot but does NOT mutate during integration.

    Parameters
    ----------
    t : float
        Time [Myr]
    y : ndarray
        State vector [R2, v2]
    snapshot : MomentumODESnapshot
        Frozen snapshot of parameters
    params : dict
        Original params for density profile lookup

    Returns
    -------
    dydt : ndarray
        Derivatives [dR2/dt, dv2/dt]
    """
    R2, v2 = y
    R2 = max(R2, 1e-10)

    # Get parameters from snapshot
    G = snapshot.G
    mCluster = snapshot.mCluster
    Lmech_total = snapshot.Lmech_total
    v_mech_total = snapshot.v_mech_total
    k_B = snapshot.k_B
    FABSi = snapshot.FABSi
    F_rad = snapshot.F_rad
    Qi = snapshot.Qi
    mShell = snapshot.mShell
    mShell_dot = snapshot.mShell_dot

    mShell = max(mShell, 1e-10)

    # Gravity
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)

    # Ram pressure (momentum phase - no thermal pressure)
    press_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)

    # HII pressures
    if FABSi < 1.0:
        rShell = snapshot.rShell
        try:
            n_r = density_profile.get_density_profile(np.array([rShell]), params)
            if hasattr(n_r, '__len__') and len(n_r) == 1:
                n_r = n_r[0]
            press_HII_in = n_r * k_B * 1e4  # TShell_ion approximation
        except Exception:
            press_HII_in = 0.0
    else:
        press_HII_in = 0.0

    # Add ambient pressure if shell is beyond cloud
    if snapshot.rShell >= snapshot.rCloud:
        press_HII_in += snapshot.PISM * k_B

    # Calculate press_HII_out
    if FABSi < 1:
        nR2 = snapshot.nISM
    else:
        caseB_alpha = snapshot.caseB_alpha
        nR2 = np.sqrt(Qi / caseB_alpha / R2**3 * 3 / FOUR_PI)
    press_HII_out = 2 * nR2 * k_B * 3e4

    # Net pressure force
    F_pressure = FOUR_PI * R2**2 * (press_ram - press_HII_in + press_HII_out)

    # Derivatives
    rd = v2
    vd = (F_pressure - mShell_dot * v2 - F_grav + F_rad) / mShell

    return np.array([rd, vd])


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
    T0 = params['T0'].value

    params['Eb'].value = 0.0

    # Pre-allocate results
    t_results = [tmin]
    R2_results = [R2]
    v2_results = [v2]

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
        params['Eb'].value = 0.0
        params['T0'].value = T0

        # ---------------------------------------------------------------------
        # Get feedback and shell structure
        # ---------------------------------------------------------------------
        feedback = get_currentSB99feedback(t_now, params)
        updateDict(params, feedback)

        # Calculate shell structure using pure function
        shell_props = shell_structure_pure(params)
        updateDict(params, shell_props)

        # Set R1 = R2 (no inner shock in momentum phase)
        params['R1'].value = R2

        # ---------------------------------------------------------------------
        # Get shell mass
        # ---------------------------------------------------------------------
        is_collapse = params.get('isCollapse', None)
        if is_collapse and hasattr(is_collapse, 'value') and is_collapse.value:
            mShell = params['shell_mass'].value
            mShell_dot = 0.0
        else:
            mShell, mShell_dot = mass_profile.get_mass_profile(
                R2, params, return_mdot=True, rdot=v2
            )
            # Handle array returns
            if hasattr(mShell, '__len__') and len(mShell) == 1:
                mShell = float(mShell[0])
            if hasattr(mShell_dot, '__len__') and len(mShell_dot) == 1:
                mShell_dot = float(mShell_dot[0])

        params['shell_mass'].value = mShell
        params['shell_massDot'].value = mShell_dot

        # ---------------------------------------------------------------------
        # Compute and store forces BEFORE ODE - all values consistent at t_now
        # ---------------------------------------------------------------------
        Lmech_total = feedback.Lmech_total
        v_mech_total = feedback.v_mech_total

        force_props = compute_forces_momentum_pure(
            R2, mShell, Lmech_total, v_mech_total, shell_props, params
        )
        params['F_grav'].value = force_props.F_grav
        params['F_ion_in'].value = force_props.F_ion_in
        params['F_ion_out'].value = force_props.F_ion_out
        params['F_ram'].value = force_props.F_ram
        params['F_rad'].value = force_props.F_rad

        # Store Pb (ram pressure)
        press_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
        params['Pb'].value = press_ram

        # ---------------------------------------------------------------------
        # Save snapshot BEFORE ODE - all values are consistent at t_now
        # ---------------------------------------------------------------------
        # At this point: t_now, R2, v2, feedback, shell_props, mShell, forces,
        # Pb are all computed for the SAME t_now
        params.save_snapshot()

        # ---------------------------------------------------------------------
        # Build snapshot and integrate segment
        # ---------------------------------------------------------------------
        snapshot = create_momentum_snapshot(params, shell_props, mShell, mShell_dot)

        t_segment_end = min(t_now + DT_SEGMENT, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2])

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_momentum_pure(t, y, snapshot, params),
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
        # Check termination conditions
        # ---------------------------------------------------------------------

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

    logger.info(f"Momentum phase completed: {termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr, Final R2: {R2:.6e} pc, Segments: {segment_count}")

    return MomentumPhaseResults(
        t=np.array(t_results),
        R2=np.array(R2_results),
        v2=np.array(v2_results),
        termination_reason=termination_reason,
        final_time=t_now,
    )
