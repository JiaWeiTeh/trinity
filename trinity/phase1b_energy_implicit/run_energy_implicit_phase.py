#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Implicit Phase Runner for TRINITY
=================================================

This module continues the energy phase with real-time beta/delta calculations,
using scipy.integrate.solve_ivp instead of manual Euler stepping.

Key Features
------------
1. **Pure ODE functions**: No dictionary mutations during integration
2. **scipy.integrate.solve_ivp(LSODA)**: Adaptive integration for accuracy
3. **Segment-based integration**: Beta/delta updates between segments
4. **Pre-allocated result arrays**: Efficient memory usage
5. **Consistent snapshots**: All values saved at consistent timestamps

Snapshot Consistency (January 2026)
-----------------------------------
Snapshots are saved BEFORE ODE integration to ensure all values correspond
to the same timestamp (t_now). The snapshot includes:
- t_now, R2, v2, Eb, T0 (current state)
- feedback properties (Lmech, pdot, etc.)
- shell_props (shell structure)
- bubble_props (bubble structure from beta-delta solver)
- beta, delta (cooling parameters)
- R1, Pb (inner radius and pressure)
- forces (F_grav, F_ram, F_ion, F_rad)
- residual diagnostics (Edot and T residuals)

Beta-Delta Solver
-----------------
The beta-delta solver (get_betadelta.py) uses:
1. Grid search first (4x4 grid by default)
2. L-BFGS-B fallback only if grid residual > LBFGSB_FALLBACK_THRESHOLD
3. Best result selection from all candidates

The solver returns a BetaDeltaResult dataclass with:
- beta, delta: Cooling parameters
- Edot_residual, T_residual: Normalized residuals
- Edot_from_beta, Edot_from_balance: Raw Edot values for diagnostics
- T_bubble, T0: Raw temperature values for diagnostics

Main Function
-------------
run_phase_energy(params) -> ImplicitPhaseResults

Returns
-------
ImplicitPhaseResults : dataclass
    Contains t, R2, v2, Eb, T0, beta, delta arrays and termination info

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import trinity.cloud_properties.mass_profile as mass_profile
import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity._functions.unit_conversions as cvt
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity._functions.operations as operations
from trinity.sps.update_feedback import get_current_sps_feedback
from trinity._input.dictionary import updateDict

# Import pure functions
from trinity.phase1_energy.energy_phase_ODEs import (
    ODESnapshot,
    get_ODE_Edot_pure,
    create_ODE_snapshot,
    ODEResult,
    compute_derived_quantities,
)
from trinity.phase1b_energy_implicit.get_betadelta import (
    solve_betadelta_pure,
    cool_beta_to_Ebdot_pure,
    delta2dTdt_pure,
    compute_R1_Pb,
    BetaDeltaResult,
)
from trinity.shell_structure.shell_structure import (
    shell_structure_pure,
    ShellProperties,
)

# Import centralized event functions
from trinity.phase_general.phase_events import (
    build_implicit_phase_events,
    check_event_termination,
    apply_event_result,
)
from trinity._output.simulation_end import SimulationEndCode

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# TODO: very fine grid in this phase. Only in transition phase it goes coarse. 

COOLING_UPDATE_INTERVAL = 5e-3  # Myr - recalculate cooling
DT_SEGMENT_INIT = 5e-4  # Myr - initial segment duration
DT_SEGMENT_MIN = 1e-4   # Myr - minimum segment duration
DT_SEGMENT_MAX = 5e-2   # Myr - maximum segment duration
MAX_SEGMENTS = 5000
FOUR_PI = 4.0 * np.pi

# Adaptive stepping parameters
ADAPTIVE_THRESHOLD_DEX = 0.05  # dex - threshold for parameter change (10^0.1 ≈ 1.26x)
ADAPTIVE_FACTOR = 10**0.1     # Factor to increase/decrease DT_SEGMENT (~1.26)

# Velocity-based proactive timestep control (for rapid collapse)
# When |v2| exceeds threshold, reduce dt_segment to ensure fine temporal resolution
VELOCITY_THRESHOLD_COLLAPSE = 50.0   # pc/Myr - proactively reduce step when |v2| > this
VELOCITY_THRESHOLD_EXTREME = 150.0   # pc/Myr - use minimum step when |v2| > this
DT_SEGMENT_COLLAPSE = 5e-5           # Myr - segment duration during collapse (50 years, tighter than other phases)

# Parameters to monitor for adaptive stepping (keys in params dict)
# Only scalar (float/int) parameters - no arrays
# Based on analysis of the top 30 most variable parameters
ADAPTIVE_MONITOR_KEYS = [
    # Core state variables
    'R2', 'v2', 'Eb', 'T0', 'Pb', 'R1',
    # Feedback values
    'pdot_SN', 'Lmech_SN', 'pdotdot_total',
    # Cooling parameters
    'cool_delta', 'cool_beta',
    # Bubble properties
    'bubble_mass', 'bubble_r_Tb', 'bubble_LTotal',
    'bubble_L1Bubble', 'bubble_Lloss', 'bubble_dMdt',
    'bubble_L2Conduction', 'bubble_L3Intermediate',
    # Shell parameters
    'shell_mass', 'shell_massDot', 'shell_n0', 'shell_nMax',
    'shell_thickness', 'shell_tauKappaRatio', 'shell_fIonisedDust', 'rShell',
    # Force parameters
    'F_grav', 'F_ram', 'F_ram_wind', 'F_ram_SN',
    'F_ion_in', 'F_HII', 'F_rad', 'F_ISM',
]

# ODE solver settings
ODE_RTOL = 1e-6      # Relative tolerance
ODE_ATOL = 1e-8      # Absolute tolerance (relaxed from 1e-9)
ODE_MIN_STEP = 1e-6  # Minimum step size (Myr)
ODE_MAX_STEP = DT_SEGMENT_MIN / 5  # Max step = 2e-5 Myr (ensures >=5 steps per segment)

# Solver method: 'LSODA' for stiff/non-stiff switching
ODE_METHOD = 'LSODA'


# =============================================================================
# Force Properties Dataclass
# =============================================================================
# Adaptive Stepping Helper
# =============================================================================

def compute_max_dex_change(params_before: dict, params_after: dict, keys: list) -> float:
    """
    Compute the maximum dex (log10) change across monitored parameters.

    Parameters
    ----------
    params_before : dict
        Parameter values before the segment
    params_after : dict
        Parameter values after the segment
    keys : list
        List of parameter keys to monitor

    Returns
    -------
    float
        Maximum absolute dex change across all monitored parameters
    """
    max_dex = 0.0
    for key in keys:
        old_val = params_before.get(key)
        new_val = params_after.get(key)

        # Skip if values are missing, zero, or opposite signs
        if old_val is None or new_val is None:
            continue
        if old_val == 0 or new_val == 0:
            continue
        if (old_val > 0) != (new_val > 0):  # Sign change
            # Large change if sign flips
            max_dex = max(max_dex, 1.0)
            continue

        # Compute dex change: |log10(new/old)|
        try:
            dex_change = abs(np.log10(abs(new_val) / abs(old_val)))
            max_dex = max(max_dex, dex_change)
        except (ValueError, ZeroDivisionError):
            continue

    return max_dex


def get_monitor_values(params) -> dict:
    """
    Extract current values of monitored parameters.

    Parameters
    ----------
    params : DescribedDict
        Parameter dictionary

    Returns
    -------
    dict
        Dictionary of parameter key -> value
    """
    values = {}
    for key in ADAPTIVE_MONITOR_KEYS:
        try:
            val = params.get(key, None)
            if val is not None and hasattr(val, 'value'):
                values[key] = val.value
            elif val is not None:
                values[key] = val
        except Exception:
            pass
    return values


# =============================================================================
# Force Properties Dataclass
# =============================================================================

@dataclass
class ForceProperties:
    """Container for force calculations (pure function output)."""
    F_grav: float       # Gravitational force
    F_ion_in: float     # Inward ionization pressure force
    F_HII: float        # Outward HII pressure force (from n_IF_Str)
    F_ram: float        # Ram pressure force (from bubble pressure)
    F_rad: float        # Radiation pressure force
    # Pressure quantities
    n_IF: float = 0.0
    R_IF: float = 0.0
    P_HII: float = 0.0
    P_drive: float = 0.0
    P_ram: float = 0.0
    press_HII_in: float = 0.0


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

    # Inward pressure from photoionized gas outside shell (evaluated at rShell)
    FABSi = shell_props.shell_fAbsorbedIon
    if FABSi < 1.0:
        from trinity.cloud_properties import density_profile
        try:
            n_r = density_profile.get_density_profile(np.array([rShell]), params)
            if hasattr(n_r, '__len__') and len(n_r) == 1:
                n_r = n_r[0]
            P_ext = 2.0 * n_r * k_B * TShell_ion
        except Exception:
            P_ext = 0.0
    else:
        P_ext = 0.0

    # Add ISM pressure if shell extends beyond cloud
    if rShell >= rCloud:
        P_ext += PISM * k_B

    # ==========================================================================
    # WARM IONIZED GAS PRESSURE (implicit phase)
    # P_HII from Strömgren ionization balance in shell (n_IF_Str)
    # ==========================================================================
    n_IF = shell_props.n_IF
    R_IF = shell_props.R_IF

    # P_HII pre-computed in phase runner from n_IF_Str
    P_HII = params['P_HII'].value
    P_drive = max(Pb, P_HII)

    # Forces
    F_ion_in = P_ext * FOUR_PI * R2**2
    F_HII = FOUR_PI * R2**2 * P_HII

    # Ram pressure force (from bubble pressure)
    F_ram = Pb * FOUR_PI * R2**2

    # Radiation pressure force (direct + IR-trapped)
    if shell_props.isDissolved:
        F_rad = 0.0
    else:
        F_rad = (shell_props.shell_fAbsorbedWeightedTotal
                 * params['Lbol'].value / params['c_light'].value
                 * (1.0 + shell_props.shell_tauKappaRatio * params['dust_KappaIR'].value))

    return ForceProperties(
        F_grav=F_grav,
        F_ion_in=F_ion_in,
        F_HII=F_HII,
        F_ram=F_ram,
        F_rad=F_rad,
        n_IF=n_IF,
        R_IF=R_IF,
        P_HII=P_HII,
        P_drive=P_drive,
        P_ram=0.0,  # no ram pressure in implicit phase
        press_HII_in=P_ext,
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
    # try to fix kink (see paper/rCloud_bump)
    # Ed_from_balance = dydt_energy[2]
    # Ed = min(Ed_from_beta, Ed_from_balance)
    # return np.array([rd, vd, Ed, Td_from_delta])


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

    # --- PHASE BOUNDARY DIAGNOSTIC ---
    v2_from_ODE = params['v2'].value
    v2_from_alpha = params['cool_alpha'].value * params['R2'].value / params['t_now'].value
    logger.info(f"PHASE BOUNDARY [energy->implicit]: "
                f"v2_ODE={v2_from_ODE:.6e}, v2_alpha={v2_from_alpha:.6e}, "
                f"ratio={v2_from_alpha/v2_from_ODE if v2_from_ODE != 0 else float('inf'):.4f}")
    logger.info(f"  R2={params['R2'].value:.6e}, shell_mass={params['shell_mass'].value:.6e}, "
                f"Eb={params['Eb'].value:.6e}, t_now={params['t_now'].value:.6e}")
    # --- END DIAGNOSTIC ---

    # Update cool_alpha to match ODE-evolved v2 (preserves ODE continuity)
    params['cool_alpha'].value = params['t_now'].value / params['R2'].value * params['v2'].value

    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # If the prior phase already advanced past stop_t, there is no work to
    # do here.  Surface that explicitly instead of silently looping zero
    # times and reporting termination_reason="unknown".
    if tmin >= tmax:
        logger.info(
            f"Implicit phase skipped: t_now={tmin:.6e} Myr >= stop_t={tmax:.6e} Myr "
            f"(simulation time limit reached in prior phase)"
        )
        params['SimulationEndCode'].value = SimulationEndCode.STOPPING_TIME.code
        params['SimulationEndReason'].value = (
            f"Reached stop_t={tmax:.6e} Myr during prior phase"
        )
        params['EndSimulationDirectly'].value = True
        return ImplicitPhaseResults(
            t=np.array([tmin]),
            R2=np.array([params['R2'].value]),
            v2=np.array([params['v2'].value]),
            Eb=np.array([params['Eb'].value]),
            T0=np.array([params['T0'].value]),
            beta=np.array([params['cool_beta'].value]),
            delta=np.array([params['cool_delta'].value]),
            termination_reason="skipped_past_stop_t",
            final_time=tmin,
        )

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

    # Initial values are appended inside the loop after all derived
    # quantities (beta, delta, shell, etc.) are computed at t_now,
    # ensuring every result entry is self-consistent.

    t_now = tmin
    segment_count = 0
    termination_reason = None
    beta = params['cool_beta'].value
    delta = params['cool_delta'].value

    # Track previous R2 for collapse detection
    R2_prev = R2

    # Adaptive time stepping
    dt_segment = DT_SEGMENT_INIT

    # =============================================================================
    # Build events for safe termination
    # =============================================================================

    # Build events using centralized module
    # Returns (events_list, cooling_balance_factory)
    ode_events, cooling_balance_factory = build_implicit_phase_events(params)

    # =============================================================================
    # Main loop (segment-based with adaptive stepping)
    # =============================================================================

    while t_now <= tmax and segment_count < MAX_SEGMENTS:
        segment_count += 1

        # stop_at_rCloud_nSnap: terminate once we've already saved enough
        # post-rCloud segment-loop snapshots.  Checked at top of loop so we
        # break BEFORE this iteration's save_snapshot fires.
        nSnap_rCloud = params['stop_at_rCloud_nSnap'].value
        if (nSnap_rCloud is not None
                and R2 > params['rCloud'].value
                and params['_snapshots_after_rCloud'].value >= nSnap_rCloud):
            termination_reason = "stop_at_rCloud"
            params['SimulationEndReason'].value = (
                f"Reached {nSnap_rCloud} segment(s) past rCloud "
                f"(stop_at_rCloud_nSnap)"
            )
            params['SimulationEndCode'].value = SimulationEndCode.RCLOUD_BOUNDARY.code
            params['EndSimulationDirectly'].value = True
            break

        # Log current state at beginning of each segment
        logger.debug(f"[Implicit] t={t_now:.6e} Myr, R2={R2:.4e} pc, v2={v2:.4e} pc/Myr, Eb={Eb:.4e}, T0={T0:.4e} K")

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
        # Get feedback
        # ---------------------------------------------------------------------
        feedback = get_current_sps_feedback(t_now, params)
        updateDict(params, feedback)

        # ---------------------------------------------------------------------
        # Covering-fraction energy leak (Eq. leak), consumed by the
        # energy-balance branch of solve_betadelta_pure below. Pb and c_sound
        # are carried from the previous segment (1-step frozen), consistent
        # with this phase's slowly-varying Lloss treatment. Cf=1 -> 0, so the
        # sealed-bubble trajectory is reproduced exactly.
        # ---------------------------------------------------------------------
        params['bubble_Leak'].value = get_bubbleParams.get_leak_luminosity(
            params['coverFraction'].value,
            params['R2'].value,
            params['Pb'].value,
            params['c_sound'].value,
            params['gamma_adia'].value,
        )

        # ---------------------------------------------------------------------
        # Calculate beta/delta and bubble properties BEFORE shell structure,
        # so that Pb and bubble_mass are current when shell_structure_pure
        # reads them (bubble computation does not depend on shell).
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

        # Energy balance: luminosity gain and loss
        if betadelta_result.L_gain is not None:
            params['bubble_Lgain'].value = betadelta_result.L_gain
        if betadelta_result.L_loss is not None:
            params['bubble_Lloss'].value = betadelta_result.L_loss

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
        # Compute shell mass BEFORE shell structure so that the shell
        # termination condition uses the current R2's swept-up mass.
        # Two conditions for freezing shell mass:
        # 1. During collapse (isCollapse=True): shell mass is frozen
        # 2. Shell mass can NEVER decrease - once mass is swept up, it stays in shell
        # ---------------------------------------------------------------------
        prev_mShell = params['shell_mass'].value
        is_collapse = params.get('isCollapse', None)
        is_collapse_val = is_collapse.value if is_collapse and hasattr(is_collapse, 'value') else False

        if is_collapse_val:
            # During collapse, shell mass is frozen
            mShell = prev_mShell
            mShell_dot = 0.0
        else:
            mShell_new, mShell_dot = mass_profile.get_mass_profile(R2, params, return_mdot=True, rdot=v2)
            # Ensure shell mass never decreases
            if prev_mShell > 0 and mShell_new < prev_mShell:
                mShell = prev_mShell
                mShell_dot = 0.0
            else:
                mShell = mShell_new
        params['shell_mass'].value = mShell
        params['shell_massDot'].value = mShell_dot

        # ---------------------------------------------------------------------
        # Calculate shell structure (now with current Pb, bubble_mass, shell_mass)
        # ---------------------------------------------------------------------
        shell_props = shell_structure_pure(params)
        updateDict(params, shell_props)

        # Compute P_HII from Strömgren ionization balance in shell (n_IF_Str)
        n_IF_Str = shell_props.n_IF_Str
        if params['include_PHII'].value and n_IF_Str > 0:
            P_HII = 2.0 * n_IF_Str * params['k_B'].value * params['TShell_ion'].value
        else:
            P_HII = 0.0
        params['P_HII'].value = P_HII
        F_HII = 4.0 * np.pi * R2**2 * P_HII
        params['F_HII'].value = F_HII

        # ---------------------------------------------------------------------
        # Convert beta/delta to Ed, Td using pure functions
        # ---------------------------------------------------------------------

        Ed = cool_beta_to_Ebdot_pure(beta, Pb, t_now, R1, R2, v2, Eb, feedback.pdot_total, feedback.pdotdot_total)
        Td = delta2dTdt_pure(t_now, T0, delta)

        force_props = compute_forces_pure(R2, mShell, Pb, shell_props, params)
        params['F_grav'].value = force_props.F_grav
        params['F_ion_in'].value = force_props.F_ion_in
        params['F_HII'].value = force_props.F_HII
        params['F_ram'].value = force_props.F_ram
        params['F_rad'].value = force_props.F_rad
        # Pressure diagnostic quantities
        params['n_IF'].value = force_props.n_IF
        params['R_IF'].value = force_props.R_IF
        params['P_HII'].value = force_props.P_HII
        params['P_drive'].value = force_props.P_drive
        params['P_ram'].value = force_props.P_ram
        params['press_HII_in'].value = force_props.press_HII_in
        params['F_ram_wind'].value = feedback.pdot_W
        params['F_ram_SN'].value = feedback.pdot_SN

        # ---------------------------------------------------------------------
        # Save snapshot BEFORE ODE - all values are consistent at t_now
        # ---------------------------------------------------------------------
        # At this point: t_now, R2, v2, Eb, T0, feedback, shell_props, bubble_props,
        # beta, delta, R1, Pb, forces, residuals are all computed for the SAME t_now
        _save_count_before = params.save_count
        params.save_snapshot()

        # stop_at_rCloud_nSnap: increment past-rCloud counter only when the
        # save actually wrote (save_count delta > 0); the duplicate guard
        # in save_snapshot can silently skip the first segment of a phase
        # when its (t_now, R2) match the previous phase's reconciliation.
        if (params['stop_at_rCloud_nSnap'].value is not None
                and params.save_count > _save_count_before
                and R2 > params['rCloud'].value):
            params['_snapshots_after_rCloud'].value += 1

        # Store results at the same consistent point as the snapshot
        t_results.append(t_now)
        R2_results.append(R2)
        v2_results.append(v2)
        Eb_results.append(Eb)
        T0_results.append(T0)
        beta_results.append(beta)
        delta_results.append(delta)

        # ---------------------------------------------------------------------
        # Check if we've reached stop_t - if so, terminate successfully
        # ---------------------------------------------------------------------
        if tmax is not None and t_now >= tmax:
            termination_reason = "reached_tmax"
            params['SimulationEndReason'].value = 'Stopping time reached'
            params['SimulationEndCode'].value = SimulationEndCode.STOPPING_TIME.code
            params['EndSimulationDirectly'].value = True
            logger.info(f"Simulation reached stop_t={tmax} Myr successfully")
            break

        # ---------------------------------------------------------------------
        # Build snapshot and integrate segment
        # ---------------------------------------------------------------------
        snapshot = create_ODE_snapshot(params, shell_props)

        # Capture parameter values BEFORE integration for adaptive stepping
        values_before = get_monitor_values(params)

        t_segment_end = min(t_now + dt_segment, tmax)
        t_span = (t_now, t_segment_end)
        y0 = np.array([R2, v2, Eb, T0])

        # Debug: log the solve_ivp inputs
        logger.debug(f"solve_ivp: t_span=({t_span[0]:.10e}, {t_span[1]:.10e}), dt={dt_segment:.3e}, "
                     f"y0=[R2={y0[0]:.10e}, v2={y0[1]:.6e}, Eb={y0[2]:.6e}, T0={y0[3]:.6e}]")

        try:
            # Build solver kwargs (min_step only supported by LSODA)
            solver_kwargs = {
                'fun': lambda t, y: get_ODE_implicit_pure(t, y, snapshot, params, Ed, Td),
                't_span': t_span,
                'y0': y0,
                'method': ODE_METHOD,
                'rtol': ODE_RTOL,
                'atol': ODE_ATOL,
                'max_step': ODE_MAX_STEP,
                'events': ode_events,  # Event functions for safe termination
            }
            if ODE_METHOD == 'LSODA':
                solver_kwargs['min_step'] = ODE_MIN_STEP

            sol = scipy.integrate.solve_ivp(**solver_kwargs)
        except Exception as e:
            logger.error(f"solve_ivp failed at t={t_now:.6e}: {e}")
            termination_reason = f"solver_error: {e}"
            break

        if not sol.success or len(sol.t) == 0:
            logger.error(f"Solver did not succeed: {sol.message}")
            logger.error(f"  t_span was: ({t_span[0]:.10e}, {t_span[1]:.10e})")
            logger.error(f"  y0 was: R2={y0[0]:.10e}, v2={y0[1]:.6e}")
            termination_reason = f"solver_failed: {sol.message}"
            break

        # ---------------------------------------------------------------------
        # Check if an event terminated the integration
        # ---------------------------------------------------------------------
        event_result = check_event_termination(sol, ode_events)
        if event_result.triggered:
            logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr: "
                          f"R2={event_result.y[0]:.4e} pc, v2={event_result.y[1]:.4e} pc/Myr")
            termination_reason = event_result.reason_code
            # Update state from event
            R2 = float(event_result.y[0])
            v2 = float(event_result.y[1])
            Eb = float(event_result.y[2])
            T0 = float(event_result.y[3])
            t_now = event_result.t
            # Add final state to results.
            # beta/delta are from the start of this segment (best available;
            # the event occurred within one segment of their computation).
            t_results.append(t_now)
            R2_results.append(R2)
            v2_results.append(v2)
            Eb_results.append(Eb)
            T0_results.append(T0)
            beta_results.append(beta)
            delta_results.append(delta)
            # Apply event result to params
            apply_event_result(params, event_result, t_now, event_result.y,
                              state_keys=['R2', 'v2', 'Eb', 'T0'])
            break

        # ---------------------------------------------------------------------
        # Extract final state
        # ---------------------------------------------------------------------
        R2 = float(sol.y[0, -1])
        v2 = float(sol.y[1, -1])
        Eb = float(sol.y[2, -1])
        T0 = float(sol.y[3, -1])
        t_now = float(sol.t[-1])

        # ---------------------------------------------------------------------
        # Adaptive stepping: adjust dt_segment based on parameter changes
        # ---------------------------------------------------------------------
        # Update params with new state for comparison
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = Eb
        params['T0'].value = T0
        
        # Shell mass update for adaptive stepping comparison.
        # Apply the same collapse-freeze and never-decrease guards as the
        # primary shell mass block (lines 642-662).
        prev_mShell_post = params['shell_mass'].value
        is_collapse_post = params.get('isCollapse', None)
        is_collapse_post_val = is_collapse_post.value if is_collapse_post and hasattr(is_collapse_post, 'value') else False

        if is_collapse_post_val:
            pass  # keep params['shell_mass'] at its previous value
        else:
            mShell_post = mass_profile.get_mass_profile(R2, params, return_mdot=False)
            # Ensure shell mass never decreases
            if prev_mShell_post > 0 and mShell_post < prev_mShell_post:
                pass  # keep params['shell_mass'] at its previous value
            else:
                params['shell_mass'].value = mShell_post

        values_after = get_monitor_values(params)
        max_dex_change = compute_max_dex_change(values_before, values_after, ADAPTIVE_MONITOR_KEYS)

        if max_dex_change > ADAPTIVE_THRESHOLD_DEX:
            # Large change: decrease dt_segment
            dt_segment_old = dt_segment
            dt_segment = max(dt_segment / ADAPTIVE_FACTOR, DT_SEGMENT_MIN)
            logger.debug(f"Adaptive: max_dex={max_dex_change:.3f} > {ADAPTIVE_THRESHOLD_DEX}, "
                        f"dt: {dt_segment_old:.3e} -> {dt_segment:.3e}")
        else:
            # Small change: increase dt_segment
            dt_segment_old = dt_segment
            dt_segment = min(dt_segment * ADAPTIVE_FACTOR, DT_SEGMENT_MAX)
            if dt_segment != dt_segment_old:
                logger.debug(f"Adaptive: max_dex={max_dex_change:.3f} < {ADAPTIVE_THRESHOLD_DEX}, "
                            f"dt: {dt_segment_old:.3e} -> {dt_segment:.3e}")

        # ---------------------------------------------------------------------
        # Proactive velocity-based timestep control during collapse
        # This ensures fine temporal resolution when shell is collapsing rapidly
        # ---------------------------------------------------------------------
        abs_v2 = abs(v2)
        if v2 < 0:  # Only during collapse (negative velocity = inward motion)
            if abs_v2 > VELOCITY_THRESHOLD_EXTREME:
                # Extreme collapse velocity: use minimum segment duration
                dt_segment = DT_SEGMENT_COLLAPSE
                logger.debug(f"Velocity-based: |v2|={abs_v2:.1f} > {VELOCITY_THRESHOLD_EXTREME}, "
                             f"dt -> {dt_segment:.3e} Myr (collapse mode)")
            elif abs_v2 > VELOCITY_THRESHOLD_COLLAPSE:
                # Moderate collapse velocity: use intermediate segment duration
                dt_segment = min(dt_segment, DT_SEGMENT_MIN)
                logger.debug(f"Velocity-based: |v2|={abs_v2:.1f} > {VELOCITY_THRESHOLD_COLLAPSE}, "
                            f"dt -> {dt_segment:.3e} Myr")

        # (Results already appended before ODE at the consistent snapshot point.)

        # ---------------------------------------------------------------------
        # Check termination conditions
        # ---------------------------------------------------------------------

        # Re-fetch feedback at post-ODE time for the termination check.
        # Lgain must reflect the current Lmech_total at the new t_now,
        # especially across SN turn-on boundaries.
        feedback_post = get_current_sps_feedback(t_now, params)
        Lgain = feedback_post.Lmech_total

        # Lloss from pre-ODE bubble properties (cannot cheaply recompute
        # without the betadelta solver; acceptable since Lloss changes slowly)
        if bubble_props is not None:
            Lloss = bubble_props.bubble_LTotal
            bubble_Leak = params.get('bubble_Leak', None)
            if bubble_Leak is not None and hasattr(bubble_Leak, 'value'):
                Lloss += bubble_Leak.value
        else:
            Lloss_param = params.get('bubble_Lloss', None)
            Lloss = Lloss_param.value if Lloss_param and hasattr(Lloss_param, 'value') else 0.0

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

        # Stop time check (skip if tmax is None)
        if tmax is not None and t_now > tmax:
            termination_reason = "reached_tmax"
            params['SimulationEndReason'].value = 'Stopping time reached'
            params['SimulationEndCode'].value = SimulationEndCode.STOPPING_TIME.code
            params['EndSimulationDirectly'].value = True
            break

        is_collapse = params.get('isCollapse', None)
        if is_collapse and hasattr(is_collapse, 'value') and is_collapse.value:
            coll_r = params['coll_r'].value
            if R2 < coll_r:
                termination_reason = "small_radius"
                params['SimulationEndReason'].value = 'Small radius reached'
                params['SimulationEndCode'].value = SimulationEndCode.SHELL_COLLAPSED.code
                params['EndSimulationDirectly'].value = True
                break

        # Stop radius check (skip if stop_r is None)
        stop_r = params['stop_r'].value
        if stop_r is not None and R2 > stop_r:
            termination_reason = "large_radius"
            params['SimulationEndReason'].value = 'Large radius reached'
            params['SimulationEndCode'].value = SimulationEndCode.LARGE_RADIUS.code
            params['EndSimulationDirectly'].value = True
            break

    # =============================================================================
    # Capture final post-ODE state if not already recorded.
    # The main result append is now pre-ODE, so termination paths that break
    # after ODE (cooling_balance, small_radius, etc.) would otherwise lose the
    # final state.  Event terminations already append at the event point.
    # =============================================================================
    if len(t_results) == 0 or t_now != t_results[-1]:
        t_results.append(t_now)
        R2_results.append(R2)
        v2_results.append(v2)
        Eb_results.append(Eb)
        T0_results.append(T0)
        # beta/delta are from the last pre-ODE computation (best available)
        beta_results.append(beta)
        delta_results.append(delta)

    # =========================================================================
    # Phase-boundary reconciliation snapshot.
    # Recompute derived properties (Pb, shell structure, forces) with the
    # post-ODE state so the snapshot is fully consistent.  A bare
    # save_snapshot() here would save stale derived values AND block the
    # next phase's correct first snapshot via the duplicate guard.
    # =========================================================================
    try:
        feedback_final = get_current_sps_feedback(t_now, params)
        updateDict(params, feedback_final)
        gamma_adia = params['gamma_adia'].value
        R1_f, Pb_f = compute_R1_Pb(R2, Eb, feedback_final.Lmech_total,
                                    feedback_final.v_mech_total, gamma_adia)
        params['R1'].value = R1_f
        params['Pb'].value = Pb_f
        shell_props_f = shell_structure_pure(params)
        updateDict(params, shell_props_f)
        # P_HII from Strömgren ionization balance
        n_IF_Str_f = shell_props_f.n_IF_Str
        if params['include_PHII'].value and n_IF_Str_f > 0:
            P_HII_f = 2.0 * n_IF_Str_f * params['k_B'].value * params['TShell_ion'].value
        else:
            P_HII_f = 0.0
        params['P_HII'].value = P_HII_f
        params['F_HII'].value = FOUR_PI * R2**2 * P_HII_f
        force_f = compute_forces_pure(R2, params['shell_mass'].value,
                                      Pb_f, shell_props_f, params)
        params['F_grav'].value = force_f.F_grav
        params['F_ion_in'].value = force_f.F_ion_in
        params['F_HII'].value = force_f.F_HII
        params['F_ram'].value = force_f.F_ram
        params['F_rad'].value = force_f.F_rad
        params['P_HII'].value = force_f.P_HII
        params['P_drive'].value = force_f.P_drive
        params['P_ram'].value = force_f.P_ram
        params.save_snapshot()
    except Exception as e:
        logger.warning(f"Phase-boundary reconciliation failed: {e}")

    # =============================================================================
    # Build results
    # =============================================================================

    if termination_reason is None:
        # If we get here with no termination reason, it's either max_segments or unknown
        termination_reason = "max_segments" if segment_count >= MAX_SEGMENTS else "unknown"

    # "unknown" means we fell through every known exit path — a real bug
    # surface, not a routine completion.  Surface it loudly.
    completion_log = logger.warning if termination_reason == "unknown" else logger.info
    completion_log(f"Implicit phase completed: {termination_reason}")
    completion_log(f"  Final time: {t_now:.6e} Myr, Segments: {segment_count}")

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
