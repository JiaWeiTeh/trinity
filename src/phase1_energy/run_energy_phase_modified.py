#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified energy phase runner for TRINITY.

This module implements the energy-driven phase (Phase 1) using scipy.integrate.solve_ivp
with LSODA method instead of manual Euler stepping.

Key improvements:
1. Pre-allocated result arrays (fixes O(n²) concatenation)
2. scipy.integrate.solve_ivp(LSODA) for adaptive ODE integration
3. Segment-based integration with bubble_luminosity calls between segments
4. ODE reads params but does NOT mutate during integration
5. update_params_after_segment() called after each successful segment

State vector: y = [R2, v2, Eb] (3 variables)
Note: T0 is NOT integrated - it's calculated externally via bubble_luminosity.

@author: TRINITY Team (refactored for solve_ivp)
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.interpolate
import logging
from typing import Dict, Tuple, Optional, List

import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
import src.bubble_structure.bubble_luminosity_modified as bubble_luminosity_modified
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src._input.dictionary import updateDict
import src._functions.unit_conversions as cvt
from src.sb99.update_feedback import get_currentSB99feedback

# Import ODE functions
from src.phase1_energy.energy_phase_ODEs_modified import (
    get_ODE_Edot_pure,
    update_params_after_segment,
    R1Cache,
    radius_exceeds_cloud_event,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

TFINAL_ENERGY_PHASE = 3e-3  # Myr - max duration (~3000 years)
DT_SEGMENT = 5e-5  # Myr - segment duration (comparable to original's 30*1e-6=3e-5)
DT_MIN_SEGMENT = 1e-6  # Myr - minimum segment duration for first step
COOLING_UPDATE_INTERVAL = 5e-2  # Myr - recalculate cooling every 50k years
MAX_SEGMENTS = 10000  # Maximum number of segments to prevent infinite loops


# =============================================================================
# Result Container
# =============================================================================

class EnergyPhaseResults:
    """Container for energy phase results with pre-allocated arrays."""

    def __init__(self, max_points: int = 10000):
        """Initialize with pre-allocated arrays."""
        self.max_points = max_points
        self.n_points = 0

        # Pre-allocate arrays
        self.t = np.zeros(max_points)
        self.R2 = np.zeros(max_points)
        self.v2 = np.zeros(max_points)
        self.Eb = np.zeros(max_points)
        self.T0 = np.zeros(max_points)
        self.R1 = np.zeros(max_points)
        self.mShell = np.zeros(max_points)
        self.Pb = np.zeros(max_points)

        # Termination info
        self.termination_reason = None
        self.final_time = 0.0

    def append(self, t: float, R2: float, v2: float, Eb: float,
               T0: float, R1: float, mShell: float, Pb: float):
        """Append a single point to results."""
        if self.n_points >= self.max_points:
            # Extend arrays if needed
            self._extend_arrays()

        i = self.n_points
        self.t[i] = t
        self.R2[i] = R2
        self.v2[i] = v2
        self.Eb[i] = Eb
        self.T0[i] = T0
        self.R1[i] = R1
        self.mShell[i] = mShell
        self.Pb[i] = Pb
        self.n_points += 1

    def _extend_arrays(self):
        """Double the array capacity."""
        new_max = self.max_points * 2
        for attr in ['t', 'R2', 'v2', 'Eb', 'T0', 'R1', 'mShell', 'Pb']:
            old_arr = getattr(self, attr)
            new_arr = np.zeros(new_max)
            new_arr[:self.max_points] = old_arr
            setattr(self, attr, new_arr)
        self.max_points = new_max

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Return trimmed arrays as dict."""
        n = self.n_points
        return {
            't': self.t[:n].copy(),
            'R2': self.R2[:n].copy(),
            'v2': self.v2[:n].copy(),
            'Eb': self.Eb[:n].copy(),
            'T0': self.T0[:n].copy(),
            'R1': self.R1[:n].copy(),
            'mShell': self.mShell[:n].copy(),
            'Pb': self.Pb[:n].copy(),
        }


# =============================================================================
# Main Energy Phase Function
# =============================================================================

def run_energy(params) -> EnergyPhaseResults:
    """
    Run the energy-driven phase (Phase 1) using solve_ivp.

    This function implements the Weaver+77 bubble expansion model with:
    - Adaptive ODE integration via scipy.integrate.solve_ivp(LSODA)
    - Segment-based integration for updating T0, L_bubble between segments
    - Pre-allocated result arrays for efficiency
    - ODE reads params but does NOT mutate during integration
    - update_params_after_segment() called after each successful segment

    Parameters
    ----------
    params : ParameterDict
        Parameter dictionary with .value attributes

    Returns
    -------
    results : EnergyPhaseResults
        Container with result arrays and termination info
    """
    # =============================================================================
    # Initialization
    # =============================================================================

    t_now = params['t_now'].value
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    rCloud = params['rCloud'].value
    t_neu = params['TShell_neu'].value
    t_ion = params['TShell_ion'].value
    tSF = params['tSF'].value

    # Initialize results container
    results = EnergyPhaseResults()

    # Initialize R1 cache for efficient brentq caching
    r1_cache = R1Cache()

    # =============================================================================
    # Get initial feedback values
    # =============================================================================

    feedback = get_currentSB99feedback(t_now, params)
    (t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
     pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total) = feedback
    
    updateDict(params, ['Qi', 'Li', 'Ln', 'Lbol', 'Lmech_W', 'Lmech_SN', 'Lmech_total', 'pdot_W', 'pdot_SN', 'pdot_total', 'pdotdot_total', 'v_mech_total'],
               [Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total])

    # =============================================================================
    # Calculate initial R1 and Pb
    # =============================================================================

    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )
    r1_cache.update(t_now, R2, Eb, Lmech_total, v_mech_total)

    # Initial shell mass and bubble pressure
    Msh0 = mass_profile.get_mass_profile(R2, params, return_mdot=False)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

    # Update params with initial values
    params['Pb'].value = Pb
    params['R1'].value = R1

    logger.info('Energy phase initialization:')
    logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
    logger.info(f'  Initial shell mass: {Msh0:.6e} Msun')
    logger.info(f'  Initial bubble pressure: {Pb:.6e} Msun/pc/Myr^2')

    # Store initial point
    results.append(t_now, R2, v2, Eb, T0, R1, Msh0, Pb)

    # =============================================================================
    # Build feedback interpolators
    # =============================================================================

    # Get SB99 feedback arrays for interpolation
    # (These should already be in params from initialization)
    t_fb = params['SB99_t'].value
    L_mech_arr = params['SB99_Lmech'].value
    v_mech_arr = params['SB99_vmech'].value

    # Create interpolators (only if arrays exist)
    if len(t_fb) > 1:
        L_mech_interp = scipy.interpolate.interp1d(
            t_fb, L_mech_arr, kind='linear',
            bounds_error=False, fill_value=(L_mech_arr[0], L_mech_arr[-1])
        )
        v_mech_interp = scipy.interpolate.interp1d(
            t_fb, v_mech_arr, kind='linear',
            bounds_error=False, fill_value=(v_mech_arr[0], v_mech_arr[-1])
        )
    else:
        # Fallback to constant values
        L_mech_interp = lambda t: Lmech_total
        v_mech_interp = lambda t: v_mech_total

    # =============================================================================
    # Main integration loop (segment-based)
    # =============================================================================

    tfinal = t_now + TFINAL_ENERGY_PHASE
    segment_count = 0
    continueWeaver = True
    loop_count = 0

    # Track time of last cooling structure update
    t_last_cooling_update = t_now

    # Initialize cooling structure before main loop
    # (Original version does this on first iteration via t_previousCoolingUpdate check)
    cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
    params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
    params['cStruc_heating_nonCIE'].value = heating_nonCIE
    params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
    params['t_previousCoolingUpdate'].value = t_now

    while R2 < rCloud and t_now < tfinal and continueWeaver:
        segment_count += 1

        if segment_count > MAX_SEGMENTS:
            logger.warning(f"Exceeded maximum segments ({MAX_SEGMENTS}), stopping")
            results.termination_reason = "max_segments"
            break

        # =============================================================================
        # Update cooling structure periodically
        # =============================================================================

        if abs(t_now - t_last_cooling_update) > COOLING_UPDATE_INTERVAL:
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
            params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            params['cStruc_heating_nonCIE'].value = heating_nonCIE
            params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            t_last_cooling_update = t_now
            params['t_previousCoolingUpdate'].value = t_now

        # =============================================================================
        # Calculate bubble and shell structure (after first loop)
        # =============================================================================

        calculate_bubble_shell = loop_count > 0

        if calculate_bubble_shell:
            # Update params with current state BEFORE bubble calculations
            # (Original ODE updates params at start of each call, but modified
            # version only updates after segment. Functions like bubble_luminosity
            # read from params, so we must update them first.)
            params['t_now'].value = t_now
            params['R2'].value = R2
            params['v2'].value = v2
            params['Eb'].value = Eb

            # Calculate bubble properties using pure function
            bubble_props = bubble_luminosity_modified.get_bubbleproperties_pure(
                R2=R2,
                v2=v2,
                Eb=Eb,
                t_now=t_now,
                params=params
            )

            # Update params with bubble properties
            params['R1'].value = bubble_props.R1
            params['Pb'].value = bubble_props.Pb
            params['bubble_dMdt'].value = bubble_props.dMdt
            params['bubble_T_r_Tb'].value = bubble_props.T_rgoal
            params['bubble_LTotal'].value = bubble_props.L_total
            params['bubble_Tavg'].value = bubble_props.Tavg
            params['bubble_T_arr'].value = bubble_props.T_arr
            params['bubble_v_arr'].value = bubble_props.v_arr
            params['bubble_r_arr'].value = bubble_props.r_arr
            params['bubble_n_arr'].value = bubble_props.n_arr

            # Get updated T0 from bubble calculation
            T0 = bubble_props.T_rgoal
            params['T0'].value = T0
            Tavg = bubble_props.Tavg

            # Calculate shell structure
            shell_structure.shell_structure(params)
        else:
            # Early phase: use initial T0
            Tavg = T0

        # Update sound speed
        c_sound = operations.get_soundspeed(Tavg, params)
        params['c_sound'].value = c_sound

        # =============================================================================
        # Update R1 cache and get current feedback
        # =============================================================================

        L_mech_now = float(L_mech_interp(t_now))
        v_mech_now = float(v_mech_interp(t_now))

        # Update R1 for this segment
        R1 = r1_cache.update(t_now, R2, Eb, L_mech_now, v_mech_now)
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

        # Update params for this segment (these are read by ODE but not modified)
        params['Lmech_total'].value = L_mech_now
        params['v_mech_total'].value = v_mech_now
        params['R1'].value = R1
        params['Pb'].value = Pb

        # =============================================================================
        # Define segment time span
        # =============================================================================

        dt_segment = DT_SEGMENT
        if loop_count == 0:
            # Smaller first segment
            dt_segment = DT_MIN_SEGMENT

        t_segment_end = min(t_now + dt_segment, tfinal)
        t_span = (t_now, t_segment_end)

        # =============================================================================
        # Integrate segment with solve_ivp
        # =============================================================================

        y0 = np.array([R2, v2, Eb])

        # Set up event for cloud boundary
        def cloud_event(t, y):
            return radius_exceeds_cloud_event(t, y, params)
        cloud_event.terminal = False
        cloud_event.direction = -1

        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda t, y: get_ODE_Edot_pure(t, y, params, R1),
                t_span=t_span,
                y0=y0,
                method='LSODA',
                events=[cloud_event],
                rtol=1e-6,
                atol=1e-9,
                max_step=dt_segment,
            )
        except Exception as e:
            logger.error(f"solve_ivp failed at t={t_now:.6e}: {e}")
            results.termination_reason = f"solver_error: {e}"
            break

        if not sol.success:
            logger.warning(f"solve_ivp did not succeed: {sol.message}")
            # Try to continue with partial results if available
            if len(sol.t) == 0:
                results.termination_reason = f"solver_failed: {sol.message}"
                break

        # =============================================================================
        # Extract results from this segment
        # =============================================================================

        # Get final state from segment
        R2 = float(sol.y[0, -1])
        v2 = float(sol.y[1, -1])
        Eb = float(sol.y[2, -1])
        t_now = float(sol.t[-1])

        # Check for events
        if len(sol.t_events) > 0 and sol.t_events[0].size > 0:
            logger.info(f"Shell reached cloud boundary at t={sol.t_events[0][0]:.6e}")

        # =============================================================================
        # Update params after successful segment
        # =============================================================================

        # This is the ONLY place where params is mutated during integration
        update_params_after_segment(t_now, R2, v2, Eb, params, R1)

        # Get values for results storage
        mShell = params['shell_mass'].value
        Pb = params['Pb'].value

        # Store final point of this segment
        results.append(t_now, R2, v2, Eb, T0, R1, mShell, Pb)

        # =============================================================================
        # Update history arrays
        # =============================================================================

        params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
        params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
        params['array_R1'].value = np.concatenate([params['array_R1'].value, [R1]])
        params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
        params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
        params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])

        # Save snapshot
        params.save_snapshot()

        # Update loop counter
        loop_count += 1

    # =============================================================================
    # Finalize results
    # =============================================================================

    results.final_time = t_now

    if results.termination_reason is None:
        if R2 >= rCloud:
            results.termination_reason = "reached_cloud_boundary"
        elif t_now >= tfinal:
            results.termination_reason = "reached_tfinal"
        else:
            results.termination_reason = "unknown"

    logger.info(f"Energy phase completed: {results.termination_reason}")
    logger.info(f"  Final time: {t_now:.6e} Myr")
    logger.info(f"  Final R2: {R2:.6e} pc")
    logger.info(f"  Segments: {segment_count}")

    return results


# =============================================================================
# Wrapper for backward compatibility
# =============================================================================

def run_energy_compat(params):
    """
    Backward-compatible wrapper that matches original function signature.

    Calls run_energy() internally but updates params dict as original code did.

    Parameters
    ----------
    params : ParameterDict
        Parameter dictionary
    """
    results = run_energy(params)

    # Update params with final values (for compatibility)
    arrays = results.get_arrays()

    if len(arrays['t']) > 0:
        params['t_now'].value = arrays['t'][-1]
        params['R2'].value = arrays['R2'][-1]
        params['v2'].value = arrays['v2'][-1]
        params['Eb'].value = arrays['Eb'][-1]
        params['T0'].value = arrays['T0'][-1]
        params['R1'].value = arrays['R1'][-1]
        params['shell_mass'].value = arrays['mShell'][-1]

    return
