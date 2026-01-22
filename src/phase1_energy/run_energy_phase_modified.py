#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified energy phase with adaptive ODE solver.

This module implements the energy-driven phase using scipy.integrate.solve_ivp
with adaptive stepping, replacing the manual Euler integration in run_energy_phase.py.

Key differences from run_energy_phase.py:
1. Uses solve_ivp with RK45 adaptive solver instead of manual Euler
2. Segment-based integration: short segments with params updates only after success
3. Pure ODE functions that don't mutate params during integration
4. Uses dataclass returns from bubble_luminosity_modified

The dictionary mutation problem:
- Original ODE functions write to params during evaluation
- Adaptive solvers take trial steps that can be rejected
- Rejected trial steps leave params in corrupted state
- Solution: Pure ODE functions + update params only after successful segments

@author: Jia Wei Teh
"""

import numpy as np
import scipy.integrate
import scipy.optimize
import logging

import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure_modified as shell_structure_modified
import src.cloud_properties.mass_profile as mass_profile
import src.phase1_energy.energy_phase_ODEs_modified as energy_phase_ODEs_modified
import src.bubble_structure.bubble_luminosity_modified as bubble_luminosity_modified
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src._input.dictionary import updateDict
import src._functions.unit_conversions as cvt
from src.sb99.update_feedback import get_currentSB99feedback

# Import centralized event functions
from src.phase_general.phase_events import (
    build_energy_phase_events,
    check_event_termination,
    apply_event_result,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

TFINAL_ENERGY_PHASE = 3e-3  # Myr - max duration (~3000 years)
SEGMENT_DURATION = 3e-5  # Myr - duration of each integration segment (~30 years)
DT_EXIT_THRESHOLD = 1e-4  # Myr - exit when this close to tfinal
COOLING_UPDATE_INTERVAL = 5e-2  # Myr - recalculate cooling every 50k years
RTOL = 1e-6  # Relative tolerance for solve_ivp
ATOL = 1e-9  # Absolute tolerance for solve_ivp


def run_energy(params):
    """
    Run the energy-driven phase (Phase 1) using adaptive ODE integration.

    This implements the Weaver+77 bubble expansion model with solve_ivp
    instead of manual Euler integration. The key improvement is that
    ODE functions are pure (no dictionary mutations) and params is only
    updated after successful integration segments.

    Parameters
    ----------
    params : DescribedDict
        Main parameter dictionary
    """
    logger.info('Starting modified energy phase with adaptive solver')

    # =============================================================================
    # Initialization
    # =============================================================================

    t_now = params['t_now'].value
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    rCloud = params['rCloud'].value

    # =============================================================================
    # Initial feedback and bubble parameters
    # =============================================================================

    feedback = get_currentSB99feedback(t_now, params)
    updateDict(params, feedback)

    # Calculate initial R1 and Pb
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-4 * R2, R2,
        args=([feedback.Lmech_total, Eb, feedback.v_mech_total, R2])
    )

    mShell = mass_profile.get_mass_profile(R2, params, return_mdot=False)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

    logger.info('Energy phase initialization (modified):')
    logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
    logger.info(f'  Initial shell mass: {mShell:.6e} Msun')
    logger.info(f'  Initial bubble pressure: {Pb:.6e} Msun/pc/Myr^2')

    params['Pb'].value = Pb
    params['R1'].value = R1

    loop_count = 0

    # =============================================================================
    # Build events for safe termination
    # =============================================================================

    ode_events = build_energy_phase_events(params)

    # =============================================================================
    # Cooling structure (computed periodically)
    # =============================================================================

    if np.abs(params['t_previousCoolingUpdate'] - params['t_now']) > COOLING_UPDATE_INTERVAL:
        cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
        params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
        params['cStruc_heating_nonCIE'].value = heating_nonCIE
        params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
        params['t_previousCoolingUpdate'].value = params['t_now'].value

    # =============================================================================
    # Main loop: segment-based integration
    # =============================================================================

    continueWeaver = True

    while R2 < rCloud and (TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD and continueWeaver:

        calculate_bubble_shell = loop_count > 0

        # Define segment time span
        t_segment_end = min(t_now + SEGMENT_DURATION, TFINAL_ENERGY_PHASE)

        logger.debug(f'Segment: t={t_now:.6e} to {t_segment_end:.6e} Myr')

        # =============================================================================
        # Calculate bubble and shell structure (between segments, not during ODE)
        # =============================================================================

        if calculate_bubble_shell:
            # Use modified bubble_luminosity that returns dataclass
            bubble_data = bubble_luminosity_modified.get_bubbleproperties_pure(params)

            # Update params with bubble properties
            params['bubble_LTotal'].value = bubble_data.bubble_LTotal
            params['bubble_T_r_Tb'].value = bubble_data.bubble_T_r_Tb
            params['bubble_Tavg'].value = bubble_data.bubble_Tavg
            params['bubble_mass'].value = bubble_data.bubble_mass
            params['bubble_L1Bubble'].value = bubble_data.bubble_L1Bubble
            params['bubble_L2Conduction'].value = bubble_data.bubble_L2Conduction
            params['bubble_L3Intermediate'].value = bubble_data.bubble_L3Intermediate
            params['bubble_v_arr'].value = bubble_data.bubble_v_arr
            params['bubble_T_arr'].value = bubble_data.bubble_T_arr
            params['bubble_dTdr_arr'].value = bubble_data.bubble_dTdr_arr
            params['bubble_r_arr'].value = bubble_data.bubble_r_arr
            params['bubble_n_arr'].value = bubble_data.bubble_n_arr
            params['bubble_dMdt'].value = bubble_data.bubble_dMdt
            params['R1'].value = bubble_data.R1
            params['Pb'].value = bubble_data.Pb
            params['bubble_r_Tb'].value = bubble_data.bubble_r_Tb

            logger.info('bubble complete (modified)')

            T0 = params['bubble_T_r_Tb'].value
            params['T0'].value = T0
            Tavg = params['bubble_Tavg'].value

            # Compute shell structure
            shell_data = shell_structure_modified.shell_structure_pure(params)
            # Update params with shell properties
            params['shell_n0'].value = shell_data.shell_n0
            params['rShell'].value = shell_data.rShell
            params['isDissolved'].value = shell_data.isDissolved
            params['shell_fAbsorbedIon'].value = shell_data.shell_fAbsorbedIon
            params['shell_fAbsorbedNeu'].value = shell_data.shell_fAbsorbedNeu
            params['shell_fAbsorbedWeightedTotal'].value = shell_data.shell_fAbsorbedWeightedTotal
            params['shell_fIonisedDust'].value = shell_data.shell_fIonisedDust
            params['shell_thickness'].value = shell_data.shell_thickness
            params['shell_nMax'].value = shell_data.shell_nMax
            params['shell_tauKappaRatio'].value = shell_data.shell_tauKappaRatio
            params['shell_F_rad'].value = shell_data.shell_F_rad
            params['shell_grav_r'].value = shell_data.shell_grav_r
            params['shell_grav_phi'].value = shell_data.shell_grav_phi
            params['shell_grav_force_m'].value = shell_data.shell_grav_force_m
            logger.info('shell complete (modified)')
        else:
            Tavg = T0

        # Calculate sound speed
        c_sound = operations.get_soundspeed(Tavg, params)
        params['c_sound'].value = c_sound

        # =============================================================================
        # Create frozen snapshot for ODE integration
        # =============================================================================

        snapshot = energy_phase_ODEs_modified.create_ODE_snapshot(params)

        # =============================================================================
        # Solve ODE using adaptive solver (solve_ivp)
        # =============================================================================

        y0 = [R2, v2, Eb]

        # Define ODE function wrapper for solve_ivp (t, y order)
        def ode_func(t, y):
            return energy_phase_ODEs_modified.get_ODE_Edot_pure(t, y, snapshot, params)

        # Integrate segment with adaptive solver
        solution = scipy.integrate.solve_ivp(
            ode_func,
            t_span=(t_now, t_segment_end),
            y0=y0,
            method='RK45',
            events=ode_events,
            rtol=RTOL,
            atol=ATOL,
            dense_output=True
        )

        if not solution.success:
            logger.warning(f'solve_ivp failed: {solution.message}')
            # Fallback: take smaller segment
            t_segment_end = t_now + SEGMENT_DURATION / 10
            solution = scipy.integrate.solve_ivp(
                ode_func,
                t_span=(t_now, t_segment_end),
                y0=y0,
                method='RK23',  # More robust method
                events=ode_events,
                rtol=RTOL * 10,
                atol=ATOL * 10
            )

        # Check if an event terminated the integration
        event_result = check_event_termination(solution, ode_events)
        if event_result.triggered:
            logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr")
            apply_event_result(params, event_result, event_result.t, event_result.y,
                              state_keys=['R2', 'v2', 'Eb'])
            if event_result.is_simulation_ending:
                return  # Exit immediately for simulation-ending events
            # For phase-ending events (cloud_boundary), exit the loop normally
            break

        # Extract final state
        R2_new, v2_new, Eb_new = solution.y[:, -1]
        t_new = solution.t[-1]

        logger.debug(f'solve_ivp: {len(solution.t)} steps, final t={t_new:.6e}')

        # =============================================================================
        # Handle early phase approximation switch
        # =============================================================================

        if loop_count == 0 and params['EarlyPhaseApproximation'].value:
            # After first successful segment, disable approximation
            params['EarlyPhaseApproximation'].value = False
            logger.info('Switching to no approximation')

        # =============================================================================
        # Update params with results (only after successful integration)
        # =============================================================================

        # Compute derived quantities at final state
        ode_result = energy_phase_ODEs_modified.compute_derived_quantities(
            t_new, [R2_new, v2_new, Eb_new], snapshot, params
        )

        # Update state variables
        t_now = t_new
        R2 = R2_new
        v2 = v2_new
        Eb = Eb_new

        logger.info(f'Phase values: t: {t_now:.6e}, R2: {R2:.6e}, v2: {v2:.6e}, Eb: {Eb:.6e}, T0: {T0:.2e}')

        # Get shell mass
        mShell = mass_profile.get_mass_profile(R2, params, return_mdot=False)

        # Save snapshot
        params.save_snapshot()

        # Update feedback
        feedback = get_currentSB99feedback(t_now, params)
        updateDict(params, feedback)

        # Update R1 and Pb
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-3 * R2, R2,
            args=([feedback.Lmech_total, Eb, feedback.v_mech_total, R2])
        )
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

        # Update params dictionary
        updateDict(params,
                   ['R1', 'R2', 'v2', 'Eb', 't_now', 'Pb', 'shell_mass'],
                   [R1, R2, v2, Eb, t_now, Pb, mShell])

        # Also update forces from ODE result
        if ode_result.F_grav is not None:
            params['F_grav'].value = ode_result.F_grav
        if ode_result.F_ion_in is not None:
            params['F_ion_in'].value = ode_result.F_ion_in
        if ode_result.F_ion_out is not None:
            params['F_ion_out'].value = ode_result.F_ion_out
        if ode_result.F_ram is not None:
            params['F_ram'].value = ode_result.F_ram
        if ode_result.F_rad is not None:
            params['F_rad'].value = ode_result.F_rad
        if ode_result.shell_mass is not None:
            params['shell_mass'].value = ode_result.shell_mass
        if ode_result.shell_massDot is not None:
            params['shell_massDot'].value = ode_result.shell_massDot

        loop_count += 1

    logger.info(f'Modified energy phase complete: {loop_count} segments')
    return


# =============================================================================
# Alternative: Continuous integration with events
# =============================================================================

def run_energy_continuous(params):
    """
    Alternative implementation using continuous integration with event detection.

    This version integrates the entire phase at once using solve_ivp's event
    detection to stop at key points (bubble/shell structure updates).

    Note: This is more efficient but less flexible than segment-based approach.
    """
    logger.info('Starting continuous energy phase integration')

    t_now = params['t_now'].value
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value

    # Initial setup
    feedback = get_currentSB99feedback(t_now, params)
    updateDict(params, feedback)

    # Build events using centralized module
    # Energy phase events: cloud_boundary (phase ending), min_radius, velocity_runaway
    ode_events = build_energy_phase_events(params)

    # Create snapshot
    snapshot = energy_phase_ODEs_modified.create_ODE_snapshot(params)

    def ode_func(t, y):
        return energy_phase_ODEs_modified.get_ODE_Edot_pure(t, y, snapshot, params)

    # Integrate entire phase
    solution = scipy.integrate.solve_ivp(
        ode_func,
        t_span=(t_now, TFINAL_ENERGY_PHASE),
        y0=[R2, v2, Eb],
        method='RK45',
        events=ode_events,
        rtol=RTOL,
        atol=ATOL,
        dense_output=True
    )

    logger.info(f'Continuous integration: {len(solution.t)} steps')

    # Check if an event terminated the integration
    event_result = check_event_termination(solution, ode_events)
    if event_result.triggered:
        logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr")
        # Apply event result to params
        apply_event_result(params, event_result, event_result.t, event_result.y,
                          state_keys=['R2', 'v2', 'Eb'])
    else:
        # Update final state normally
        R2_final, v2_final, Eb_final = solution.y[:, -1]
        t_final = solution.t[-1]
        updateDict(params,
                   ['R2', 'v2', 'Eb', 't_now'],
                   [R2_final, v2_final, Eb_final, t_final])

    return solution
