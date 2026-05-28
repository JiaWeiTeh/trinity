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
from src.sps.update_feedback import get_current_sps_feedback

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

    feedback = get_current_sps_feedback(t_now, params)
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
    # Follows the same compute → save → ODE pattern as phases 1b/1c/2.
    # =============================================================================

    continueWeaver = True

    while R2 < rCloud and (TFINAL_ENERGY_PHASE - t_now) > DT_EXIT_THRESHOLD and continueWeaver:

        # Define segment time span
        t_segment_end = min(t_now + SEGMENT_DURATION, TFINAL_ENERGY_PHASE)

        logger.debug(f'Segment: t={t_now:.6e} to {t_segment_end:.6e} Myr')

        # =============================================================================
        # 1. Update params with current state
        # =============================================================================
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = Eb
        params['T0'].value = T0

        # =============================================================================
        # 2. Get feedback
        # =============================================================================
        feedback = get_current_sps_feedback(t_now, params)
        updateDict(params, feedback)

        # =============================================================================
        # 3. Compute bubble structure (always, not conditional on loop_count)
        # =============================================================================
        bubble_data = bubble_luminosity_modified.get_bubbleproperties_pure(params)
        updateDict(params, bubble_data)

        T0 = bubble_data.bubble_T_r_Tb
        params['T0'].value = T0
        Tavg = bubble_data.bubble_Tavg
        R1 = bubble_data.R1
        Pb = bubble_data.Pb
        params['R1'].value = R1
        params['Pb'].value = Pb

        logger.debug('bubble complete (modified)')

        # =============================================================================
        # 3b. Compute shell mass BEFORE shell structure so that the shell
        #     termination condition uses the current R2's swept-up mass
        #     rather than the previous iteration's stale value.
        # =============================================================================
        mShell = mass_profile.get_mass_profile(R2, params, return_mdot=False)
        params['shell_mass'].value = mShell

        # =============================================================================
        # 3c. Compute shell structure
        # =============================================================================
        shell_data = shell_structure_modified.shell_structure_pure(params)
        updateDict(params, shell_data)
        logger.debug('shell complete (modified)')

        # Compute P_HII from Strömgren ionization balance in shell (n_IF_Str)
        n_IF_Str = shell_data.n_IF_Str
        if params['include_PHII'].value and n_IF_Str > 0:
            P_HII = 2.0 * n_IF_Str * params['k_B'].value * params['TShell_ion'].value
        else:
            P_HII = 0.0
        params['P_HII'].value = P_HII
        F_HII = 4.0 * np.pi * R2**2 * P_HII
        params['F_HII'].value = F_HII

        # Calculate sound speed
        c_sound = operations.get_soundspeed(Tavg, params)
        params['c_sound'].value = c_sound

        # =============================================================================
        # 5. Compute forces and diagnostics
        # =============================================================================
        snapshot_for_forces = energy_phase_ODEs_modified.create_ODE_snapshot(params, shell_data)
        ode_result = energy_phase_ODEs_modified.compute_derived_quantities(
            t_now, [R2, v2, Eb], snapshot_for_forces, params
        )
        if ode_result.F_grav is not None:
            params['F_grav'].value = ode_result.F_grav
        if ode_result.F_ion_in is not None:
            params['F_ion_in'].value = ode_result.F_ion_in
        if ode_result.F_HII is not None:
            params['F_HII'].value = ode_result.F_HII
        if ode_result.F_ram is not None:
            params['F_ram'].value = ode_result.F_ram
        if ode_result.F_rad is not None:
            params['F_rad'].value = ode_result.F_rad
        if ode_result.P_HII is not None:
            params['P_HII'].value = ode_result.P_HII
        if ode_result.P_drive is not None:
            params['P_drive'].value = ode_result.P_drive
        if ode_result.P_ram is not None:
            params['P_ram'].value = ode_result.P_ram
        if ode_result.press_HII_in is not None:
            params['press_HII_in'].value = ode_result.press_HII_in
        if ode_result.shell_mass is not None:
            params['shell_mass'].value = ode_result.shell_mass
        if ode_result.shell_massDot is not None:
            params['shell_massDot'].value = ode_result.shell_massDot
        params['F_ram_wind'].value = feedback.pdot_W
        params['F_ram_SN'].value = feedback.pdot_SN

        # ζ diagnostic (Lancaster+2025)
        _Qi_zeta   = params['Qi'].value
        _alphaB    = params['caseB_alpha'].value
        _k_B       = params['k_B'].value
        _T_ion     = params['TShell_ion'].value
        _mu_ion    = params['mu_ion'].value
        _R2_now    = R2
        _rCloud_z  = rCloud
        _pdot_W    = feedback.pdot_W
        if _R2_now < _rCloud_z:
            from src.cloud_properties import density_profile as _dp
            _n_amb = float(np.atleast_1d(
                _dp.get_density_profile(np.array([_R2_now]), params)
            )[0])
        else:
            _n_amb = params['nISM'].value
        if _Qi_zeta > 0.0 and _n_amb > 0.0:
            _R_St = (3.0 * _Qi_zeta /
                     (4.0 * np.pi * _alphaB * _n_amb**2))**(1.0 / 3.0)
        else:
            _R_St = np.inf
        _c_i2 = _k_B * _T_ion / _mu_ion
        _rho_amb = _n_amb * params['mu_atom'].value
        if _pdot_W > 0.0 and _rho_amb > 0.0 and _c_i2 > 0.0:
            _R_eq = np.sqrt(_pdot_W /
                            (4.0 * np.pi * _rho_amb * _c_i2))
        else:
            _R_eq = 0.0
        _zeta = _R_eq / _R_St if (np.isfinite(_R_St) and _R_St > 0.0) else 0.0
        params['zeta'].value = _zeta
        if _zeta < 0.5:
            logger.debug(f'ζ={_zeta:.3f} < 0.5: PIR-dominated, n_IF_Str active '
                         f'(n_IF={params["n_IF"].value:.3e}, '
                         f'n_IF_Str={params["n_IF_Str"].value:.3e})')

        # =============================================================================
        # 6. Save snapshot BEFORE ODE — all values consistent at t_now
        # =============================================================================
        params.save_snapshot()

        # =============================================================================
        # 7. Create ODE snapshot and integrate
        # =============================================================================
        snapshot = energy_phase_ODEs_modified.create_ODE_snapshot(params, shell_data)

        y0 = [R2, v2, Eb]

        def ode_func(t, y):
            return energy_phase_ODEs_modified.get_ODE_Edot_pure(t, y, snapshot, params)

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
            t_segment_end = t_now + SEGMENT_DURATION / 10
            solution = scipy.integrate.solve_ivp(
                ode_func,
                t_span=(t_now, t_segment_end),
                y0=y0,
                method='RK23',
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
                return
            break

        # =============================================================================
        # 8. Extract new state and update local variables
        # =============================================================================
        R2_new, v2_new, Eb_new = solution.y[:, -1]
        t_new = solution.t[-1]

        logger.debug(f'solve_ivp: {len(solution.t)} steps, final t={t_new:.6e}')

        # Handle early phase approximation switch
        if loop_count == 0 and params['EarlyPhaseApproximation'].value:
            params['EarlyPhaseApproximation'].value = False
            logger.info('Switching to no approximation')

        t_now = t_new
        R2 = R2_new
        v2 = v2_new
        Eb = Eb_new

        logger.debug(f'Phase values: t: {t_now:.6e}, R2: {R2:.6e}, v2: {v2:.6e}, Eb: {Eb:.6e}, T0: {T0:.2e}')

        # Update params with new state (for next iteration's bubble/shell)
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = Eb

        loop_count += 1

    # =========================================================================
    # Phase-boundary reconciliation snapshot.
    # Recompute derived properties (Pb, shell structure) with the post-ODE
    # state so the snapshot is fully consistent.  A bare save_snapshot()
    # would save stale derived values AND block the next phase's correct
    # first snapshot via the duplicate guard.
    # =========================================================================
    try:
        feedback_final = get_current_sps_feedback(t_now, params)
        updateDict(params, feedback_final)
        R1_f = scipy.optimize.brentq(
            get_bubbleParams.get_r1, 1e-3 * R2, R2,
            args=([feedback_final.Lmech_total, Eb, feedback_final.v_mech_total, R2])
        )
        Pb_f = get_bubbleParams.bubble_E2P(Eb, R2, R1_f, params['gamma_adia'].value)
        params['R1'].value = R1_f
        params['Pb'].value = Pb_f
        mShell_f = mass_profile.get_mass_profile(R2, params, return_mdot=False)
        params['shell_mass'].value = mShell_f
        shell_f = shell_structure_modified.shell_structure_pure(params)
        updateDict(params, shell_f)
        params.save_snapshot()
    except Exception as e:
        logger.warning(f"Phase-boundary reconciliation failed: {e}")

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
    feedback = get_current_sps_feedback(t_now, params)
    updateDict(params, feedback)

    # Build events using centralized module
    # Energy phase events: cloud_boundary (phase ending), min_radius, velocity_runaway
    ode_events = build_energy_phase_events(params)

    # Create snapshot (needs current shell_props for F_rad)
    shell_data = shell_structure_modified.shell_structure_pure(params)
    updateDict(params, shell_data)
    snapshot = energy_phase_ODEs_modified.create_ODE_snapshot(params, shell_data)

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
