#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy-driven phase with adaptive ODE solver.

This module implements the energy-driven phase using scipy.integrate.solve_ivp
with adaptive stepping (rather than manual Euler integration).

Integration approach:
1. Uses solve_ivp with RK45 adaptive solver instead of manual Euler
2. Segment-based integration: short segments with params updates only after success
3. Pure ODE functions that don't mutate params during integration
4. Uses dataclass returns from bubble_luminosity

The dictionary mutation problem:
- Original ODE functions write to params during evaluation
- Adaptive solvers take trial steps that can be rejected
- Rejected trial steps leave params in corrupted state
- Solution: Pure ODE functions + update params only after successful segments

@author: Jia Wei Teh
"""

import numpy as np
import scipy.integrate
import logging

import trinity.bubble_structure.get_bubbleParams as get_bubbleParams
import trinity.shell_structure.shell_structure as shell_structure
import trinity.cloud_properties.mass_profile as mass_profile
import trinity.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import trinity.bubble_structure.bubble_luminosity as bubble_luminosity
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity._functions.operations as operations
from trinity._input.dictionary import updateDict
import trinity._functions.unit_conversions as cvt
import trinity._output.terminal_prints as terminal_prints
from trinity._output.simulation_end import SimulationEndCode
from trinity.sps.update_feedback import get_current_sps_feedback

# Import centralized event functions
from trinity.phase_general.phase_events import (
    build_energy_phase_events,
    check_event_termination,
    apply_event_result,
    make_energy_collapse_event,
    update_collapse_radius,
    add_solver_flag,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

TFINAL_ENERGY_PHASE = 3e-3  # Myr - max duration (~3000 years)
PHASE1A_WINDOW_FACTOR = 3.0  # phase-1a length / ramp window when the window is stretched (= 3e-3 / 1e-3, the shipped ratio)
SEGMENT_DURATION = 3e-5  # Myr - fixed-segment fallback, used when phase1a_segFrac = 0
DT_EXIT_THRESHOLD = 1e-4  # Myr - exit when this close to tfinal
COOLING_UPDATE_INTERVAL = 5e-2  # Myr - recalculate cooling every 50k years
COOLING_UPDATE_INTERVAL_STRETCHED = 5e-3  # Myr - in-loop refresh for a stretched 1a; = phase 1b's interval
RTOL = 1e-6  # Relative tolerance for solve_ivp
ATOL = 1e-9  # Absolute tolerance for solve_ivp


def _handoff_spent_bubble(params, t_now, R2, v2, why, channel="energy_to_momentum"):
    """Hand a spent phase-1a bubble to momentum instead of ending the run.

    Also used for a turnaround (channel 'velocity_sign_change', ruled 2026-10-01:
    v2 < 0 goes straight to momentum whatever Eb is left). The Eb thrown away is
    kept in Eb_handoff and the channel in transition_channel.

    Phase-1a counterpart of 1b's energy_to_momentum routing
    (run_energy_implicit_phase.classify_energy_collapse): once Eb is gone, R1 -> R2
    and the shell is already pushed by the wind's ram pressure, so the momentum
    phase is the right model. Sets Eb to 1b's ENERGY_HANDOFF_FLOOR (1c is then
    skipped and phase 2 starts at once), clears the end flags the collapse
    event set, and marks energy_handoff_1a so phase 1b does not integrate a
    bubble that no longer exists. Returns the new Eb.
    docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md, 2026-09-30.
    """
    from trinity.phase1b_energy_implicit.run_energy_implicit_phase import ENERGY_HANDOFF_FLOOR
    params['t_now'].value = t_now
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb_handoff'].value = params['Eb'].value
    params['Eb'].value = ENERGY_HANDOFF_FLOOR
    params['energy_handoff_1a'].value = True
    params['transition_channel'].value = channel
    params['EndSimulationDirectly'].value = False
    params['SimulationEndReason'].value = ''
    params['SimulationEndCode'].value = None
    params['isCollapse'].value = False
    logger.warning(
        f"Phase 1a hand-off to momentum at t={t_now:.6e} Myr ({why}; R2={R2:.4f} pc, "
        f"v2={v2:.3e} pc/Myr, Eb={params['Eb_handoff'].value:.3e} dropped to the floor): "
        f"routing to momentum via 1c, phase 1b skipped [{channel}]."
    )
    return ENERGY_HANDOFF_FLOOR


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
    logger.info('Starting energy phase with adaptive solver')

    # =============================================================================
    # Initialization
    # =============================================================================

    t_now = params['t_now'].value
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    rCloud = params['rCloud'].value
    # Segment schedule: segments are a fixed fraction of the bubble's age, so the
    # per-segment freezing of the driving terms carries the same relative staleness
    # at every object scale. A fixed duration cannot: 30 yr is a small step for a
    # GMC and spans the whole free-streaming->Weaver relaxation of a compact HII
    # region. See docs/dev/phase1a-init/FINDINGS.md.
    segFrac = params['phase1a_segFrac'].value
    tSF = params['tSF'].value

    # R1 switch-on window, and with it the length of phase 1a, scale with the seed
    # age once the seed is older than 1/3 kyr; younger seeds keep the shipped 1e-3
    # and 3e-3 Myr exactly. docs/dev/switchon-successor/PLAN.md, 2026-09-30.
    params['energy_handoff_1a'].value = False
    dt_switchon = get_bubbleParams.switchon_window(t_now - tSF)
    params['dt_switchon'].value = dt_switchon
    if dt_switchon == get_bubbleParams.DT_SWITCHON:
        t_end_1a = TFINAL_ENERGY_PHASE
    else:
        # A stretched phase 1a stops at stop_t: unlike the shipped 3 kyr it can be long.
        t_end_1a = max(TFINAL_ENERGY_PHASE, tSF + PHASE1A_WINDOW_FACTOR * dt_switchon)
        if params['stop_t'].value is not None:
            t_end_1a = min(t_end_1a, params['stop_t'].value)
        logger.info(f'Old seed (dt_phase0={t_now - tSF:.4e} Myr): R1 ramp over {dt_switchon:.4e} Myr, '
                    f'phase 1a to t={t_end_1a:.4e} Myr')

    # =============================================================================
    # Initial feedback and bubble parameters
    # =============================================================================

    feedback = get_current_sps_feedback(t_now, params)
    updateDict(params, feedback)

    # Calculate initial R1 and Pb
    R1 = get_bubbleParams.solve_R1(R2, Eb, feedback.Lmech_total, feedback.v_mech_total)

    mShell = mass_profile.get_mass_profile(R2, params, return_mdot=False)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

    logger.info('Energy phase initialization:')
    logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
    logger.info(f'  Initial shell mass: {mShell:.6e} Msun')
    logger.info(f'  Initial bubble pressure: {Pb*cvt.Pb_au2_KcmInv:.6e} K cm⁻³ (P/k_B)')

    params['Pb'].value = Pb
    params['R1'].value = R1

    logger.info(terminal_prints.format_state(params, label="energy phase entry"))

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

    while R2 < rCloud and (t_end_1a - t_now) > DT_EXIT_THRESHOLD and continueWeaver:

        # Define segment time span
        dt_segment = segFrac * (t_now - tSF)
        if dt_segment <= 0:  # phase1a_segFrac=0 (fixed-segment fallback), or a degenerate age
            dt_segment = SEGMENT_DURATION
        t_segment_end = min(t_now + dt_segment, t_end_1a)

        logger.debug(f'Segment: t={t_now:.6e} to {t_segment_end:.6e} Myr')

        # =============================================================================
        # 1. Update params with current state
        # =============================================================================
        params['t_now'].value = t_now
        params['R2'].value = R2
        params['v2'].value = v2
        params['Eb'].value = Eb
        params['T0'].value = T0
        update_collapse_radius(params, R2)

        # Refresh the non-CIE cooling cube on phase 1b's interval. The cube was
        # set at 1a entry (t_now = t0), and the shipped phase 1a ends at 3e-3 Myr,
        # so this cannot fire unless phase 1a was stretched (old seeds).
        if t_now - params['t_previousCoolingUpdate'].value > COOLING_UPDATE_INTERVAL_STRETCHED:
            cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
            params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
            params['cStruc_heating_nonCIE'].value = heating_nonCIE
            params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
            params['t_previousCoolingUpdate'].value = t_now

        # =============================================================================
        # 2. Get feedback
        # =============================================================================
        feedback = get_current_sps_feedback(t_now, params)
        updateDict(params, feedback)

        # =============================================================================
        # 3. Compute bubble structure (always, not conditional on loop_count)
        # =============================================================================
        # A failed structure solve ends phase 1a early and hands the state to 1b,
        # which has a rescue ladder (and hands a bubble it cannot solve to momentum).
        # Until 2026-10-01 this ended the run as ENERGY_COLLAPSED "Eb -> 0", but the
        # pilot_v5 logs show it firing on GROWING bubbles: all 7 such runs (5e9 Msun,
        # n 1e5, 0.5-1.7 kyr, Eb ~6e53 erg) died on a scipy LSODA error, now retried
        # with Radau in bubble_luminosity. Flagged in solver_flags.
        # docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md
        try:
            bubble_data = bubble_luminosity.get_bubbleproperties_pure(params)
        except (ValueError, RuntimeError, bubble_luminosity.BubbleSolverError) as e:
            add_solver_flag(params, '1a_structure_failure')
            logger.warning(
                f"Phase 1a bubble solve failed at t={t_now:.6e} Myr "
                f"(Eb={Eb:.3e}, R2={R2:.4f} pc; {type(e).__name__}: {e}): "
                f"ending phase 1a early, phase 1b continues [1a_structure_failure]."
            )
            break
        updateDict(params, bubble_data)

        T0 = bubble_data.bubble_T_r_Tb
        params['T0'].value = T0
        Tavg = bubble_data.bubble_Tavg
        R1 = bubble_data.R1
        Pb = bubble_data.Pb
        params['R1'].value = R1
        params['Pb'].value = Pb

        logger.debug('bubble complete')

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
        shell_data = shell_structure.shell_structure_pure(params)
        updateDict(params, shell_data)
        logger.debug('shell complete')

        # Compute P_HII: photoionised pressure (get_bubbleParams.get_phii_c3c) -- exactly 0.0 while confined
        n_IF_Str = shell_data.n_IF_Str
        if params['include_PHII'].value and get_bubbleParams.phii_is_active(params, shell_data):
            # Photoionised pressure is a regime switch, not the capped Stromgren
            # relabelling of Pb; see get_bubbleParams.get_phii_c3c.
            P_HII = get_bubbleParams.get_phii(params, shell_data)
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
        snapshot_for_forces = energy_phase_ODEs.create_ODE_snapshot(params, shell_data)
        ode_result = energy_phase_ODEs.compute_derived_quantities(
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
        if ode_result.bubble_Leak is not None:
            params['bubble_Leak'].value = ode_result.bubble_Leak
        params['F_ram_wind'].value = feedback.pdot_W
        params['F_ram_SN'].value = feedback.pdot_SN

        # =============================================================================
        # 6. Save snapshot BEFORE ODE — all values consistent at t_now
        # =============================================================================
        shell_structure.record_validity(params)
        params.save_snapshot()

        # =============================================================================
        # 6b. Transition-trigger parity with phase 1b (cooling_balance).
        # A violently cooling cloud can reach the energy->momentum cooling balance
        # WITHIN this fixed ~3000-yr early phase; without a check here it would either
        # wait for the 1a->1b boundary or, if cooling drives Eb<=0 first, hit the
        # collapse routing below. Evaluated at the consistent pre-ODE snapshot with the
        # SAME formula as run_energy_implicit_phase.py (Lgain=Lmech_total,
        # Lloss=effective_Lloss(Lcool=bubble_LTotal, leak)). No-op for healthy bubbles
        # (early cooling is negligible, ratio ~1 >> threshold) -> byte-identical (G0).
        from trinity.phase1b_energy_implicit.run_energy_implicit_phase import parse_transition_triggers
        from trinity.phase1b_energy_implicit.get_betadelta import effective_Lloss_from_params
        _active_triggers = parse_transition_triggers(params['transition_trigger'].value)
        if 'cooling_balance' in _active_triggers:
            _Lgain = feedback.Lmech_total
            _leak = ode_result.bubble_Leak if ode_result.bubble_Leak is not None else 0.0
            _Lloss = effective_Lloss_from_params(params, bubble_data.bubble_LTotal, _leak, _Lgain)
            _thr = params['phaseSwitch_LlossLgain'].value
            _thr = _thr if _thr else 0.05
            if _Lgain > 0 and (_Lgain - _Lloss) / _Lgain < _thr:
                logger.info(
                    f"Phase 1a cooling_balance reached (Lloss/Lgain > {1 - _thr:.2f}) at "
                    f"t={t_now:.6e} Myr -> ending early phase (hands off via 1b -> 1c -> momentum)."
                )
                break

        # =============================================================================
        # 7. Create ODE snapshot and integrate
        # =============================================================================
        snapshot = energy_phase_ODEs.create_ODE_snapshot(params, shell_data)

        y0 = [R2, v2, Eb]

        def ode_func(t, y):
            return energy_phase_ODEs.get_ODE_Edot_pure(t, y, snapshot, params)

        # In-band energy-collapse guard, rebuilt per segment because its
        # reference is this segment's starting Eb. The Eb<=0 check below runs
        # only *between* segments and Eb never actually reaches 0 -- it pins at
        # a small positive value on a stiff manifold the explicit solver cannot
        # cross, so without this the run grinds instead of stopping.
        # Inert on healthy runs (Eb grows every segment). See
        # docs/dev/phase1a-stiffness/PLAN.md.
        segment_events = ode_events + [make_energy_collapse_event(Eb)]

        solution = scipy.integrate.solve_ivp(
            ode_func,
            t_span=(t_now, t_segment_end),
            y0=y0,
            method='RK45',
            events=segment_events,
            rtol=RTOL,
            atol=ATOL,
            dense_output=True
        )

        if not solution.success:
            logger.warning(f'solve_ivp failed: {solution.message}')
            t_segment_end = t_now + dt_segment / 10
            solution = scipy.integrate.solve_ivp(
                ode_func,
                t_span=(t_now, t_segment_end),
                y0=y0,
                method='RK23',
                events=segment_events,
                rtol=RTOL * 10,
                atol=ATOL * 10
            )

        # Check if an event terminated the integration
        event_result = check_event_termination(solution, segment_events)
        if event_result.triggered:
            logger.info(f"Event '{event_result.name}' triggered at t={event_result.t:.6e} Myr")
            apply_event_result(params, event_result, event_result.t, event_result.y,
                              state_keys=['R2', 'v2', 'Eb'])
            if event_result.name == 'velocity_sign':
                # Turnaround: straight to momentum, whatever Eb is left (ruled
                # 2026-10-01, an approximation; Eb_handoff keeps what was dropped).
                t_now = float(event_result.t)
                R2 = float(event_result.y[0])
                v2 = float(event_result.y[1])
                Eb = _handoff_spent_bubble(params, t_now, R2, v2, "shell turned around",
                                           channel="velocity_sign_change")
                break
            if event_result.name == 'energy_collapse':
                # Spent bubble: continue in momentum instead of ending the run. The
                # locals still hold the segment start; move them to the event state
                # so the reconciliation snapshot below is consistent.
                t_now = float(event_result.t)
                R2 = float(event_result.y[0])
                v2 = float(event_result.y[1])
                Eb = _handoff_spent_bubble(params, t_now, R2, v2, "Eb fell to 1e-3 of segment start")
                break
            if event_result.is_simulation_ending:
                return
            break

        # =============================================================================
        # 8. Extract new state and update local variables
        # =============================================================================
        R2_new, v2_new, Eb_new = solution.y[:, -1]
        t_new = solution.t[-1]

        logger.debug(f'solve_ivp: {len(solution.t)} steps, final t={t_new:.6e}')

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

        # Energy-driven collapse in the early (1a) phase: a massive/dense cloud can
        # lose the bubble's thermal energy (PdV work on a heavy shell, radiation
        # pushing the shell past v_wind/2, or radiative cooling) faster than the wind
        # resupplies it, so Eb falls through zero. The energy-driven model is then
        # invalid (it would drive R1->R2 and divide-by-zero -> Eb=nan). A finite
        # collapse is routed to the momentum phase exactly as 1b routes it
        # (classify_energy_collapse); only a non-finite Eb stops the run. Routing was
        # deferred as "rare" until 2026-09-30; the v4 survey grid had 7,968 such runs.
        # See docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md.
        if np.isfinite(Eb) and Eb <= 0:
            Eb = _handoff_spent_bubble(params, t_now, R2, v2, "Eb <= 0 after a segment")
            break
        if not np.isfinite(Eb):
            # reason text kept verbatim from before the handoff (read by tools/bubble_fate.py)
            params['EndSimulationDirectly'].value = True
            params['SimulationEndReason'].value = (
                "Energy-driven bubble collapsed: Eb fell to <= 0 "
                "(energy-driven phase no longer self-sustains)"
            )
            params['SimulationEndCode'].value = SimulationEndCode.ENERGY_COLLAPSED.code
            logger.warning(
                f"Energy-driven bubble collapsed at t={t_now:.6e} Myr "
                f"(Eb={Eb:.3e}, R2={R2:.4f} pc): stopping run cleanly."
            )
            break

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
        R1_f = get_bubbleParams.solve_R1(R2, Eb, feedback_final.Lmech_total,
                                         feedback_final.v_mech_total)
        Pb_f = get_bubbleParams.bubble_E2P(Eb, R2, R1_f, params['gamma_adia'].value)
        params['R1'].value = R1_f
        params['Pb'].value = Pb_f
        mShell_f = mass_profile.get_mass_profile(R2, params, return_mdot=False)
        params['shell_mass'].value = mShell_f
        shell_f = shell_structure.shell_structure_pure(params)
        updateDict(params, shell_f)
        shell_structure.record_validity(params)
        params.save_snapshot()
    except Exception as e:
        logger.warning(f"Phase-boundary reconciliation failed: {e}")

    logger.info(f'Energy phase complete: {loop_count} segments')
    logger.info(terminal_prints.format_state(params, label="energy phase exit"))
    return
