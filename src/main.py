#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:46:31 2022

@author: Jia Wei Teh

This script contains a wrapper that initialises the expansion of
shell.
"""

# libraries
import logging
import numpy as np
import datetime
import scipy
import src._functions.unit_conversions as cvt

#--
from src.sb99 import read_SB99
from src.phase0_init import (get_InitCloudProp, get_InitPhaseParam)
from src._output.simulation_end import write_simulation_end, SimulationEndCode
from src.phase1_energy import run_energy_phase_modified
from src.phase1b_energy_implicit import run_energy_implicit_phase_modified
from src.phase1c_transition import run_transition_phase_modified
from src.phase2_momentum import run_momentum_phase_modified
import src._output.terminal_prints as terminal_prints
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src._input.dictionary import DescribedItem, DescribedDict, COOLING_PHASE_KEYS

# Initialize logger for this module
logger = logging.getLogger(__name__)


# Threshold above which a stop_r value is considered "comfortably above"
# rCloud — meaning stop_at_rCloud_nSnap will almost certainly fire first.
# Below this multiple of rCloud, the two termination conditions race.
_STOP_R_RCLOUD_RACE_FACTOR = 1.5


def _check_stop_r_rCloud_interaction(nSnap_rCloud, stop_r, rCloud):
    """
    Decide whether stop_r conflicts with stop_at_rCloud_nSnap.

    rCloud is derived from the cloud properties at init time, so a user
    setting stop_r in a .param file may accidentally pick a value
    smaller than rCloud and silently disable their stop_at_rCloud_nSnap
    termination.  Both knobs are valid independently — this is a UX
    warning, not an error.

    Returns
    -------
    (level, message) : tuple
        level is "warning", "info", or None.  message is the log
        text (or None when no log is needed).
    """
    if nSnap_rCloud is None or stop_r is None:
        return (None, None)

    if stop_r <= rCloud:
        return (
            "warning",
            f"stop_at_rCloud_nSnap={nSnap_rCloud} but stop_r={stop_r} pc "
            f"<= rCloud={rCloud:.4f} pc; stop_r will terminate the run "
            f"before stop_at_rCloud_nSnap can fire.  Increase stop_r "
            f"or set it to None to use stop_at_rCloud_nSnap."
        )

    if stop_r <= _STOP_R_RCLOUD_RACE_FACTOR * rCloud:
        return (
            "info",
            f"stop_at_rCloud_nSnap={nSnap_rCloud} and stop_r={stop_r} pc "
            f"are close to rCloud={rCloud:.4f} pc (within "
            f"{_STOP_R_RCLOUD_RACE_FACTOR}x); whichever fires first "
            f"will terminate the run."
        )

    return (None, None)


def start_expansion(params):
    """
    This wrapper takes in the parameters and feed them into smaller
    functions.

    Parameters
    ----------
    params : Object
        An object describing TRINITY parameters.

    Returns
    -------
    None.

    """
    
    # =============================================================================
    # Step 0: Preliminary
    # =============================================================================

    # TODO: put this in read_param, and make it depend on param file.
    # Note: Logging is configured in run.py before this function is called.
    # This fallback ensures logging works if main.py is called directly.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    # Record timestamp
    startdatetime = datetime.datetime.now()
    logger.info("=" * 16 + " TRINITY Simulation Starting " + "=" * 15)
    logger.info(f"Start time: {startdatetime}")
    terminal_prints.phase0(startdatetime)
    
    # =============================================================================
    # A: Initialising cloud properties.
    # =============================================================================

    # Step 1: Obtain initial cloud properties
    logger.info("Step 1: Initializing cloud properties...")
    get_InitCloudProp.get_InitCloudProp(params)
    logger.debug(f"Cloud radius: {params['rCloud'].value:.4f} pc")
    logger.debug(f"Core density: {params['nCore'].value*cvt.ndens_au2cgs:.4e} cm-3")

    # rCloud is now known; warn the user if stop_r will starve
    # stop_at_rCloud_nSnap of any chance to fire (or race with it).
    _level, _msg = _check_stop_r_rCloud_interaction(
        params['stop_at_rCloud_nSnap'].value,
        params['stop_r'].value,
        params['rCloud'].value,
    )
    if _level == "warning":
        logger.warning(_msg)
    elif _level == "info":
        logger.info(_msg)

    # Step 2: Obtain SPS feedback parameters
    # The loader handles both the legacy SB99 grammar (sps_path = def_path)
    # and user-defined sps_path files; see read_SB99 module docstring.
    logger.info("Step 2: Loading SPS stellar feedback data...")
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    f_mass = params['mCluster'] / params['sps_refmass']
    logger.debug(f"SPS mass scaling factor: {f_mass:.4f}")
    # Get SPS data and interpolation functions.
    sps_data = read_SB99.read_SB99(f_mass, params)
    sps_f = read_SB99.get_interpolation(sps_data)
    # TODO:
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    # update
    params['sps_data'].value = sps_data
    params['sps_f'].value = sps_f
    logger.info("SPS data loaded and interpolation functions created")
    
    # Step 3: get cooling structure for CIE (since it is non time dependent).
    logger.info("Step 3: Loading CIE cooling curve...")
    # Values for non-CIE cooling curve will be calculated along the simulation, since it depends on time evolution.

    # for metallicity, here we need to take care of both CIE and nonCIE part.

    cooling_path = params['path_cooling_CIE'].value
    logger.debug(f"Loading cooling curve from: {cooling_path}")
    # unpack from file
    logT, logLambda = np.loadtxt(cooling_path, unpack=True)
    # create interpolation
    cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind='linear')
    # update
    params['cStruc_cooling_CIE_logT'].value = logT
    params['cStruc_cooling_CIE_logLambda'].value = logLambda
    params['cStruc_cooling_CIE_interpolation'].value = cooling_CIE_interpolation

    logger.info(f"Loaded CIE cooling curve (T range: 10^{logT.min():.1f} - 10^{logT.max():.1f} K)")
    
    # =============================================================================
    # Begin simulation.
    # =============================================================================
    logger.info("=" * 5 + " Initialization complete. Starting bubble expansion simulation... " + "=" * 5)

    run_expansion(params)

    # Log completion
    enddatetime = datetime.datetime.now()
    elapsed = enddatetime - startdatetime
    logger.info("=" * 16 + " TRINITY Simulation Complete " + "=" * 15)
    logger.info(f"End time: {enddatetime}")
    logger.info(f"Total elapsed time: {elapsed}")

    # Write simulation end report to file
    try:
        exit_code = write_simulation_end(params)
        logger.info(f"Simulation end report written (exit code: {exit_code})")
    except Exception as e:
        logger.warning(f"Could not write simulation end report: {e}")
        exit_code = 99

    # Write termination debug report (last 2 snapshots with comparison)
    try:
        reason = params['SimulationEndReason'].value if 'SimulationEndReason' in params else "Unknown"
        debug_path = params.write_termination_report(reason=reason)
        if debug_path:
            logger.info(f"Termination debug report written: {debug_path}")
    except Exception as e:
        logger.warning(f"Could not write termination debug report: {e}")


    # ########### STEP 2: In case of recollapse, prepare next expansion ##########################
    # TODO: add loop so that this simulation starts over with old generation of parameter to simulate new starburst environment
    
    return 0
    
    
#%%
    
def run_expansion(params):
    """
    Model evolution of the cloud (both energy- and momentum-phase) until next recollapse or (if no re-collapse) until end of simulation
    """

    # =============================================================================
    # Prep for phases
    # =============================================================================
    # t0 = start time for Weaver phase
    # y0 = [r0, v0, E0, T0]
    # R2 = initial outer bubble radius (= inner shell edge) (pc)
    # v2 = initial velocity (pc/Myr)
    # Eb = initial energy
    # T0 = initial temperature (K)
    logger.info("Computing initial phase parameters (free-streaming -> Weaver transition)...")

    t0, r0, v0, E0, T0 = get_InitPhaseParam.get_y0(params)

    params['t_now'].value = t0
    params['R2'].value = r0
    params['v2'].value = v0
    params['Eb'].value = E0
    params['T0'].value = T0

    # =============================================================================
    # Phase 1a: Energy driven phase.
    # =============================================================================
    
    params['current_phase'].value = 'energy'

    logger.info("-" * 5 + " PHASE 1a: Energy-driven phase (constant cooling) " + "-" * 5)
    terminal_prints.phase('Entering energy driven phase (constant cooling)')

    phase1a_starttime = datetime.datetime.now()

    run_energy_phase_modified.run_energy(params)

    phase1a_endtime = datetime.datetime.now()
    phase1a_elapsed = phase1a_endtime - phase1a_starttime

    logger.info(f"Phase 1a complete. Duration: {phase1a_elapsed}")
    logger.debug(f"  Final R2 = {params['R2'].value:.6e} pc")
    logger.debug(f"  Final v2 = {params['v2'].value:.6e} pc/Myr")

    # stop_at_rCloud_nSnap == 0: terminate now if R2 has reached the cloud edge.
    # The energy-phase reconciliation snapshot already captured R2 = rCloud, and
    # we explicitly do NOT want phases 1b/1c/2 to advance past it.
    nSnap_rCloud = params['stop_at_rCloud_nSnap'].value
    if (nSnap_rCloud is not None and nSnap_rCloud == 0
            and params['R2'].value >= params['rCloud'].value):
        params['EndSimulationDirectly'].value = True
        params['SimulationEndReason'].value = (
            "Reached cloud edge (stop_at_rCloud_nSnap=0)"
        )
        params['SimulationEndCode'].value = SimulationEndCode.RCLOUD_BOUNDARY.code
        logger.info("stop_at_rCloud_nSnap=0 and R2 >= rCloud at end of phase 1a; "
                    "skipping subsequent phases.")

    # =============================================================================
    # Phase 1b: implicit energy phase
    # =============================================================================

    params['current_phase'].value = 'implicit'

    logger.info("-" * 5 + " PHASE 1b: Energy-driven phase (adaptive cooling) " + "-" * 5)
    terminal_prints.phase('Entering energy driven phase (adaptive cooling)')

    if params['EndSimulationDirectly'].value == False:
        phase1b_starttime = datetime.datetime.now()

        run_energy_implicit_phase_modified.run_phase_energy(params)

        phase1b_endtime = datetime.datetime.now()
        phase1b_elapsed = phase1b_endtime - phase1b_starttime
        logger.info(f"Phase 1b complete. Duration: {phase1b_elapsed}")
    else:
        logger.info("EndSimulationDirectly=True, skipping implicit phase")

    # =============================================================================
    # Phase 1c: transition phase
    # =============================================================================

    logger.info("-" * 5 + " PHASE 1c: Transition phase (energy -> momentum) " + "-" * 5)
    terminal_prints.phase('Entering transition phase (decreasing energy before momentum)')

    params['current_phase'].value = 'transition'

    if params['EndSimulationDirectly'].value == False:
        phase1c_starttime = datetime.datetime.now()

        run_transition_phase_modified.run_phase_transition(params)

        phase1c_endtime = datetime.datetime.now()
        phase1c_elapsed = phase1c_endtime - phase1c_starttime
        logger.info(f"Phase 1c complete. Duration: {phase1c_elapsed}")
    else:
        logger.info("EndSimulationDirectly=True, skipping transition phase")

    # Since cooling is not needed anymore after this phase, we reset values.
    # COOLING_PHASE_KEYS contains all cooling-related parameters that can be cleared.
    logger.debug("Resetting cooling-related parameters (no longer needed)...")
    params.reset_keys(COOLING_PHASE_KEYS)


    # =============================================================================
    # Phase 2: momentum phase
    # =============================================================================

    logger.info("-" * 5 + " PHASE 2: Momentum-driven phase " + "-" * 5)
    terminal_prints.phase('Entering momentum phase')

    params['current_phase'].value = 'momentum'

    # Eb is inherited from the transition phase (near ENERGY_FLOOR = 1e3).
    # The momentum phase runner sets Eb = 0 at its initialization (line 522).
    # Do NOT set Eb=1 here — it was dead code that caused confusion.
    logger.info(f"Entering momentum phase: Eb={params['Eb'].value:.4e} "
                f"(will be set to 0 by momentum runner)")

    # --- Transition -> Momentum boundary diagnostic ---
    R2_bnd = params['R2'].value
    Lmech_bnd = params['Lmech_total'].value
    v_mech_bnd = params['v_mech_total'].value
    P_ram_bnd = get_bubbleParams.pRam(R2_bnd, Lmech_bnd, v_mech_bnd)
    logger.info(f"T->M boundary check: P_ram={P_ram_bnd:.4e} "
                f"(momentum phase will use this)")

    if params['EndSimulationDirectly'].value == False:
        phase2_starttime = datetime.datetime.now()

        run_momentum_phase_modified.run_phase_momentum(params)

        phase2_endtime = datetime.datetime.now()
        phase2_elapsed = phase2_endtime - phase2_starttime
        logger.info(f"Phase 2 (momentum) complete. Duration: {phase2_elapsed}")
    else:
        logger.info("EndSimulationDirectly=True, skipping momentum phase")

    # Flush parameters to disk
    try:
        params.flush()
        logger.debug("Parameters flushed to disk")
    except Exception as e:
        logger.warning(f"Could not flush parameters: {e}")

    logger.info("All expansion phases complete")

    return 


def expansion_next(tStart, ODEpar, sps_data_old, sps_f_old, mypath, cloudypath, ii_coll):

    return


