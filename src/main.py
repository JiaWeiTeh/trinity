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
from src._output.simulation_end import write_simulation_end



from src.phase1_energy import run_energy_phase_modified
from src.phase1b_energy_implicit import run_energy_implicit_phase_modified
from src.phase1c_transition import run_transition_phase_modified
from src.phase2_momentum import run_momentum_phase_modified
import src._output.terminal_prints as terminal_prints
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src._input.dictionary import DescribedItem, DescribedDict, COOLING_PHASE_KEYS

# Initialize logger for this module
logger = logging.getLogger(__name__)


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
    logger.info("=" * 60)
    logger.info("TRINITY Simulation Starting")
    logger.info("=" * 60)
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

    # Step 2: Obtain parameters from Starburst99
    logger.info("Step 2: Loading Starburst99 stellar feedback data...")
    # Scaling factor for cluster masses. Though this might only be accurate for
    # high mass clusters (~>1e5) in which the IMF is fully sampled.
    f_mass = params['mCluster'] / params['SB99_mass']
    logger.debug(f"SB99 mass scaling factor: {f_mass:.4f}")
    # Get SB99 data and interpolation functions.
    SB99_data = read_SB99.read_SB99(f_mass, params)
    SB99f = read_SB99.get_interpolation(SB99_data)
    # TODO:
    # if tSF != 0.: we would actually need to shift the feedback parameters by tSF
    # update
    params['SB99_data'].value = SB99_data
    params['SB99f'].value = SB99f
    logger.info("SB99 data loaded and interpolation functions created")
    
    # Step 3: get cooling structure for CIE (since it is non time dependent).
    logger.info("Step 3: Loading CIE cooling curve...")
    # Values for non-CIE cooling curve will be calculated along the simulation, since it depends on time evolution.

    # for metallicity, here we need to take care of both CIE and nonCIE part.

    # maybe move this to read_params
    # if params['metallicity'].value != 1:
    #     sys.exit('Need to implement non-solar metallicity.')

    # get path to library
    # See example_pl.param for more information.
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
    # These two are currently not being needed. 
    # =============================================================================
    # create density law for cloudy
    # TODO: add CLOUDY support in the future.
    
    # =============================================================================
    # Begin simulation.
    # =============================================================================
    logger.info("=" * 60)
    logger.info("Initialization complete. Starting bubble expansion simulation...")
    logger.info("=" * 60)

    run_expansion(params)

    # Log completion
    enddatetime = datetime.datetime.now()
    elapsed = enddatetime - startdatetime
    logger.info("=" * 60)
    logger.info("TRINITY Simulation Complete")
    logger.info(f"End time: {enddatetime}")
    logger.info(f"Total elapsed time: {elapsed}")
    logger.info("=" * 60)

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
    # R2 = initial outer bubble/shell radius (pc)
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

    logger.info("-" * 60)
    logger.info("PHASE 1a: Energy-driven phase (constant cooling)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering energy driven phase (constant cooling)')

    phase1a_starttime = datetime.datetime.now()

    run_energy_phase_modified.run_energy(params)

    phase1a_endtime = datetime.datetime.now()
    phase1a_elapsed = phase1a_endtime - phase1a_starttime

    logger.info(f"Phase 1a complete. Duration: {phase1a_elapsed}")
    logger.debug(f"  Final R2 = {params['R2'].value:.6e} pc")
    logger.debug(f"  Final v2 = {params['v2'].value:.6e} pc/Myr")
    
    
    # record
    # try:
    #     params.flush()
    # except:
    #     pass

    # sys.exit('done with phase 1a')

    # ------------
    # checkout load_dict_halfway.py to understand how to load dictionaries
    # halfway through the simulation.
    # ------------

    # =============================================================================
    # Phase 1b: implicit energy phase
    # =============================================================================

    params['current_phase'].value = 'implicit'

    logger.info("-" * 60)
    logger.info("PHASE 1b: Energy-driven phase (adaptive cooling)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering energy driven phase (adaptive cooling)')

    phase1b_starttime = datetime.datetime.now()


    run_energy_implicit_phase_modified.run_phase_energy(params)
        

    phase1b_endtime = datetime.datetime.now()
    phase1b_elapsed = phase1b_endtime - phase1b_starttime
    logger.info(f"Phase 1b complete. Duration: {phase1b_elapsed}")

    # record
    # try:
    #     params.flush()
    # except:
    #     pass

    # make a function that interpolates density so that it goes from top to end of cloud.

    
    # =============================================================================
    # Phase 1c: transition phase
    # =============================================================================

    logger.info("-" * 60)
    logger.info("PHASE 1c: Transition phase (energy -> momentum)")
    logger.info("-" * 60)
    terminal_prints.phase('Entering transition phase (decreasing energy before momentum)')

    params['current_phase'].value = 'transition'

    if params['EndSimulationDirectly'].value == False:
        phase1c_starttime = datetime.datetime.now()

        run_transition_phase_modified.run_phase_transition(params)

        phase1c_endtime = datetime.datetime.now()
        phase1c_elapsed = phase1c_endtime - phase1c_starttime
        logger.info(f"Phase 1c complete. Duration: {phase1c_elapsed}")
    else:
        logger.warning("EndSimulationDirectly=True, skipping transition phase")

    # try:
    #     params.flush()
    # except:
    #     pass

    # Since cooling is not needed anymore after this phase, we reset values.
    # COOLING_PHASE_KEYS contains all cooling-related parameters that can be cleared.
    logger.debug("Resetting cooling-related parameters (no longer needed)...")
    params.reset_keys(COOLING_PHASE_KEYS)


    # =============================================================================
    # Phase 2: momentum phase
    # =============================================================================

    logger.info("-" * 60)
    logger.info("PHASE 2: Momentum-driven phase")
    logger.info("-" * 60)
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
        logger.warning("EndSimulationDirectly=True, skipping momentum phase")

    # Flush parameters to disk
    try:
        params.flush()
        logger.debug("Parameters flushed to disk")
    except Exception as e:
        logger.warning(f"Could not flush parameters: {e}")

    logger.info("All expansion phases complete")

    return 


def expansion_next(tStart, ODEpar, SB99_data_old, SB99f_old, mypath, cloudypath, ii_coll):

    return


