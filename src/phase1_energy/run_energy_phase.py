#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 23:16:58 2022

@author: Jia Wei Teh

"""
# libraries
import numpy as np
import scipy.interpolate
#--
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
import src.cloud_properties.mass_profile as mass_profile
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.bubble_structure.bubble_luminosity as bubble_luminosity
import src.cooling.non_CIE.read_cloudy as non_CIE
import src._functions.operations as operations
from src._input.dictionary import updateDict
import src._functions.unit_conversions as cvt
from src.sb99.update_feedback import get_currentSB99feedback
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# Constants 
# =============================================================================

TFINAL_ENERGY_PHASE = 3e-3  # Myr - max duration (~3000 years)
DT_MIN = 1e-6  # Myr - minimum timestep for manual method (not needed with odeint)
DT_EXIT_THRESHOLD = 1e-4  # Myr - exit when this close to tfinal
COOLING_UPDATE_INTERVAL = 5e-2  # Myr - recalculate cooling every 50k years
TIMESTEPS_PER_LOOP = 30  # Timesteps per outer loop


def run_energy(params):
    
    # This function runs the energy-driven phase (Phase 1), implementing
    # the Weaver+77 bubble expansion model. 
    # This phase is run until the first ~3000 year. Then it should go into implicit phase
    # since cooling becomes important. 
    
    # This phase includes bubble structure calculation, shell structure calculation.
    # This function solves ODEs to calculate bubble expansion. 
    
    # TODO: add CLOUDY

    # =============================================================================
    # Initialization
    # =============================================================================
    # here we grab parameter from the main dictioanry
    t_now = params['t_now'].value #time
    R2 = params['R2'].value #outer bubble radius
    v2 = params['v2'].value #bubble velocity
    Eb = params['Eb'].value #bubble energy
    T0 = params['T0'].value #temperature at xi_Tb (T_rgoal in bubble_luminosity)
    rCloud = params['rCloud'].value #cloud radius
    t_neu = params['TShell_neu'].value #temperature neutral
    t_ion = params['TShell_ion'].value #temperature ionised
    
    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # -----------
    # Step1: Obtain initial feedback values
    # -----------
    # we take starburst99 feedback values here
    [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total] = get_currentSB99feedback(t_now, params)
    # update them to dictionary.
    updateDict(params, ['Qi', 'Li', 'Ln', 'Lbol', 'Lmech_W', 'Lmech_SN', 'Lmech_total', 'pdot_W', 'pdot_SN', 'pdot_total', 'pdotdot_total', 'v_mech_total'],
               [Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total])
    
    
    # -----------
    # Solve equation for inner radius of the inner shock.
    # -----------
    # [pc]
    
    # Calculate initial R1 and Pb
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-4 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # -----------
    # Solve equation for mass and pressure within bubble
    # -----------
    mShell = mass_profile.get_mass_profile(R2, params, return_mdot=False)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

    logger.info('Energy phase initialization:')
    logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
    logger.info(f'  Initial shell mass: {mShell:.6e} Msun')
    logger.info(f'  Initial bubble pressure: {Pb:.6e} Msun/pc/Myr^2')

    # Update params with initial values
    params['Pb'].value = Pb
    params['R1'].value = R1

    # initialisation
    continueWeaver = True
    # how many times had the main loop being ran?
    loop_count = 0

    # What is the maximum [Myr] this phase can run? (~<sedov taylor cooling time 1e4yr)
    tfinal = 3e-3
    # What is the smallest timestep [Myr]?
    dt_min = 1e-6

    # =============================================================================
    # Prelude: prepare cooling structures so that it doesnt have to run every loop.
    # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
    # =============================================================================
    
    if np.abs(params['t_previousCoolingUpdate'] - params['t_now']) > 5e-2:
        # recalculate non-CIE
        cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
        # save
        params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
        params['cStruc_heating_nonCIE'].value = heating_nonCIE
        params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
        # update current value
        params['t_previousCoolingUpdate'].value = params['t_now'].value
        
        
    while R2 < rCloud and (tfinal - t_now) > 1e-4 and continueWeaver:

        # no need to calculate bubble structure and shell structure at very early times, since we say that no bubble or shell is being
        # created at this time.  
        calculate_bubble_shell = loop_count > 0
        
        # eventhough we have an array of time t_arr, we dont have to do bubble calculation
        # every single time. We just assume that since the timesteps are small enough, 
        # we can approximate the bubble as having the same properties
        
        tsteps = 30
        t_arr = np.arange(t_now, t_now +  (dt_min * tsteps), dt_min)[1:]  
        
        logger.debug(f't_arr is this {t_arr[0]}-{t_arr[-1]} Myr')
        
        # =============================================================================
        # Calculate shell structure
        # =============================================================================
        
        if calculate_bubble_shell:
            
            # compuete bubble structure
            _ = bubble_luminosity.get_bubbleproperties(params)
            logger.info('bubble complete')
            
            T0 = params['bubble_T_r_Tb'].value
            params['T0'].value = T0
            Tavg = params['bubble_Tavg'].value

            # compute shell structure
            shell_structure.shell_structure(params)
            logger.info('shell complete')
            
            
        # TODO:
            # bubbleData = bubble_luminosity.get_bubbleproperties(params)
            # updateDict(params, bubbleData) <- this will automatically retrieve key,value pair and update them to params dict.
            # same goes with 
            # shellData = shell_structure.shell_structure(params)
            
        elif not calculate_bubble_shell:
            # if bubble and shell is not calculated, average temperature will just be T0.
            Tavg = T0
            
        # calculate sound speed [Myr/pc]
        c_sound = operations.get_soundspeed(Tavg, params)
        # update
        params['c_sound'].value = c_sound
            
        # =============================================================================
        # call ODE solver to solve for equation of motion (r, v (rdot), Eb). 
        # =============================================================================
        
        # This is an Euler approach, which is not very good. 
        # However, it is used because the dictionary `params` are passed in. The functions
        # within mutates it, and this is terrible because scipy ODE solver 
        # is adaptive and will create 'trial' states: these can go backwards in steps, duplicate steps. 
        # Since functions within is time sensitive and is based on previous steps, this will
        # create nonsense and non-reproducible results.
        
        # initialise list
        r_arr = []
        v_arr = []
        Eb_arr = []
        
        for ii, time in enumerate(t_arr):
            
            # new inputs
            y = [R2, v2, Eb, T0]
            
            # print('original y', y)
        
            try:
                params['t_next'].value = t_arr[ii+1]
            except:
                params['t_next'].value = time + dt_min
                
                
            # print('time', time)
                    
            rd, vd, Ed=  energy_phase_ODEs.get_ODE_Edot(y, time, params)
            
            if ii != (len(t_arr) - 1):
                R2 += rd * dt_min 
                v2 += vd * dt_min 
                Eb += Ed * dt_min 
                
            r_arr.append(R2)
            v_arr.append(v2)
            Eb_arr.append(Eb)
            
            
            if ii == 10 and params['EarlyPhaseApproximation'].value == True:
                params['EarlyPhaseApproximation'].value = False
                print('\n\n\n\n\n\n\nswitch to no approximation\n\n\n\n\n\n')
            
        # =============================================================================
        # Here, we perform checks to see if we should continue the branch (i.e., increasing steps)
        # =============================================================================
            
        #----------------------------
        # 2. When does fragmentation occur?
        #----------------------------
            # -----------
            # Option1 : Gravitational instability
            # -----------
            
            
            # TODO
            # -----------
            # Option2 : Rayleigh-Taylor isntability (not yet implemented)
            # -----------    
            
            
        # which temperature?
        # this is obtained from shell_structure
        if params['shell_fAbsorbedIon'].value < 0.99:
            T_shell = t_neu
        else:
            T_shell = t_ion
        # sound speed
        c_sound = operations.get_soundspeed(T_shell, params)
        params['c_sound'].value = c_sound
    
        
        
        # =============================================================================
        # Prepare for next loop
        # =============================================================================

        # new initial values
        # time
        t_now = t_arr[-1]
        # shell radius
        R2 = r_arr[-1]
        # shell velocity
        v2 = v_arr[-1]
        # bubble energy
        Eb = Eb_arr[-1]
        
        logger.info(f'Phase values: t: {t_now}, R2: {R2}, v2: {v2}, Eb: {Eb}, T0: {T0}')
        
        
        
        # get shell mass
        mShell_arr = mass_profile.get_mass_profile(r_arr, params,
                                                    return_mdot = False)
        
        mShell = mShell_arr[-1] # shell mass
        
        # -- end new
        
        # save here
        print('saving snapshot')
        params.save_snapshot()
        
        'pdot_SN', 'pdot_total', 'pdotdot_total'
    
        [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total] = get_currentSB99feedback(t_now, params)
        # Extract derived values from params for backward compatibility
        Lmech_total = params['Lmech_total'].value
        v_mech_total = params['v_mech_total'].value
        pdot_total = params['pdot_total'].value
        pdotdot_total = params['pdotdot_total'].value

        # # if we are going to the momentum phase next, do not have to 
        # # calculate the discontinuity for the next loop
        # if immediately_to_momentumphase:
        #     R1 = R2 # why?
        #     # bubble pressure
        #     Pb = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
        # # else, if we are continuing this loop and staying in energy
        # else:
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       1e-3 * R2, R2, 
                       args=([Lmech_total, Eb, v_mech_total, R2]))
        # bubble pressure
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        
        updateDict(params, 
                    ['R1', 'R2', 'v2', 'Eb', 't_now', 'Pb', 'shell_mass'], 
                    [R1, R2, v2, Eb, t_now, Pb, mShell])
            
        # update loop counter
        loop_count += 1
        
    return 
    

# TODO: 
# 1) Add fragmentation mechanics (fragmentation time)
#    Gravitational instability
#    Another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
# 2) Add cover fraction.
# 3) Add stop events. Now we assume it will never stop here, but who knows.



