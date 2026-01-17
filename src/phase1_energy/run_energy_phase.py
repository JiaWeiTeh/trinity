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
    
    # TODO: add CLOUDY
    # the energy-driven phase


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
    
    # =============================================================================
    # Now, we begin Energy-driven calculations (Phase 1)
    # =============================================================================
    # -----------
    # Step1: Obtain initial feedback values
    # -----------
    [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total] = get_currentSB99feedback(t_now, params)
    
    updateDict(params, ['Qi', 'Li', 'Ln', 'Lbol', 'Lmech_W', 'Lmech_SN', 'Lmech_total', 'pdot_W', 'pdot_SN', 'pdot_total', 'pdotdot_total', 'v_mech_total'],
               [Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total])
    
    
    # -----------
    # Solve equation for inner radius of the inner shock.
    # -----------
    # [pc]
    
    # Calculate initial R1 and Pb
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * R2, R2,
        args=([Lmech_total, Eb, v_mech_total, R2])
    )

    # -----------
    # Solve equation for mass and pressure within bubble
    # -----------
    Msh0 = mass_profile.get_mass_profile(R2, params, return_mdot=False)
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

    logger.info('Energy phase initialization:')
    logger.info(f'  Inner discontinuity (R1): {R1:.6e} pc')
    logger.info(f'  Initial shell mass: {Msh0:.6e} Msun')
    logger.info(f'  Initial bubble pressure: {Pb:.6e} Msun/pc/Myr^2')

    # Update params with initial values
    params['Pb'].value = Pb
    params['R1'].value = R1

    
    # CONTINUE here
    # LWind = params['LWind'].value
    # vWind = params['vWind'].value
    # pWindDot = params['pWindDot'].value
    # pWindDotDot = params['pWindDotDot'].value
    
    
    # old code: mom_phase
    immediately_to_momentumphase = False
    # record the initial Lw0. This value will be changed in the loop. 
    # old code: Lw_old
    Lw_previous = Lmech_total


    continueWeaver = True
    # how many times had the main loop being ran?
    loop_count = 0

    # Lets make this phase at max 16 Myr according to Eq4, Rahner thesis pg44.
    # actually lets make it less than 1e4 yr (sedov taylor cooling time i think)
    # in previous code this is 3e-3 (~3000 yr)
    # Myr
    # tfinal = 1e-2
    tfinal = 3e-3
    
    # dt_min = 1e-6
    dt_min = 1e-6


# =============================================================================
#   this energy phase persists if:
#       1) radius is less than cloud radius
#       2) total time change is less than 1e-5/-4 Myr (~1e4yr is Sedov Taylor cooling).
# =============================================================================

    while R2 < rCloud and\
        (tfinal - t_now) > 1e-4 and\
            continueWeaver:

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
        
        
        # calculate bubble structure and shell structure?
        # no need to calculate them at very early times, since we say that no bubble or shell is being
        # created at this time. Old code: structure_switch
        
        # an initially small dt value if it is the first loop.
        calculate_bubble_shell = loop_count > 0
        
        # eventhough we have an array of time t_arr, we dont have to do bubble calculation
        # every single time. We just assume that since the timesteps are small enough, 
        # we can approximate the bubble as having the same properties
        # Since also this phase cooling is not important, its probably a good approximation
        # 
        
        # something is wrong here: the loop seems to record t twice, thus
        # creating a duplicate which causes problems in extrapolation in mshelldot.
        # tsteps = 50
        tsteps = 30
        t_arr = np.arange(t_now, t_now +  (dt_min * tsteps), dt_min)[1:]  
        
        print('t_arr is this', t_arr)
        
        # =============================================================================
        # Calculate shell structure
        # =============================================================================
        
        if calculate_bubble_shell:
            
            
            _ = bubble_luminosity.get_bubbleproperties(params)
            
            # update this here instead of in bubble_luminosity so that 
            # T0 will not be overwrite when we are dealing with phase1b.
            T0 = params['bubble_T_r_Tb'].value
            params['T0'].value = T0
            Tavg = params['bubble_Tavg'].value

            print('\n\nFinish bubble\n\n')

            shell_structure.shell_structure(params)
            
            print('\n\nShell structure calculated.\n\n')
            
        elif not calculate_bubble_shell:
            print('bubble and shell not calculated.')
            # TODO: redefine these values so that they are more physically similar to the environments
            # TODO: what about those that are inherited by values from previous entries in dictionary?
            # make sure to also initialise them properly.
            Tavg = T0
            
        
        c_sound = operations.get_soundspeed(Tavg, params)
        params['c_sound'].value = c_sound
            
        # update
        
        # =============================================================================
        # call ODE solver to solve for equation of motion (r, v (rdot), Eb). 
        # =============================================================================
        
        
        
        
        
        
            # # METHOD 1 ODE solver
            
            # # radiation pressure coupled to the shell
            # # f_absorbed_ion calculated from shell_structure.
            # # F_rad = f_absorbed_ion * Lbol / params['c_au'].value
            
            # y0 = [R2, v2, Eb, T0]
            
            # # call ODE solver
            # psoln = scipy.integrate.odeint(energy_phase_ODEs.get_ODE_Edot, y0, t_arr, args=(params,))
            
            # # if calculate_bubble_shell:
            # #     import sys
            # #     sys.exit()
            
            # # [pc]
            # r_arr = psoln[:,0] 
            # v_arr = psoln[:, 1]
            # Eb_arr = psoln[:, 2] 


        # METHOD 2 own equations, this solves problem with dictionary
        
        
        r_arr = []
        v_arr = []
        Eb_arr = []
        
        
        print('R2', R2)
        print('v2', v2)
        print('Eb', Eb)
        print('T0', T0)
        
        
        for ii, time in enumerate(t_arr):
            
            # new inputs
            y = [R2, v2, Eb, T0]
            
            # print('original y', y)
        
            try:
                params['t_next'].value = t_arr[ii+1]
            except:
                params['t_next'].value = time + dt_min
                
                
            # print('time', time)
                    
            rd, vd, Ed, Td =  energy_phase_ODEs.get_ODE_Edot(y, time, params)
            
            if ii != (len(t_arr) - 1):
                R2 += rd * dt_min 
                v2 += vd * dt_min 
                Eb += Ed * dt_min 
                T0 += Td * dt_min 
                
                print('new rd values in run_energy_phase')
                print('rd', rd)
                print('vd', vd)
                print('Ed', Ed)
                print('Td', Td)
                print('R2', R2)
                print('v2', v2)
                print('Eb', Eb)
                print('T0', T0)
                
            r_arr.append(R2)
            v_arr.append(v2)
            Eb_arr.append(Eb)
            
            
            if ii == 10:
                params['EarlyPhaseApproximation'].value = False
                print('\n\n\n\n\n\n\nswitch to no approximation\n\n\n\n\n\n')
            
            # if ii == 20:
            #     import sys
            #     sys.exit('loop test done')
            
            
        # print(v_arr)

            
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
        
        # -- new, record only once
        params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t_now]])
        params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
        params['array_R1'].value = np.concatenate([params['array_R1'].value, [params['R1'].value]])
        params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
        params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
        
        # get shell mass
        mShell_arr = mass_profile.get_mass_profile(r_arr, params,
                                                    return_mdot = False)
        
        Msh0 = mShell_arr[-1] # shell mass
        
        params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [Msh0]])
        
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
        #     Pb = get_bubbleParams.pRam(R2, LWind, vWind)
        # # else, if we are continuing this loop and staying in energy
        # else:
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       1e-3 * R2, R2, 
                       args=([Lmech_total, Eb, v_mech_total, R2]))
        # bubble pressure
        Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        
        updateDict(params, 
                    ['R1', 'R2', 'v2', 'Eb', 't_now', 'Pb', 'shell_mass'], 
                    [R1, R2, v2, Eb, t_now, Pb, Msh0])
            
        # renew constants
        # Lw_previous = LWind
        
        
        # update loop counter
        loop_count += 1
        
        # print(params)
        
        # print('t_now', t_now)
        # print('R2', R2)
        # print('v2 array', v2)
        
        # if loop_count > 1:
        #     import sys
        #     sys.exit('energy loop check')
        
        # import sys
        # sys.exit('energy loop check')
        
        pass

    # import sys
    # sys.exit('completed energy early.')

    return 
    

# TODO: 
# 1) Add fragmentation mechanics (fragmentation time)
#    Gravitational instability
#    Another way to estimate when fragmentation occurs: Raylor-Taylor instabilities, see Baumgartner 2013, eq. (48)
# 2) Add cover fraction.
# 3) Add stop events. Now we assume it will never stop here, but who knows.



