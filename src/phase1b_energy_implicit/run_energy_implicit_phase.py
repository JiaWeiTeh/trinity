# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:12:37 2023

@author: Jia Wei Teh

This scritp continues to compute the energy phase, but solving real-time values of beta and delta,
which was omitted and neglected in the previous run_energy_phase (due to low impact during early phases.).

"""


import scipy.optimize
import numpy as np
import astropy.units as u
import sys
import os
#--
import src.phase_general.phase_ODEs as phase_ODEs
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs
import src.cloud_properties.mass_profile as mass_profile

import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.phase1b_energy_implicit.get_betadelta as get_betadelta
import src._functions.unit_conversions as cvt
import src.cooling.non_CIE.read_cloudy as non_CIE
from src.sb99.update_feedback import get_currentSB99feedback
import src.shell_structure.shell_structure as shell_structure
import src._functions.operations as operations
#--


def run_phase_energy(params):

    # TODO: add fragmentation mechanics in events 

    # what is the current outer bubble velocity?
    params['v2'].value = params['cool_alpha'].value * params['R2'].value / (params['t_now'].value) 
    
    #-- theoretical minimum and maximum of this phase
    tmin = params['t_now'].value
    tmax = params['stop_t'].value

        
    # how many timesteps from tmin to tmax? set about 200 timesteps per dex
    nmin = int(200 * np.log10(tmax/tmin))
    # create array for these timesteps
    time_range = np.logspace(np.log10(tmin), np.log10(tmax), nmin)[1:] # [1:] to avoid duplicate starting values
    # what is the dt for this?
    dt = np.diff(time_range)

    # main parameters required for bubble evolution calculation: outer radius, outer velocity, bubble energy, temperature.
    r2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    
    
    # Here, we run through timestep and evolve bubble. If any conditions (events) are 
    # met, we stop the simulation
    
    # initialise stop condition
    stop_condition = False

    for ii, time in enumerate(time_range):
        # do not calculate final element
        if ii == (len(time_range) - 1):
            break
        
        # new inputs
        y = [r2, v2, Eb, T0]
    
        try:
            params['t_next'].value = time_range[ii+1]
        except:
            params['t_next'].value = time + dt
                
        # this functino returns time derivative of parameters, which can be used 
        # to calculate the next loop
        rd, vd, Ed, Td =  ODE_equations(time, y, params)
        
        # collect values to feed into stop condition calculation
        dt_params = [dt[ii], rd, vd, Ed, Td]
            
        # if true, leave loop
        if check_events(params, dt_params):
            stop_condition = True
            break
        
        # otherwise calculate values for next loop
        if ii != (len(time_range) - 1):
            r2 += rd * dt[ii]
            v2 += vd * dt[ii]
            Eb += Ed * dt[ii]
            T0 += Td * dt[ii]
            
    return



# main equation that calculates time derivative of the values
def ODE_equations(t, y, params):
    
    # t [yr], R2 [pc], v2 [pc/yr], Eb [au]
    # --- These are R2, v2, Eb and T0 (Trgoal).
    R2, v2, Eb, T0 = y
    
    # record new values into dictionary
    params['t_now'].value = t
    params['v2'].value = v2
    params['Eb'].value = Eb
    params['T0'].value = T0
    params['R2'].value = R2
    # print into terminal
    print(f'current stage: t:{t}, r:{R2}, v:{v2}, E:{Eb}, T:{T0}')
    
    # ---  
    
    #-- updating values in the loop; make sure to include all values of parameters
    # update new alpha value
    params['cool_alpha'].value = t / R2 * v2
    
    # =============================================================================
    # Prelude: prepare cooling structures so that it doesnt have to run every loop.
    # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
    # =============================================================================
    if np.abs(params['t_previousCoolingUpdate'].value - params['t_now'].value) > 5e-3: # in Myr
        # recalculate non-CIE cooling structure
        cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
        # save into dictionary
        params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
        params['cStruc_heating_nonCIE'].value = heating_nonCIE
        params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
        # update current value
        params['t_previousCoolingUpdate'].value = params['t_now'].value 
        
    # =============================================================================
    # Part 1: find acceleration and velocity
    # =============================================================================
    # get current feedback value
    [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] =  get_currentSB99feedback(t, params)
    # run shell structure calculations
    shell_structure.shell_structure(params)
    # get time derivative of radius and velocity from ode equations
    rd, vd, _, _ = energy_phase_ODEs.get_ODE_Edot(y, t, params)
    
    # here we record radius trends across time, so that we can interpolate in the future
    params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t]])
    params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
    params['array_R1'].value = np.concatenate([params['array_R1'].value, [params['R1'].value]])
    params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
    params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
    # here is where the interpolation is done
    mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                   return_mdot = True, 
                                                   rdot_arr = v2)

    # if mShell or mShell_dot is a single element list, extract value
    if hasattr(mShell, '__len__'):
        if len(mShell) == 1:
            mShell = mShell[0]
        
    if hasattr(mShell_dot, '__len__'):
        if len(mShell_dot) == 1:
            mShell_dot = mShell_dot[0]
    
    # update
    params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])
            
    # =============================================================================
    # Part 2: find beta, delta and convert them to dEdt and dTdt
    # =============================================================================
        
    (beta, delta), result_params = get_betadelta.get_beta_delta_wrapper(params['cool_beta'].value, params['cool_delta'].value, params)
           
    # update
    result_params["cool_beta"].value = beta
    result_params["cool_delta"].value = delta
    print('beta found:', beta, 'delta found', delta)
    print('current state', result_params)
    
    # sound speed for future dEdt calculation.
    result_params['c_sound'].value = operations.get_soundspeed(result_params['bubble_Tavg'].value, result_params)

    #------ convert them to dEdt and dTdt.
    def get_EdotTdot(params_dict
                  ):
        # convert beta and delta to dE/dt and dT/dt.
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       1e-3 * params_dict['R2'].value, params_dict['R2'].value, 
                       args=([params_dict['LWind'].value, 
                              params_dict['Eb'].value, 
                              params_dict['vWind'].value,
                              params_dict['R2'].value,
                              ]))
    
        params_dict['R1'].value = R1
        
        # The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]
        Pb = get_bubbleParams.bubble_E2P(params_dict['Eb'].value,
                                        params_dict['R2'].value, 
                                        params_dict['R1'].value,
                                        params_dict['gamma_adia'].value)
    
        params_dict['Pb'].value = Pb

        # get new beta value
        Edot = get_bubbleParams.beta2Edot(params_dict)
        # get dTdt
        Tdot = get_bubbleParams.delta2dTdt(params_dict['t_now'].value, params_dict['T0'].value, params_dict['cool_delta'].value)
        
        return Edot, Tdot

    
    Ed, Td = get_EdotTdot(result_params)
    
    rd = v2
    
    print('completed a phase in ODE_equations in implicit_phase')
    print(f'rd: {rd}, vd: {vd}, Ed: {Ed}, Td: {Td}')

   
    # save snapshot
    result_params.save_snapshot()
    
    return [rd, vd, Ed, Td]



def check_events(params, dt_params):
    
    [dt, rd, vd, Ed, Td] = dt_params
    
    t_next = params['t_now'].value + dt
    R2_next = params['R2'].value + rd * dt
    v2_next = params['v2'].value + vd * dt
    Eb_next = params['Eb'].value + Ed * dt
    T0_next = params['T0'].value + Td * dt
        
    # =============================================================================
    # Non terminating events
    # =============================================================================
        
    # check if it is collapsing
    if np.sign(v2_next) == -1:
        if R2_next < params['R2'].value:
            print(f'Bubble currently collapsing because the next velocity is {v2_next / cvt.v_kms2au} km/s and current radius is {params["R2"].value} pc.')
            params['isCollapse'].value = True
        else:
            params['isCollapse'].value = False
            
    # =============================================================================
    # Terminating events
    # =============================================================================
    # TODO add this percent thing into params as well
    
    # Main event: when Lcool approaches 10(?) percent of Lgain.
    if (params['bubble_Lgain'].value - params['bubble_Lloss'].value)/params['bubble_Lgain'].value < 0.05:
        print(f"Phase ended because Lloss: {params['bubble_Lloss'].value} is within {(params['bubble_Lgain'].value - params['bubble_Lloss'].value)/params['bubble_Lgain'].value * 100}% of Lgain: {params['bubble_Lgain'].value}")
        
        return True
    
    #--- 1) Stopping time reached
    if t_next > params['stop_t'].value:
        print(f"Phase ended because t reaches {t_next} Myr (> tStop: {params['stop_t'].value}) in the next iteration.")
        params['SimulationEndReason'].value = 'Stopping time reached'
        params['EndSimulationDirectly'].value = True
        return True
    
    #--- 2) Small radius reached during collapse.
    if params['isCollapse'].value == True and R2_next < params['coll_r'].value:
        print(f"Phase ended because collapse is {params['isCollapse'].value} and r reaches {R2_next} pc (< r_coll: {params['coll_r'].value} pc)")
        params['SimulationEndReason'].value = 'Small radius reached'
        params['EndSimulationDirectly'].value = True
        return True
    
    #--- 3) Large radius reached during expansion.
    if R2_next > params['stop_r'].value:
        print(f"Phase ended because r reaches {R2_next} pc (> stop_r: {params['stop_r'].value} pc)")
        params['SimulationEndReason'].value = 'Large radius reached'
        params['EndSimulationDirectly'].value = True
        return True
        
    #--- 4) dissolution after certain period of low density
    # if params['t_now'].value - params['t_Lowdense'].value > params['stop_t_diss'].value:
    if params['shell_nMax'].value < params['stop_n_diss'].value:
        params['isDissolved'].value = True
        # print(f"Phase ended because {params['t_now'].value - params['t_Lowdense'].value} Myr passed since low density of {params['shell_nShell_max'].value/cvt.ndens_cgs2au} /cm3")
        params['completed_reason'].value = 'Shell dissolved'
        params['EndSimulationDirectly'].value = True
        return True
    
    #--- 5) exceeds cloud radius
    if params['expansionBeyondCloud'] == False:
        if params['R2'].value > params['rCloud'].value:
            print(f"Bubble radius ({params['R2'].value} pc) exceeds cloud radius ({params['rCloud'].value} pc)")
            params['SimulationEndReason'].value = 'Bubble radius larger than cloud'
            params['EndSimulationDirectly'].value = True
            return True
    
    return False



