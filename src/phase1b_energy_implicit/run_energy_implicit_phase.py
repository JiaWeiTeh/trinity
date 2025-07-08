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
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.phase1b_energy_implicit.get_betadelta as get_betadelta
import src._functions.unit_conversions as cvt
import src.cooling.non_CIE.read_cloudy as non_CIE
from src.sb99.update_feedback import get_currentSB99feedback

import src._functions.operations as operations
#--

def run_phase_energy(params):

    # TODO: add fragmentation mechanics in events 

    # what is the current v2?
    params['v2'].value = params['cool_alpha'].value * params['R2'].value / (params['t_now'].value) 
    
    
    #-- theoretical minimum and maximum of this phase
    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # =============================================================================
    # List of possible events and ODE terminating conditions
    # =============================================================================
        
    # how many timesteps? about 200 timesteps per dex
    nmin = int(200 * np.log10(tmax/tmin))

    time_range = np.logspace(np.log10(tmin), np.log10(tmax), nmin)
    
    dt = np.diff(time_range)


    r2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    
    stop_condition = False

    for ii, time in enumerate(time_range):
        
        # new inputs
        y = [r2, v2, Eb, T0]
    
        rd, vd, Ed, Td =  ODE_equations(time, y, params)
        
                
        try:
            params['t_next'].value = time_range[ii+1]
        except:
            params['t_next'].value = time + dt
                
        
        # if hasattr(vd, '__len__') and len(vd) == 1:
        #     vd = vd[0]
        # else:
        #     sys.exit('weird vd behaviour in implicit')
        
        
        dt_params = [dt[ii], rd, vd, Ed, Td]
            
        if check_events(params, dt_params):
            stop_condition = True
            break
        
        
        if ii != (len(time_range) - 1):
            r2 += rd * dt[ii]
            v2 += vd * dt[ii]
            Eb += Ed * dt[ii]
            T0 += Td * dt[ii]
            
            
    # if break, maybe something happened. Decrease dt
    if stop_condition:
        
        tmin = time_range[ii]
        tmax = time_range[ii+1] # this is the final moment
        
        
        # reverse log space so that we have more point towards the end.
        time_range = (tmin + tmax) - np.logspace(np.log10(tmin), np.log10(tmax), 50)
        
        
        for ii, time in enumerate(time_range):
        
            # new inputs
            y = [r2, v2, Eb, T0]
            
            try:
                params['t_next'].value = time_range[ii+1]
            except:
                params['t_next'].value = time + dt
        
            rd, vd, Ed, Td =  ODE_equations(time, y, params)
            
            dt_params = [dt[ii], rd, vd, Ed, Td]
                
            if check_events(params, dt_params):
                break
            
            
            if ii != (len(time_range) - 1):
                r2 += rd * dt[ii]
                v2 += vd * dt[ii]
                Eb += Ed * dt[ii]
                T0 += Td * dt[ii]
        
    return






def ODE_equations(t, y, params):
    
    # t [yr], R2 [pc], v2 [pc/yr], Eb [au]
    # --- These are R2, v2, Eb and T0 (Trgoal).
    R2, v2, Eb, T0 = y
    
    # record
    params['t_now'].value = t
    params['v2'].value = v2
    params['Eb'].value = Eb
    params['T0'].value = T0
    params['R2'].value = R2
    
    print(f'current stage: t:{t}, r:{R2}, v:{v2}, E:{Eb}, T:{T0}')
    
    
    # ---  
    
    #-- updating values in the loop; make sure to include all values of parameters
    # Take note that alpha is defined as a = t/r*v, where t[Myr], v[kms], r[pc]
    # this is kinda wrong. In future lets make it all in pc and Myr before converting
    params['cool_alpha'].value = t / R2 * v2
    
    # =============================================================================
    # Prelude: prepare cooling structures so that it doesnt have to run every loop.
    # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
    # =============================================================================
    if np.abs(params['t_previousCoolingUpdate'].value - params['t_now'].value) > 5e-3: # in Myr
        # recalculate non-CIE
        cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params)
        # save
        params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
        params['cStruc_heating_nonCIE'].value = heating_nonCIE
        params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
        # update current value
        params['t_previousCoolingUpdate'].value = params['t_now'].value 
        
    # =============================================================================
    # Part 1: find acceleration and velocity
    # =============================================================================
    
    # returns in pc/yr2
    vd = phase_ODEs.get_vdot(t, y, params)
    rd = v2
        
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
    
    print('completed a phase in ODE_equations in implicit_phase')
    print(f'rd: {rd}, vd: {vd}, Ed: {Ed}, Td: {Td}')

   
    # save snapshot
    result_params.save_snapshot()
    
    # return [rd.to(u.pc/u.Myr).value, vd.to(u.km/u.s/u.Myr).value, Ed.to(u.erg/u.Myr).value, Td.to(u.K/u.Myr).value]
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
        
    # check if there is a change in sign 
    if np.sign(v2_next) != np.sign(params['v2'].value):
        if np.sign(v2_next) == -1:
            print(f'Bubble currently collapsing because the next velocity is {v2_next / cvt.v_kms2au} km/s.')
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
        
    # #--- 4) dissolution after certain period of low density
    # if params['t_now'].value - params['t_Lowdense'].value > params['stop_t_diss'].value:
    #     print(f"Phase ended because {params['t_now'].value - params['t_Lowdense'].value} Myr passed since low density of {params['shell_nShell_max'].value/cvt.ndens_cgs2au} /cm3")
    #     params['completed_reason'].value = 'Shell dissolved'
    #     return True
    
    #--- 5) exceeds cloud radius
    if params['expansionBeyondCloud'] == False:
        if params['R2'].value > params['rCloud'].value:
            print(f"Bubble radius ({params['R2'].value} pc) exceeds cloud radius ({params['rCloud'].value} pc)")
            params['SimulationEndReason'].value = 'Bubble radius larger than cloud'
            params['EndSimulationDirectly'].value = True
            return True
    
    return False



