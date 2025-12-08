#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:13:28 2023

@author: Jia Wei Teh
"""

import scipy.optimize
import numpy as np
import astropy.units as u
import astropy.constants as c
import sys
import os
#--
import src.phase_general.phase_ODEs as phase_ODEs
import src.shell_structure.shell_structure as shell_structure
import src.phase1_energy.energy_phase_ODEs as energy_phase_ODEs

import src._functions.unit_conversions as cvt


def run_phase_transition(params):
    
    
    # TODO: add fragmentation mechanics in events 

    # what is the current v2?
    params['v2'].value = params['cool_alpha'].value * params['R2'].value / (params['t_now'].value) 
    
    
    #-- theoretical minimum and maximum of this phase
    tmin = params['t_now'].value
    tmax = params['stop_t'].value

    # =============================================================================
    # List of possible events and ODE terminating conditions
    # =============================================================================
     
    
    nmin = int(200 * np.log10(tmax/tmin))

    time_range = np.logspace(np.log10(tmin), np.log10(tmax), nmin)[1:] #to avoid duplicate starting values
    dt = np.diff(time_range)


    r2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    stop_condition = False
    
    for ii, time in enumerate(time_range):
        
        # new inputs
        y = [r2, v2, Eb, T0]
        
        try:
            params['t_next'].value = time_range[ii+1]
        except:
            params['t_next'].value = time + dt
            
    
        rd, vd, Ed, Td =  ODE_equations_transition(time, y, params)
        
        
        dt_params = [dt[ii], rd, vd, Ed, Td]
            
        if check_events(params, dt_params):
            stop_condition = True
            break
        
        
        if ii != (len(time_range) - 1):
            r2 += rd * dt[ii]
            v2 += vd * dt[ii]
            Eb += Ed * dt[ii]
            T0 += Td * dt[ii]
    
    
                       
    # # if break, maybe something happened. Decrease dt
    # if stop_condition:
        
    #     tmin = time_range[ii]
    #     tmax = time_range[ii+1] # this is the final moment
        
        
    #     # reverse log space so that we have more point towards the end.
    #     time_range = (tmin + tmax) - np.logspace(np.log10(tmin), np.log10(tmax), 50)
    #     time_range = time_range[1:]
        
        
    #     for ii, time in enumerate(time_range):
        
    #         # new inputs
    #         y = [r2, v2, Eb, T0]
            
    #         try:
    #             params['t_next'].value = time_range[ii+1]
    #         except:
    #             params['t_next'].value = time + dt

            
        
    #         rd, vd, Ed, Td =  ODE_equations_transition(time, y, params)
            
    #         # if hasattr(vd, '__len__') and len(vd) == 1:
    #         #     vd = vd[0]
    #         # else:
    #         #     sys.exit('weird vd behaviour in implicit')
            
            
    #         dt_params = [dt[ii], rd, vd, Ed, Td]
                
    #         if check_events(params, dt_params):
    #             break
            
            
    #         if ii != (len(time_range) - 1):
    #             r2 += rd * dt[ii]
    #             v2 += vd * dt[ii]
    #             Eb += Ed * dt[ii]
    #             T0 += Td * dt[ii]
            
    return




def ODE_equations_transition(t, y, params):
    
    
    # --- These are R2, v2, Eb and T0 (Trgoal).
    R2, v2, Eb, T0 = y    
    
    
    # =============================================================================
    # Part 1: find acceleration and velocity
    # =============================================================================
    print(f'current stage: t:{t}, r:{R2}, v:{v2}, E:{Eb}, T:{T0}')

    # record
    params['t_now'].value = t
    params['v2'].value = v2
    params['Eb'].value = Eb
    params['T0'].value = T0
    params['R2'].value = R2
    
    # returns in pc/yr2
    # vd = phase_ODEs.get_vdot(t, y, params)
    # new
    from src.sb99.update_feedback import get_currentSB99feedback

    [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] =  get_currentSB99feedback(t, params)
    shell_structure.shell_structure(params)
    
    _, vd, _, _ = energy_phase_ODEs.get_ODE_Edot(y, t, params)
    
    
        
    # ILL MAYBE PLACE THIS AFTER SO THAT WE WONT RECORD THE INTRIDCACIES INSIDE THIS LOOP
    # # print('t here is t=', t)
    params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t]])
    params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
    params['array_R1'].value = np.concatenate([params['array_R1'].value, [params['R1'].value]])
    params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
    params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
    
    # print('mshell problems', mShell)
    
    import src.cloud_properties.mass_profile as mass_profile
    mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                   return_mdot = True, 
                                                   rdot_arr = v2)
    

    
    # NEW CODE ---
    # just artifacts. 
    # TODO: fix this in the future
    if hasattr(mShell, '__len__'):
        if len(mShell) == 1:
            mShell = mShell[0]
        
    if hasattr(mShell_dot, '__len__'):
        if len(mShell_dot) == 1:
            mShell_dot = mShell_dot[0]
    
    
    
    params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])
            
    
    
    rd = v2
    
    # shouldnt we also balance r1?

    t_soundcrossing = params['R2'].value/params['c_sound'].value

    # params['dEdt'].value = - Eb / t_soundcrossing
    dEdt = - Eb / t_soundcrossing
    
    # print('dEdt is', params['dEdt'].value)

    
    
    params.save_snapshot()

    return [rd, vd, dEdt, 0]



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
            print(f'Bubble currently collapsing because the next velocity is {v2_next / cvt.v_kms2au} km/s.')
            params['isCollapse'].value = True
        else:
            params['isCollapse'].value = False
            
    # =============================================================================
    # Terminating events
    # =============================================================================
    
    # Main event: when energy is close to zero.
    if Eb_next < 1:
        print(f"Phase ended because energy crosses from E: {params['Eb'].value} to E: {Eb_next} in the next iteration.")
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
    if params['isDissolved'].value == True:
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





