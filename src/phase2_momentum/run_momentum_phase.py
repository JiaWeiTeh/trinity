#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:53:30 2023

@author: Jia Wei Teh
"""
# libraries
import scipy.optimize
import numpy as np
import astropy.units as u
import astropy.constants as c
import sys
import os
#--
from src.phase_general import phase_ODEs
import src._functions.unit_conversions as cvt



def run_phase_momentum(params):
    
    
    # initial conditions for the ODE equation.
    # [pc], [pc/yr], [u.M_sun*u.pc**2/u.yr**2], [K]
    y0 = [params['R2'].value, params['v2'].value, params['Eb'].value, params['T0'].value]


    #-- theoretical minimum and maximum of this phase
    tmin = params['t_now'].value
    tmax = params['tStop'].value
    
    
    
    # =============================================================================
    # List of possible events and ODE terminating conditions
    # =============================================================================
        
    nmin = int(200 * np.log10(tmax/tmin))

    time_range = np.logspace(np.log10(tmin), np.log10(tmax), nmin)
    dt = np.diff(time_range)


    r2 = params['R2'].value
    v2 = params['v2'].value
    Eb = 0
    T0 = params['T0'].value

    for ii, time in enumerate(time_range):
        
        # new inputs
        y = [r2, v2, Eb, T0]
    
        rd, vd, Ed, Td =  ODE_equations_momentum(time, y, params)
        
        if hasattr(vd, '__len__') and len(vd) == 1:
            vd = vd[0]
        else:
            sys.exit('weird vd behaviour in implicit')
        
        
        dt_params = [dt[ii], rd, vd, Ed, Td]
            
        if check_events(params, dt_params):
            break
        
        
        if ii != (len(time_range) - 1):
            r2 += rd * dt[ii]
            v2 += vd * dt[ii]
            Eb += Ed * dt[ii]
            T0 += Td * dt[ii]
            
        
    return 
    
    


def ODE_equations_momentum(t, y, params):
    
    # --- These are R2, v2, Eb and T0 (Trgoal).
    R2, v2, Eb, T0 = y    
    
    print(f'current stage: t:{t}, r:{R2}, v:{v2}, E:{Eb}, T:{T0}')

    # record
    params['t_now'].value = t
    params['v2'].value = v2
    params['Eb'].value = Eb
    params['T0'].value = T0
    params['R2'].value = R2
    
    SB99f = params['SB99f'].value
    
    # idea: maybe E is not pure 0 and Ed should be previous value? 
    
    
    # --- in this phase we are not solving for E and T. 
    # However, phase_ODEs.get_vdot still expects values:
    
    # vd, _ = phase_ODEs.get_vdot(t, [R2, v2, 0, 0], params, SB99f)
    # idea
    vd = phase_ODEs.get_vdot(t, [R2, v2, Eb, T0], params, SB99f)
    rd = v2
    
    params.save_snapShot()
    
    # return [rd, vd, 0, 0]
    return [rd, vd, 0, 0]



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
    
    #--- 1) Stopping time reached
    if t_next > params['tStop'].value:
        print(f"Phase ended because t reaches {t_next} Myr (> tStop: {params['tStop'].value}) in the next iteration.")
        params['completed_reason'].value = 'Stopping time reached'
        return True
    
    #--- 2) Small radius reached during collapse.
    if params['isCollapse'].value == True and R2_next < params['r_coll'].value:
        print(f"Phase ended because collapse is {params['isCollapse'].value} and r reaches {R2_next} pc (< r_coll: {params['r_coll'].value} pc)")
        params['completed_reason'].value = 'Small radius reached'
        return True
    
    #--- 3) Large radius reached during expansion.
    if R2_next > params['stop_r'].value:
        print(f"Phase ended because r reaches {R2_next} pc (> stop_r: {params['stop_r'].value} pc)")
        params['completed_reason'].value = 'Large radius reached'
        return True
        
    #--- 4) dissolution after certain period of low density
    if params['t_now'].value - params['t_Lowdense'].value > params['shell_nShell_max'].value:
        print(f"Phase ended because {params['t_now'].value - params['t_Lowdense'].value} Myr passed since low density of {params['shell_nShell_max'].value/cvt.ndens_cgs2au} /cm3")
        params['completed_reason'].value = 'Shell dissolved'
        return True
    
    
    return False




