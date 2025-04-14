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
import src._functions.operations as operations
#--

def run_phase_energy(params):

    # TODO: add fragmentation mechanics in events 

    # what is the current v2?
    params['v2'].value = params['alpha'].value * params['R2'].value / (params['t_now'].value) 
    
    
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
    Eb = params['Eb'].value
    T0 = params['T0'].value
    
    stop_iteration = False

    for ii, time in enumerate(time_range):
        
        # new inputs
        y = [r2, v2, Eb, T0]
    
        rd, vd, Ed, Td =  ODE_equations(time, y, params)
        
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
    
    SB99f = params['SB99f'].value
    
    print(f'current stage: t:{t}, r:{R2}, v:{v2}, E:{Eb}, T:{T0}')
    
    # --- feedback parameters required to find beta/delta etc
    # Interpolate SB99 to get feedback parameters
    # mechanical luminosity at time t  
    L_wind = SB99f['fLw_cgs'](t) * cvt.L_cgs2au
    # momentum of stellar winds at time t  
    pdot_wind = SB99f['fpdot_cgs'](t) * cvt.pdot_cgs2au
    # get the slope via mini interpolation for some dt.
    dt = 1e-9 #*Myr
    pdotdot_wind = (SB99f['fpdot_cgs'](t + dt) - SB99f['fpdot_cgs'](t - dt))/ ((dt+dt)/cvt.s2Myr) #this is still in cgs
    # and then add units
    pdotdot_wind *= cvt.pdotdot_cgs2au
    # print('pdotdot_wind', pdotdot_wind)
    # other luminosities
    Qi = SB99f['fQi_cgs'](t) / cvt.s2Myr
    # velocity from luminosity and change of momentum (pc/Myr)
    v_wind = (2.*L_wind/pdot_wind)
    # ---  
    
    #-- updating values in the loop; make sure to include all values of parameters
    # Take note that alpha is defined as a = t/r*v, where t[Myr], v[kms], r[pc]
    # this is kinda wrong. In future lets make it all in pc and Myr before converting
    params['alpha'].value = t / R2 * v2
    params['Qi'].value = Qi
    params['v_wind'].value = v_wind
    params['pwdot'].value = pdot_wind
    params['pwdot_dot'].value = pdotdot_wind
    params['L_wind'].value = L_wind
    
    # =============================================================================
    # Prelude: prepare cooling structures so that it doesnt have to run every loop.
    # Tip: Get cooling structure every 50k years (or 1e5?) or so. 
    # =============================================================================
    if np.abs(params['time_last_cooling_update'].value - params['t_now'].value) > 5e-3: # in Myr
        # recalculate non-CIE
        cooling_nonCIE, heating_nonCIE, netcooling_interpolation = non_CIE.get_coolingStructure(params['t_now'].value * 1e6)
        # save
        params['cStruc_cooling_nonCIE'].value = cooling_nonCIE
        params['cStruc_heating_nonCIE'].value = heating_nonCIE
        params['cStruc_net_nonCIE_interpolation'].value = netcooling_interpolation
        # update current value
        params['time_last_cooling_update'].value = params['t_now'].value 
        
    # =============================================================================
    # Part 1: find acceleration and velocity
    # =============================================================================
    
    # returns in pc/yr2
    vd = phase_ODEs.get_vdot(t, y, params, SB99f)
    rd = v2
        
    # =============================================================================
    # Part 2: find beta, delta and convert them to dEdt and dTdt
    # =============================================================================
        
    (beta, delta), result_params = get_betadelta.get_beta_delta_wrapper(params['beta'].value, params['delta'].value, params)
           
    # update
    result_params["beta"].value = beta
    result_params["delta"].value = delta
    print('beta found:', beta, 'delta found', delta)
    
    # sound speed for future dEdt calculation.
    result_params['cs_avg'].value = operations.get_soundspeed(result_params['bubble_Tavg'].value, result_params)

    #------ convert them to dEdt and dTdt.
    def get_EdotTdot(params_dict
                  ):
        # convert beta and delta to dE/dt and dT/dt.
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 
                       1e-3 * params_dict['R2'].value, params_dict['R2'].value, 
                       args=([params_dict['L_wind'].value, 
                              params_dict['Eb'].value, 
                              params_dict['v_wind'].value,
                              params_dict['R2'].value,
                              ]))
    
        params_dict['R1'].value = R1
        
        # The bubble Pbure [cgs - g/cm/s2, or dyn/cm2]
        Pb = get_bubbleParams.bubble_E2P(params_dict['Eb'].value,
                                        params_dict['R2'].value, 
                                        params_dict['R1'].value)
    
        params_dict['Pb'].value = Pb

        # get new beta value
        Edot = get_bubbleParams.beta2Edot(params_dict)
        # get dTdt
        Tdot = get_bubbleParams.delta2dTdt(params_dict['t_now'].value, params_dict['T0'].value, params_dict['delta'].value)
        
        return Edot, Tdot

    
    Ed, Td = get_EdotTdot(result_params)
    
    print('completed a phase in ODE_equations in implicit_phase')
    print(f'rd: {rd}, vd: {vd}, Ed: {Ed}, Td: {Td}')
    
    # save snapshot
    result_params.save_snapShot()
    
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
    if (params['Lgain'].value - params['Lloss'].value)/params['Lgain'].value < 0.05:
        print(f"Phase ended because Lloss: {params['Lloss'].value} is within {(params['Lgain'].value - params['Lloss'].value)/params['Lgain'].value * 100}% of Lgain: {params['Lgain'].value}")
        
        return True
    
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
    if params['t_now'].value - params['t_Lowdense'].value > params['stop_t_diss'].value:
        print(f"Phase ended because {params['t_now'].value - params['t_Lowdense'].value} Myr passed since low density of {params['shell_nShell_max'].value/cvt.ndens_cgs2au} /cm3")
        params['completed_reason'].value = 'Shell dissolved'
        return True
    
    
    return False



