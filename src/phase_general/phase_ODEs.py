#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:28:20 2023

@author: Jia Wei Teh

"""


import numpy as np
import scipy.optimize
#--
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import src.shell_structure.shell_structure as shell_structure
# import src.shell_structure.shell_structure_old as shell_structure
import src._functions.unit_conversions as cvt

    # TODO: add cover fraction cf
    
def get_vdot(t, y, 
                 params, SB99f):
    
    # unpack current values of y (r, rdot, E, T)
    R2, v2, Eb, T0 = y  
    
    tSF = params['tSF'].value
    
    # Interpolate SB99 to get feedback parameters
    # can't quite use what's in the dictionary because it could be changed.
    # TODO: in future just add them into dictionary
    # mechanical luminosity at time t (erg)
    L_wind = SB99f['fLw'](t)[()]
    # momentum of stellar winds at time t (cgs)
    pdot_wind = SB99f['fpdot'](t)[()] 
    # other luminosities
    Lbol = SB99f['fLbol'](t)[()]
    Ln = SB99f['fLn'](t)[()]
    Li = SB99f['fLi'](t)[()]
    Qi = SB99f['fQi'](t)[()]
    
    
    # velocity from luminosity and change of momentum (au)
    v_wind = (2.*L_wind/pdot_wind)
    
    # =============================================================================
    # Shell mass, where radius = maximum extent of shell.
    # =============================================================================

    if params['isCollapse'].value == True:
        # stays constant during collapse
        mShell = params['mShell'].value
        mShell_dot = 0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, rdot_arr = v2)
    
    # just artifacts. 
    # TODO: fix this in the future
    if hasattr(mShell, '__len__'):
        if len(mShell) == 1:
            mShell = mShell[0]
        
    if hasattr(mShell_dot, '__len__'):
        if len(mShell_dot) == 1:
            mShell_dot = mShell_dot[0]
    
    params['mShell'].value = mShell
    params['mShell_dot'].value = mShell_dot
        
    cf = 1
    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    # if you don't want gravity, set .inc_grav to zero
    F_grav = (params['G_au'].value * mShell / R2**2 * (params['mCluster_au'].value + mShell/2)  * params['inc_grav'].value) 
  
    # get pressure from energy. 
    if Eb > 0:
        # calculate radius of inner discontinuity (inner radius of bubble)
        R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, R2,
                                   args=([L_wind, Eb, v_wind, R2])) 
        # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
        # IDEA: move R1 gradually outwards
        dt_switchon = 1e-3 # in Myr, gradually switch on things during this time period
        tmin = dt_switchon
        # remember t is in year
        if (t > (tmin + tSF)):
            # equation of state PB is in cgs
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        elif (t <= (tmin + tSF)):
            R1_tmp = (t-tSF)/tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
    else: # energy is very small: case of pure momentum driving
        R1 = R2 # there is no bubble --> inner bubble radius R1 equals shell radius r
        # ram pressure from winds
        press_bubble = get_bubbleParams.pRam(R2, L_wind, v_wind)
            
        
    params['Ln'].value = Ln  
    params['Li'].value = Li  
    params['Qi'].value = Qi  
    params['Lbol'].value = Lbol  
    params['mShell'].value = mShell  
    params['Pb'].value = press_bubble  
        
    # =============================================================================
    # Shell structure
    # =============================================================================
    
    # calculate simplified shell structure (warpfield-internal shell structure, not cloudy)
    # We are setting mBubble = 0 here, since we are not interested in the potential. This can skip some calculations.
    shell_structure.shell_structure(params)
    
    # units right
    
    # radiation pressure coupled to the shell
    fRad = params['shell_f_absorbed'].value * Lbol / (params['c_au'].value)
    params['shell_f_rad'].value = fRad

    isLowdense = params['shell_nShell_max'].value < params['stop_n_diss'].value

    if isLowdense != params['isLowdense'].value:
        # update if it is low dense now
        if isLowdense == True:
            params['t_Lowdense'].value = params['t_now'].value
            params['isLowdense'].value = True
        else:
            # set so that t_lowdense - t_now in check_event() will never be positive (stop_t_diss)
            params['t_Lowdense'].value = 1e30
            params['isLowdense'].value = False    

    
    def get_press_ion(r, ion_dict):
        """
        calculates pressure from photoionized part of cloud at radius r
        :return: pressure of ionized gas outside shell
        """
        # old code: ODE.calc_ionpress()
        
        # n_r: total number density of particles (H+, He++, electrons)
        try:
            r = np.array([r])
        except:
            pass
        
        n_r = density_profile.get_density_profile(r, ion_dict)
        
        P_ion = n_r * ion_dict['k_B_au'].value * ion_dict['t_ion'].value
        
        if len(P_ion) == 1:
            P_ion = P_ion[0]
        
        return P_ion
    

    # calculate inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if params['shell_f_absorbed_ion'].value < 1.0:
        press_HII = get_press_ion(R2, params)
    else:
        press_HII = 0.0
        
    # =============================================================================
    # calculate the ODE part: Acceleration 
    # =============================================================================
    vd = (cf * 4 * np.pi * R2**2 * (press_bubble - press_HII)\
            - mShell_dot * v2\
                - F_grav + cf * fRad) / mShell
    
    # force calculation
    params['F_grav'].value = F_grav
    params['F_ion'].value = press_HII
    # params['F_ram'].value = (4 * np.pi * R2**2 * (press_bubble - press_HII))
    params['F_rad'].value = fRad
    # params['F_SN'].value = fRad
    
    return vd
