#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh
"""
# libraries
import numpy as np
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
import scipy.optimize
import sys 
from src.sb99.update_feedback import get_currentSB99feedback

import src._functions.unit_conversions as cvt


def get_ODE_Edot(y, t, params):
    """
    This ODE solver solves for y (see below). This is being used in run_energy_phase(), after 
    bubble and shell structure is calculated. This is a general ODE, with more specific ones
    for other stages in other scripts. 

    old code: fE_gen()

    Parameters
    ----------
    y contains:
        - r (R2), shell radius [pc].
        - v (v2), shell velocity [au]
        - E (Eb), bubble energy [au]
    t : time [Myr]

    params : see run_energy_phase() for more description of params. Here unit does not matter, because
            astropy will take care of it. Units do matter for y and t, because scipy strips them off. 
            In case the naming does not makes sense:
        
        - Lw: mechanical luminosity
        - pdot_wind: momentum rate 
        - L_bubble: luminosity loss to cooling (see get_bubbleproperties() in bubble_luminosity.py)
        - FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
        - tFRAG: time takes to fragmentation
        - tSCR: sound crossing time

    Returns
    -------
    time derivatives of the ODE: 
        - drdt  
        - dvdt  what units?
        - dEdt  

    """
    
    # unpack current values of y (r, rdot, E)
    R2, v2, Eb = y 
    
    
    [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] =  get_currentSB99feedback(t, params)
    
    
    FABSi = params['shell_fAbsorbedIon'].value
    FRAD = params['shell_fAbsorbedWeightedTotal'].value
    mCluster = params['mCluster'].value
    LWind = params['LWind'].value
    L_bubble = params['bubble_LTotal'].value
    pWindDot = params['pWindDot'].value 
    
    # calculate shell mass and time derivative of shell mass [au]
    mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, 
                                                       rdot_arr = v2)
    
    if len(mShell_dot) == 1:
        mShell = mShell[0]
        mShell_dot = mShell_dot[0]
    
    params['shell_mass'].value = mShell
    params['shell_massDot'].value = mShell_dot
    
    def get_press_ion(r, ion_dict):
        """
        calculates pressure from photoionized part of cloud at radius r
        :return: pressure of ionized gas outside shell
        
        returns in [au]
        """
        # old code: ODE.calc_ionpress()
        
        # n_r: total number density of particles (H+, He++, electrons)
        try:
            r = np.array([r])
        except:
            pass
        n_r = density_profile.get_density_profile(r, ion_dict)
        
        P_ion = n_r * ion_dict['k_B'].value * ion_dict['TShell_ion'].value
        
        return P_ion

    # calc inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        press_HII = get_press_ion(R2, params)
    else:
        press_HII = 0.0
        
    if params['R2'].value >= params['rCloud'].value:
        # TODO: add this more for ambient pressure
        press_HII += params['PISM'] * params['k_B']   
    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    F_grav = params['G'].value * mShell / R2**2 * (mCluster + mShell/2) 
    
    # get pressure from energy
    # calculate radius of inner discontinuity (inner radius of bubble)
    
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, R2,
                               args=([LWind, Eb, vWind, R2])) 
    
    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 #* u.Myr # gradually switch on things during this time period
    tmin = dt_switchon
    if (t > (tmin + params['tSF'].value)):
        
        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        
    elif (t <= (tmin + params['tSF'].value)):
        
        R1_tmp = (t-params['tSF'].value)/tmin * R1
        press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
    
    
    # TODO: Future-------
    # def calc_coveringf(t,tFRAG,ts):
    #     """
    #     estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
    #     if the shell covers the whole sphere: cf = 1
    #     if there is no shell: cf = 0
        
    #     Note: I think, since we set tFRAG ultra high, that means cf will almost always be 1. 
    #     """
    #     cfmin = 0.4
    #     # simple slope
    #     cf = 1. - ((t - tFRAG) / ts)**1.
    #     cf[cf>1.0] = 1.0
    #     cf[cf<cfmin] = cfmin
    #     # return
    #     return cf

    # calculate covering fraction
    # cf = calc_coveringf(np.array([t.value])*u.s,tFRAG,tSCR)
    # TODO: finish this in the future
    # cf = 1
    # leaked luminosity
    # if cf < 1:
    #     L_leak = (1. - cf)  * 4. * np.pi * R2 ** 2 * press_bubble * CS / (gamma_adia - 1.)
    # else:
    L_leak = 0  
    #--------
        
    # time derivatives￼￼
    rd = v2
    vd = (4 * np.pi * R2**2 * (press_bubble-press_HII) - mShell_dot * v2 - F_grav + FRAD) / mShell
    Ed = (LWind - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak 

    derivs = [rd, vd, Ed]
    
    params['F_grav'].value = F_grav
    params['F_ion'].value = press_HII
    # params['F_ram'].value = (4 * np.pi * R2**2 * (press_bubble - press_HII))
    fRad = params['shell_fAbsorbedWeightedTotal'].value * params['Lbol'].value  / (params['c_light'].value)
    params['F_rad'].value = fRad
    
    
    # return
    return derivs





