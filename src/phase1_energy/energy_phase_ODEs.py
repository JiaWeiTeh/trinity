 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh
"""
# libraries
import sys 
import scipy
import numpy as np
import scipy.optimize
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.sb99.update_feedback import get_currentSB99feedback

# -- constant
FOUR_PI = 4.0 * np.pi

def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x

def get_press_ion_outside(r, params):
    """
    Pressure of photoionized gas at radius r (outside shell).
    Returns scalar in your code units.
    """
    r = np.atleast_1d(r)
    n_r = density_profile.get_density_profile(r, params)  # assumes it accepts params
    P = 2.0 * n_r * params["k_B"].value * params["TShell_ion"].value
    return _scalar(P)


def get_ODE_Edot(y, t, params):
    """
    old code: fE_gen()

    ODE system for bubble expansion.
    y = [R2, v2, Eb, T0]  (T0 is carried as a constant here)
    t in Myr
    """

    R2, v2, Eb, T0 = y
    R2 = float(R2)
    v2 = float(v2)
    Eb = float(Eb)
    
    # update values
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb
    
    
    # --- pull frequently-used parameters once
    FABSi     = params["shell_fAbsorbedIon"].value
    F_rad     = params["shell_F_rad"].value
    mCluster  = params["mCluster"].value
    L_bubble  = params["bubble_LTotal"].value
    gamma     = params["gamma_adia"].value
    tSF       = params["tSF"].value
    G         = params["G"].value
    Qi        = params["Qi"].value
    LWind     = params["LWind"].value
    vWind     = params["vWind"].value
    
    
    # --- calculate shell mass and time derivative of shell mass [au]
    mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, 
                                                       rdot_arr = v2)
    
    
    if params['isCollapse'].value == True:
        # stays constant during collapse
        mShell = params['shell_mass'].value
        mShell_dot = 0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(
            R2, params, return_mdot = True, rdot_arr = v2
            )
        mShell = _scalar(mShell)
        mShell_dot = _scalar(mShell_dot)
        
    params['shell_mass'].value = mShell
    params['shell_massDot'].value = mShell_dot
            
    
    # --- gravity force (self + cluster)
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)
    
    
    # calculate radius of inner discontinuity (inner radius of bubble)
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, R2,
                               args=([LWind, Eb, vWind, R2])) 
    
    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 #* u.Myr # gradually switch on things during this time period
    tmin = dt_switchon

        
    if params['current_phase'].value in ['momentum']:
        press_bubble = get_bubbleParams.pRam(R2, LWind, vWind)
    else:
        if (t > (tmin + params['tSF'].value)):
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
            
        elif (t <= (tmin + params['tSF'].value)):
            R1_tmp = (t-params['tSF'].value)/tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
       
    #     press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
    params['Pb'].value = press_bubble  
    
    
    def get_press_ion(r, ion_dict):
        """
        calculates pressure from photoionized part of cloud at radius r
        :return: pressure of ionized gas outside shell
        
        returns in [au]
        """
        
        # n_r: total number density of particles (H+, He++, electrons)
        try:
            r = np.array([r])
        except:
            pass
        n_r = density_profile.get_density_profile(r, ion_dict)
        
        print('n_r', n_r)
        
        P_ion = 2 * n_r * ion_dict['k_B'].value * ion_dict['TShell_ion'].value
        
        # should always be true?
        if hasattr(P_ion, '__len__'):
            if len(P_ion) == 1:
                P_ion = P_ion[0]
                
        return P_ion


    # Question: maybe 0.5?

    # calc inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        press_HII_in = get_press_ion(params['rShell'].value, params)
    else:
        press_HII_in = 0.0
        
    if params['rShell'].value >= params['rCloud'].value:
        # TODO: add this more for ambient pressure
        press_HII_in += params['PISM'] * params['k_B']   
        
    # this should follow density profile inside the bubble, hence interpolation
    bubble_n_arr = params['bubble_n_arr'].value
    bubble_r_arr = params['bubble_r_arr'].value
    
    print('bubble bubble_r_arr, bubble_n_arr', bubble_r_arr, bubble_n_arr)
    # method 2: all hii region approximation
    if FABSi < 1:
        nR2 = params['nISM']
    else:
        nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3 * 3 / 4 / np.pi)
    press_HII_out = 2 * nR2 * params['k_B'].value * 3e4
    #---
    
     
    # TODO: Future------- add cover fraction after fragmentation
    L_leak = 0  
    #--------
        
    # time derivatives￼￼
    rd = v2
    vd = (4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out) - mShell_dot * v2 - F_grav + F_rad) / mShell
    
    # lets say that within the first few runs, mShell_dot is too large (explosive) so that negative value is 
    # too much. Let's make this instead 0.?
    # if params['EarlyPhaseApproximation'].value == True:
    #     mShell_dot = 0
        # vd = -1e8
    if params['EarlyPhaseApproximation'].value == True:
        vd = -1e8
    
    print(f'vd is {4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out)} - {mShell_dot * v2} - {F_grav} + {F_rad} divide {mShell} equals {vd}')
    
    # but this isnt used I think - Ed is obtained via conversion of beta/delta in run_implicit_energy.py.
    Ed = (LWind - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak 

    derivs = [rd, vd, Ed, 0]
    
    # calculate forces
    params['F_grav'].value = F_grav
    params['F_ion_in'].value = press_HII_in * 4 * np.pi * R2**2 
    params['F_ion_out'].value = press_HII_out * 4 * np.pi * R2**2 
    params['F_ram'].value = press_bubble * 4 * np.pi * R2**2 
    params['F_rad'].value = F_rad
    
    
    # return
    return derivs







