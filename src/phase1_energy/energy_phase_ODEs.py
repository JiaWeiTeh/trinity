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
import scipy

import sys 
from src.sb99.update_feedback import get_currentSB99feedback

import src._functions.unit_conversions as cvt


def get_ODE_Edot(y, t, params):
    """
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
    R2, v2, Eb, T0 = y 
    
    # print('t value is ', t)
    # import sys
    # sys.exit()
    
    [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] =  get_currentSB99feedback(t, params)
    
    # update values
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb
    
    
    FABSi = params['shell_fAbsorbedIon'].value
    FRAD = params['shell_fRad'].value
    mCluster = params['mCluster'].value
    LWind = params['LWind'].value
    L_bubble = params['bubble_LTotal'].value
    pWindDot = params['pWindDot'].value 
    
    # calculate shell mass and time derivative of shell mass [au]
    mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, 
                                                       rdot_arr = v2)
    
    
    if params['isCollapse'].value == True:
        # stays constant during collapse
        mShell = params['shell_mass'].value
        mShell_dot = 0
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
                                                       return_mdot = True, rdot_arr = v2)
        
        
    # ADD IF ELSE FOR IF CASE 1b AND ALSO TIME CONSTRAIN BOTH HERE AND ALSO IN MAS PROFILE CALCULATION
    
    # NEW CODE ---
    # just artifacts. 
    # TODO: fix this in the future
    if hasattr(mShell, '__len__'):
        if len(mShell) == 1:
            mShell = mShell[0]
        
    if hasattr(mShell_dot, '__len__'):
        if len(mShell_dot) == 1:
            mShell_dot = mShell_dot[0]
    
    params['shell_mass'].value = mShell
    
    
    
    params['shell_massDot'].value = mShell_dot
            
    
            # ILL MAYBE PLACE THIS AFTER SO THAT WE WONT RECORD THE INTRIDCACIES INSIDE THIS LOOP
            # # print('t here is t=', t)
            # params['array_t_now'].value = np.concatenate([params['array_t_now'].value, [t]])
            # params['array_R2'].value = np.concatenate([params['array_R2'].value, [R2]])
            # params['array_R1'].value = np.concatenate([params['array_R1'].value, [params['R1'].value]])
            # params['array_v2'].value = np.concatenate([params['array_v2'].value, [v2]])
            # params['array_T0'].value = np.concatenate([params['array_T0'].value, [T0]])
            
            # # print('mshell problems', mShell)
            
            # params['array_mShell'].value = np.concatenate([params['array_mShell'].value, [mShell]])
    
    # ---

    
    # gravity correction (self-gravity and gravity between shell and star cluster)
    F_grav = params['G'].value * mShell / R2**2 * (mCluster + mShell/2) 
    
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

    # NEW SSTUFFS HEREE

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
    try:
        # This is only a lazy way of achieving this, utilising the fact that the bubble_r_arr is
        # empty because the first loop does not take into account the calculation of bubble/shell strcuture. 
        # This is not error-proof: in future there needs to be a way to tell apart other than using this try/except method.
        
        # Interpolation is not used, because at very sharp end the values will blow up. Better to use instead
        # the end of array.
        
        if params['current_phase'].value in ['transition', 'momentum']:
            print('in try if')
            # from src.shell_structure import get_shellODE, get_shellParams
            # nShell0 = get_shellParams.get_nShell0(params)
            #     nShell0 = params['mu_neu'].value/params['mu_ion'].value/(params['k_B'].value * params['TShell_ion'].value) * params['Pb'].value
            # simply assume that the buble temperature is 10k at transition and momentum phasew
            # press_HII_out = 2 * params['shell_n0'].value * params['k_B'].value * params['TShell_ion'].value
            # experiment
            nR2 = (params['shell_n0'].value * params['TShell_ion'].value)/3e4
            press_HII_out = 2 * nR2 * params['k_B'].value * 3e4
        elif params['current_phase'].value in ['energy', 'implicit']:
            print('in try else')
            nR2 = bubble_n_arr[0] #takes the first element because the radius array is inversed.
            print('nR2', nR2)
            press_HII_out = 2 * nR2 * params['k_B'].value * params['TShell_ion'].value
        else: 
            sys.exit('phase not found')
            
    except:
        print('in press_HII_out except')
        press_HII_out = get_press_ion(R2, params)
        
    #---
    
    
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
    
    # add: outwards ionising pressure
    
    vd = (4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out) - mShell_dot * v2 - F_grav + FRAD) / mShell
    
    
    # lets say that within the first few runs, mShell_dot is too large (explosive) so that negative value is 
    # too much. Let's make this instead 0.?
    # if params['EarlyPhaseApproximation'].value == True:
    #     mShell_dot = 0
        # vd = -1e8
    
    if params['EarlyPhaseApproximation'].value == True:
        vd = -1e8
    
    print(f'vd is {4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out)} - {mShell_dot * v2} - {F_grav} + {FRAD} divide {mShell} equals {vd}')
    
    Ed = (LWind - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak 

    derivs = [rd, vd, Ed, 0]
    
    params['F_grav'].value = F_grav
    params['F_ion_in'].value = press_HII_in * 4 * np.pi * R2**2 
    params['F_ion_out'].value = press_HII_out * 4 * np.pi * R2**2 
    params['F_ram'].value = press_bubble * 4 * np.pi * R2**2 
    params['F_rad'].value = FRAD
    
    
    # return
    return derivs





# # before merging
# def get_ODE_Edot_OLD(y, t, params):
#     """
#     This ODE solver solves for y (see below). This is being used in run_energy_phase(), after 
#     bubble and shell structure is calculated. This is a general ODE, with more specific ones
#     for other stages in other scripts. 

#     old code: fE_gen()

#     Parameters
#     ----------
#     y contains:
#         - r (R2), shell radius [pc].
#         - v (v2), shell velocity [au]
#         - E (Eb), bubble energy [au]
#     t : time [Myr]

#     params : see run_energy_phase() for more description of params. Here unit does not matter, because
#             astropy will take care of it. Units do matter for y and t, because scipy strips them off. 
#             In case the naming does not makes sense:
        
#         - Lw: mechanical luminosity
#         - pdot_wind: momentum rate 
#         - L_bubble: luminosity loss to cooling (see get_bubbleproperties() in bubble_luminosity.py)
#         - FRAD: radiation pressure coupled to the shell, i.e. Lbol/c * fabs (calculate fabs from shell structure)
#         - tFRAG: time takes to fragmentation
#         - tSCR: sound crossing time

#     Returns
#     -------
#     time derivatives of the ODE: 
#         - drdt  
#         - dvdt  what units?
#         - dEdt  

#     """
    
#     # unpack current values of y (r, rdot, E)
#     R2, v2, Eb = y 
    
    
#     [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot] =  get_currentSB99feedback(t, params)
    
    
#     FABSi = params['shell_fAbsorbedIon'].value
#     FRAD = params['shell_fRad'].value
#     mCluster = params['mCluster'].value
#     LWind = params['LWind'].value
#     L_bubble = params['bubble_LTotal'].value
#     pWindDot = params['pWindDot'].value 
    
#     # calculate shell mass and time derivative of shell mass [au]
#     mShell, mShell_dot = mass_profile.get_mass_profile(R2, params,
#                                                        return_mdot = True, 
#                                                        rdot_arr = v2)
    
#     if len(mShell_dot) == 1:
#         mShell = mShell[0]
#         mShell_dot = mShell_dot[0]
    
#     params['shell_mass'].value = mShell
#     params['shell_massDot'].value = mShell_dot
    
#     def get_press_ion(r, ion_dict):
#         """
#         calculates pressure from photoionized part of cloud at radius r
#         :return: pressure of ionized gas outside shell
        
#         returns in [au]
#         """
        
#         # n_r: total number density of particles (H+, He++, electrons)
#         try:
#             r = np.array([r])
#         except:
#             pass
#         n_r = density_profile.get_density_profile(r, ion_dict)
        
#         P_ion = n_r * ion_dict['k_B'].value * ion_dict['TShell_ion'].value
        
#         return P_ion

#     # calc inward pressure from photoionized gas outside the shell 
#     # (is zero if no ionizing radiation escapes the shell)
#     if FABSi < 1.0:
#         press_HII = get_press_ion(R2, params)
#     else:
#         press_HII = 0.0
        
#     if params['R2'].value >= params['rCloud'].value:
#         # TODO: add this more for ambient pressure
#         press_HII += params['PISM'] * params['k_B']   
    
#     # gravity correction (self-gravity and gravity between shell and star cluster)
#     F_grav = params['G'].value * mShell / R2**2 * (mCluster + mShell/2) 
    
#     # get pressure from energy
#     # calculate radius of inner discontinuity (inner radius of bubble)
    
#     R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 0.0, R2,
#                                args=([LWind, Eb, vWind, R2])) 
    
#     # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
#     # IDEA: move R1 gradually outwards
#     dt_switchon = 1e-3 #* u.Myr # gradually switch on things during this time period
#     tmin = dt_switchon
#     if (t > (tmin + params['tSF'].value)):
#         press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        
#     elif (t <= (tmin + params['tSF'].value)):
#         R1_tmp = (t-params['tSF'].value)/tmin * R1
#         press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
    
    
#     # TODO: Future-------
#     # def calc_coveringf(t,tFRAG,ts):
#     #     """
#     #     estimate covering fraction cf (assume that after fragmentation, during 1 sound crossing time cf goes from 1 to 0)
#     #     if the shell covers the whole sphere: cf = 1
#     #     if there is no shell: cf = 0
        
#     #     Note: I think, since we set tFRAG ultra high, that means cf will almost always be 1. 
#     #     """
#     #     cfmin = 0.4
#     #     # simple slope
#     #     cf = 1. - ((t - tFRAG) / ts)**1.
#     #     cf[cf>1.0] = 1.0
#     #     cf[cf<cfmin] = cfmin
#     #     # return
#     #     return cf

#     # calculate covering fraction
#     # cf = calc_coveringf(np.array([t.value])*u.s,tFRAG,tSCR)
#     # TODO: finish this in the future
#     # cf = 1
#     # leaked luminosity
#     # if cf < 1:
#     #     L_leak = (1. - cf)  * 4. * np.pi * R2 ** 2 * press_bubble * CS / (gamma_adia - 1.)
#     # else:
#     L_leak = 0  
#     #--------
        
#     # time derivatives￼￼
#     rd = v2
#     vd = (4 * np.pi * R2**2 * (press_bubble-press_HII) - mShell_dot * v2 - F_grav + FRAD) / mShell
#     Ed = (LWind - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak 

#     derivs = [rd, vd, Ed]
    
#     params['F_grav'].value = F_grav
#     params['F_ion'].value = press_HII * 4 * np.pi * R2**2 
#     # params['F_ram'].value = (4 * np.pi * R2**2 * (press_bubble - press_HII))
#     # fRad = params['shell_fAbsorbedWeightedTotal'].value * params['Lbol'].value  / (params['c_light'].value)
#     params['F_rad'].value = FRAD
    
    
#     # return
#     return derivs



