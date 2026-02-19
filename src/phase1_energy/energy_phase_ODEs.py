 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh
"""
# libraries
import scipy
import numpy as np
import scipy.optimize
import src.cloud_properties.mass_profile as mass_profile
import src.cloud_properties.density_profile as density_profile
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.sb99.update_feedback import get_currentSB99feedback

import logging
# Initialize logger for this module
logger = logging.getLogger(__name__)


def _scalar(x):
    """Convert len-1 arrays / 0-d arrays to Python scalars; otherwise return x."""
    a = np.asarray(x)
    return a.item() if a.size == 1 else x

def get_press_ion(r, params):
    """
    Pressure from photoionized part of cloud at radius r.

    Parameters
    ----------
    r : float
        Radius in pc.
    params : DescribedDict
        Parameter dictionary.

    Returns
    -------
    float
        Pressure of ionized gas in code units.
    """
    r = np.atleast_1d(r)
    n_r = density_profile.get_density_profile(r, params)
    P_ion = 2.0 * n_r * params['k_B'].value * params['TShell_ion'].value
    return _scalar(P_ion)


def get_ODE_Edot(y, t, params):
    """
    ODE system for bubble expansion.
    y = [R2, v2, Eb]   
    t in Myr
    """
    # radius, velocity, energy
    R2, v2, Eb = y
    
    # update values, because other calculations may depend on dictionary params.
    # parameter should always reflect the current time
    params['t_now'].value = t
    params['R2'].value = R2
    params['v2'].value = v2
    params['Eb'].value = Eb
    
    # --- pull frequently-used parameters once
    FABSi     = params["shell_fAbsorbedIon"].value
    F_rad     = params["shell_F_rad"].value
    mCluster  = params["mCluster"].value
    L_bubble  = params["bubble_LTotal"].value
    G         = params["G"].value
    Qi        = params["Qi"].value
    Lmech_total     = params["Lmech_total"].value
    v_mech_total     = params["v_mech_total"].value
    
    
    # --- calculate shell mass and time derivative of shell mass 
    # if collapse, do not calculate and take previous run value.
    if params['isCollapse'].value == True:
        mShell = params['shell_mass'].value
        mShell_dot = 0
    # if not, calculate
    else:
        mShell, mShell_dot = mass_profile.get_mass_profile(
            R2, params, return_mdot = True, rdot = v2
            )
        mShell = _scalar(mShell)
        mShell_dot = _scalar(mShell_dot)
    # update value
    params['shell_mass'].value = mShell
    params['shell_massDot'].value = mShell_dot
            
    
    # --- gravity force (self + cluster)
    F_grav = G * mShell / (R2**2) * (mCluster + 0.5 * mShell)
    
    # calculate radius of inner discontinuity (inner radius of bubble)
    # grab feedback
    feedback = get_currentSB99feedback(t, params)
    # calculate
    R1 = scipy.optimize.brentq(get_bubbleParams.get_r1, 1e-3*R2, R2,
                               args=([feedback.Lmech_total, Eb, feedback.v_mech_total, R2])) 
    
    # --- ram pressure
    # the following if-clause needs to be rethought. for now, this prevents negative energies at very early times
    # IDEA: move R1 gradually outwards
    dt_switchon = 1e-3 #* u.Myr # gradually switch on things during this time period
    tmin = dt_switchon

    # if momentum phase, it's just ram
    if params['current_phase'].value in ['momentum']:
        press_bubble = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
    # transition phase: max(P_thermal, P_ram) for smooth handoff to momentum
    elif params['current_phase'].value == 'transition':
        P_thermal = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)
        P_ram = get_bubbleParams.pRam(R2, Lmech_total, v_mech_total)
        press_bubble = max(P_thermal, P_ram)
    # energy/implicit phases: calculated from bubble energy
    else:
        if (t > (tmin + params['tSF'].value)):
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1, params['gamma_adia'].value)

        elif (t <= (tmin + params['tSF'].value)):
            R1_tmp = (t-params['tSF'].value)/tmin * R1
            press_bubble = get_bubbleParams.bubble_E2P(Eb, R2, R1_tmp, params['gamma_adia'].value)
    # update
    params['Pb'].value = press_bubble  
    
    
    # --- inward pressure from photoionized gas outside the shell 
    # (is zero if no ionizing radiation escapes the shell)
    if FABSi < 1.0:
        press_HII_in = get_press_ion(params['rShell'].value, params)
    else:
        press_HII_in = 0.0
    
    # if shell is beyond cloud, add also ISM pressure
    if params['rShell'].value >= params['rCloud'].value:
        # TODO: add this more for ambient pressure
        press_HII_in += params['PISM'].value * params['k_B'].value   
        
        
    # --- photoionised gas from HII region approximation
    # if ions escape cloud, HII region has density of outer ISM
    if FABSi < 1:
        nR2 = params['nISM'].value
    # otherwise, it has density assuming radius = bubble radius
    else:
        nR2 = np.sqrt(Qi/params['caseB_alpha'].value/R2**3 * 3 / 4 / np.pi)
    # calculate final
    press_HII_out = 2 * nR2 * params['k_B'].value * 3e4
    #---
    
     
    # TODO: Future------- add cover fraction after fragmentation
    L_leak = 0  
    #--------
        
    # time derivatives￼￼ via energy and momentum equation
    rd = v2
    vd = (4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out) - mShell_dot * v2 - F_grav + F_rad) / mShell
    
    # lets say that within the first few runs, mShell_dot is too large (explosive) so that negative value is 
    # too much. Let's make this instead 0.?
    if params['EarlyPhaseApproximation'].value == True:
        vd = -1e8
    
    # Ed is obtained via conversion of beta/delta in run_implicit_energy.py, but calculated here for run_energy_phase.
    Ed = (Lmech_total - L_bubble) - (4 * np.pi * R2**2 * press_bubble) * v2 - L_leak 
    
    # debug message
    logger.debug(f'vd is {4 * np.pi * R2**2 * (press_bubble-press_HII_in+press_HII_out)} - {mShell_dot * v2} - {F_grav} + {F_rad} divide {mShell} equals {vd}')

    # calculate forces
    params['F_grav'].value = F_grav
    params['F_ion_in'].value = press_HII_in * 4 * np.pi * R2**2 
    params['F_ion_out'].value = press_HII_out * 4 * np.pi * R2**2 
    params['F_ram'].value = press_bubble * 4 * np.pi * R2**2 
    params['F_rad'].value = F_rad
    
    # return
    return [rd, vd, Ed]







