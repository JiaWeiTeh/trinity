#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:44:00 2022

@author: Jia Wei Teh

This script contains a function that computes the initial values for the
energy-driven phase (from a short free-streaming phase).
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
#--
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def get_y0(tSF, SB99f):
    """
    Obtain initial values for the energy driven phase.

    Parameters
    ----------
    tSF : float [Myr]
        time of last star formation event (or - if no SF ocurred - time of last recollapse).
    SB99f : func
        starburst99 interpolation functions.

    Returns
    -------
    t0 [Myr] : starting time for Weaver phase (free_expansion phase)
    y0 : An array of initial values. Check comments below for references in the literature
        r0 [pc]: initial separation of bubble edge calculated using (terminal velocity / duration of free expansion phase)
        v0 [km/s]: velocity of expanding bubble (terminal velocity) 
        E0 [erg/s]: energy contained within the bubble
        T0: temperature
        
    """
    # Note:
        # old code: get_startvalues.get_y0()
        
    
    # Make sure it is in the right unit
    tSF = tSF.to(u.Myr)

    Lw_evo0 = SB99f['fLw_cgs'](tSF) * u.erg/u.s
    pdot_evo0 = SB99f['fpdot_cgs'](tSF) * u.g * u.cm / u.s**2
    
    # print(Lw_evo0, pdot_evo0)

    # mass loss rate from winds and SNe (cgs)
    Mdot0_cgs = pdot_evo0**2/(2.*Lw_evo0) 
    # terminal velocity from winds and SNe (cgs)
    vterminal0_cgs = 2.*Lw_evo0/pdot_evo0 

    rhoa =  warpfield_params.nCore * warpfield_params.mu_n
    # duration of inital free-streaming phase (Myr)
    # see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17 Eq 1.15
    dt_phase0 = np.sqrt(3. * Mdot0_cgs / (4. * np.pi * rhoa * vterminal0_cgs ** 3)).to(u.Myr)  
    # print(dt_phase0)
    # start time for Weaver phase (Myr)
    t0 = tSF + dt_phase0  
    # initial separation (pc)
    r0 = vterminal0_cgs * dt_phase0 
    r0 = r0.to(u.pc)
    # initial velocity (km/s)
    v0 = vterminal0_cgs.to(u.km/u.s)
    # The energy contained within the bubble (calculated using wind luminosity)
    # see Weaver+77, eq. (20)
    # New, now changed to erg
    E0 = (5. / 11. * Lw_evo0  * dt_phase0).to(u.erg)
    # Make sure the units are right! see Weaver+77, eq. (37)
    # TODO: isn't it 2.07?
    T0 = 1.51e6 * (Lw_evo0.value/1e36)**(8./35.) * warpfield_params.nCore.value**(2./35.) * (dt_phase0.value)**(-6./35.) * (1.-warpfield_params.xi_Tb)**0.4 * u.K
    # pack into list.
    y0 = [r0, v0, E0, T0]

    return t0, y0    











