#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:44:00 2022

@author: Jia Wei Teh

This script contains a function that computes the initial values for the
energy-driven phase (from a short free-streaming phase).
"""

import numpy as np
import src._functions.unit_conversions as cvt

def get_y0(params):
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
    tSF = params['tSF'].value
    SB99f = params['SB99f'].value

    Lw_evo0 = SB99f['fLw_cgs'](tSF) * cvt.L_cgs2au
    pdot_evo0 = SB99f['fpdot_cgs'](tSF) * cvt.pdot_cgs2au
    
    # print(Lw_evo0, pdot_evo0)

    # mass loss rate from winds and SNe (cgs)
    Mdot0 = pdot_evo0**2/(2.*Lw_evo0) 
    # terminal velocity from winds and SNe (cgs)
    # initial valocity (pc/Myr)
    v0 = 2.*Lw_evo0/pdot_evo0 

    rhoa =  params['nCore_au'].value * params['mu_n_au'].value
    # duration of inital free-streaming phase (Myr)
    # see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17 Eq 1.15
    dt_phase0 = np.sqrt(3. * Mdot0 / (4. * np.pi * rhoa * v0 ** 3))
    # print(dt_phase0)
    # start time for Weaver phase (Myr)
    t0 = tSF + dt_phase0  
    # initial separation (pc)
    r0 = v0 * dt_phase0 
    # The energy contained within the bubble (calculated using wind luminosity)
    # see Weaver+77, eq. (20)
    # In au units (Myr, pc, Msun)
    E0 = 5. / 11. * Lw_evo0  * dt_phase0
    # Make sure the units are right! see Weaver+77, eq. (37)
    # TODO: isn't it 2.07?
    # entry changed to L [erg/s], t [Myr] and n [/cm3]
    T0 = 1.51e6 * (Lw_evo0 / cvt.L_cgs2au / 1e36)**(8/35) * \
                (params['nCore_au'].value / cvt.ndens_cgs2au)**(2./35.) * \
                    (dt_phase0)**(-6./35.) * \
                        (1 - params['xi_Tb'].value)**0.4
    # update
    params['t_now'].value = t0
    params['R2'].value = r0
    params['v2'].value = v0
    params['Eb'].value = E0 
    params['T0'].value = T0 

    return 











