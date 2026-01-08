#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 23:14:53 2025

@author: Jia Wei Teh

Update SB99 feedback values across dictionary
"""

from src._input.dictionary import updateDict
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


def get_currentSB99feedback(t, params):
    
    SB99f = params['SB99f'].value
    
    # mechanical luminosity at time t_midpoint (erg)
    LWind = SB99f['fLw'](t)[()]  
    Lbol = SB99f['fLbol'](t)[()]
    Ln = SB99f['fLn'](t)[()]
    Li = SB99f['fLi'](t)[()]
    # get the slope via mini interpolation for some dt.
    dt = 1e-9 #*Myr
    # force of SN
    pdot_SNe = SB99f['fpdot_SNe'](t)[()]
    # force of stellar winds at time t0 (cgs)
    pWindDot = SB99f['fpdot'](t)[()]
    pWindDotDot = (SB99f['fpdot'](t + dt)[()] - SB99f['fpdot'](t - dt)[()])/ (dt+dt)
    # terminal wind velocity at time t0 (pc/Myr)
    vWind = (2. * LWind / pWindDot)[()]
    # ionizing
    Qi = SB99f['fQi'](t)[()]
    
    # dont really have to return because dictionaries update themselves, but still, for clarity
    updateDict(params, ['Qi', 'LWind', 'Lbol', 'Ln', 'Li', 'vWind', 'pWindDot', 'pWindDotDot'],
                       [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot],
               )
    
    # also
    # collect values
    # this pWindDot is actually pRamDot=pWindDot+pSNeDot (see read_SB99. this is a huge misname)
    params['F_ram_wind'].value = pWindDot - pdot_SNe
    params['F_ram_SN'].value = pdot_SNe
    
    return [Qi, LWind, Lbol, Ln, Li, vWind, pWindDot, pWindDotDot]

