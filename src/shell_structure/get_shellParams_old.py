#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 22:25:55 2022

@author: Jia Wei Teh

This script includes a mini function that helps compute density of the 
shell at the innermost radius.
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import scipy.optimize
import sys

# get parameter
import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()


def get_nShell0(pBubble, T):
    """
    This function computes density of the shell at the innermost radius.

    Parameters
    ----------
    pBubble : pressure of the bubble
        DESCRIPTION.
    T : float (units: K)
        Temperature of at inner edge of shell.
            
    Returns
    -------
    nShell0 : float
        The density of shell at inner edge/radius.
    nShell0_cloudy : float
        The density of shell at inner edge/radius, but including B-field, as
        this will be passed to CLOUDY.

    """
    # TODO: BMW and nMW are given in log units. Is this the same?
    # TODO: Add description for BMW nMW
    
    
    pBubble = pBubble.decompose(bases=u.cgs.bases)
    # The density of shell at inner edge/radius
    nShell0 = warpfield_params.mu_p/warpfield_params.mu_n/(c.k_B.cgs * T.to(u.K)) * pBubble
    
    # TODO: here, CLOUDY stuffs are removed.
    
    # The density of shell at inner edge/radius that is passed to cloudy (usually includes B-field)
    # Note: this is only used to pass on to CLOUDY and does not affect WARPFIELD.
    # Assuming equipartition and pressure equilibrium, such that
    # Pwind = Pshell, where Pshell = Ptherm + Pturb + Pmag
    #                              = Ptherm + 2Pmag
    # where Pmag \propro n^(gamma/2)
    
    # BMW = 10**(warpfield_params.log_BMW)
    # nMW = 10**(warpfield_params.log_nMW)
    
    # def pShell(n, pBubble, T):
    #     # return function
    #     return warpfield_params.mu_n/warpfield_params.mu_p * c.k_B.cgs * T * n +\
    #                 BMW**2 / (4 * np.pi * nMW**warpfield_params.gamma_mag) * n ** (4/3) - pBubble
  
    # nShell0_cloudy = scipy.optimize.fsolve(pShell, x0 = 10,
    #                                        args = (pBubble, T))[0]
    
    # return nShell0, nShell0_cloudy
    return nShell0.to(1/u.cm**3)







