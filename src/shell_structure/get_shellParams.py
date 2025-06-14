#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 22:25:55 2022

@author: Jia Wei Teh

This script includes a mini function that helps compute density of the 
shell at the innermost radius.
"""


def get_nShell0(params):
    
    """
    This function computes density of the shell at the innermost radius via pressure balance(?).

    Returns
    -------
    nShell0 : float
        The density of shell at inner edge/radius.
    nShell0_cloudy : float (TBD)
        The density of shell at inner edge/radius, but including B-field, as
        this will be passed to CLOUDY.

    """
    # TODO: BMW and nMW are given in log units. Is this the same?
    # TODO: Add description for BMW nMW
    
    nShell0 = params['mu_p_au'].value/params['mu_n_au'].value/(params['k_B_au'].value * params['TShell_ion'].value) * params['Pb'].value
    
    # old code
    # pBubble = pBubble.decompose(bases=u.cgs.bases)
    # The density of shell at inner edge/radius
    # nShell0 = warpfield_params.mu_p/warpfield_params.mu_n/(c.k_B.cgs * T.to(u.K)) * pBubble
    
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
    # return nShell0.to(1/u.cm**3)
    return nShell0







