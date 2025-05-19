#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:22:09 2022

@author: Jia Wei Teh

This script contains a function that returns ODE of the ionised number density (n), 
fraction of ionizing photons that reaches a surface with radius r (phi), and the
optical depth (tau) of the shell.
"""

import numpy as np
import sys

import src._functions.unit_conversions as cvt


# TODO: add cover fraction cf (f_cover)

def get_shellODE(y, 
                 r, 
                 f_cover,
                 is_ionised,
                 params,
                 ):
    """
    A function that returns ODE of the ionised number density (n), 
    fraction of ionizing photons that reaches a surface with radius r (phi), and the
    optical depth of dust (tau) of the shell.
    
    This routine assumes cgs
    
    Parameters
    ----------
    y : list
        A list of ODE variable, including:
        # nShell [1/pc3]: float
            the number density of the shell.
        # phi [unitless]: float
            fraction of ionizing photons that reaches a surface with radius r.
        # tau [unitless]]: float
            the optical depth of dust in the shell.               
    r [pc]: list
        An array of radii where y is evaluated.
                
    f_cover: float, 0 < f_cover <= 1
            The fraction of shell that remained after fragmentation process.
            f_cover = 1: all remained.
    is_ionised: boolean
            Is this part of the shell ionised? If not, then phi = Li = 0, where
            r > R_ionised.

    Returns
    -------
    dndr [1/pc4]: ODE 
    dphidr [1/pc]: ODE (only in ionised region)
    dtaudr [1/pc]: ODE

    """

    sigma_dust = params['sigma_d_au'].value
    mu_n = params['mu_n_au'].value   
    mu_p = params['mu_p_au'].value 
    t_ion = params['t_ion'].value
    t_neu = params['t_neu'].value
    alpha_B = params['alpha_B_au'].value  #cm3/s (au)
    k_B = params['k_B_au'].value  
    c = params['c_au'].value  
    Ln = params['Ln'].value  
    Li = params['Li'].value 
    Qi = params['Qi'].value  
    
    # Is this region of the shell ionised?
    # If yes:
    if is_ionised:
        # unravel, and make sure they are in the right units
        nShell, phi, tau = y
        
        # prevent underflow for very large tau values
        if tau > 500:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)
        
        # number density
        dndr = mu_p/mu_n/(k_B * t_ion) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau + Li * phi)\
                + nShell**2 * alpha_B * Li / Qi / c
            )
        # ionising photons
        dphidr = - 4 * np.pi * r**2 * alpha_B * nShell**2 / Qi - nShell * sigma_dust * phi
        # optical depth
        dtaudr = nShell * sigma_dust * f_cover
        
        # return
        return dndr, dphidr, dtaudr

    
    # If not, omit ionised paramters such as Li and phi.
    else:
        # unravel
        nShell, tau = y
        
        # prevent underflow for very large tau values
        if tau > 500:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)        
            
        # number density
        dndr = 1/(k_B * t_neu) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau) 
            )
        # optical depth
        dtaudr = nShell * sigma_dust
        
        # return
        return dndr, dtaudr






