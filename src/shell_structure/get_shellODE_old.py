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
import astropy.constants as c
import astropy.units as u

import os
import importlib
warpfield_params = importlib.import_module(os.environ['WARPFIELD3_SETTING_MODULE'])

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def get_shellODE(y, 
                 r, 
                 cons,
                 f_cover = 1,
                 is_ionised = True
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
        # nShell [1/cm3]: float
            the number density of the shell.
        # phi [unitless]: float
            fraction of ionizing photons that reaches a surface with radius r.
        # tau [unitless]]: float
            the optical depth of dust in the shell.               
    r [pc]: list
        An array of radii where y is evaluated.
    cons : list
        A list of constants used in the ODE, including:
            Ln, Li and Qi. In erg/s and 1/s
                
    f_cover: float, 0 < f_cover <= 1
            The fraction of shell that remained after fragmentation process.
            f_cover = 1: all remained.
    is_ionised: boolean
            Is this part of the shell ionised? If not, then phi = Li = 0, where
            r > R_ionised.

    Returns
    -------
    dndr [1/cm4]: ODE 
    dphidr [1/cm]: ODE (only in ionised region)
    dtaudr [1/cm]: ODE

    """
    
    sigma_dust = warpfield_params.sigma_d 
    mu_n = warpfield_params.mu_n 
    mu_p = warpfield_params.mu_p
    t_ion = warpfield_params.t_ion
    t_neu = warpfield_params.t_neu
    alpha_B = warpfield_params.alpha_B
    # UNITS
    r *= u.cm
    # TODO: Add f_cover
    
    # Is this region of the shell ionised?
    # If yes:
    if is_ionised:
        # unravel, and make sure they are in the right units
        nShell, phi, tau = y
        nShell *= (1/u.cm**3)
        Ln, Li, Qi = cons
        
        # prevent underflow for very large tau values
        if tau > 700:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)
            
        
        # number density
        dndr = mu_p/mu_n/(c.k_B.cgs * t_ion) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c.c.cgs) * (Ln * neg_exp_tau + Li * phi)\
                + nShell**2 * alpha_B * Li / Qi / c.c.cgs
            )
    
        dphidr = - 4 * np.pi * r**2 * alpha_B * nShell**2 / Qi - nShell * sigma_dust * phi
        # optical depth
        dtaudr = nShell * sigma_dust * f_cover
        
        # return
        return dndr.to(1/u.cm**4).value, dphidr.to(1/u.cm).value, dtaudr.to(1/u.cm).value
    
    
    # If not, omit ionised paramters such as Li and phi.
    else:
        # unravel
        nShell, tau = y
        nShell *= (1/u.cm**3)
        Ln, Qi = cons
        
        # prevent underflow for very large tau values
        if tau > 700:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)        
            
        # number density
        dndr = 1/(c.k_B.cgs * t_neu) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c.c.cgs) * (Ln * neg_exp_tau) 
            )
        # optical depth
        dtaudr = nShell * sigma_dust
        # return
        return dndr.to(1/u.cm**4).value, dtaudr.to(1/u.cm).value






