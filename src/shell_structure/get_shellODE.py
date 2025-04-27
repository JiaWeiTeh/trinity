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
                  # cons,
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
    
    # make sure it is all in cgs
    
    # sigma_dust = params['sigma_d_au'].value / cvt.cm2pc**2
    # mu_n = params['mu_n_au'].value / cvt.g2Msun
    # mu_p = params['mu_p_au'].value / cvt.g2Msun
    # t_ion = params['t_ion'].value
    # t_neu = params['t_neu'].value
    # alpha_B = params['alpha_B_au'].value / cvt.cm2pc**3 * cvt.s2Myr #cm3/s
    # k_B = params['k_B_au'].value / cvt.k_B_cgs2au
    # c = params['c_au'].value / cvt.v_cms2au
    # Ln = params['Ln'].value / cvt.L_cgs2au 
    # Li = params['Li'].value / cvt.L_cgs2au 
    # Qi = params['Qi'].value * cvt.s2Myr
    
    # try non cgs here
    sigma_dust = params['sigma_d_au'].value
    mu_n = params['mu_n_au'].value   
    mu_p = params['mu_p_au'].value 
    t_ion = params['t_ion'].value
    t_neu = params['t_neu'].value
    alpha_B = params['alpha_B_au'].value  #cm3/s
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
        
        # nShell *= (1/u.cm**3)
        # Ln, Li, Qi = cons
        # Ln = Ln.to(u.)
        
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
        # return dndr.to(1/u.cm**4).value, dphidr.to(1/u.cm).value, dtaudr.to(1/u.cm).value

    
    # If not, omit ionised paramters such as Li and phi.
    else:
        # unravel
        nShell, tau = y
        
        # nShell *= (1/u.cm**3)
        # Ln, Qi = cons
        
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
        # return dndr.to(1/u.cm**4).value, dtaudr.to(1/u.cm).value






