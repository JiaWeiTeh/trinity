#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile, given 
"""

import numpy as np
from src.cloud_properties import bonnorEbertSphere


def get_density_profile(r_arr,
                         params,
                         ):
    """
    Density profile (if r_arr is an array), otherwise the density at point r.
    """
    
    nISM = params['nISM'].value
    rCloud = params['rCloud'].value
    nCore = params['nCore'].value
    rCore = params['rCore'].value
    nCore = params['nCore'].value

    if type(r_arr) is not np.ndarray:
        r_arr = np.array([r_arr])
        
    # =============================================================================
    # For a power-law profile
    # =============================================================================
    
    if params['dens_profile'].value == 'densPL':
        alpha = params['densPL_alpha'].value
        # Initialise with power-law
        # for different alphas:
        if alpha == 0:
            n_arr = nISM * r_arr ** alpha
            n_arr[r_arr <= rCloud] = nCore
        else:
            n_arr = nCore * (r_arr/rCore)**alpha
            n_arr[r_arr <= rCore] = nCore
            n_arr[r_arr > rCloud] = nISM
        
        
    elif params['dens_profile'].value == 'densBE':
        
        f_rho_rhoc = params['densBE_f_rho_rhoc'].value
        
        xi_arr = bonnorEbertSphere.r2xi(r_arr, params)
        
        # print(xi_arr)

        rho_rhoc = f_rho_rhoc(xi_arr)
        
        n_arr = rho_rhoc * params['nCore'] 
        
        n_arr[r_arr > rCloud] = nISM
        
        # print(n_arr)
        
    # return n(r)
    return n_arr
        






