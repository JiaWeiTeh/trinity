#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile.
"""

# from src.input_tools import get_param
# warpfield_params = get_param.get_param()

def get_density_profile(r_arr,
                         density_params,
                         ):
    """
    This function takes in a list of radius and evaluates the density profile
    based on the points given in the list. The output will depend on selected
    type of density profile describing the sphere.
    
    Watch out the units!

    Parameters
    ----------
    r_arr : list/array of radius of interest
        Radius at which we are interested in the density (Units: pc).
    rCloud : float
        Cloud radius. (Units: pc)

    Returns
    -------
    dens_arr : array of float
        NUMBER DENSITY profile for given radius profile. n(r). (Units: 1/pc^3)

    """
    
    # Note:
        # old code: f_dens and f_densBE().
        
        
    nISM = density_params['nISM_au'].value
    alpha = density_params['alpha_pL'].value
    rCloud = density_params['rCloud_au'].value
    nCore = density_params['nCore_au'].value
    nAvg = density_params['nAvg_au'].value
    rCore = density_params['rCore_au'].value
    nCore = density_params['nCore_au'].value

    # =============================================================================
    # For a power-law profile
    # =============================================================================
    
    # Initialise with power-law
    # for different alphas:
    if alpha == 0:
        dens_arr = nISM * r_arr ** alpha
        dens_arr[r_arr <= rCloud] = nAvg
    else:
        dens_arr = nCore * (r_arr/rCore)**alpha
        dens_arr[r_arr <= rCore] = nCore
        dens_arr[r_arr > rCloud] = nISM
    
    # return n(r)
    return dens_arr
        






