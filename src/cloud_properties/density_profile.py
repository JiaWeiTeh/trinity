#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile.
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
        


def make_density_1_over_t_ism(t_init, n_init, P_ism, T_ism, k_B=None):
    """
    Create n(t) assuming n(t) ∝ 1/t, with final density set by ISM pressure:
        n_final = P_ism / (k_B * T_ism).

    Parameters
    ----------
    t_init : float
        Initial time.
    n_init : float
        Density at t_init.
    P_ism : float
        ISM pressure (in consistent units with k_B and T_ism).
    T_ism : float
        ISM temperature (Kelvin).
    k_B : float, optional
        Boltzmann constant. If None, uses scipy.constants.k (SI units).

    Returns
    -------
    n_of_t : callable
        Function n_of_t(t_now) -> n(t_now) using n(t) = C / t.
    t_final : float
        Final time implied by the 1/t scaling and ISM pressure confinement.
    n_final : float
        Final density set by P_ism / (k_B * T_ism).

    Notes
    -----
    - If you use SI units:
        P_ism in Pa (J/m^3), T_ism in K, k_B in J/K -> n in m^-3.
    - Make sure all quantities are in a consistent unit system.
    """
    if k_B is None:
        from scipy.constants import k as k_B  # Boltzmann constant in J/K (SI)

    if t_init <= 0:
        raise ValueError("t_init must be positive for n ∝ 1/t to make sense here.")
    if n_init <= 0:
        raise ValueError("n_init must be positive.")
    if P_ism <= 0 or T_ism <= 0:
        raise ValueError("P_ism and T_ism must be positive.")

    # Final density from ISM pressure confinement
    n_final = P_ism / (k_B * T_ism)

    if n_final >= n_init:
        raise ValueError(
            "For a decreasing density with n ∝ 1/t, "
            "the ISM-confined n_final must be < n_init."
        )

    # n(t) = C / t, with C fixed by initial condition
    C = n_init * t_init

    # Final time implied by n_final = C / t_final
    t_final = C / n_final  # = n_init * t_init / n_final

    if t_final <= t_init:
        raise RuntimeError(
            "Inconsistent parameters: t_final came out ≤ t_init. "
            "Check units and values of P_ism, T_ism, and k_B."
        )

    def n_of_t(t_now):
        """
        Density at time t_now using the exact 1/t scaling.
        """
        t_arr = np.asarray(t_now)
        n_arr = C / t_arr
        return float(n_arr) if np.isscalar(t_now) else n_arr

    return n_of_t, t_final, n_final







