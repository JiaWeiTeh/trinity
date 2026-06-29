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

import trinity._functions.unit_conversions as cvt


# Numerical guard for the ionised RHS. The ionised shell ODE has a dn/dr ∝ +nShell**2
# recombination term, which is a finite-radius pole: just past the ionisation front
# nShell runs away toward infinity. shell_structure truncates the profile AT the front
# (first phi<=1e-9 / mass-limited row), so that runaway tail is DISCARDED -- but odeint
# still integrates through it, and nShell**2 overflows float64 (1.8e308) -> inf/nan ->
# LSODA is driven to machine-precision steps and floods "t + h = t" warnings.
# Capping nShell keeps nShell**2 finite in that discarded tail so the integrator steps
# cleanly. This is a NUMERICAL safety rail, NOT a physics cutoff: the cap is ~55 orders
# of magnitude above any physical shell density (the ionisation front peaks at ~1e65 in
# code units, i.e. ~1e10 cm^-3; a neutron star is ~1e38 cm^-3), so it never bites in the
# used region -- the consumed shell profile is bit-identical to the unguarded solve
# (verified end-to-end, docs/dev/shell-solver/OVERFLOW_FIX_PLAN.md). The value keeps
# nShell**2 times the ~1e55 dndr prefactor well under float64's ceiling.
_NSHELL_MAX = 1e120


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
    
    All quantities are in code units [Msun, pc, Myr] (see the parameter annotations below).
    
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

    sigma_dust = params['dust_sigma'].value
    mu_n = params['mu_atom'].value
    mu_p = params['mu_ion_shell'].value   # shell HII is singly ionised (Z_He_shell)
    mu_H = params['mu_convert'].value
    chi_e = params['chi_e_shell'].value   # shell electron factor (singly ionised)
    t_ion = params['TShell_ion'].value
    t_neu = params['TShell_neu'].value
    alpha_B = params['caseB_alpha'].value  # case-B recombination coeff [code units; physically cm^3/s]
    k_B = params['k_B'].value  
    c = params['c_light'].value  
    Ln = params['Ln'].value  
    Li = params['Li'].value 
    Qi = params['Qi'].value  
    
    # Is this region of the shell ionised?
    # If yes:
    if is_ionised:
        # unravel, and make sure they are in the right units
        nShell, phi, tau = y

        # numerical guard: cap nShell so the +nShell**2 pole in the discarded
        # post-front tail cannot overflow float64 (see _NSHELL_MAX above).
        nShell = min(nShell, _NSHELL_MAX)
        
        # prevent underflow for very large tau values
        if tau > 500:
            neg_exp_tau = 0
        else:
            neg_exp_tau = np.exp(-tau)
        
        # Clamp phi: negative values are unphysical (ionizing photons cannot be regenerated).
        # This prevents the -n*sigma_d*phi term from acting as a photon source
        # and Li*phi from inverting the radiation pressure gradient.
        phi = max(0.0, phi)   # <-- add this line


        # number density
        dndr = mu_p/mu_H/(k_B * t_ion) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau + Li * phi)\
                + chi_e * nShell**2 * alpha_B * Li / Qi / c
            )
        # ionising photons
        dphidr = - 4 * np.pi * r**2 * chi_e * alpha_B * nShell**2 / Qi - nShell * sigma_dust * phi
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
        dndr = mu_n/mu_H/(k_B * t_neu) * (
            nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau) 
            )
        # optical depth
        dtaudr = nShell * sigma_dust
        
        # return
        return dndr, dtaudr






