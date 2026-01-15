#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:53:59 2025

@author: Jia Wei Teh

Power-Law Sphere Density Profile
================================

Creates power-law density profiles for molecular clouds:

    n(r) = nCore                    for r ≤ rCore
    n(r) = nCore × (r/rCore)^α      for rCore < r ≤ rCloud
    n(r) = nISM                     for r > rCloud

Physics:
--------
For α = 0 (homogeneous): rCloud = (3M/(4πρ))^(1/3)
For α ≠ 0: M = 4πρc[rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]

Units:
------
- nCore: number density [cm⁻³]
- mCloud: cloud mass [Msun]
- rCloud, rCore: radii [pc]
- mu_ion: mean molecular weight (dimensionless)
"""

import sys
import numpy as np
from scipy.optimize import brentq

# Import unit conversion constants
import src._functions.unit_conversions as cvt

# Density conversion: n [cm⁻³] × μ → ρ [Msun/pc³]
DENSITY_CONVERSION = cvt.CGS.m_H * cvt.INV_CONV.pc2cm**3 / cvt.INV_CONV.Msun2g


# =============================================================================
# Utility functions for computing rCloud from physical parameters
# =============================================================================

def compute_rCloud_homogeneous(M_cloud, nCore, mu=1.4):
    """
    Compute cloud radius for homogeneous (α=0) density profile.

    For homogeneous cloud: M = (4/3)πr³ρ → r = (3M/(4πρ))^(1/3)

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm⁻³]
    mu : float
        Mean molecular weight (default 1.4 for ionized gas)

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    """
    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc³
    rCloud = (3 * M_cloud / (4 * np.pi * rhoCore)) ** (1.0/3.0)
    return rCloud


def compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore=None, rCore_fraction=0.1, mu=1.4):
    """
    Compute cloud radius for power-law density profile.

    Solves for rCloud given M_cloud, nCore, alpha, and either:
    - Fixed rCore value, or
    - rCore = rCore_fraction × rCloud (solved iteratively)

    Mass formula (Rahner+ 2018 Eq 25):
        M = 4πρc [rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm⁻³]
    alpha : float
        Power-law exponent (typically negative, e.g., -2 for isothermal)
    rCore : float, optional
        Fixed core radius [pc]. If None, uses rCore_fraction.
    rCore_fraction : float
        Ratio rCore/rCloud (default 0.1). Used when rCore is None.
    mu : float
        Mean molecular weight (default 1.4)

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]

    Examples
    --------
    >>> # Compute rCloud for 1e5 Msun cloud with nCore=1000 cm⁻³
    >>> rCloud, rCore = compute_rCloud_powerlaw(1e5, 1000, alpha=-2)
    >>> print(f"rCloud = {rCloud:.2f} pc, rCore = {rCore:.2f} pc")
    """
    # Special case: homogeneous
    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(M_cloud, nCore, mu)
        if rCore is None:
            rCore = rCloud * rCore_fraction
        return rCloud, rCore

    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc³

    def mass_at_radius(rCloud_guess, rCore_val):
        """Compute enclosed mass at rCloud given rCore."""
        return 4.0 * np.pi * rhoCore * (
            rCore_val**3 / 3.0 +
            (rCloud_guess**(3.0 + alpha) - rCore_val**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore_val**alpha)
        )

    if rCore is not None:
        # Fixed rCore: solve directly for rCloud
        def mass_residual(rCloud_guess):
            return mass_at_radius(rCloud_guess, rCore) - M_cloud

        # Initial bracket based on homogeneous estimate
        rCloud_homo = compute_rCloud_homogeneous(M_cloud, nCore, mu)
        r_min = max(rCore * 1.01, 0.1 * rCloud_homo)  # rCloud must be > rCore
        r_max = 100.0 * rCloud_homo

        try:
            rCloud = brentq(mass_residual, r_min, r_max, xtol=1e-10)
        except ValueError:
            # Fallback to homogeneous approximation
            rCloud = rCloud_homo

        return rCloud, rCore

    else:
        # rCore = fraction × rCloud: solve iteratively
        def mass_residual_fractional(rCloud_guess):
            rCore_val = rCloud_guess * rCore_fraction
            return mass_at_radius(rCloud_guess, rCore_val) - M_cloud

        # Initial bracket
        rCloud_homo = compute_rCloud_homogeneous(M_cloud, nCore, mu)
        r_min = 0.1 * rCloud_homo
        r_max = 10.0 * rCloud_homo

        try:
            rCloud = brentq(mass_residual_fractional, r_min, r_max, xtol=1e-10)
        except ValueError:
            rCloud = rCloud_homo

        rCore_out = rCloud * rCore_fraction
        return rCloud, rCore_out


def compute_consistent_params(M_cloud, nCore, alpha, rCore_fraction=0.1, mu=1.4, nISM=1.0):
    """
    Compute self-consistent cloud parameters from (M_cloud, nCore, alpha).

    This is the recommended way to set up test parameters, ensuring
    rCloud is computed from the fundamental inputs rather than hardcoded.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm⁻³]
    alpha : float
        Power-law exponent
    rCore_fraction : float
        Ratio rCore/rCloud (default 0.1)
    mu : float
        Mean molecular weight (default 1.4)
    nISM : float
        ISM number density [cm⁻³] (default 1.0)

    Returns
    -------
    dict with keys:
        'rCloud': float [pc]
        'rCore': float [pc]
        'nEdge': float [cm⁻³]
        'M_cloud': float [Msun]
        'nCore': float [cm⁻³]
        'alpha': float
        'mu': float

    Examples
    --------
    >>> params = compute_consistent_params(M_cloud=1e5, nCore=1000, alpha=-2)
    >>> print(f"rCloud = {params['rCloud']:.2f} pc")
    """
    rCloud, rCore = compute_rCloud_powerlaw(M_cloud, nCore, alpha,
                                             rCore_fraction=rCore_fraction, mu=mu)

    # Compute edge density
    if alpha == 0:
        nEdge = nCore
    else:
        nEdge = nCore * (rCloud / rCore) ** alpha

    return {
        'rCloud': rCloud,
        'rCore': rCore,
        'nEdge': nEdge,
        'M_cloud': M_cloud,
        'nCore': nCore,
        'alpha': alpha,
        'mu': mu,
        'nISM': nISM,
    }


# =============================================================================
# Main function for TRINITY pipeline
# =============================================================================

def create_PLSphere(params):
    """
    """

    alpha = params['densPL_alpha'].value
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    rCore = params['rCore'].value
    mu_ion = params['mu_ion'].value
    nISM = params['nISM'].value
    rhoCore = nCore * mu_ion
    
    print(rCore)
    
    # compute cloud radius
    # use core radius/density if there is a power law. If not, use average density.
    if alpha != 0:
        # rCloud = (
        #             (
        #                 mCloud/(4 * np.pi * nCore * mu_atom) - rCore**3/3
        #             ) * rCore ** alpha * (alpha + 3) + rCore**(alpha + 3)
        #           )**(1/(alpha + 3))
        
        
        rCloud = (mCloud * rCore**alpha * (3+alpha) / (4 * np.pi * rhoCore) -\
                    rCore**3 * rCore**alpha * (3 + alpha) / 3 +\
                        rCore**(3+alpha)) ** (1 / (3 + alpha))
                        
        
            
        print(mCloud, rCore, alpha)
        
        print('cloud radius is [pc]', rCloud)
            
        # density at edge
        nEdge = nCore * (rCloud/rCore)**alpha
        
    elif alpha == 0:
        rCloud = (3 * mCloud / 4 / np.pi / (nCore * mu_ion))**(1/3)
        # density at edge should just be the average density
        nEdge = nCore
    
    # sanity check
    if nEdge < nISM:
        print(f'nCore: {nCore}, nISM: {nISM}')
        sys.exit(f"The density at the edge of the cloud ({nEdge}) is lower than the ISM ({nISM}); please consider increasing nCore, or decreasing rCore")
    # return
    return rCloud, nEdge