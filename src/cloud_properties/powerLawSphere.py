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
    
    # Validate cloud parameters
    validation = validate_cloud_params(
        mCloud=mCloud,
        nCore=nCore,
        rCore=rCore,
        rCloud=rCloud,
        nEdge=nEdge,
        nISM=nISM,
        alpha=alpha,
        mu=mu_ion
    )

    # Print warnings
    for warning in validation['warnings']:
        print(warning)

    # Stop on critical errors
    if validation['errors']:
        for error in validation['errors']:
            print(error)
        sys.exit("Simulation stopped due to invalid cloud parameters.")

    # return
    return rCloud, nEdge


# =============================================================================
# Validation Functions
# =============================================================================

def validate_cloud_params(mCloud, nCore, rCore, rCloud, nEdge, nISM, alpha, mu,
                          tolerance=0.001, r_max=200.0):
    """
    Validate cloud parameters for physical consistency.

    Checks:
    1. Mass consistency: M(rCloud) within tolerance of mCloud
    2. Radius limit: rCloud <= r_max (default 200 pc, typical GMC)
    3. Edge density: nEdge >= nISM

    Parameters
    ----------
    mCloud : float
        Expected cloud mass [Msun]
    nCore : float
        Core number density [cm⁻³]
    rCore : float
        Core radius [pc]
    rCloud : float
        Cloud radius [pc]
    nEdge : float
        Edge density [cm⁻³]
    nISM : float
        ISM density [cm⁻³]
    alpha : float
        Power-law exponent
    mu : float
        Mean molecular weight
    tolerance : float
        Maximum allowed relative mass error (default 0.1% = 0.001)
    r_max : float
        Maximum cloud radius [pc] (default 200)

    Returns
    -------
    dict with keys:
        'valid': bool - All checks passed (no errors)
        'errors': list[str] - Critical errors (simulation should stop)
        'warnings': list[str] - Non-critical warnings
        'mass_error': float - Relative mass error
        'M_computed': float - Computed mass at rCloud [Msun]
    """
    errors = []
    warnings = []

    # 1. Compute mass at rCloud and check consistency
    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc³

    if alpha == 0:
        M_computed = (4.0/3.0) * np.pi * rCloud**3 * rhoCore
    else:
        M_computed = 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0 +
            (rCloud**(3.0 + alpha) - rCore**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore**alpha)
        )

    mass_error = abs(M_computed - mCloud) / mCloud if mCloud > 0 else 0

    # Mass check (CRITICAL - stops simulation)
    if mass_error > tolerance:
        errors.append(
            f"CRITICAL: Mass inconsistency detected!\n"
            f"  Expected mCloud = {mCloud:.4e} Msun\n"
            f"  Computed M(rCloud) = {M_computed:.4e} Msun\n"
            f"  Relative error = {mass_error*100:.4f}% (tolerance: {tolerance*100:.3f}%)\n"
            f"  Check: nCore, rCore, mCloud, densPL_alpha combination"
        )

    # 2. Radius check (WARNING)
    if rCloud > r_max:
        warnings.append(
            f"WARNING: Cloud radius ({rCloud:.1f} pc) exceeds typical single GMC size ({r_max:.0f} pc).\n"
            f"  Such large clouds may be subject to galactic shear."
        )

    # 3. Edge density check (CRITICAL)
    if nEdge < nISM:
        errors.append(
            f"CRITICAL: Edge density ({nEdge:.2e} cm⁻³) < ISM density ({nISM:.2e} cm⁻³)!\n"
            f"  Consider: increasing nCore, decreasing rCore, or reducing |alpha|"
        )

    # If critical errors, find and suggest valid alternatives
    if errors:
        suggestions = find_valid_alternatives(
            mCloud_orig=mCloud,
            nCore_orig=nCore,
            rCore_orig=rCore,
            alpha=alpha,
            nISM=nISM,
            mu=mu,
            n_suggestions=3
        )
        if suggestions:
            errors.append("\n" + "=" * 50)
            errors.append("SUGGESTED VALID PARAMETER COMBINATIONS:")
            errors.append("=" * 50)
            for i, s in enumerate(suggestions, 1):
                errors.append(
                    f"  {i}. mCloud = {s['mCloud']:.2e} Msun, "
                    f"nCore = {s['nCore']:.2e} cm⁻³, "
                    f"rCore = {s['rCore']:.3f} pc\n"
                    f"     → rCloud = {s['rCloud']:.2f} pc, "
                    f"nEdge = {s['nEdge']:.2e} cm⁻³, "
                    f"mass_error = {s['mass_error']*100:.4f}%"
                )
        else:
            errors.append("\nNo valid alternatives found within ±50% of current parameters.")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'mass_error': mass_error,
        'M_computed': M_computed,
        'rCloud': rCloud,
        'nEdge': nEdge
    }


def find_valid_alternatives(mCloud_orig, nCore_orig, rCore_orig, alpha, nISM, mu,
                            n_suggestions=3, search_range=0.5, r_max=200.0,
                            mass_tolerance=0.001):
    """
    Search nearby parameter space for valid (mCloud, nCore, rCore) triplets.

    Searches ±search_range (default 50%) around current values to find
    combinations that satisfy all constraints:
    - Mass consistency (M(rCloud) matches mCloud within tolerance)
    - Edge density >= nISM
    - rCloud <= r_max

    Parameters
    ----------
    mCloud_orig : float
        Original cloud mass [Msun]
    nCore_orig : float
        Original core density [cm⁻³]
    rCore_orig : float
        Original core radius [pc]
    alpha : float
        Power-law exponent
    nISM : float
        ISM density [cm⁻³]
    mu : float
        Mean molecular weight
    n_suggestions : int
        Maximum number of suggestions to return (default 3)
    search_range : float
        Search range as fraction (default 0.5 = ±50%)
    r_max : float
        Maximum cloud radius [pc] (default 200)
    mass_tolerance : float
        Maximum relative mass error (default 0.1% = 0.001)

    Returns
    -------
    list of dict
        Each dict contains 'mCloud', 'nCore', 'rCore', 'rCloud', 'nEdge', 'mass_error'
        Sorted by smallest change from original parameters.
    """
    # Generate search grid
    # For mCloud and nCore: ±50% factors
    mCloud_factors = np.array([1.0 - search_range, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0 + search_range])
    nCore_factors = np.array([1.0 - search_range, 0.8, 0.9, 1.0, 1.1, 1.2, 1.0 + search_range])
    # For rCore: finer grid since it's often the key parameter
    rCore_factors = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    valid_combinations = []

    for mf in mCloud_factors:
        for nf in nCore_factors:
            for rf in rCore_factors:
                # Skip original combination
                if mf == 1.0 and nf == 1.0 and rf == 1.0:
                    continue

                mCloud_test = mCloud_orig * mf
                nCore_test = nCore_orig * nf
                rCore_test = rCore_orig * rf

                # Compute rCloud for this combination
                if alpha == 0:
                    rCloud_test = compute_rCloud_homogeneous(mCloud_test, nCore_test, mu)
                    nEdge_test = nCore_test
                else:
                    try:
                        rCloud_test, _ = compute_rCloud_powerlaw(
                            mCloud_test, nCore_test, alpha,
                            rCore=rCore_test, mu=mu
                        )
                        nEdge_test = nCore_test * (rCloud_test / rCore_test) ** alpha
                    except:
                        continue  # Skip if computation fails

                # Check radius constraint
                if rCloud_test > r_max:
                    continue

                # Check density constraint
                if nEdge_test < nISM:
                    continue

                # Compute mass error
                rhoCore = nCore_test * mu * DENSITY_CONVERSION
                if alpha == 0:
                    M_computed = (4.0/3.0) * np.pi * rCloud_test**3 * rhoCore
                else:
                    M_computed = 4.0 * np.pi * rhoCore * (
                        rCore_test**3 / 3.0 +
                        (rCloud_test**(3.0 + alpha) - rCore_test**(3.0 + alpha)) /
                        ((3.0 + alpha) * rCore_test**alpha)
                    )
                mass_error = abs(M_computed - mCloud_test) / mCloud_test

                # Check mass constraint
                if mass_error <= mass_tolerance:
                    valid_combinations.append({
                        'mCloud': mCloud_test,
                        'nCore': nCore_test,
                        'rCore': rCore_test,
                        'rCloud': rCloud_test,
                        'nEdge': nEdge_test,
                        'mass_error': mass_error
                    })

    # Sort by smallest change from original (in log space for mCloud/nCore)
    def distance_from_original(combo):
        log_m_diff = abs(np.log10(combo['mCloud'] / mCloud_orig)) if mCloud_orig > 0 else 0
        log_n_diff = abs(np.log10(combo['nCore'] / nCore_orig)) if nCore_orig > 0 else 0
        r_diff = abs(combo['rCore'] / rCore_orig - 1) if rCore_orig > 0 else 0
        return log_m_diff + log_n_diff + r_diff

    valid_combinations.sort(key=distance_from_original)
    return valid_combinations[:n_suggestions]