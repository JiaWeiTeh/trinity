#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Strömgren sphere calculation for TRINITY.

Computes the Strömgren radius and HII pressure from the ambient cloud
density profile, accounting for the transparent bubble cavity.

This provides an independent P_HII that is NOT anchored to the bubble
pressure Pb, unlike the shell-structure-derived P_HII which is ≈ Pb
by construction.

@author: Jia Wei Teh
"""

import numpy as np
import scipy.integrate
import scipy.optimize

from src.cloud_properties import density_profile


def compute_P_HII_Stromgren(Qi, R2, params):
    """
    Compute standalone Strömgren HII pressure from ambient cloud profile,
    accounting for the transparent bubble cavity.

    Solves:  Qi = alpha_B * integral_{R2}^{R_St} [ n_cloud^2(r) 4 pi r^2 dr ]
    Returns: P_HII_St = 2 * n_cloud(R_St) * k_B * T_ion

    The lower integration limit is R2 (bubble outer radius), not 0:
    the hot bubble interior is transparent to ionizing photons, so the
    full Qi budget arrives at R2 undiminished.

    Parameters
    ----------
    Qi : float
        Ionising photon rate [1/Myr] (code units).
    R2 : float
        Current bubble outer radius [pc]. Used as lower integration limit.
    params : dict-like
        Must contain keys needed by density_profile.get_density_profile(),
        plus 'caseB_alpha', 'k_B', 'TShell_ion', 'rCloud'.
        All values in code units [Msun, pc, Myr].

    Returns
    -------
    P_HII_St : float
        Standalone Strömgren HII pressure [Msun/Myr^2/pc] (code units).
    R_St : float
        Strömgren radius [pc] (measured from centre, not from R2).
    n_St : float
        Ambient cloud density at R_St [1/pc^3] (code units).

    Notes
    -----
    If Qi <= 0, returns (0, 0, 0).
    If R_St > rCloud, clamps to rCloud and uses n_cloud(rCloud).
    """
    if Qi <= 0:
        return 0.0, 0.0, 0.0

    # Respect include_PHII flag: when False, HII pressure is disabled entirely
    if not params['include_PHII'].value:
        return 0.0, 0.0, 0.0

    alpha_B = params['caseB_alpha'].value
    k_B = params['k_B'].value
    T_ion = params['TShell_ion'].value
    rCloud = params['rCloud'].value

    # Floor R2 to avoid numerical issues with power-law profiles near r=0
    R2_floor = max(R2, 1e-6)

    # Target: integral from R2 to R_St of n^2(r) * 4*pi*r^2 dr = Qi / alpha_B
    target = Qi / alpha_B

    def integrand(r):
        n = density_profile.get_density_profile(r, params)
        return n**2 * 4.0 * np.pi * r**2

    # Check if the entire cloud can balance Qi
    total_integral, _ = scipy.integrate.quad(
        integrand, R2_floor, rCloud,
        limit=100, epsrel=1e-8
    )

    if total_integral <= target:
        # HII region extends beyond cloud — clamp to rCloud
        R_St = rCloud
        n_St = density_profile.get_density_profile(rCloud, params)
        P_HII_St = 2.0 * n_St * k_B * T_ion
        return P_HII_St, R_St, n_St

    # Find R_St via root-finding: cumulative integral = target
    def residual(r_upper):
        integral, _ = scipy.integrate.quad(
            integrand, R2_floor, r_upper,
            limit=100, epsrel=1e-8
        )
        return integral - target

    # R_St must be between R2 and rCloud
    R_St = scipy.optimize.brentq(
        residual, R2_floor, rCloud,
        xtol=1e-10, rtol=1e-10
    )

    n_St = density_profile.get_density_profile(R_St, params)
    P_HII_St = 2.0 * n_St * k_B * T_ion

    return P_HII_St, R_St, n_St
