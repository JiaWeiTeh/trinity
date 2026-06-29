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
All quantities are in code units [Msun, pc, Myr] after conversion by read_param.py:
- nCore: number density [1/pc³] (converted from cm⁻³ via ndens_cgs2au)
- mCloud: cloud mass [Msun]
- rCloud, rCore: radii [pc]
- mu: mean molecular weight [Msun] (converted from m_H)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Note on units
# =============================================================================
# All parameters are expected to be in internal units [Msun, pc, Myr] as
# converted by read_param.py:
#   - nCore, nISM: [1/pc³] (converted from cm⁻³ via ndens_cgs2au)
#   - mu: [Msun] (converted from m_H units via m_H * g2Msun)
#
# Therefore: rho = n * mu directly gives [Msun/pc³]


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
        Core number density [1/pc³] (internal units, converted from cm⁻³)
    mu : float
        Mean molecular weight [Msun] (internal units, converted from m_H)

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    """
    # In internal units: n [1/pc³] * mu [Msun] = rho [Msun/pc³]
    rhoCore = nCore * mu
    rCloud = (3 * M_cloud / (4 * np.pi * rhoCore)) ** (1.0/3.0)
    return rCloud


def compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore=None, rCore_fraction=0.1, mu=1.4):
    """
    Compute cloud radius for power-law density profile.

    Uses the analytical inversion of the enclosed-mass formula
    (Rahner+ 2018 Eq 25) — no numerical root-finding required.

    Mass formula:
        M = 4πρc [rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]

    Solving for rCloud (valid for α ≠ 0, α ≠ -3):
        rCloud = { [M/(4πρc) - rCore³/3] × (3+α) × rCore^α + rCore^(3+α) }^(1/(3+α))

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [1/pc³] (code units)
    alpha : float
        Power-law exponent (typically negative, e.g., -2 for isothermal)
    rCore : float, optional
        Fixed core radius [pc]. If None, uses rCore_fraction.
    rCore_fraction : float
        Ratio rCore/rCloud (default 0.1). Used when rCore is None.
    mu : float
        Mean molecular weight [Msun] (code units)

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]

    Raises
    ------
    ValueError
        If α ≈ -3 (mass integral diverges) or if the parameters are
        unphysical (core mass alone exceeds cloud mass).

    Examples
    --------
    >>> # Compute rCloud for 1e5 Msun cloud with nCore and mu in code units
    >>> rCloud, rCore = compute_rCloud_powerlaw(1e5, nCore, alpha=-2)
    >>> print(f"rCloud = {rCloud:.2f} pc, rCore = {rCore:.2f} pc")
    """
    # Special case: homogeneous
    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(M_cloud, nCore, mu)
        if rCore is None:
            rCore = rCloud * rCore_fraction
        return rCloud, rCore

    # In internal units: n [1/pc³] * mu [Msun] = rho [Msun/pc³]
    rhoCore = nCore * mu

    def mass_at_radius(rCloud_guess, rCore_val):
        """Compute enclosed mass at rCloud given rCore."""
        return 4.0 * np.pi * rhoCore * (
            rCore_val**3 / 3.0 +
            (rCloud_guess**(3.0 + alpha) - rCore_val**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore_val**alpha)
        )

    # Guard: α = -3 makes the mass integral diverge
    if abs(3.0 + alpha) < 1e-14:
        raise ValueError(
            "alpha approx -3: mass integral diverges, cannot compute rCloud."
        )

    if rCore is not None:
        # ----- Fixed rCore: analytical inversion (Rahner+ 2018 Eq 25) -----
        # M = 4πρc [rc³/3 + (rCl^(3+α) - rc^(3+α)) / ((3+α) rc^α)]
        # => rCl^(3+α) = [M/(4πρc) - rc³/3] (3+α) rc^α  +  rc^(3+α)
        A = M_cloud / (4.0 * np.pi * rhoCore) - rCore**3 / 3.0
        rhs = A * (3.0 + alpha) * rCore**alpha + rCore**(3.0 + alpha)

        if rhs <= 0:
            raise ValueError(
                f"Unphysical parameters: the uniform core (r<={rCore:.3f} pc) "
                f"already exceeds the cloud mass budget. "
                f"M_cloud={M_cloud:.3e}, nCore={nCore:.3e}, alpha={alpha}, "
                f"rCore={rCore:.3f}"
            )

        rCloud = rhs ** (1.0 / (3.0 + alpha))

        # Forward mass check
        M_check = mass_at_radius(rCloud, rCore)
        rel_err = abs(M_check - M_cloud) / M_cloud
        if rel_err > 1e-6:
            raise RuntimeError(
                f"Analytical rCloud failed consistency check: "
                f"M(rCloud)={M_check:.6e} vs M_cloud={M_cloud:.6e}, "
                f"rel_err={rel_err:.2e}"
            )
        logger.debug(
            f"rCloud_powerlaw (fixed rCore): alpha={alpha}, "
            f"rCore={rCore:.4f}, rCloud={rCloud:.4f}, rel_err={rel_err:.2e}"
        )
        return rCloud, rCore

    else:
        # ----- Fractional rCore: rCore = f × rCloud ----------------------
        # Substituting rCore = f rCl into the mass formula gives
        #   M = 4πρc rCl³ [ f³/3 + (1 - f^(3+α)) / ((3+α) f^α) ]
        # so  rCl = [ M / (4πρc g) ]^(1/3)
        f = rCore_fraction
        g = f**3 / 3.0 + (1.0 - f**(3.0 + alpha)) / ((3.0 + alpha) * f**alpha)

        if g <= 0:
            raise ValueError(
                f"Unphysical parameters: geometric factor g={g:.6e} <= 0 "
                f"for alpha={alpha}, rCore_fraction={f}."
            )

        rCloud = (M_cloud / (4.0 * np.pi * rhoCore * g)) ** (1.0 / 3.0)
        rCore_out = rCloud * f

        # Forward mass check
        M_check = mass_at_radius(rCloud, rCore_out)
        rel_err = abs(M_check - M_cloud) / M_cloud
        if rel_err > 1e-6:
            raise RuntimeError(
                f"Analytical rCloud failed consistency check: "
                f"M(rCloud)={M_check:.6e} vs M_cloud={M_cloud:.6e}, "
                f"rel_err={rel_err:.2e}"
            )
        logger.debug(
            f"rCloud_powerlaw (fractional rCore): alpha={alpha}, "
            f"f={f}, rCloud={rCloud:.4f}, rCore={rCore_out:.4f}, "
            f"rel_err={rel_err:.2e}"
        )
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
        Core number density [1/pc³] (code units)
    alpha : float
        Power-law exponent
    rCore_fraction : float
        Ratio rCore/rCloud (default 0.1)
    mu : float
        Mean molecular weight [Msun] (code units)
    nISM : float
        ISM number density [1/pc³] (code units)

    Returns
    -------
    dict with keys:
        'rCloud': float [pc]
        'rCore': float [pc]
        'nEdge': float [1/pc³] (code units)
        'M_cloud': float [Msun]
        'nCore': float [1/pc³] (code units)
        'alpha': float
        'mu': float

    Examples
    --------
    >>> params = compute_consistent_params(M_cloud=1e5, nCore=nCore, alpha=-2)
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

