#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated mass profile module for TRINITY.

Calculates mass profiles M(r) and mass accretion rates dM/dt.
Supports both scalar and array inputs with consistent output types.

Physics:
    M(r) = integral[0 to r] 4*pi*r'^2 * rho(r') dr'
    dM/dt = dM/dr * dr/dt = 4*pi*r^2 * rho(r) * v(r)

Key features:
- Scalar input -> Scalar output
- Array input -> Array output
- Correct formula: dM/dt = 4*pi*r^2 * rho(r) * v(r) for ALL profiles
- No solver coupling (no dependency on array_t_now, etc.)
- Clean separation: density calculation -> mass integration -> rate

@author: Jia Wei
"""

import numpy as np
import scipy.integrate
import logging
from typing import Union, Tuple

# Import density profile from integrated module
from src.cloud_properties.density_profile_integrated import get_density_profile

# Import unit conversions and physical constants from central module
from src._functions.unit_conversions import CGS, INV_CONV

# Import utility for computing rCloud from physical parameters
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw
)

logger = logging.getLogger(__name__)


# =============================================================================
# Physical constants for unit conversions (from central module)
# =============================================================================

PC_TO_CM = INV_CONV.pc2cm        # [cm/pc]
MSUN_TO_G = INV_CONV.Msun2g      # [g/Msun]
M_H_CGS = CGS.m_H                # [g] hydrogen mass

# Conversion factor: n [cm^-3] * mu -> rho [Msun/pc^3]
# rho [g/cm^3] = n [cm^-3] * mu * m_H [g]
# rho [Msun/pc^3] = rho [g/cm^3] * (pc_to_cm)^3 / Msun_to_g
#              = n * mu * m_H * (pc_to_cm)^3 / Msun_to_g
DENSITY_CONVERSION = M_H_CGS * PC_TO_CM**3 / MSUN_TO_G  # approx 2.47e-2


# Type aliases for clarity
Scalar = float
Array = np.ndarray
ScalarOrArray = Union[float, int, np.ndarray, list]


# =============================================================================
# Helper functions for scalar/array consistency
# =============================================================================

def _is_scalar(x) -> bool:
    """Check if input is scalar (not array-like)."""
    return np.ndim(x) == 0


def _to_array(x) -> np.ndarray:
    """Convert input to numpy array, preserving dtype."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _to_output(result: np.ndarray, was_scalar: bool):
    """Convert result back to scalar if input was scalar."""
    if was_scalar:
        return float(result[0])
    return result


# =============================================================================
# Main functions
# =============================================================================

def get_mass_density(
    r: ScalarOrArray,
    params
) -> ScalarOrArray:
    """
    Get mass density rho(r) from number density n(r).

    This function wraps get_density_profile() from density_profile_integrated.py
    and converts number density [cm^-3] to mass density [Msun/pc^3].

    Parameters
    ----------
    r : float or array-like
        Radius/radii [pc]
    params : dict
        Parameter dictionary

    Returns
    -------
    rho : float or array
        Mass density at radius r [Msun/pc^3]

    Notes
    -----
    The conversion from n [cm^-3] to rho [Msun/pc^3]:
        rho = n * mu * m_H * (pc/cm)^3 / Msun_to_g
            = n * mu * DENSITY_CONVERSION
    """
    # Get number density from density_profile module
    n = get_density_profile(r, params)

    # Use mu_convert for mass density calculation
    # mu_convert = 1.4 is independent of ionization state
    # (ionization changes particle counts but NOT total mass)
    was_scalar = _is_scalar(r)
    r_arr = _to_array(r)
    n_arr = _to_array(n)

    mu_convert = params['mu_convert'].value

    # Mass density = number density * mean molecular weight * unit conversion
    rho_arr = n_arr * mu_convert * DENSITY_CONVERSION

    return _to_output(rho_arr, was_scalar)


def get_mass_profile(
    r: ScalarOrArray,
    params,
    return_mdot: bool = False,
    rdot: ScalarOrArray = None
) -> Union[ScalarOrArray, Tuple[ScalarOrArray, ScalarOrArray]]:
    """
    Calculate mass profile M(r) and optionally dM/dt.

    This function handles both scalar and array inputs consistently:
    - Scalar input -> Scalar output
    - Array input -> Array output

    Parameters
    ----------
    r : float or array-like
        Radius/radii at which to evaluate mass [pc]
    params : dict
        Parameter dictionary with density profile info
        Required keys:
        - 'dens_profile': Profile type ('densPL' or 'densBE')
        - 'nCore', 'nISM': Number densities
        - 'mu_convert': Mean molecular weight for mass conversion (=1.4)
        - 'mCloud', 'rCloud', 'rCore': Cloud parameters
        - Profile-specific parameters (see get_density_profile)
    return_mdot : bool, optional
        Whether to compute dM/dt (default False)
    rdot : float or array-like, optional
        dr/dt (shell velocities) - required if return_mdot=True
        Must be same shape as r

    Returns
    -------
    M : float or array
        Mass enclosed within radius/radii [Msun]
        Returns same type as input r
    dMdt : float or array (if return_mdot=True)
        Mass accretion rate dM/dt at radius/radii
        Returns same type as input r

    Examples
    --------
    >>> # Scalar input -> scalar output
    >>> M = get_mass_profile(5.0, params)
    >>> print(type(M))  # <class 'float'>
    >>>
    >>> # Array input -> array output
    >>> r_arr = np.array([1.0, 2.0, 5.0, 10.0])
    >>> M_arr = get_mass_profile(r_arr, params)
    >>> print(type(M_arr))  # <class 'numpy.ndarray'>
    >>>
    >>> # With mass accretion rate
    >>> M, dMdt = get_mass_profile(5.0, params, return_mdot=True, rdot=10.0)
    """
    # Track if input was scalar for output conversion
    r_was_scalar = _is_scalar(r)
    rdot_was_scalar = _is_scalar(rdot) if rdot is not None else None

    # Convert to array for internal computation
    r_arr = _to_array(r)

    # Validate inputs
    if return_mdot:
        if rdot is None:
            raise ValueError("rdot required when return_mdot=True")
        rdot_arr = _to_array(rdot)
        if len(rdot_arr) != len(r_arr):
            raise ValueError(
                f"rdot length ({len(rdot_arr)}) must match r ({len(r_arr)})"
            )
    else:
        rdot_arr = None

    logger.debug(f"Computing mass profile for {len(r_arr)} radii (scalar={r_was_scalar})")

    # =========================================================================
    # Step 1: Get mass density rho(r) [Msun/pc^3]
    # =========================================================================
    rho_arr = _to_array(get_mass_density(r_arr, params))

    # =========================================================================
    # Step 2: Compute enclosed mass M(r) [Msun]
    # =========================================================================
    M_arr = compute_enclosed_mass(r_arr, rho_arr, params)

    # =========================================================================
    # Step 3: Convert output and return
    # =========================================================================
    if not return_mdot:
        return _to_output(M_arr, r_was_scalar)

    # Compute dM/dt using correct formula
    # dM/dt = dM/dr * dr/dt = 4*pi*r^2 * rho(r) * v(r)
    dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr

    logger.debug(f"dM/dt range: [{dMdt_arr.min():.3e}, {dMdt_arr.max():.3e}]")

    return _to_output(M_arr, r_was_scalar), _to_output(dMdt_arr, r_was_scalar)


def compute_enclosed_mass(
    r_arr: np.ndarray,
    rho_arr: np.ndarray,
    params
) -> np.ndarray:
    """
    Compute enclosed mass M(r) = integral[0 to r] 4*pi*r'^2 * rho(r') dr'.

    Uses appropriate method based on profile type:
    - Power-law: Analytical formula
    - Bonnor-Ebert: Analytical Lane-Emden or numerical integration

    Parameters
    ----------
    r_arr : array
        Radii [pc]
    rho_arr : array
        Mass density at each radius [Msun/pc^3] (from get_mass_density)
    params : dict
        Parameter dictionary

    Returns
    -------
    M_arr : array
        Mass enclosed within each radius [Msun]
    """
    profile_type = params['dens_profile'].value

    if profile_type == 'densPL':
        return compute_enclosed_mass_powerlaw(r_arr, params)
    elif profile_type == 'densBE':
        return compute_enclosed_mass_bonnor_ebert(r_arr, rho_arr, params)
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


def compute_enclosed_mass_powerlaw(
    r_arr: np.ndarray,
    params
) -> np.ndarray:
    """
    Analytical enclosed mass for power-law profile.

    For alpha=0 (homogeneous):
        M(r) = (4/3)*pi*r^3*rho_core        for r <= r_cloud
        M(r) = M_cloud + (4/3)*pi*(r^3-r_cloud^3)*rho_ISM   for r > r_cloud

    For alpha!=0 (power-law):
        M(r) = (4/3)*pi*r^3*rho_core        for r <= r_core
        M(r) = 4*pi*rho_core [r_core^3/3 + (r^(3+alpha) - r_core^(3+alpha))/((3+alpha)*r_core^alpha)]
               for r_core < r <= r_cloud
        M(r) = M_cloud + (4/3)*pi*(r^3-r_cloud^3)*rho_ISM   for r > r_cloud

    Parameters
    ----------
    r_arr : array
        Radii [pc]
    params : dict
        Parameter dictionary

    Returns
    -------
    M_arr : array
        Enclosed mass at each radius [Msun]
    """
    # Extract parameters
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    mu_convert = params['mu_convert'].value
    rCore = params['rCore'].value
    rCloud = params['rCloud'].value
    mCloud = params['mCloud'].value
    alpha = params['densPL_alpha'].value

    # Physical density units: [Msun/pc^3]
    # mu_convert = 1.4 is independent of ionization state
    rhoCore = nCore * mu_convert * DENSITY_CONVERSION
    rhoISM = nISM * mu_convert * DENSITY_CONVERSION

    M_arr = np.zeros_like(r_arr, dtype=float)

    if alpha == 0:
        # Special case: Homogeneous cloud (alpha=0)
        # Simple sphere formula everywhere inside cloud
        inside_cloud = r_arr <= rCloud
        M_arr[inside_cloud] = (4.0/3.0) * np.pi * r_arr[inside_cloud]**3 * rhoCore

        # ISM region - mCloud should be in Msun
        outside_cloud = r_arr > rCloud
        M_arr[outside_cloud] = mCloud + (4.0/3.0) * np.pi * rhoISM * (
            r_arr[outside_cloud]**3 - rCloud**3
        )
    else:
        # General case: Power-law profile
        # Region 1: r <= r_core (uniform density)
        region1 = r_arr <= rCore
        M_arr[region1] = (4.0/3.0) * np.pi * r_arr[region1]**3 * rhoCore

        # Region 2: r_core < r <= r_cloud (power-law)
        region2 = (r_arr > rCore) & (r_arr <= rCloud)
        # Analytical result (Rahner+ 2018, Eq 25):
        M_arr[region2] = 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0 +
            (r_arr[region2]**(3.0 + alpha) - rCore**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore**alpha)
        )

        # Region 3: r > r_cloud (ISM) - mCloud should be in Msun
        region3 = r_arr > rCloud
        M_arr[region3] = mCloud + (4.0/3.0) * np.pi * rhoISM * (
            r_arr[region3]**3 - rCloud**3
        )

    return M_arr


def compute_enclosed_mass_bonnor_ebert(
    r_arr: np.ndarray,
    rho_arr: np.ndarray,
    params
) -> np.ndarray:
    """
    Enclosed mass for Bonnor-Ebert sphere using analytical Lane-Emden formula.

    Uses M(r)/M_cloud = m(xi)/m(xi_out) where m(xi) = xi^2 du/dxi from Lane-Emden.
    This gives EXACT results: M(rCloud) = mCloud guaranteed.

    Falls back to numerical integration if Lane-Emden mass function not available.

    Parameters
    ----------
    r_arr : array
        Radii (must be sorted!)
    rho_arr : array
        Mass density at each radius (used for fallback numerical integration)
    params : dict
        Parameter dictionary. For analytical method, needs 'densBE_f_m' and 'densBE_xi_out'.

    Returns
    -------
    M_arr : array
        Enclosed mass at each radius [Msun]
    """
    rCloud = params['rCloud'].value
    mCloud = params['mCloud'].value
    nISM = params['nISM'].value
    mu_convert = params['mu_convert'].value
    rhoISM = nISM * mu_convert * DENSITY_CONVERSION  # Physical units [Msun/pc^3]

    M_arr = np.zeros_like(r_arr, dtype=float)

    # Check if we have Lane-Emden mass function for analytical calculation
    has_analytical = 'densBE_f_m' in params and 'densBE_xi_out' in params

    inside_cloud = r_arr <= rCloud

    if np.any(inside_cloud):
        r_inside = r_arr[inside_cloud]

        if has_analytical:
            # === ANALYTICAL METHOD (exact) ===
            # Use Lane-Emden mass function: M(r)/M_cloud = m(xi)/m(xi_out)
            f_m = params['densBE_f_m'].value          # m(xi) interpolator
            xi_out = params['densBE_xi_out'].value    # xi at cloud edge

            # Get m(xi_out) for normalization
            m_dim_out = float(f_m(xi_out))

            # Convert r -> xi (linear scaling: xi/xi_out = r/rCloud)
            xi_inside = xi_out * (r_inside / rCloud)

            # Get m(xi) from Lane-Emden solution
            m_inside = f_m(xi_inside)

            # Scale to physical mass: M(r) = mCloud * m(xi)/m(xi_out)
            # This guarantees M(rCloud) = mCloud exactly
            M_arr[inside_cloud] = mCloud * (m_inside / m_dim_out)

        else:
            # === NUMERICAL FALLBACK ===
            # Use trapezoidal integration (less accurate, ~0.5% error)
            rho_inside = rho_arr[inside_cloud]

            for i, (r, rho) in enumerate(zip(r_inside, rho_inside)):
                if i == 0:
                    M_arr[i] = 0.0
                else:
                    M_arr[i] = scipy.integrate.trapezoid(
                        4.0 * np.pi * r_inside[:i+1]**2 * rho_inside[:i+1],
                        r_inside[:i+1]
                    )

    # ISM region (r > r_cloud): add ISM contribution
    outside_cloud = r_arr > rCloud
    M_arr[outside_cloud] = mCloud + (4.0/3.0) * np.pi * rhoISM * (
        r_arr[outside_cloud]**3 - rCloud**3
    )

    return M_arr


# =============================================================================
# Mass Accretion Rate (dM/dt)
# =============================================================================

def compute_mass_accretion_rate(
    r_arr: np.ndarray,
    rdot_arr: np.ndarray,
    params
) -> np.ndarray:
    """
    Compute mass accretion rate dM/dt = 4*pi*r^2*rho(r)*v(r).

    This is the rate at which mass flows through a spherical shell moving
    at velocity v(r) = dr/dt. It follows directly from the chain rule:

        dM/dt = dM/dr * dr/dt = 4*pi*r^2*rho(r) * v(r)

    This formula is EXACT for any smooth density profile, including:
    - Power-law profiles (analytical)
    - Bonnor-Ebert spheres (using Lane-Emden interpolation)

    NO SOLVER HISTORY NEEDED - just instantaneous rho(r) and v(r).

    Parameters
    ----------
    r_arr : array
        Radii [pc]
    rdot_arr : array
        Shell velocities dr/dt [pc/Myr] at each radius
    params : dict
        Parameter dictionary with density profile info

    Returns
    -------
    dMdt_arr : array
        Mass accretion rate at each radius [Msun/Myr]

    See Also
    --------
    get_mass_density : Computes rho(r) for any profile type
    """
    # Get density at each radius [Msun/pc^3]
    rho_arr = _to_array(get_mass_density(r_arr, params))

    # The universal formula: dM/dt = 4*pi*r^2*rho(r)*v(r)
    # This works for ALL density profiles!
    dMdt_arr = 4.0 * np.pi * r_arr**2 * rho_arr * rdot_arr

    return dMdt_arr


# =============================================================================
# Mass Validation
# =============================================================================

def validate_mass_at_rCloud(params, tolerance=0.001):
    """
    Validate that computed M(rCloud) matches expected mCloud.

    This function provides an independent check that the cloud parameters
    (mCloud, nCore, rCore, alpha) are self-consistent. If the computed
    mass at rCloud differs from mCloud by more than the tolerance, it
    indicates inconsistent parameters.

    Parameters
    ----------
    params : dict
        Parameter dictionary with cloud properties.
        Required keys: 'rCloud', 'mCloud', 'dens_profile', etc.
    tolerance : float
        Maximum allowed relative error (default 0.1% = 0.001)

    Returns
    -------
    dict with keys:
        'valid': bool - True if error within tolerance
        'M_computed': float - Computed mass at rCloud [Msun]
        'M_expected': float - Expected mCloud [Msun]
        'relative_error': float - |M_computed - M_expected| / M_expected
        'message': str - Human-readable summary

    Examples
    --------
    >>> result = validate_mass_at_rCloud(params)
    >>> if not result['valid']:
    ...     print(f"Mass error: {result['relative_error']*100:.4f}%")
    ...     print(result['message'])
    """
    rCloud = params['rCloud'].value
    mCloud = params['mCloud'].value

    # Compute mass at rCloud using our analytical formula
    M_computed = get_mass_profile(rCloud, params)

    # Handle edge case of zero expected mass
    if mCloud <= 0:
        return {
            'valid': False,
            'M_computed': M_computed,
            'M_expected': mCloud,
            'relative_error': float('inf'),
            'message': f"ERROR: mCloud = {mCloud} is not positive"
        }

    relative_error = abs(M_computed - mCloud) / mCloud

    is_valid = relative_error <= tolerance

    if is_valid:
        status = "PASS"
    else:
        status = "FAIL"

    message = (
        f"Mass validation {status}: "
        f"M(rCloud) = {M_computed:.4e} Msun vs "
        f"mCloud = {mCloud:.4e} Msun "
        f"(error: {relative_error*100:.4f}%, tolerance: {tolerance*100:.3f}%)"
    )

    return {
        'valid': is_valid,
        'M_computed': M_computed,
        'M_expected': mCloud,
        'relative_error': relative_error,
        'message': message
    }


# =============================================================================
# Utility functions
# =============================================================================

def compute_minimum_rCore(nCore, nISM, rCloud, alpha, margin=1.1):
    """
    Compute minimum rCore such that edge density nEdge >= nISM.

    For power-law profile: n(r) = nCore * (r/rCore)^alpha
    At cloud edge: nEdge = nCore * (rCloud/rCore)^alpha

    For alpha < 0 (density decreasing outward), require nEdge >= nISM:
        nCore * (rCloud/rCore)^alpha >= nISM
        (rCloud/rCore)^alpha >= nISM/nCore

    Since alpha < 0, raising to power 1/alpha flips inequality:
        rCloud/rCore <= (nISM/nCore)^(1/alpha)
        rCore >= rCloud * (nCore/nISM)^(1/alpha)

    Therefore: rCore_min = rCloud * (nCore/nISM)^(1/alpha)

    Parameters
    ----------
    nCore : float
        Core number density [cm^-3]
    nISM : float
        ISM number density [cm^-3]
    rCloud : float
        Cloud outer radius [pc]
    alpha : float
        Power-law exponent (typically negative)
    margin : float
        Safety margin factor (rCore = rCore_min * margin), default 1.1

    Returns
    -------
    rCore_suggested : float
        Suggested rCore value [pc]
    nEdge : float
        Edge density at suggested rCore [cm^-3]
    is_valid : bool
        Whether nEdge >= nISM
    rCore_min : float
        Minimum valid rCore (without margin) [pc]
    """
    if alpha == 0:
        # Homogeneous: nEdge = nCore, always valid if nCore > nISM
        rCore_suggested = rCloud * 0.1  # Default: 10% of cloud radius
        return rCore_suggested, nCore, nCore >= nISM, rCore_suggested

    # For alpha < 0: compute minimum rCore
    # rCore_min = rCloud * (nCore/nISM)^(1/alpha)
    ratio = (nCore / nISM) ** (1.0 / alpha)
    rCore_min = rCloud * ratio

    # Apply safety margin
    rCore_suggested = rCore_min * margin

    # Ensure rCore doesn't exceed rCloud (pathological case)
    if rCore_suggested >= rCloud:
        rCore_suggested = rCloud * 0.9

    # Compute resulting edge density
    nEdge = nCore * (rCloud / rCore_suggested) ** alpha

    return rCore_suggested, nEdge, nEdge >= nISM, rCore_min
