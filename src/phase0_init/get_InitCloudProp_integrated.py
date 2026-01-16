#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Cloud Property Initialization for TRINITY.

Initialize cloud properties (radius, density, mass profiles) using
the integrated density and mass profile modules.

Key features:
- Uses mu_convert = 1.4 consistently for mass density calculations
- Analytical Lane-Emden mass for BE spheres (exact M(rCloud) = mCloud)
- Self-consistent rCloud computation from fundamental inputs
- Key radii (rCloud, rCore) included exactly in arrays
- Vectorized profile computation for efficiency

Supported density profiles:
- densPL: Power-law profile n(r) = nCore * (r/rCore)^alpha
- densBE: Bonnor-Ebert sphere from Lane-Emden equation

Author: TRINITY Team
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

# Absolute imports from integrated modules
from src.cloud_properties.density_profile_integrated import get_density_profile
from src.cloud_properties.mass_profile_integrated import get_mass_profile
from src.cloud_properties.bonnorEbertSphere_v2 import (
    create_BE_sphere,
    solve_lane_emden,
)
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CloudProperties:
    """
    Container for computed cloud properties.

    Attributes
    ----------
    rCloud : float
        Cloud outer radius [pc]
    rCore : float
        Core radius [pc]
    nEdge : float
        Edge number density [cm^-3]
    r_arr : np.ndarray
        Radius array [pc], includes rCore and rCloud exactly
    n_arr : np.ndarray
        Number density profile [cm^-3]
    M_arr : np.ndarray
        Enclosed mass profile [Msun]
    T_eff : float, optional
        Effective temperature [K] (BE sphere only)
    xi_out : float, optional
        Dimensionless outer radius (BE sphere only)
    """
    rCloud: float
    rCore: float
    nEdge: float
    r_arr: np.ndarray
    n_arr: np.ndarray
    M_arr: np.ndarray
    # BE sphere specific
    T_eff: Optional[float] = None
    xi_out: Optional[float] = None


# =============================================================================
# Main entry point
# =============================================================================

def get_InitCloudProp(params) -> CloudProperties:
    """
    Initialize cloud properties based on density profile type.

    Parameters
    ----------
    params : dict-like
        Parameter dictionary with .value attribute access.
        Required keys:
        - dens_profile: 'densPL' or 'densBE'
        - mCloud: cloud mass [Msun]
        - nCore: core density [cm^-3]
        - nISM: ISM density [cm^-3]
        - mu_convert: mean molecular weight for mass (=1.4)
        - rCore: core radius [pc] (user-specified)
        For densPL:
        - densPL_alpha: power-law exponent
        For densBE:
        - densBE_Omega: density contrast (rho_core/rho_edge)
        - gamma_adia: adiabatic index

    Returns
    -------
    CloudProperties
        Dataclass containing rCloud, rCore, nEdge, r_arr, n_arr, M_arr

    Notes
    -----
    This function also updates params in-place with computed values:
    - rCloud, rCore, nEdge
    - initial_cloud_r_arr, initial_cloud_n_arr, initial_cloud_m_arr
    - For BE spheres: densBE_Teff, densBE_xi_out, densBE_f_rho_rhoc, densBE_f_m
    """
    _validate_params(params)

    profile_type = params['dens_profile'].value

    if profile_type == 'densPL':
        return _init_powerlaw_cloud(params)
    elif profile_type == 'densBE':
        return _init_bonnor_ebert_cloud(params)
    else:
        raise ValueError(f"Unknown density profile: {profile_type}")


# =============================================================================
# Power-law cloud initialization
# =============================================================================

def _init_powerlaw_cloud(params) -> CloudProperties:
    """
    Initialize power-law density profile cloud.

    For power-law: n(r) = nCore * (r/rCore)^alpha
    - alpha = 0: homogeneous (constant density)
    - alpha = -1: intermediate
    - alpha = -2: isothermal

    Uses mu_convert = 1.4 for mass density calculations.
    """
    # Extract parameters
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    alpha = params['densPL_alpha'].value
    mu = params['mu_convert'].value  # Use mu_convert, NOT mu_neu or mu_ion
    rCore = params['rCore'].value

    # Compute rCloud from physics (not hardcoded!)
    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu)
    else:
        rCloud, _ = compute_rCloud_powerlaw(
            mCloud, nCore, alpha, rCore=rCore, mu=mu
        )

    # Compute edge density
    nEdge = nCore * (rCloud / rCore) ** alpha if alpha != 0 else nCore

    # Validate edge density - warn user if rCore is too small
    if nEdge < nISM and alpha != 0:
        # Calculate minimum rCore such that nEdge = nISM
        rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)
        # Calculate alternative: minimum nCore to keep rCore fixed
        nCore_min = nISM * (rCloud / rCore) ** (-alpha)
        logger.warning(
            f"nEdge ({nEdge:.2e} cm^-3) < nISM ({nISM:.2e} cm^-3)!\n"
            f"  Current rCore = {rCore:.3f} pc is too small.\n"
            f"  Option 1: Increase rCore to at least {rCore_min:.3f} pc\n"
            f"  Option 2: Increase nCore to at least {nCore_min:.2e} cm^-3"
        )
        raise ValueError(
            f"rCore={rCore:.3f} pc too small: nEdge={nEdge:.2e} < nISM={nISM:.2e}. "
            f"Min rCore={rCore_min:.3f} pc OR min nCore={nCore_min:.2e} cm^-3"
        )

    # Store computed values back to params
    params['rCloud'].value = rCloud
    params['rCore'].value = rCore
    params['nEdge'].value = nEdge

    # Create radius array with key radii included exactly
    r_arr = _create_radius_array(rCloud, rCore)

    # Compute profiles using VECTORIZED integrated modules
    n_arr = get_density_profile(r_arr, params)  # Returns array
    M_arr = get_mass_profile(r_arr, params)     # Returns array

    logger.info(
        f"Power-law cloud (alpha={alpha}): "
        f"rCloud={rCloud:.3f} pc, nEdge={nEdge:.2e} cm^-3"
    )

    # Store arrays in params for compatibility with downstream code
    params['initial_cloud_r_arr'].value = r_arr
    params['initial_cloud_n_arr'].value = n_arr
    params['initial_cloud_m_arr'].value = M_arr

    return CloudProperties(
        rCloud=rCloud, rCore=rCore, nEdge=nEdge,
        r_arr=r_arr, n_arr=n_arr, M_arr=M_arr
    )


# =============================================================================
# Bonnor-Ebert sphere initialization
# =============================================================================

def _init_bonnor_ebert_cloud(params) -> CloudProperties:
    """
    Initialize Bonnor-Ebert sphere cloud with analytical mass.

    Uses analytical Lane-Emden mass formula: M(r)/M_cloud = m(xi)/m(xi_out)
    This gives EXACT results: M(rCloud) = mCloud guaranteed.

    Uses mu_convert = 1.4 for mass density calculations.
    """
    # Extract parameters
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    Omega = params['densBE_Omega'].value
    mu = params['mu_convert'].value  # Use mu_convert for mass density
    gamma = params['gamma_adia'].value
    rCore = params['rCore'].value  # User-specified

    # Solve Lane-Emden equation (can be cached for efficiency)
    le_solution = solve_lane_emden()

    # Create BE sphere (computes rCloud, T_eff analytically)
    be_result = create_BE_sphere(
        M_cloud=mCloud,
        n_core=nCore,
        Omega=Omega,
        mu=mu,
        gamma=gamma,
        lane_emden_solution=le_solution
    )

    rCloud = be_result.r_out
    nEdge = be_result.n_out
    T_eff = be_result.T_eff
    xi_out = be_result.xi_out

    # Store computed values in params
    params['rCloud'].value = rCloud
    params['rCore'].value = rCore
    params['nEdge'].value = nEdge
    params['densBE_Teff'].value = T_eff

    # Ensure BE-specific params exist (may not be in read_param.py)
    _ensure_be_params_exist(params)

    # Store Lane-Emden interpolation functions
    params['densBE_xi_out'].value = xi_out
    params['densBE_f_rho_rhoc'].value = le_solution.f_rho_rhoc
    params['densBE_f_m'].value = le_solution.f_m  # Critical for analytical mass!

    # Create radius array
    r_arr = _create_radius_array(rCloud, rCore)

    # Compute profiles using VECTORIZED integrated modules
    n_arr = get_density_profile(r_arr, params)  # Returns array
    M_arr = get_mass_profile(r_arr, params)     # Returns array (analytical for BE)

    stability = "STABLE" if be_result.is_stable else "UNSTABLE"
    logger.info(
        f"BE sphere (Omega={Omega:.2f}): "
        f"rCloud={rCloud:.3f} pc, T={T_eff:.1f} K [{stability}]"
    )

    # Store arrays in params
    params['initial_cloud_r_arr'].value = r_arr
    params['initial_cloud_n_arr'].value = n_arr
    params['initial_cloud_m_arr'].value = M_arr

    return CloudProperties(
        rCloud=rCloud, rCore=rCore, nEdge=nEdge,
        r_arr=r_arr, n_arr=n_arr, M_arr=M_arr,
        T_eff=T_eff, xi_out=xi_out
    )


# =============================================================================
# Helper functions
# =============================================================================

def _validate_params(params) -> None:
    """
    Validate input parameters.

    Raises ValueError if required parameters are missing or invalid.
    """
    required = ['dens_profile', 'mCloud', 'nCore', 'nISM', 'mu_convert', 'rCore']

    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
        if params[key].value is None:
            raise ValueError(f"Parameter {key} is None")

    if params['mCloud'].value <= 0:
        raise ValueError(f"mCloud must be positive, got {params['mCloud'].value}")
    if params['nCore'].value <= 0:
        raise ValueError(f"nCore must be positive, got {params['nCore'].value}")
    if params['rCore'].value <= 0:
        raise ValueError(f"rCore must be positive, got {params['rCore'].value}")

    profile = params['dens_profile'].value
    if profile == 'densPL':
        if 'densPL_alpha' not in params:
            raise ValueError("densPL profile requires densPL_alpha")
    elif profile == 'densBE':
        if 'densBE_Omega' not in params:
            raise ValueError("densBE profile requires densBE_Omega")
        if 'gamma_adia' not in params:
            raise ValueError("densBE profile requires gamma_adia")


def _create_radius_array(
    rCloud: float,
    rCore: float,
    n_inside: int = 1000,
    n_outside: int = 100
) -> np.ndarray:
    """
    Create radius array with key radii included exactly.

    Parameters
    ----------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]
    n_inside : int
        Number of points inside cloud
    n_outside : int
        Number of points beyond cloud

    Returns
    -------
    r_arr : np.ndarray
        Sorted unique radius array including rCore and rCloud exactly
    """
    r_min = 1e-3  # pc

    # Inside cloud: logspace from small radius to rCloud
    r_inside = np.logspace(np.log10(r_min), np.log10(rCloud), n_inside)

    # Beyond cloud: up to 1.5 * rCloud
    r_outside = np.logspace(np.log10(rCloud), np.log10(1.5 * rCloud), n_outside)

    # Combine with near-origin point
    r_arr = np.concatenate([
        [1e-10],     # Near-origin point for mass profile
        r_inside,
        r_outside
    ])

    # Add key radii exactly and ensure unique sorted array
    r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud])))

    return r_arr


def _ensure_be_params_exist(params) -> None:
    """
    Ensure BE-specific parameters exist in params dictionary.

    Some BE params are not defined in read_param.py and must be
    created dynamically when computing BE sphere properties.
    """
    # Simple wrapper class for dynamic params
    class DynamicParam:
        def __init__(self, value=None):
            self.value = value

    be_params_needed = [
        'densBE_f_m',         # Lane-Emden mass interpolation function
        'densBE_xi_out',      # Dimensionless outer radius
        'densBE_f_rho_rhoc',  # Density ratio interpolation function
    ]

    for key in be_params_needed:
        if key not in params:
            params[key] = DynamicParam(None)


# =============================================================================
# Verification functions
# =============================================================================

def verify_mass_at_rCloud(props: CloudProperties, mCloud: float) -> float:
    """
    Verify that M(rCloud) = mCloud.

    Parameters
    ----------
    props : CloudProperties
        Computed cloud properties
    mCloud : float
        Expected cloud mass [Msun]

    Returns
    -------
    rel_error : float
        Relative error |M(rCloud) - mCloud| / mCloud
    """
    # Find index of rCloud in array (should be exact due to _create_radius_array)
    idx = np.searchsorted(props.r_arr, props.rCloud)

    # Check if rCloud is exactly in array
    if idx < len(props.r_arr) and np.isclose(props.r_arr[idx], props.rCloud):
        M_at_rCloud = props.M_arr[idx]
    else:
        # Fallback to interpolation
        M_at_rCloud = np.interp(props.rCloud, props.r_arr, props.M_arr)

    rel_error = abs(M_at_rCloud - mCloud) / mCloud

    if rel_error > 0.01:  # > 1% error
        logger.warning(f"M(rCloud) error: {rel_error*100:.2f}%")
    else:
        logger.info(f"M(rCloud) = {M_at_rCloud:.4e} Msun (error: {rel_error*100:.4f}%)")

    return rel_error


def verify_key_radii_in_array(props: CloudProperties) -> bool:
    """
    Verify that rCloud and rCore are exactly in the radius array.

    Returns
    -------
    success : bool
        True if both radii are in array exactly
    """
    rCloud_in_array = np.any(np.isclose(props.r_arr, props.rCloud))
    rCore_in_array = np.any(np.isclose(props.r_arr, props.rCore))

    if not rCloud_in_array:
        logger.warning(f"rCloud={props.rCloud:.6f} not found exactly in r_arr")
    if not rCore_in_array:
        logger.warning(f"rCore={props.rCore:.6f} not found exactly in r_arr")

    success = rCloud_in_array and rCore_in_array
    if success:
        logger.info("rCloud and rCore are both exactly in r_arr")

    return success


# =============================================================================
# Test / Example usage
# =============================================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 70)
    print("TESTING get_InitCloudProp_integrated")
    print("=" * 70)

    # Mock value class for testing
    class MockParam:
        def __init__(self, v):
            self.value = v

    # -------------------------------------------------------------------------
    # Test 1: Power-law alpha = -2
    # -------------------------------------------------------------------------
    print("\n[Test 1] Power-law alpha = -2")

    params_PL = {
        'dens_profile': MockParam('densPL'),
        'mCloud': MockParam(1e5),
        'nCore': MockParam(1e3),
        'nISM': MockParam(1.0),
        'mu_convert': MockParam(1.4),
        'rCore': MockParam(1.0),
        'densPL_alpha': MockParam(-2),
        'rCloud': MockParam(None),
        'nEdge': MockParam(None),
        'initial_cloud_r_arr': MockParam(None),
        'initial_cloud_n_arr': MockParam(None),
        'initial_cloud_m_arr': MockParam(None),
    }

    props_PL = get_InitCloudProp(params_PL)
    print(f"  rCloud = {props_PL.rCloud:.3f} pc")
    print(f"  rCore = {props_PL.rCore:.3f} pc")
    print(f"  nEdge = {props_PL.nEdge:.2e} cm^-3")

    error_PL = verify_mass_at_rCloud(props_PL, 1e5)
    verify_key_radii_in_array(props_PL)

    # -------------------------------------------------------------------------
    # Test 2: Homogeneous cloud (alpha = 0)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Homogeneous cloud (alpha = 0)")

    params_homo = {
        'dens_profile': MockParam('densPL'),
        'mCloud': MockParam(1e5),
        'nCore': MockParam(1e3),
        'nISM': MockParam(1.0),
        'mu_convert': MockParam(1.4),
        'rCore': MockParam(1.0),
        'densPL_alpha': MockParam(0),
        'rCloud': MockParam(None),
        'nEdge': MockParam(None),
        'initial_cloud_r_arr': MockParam(None),
        'initial_cloud_n_arr': MockParam(None),
        'initial_cloud_m_arr': MockParam(None),
    }

    props_homo = get_InitCloudProp(params_homo)
    print(f"  rCloud = {props_homo.rCloud:.3f} pc")
    print(f"  nEdge = {props_homo.nEdge:.2e} cm^-3 (should equal nCore)")

    error_homo = verify_mass_at_rCloud(props_homo, 1e5)
    verify_key_radii_in_array(props_homo)

    # -------------------------------------------------------------------------
    # Test 3: Bonnor-Ebert sphere
    # -------------------------------------------------------------------------
    print("\n[Test 3] BE sphere at Omega=10")

    params_BE = {
        'dens_profile': MockParam('densBE'),
        'mCloud': MockParam(100.0),
        'nCore': MockParam(1e4),
        'nISM': MockParam(1.0),
        'mu_convert': MockParam(1.4),
        'gamma_adia': MockParam(5.0/3.0),
        'rCore': MockParam(0.1),
        'densBE_Omega': MockParam(10.0),
        'rCloud': MockParam(None),
        'nEdge': MockParam(None),
        'densBE_Teff': MockParam(None),
        'initial_cloud_r_arr': MockParam(None),
        'initial_cloud_n_arr': MockParam(None),
        'initial_cloud_m_arr': MockParam(None),
    }

    props_BE = get_InitCloudProp(params_BE)
    print(f"  rCloud = {props_BE.rCloud:.4f} pc")
    print(f"  nEdge = {props_BE.nEdge:.2e} cm^-3")
    print(f"  T_eff = {props_BE.T_eff:.1f} K")
    print(f"  xi_out = {props_BE.xi_out:.3f}")

    error_BE = verify_mass_at_rCloud(props_BE, 100.0)
    verify_key_radii_in_array(props_BE)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Profile':<25} {'rCloud [pc]':>12} {'M(rCloud) error':>18}")
    print("-" * 70)
    print(f"{'Power-law (a=-2)':<25} {props_PL.rCloud:>12.3f} {error_PL*100:>17.6f}%")
    print(f"{'Homogeneous (a=0)':<25} {props_homo.rCloud:>12.3f} {error_homo*100:>17.6f}%")
    print(f"{'BE (Omega=10)':<25} {props_BE.rCloud:>12.4f} {error_BE*100:>17.6f}%")
    print("=" * 70)

    # Check all tests passed
    all_passed = error_PL < 0.01 and error_homo < 0.01 and error_BE < 0.01
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
