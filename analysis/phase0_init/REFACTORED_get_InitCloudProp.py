#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED Cloud Property Initialization
========================================

Initialize cloud properties (radius, density, mass profiles) for TRINITY simulations.

Key improvements over original:
- Analytical Lane-Emden mass for BE spheres (exact M(rCloud)=mCloud)
- Self-consistent rCloud computation from fundamental inputs
- Key radii (rCloud, rCore) included exactly in arrays
- rCore read from params (user-specified, not computed as fraction)
- nEdge validation: warns if nEdge < nISM and calculates minimum rCore
- Proper scalar/array consistency throughout
- Comprehensive logging and error handling

Supported density profiles:
- densPL: Power-law profile n(r) = nCore * (r/rCore)^alpha
- densBE: Bonnor-Ebert sphere from Lane-Emden equation

Author: Claude Code (refactored from original)
Date: 2026-01-14
"""

import numpy as np
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional
import importlib.util

# =============================================================================
# Setup paths and imports
# =============================================================================

_this_dir = os.path.dirname(os.path.abspath(__file__))
_analysis_dir = os.path.dirname(_this_dir)
_project_root = os.path.dirname(_analysis_dir)

for path in [_project_root, _analysis_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import refactored modules using explicit paths
_density_module = _import_from_path(
    "REFACTORED_density_profile",
    os.path.join(_analysis_dir, "density_profile", "REFACTORED_density_profile.py")
)
_mass_module = _import_from_path(
    "REFACTORED_mass_profile",
    os.path.join(_analysis_dir, "mass_profile", "REFACTORED_mass_profile.py")
)

get_density_profile = _density_module.get_density_profile
get_mass_profile = _mass_module.get_mass_profile

# Import utilities from src
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_powerlaw,
    compute_rCloud_homogeneous,
)

# Import BE sphere module
_bonnor_ebert_module = _import_from_path(
    "bonnorEbertSphere_v2",
    os.path.join(_analysis_dir, "bonnorEbert", "bonnorEbertSphere_v2.py")
)
create_BE_sphere = _bonnor_ebert_module.create_BE_sphere
solve_lane_emden = _bonnor_ebert_module.solve_lane_emden

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CloudProperties:
    """Container for computed cloud properties.

    Attributes
    ----------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]
    nEdge : float
        Edge density [cm^-3]
    r_arr : np.ndarray
        Radius array [pc], includes rCore and rCloud exactly
    n_arr : np.ndarray
        Density profile [cm^-3]
    M_arr : np.ndarray
        Mass profile [Msun]
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

def get_InitCloudProp(params: Dict[str, Any]) -> CloudProperties:
    """
    Initialize cloud properties based on density profile type.

    Parameters
    ----------
    params : dict
        Parameter dictionary with keys:
        - dens_profile: 'densPL' or 'densBE'
        - mCloud: cloud mass [Msun]
        - nCore: core density [cm^-3]
        - nISM: ISM density [cm^-3]
        - mu_ion: mean molecular weight
        - rCore: core radius [pc] (required, user-specified)
        For densPL:
        - densPL_alpha: power-law exponent
        For densBE:
        - densBE_Omega: density contrast (rho_core/rho_edge)

    Returns
    -------
    CloudProperties
        Dataclass containing rCloud, rCore, nEdge, r_arr, n_arr, M_arr
    """
    # Validate inputs
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

# def _init_powerlaw_cloud(params: Dict[str, Any]) -> CloudProperties:
#     """Initialize power-law density profile cloud.

#     For power-law: n(r) = nCore * (r/rCore)^alpha
#     - alpha = 0: homogeneous
#     - alpha = -1: intermediate
#     - alpha = -2: isothermal
#     """
#     # Extract parameters
#     mCloud = params['mCloud'].value
#     nCore = params['nCore'].value
#     nISM = params['nISM'].value
#     alpha = params['densPL_alpha'].value
#     mu = params['mu_ion'].value
#     rCore = params['rCore'].value  # Read rCore directly from params

#     # Compute rCloud from physics (not hardcoded!)
#     if alpha == 0:
#         rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu)
#     else:
#         rCloud, _ = compute_rCloud_powerlaw(
#             mCloud, nCore, alpha, rCore=rCore, mu=mu
#         )

#     # Compute edge density
#     nEdge = nCore * (rCloud / rCore) ** alpha if alpha != 0 else nCore

#     # Validate edge density - warn user if rCore is too small
#     if nEdge < nISM and alpha != 0:
#         # Calculate minimum rCore such that nEdge = nISM
#         # nISM = nCore * (rCloud/rCore_min)^alpha
#         # rCore_min = rCloud * (nCore/nISM)^(1/alpha)
#         rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)
#         # Calculate alternative: minimum nCore to keep rCore fixed
#         # nCore_min = nISM * (rCloud/rCore)^(-alpha)
#         nCore_min = nISM * (rCloud / rCore) ** (-alpha)
#         logger.warning(
#             f"nEdge ({nEdge:.2e} cm^-3) < nISM ({nISM:.2e} cm^-3)!\n"
#             f"  Current rCore = {rCore:.3f} pc is too small.\n"
#             f"  Option 1: Increase rCore to at least {rCore_min:.3f} pc\n"
#             f"  Option 2: Increase nCore to at least {nCore_min:.2e} cm^-3 (to keep rCore={rCore:.3f} pc)"
#         )
#         # Don't silently adjust - raise error so user fixes input
#         raise ValueError(
#             f"rCore={rCore:.3f} pc too small: nEdge={nEdge:.2e} < nISM={nISM:.2e}. "
#             f"Min rCore={rCore_min:.3f} pc OR min nCore={nCore_min:.2e} cm^-3"
#         )

#     # Store computed values back to params
#     params['rCloud'].value = rCloud
#     params['rCore'].value = rCore
#     params['nEdge'].value = nEdge

#     # Create radius array with key radii included exactly
#     r_arr = _create_radius_array(rCloud, rCore)

#     # Compute profiles
#     n_arr = np.array([get_density_profile(r, params) for r in r_arr])
#     M_arr = np.array([get_mass_profile(r, params, return_mdot=False) for r in r_arr])

#     logger.info(f"Power-law cloud (alpha={alpha}): rCloud={rCloud:.3f} pc, nEdge={nEdge:.2e} cm^-3")

#     # Store arrays in params for compatibility
#     params['initial_cloud_r_arr'].value = r_arr
#     params['initial_cloud_n_arr'].value = n_arr
#     params['initial_cloud_m_arr'].value = M_arr

#     return CloudProperties(
#         rCloud=rCloud, rCore=rCore, nEdge=nEdge,
#         r_arr=r_arr, n_arr=n_arr, M_arr=M_arr
#     )


# =============================================================================
# Bonnor-Ebert sphere initialization
# =============================================================================

def _init_bonnor_ebert_cloud(params: Dict[str, Any]) -> CloudProperties:
    """Initialize Bonnor-Ebert sphere cloud with analytical mass.

    Uses analytical Lane-Emden mass formula: M(r)/M_cloud = m(xi)/m(xi_out)
    This gives EXACT results: M(rCloud) = mCloud guaranteed.
    """
    # Extract parameters
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    Omega = params['densBE_Omega'].value
    mu = params['mu_ion'].value

    # Solve Lane-Emden equation
    le_solution = solve_lane_emden()

    # Create BE sphere (computes rCloud, T_eff analytically)
    be_result = create_BE_sphere(
        M_cloud=mCloud,
        n_core=nCore,
        Omega=Omega,
        mu=mu,
        lane_emden_solution=le_solution
    )

    rCloud = be_result.r_out
    nEdge = be_result.n_out
    T_eff = be_result.T_eff
    xi_out = be_result.xi_out

    # rCore read from params (user-specified)
    rCore = params['rCore'].value

    # Store computed values in params
    params['rCloud'].value = rCloud
    params['rCore'].value = rCore
    params['nEdge'].value = nEdge
    params['densBE_Teff'].value = T_eff
    params['densBE_xi_out'].value = xi_out
    params['densBE_f_rho_rhoc'].value = le_solution.f_rho_rhoc
    params['densBE_f_m'].value = le_solution.f_m  # For analytical mass!

    # Create radius array
    r_arr = _create_radius_array(rCloud, rCore)

    # Compute density profile
    n_arr = np.array([get_density_profile(r, params) for r in r_arr])

    # Compute mass profile using analytical Lane-Emden formula
    # Pass full array for BE sphere (analytical formula handles this)
    M_arr = get_mass_profile(r_arr, params, return_mdot=False)

    stability = "STABLE" if be_result.is_stable else "UNSTABLE"
    logger.info(f"BE sphere (Omega={Omega:.2f}): rCloud={rCloud:.3f} pc, T={T_eff:.1f} K [{stability}]")

    # Store arrays in params
    params['initial_cloud_r_arr'].value = r_arr
    params['initial_cloud_n_arr'].value = n_arr
    params['initial_cloud_m_arr'].value = M_arr

    return CloudProperties(
        rCloud=rCloud, rCore=rCore, nEdge=nEdge,
        r_arr=r_arr, n_arr=n_arr, M_arr=M_arr,
        T_eff=T_eff, xi_out=xi_out
    )


# # =============================================================================
# # Helper functions
# # =============================================================================

# def _create_radius_array(rCloud: float, rCore: float,
#                          n_inside: int = 1000,
#                          n_outside: int = 100) -> np.ndarray:
#     """
#     Create radius array with key radii included exactly.

#     Parameters
#     ----------
#     rCloud : float
#         Cloud radius [pc]
#     rCore : float
#         Core radius [pc]
#     n_inside : int
#         Number of points inside cloud
#     n_outside : int
#         Number of points beyond cloud

#     Returns
#     -------
#     r_arr : array
#         Sorted unique radius array including rCore and rCloud exactly
#     """
#     # Inside cloud: logspace from small radius to rCloud
#     r_min = 1e-3  # pc
#     r_inside = np.logspace(np.log10(r_min), np.log10(rCloud), n_inside)

#     # Beyond cloud: up to 1.5 * rCloud
#     r_outside = np.logspace(np.log10(rCloud), np.log10(1.5 * rCloud), n_outside)

#     # Combine and add key radii explicitly
#     r_arr = np.concatenate([
#         [1e-10],     # Near-origin point
#         r_inside,
#         r_outside
#     ])

#     # Add key radii exactly and sort
#     r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud])))

#     return r_arr


# def _validate_params(params: Dict[str, Any]) -> None:
#     """Validate input parameters."""
#     required = ['dens_profile', 'mCloud', 'nCore', 'nISM', 'mu_ion', 'rCore']

#     for key in required:
#         if key not in params:
#             raise ValueError(f"Missing required parameter: {key}")
#         if params[key].value is None:
#             raise ValueError(f"Parameter {key} is None")

#     if params['mCloud'].value <= 0:
#         raise ValueError(f"mCloud must be positive, got {params['mCloud'].value}")
#     if params['nCore'].value <= 0:
#         raise ValueError(f"nCore must be positive, got {params['nCore'].value}")
#     if params['rCore'].value <= 0:
#         raise ValueError(f"rCore must be positive, got {params['rCore'].value}")

#     profile = params['dens_profile'].value
#     if profile == 'densPL':
#         if 'densPL_alpha' not in params:
#             raise ValueError("densPL profile requires densPL_alpha")
#     elif profile == 'densBE':
#         if 'densBE_Omega' not in params:
#             raise ValueError("densBE profile requires densBE_Omega")


# =============================================================================
# Verification functions
# =============================================================================

def verify_mass_at_rCloud(props: CloudProperties, mCloud: float) -> float:
    """Verify that M(rCloud) = mCloud.

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
    """Verify that rCloud and rCore are exactly in the radius array.

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
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("=" * 70)
    print("TESTING REFACTORED get_InitCloudProp")
    print("=" * 70)

    # Mock value class for testing
    class MockValue:
        def __init__(self, v):
            self.value = v

    # -------------------------------------------------------------------------
    # Test 1: Bonnor-Ebert sphere at critical omega
    # -------------------------------------------------------------------------
    print("\n[Test 1] BE sphere at critical Omega=14.04")

    params_BE = {
        'dens_profile': MockValue('densBE'),
        'mCloud': MockValue(1e7),
        'nCore': MockValue(1e3),
        'nISM': MockValue(1.0),
        'mu_ion': MockValue(1.4),
        'mu_neu': MockValue(1.4),
        'gamma_adia': MockValue(5.0/3.0),
        'rCore': MockValue(10.0),  # User-specified core radius [pc]
        'densBE_Omega': MockValue(14.04),
        # Placeholders for computed values
        'rCloud': MockValue(None),
        'nEdge': MockValue(None),
        'densBE_Teff': MockValue(None),
        'densBE_xi_out': MockValue(None),
        'densBE_f_rho_rhoc': MockValue(None),
        'densBE_f_m': MockValue(None),
        'initial_cloud_r_arr': MockValue(None),
        'initial_cloud_n_arr': MockValue(None),
        'initial_cloud_m_arr': MockValue(None),
    }

    props_BE = get_InitCloudProp(params_BE)

    print(f"  rCloud = {props_BE.rCloud:.3f} pc")
    print(f"  rCore = {props_BE.rCore:.3f} pc")
    print(f"  nEdge = {props_BE.nEdge:.2e} cm^-3")
    print(f"  T_eff = {props_BE.T_eff:.1f} K")
    print(f"  xi_out = {props_BE.xi_out:.3f}")

    error_BE = verify_mass_at_rCloud(props_BE, 1e7)
    print(f"  M(rCloud)/mCloud error: {error_BE*100:.6f}%")

    verify_key_radii_in_array(props_BE)

    # -------------------------------------------------------------------------
    # Test 2: Power-law alpha = -2
    # -------------------------------------------------------------------------
    print("\n[Test 2] Power-law alpha = -2")

    params_PL = {
        'dens_profile': MockValue('densPL'),
        'mCloud': MockValue(1e7),
        'nCore': MockValue(1e3),
        'nISM': MockValue(1.0),
        'mu_ion': MockValue(1.4),
        'mu_neu': MockValue(1.4),
        'rCore': MockValue(10.0),  # User-specified
        'densPL_alpha': MockValue(-2),
        # Placeholders
        'rCloud': MockValue(None),
        'nEdge': MockValue(None),
        'initial_cloud_r_arr': MockValue(None),
        'initial_cloud_n_arr': MockValue(None),
        'initial_cloud_m_arr': MockValue(None),
    }

    props_PL = get_InitCloudProp(params_PL)

    print(f"  rCloud = {props_PL.rCloud:.3f} pc")
    print(f"  rCore = {props_PL.rCore:.3f} pc")
    print(f"  nEdge = {props_PL.nEdge:.2e} cm^-3")

    error_PL = verify_mass_at_rCloud(props_PL, 1e7)
    print(f"  M(rCloud)/mCloud error: {error_PL*100:.6f}%")

    verify_key_radii_in_array(props_PL)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Profile':<20} {'rCloud [pc]':>12} {'M(rCloud) error':>18}")
    print("-" * 70)
    print(f"{'BE (critical)':<20} {props_BE.rCloud:>12.3f} {error_BE*100:>17.6f}%")
    print(f"{'Power-law (a=-2)':<20} {props_PL.rCloud:>12.3f} {error_PL*100:>17.6f}%")
    print("=" * 70)
