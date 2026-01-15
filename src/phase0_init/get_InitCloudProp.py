# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sun Jul 24 23:42:14 2022

@author: Jia Wei Teh

Initialize cloud properties (radius, density, mass profiles) for TRINITY simulations.

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


"""
import numpy as np
import sys
import src.cloud_properties.bonnorEbertSphere as bonnorEbertSphere
import src.cloud_properties.powerLawSphere as powerLawSphere
import src.cloud_properties.density_profile as density_profile
import src.cloud_properties.mass_profile as mass_profile
import src._functions.unit_conversions as cvt


import numpy as np
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional
import importlib.util

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# Main entry point
# =============================================================================

def get_InitCloudProp(params):
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
    # get profile
    profile_type = params['dens_profile'].value

    if profile_type == 'densPL':
        return _init_powerlaw_cloud(params)
    elif profile_type == 'densBE':
        return _init_bonnor_ebert_cloud(params)
    else:
        raise ValueError(f"Unknown density profile: {profile_type}")
        
    return
        

def _validate_params(params):
    """Validate input parameters."""
    required = ['dens_profile', 'mCloud', 'nCore', 'nISM', 'mu_ion', 'rCore']

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
            
    return





# =============================================================================
# Power-law cloud initialization
# =============================================================================


def _init_powerlaw_cloud(params):
    """Initialize power-law density profile cloud.

    For power-law: n(r) = nCore * (r/rCore)^alpha
    - alpha = 0: homogeneous
    - alpha = -1: intermediate
    - alpha = -2: isothermal
    """
    # Extract parameters
    mCloud = params['mCloud'].value
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    alpha = params['densPL_alpha'].value
    mu = params['mu_neu'].value
    rCore = params['rCore'].value  

    # Compute rCloud from physics
    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu)
    else:
        rCloud, _ = compute_rCloud_powerlaw(
            mCloud, nCore, alpha, rCore, mu
        )

    # Compute edge density
    nEdge = nCore * (rCloud / rCore) ** alpha if alpha != 0 else nCore

    # Validate edge density - warn user if rCore is too small
    if nEdge < nISM and alpha != 0:
        # Calculate minimum rCore such that nEdge = nISM
        # nISM = nCore * (rCloud/rCore_min)^alpha
        # rCore_min = rCloud * (nCore/nISM)^(1/alpha)
        rCore_min = rCloud * (nCore / nISM) ** (1.0 / alpha)
        # Calculate alternative: minimum nCore to keep rCore fixed
        # nCore_min = nISM * (rCloud/rCore)^(-alpha)
        nCore_min = nISM * (rCloud / rCore) ** (-alpha)
        logger.warning(
            f"nEdge ({nEdge:.2e} cm^-3) < nISM ({nISM:.2e} cm^-3)!\n"
            f"  Current rCore = {rCore:.3f} pc is too small.\n"
            f"  Option 1: Increase rCore to at least {rCore_min:.3f} pc\n"
            f"  Option 2: Increase nCore to at least {nCore_min:.2e} cm^-3 (to keep rCore={rCore:.3f} pc)"
        )
        # Don't silently adjust - raise error so user fixes input
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

    # Compute profiles
    n_arr = np.array([get_density_profile(r, params) for r in r_arr])
    M_arr = np.array([mass_profile.get_mass_profile(r, params, return_mdot=False) for r in r_arr])

    logger.info(f"Power-law cloud (alpha={alpha}): rCloud={rCloud:.3f} pc, nEdge={nEdge:.2e} cm^-3")

    # Store arrays in params for compatibility
    params['initial_cloud_r_arr'].value = r_arr
    params['initial_cloud_n_arr'].value = n_arr
    params['initial_cloud_m_arr'].value = M_arr

    return CloudProperties(
        rCloud=rCloud, rCore=rCore, nEdge=nEdge,
        r_arr=r_arr, n_arr=n_arr, M_arr=M_arr
    )










# =============================================================================
# Helper functions
# =============================================================================

def _create_radius_array(rCloud: float, rCore: float,
                         n_inside: int = 1000,
                         n_outside: int = 100) -> np.ndarray:
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
    r_arr : array
        Sorted unique radius array including rCore and rCloud exactly
    """
    # Inside cloud: logspace from small radius to rCloud
    r_min = 1e-3  # pc
    r_inside = np.logspace(np.log10(r_min), np.log10(rCloud), n_inside)

    # Beyond cloud: up to 1.5 * rCloud
    r_outside = np.logspace(np.log10(rCloud), np.log10(1.5 * rCloud), n_outside)

    # Combine and add key radii explicitly
    r_arr = np.concatenate([
        [1e-10],     # Near-origin point
        r_inside,
        r_outside
    ])

    # Add key radii exactly and sort
    r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud])))

    return r_arr












#--

# TODO: add ability to produce cloud mass and density radial profile as output. Use plotting function from /compare_profiles_radial.py

def get_InitCloudProp_OLD(params):
    
    # get cloud radius and number density at cloud radius
    # get initial density profile
    
    
    if params['dens_profile'].value == 'densBE':
        _, rCloud, nEdge, _ = bonnorEbertSphere.create_BESphere(params)
        # _, rCloud, nEdge, _ = bonnorEbertSphere.create_BESphereVersion2(params)
        
    
    elif params['dens_profile'].value == 'densPL':
        rCloud, nEdge = powerLawSphere.create_PLSphere(params)
        
    # logspace radius array
    r_arr_inCloud = np.logspace(-3, np.log10(rCloud), 1000)
    r_arr_beyondCloud = np.logspace(np.log10(rCloud), np.log10(rCloud*1.5), 100)
    # start with zero
    r_arr = np.concatenate(([1e-10], r_arr_inCloud, r_arr_beyondCloud))
    
    r_arr = np.unique(r_arr)
    
    # initial cloud values
    params['rCloud'].value = rCloud
    params['nEdge'].value = nEdge
    print(f"Cloud radius is {np.round(rCloud, 3)}pc.")
    print(f"Cloud edge density is {np.round(nEdge * cvt.ndens_au2cgs, 3)} cm-3.")
    # radius array
    params['initial_cloud_r_arr'].value = r_arr
    # density array
    params['initial_cloud_n_arr'].value = density_profile.get_density_profile(r_arr, params)
    print(params['initial_cloud_n_arr'].value)
    # mass array
    params['initial_cloud_m_arr'].value = mass_profile.get_mass_profile(r_arr, params, return_mdot = False)
    
    return 
    

