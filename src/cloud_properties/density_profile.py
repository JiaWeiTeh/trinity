#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:37:53 2022

@author: Jia Wei Teh

This script includes function that calculates the density profile, given 

REFACTORED VERSION:
- Scalar/array input-output consistency: scalar input → scalar output, array input → array output
- Cleaner code structure with helper functions
- Comprehensive tests included at bottom
"""

import numpy as np
from src.cloud_properties import bonnorEbertSphere


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
# Main function
# =============================================================================

def get_density_profile(r, params):
    """
    Calculate the number density profile n(r) at given radius/radii.

    This function computes the gas number density at position r, supporting
    both power-law (densPL) and Bonnor-Ebert sphere (densBE) density profiles.

    Parameters
    ----------
    r : float or array-like
        Radius/radii of interest [pc]. Can be scalar or array.
    params : dict
        Dictionary containing cloud parameters:
        - nISM: ISM number density [cm^-3]
        - nCore: Core number density [cm^-3]
        - rCloud: Cloud radius [pc]
        - rCore: Core radius [pc]
        - dens_profile: 'densPL' or 'densBE'
        - densPL_alpha: power-law exponent (for densPL)
        - densBE_f_rho_rhoc: interpolation function (for densBE)

    Returns
    -------
    n : float or np.ndarray
        Number density at radius r [cm^-3].
        Returns scalar if input r is scalar, array if input r is array.

    Notes
    -----
    For power-law profile (densPL):
        - alpha = 0 (homogeneous): n = nCore for r <= rCloud, n = nISM for r > rCloud
        - alpha != 0: n = nCore * (r/rCore)^alpha with boundary conditions

    For Bonnor-Ebert sphere (densBE):
        - Uses tabulated density profile from bonnorEbertSphere module
        - n = nCore * f_rho_rhoc(xi) for r <= rCloud, n = nISM for r > rCloud

    Examples
    --------
    >>> # Scalar input returns scalar output
    >>> n = get_density_profile(0.5, params)
    >>> type(n)
    <class 'float'>

    >>> # Array input returns array output
    >>> n_arr = get_density_profile([0.5, 1.0, 2.0], params)
    >>> type(n_arr)
    <class 'numpy.ndarray'>
    """
    # Track if input was scalar for output conversion
    was_scalar = _is_scalar(r)
    r_arr = _to_array(r)

    # Extract parameters
    nISM = params['nISM'].value
    nCore = params['nCore'].value
    rCloud = params['rCloud'].value
    rCore = params['rCore'].value

    # Initialize output array
    n_arr = np.zeros_like(r_arr)

    # =============================================================================
    # Power-law profile
    # =============================================================================
    if params['dens_profile'].value == 'densPL':
        alpha = params['densPL_alpha'].value

        if alpha == 0:
            # Homogeneous cloud: constant density inside, ISM outside
            n_arr[:] = nISM  # Default to ISM
            n_arr[r_arr <= rCloud] = nCore
        else:
            # Power-law profile: n = nCore * (r/rCore)^alpha
            # with boundary conditions at rCore and rCloud

            # Default: power-law region
            n_arr = nCore * (r_arr / rCore) ** alpha

            # Inner core: constant density
            n_arr[r_arr <= rCore] = nCore

            # Outer ISM: constant density
            n_arr[r_arr > rCloud] = nISM

    # =============================================================================
    # Bonnor-Ebert sphere profile
    # =============================================================================
    elif params['dens_profile'].value == 'densBE':
        f_rho_rhoc = params['densBE_f_rho_rhoc'].value

        # Convert radius to dimensionless xi coordinate
        xi_arr = bonnorEbertSphere.r2xi(r_arr, params)

        # Get density ratio from interpolation function
        rho_rhoc = f_rho_rhoc(xi_arr)

        # Calculate number density
        n_arr = rho_rhoc * nCore

        # Outside cloud: ISM density
        n_arr[r_arr > rCloud] = nISM

    else:
        raise ValueError(f"Unknown density profile: {params['dens_profile'].value}")

    # Convert back to scalar if input was scalar
    return _to_output(n_arr, was_scalar)


# =============================================================================
# Tests
# =============================================================================

def test_scalar_array_consistency():
    """Test that scalar input returns scalar, array input returns array."""

    # Create mock params
    class MockValue:
        def __init__(self, val):
            self.value = val

    params = {
        'nISM': MockValue(1.0),
        'nCore': MockValue(1000.0),
        'rCloud': MockValue(10.0),
        'rCore': MockValue(1.0),
        'dens_profile': MockValue('densPL'),
        'densPL_alpha': MockValue(0),
    }

    # Test scalar input
    r_scalar = 5.0
    n_scalar = get_density_profile(r_scalar, params)
    assert isinstance(n_scalar, float), f"Expected float, got {type(n_scalar)}"
    print(f"✓ Scalar input (r={r_scalar}) → scalar output (n={n_scalar})")

    # Test array input
    r_array = np.array([0.5, 5.0, 15.0])
    n_array = get_density_profile(r_array, params)
    assert isinstance(n_array, np.ndarray), f"Expected ndarray, got {type(n_array)}"
    assert len(n_array) == len(r_array), f"Output length mismatch"
    print(f"✓ Array input (r={r_array}) → array output (n={n_array})")

    # Test list input (should behave like array)
    r_list = [0.5, 5.0, 15.0]
    n_list = get_density_profile(r_list, params)
    assert isinstance(n_list, np.ndarray), f"Expected ndarray, got {type(n_list)}"
    print(f"✓ List input (r={r_list}) → array output (n={n_list})")

    # Test single-element array (should still be array, not scalar)
    r_single_arr = np.array([5.0])
    n_single_arr = get_density_profile(r_single_arr, params)
    assert isinstance(n_single_arr, np.ndarray), f"Expected ndarray, got {type(n_single_arr)}"
    print(f"✓ Single-element array input → array output (n={n_single_arr})")

    print("\n✓ All scalar/array consistency tests passed!")


def test_homogeneous_cloud():
    """Test homogeneous cloud profile (alpha=0)."""

    class MockValue:
        def __init__(self, val):
            self.value = val

    params = {
        'nISM': MockValue(1.0),
        'nCore': MockValue(1000.0),
        'rCloud': MockValue(10.0),
        'rCore': MockValue(1.0),
        'dens_profile': MockValue('densPL'),
        'densPL_alpha': MockValue(0),
    }

    # Test inside cloud
    r_inside = 5.0
    n_inside = get_density_profile(r_inside, params)
    assert n_inside == params['nCore'].value, f"Expected nCore, got {n_inside}"
    print(f"✓ Inside cloud (r={r_inside}): n = nCore = {n_inside}")

    # Test exactly at cloud boundary
    r_boundary = 10.0
    n_boundary = get_density_profile(r_boundary, params)
    assert n_boundary == params['nCore'].value, f"Expected nCore at boundary, got {n_boundary}"
    print(f"✓ At cloud boundary (r={r_boundary}): n = nCore = {n_boundary}")

    # Test outside cloud
    r_outside = 15.0
    n_outside = get_density_profile(r_outside, params)
    assert n_outside == params['nISM'].value, f"Expected nISM, got {n_outside}"
    print(f"✓ Outside cloud (r={r_outside}): n = nISM = {n_outside}")

    print("\n✓ All homogeneous cloud tests passed!")


def test_power_law_profile():
    """Test power-law density profile (alpha != 0)."""

    class MockValue:
        def __init__(self, val):
            self.value = val

    alpha = -2.0  # Typical value for molecular clouds

    params = {
        'nISM': MockValue(1.0),
        'nCore': MockValue(1000.0),
        'rCloud': MockValue(10.0),
        'rCore': MockValue(1.0),
        'dens_profile': MockValue('densPL'),
        'densPL_alpha': MockValue(alpha),
    }

    nCore = params['nCore'].value
    rCore = params['rCore'].value
    rCloud = params['rCloud'].value
    nISM = params['nISM'].value

    # Test inside core (should be constant nCore)
    r_core = 0.5
    n_core = get_density_profile(r_core, params)
    assert n_core == nCore, f"Expected nCore inside core, got {n_core}"
    print(f"✓ Inside core (r={r_core}): n = nCore = {n_core}")

    # Test in power-law region
    r_pl = 5.0
    n_pl = get_density_profile(r_pl, params)
    expected_n_pl = nCore * (r_pl / rCore) ** alpha
    assert np.isclose(n_pl, expected_n_pl), f"Expected {expected_n_pl}, got {n_pl}"
    print(f"✓ Power-law region (r={r_pl}): n = {n_pl} (expected {expected_n_pl})")

    # Test outside cloud (should be nISM)
    r_outside = 15.0
    n_outside = get_density_profile(r_outside, params)
    assert n_outside == nISM, f"Expected nISM outside cloud, got {n_outside}"
    print(f"✓ Outside cloud (r={r_outside}): n = nISM = {n_outside}")

    print("\n✓ All power-law profile tests passed!")


def test_array_regions():
    """Test array input spanning multiple regions."""

    class MockValue:
        def __init__(self, val):
            self.value = val

    params = {
        'nISM': MockValue(1.0),
        'nCore': MockValue(1000.0),
        'rCloud': MockValue(10.0),
        'rCore': MockValue(1.0),
        'dens_profile': MockValue('densPL'),
        'densPL_alpha': MockValue(0),
    }

    # Array spanning inside and outside cloud
    r_arr = np.array([0.5, 5.0, 10.0, 15.0, 20.0])
    n_arr = get_density_profile(r_arr, params)

    nCore = params['nCore'].value
    nISM = params['nISM'].value

    # Check each element
    assert n_arr[0] == nCore, f"r=0.5 should give nCore"
    assert n_arr[1] == nCore, f"r=5.0 should give nCore"
    assert n_arr[2] == nCore, f"r=10.0 (boundary) should give nCore"
    assert n_arr[3] == nISM, f"r=15.0 should give nISM"
    assert n_arr[4] == nISM, f"r=20.0 should give nISM"

    print(f"✓ Array spanning regions: r={r_arr} → n={n_arr}")
    print("\n✓ All array region tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running density_profile tests")
    print("=" * 60)

    print("\n--- Test 1: Scalar/Array Consistency ---")
    test_scalar_array_consistency()

    print("\n--- Test 2: Homogeneous Cloud (alpha=0) ---")
    test_homogeneous_cloud()

    print("\n--- Test 3: Power-law Profile (alpha≠0) ---")
    test_power_law_profile()

    print("\n--- Test 4: Array Spanning Multiple Regions ---")
    test_array_regions()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def get_density_profile_OLD(r_arr,
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
        






