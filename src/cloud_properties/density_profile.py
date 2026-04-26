#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated density profile module for TRINITY.

Calculates number density n(r) for power-law and Bonnor-Ebert profiles.
Supports both scalar and array inputs with consistent output types.

For power-law density profile:
    n(r) = nCore                    for r <= rCore
    n(r) = nCore * (r/rCore)^alpha  for rCore < r <= rCloud
    n(r) = nISM                     for r > rCloud

For Bonnor-Ebert sphere:
    n(r) = nCore * f_rho_rhoc(xi)   for r <= rCloud
    n(r) = nISM                     for r > rCloud

The discontinuous jump from cloud density to nISM at r=rCloud is replaced
by a tanh bridge of width SMOOTH_FRAC * rCloud (1% by default) so the ODE
solvers in the phase modules see a C^infty rhs at the cloud boundary.

@author: Jia Wei
"""

import numpy as np

# Import Bonnor-Ebert sphere module for r2xi conversion
from src.cloud_properties.bonnorEbertSphere import r2xi as be_r2xi

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
        - nISM: ISM number density [1/pc^3] (code units)
        - nCore: Core number density [1/pc^3] (code units)
        - rCloud: Cloud radius [pc]
        - rCore: Core radius [pc]
        - dens_profile: 'densPL' or 'densBE'
        - densPL_alpha: power-law exponent (for densPL)
        - densBE_f_rho_rhoc: interpolation function (for densBE)

    Returns
    -------
    n : float or np.ndarray
        Number density at radius r [1/pc^3] (code units).
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
    # Numerical regularization at rCloud
    # ---------------------------------
    # The cloud->ISM transition is mathematically idealized as a step. That
    # step makes mShell_dot = 4*pi*r^2*rho(r)*v in the phase ODEs jump by
    # ~10^3 across r=rCloud, which can cause LSODA to stall trying to refine
    # its step below min_step. We replace the step with a tanh bridge of
    # width SMOOTH_FRAC*rCloud so the rhs is C^infty everywhere. The width
    # is well below physical uncertainty in cloud-edge structure; mass
    # conservation holds to O(SMOOTH_FRAC^2).
    # =============================================================================
    SMOOTH_FRAC = 0.01
    delta = SMOOTH_FRAC * rCloud
    w_outside = 0.5 * (1.0 + np.tanh((r_arr - rCloud) / delta))

    # =============================================================================
    # Power-law profile
    # =============================================================================
    if params['dens_profile'].value == 'densPL':
        alpha = params['densPL_alpha'].value

        if alpha == 0:
            # Homogeneous cloud: constant nCore inside, blends to nISM at rCloud
            n_inside = np.full_like(r_arr, nCore)
        else:
            # Power-law profile: n = nCore * (r/rCore)^alpha
            n_inside = nCore * (r_arr / rCore) ** alpha
            # Inner core: constant density (rCore is far below rCloud, so the
            # smoothing band does not reach rCore in any realistic setup)
            n_inside[r_arr <= rCore] = nCore

        n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside

    # =============================================================================
    # Bonnor-Ebert sphere profile
    # =============================================================================
    elif params['dens_profile'].value == 'densBE':
        f_rho_rhoc = params['densBE_f_rho_rhoc'].value

        # Convert radius to dimensionless xi coordinate
        xi_arr = be_r2xi(r_arr, params)

        # Get density ratio from interpolation function (interpolator already
        # handles xi values for r > rCloud — same as the original code path)
        rho_rhoc = f_rho_rhoc(xi_arr)

        n_inside = rho_rhoc * nCore
        n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside

    else:
        raise ValueError(f"Unknown density profile: {params['dens_profile'].value}")

    # Convert back to scalar if input was scalar
    return _to_output(n_arr, was_scalar)
