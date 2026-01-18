#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified bubble luminosity module for TRINITY.

This module provides pure (side-effect-free) functions for calculating bubble
properties, with fixes for the 1/r singularities in the ODE system.

Key changes from original:
1. Fixed 1/r singularities in get_bubble_ODE with regularization
2. Pure functions that return dicts instead of mutating params
3. Factored out duplicate radius array construction
4. Improved convergence handling for dMdt solver

@author: TRINITY Team (refactored for pure functions)
"""

import numpy as np
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import src._functions.operations as operations
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.cooling import net_coolingcurve
import src._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

FOUR_PI = 4.0 * np.pi
R_MIN = 1e-6  # Minimum radius for regularization [pc]


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class BubbleProperties:
    """Container for computed bubble properties."""
    R1: float  # Inner bubble radius [pc]
    Pb: float  # Bubble pressure [internal units]
    dMdt: float  # Mass flux [Msun/Myr]
    T_rgoal: float  # Temperature at r_goal [K]
    L_total: float  # Total cooling luminosity [internal units]
    L_bubble: float  # Bubble zone cooling [internal units]
    L_conduction: float  # Conduction zone cooling [internal units]
    L_intermediate: float  # Intermediate zone cooling [internal units]
    T_arr: np.ndarray  # Temperature profile [K]
    v_arr: np.ndarray  # Velocity profile [pc/Myr]
    r_arr: np.ndarray  # Radius array [pc]
    n_arr: np.ndarray  # Density profile [1/pc³]
    dTdr_arr: np.ndarray  # Temperature gradient profile [K/pc]
    Tavg: float  # Average temperature [K]


# =============================================================================
# Regularized ODE Function
# =============================================================================

def get_bubble_ODE_regularized(r: float, y: np.ndarray, params_dict: Dict) -> np.ndarray:
    """
    Regularized ODE for bubble structure - fixes 1/r singularity.

    This function implements Weaver+77 Eqs 42-43 with regularization
    near r=0 to prevent numerical instability.

    Parameters
    ----------
    r : float
        Radius [pc]
    y : ndarray
        State vector [v, T, dTdr]
    params_dict : dict
        Parameters needed for ODE (plain dict, not DescribedItem)

    Returns
    -------
    dydr : ndarray
        Derivatives [dvdr, dTdr, dTdrr]
    """
    v, T, dTdr = y

    # Regularization: use small cutoff near r=0
    r_safe = max(r, R_MIN)

    # Ensure positive temperature
    T = max(T, 1e3)

    # Extract parameters
    Pb = params_dict['Pb']
    k_B = params_dict['k_B']
    Qi = params_dict['Qi']
    t_now = params_dict['t_now']
    C_thermal = params_dict['C_thermal']
    cool_alpha = params_dict['cool_alpha']
    cool_beta = params_dict['cool_beta']
    cool_delta = params_dict['cool_delta']

    # Get density and ionizing flux
    ndens = Pb / (2 * k_B * T)
    phi = Qi / (FOUR_PI * r_safe**2)

    # Cooling rate
    dudt = net_coolingcurve.get_dudt(t_now, ndens, T, phi, params_dict)

    # Terminal velocity term
    v_term = cool_alpha * r_safe / t_now

    # Temperature gradient derivative (Weaver+77 Eq 42)
    # FIXED: Use r_safe to avoid 1/r singularity
    dTdrr = (Pb / (C_thermal * T**(5/2))) * (
        (cool_beta + 2.5 * cool_delta) / t_now +
        2.5 * (v - v_term) * dTdr / T - dudt / Pb
    ) - 2.5 * dTdr**2 / T - 2 * dTdr / r_safe

    # Velocity gradient (Weaver+77 Eq 43)
    # FIXED: Use r_safe to avoid 1/r singularity
    dvdr = ((cool_beta + cool_delta) / t_now +
            (v - v_term) * dTdr / T - 2 * v / r_safe)

    return np.array([dvdr, dTdr, dTdrr])


# =============================================================================
# Helper Functions
# =============================================================================

def create_radius_array(R1: float, R2: float, n_points: int = 2000) -> np.ndarray:
    """
    Create radius array for ODE integration.

    Array is in DECREASING order (from R2 to R1) with higher
    resolution near the boundaries.

    Parameters
    ----------
    R1 : float
        Inner radius [pc]
    R2 : float
        Outer radius [pc]
    n_points : int
        Number of points

    Returns
    -------
    r_arr : ndarray
        Radius array [pc], decreasing order
    """
    # Log-spaced from R1 to R2
    r_arr = np.logspace(np.log10(R1), np.log10(R2), n_points)

    # Reverse to get decreasing order (R2 -> R1)
    r_arr = r_arr[::-1]

    return r_arr


def get_initial_conditions(dMdt: float, params_dict: Dict) -> Tuple[float, float, float, float]:
    """
    Compute initial conditions for ODE integration.

    Based on Weaver+77 boundary conditions at r ≈ R2.

    Parameters
    ----------
    dMdt : float
        Mass flux [Msun/Myr]
    params_dict : dict
        Parameters dictionary

    Returns
    -------
    r2_prime : float
        Starting radius [pc]
    T : float
        Initial temperature [K]
    dTdr : float
        Initial temperature gradient [K/pc]
    v : float
        Initial velocity [pc/Myr]
    """
    R2 = params_dict['R2']
    Pb = params_dict['Pb']
    t_now = params_dict['t_now']
    k_B = params_dict['k_B']
    mu_ion = params_dict['mu_ion']
    C_thermal = params_dict['C_thermal']
    cool_alpha = params_dict['cool_alpha']

    # Small offset from R2
    dR2 = max(1e-11, 1e-5 * R2)

    # Constant for temperature calculation
    constant = (25 * Pb / (8 * np.pi * C_thermal))**2

    # Temperature at r2_prime
    T = (constant * dMdt * dR2 / (FOUR_PI * R2**2))**(2/5)
    T = max(T, 1e4)  # Floor at 10^4 K

    # Velocity at r2_prime
    v = cool_alpha * R2 / t_now - dMdt / (FOUR_PI * R2**2) * k_B * T / (mu_ion * Pb)

    # Temperature gradient
    dTdr = -2/5 * T / dR2

    # Starting radius
    r2_prime = R2 - dR2

    return r2_prime, T, dTdr, v


def compute_velocity_residual(dMdt: float, params_dict: Dict) -> float:
    """
    Compute residual for dMdt solver.

    Integrates ODE and checks if boundary condition v(R1) → 0 is satisfied.

    Parameters
    ----------
    dMdt : float
        Mass flux guess [Msun/Myr]
    params_dict : dict
        Parameters dictionary

    Returns
    -------
    residual : float
        Residual (should be ~0 when converged)
    """
    # Ensure dMdt is a scalar (fsolve passes arrays)
    dMdt = float(np.atleast_1d(dMdt)[0])
    # Ensure positive dMdt
    dMdt = max(dMdt, 1e-20)

    # Get initial conditions
    r2_prime, T_init, dTdr_init, v_init = get_initial_conditions(dMdt, params_dict)

    # Create radius array
    R1 = params_dict['R1']
    r_arr = create_radius_array(R1, r2_prime, n_points=2000)

    # Initial state
    y0 = [v_init, T_init, dTdr_init]

    # Integrate ODE
    try:
        solution = scipy.integrate.odeint(
            lambda y, r: get_bubble_ODE_regularized(r, y, params_dict),
            y0,
            r_arr,
            full_output=0
        )

        v_arr = solution[:, 0]
        T_arr = solution[:, 1]

        # Check for valid solution
        min_T = np.min(T_arr)
        if min_T < 3e4:
            # Penalize solutions with too-low temperature
            return (3e4 / (min_T + 1e-1))**2

        if np.isnan(min_T):
            return -1e3

        if not operations.monotonic(T_arr):
            return 1e2

        # Residual: v should approach 0 at inner boundary
        residual = (v_arr[-1] - 0) / (v_arr[0] + 1e-4)

        return residual

    except Exception as e:
        logger.warning(f"ODE integration failed: {e}")
        return 1e3


# =============================================================================
# Cooling Luminosity Functions (Pure)
# =============================================================================

# Temperature thresholds for cooling zones
T_COOLING_SWITCH = 1e4      # Below this, no cooling [K]
T_CIE_SWITCH = 10**5.5      # Above this, use CIE cooling [K]


def _compute_L_bubble(
    T_arr: np.ndarray,
    r_arr: np.ndarray,
    n_arr: np.ndarray,
    idx_CIE_switch: int,
    cooling_CIE_interp,
) -> Tuple[float, float]:
    """
    Compute CIE cooling luminosity for the hot bubble zone (T > 10^5.5 K).

    Parameters
    ----------
    T_arr : ndarray
        Temperature profile [K]
    r_arr : ndarray
        Radius array [pc], decreasing order
    n_arr : ndarray
        Density profile [1/pc³]
    idx_CIE_switch : int
        Index where T crosses CIE threshold
    cooling_CIE_interp : callable
        CIE cooling curve interpolator (log10(T) -> log10(Λ))

    Returns
    -------
    L_bubble : float
        Cooling luminosity in bubble zone [internal units]
    Tavg_bubble : float
        Volume-weighted temperature contribution [K pc³]
    """
    if idx_CIE_switch >= len(T_arr):
        return 0.0, 0.0

    # Slice arrays for T > T_CIE_SWITCH (r is decreasing, T is increasing)
    T_bubble = T_arr[idx_CIE_switch:]
    r_bubble = r_arr[idx_CIE_switch:]
    n_bubble = n_arr[idx_CIE_switch:]

    if len(T_bubble) == 0 or cooling_CIE_interp is None:
        return 0.0, 0.0

    # CIE cooling rate [internal units]
    # Λ(T) from interpolation: log10(Λ_cgs) = interp(log10(T))
    Lambda_bubble = 10**(cooling_CIE_interp(np.log10(T_bubble))) * cvt.Lambda_cgs2au

    # Integrand: n² Λ 4πr²
    integrand = n_bubble**2 * Lambda_bubble * FOUR_PI * r_bubble**2

    # Integrate (r is decreasing, so use abs)
    L_bubble = np.abs(np.trapz(integrand, x=r_bubble))

    # Volume-weighted temperature for average calculation
    Tavg_bubble = np.abs(np.trapz(r_bubble**2 * T_bubble, x=r_bubble))

    return L_bubble, Tavg_bubble


def _compute_L_conduction(
    T_arr: np.ndarray,
    r_arr: np.ndarray,
    n_arr: np.ndarray,
    dTdr_arr: np.ndarray,
    idx_cooling_switch: int,
    idx_CIE_switch: int,
    Pb: float,
    k_B: float,
    Qi: float,
    cooling_nonCIE,
    heating_nonCIE,
    params_dict: Dict,
) -> Tuple[float, float, float]:
    """
    Compute non-CIE cooling luminosity for conduction zone (10^4 < T < 10^5.5 K).

    Parameters
    ----------
    T_arr, r_arr, n_arr, dTdr_arr : ndarray
        Bubble structure profiles
    idx_cooling_switch : int
        Index where T crosses 10^4 K threshold
    idx_CIE_switch : int
        Index where T crosses 10^5.5 K threshold
    Pb : float
        Bubble pressure [internal units]
    k_B : float
        Boltzmann constant [internal units]
    Qi : float
        Ionizing photon rate [1/Myr]
    cooling_nonCIE, heating_nonCIE : object
        CLOUDY interpolators for non-CIE cooling/heating
    params_dict : dict
        Additional parameters for ODE re-integration if needed

    Returns
    -------
    L_conduction : float
        Cooling luminosity in conduction zone [internal units]
    Tavg_conduction : float
        Volume-weighted temperature contribution [K pc³]
    dTdr_at_cooling_switch : float
        Temperature gradient at cooling switch [K/pc]
    """
    # No conduction zone if indices are equal (steep shock front)
    if idx_cooling_switch == idx_CIE_switch:
        if idx_CIE_switch < len(dTdr_arr):
            return 0.0, 0.0, dTdr_arr[idx_CIE_switch]
        return 0.0, 0.0, 0.0

    if cooling_nonCIE is None or heating_nonCIE is None:
        return 0.0, 0.0, dTdr_arr[idx_cooling_switch] if idx_cooling_switch < len(dTdr_arr) else 0.0

    # Check if zone is well-resolved (need at least 100 points for accuracy)
    n_points_in_zone = idx_CIE_switch - idx_cooling_switch

    if n_points_in_zone < 100:
        # Re-solve ODE with higher resolution in this zone
        lowres_r = r_arr[:idx_CIE_switch + 1]
        original_rmax = np.max(lowres_r)
        original_rmin = np.min(lowres_r)

        _highres = 100
        r_conduction = np.linspace(original_rmax, original_rmin, _highres)

        # Re-integrate ODE for this zone
        try:
            psoln = scipy.integrate.odeint(
                lambda y, r: get_bubble_ODE_regularized(r, y, params_dict),
                [T_arr[idx_cooling_switch], T_arr[idx_cooling_switch], dTdr_arr[idx_cooling_switch]],
                r_conduction,
                tfirst=True
            )
            v_conduction = psoln[:, 0]
            T_conduction = psoln[:, 1]
            dTdr_conduction = psoln[:, 2]
        except Exception:
            # Fall back to low-res data
            r_conduction = r_arr[:idx_CIE_switch + 1]
            T_conduction = T_arr[:idx_CIE_switch + 1]
            dTdr_conduction = dTdr_arr[:idx_CIE_switch + 1]

        # Mask out values above CIE switch
        mask = T_conduction < T_CIE_SWITCH
        r_conduction = r_conduction[mask]
        T_conduction = T_conduction[mask]
        dTdr_conduction = dTdr_conduction[mask]

        dTdr_at_cooling_switch = dTdr_conduction[0] if len(dTdr_conduction) > 0 else 0.0
    else:
        # Use existing arrays (well-resolved)
        r_conduction = r_arr[:idx_CIE_switch + 1]
        T_conduction = T_arr[:idx_CIE_switch + 1]
        dTdr_conduction = dTdr_arr[:idx_CIE_switch + 1]
        dTdr_at_cooling_switch = dTdr_conduction[0] if len(dTdr_conduction) > 0 else 0.0

    if len(r_conduction) == 0:
        return 0.0, 0.0, dTdr_at_cooling_switch

    # Calculate density and ionizing flux
    n_conduction = Pb / (2 * k_B * T_conduction)
    phi_conduction = Qi / (FOUR_PI * r_conduction**2)

    # Get cooling/heating rates from CLOUDY interpolators [cgs]
    try:
        cooling = 10**cooling_nonCIE.interp(
            np.transpose(np.log10([
                n_conduction / cvt.ndens_cgs2au,
                T_conduction,
                phi_conduction / cvt.phi_cgs2au
            ]))
        )
        heating = 10**heating_nonCIE.interp(
            np.transpose(np.log10([
                n_conduction / cvt.ndens_cgs2au,
                T_conduction,
                phi_conduction / cvt.phi_cgs2au
            ]))
        )
    except Exception as e:
        logger.warning(f"Non-CIE interpolation failed: {e}")
        return 0.0, 0.0, dTdr_at_cooling_switch

    # Net cooling rate [internal units]
    dudt_conduction = (heating - cooling) * cvt.dudt_cgs2au

    # Integrand
    integrand = dudt_conduction * FOUR_PI * r_conduction**2

    # Integrate
    L_conduction = np.abs(np.trapz(integrand, x=r_conduction))

    # Volume-weighted temperature
    Tavg_conduction = np.abs(np.trapz(r_conduction**2 * T_conduction, x=r_conduction))

    return L_conduction, Tavg_conduction, dTdr_at_cooling_switch


def _compute_L_intermediate(
    T_arr: np.ndarray,
    r_arr: np.ndarray,
    dTdr_arr: np.ndarray,
    idx_cooling_switch: int,
    dTdr_at_cooling_switch: float,
    Pb: float,
    k_B: float,
    Qi: float,
    cooling_CIE_interp,
    cooling_nonCIE,
    heating_nonCIE,
) -> Tuple[float, float]:
    """
    Compute cooling luminosity for intermediate region (near shell, T ~ 10^4 K).

    This region is between r_arr[idx_cooling_switch] and R2_coolingswitch
    where T crosses exactly 10^4 K.

    Parameters
    ----------
    T_arr, r_arr, dTdr_arr : ndarray
        Bubble structure profiles
    idx_cooling_switch : int
        Index where T crosses 10^4 K threshold
    dTdr_at_cooling_switch : float
        Temperature gradient at cooling switch [K/pc]
    Pb : float
        Bubble pressure [internal units]
    k_B : float
        Boltzmann constant [internal units]
    Qi : float
        Ionizing photon rate [1/Myr]
    cooling_CIE_interp : callable
        CIE cooling curve interpolator
    cooling_nonCIE, heating_nonCIE : object
        CLOUDY interpolators for non-CIE cooling/heating

    Returns
    -------
    L_intermediate : float
        Cooling luminosity in intermediate zone [internal units]
    Tavg_intermediate : float
        Volume-weighted temperature contribution [K pc³]
    """
    if idx_cooling_switch >= len(T_arr) or dTdr_at_cooling_switch == 0:
        return 0.0, 0.0

    # Find R2_coolingswitch where T = T_COOLING_SWITCH exactly
    # Using linear extrapolation: T = T0 + dTdr * (r - r0)
    T_at_switch = T_arr[idx_cooling_switch]
    r_at_switch = r_arr[idx_cooling_switch]

    # R2_coolingswitch = r where T = T_COOLING_SWITCH
    R2_coolingswitch = (T_COOLING_SWITCH - T_at_switch) / dTdr_at_cooling_switch + r_at_switch

    # If this region is tiny or inverted, skip
    if R2_coolingswitch >= r_at_switch:
        return 0.0, 0.0

    # Create fine radius array in this small region
    r_intermediate = np.linspace(r_at_switch, R2_coolingswitch, num=1000, endpoint=True)

    # Interpolate temperature
    T_interp_func = interp1d(
        np.array([r_at_switch, R2_coolingswitch]),
        np.array([T_at_switch, T_COOLING_SWITCH]),
        kind='linear'
    )
    T_intermediate = T_interp_func(r_intermediate)

    # Calculate density and ionizing flux
    n_intermediate = Pb / (2 * k_B * T_intermediate)
    phi_intermediate = Qi / (FOUR_PI * r_intermediate**2)

    # Split into CIE and non-CIE regimes based on local temperature
    mask_nonCIE = T_intermediate < T_CIE_SWITCH
    mask_CIE = T_intermediate >= T_CIE_SWITCH

    L_intermediate = 0.0

    # Non-CIE contribution
    if np.any(mask_nonCIE) and cooling_nonCIE is not None and heating_nonCIE is not None:
        try:
            cooling = 10**cooling_nonCIE.interp(
                np.transpose(np.log10([
                    n_intermediate[mask_nonCIE] / cvt.ndens_cgs2au,
                    T_intermediate[mask_nonCIE],
                    phi_intermediate[mask_nonCIE] / cvt.phi_cgs2au
                ]))
            )
            heating = 10**heating_nonCIE.interp(
                np.transpose(np.log10([
                    n_intermediate[mask_nonCIE] / cvt.ndens_cgs2au,
                    T_intermediate[mask_nonCIE],
                    phi_intermediate[mask_nonCIE] / cvt.phi_cgs2au
                ]))
            )
            dudt = (heating - cooling) * cvt.dudt_cgs2au
            integrand = dudt * FOUR_PI * r_intermediate[mask_nonCIE]**2
            L_intermediate += np.abs(np.trapz(integrand, x=r_intermediate[mask_nonCIE]))
        except Exception as e:
            logger.warning(f"Non-CIE intermediate calc failed: {e}")

    # CIE contribution
    if np.any(mask_CIE) and cooling_CIE_interp is not None:
        try:
            Lambda = 10**(cooling_CIE_interp(np.log10(T_intermediate[mask_CIE]))) * cvt.Lambda_cgs2au
            integrand = n_intermediate[mask_CIE]**2 * Lambda * FOUR_PI * r_intermediate[mask_CIE]**2
            L_intermediate += np.abs(np.trapz(integrand, x=r_intermediate[mask_CIE]))
        except Exception as e:
            logger.warning(f"CIE intermediate calc failed: {e}")

    # Volume-weighted temperature
    Tavg_intermediate = np.abs(np.trapz(r_intermediate**2 * T_intermediate, x=r_intermediate))

    return L_intermediate, Tavg_intermediate


def compute_cooling_luminosity_pure(
    T_arr: np.ndarray,
    r_arr: np.ndarray,
    n_arr: np.ndarray,
    dTdr_arr: np.ndarray,
    Pb: float,
    k_B: float,
    Qi: float,
    R1: float,
    cooling_CIE_interp,
    cooling_nonCIE,
    heating_nonCIE,
    params_dict: Dict,
) -> Tuple[float, float, float, float, float]:
    """
    Compute cooling luminosity in all three zones - PURE FUNCTION.

    Parameters
    ----------
    T_arr, r_arr, n_arr, dTdr_arr : ndarray
        Bubble structure profiles (r is decreasing, T is increasing)
    Pb : float
        Bubble pressure [internal units]
    k_B : float
        Boltzmann constant [internal units]
    Qi : float
        Ionizing photon rate [1/Myr]
    R1 : float
        Inner bubble radius [pc]
    cooling_CIE_interp : callable
        CIE cooling curve interpolator
    cooling_nonCIE, heating_nonCIE : object
        CLOUDY interpolators
    params_dict : dict
        Parameters for potential ODE re-integration

    Returns
    -------
    L_total : float
        Total cooling luminosity [internal units]
    L_bubble : float
        CIE zone cooling [internal units]
    L_conduction : float
        Non-CIE zone cooling [internal units]
    L_intermediate : float
        Transition zone cooling [internal units]
    Tavg : float
        Volume-weighted average temperature [K]
    """
    # Find zone boundary indices
    # r is decreasing, T is increasing
    idx_CIE_switch = operations.find_nearest_higher(T_arr, T_CIE_SWITCH)
    idx_cooling_switch = operations.find_nearest_higher(T_arr, T_COOLING_SWITCH)

    # Zone 1: Bubble (CIE) - T > 10^5.5 K
    L_bubble, Tavg_bubble = _compute_L_bubble(
        T_arr, r_arr, n_arr, idx_CIE_switch, cooling_CIE_interp
    )

    # Zone 2: Conduction (non-CIE) - 10^4 < T < 10^5.5 K
    L_conduction, Tavg_conduction, dTdr_at_cooling_switch = _compute_L_conduction(
        T_arr, r_arr, n_arr, dTdr_arr,
        idx_cooling_switch, idx_CIE_switch,
        Pb, k_B, Qi,
        cooling_nonCIE, heating_nonCIE,
        params_dict
    )

    # Zone 3: Intermediate - transition to shell
    L_intermediate, Tavg_intermediate = _compute_L_intermediate(
        T_arr, r_arr, dTdr_arr,
        idx_cooling_switch, dTdr_at_cooling_switch,
        Pb, k_B, Qi,
        cooling_CIE_interp, cooling_nonCIE, heating_nonCIE
    )

    # Total cooling
    L_total = L_bubble + L_conduction + L_intermediate

    # Compute volume-weighted average temperature
    r_bubble = r_arr[idx_CIE_switch:] if idx_CIE_switch < len(r_arr) else np.array([])
    r_conduction = r_arr[:idx_CIE_switch + 1] if idx_CIE_switch < len(r_arr) else np.array([])

    Tavg = 0.0
    if len(r_bubble) > 1:
        vol_bubble = r_bubble[0]**3 - r_bubble[-1]**3
        if vol_bubble > 0:
            Tavg += 3 * Tavg_bubble / vol_bubble
    if len(r_conduction) > 1 and idx_cooling_switch != idx_CIE_switch:
        vol_conduction = r_conduction[0]**3 - r_conduction[-1]**3
        if vol_conduction > 0:
            Tavg += 3 * Tavg_conduction / vol_conduction

    return L_total, L_bubble, L_conduction, L_intermediate, Tavg


# =============================================================================
# Pure Interface Functions
# =============================================================================

def get_bubbleproperties_pure(R2: float, v2: float, Eb: float, t_now: float,
                               params) -> BubbleProperties:
    """
    Calculate bubble properties - PURE FUNCTION.

    This function returns a BubbleProperties object instead of
    mutating the params dictionary.

    Parameters
    ----------
    R2 : float
        Shell radius [pc]
    v2 : float
        Shell velocity [pc/Myr]
    Eb : float
        Bubble energy [internal units]
    t_now : float
        Current time [Myr]
    params : dict
        Parameter dictionary (read-only)

    Returns
    -------
    props : BubbleProperties
        Computed bubble properties
    """
    # Extract needed parameters into plain dict
    def get_val(key, default=0.0):
        if key in params and hasattr(params[key], 'value'):
            return params[key].value
        elif key in params:
            return params[key]
        return default

    Lmech_total = get_val('Lmech_total')
    v_mech_total = get_val('v_mech_total')
    gamma_adia = get_val('gamma_adia', 5/3)

    # Calculate R1
    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-3 * R2, R2 * 0.999,
            args=([Lmech_total, Eb, v_mech_total, R2])
        )
    except ValueError:
        R1 = 0.01 * R2
        logger.warning(f"R1 brentq failed, using R1=0.01*R2")

    # Calculate bubble pressure
    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, gamma_adia)

    # Simple wrapper for values that need .value attribute
    class ValWrapper:
        def __init__(self, v):
            self.value = v

    # Build params dict for ODE solver
    # Some fields need .value wrapper for compatibility with net_coolingcurve.get_dudt
    params_dict = {
        'R1': R1,
        'R2': R2,
        'Pb': Pb,
        't_now': t_now,
        'k_B': get_val('k_B'),
        'mu_ion': get_val('mu_ion'),
        'C_thermal': get_val('C_thermal'),
        'cool_alpha': get_val('cool_alpha'),
        'cool_beta': get_val('cool_beta'),
        'cool_delta': get_val('cool_delta'),
        'Qi': get_val('Qi'),
        'Lmech_total': Lmech_total,
        'v_mech_total': v_mech_total,
        # Fields needed by net_coolingcurve.get_dudt (require .value attribute)
        'cStruc_cooling_nonCIE': ValWrapper(get_val('cStruc_cooling_nonCIE', None)),
        'cStruc_heating_nonCIE': ValWrapper(get_val('cStruc_heating_nonCIE', None)),
        'cStruc_net_nonCIE_interpolation': ValWrapper(get_val('cStruc_net_nonCIE_interpolation', None)),
        'cStruc_cooling_CIE_interpolation': ValWrapper(get_val('cStruc_cooling_CIE_interpolation', None)),
        'cStruc_cooling_CIE_logT': ValWrapper(get_val('cStruc_cooling_CIE_logT', None)),
        'ZCloud': ValWrapper(get_val('ZCloud', 1.0)),
    }

    # Get initial dMdt guess
    dMdt_init = get_val('bubble_dMdt', np.nan)
    if np.isnan(dMdt_init):
        dMdt_init = _compute_init_dMdt(params_dict)

    # Solve for dMdt
    try:
        dMdt = scipy.optimize.fsolve(
            lambda x: compute_velocity_residual(x, params_dict),
            dMdt_init,
            xtol=1e-4,
            factor=50,
            epsfcn=1e-4
        )[0]
    except Exception as e:
        logger.warning(f"dMdt solver failed: {e}, using initial guess")
        dMdt = dMdt_init

    # Compute final profiles
    r2_prime, T_init, dTdr_init, v_init = get_initial_conditions(dMdt, params_dict)
    r_arr = create_radius_array(R1, r2_prime, n_points=2000)

    y0 = [v_init, T_init, dTdr_init]
    solution = scipy.integrate.odeint(
        lambda y, r: get_bubble_ODE_regularized(r, y, params_dict),
        y0,
        r_arr
    )

    v_arr = solution[:, 0]
    T_arr = solution[:, 1]
    dTdr_arr = solution[:, 2]

    # Density profile
    n_arr = Pb / (2 * params_dict['k_B'] * T_arr)

    # Find temperature at goal radius
    xi_Tb = get_val('bubble_xi_Tb', 0.9)
    r_Tb = R1 + xi_Tb * (R2 - R1)

    # Interpolate to find T at r_Tb
    T_interp = interp1d(r_arr, T_arr, kind='linear', fill_value='extrapolate')
    T_rgoal = float(T_interp(r_Tb))

    # Extract cooling interpolators from params
    cooling_CIE_interp = get_val('cStruc_cooling_CIE_interpolation', None)
    cooling_nonCIE = get_val('cStruc_cooling_nonCIE', None)
    heating_nonCIE = get_val('cStruc_heating_nonCIE', None)

    # Compute cooling luminosities using pure function
    L_total, L_bubble, L_conduction, L_intermediate, Tavg = compute_cooling_luminosity_pure(
        T_arr=T_arr,
        r_arr=r_arr,
        n_arr=n_arr,
        dTdr_arr=dTdr_arr,
        Pb=Pb,
        k_B=params_dict['k_B'],
        Qi=params_dict['Qi'],
        R1=R1,
        cooling_CIE_interp=cooling_CIE_interp,
        cooling_nonCIE=cooling_nonCIE,
        heating_nonCIE=heating_nonCIE,
        params_dict=params_dict,
    )

    return BubbleProperties(
        R1=R1,
        Pb=Pb,
        dMdt=dMdt,
        T_rgoal=T_rgoal,
        L_total=L_total,
        L_bubble=L_bubble,
        L_conduction=L_conduction,
        L_intermediate=L_intermediate,
        T_arr=T_arr,
        v_arr=v_arr,
        r_arr=r_arr,
        n_arr=n_arr,
        dTdr_arr=dTdr_arr,
        Tavg=Tavg,
    )


def _compute_init_dMdt(params_dict: Dict) -> float:
    """
    Compute initial dMdt guess using Weaver+77 Eq. 33.

    Parameters
    ----------
    params_dict : dict
        Parameters dictionary

    Returns
    -------
    dMdt_init : float
        Initial guess for mass flux [Msun/Myr]
    """
    R2 = params_dict['R2']
    Pb = params_dict['Pb']
    t_now = params_dict['t_now']
    k_B = params_dict['k_B']
    C_thermal = params_dict['C_thermal']
    mu_ion = params_dict.get('mu_ion', params_dict.get('mu_atom', 1e-57))

    # Weaver+77 Eq. 33 (with empirical factor)
    dMdt_factor = 1.646

    dMdt_init = (12/75 * dMdt_factor**(5/2) * FOUR_PI * R2**3 / t_now *
                 mu_ion / k_B * (t_now * C_thermal / R2**2)**(2/7) *
                 Pb**(5/7))

    return dMdt_init


# =============================================================================
# Wrapper for backward compatibility
# =============================================================================

def get_bubbleproperties_wrapper(params):
    """
    Wrapper that provides interface compatible with old code.

    DEPRECATED: Use get_bubbleproperties_pure instead.

    This wrapper:
    1. Calls the pure function
    2. Updates params dict with computed values (for compatibility)

    Parameters
    ----------
    params : dict
        Parameter dictionary

    Returns
    -------
    None (updates params in place for compatibility)
    """
    import warnings
    warnings.warn(
        "get_bubbleproperties_wrapper is deprecated. Use get_bubbleproperties_pure.",
        DeprecationWarning,
        stacklevel=2
    )

    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    t_now = params['t_now'].value

    props = get_bubbleproperties_pure(R2, v2, Eb, t_now, params)

    # Update params dict (for compatibility)
    params['R1'].value = props.R1
    params['Pb'].value = props.Pb
    params['bubble_dMdt'].value = props.dMdt
    params['bubble_T_r_Tb'].value = props.T_rgoal
    params['bubble_LTotal'].value = props.L_total
    params['bubble_Tavg'].value = props.Tavg
    params['bubble_T_arr'].value = props.T_arr
    params['bubble_v_arr'].value = props.v_arr
    params['bubble_r_arr'].value = props.r_arr
    params['bubble_n_arr'].value = props.n_arr
