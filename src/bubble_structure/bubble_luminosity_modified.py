#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified bubble luminosity with dataclass returns.

This module provides bubble property calculations that return a dataclass
instead of mutating the params dictionary. This is essential for use with
adaptive ODE solvers.

Key difference from bubble_luminosity.py:
- get_bubbleproperties_pure() returns a BubbleProperties dataclass
- No dictionary mutations during calculation
- Use updateDict(params, bubble_data) after call returns

TODO: add docstrings for each function

@author: Jia Wei Teh
"""

import numpy as np
import sys
import scipy.optimize
import scipy.integrate
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Optional
import logging

import src._functions.operations as operations
import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.cooling import net_coolingcurve
import src._functions.unit_conversions as cvt
from src.sb99.update_feedback import get_currentSB99feedback

logger = logging.getLogger(__name__)

# NumPy compatibility: trapz was renamed to trapezoid in NumPy 2.0
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


# =============================================================================
# Grid construction utilities
# =============================================================================

def _clean_radius_grid(r_array: np.ndarray, min_rel_spacing: float = 1e-10) -> np.ndarray:
    """
    Clean radius grid by removing near-duplicates and enforcing strict monotonicity.

    This fixes LSODA 'intdy-- t illegal' warnings caused by consecutive grid points
    being too close together (within floating-point precision).

    Parameters
    ----------
    r_array : np.ndarray
        Radius array (expected to be in decreasing order)
    min_rel_spacing : float
        Minimum relative spacing between consecutive points

    Returns
    -------
    np.ndarray
        Cleaned radius array in strictly decreasing order
    """
    # Handle empty or single-element arrays
    if len(r_array) <= 1:
        return r_array

    # Determine if array is decreasing or increasing
    is_decreasing = r_array[0] > r_array[-1]

    if is_decreasing:
        # Work with decreasing array
        result = [r_array[0]]
        for r in r_array[1:]:
            spacing = result[-1] - r
            if spacing > min_rel_spacing * abs(result[-1]):
                result.append(r)
        return np.array(result)
    else:
        # Work with increasing array
        result = [r_array[0]]
        for r in r_array[1:]:
            spacing = r - result[-1]
            if spacing > min_rel_spacing * abs(result[-1]) if result[-1] != 0 else spacing > min_rel_spacing:
                result.append(r)
        return np.array(result)


def _create_legacy_radius_grid(R1: float, r2Prime: float) -> np.ndarray:
    """
    Create radius grid using the legacy method (three logspace chunks).

    This is the original implementation that creates ~60k points.
    Use this as fallback if adaptive refinement fails.

    Parameters
    ----------
    R1 : float
        Inner bubble radius
    r2Prime : float
        Outer bubble radius (slightly inside R2)

    Returns
    -------
    np.ndarray
        Radius array in decreasing order (~60k points)
    """
    # Step 1: Base logspace (reverse logspace for higher density at outer edge)
    r_array = (r2Prime + R1) - np.logspace(np.log10(R1), np.log10(r2Prime), int(2e4))

    # Step 2: Improve resolution at the start (outer edge)
    r_improve = np.logspace(np.log10(r_array[0]), np.log10(r_array[2]), int(2e4))
    r_array = np.insert(r_array[3:], 0, r_improve)

    # Step 3: Further improve resolution at the end (inner edge)
    r_further = (r_array[-1] + r_array[-5]) - np.logspace(
        np.log10(r_array[-1]), np.log10(r_array[-5]), int(2e4)
    )
    r_array = np.insert(r_array[:-5], len(r_array[:-5]), r_further)

    # Clean the grid to remove near-duplicates from np.insert operations
    return _clean_radius_grid(r_array)


def _create_adaptive_radius_grid(
    R1: float,
    r2Prime: float,
    initial_conditions: tuple,
    params,
    Pb: float,
    coarse_points: int = 2000,
    refinement_points: int = 500,
    gradient_threshold_percentile: float = 85
) -> tuple:
    """
    Create radius grid with adaptive refinement around shock fronts.

    Strategy:
    1. Solve ODE on coarse grid using solve_ivp with dense_output
    2. Identify shock regions (large |dT/dr|)
    3. Build refined grid concentrated around those regions
    4. Evaluate dense output on refined grid

    Parameters
    ----------
    R1 : float
        Inner bubble radius
    r2Prime : float
        Outer bubble radius (slightly inside R2)
    initial_conditions : tuple
        (v_r2Prime, T_r2Prime, dTdr_r2Prime) initial values
    params : DescribedDict
        Parameter dictionary
    Pb : float
        Bubble pressure
    coarse_points : int
        Number of points in coarse grid
    refinement_points : int
        Number of points to add around each shock region
    gradient_threshold_percentile : float
        Percentile of |dT/dr| above which to refine

    Returns
    -------
    tuple
        (r_array, psoln) where psoln is the ODE solution array
        Returns (None, None) if adaptive method fails
    """
    from scipy.integrate import solve_ivp

    v_init, T_init, dTdr_init = initial_conditions

    # Ensure scalars
    v_init = float(v_init) if hasattr(v_init, '__len__') else v_init
    T_init = float(T_init) if hasattr(T_init, '__len__') else T_init
    dTdr_init = float(dTdr_init) if hasattr(dTdr_init, '__len__') else dTdr_init
    r2Prime = float(r2Prime) if hasattr(r2Prime, '__len__') else r2Prime

    try:
        # Step 1: Coarse grid (decreasing order, so we integrate "backwards")
        r_start = r2Prime  # Start at outer edge
        r_end = R1 * 1.001  # End just above inner edge (avoid singularity)

        # Use solve_ivp with dense_output for efficient interpolation
        sol = solve_ivp(
            _get_bubble_ODE,
            [r_start, r_end],
            [v_init, T_init, dTdr_init],
            args=(params, Pb),
            method='LSODA',
            dense_output=True,
            max_step=(r_start - r_end) / 100  # Ensure reasonable step sizes
        )

        if not sol.success:
            logger.warning(f"Adaptive grid: solve_ivp failed: {sol.message}")
            return None, None

        # Step 2: Evaluate on coarse grid to find gradients
        r_coarse = np.linspace(r_start, r_end, coarse_points)
        y_coarse = sol.sol(r_coarse)
        T_coarse = y_coarse[1]

        # Check for negative temperatures
        if np.any(T_coarse < 0):
            logger.warning("Adaptive grid: negative temperature in coarse solution")
            return None, None

        # Step 3: Find shock regions (large temperature gradient)
        dTdr_coarse = np.abs(np.gradient(T_coarse, r_coarse))
        threshold = np.percentile(dTdr_coarse, gradient_threshold_percentile)
        shock_mask = dTdr_coarse > threshold

        # Step 4: Build refined grid
        # Start with coarse grid
        r_list = list(r_coarse)

        # Add refinement points around shock regions
        shock_indices = np.where(shock_mask)[0]
        if len(shock_indices) > 0:
            # Group consecutive shock indices into regions
            regions = []
            start_idx = shock_indices[0]
            for i in range(1, len(shock_indices)):
                if shock_indices[i] - shock_indices[i-1] > 5:  # Gap > 5 points = new region
                    regions.append((start_idx, shock_indices[i-1]))
                    start_idx = shock_indices[i]
            regions.append((start_idx, shock_indices[-1]))

            # Add fine points around each region
            for start_idx, end_idx in regions:
                # Expand region slightly
                start_idx = max(0, start_idx - 5)
                end_idx = min(len(r_coarse) - 1, end_idx + 5)

                r_region_start = r_coarse[start_idx]
                r_region_end = r_coarse[end_idx]

                # Add fine grid in this region
                r_fine = np.linspace(r_region_start, r_region_end, refinement_points)
                r_list.extend(r_fine)

        # Also add refinement near the boundaries (where shocks often occur)
        # Near outer boundary
        r_outer_fine = np.linspace(r_start, r_coarse[min(50, len(r_coarse)-1)], refinement_points)
        r_list.extend(r_outer_fine)

        # Near inner boundary
        r_inner_fine = np.linspace(r_coarse[max(0, len(r_coarse)-50)], r_end, refinement_points)
        r_list.extend(r_inner_fine)

        # Sort and remove duplicates (want decreasing order)
        r_array = np.unique(r_list)[::-1]
        r_array = _clean_radius_grid(r_array)

        # Step 5: Evaluate dense output on refined grid
        y_refined = sol.sol(r_array)

        # Check solution quality
        T_refined = y_refined[1]
        if np.any(T_refined < 0) or np.any(np.isnan(T_refined)):
            logger.warning("Adaptive grid: invalid temperature in refined solution")
            return None, None

        # Return in format compatible with odeint output
        psoln = np.column_stack([y_refined[0], y_refined[1], y_refined[2]])

        logger.info(f"Adaptive grid: {len(r_array)} points (vs ~60k legacy)")
        return r_array, psoln

    except Exception as e:
        logger.warning(f"Adaptive grid failed: {e}")
        return None, None


def _solve_bubble_ode_with_ivp(
    r_array: np.ndarray,
    initial_conditions: tuple,
    params,
    Pb: float,
    method: str = 'LSODA'
) -> np.ndarray:
    """
    Solve bubble ODE using solve_ivp instead of odeint.

    This is an alternative solver that may have better behavior for
    some parameter regimes. Kept for future use and experimentation.

    Parameters
    ----------
    r_array : np.ndarray
        Radius array (evaluation points, decreasing order)
    initial_conditions : tuple
        (v_init, T_init, dTdr_init)
    params : DescribedDict
        Parameter dictionary
    Pb : float
        Bubble pressure
    method : str
        Integration method ('LSODA', 'RK45', 'BDF', etc.)

    Returns
    -------
    np.ndarray
        Solution array with shape (len(r_array), 3) for [v, T, dTdr]
        Returns None if solve fails
    """
    from scipy.integrate import solve_ivp

    v_init, T_init, dTdr_init = initial_conditions

    # Ensure scalars
    v_init = float(v_init) if hasattr(v_init, '__len__') else v_init
    T_init = float(T_init) if hasattr(T_init, '__len__') else T_init
    dTdr_init = float(dTdr_init) if hasattr(dTdr_init, '__len__') else dTdr_init

    try:
        sol = solve_ivp(
            _get_bubble_ODE,
            [r_array[0], r_array[-1]],
            [v_init, T_init, dTdr_init],
            args=(params, Pb),
            method=method,
            t_eval=r_array,
            dense_output=False
        )

        if not sol.success:
            logger.warning(f"solve_ivp failed: {sol.message}")
            return None

        # Return in same format as odeint
        return np.column_stack([sol.y[0], sol.y[1], sol.y[2]])

    except Exception as e:
        logger.warning(f"solve_ivp error: {e}")
        return None


@dataclass
class BubbleProperties:
    """
    Dataclass containing all bubble properties.

    This can be used with updateDict(params, bubble_properties) to
    update the params dictionary after bubble calculation completes.
    """
    # Core bubble properties
    bubble_LTotal: float  # Total luminosity loss
    bubble_T_r_Tb: float  # Temperature at bubble_r_Tb
    bubble_Tavg: float  # Average temperature
    bubble_mass: float  # Bubble mass

    # Luminosity components
    bubble_L1Bubble: float  # Luminosity in bubble region (CIE)
    bubble_L2Conduction: float  # Luminosity in conduction zone (non-CIE)
    bubble_L3Intermediate: float  # Luminosity in intermediate region

    # Bubble structure arrays
    bubble_v_arr: np.ndarray  # Velocity array
    bubble_T_arr: np.ndarray  # Temperature array
    bubble_dTdr_arr: np.ndarray  # Temperature gradient array
    bubble_r_arr: np.ndarray  # Radius array
    bubble_n_arr: np.ndarray  # Density array

    # Mass flux
    bubble_dMdt: float  # Mass flux from shell into hot region

    # Derived quantities
    R1: float  # Inner bubble radius
    Pb: float  # Bubble pressure
    bubble_r_Tb: float  # Radius at T_goal


def get_bubbleproperties_pure(params) -> BubbleProperties:
    """
    Calculate bubble properties and return as a dataclass.

    This is the pure version that does NOT mutate params.
    All calculated values are returned in a BubbleProperties dataclass.

    Parameters
    ----------
    params : DescribedDict
        Parameter dictionary (read-only access)

    Returns
    -------
    BubbleProperties
        Dataclass containing all bubble properties
    """
    logger.info('Entering get_bubbleproperties_pure')

    # =============================================================================
    # Step 1: Get necessary parameters
    # =============================================================================

    # Inner bubble radius
    R1 = scipy.optimize.brentq(
        get_bubbleParams.get_r1,
        1e-3 * params['R2'].value, params['R2'].value,
        args=([params['Lmech_total'], params['Eb'],
               params['v_mech_total'], params['R2']])
    )

    # Bubble pressure
    Pb = get_bubbleParams.bubble_E2P(
        params['Eb'].value,
        params['R2'].value,
        R1,
        params['gamma_adia'].value
    )

    # =============================================================================
    # Step 2: Calculate dMdt
    # The mass flux from the shell back into the hot region (b, hot stellar wind)
    # if it isn't yet computed, set it via estimation from Equation 33 in Weaver+77.
    # =============================================================================

    # Get initial dMdt guess
    bubble_dMdt = params['bubble_dMdt'].value
    if np.isnan(bubble_dMdt):
        # if no guesses, begin from formula
        # if value already exist, use previous as current guess.
        bubble_dMdt = _get_init_dMdt(params, Pb)
        logger.info(f"Initial dMdt guess: {bubble_dMdt}")

    # Calculate r_Tb, i.e., radius at which we evaluate bubble temperature
    xi_Tb = params['bubble_xi_Tb'].value
    # this is relative to bubble thickness.
    bubble_r_Tb = R1 + xi_Tb * (params['R2'].value - R1)
    # sanity check: r_Tb cannot be smaller than the inner bubble radius R1
    assert bubble_r_Tb > R1, f"r_Tb ({bubble_r_Tb}) < R1 ({R1})"

    # Solve for dMdt using boundary condition
    # Create a wrapper that uses local Pb instead of params['Pb']
    def velocity_residuals_wrapper(dMdt_guess):
        return _get_velocity_residuals(dMdt_guess, params, Pb, R1)

    bubble_dMdt = scipy.optimize.fsolve(
            velocity_residuals_wrapper,
            bubble_dMdt,
            xtol=1e-4,
            factor=50,
            epsfcn=1e-4
        )[0]

    # =============================================================================
    # Step 3: Calculate bubble structure arrays
    # We note here that T_array, the temperature structure, is in increasing order.
    # This means r_array, the radius, is in DECREASING order.
    # =============================================================================

    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        bubble_dMdt, params, Pb, R1
    )

    # Create radius array and solve ODE
    # Strategy: Try adaptive refinement first (fewer points, concentrated at shocks)
    # Fall back to legacy method (60k points) if adaptive fails
    initial_conditions = (v_r2Prime, T_r2Prime, dTdr_r2Prime)

    # Try adaptive refinement first
    r_array, psoln = _create_adaptive_radius_grid(
        R1, r2Prime, initial_conditions, params, Pb
    )

    if r_array is None:
        # Adaptive failed, fall back to legacy method with grid cleaning
        logger.info("Adaptive grid failed, using legacy method with cleaned grid")
        r_array = _create_legacy_radius_grid(R1, r2Prime)

        # Solve ODE with cleaned legacy grid
        psoln = scipy.integrate.odeint(
            _get_bubble_ODE,
            [v_r2Prime, T_r2Prime, dTdr_r2Prime],
            r_array,
            args=(params, Pb),
            tfirst=True
        )

    v_array = psoln[:, 0]
    T_array = psoln[:, 1]
    dTdr_array = psoln[:, 2]
    n_array = Pb / (2 * params['k_B'].value * T_array)

    logger.info(f'Bubble structure: r=[{r_array[0]:.4f}, {r_array[-1]:.4f}], '
                f'T=[{T_array[0]:.2e}, {T_array[-1]:.2e}]'
                f'n=[{n_array[0]:.2e}, {n_array[-1]:.2e}]')

    # =============================================================================
    # Step 4: Calculate cooling losses
    # The bubble will have these regions:
    #   1. Low resolution (bubble) region. This is the CIE region, where T > 10**5.5 K.
    #   2. High resolution (conduction zone) region. This is the non-CIE region.
    #   3. Intermediate region. This is between 1e4 and T[index_cooling_switch].
    # -----
    # Goal: calculate power-loss (luminosity) in these regions due to cooling. To do this
    #       for each zone we calculate T, dTdr and n. 
    # Remember r is monotonically decreasing, so temperature increases!
    # 
    # 
    # Two things to make sure in this section:
    #   1. All cooling calculations take in au values, but the inner operations and outputs are cgs.
    #      The exception is get_dudt(), which takes in au and returns in au.
    #   2. Dobule check units when using interpolation functions; some of them also only take log10 or 10^.
    # 
    # =============================================================================

    # Temperature at which any lower will have no cooling
    _coolingswitch = 1e4
    # Temperature of switching between CIE and non-CIE. Temperatures higher than 
    # _CIEswitch results in switching on CIE.
    _CIEswitch = 10**5.5

    index_CIE_switch = operations.find_nearest_higher(T_array, _CIEswitch)
    index_cooling_switch = operations.find_nearest_higher(T_array, _coolingswitch)

    if any(T_array < 0):
        logger.error('Negative temperature detected')
        sys.exit('Negative temperature in bubble structure')

    # Process CIE/non-CIE transition
    if index_cooling_switch != index_CIE_switch:
        # extra points to add
        _xtra = 20
        # array sliced from beginning until somewhere after _CIEswitch
        r_interp = r_array[:index_CIE_switch + _xtra]
        # interpolation function for T and dTdr.
        fdTdr_interp = interp1d(r_interp, dTdr_array[:index_CIE_switch + _xtra], kind='linear')
        # subtract so that it is zero at _CIEswitch
        fT_interp = interp1d(r_interp, T_array[:index_CIE_switch + _xtra] - _CIEswitch, kind='cubic')
        fv_interp = interp1d(r_interp, v_array[:index_CIE_switch + _xtra], kind='linear')

        # calculate quantities
        r_CIEswitch = scipy.optimize.brentq(fT_interp, np.min(r_interp), np.max(r_interp), xtol=1e-8)
        n_CIEswitch = Pb / (2 * params['k_B'].value * _CIEswitch)
        dTdr_CIEswitch = fdTdr_interp(r_CIEswitch)
        v_CIEswitch = fv_interp(r_CIEswitch)

        # insert into array
        T_array = np.insert(T_array, index_CIE_switch, _CIEswitch)
        r_array = np.insert(r_array, index_CIE_switch, r_CIEswitch)
        n_array = np.insert(n_array, index_CIE_switch, n_CIEswitch)
        dTdr_array = np.insert(dTdr_array, index_CIE_switch, dTdr_CIEswitch)
        v_array = np.insert(v_array, index_CIE_switch, v_CIEswitch)

    #---------------- 1. Bubble. Low resolution region, T > 10**5.5 K. CIE is used. 
    T_bubble = T_array[index_CIE_switch:]
    r_bubble = r_array[index_CIE_switch:]
    n_bubble = n_array[index_CIE_switch:]
    dTdr_bubble = dTdr_array[index_CIE_switch:]
    # import values from two cooling curves
    cooling_CIE = params['cStruc_cooling_CIE_interpolation'].value
    # cooling rate [au]
    Lambda_bubble = 10**(cooling_CIE(np.log10(T_bubble))) * cvt.Lambda_cgs2au

    integrand_bubble = n_bubble**2 * Lambda_bubble * 4 * np.pi * r_bubble**2
    # calculate power loss due to cooling
    L_bubble = np.abs(_trapezoid(integrand_bubble, x=r_bubble))
    # intermediate result for calculation of average temperature [K pc3]
    Tavg_bubble = np.abs(_trapezoid(r_bubble**2 * T_bubble, x=r_bubble))

    #---------------- 2. Conduction zone. High resolution region, 10**4 < T < 10**5.5 K. 
    L_conduction = 0.0
    Tavg_conduction = 0.0
    dTdR_coolingswitch = dTdr_bubble[0]

    if index_cooling_switch != index_CIE_switch:
        # if this zone is not well resolved, solve ODE again with high resolution (IMPROVE BY ALWAYS INTERPOLATING)
        if index_CIE_switch - index_cooling_switch < 100:
             # This is the original array that is too short
            lowres_r = r_array[:index_CIE_switch + 1]
            _highres = 1e2
             # how many intervales in high-res version? [::-1] included because r is reversed.
            r_conduction = np.arange(
                min(lowres_r), max(lowres_r),
                (max(lowres_r) - min(lowres_r)) / _highres
            )[::-1]
            # rerun structure with greater precision
            psoln_cond = scipy.integrate.odeint(
                _get_bubble_ODE,
                [v_array[index_cooling_switch], T_array[index_cooling_switch], dTdr_array[index_cooling_switch]],
                r_conduction,
                args=(params, Pb),
                tfirst=True
            )

            # Here, something needs to be done. Because of the precision of the solver, 
            # it may return temperature with values > 10**5.5K eventhough that was the maximum limit (i.e., 10**5.500001).
            # This will crash the interpolator. To fix this, we simple shave away values in the array where T > 10**5.5, 
            # and concatenate to the low-rez limit. 
            
            # Actually, the final value may not be required; the value is already included
            # in the first zone, so we don't have to worry about them here.
            
            v_cond = psoln_cond[:, 0]
            T_cond = psoln_cond[:, 1]
            dTdr_cond = psoln_cond[:, 2]

            mask = T_cond < _CIEswitch
            r_conduction = r_conduction[mask]
            T_cond = T_cond[mask]
            dTdr_cond = dTdr_cond[mask]
            dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]
        else:
            r_conduction = r_array[:index_CIE_switch + 1]
            T_cond = T_array[:index_CIE_switch + 1]
            dTdr_cond = dTdr_array[:index_CIE_switch + 1]
            dTdR_coolingswitch = dTdr_cond[0]
        # calculate array [au]
        n_cond = Pb / (2 * params['k_B'].value * T_cond)
        phi_cond = params['Qi'].value / (4 * np.pi * r_conduction**2)
        # import values from two cooling curves
        cooling_nonCIE = params['cStruc_cooling_nonCIE'].value
        heating_nonCIE = params['cStruc_heating_nonCIE'].value
        # cooling rate [cgs]
        cool_cond = 10 ** cooling_nonCIE.interp(
            np.transpose(np.log10([n_cond / cvt.ndens_cgs2au, T_cond, phi_cond / cvt.phi_cgs2au]))
        )
        heat_cond = 10 ** heating_nonCIE.interp(
            np.transpose(np.log10([n_cond / cvt.ndens_cgs2au, T_cond, phi_cond / cvt.phi_cgs2au]))
        )
        # net cooling rate [au]
        dudt_cond = (heat_cond - cool_cond) * cvt.dudt_cgs2au
        # integrand [au]
        integrand_cond = dudt_cond * 4 * np.pi * r_conduction**2
        # calculate power loss due to cooling [au]
        L_conduction = np.abs(_trapezoid(integrand_cond, x=r_conduction))
        # intermediate result for calculation of average temperature
        Tavg_conduction = np.abs(_trapezoid(r_conduction**2 * T_cond, x=r_conduction))

    #---------------- 3. Region between 1e4 K and T_array[index_cooling_switch]
    # If R2_prime is very close to R2 (i.e., where T ~ 1e4K), then this region is tiny (or non-existent)
    R2_coolingswitch = (_coolingswitch - T_array[index_cooling_switch]) / dTdR_coolingswitch + r_array[index_cooling_switch]
    # interpolate between R2_prime and R2_1e4, important because the cooling function varies a lot between 1e4 and 1e5K (R2_prime is above 1e4)
    fT_interp_interm = interp1d(
        np.array([r_array[index_cooling_switch], R2_coolingswitch]),
        np.array([T_array[index_cooling_switch], _coolingswitch]),
        kind='linear'
    )

    r_interm = np.linspace(r_array[index_cooling_switch], R2_coolingswitch, num=1000, endpoint=True)
    T_interm = fT_interp_interm(r_interm)
    n_interm = Pb / (2 * params['k_B'].value * T_interm)
    phi_interm = params['Qi'].value / (4 * np.pi * r_interm**2)
    # get cooling, taking into account for both CIE and non-CIE regimes
    L_intermediate = 0.0
    for regime in ['non-CIE', 'CIE']:
        mask = T_interm < _CIEswitch if regime == 'non-CIE' else T_interm >= _CIEswitch
        if not np.any(mask):
            continue

        if regime == 'non-CIE':
            cooling_nonCIE = params['cStruc_cooling_nonCIE'].value
            heating_nonCIE = params['cStruc_heating_nonCIE'].value
            cool_int = 10 ** cooling_nonCIE.interp(
                np.transpose(np.log10([n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask], phi_interm[mask] / cvt.phi_cgs2au]))
            )
            heat_int = 10 ** heating_nonCIE.interp(
                np.transpose(np.log10([n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask], phi_interm[mask] / cvt.phi_cgs2au]))
            )
            dudt_int = (heat_int - cool_int) * cvt.dudt_cgs2au
            integrand_int = dudt_int * 4 * np.pi * r_interm[mask]**2
        else:
            Lambda_int = 10**(cooling_CIE(np.log10(T_interm[mask]))) * cvt.Lambda_cgs2au
            integrand_int = n_interm[mask]**2 * Lambda_int * 4 * np.pi * r_interm[mask]**2
        # calculate power loss due to cooling
        L_intermediate += np.abs(_trapezoid(integrand_int, x=r_interm[mask]))

    Tavg_intermediate = np.abs(_trapezoid(r_interm**2 * T_interm, x=r_interm))

    # Total luminosity
    L_total = L_bubble + L_conduction + L_intermediate

    # Average temperature
    if index_cooling_switch != index_CIE_switch:
        Tavg = 3 * (
            Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +
            Tavg_conduction / (r_conduction[0]**3 - r_conduction[-1]**3) +
            Tavg_intermediate / (r_interm[0]**3 - r_interm[-1]**3)
        )
    else:
        Tavg = 3 * (
            Tavg_bubble / (r_bubble[0]**3 - r_bubble[-1]**3) +
            Tavg_intermediate / (r_interm[0]**3 - r_interm[-1]**3)
        )

    # Temperature at r_Tb
    # If rgoal is smaller than the radius of cooling threshold, i.e., larger than the index,
    if bubble_r_Tb > r_array[index_cooling_switch]:# looking for the smallest value in r_cz
        T_rgoal = fT_interp_interm(bubble_r_Tb)
    elif bubble_r_Tb > r_array[index_CIE_switch]:
        if index_cooling_switch != index_CIE_switch: # looking for the largest value in r_cz
            idx = operations.find_nearest(r_conduction, bubble_r_Tb)
            T_rgoal = T_cond[idx] + dTdr_cond[idx] * (bubble_r_Tb - r_conduction[idx])
        else:
            T_rgoal = T_bubble[0]
    # otherwise, interpolate.
    else:
        idx = operations.find_nearest(r_bubble, bubble_r_Tb)
        T_rgoal = T_bubble[idx] + dTdr_bubble[idx] * (bubble_r_Tb - r_bubble[idx])

    # =============================================================================
    # Step 5: Calculate mass
    # =============================================================================

    m_cumulative, _, _ = _get_mass_and_grav(n_array, r_array, params)
    mBubble = m_cumulative[-1]

    # Return dataclass
    return BubbleProperties(
        bubble_LTotal=L_total,
        bubble_T_r_Tb=T_rgoal,
        bubble_Tavg=Tavg,
        bubble_mass=mBubble,
        bubble_L1Bubble=L_bubble,
        bubble_L2Conduction=L_conduction,
        bubble_L3Intermediate=L_intermediate,
        bubble_v_arr=v_array,
        bubble_T_arr=T_array,
        bubble_dTdr_arr=dTdr_array,
        bubble_r_arr=r_array,
        bubble_n_arr=n_array,
        bubble_dMdt=bubble_dMdt,
        R1=R1,
        Pb=Pb,
        bubble_r_Tb=bubble_r_Tb,
    )


def _get_init_dMdt(params, Pb: float) -> float:
    """Initial guess for dMdt (Equation 33 in Weaver+77)."""
    dMdt_factor = 1.646
    return (12 / 75 * dMdt_factor**(5/2) * 4 * np.pi * params['R2']**3 / params['t_now']
            * params['mu_atom'] / params['k_B']
            * (params['t_now'] * params['C_thermal'] / params['R2']**2)**(2/7)
            * Pb**(5/7))


def _get_velocity_residuals(dMdt_init, params, Pb: float, R1: float) -> float:
    """Calculate velocity residual for dMdt solver."""
    # =============================================================================
    # Get initial bubble values for integration
    # =============================================================================
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = _get_bubble_ODE_initial_conditions(
        dMdt_init, params, Pb, R1
    )

    # Ensure scalar values (in case arrays are returned)
    r2Prime_val = float(r2Prime[0]) if hasattr(r2Prime, '__len__') else float(r2Prime)
    T_r2Prime_val = float(T_r2Prime[0]) if hasattr(T_r2Prime, '__len__') else float(T_r2Prime)
    dTdr_r2Prime_val = float(dTdr_r2Prime[0]) if hasattr(dTdr_r2Prime, '__len__') else float(dTdr_r2Prime)
    v_r2Prime_val = float(v_r2Prime[0]) if hasattr(v_r2Prime, '__len__') else float(v_r2Prime)

    # =============================================================================
    # radius array at which bubble structure is being evaluated.
    # =============================================================================
    # Use legacy grid with cleaning (simpler and faster for residual evaluation)
    # This function is called many times during fsolve, so we avoid adaptive overhead
    r_array = _create_legacy_radius_grid(R1, r2Prime_val)

    # Solve ODE with cleaned grid (no LSODA interpolation warnings)
    psoln = scipy.integrate.odeint(
        _get_bubble_ODE,
        [v_r2Prime_val, T_r2Prime_val, dTdr_r2Prime_val],
        r_array,
        args=(params, Pb),
        tfirst=True
    )

    v_array = psoln[:, 0]
    T_array = psoln[:, 1]

    residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)

    min_T = np.min(T_array)
    if min_T < 3e4:
        logger.debug(f'Rejected. min T: {min_T}')
        return residual * (3e4 / (min_T + 1e-1))**2

    if np.isnan(min_T):
        logger.debug('Rejected. nan temperature')
        return -1e3

    if not operations.monotonic(T_array):
        logger.debug('Temperature not monotonic')
        return 1e2

    return residual


def _get_bubble_ODE_initial_conditions(dMdt, params, Pb: float, R1: float):
    """Get initial conditions for bubble ODE (Eq 44 in Weaver+77)."""
    T_init = 3e4

    constant = (25/4 * params['k_B'] / params['mu_ion'] / params['C_thermal'])
    dR2 = T_init**(5/2) / (constant * dMdt / (4 * np.pi * params['R2'].value**2))

    T = (constant * dMdt * dR2 / (4 * np.pi * params['R2'].value**2))**(2/5)
    v = (params['cool_alpha'].value * params['R2'].value / params['t_now'].value
         - dMdt / (4 * np.pi * params['R2'].value**2)
         * params['k_B'].value * T / params['mu_ion'].value / Pb)
    dTdr = -2/5 * T / dR2
    r2_prime = params['R2'].value - dR2

    return r2_prime, T, dTdr, v


def _get_bubble_ODE(r_arr, initial_ODEs, params, Pb: float):
    """Bubble structure ODE (Equations 42-43 in Weaver+77)."""
    v, T, dTdr = initial_ODEs

    if np.abs(T - 0) < 1e-5:
        logger.error('T is zero in bubble ODE')
        sys.exit()

    ndens = Pb / (2 * params['k_B'].value * T)
    phi = params['Qi'].value / (4 * np.pi * r_arr**2)

    dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)

    v_term = params['cool_alpha'].value * r_arr / params['t_now'].value

    dTdrr = (Pb / (params['C_thermal'].value * T**(5/2)) * (
        (params['cool_beta'].value + 2.5 * params['cool_delta'].value) / params['t_now'].value
        + 2.5 * (v - v_term) * dTdr / T - dudt / Pb
    ) - 2.5 * dTdr**2 / T - 2 * dTdr / r_arr)

    dvdr = ((params['cool_beta'].value + params['cool_delta'].value) / params['t_now'].value
            + (v - v_term) * dTdr / T - 2 * v / r_arr)

    return [dvdr, dTdr, dTdrr]


def _get_mass_and_grav(n, r, params):
    """Calculate cumulative mass and gravitational potential."""
    # Flip arrays to be monotonically increasing
    r_new = r[::-1]
    rho_new = n[::-1] * params['mu_ion'].value # Mass density [Msun/pc³]

    # Calculate cumulative mass using O(n) cumulative integration
    # instead of O(n²) loop with simps
    # M(r) = ∫[0 to r] 4πr'² ρ(r') dr'
    integrand = rho_new * r_new**2
    m_cumulative = 4 * np.pi * scipy.integrate.cumulative_trapezoid(
        integrand, x=r_new, initial=0
    )
    # Gravitational potential [pc²/Myr²]
    grav_phi = -4 * np.pi * params['G'].value * scipy.integrate.simpson(
        r_new * rho_new, x=r_new
    )
    # Gravitational force per unit mass [pc/Myr²]
    # Add small number to avoid division by zero at r=0
    grav_force_m = params['G'].value * m_cumulative / (r_new**2 + 1e-10)

    return m_cumulative, grav_phi, grav_force_m
