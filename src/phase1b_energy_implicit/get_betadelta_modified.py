#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified beta-delta solver with pure functions.

This module provides pure function implementations for finding optimal
beta and delta values without mutating the params dictionary.

Key improvements over get_betadelta.py:
1. Pure functions that return results instead of mutating params
2. scipy.optimize.minimize instead of brute-force grid search
3. Reuses get_bubbleproperties_pure from bubble_luminosity_modified
4. Proper logging instead of print statements

@author: TRINITY Team (refactored for pure functions)
"""

import numpy as np
import scipy.optimize
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import src.bubble_structure.get_bubbleParams as get_bubbleParams
from src.bubble_structure.bubble_luminosity_modified import (
    get_bubbleproperties_pure,
    BubbleProperties,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Bounds for beta and delta
BETA_MIN = 0.0
BETA_MAX = 1.0
DELTA_MIN = -1.0
DELTA_MAX = 0.0

# Convergence thresholds
RESIDUAL_THRESHOLD = 1e-4
MAX_ITERATIONS = 50


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class BetaDeltaResult:
    """Container for beta-delta solver results."""
    beta: float
    delta: float
    Edot_residual: float
    T_residual: float
    total_residual: float
    converged: bool
    iterations: int
    bubble_properties: Optional[BubbleProperties] = None


# =============================================================================
# Pure Helper Functions
# =============================================================================

def beta2Edot_pure(
    beta: float,
    Pb: float,
    t_now: float,
    R1: float,
    R2: float,
    v2: float,
    Eb: float,
    pdot_total: float,
    pdotdot_total: float,
) -> float:
    """
    Convert beta to dE/dt (pure function version).

    See pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Parameters
    ----------
    beta : float
        -(t/Pb)*(dPb/dt), cooling parameter
    Pb : float
        Bubble pressure [cgs]
    t_now : float
        Current time [Myr]
    R1 : float
        Inner bubble radius [pc]
    R2 : float
        Outer bubble radius [pc]
    v2 : float
        Outer bubble velocity [pc/Myr]
    Eb : float
        Bubble energy [erg]
    pdot_total : float
        Momentum injection rate
    pdotdot_total : float
        Second derivative of momentum injection

    Returns
    -------
    Edot : float
        Time derivative of bubble energy [erg/Myr]
    """
    # dp/dt from beta
    press_dot = -Pb * beta / t_now

    # Define terms
    a = np.sqrt(pdot_total / 2)
    b = 1.5 * a**2 * R1
    d = R2**3 - R1**3
    adot = 0.25 * pdotdot_total / a if a > 0 else 0.0

    e = b / (b + Eb) if (b + Eb) > 0 else 0.0

    # Main equation (Rahner thesis eq A12)
    numerator = (
        2 * np.pi * press_dot * d**2
        + 3 * Eb * v2 * R2**2 * (1 - e)
        - 3 * (adot / a) * R1**3 * Eb**2 / (Eb + b) if a > 0 else 0.0
    )
    denominator = d * (1 - e)

    if abs(denominator) < 1e-300:
        return 0.0

    return numerator / denominator


def delta2dTdt_pure(t: float, T: float, delta: float) -> float:
    """
    Convert delta to dT/dt (pure function version).

    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Parameters
    ----------
    t : float
        Current time [Myr]
    T : float
        Temperature at xi = r/R2 [K]
    delta : float
        Cooling parameter

    Returns
    -------
    dTdt : float
        Time derivative of temperature [K/Myr]
    """
    if t <= 0:
        return 0.0
    return (T / t) * delta


def compute_R1_Pb(
    R2: float,
    Eb: float,
    Lmech_total: float,
    v_mech_total: float,
    gamma_adia: float,
) -> Tuple[float, float]:
    """
    Compute inner radius R1 and bubble pressure Pb.

    Parameters
    ----------
    R2 : float
        Outer bubble radius [pc]
    Eb : float
        Bubble energy [erg]
    Lmech_total : float
        Total mechanical luminosity
    v_mech_total : float
        Mechanical velocity
    gamma_adia : float
        Adiabatic index

    Returns
    -------
    R1 : float
        Inner bubble radius [pc]
    Pb : float
        Bubble pressure [cgs]
    """
    try:
        R1 = scipy.optimize.brentq(
            get_bubbleParams.get_r1,
            1e-3 * R2,
            R2,
            args=([Lmech_total, Eb, v_mech_total, R2])
        )
    except (ValueError, RuntimeError):
        # Fallback if root finding fails
        R1 = 0.01 * R2
        logger.warning(f"R1 root finding failed, using fallback R1 = {R1:.4e}")

    Pb = get_bubbleParams.bubble_E2P(Eb, R2, R1, gamma_adia)

    return R1, Pb


# =============================================================================
# Pure Residual Calculation
# =============================================================================

def get_residual_pure(
    beta: float,
    delta: float,
    params,
    return_bubble_props: bool = False,
) -> Tuple[float, float, Optional[BubbleProperties]]:
    """
    Calculate residuals for beta and delta without mutating params.

    The residuals measure:
    1. Edot_residual: difference between Edot from beta and Edot from energy balance
    2. T_residual: difference between bubble temperature and target temperature T0

    Parameters
    ----------
    beta : float
        Beta cooling parameter
    delta : float
        Delta cooling parameter
    params : dict-like
        Parameter dictionary (read-only)
    return_bubble_props : bool
        If True, also return the BubbleProperties

    Returns
    -------
    Edot_residual : float
        Relative residual in energy derivative
    T_residual : float
        Relative residual in temperature
    bubble_props : BubbleProperties or None
        Bubble properties if return_bubble_props=True
    """
    import copy

    # Create a temporary copy for bubble calculation
    # This is necessary because get_bubbleproperties_pure still reads from params
    temp_params = copy.deepcopy(params)
    temp_params['cool_beta'].value = beta
    temp_params['cool_delta'].value = delta

    # Calculate bubble properties
    try:
        bubble_props = get_bubbleproperties_pure(temp_params)
    except Exception as e:
        logger.warning(f"Bubble properties calculation failed: {e}")
        return 100.0, 100.0, None

    # Extract needed values
    R2 = temp_params['R2'].value
    v2 = temp_params['v2'].value
    Eb = temp_params['Eb'].value
    T0 = temp_params['T0'].value
    t_now = temp_params['t_now'].value
    gamma_adia = temp_params['gamma_adia'].value
    Lmech_total = temp_params['Lmech_total'].value
    v_mech_total = temp_params['v_mech_total'].value
    pdot_total = temp_params['pdot_total'].value
    pdotdot_total = temp_params['pdotdot_total'].value

    # Compute R1 and Pb
    R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)

    # =============================================================================
    # Part 1: Calculate Edot residual for beta
    # =============================================================================

    # Method 1: Edot from beta
    Edot_from_beta = beta2Edot_pure(
        beta, Pb, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total
    )

    # Method 2: Edot from energy balance (gain - loss - work)
    L_gain = Lmech_total
    L_loss = bubble_props.bubble_LTotal
    # Add leak if available
    bubble_Leak = getattr(temp_params.get('bubble_Leak', None), 'value', 0.0)
    if bubble_Leak is None:
        bubble_Leak = 0.0
    L_loss += bubble_Leak

    Edot_from_balance = L_gain - L_loss - 4 * np.pi * R2**2 * v2 * Pb

    # Relative residual
    if abs(Edot_from_beta) > 1e-300:
        Edot_residual = (Edot_from_beta - Edot_from_balance) / Edot_from_beta
    else:
        Edot_residual = Edot_from_balance if abs(Edot_from_balance) > 0 else 0.0

    # =============================================================================
    # Part 2: Calculate T residual for delta
    # =============================================================================

    # Temperature at measurement point vs target temperature
    T_bubble = bubble_props.bubble_T_r_Tb

    if abs(T0) > 1e-300:
        T_residual = (T_bubble - T0) / T0
    else:
        T_residual = T_bubble if abs(T_bubble) > 0 else 0.0

    if return_bubble_props:
        return Edot_residual, T_residual, bubble_props
    return Edot_residual, T_residual, None


# =============================================================================
# Main Solver
# =============================================================================

def solve_betadelta_pure(
    beta_guess: float,
    delta_guess: float,
    params,
) -> BetaDeltaResult:
    """
    Solve for optimal beta and delta using scipy.optimize.minimize.

    This replaces the brute-force grid search with an efficient optimizer.

    Parameters
    ----------
    beta_guess : float
        Initial guess for beta
    delta_guess : float
        Initial guess for delta
    params : dict-like
        Parameter dictionary (not mutated)

    Returns
    -------
    result : BetaDeltaResult
        Container with optimal beta, delta, residuals, and convergence info
    """
    # First check if current guess is already good enough
    Edot_res, T_res, bubble_props = get_residual_pure(
        beta_guess, delta_guess, params, return_bubble_props=True
    )
    total_res = Edot_res**2 + T_res**2

    if total_res < RESIDUAL_THRESHOLD:
        logger.debug(f"Initial guess already converged: residual={total_res:.2e}")
        return BetaDeltaResult(
            beta=beta_guess,
            delta=delta_guess,
            Edot_residual=Edot_res,
            T_residual=T_res,
            total_residual=total_res,
            converged=True,
            iterations=0,
            bubble_properties=bubble_props,
        )

    # Define objective function for optimizer
    def objective(x):
        beta, delta = x
        # Enforce bounds
        beta = np.clip(beta, BETA_MIN, BETA_MAX)
        delta = np.clip(delta, DELTA_MIN, DELTA_MAX)

        try:
            Edot_res, T_res, _ = get_residual_pure(beta, delta, params)
            return Edot_res**2 + T_res**2
        except Exception as e:
            logger.warning(f"Residual calculation failed: {e}")
            return 1e10

    # Use scipy.optimize.minimize with bounds
    bounds = [(BETA_MIN, BETA_MAX), (DELTA_MIN, DELTA_MAX)]
    x0 = np.array([beta_guess, delta_guess])

    try:
        result = scipy.optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': MAX_ITERATIONS,
                'ftol': 1e-8,
                'gtol': 1e-6,
            }
        )

        beta_opt, delta_opt = result.x
        iterations = result.nit

    except Exception as e:
        logger.warning(f"Optimizer failed: {e}, using initial guess")
        beta_opt, delta_opt = beta_guess, delta_guess
        iterations = 0

    # Get final residuals with bubble properties
    Edot_res, T_res, bubble_props = get_residual_pure(
        beta_opt, delta_opt, params, return_bubble_props=True
    )
    total_res = Edot_res**2 + T_res**2

    converged = total_res < RESIDUAL_THRESHOLD

    logger.debug(
        f"Beta-delta solved: beta={beta_opt:.4f}, delta={delta_opt:.4f}, "
        f"residual={total_res:.2e}, converged={converged}, iter={iterations}"
    )

    return BetaDeltaResult(
        beta=beta_opt,
        delta=delta_opt,
        Edot_residual=Edot_res,
        T_residual=T_res,
        total_residual=total_res,
        converged=converged,
        iterations=iterations,
        bubble_properties=bubble_props,
    )


# =============================================================================
# Wrapper for Backward Compatibility
# =============================================================================

def get_beta_delta_wrapper_pure(
    beta_guess: float,
    delta_guess: float,
    params,
) -> Tuple[Tuple[float, float], BetaDeltaResult]:
    """
    Wrapper that matches the interface of the original get_beta_delta_wrapper.

    This allows drop-in replacement while using pure functions internally.

    Parameters
    ----------
    beta_guess : float
        Initial guess for beta
    delta_guess : float
        Initial guess for delta
    params : dict-like
        Parameter dictionary

    Returns
    -------
    (beta, delta) : tuple
        Optimal values
    result : BetaDeltaResult
        Full result object
    """
    result = solve_betadelta_pure(beta_guess, delta_guess, params)
    return (result.beta, result.delta), result
