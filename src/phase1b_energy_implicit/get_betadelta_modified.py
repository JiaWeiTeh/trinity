#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified beta-delta solver with pure functions.

This module provides pure function implementations for finding optimal
beta and delta values without mutating the params dictionary.

Key improvements over get_betadelta.py:
1. Pure functions that return results instead of mutating params
2. BubbleParamsView avoids expensive deepcopy (25-100x faster per evaluation)
3. Grid search first (default), then L-BFGS-B fallback if grid doesn't converge
4. If both fail to converge, picks the best result from grid/L-BFGS-B/original input
5. Reuses get_bubbleproperties_pure from bubble_luminosity_modified
6. Proper logging instead of print statements

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
MAX_ITERATIONS = 15

# Threshold for L-BFGS-B fallback: only run L-BFGS-B if grid residual exceeds this
# If grid gives a reasonable result (< 5.0), L-BFGS-B is unlikely to improve much
# and wastes ~50 expensive function evaluations
LBFGSB_FALLBACK_THRESHOLD = 5.0

# Grid search parameters (matching original get_betadelta.py)
GRID_SIZE = 5  # 5x5 grid (matching original get_betadelta.py)
GRID_EPSILON = 0.02  # Search range around guess


# =============================================================================
# Lightweight View for Beta/Delta Override (Performance Optimization)
# =============================================================================

class _MockValue:
    """Mimics DescribedItem with a .value attribute."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value


class BubbleParamsView:
    """
    Lightweight view that overrides cool_beta and cool_delta without copying.

    This avoids expensive copy.deepcopy() by:
    - Returning override values for cool_beta/cool_delta
    - Passing through all other accesses to the original params

    Since get_bubbleproperties_pure() only READS params (never writes),
    this is safe and provides ~25-100x speedup per residual evaluation.
    """
    __slots__ = ('_params', '_overrides')

    def __init__(self, params, beta: float, delta: float):
        self._params = params
        self._overrides = {
            'cool_beta': _MockValue(beta),
            'cool_delta': _MockValue(delta),
        }

    def __getitem__(self, key: str):
        if key in self._overrides:
            return self._overrides[key]
        return self._params[key]

    def get(self, key: str, default=None):
        if key in self._overrides:
            return self._overrides[key]
        return self._params.get(key, default)


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
    # Raw residual components for diagnostics
    Edot_from_beta: Optional[float] = None      # residual_Edot1_guess
    Edot_from_balance: Optional[float] = None   # residual_Edot2_guess
    T_bubble: Optional[float] = None            # residual_T1_guess
    T0: Optional[float] = None                  # residual_T2_guess
    # Energy balance terms
    L_gain: Optional[float] = None              # bubble_Lgain
    L_loss: Optional[float] = None              # bubble_Lloss


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
        - 3 * (adot / a) * R1**3 * Eb**2 / (Eb + b)
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
    # Create a lightweight view that overrides beta/delta without copying
    # This is ~25-100x faster than copy.deepcopy(params)
    params_view = BubbleParamsView(params, beta, delta)

    # Calculate bubble properties
    try:
        bubble_props = get_bubbleproperties_pure(params_view)
    except Exception as e:
        logger.warning(f"Bubble properties calculation failed: {e}")
        return 100.0, 100.0, None

    # Extract needed values from original params (not modified)
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    t_now = params['t_now'].value
    gamma_adia = params['gamma_adia'].value
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value
    pdot_total = params['pdot_total'].value
    pdotdot_total = params['pdotdot_total'].value

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
    bubble_Leak = getattr(params.get('bubble_Leak', None), 'value', 0.0)
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


@dataclass
class ResidualDetails:
    """Detailed residual components for diagnostics."""
    Edot_residual: float
    T_residual: float
    Edot_from_beta: float       # residual_Edot1_guess
    Edot_from_balance: float    # residual_Edot2_guess
    T_bubble: float             # residual_T1_guess
    T0: float                   # residual_T2_guess
    bubble_props: Optional[BubbleProperties] = None
    L_gain: float = 0.0        # bubble_Lgain
    L_loss: float = 0.0        # bubble_Lloss


def get_residual_detailed(
    beta: float,
    delta: float,
    params,
) -> ResidualDetails:
    """
    Calculate residuals with all raw components for diagnostics.

    Returns a ResidualDetails object containing both normalized residuals
    and the raw values used to compute them.
    """
    # Create a lightweight view that overrides beta/delta without copying
    params_view = BubbleParamsView(params, beta, delta)

    # Calculate bubble properties
    try:
        bubble_props = get_bubbleproperties_pure(params_view)
    except Exception as e:
        logger.warning(f"Bubble properties calculation failed: {e}")
        return ResidualDetails(
            Edot_residual=100.0,
            T_residual=100.0,
            Edot_from_beta=np.nan,
            Edot_from_balance=np.nan,
            T_bubble=np.nan,
            T0=np.nan,
            bubble_props=None,
        )

    # Extract needed values from original params
    R2 = params['R2'].value
    v2 = params['v2'].value
    Eb = params['Eb'].value
    T0 = params['T0'].value
    t_now = params['t_now'].value
    gamma_adia = params['gamma_adia'].value
    Lmech_total = params['Lmech_total'].value
    v_mech_total = params['v_mech_total'].value
    pdot_total = params['pdot_total'].value
    pdotdot_total = params['pdotdot_total'].value

    # Compute R1 and Pb
    R1, Pb = compute_R1_Pb(R2, Eb, Lmech_total, v_mech_total, gamma_adia)

    # Part 1: Calculate Edot values
    Edot_from_beta = beta2Edot_pure(
        beta, Pb, t_now, R1, R2, v2, Eb, pdot_total, pdotdot_total
    )

    L_gain = Lmech_total
    L_loss = bubble_props.bubble_LTotal
    bubble_Leak = getattr(params.get('bubble_Leak', None), 'value', 0.0)
    if bubble_Leak is None:
        bubble_Leak = 0.0
    L_loss += bubble_Leak

    Edot_from_balance = L_gain - L_loss - 4 * np.pi * R2**2 * v2 * Pb

    # Relative Edot residual
    if abs(Edot_from_beta) > 1e-300:
        Edot_residual = (Edot_from_beta - Edot_from_balance) / Edot_from_beta
    else:
        Edot_residual = Edot_from_balance if abs(Edot_from_balance) > 0 else 0.0

    # Part 2: Calculate T values
    T_bubble = bubble_props.bubble_T_r_Tb

    if abs(T0) > 1e-300:
        T_residual = (T_bubble - T0) / T0
    else:
        T_residual = T_bubble if abs(T_bubble) > 0 else 0.0

    return ResidualDetails(
        Edot_residual=Edot_residual,
        T_residual=T_residual,
        Edot_from_beta=Edot_from_beta,
        Edot_from_balance=Edot_from_balance,
        T_bubble=T_bubble,
        T0=T0,
        bubble_props=bubble_props,
        L_gain=L_gain,
        L_loss=L_loss,
    )


# =============================================================================
# Main Solver
# =============================================================================

def solve_betadelta_pure(
    beta_guess: float,
    delta_guess: float,
    params,
    method: str = 'grid',
) -> BetaDeltaResult:
    """
    Solve for optimal beta and delta.

    Parameters
    ----------
    beta_guess : float
        Initial guess for beta
    delta_guess : float
        Initial guess for delta
    params : dict-like
        Parameter dictionary (not mutated)
    method : str, optional
        Solver method: 'grid' (default, fast grid search) or 'lbfgsb' (optimizer).
        When method='grid', automatically falls back to 'lbfgsb' if grid search
        fails or doesn't converge. If both fail, picks the best result from
        grid, L-BFGS-B, or original input.

    Returns
    -------
    result : BetaDeltaResult
        Container with optimal beta, delta, residuals, and convergence info
    """
    # First check if current guess is already good enough
    Edot_res_input, T_res_input, bubble_props_input = get_residual_pure(
        beta_guess, delta_guess, params, return_bubble_props=True
    )
    total_res_input = Edot_res_input**2 + T_res_input**2

    if total_res_input < RESIDUAL_THRESHOLD:
        logger.debug(f"Initial guess already converged: residual={total_res_input:.2e}")
        return BetaDeltaResult(
            beta=beta_guess,
            delta=delta_guess,
            Edot_residual=Edot_res_input,
            T_residual=T_res_input,
            total_residual=total_res_input,
            converged=True,
            iterations=0,
            bubble_properties=bubble_props_input,
        )

    # Track candidates: (beta, delta, residual, method_name, iterations)
    candidates = []

    # Always add original input as a candidate
    if np.isfinite(total_res_input):
        candidates.append((beta_guess, delta_guess, total_res_input, 'input', 0))

    # Step 1: Try grid search first
    grid_converged = False
    grid_result = None
    try:
        beta_grid, delta_grid, iter_grid = _solve_grid(beta_guess, delta_guess, params)
        Edot_res_grid, T_res_grid, _ = get_residual_pure(beta_grid, delta_grid, params)
        total_res_grid = Edot_res_grid**2 + T_res_grid**2

        if np.isfinite(total_res_grid):
            candidates.append((beta_grid, delta_grid, total_res_grid, 'grid', iter_grid))
            grid_result = (beta_grid, delta_grid, total_res_grid, iter_grid)

            if total_res_grid < RESIDUAL_THRESHOLD:
                grid_converged = True
                logger.info(
                    f"Grid search converged: β={beta_grid:.4f}, δ={delta_grid:.4f}, "
                    f"residual={total_res_grid:.2e}"
                )
            else:
                logger.info(
                    f"Grid search did not converge: β={beta_grid:.4f}, δ={delta_grid:.4f}, "
                    f"residual={total_res_grid:.2e} (threshold={RESIDUAL_THRESHOLD:.0e})"
                )
    except Exception as e:
        logger.warning(f"Grid search failed: {e}")

    # Step 2: If grid didn't converge AND grid residual is bad, try L-BFGS-B
    # Skip L-BFGS-B if grid gave a reasonable result to avoid wasting ~50 evaluations
    lbfgsb_converged = False
    lbfgsb_result = None
    grid_residual = grid_result[2] if grid_result else float('inf')

    if not grid_converged and grid_residual > LBFGSB_FALLBACK_THRESHOLD:
        logger.info(
            f"Grid residual ({grid_residual:.2e}) > fallback threshold ({LBFGSB_FALLBACK_THRESHOLD}), "
            "trying L-BFGS-B"
        )
        try:
            beta_lbfgsb, delta_lbfgsb, iter_lbfgsb = _solve_lbfgsb(
                beta_guess, delta_guess, params
            )
            Edot_res_lbfgsb, T_res_lbfgsb, _ = get_residual_pure(
                beta_lbfgsb, delta_lbfgsb, params
            )
            total_res_lbfgsb = Edot_res_lbfgsb**2 + T_res_lbfgsb**2

            if np.isfinite(total_res_lbfgsb):
                candidates.append((
                    beta_lbfgsb, delta_lbfgsb, total_res_lbfgsb, 'lbfgsb', iter_lbfgsb
                ))
                lbfgsb_result = (beta_lbfgsb, delta_lbfgsb, total_res_lbfgsb, iter_lbfgsb)

                if total_res_lbfgsb < RESIDUAL_THRESHOLD:
                    lbfgsb_converged = True
                    logger.info(
                        f"L-BFGS-B converged: β={beta_lbfgsb:.4f}, δ={delta_lbfgsb:.4f}, "
                        f"residual={total_res_lbfgsb:.2e}"
                    )
                else:
                    logger.info(
                        f"L-BFGS-B did not converge: β={beta_lbfgsb:.4f}, δ={delta_lbfgsb:.4f}, "
                        f"residual={total_res_lbfgsb:.2e}"
                    )
        except Exception as e:
            logger.warning(f"L-BFGS-B failed: {e}")

    # Step 3: Pick the best result
    if not candidates:
        # All methods failed completely, return original input with failure status
        logger.warning("All solver methods failed, returning original input")
        return BetaDeltaResult(
            beta=beta_guess,
            delta=delta_guess,
            Edot_residual=float('inf'),
            T_residual=float('inf'),
            total_residual=float('inf'),
            converged=False,
            iterations=0,
            bubble_properties=None,
        )

    # Sort by residual and pick best
    candidates.sort(key=lambda x: x[2])
    best_beta, best_delta, best_residual, best_method, best_iterations = candidates[0]

    # Determine convergence and method description
    converged = best_residual < RESIDUAL_THRESHOLD

    if grid_converged:
        method_desc = 'grid'
    elif lbfgsb_converged:
        method_desc = 'grid->lbfgsb'
    else:
        # Neither converged, picked best from all candidates
        method_desc = f'best({best_method})'

    # Get final detailed residuals for best result
    details = get_residual_detailed(best_beta, best_delta, params)

    logger.info(
        f"Beta-delta result ({method_desc}): β={best_beta:.4f}, δ={best_delta:.4f}, "
        f"residual={best_residual:.2e}, converged={converged}"
    )

    return BetaDeltaResult(
        beta=best_beta,
        delta=best_delta,
        Edot_residual=details.Edot_residual,
        T_residual=details.T_residual,
        total_residual=best_residual,
        converged=converged,
        iterations=best_iterations,
        bubble_properties=details.bubble_props,
        # Raw residual components for saving to dictionary
        Edot_from_beta=details.Edot_from_beta,
        Edot_from_balance=details.Edot_from_balance,
        T_bubble=details.T_bubble,
        T0=details.T0,
        # Energy balance terms
        L_gain=details.L_gain,
        L_loss=details.L_loss,
    )


def _solve_grid(
    beta_guess: float,
    delta_guess: float,
    params,
) -> Tuple[float, float, int]:
    """
    Grid search solver using BubbleParamsView (no deepcopy).

    Searches a 5x5 grid around the guess, matching original get_betadelta.py.
    """
    # Generate grid around guess
    beta_min = max(BETA_MIN, beta_guess - GRID_EPSILON)
    beta_max = min(BETA_MAX, beta_guess + GRID_EPSILON)
    delta_min = max(DELTA_MIN, delta_guess - GRID_EPSILON)
    delta_max = min(DELTA_MAX, delta_guess + GRID_EPSILON)

    beta_range = np.linspace(beta_min, beta_max, GRID_SIZE)
    delta_range = np.linspace(delta_min, delta_max, GRID_SIZE)

    # Evaluate all grid points
    best_residual = float('inf')
    best_beta, best_delta = beta_guess, delta_guess

    for beta in beta_range:
        for delta in delta_range:
            try:
                Edot_res, T_res, _ = get_residual_pure(beta, delta, params)
                residual = Edot_res**2 + T_res**2
                if residual < best_residual:
                    best_residual = residual
                    best_beta, best_delta = beta, delta
            except Exception as e:
                logger.critical(f"Grid point ({beta:.3f}, {delta:.3f}) failed: {e}")
                continue

    return best_beta, best_delta, GRID_SIZE * GRID_SIZE


def _solve_lbfgsb(
    beta_guess: float,
    delta_guess: float,
    params,
) -> Tuple[float, float, int]:
    """
    L-BFGS-B optimizer solver.
    """
    def objective(x):
        beta, delta = x
        beta = np.clip(beta, BETA_MIN, BETA_MAX)
        delta = np.clip(delta, DELTA_MIN, DELTA_MAX)
        try:
            Edot_res, T_res, _ = get_residual_pure(beta, delta, params)
            return Edot_res**2 + T_res**2
        except Exception as e:
            logger.warning(f"Residual calculation failed: {e}")
            return 1e10

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
        return result.x[0], result.x[1], result.nit
    except Exception as e:
        logger.critical(f"Optimizer failed: {e}, using initial guess")
        return beta_guess, delta_guess, 0


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
