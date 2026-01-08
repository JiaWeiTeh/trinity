#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED: read_coolingcurve.py

CIE (Collisional Ionization Equilibrium) cooling function Lambda(T).

KEY IMPROVEMENTS FROM ORIGINAL:
================================

1. INPUT VALIDATION - Checks T > 0, not NaN, not inf
2. BOUNDS CHECKING - Warns if T outside interpolation range with fallback
3. DEAD CODE REMOVED - No commented-out sections
4. UNUSED IMPORTS REMOVED - Removed sys, astropy.units
5. PROPER LOGGING - Uses logging module instead of silent failures
6. TYPE HINTS - Clear function signatures
7. DOCSTRING IMPROVEMENTS - Clear, accurate documentation
8. OPTIONAL EXTRAPOLATION - Can handle T outside range gracefully

@author: Refactored version by Claude Code
@date: 2026-01-07
@original_author: Jia Wei Teh
"""

import numpy as np
import logging
from typing import Union, Callable

logger = logging.getLogger(__name__)


def get_Lambda(
    T: Union[float, np.ndarray],
    cooling_CIE_interpolation: Callable,
    metallicity: float = 1.0,
    allow_extrapolation: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate cooling function Lambda(T) for CIE conditions.

    In Collisional Ionization Equilibrium (CIE), the cooling function depends
    only on temperature, not on density or ionization state.

    Parameters
    ----------
    T : float or np.ndarray [K]
        Temperature. Must be positive.
    cooling_CIE_interpolation : callable
        Interpolation function: log10(T) → log10(Lambda)
        Created from cooling curve data file.
    metallicity : float, optional
        Metallicity relative to solar (default: 1.0 = solar)
        NOTE: Currently not implemented for non-solar values.
        Reserved for future use.
    allow_extrapolation : bool, optional
        If True, extrapolate beyond interpolation range using nearest value.
        If False, raise ValueError for out-of-range temperatures.
        Default: False

    Returns
    -------
    Lambda : float or np.ndarray [erg s^-1 cm^3]
        Cooling function.

    Raises
    ------
    ValueError
        If T <= 0, T is NaN/inf, or T outside interpolation range
        (when allow_extrapolation=False)

    Notes
    -----
    Available cooling curves:
    1. CLOUDY HII region, solar metallicity
    2. CLOUDY HII region with grain sublimation cooling
    3. Gnat & Ferland 2012 (interpolated)
    4. Sutherland & Dopita 1993 ([Fe/H] = -1)

    Examples
    --------
    >>> # Assuming cooling_CIE_interpolation is loaded
    >>> T = 1e6  # K
    >>> Lambda = get_Lambda(T, cooling_CIE_interpolation, metallicity=1.0)
    >>> print(f"Lambda = {Lambda:.3e} erg s^-1 cm^3")

    >>> # Array input
    >>> T_arr = np.logspace(4, 7, 100)
    >>> Lambda_arr = get_Lambda(T_arr, cooling_CIE_interpolation)
    """

    # ==========================================================================
    # INPUT VALIDATION
    # ==========================================================================

    # Check for NaN or inf
    if np.any(np.isnan(T)):
        raise ValueError("Temperature contains NaN values")

    if np.any(np.isinf(T)):
        raise ValueError("Temperature contains inf values")

    # Check T > 0
    if np.any(T <= 0):
        if np.isscalar(T):
            raise ValueError(f"Temperature must be positive, got T = {T} K")
        else:
            bad_indices = np.where(T <= 0)[0]
            raise ValueError(
                f"Temperature must be positive. "
                f"Found {len(bad_indices)} negative/zero values. "
                f"Example: T[{bad_indices[0]}] = {T[bad_indices[0]]} K"
            )

    # ==========================================================================
    # CHECK INTERPOLATION BOUNDS
    # ==========================================================================

    # Convert to log10
    log10_T = np.log10(T)

    # Get interpolation range from the interpolator
    # Assuming scipy.interpolate.interp1d structure
    try:
        T_min_log = cooling_CIE_interpolation.x[0]
        T_max_log = cooling_CIE_interpolation.x[-1]
    except AttributeError:
        # If different interpolator type, skip bounds checking
        logger.debug("Could not extract interpolation bounds from interpolator")
        T_min_log = -np.inf
        T_max_log = np.inf

    # Check bounds
    out_of_bounds_low = np.any(log10_T < T_min_log)
    out_of_bounds_high = np.any(log10_T > T_max_log)

    if out_of_bounds_low or out_of_bounds_high:
        T_min = 10**T_min_log
        T_max = 10**T_max_log

        if allow_extrapolation:
            # Clamp to valid range with warning
            logger.warning(
                f"Temperature outside interpolation range "
                f"[{T_min:.2e}, {T_max:.2e}] K. "
                f"Clamping to nearest bound (extrapolation enabled)."
            )
            log10_T = np.clip(log10_T, T_min_log, T_max_log)
        else:
            # Raise error
            if np.isscalar(T):
                raise ValueError(
                    f"Temperature T = {T:.2e} K outside interpolation range "
                    f"[{T_min:.2e}, {T_max:.2e}] K. "
                    f"Set allow_extrapolation=True to clamp to nearest bound."
                )
            else:
                bad_low = np.sum(log10_T < T_min_log)
                bad_high = np.sum(log10_T > T_max_log)
                raise ValueError(
                    f"Temperature outside interpolation range "
                    f"[{T_min:.2e}, {T_max:.2e}] K. "
                    f"Found {bad_low} values too low, {bad_high} values too high. "
                    f"Set allow_extrapolation=True to clamp to nearest bound."
                )

    # ==========================================================================
    # CALCULATE LAMBDA
    # ==========================================================================

    # Interpolate log(Lambda) from log(T)
    log10_Lambda = cooling_CIE_interpolation(log10_T)

    # Convert back to linear
    Lambda = 10**log10_Lambda

    # ==========================================================================
    # METALLICITY HANDLING (Future feature)
    # ==========================================================================

    if metallicity != 1.0:
        logger.warning(
            f"Non-solar metallicity (Z = {metallicity}) requested, "
            f"but not yet implemented. Using solar metallicity instead."
        )
        # TODO: Implement metallicity scaling
        # Possible approaches:
        # 1. Load different cooling curves for different Z
        # 2. Scale Lambda by Z factor
        # 3. Interpolate between Z values

    return Lambda


# ==============================================================================
# EXAMPLE USAGE AND TESTING
# ==============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    print("="*80)
    print("REFACTORED read_coolingcurve.py")
    print("="*80)
    print()
    print("IMPROVEMENTS:")
    print("  1. Input validation (T > 0, not NaN/inf)")
    print("  2. Bounds checking with clear error messages")
    print("  3. Optional extrapolation with warnings")
    print("  4. Proper logging instead of silent failures")
    print("  5. Type hints and comprehensive docstrings")
    print("  6. No dead code or unused imports")
    print("="*80)
    print()

    # Example: Create a mock interpolation function for testing
    from scipy.interpolate import interp1d

    # Mock cooling curve data (log10(T) vs log10(Lambda))
    # Typical range: 10^4 K to 10^9 K
    log_T_data = np.linspace(4.0, 9.0, 100)
    # Typical Lambda: ~10^-23 to 10^-21 erg s^-1 cm^3
    # Simplified model: Lambda ∝ T^(-0.5) in middle range
    log_Lambda_data = -22.0 - 0.5 * (log_T_data - 6.5)

    # Create interpolator
    cooling_CIE_interpolation = interp1d(
        log_T_data,
        log_Lambda_data,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )

    print("EXAMPLE 1: Valid temperature")
    print("-" * 40)
    T1 = 1e6  # K
    Lambda1 = get_Lambda(T1, cooling_CIE_interpolation)
    print(f"T = {T1:.2e} K → Lambda = {Lambda1:.3e} erg s^-1 cm^3")
    print()

    print("EXAMPLE 2: Array of temperatures")
    print("-" * 40)
    T_arr = np.logspace(5, 7, 5)
    Lambda_arr = get_Lambda(T_arr, cooling_CIE_interpolation)
    for T, Lambda in zip(T_arr, Lambda_arr):
        print(f"T = {T:.2e} K → Lambda = {Lambda:.3e} erg s^-1 cm^3")
    print()

    print("EXAMPLE 3: Temperature at boundary")
    print("-" * 40)
    T_min = 10**log_T_data[0]
    Lambda_min = get_Lambda(T_min, cooling_CIE_interpolation)
    print(f"T = {T_min:.2e} K (minimum) → Lambda = {Lambda_min:.3e} erg s^-1 cm^3")
    print()

    print("EXAMPLE 4: Temperature slightly out of range (with extrapolation)")
    print("-" * 40)
    T_low = 1e3  # Below minimum (10^4)
    try:
        Lambda_low = get_Lambda(T_low, cooling_CIE_interpolation, allow_extrapolation=True)
        print(f"T = {T_low:.2e} K → Lambda = {Lambda_low:.3e} erg s^-1 cm^3 (extrapolated)")
    except ValueError as e:
        print(f"ERROR: {e}")
    print()

    print("EXAMPLE 5: Invalid temperature (should raise error)")
    print("-" * 40)
    try:
        T_invalid = -1000.0  # Negative!
        Lambda_invalid = get_Lambda(T_invalid, cooling_CIE_interpolation)
        print(f"T = {T_invalid} K → Lambda = {Lambda_invalid:.3e} (should not reach here!)")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    print()

    print("EXAMPLE 6: NaN temperature (should raise error)")
    print("-" * 40)
    try:
        T_nan = np.nan
        Lambda_nan = get_Lambda(T_nan, cooling_CIE_interpolation)
        print(f"T = {T_nan} K → Lambda = {Lambda_nan} (should not reach here!)")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    print()

    print("="*80)
    print("All examples completed successfully!")
    print("="*80)
