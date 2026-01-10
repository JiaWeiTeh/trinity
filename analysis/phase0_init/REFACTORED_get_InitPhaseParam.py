#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED VERSION of get_InitPhaseParam.py

Original Author: Jia Wei Teh
Refactored: 2026-01-10

This script computes the initial values for the energy-driven phase
(from a short free-streaming phase).

CHANGES FROM ORIGINAL:
======================

1. WIND VELOCITY BUG FIX (Critical)
   Original used ambiguous 'fpdot' which may include SNe momentum.
   Refactored explicitly uses wind-only quantities for wind velocity:

   v0_wind = 2 * Lmech_W / pdot_W  (CORRECT - wind only)

   NOT:
   v0 = 2 * Lmech / pdot_total  (WRONG - includes SNe)

2. NEW NAMING CONVENTION
   Old: SB99f['fLw'], SB99f['fpdot']
   New: SB99f['fLmech_W'], SB99f['fLmech_SN'], SB99f['fLmech_total']
        SB99f['fpdot_W'], SB99f['fpdot_SN'], SB99f['fpdot_total']

3. FIXED .value ACCESS INCONSISTENCY
   Original code inconsistently used .value for some params but not others.
   Refactored consistently uses .value for all DescribedItem objects.

4. DOCSTRING MATCHES IMPLEMENTATION
   Original docstring documented wrong parameters and return values.
   Refactored docstring accurately reflects the params-based interface.

5. INPUT VALIDATION
   Added checks for invalid values that would cause physics errors.

6. NAMED CONSTANTS
   Magic numbers replaced with named constants with literature references.

7. OPTIONAL RETURN VALUES
   Function now optionally returns computed values for testing/debugging,
   while still updating params in-place for compatibility.

PHYSICS REFERENCE:
==================
- Free-streaming phase duration: Rahner thesis Eq. 1.15
  https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17

- Bubble energy: Weaver+77, Eq. 20
- Bubble temperature: Weaver+77, Eq. 37
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

import src._functions.unit_conversions as cvt

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS (with literature references)
# =============================================================================

# Energy fraction in bubble interior: E0 = (5/11) * Lw * dt
# From Weaver+77, Eq. 20 - assumes adiabatic index gamma = 5/3
WEAVER_ENERGY_FRACTION = 5.0 / 11.0

# Temperature coefficient in Weaver+77, Eq. 37
# T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4
# NOTE: Original code has TODO asking "isn't it 2.07?" - needs verification
WEAVER_TEMP_COEFFICIENT = 1.51e6  # Kelvin

# Reference luminosity for temperature scaling [erg/s]
WEAVER_L_REF = 1e36

# Minimum valid values to prevent division by zero
MIN_LUMINOSITY = 1e-100  # Prevent div by zero in Mdot calculation
MIN_MOMENTUM = 1e-100    # Prevent div by zero in velocity calculation
MIN_VELOCITY = 1e-100    # Prevent div by zero in dt_phase0 calculation


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_y0(
    params: Dict[str, Any],
    return_values: bool = False
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Compute initial values for the energy-driven (Weaver) phase.

    This function calculates the initial conditions at the end of the
    free-streaming phase, which marks the beginning of the energy-driven
    expansion phase.

    Parameters
    ----------
    params : dict
        Simulation parameters dictionary containing:

        Required keys (with .value attribute):
        - 'tSF' : float [Myr]
            Time of last star formation event (or recollapse).
        - 'SB99f' : dict
            Starburst99 interpolation functions with new naming convention:
            - 'fLmech_W': Wind mechanical luminosity interpolator
            - 'fLmech_SN': SN mechanical luminosity interpolator
            - 'fLmech_total': Total mechanical luminosity interpolator
            - 'fpdot_W': Wind momentum rate interpolator
            - 'fpdot_SN': SN momentum rate interpolator
            - 'fpdot_total': Total momentum rate interpolator

        Required keys (with .value attribute for DescribedItem, or raw values):
        - 'nCore' : float [AU units]
            Core number density.
        - 'mu_neu' : float [AU units]
            Mean molecular weight of neutral gas.
        - 'bubble_xi_Tb' : float [dimensionless]
            Thermal conduction efficiency parameter (0-1).

        Keys updated by this function:
        - 't_now' : float [Myr] - Starting time for Weaver phase
        - 'R2' : float [pc] - Initial bubble radius
        - 'v2' : float [pc/Myr] - Initial expansion velocity
        - 'Eb' : float [AU units] - Initial bubble energy
        - 'T0' : float [K] - Initial temperature

    return_values : bool, optional
        If True, return computed values as tuple. Default False.

    Returns
    -------
    None or tuple
        If return_values=False: None (updates params in-place only)
        If return_values=True: (t0, r0, v0, E0, T0) tuple

    Raises
    ------
    ValueError
        If input values are physically invalid (negative, zero, etc.)

    Notes
    -----
    CRITICAL: Wind velocity (v0) must be calculated using WIND-ONLY
    quantities (Lmech_W, pdot_W), not totals that include SNe.

    The free-streaming phase duration formula (dt_phase0) comes from
    Rahner thesis Eq. 1.15.

    See Also
    --------
    analysis.update_feedback.REFACTORED_update_feedback : SB99 feedback with bug fix

    References
    ----------
    .. [1] Weaver et al. 1977, ApJ, 218, 377
    .. [2] Rahner PhD Thesis, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf
    """

    # =========================================================================
    # EXTRACT PARAMETERS
    # =========================================================================

    # Time of star formation [Myr]
    tSF = params['tSF'].value

    # SB99 interpolation functions (with new naming convention)
    SB99f = params['SB99f'].value

    # Core properties - handle both DescribedItem and raw value access
    nCore = _get_param_value(params, 'nCore')
    mu_neu = _get_param_value(params, 'mu_neu')
    bubble_xi_Tb = _get_param_value(params, 'bubble_xi_Tb')

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if tSF < 0:
        raise ValueError(f"tSF must be non-negative, got {tSF}")

    if nCore <= 0:
        raise ValueError(f"nCore must be positive, got {nCore}")

    if mu_neu <= 0:
        raise ValueError(f"mu_neu must be positive, got {mu_neu}")

    if not (0 <= bubble_xi_Tb <= 1):
        raise ValueError(f"bubble_xi_Tb must be in [0,1], got {bubble_xi_Tb}")

    # =========================================================================
    # GET SB99 FEEDBACK VALUES AT tSF
    # =========================================================================

    # CRITICAL: Use WIND-ONLY quantities for wind velocity calculation
    # Using new naming convention
    Lmech_W = SB99f['fLmech_W'](tSF)
    pdot_W = SB99f['fpdot_W'](tSF)

    # Validate SB99 values
    if Lmech_W < MIN_LUMINOSITY:
        logger.warning(f"Lmech_W={Lmech_W} is very small at tSF={tSF} Myr")
        Lmech_W = MIN_LUMINOSITY

    if pdot_W < MIN_MOMENTUM:
        logger.warning(f"pdot_W={pdot_W} is very small at tSF={tSF} Myr")
        pdot_W = MIN_MOMENTUM

    # =========================================================================
    # COMPUTE WIND PROPERTIES (WIND-ONLY - BUG FIX)
    # =========================================================================

    # Mass loss rate from winds [AU units]
    # From: L = 0.5 * Mdot * v^2 and pdot = Mdot * v
    # => Mdot = pdot^2 / (2 * L)
    Mdot0 = pdot_W**2 / (2.0 * Lmech_W)

    # Terminal velocity from winds [pc/Myr in AU units]
    # From: v = 2 * L / pdot
    #
    # CRITICAL BUG FIX: Use wind-only quantities!
    # WRONG:  v0 = 2 * Lmech_total / pdot_total  (includes SNe)
    # RIGHT:  v0 = 2 * Lmech_W / pdot_W          (wind only)
    v0 = 2.0 * Lmech_W / pdot_W

    if v0 < MIN_VELOCITY:
        logger.warning(f"v0={v0} is very small, may cause numerical issues")
        v0 = MIN_VELOCITY

    # =========================================================================
    # COMPUTE FREE-STREAMING PHASE DURATION
    # =========================================================================

    # Ambient density [AU units: Msun/pc^3]
    rhoa = nCore * mu_neu

    # Duration of free-streaming phase [Myr]
    # From Rahner thesis Eq. 1.15:
    # dt = sqrt(3 * Mdot / (4 * pi * rho_a * v^3))
    dt_phase0 = np.sqrt(3.0 * Mdot0 / (4.0 * np.pi * rhoa * v0**3))

    logger.debug(f"Free-streaming phase duration: dt_phase0 = {dt_phase0:.6e} Myr")

    # =========================================================================
    # COMPUTE INITIAL VALUES FOR WEAVER PHASE
    # =========================================================================

    # Start time for Weaver phase [Myr]
    t0 = tSF + dt_phase0

    # Initial separation / bubble radius [pc]
    r0 = v0 * dt_phase0

    # Initial bubble energy [AU units]
    # From Weaver+77, Eq. 20: E = (5/11) * L * t
    E0 = WEAVER_ENERGY_FRACTION * Lmech_W * dt_phase0

    # Initial temperature [K]
    # From Weaver+77, Eq. 37:
    # T = 1.51e6 * (L/10^36)^(8/35) * (n)^(2/35) * t^(-6/35) * (1-xi)^0.4
    T0 = WEAVER_TEMP_COEFFICIENT * \
         (Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0) * \
         (nCore * cvt.ndens_au2cgs)**(2.0/35.0) * \
         (dt_phase0)**(-6.0/35.0) * \
         (1.0 - bubble_xi_Tb)**0.4

    # =========================================================================
    # UPDATE PARAMS DICTIONARY
    # =========================================================================

    _update_param_value(params, 't_now', t0)
    _update_param_value(params, 'R2', r0)
    _update_param_value(params, 'v2', v0)
    _update_param_value(params, 'Eb', E0)
    _update_param_value(params, 'T0', T0)

    logger.info(
        f"Initial Weaver phase values: "
        f"t0={t0:.6f} Myr, r0={r0:.6e} pc, v0={v0:.6e} pc/Myr, "
        f"E0={E0:.6e}, T0={T0:.2e} K"
    )

    if return_values:
        return t0, r0, v0, E0, T0

    return None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_param_value(params: Dict[str, Any], key: str) -> float:
    """
    Safely extract parameter value, handling both DescribedItem and raw values.

    Parameters
    ----------
    params : dict
        Parameters dictionary
    key : str
        Key to extract

    Returns
    -------
    value : float
        The parameter value
    """
    param = params[key]

    # Check if it's a DescribedItem (has .value attribute)
    if hasattr(param, 'value'):
        return param.value
    else:
        # Raw value
        return param


def _update_param_value(params: Dict[str, Any], key: str, value: float) -> None:
    """
    Safely update parameter value, handling both DescribedItem and raw values.

    Parameters
    ----------
    params : dict
        Parameters dictionary
    key : str
        Key to update
    value : float
        New value to set
    """
    if key not in params:
        logger.warning(f"Key '{key}' not found in params, skipping update")
        return

    param = params[key]

    # Check if it's a DescribedItem (has .value attribute)
    if hasattr(param, 'value'):
        param.value = value
    else:
        # Raw value - update directly
        params[key] = value


# =============================================================================
# LEGACY COMPATIBILITY WRAPPER
# =============================================================================

def get_y0_legacy(params: Dict[str, Any]) -> None:
    """
    Legacy wrapper for backward compatibility with old SB99 naming.

    DEPRECATED: Use get_y0() with new SB99 naming convention.

    This wrapper handles the old naming convention:
    - SB99f['fLw'] -> SB99f['fLmech_W']
    - SB99f['fpdot'] -> SB99f['fpdot_W'] (assuming wind-only was intended)

    Parameters
    ----------
    params : dict
        Parameters with old-style SB99f naming

    Warnings
    --------
    This function is DEPRECATED. The old naming convention ('fLw', 'fpdot')
    was ambiguous about whether wind-only or total values were used.
    Migrate to get_y0() with explicit naming convention.
    """

    import warnings
    warnings.warn(
        "get_y0_legacy() is deprecated. "
        "Use get_y0() with new SB99 naming convention "
        "(fLmech_W, fpdot_W, etc.).",
        DeprecationWarning
    )

    SB99f = params['SB99f'].value

    # Check if old naming is present
    if 'fLw' in SB99f and 'fLmech_W' not in SB99f:
        # Map old names to new names
        SB99f['fLmech_W'] = SB99f['fLw']
        logger.info("Mapped legacy 'fLw' -> 'fLmech_W'")

    if 'fpdot' in SB99f and 'fpdot_W' not in SB99f:
        # ASSUMPTION: Old 'fpdot' was total, but we need wind-only
        # This is a potential source of bugs if the original code
        # incorrectly used total momentum for wind velocity
        SB99f['fpdot_W'] = SB99f['fpdot']
        logger.warning(
            "Mapped legacy 'fpdot' -> 'fpdot_W'. "
            "Verify this is wind-only momentum rate!"
        )

    # Call the new function
    return get_y0(params)


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_wind_velocity_error(
    params: Dict[str, Any],
    tSF: Optional[float] = None
) -> Dict[str, float]:
    """
    Diagnose potential wind velocity calculation errors.

    Compares wind velocity calculated with wind-only vs total momentum.

    Parameters
    ----------
    params : dict
        Simulation parameters
    tSF : float, optional
        Time to evaluate. If None, uses params['tSF'].value

    Returns
    -------
    diagnostic : dict
        Dictionary containing:
        - 'v0_correct': Wind velocity using pdot_W (correct)
        - 'v0_wrong': Wind velocity using pdot_total (incorrect)
        - 'error_percent': Relative error (%)
        - 'pdot_W': Wind momentum rate
        - 'pdot_SN': SN momentum rate
        - 'pdot_total': Total momentum rate
    """

    if tSF is None:
        tSF = params['tSF'].value

    SB99f = params['SB99f'].value

    Lmech_W = SB99f['fLmech_W'](tSF)
    pdot_W = SB99f['fpdot_W'](tSF)
    pdot_SN = SB99f['fpdot_SN'](tSF)
    pdot_total = SB99f['fpdot_total'](tSF)

    # Correct calculation (wind-only)
    v0_correct = 2.0 * Lmech_W / pdot_W

    # Incorrect calculation (total momentum)
    v0_wrong = 2.0 * Lmech_W / pdot_total

    # Error
    if v0_correct > 0:
        error_percent = 100.0 * (v0_correct - v0_wrong) / v0_correct
    else:
        error_percent = 0.0

    return {
        'v0_correct': v0_correct,
        'v0_wrong': v0_wrong,
        'error_percent': error_percent,
        'Lmech_W': Lmech_W,
        'pdot_W': pdot_W,
        'pdot_SN': pdot_SN,
        'pdot_total': pdot_total,
    }


def print_diagnostic_report(params: Dict[str, Any]) -> None:
    """
    Print a diagnostic report for the initial phase parameters.

    Parameters
    ----------
    params : dict
        Simulation parameters (after get_y0 has been called)
    """

    print("=" * 70)
    print("INITIAL PHASE PARAMETER DIAGNOSTIC REPORT")
    print("=" * 70)

    tSF = params['tSF'].value
    print(f"\nStar formation time: tSF = {tSF:.6f} Myr")

    # Get diagnostic info
    diag = diagnose_wind_velocity_error(params, tSF)

    print(f"\nSB99 Feedback at tSF:")
    print(f"  Lmech_W     = {diag['Lmech_W']:.6e} [AU units]")
    print(f"  pdot_W      = {diag['pdot_W']:.6e} [AU units]")
    print(f"  pdot_SN     = {diag['pdot_SN']:.6e} [AU units]")
    print(f"  pdot_total  = {diag['pdot_total']:.6e} [AU units]")

    print(f"\nWind Velocity Calculation:")
    print(f"  v0 (correct, pdot_W):     {diag['v0_correct']:.6e} [pc/Myr]")
    print(f"  v0 (wrong, pdot_total):   {diag['v0_wrong']:.6e} [pc/Myr]")
    print(f"  Error if using total:     {diag['error_percent']:.1f}%")

    # Print computed initial values
    print(f"\nComputed Initial Values:")
    print(f"  t0 (start time) = {_get_param_value(params, 't_now'):.6f} Myr")
    print(f"  r0 (radius)     = {_get_param_value(params, 'R2'):.6e} pc")
    print(f"  v0 (velocity)   = {_get_param_value(params, 'v2'):.6e} pc/Myr")
    print(f"  E0 (energy)     = {_get_param_value(params, 'Eb'):.6e} [AU units]")
    print(f"  T0 (temp)       = {_get_param_value(params, 'T0'):.2e} K")

    print("\n" + "=" * 70)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("REFACTORED_get_InitPhaseParam.py")
    print("This module provides get_y0() with fixed wind velocity calculation.")
    print("\nKey changes from original:")
    print("  1. Uses wind-only (Lmech_W, pdot_W) for velocity calculation")
    print("  2. New SB99 naming convention (_W, _SN, _total suffixes)")
    print("  3. Consistent .value access for DescribedItem params")
    print("  4. Input validation and logging")
    print("  5. Named constants with literature references")
    print("\nRun diagnose_wind_velocity_error() to check for potential bugs.")
