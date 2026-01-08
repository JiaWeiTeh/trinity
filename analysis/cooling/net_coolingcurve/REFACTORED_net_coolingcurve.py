#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED VERSION of net_coolingcurve.py

Author: Jia Wei Teh (original)
Refactored: 2026-01-08

This module creates a NET cooling rate (dudt) curve containing both CIE and
non-CIE conditions.

IMPROVEMENTS OVER ORIGINAL:
1. ✓ Added proper logging instead of silent operations
2. ✓ Added comprehensive input validation
3. ✓ Removed silent temperature clamping (now logs warning)
4. ✓ Added type hints
5. ✓ Added proper error messages
6. ✓ Removed commented-out code
7. ✓ Added docstring improvements
8. ✓ Added physical constants validation

CRITICAL FIXES:
- Lines 85-86: Silent temperature clamping now logs WARNING
- Added validation for ndens > 0, T > 0, phi > 0
- Added validation that cooling structures exist before use
- Better error messages for out-of-range values
"""

import logging
from typing import Dict, Any
import scipy.interpolate
import numpy as np

import src.cooling.CIE.read_coolingcurve as CIE
import src._functions.unit_conversions as cvt

# Set up logging
logger = logging.getLogger(__name__)


def get_dudt(
    age: float,
    ndens: float,
    T: float,
    phi: float,
    params_dict: Dict[str, Any]
) -> float:
    """
    Calculate net cooling rate dudt.

    This function switches between CIE and non-CIE cooling based on temperature,
    with smooth interpolation in the transition region.

    Parameters
    ----------
    age : float
        Age in Myr
    ndens : float
        Number density in 1/pc³
    T : float
        Temperature in K
    phi : float
        Ionizing photon flux in 1/pc²/Myr
    params_dict : dict
        Dictionary containing cooling structures and parameters

    Returns
    -------
    dudt : float
        Net cooling rate in M_sun/pc/yr³ (equivalent to erg/cm³/s)
        Returns NEGATIVE value (energy loss)

    Raises
    ------
    ValueError
        If inputs are invalid (negative, zero, or NaN)
    KeyError
        If required parameters missing from params_dict

    Notes
    -----
    - CIE cooling: Lambda depends only on T (T > ~10^5.5 K)
    - Non-CIE cooling: Lambda depends on (n, T, phi) triplet (T < ~10^5.5 K)
    - Smooth interpolation between the two regimes

    Physics:
    - Cooling rate: dudt = -n² * Lambda(T) [CIE]
    - Cooling rate: dudt = -Lambda(n, T, phi) [non-CIE]
    - Sign convention: NEGATIVE for energy loss

    Unit conversions:
    - ndens: pc⁻³ → cm⁻³ (divide by cvt.ndens_cgs2au)
    - phi: pc⁻²·yr⁻¹ → cm⁻²·s⁻¹ (divide by cvt.phi_cgs2au)
    - dudt: erg·cm⁻³·s⁻¹ → M_sun·pc⁻¹·yr⁻³ (multiply by cvt.dudt_cgs2au)
    """

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    # Validate inputs are positive and finite
    if not np.isfinite(age) or age < 0:
        raise ValueError(f"Invalid age: {age} Myr (must be finite and >= 0)")

    if not np.isfinite(ndens) or ndens <= 0:
        raise ValueError(f"Invalid ndens: {ndens} pc⁻³ (must be finite and > 0)")

    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"Invalid T: {T} K (must be finite and > 0)")

    if not np.isfinite(phi) or phi <= 0:
        raise ValueError(f"Invalid phi: {phi} pc⁻²·Myr⁻¹ (must be finite and > 0)")

    # Validate required parameters exist
    required_keys = [
        'cStruc_cooling_nonCIE',
        'cStruc_net_nonCIE_interpolation',
        'cStruc_cooling_CIE_interpolation',
        'cStruc_cooling_CIE_logT',
        'ZCloud'
    ]

    for key in required_keys:
        if key not in params_dict:
            raise KeyError(f"Required parameter '{key}' missing from params_dict")

    # =========================================================================
    # TEMPERATURE VALIDATION AND CLAMPING
    # =========================================================================

    # Minimum temperature constraint
    # NOTE: This is a workaround. The root cause should be investigated!
    # The temperature should be ~10^7 K in feedback regions, not 10^4 K.
    T_min = 1e4  # K

    if T < T_min:
        logger.warning(
            f"Temperature T = {T:.2e} K is below minimum {T_min:.2e} K. "
            f"Clamping to {T_min:.2e} K. "
            "This may indicate an underlying physics problem that should be investigated!"
        )
        T = T_min

    # =========================================================================
    # UNIT CONVERSIONS
    # =========================================================================

    # Convert from AU units to CGS
    ndens_cgs = ndens / cvt.ndens_cgs2au  # pc⁻³ → cm⁻³
    phi_cgs = phi / cvt.phi_cgs2au        # pc⁻²·yr⁻¹ → cm⁻²·s⁻¹

    # Validate conversions
    if not np.isfinite(ndens_cgs) or ndens_cgs <= 0:
        raise ValueError(f"Unit conversion failed for ndens: {ndens} → {ndens_cgs}")

    if not np.isfinite(phi_cgs) or phi_cgs <= 0:
        raise ValueError(f"Unit conversion failed for phi: {phi} → {phi_cgs}")

    # =========================================================================
    # EXTRACT COOLING STRUCTURES
    # =========================================================================

    cooling_nonCIE = params_dict['cStruc_cooling_nonCIE'].value
    netcool_interp = params_dict['cStruc_net_nonCIE_interpolation'].value
    CIE_interp = params_dict['cStruc_cooling_CIE_interpolation'].value
    logT_CIE = params_dict['cStruc_cooling_CIE_logT'].value
    ZCloud = params_dict['ZCloud'].value

    # =========================================================================
    # GET CIE COOLING RATE
    # =========================================================================

    # Lambda in erg·s⁻¹·cm³
    Lambda_CIE = CIE.get_Lambda(T, CIE_interp, ZCloud)

    if not np.isfinite(Lambda_CIE):
        raise ValueError(f"CIE cooling function returned invalid Lambda: {Lambda_CIE}")

    # =========================================================================
    # DETERMINE TEMPERATURE REGIME
    # =========================================================================

    # Cutoff temperatures (in log₁₀ space)
    # Below nonCIE_Tcutoff: use non-CIE cooling
    # Above CIE_Tcutoff: use CIE cooling
    # In between: interpolate

    # Maximum temperature in non-CIE table (should be ~10^5.5 K)
    nonCIE_Tcutoff = max(cooling_nonCIE.temp[cooling_nonCIE.temp <= 5.5])

    # Minimum temperature in CIE table (should be ~10^5.5 K)
    CIE_Tcutoff = min(logT_CIE[logT_CIE > 5.5])

    logT = np.log10(T)

    # =========================================================================
    # CASE 1: NON-CIE REGIME (T <= 10^5.5 K)
    # =========================================================================

    if logT <= nonCIE_Tcutoff and logT >= min(cooling_nonCIE.temp):

        logger.debug(f"Using non-CIE cooling (T = {T:.2e} K)")

        # Interpolate net cooling from (n, T, phi) cube
        # Input must be in log₁₀ space
        try:
            dudt_cgs = netcool_interp([
                np.log10(ndens_cgs),
                np.log10(T),
                np.log10(phi_cgs)
            ])[0]
        except ValueError as e:
            raise ValueError(
                f"Non-CIE interpolation failed for "
                f"(log n, log T, log phi) = ({np.log10(ndens_cgs):.2f}, "
                f"{np.log10(T):.2f}, {np.log10(phi_cgs):.2f}): {e}"
            )

        # Convert to AU units and return (negative sign for energy loss)
        return -1.0 * dudt_cgs * cvt.dudt_cgs2au

    # =========================================================================
    # CASE 2: CIE REGIME (T >= 10^5.5 K)
    # =========================================================================

    elif logT >= CIE_Tcutoff:

        logger.debug(f"Using CIE cooling (T = {T:.2e} K)")

        # CIE cooling: dudt = n² * Lambda(T)
        dudt_cgs = ndens_cgs**2 * Lambda_CIE

        # Convert to AU units and return (negative sign for energy loss)
        return -1.0 * dudt_cgs * cvt.dudt_cgs2au

    # =========================================================================
    # CASE 3: TRANSITION REGIME (10^5.5 K < T < 10^5.5 K)
    # =========================================================================

    elif logT > nonCIE_Tcutoff and logT < CIE_Tcutoff:

        logger.debug(
            f"Using interpolated cooling (T = {T:.2e} K, "
            f"between {10**nonCIE_Tcutoff:.2e} and {10**CIE_Tcutoff:.2e} K)"
        )

        # --- Non-CIE contribution (at maximum non-CIE temperature) ---

        try:
            dudt_nonCIE = netcool_interp([
                np.log10(ndens_cgs),
                nonCIE_Tcutoff,  # Use max non-CIE temperature
                np.log10(phi_cgs)
            ])[0]
        except ValueError as e:
            raise ValueError(
                f"Non-CIE interpolation failed at transition: {e}"
            )

        # --- CIE contribution (at minimum CIE temperature) ---

        Lambda_CIE_cutoff = CIE.get_Lambda(10**CIE_Tcutoff, CIE_interp, ZCloud)
        dudt_CIE = ndens_cgs**2 * Lambda_CIE_cutoff

        # --- Linear interpolation in log T space ---

        dudt_cgs = np.interp(
            logT,
            [nonCIE_Tcutoff, CIE_Tcutoff],
            [dudt_nonCIE, dudt_CIE]
        )

        # Convert to AU units and return (negative sign for energy loss)
        return -1.0 * dudt_cgs * cvt.dudt_cgs2au

    # =========================================================================
    # CASE 4: OUT OF RANGE
    # =========================================================================

    else:
        raise ValueError(
            f"Temperature T = {T:.2e} K (log₁₀ T = {logT:.2f}) is outside "
            f"the valid range of cooling tables:\n"
            f"  Non-CIE range: 10^{min(cooling_nonCIE.temp):.2f} - "
            f"10^{nonCIE_Tcutoff:.2f} K\n"
            f"  CIE range: 10^{CIE_Tcutoff:.2f}+ K\n"
            f"Cannot compute cooling rate."
        )
