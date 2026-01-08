#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED VERSION of read_SB99.py

Author: Jia Wei Teh (original)
Refactored: 2026-01-08

This script reads Starburst99 stellar evolution model data.

IMPROVEMENTS OVER ORIGINAL:
1. ✓ FIXED broken exception handling (no more undefined variable references)
2. ✓ Added explicit metallicity validation (clear error for unsupported Z)
3. ✓ Added div-by-zero protection (EPSILON for Lmech and pdot)
4. ✓ Added comprehensive input validation
5. ✓ Added proper logging instead of silent operations
6. ✓ Added type hints
7. ✓ Documented unit conversions explicitly
8. ✓ Removed unused imports (sys)
9. ✓ Better error messages

CRITICAL FIXES:
- Lines 180-182: Broken exception handling
  - Original: Referenced undefined 'filename' variable in except block
  - Fixed: Construct filename before try block, proper error messages
- Lines 164-176: Hardcoded metallicity
  - Original: Only Z=1.0 or 0.15, silent failure for others (z_str undefined)
  - Fixed: Explicit ValueError for unsupported metallicities
- Lines 93-94, 113: Division by zero
  - Original: Mdot = pdot²/(2*Lmech) can be inf/NaN if Lmech=0
  - Fixed: Use EPSILON to prevent division by zero
- Lines 65-77: Silent unit conversions
  - Original: Magic conversion factors, no documentation
  - Fixed: Explicit documentation of all conversions
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.interpolate

import src._functions.unit_conversions as cvt

# Set up logging
logger = logging.getLogger(__name__)

# Physical constants
EPSILON = 1e-100  # Small number to prevent division by zero


def read_SB99(f_mass: float, params: Dict[str, Any]) -> List[np.ndarray]:
    """
    Read Starburst99 stellar evolution model data.

    This function loads SB99 feedback parameters (ionizing photons, luminosities,
    mechanical energy) as a function of time for a stellar cluster.

    Parameters
    ----------
    f_mass : float
        Cluster mass fraction relative to SB99 file mass
        (e.g., f_mass = 0.1 for 1e5 Msun cluster if SB99 file is for 1e6 Msun)
    params : dict
        Dictionary containing SB99 parameters:
        - 'path_sps': Path to SB99 files
        - 'SB99_mass': Reference mass in SB99 file [Msun]
        - 'SB99_rotation': Boolean for stellar rotation
        - 'ZCloud': Metallicity in solar units
        - 'SB99_BHCUT': Black hole cutoff mass [Msun]
        - 'FB_mColdWindFrac': Cold mass fraction in winds
        - 'FB_thermCoeffWind': Thermal efficiency of winds
        - 'FB_mColdSNFrac': Cold mass fraction in SNe
        - 'FB_thermCoeffSN': Thermal efficiency of SNe
        - 'FB_vSN': SN ejecta velocity

    Returns
    -------
    [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN] : list of np.ndarray
        t : Time [Myr]
        Qi : Ionizing photon emission rate [1/Myr] (AU units)
        Li : Ionizing luminosity (>13.6 eV) [erg/s] (AU units)
        Ln : Non-ionizing luminosity (<13.6 eV) [erg/s] (AU units)
        Lbol : Bolometric luminosity [erg/s] (AU units)
        Lmech : Total mechanical luminosity [erg/s] (AU units)
        pdot : Total momentum injection rate [g·cm/s²] (AU units)
        pdot_SN : SN momentum injection rate [g·cm/s²] (AU units)

    Raises
    ------
    ValueError
        If f_mass <= 0 or is NaN/inf
        If unsupported metallicity
        If unsupported black hole cutoff mass
    FileNotFoundError
        If SB99 file not found

    Notes
    -----
    SB99 file format (7 columns):
    1. Time [yr]
    2. log₁₀(Qi) [1/s]
    3. log₁₀(fi) [fraction of ionizing radiation]
    4. log₁₀(Lbol) [erg/s]
    5. log₁₀(Lmech) [erg/s] (winds + SNe)
    6. log₁₀(pdot_W) [g·cm/s²] (winds only)
    7. log₁₀(Lmech_W) [erg/s] (winds only)

    Unit conversions (CGS → AU):
    - Time: yr → Myr (divide by 1e6)
    - Qi: 1/s → 1/Myr (divide by cvt.s2Myr)
    - Luminosity: erg/s → AU (multiply by cvt.L_cgs2au)
    - Momentum rate: g·cm/s² → AU (multiply by cvt.pdot_cgs2au)

    Wind/SN modifications:
    - Mass injection: Mdot × (1 + f_cold)
    - Terminal velocity: v × √(η_therm / (1 + f_cold))
    where η_therm is thermal efficiency, f_cold is cold mass fraction
    """

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if not np.isfinite(f_mass) or f_mass <= 0:
        raise ValueError(
            f"Invalid f_mass: {f_mass}. Must be finite and > 0."
        )

    required_keys = [
        'path_sps', 'SB99_rotation', 'ZCloud', 'SB99_BHCUT',
        'FB_mColdWindFrac', 'FB_thermCoeffWind',
        'FB_mColdSNFrac', 'FB_thermCoeffSN', 'FB_vSN'
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Required parameter '{key}' missing from params")

    logger.info(f"Reading SB99 data with f_mass = {f_mass}")

    # =========================================================================
    # STEP 1: GET FILENAME AND READ FILE
    # =========================================================================

    filename = get_filename(params)
    path2sps = params['path_sps']
    filepath = path2sps + filename

    logger.debug(f"Loading SB99 file: {filepath}")

    try:
        SB99_file = np.loadtxt(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SB99 file not found: {filepath}\n"
            f"Check parameters:\n"
            f"  - SB99_rotation: {params['SB99_rotation']}\n"
            f"  - ZCloud: {params['ZCloud']}\n"
            f"  - SB99_BHCUT: {params['SB99_BHCUT']}"
        )
    except Exception as e:
        raise IOError(f"Error reading SB99 file {filepath}: {e}")

    # Validate file format
    if SB99_file.ndim != 2 or SB99_file.shape[1] < 7:
        raise ValueError(
            f"Invalid SB99 file format in {filename}. "
            f"Expected 2D array with >= 7 columns, got shape {SB99_file.shape}"
        )

    logger.debug(f"Loaded {len(SB99_file)} time steps from SB99 file")

    # =========================================================================
    # STEP 2: READ COLUMNS AND CONVERT UNITS
    # =========================================================================

    # Time: yr → Myr
    t = SB99_file[:, 0] / 1e6

    # Convert from log space and scale by cluster mass
    # Ionizing photon rate: log₁₀(1/s) → 1/Myr
    Qi = 10**SB99_file[:, 1] * f_mass / cvt.s2Myr

    # Ionizing fraction (linear, not log)
    fi = 10**SB99_file[:, 2]

    # Bolometric luminosity: log₁₀(erg/s) → erg/s (AU)
    Lbol = 10**SB99_file[:, 3] * f_mass * cvt.L_cgs2au

    # Mechanical luminosity (winds + SNe): log₁₀(erg/s) → erg/s (AU)
    Lmech = 10**SB99_file[:, 4] * f_mass * cvt.L_cgs2au

    # Wind momentum rate: log₁₀(g·cm/s²) → g·cm/s² (AU)
    pdot_W = 10**SB99_file[:, 5] * f_mass * cvt.pdot_cgs2au

    # Wind mechanical luminosity: log₁₀(erg/s) → erg/s (AU)
    Lmech_W = 10**SB99_file[:, 6] * f_mass * cvt.L_cgs2au

    # Validate all arrays are finite
    for name, arr in [('t', t), ('Qi', Qi), ('fi', fi), ('Lbol', Lbol),
                       ('Lmech', Lmech), ('pdot_W', pdot_W), ('Lmech_W', Lmech_W)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in {name} array from SB99 file")

    # =========================================================================
    # STEP 3: CALCULATE DERIVED VALUES
    # =========================================================================

    # Ionizing and non-ionizing luminosity (13.6 eV threshold)
    Li = Lbol * fi
    Ln = Lbol * (1 - fi)

    # Mechanical luminosity from SNe only
    Lmech_SN = Lmech - Lmech_W

    # Validate no negative values
    if np.any(Lmech_SN < 0):
        logger.warning(
            "Negative SN mechanical luminosity detected. "
            "This may indicate Lmech_W > Lmech in SB99 file. "
            "Setting negative values to zero."
        )
        Lmech_SN = np.maximum(Lmech_SN, 0)

    # =========================================================================
    # STEP 4: SCALE WIND PARAMETERS
    # =========================================================================

    # Break down into mass loss rate and terminal velocity
    # Mdot = pdot² / (2 * Lmech)
    # v = 2 * Lmech / pdot
    #
    # CRITICAL: Add EPSILON to prevent division by zero!

    Mdot_W = pdot_W**2 / (2 * (Lmech_W + EPSILON))
    velocity_W = 2 * (Lmech_W + EPSILON) / (pdot_W + EPSILON)

    # Validate results
    if not np.all(np.isfinite(Mdot_W)):
        logger.error("Non-finite Mdot_W detected after calculation")
        raise ValueError("Failed to calculate wind mass loss rate")

    if not np.all(np.isfinite(velocity_W)):
        logger.error("Non-finite velocity_W detected after calculation")
        raise ValueError("Failed to calculate wind velocity")

    # Apply scaling factors
    # 1) Cold mass fraction: increases mass injection
    Mdot_W *= (1 + params['FB_mColdWindFrac'])

    # 2) Thermal efficiency: modifies velocity
    velocity_W *= np.sqrt(
        params['FB_thermCoeffWind'] / (1.0 + params['FB_mColdWindFrac'])
    )

    # Recalculate momentum and energy rates
    pdot_W = Mdot_W * velocity_W
    Lmech_W = 0.5 * Mdot_W * velocity_W**2

    # =========================================================================
    # STEP 5: SCALE SN PARAMETERS
    # =========================================================================

    # SN ejecta velocity (constant, from parameters)
    velocity_SN = params['FB_vSN']

    if velocity_SN <= 0:
        raise ValueError(
            f"Invalid SN velocity: {velocity_SN}. Must be > 0."
        )

    # Mass loss rate from energy: Mdot = 2 * Lmech / v²
    Mdot_SN = 2 * Lmech_SN / (velocity_SN**2 + EPSILON)

    # Apply cold mass fraction
    Mdot_SN *= (1 + params['FB_mColdSNFrac'])

    # Apply thermal efficiency to velocity
    velocity_SN *= np.sqrt(
        params['FB_thermCoeffSN'] / (1.0 + params['FB_mColdSNFrac'])
    )

    # Recalculate momentum and energy rates
    pdot_SN = Mdot_SN * velocity_SN
    Lmech_SN = 0.5 * Mdot_SN * velocity_SN**2

    # =========================================================================
    # STEP 6: COMBINE WIND + SN CONTRIBUTIONS
    # =========================================================================

    Lmech = Lmech_SN + Lmech_W
    pdot = pdot_SN + pdot_W

    # =========================================================================
    # STEP 7: INSERT t=0 FOR INTERPOLATION
    # =========================================================================

    # Add initial values at t=0 for better interpolation behavior
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])
    Lmech = np.insert(Lmech, 0, Lmech[0])
    pdot = np.insert(pdot, 0, pdot[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])

    logger.info(
        f"SB99 data loaded successfully. "
        f"Time range: {t[0]:.2f} - {t[-1]:.2f} Myr"
    )

    return [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN]


def get_filename(params: Dict[str, Any]) -> str:
    """
    Construct SB99 filename from simulation parameters.

    Parameters
    ----------
    params : dict
        Must contain:
        - 'SB99_mass': Cluster mass in SB99 file [Msun]
        - 'SB99_rotation': Boolean for stellar rotation
        - 'ZCloud': Metallicity in solar units
        - 'SB99_BHCUT': Black hole cutoff mass [Msun]

    Returns
    -------
    filename : str
        SB99 filename (e.g., "1e6cluster_rot_Z0014_BH120.txt")

    Raises
    ------
    ValueError
        If metallicity or BH cutoff not supported

    Notes
    -----
    Filename format: {mass}cluster_{rotation}_{metallicity}_{BHcutoff}.txt

    Supported combinations:
    - Metallicity: 1.0 (Z0014, solar) or 0.15 (Z0002, 0.15 solar)
    - BH cutoff: 120 Msun or 40 Msun
    - Rotation: 'rot' or 'norot'
    """

    # Extract parameters with validation
    SB99_mass = params.get('SB99_mass')
    SB99_rotation = params.get('SB99_rotation')
    ZCloud = params.get('ZCloud')
    SB99_BHCUT = params.get('SB99_BHCUT')

    if SB99_mass is None or SB99_mass <= 0:
        raise ValueError(f"Invalid SB99_mass: {SB99_mass}")

    # Format mass (e.g., 1000000 → "1e6")
    def format_e(n: float) -> str:
        """Format number in simplified scientific notation."""
        a = '%E' % n
        mantissa = a.split('E')[0].rstrip('0').rstrip('.')
        exponent = a.split('E')[1].strip('+').lstrip('0') or '0'
        return f"{mantissa}e{exponent}"

    SBmass_str = format_e(SB99_mass)

    # Rotation string
    if SB99_rotation:
        rot_str = 'rot'
    else:
        rot_str = 'norot'

    # Metallicity string
    if ZCloud == 1.0:
        z_str = 'Z0014'  # Solar metallicity
    elif ZCloud == 0.15:
        z_str = 'Z0002'  # 0.15 solar
    else:
        raise ValueError(
            f"Unsupported metallicity: ZCloud = {ZCloud}. "
            "Only 1.0 (solar) and 0.15 (0.15 solar) are currently supported. "
            "Available SB99 files must be added for other metallicities."
        )

    # Black hole cutoff string
    if SB99_BHCUT == 120:
        BH_str = 'BH120'
    elif SB99_BHCUT == 40:
        BH_str = 'BH40'
    else:
        raise ValueError(
            f"Unsupported black hole cutoff mass: SB99_BHCUT = {SB99_BHCUT}. "
            "Only 120 Msun and 40 Msun are currently supported. "
            "Available SB99 files must be added for other cutoff masses."
        )

    # Construct filename
    filename = f"{SBmass_str}cluster_{rot_str}_{z_str}_{BH_str}.txt"

    logger.debug(f"SB99 filename: {filename}")

    return filename


def get_interpolation(
    SB99: List[np.ndarray],
    ftype: str = 'cubic'
) -> Dict[str, scipy.interpolate.interp1d]:
    """
    Create interpolation functions for SB99 data.

    Parameters
    ----------
    SB99 : list of np.ndarray
        SB99 data array from read_SB99()
    ftype : str, optional
        Interpolation type ('linear', 'cubic', etc.)
        Default is 'cubic' for smooth interpolation

    Returns
    -------
    SB99f : dict
        Dictionary of interpolation functions:
        - 'fQi': Ionizing photon rate
        - 'fLi': Ionizing luminosity
        - 'fLn': Non-ionizing luminosity
        - 'fLbol': Bolometric luminosity
        - 'fLw': Mechanical luminosity
        - 'fpdot': Momentum injection rate
        - 'fpdot_SNe': SN momentum injection rate

    Notes
    -----
    Cubic interpolation is important for accurate small-value interpolations.
    """

    # Unpack SB99 data
    [t_Myr, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe] = SB99

    # Validate time array
    if not np.all(np.diff(t_Myr) > 0):
        logger.warning("Time array is not strictly increasing. Sorting...")
        sort_idx = np.argsort(t_Myr)
        t_Myr = t_Myr[sort_idx]
        Qi = Qi[sort_idx]
        Li = Li[sort_idx]
        Ln = Ln[sort_idx]
        Lbol = Lbol[sort_idx]
        Lw = Lw[sort_idx]
        pdot = pdot[sort_idx]
        pdot_SNe = pdot_SNe[sort_idx]

    # Create interpolation functions
    try:
        fQi = scipy.interpolate.interp1d(t_Myr, Qi, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fLi = scipy.interpolate.interp1d(t_Myr, Li, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fLn = scipy.interpolate.interp1d(t_Myr, Ln, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fLbol = scipy.interpolate.interp1d(t_Myr, Lbol, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fLw = scipy.interpolate.interp1d(t_Myr, Lw, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fpdot = scipy.interpolate.interp1d(t_Myr, pdot, kind=ftype, bounds_error=False, fill_value='extrapolate')
        fpdot_SNe = scipy.interpolate.interp1d(t_Myr, pdot_SNe, kind=ftype, bounds_error=False, fill_value='extrapolate')
    except ValueError as e:
        raise ValueError(f"Failed to create interpolation functions: {e}")

    SB99f = {
        'fQi': fQi,
        'fLi': fLi,
        'fLn': fLn,
        'fLbol': fLbol,
        'fLw': fLw,
        'fpdot': fpdot,
        'fpdot_SNe': fpdot_SNe
    }

    logger.debug(f"Created {ftype} interpolation functions for SB99 data")

    return SB99f
