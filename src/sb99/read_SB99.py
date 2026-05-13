#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:06:39 2023

@author: Jia Wei Teh

This script contains functions that will help reading in Starburst99 data.

"""

import numpy as np
import scipy
import sys
import logging

import src._functions.unit_conversions as cvt

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Physical constants
EPSILON = 1e-100  # Small number to prevent division by zero


# TODO: Support in-between metallicities (currently only Z=1 and Z=0.15 solar are supported).

def read_SB99(f_mass, params):
    """
    Read and process Starburst99 stellar feedback data.

    This function reads SB99 files, scales by cluster mass, applies thermal
    efficiency and cold mass corrections, and returns time series data with
    properly separated wind and SN components.
    
    Original units from SB99 files are in cgs (except t = Myr). We will convert
    all of them to astronomical units (AU) here. 

    Parameters
    ----------
    f_mass : float
        Cluster mass fraction (f_mass = M_cluster / SB99_mass)
    params : DescribedDict
        TRINITY parameters dictionary containing:
        - sps_path : str, full path to the SPS data file (already resolved
          by read_param.py; legacy SB99 grammar is the permanent fallback
          when the user hasn't overridden sps_path — see
          analysis/sb99-refactor-audit.md §9)
        - FB_mColdWindFrac, FB_thermCoeffWind : Wind corrections
        - FB_mColdSNFrac, FB_thermCoeffSN, FB_vSN : SN corrections

    Returns
    -------
    list : [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]
        Time series arrays (all with t=0 prepended):

        t : ndarray
            Time [Myr]
        Qi : ndarray
            Ionizing photon rate [s⁻¹] (AU)
        Li : ndarray
            Ionizing luminosity (>13.6 eV) [erg/s] (AU)
        Ln : ndarray
            Non-ionizing luminosity (<13.6 eV) [erg/s] (AU)
        Lbol : ndarray
            Bolometric luminosity [erg/s] (AU)
        Lmech_W : ndarray
            Wind mechanical luminosity [erg/s] (AU)
        Lmech_SN : ndarray
            SN mechanical luminosity [erg/s] (AU)
        Lmech_total : ndarray
            Total mechanical luminosity (winds + SN) [erg/s] (AU)
        pdot_W : ndarray
            Wind momentum rate [M_sun·pc/Myr²] (AU)
        pdot_SN : ndarray
            SN momentum rate [M_sun·pc/Myr²] (AU)
        pdot_total : ndarray
            Total momentum rate (winds + SN) [M_sun·pc/Myr²] (AU)

    Raises
    ------
    ValueError
        If f_mass <= 0 or is NaN/inf
        If unsupported metallicity or black hole cutoff mass
    FileNotFoundError
        If SB99 file not found

    Notes
    -----
    - All arrays have t=0 prepended with initial values for interpolation
    - Thermal efficiency and cold mass corrections are applied to winds and SN
    - Wind velocity: v_wind = 2 * Lmech_W / pdot_W (after corrections)
    - SN velocity: from params['FB_vSN'] (after corrections)

    Examples
    --------
    >>> SB99_data = read_SB99(f_mass=1.0, params=params)
    >>> t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total = SB99_data
    >>> print(f"At t={t[50]:.3f} Myr: Wind Lmech={Lmech_W[50]:.3e}, SN Lmech={Lmech_SN[50]:.3e}")
    """

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if not np.isfinite(f_mass) or f_mass <= 0:
        raise ValueError(
            f"Invalid f_mass: {f_mass}. Must be finite and > 0."
        )

    required_keys = [
        'sps_path',
        'FB_mColdWindFrac', 'FB_thermCoeffWind',
        'FB_mColdSNFrac', 'FB_thermCoeffSN', 'FB_vSN'
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Required parameter '{key}' missing from params")

    logger.info(f"Reading SB99 data with f_mass = {f_mass}")

    # =========================================================================
    # STEP 1: READ FILE
    # =========================================================================
    # sps_path is already a resolved file path by the time we get here — see
    # _get_legacy_sb99_filename in read_param.py for the legacy fallback path.

    filepath = params['sps_path'].value

    logger.debug(f"Loading SB99 file: {filepath}")

    try:
        SB99_file = np.loadtxt(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SB99 file not found: {filepath}\n"
            "If sps_path was set explicitly, verify the path exists.\n"
            "If the legacy SB99 grammar was used (sps_path = def_path),\n"
            "check SB99_mass, SB99_rotation, ZCloud, and SB99_BHCUT in your .param."
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

    # Bolometric luminosity: log₁₀(erg/s) → Msun·pc²/Myr³ (AU)
    Lbol = 10**SB99_file[:, 3] * f_mass * cvt.L_cgs2au

    # Mechanical luminosity (winds + SN): log₁₀(erg/s) → Msun·pc²/Myr³ (AU)
    Lmech = 10**SB99_file[:, 4] * f_mass * cvt.L_cgs2au

    # Wind momentum rate: log₁₀(g·cm/s²) → Msun·pc/Myr² (AU)
    pdot_wind_raw = 10**SB99_file[:, 5] * f_mass * cvt.pdot_cgs2au

    # Wind mechanical luminosity: log₁₀(erg/s) → Msun·pc²/Myr³ (AU)
    Lmech_wind_raw = 10**SB99_file[:, 6] * f_mass * cvt.L_cgs2au

    # Validate all arrays are finite
    for name, arr in [('t', t), ('Qi', Qi), ('fi', fi), ('Lbol', Lbol),
                       ('Lmech', Lmech), ('pdot_wind_raw', pdot_wind_raw),
                       ('Lmech_wind_raw', Lmech_wind_raw)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in {name} array from SB99 file")

    # =========================================================================
    # STEP 3: CALCULATE DERIVED VALUES
    # =========================================================================

    # Ionizing and non-ionizing luminosity (13.6 eV threshold)
    Li = Lbol * fi
    Ln = Lbol * (1 - fi)

    # Mechanical luminosity from SN only (before corrections)
    Lmech_SN_raw = Lmech - Lmech_wind_raw

    # Validate no negative values
    if np.any(Lmech_SN_raw < 0):
        logger.warning(
            "Negative SN mechanical luminosity detected. "
            "This may indicate Lmech_wind > Lmech in SB99 file. "
            "Setting negative values to zero."
        )
        Lmech_SN_raw = np.maximum(Lmech_SN_raw, 0)

    # =========================================================================
    # STEP 4: SCALE WIND PARAMETERS (thermal efficiency + cold mass)
    # =========================================================================

    # Break down into mass loss and velocity
    # Protect against division by zero
    Mdot_wind = pdot_wind_raw ** 2 / (2 * np.maximum(Lmech_wind_raw, EPSILON))
    velocity_wind = 2 * Lmech_wind_raw / np.maximum(pdot_wind_raw, EPSILON)

    # Add fraction of mass injected due to sweeping cold material
    Mdot_wind *= (1 + params['FB_mColdWindFrac'].value)

    # Modify terminal velocity according to:
    # 1) thermal efficiency and 2) cold mass content
    velocity_wind *= np.sqrt(params['FB_thermCoeffWind'].value / (1. + params['FB_mColdWindFrac'].value))

    # Convert back to momentum rate and luminosity
    pdot_wind = Mdot_wind * velocity_wind
    Lmech_wind = 0.5 * Mdot_wind * velocity_wind**2

    # =========================================================================
    # STEP 5: SCALE SN PARAMETERS (thermal efficiency + cold mass)
    # =========================================================================

    # Get SN velocity from params
    velocity_SN = params['FB_vSN'].value

    # Break down into mass loss rate
    # Protect against division by zero
    Mdot_SN = 2 * Lmech_SN_raw / np.maximum(velocity_SN**2, EPSILON)

    # Add fraction of mass injected due to sweeping cold material
    Mdot_SN *= (1 + params['FB_mColdSNFrac'].value)

    # Modify terminal velocity according to thermal efficiency and cold mass
    velocity_SN *= np.sqrt(params['FB_thermCoeffSN'].value / (1. + params['FB_mColdSNFrac'].value))

    # Convert back to momentum rate and luminosity
    pdot_SN = Mdot_SN * velocity_SN
    Lmech_SN = 0.5 * Mdot_SN * velocity_SN**2

    # =========================================================================
    # STEP 6: CALCULATE TOTALS
    # =========================================================================

    # Total mechanical luminosity and momentum injection rate
    Lmech_total = Lmech_SN + Lmech_wind
    pdot_total = pdot_SN + pdot_wind

    # =========================================================================
    # STEP 7: INSERT t=0 FOR INTERPOLATION
    # =========================================================================

    # Insert initial values at t=0 for proper interpolation
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])

    # Insert separated wind and SN components
    Lmech_W = np.insert(Lmech_wind, 0, Lmech_wind[0])
    Lmech_SN = np.insert(Lmech_SN, 0, Lmech_SN[0])
    Lmech_total = np.insert(Lmech_total, 0, Lmech_total[0])

    pdot_W = np.insert(pdot_wind, 0, pdot_wind[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])
    pdot_total = np.insert(pdot_total, 0, pdot_total[0])

    logger.info(
        f"SB99 data processed: {len(t)} time points, "
        f"t_max={t[-1]:.2f} Myr"
    )

    # Return all separated components
    return [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
            pdot_W, pdot_SN, pdot_total]


def get_interpolation(SB99, ftype='cubic'):
    """
    Create cubic interpolation functions for SB99 feedback data.

    This function takes the output from read_SB99() and creates scipy
    interpolation functions for all feedback parameters, with properly
    separated wind and SN components.

    Parameters
    ----------
    SB99 : list
        Data array from read_SB99():
        [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]
    ftype : str, optional
        Interpolation type for scipy.interpolate.interp1d.
        Options: 'linear', 'cubic' (default), 'quadratic', etc.
        Cubic is recommended for small-value interpolations.

    Returns
    -------
    SB99f : dict
        Dictionary of interpolation functions with keys matching variable names:
        - 'fQi' : Ionizing photon rate interpolator
        - 'fLi' : Ionizing luminosity interpolator
        - 'fLn' : Non-ionizing luminosity interpolator
        - 'fLbol' : Bolometric luminosity interpolator
        - 'fLmech_W' : Wind mechanical luminosity interpolator
        - 'fLmech_SN' : SN mechanical luminosity interpolator
        - 'fLmech_total' : Total mechanical luminosity interpolator
        - 'fpdot_W' : Wind momentum rate interpolator
        - 'fpdot_SN' : SN momentum rate interpolator
        - 'fpdot_total' : Total momentum rate interpolator

    Notes
    -----
    - All interpolator keys match variable naming convention
    - Wind components use '_W' suffix
    - SN components use '_SN' suffix
    - Total components use '_total' suffix

    Examples
    --------
    >>> SB99_data = read_SB99(f_mass=1.0, params=params)
    >>> SB99f = get_interpolation(SB99_data, ftype='cubic')
    >>> t = 5.0  # Myr
    >>> Lmech_W = SB99f['fLmech_W'](t)
    >>> Lmech_SN = SB99f['fLmech_SN'](t)
    >>> Lmech_total = SB99f['fLmech_total'](t)
    >>> v_mech_total = 2.0 * Lmech_W / SB99f['fpdot_W'](t)  # Correct formula!
    """

    # Unpack all SB99 values (with separated components)
    [t_Myr, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
     pdot_W, pdot_SN, pdot_total] = SB99

    # Create interpolation functions for all feedback parameters
    fQi = scipy.interpolate.interp1d(t_Myr, Qi, kind=ftype)
    fLi = scipy.interpolate.interp1d(t_Myr, Li, kind=ftype)
    fLn = scipy.interpolate.interp1d(t_Myr, Ln, kind=ftype)
    fLbol = scipy.interpolate.interp1d(t_Myr, Lbol, kind=ftype)

    # Mechanical luminosity interpolators (separated with consistent naming!)
    fLmech_W = scipy.interpolate.interp1d(t_Myr, Lmech_W, kind=ftype)
    fLmech_SN = scipy.interpolate.interp1d(t_Myr, Lmech_SN, kind=ftype)
    fLmech_total = scipy.interpolate.interp1d(t_Myr, Lmech_total, kind=ftype)

    # Momentum rate interpolators (separated with consistent naming!)
    fpdot_W = scipy.interpolate.interp1d(t_Myr, pdot_W, kind=ftype)
    fpdot_SN = scipy.interpolate.interp1d(t_Myr, pdot_SN, kind=ftype)
    fpdot_total = scipy.interpolate.interp1d(t_Myr, pdot_total, kind=ftype)

    # Dictionary of all interpolators with consistent key naming
    SB99f = {
        'fQi': fQi,
        'fLi': fLi,
        'fLn': fLn,
        'fLbol': fLbol,
        'fLmech_W': fLmech_W,            # Consistent: variable is Lmech_W, key is fLmech_W
        'fLmech_SN': fLmech_SN,
        'fLmech_total': fLmech_total,
        'fpdot_W': fpdot_W,              # Consistent: variable is pdot_W, key is fpdot_W
        'fpdot_SN': fpdot_SN,
        'fpdot_total': fpdot_total
    }

    return SB99f
