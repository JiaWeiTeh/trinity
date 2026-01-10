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


# TODO: Implement interpolation function for in-between metallicities/cluster
    # : Add fmet, where metallicity scaling due to non-existent SB99 file

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
        - path_sps : str, path to SB99 files
        - SB99_mass, SB99_rotation, SB99_BHCUT, ZCloud : SB99 file selection
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

    # Mechanical luminosity (winds + SN): log₁₀(erg/s) → erg/s (AU)
    Lmech = 10**SB99_file[:, 4] * f_mass * cvt.L_cgs2au

    # Wind momentum rate: log₁₀(g·cm/s²) → g·cm/s² (AU)
    pdot_wind_raw = 10**SB99_file[:, 5] * f_mass * cvt.pdot_cgs2au

    # Wind mechanical luminosity: log₁₀(erg/s) → erg/s (AU)
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


def get_filename(params):
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
    SB99_mass = params.get('SB99_mass').value
    SB99_rotation = params.get('SB99_rotation').value
    ZCloud = params.get('ZCloud').value
    SB99_BHCUT = params.get('SB99_BHCUT').value

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
    >>> vWind = 2.0 * Lmech_W / SB99f['fpdot_W'](t)  # Correct formula!
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
