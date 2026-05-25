#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:06:39 2023

@author: Jia Wei Teh

Loader for SPS (stellar-population-synthesis) feedback time-series data.

Two entry points into the file:

  read_SB99(f_mass, params)
      Top-level dispatcher. Picks _read_sb99_legacy or _read_sb99_user
      based on params['sps_layout_is_legacy'] (set by read_param.py when
      it resolved sps_path). User-mode column maps may now contain
      either integer indices OR header-row names (per ColumnSpec), so
      file_column type alone is no longer a reliable dispatch signal.

  get_interpolation(sps, ftype='cubic')
      Wraps the 11-array tuple returned by read_SB99() in scipy cubic
      interpolators on `params['sps_f']`.

The legacy branch is byte-equivalent to the pre-refactor SB99-only
loader (see analysis/sb99-refactor-audit.md §8 Invariants). SB99
specifics that survive in this module are intentional: the file is
still called `read_SB99.py`, the helper is `_read_sb99_legacy`, and
the symbol `read_SB99` is the public entry point, per audit §14
question 1 ("symbols/file path stay; SB99 remains the canonical SPS
in this codebase").
"""

import numpy as np
import scipy
import sys
import logging

import src._functions.unit_conversions as cvt
import src.sb99.sps_columns as sps_columns

# Initialize logger for this module
logger = logging.getLogger(__name__)

# Physical constants
EPSILON = 1e-100  # Small number to prevent division by zero


def read_SB99(f_mass, params):
    """
    Read and process Starburst99 stellar feedback data.

    Dispatches to either the legacy SB99 7-column positional loader
    (`_read_sb99_legacy`) or the user-defined column-map loader
    (`_read_sb99_user`) based on the `params['sps_column_map']` layout.
    The legacy branch is byte-equivalent to the pre-refactor SB99-only
    implementation (see analysis/sb99-refactor-audit.md §8 Invariants).

    Files for the legacy SB99 grid use cgs units (with time in years);
    every output of this function is converted to astronomical units
    (Msun, pc, Myr).

    Parameters
    ----------
    f_mass : float
        Cluster mass fraction (f_mass = M_cluster / sps_refmass). Computed
        by main.py before this function is called.
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
        If f_mass <= 0 or is NaN/inf, or if the file shape is invalid.
        (Unsupported legacy-grammar metallicity / BH cutoff is raised
        earlier, by _get_legacy_sb99_filename in read_param.py.)
    FileNotFoundError
        If sps_path points to a file that does not exist.

    Notes
    -----
    - All arrays have t=0 prepended with initial values for interpolation
    - Thermal efficiency and cold mass corrections are applied to winds and SN
    - Wind velocity: v_wind = 2 * Lmech_W / pdot_W (after corrections)
    - SN velocity: from params['FB_vSN'] (after corrections)

    Examples
    --------
    >>> sps_data = read_SB99(f_mass=1.0, params=params)
    >>> t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total = sps_data
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
        'sps_path', 'sps_column_map', 'sps_layout_is_legacy',
        'FB_mColdWindFrac', 'FB_thermCoeffWind',
        'FB_mColdSNFrac', 'FB_thermCoeffSN', 'FB_vSN'
    ]

    for key in required_keys:
        if key not in params:
            raise KeyError(f"Required parameter '{key}' missing from params")

    logger.info(f"Reading SPS data with f_mass = {f_mass}")

    # =========================================================================
    # DISPATCH: legacy positional layout vs user-defined column layout
    # =========================================================================
    # Dispatch on the explicit sps_layout_is_legacy flag set by read_param.py
    # at the time sps_path was resolved. We can't infer this from the
    # column_map's ColumnSpec types anymore because, as of the dual-mode
    # change, user-mode column maps may also contain integer file_column
    # values (when the user writes sps_col_t 0 yr linear etc.). The legacy
    # code path stays byte-equivalent to the pre-refactor SB99-only loader;
    # operator order matters for ULP-level equivalence (see audit §8
    # Invariants).

    column_map = params['sps_column_map'].value
    filepath = params['sps_path'].value
    is_legacy_layout = params['sps_layout_is_legacy'].value

    if is_legacy_layout:
        return _read_sb99_legacy(filepath, f_mass, params)
    else:
        return _read_sb99_user(filepath, f_mass, params, column_map)


def _read_sb99_legacy(filepath, f_mass, params):
    """Legacy SB99 7-column positional loader.

    Byte-equivalent to the pre-refactor SB99-only loader for every
    (mCluster, SB99_mass, SB99_rotation, SB99_BHCUT, ZCloud, FB_*)
    combination. The operator ordering and arithmetic here is
    intentionally preserved verbatim so `np.array_equal` against
    pre-refactor goldens passes (audit §8 Invariants).

    Used whenever sps_path was resolved from the legacy SB99 grammar
    (sps_path = def_path at config-load time).
    """

    logger.debug(f"Loading SB99 file (legacy positional layout): {filepath}")

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
            f"Invalid SB99 file format in {filepath}. "
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

    # Ionizing fraction: SB99 stores log10(fi), we want linear fi.
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

    # Validate strict monotonicity of t — scipy.interpolate.interp1d
    # (called later in get_interpolation) requires it, and its native
    # error is cryptic. This produces a useful message at load time
    # pointing at the file + the first offending row.
    sps_columns.validate_t_monotonic(t, filepath)

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


def _read_sb99_user(filepath, f_mass, params, column_map):
    """User-defined SPS column layout via sps_col_* declarations.

    Loads any .txt or .csv file (delimiter auto-sniffed, header auto-
    detected, '#'-comment lines skipped), applies per-column unit
    conversion and mass scaling using the canonical registry in
    sps_columns.py, then runs the same FB_* correction pipeline as the
    legacy loader. Each ColumnSpec.file_column is either a 0-based
    integer index (works on any file) or a string name resolved against
    the file's header row.

    Missing optional canonicals fall back to the existing derivations:

      - Li, Ln       <- Lbol * fi, Lbol * (1 - fi)      [if not supplied]
      - Lmech_SN_raw <- Lmech_total - Lmech_W           [if Lmech_SN absent]
      - Mdot_SN      <- 2 * Lmech_SN_raw / v_SN^2       [if Mdot_SN absent]
      - v_SN         <- params['FB_vSN'].value          [if v_SN absent]
      - pdot_SN      <- Mdot_SN_modified * v_SN_mod     [if pdot_SN absent]

    User-supplied columns plug into the pipeline at the points indicated
    above; FB_mColdSNFrac / FB_thermCoeffSN still apply on top (see audit
    §10 PR-2). Note this path does NOT guarantee bitwise equivalence
    to the legacy loader because the operator ordering of unit
    conversions differs by ULPs — equivalence to legacy is tested at
    the JSONL snapshot level (rtol=1e-12), not at the loader level
    (audit §11 Test strategy).
    """

    logger.debug(f"Loading SPS file (user-defined column layout): {filepath}")

    try:
        raw_cols = sps_columns.load_user_columns(filepath, column_map)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SPS file not found: {filepath}\n"
            "sps_path is set explicitly; verify the path exists and points "
            "to a readable file."
        )

    # Per-column conversion + mass scaling.
    cols = {}
    for canonical, spec in column_map.items():
        arr = sps_columns.convert_to_canonical_au(
            raw_cols[canonical], canonical, spec.units, spec.log
        )
        if sps_columns.CANONICALS[canonical].mass_scaled:
            arr = arr * f_mass
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Non-finite values in '{canonical}' from {filepath}"
            )
        cols[canonical] = arr

    # Validate strict monotonicity of t — scipy.interpolate.interp1d
    # (called later in get_interpolation) requires it, and its native
    # error is cryptic. This produces a useful message at load time
    # pointing at the file + the first offending row. Common cause:
    # the file's time column was written with too few significant
    # figures (e.g. '%.2E' collapses 1.001e7, 1.002e7 to '1.00E+007').
    sps_columns.validate_t_monotonic(cols['t'], filepath)

    # Derive Li, Ln if not supplied (matches legacy 13.6 eV behaviour when
    # only fi is given; bypassed entirely when both Li and Ln are present —
    # this is what closes audit hot-spot #5).
    if 'Li' not in cols:
        cols['Li'] = cols['Lbol'] * cols['fi']
    if 'Ln' not in cols:
        cols['Ln'] = cols['Lbol'] * (1.0 - cols['fi'])

    # Derive Lmech_SN_raw if not supplied. Validation in read_param.py
    # ensures at least one of (Lmech_SN, Lmech_total) is present.
    if 'Lmech_SN' in cols:
        Lmech_SN_raw = cols['Lmech_SN']
    else:
        Lmech_SN_raw = cols['Lmech_total'] - cols['Lmech_W']

    if np.any(Lmech_SN_raw < 0):
        logger.warning(
            "Negative SN mechanical luminosity detected; clamping to zero. "
            "Check sps_col_Lmech_SN / sps_col_Lmech_total inputs."
        )
        Lmech_SN_raw = np.maximum(Lmech_SN_raw, 0)

    # === WIND corrections (same math as the legacy path) ===
    Lmech_wind_raw = cols['Lmech_W']
    pdot_wind_raw = cols['pdot_W']

    Mdot_wind = pdot_wind_raw ** 2 / (2 * np.maximum(Lmech_wind_raw, EPSILON))
    velocity_wind = 2 * Lmech_wind_raw / np.maximum(pdot_wind_raw, EPSILON)
    Mdot_wind = Mdot_wind * (1 + params['FB_mColdWindFrac'].value)
    velocity_wind = velocity_wind * np.sqrt(
        params['FB_thermCoeffWind'].value /
        (1.0 + params['FB_mColdWindFrac'].value)
    )
    pdot_wind = Mdot_wind * velocity_wind
    Lmech_wind = 0.5 * Mdot_wind * velocity_wind ** 2

    # === SN corrections (with user-override pluggability) ===
    if 'v_SN' in cols:
        velocity_SN_base = cols['v_SN']
    else:
        velocity_SN_base = params['FB_vSN'].value

    if 'Mdot_SN' in cols:
        Mdot_SN = np.array(cols['Mdot_SN'], copy=True)
    else:
        Mdot_SN = 2 * Lmech_SN_raw / np.maximum(velocity_SN_base ** 2, EPSILON)
    Mdot_SN = Mdot_SN * (1 + params['FB_mColdSNFrac'].value)

    velocity_SN_modified = velocity_SN_base * np.sqrt(
        params['FB_thermCoeffSN'].value /
        (1.0 + params['FB_mColdSNFrac'].value)
    )

    if 'pdot_SN' in cols:
        pdot_SN = cols['pdot_SN']
    else:
        pdot_SN = Mdot_SN * velocity_SN_modified

    Lmech_SN_final = 0.5 * Mdot_SN * velocity_SN_modified ** 2

    # === Totals ===
    Lmech_total = Lmech_SN_final + Lmech_wind
    pdot_total = pdot_SN + pdot_wind

    # Convenience aliases
    t = cols['t']
    Qi = cols['Qi']
    Li = cols['Li']
    Ln = cols['Ln']
    Lbol = cols['Lbol']
    Lmech_W = Lmech_wind
    Lmech_SN = Lmech_SN_final
    pdot_W = pdot_wind

    # === t=0 prepend (idempotent — skip if the file already starts at t=0) ===
    if len(t) == 0 or t[0] != 0.0:
        t = np.insert(t, 0, 0.0)
        Qi = np.insert(Qi, 0, Qi[0])
        Li = np.insert(Li, 0, Li[0])
        Ln = np.insert(Ln, 0, Ln[0])
        Lbol = np.insert(Lbol, 0, Lbol[0])
        Lmech_W = np.insert(Lmech_W, 0, Lmech_W[0])
        Lmech_SN = np.insert(Lmech_SN, 0, Lmech_SN[0])
        Lmech_total = np.insert(Lmech_total, 0, Lmech_total[0])
        pdot_W = np.insert(pdot_W, 0, pdot_W[0])
        pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])
        pdot_total = np.insert(pdot_total, 0, pdot_total[0])

    logger.info(
        f"SPS data processed: {len(t)} time points, "
        f"t_max={t[-1]:.2f} Myr"
    )

    return [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
            pdot_W, pdot_SN, pdot_total]


def get_interpolation(sps, ftype='cubic'):
    """
    Create cubic interpolation functions for SPS feedback data.

    This function takes the output from read_SB99() and creates scipy
    interpolation functions for all feedback parameters, with properly
    separated wind and SN components.

    Parameters
    ----------
    sps : list
        Data array from read_SB99():
        [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]
    ftype : str, optional
        Interpolation type for scipy.interpolate.interp1d.
        Options: 'linear', 'cubic' (default), 'quadratic', etc.
        Cubic is recommended for small-value interpolations.

    Returns
    -------
    sps_f : dict
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
    >>> sps_data = read_SB99(f_mass=1.0, params=params)
    >>> sps_f = get_interpolation(sps_data, ftype='cubic')
    >>> t = 5.0  # Myr
    >>> Lmech_W = sps_f['fLmech_W'](t)
    >>> Lmech_SN = sps_f['fLmech_SN'](t)
    >>> Lmech_total = sps_f['fLmech_total'](t)
    >>> v_mech_total = 2.0 * Lmech_W / sps_f['fpdot_W'](t)  # Correct formula!
    """

    # Unpack all SPS values (with separated components)
    [t_Myr, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
     pdot_W, pdot_SN, pdot_total] = sps

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
    sps_f = {
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

    return sps_f
