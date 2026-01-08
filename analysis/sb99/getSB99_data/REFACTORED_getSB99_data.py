#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFACTORED VERSION of getSB99_data.py

WARNING: This file is LEGACY CODE from WARPFIELD and is likely UNUSED in TRINITY.
         Use read_SB99.py instead (simpler, working, TRINITY-native version).

Original Author: WARPFIELD team
Refactored: 2026-01-08

IMPROVEMENTS OVER ORIGINAL:
1. ✓ FIXED syntax error (Lines 99-102) - incomplete function call
2. ✓ FIXED array indexing bug (Line 261) - t2[-2] → t2[-1]
3. ✓ FIXED deprecated collections.Mapping → collections.abc.Mapping
4. ✓ Added logging instead of print statements
5. ✓ Added type hints
6. ✓ Better error messages
7. ✓ Removed commented-out code

CRITICAL FIXES:
- Lines 99-102: SYNTAX ERROR - incomplete getSB99_data_interp() call
  - Original: Missing closing parenthesis, arguments 4 and 5
  - Fixed: Completed function call with SB99_file_Z0014 and 1.0
- Line 261: ARRAY INDEXING BUG
  - Original: tend2 = t2[-2] (second-to-last element)
  - Fixed: tend2 = t2[-1] (last element, consistent with tend1)
- Lines 445, 460: DEPRECATED IMPORT
  - Original: collections.Mapping (removed in Python 3.10)
  - Fixed: collections.abc.Mapping

MISSING DEPENDENCIES (from WARPFIELD):
This file requires the following WARPFIELD modules that are NOT in TRINITY:
- auxiliary_functions (aliased as 'aux')
  - aux.printl() - logging function
  - aux.find_nearest_lower() - array search function
- warp_nameparser
  - get_SB99_filename() - filename constructor
- init (aliased as 'i')
  - Multiple configuration parameters

TO USE THIS FILE:
Either:
1. Port the WARPFIELD modules to TRINITY, OR
2. Replace with stubs/alternatives, OR
3. Use read_SB99.py instead (RECOMMENDED)

RECOMMENDATION: Delete this file and use read_SB99.py for all SB99 operations.
"""

import logging
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import sys
import os
import scipy.interpolate
import warnings
import collections.abc
import pathlib

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# WARPFIELD DEPENDENCY STUBS
# ============================================================================
# These are PLACEHOLDERS for missing WARPFIELD modules.
# Replace with actual implementations if porting to TRINITY.

class AuxiliaryFunctionsStub:
    """Stub for WARPFIELD auxiliary_functions module."""

    @staticmethod
    def printl(message: str, verbose: int = 0):
        """Stub for aux.printl()."""
        if verbose > 0:
            logger.info(message)

    @staticmethod
    def find_nearest_lower(arr: np.ndarray, val: float) -> int:
        """
        Find index of largest array element that is <= val.

        Stub for aux.find_nearest_lower().
        """
        idx = np.where(arr <= val)[0]
        if len(idx) == 0:
            return 0
        return idx[-1]


class WarpNameparserStub:
    """Stub for WARPFIELD warp_nameparser module."""

    @staticmethod
    def get_SB99_filename(
        Z: float,
        rotation: bool,
        BHcutoff: float,
        SB99_mass: float
    ) -> str:
        """
        Stub for warp_nameparser.get_SB99_filename().

        This should construct SB99 filename from parameters.
        Replace with actual implementation.
        """
        raise NotImplementedError(
            "warp_nameparser.get_SB99_filename() is not implemented in TRINITY. "
            "Use read_SB99.get_filename() instead."
        )


class InitStub:
    """Stub for WARPFIELD init module."""

    force_SB99file = 0
    SB99_mass = 1e6
    SB99cloudy_file = ''
    f_Mcold_W = 0.0
    thermcoeff_clwind = 1.0
    f_Mcold_SN = 0.0
    thermcoeff_SN = 1.0
    v_SN = 1e4  # cm/s


# Create stub instances
aux = AuxiliaryFunctionsStub()
warp_nameparser = WarpNameparserStub()
i = InitStub()

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================


def getSB99_main(
    Zism: float,
    rotation: bool = True,
    f_mass: float = 1e6,
    BHcutoff: float = 120.0,
    return_format: str = "array"
) -> Tuple[Union[List, Dict], Dict]:
    """
    Get Starburst99 data and interpolation functions.

    Parameters
    ----------
    Zism : float
        Metallicity in solar units
    rotation : bool, optional
        Include stellar rotation (default: True)
    f_mass : float, optional
        Cluster mass in solar masses (default: 1e6)
    BHcutoff : float, optional
        Black hole cutoff mass in solar masses (default: 120)
    return_format : str, optional
        Return format: "array" or "dict" (default: "array")

    Returns
    -------
    SB99_data : list or dict
        SB99 feedback data
    SB99f : dict
        Interpolation functions

    Notes
    -----
    This is the main entry point for loading SB99 data.
    """

    SB99_data = load_stellar_tracks(
        Zism,
        rotation=rotation,
        f_mass=f_mass,
        BHcutoff=BHcutoff
    )

    SB99f = make_interpfunc(SB99_data)

    if return_format == "dict":
        SB99_data = return_as_dict(SB99_data)
    elif return_format == "array":
        SB99_data = return_as_array(SB99_data)

    return SB99_data, SB99f


def load_stellar_tracks(
    Zism: float,
    rotation: bool = True,
    f_mass: float = 1.0,
    BHcutoff: float = 120.0,
    force_file: Union[str, int] = 0,
    test_plot: bool = False,
    log_t: bool = False,
    tmax: float = 30.0,
    return_format: str = "array"
) -> Union[List, Dict]:
    """
    Load stellar evolution tracks from SB99 files.

    Parameters
    ----------
    Zism : float
        Metallicity in solar units
    rotation : bool, optional
        Rotating stars (default: True)
    f_mass : float, optional
        Cluster mass in units of 1e6 Msun (default: 1.0)
    BHcutoff : float, optional
        Black hole cutoff mass (default: 120 Msun)
    force_file : str or int, optional
        Force specific SB99 file (default: 0, auto-select)
    test_plot : bool, optional
        Show debug plot (default: False)
    log_t : bool, optional
        Logarithmic time axis in plot (default: False)
    tmax : float, optional
        Maximum time for plot in Myr (default: 30)
    return_format : str, optional
        Return format: "array" or "dict" (default: "array")

    Returns
    -------
    Data : list or dict
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo]

    Notes
    -----
    CRITICAL FIX: Lines 99-102 had incomplete function call (SYNTAX ERROR).
    Now properly calls getSB99_data_interp() with all 5 arguments.
    """

    # Get SB99 filenames
    # NOTE: This will fail unless warp_nameparser is properly implemented
    try:
        SB99_file_Z0002 = warp_nameparser.get_SB99_filename(
            0.15, rotation, BHcutoff, SB99_mass=i.SB99_mass
        )
        SB99_file_Z0014 = warp_nameparser.get_SB99_filename(
            1.0, rotation, BHcutoff, SB99_mass=i.SB99_mass
        )
    except NotImplementedError:
        raise NotImplementedError(
            "This file requires WARPFIELD's warp_nameparser module. "
            "Use read_SB99.py instead for TRINITY-native SB99 loading."
        )

    # Case 1: Specific file is forced
    if force_file != 0:
        SB99file = force_file
        warnings.warn(f"Forcing WARPFIELD to use SB99 file: {force_file}")
        warnings.warn("Make sure you provided correct metallicity and mass scaling")

        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = \
            getSB99_data(SB99file, f_mass=f_mass, test_plot=test_plot, log_t=log_t, tmax=tmax)

        if SB99file != i.SB99cloudy_file + '.txt':
            warnings.warn("SB99 file and SB99cloudy_file do not agree!")
            logger.warning(f"SB99file: {SB99file}")
            logger.warning(f"SB99cloudy_file: {i.SB99cloudy_file}")

    # Case 2: Metallicity interpolation (CRITICAL FIX: was incomplete!)
    elif 0.15 < Zism < 1.0:

        # FIXED: Lines 99-102 had SYNTAX ERROR (incomplete function call)
        # Original was missing:
        #   - Closing parenthesis
        #   - 4th argument: SB99_file_Z0014
        #   - 5th argument: 1.0
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = \
            getSB99_data_interp(
                Zism,
                SB99_file_Z0002,
                0.15,
                SB99_file_Z0014,  # ADDED: missing 4th argument
                1.0,              # ADDED: missing 5th argument
                f_mass=f_mass,
                test_plot=test_plot,
                log_t=log_t,
                tmax=tmax
            )

    # Case 3: Exact metallicity match
    elif Zism == 1.0 or (0.14 <= Zism <= 0.15):

        if Zism == 1.0:
            SB99file = SB99_file_Z0014
        elif 0.14 <= Zism <= 0.15:
            SB99file = SB99_file_Z0002

        if SB99file != i.SB99cloudy_file + '.txt':
            warnings.warn("SB99 file and SB99cloudy_file do not agree!")
            logger.warning(f"SB99file: {SB99file}")
            logger.warning(f"SB99cloudy_file: {i.SB99cloudy_file}")

        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = \
            getSB99_data(SB99file, f_mass=f_mass, test_plot=test_plot, log_t=log_t, tmax=tmax)

    # Case 4: Metallicity out of range (extrapolation)
    else:
        warnings.warn(
            f"Metallicity Z = {Zism} is out of range. "
            "Using closest track and scaling winds linearly."
        )

        if Zism < 0.15:
            SB99file = SB99_file_Z0002
            Zism_rel = 0.15
        elif Zism > 1.0:
            SB99file = SB99_file_Z0014
            Zism_rel = 1.0

        logger.info(f"Using SB99 file: {SB99file}")

        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = \
            getSB99_data(SB99file, f_mass=f_mass, f_met=Zism / Zism_rel)

    Data = [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo]

    if return_format == 'dict':
        Data_out = return_as_dict(Data)
    elif return_format == 'array':
        Data_out = return_as_array(Data)
    else:
        Data_out = Data

    return Data_out


def getSB99_data(
    file: str,
    f_mass: float = 1.0,
    f_met: float = 1.0,
    test_plot: bool = False,
    log_t: bool = False,
    tmax: float = 30.0,
    verbose: int = 0,
    ylim: List[float] = [37.0, 43.0]
) -> List[np.ndarray]:
    """
    Load SB99 data from file.

    Similar to read_SB99.read_SB99() but with WARPFIELD conventions.

    Parameters
    ----------
    file : str
        Path to SB99 file
    f_mass : float, optional
        Mass scaling factor (default: 1.0)
    f_met : float, optional
        Metallicity scaling factor (default: 1.0)
    test_plot : bool, optional
        Show debug plot (default: False)
    log_t : bool, optional
        Logarithmic time axis (default: False)
    tmax : float, optional
        Maximum plot time in Myr (default: 30)
    verbose : int, optional
        Verbosity level (default: 0)
    ylim : list, optional
        Y-axis limits for plot (default: [37, 43])

    Returns
    -------
    [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN] : list of np.ndarray
    """

    aux.printl(f"getSB99_data: mass scaling f_mass = {f_mass:.3f}", verbose=verbose)

    # Try to load file
    if os.path.isfile(file):
        data = np.loadtxt(file)
    elif os.path.isfile(pathlib.Path(__file__).parent / file):
        data = np.loadtxt(pathlib.Path(__file__).parent / file)
    else:
        sys.exit(f"Specified SB99 file does not exist: {file}")

    # Read columns
    t = data[:, 0] / 1e6  # yr → Myr
    Qi = 10.0**data[:, 1] * f_mass
    fi = 10**data[:, 2]
    Lbol = 10**data[:, 3] * f_mass
    Li = fi * Lbol
    Ln = (1.0 - fi) * Lbol

    # Mechanical luminosity before scaling
    pdot_W_tmp = 10**data[:, 5] * f_mass
    Lmech_tmp = 10**data[:, 4] * f_mass
    Lmech_W_tmp = 10**data[:, 6] * f_mass
    Lmech_SN_tmp = Lmech_tmp - Lmech_W_tmp

    # Wind scaling
    Mdot_W, v_W = getMdotv(pdot_W_tmp, Lmech_W_tmp)
    Mdot_W *= f_met * (1.0 + i.f_Mcold_W)
    v_W *= np.sqrt(i.thermcoeff_clwind / (1.0 + i.f_Mcold_W))
    pdot_W, Lmech_W = getpdotLmech(Mdot_W, v_W)

    # SN scaling
    v_SN = i.v_SN
    Mdot_SN = 2.0 * Lmech_SN_tmp / v_SN**2
    Mdot_SN *= (1.0 + i.f_Mcold_SN)
    v_SN *= np.sqrt(i.thermcoeff_SN / (1.0 + i.f_Mcold_SN))
    pdot_SN, Lmech_SN = getpdotLmech(Mdot_SN, v_SN)

    # Total
    Lmech = Lmech_W + Lmech_SN
    pdot = pdot_W + pdot_SN

    # Insert t=0
    t = np.insert(t, 0, 0.0)
    Qi = np.insert(Qi, 0, Qi[0])
    Li = np.insert(Li, 0, Li[0])
    Ln = np.insert(Ln, 0, Ln[0])
    Lbol = np.insert(Lbol, 0, Lbol[0])
    Lmech = np.insert(Lmech, 0, Lmech[0])
    pdot = np.insert(pdot, 0, pdot[0])
    pdot_SN = np.insert(pdot_SN, 0, pdot_SN[0])

    if test_plot:
        testplot(t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN, log_t=log_t, t_max=tmax, ylim=ylim)

    return [t, Qi, Li, Ln, Lbol, Lmech, pdot, pdot_SN]


def getSB99_data_interp(
    Zism: float,
    file1: str,
    Zfile1: float,
    file2: str,
    Zfile2: float,
    f_mass: float = 1.0,
    test_plot: bool = False,
    log_t: bool = False,
    tmax: float = 30.0
) -> List[np.ndarray]:
    """
    Interpolate SB99 data between two metallicities.

    Parameters
    ----------
    Zism : float
        Target metallicity (between Zfile1 and Zfile2)
    file1 : str
        Path to SB99 file for metallicity 1
    Zfile1 : float
        Metallicity 1
    file2 : str
        Path to SB99 file for metallicity 2
    Zfile2 : float
        Metallicity 2 (NOTE: docstring typo fixed - was "Zfile")
    f_mass : float, optional
        Mass scaling factor (default: 1.0)
    test_plot : bool, optional
        Show debug plot (default: False)
    log_t : bool, optional
        Logarithmic time axis (default: False)
    tmax : float, optional
        Maximum plot time in Myr (default: 30)

    Returns
    -------
    [t, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe] : list of np.ndarray
        Interpolated SB99 data

    Notes
    -----
    CRITICAL FIX: Line 261 had array indexing bug.
    Original: tend2 = t2[-2] (second-to-last element)
    Fixed: tend2 = t2[-1] (last element, consistent with tend1)
    """

    # Ensure index 1 is lower metallicity
    if Zfile1 < Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = \
            getSB99_data(file1, f_mass=f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = \
            getSB99_data(file2, f_mass=f_mass)
        Z1 = Zfile1
        Z2 = Zfile2
    elif Zfile1 > Zfile2:
        [t1, Qi1, Li1, Ln1, Lbol1, Lw1, pdot1, pdot_SNe1] = \
            getSB99_data(file2, f_mass=f_mass)
        [t2, Qi2, Li2, Ln2, Lbol2, Lw2, pdot2, pdot_SNe2] = \
            getSB99_data(file1, f_mass=f_mass)
        Z1 = Zfile2
        Z2 = Zfile1
    else:
        raise ValueError(f"Zfile1 and Zfile2 are equal: {Zfile1}")

    # Get end times
    tend1 = t1[-1]

    # CRITICAL FIX: Line 261 had bug
    # Original: tend2 = t2[-2]  (second-to-last element)
    # Fixed:    tend2 = t2[-1]  (last element)
    # Reason: Should be consistent with tend1 = t1[-1]
    tend2 = t2[-1]  # FIXED: was t2[-2]

    tend = np.min([tend1, tend2])

    # Trim arrays to same length
    Qi1 = Qi1[t1 <= tend]
    Li1 = Li1[t1 <= tend]
    Ln1 = Ln1[t1 <= tend]
    Lbol1 = Lbol1[t1 <= tend]
    Lw1 = Lw1[t1 <= tend]
    pdot1 = pdot1[t1 <= tend]
    pdot_SNe1 = pdot_SNe1[t1 <= tend]

    Qi2 = Qi2[t2 <= tend]
    Li2 = Li2[t2 <= tend]
    Ln2 = Ln2[t2 <= tend]
    Lbol2 = Lbol2[t2 <= tend]
    Lw2 = Lw2[t2 <= tend]
    pdot2 = pdot2[t2 <= tend]
    pdot_SNe2 = pdot_SNe2[t2 <= tend]

    t1 = t1[t1 <= tend]
    t2 = t2[t2 <= tend]

    # Verify time arrays match
    if not np.array_equal(t1, t2):
        raise ValueError(
            "Time arrays do not match! "
            "Cannot interpolate between SB99 files with different time grids."
        )

    t = t1

    # Linear interpolation in metallicity
    Qi = (Qi1 * (Z2 - Zism) + Qi2 * (Zism - Z1)) / (Z2 - Z1)
    Li = (Li1 * (Z2 - Zism) + Li2 * (Zism - Z1)) / (Z2 - Z1)
    Ln = (Ln1 * (Z2 - Zism) + Ln2 * (Zism - Z1)) / (Z2 - Z1)
    Lbol = (Lbol1 * (Z2 - Zism) + Lbol2 * (Zism - Z1)) / (Z2 - Z1)
    Lw = (Lw1 * (Z2 - Zism) + Lw2 * (Zism - Z1)) / (Z2 - Z1)
    pdot = (pdot1 * (Z2 - Zism) + pdot2 * (Zism - Z1)) / (Z2 - Z1)
    pdot_SNe = (pdot_SNe1 * (Z2 - Zism) + pdot_SNe2 * (Zism - Z1)) / (Z2 - Z1)

    return [t, Qi, Li, Ln, Lbol, Lw, pdot, pdot_SNe]


def getMdotv(pdot: np.ndarray, Lmech: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mass loss rate and velocity from momentum and energy.

    Parameters
    ----------
    pdot : np.ndarray
        Momentum injection rate [g·cm/s²]
    Lmech : np.ndarray
        Mechanical luminosity [erg/s]

    Returns
    -------
    Mdot : np.ndarray
        Mass loss rate [g/s]
    v : np.ndarray
        Terminal velocity [cm/s]
    """
    Mdot = pdot**2 / (2.0 * Lmech)
    v = 2.0 * Lmech / pdot
    return Mdot, v


def getpdotLmech(Mdot: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate momentum and energy rates from mass loss and velocity.

    Parameters
    ----------
    Mdot : np.ndarray
        Mass loss rate [g/s]
    v : np.ndarray
        Terminal velocity [cm/s]

    Returns
    -------
    pdot : np.ndarray
        Momentum injection rate [g·cm/s²]
    Lmech : np.ndarray
        Mechanical luminosity [erg/s]
    """
    pdot = Mdot * v
    Lmech = 0.5 * Mdot * v**2
    return pdot, Lmech


def make_interpfunc(SB99_data_IN: Union[List, Dict]) -> Dict:
    """
    Create interpolation functions for SB99 data.

    Parameters
    ----------
    SB99_data_IN : list or dict
        SB99 data

    Returns
    -------
    SB99f : dict
        Dictionary of interpolation functions
    """

    SB99_data = return_as_array(SB99_data_IN)
    [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs] = SB99_data

    fQi_cgs = scipy.interpolate.interp1d(t_Myr, Qi_cgs, kind='cubic')
    fLi_cgs = scipy.interpolate.interp1d(t_Myr, Li_cgs, kind='cubic')
    fLn_cgs = scipy.interpolate.interp1d(t_Myr, Ln_cgs, kind='cubic')
    fLbol_cgs = scipy.interpolate.interp1d(t_Myr, Lbol_cgs, kind='cubic')
    fLw_cgs = scipy.interpolate.interp1d(t_Myr, Lw_cgs, kind='cubic')
    fpdot_cgs = scipy.interpolate.interp1d(t_Myr, pdot_cgs, kind='cubic')
    fpdot_SNe_cgs = scipy.interpolate.interp1d(t_Myr, pdot_SNe_cgs, kind='cubic')

    SB99f = {
        'fQi_cgs': fQi_cgs,
        'fLi_cgs': fLi_cgs,
        'fLn_cgs': fLn_cgs,
        'fLbol_cgs': fLbol_cgs,
        'fLw_cgs': fLw_cgs,
        'fpdot_cgs': fpdot_cgs,
        'fpdot_SNe_cgs': fpdot_SNe_cgs
    }

    return SB99f


def return_as_dict(SB99_data: Union[List, Dict]) -> Dict:
    """
    Return SB99 data as dictionary.

    FIXED: collections.Mapping → collections.abc.Mapping (Line 445)

    Parameters
    ----------
    SB99_data : list or dict
        SB99 data

    Returns
    -------
    dict
        SB99 data dictionary
    """

    if isinstance(SB99_data, collections.abc.Mapping):  # FIXED: was collections.Mapping
        return SB99_data
    else:
        [t_evo, Qi_evo, Li_evo, Ln_evo, Lbol_evo, Lw_evo, pdot_evo, pdot_SNe_evo] = SB99_data
        SB99_data_out = {
            't_Myr': t_evo,
            'Qi_cgs': Qi_evo,
            'Li_cgs': Li_evo,
            'Ln_cgs': Ln_evo,
            'Lbol_cgs': Lbol_evo,
            'Lw_cgs': Lw_evo,
            'pdot_cgs': pdot_evo,
            'pdot_SNe_cgs': pdot_SNe_evo
        }
        return SB99_data_out


def return_as_array(SB99_data: Union[List, Dict]) -> List:
    """
    Return SB99 data as array.

    FIXED: collections.Mapping → collections.abc.Mapping (Line 460)

    Parameters
    ----------
    SB99_data : list or dict
        SB99 data

    Returns
    -------
    list
        SB99 data array
    """

    if isinstance(SB99_data, collections.abc.Mapping):  # FIXED: was collections.Mapping
        t_Myr = SB99_data['t_Myr']
        Qi_cgs = SB99_data['Qi_cgs']
        Li_cgs = SB99_data['Li_cgs']
        Ln_cgs = SB99_data['Ln_cgs']
        Lbol_cgs = SB99_data['Lbol_cgs']
        Lw_cgs = SB99_data['Lw_cgs']
        pdot_cgs = SB99_data['pdot_cgs']
        pdot_SNe_cgs = SB99_data['pdot_SNe_cgs']
        SB99_data_out = [t_Myr, Qi_cgs, Li_cgs, Ln_cgs, Lbol_cgs, Lw_cgs, pdot_cgs, pdot_SNe_cgs]
        return SB99_data_out
    else:
        return SB99_data


def testplot(
    t: np.ndarray,
    Qi: np.ndarray,
    Li: np.ndarray,
    Ln: np.ndarray,
    Lbol: np.ndarray,
    Lw: np.ndarray,
    pdot: np.ndarray,
    pdot_SNe: np.ndarray,
    log_t: bool = False,
    t_max: float = 30.0,
    ylim: List[float] = [39.0, 43.0]
):
    """
    Debug plot for SB99 data.

    Parameters
    ----------
    t : np.ndarray
        Time [Myr]
    Qi : np.ndarray
        Ionizing photon rate
    Li : np.ndarray
        Ionizing luminosity
    Ln : np.ndarray
        Non-ionizing luminosity
    Lbol : np.ndarray
        Bolometric luminosity
    Lw : np.ndarray
        Mechanical luminosity
    pdot : np.ndarray
        Momentum injection rate
    pdot_SNe : np.ndarray
        SN momentum injection rate
    log_t : bool, optional
        Logarithmic time axis (default: False)
    t_max : float, optional
        Maximum time (default: 30 Myr)
    ylim : list, optional
        Y-axis limits (default: [39, 43])
    """

    import matplotlib.pyplot as plt

    if log_t:
        plt.semilogx(t, np.log10(Li), 'b', label="$L_i$")
        plt.semilogx(t, np.log10(Ln), 'r', label="$L_n$")
        plt.semilogx(t, np.log10(Lbol), 'g--', label="$L_{bol}$")
        plt.semilogx(t, np.log10(Lw), 'k', label="$L_{wind}$")
        plt.semilogx(t, np.log10(Qi) - 10.0, 'm', label="$Q_{i}-10$")
        plt.semilogx(t, np.log10(pdot) + 10.0, 'c', label="$\\dot{p_{w}}+10$")
    else:
        plt.plot(t, np.log10(Li), 'b', label="$L_i$")
        plt.plot(t, np.log10(Ln), 'r', label="$L_n$")
        plt.plot(t, np.log10(Lbol), 'g--', label="$L_{bol}$")
        plt.plot(t, np.log10(Lw), 'k', label="$L_{wind}$")
        plt.plot(t, np.log10(Qi) - 10.0, 'm', label="$Q_{i}-10$")
        plt.plot(t, np.log10(pdot) + 10.0, 'c', label="$\\dot{p_{w}}+10$")

    plt.xlabel("t in Myr")
    plt.xlim([0.9, t_max])
    plt.ylim(ylim)
    plt.ylabel("log10(L) in erg/s")
    plt.legend()
    plt.show()

    return 0


# ============================================================================
# NOTE: Removed deprecated functions (sum_SB99_old, time_shift_old)
# ============================================================================
