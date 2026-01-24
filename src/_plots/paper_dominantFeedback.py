#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dominant Feedback Grid Plot
===========================

Creates a grid of subplots showing which feedback mechanism dominates
at different time snapshots across a parameter space of cloud mass (mCloud)
and star formation efficiency (SFE).

Each subplot (one per time snapshot):
- X-axis: cloud mass (mCloud)
- Y-axis: star formation efficiency (SFE)
- Color: dominant feedback force (F_grav, F_ram_wind, F_ram_SN, F_ion_out, F_rad)
- White: no data (simulation ended before this time or hasn't reached it)

Usage
-----
    # Default settings
    python paper_dominantFeedback.py

    # Custom parameters
    python paper_dominantFeedback.py --mCloud 1e7 1e8 --sfe 001 020 --times 1.0 1.5 2.0 2.5

    # Full example
    python paper_dominantFeedback.py \\
        --mCloud 1e5 1e6 1e7 1e8 \\
        --sfe 001 010 020 030 \\
        --nCore 1e4 \\
        --times 1.0 1.5 2.0 2.5 \\
        --output-dir /path/to/outputs

Author: TRINITY Team
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import tempfile
import shutil

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


# =============================================================================
# Configuration
# =============================================================================

# Switch to use *_modified/ output folders and save as *_modified.pdf
USE_MODIFIED = False

# F_ram decomposition mode: 'decomposed', 'combined', or 'sn_highlight'
# - 'decomposed': F_ram_wind (blue), F_ram_SN (yellow), F_ram_residual (gray) separately
# - 'combined': single F_ram category (blue)
# - 'sn_highlight': F_ram_thermal (blue, wind+residual), F_ram_SN (yellow) - highlights SN only
DECOMPOSE_MODE = 'decomposed'

# Force field definitions: (key, label, color)
# DECOMPOSED mode: F_ram split into wind/SN/residual (6 force types, indices 0-5)
FORCE_FIELDS_DECOMPOSED = [
    ("F_grav",         "Gravity",           "#2c3e50"),  # Dark blue-gray
    ("F_ram_wind",     "Winds",             "#3498db"),  # Blue (ram from winds)
    ("F_ram_SN",       "Supernovae",        "#DAA520"),  # Golden yellow for SN
    ("F_ram_residual", "Other thermal",     "#7f8c8d"),  # Gray (F_ram - wind - SN)
    ("F_ion_out",      "Photoionised gas",  "#e74c3c"),  # Red
    ("F_rad",          "Radiation",         "#9b59b6"),  # Purple
]

# COMBINED mode: F_ram as single category (4 force types, indices 0-3)
FORCE_FIELDS_COMBINED = [
    ("F_grav",    "Gravity",           "#2c3e50"),  # Dark blue-gray
    ("F_ram",     "Ram pressure",      "#3498db"),  # Blue (combined ram)
    ("F_ion_out", "Photoionised gas",  "#e74c3c"),  # Red
    ("F_rad",     "Radiation",         "#9b59b6"),  # Purple
]

# SN_HIGHLIGHT mode: Only F_ram_SN highlighted, wind+residual grouped as thermal (5 force types)
FORCE_FIELDS_SN_HIGHLIGHT = [
    ("F_grav",         "Gravity",              "#2c3e50"),  # Dark blue-gray
    ("F_ram_thermal",  "Thermal (non-SN)",     "#3498db"),  # Blue (wind + residual combined)
    ("F_ram_SN",       "Supernovae",           "#DAA520"),  # Golden yellow for SN
    ("F_ion_out",      "Photoionised gas",     "#e74c3c"),  # Red
    ("F_rad",          "Radiation",            "#9b59b6"),  # Purple
]

# Active force fields (set based on DECOMPOSE_MODE)
if DECOMPOSE_MODE == 'decomposed':
    FORCE_FIELDS = FORCE_FIELDS_DECOMPOSED
elif DECOMPOSE_MODE == 'sn_highlight':
    FORCE_FIELDS = FORCE_FIELDS_SN_HIGHLIGHT
else:
    FORCE_FIELDS = FORCE_FIELDS_COMBINED

# Special values for missing data (must be negative to distinguish from force indices)
FILE_NOT_FOUND = -1      # Gray: simulation file not found
TIME_OUT_OF_RANGE = -2   # White: time outside simulation range (t > t_max or t < t_min)

# Missing data colors
COLOR_FILE_NOT_FOUND = "#808080"      # Gray
COLOR_TIME_OUT_OF_RANGE = "#ffffff"   # White

# Default parameters
DEFAULT_MCLOUD = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
# DEFAULT_SFE = ["001", "010", "030", "050", "080"] #"020"
DEFAULT_SFE = ["001", "005", "010", "020", "030", "050", "070", "080"] #"020"
# DEFAULT_NCORE = ["1e2", "1e4"]  # List of nCore values - produces one plot per nCore
DEFAULT_NCORE = ["1e3"]  # List of nCore values - produces one plot per nCore
# DEFAULT_TIMES = [1.0, 1.5, 2.0, 2.5]  # Myr
# Note: SN typically starts ~3-4 Myr after star formation
DEFAULT_TIMES = [1.0, 3.0, 5.0]  # Myr - includes times when SN should be active


# Axis mode options:
#   'discrete': equal spacing, categorical labels (default)
#   'continuous': real value spacing (log for mCloud, linear for SFE)
DEFAULT_AXIS_MODE = 'continuous'

# Smoothing options: 'none' or 'interp' (only effective with axis_mode='continuous')
DEFAULT_SMOOTH = 'interp'

# Default directories
DEFAULT_OUTPUT_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"
FIG_DIR = Path(__file__).parent.parent.parent / "fig"


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_snapshot_at_time(output, target_time):
    """
    Get snapshot data at a specific time.

    Returns None if the target time is outside the simulation's data range.
    This prevents interpolation into "unknown territory".

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output
    target_time : float
        Target time in Myr

    Returns
    -------
    dict or None
        Snapshot data dict, or None if time is out of range
    """
    t_min = output.t_min
    t_max = output.t_max

    # Strict bounds checking - no extrapolation
    if target_time < t_min or target_time > t_max:
        return None

    # Use closest snapshot mode to avoid interpolation artifacts
    try:
        snap = output.get_at_time(target_time, mode='closest', quiet=True)
        return snap.data
    except (ValueError, IndexError):
        return None


def get_dominant_force(snapshot, force_fields=None):
    """
    Determine the dominant feedback force from a snapshot.

    Parameters
    ----------
    snapshot : dict
        Snapshot data dictionary
    force_fields : list, optional
        List of (key, label, color) tuples. If None, uses global FORCE_FIELDS.

    Returns
    -------
    int or np.nan
        Index of dominant force, or np.nan if no valid forces
    """
    if snapshot is None:
        return np.nan

    if force_fields is None:
        force_fields = FORCE_FIELDS

    # Use get_force_values to handle computed fields like F_ram_residual
    force_vals = get_force_values(snapshot, force_fields)
    if force_vals is None:
        return np.nan

    forces = []
    for key, _, _ in force_fields:
        val = force_vals.get(key, 0.0)
        if val is None or not np.isfinite(val):
            forces.append(0.0)
        else:
            forces.append(abs(val))

    # Check if we have any valid force data
    total = sum(forces)
    if total == 0 or not np.isfinite(total):
        return np.nan

    return int(np.argmax(forces))


def get_force_values(snapshot, force_fields=None):
    """
    Extract all force values from a snapshot.

    Parameters
    ----------
    snapshot : dict
        Snapshot data dictionary
    force_fields : list, optional
        List of (key, label, color) tuples defining which forces to extract.
        If None, uses global FORCE_FIELDS.

    Returns
    -------
    dict or None
        Dictionary of {force_key: value} or None if snapshot is None
    """
    if snapshot is None:
        return None

    if force_fields is None:
        force_fields = FORCE_FIELDS

    forces = {}
    force_keys = [key for key, _, _ in force_fields]

    # First pass: get raw values for standard fields
    # Skip computed fields (F_ram_residual, F_ram_thermal) - they're computed below
    computed_fields = {"F_ram_residual", "F_ram_thermal"}
    for key, _, _ in force_fields:
        if key in computed_fields:
            continue
        val = snapshot.get(key)
        if val is None or not np.isfinite(val):
            forces[key] = np.nan
        else:
            forces[key] = abs(val)

    # Compute F_ram_residual only if it's in the force fields (decomposed mode)
    if "F_ram_residual" in force_keys:
        # F_ram_residual = F_ram - F_ram_wind - F_ram_SN
        # This captures any thermal pressure not from direct wind/SN momentum input
        F_ram = snapshot.get("F_ram")
        F_ram_wind = forces.get("F_ram_wind", 0.0)
        F_ram_SN = forces.get("F_ram_SN", 0.0)

        if F_ram is not None and np.isfinite(F_ram):
            # Use nan_to_num to handle NaN in wind/SN values
            F_wind_val = np.nan_to_num(F_ram_wind, nan=0.0)
            F_SN_val = np.nan_to_num(F_ram_SN, nan=0.0)
            residual = abs(F_ram) - F_wind_val - F_SN_val
            # Residual should be non-negative (force magnitude)
            forces["F_ram_residual"] = max(0.0, residual)
        else:
            forces["F_ram_residual"] = np.nan

    # Compute F_ram_thermal only if it's in the force fields (sn_highlight mode)
    if "F_ram_thermal" in force_keys:
        # F_ram_thermal = F_ram - F_ram_SN (combines wind + residual into one blue category)
        # This highlights SN contribution while grouping all other thermal feedback
        F_ram = snapshot.get("F_ram")
        F_ram_SN_raw = snapshot.get("F_ram_SN")

        if F_ram is not None and np.isfinite(F_ram):
            F_SN_val = np.nan_to_num(F_ram_SN_raw, nan=0.0) if F_ram_SN_raw is not None else 0.0
            thermal = abs(F_ram) - abs(F_SN_val)
            # Thermal should be non-negative (force magnitude)
            forces["F_ram_thermal"] = max(0.0, thermal)
        else:
            forces["F_ram_thermal"] = np.nan

    return forces


def load_simulation(base_dir, run_name, use_modified=False):
    """
    Load simulation output for a given run name.

    Parameters
    ----------
    base_dir : Path
        Base directory containing output folders
    run_name : str
        Run name (e.g., "1e7_sfe001_n1e4")
    use_modified : bool
        If True, look in *_modified/ folders instead of regular folders

    Returns
    -------
    TrinityOutput or None
        Loaded output, or None if not found
    """
    # Modify run_name to look in *_modified folder if requested
    search_name = f"{run_name}_modified" if use_modified else run_name

    data_path = find_data_file(base_dir, search_name)
    if data_path is None:
        return None

    try:
        return load_output(data_path)
    except Exception as e:
        print(f"    Warning: Failed to load {search_name}: {e}")
        return None


# =============================================================================
# Grid Building
# =============================================================================

def build_dominance_grid(target_time, mCloud_list, sfe_list, nCore, base_dir, use_modified=False):
    """
    Build 2D grid of dominant feedback indices for a given time.

    Parameters
    ----------
    target_time : float
        Target time in Myr
    mCloud_list : list
        List of cloud mass strings (e.g., ["1e7", "1e8"])
    sfe_list : list
        List of SFE strings (e.g., ["001", "010", "020"])
    nCore : str
        Core density string (e.g., "1e4")
    base_dir : Path
        Base directory for output folders
    use_modified : bool
        If True, look in *_modified/ folders

    Returns
    -------
    np.ndarray
        2D array shape (len(sfe_list), len(mCloud_list))
        Values: 0-3 for force index, FILE_NOT_FOUND (-1), TIME_OUT_OF_RANGE (-2)
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)
    # Initialize with FILE_NOT_FOUND as default
    grid = np.full((n_sfe, n_mass), FILE_NOT_FOUND, dtype=float)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            run_name = f"{mCloud}_sfe{sfe}_n{nCore}"
            display_name = f"{run_name}_modified" if use_modified else run_name

            output = load_simulation(base_dir, run_name, use_modified=use_modified)
            if output is None:
                print(f"    {display_name}: not found")
                grid[i, j] = FILE_NOT_FOUND
                continue

            # Check time bounds before attempting to get snapshot
            if target_time < output.t_min or target_time > output.t_max:
                print(f"    {display_name}: t={target_time} outside range [{output.t_min:.3f}, {output.t_max:.3f}]")
                grid[i, j] = TIME_OUT_OF_RANGE
                continue

            snapshot = get_snapshot_at_time(output, target_time)
            dominant = get_dominant_force(snapshot)

            if np.isnan(dominant):
                # Valid file, valid time, but no force data
                grid[i, j] = TIME_OUT_OF_RANGE
            else:
                grid[i, j] = dominant
                force_name = FORCE_FIELDS[int(dominant)][1]
                print(f"    {display_name}: {force_name}")

    return grid


def build_force_grids(target_time, mCloud_list, sfe_list, nCore, base_dir, use_modified=False):
    """
    Build 2D grids of force values for a given time (for proper interpolation).

    Unlike build_dominance_grid which returns dominant indices, this returns
    the actual force values for each type, allowing proper interpolation
    before taking argmax.

    Parameters
    ----------
    target_time : float
        Target time in Myr
    mCloud_list : list
        List of cloud mass strings (e.g., ["1e7", "1e8"])
    sfe_list : list
        List of SFE strings (e.g., ["001", "010", "020"])
    nCore : str
        Core density string (e.g., "1e4")
    base_dir : Path
        Base directory for output folders
    use_modified : bool
        If True, look in *_modified/ folders

    Returns
    -------
    forces_dict : dict
        Dictionary {force_key: 2D array} for each force type
    mask : np.ndarray
        Boolean 2D array where True = invalid (file not found or time out of range)
    status_grid : np.ndarray
        2D array with FILE_NOT_FOUND (-1) or TIME_OUT_OF_RANGE (-2) for invalid cells
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)

    # Initialize force grids with NaN
    forces_dict = {}
    for key, _, _ in FORCE_FIELDS:
        forces_dict[key] = np.full((n_sfe, n_mass), np.nan, dtype=float)

    # Status grid for missing data types
    status_grid = np.zeros((n_sfe, n_mass), dtype=float)
    mask = np.zeros((n_sfe, n_mass), dtype=bool)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            run_name = f"{mCloud}_sfe{sfe}_n{nCore}"
            display_name = f"{run_name}_modified" if use_modified else run_name

            output = load_simulation(base_dir, run_name, use_modified=use_modified)
            if output is None:
                print(f"    {display_name}: not found")
                mask[i, j] = True
                status_grid[i, j] = FILE_NOT_FOUND
                continue

            # Check time bounds
            if target_time < output.t_min or target_time > output.t_max:
                print(f"    {display_name}: t={target_time} outside range [{output.t_min:.3f}, {output.t_max:.3f}]")
                mask[i, j] = True
                status_grid[i, j] = TIME_OUT_OF_RANGE
                continue

            snapshot = get_snapshot_at_time(output, target_time)
            force_vals = get_force_values(snapshot)

            if force_vals is None or all(np.isnan(v) for v in force_vals.values()):
                mask[i, j] = True
                status_grid[i, j] = TIME_OUT_OF_RANGE
            else:
                for key in forces_dict:
                    forces_dict[key][i, j] = force_vals.get(key, np.nan)

                # Report dominant force
                valid_forces = {k: v for k, v in force_vals.items() if np.isfinite(v)}
                if valid_forces:
                    dominant_key = max(valid_forces, key=valid_forces.get)
                    for idx, (k, label, _) in enumerate(FORCE_FIELDS):
                        if k == dominant_key:
                            print(f"    {display_name}: {label}")
                            break

    return forces_dict, mask, status_grid


def refine_dominant_map(logM, eps, forces_dict, mask, nref_M=300, nref_eps=300):
    """
    Interpolate continuous force fields, then take argmax for smooth boundaries.

    This is the correct approach: interpolate the underlying forces FIRST,
    then determine the dominant force at each point. This avoids the color
    bleeding issues from smoothing categorical labels.

    Parameters
    ----------
    logM : np.ndarray
        1D array of log10(Mcloud/Msun) values
    eps : np.ndarray
        1D array of SFE values
    forces_dict : dict
        Dictionary {force_key: 2D array (n_eps, n_mass)} for each force
    mask : np.ndarray
        Boolean 2D array where True = invalid data
    nref_M : int
        Number of points in refined mass grid
    nref_eps : int
        Number of points in refined SFE grid

    Returns
    -------
    logM_f : np.ndarray
        Fine mass grid (log10 values)
    eps_f : np.ndarray
        Fine SFE grid
    dom_f : np.ma.MaskedArray
        Dominant force index at each fine grid point (masked where invalid)
    """
    keys = list(forces_dict.keys())

    # Create fine grids
    logM_f = np.linspace(logM.min(), logM.max(), nref_M)
    eps_f = np.linspace(eps.min(), eps.max(), nref_eps)
    E, L = np.meshgrid(eps_f, logM_f, indexing="ij")  # shapes (nref_eps, nref_M)
    pts = np.stack([E.ravel(), L.ravel()], axis=-1)

    # Interpolate each force field
    fine_fields = []
    for k in keys:
        arr = np.array(forces_dict[k], dtype=float)  # (n_eps, n_mass)
        # Set masked regions to NaN to prevent interpolation across holes
        arr_masked = arr.copy()
        arr_masked[mask] = np.nan

        itp = RegularGridInterpolator(
            (eps, logM), arr_masked,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        fine_fields.append(itp(pts).reshape(nref_eps, nref_M))

    stack = np.stack(fine_fields, axis=-1)  # (nref_eps, nref_M, nF)

    # Identify invalid cells (where ALL forces are NaN)
    invalid_f = np.all(~np.isfinite(stack), axis=-1)

    # For cells with all NaN, temporarily fill with 0 to avoid nanargmax error
    # We'll mask these cells afterward
    stack_filled = stack.copy()
    stack_filled[invalid_f] = 0  # Temporary fill for argmax

    # Take argmax to get dominant force
    with np.errstate(invalid='ignore'):
        dom_f = np.nanargmax(stack_filled, axis=-1).astype(float)

    # Mask invalid cells
    dom_f = np.ma.array(dom_f, mask=invalid_f)

    return logM_f, eps_f, dom_f


# =============================================================================
# Plotting Functions
# =============================================================================

def create_colormap(force_fields=None):
    """
    Create discrete colormap for force types and missing data.

    Color mapping depends on force_fields:
    - -2 (TIME_OUT_OF_RANGE): white
    - -1 (FILE_NOT_FOUND): gray
    - 0, 1, 2, ... : force colors from force_fields list
    """
    if force_fields is None:
        force_fields = FORCE_FIELDS

    n_forces = len(force_fields)

    # Colors in order: TIME_OUT_OF_RANGE, FILE_NOT_FOUND, then force colors
    colors = [
        COLOR_TIME_OUT_OF_RANGE,  # -2
        COLOR_FILE_NOT_FOUND,     # -1
    ] + [field[2] for field in force_fields]

    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # NaN -> white (fallback)

    # Bounds: -2.5 to -1.5 -> color[0], -1.5 to -0.5 -> color[1], etc.
    # Dynamic based on number of force types
    bounds = [-2.5, -1.5, -0.5] + [i + 0.5 for i in range(n_forces)]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def create_force_colormap(force_fields=None):
    """
    Create discrete colormap for just force types.

    Used for interpolated plots where missing data is handled via masking.
    """
    if force_fields is None:
        force_fields = FORCE_FIELDS

    n_forces = len(force_fields)
    colors = [field[2] for field in force_fields]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # Masked values -> white
    norm = BoundaryNorm(np.arange(n_forces + 1) - 0.5, n_forces)
    return cmap, norm


def compute_grid_layout(n_plots):
    """Compute optimal subplot grid layout."""
    layouts = {
        1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2),
        5: (2, 3), 6: (2, 3), 7: (2, 4), 8: (2, 4),
        9: (3, 3), 10: (2, 5), 11: (3, 4), 12: (3, 4)
    }
    if n_plots in layouts:
        return layouts[n_plots]

    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    return nrows, ncols


def plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                     smooth='none', axis_mode='discrete', forces_dict=None, mask=None,
                     nref_M=300, nref_eps=300):
    """
    Plot a single dominance grid on an axis.

    For smooth mode, uses the proper approach: interpolate continuous force values
    on a fine grid, then take argmax. This avoids color bleeding issues.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    grid : np.ndarray
        2D array of dominant force indices (used for discrete mode or as fallback)
    mCloud_list : list
        Cloud mass labels (strings like "1e7")
    sfe_list : list
        SFE labels (strings like "010")
    target_time : float
        Time in Myr (for title)
    cmap : ListedColormap
        Colormap (full colormap with missing data colors)
    norm : BoundaryNorm
        Color normalization
    smooth : str
        Smoothing method: 'none' or 'interp'
        Note: 'interp' requires forces_dict and mask to be provided
    axis_mode : str
        'discrete': equal spacing with categorical labels
        'continuous': real value spacing (log for mCloud, linear for SFE)
    forces_dict : dict, optional
        Dictionary {force_key: 2D array} for interpolation (required for smooth='interp')
    mask : np.ndarray, optional
        Boolean mask where True = invalid data (required for smooth='interp')
    nref_M : int
        Number of points in refined mass grid for interpolation
    nref_eps : int
        Number of points in refined SFE grid for interpolation
    """
    n_mass = len(mCloud_list)
    n_sfe = len(sfe_list)

    # Convert to real values
    logM = np.array([np.log10(float(m)) for m in mCloud_list])  # log10(Msun)
    eps = np.array([int(s) / 100.0 for s in sfe_list])  # decimal SFE

    # Smoothing only works with continuous mode and requires force data
    if smooth == 'interp' and (forces_dict is None or mask is None):
        print("    Warning: smooth='interp' requires forces_dict and mask, falling back to 'none'")
        smooth = 'none'
    if axis_mode == 'discrete' and smooth != 'none':
        smooth = 'none'

    if axis_mode == 'continuous' and smooth == 'interp':
        # PROPER APPROACH: Interpolate continuous force values, then argmax
        # This gives smooth boundaries without color bleeding
        logM_f, eps_f, dom_f = refine_dominant_map(logM, eps, forces_dict, mask,
                                                    nref_M=nref_M, nref_eps=nref_eps)

        # Create colormap for just forces (0-3), masking handles invalid data
        force_cmap, force_norm = create_force_colormap()

        # Use pcolormesh with nearest shading for categorical data
        # This ensures each pixel is exactly one color with no interpolation artifacts
        im = ax.pcolormesh(
            10**logM_f,  # Convert back to linear mass for display
            eps_f,
            dom_f,
            cmap=force_cmap,
            norm=force_norm,
            shading='nearest',
            linewidth=0,
            antialiased=False
        )

        # Use log scale for x-axis
        ax.set_xscale('log')

        # Set axis limits exactly to data range (no padding for square plot)
        ax.set_xlim(10**logM.min(), 10**logM.max())
        ax.set_ylim(eps.min(), eps.max())

        # Auto-generate nice tick locations for y-axis
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    elif axis_mode == 'continuous':
        # Continuous mode without interpolation - use pcolormesh with nearest
        mass_min, mass_max = logM.min(), logM.max()
        eps_min, eps_max = eps.min(), eps.max()

        # Add padding (half cell width at edges)
        if n_mass > 1:
            mass_pad = (logM[1] - logM[0]) / 2
        else:
            mass_pad = 0.5
        if n_sfe > 1:
            eps_pad = (eps[1] - eps[0]) / 2
        else:
            eps_pad = 0.05

        extent = (mass_min - mass_pad, mass_max + mass_pad,
                  eps_min - eps_pad, eps_max + eps_pad)

        # Mask invalid values
        grid_masked = np.ma.masked_invalid(grid)

        # Use pcolormesh with nearest shading
        im = ax.pcolormesh(
            logM, eps, grid_masked,
            cmap=cmap,
            norm=norm,
            shading='nearest',
            linewidth=0,
            antialiased=False
        )

        # X-axis: use standard log-scale ticks at integer powers
        min_power = int(np.floor(extent[0]))
        max_power = int(np.ceil(extent[1]))
        x_ticks = np.arange(min_power, max_power + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([rf"$10^{{{int(p)}}}$" for p in x_ticks])
        ax.set_xlim(extent[0], extent[1])

        # Y-axis: linear SFE with reasonable tick spacing
        ax.set_ylim(extent[2], extent[3])
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    else:
        # Discrete mode - categorical grid with cell borders
        X, Y = np.meshgrid(
            np.arange(n_mass + 1) - 0.5,
            np.arange(n_sfe + 1) - 0.5
        )

        # Mask invalid values
        grid_masked = np.ma.masked_invalid(grid)

        im = ax.pcolormesh(
            X, Y, grid_masked,
            cmap=cmap,
            norm=norm,
            edgecolors='white',
            linewidths=1.0
        )

        # Discrete axis labels - handle non-power-of-10 masses (e.g., 5e6)
        ax.set_xticks(np.arange(n_mass))
        xlabels = []
        for m in mCloud_list:
            mval = float(m)
            mexp = int(np.floor(np.log10(mval)))
            mcoeff = mval / (10 ** mexp)
            mcoeff = round(mcoeff)
            if mcoeff == 10:
                mcoeff = 1
                mexp += 1
            if mcoeff == 1:
                xlabels.append(rf"$10^{{{mexp}}}$")
            else:
                xlabels.append(rf"${mcoeff}\times10^{{{mexp}}}$")
        ax.set_xticklabels(xlabels)

        ax.set_yticks(np.arange(n_sfe))
        ax.set_yticklabels([f"{int(s)/100:.2f}" for s in sfe_list])

    ax.set_title(rf"$t = {target_time}$ Myr", fontsize=11)

    return im


def create_legend(force_fields=None):
    """Create legend handles for force types and missing data."""
    if force_fields is None:
        force_fields = FORCE_FIELDS

    handles = [
        Patch(facecolor=color, edgecolor='gray', label=label)
        for _, label, color in force_fields
    ]
    # Add missing data types
    handles.append(
        Patch(facecolor=COLOR_TIME_OUT_OF_RANGE, edgecolor='gray', label='Beyond $t_{\\rm max}$')
    )
    handles.append(
        Patch(facecolor=COLOR_FILE_NOT_FOUND, edgecolor='gray', label='File not found')
    )
    return handles


# =============================================================================
# Main Function
# =============================================================================

def main(mCloud_list, sfe_list, nCore_list, target_times, base_dir, fig_dir=None,
         use_modified=False, smooth='none', axis_mode='discrete'):
    """
    Generate dominant feedback grid plot(s).

    Parameters
    ----------
    mCloud_list : list
        Cloud mass values (strings, e.g., ["1e7", "1e8"])
    sfe_list : list
        SFE values (strings, e.g., ["001", "010", "020"])
    nCore_list : list
        List of core density values (e.g., ["1e4", "1e5"]).
        Produces one plot per nCore.
    target_times : list
        Target times in Myr (e.g., [1.0, 1.5, 2.0, 2.5])
    base_dir : Path
        Base directory for output folders
    fig_dir : Path, optional
        Directory to save figures
    use_modified : bool
        If True, look in *_modified/ folders and save as *_modified.pdf
    smooth : str
        Smoothing method: 'none' or 'interp' (interpolate force fields)
    axis_mode : str
        'discrete': equal spacing with categorical labels
        'continuous': real value spacing (log for mCloud, linear for SFE)
    """
    print("=" * 60)
    print("Dominant Feedback Grid Plot")
    print("=" * 60)
    print(f"  mCloud: {mCloud_list}")
    print(f"  SFE: {sfe_list}")
    print(f"  nCore: {nCore_list}")
    print(f"  Times: {target_times} Myr")
    print(f"  Base dir: {base_dir}")
    print(f"  Use modified: {use_modified}")
    print(f"  Smooth: {smooth}")
    print(f"  Axis mode: {axis_mode}")
    if axis_mode == 'discrete' and smooth != 'none':
        print(f"  Warning: Smoothing only works with axis_mode='continuous', ignoring --smooth {smooth}")
        smooth = 'none'
    print()

    # Set up figure directory
    if fig_dir is None:
        fig_dir = FIG_DIR
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Compute layout
    n_times = len(target_times)
    nrows, ncols = compute_grid_layout(n_times)

    cmap, norm = create_colormap()

    # Generate one plot per nCore
    for nCore in nCore_list:
        print("-" * 60)
        print(f"Processing nCore = {nCore}")
        print("-" * 60)

        # Create figure (square aspect for each subplot)
        fig_width = 5.0 * ncols
        fig_height = 5.0 * nrows + 0.8  # Extra space for legend
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
            squeeze=False
        )

        # Build and plot each time snapshot
        for idx, target_time in enumerate(target_times):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            print(f"Building grid for t = {target_time} Myr...")

            # For interpolated smooth mode, get force values; otherwise just dominance grid
            if smooth == 'interp' and axis_mode == 'continuous':
                forces_dict, mask, status_grid = build_force_grids(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                # Build dominance grid from forces for fallback/status display
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode,
                               forces_dict=forces_dict, mask=mask)
            else:
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode)
            print()

        # Hide unused subplots
        for idx in range(n_times, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)

        # Axis labels
        fig.supxlabel(r"$M_{\rm cloud}$ [$M_\odot$]", fontsize=12)
        fig.supylabel(r"Star Formation Efficiency $\epsilon$", fontsize=12)

        # Title
        nlog = int(np.log10(float(nCore)))
        title_suffix = " (modified)" if use_modified else ""
        fig.suptitle(
            rf"Dominant Feedback ($n_{{\rm core}} = 10^{{{nlog}}}$ cm$^{{-3}}$){title_suffix}",
            fontsize=13
        )

        # Legend
        handles = create_legend()
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
            facecolor='white',
            edgecolor='gray'
        )

        # Save with appropriate suffix
        # Format: dominant_feedback_n{nCore}[_modified]_{axis_mode}[_{smooth}].pdf
        filename = f"dominant_feedback_n{nCore}"
        if use_modified:
            filename = f"{filename}_modified"
        filename = f"{filename}_{axis_mode}"
        if axis_mode == 'continuous' and smooth != 'none':
            filename = f"{filename}_{smooth}"
        out_pdf = fig_dir / f"{filename}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Saved: {out_pdf}")

        plt.show()
        # plt.close(fig)
        print()


# =============================================================================
# Movie Generation
# =============================================================================

def make_movie(mCloud_list, sfe_list, nCore, base_dir, fig_dir=None,
               use_modified=False, smooth='none', axis_mode='discrete',
               t_start=0.0, t_end=5.0, dt=0.05, fps=10):
    """
    Create an animated GIF showing the evolution of dominant feedback over time.

    Parameters
    ----------
    mCloud_list : list
        Cloud mass values (strings, e.g., ["1e7", "1e8"])
    sfe_list : list
        SFE values (strings, e.g., ["001", "010", "020"])
    nCore : str
        Core density value (single value, e.g., "1e4")
    base_dir : Path
        Base directory for output folders
    fig_dir : Path, optional
        Directory to save the GIF
    use_modified : bool
        If True, look in *_modified/ folders
    smooth : str
        Smoothing method: 'none' or 'interp' (interpolate force fields)
    axis_mode : str
        'discrete' or 'continuous'
    t_start : float
        Start time in Myr (default: 0.0)
    t_end : float
        End time in Myr (default: 5.0)
    dt : float
        Time step in Myr (default: 0.05)
    fps : int
        Frames per second in the output GIF (default: 10)
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL (Pillow) is required for movie generation.")
        print("Install with: pip install Pillow")
        return

    print("=" * 60)
    print("Dominant Feedback Movie Generation")
    print("=" * 60)
    print(f"  mCloud: {mCloud_list}")
    print(f"  SFE: {sfe_list}")
    print(f"  nCore: {nCore}")
    print(f"  Time range: {t_start} - {t_end} Myr, dt = {dt} Myr")
    print(f"  FPS: {fps} (frame duration: {1000/fps:.0f} ms)")
    print(f"  Base dir: {base_dir}")
    print(f"  Use modified: {use_modified}")
    print(f"  Smooth: {smooth}, Axis mode: {axis_mode}")
    print()

    # Set up directories
    if fig_dir is None:
        fig_dir = FIG_DIR
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate time steps
    times = np.arange(t_start, t_end + dt/2, dt)
    n_frames = len(times)
    print(f"Generating {n_frames} frames...")

    # Handle smoothing warning
    if axis_mode == 'discrete' and smooth != 'none':
        print(f"  Warning: Smoothing only works with axis_mode='continuous', ignoring --smooth {smooth}")
        smooth = 'none'

    cmap, norm = create_colormap()

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix='dominant_feedback_movie_')
    frame_paths = []

    try:
        for frame_idx, target_time in enumerate(times):
            print(f"  Frame {frame_idx + 1}/{n_frames}: t = {target_time:.3f} Myr", end='\r')

            # Create single-panel figure
            fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

            # For interpolated smooth mode, get force values; otherwise just dominance grid
            if smooth == 'interp' and axis_mode == 'continuous':
                forces_dict, mask, status_grid = build_force_grids(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode,
                               forces_dict=forces_dict, mask=mask)
            else:
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, nCore, base_dir,
                    use_modified=use_modified
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode)

            # Labels
            ax.set_xlabel(r"$M_{\rm cloud}$ [$M_\odot$]", fontsize=12)
            ax.set_ylabel(r"Star Formation Efficiency $\epsilon$", fontsize=12)

            # Title with nCore info
            nlog = int(np.log10(float(nCore)))
            title_suffix = " (modified)" if use_modified else ""
            ax.set_title(
                rf"Dominant Feedback at $t = {target_time:.2f}$ Myr "
                rf"($n_{{\rm core}} = 10^{{{nlog}}}$ cm$^{{-3}}$){title_suffix}",
                fontsize=12
            )

            # Legend
            handles = create_legend()
            ax.legend(
                handles=handles,
                loc='upper center',
                ncol=3,
                bbox_to_anchor=(0.5, -0.12),
                frameon=True,
                facecolor='white',
                edgecolor='gray',
                fontsize=9
            )

            # Save frame
            frame_path = Path(temp_dir) / f"frame_{frame_idx:04d}.png"
            fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            frame_paths.append(frame_path)
            plt.close(fig)

        print()  # New line after progress
        print(f"Assembling GIF from {len(frame_paths)} frames...")

        # Load frames and create GIF
        frames = [Image.open(fp) for fp in frame_paths]

        # Calculate frame duration in milliseconds
        frame_duration = int(1000 / fps)

        # Build output filename
        filename = f"dominant_feedback_movie_n{nCore}"
        if use_modified:
            filename = f"{filename}_modified"
        filename = f"{filename}_{axis_mode}"
        if axis_mode == 'continuous' and smooth != 'none':
            filename = f"{filename}_{smooth}"
        filename = f"{filename}_t{t_start:.1f}-{t_end:.1f}"
        out_gif = fig_dir / f"{filename}.gif"

        # Save GIF
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0  # 0 = infinite loop
        )

        print(f"Saved: {out_gif}")
        print(f"  Duration: {len(frames) * frame_duration / 1000:.1f} seconds")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot dominant feedback grid across mCloud-SFE parameter space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Static plots
  python paper_dominantFeedback.py
  python paper_dominantFeedback.py --mCloud 1e7 1e8 --sfe 001 020 --times 1 1.5 2 2.5
  python paper_dominantFeedback.py --nCore 1e4 1e5 --modified
  python paper_dominantFeedback.py --axis-mode continuous --smooth interp

  # Movie generation
  python paper_dominantFeedback.py --movie --nCore 1e4 --dt 0.05 --fps 10
  python paper_dominantFeedback.py --movie --t-start 0.5 --t-end 3.0 --dt 0.1 --fps 5

Smoothing methods (only with --axis-mode continuous):
  none   - Discrete grid, each cell shows dominant force at that point
  interp - Interpolate underlying force fields on fine grid, then take argmax
           This gives smooth boundaries without color bleeding artifacts

Axis modes:
  discrete   - Equal spacing with categorical labels (default, no smoothing)
  continuous - Real value spacing (log scale for mCloud, linear for SFE)

Movie mode:
  --movie    - Generate animated GIF instead of static plots
  --dt       - Time step between frames in Myr (default: 0.05)
  --fps      - Frames per second / playback speed (default: 10)
  --t-start  - Start time in Myr (default: 0.0)
  --t-end    - End time in Myr (default: 5.0)

F_ram decomposition modes:
  --decompose-ram     - Show F_ram_wind (winds), F_ram_SN (supernovae), F_ram_residual separately
  --sn-highlight      - Highlight only SN (yellow); group winds+residual as thermal (blue)
  --no-decompose-ram  - Show combined F_ram as single category (original behavior)
        """
    )

    parser.add_argument(
        '--mCloud', '-m', nargs='+', default=None,
        help=f'Cloud masses (e.g., 1e7 1e8). Default: {DEFAULT_MCLOUD}'
    )
    parser.add_argument(
        '--sfe', '-s', nargs='+', default=None,
        help=f'Star formation efficiencies (e.g., 001 010 020). Default: {DEFAULT_SFE}'
    )
    parser.add_argument(
        '--nCore', '-n', nargs='+', default=None,
        help=f'Core densities (e.g., 1e4 1e5). Produces one plot per nCore. Default: {DEFAULT_NCORE}'
    )
    parser.add_argument(
        '--times', '-t', nargs='+', type=float, default=None,
        help=f'Target times in Myr (e.g., 1.0 1.5 2.0 2.5). Default: {DEFAULT_TIMES}'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help=f'Base directory for simulation outputs. Default: {DEFAULT_OUTPUT_DIR}'
    )
    parser.add_argument(
        '--fig-dir', '-f', default=None,
        help=f'Directory to save figures. Default: {FIG_DIR}'
    )
    parser.add_argument(
        '--modified', action='store_true',
        help='Use *_modified/ output folders and save as *_modified.pdf'
    )
    parser.add_argument(
        '--smooth', choices=['none', 'interp'], default=None,
        help=f"Smoothing method: 'none' for discrete grid, 'interp' for interpolated force fields "
             f"(only with --axis-mode continuous). Default: {DEFAULT_SMOOTH}"
    )
    parser.add_argument(
        '--axis-mode', choices=['discrete', 'continuous'], default=None,
        help=f"Axis spacing mode: 'discrete' for equal spacing with categorical labels, "
             f"'continuous' for real value spacing (log for mCloud, linear for SFE). Default: {DEFAULT_AXIS_MODE}"
    )

    # Movie generation arguments
    parser.add_argument(
        '--movie', action='store_true',
        help='Generate animated GIF instead of static plots'
    )
    parser.add_argument(
        '--dt', type=float, default=0.05,
        help='Time step between frames in Myr (default: 0.05)'
    )
    parser.add_argument(
        '--fps', type=int, default=10,
        help='Frames per second / playback speed (default: 10, higher = faster)'
    )
    parser.add_argument(
        '--t-start', type=float, default=0.0,
        help='Start time for movie in Myr (default: 0.0)'
    )
    parser.add_argument(
        '--t-end', type=float, default=5.0,
        help='End time for movie in Myr (default: 5.0)'
    )

    # F_ram decomposition mode
    parser.add_argument(
        '--decompose-ram', action='store_true', default=None,
        help='Decompose F_ram into F_ram_wind (winds), F_ram_SN (supernovae), F_ram_residual (other thermal)'
    )
    parser.add_argument(
        '--sn-highlight', action='store_true',
        help='Highlight only SN (yellow); group winds+residual as thermal (blue)'
    )
    parser.add_argument(
        '--no-decompose-ram', action='store_true',
        help='Use combined F_ram as single category (original behavior)'
    )

    args = parser.parse_args()

    # Apply defaults
    mCloud_list = args.mCloud if args.mCloud else DEFAULT_MCLOUD
    sfe_list = args.sfe if args.sfe else DEFAULT_SFE
    nCore_list = args.nCore if args.nCore else DEFAULT_NCORE
    target_times = args.times if args.times else DEFAULT_TIMES
    base_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    fig_dir = Path(args.fig_dir) if args.fig_dir else None
    smooth = args.smooth if args.smooth else DEFAULT_SMOOTH
    axis_mode = args.axis_mode if args.axis_mode else DEFAULT_AXIS_MODE

    # Use module-level USE_MODIFIED if --modified not explicitly set
    use_modified = args.modified or USE_MODIFIED

    # Determine F_ram decomposition mode
    # Priority: CLI flags > module-level DECOMPOSE_MODE
    if args.no_decompose_ram:
        decompose_mode = 'combined'
    elif args.sn_highlight:
        decompose_mode = 'sn_highlight'
    elif args.decompose_ram:
        decompose_mode = 'decomposed'
    else:
        decompose_mode = DECOMPOSE_MODE

    # Update global FORCE_FIELDS based on decomposition mode
    global FORCE_FIELDS
    if decompose_mode == 'decomposed':
        FORCE_FIELDS = FORCE_FIELDS_DECOMPOSED
        print("F_ram decomposition: enabled (wind/SN/residual separately)")
    elif decompose_mode == 'sn_highlight':
        FORCE_FIELDS = FORCE_FIELDS_SN_HIGHLIGHT
        print("F_ram decomposition: SN highlight (thermal in blue, SN in yellow)")
    else:
        FORCE_FIELDS = FORCE_FIELDS_COMBINED
        print("F_ram decomposition: disabled (combined)")

    if args.movie:
        # Movie mode: generate one GIF per nCore
        for nCore in nCore_list:
            make_movie(
                mCloud_list, sfe_list, nCore, base_dir, fig_dir,
                use_modified=use_modified, smooth=smooth, axis_mode=axis_mode,
                t_start=args.t_start, t_end=args.t_end, dt=args.dt, fps=args.fps
            )
    else:
        # Static plot mode
        main(mCloud_list, sfe_list, nCore_list, target_times, base_dir, fig_dir,
             use_modified, smooth, axis_mode)
