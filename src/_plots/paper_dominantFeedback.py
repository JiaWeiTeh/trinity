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
- White: collapsed (simulation ended in shell collapse)
- Gray: no data (simulation file not found)

F_ram competes as a whole first, then subclassifies to wind (blue) or SN (yellow)
if it wins. This is the same logic as paper_momentum.py's dominant bar.

For simulations that have ended (t > t_max): if the cloud collapsed, the cell is
shown as white ("Collapse"); otherwise, the final dominant feedback is persisted.

Usage
-----
    # Plot from folder (auto-discovers simulations)
    python paper_dominantFeedback.py -F /path/to/outputs/sweep_test

    # Filter by density
    python paper_dominantFeedback.py -F /path/to/outputs --nCore 1e4

    # Custom time snapshots
    python paper_dominantFeedback.py -F /path/to/outputs --times 1.0 3.0 5.0

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
from src._output.trinity_reader import (
    load_output, find_all_simulations, organize_simulations_for_grid, get_unique_ndens,
    info_simulations
)

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


# =============================================================================
# Configuration
# =============================================================================

# Force field definitions: (key, label, color)
# F_ram competes as a whole first, then subclassifies to wind or SN if it wins
FORCE_FIELDS = [
    ("F_grav",     "Gravity",           "#2c3e50"),  # Dark blue-gray
    ("F_ram_wind", "Winds",             "#3498db"),  # Blue for winds (when F_ram wins and wind > SN)
    ("F_ram_SN",   "Supernovae",        "#DAA520"),  # Golden yellow for SN (when F_ram wins and SN > wind)
    ("F_ion_out",  "Photoionised gas",  "#e74c3c"),  # Red
    ("F_rad",      "Radiation",         "#9b59b6"),  # Purple
]

# Special values for missing data (must be negative to distinguish from force indices)
FILE_NOT_FOUND = -1      # Gray: simulation file not found
COLLAPSED = -2           # White: simulation collapsed

# Missing data colors
COLOR_FILE_NOT_FOUND = "#808080"      # Gray
COLOR_COLLAPSED = "#FFFFFF"           # White

# Default parameters
DEFAULT_TIMES = [1.0, 3.0, 5.0]  # Myr - includes times when SN should be active

# Axis mode options:
#   'discrete': equal spacing, categorical labels
#   'continuous': real value spacing (log for mCloud, linear for SFE)
DEFAULT_AXIS_MODE = 'continuous'

# Smoothing options: 'none' or 'interp' (only effective with axis_mode='continuous')
DEFAULT_SMOOTH = 'interp'

# Default directories
FIG_DIR = Path(__file__).parent.parent.parent / "fig"


def build_filename(base_name, **kwargs):
    """Build output filename from base name and keyword arguments."""
    parts = [base_name]

    flag_order = [
        ('nCore', lambda v: f"n{v}"),
        ('axis_mode', lambda v: v if v and v != 'discrete' else None),
        ('smooth', lambda v: v if v and v != 'none' else None),
        ('t_range', lambda v: f"t{v[0]:.1f}-{v[1]:.1f}" if v else None),
    ]

    for flag_name, formatter in flag_order:
        if flag_name in kwargs:
            value = kwargs[flag_name]
            if value is not None and value is not False:
                formatted = formatter(value)
                if formatted:
                    parts.append(formatted)

    return "_".join(parts)


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_snapshot_at_time(output, target_time, persist_final=True):
    """
    Get snapshot data at a specific time.

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output
    target_time : float
        Target time in Myr
    persist_final : bool
        If True and target_time > t_max, return the final snapshot instead of None.

    Returns
    -------
    dict or None
        Snapshot data dict, or None if time is before t_min or file error
    """
    t_min = output.t_min
    t_max = output.t_max

    # If target time is before simulation starts, return None
    if target_time < t_min:
        return None

    # If target time is after simulation ends, return final snapshot (persist)
    if target_time > t_max:
        if persist_final:
            try:
                snap = output.get_at_time(t_max, mode='closest', quiet=True)
                return snap.data
            except (ValueError, IndexError):
                return None
        else:
            return None

    # Normal case: target time is within range
    try:
        snap = output.get_at_time(target_time, mode='closest', quiet=True)
        return snap.data
    except (ValueError, IndexError):
        return None


def get_dominant_force(snapshot):
    """
    Determine the dominant feedback force from a snapshot.

    F_ram competes as a whole first, then subclassifies to wind vs SN if it wins.

    Returns
    -------
    int or np.nan
        Index of dominant force (0-4), or np.nan if no valid forces
    """
    if snapshot is None:
        return np.nan

    force_vals = get_force_values(snapshot)
    if force_vals is None:
        return np.nan

    # Get individual force values
    F_grav = abs(force_vals.get("F_grav", 0.0) or 0.0)
    F_ram_wind = abs(force_vals.get("F_ram_wind", 0.0) or 0.0)
    F_ram_SN = abs(force_vals.get("F_ram_SN", 0.0) or 0.0)
    F_ion_out = abs(force_vals.get("F_ion_out", 0.0) or 0.0)
    F_rad = abs(force_vals.get("F_rad", 0.0) or 0.0)

    # F_ram competes as total
    F_ram_total = F_ram_wind + F_ram_SN

    # Compete: {F_grav, F_ram_total, F_ion_out, F_rad}
    competitors = [F_grav, F_ram_total, F_ion_out, F_rad]
    total = sum(competitors)
    if total == 0 or not np.isfinite(total):
        return np.nan

    winner_idx = int(np.argmax(competitors))

    # Map winner to FORCE_FIELDS indices:
    # 0 = F_grav, 1 = F_ram_wind, 2 = F_ram_SN, 3 = F_ion_out, 4 = F_rad
    if winner_idx == 0:  # F_grav wins
        return 0
    elif winner_idx == 1:  # F_ram_total wins -> subclassify
        # Return wind (1) or SN (2) based on which is larger
        if F_ram_SN > F_ram_wind:
            return 2  # F_ram_SN (yellow)
        else:
            return 1  # F_ram_wind (blue)
    elif winner_idx == 2:  # F_ion_out wins
        return 3
    else:  # F_rad wins
        return 4


def get_force_values(snapshot):
    """
    Extract all force values from a snapshot.

    Returns
    -------
    dict or None
        Dictionary of {force_key: value} or None if snapshot is None
    """
    if snapshot is None:
        return None

    forces = {}
    for key, _, _ in FORCE_FIELDS:
        val = snapshot.get(key)
        if val is None or not np.isfinite(val):
            forces[key] = np.nan
        else:
            forces[key] = abs(val)

    return forces


# =============================================================================
# Grid Building
# =============================================================================

def build_dominance_grid(target_time, mCloud_list, sfe_list, sim_grid):
    """
    Build 2D grid of dominant feedback indices for a given time.

    Parameters
    ----------
    target_time : float
        Target time in Myr
    mCloud_list : list
        List of cloud mass strings (sorted)
    sfe_list : list
        List of SFE strings (sorted)
    sim_grid : dict
        Dictionary mapping (mCloud, sfe) -> data_path

    Returns
    -------
    np.ndarray
        2D array shape (len(sfe_list), len(mCloud_list))
        Values: 0-4 for force index, FILE_NOT_FOUND (-1)
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)
    grid = np.full((n_sfe, n_mass), FILE_NOT_FOUND, dtype=float)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            data_path = sim_grid.get((mCloud, sfe))

            if data_path is None:
                grid[i, j] = FILE_NOT_FOUND
                continue

            try:
                output = load_output(data_path)
            except Exception as e:
                print(f"    Error loading {data_path}: {e}")
                grid[i, j] = FILE_NOT_FOUND
                continue

            # If past simulation end and cloud collapsed, mark as COLLAPSED
            if target_time > output.t_max:
                last_snap = output[-1]
                is_collapse = last_snap.data.get("isCollapse", False)
                end_reason = str(last_snap.data.get("SimulationEndReason", "")).lower()
                if is_collapse or "small radius" in end_reason or "collapsed" in end_reason:
                    grid[i, j] = COLLAPSED
                    continue

            # Get snapshot (persists final value if t > t_max)
            snapshot = get_snapshot_at_time(output, target_time, persist_final=True)
            dominant = get_dominant_force(snapshot)

            if np.isnan(dominant):
                grid[i, j] = FILE_NOT_FOUND
            else:
                grid[i, j] = dominant

    return grid


def build_force_grids(target_time, mCloud_list, sfe_list, sim_grid):
    """
    Build 2D grids of force values for a given time (for proper interpolation).

    Parameters
    ----------
    target_time : float
        Target time in Myr
    mCloud_list : list
        List of cloud mass strings
    sfe_list : list
        List of SFE strings
    sim_grid : dict
        Dictionary mapping (mCloud, sfe) -> data_path

    Returns
    -------
    forces_dict : dict
        Dictionary {force_key: 2D array} for each force type
    mask : np.ndarray
        Boolean 2D array where True = invalid (file not found)
    collapse_mask : np.ndarray
        Boolean 2D array where True = simulation collapsed before target_time
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)

    forces_dict = {}
    for key, _, _ in FORCE_FIELDS:
        forces_dict[key] = np.full((n_sfe, n_mass), np.nan, dtype=float)

    mask = np.zeros((n_sfe, n_mass), dtype=bool)
    collapse_mask = np.zeros((n_sfe, n_mass), dtype=bool)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            data_path = sim_grid.get((mCloud, sfe))

            if data_path is None:
                mask[i, j] = True
                continue

            try:
                output = load_output(data_path)
            except Exception:
                mask[i, j] = True
                continue

            # If past simulation end and cloud collapsed, mark as collapsed
            if target_time > output.t_max:
                last_snap = output[-1]
                is_collapse = last_snap.data.get("isCollapse", False)
                end_reason = str(last_snap.data.get("SimulationEndReason", "")).lower()
                if is_collapse or "small radius" in end_reason or "collapsed" in end_reason:
                    mask[i, j] = True
                    collapse_mask[i, j] = True
                    continue

            # Get snapshot (persists final value if t > t_max)
            snapshot = get_snapshot_at_time(output, target_time, persist_final=True)
            force_vals = get_force_values(snapshot)

            if force_vals is None or all(np.isnan(v) for v in force_vals.values()):
                mask[i, j] = True
            else:
                for key in forces_dict:
                    forces_dict[key][i, j] = force_vals.get(key, np.nan)

    return forces_dict, mask, collapse_mask


def refine_dominant_map(logM, eps, forces_dict, mask, nref_M=300, nref_eps=300,
                        collapse_mask=None):
    """
    Interpolate continuous force fields, then take argmax for smooth boundaries.

    F_ram competes as a whole, then subclassifies to wind vs SN if it wins.
    Collapsed cells (from collapse_mask) are shown as COLLAPSED, not interpolated.
    """
    keys = list(forces_dict.keys())

    logM_f = np.linspace(logM.min(), logM.max(), nref_M)
    eps_f = np.linspace(eps.min(), eps.max(), nref_eps)
    E, L = np.meshgrid(eps_f, logM_f, indexing="ij")
    pts = np.stack([E.ravel(), L.ravel()], axis=-1)

    # Interpolate each force field
    fine_fields = {}
    for k in keys:
        arr = np.array(forces_dict[k], dtype=float)
        arr_masked = arr.copy()
        arr_masked[mask] = np.nan

        itp = RegularGridInterpolator(
            (eps, logM), arr_masked,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        fine_fields[k] = itp(pts).reshape(nref_eps, nref_M)

    # Interpolate collapse_mask to fine grid (nearest-neighbor)
    fine_collapse = None
    if collapse_mask is not None and np.any(collapse_mask):
        collapse_float = collapse_mask.astype(float)
        itp_c = RegularGridInterpolator(
            (eps, logM), collapse_float,
            method="nearest",
            bounds_error=False,
            fill_value=0.0
        )
        fine_collapse = itp_c(pts).reshape(nref_eps, nref_M) > 0.5

    # Get interpolated force arrays
    F_grav = np.abs(np.nan_to_num(fine_fields.get("F_grav", 0), nan=0.0))
    F_ram_wind = np.abs(np.nan_to_num(fine_fields.get("F_ram_wind", 0), nan=0.0))
    F_ram_SN = np.abs(np.nan_to_num(fine_fields.get("F_ram_SN", 0), nan=0.0))
    F_ion_out = np.abs(np.nan_to_num(fine_fields.get("F_ion_out", 0), nan=0.0))
    F_rad = np.abs(np.nan_to_num(fine_fields.get("F_rad", 0), nan=0.0))

    # F_ram competes as total
    F_ram_total = F_ram_wind + F_ram_SN

    # Stack competitors: {F_grav, F_ram_total, F_ion_out, F_rad}
    stack = np.stack([F_grav, F_ram_total, F_ion_out, F_rad], axis=-1)

    # Identify invalid cells
    invalid_f = np.all(stack == 0, axis=-1)

    # Get winner index (0=grav, 1=ram, 2=ion, 3=rad)
    winner_idx = np.argmax(stack, axis=-1)

    # Initialize dominant array
    dom_f = np.zeros_like(winner_idx, dtype=float)

    # Map winners to FORCE_FIELDS indices:
    # 0 = F_grav, 1 = F_ram_wind, 2 = F_ram_SN, 3 = F_ion_out, 4 = F_rad
    dom_f[winner_idx == 0] = 0  # F_grav
    dom_f[winner_idx == 2] = 3  # F_ion_out
    dom_f[winner_idx == 3] = 4  # F_rad

    # For F_ram winners (winner_idx == 1), subclassify
    ram_wins = (winner_idx == 1)
    sn_dominates = (F_ram_SN > F_ram_wind) & ram_wins
    wind_dominates = ~sn_dominates & ram_wins

    dom_f[sn_dominates] = 2    # F_ram_SN (yellow)
    dom_f[wind_dominates] = 1  # F_ram_wind (blue)

    # Overlay collapsed cells
    if fine_collapse is not None:
        dom_f[fine_collapse] = COLLAPSED
        # Un-invalidate collapsed cells so they render (not masked out)
        invalid_f = invalid_f & ~fine_collapse

    # Mask invalid cells
    dom_f = np.ma.array(dom_f, mask=invalid_f)

    return logM_f, eps_f, dom_f


# =============================================================================
# Plotting Functions
# =============================================================================

def create_colormap():
    """Create discrete colormap for force types, missing data, and collapse."""
    n_forces = len(FORCE_FIELDS)

    colors = [
        COLOR_COLLAPSED,          # -2 (white)
        COLOR_FILE_NOT_FOUND,     # -1 (gray)
    ] + [field[2] for field in FORCE_FIELDS]

    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')

    bounds = [-2.5, -1.5, -0.5] + [i + 0.5 for i in range(n_forces)]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def create_force_colormap():
    """Create discrete colormap for force types + collapse (for interpolated plots)."""
    n_forces = len(FORCE_FIELDS)
    colors = [
        COLOR_COLLAPSED,      # COLLAPSED = -2
    ] + [field[2] for field in FORCE_FIELDS]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')
    # Bounds: [-2.5, -0.5] -> collapsed, [-0.5, 0.5] -> force0, ...
    bounds = [-2.5, -0.5] + [i + 0.5 for i in range(n_forces)]
    norm = BoundaryNorm(bounds, len(colors))
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
                     collapse_mask=None, nref_M=300, nref_eps=300):
    """Plot a single dominance grid on an axis."""
    n_mass = len(mCloud_list)
    n_sfe = len(sfe_list)

    logM = np.array([np.log10(float(m)) for m in mCloud_list])
    eps = np.array([int(s) / 100.0 for s in sfe_list])

    if smooth == 'interp' and (forces_dict is None or mask is None):
        print("    Warning: smooth='interp' requires forces_dict and mask, falling back to 'none'")
        smooth = 'none'
    if axis_mode == 'discrete' and smooth != 'none':
        smooth = 'none'

    if axis_mode == 'continuous' and smooth == 'interp':
        logM_f, eps_f, dom_f = refine_dominant_map(logM, eps, forces_dict, mask,
                                                    nref_M=nref_M, nref_eps=nref_eps,
                                                    collapse_mask=collapse_mask)

        force_cmap, force_norm = create_force_colormap()

        ax.pcolormesh(
            10**logM_f, eps_f, dom_f,
            cmap=force_cmap,
            norm=force_norm,
            shading='nearest',
            linewidth=0,
            antialiased=False
        )

        ax.set_xscale('log')
        ax.set_xlim(10**logM.min(), 10**logM.max())
        ax.set_ylim(eps.min(), eps.max())
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    elif axis_mode == 'continuous':
        mass_min, mass_max = logM.min(), logM.max()
        eps_min, eps_max = eps.min(), eps.max()

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

        grid_masked = np.ma.masked_invalid(grid)

        ax.pcolormesh(
            logM, eps, grid_masked,
            cmap=cmap,
            norm=norm,
            shading='nearest',
            linewidth=0,
            antialiased=False
        )

        min_power = int(np.floor(extent[0]))
        max_power = int(np.ceil(extent[1]))
        x_ticks = np.arange(min_power, max_power + 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([rf"$10^{{{int(p)}}}$" for p in x_ticks])
        ax.set_xlim(extent[0], extent[1])

        ax.set_ylim(extent[2], extent[3])
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

    else:
        X, Y = np.meshgrid(
            np.arange(n_mass + 1) - 0.5,
            np.arange(n_sfe + 1) - 0.5
        )

        grid_masked = np.ma.masked_invalid(grid)

        ax.pcolormesh(
            X, Y, grid_masked,
            cmap=cmap,
            norm=norm,
            edgecolors='white',
            linewidths=1.0
        )

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


def create_legend():
    """Create legend handles for force types, collapse, and missing data."""
    handles = [
        Patch(facecolor=color, edgecolor='gray', label=label)
        for _, label, color in FORCE_FIELDS
    ]
    handles.append(
        Patch(facecolor=COLOR_COLLAPSED, edgecolor='gray', label='Collapse')
    )
    handles.append(
        Patch(facecolor=COLOR_FILE_NOT_FOUND, edgecolor='gray', label='No data')
    )
    return handles


# =============================================================================
# Main Function
# =============================================================================

def plot_grid(folder_path, target_times=None, output_dir=None, ndens_filter=None,
              smooth='none', axis_mode='discrete', mCloud_filter=None, sfe_filter=None):
    """
    Plot dominant feedback grid from simulations in a folder.

    Dynamically discovers simulations from the folder, organizes them into
    a grid by mCloud (x-axis) and SFE (y-axis).

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    target_times : list, optional
        Target times in Myr. Default: [1.0, 3.0, 5.0]
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
    smooth : str
        Smoothing method: 'none' or 'interp'
    axis_mode : str
        'discrete' or 'continuous'
    mCloud_filter : list of str, optional
        Filter simulations by cloud mass (e.g., ["1e6", "1e7"]).
    sfe_filter : list of str, optional
        Filter simulations by SFE (e.g., ["001", "010"]).
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    if target_times is None:
        target_times = DEFAULT_TIMES

    # Find all simulations
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    # Determine which densities to plot
    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print("=" * 60)
    print("Dominant Feedback Grid Plot")
    print("=" * 60)
    print(f"  Folder: {folder_path}")
    print(f"  Found {len(sim_files)} simulations")
    print(f"  Densities to plot: {ndens_to_plot}")
    print(f"  Times: {target_times} Myr")
    print(f"  Smooth: {smooth}")
    print(f"  Axis mode: {axis_mode}")

    if axis_mode == 'discrete' and smooth != 'none':
        print(f"  Warning: Smoothing only works with axis_mode='continuous', ignoring --smooth {smooth}")
        smooth = 'none'
    print()

    if output_dir is None:
        output_dir = FIG_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute layout for time snapshots
    n_times = len(target_times)
    nrows, ncols = compute_grid_layout(n_times)

    cmap, norm = create_colormap()

    # Create one grid per density
    for ndens in ndens_to_plot:
        print("-" * 60)
        print(f"Processing n={ndens}...")
        print("-" * 60)

        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
        mCloud_list = organized['mCloud_list']
        sfe_list = organized['sfe_list']
        sim_grid = organized['grid']

        if not mCloud_list or not sfe_list:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        # Create figure
        fig_width = 5.0 * ncols
        fig_height = 5.0 * nrows + 0.8
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
            squeeze=False
        )

        for idx, target_time in enumerate(target_times):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            print(f"  Building grid for t = {target_time} Myr...")

            if smooth == 'interp' and axis_mode == 'continuous':
                forces_dict, mask, collapse_mask = build_force_grids(
                    target_time, mCloud_list, sfe_list, sim_grid
                )
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, sim_grid
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode,
                               forces_dict=forces_dict, mask=mask,
                               collapse_mask=collapse_mask)
            else:
                grid = build_dominance_grid(
                    target_time, mCloud_list, sfe_list, sim_grid
                )
                plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                               smooth=smooth, axis_mode=axis_mode)

        # Hide unused subplots
        for idx in range(n_times, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)

        # Axis labels
        fig.supxlabel(r"$M_{\rm cloud}$ [$M_\odot$]", fontsize=12)
        fig.supylabel(r"Star Formation Efficiency $\epsilon$", fontsize=12)

        # Title
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(
            rf"Dominant Feedback ($n_{{\rm core}} = 10^{{{nlog}}}$ cm$^{{-3}}$)",
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

        # Save
        save_dir = output_dir / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = build_filename(
            'dominantFeedback',
            nCore=ndens,
            axis_mode=axis_mode,
            smooth=smooth if axis_mode == 'continuous' else None
        )
        out_pdf = save_dir / f"{filename}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"  Saved: {out_pdf}")

        plt.show()
        print()


# =============================================================================
# Movie Generation
# =============================================================================

def make_movie(folder_path, output_dir=None, ndens_filter=None,
               smooth='none', axis_mode='discrete',
               t_start=0.0, t_end=5.0, dt=0.05, fps=10,
               mCloud_filter=None, sfe_filter=None):
    """Create an animated GIF showing the evolution of dominant feedback over time."""
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL (Pillow) is required for movie generation.")
        print("Install with: pip install Pillow")
        return

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # Find all simulations
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    # Determine which densities to plot
    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print("=" * 60)
    print("Dominant Feedback Movie Generation")
    print("=" * 60)
    print(f"  Folder: {folder_path}")
    print(f"  Time range: {t_start} - {t_end} Myr, dt = {dt} Myr")
    print(f"  FPS: {fps} (frame duration: {1000/fps:.0f} ms)")
    print(f"  Smooth: {smooth}, Axis mode: {axis_mode}")
    print()

    if output_dir is None:
        output_dir = FIG_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.arange(t_start, t_end + dt/2, dt)
    n_frames = len(times)
    print(f"Generating {n_frames} frames...")

    if axis_mode == 'discrete' and smooth != 'none':
        print(f"  Warning: Smoothing only works with axis_mode='continuous', ignoring --smooth {smooth}")
        smooth = 'none'

    cmap, norm = create_colormap()

    # Create one movie per density
    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")

        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
        mCloud_list = organized['mCloud_list']
        sfe_list = organized['sfe_list']
        sim_grid = organized['grid']

        if not mCloud_list or not sfe_list:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        temp_dir = tempfile.mkdtemp(prefix='dominant_feedback_movie_')
        frame_paths = []

        try:
            for frame_idx, target_time in enumerate(times):
                print(f"  Frame {frame_idx + 1}/{n_frames}: t = {target_time:.3f} Myr", end='\r')

                fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

                if smooth == 'interp' and axis_mode == 'continuous':
                    forces_dict, mask, collapse_mask = build_force_grids(
                        target_time, mCloud_list, sfe_list, sim_grid
                    )
                    grid = build_dominance_grid(
                        target_time, mCloud_list, sfe_list, sim_grid
                    )
                    plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                                   smooth=smooth, axis_mode=axis_mode,
                                   forces_dict=forces_dict, mask=mask,
                                   collapse_mask=collapse_mask)
                else:
                    grid = build_dominance_grid(
                        target_time, mCloud_list, sfe_list, sim_grid
                    )
                    plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm,
                                   smooth=smooth, axis_mode=axis_mode)

                ax.set_xlabel(r"$M_{\rm cloud}$ [$M_\odot$]", fontsize=12)
                ax.set_ylabel(r"Star Formation Efficiency $\epsilon$", fontsize=12)

                nlog = int(np.log10(float(ndens)))
                ax.set_title(
                    rf"Dominant Feedback at $t = {target_time:.2f}$ Myr "
                    rf"($n_{{\rm core}} = 10^{{{nlog}}}$ cm$^{{-3}}$)",
                    fontsize=12
                )

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

                frame_path = Path(temp_dir) / f"frame_{frame_idx:04d}.png"
                fig.savefig(frame_path, dpi=150, bbox_inches='tight')
                frame_paths.append(frame_path)
                plt.close(fig)

            print()
            print(f"  Assembling GIF from {len(frame_paths)} frames...")

            frames = [Image.open(fp) for fp in frame_paths]

            frame_duration = int(1000 / fps)

            save_dir = output_dir / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = build_filename(
                'dominantFeedback_movie',
                nCore=ndens,
                axis_mode=axis_mode,
                smooth=smooth if axis_mode == 'continuous' else None,
                t_range=(t_start, t_end)
            )
            out_gif = save_dir / f"{filename}.gif"

            frames[0].save(
                out_gif,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0
            )

            print(f"  Saved: {out_gif}")
            print(f"  Duration: {len(frames) * frame_duration / 1000:.1f} seconds")

        finally:
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
  # Plot from folder (auto-discovers simulations)
  python paper_dominantFeedback.py -F /path/to/outputs/sweep_test

  # Filter by density
  python paper_dominantFeedback.py -F /path/to/outputs --nCore 1e4

  # Custom time snapshots
  python paper_dominantFeedback.py -F /path/to/outputs --times 1.0 3.0 5.0

  # Filter by cloud mass and/or SFE
  python paper_dominantFeedback.py -F /path/to/outputs --mCloud 1e6 1e7
  python paper_dominantFeedback.py -F /path/to/outputs --sfe 001 010 020
  python paper_dominantFeedback.py -F /path/to/outputs -n 1e4 --mCloud 1e7 --sfe 010

  # Scan folder for available parameters
  python paper_dominantFeedback.py -F /path/to/outputs --info

  # With interpolation smoothing
  python paper_dominantFeedback.py -F /path/to/outputs --axis-mode continuous --smooth interp

  # Movie generation
  python paper_dominantFeedback.py -F /path/to/outputs --movie --dt 0.05 --fps 10

Smoothing methods (only with --axis-mode continuous):
  none   - Discrete grid, each cell shows dominant force at that point
  interp - Interpolate underlying force fields on fine grid, then take argmax

Axis modes:
  discrete   - Equal spacing with categorical labels
  continuous - Real value spacing (log scale for mCloud, linear for SFE)

Note: F_ram competes as a whole first, then subclassifies to wind (blue) or
SN (yellow) if it wins. Simulations that ended in collapse are shown as white;
others that have ended persist their final dominant feedback.
        """
    )

    parser.add_argument(
        '--folder', '-F', required=True,
        help='Path to folder containing simulation subfolders'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by density (e.g., "1e4"). If not specified, '
             'generates one PDF per density found.'
    )
    parser.add_argument(
        '--mCloud', nargs='+', default=None,
        help='Filter simulations by cloud mass (e.g., --mCloud 1e6 1e7). '
             'Only shows specified mCloud values on the grid.'
    )
    parser.add_argument(
        '--sfe', nargs='+', default=None,
        help='Filter simulations by SFE (e.g., --sfe 001 010). '
             'Only shows specified SFE values on the grid.'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Scan folder and print available mCloud, SFE, and nCore values, then exit.'
    )
    parser.add_argument(
        '--times', '-t', nargs='+', type=float, default=None,
        help=f'Target times in Myr (e.g., 1.0 3.0 5.0). Default: {DEFAULT_TIMES}'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help=f'Directory to save figures. Default: {FIG_DIR}'
    )
    parser.add_argument(
        '--smooth', choices=['none', 'interp'], default=DEFAULT_SMOOTH,
        help=f"Smoothing method (only with --axis-mode continuous). Default: {DEFAULT_SMOOTH}"
    )
    parser.add_argument(
        '--axis-mode', choices=['discrete', 'continuous'], default=DEFAULT_AXIS_MODE,
        help=f"Axis spacing mode. Default: {DEFAULT_AXIS_MODE}"
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
        help='Frames per second / playback speed (default: 10)'
    )
    parser.add_argument(
        '--t-start', type=float, default=0.0,
        help='Start time for movie in Myr (default: 0.0)'
    )
    parser.add_argument(
        '--t-end', type=float, default=5.0,
        help='End time for movie in Myr (default: 5.0)'
    )

    args = parser.parse_args()

    if args.info:
        # Info mode: scan folder and print available parameters
        info = info_simulations(args.folder)
        print("=" * 50)
        print(f"Simulation parameters in: {args.folder}")
        print("=" * 50)
        print(f"  Total simulations: {info['count']}")
        print(f"  mCloud values: {info['mCloud']}")
        print(f"  SFE values: {info['sfe']}")
        print(f"  nCore values: {info['ndens']}")
    elif args.movie:
        make_movie(
            args.folder,
            output_dir=args.output_dir,
            ndens_filter=args.nCore,
            smooth=args.smooth,
            axis_mode=args.axis_mode,
            t_start=args.t_start,
            t_end=args.t_end,
            dt=args.dt,
            fps=args.fps,
            mCloud_filter=args.mCloud,
            sfe_filter=args.sfe
        )
    else:
        plot_grid(
            args.folder,
            target_times=args.times,
            output_dir=args.output_dir,
            ndens_filter=args.nCore,
            smooth=args.smooth,
            axis_mode=args.axis_mode,
            mCloud_filter=args.mCloud,
            sfe_filter=args.sfe
        )
