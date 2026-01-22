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
- Color: dominant feedback force (F_grav, F_ram, F_ion_out, F_rad)
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
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

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

# Force field definitions: (key, label, color)
# Order determines the index (0, 1, 2, 3) used in the grid
FORCE_FIELDS = [
    ("F_grav",     "Gravity",           "#2c3e50"),  # Dark blue-gray
    ("F_ram",      "Ram pressure",      "#3498db"),  # Blue
    ("F_ion_out",  "Photoionised gas",  "#e74c3c"),  # Red
    ("F_rad",      "Radiation",         "#9b59b6"),  # Purple
]

# Special values for missing data (must be negative to distinguish from force indices 0-3)
FILE_NOT_FOUND = -1      # Gray: simulation file not found
TIME_OUT_OF_RANGE = -2   # White: time outside simulation range (t > t_max or t < t_min)

# Missing data colors
COLOR_FILE_NOT_FOUND = "#808080"      # Gray
COLOR_TIME_OUT_OF_RANGE = "#ffffff"   # White

# Default parameters
DEFAULT_MCLOUD = ["1e5", "1e7", "1e8"]
DEFAULT_SFE = ["001", "010", "020", "030", "050", "080"]
DEFAULT_NCORE = ["1e4"]  # List of nCore values - produces one plot per nCore
DEFAULT_TIMES = [1.0, 1.5, 2.0, 2.5]  # Myr

# Smoothing options: 'none', 'gaussian', 'contour'
DEFAULT_SMOOTH = 'none'

# Axis mode options:
#   'discrete': equal spacing, categorical labels (default)
#   'continuous': real value spacing (log for mCloud, linear for SFE)
DEFAULT_AXIS_MODE = 'discrete'

# Default directories
DEFAULT_OUTPUT_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"
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


def get_dominant_force(snapshot):
    """
    Determine the dominant feedback force from a snapshot.

    Parameters
    ----------
    snapshot : dict
        Snapshot data dictionary

    Returns
    -------
    int or np.nan
        Index of dominant force (0-3), or np.nan if no valid forces
    """
    if snapshot is None:
        return np.nan

    forces = []
    for key, _, _ in FORCE_FIELDS:
        val = snapshot.get(key)
        if val is None or not np.isfinite(val):
            forces.append(0.0)
        else:
            forces.append(abs(val))

    # Check if we have any valid force data
    total = sum(forces)
    if total == 0 or not np.isfinite(total):
        return np.nan

    return int(np.argmax(forces))


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


# =============================================================================
# Plotting Functions
# =============================================================================

def create_colormap():
    """
    Create discrete colormap for force types and missing data.

    Color mapping:
    - -2 (TIME_OUT_OF_RANGE): white
    - -1 (FILE_NOT_FOUND): gray
    - 0 (F_grav): dark blue-gray
    - 1 (F_ram): blue
    - 2 (F_ion_out): red
    - 3 (F_rad): purple
    """
    # Colors in order: TIME_OUT_OF_RANGE, FILE_NOT_FOUND, then force colors
    colors = [
        COLOR_TIME_OUT_OF_RANGE,  # -2
        COLOR_FILE_NOT_FOUND,     # -1
    ] + [field[2] for field in FORCE_FIELDS]  # 0, 1, 2, 3

    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # NaN -> white (fallback)

    # Bounds: -2.5 to -1.5 -> color[0], -1.5 to -0.5 -> color[1], etc.
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def apply_smoothing(grid, method='none', sigma=0.8, upsample=4):
    """
    Apply smoothing to the dominance grid for contour-like appearance.

    Parameters
    ----------
    grid : np.ndarray
        2D array of dominant force indices
    method : str
        'none': no smoothing (discrete grid)
        'gaussian': Gaussian blur for soft transitions
        'contour': upsampled nearest-neighbor for smooth boundaries
    sigma : float
        Gaussian sigma for 'gaussian' method
    upsample : int
        Upsampling factor for 'contour' method

    Returns
    -------
    grid_smooth : np.ndarray
        Smoothed grid (may be larger if upsampled)
    extent : tuple or None
        New extent for imshow if upsampled, else None
    """
    if method == 'none':
        return grid, None

    n_sfe, n_mass = grid.shape

    if method == 'gaussian':
        # Create separate binary masks for each force type
        # Then smooth each and take argmax
        n_forces = len(FORCE_FIELDS)
        masks = np.zeros((n_forces, n_sfe, n_mass))

        for force_idx in range(n_forces):
            masks[force_idx] = (grid == force_idx).astype(float)

        # Apply Gaussian filter to each mask
        smoothed_masks = np.array([
            gaussian_filter(mask, sigma=sigma, mode='nearest')
            for mask in masks
        ])

        # Handle missing data: where original was negative, keep it
        missing_mask = grid < 0

        # Take argmax of smoothed masks
        grid_smooth = np.argmax(smoothed_masks, axis=0).astype(float)

        # Restore missing data markers
        grid_smooth[missing_mask] = grid[missing_mask]

        return grid_smooth, None

    elif method == 'contour':
        # Upsample using nearest-neighbor for sharp but smooth boundaries
        new_sfe = n_sfe * upsample
        new_mass = n_mass * upsample

        # Create coordinate grids
        x_old = np.arange(n_mass)
        y_old = np.arange(n_sfe)
        x_new = np.linspace(0, n_mass - 1, new_mass)
        y_new = np.linspace(0, n_sfe - 1, new_sfe)

        # Use nearest-neighbor interpolation for each force type
        # to maintain discrete categories
        grid_smooth = np.zeros((new_sfe, new_mass))

        for iy, y in enumerate(y_new):
            for ix, x in enumerate(x_new):
                # Find nearest original cell
                orig_y = int(round(y))
                orig_x = int(round(x))
                orig_y = min(max(orig_y, 0), n_sfe - 1)
                orig_x = min(max(orig_x, 0), n_mass - 1)
                grid_smooth[iy, ix] = grid[orig_y, orig_x]

        # Extent for plotting
        extent = (-0.5, n_mass - 0.5, -0.5, n_sfe - 0.5)

        return grid_smooth, extent

    return grid, None


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
                     smooth='none', axis_mode='discrete'):
    """
    Plot a single dominance grid on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    grid : np.ndarray
        2D array of dominant force indices
    mCloud_list : list
        Cloud mass labels (strings like "1e7")
    sfe_list : list
        SFE labels (strings like "010")
    target_time : float
        Time in Myr (for title)
    cmap : ListedColormap
        Colormap
    norm : BoundaryNorm
        Color normalization
    smooth : str
        Smoothing method: 'none', 'gaussian', 'contour'
    axis_mode : str
        'discrete': equal spacing with categorical labels
        'continuous': real value spacing (log for mCloud, linear for SFE)
    """
    n_mass = len(mCloud_list)
    n_sfe = len(sfe_list)

    # Convert to real values for continuous mode
    mass_values = np.array([np.log10(float(m)) for m in mCloud_list])  # log10(Msun)
    sfe_values = np.array([int(s) / 100.0 for s in sfe_list])  # decimal SFE

    # Apply smoothing if requested
    grid_plot, extent = apply_smoothing(grid, method=smooth)

    if axis_mode == 'continuous':
        # Real-value axis spacing
        # For pcolormesh, we need cell edges
        # Create edges at midpoints between values, plus outer edges
        if n_mass > 1:
            mass_edges = np.zeros(n_mass + 1)
            mass_edges[0] = mass_values[0] - (mass_values[1] - mass_values[0]) / 2
            mass_edges[-1] = mass_values[-1] + (mass_values[-1] - mass_values[-2]) / 2
            for i in range(1, n_mass):
                mass_edges[i] = (mass_values[i-1] + mass_values[i]) / 2
        else:
            mass_edges = np.array([mass_values[0] - 0.5, mass_values[0] + 0.5])

        if n_sfe > 1:
            sfe_edges = np.zeros(n_sfe + 1)
            sfe_edges[0] = sfe_values[0] - (sfe_values[1] - sfe_values[0]) / 2
            sfe_edges[-1] = sfe_values[-1] + (sfe_values[-1] - sfe_values[-2]) / 2
            for i in range(1, n_sfe):
                sfe_edges[i] = (sfe_values[i-1] + sfe_values[i]) / 2
        else:
            sfe_edges = np.array([sfe_values[0] - 0.05, sfe_values[0] + 0.05])

        X, Y = np.meshgrid(mass_edges, sfe_edges)

        # Mask NaN values
        grid_masked = np.ma.masked_invalid(grid_plot)

        im = ax.pcolormesh(
            X, Y, grid_masked,
            cmap=cmap,
            norm=norm,
            edgecolors='white' if smooth == 'none' else 'none',
            linewidths=0.5 if smooth == 'none' else 0
        )

        # Set axis to real values
        ax.set_xticks(mass_values)
        ax.set_xticklabels([rf"$10^{{{int(m)}}}$" for m in mass_values])
        ax.set_xlim(mass_edges[0], mass_edges[-1])

        ax.set_yticks(sfe_values)
        ax.set_yticklabels([f"{s:.2f}" for s in sfe_values])
        ax.set_ylim(sfe_edges[0], sfe_edges[-1])

    else:
        # Discrete mode (original behavior)
        if smooth == 'none':
            # Original discrete grid with cell borders
            X, Y = np.meshgrid(
                np.arange(n_mass + 1) - 0.5,
                np.arange(n_sfe + 1) - 0.5
            )

            # Mask NaN values
            grid_masked = np.ma.masked_invalid(grid_plot)

            im = ax.pcolormesh(
                X, Y, grid_masked,
                cmap=cmap,
                norm=norm,
                edgecolors='white',
                linewidths=1.0
            )
        else:
            # Smoothed grid - use imshow for better appearance
            grid_masked = np.ma.masked_invalid(grid_plot)

            if extent is None:
                extent = (-0.5, n_mass - 0.5, -0.5, n_sfe - 0.5)

            im = ax.imshow(
                grid_masked,
                cmap=cmap,
                norm=norm,
                origin='lower',
                extent=extent,
                aspect='auto',
                interpolation='nearest' if smooth == 'contour' else 'bilinear'
            )

        # Discrete axis labels
        ax.set_xticks(np.arange(n_mass))
        ax.set_xticklabels([rf"$10^{{{int(np.log10(float(m)))}}}$" for m in mCloud_list])

        ax.set_yticks(np.arange(n_sfe))
        ax.set_yticklabels([f"{int(s)/100:.2f}" for s in sfe_list])

    ax.set_title(rf"$t = {target_time}$ Myr", fontsize=11)

    return im


def create_legend():
    """Create legend handles for force types and missing data."""
    handles = [
        Patch(facecolor=color, edgecolor='gray', label=label)
        for _, label, color in FORCE_FIELDS
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
        Smoothing method: 'none', 'gaussian', 'contour'
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

        # Create figure
        fig_width = 3.5 * ncols
        fig_height = 3.0 * nrows + 0.5  # Extra space for legend
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
        filename = f"dominant_feedback_n{nCore}"
        if use_modified:
            filename = f"{filename}_modified"
        if smooth != 'none':
            filename = f"{filename}_{smooth}"
        out_pdf = fig_dir / f"{filename}.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)
        print()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot dominant feedback grid across mCloud-SFE parameter space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_dominantFeedback.py
  python paper_dominantFeedback.py --mCloud 1e7 1e8 --sfe 001 020 --times 1 1.5 2 2.5
  python paper_dominantFeedback.py --nCore 1e4 1e5 --modified
  python paper_dominantFeedback.py --smooth gaussian
  python paper_dominantFeedback.py --axis-mode continuous
  python paper_dominantFeedback.py --output-dir /path/to/outputs --fig-dir /path/to/figs

Smoothing methods:
  none     - Discrete grid with cell borders (default)
  gaussian - Gaussian blur for soft color transitions
  contour  - Upsampled nearest-neighbor for smooth region boundaries

Axis modes:
  discrete   - Equal spacing with categorical labels (default)
  continuous - Real value spacing (log scale for mCloud, linear for SFE)
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
        '--smooth', choices=['none', 'gaussian', 'contour'], default=None,
        help=f"Smoothing method for contour-like appearance. Default: {DEFAULT_SMOOTH}"
    )
    parser.add_argument(
        '--axis-mode', choices=['discrete', 'continuous'], default=None,
        help=f"Axis spacing mode: 'discrete' for equal spacing with categorical labels, "
             f"'continuous' for real value spacing (log for mCloud, linear for SFE). Default: {DEFAULT_AXIS_MODE}"
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

    main(mCloud_list, sfe_list, nCore_list, target_times, base_dir, fig_dir, use_modified, smooth, axis_mode)
