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

# Default parameters
DEFAULT_MCLOUD = ["1e5", "1e7", "1e8"]
DEFAULT_SFE = ["001", "010", "020", "030", "050", "080"]
DEFAULT_NCORE = ["1e4"]  # List of nCore values - produces one plot per nCore
DEFAULT_TIMES = [1.0, 1.5, 2.0, 2.5]  # Myr

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
        Values: 0-3 for force index, NaN for no data
    """
    n_sfe = len(sfe_list)
    n_mass = len(mCloud_list)
    grid = np.full((n_sfe, n_mass), np.nan, dtype=float)

    for j, mCloud in enumerate(mCloud_list):
        for i, sfe in enumerate(sfe_list):
            run_name = f"{mCloud}_sfe{sfe}_n{nCore}"
            display_name = f"{run_name}_modified" if use_modified else run_name

            output = load_simulation(base_dir, run_name, use_modified=use_modified)
            if output is None:
                print(f"    {display_name}: not found")
                continue

            # Check time bounds before attempting to get snapshot
            if target_time < output.t_min or target_time > output.t_max:
                print(f"    {display_name}: t={target_time} outside range [{output.t_min:.3f}, {output.t_max:.3f}]")
                continue

            snapshot = get_snapshot_at_time(output, target_time)
            dominant = get_dominant_force(snapshot)
            grid[i, j] = dominant

            if np.isfinite(dominant):
                force_name = FORCE_FIELDS[int(dominant)][1]
                print(f"    {display_name}: {force_name}")

    return grid


# =============================================================================
# Plotting Functions
# =============================================================================

def create_colormap():
    """Create discrete colormap for force types."""
    colors = [field[2] for field in FORCE_FIELDS]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')  # NaN -> white

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

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


def plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm):
    """
    Plot a single dominance grid on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    grid : np.ndarray
        2D array of dominant force indices
    mCloud_list : list
        Cloud mass labels
    sfe_list : list
        SFE labels
    target_time : float
        Time in Myr (for title)
    cmap : ListedColormap
        Colormap
    norm : BoundaryNorm
        Color normalization
    """
    n_mass = len(mCloud_list)
    n_sfe = len(sfe_list)

    # Create pcolormesh grid
    X, Y = np.meshgrid(
        np.arange(n_mass + 1) - 0.5,
        np.arange(n_sfe + 1) - 0.5
    )

    # Mask NaN values for white display
    grid_masked = np.ma.masked_invalid(grid)

    # Plot
    im = ax.pcolormesh(
        X, Y, grid_masked,
        cmap=cmap,
        norm=norm,
        edgecolors='white',
        linewidths=1.0
    )

    # Axis labels
    ax.set_xticks(np.arange(n_mass))
    ax.set_xticklabels([rf"$10^{{{int(np.log10(float(m)))}}}$" for m in mCloud_list])

    ax.set_yticks(np.arange(n_sfe))
    ax.set_yticklabels([f"{int(s)/100:.2f}" for s in sfe_list])

    ax.set_title(rf"$t = {target_time}$ Myr", fontsize=11)

    return im


def create_legend():
    """Create legend handles for force types."""
    handles = [
        Patch(facecolor=color, edgecolor='gray', label=label)
        for _, label, color in FORCE_FIELDS
    ]
    handles.append(
        Patch(facecolor='white', edgecolor='gray', label='No data')
    )
    return handles


# =============================================================================
# Main Function
# =============================================================================

def main(mCloud_list, sfe_list, nCore_list, target_times, base_dir, fig_dir=None, use_modified=False):
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

            plot_single_grid(ax, grid, mCloud_list, sfe_list, target_time, cmap, norm)
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
  python paper_dominantFeedback.py --output-dir /path/to/outputs --fig-dir /path/to/figs
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

    args = parser.parse_args()

    # Apply defaults
    mCloud_list = args.mCloud if args.mCloud else DEFAULT_MCLOUD
    sfe_list = args.sfe if args.sfe else DEFAULT_SFE
    nCore_list = args.nCore if args.nCore else DEFAULT_NCORE
    target_times = args.times if args.times else DEFAULT_TIMES
    base_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    fig_dir = Path(args.fig_dir) if args.fig_dir else None

    # Use module-level USE_MODIFIED if --modified not explicitly set
    use_modified = args.modified or USE_MODIFIED

    main(mCloud_list, sfe_list, nCore_list, target_times, base_dir, fig_dir, use_modified)
