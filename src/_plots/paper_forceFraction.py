#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Fraction Stacked Area Plot for TRINITY

Shows the relative importance of different feedback forces as fractions
of the total force budget over time.

Forces shown (all mechanically distinct and additive):
- F_thermal: Thermal pressure force (4*pi*R2^2 * P_drive) - combines hot bubble + warm HII
- F_rad: Radiation pressure force
- F_grav: Gravitational force

This is physically correct because these forces ARE additive in the equation
of motion. The thermal force is the combined driving pressure, not the
individual P_b and P_IF pressures (which are not additive).

Fractions are computed as: F_i / F_tot where F_tot = sum(|F_i|)
All fractions sum to 1.0 at each timestep.

Author: TRINITY Team
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting force fractions (F_thermal, F_rad, F_grav)")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 11  # None or 1 disables
PHASE_CHANGE = True
USE_LOG_X = False  # Use log scale for x-axis (time)

# Force colors (consistent with existing TRINITY plots)
C_GRAV = "black"
C_THERMAL = "blue"  # Thermal = bubble + HII combined
C_RAD = "#9467bd"   # Purple for radiation

# Force fields to show
FORCE_FIELDS = [
    ("F_grav",    r"$F_{\rm grav}$",    C_GRAV),
    ("F_thermal", r"$F_{\rm thermal}$", C_THERMAL),
    ("F_rad",     r"$F_{\rm rad}$",     C_RAD),
]

# --- output
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


def smooth_1d(y, window, mode="edge"):
    """Apply 1D smoothing with moving average."""
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def smooth_2d(arr, window, mode="edge"):
    """Apply smoothing to 2D array (each row separately)."""
    if window is None or window <= 1:
        return arr
    return np.vstack([smooth_1d(row, window, mode=mode) for row in arr])


def load_run(data_path: Path):
    """Load run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Core time series
    t = output.get('t_now')
    R2 = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))

    # Helper to get field with default
    def get_field(field, default=np.nan):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    # Get individual forces
    F_grav = get_field('F_grav', 0.0)
    F_rad = get_field('F_rad', 0.0)

    # Thermal force: F_ram (from Pb) + F_HII (from P_IF contribution)
    # F_ram = Pb * 4*pi*R2^2, F_HII captures the P_IF excess
    F_ram = get_field('F_ram', 0.0)
    F_HII = get_field('F_HII', 0.0)

    # Alternative: compute from P_drive directly if available
    P_drive = get_field('P_drive', np.nan)
    if not np.all(np.isnan(P_drive)):
        # F_thermal = P_drive * 4*pi*R2^2
        F_thermal = P_drive * 4 * np.pi * R2**2
    else:
        # Fallback: use F_ram + F_HII
        F_thermal = np.nan_to_num(F_ram, nan=0.0) + np.nan_to_num(F_HII, nan=0.0)

    # Clean up NaN
    F_grav = np.nan_to_num(F_grav, nan=0.0)
    F_rad = np.nan_to_num(F_rad, nan=0.0)
    F_thermal = np.nan_to_num(F_thermal, nan=0.0)

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        F_grav = F_grav[order]
        F_thermal = F_thermal[order]
        F_rad = F_rad[order]
        isCollapse = isCollapse[order]

    # Stack forces in order: grav, thermal, rad
    forces = np.vstack([F_grav, F_thermal, F_rad])

    return {
        't': t, 'R2': R2, 'phase': phase,
        'forces': forces,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, alpha=0.75,
                   use_log_x=False):
    """Plot force fractions on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']
    forces = data['forces'].copy()

    # Add phase markers
    add_plot_markers(
        ax, t,
        phase=phase if phase_change else None,
        R2=R2 if show_rcloud else None,
        rcloud=rcloud if show_rcloud else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=phase_change,
        show_rcloud=show_rcloud,
        show_collapse=show_collapse
    )

    # Apply smoothing
    forces = smooth_2d(forces, smooth_window)

    # Use absolute values for force fractions
    forces_abs = np.abs(forces)

    # Compute total and fractions
    ftotal = forces_abs.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)
    frac = forces_abs / ftotal

    # Cumulative sum for stacking
    cum = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    # Fill stacked areas
    for (field, label, color), y0, y1 in zip(FORCE_FIELDS, prev, cum):
        ax.fill_between(t, y0, y1, color=color, alpha=alpha, lw=0, zorder=2)

    # Add reference line at 0.5
    ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.5, zorder=1)

    ax.set_ylim(0, 1)

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot force fractions from a direct data path/folder."""
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading data from: {data_path}")

    try:
        data = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$|F_i|/F_{\rm tot}$")
    ax.set_title(f"Force Fractions: {data_path.parent.name}")

    # Legend
    handles = [
        Patch(facecolor=C_GRAV, alpha=0.75, label=r"$F_{\rm grav}$"),
        Patch(facecolor=C_THERMAL, alpha=0.75, label=r"$F_{\rm thermal}$"),
        Patch(facecolor=C_RAD, alpha=0.75, label=r"$F_{\rm rad}$"),
    ]
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_forceFraction_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid(folder_path, output_dir=None, ndens_filter=None):
    """
    Plot grid of force fractions from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid, get_unique_ndens

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    # Determine which densities to plot
    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print(f"Found {len(sim_files)} simulations")
    print(f"  Densities to plot: {ndens_to_plot}")

    # Create one grid per density
    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")
        organized = organize_simulations_for_grid(sim_files, ndens_filter=ndens)
        mCloud_list_found = organized['mCloud_list']
        sfe_list_found = organized['sfe_list']
        grid = organized['grid']

        if not mCloud_list_found or not sfe_list_found:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list_found}")
        print(f"  SFE: {sfe_list_found}")

        nrows, ncols = len(mCloud_list_found), len(sfe_list_found)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.0 * ncols, 2.4 * nrows),
            sharex=False, sharey=True,
            dpi=300,
            squeeze=False,
            constrained_layout=False
        )

        for i, mCloud in enumerate(mCloud_list_found):
            for j, sfe in enumerate(sfe_list_found):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"    Loading: {data_path}")
                try:
                    data = load_run(data_path)
                    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                                   phase_change=PHASE_CHANGE, use_log_x=USE_LOG_X)
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                ax.tick_params(axis="x", which="both", bottom=True)
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                if j == 0:
                    mval = float(mCloud)
                    mexp = int(np.floor(np.log10(mval)))
                    mcoeff = round(mval / (10 ** mexp))
                    if mcoeff == 10:
                        mcoeff = 1
                        mexp += 1
                    if mcoeff == 1:
                        mlabel = rf"$M_{{\rm cl}}=10^{{{mexp}}}$"
                    else:
                        mlabel = rf"$M_{{\rm cl}}={mcoeff}\times10^{{{mexp}}}$"
                    ax.set_ylabel(mlabel + "\n" + r"$|F_i|/F_{\rm tot}$")
                else:
                    ax.tick_params(labelleft=False)

        handles = [
            Patch(facecolor=C_GRAV, alpha=0.75, label=r"$F_{\rm grav}$ (Gravity)"),
            Patch(facecolor=C_THERMAL, alpha=0.75, label=r"$F_{\rm thermal}$ (Thermal pressure)"),
            Patch(facecolor=C_RAD, alpha=0.75, label=r"$F_{\rm rad}$ (Radiation)"),
        ]
        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)
        ndens_tag = f"n{ndens}"
        fig.suptitle(f"{folder_name} ({ndens_tag})", fontsize=14, y=1.02)

        leg = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.0)
        )
        leg.set_zorder(10)

        # Save figure to ./fig/{folder_name}/forceFraction_{ndens_tag}.pdf
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"forceFraction_{ndens_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY force fractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_forceFraction.py 1e7_sfe020_n1e4
  python paper_forceFraction.py /path/to/outputs/1e7_sfe020_n1e4

  # Grid plot from folder (auto-discovers simulations)
  python paper_forceFraction.py --folder /path/to/my_experiment/
  python paper_forceFraction.py -F /path/to/simulations/ -n 1e4
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path (for single simulation)'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Directory to save output figures (default: fig/)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Create grid plot from all simulations in folder.'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3").'
    )

    args = parser.parse_args()

    if args.folder:
        plot_grid(args.folder, args.output_dir, ndens_filter=args.nCore)
    elif args.data:
        plot_from_path(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
