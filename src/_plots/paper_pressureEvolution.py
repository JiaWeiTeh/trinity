#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure Evolution Plot for TRINITY

Shows how P_drive transitions between P_b (hot bubble) and P_IF (ionization front)
over time. This illustrates the convex blend model:

    P_drive = (1 - w) * P_b + w * P_IF

where w = f_abs_ion * P_IF / (P_IF + P_b)

Plot shows:
- P_b(t): Hot bubble pressure (blue solid)
- P_IF(t): Ionization front pressure (red solid)
- P_drive(t): Effective driving pressure (black dashed)
- Optionally: P_ext (external pressure, gray dotted)

Author: TRINITY Team
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV, CGS

print("...plotting pressure evolution (P_b, P_IF, P_drive)")

# Unit conversion: code units (Msun/pc/Myr²) → K/cm³ (= P/k_B = n*T)
# P[K/cm³] = P[code] * Pb_au2cgs / k_B
P_AU_TO_K_CM3 = INV_CONV.Pb_au2cgs / CGS.k_B

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
SHOW_PEXT = True  # Show external pressure
USE_LOG_X = False  # Use log scale for x-axis (time)

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

# Pressure field styling
PRESSURE_FIELDS = [
    ("Pb",      r"$P_b$ (bubble)",      "blue",  "-",  1.8),
    ("P_IF",    r"$P_{\rm IF}$",        "red",   "-",  1.8),
    ("P_drive", r"$P_{\rm drive}$",     "black", "--", 2.2),
]

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

    # Pressure fields - convert from code units to K/cm³
    Pb = get_field('Pb', np.nan) * P_AU_TO_K_CM3
    P_IF = get_field('P_IF', np.nan) * P_AU_TO_K_CM3
    P_drive = get_field('P_drive', np.nan) * P_AU_TO_K_CM3
    press_HII_in = get_field('press_HII_in', np.nan) * P_AU_TO_K_CM3

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        Pb = Pb[order]
        P_IF = P_IF[order]
        P_drive = P_drive[order]
        press_HII_in = press_HII_in[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'Pb': Pb, 'P_IF': P_IF, 'P_drive': P_drive,
        'press_HII_in': press_HII_in,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, show_pext=True,
                   use_log_x=False):
    """Plot pressure evolution on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']

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

    # Plot pressures
    for field, label, color, ls, lw in PRESSURE_FIELDS:
        y = data[field]
        if smooth_window:
            y = smooth_1d(y, smooth_window)

        # Skip if all NaN
        if np.all(~np.isfinite(y)):
            continue

        ax.plot(t, y, color=color, ls=ls, lw=lw, label=label, zorder=3)

    # Optionally plot external pressure
    if show_pext and 'press_HII_in' in data:
        y = data['press_HII_in']
        if smooth_window:
            y = smooth_1d(y, smooth_window)
        if not np.all(~np.isfinite(y)):
            ax.plot(t, y, color='gray', ls=':', lw=1.5,
                    label=r'$P_{\rm ext}$', alpha=0.7, zorder=2)

    ax.set_yscale('log')

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        # For log scale, start from first positive time
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_pressures = np.concatenate([
        data['Pb'], data['P_IF'], data['P_drive']
    ])
    valid = all_pressures[np.isfinite(all_pressures) & (all_pressures > 0)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        ax.set_ylim(ymin * 0.3, ymax * 3)


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot pressure evolution from a direct data path/folder."""
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
                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$P/k_B$ [K cm$^{-3}$]")
    ax.set_title(f"Pressure Evolution: {data_path.parent.name}")

    # Legend
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_pressureEvolution_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid(folder_path, output_dir=None, ndens_filter=None):
    """
    Create grid plot from all simulations found in a folder.

    Searches subfolders for dictionary.jsonl files, parses simulation
    parameters from folder names (e.g., "1e7_sfe020_n1e4"), and arranges
    them in a grid sorted by:
    - Rows: increasing mCloud (top to bottom)
    - Columns: increasing SFE (left to right)

    Saves PDF as {folder_name}_{ndens}.pdf without displaying.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        If provided, only plot simulations with this density (e.g., "1e4").
        If None, creates one PDF per unique density found.

    Notes
    -----
    Folder names must follow the pattern: {mCloud}_sfe{sfe}_n{ndens}
    Examples: "1e7_sfe020_n1e4", "5e6_sfe010_n1e3"
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
            sharex=False, sharey=False,
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
                                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT,
                                   use_log_x=USE_LOG_X)
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
                    ax.set_ylabel(mlabel + "\n" + r"$P/k_B$ [K cm$^{-3}$]")
                else:
                    ax.tick_params(labelleft=False)

        handles = [
            Line2D([0], [0], color="blue", ls="-", lw=1.8, label=r"$P_b$ (bubble)"),
            Line2D([0], [0], color="red", ls="-", lw=1.8, label=r"$P_{\rm IF}$"),
            Line2D([0], [0], color="black", ls="--", lw=2.2, label=r"$P_{\rm drive}$"),
        ]
        if SHOW_PEXT:
            handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.5,
                                  alpha=0.7, label=r"$P_{\rm ext}$"))
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

        fig_dir = Path(output_dir) if output_dir else FIG_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"{folder_name}_{ndens_tag}_pressure.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY pressure evolution (P_b, P_IF, P_drive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_pressureEvolution.py 1e7_sfe020_n1e4
  python paper_pressureEvolution.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_pressureEvolution.py /path/to/dictionary.jsonl

  # Grid plot from folder (auto-discovers simulations)
  python paper_pressureEvolution.py --folder /path/to/my_experiment/
  python paper_pressureEvolution.py -F /path/to/simulations/
  python paper_pressureEvolution.py -F /path/to/simulations/ -n 1e4  # filter by density
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
        '--log-x', action='store_true',
        help='Use log scale for x-axis (time)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Create grid plot from all simulations in folder. '
             'Auto-organizes by mCloud (rows) and SFE (columns).'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3"). '
             'If not specified, generates one PDF per density found.'
    )

    args = parser.parse_args()

    if args.log_x:
        USE_LOG_X = True

    if args.folder:
        # Grid mode: create grid from all simulations in folder
        plot_grid(args.folder, args.output_dir, ndens_filter=args.nCore)
    elif args.data:
        # Single simulation mode
        plot_from_path(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
