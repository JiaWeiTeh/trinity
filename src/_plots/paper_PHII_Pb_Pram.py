#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure Components Plot for TRINITY

Shows the three key pressure components as a function of time:

- P_HII(t): HII pressure at the ionization front (Strömgren-based)
- Pb(t): Hot bubble pressure
- P_ram(t): Ram pressure from freely-streaming wind

Optionally also shows P_drive (total driving pressure) for reference.

Author: TRINITY Team
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, resolve_data_input, info_simulations
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV, CGS

print("...plotting pressure components (P_HII, Pb, P_ram)")

# Unit conversion: code units (Msun/pc/Myr²) → K/cm³ (= P/k_B = n*T)
P_AU_TO_K_CM3 = INV_CONV.Pb_au2cgs / CGS.k_B

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
SHOW_PDRIVE = True  # Show P_drive for reference
USE_LOG_X = False  # Use log scale for x-axis (time)

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

# Pressure field styling: (output_key, label, color, linestyle, linewidth)
PRESSURE_FIELDS = [
    ("P_HII", r"$P_{\rm HII}$",  "red",    "-",  1.8),
    ("Pb",    r"$P_b$ (bubble)",  "blue",   "-",  1.8),
    ("P_ram", r"$P_{\rm ram}$",   "green",  "-",  1.8),
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
    P_HII = get_field('P_HII', np.nan) * P_AU_TO_K_CM3
    Pb = get_field('Pb', np.nan) * P_AU_TO_K_CM3
    P_ram = get_field('P_ram', np.nan) * P_AU_TO_K_CM3
    P_drive = get_field('P_drive', np.nan) * P_AU_TO_K_CM3

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        P_HII = P_HII[order]
        Pb = Pb[order]
        P_ram = P_ram[order]
        P_drive = P_drive[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'P_HII': P_HII, 'Pb': Pb, 'P_ram': P_ram, 'P_drive': P_drive,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, show_pdrive=True,
                   use_log_x=False):
    """Plot pressure components on given axes."""
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

    # Plot main pressure components
    for field, label, color, ls, lw in PRESSURE_FIELDS:
        y = data[field]
        if smooth_window:
            y = smooth_1d(y, smooth_window)

        # Skip if all NaN
        if np.all(~np.isfinite(y)):
            continue

        ax.plot(t, y, color=color, ls=ls, lw=lw, label=label, zorder=3)

    # Optionally plot P_drive for reference
    if show_pdrive:
        y = data['P_drive']
        if smooth_window:
            y = smooth_1d(y, smooth_window)
        if not np.all(~np.isfinite(y)):
            ax.plot(t, y, color='black', ls='--', lw=2.2,
                    label=r'$P_{\rm drive}$', alpha=0.8, zorder=4)

    ax.set_yscale('log')

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_pressures = np.concatenate([
        data['P_HII'], data['Pb'], data['P_ram']
    ])
    valid = all_pressures[np.isfinite(all_pressures) & (all_pressures > 0)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        ax.set_ylim(ymin * 0.3, ymax * 3)


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot pressure components from a direct data path/folder."""
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        data = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, show_pdrive=SHOW_PDRIVE,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$P/k_B$ [K cm$^{-3}$]")
    ax.set_title(f"Pressure Components: {data_path.parent.name}")

    # Legend
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    parent_folder = data_path.parent.parent.name
    fig_dir = FIG_DIR / parent_folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"PHII_Pb_Pram_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of pressure components from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
    mCloud_filter : list of str, optional
        Filter simulations by cloud mass (e.g., ["1e6", "1e7"]).
    sfe_filter : list of str, optional
        Filter simulations by SFE (e.g., ["001", "010"]).
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
        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
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

                try:
                    data = load_run(data_path)
                    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                                   phase_change=PHASE_CHANGE, show_pdrive=SHOW_PDRIVE,
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

        # Global legend
        handles = [
            Line2D([0], [0], color="red", ls="-", lw=1.8, label=r"$P_{\rm HII}$"),
            Line2D([0], [0], color="blue", ls="-", lw=1.8, label=r"$P_b$ (bubble)"),
            Line2D([0], [0], color="green", ls="-", lw=1.8, label=r"$P_{\rm ram}$"),
        ]
        if SHOW_PDRIVE:
            handles.append(Line2D([0], [0], color="black", ls="--", lw=2.2,
                                  alpha=0.8, label=r"$P_{\rm drive}$"))
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

        # Save figure
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"PHII_Pb_Pram_{ndens_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY pressure components (P_HII, Pb, P_ram)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_PHII_Pb_Pram.py 1e7_sfe020_n1e4
  python paper_PHII_Pb_Pram.py /path/to/outputs/1e7_sfe020_n1e4

  # Grid plot from folder (auto-discovers simulations)
  python paper_PHII_Pb_Pram.py --folder /path/to/my_experiment/
  python paper_PHII_Pb_Pram.py -F /path/to/simulations/ -n 1e4
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
    parser.add_argument(
        '--mCloud', nargs='+', default=None,
        help='Filter simulations by cloud mass (e.g., --mCloud 1e6 1e7).'
    )
    parser.add_argument(
        '--sfe', nargs='+', default=None,
        help='Filter simulations by SFE (e.g., --sfe 001 010).'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Scan folder and print available mCloud, SFE, and nCore values.'
    )

    args = parser.parse_args()

    if args.info:
        if not args.folder:
            parser.print_help()
            print("\nError: --info requires --folder to be specified.")
        else:
            info = info_simulations(args.folder)
            print("=" * 50)
            print(f"Simulation parameters in: {args.folder}")
            print("=" * 50)
            print(f"  Total simulations: {info['count']}")
            print(f"  mCloud values: {info['mCloud']}")
            print(f"  SFE values: {info['sfe']}")
            print(f"  nCore values: {info['ndens']}")
    elif args.folder:
        plot_grid(args.folder, args.output_dir, ndens_filter=args.nCore,
                  mCloud_filter=args.mCloud, sfe_filter=args.sfe)
    elif args.data:
        plot_from_path(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
