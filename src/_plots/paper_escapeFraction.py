#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:13:21 2025

@author: Jia Wei Teh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_collapse_marker, get_marker_legend_handles

print("...plotting escape fraction comparison")


# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # one subplot per mCloud
ndens_list  = ["1e4", "1e2", "1e3"]                        # one figure per ndens
# ndens_list  = ["1e4"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # multiple lines per subplot

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 7

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"


def smooth_1d(y, window, mode="edge"):
    """Simple moving-average smoothing. window is in number of snapshots."""
    if window is None or window <= 1:
        return y

    window = int(window)
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def load_escape_fraction(data_path: Path):
    """Return (t, fesc, isCollapse) arrays using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')

    # fesc = 1 - fAbs (fAbs stored as shell_fAbsorbedIon)
    fAbs = output.get('shell_fAbsorbedIon')
    fAbs = np.nan_to_num(fAbs, nan=0.0)
    fesc = 1.0 - fAbs

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    return t, fesc, isCollapse


import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot escape fraction from a direct data path/folder.

    Parameters
    ----------
    data_input : str
        Can be: folder name, folder path, or file path
    output_dir : str, optional
        Base directory for output folders
    """
    from matplotlib.lines import Line2D

    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading data from: {data_path}")

    try:
        t, fesc, isCollapse = load_escape_fraction(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
    fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

    ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=r"$f_{\rm esc}$")

    # --- collapse line using helper module
    add_collapse_marker(ax, t, isCollapse)

    ax.set_title(f"Escape Fraction: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$f_{\rm esc}$")
    ax.set_ylim(0, 1)
    ax.set_xscale('log')
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()

    # Save figures
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_escapeFraction_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of escape fractions."""
    for ndens in ndens_list:
        nrows = len(mCloud_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=1,
            figsize=(7.0, 2.6 * nrows),
            sharex=False,     # each subplot gets its own t_max
            sharey=True,
            dpi=200,
            constrained_layout=True
        )

        if nrows == 1:
            axes = [axes]  # make iterable

        all_line_handles = []
        all_line_labels = []

        for i, mCloud in enumerate(mCloud_list):
            ax = axes[i]

            # plot each sfe as a line on the same axis
            for sfe in sfe_list:
                run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                data_path = find_data_file(BASE_DIR, run_name)

                if data_path is None:
                    print(f"  {run_name}: missing")
                    continue

                print(f"  Loading: {data_path}")
                try:
                    t, fesc, isCollapse = load_escape_fraction(data_path)

                    # optional smoothing
                    fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
                    fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

                    eps = int(sfe) / 100.0
                    (line,) = ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=rf"$\epsilon={eps:.2f}$")

                    # store legend handles once (from first subplot only)
                    if i == 0:
                        all_line_handles.append(line)
                        all_line_labels.append(rf"$\epsilon={eps:.2f}$")

                    # --- collapse line using helper module (no label since many runs per subplot)
                    add_collapse_marker(ax, t, isCollapse, show_label=False)

                except Exception as e:
                    print(f"Error in {run_name}: {e}")

            # Handle non-power-of-10 masses (e.g., 5e6)
            mval = float(mCloud)
            mexp = int(np.floor(np.log10(mval)))
            mcoeff = mval / (10 ** mexp)
            mcoeff = round(mcoeff)
            if mcoeff == 10:
                mcoeff = 1
                mexp += 1
            if mcoeff == 1:
                mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}\,M_\odot$"
            else:
                mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"
            ax.set_ylabel(rf"$f_\mathrm{{esc}}$" + "\n" + mlabel)
            ax.set_ylim(0, 1)
            ax.set_xscale('log')

            # x label only on bottom subplot
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Escape fraction vs time  ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

        # global legend (cleaner than repeating per axis)
        if all_line_handles:
            # Add collapse indicator to legend using helper
            collapse_handles = get_marker_legend_handles(include_phase=False, include_rcloud=False, include_collapse=True)
            for h in collapse_handles:
                all_line_handles.append(h)
                all_line_labels.append(h.get_label())
            leg = fig.legend(
                handles=all_line_handles,
                labels=all_line_labels,
                loc="upper center",
                ncol=len(all_line_handles),
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="0.2",
                bbox_to_anchor=(0.5, 1.07),
            )
            leg.set_zorder(10)

        if SAVE_PDF:
            fig.savefig(FIG_DIR / f"paper_escapeFraction_n{ndens}.pdf", bbox_inches='tight')
            print(f"Saved: {FIG_DIR / f'paper_escapeFraction_n{ndens}.pdf'}")
        plt.show()
        plt.close(fig)


def plot_folder_grid(folder_path, output_dir=None):
    """
    Create grid plot from all simulations found in a folder.

    Groups SFE values as lines on each row, with mCloud defining rows.
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    organized = organize_simulations_for_grid(sim_files)
    mCloud_list_found = organized['mCloud_list']
    sfe_list_found = organized['sfe_list']
    grid = organized['grid']

    if not mCloud_list_found or not sfe_list_found:
        print(f"Could not organize simulations into grid")
        return

    print(f"Found {len(sim_files)} simulations")
    print(f"  mCloud: {mCloud_list_found}")
    print(f"  SFE: {sfe_list_found}")

    nrows = len(mCloud_list_found)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1,
        figsize=(7.0, 2.6 * nrows),
        sharex=False,
        sharey=True,
        dpi=200,
        squeeze=False,
        constrained_layout=True
    )

    all_line_handles = []
    all_line_labels = []

    for i, mCloud in enumerate(mCloud_list_found):
        ax = axes[i, 0]

        for sfe in sfe_list_found:
            data_path = grid.get((mCloud, sfe))

            if data_path is None:
                continue

            print(f"  Loading: {data_path}")
            try:
                t, fesc, isCollapse = load_escape_fraction(data_path)
                fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
                fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

                eps = int(sfe) / 100.0
                (line,) = ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=rf"$\epsilon={eps:.2f}$")

                if i == 0:
                    all_line_handles.append(line)
                    all_line_labels.append(rf"$\epsilon={eps:.2f}$")

                add_collapse_marker(ax, t, isCollapse, show_label=False)
            except Exception as e:
                print(f"Error loading {data_path}: {e}")

        mval = float(mCloud)
        mexp = int(np.floor(np.log10(mval)))
        mcoeff = round(mval / (10 ** mexp))
        if mcoeff == 10:
            mcoeff = 1
            mexp += 1
        if mcoeff == 1:
            mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}\,M_\odot$"
        else:
            mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"
        ax.set_ylabel(rf"$f_\mathrm{{esc}}$" + "\n" + mlabel)
        ax.set_ylim(0, 1)
        ax.set_xscale('log')

        if i == nrows - 1:
            ax.set_xlabel("t [Myr]")

    fig.suptitle(folder_name, fontsize=14, y=1.02)

    if all_line_handles:
        collapse_handles = get_marker_legend_handles(include_phase=False, include_rcloud=False, include_collapse=True)
        for h in collapse_handles:
            all_line_handles.append(h)
            all_line_labels.append(h.get_label())
        leg = fig.legend(
            handles=all_line_handles,
            labels=all_line_labels,
            loc="upper center",
            ncol=len(all_line_handles),
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.07),
        )
        leg.set_zorder(10)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"{folder_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY escape fraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_escapeFraction.py 1e7_sfe020_n1e4
  python paper_escapeFraction.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_escapeFraction.py /path/to/dictionary.jsonl
  python paper_escapeFraction.py  # (uses grid config at top of file)
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Base directory for output folders (default: TRINITY_OUTPUT_DIR or "outputs")'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Search folder recursively for all simulation .jsonl files'
    )

    args = parser.parse_args()

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir)
    elif args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    else:
        # Config mode: plot grid
        plot_grid()
