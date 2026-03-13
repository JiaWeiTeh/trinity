#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:13:21 2025

@author: Jia Wei Teh
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_collapse_marker, get_marker_legend_handles

print("...plotting escape fraction comparison")


# --- configuration
# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 7

SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"


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


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of escape fraction from simulations in a folder.

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
        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
        mCloud_list_found = organized['mCloud_list']
        sfe_list_found = organized['sfe_list']
        grid = organized['grid']

        if not mCloud_list_found or not sfe_list_found:
            print(f"Could not organize simulations into grid")
            continue

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

        fig.suptitle(f"{folder_name} (n{ndens})", fontsize=14, y=1.02)

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

        # Save figure to ./fig/{folder_name}/escapeFraction_{ndens_tag}.pdf
        ndens_tag = f"n{ndens}"
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"escapeFraction_{ndens_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_escapeFraction.py",
        description="Plot TRINITY escape fraction",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
