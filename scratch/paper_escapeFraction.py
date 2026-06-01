#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:13:21 2025

@author: Jia Wei Teh
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from trinity._plots.plot_base import FIG_DIR, smooth_1d
from trinity._output.trinity_reader import load_output, resolve_data_input
from trinity._plots.plot_markers import add_collapse_marker, get_marker_legend_handles
from trinity._plots.grid_template import (
    _mcloud_label,
    build_param_tag,
    iter_grid_densities,
    attach_grid_legend,
    save_grid_figure,
    phii_file_prefix,
)
from trinity._plots.cli import marker_pre_dispatch

print("...plotting escape fraction comparison")


# --- configuration
# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 7

# =============================================================================
# MARKER DEFAULTS (off for clean paper figures; enable via CLI --show-*)
# =============================================================================
SHOW_PHASE = False
SHOW_RCLOUD = False
SHOW_COLLAPSE = False

SAVE_PNG = False
SAVE_PDF = True

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

    # Suppress seed-bubble transient: discard snapshots before the shell
    # first becomes optically thick (f_esc first reaches zero).
    t_transient = np.array([])
    idx_zero = np.nonzero(fesc <= 0.0)[0]
    if len(idx_zero) > 0:
        i0 = idx_zero[0]
        t_transient = t[:i0 + 1]
        t, fesc, isCollapse = t[i0:], fesc[i0:], isCollapse[i0:]

    return t, fesc, isCollapse, t_transient



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
        t, fesc, isCollapse, t_transient = load_escape_fraction(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
    fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

    color = ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=r"$f_{\rm esc}$")[0].get_color()

    # Dashed line at f_esc=0 for the seed-bubble transient
    if len(t_transient) > 0:
        ax.plot(t_transient, np.zeros_like(t_transient), ls='--', lw=1.2, alpha=0.5, color=color)

    # --- collapse line using helper module
    if SHOW_COLLAPSE:
        add_collapse_marker(ax, t, isCollapse)

    ax.set_title(f"Escape Fraction: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$f_{\rm esc}$")
    ax.set_ylim(0, 1)
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
              mCloud_filter=None, sfe_filter=None, phii_mode="yes"):
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
    phii_mode : {"yes", "no"}
        PHII suffix variant to plot.  See ``grid_template.filter_sim_files_by_phii``.
    """
    for ndens, mCloud_list, sfe_list, grid, folder_name in iter_grid_densities(
            folder_path, ndens_filter=ndens_filter,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
            phii_mode=phii_mode):

        nrows = len(mCloud_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=1,
            figsize=(7.0, 2.6 * nrows),
            sharex=False,
            sharey=True,
            dpi=200,
            squeeze=False,
            constrained_layout=False,
        )

        all_line_handles = []
        all_line_labels = []

        for i, mCloud in enumerate(mCloud_list):
            ax = axes[i, 0]

            for sfe in sfe_list:
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    continue

                try:
                    t, fesc, isCollapse, t_transient = load_escape_fraction(data_path)
                    fesc_plot = smooth_1d(fesc, SMOOTH_WINDOW)
                    fesc_plot = np.clip(fesc_plot, 0.0, 1.0)

                    eps = int(sfe) / 100.0
                    (line,) = ax.plot(t, fesc_plot, lw=1.8, alpha=0.9, label=rf"$\epsilon={eps:.2f}$")

                    # Dashed line at f_esc=0 for the seed-bubble transient
                    if len(t_transient) > 0:
                        ax.plot(t_transient, np.zeros_like(t_transient), ls='--', lw=1.2, alpha=0.5, color=line.get_color())

                    if i == 0:
                        all_line_handles.append(line)
                        all_line_labels.append(rf"$\epsilon={eps:.2f}$")

                    if SHOW_COLLAPSE:
                        add_collapse_marker(ax, t, isCollapse, show_label=False)
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")

            ax.set_ylabel(rf"$f_\mathrm{{esc}}$" + "\n" + _mcloud_label(mCloud))
            ax.set_ylim(0, 1)

            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

        if all_line_handles:
            collapse_handles = get_marker_legend_handles(include_phase=False, include_rcloud=False, include_collapse=SHOW_COLLAPSE)
            for h in collapse_handles:
                all_line_handles.append(h)
                all_line_labels.append(h.get_label())

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, all_line_handles,
            n_rows_for_layout=nrows,
            folder_name=folder_name,
            param_tag=param_tag,
            legend_ncol=max(len(all_line_handles), 1),
        )

        save_grid_figure(
            fig, folder_name=folder_name,
            file_prefix=phii_file_prefix("escapeFraction", phii_mode),
            param_tag=param_tag, output_dir=output_dir,
        )
        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from trinity._plots.cli import dispatch
    dispatch(
        script_name="paper_escapeFraction.py",
        description="Plot TRINITY escape fraction",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
