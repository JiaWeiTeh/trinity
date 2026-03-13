# -*- coding: utf-8 -*-
"""
Shared grid-plot and single-plot infrastructure for ``src._plots``.

Most ``paper_*.py`` scripts follow the same two-mode pattern:

1. **Single mode** — load one run, plot on a single axis, save.
2. **Grid mode** — discover simulations in a folder, organise into an
   (mCloud × SFE) grid, plot each cell, add labels/legend, save one
   PDF per density.

This module provides ``plot_single`` and ``plot_grid`` templates that
encapsulate all the shared infrastructure.  Each script supplies only
the parts that differ:

- ``load_run_fn(data_path) → data``  — what to extract
- ``plot_cell_fn(ax, data, **cfg) → extra_artists``  — what to draw
- ``legend_handles_fn() → list``  — legend entries

Example usage in a paper script::

    from src._plots.grid_template import plot_single, plot_grid as _plot_grid

    def plot_from_path(data_input, output_dir=None):
        plot_single(
            data_input, output_dir,
            load_run_fn=load_run,
            plot_cell_fn=plot_run_on_ax,
            legend_handles_fn=build_legend_handles,
            file_prefix="paper_feedback",
            ylabel=r"Force fraction",
        )

    def plot_grid(folder_path, output_dir=None, **kw):
        _plot_grid(
            folder_path, output_dir, **kw,
            load_run_fn=load_run,
            plot_cell_fn=plot_run_on_ax,
            legend_handles_fn=build_legend_handles,
            file_prefix="feedback",
            ylabel=r"Force fraction",
        )
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR
from src._output.trinity_reader import (
    resolve_data_input,
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mcloud_label(mCloud: str) -> str:
    """LaTeX label for a cloud mass value, e.g. '1e7' → r'$M_{\\rm cloud}=10^{7}M_\\odot$'."""
    mval = float(mCloud)
    mexp = int(np.floor(np.log10(mval)))
    mcoeff = round(mval / (10 ** mexp))
    if mcoeff == 10:
        mcoeff = 1
        mexp += 1
    if mcoeff == 1:
        return rf"$M_{{\rm cloud}}=10^{{{mexp}}}\,M_\odot$"
    return rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"


def _mcloud_label_short(mCloud: str) -> str:
    """Short LaTeX label, e.g. '1e7' → r'$M_{\\rm cl}=10^{7}$'."""
    mval = float(mCloud)
    mexp = int(np.floor(np.log10(mval)))
    mcoeff = round(mval / (10 ** mexp))
    if mcoeff == 10:
        mcoeff = 1
        mexp += 1
    if mcoeff == 1:
        return rf"$M_{{\rm cl}}=10^{{{mexp}}}$"
    return rf"$M_{{\rm cl}}={mcoeff}\times10^{{{mexp}}}$"


def _sfe_title(sfe: str) -> str:
    """Column title for an SFE tag, e.g. '010' → r'$\\epsilon=0.10$'."""
    eps = int(sfe) / 100.0
    return rf"$\epsilon={eps:.2f}$"


# ------------------------------------------------------------------
# Single-simulation plot
# ------------------------------------------------------------------

def plot_single(
    data_input: str,
    output_dir: Optional[str] = None,
    *,
    load_run_fn: Callable,
    plot_cell_fn: Callable,
    legend_handles_fn: Optional[Callable[[], List]] = None,
    file_prefix: str,
    ylabel: str = "",
    xlabel: str = "t [Myr]",
    figsize: tuple = (8, 6),
    dpi: int = 150,
    title_fn: Optional[Callable[[Path], str]] = None,
    post_plot_fn: Optional[Callable] = None,
    legend_loc: str = "upper left",
    legend_ncol: int = 1,
) -> None:
    """Plot a single simulation run.

    Parameters
    ----------
    data_input : path-like
        Folder name, folder path, or file path.
    output_dir : str, optional
        Base directory for output folders.
    load_run_fn : callable(data_path) → data
        Returns whatever the per-cell plotter needs.
    plot_cell_fn : callable(ax, data, **cfg)
        Draws on the axis.  May return extra artists (e.g. twin axes).
    legend_handles_fn : callable() → list, optional
        Returns matplotlib legend handles.  If ``None``, uses automatic
        legend from plot labels.
    file_prefix : str
        Filename prefix, e.g. ``"paper_feedback"``.
    ylabel, xlabel : str
        Axis labels.
    figsize, dpi : plot dimensions.
    title_fn : callable(data_path) → str, optional
        Custom title builder.  Default: uses the parent folder name.
    post_plot_fn : callable(ax, data, extra_artists), optional
        Hook called after plot_cell_fn for extra customisation.
    legend_loc : str
        Legend location for single plot (default ``"upper left"``).
    legend_ncol : int
        Number of legend columns (default 1).
    """
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    try:
        data = load_run_fn(data_path)
        extra = plot_cell_fn(ax, data)
    except Exception as e:
        print(f"Error loading data: {e}")
        plt.close(fig)
        return

    if title_fn is not None:
        ax.set_title(title_fn(data_path))
    else:
        ax.set_title(f"{file_prefix}: {data_path.parent.name}")

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if post_plot_fn is not None:
        post_plot_fn(ax, data, extra)

    if legend_handles_fn is not None:
        handles = legend_handles_fn()
        ax.legend(handles=handles, loc=legend_loc, framealpha=0.9,
                  ncol=legend_ncol)
    else:
        ax.legend(loc=legend_loc, framealpha=0.9, ncol=legend_ncol)

    plt.tight_layout()

    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"{file_prefix}_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Grid plot
# ------------------------------------------------------------------

def plot_grid(
    folder_path,
    output_dir=None,
    *,
    ndens_filter=None,
    mCloud_filter=None,
    sfe_filter=None,
    load_run_fn: Callable,
    plot_cell_fn: Callable,
    legend_handles_fn: Callable[[], List],
    file_prefix: str,
    ylabel: str = "",
    xlabel: str = "t [Myr]",
    cell_width: float = 3.2,
    cell_height: float = 2.6,
    dpi: int = 500,
    sharex: bool = False,
    sharey: bool = True,
    constrained_layout: bool = False,
    legend_ncol: int = 4,
    legend_y: float = 1.0,
    suptitle: bool = True,
    subplots_adjust_top: Optional[float] = None,
    save_pdf: bool = True,
    pre_loop_fn: Optional[Callable] = None,
    post_loop_fn: Optional[Callable] = None,
    mcloud_label_fn: Optional[Callable[[str], str]] = None,
    suptitle_y: Optional[float] = None,
    hide_non_left_labels: bool = False,
) -> None:
    """Create an (mCloud × SFE) grid plot for every density found.

    Parameters
    ----------
    folder_path : path-like
        Directory containing simulation sub-folders.
    output_dir : str, optional
        Override output directory.
    ndens_filter, mCloud_filter, sfe_filter :
        Passed to ``organize_simulations_for_grid``.
    load_run_fn : callable(data_path) → data
        Returns whatever the per-cell plotter needs.
    plot_cell_fn : callable(ax, data, **cfg)
        Draws on the axis.
    legend_handles_fn : callable() → list
        Returns matplotlib legend handles.
    file_prefix : str
        Filename stem, e.g. ``"feedback"``.
    ylabel : str
        Y-axis label prepended to the mCloud row label on col 0.
    xlabel : str
        X-axis label on the bottom row.
    cell_width, cell_height : float
        Per-cell figure size multipliers.
    dpi : int
    sharex, sharey : bool
    constrained_layout : bool
    legend_ncol : int
    legend_y : float
        Vertical position of the legend anchor (in figure fraction).
    suptitle : bool
        Whether to add a top-level title with folder name and density.
    subplots_adjust_top : float, optional
        If set, call ``fig.subplots_adjust(top=...)``.
    save_pdf : bool
    pre_loop_fn : callable(fig, axes), optional
        Hook called after subplot creation but before the cell loop.
    post_loop_fn : callable(fig, axes), optional
        Hook called after the cell loop but before legend/save.
    mcloud_label_fn : callable(mCloud) → str, optional
        Custom row-label builder.  Defaults to ``_mcloud_label``.
        Use ``_mcloud_label_short`` for the compact variant.
    suptitle_y : float, optional
        Vertical position of suptitle.  If ``None``, uses matplotlib default.
    hide_non_left_labels : bool
        If ``True``, hide y-axis tick labels on non-leftmost columns.
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print(f"Found {len(sim_files)} simulations")
    print(f"  Densities to plot: {ndens_to_plot}")

    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")
        organized = organize_simulations_for_grid(
            sim_files,
            ndens_filter=ndens,
            mCloud_filter=mCloud_filter,
            sfe_filter=sfe_filter,
        )
        mCloud_list = organized["mCloud_list"]
        sfe_list = organized["sfe_list"]
        grid = organized["grid"]

        if not mCloud_list or not sfe_list:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(cell_width * ncols, cell_height * nrows),
            sharex=sharex,
            sharey=sharey,
            dpi=dpi,
            squeeze=False,
            constrained_layout=constrained_layout,
        )

        if subplots_adjust_top is not None:
            fig.subplots_adjust(top=subplots_adjust_top)

        if pre_loop_fn is not None:
            pre_loop_fn(fig, axes)

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    run_id = f"{mCloud}_sfe{sfe}_n{ndens}"
                    print(f"  {run_id}: missing")
                    ax.text(
                        0.5, 0.5, "missing",
                        ha="center", va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

                try:
                    data = load_run_fn(data_path)
                    plot_cell_fn(ax, data)
                except Exception as e:
                    print(f"  Error loading {data_path}: {e}")
                    ax.text(
                        0.5, 0.5, "error",
                        ha="center", va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

                # Column title (top row only)
                if i == 0:
                    ax.set_title(_sfe_title(sfe))

                # Row label (left column only)
                _label_fn = mcloud_label_fn or _mcloud_label
                if j == 0:
                    row_label = _label_fn(mCloud)
                    if ylabel:
                        row_label += "\n" + ylabel
                    ax.set_ylabel(row_label)
                elif hide_non_left_labels:
                    ax.tick_params(labelleft=False)

                # X-axis label (bottom row only)
                if i == nrows - 1:
                    ax.set_xlabel(xlabel)

        if post_loop_fn is not None:
            post_loop_fn(fig, axes)

        # Legend
        handles = legend_handles_fn()
        if handles:
            leg = fig.legend(
                handles=handles,
                loc="upper center",
                ncol=legend_ncol,
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                edgecolor="0.2",
                bbox_to_anchor=(0.5, legend_y),
            )
            leg.set_zorder(10)

        # Suptitle
        ndens_tag = f"n{ndens}"
        if suptitle:
            sup_kw = {"fontsize": 14}
            if suptitle_y is not None:
                sup_kw["y"] = suptitle_y
            fig.suptitle(f"{folder_name} ({ndens_tag})", **sup_kw)

        # Save
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)

        if save_pdf:
            out_pdf = fig_dir / f"{file_prefix}_{ndens_tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"  Saved: {out_pdf}")

        plt.close(fig)
