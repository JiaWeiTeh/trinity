#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 20:03:19 2025

@author: Jia Wei Teh
"""

import numpy as np
import matplotlib.transforms as mtransforms
from pathlib import Path
from matplotlib.lines import Line2D

from src._plots.plot_base import smooth_1d
from src._output.trinity_reader import load_output
from src._plots.grid_template import plot_single, plot_grid as _plot_grid

print("...plotting radius comparison")

# --- configuration
SMOOTH_WINDOW = 21
PHASE_CHANGE = True

SAVE_PDF = True

# ---------- load beta/delta + R2 ----------
def load_cooling_run(data_path: Path):
    """Load cooling run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    R2 = output.get('R2')
    additional_param = output.get('Pb')
    phase = np.array(output.get('current_phase', as_array=False))

    beta = output.get('cool_beta')
    delta = output.get('cool_delta')

    rcloud = float(output[0].get('rCloud', np.nan))

    # ensure increasing time (important for "first crossing" + plotting)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, beta, delta, additional_param = t[order], R2[order], phase[order], beta[order], delta[order], additional_param[order]

    return t, R2, phase, beta, delta, rcloud, additional_param


# ---------- plot on an axis (left: beta/delta, right: R2) ----------
def plot_cooling_on_ax(
    ax, t, R2, phase, beta, delta, rcloud, additional_param,
    smooth_window=None, smooth_mode="edge",
    show_phase_line=True,
    show_cloud_line=True,
    label_pad_points=4
):
    fig = ax.figure

    # optional smoothing (beta/delta often jittery)
    beta_s  = smooth_1d(beta,  smooth_window, mode=smooth_mode)
    delta_s = smooth_1d(delta, smooth_window, mode=smooth_mode)
    R2_s    = smooth_1d(R2,    smooth_window, mode=smooth_mode)

    # --- phase-change line:
    if show_phase_line:
        change_idx = np.flatnonzero(phase[1:] != phase[:-1]) + 1 
        for x in t[change_idx]: 
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)

    # --- cloud breakout line: first time R2 > rcloud (vertical dashed)
    if show_cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(R2_s > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)

            # padded label next to line
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8, color="k", alpha=0.8,
                rotation=90
            )

    # --- left axis: beta + delta
    ax.plot(t, beta_s,  lw=1.6, label=r"$\beta$")
    ax.plot(t, delta_s, lw=1.6, label=r"$\delta$")

    ax.set_xlim(t.min(), t.max())

    # --- right axis: R2
    axr = ax.twinx()
    axr.plot(t, additional_param, lw=1.4, alpha=0.8, c = 'k', label=r"$R_2$")
    axr.set_yscale('log')
    # axr.set_ylabel(r"$R_2$ [pc]")

    # keep the twin axis from hiding things
    ax.patch.set_visible(False)
    ax.set_zorder(2)

    return axr  # in case you want per-panel tweaks


# ---------- cell adapters for grid_template ----------

def _plot_cell_single(ax, data):
    """Adapter for single-plot mode."""
    t, R2, phase, beta, delta, rcloud, additional_param = data
    return plot_cooling_on_ax(
        ax, t, R2, phase, beta, delta, rcloud, additional_param,
        smooth_window=SMOOTH_WINDOW,
        show_phase_line=PHASE_CHANGE,
        show_cloud_line=True,
    )


def _plot_cell_grid(ax, data):
    """Adapter for grid-plot mode."""
    t, R2, phase, beta, delta, rcloud, additional_param = data
    return plot_cooling_on_ax(
        ax, t, R2, phase, beta, delta, rcloud, additional_param,
        smooth_window=7,
        show_phase_line=True,
        show_cloud_line=True,
    )


def _single_legend_handles():
    return [
        Line2D([0],[0], lw=1.6, label=r"$\beta$"),
        Line2D([0],[0], lw=1.6, label=r"$\delta$"),
        Line2D([0],[0], lw=1.4, alpha=0.8, c='k', label=r"$P_b$"),
        Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"phase change"),
    ]


def _grid_legend_handles():
    return [
        Line2D([0],[0], lw=1.6, label=r"$\beta$"),
        Line2D([0],[0], lw=1.6, label=r"$\delta$"),
        Line2D([0],[0], lw=1.4, alpha=0.8, label=r"$R_2$"),
        Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"transition$\to$momentum"),
    ]


def _post_plot_single(ax, data, extra):
    """Set right-axis label after plot_single draws."""
    axr = extra
    axr.set_ylabel(r"$P_b$")


# ---------- plot_from_path for CLI ----------
def plot_from_path(data_input: str, output_dir: str = None):
    """Plot cooling parameters from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_cooling_run,
        plot_cell_fn=_plot_cell_single,
        legend_handles_fn=_single_legend_handles,
        file_prefix="paper_betadelta",
        ylabel=r"Cooling: $\beta,\delta$",
        title_fn=lambda p: f"Cooling Parameters: {p.parent.name}",
        post_plot_fn=_post_plot_single,
    )


# ---------- GRID ----------
def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """Plot grid of cooling parameters from simulations in a folder."""
    _plot_grid(
        folder_path, output_dir,
        ndens_filter=ndens_filter,
        mCloud_filter=mCloud_filter,
        sfe_filter=sfe_filter,
        load_run_fn=load_cooling_run,
        plot_cell_fn=_plot_cell_grid,
        legend_handles_fn=_grid_legend_handles,
        file_prefix="betadelta",
        ylabel=r"Cooling: $\beta,\delta$",
        dpi=200,
        sharey=False,
        legend_ncol=3,
        legend_y=0.91,
        subplots_adjust_top=0.82,
        save_pdf=SAVE_PDF,
    )


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_betadelta.py",
        description="Plot TRINITY cooling parameters (beta, delta)",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
