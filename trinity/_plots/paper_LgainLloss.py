#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot bubble_Lgain (mechanical injection) vs bubble_Lloss (cooling + leak)
as a time series.

Both quantities are stored in code units [Msun*pc**2/Myr**3] and converted
to erg/s here via INV_CONV.L_au2cgs.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from trinity._plots.plot_base import FIG_DIR, smooth_1d
from trinity._output.trinity_reader import load_output, resolve_data_input
from trinity._functions.unit_conversions import INV_CONV
from trinity._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from trinity._plots.grid_template import (
    build_param_tag,
    iter_grid_densities,
    mark_missing_cell,
    attach_grid_legend,
    save_grid_figure,
    set_mcloud_ylabel,
)

print("...plotting bubble_Lgain vs bubble_Lloss [erg/s] vs t")

# ---------------- configuration ----------------
SHOW_PHASE    = False
SHOW_RCLOUD    = False
SHOW_COLLAPSE = False
SMOOTH_WINDOW = None
SMOOTH_MODE   = "edge"

LOG_Y = True

# colors
C_GAIN = "#1f77b4"   # blue
C_LOSS = "#d62728"   # red


def load_run(data_path: Path):
    """Load t, phase, Lgain, Lloss [erg/s], R2, rcloud, isCollapse."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    phase = np.array(output.get('current_phase', as_array=False))

    # bubble_Lgain / bubble_Lloss are stored in [Msun*pc^2/Myr^3].
    # Convert to erg/s.
    Lgain = output.get('bubble_Lgain') * INV_CONV.L_au2cgs
    Lloss = output.get('bubble_Lloss') * INV_CONV.L_au2cgs

    R2 = output.get('R2')
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        Lgain, Lloss, R2 = Lgain[order], Lloss[order], R2[order]
        isCollapse = isCollapse[order]

    return t, phase, Lgain, Lloss, R2, rcloud, isCollapse


def add_phase_and_cloud_markers(ax, t, phase, R2, rcloud, isCollapse=None,
                                 label_pad_points=4, show_collapse=SHOW_COLLAPSE):
    add_plot_markers(
        ax, t,
        phase=phase if SHOW_PHASE else None,
        R2=R2 if SHOW_RCLOUD else None,
        rcloud=rcloud if SHOW_RCLOUD else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=SHOW_PHASE,
        show_rcloud=SHOW_RCLOUD,
        show_collapse=show_collapse,
        label_pad_points=label_pad_points,
    )


def plot_panel(ax, t, phase, Lgain, Lloss, R2, rcloud, isCollapse=None):
    Lgain_s = smooth_1d(Lgain, SMOOTH_WINDOW, mode=SMOOTH_MODE)
    Lloss_s = smooth_1d(Lloss, SMOOTH_WINDOW, mode=SMOOTH_MODE)
    R2_s    = smooth_1d(R2,    SMOOTH_WINDOW, mode=SMOOTH_MODE)

    add_phase_and_cloud_markers(ax, t, phase, R2_s, rcloud, isCollapse)

    ax.plot(t, Lgain_s, lw=1.8, color=C_GAIN,
            label=r"$L_{\rm gain}$", zorder=3)
    ax.plot(t, Lloss_s, lw=1.8, color=C_LOSS,
            label=r"$L_{\rm loss}$", zorder=3)

    ax.set_xlim(t.min(), t.max())

    if LOG_Y:
        y_all = np.concatenate([Lgain_s[np.isfinite(Lgain_s)],
                                Lloss_s[np.isfinite(Lloss_s)]])
        # only switch to log if there is a strictly positive sample
        y_pos = y_all[y_all > 0]
        if y_pos.size:
            ax.set_yscale("log")


# ---------------- single-run mode ----------------

def plot_from_path(data_input: str, output_dir: str = None):
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        t, phase, Lgain, Lloss, R2, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_panel(ax, t, phase, Lgain, Lloss, R2, rcloud, isCollapse)

    ax.set_title(f"Lgain vs Lloss: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$L\ [{\rm erg\ s^{-1}}]$")
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()

    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_LgainLloss_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ---------------- grid mode ----------------

def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    for ndens, mCloud_list, sfe_list, grid, folder_name in iter_grid_densities(
            folder_path, ndens_filter=ndens_filter,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter):

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=False,
            dpi=500,
            squeeze=False,
            constrained_layout=False,
        )

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    mark_missing_cell(ax, "missing")
                    continue

                try:
                    t, phase, Lgain, Lloss, R2, rcloud, isCollapse = load_run(data_path)
                    plot_panel(ax, t, phase, Lgain, Lloss, R2, rcloud, isCollapse)
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                if j == 0:
                    set_mcloud_ylabel(ax, mCloud, extra=r"$L\ [{\rm erg\ s^{-1}}]$")
                else:
                    ax.tick_params(labelleft=False)

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        handles = [
            Line2D([0], [0], color=C_GAIN, lw=1.8, label=r"$L_{\rm gain}$"),
            Line2D([0], [0], color=C_LOSS, lw=1.8, label=r"$L_{\rm loss}$"),
        ]
        handles.extend(get_marker_legend_handles(
            include_phase=SHOW_PHASE,
            include_rcloud=SHOW_RCLOUD,
            include_collapse=SHOW_COLLAPSE,
        ))

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, handles,
            n_rows_for_layout=nrows,
            folder_name=folder_name,
            param_tag=param_tag,
            legend_ncol=2,
            legend_bbox_transform_fig=True,
        )

        save_grid_figure(
            fig, folder_name=folder_name, file_prefix="LgainLloss",
            param_tag=param_tag, output_dir=output_dir,
        )
        plt.close(fig)


plot_folder_grid = plot_grid


# ---------------- CLI ----------------
if __name__ == "__main__":
    from trinity._plots.cli import dispatch, marker_pre_dispatch
    dispatch(
        script_name="paper_LgainLloss.py",
        description="Plot TRINITY bubble Lgain vs Lloss (erg/s) time series",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
