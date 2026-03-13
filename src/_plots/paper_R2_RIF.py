#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2 vs R_IF Comparison Plot for TRINITY

Shows R2 (outer shell radius) and R_IF (ionization front radius) as a
function of time on a log-log scale.  Because the two radii are often
nearly identical, the plot highlights their relative difference:

- A bottom panel shows |R2 - R_IF| / R2  (fractional difference).
- When the difference is below a machine-precision threshold (default
  1e-10), the region is shaded to indicate that any discrepancy is
  likely numerical noise rather than physical.

Author: TRINITY Team
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR, smooth_1d
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short
from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting R2 vs R_IF comparison")

# ---------------- configuration ----------------
SMOOTH_WINDOW = None   # None or 1 disables
PHASE_CHANGE = True
USE_LOG_X = True       # log-log by default
USE_LOG_Y = True

# Fractional difference below this is considered machine precision
MACHINE_EPS = 1e-10

SAVE_PDF = True

# Colours
C_R2  = "black"
C_RIF = "#e04050"   # bright red


def load_run(data_path):
    """Load R2 and R_IF from a TRINITY output."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    phase = np.array(output.get('current_phase', as_array=False))

    def get_field(field, default=np.nan):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    R2   = get_field('R2', np.nan)
    R_IF = get_field('R_IF', np.nan)

    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        R2, R_IF = R2[order], R_IF[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'phase': phase,
        'R2': R2, 'R_IF': R_IF,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True,
                   use_log_x=True, use_log_y=True, machine_eps=MACHINE_EPS):
    """
    Plot R2 and R_IF on the top axes, fractional difference on a twin
    bottom panel created internally via inset_axes.

    Returns the residual axes so callers can style it further.
    """
    t     = data['t']
    R2    = data['R2'].copy()
    R_IF  = data['R_IF'].copy()
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']

    # --- smooth (optional) ------------------------------------------------
    if smooth_window:
        R2   = smooth_1d(R2, smooth_window)
        R_IF = smooth_1d(R_IF, smooth_window)

    # --- masks for valid data (positive, finite) --------------------------
    valid = np.isfinite(R2) & np.isfinite(R_IF) & (R2 > 0) & (R_IF > 0)

    # --- phase / cloud markers --------------------------------------------
    add_plot_markers(
        ax, t,
        phase=phase if phase_change else None,
        R2=R2 if show_rcloud else None,
        rcloud=rcloud if show_rcloud else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=phase_change,
        show_rcloud=show_rcloud,
        show_collapse=show_collapse,
    )

    # --- main radius lines ------------------------------------------------
    t_v   = t[valid]
    R2_v  = R2[valid]
    RIF_v = R_IF[valid]

    ax.plot(t_v, R2_v,  color=C_R2,  ls='-',  lw=2.0,
            label=r'$R_2$', zorder=4)
    ax.plot(t_v, RIF_v, color=C_RIF, ls='--', lw=1.6,
            label=r'$R_{\rm IF}$', zorder=3)

    # --- axes scales ------------------------------------------------------
    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    # sensible limits (skip zeros / NaNs)
    if len(t_v) > 0:
        if use_log_x:
            ax.set_xlim(t_v[t_v > 0].min() if np.any(t_v > 0) else t_v.min(),
                        t_v.max())
        else:
            ax.set_xlim(t_v.min(), t_v.max())

    # --- residual sub-panel (fractional difference) -----------------------
    # Create an inset axes at the bottom of the main axes
    ax_res = ax.inset_axes([0.0, 0.0, 1.0, 0.25])  # [x0, y0, width, height] in axes fraction
    ax.set_position(ax.get_position())  # keep original bbox

    frac_diff = np.full_like(R2, np.nan)
    frac_diff[valid] = np.abs(R2[valid] - R_IF[valid]) / R2[valid]
    # Replace exact zeros in frac_diff with a tiny number for log scale
    frac_diff_plot = frac_diff.copy()
    frac_diff_plot[(frac_diff_plot == 0) & valid] = 1e-16

    ax_res.plot(t_v, frac_diff_plot[valid], color='#555555', lw=1.0, zorder=3)

    # shade machine-precision band
    if use_log_x:
        ax_res.set_xscale('log')
    ax_res.set_yscale('log')
    ax_res.set_ylabel(r'$|\Delta R|/R_2$', fontsize=7, labelpad=2)
    ax_res.tick_params(labelsize=6)

    # match x-limits
    ax_res.set_xlim(ax.get_xlim())
    ax_res.axhspan(1e-16, machine_eps, color='#b0d0ff', alpha=0.35, zorder=1,
                   label='machine precision')

    # auto y-limits for residual
    fd_valid = frac_diff_plot[valid & np.isfinite(frac_diff_plot) & (frac_diff_plot > 0)]
    if len(fd_valid) > 0:
        ax_res.set_ylim(max(fd_valid.min() * 0.3, 1e-16), fd_valid.max() * 3)

    # hide x tick labels on main axes (shared with residual)
    ax.tick_params(labelbottom=False)
    ax_res.set_xlabel(ax.get_xlabel() or 't [Myr]', fontsize=7)

    return ax_res


# --- wrappers for grid_template -------------------------------------------

def _plot_cell(ax, data):
    """Adapter for grid_template."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE,
                   use_log_x=USE_LOG_X, use_log_y=USE_LOG_Y,
                   machine_eps=MACHINE_EPS)


def _build_grid_legend():
    handles = [
        Line2D([0], [0], color=C_R2,  ls='-',  lw=2.0, label=r'$R_2$'),
        Line2D([0], [0], color=C_RIF, ls='--', lw=1.6, label=r'$R_{\rm IF}$'),
        Patch(facecolor='#b0d0ff', alpha=0.35, edgecolor='none',
              label=r'$<10^{-10}$ (machine prec.)'),
    ]
    handles.extend(get_marker_legend_handles())
    return handles


def plot_from_path(data_input, output_dir=None):
    """Plot R2 vs R_IF from a single simulation."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_grid_legend,
        file_prefix="R2_RIF",
        ylabel='Radius [pc]',
        title_fn=lambda dp: rf"$R_2$ vs $R_{{\rm IF}}$: {dp.parent.name}",
        legend_loc='upper left',
        legend_ncol=1,
        figsize=(8, 6),
        dpi=150,
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """Plot grid of R2 vs R_IF comparisons."""
    _plot_grid(
        folder_path, output_dir,
        ndens_filter=ndens_filter,
        mCloud_filter=mCloud_filter,
        sfe_filter=sfe_filter,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_grid_legend,
        file_prefix="R2_RIF",
        ylabel='Radius [pc]',
        cell_width=3.5,
        cell_height=3.0,
        dpi=300,
        sharex=False,
        sharey=False,
        legend_ncol=4,
        legend_y=1.0,
        suptitle_y=1.02,
        subplots_adjust_top=0.9,
        save_pdf=SAVE_PDF,
        mcloud_label_fn=_mcloud_label_short,
        hide_non_left_labels=True,
    )


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_R2_RIF.py",
        description="Plot TRINITY R2 vs R_IF comparison",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
