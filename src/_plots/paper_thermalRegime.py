#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal Regime Plot for TRINITY

Shows the blending weight w_blend(t) which indicates which thermal closure
dominates the driving pressure:

    P_drive = (1 - w) * P_b + w * P_IF

where w = f_abs_ion * P_IF / (P_IF + P_b)

Interpretation:
- w ~ 0: Hot bubble dominates (energy-driven regime)
- w ~ 1: Warm ionized gas dominates (HII-driven regime)
- 0 < w < 1: Transition regime

Two visualization modes:
1. Line plot of w_blend(t)
2. Stacked area showing (1-w) vs w as fractional contributions

Author: TRINITY Team
"""

import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src._plots.plot_base import smooth_1d
from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

print("...plotting thermal regime (w_blend)")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
PLOT_MODE = "line"  # "line" or "stacked"
USE_LOG_X = False  # Use log scale for x-axis (time)

# Colors for stacked mode
C_BUBBLE = "blue"
C_HII = "#d62728"  # red

SAVE_PDF = True


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

    # Blending weight
    w_blend = get_field('w_blend', np.nan)

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        w_blend = w_blend[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'w_blend': w_blend,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, plot_mode="line",
                   use_log_x=False):
    """Plot thermal regime on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']
    w_blend = data['w_blend'].copy()

    # Apply smoothing
    if smooth_window:
        w_blend = smooth_1d(w_blend, smooth_window)

    # Clip to [0, 1] range
    w_blend = np.clip(w_blend, 0, 1)

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

    if plot_mode == "stacked":
        # Stacked area showing (1-w) and w
        bubble_frac = 1 - w_blend
        hii_frac = w_blend

        ax.fill_between(t, 0, bubble_frac, color=C_BUBBLE, alpha=0.7,
                        label=r"$(1-w)$ Bubble", lw=0, zorder=2)
        ax.fill_between(t, bubble_frac, 1, color=C_HII, alpha=0.7,
                        label=r"$w$ HII", lw=0, zorder=2)

        # Add dividing line
        ax.plot(t, bubble_frac, color='black', lw=0.8, alpha=0.5, zorder=3)

    else:  # line mode
        # Plot w_blend as line
        ax.plot(t, w_blend, color='black', lw=2, label=r'$w_{\rm blend}$', zorder=3)

        # Add reference lines
        ax.axhline(0.0, color='blue', ls='--', lw=1, alpha=0.4, zorder=1)
        ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.4, zorder=1)
        ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.4, zorder=1)

        # Shade regime regions with gradient fading to white at 0.5
        # Use multiple strips with decreasing alpha toward 0.5
        n_strips = 20
        max_alpha = 0.15
        # Bottom (bubble) gradient: 0 to 0.5, alpha decreases toward 0.5
        for i in range(n_strips):
            y0 = i * 0.5 / n_strips
            y1 = (i + 1) * 0.5 / n_strips
            # Alpha decreases linearly from max_alpha at y=0 to 0 at y=0.5
            alpha = max_alpha * (1 - (i + 0.5) / n_strips)
            ax.fill_between(t, y0, y1, color=C_BUBBLE, alpha=alpha, lw=0, zorder=0)
        # Top (HII) gradient: 0.5 to 1.0, alpha increases from 0.5 toward 1.0
        for i in range(n_strips):
            y0 = 0.5 + i * 0.5 / n_strips
            y1 = 0.5 + (i + 1) * 0.5 / n_strips
            # Alpha increases linearly from 0 at y=0.5 to max_alpha at y=1.0
            alpha = max_alpha * ((i + 0.5) / n_strips)
            ax.fill_between(t, y0, y1, color=C_HII, alpha=alpha, lw=0, zorder=0)

        # Add regime labels at edges
        ax.text(0.02, 0.05, "Bubble-dominated", transform=ax.transAxes,
                fontsize=7, color=C_BUBBLE, alpha=0.8, va='bottom')
        ax.text(0.02, 0.95, "HII-dominated", transform=ax.transAxes,
                fontsize=7, color=C_HII, alpha=0.8, va='top')

    ax.set_ylim(0, 1)

    # X-axis scale - start from where w_blend first exceeds threshold
    w_threshold = 1e-2
    valid_mask = w_blend > w_threshold
    if np.any(valid_mask):
        t_start = t[valid_mask].min()
    else:
        t_start = t[t > 0].min() if np.any(t > 0) else t.min()

    if use_log_x:
        # Use symlog: logarithmic for early times, linear for later times
        # linthresh=0.1 means linear above 0.1 Myr, giving more space to late evolution
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_xlim(max(t_start, 1e-6), t.max())
    else:
        ax.set_xlim(t_start, t.max())


# ------------------------------------------------------------------
# Adapters for grid_template
# ------------------------------------------------------------------

def _plot_cell(ax, data):
    """Adapter: call plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, plot_mode=PLOT_MODE,
                   use_log_x=USE_LOG_X)


def _build_single_legend():
    """Legend handles for single-plot mode."""
    if PLOT_MODE == "stacked":
        handles = [
            Patch(facecolor=C_BUBBLE, alpha=0.7, label=r"$(1-w)$ Bubble contribution"),
            Patch(facecolor=C_HII, alpha=0.7, label=r"$w$ HII contribution"),
        ]
    else:
        handles = [
            Line2D([0], [0], color='black', lw=2, label=r'$w_{\rm blend}$'),
            Line2D([0], [0], color='blue', ls='--', lw=1, alpha=0.4, label='w=0 (bubble)'),
            Line2D([0], [0], color='red', ls='--', lw=1, alpha=0.4, label='w=1 (HII)'),
        ]
    handles.extend(get_marker_legend_handles())
    return handles


def _build_grid_legend():
    """Legend handles for grid-plot mode."""
    if PLOT_MODE == "stacked":
        handles = [
            Patch(facecolor=C_BUBBLE, alpha=0.7, label=r"$(1-w)$ Bubble"),
            Patch(facecolor=C_HII, alpha=0.7, label=r"$w$ HII"),
        ]
    else:
        handles = [
            Line2D([0], [0], color='black', lw=2, label=r'$w_{\rm blend}$'),
            Patch(facecolor=C_BUBBLE, alpha=0.1, edgecolor='none', label='Bubble regime (w<0.3)'),
            Patch(facecolor=C_HII, alpha=0.1, edgecolor='none', label='HII regime (w>0.7)'),
        ]
    handles.extend(get_marker_legend_handles())
    return handles


# ------------------------------------------------------------------
# Public entry points
# ------------------------------------------------------------------

def plot_from_path(data_input: str, output_dir: str = None):
    """Plot thermal regime from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_single_legend,
        file_prefix="paper_thermalRegime",
        ylabel=r"$w_{\rm blend}$ (HII weight)",
        title_fn=lambda p: f"Thermal Regime: {p.parent.name}",
        figsize=(8, 5),
        legend_loc="upper right",
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of thermal regime from simulations in a folder.

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
    _plot_grid(
        folder_path, output_dir,
        ndens_filter=ndens_filter,
        mCloud_filter=mCloud_filter,
        sfe_filter=sfe_filter,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_grid_legend,
        file_prefix="thermal",
        ylabel=r"$w_{\rm blend}$",
        cell_width=3.0,
        cell_height=2.4,
        dpi=300,
        sharey=True,
        subplots_adjust_top=0.9,
        suptitle_y=1.02,
        legend_ncol=4,
        legend_y=1.0,
        hide_non_left_labels=True,
        mcloud_label_fn=_mcloud_label_short,
        save_pdf=SAVE_PDF,
    )


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_thermalRegime.py",
        description="Plot TRINITY thermal regime",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
