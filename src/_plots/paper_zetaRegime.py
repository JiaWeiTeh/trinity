#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zeta Regime Plot for TRINITY

Shows the ζ = R_eq / R_St diagnostic (Lancaster+2025) as a function of time,
indicating which pressure mechanism dominates shell dynamics:

    ζ > 1: Wind-Blown Bubble (WBB) dominated — bubble thermal/ram pressure
           exceeds what photoionisation alone can provide.
    ζ < 1: Pressure-driven by Ionised Radiation (PIR) dominated — the HII
           region is independently pressurised and the Strömgren n_IF
           correction is active.
    ζ = 1: Equipartition boundary.

Two visualization modes:
1. Line plot of ζ(t) with shaded WBB/PIR regions (default)
2. Stacked area showing WBB vs PIR fractional dominance

Author: TRINITY Team
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import smooth_1d
from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

print("...plotting zeta regime (Lancaster+2025)")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
PLOT_MODE = "line"  # "line" or "stacked"
USE_LOG_X = False  # Use log scale for x-axis (time)
USE_LOG_Y = True   # Use log scale for y-axis (zeta)

# Colors
C_WBB = "blue"       # Wind-blown bubble regime
C_PIR = "#d62728"    # Photoionisation-driven regime

SAVE_PDF = True


def load_run(data_path):
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

    # Zeta diagnostic
    zeta = get_field('zeta', np.nan)

    # Also load n_IF and n_IF_Str for optional secondary panel
    n_IF = get_field('n_IF', np.nan)
    n_IF_Str = get_field('n_IF_Str', np.nan)

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase = t[order], R2[order], phase[order]
        zeta, n_IF, n_IF_Str = zeta[order], n_IF[order], n_IF_Str[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'zeta': zeta, 'n_IF': n_IF, 'n_IF_Str': n_IF_Str,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, plot_mode="line",
                   use_log_x=False, use_log_y=True):
    """Plot ζ regime on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']
    zeta = data['zeta'].copy()

    # Apply smoothing
    if smooth_window:
        zeta = smooth_1d(zeta, smooth_window)

    # Replace non-positive with tiny value for log scale
    zeta = np.where(np.isfinite(zeta) & (zeta > 0), zeta, np.nan)

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
        # Stacked area: fraction above/below ζ=1
        # Map ζ -> weight: w = 1/(1+ζ). w>0.5 => PIR, w<0.5 => WBB
        valid = np.isfinite(zeta)
        w = np.where(valid, 1.0 / (1.0 + zeta), 0.5)
        w = np.clip(w, 0, 1)

        wbb_frac = 1 - w
        pir_frac = w

        ax.fill_between(t, 0, wbb_frac, color=C_WBB, alpha=0.7,
                        label="WBB dominated", lw=0, zorder=2)
        ax.fill_between(t, wbb_frac, 1, color=C_PIR, alpha=0.7,
                        label="PIR dominated", lw=0, zorder=2)
        ax.plot(t, wbb_frac, color='black', lw=0.8, alpha=0.5, zorder=3)

        ax.set_ylim(0, 1)

    else:  # line mode
        ax.plot(t, zeta, color='black', lw=2,
                label=r'$\zeta = R_{\rm eq}/R_{\rm St}$', zorder=3)

        # Reference lines
        ax.axhline(1.0, color='gray', ls='-', lw=1.2, alpha=0.5, zorder=1,
                    label=r'$\zeta = 1$ (equipartition)')
        ax.axhline(0.5, color=C_PIR, ls=':', lw=1, alpha=0.5, zorder=1,
                    label=r'$\zeta = 0.5$ (PIR threshold)')

        # Shade regime regions with gradient
        t_range = t
        if use_log_y:
            ax.set_yscale('log')
            # Shade below ζ=1 (PIR regime)
            ax.fill_between(t_range, 1e-3, 1.0, color=C_PIR, alpha=0.06,
                            lw=0, zorder=0)
            # Shade above ζ=1 (WBB regime)
            valid_zeta = zeta[np.isfinite(zeta)]
            y_top = max(valid_zeta.max() * 3, 10) if len(valid_zeta) > 0 else 10
            ax.fill_between(t_range, 1.0, y_top, color=C_WBB, alpha=0.06,
                            lw=0, zorder=0)

            # Auto y-limits
            if len(valid_zeta) > 0:
                ymin = max(valid_zeta[valid_zeta > 0].min() * 0.3, 1e-3) if np.any(valid_zeta > 0) else 1e-3
                ymax = valid_zeta.max() * 3
                ax.set_ylim(ymin, ymax)
        else:
            # Linear y-axis
            valid_zeta = zeta[np.isfinite(zeta)]
            y_max = valid_zeta.max() * 1.2 if len(valid_zeta) > 0 else 5
            ax.fill_between(t_range, 0, 1.0, color=C_PIR, alpha=0.06,
                            lw=0, zorder=0)
            ax.fill_between(t_range, 1.0, y_max, color=C_WBB, alpha=0.06,
                            lw=0, zorder=0)
            ax.set_ylim(0, y_max)

        # Regime labels
        ax.text(0.02, 0.95, r"WBB ($\zeta > 1$)", transform=ax.transAxes,
                fontsize=7, color=C_WBB, alpha=0.8, va='top')
        ax.text(0.02, 0.05, r"PIR ($\zeta < 1$)", transform=ax.transAxes,
                fontsize=7, color=C_PIR, alpha=0.8, va='bottom')

    # X-axis
    if use_log_x:
        ax.set_xscale('symlog', linthresh=0.1)
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


# ------------------------------------------------------------------
# Adapters for grid_template
# ------------------------------------------------------------------

def _plot_cell(ax, data):
    """Adapter: call plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, plot_mode=PLOT_MODE,
                   use_log_x=USE_LOG_X, use_log_y=USE_LOG_Y)


def _build_single_legend():
    """Legend handles for single-plot mode."""
    if PLOT_MODE == "stacked":
        handles = [
            Patch(facecolor=C_WBB, alpha=0.7, label="WBB dominated"),
            Patch(facecolor=C_PIR, alpha=0.7, label="PIR dominated"),
        ]
    else:
        handles = [
            Line2D([0], [0], color='black', lw=2,
                   label=r'$\zeta = R_{\rm eq}/R_{\rm St}$'),
            Line2D([0], [0], color='gray', ls='-', lw=1.2, alpha=0.5,
                   label=r'$\zeta = 1$ (equipartition)'),
            Line2D([0], [0], color=C_PIR, ls=':', lw=1, alpha=0.5,
                   label=r'$\zeta = 0.5$ (PIR threshold)'),
            Patch(facecolor=C_WBB, alpha=0.06, edgecolor='none',
                  label=r'WBB regime ($\zeta > 1$)'),
            Patch(facecolor=C_PIR, alpha=0.06, edgecolor='none',
                  label=r'PIR regime ($\zeta < 1$)'),
        ]
    handles.extend(get_marker_legend_handles())
    return handles


def _build_grid_legend():
    """Legend handles for grid-plot mode."""
    if PLOT_MODE == "stacked":
        handles = [
            Patch(facecolor=C_WBB, alpha=0.7, label="WBB"),
            Patch(facecolor=C_PIR, alpha=0.7, label="PIR"),
        ]
    else:
        handles = [
            Line2D([0], [0], color='black', lw=2,
                   label=r'$\zeta$'),
            Line2D([0], [0], color='gray', ls='-', lw=1.2, alpha=0.5,
                   label=r'$\zeta=1$'),
            Patch(facecolor=C_WBB, alpha=0.1, edgecolor='none',
                  label='WBB'),
            Patch(facecolor=C_PIR, alpha=0.1, edgecolor='none',
                  label='PIR'),
        ]
    handles.extend(get_marker_legend_handles())
    return handles


# ------------------------------------------------------------------
# Public entry points
# ------------------------------------------------------------------

def plot_from_path(data_input, output_dir=None):
    """Plot ζ regime from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_single_legend,
        file_prefix="paper_zetaRegime",
        ylabel=r"$\zeta = R_{\rm eq} / R_{\rm St}$",
        title_fn=lambda p: f"WBB vs PIR Regime: {p.parent.name}",
        figsize=(8, 5),
        legend_loc="upper right",
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of ζ regime from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4").
    mCloud_filter : list of str, optional
        Filter simulations by cloud mass (e.g., ["1e6", "1e7"]).
    sfe_filter : list of str, optional
        Filter simulations by SFE (e.g., ["001", "010"]).
    """
    _plot_grid(
        folder_path, output_dir,
        ndens_filter=ndens_filter,
        mCloud_filter=mCloud_filter,
        sfe_filter=sfe_filter,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_grid_legend,
        file_prefix="zetaRegime",
        ylabel=r"$\zeta$",
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
        script_name="paper_zetaRegime.py",
        description="Plot TRINITY zeta regime (WBB vs PIR, Lancaster+2025)",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
