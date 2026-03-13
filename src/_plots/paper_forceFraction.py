#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Fraction Stacked Area Plot for TRINITY

Shows the relative importance of different feedback forces as fractions
of the total force budget over time.

Forces shown (all mechanically distinct and additive):
- F_thermal: Thermal pressure force (4*pi*R2^2 * P_drive) - combines hot bubble + warm HII
- F_rad: Radiation pressure force
- F_grav: Gravitational force

This is physically correct because these forces ARE additive in the equation
of motion. The thermal force is the combined driving pressure, not the
individual P_b and P_IF pressures (which are not additive).

Fractions are computed as: F_i / F_tot where F_tot = sum(|F_i|)
All fractions sum to 1.0 at each timestep.

Author: TRINITY Team
"""

import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

from src._plots.plot_base import smooth_1d, smooth_2d
from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

print("...plotting force fractions (F_thermal, F_rad, F_grav)")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 11  # None or 1 disables
PHASE_CHANGE = True
USE_LOG_X = False  # Use log scale for x-axis (time)

# Force colors (consistent with existing TRINITY plots)
C_GRAV = "black"
C_THERMAL = "blue"  # Thermal = bubble + HII combined
C_RAD = "#9467bd"   # Purple for radiation

# Force fields to show
FORCE_FIELDS = [
    ("F_grav",    r"$F_{\rm grav}$",    C_GRAV),
    ("F_thermal", r"$F_{\rm thermal}$", C_THERMAL),
    ("F_rad",     r"$F_{\rm rad}$",     C_RAD),
]

SAVE_PNG = False
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

    # Get individual forces
    F_grav = get_field('F_grav', 0.0)
    F_rad = get_field('F_rad', 0.0)

    # Thermal force: F_ram (from Pb) + F_HII (from P_IF contribution)
    # F_ram = Pb * 4*pi*R2^2, F_HII captures the P_IF excess
    F_ram = get_field('F_ram', 0.0)
    F_HII = get_field('F_HII', 0.0)

    # Alternative: compute from P_drive directly if available
    P_drive = get_field('P_drive', np.nan)
    if not np.all(np.isnan(P_drive)):
        # F_thermal = P_drive * 4*pi*R2^2
        F_thermal = P_drive * 4 * np.pi * R2**2
    else:
        # Fallback: use F_ram + F_HII
        F_thermal = np.nan_to_num(F_ram, nan=0.0) + np.nan_to_num(F_HII, nan=0.0)

    # Clean up NaN
    F_grav = np.nan_to_num(F_grav, nan=0.0)
    F_rad = np.nan_to_num(F_rad, nan=0.0)
    F_thermal = np.nan_to_num(F_thermal, nan=0.0)

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        F_grav = F_grav[order]
        F_thermal = F_thermal[order]
        F_rad = F_rad[order]
        isCollapse = isCollapse[order]

    # Stack forces in order: grav, thermal, rad
    forces = np.vstack([F_grav, F_thermal, F_rad])

    return {
        't': t, 'R2': R2, 'phase': phase,
        'forces': forces,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, alpha=0.75,
                   use_log_x=False):
    """Plot force fractions on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']
    forces = data['forces'].copy()

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

    # Apply smoothing
    forces = smooth_2d(forces, smooth_window)

    # Use absolute values for force fractions
    forces_abs = np.abs(forces)

    # Compute total and fractions
    ftotal = forces_abs.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)
    frac = forces_abs / ftotal

    # Cumulative sum for stacking
    cum = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    # Fill stacked areas
    for (field, label, color), y0, y1 in zip(FORCE_FIELDS, prev, cum):
        ax.fill_between(t, y0, y1, color=color, alpha=alpha, lw=0, zorder=2)

    # Add reference line at 0.5
    ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.5, zorder=1)

    ax.set_ylim(0, 1)

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


# ------------------------------------------------------------------
# Adapter / legend helpers
# ------------------------------------------------------------------

def _plot_cell(ax, data):
    """Adapter: call plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, use_log_x=USE_LOG_X)


def _build_single_legend():
    """Legend handles for single-plot mode."""
    handles = [
        Patch(facecolor=C_GRAV, alpha=0.75, label=r"$F_{\rm grav}$"),
        Patch(facecolor=C_THERMAL, alpha=0.75, label=r"$F_{\rm thermal}$"),
        Patch(facecolor=C_RAD, alpha=0.75, label=r"$F_{\rm rad}$"),
    ]
    handles.extend(get_marker_legend_handles())
    return handles


def _build_grid_legend():
    """Legend handles for grid-plot mode (extended labels)."""
    handles = [
        Patch(facecolor=C_GRAV, alpha=0.75, label=r"$F_{\rm grav}$ (Gravity)"),
        Patch(facecolor=C_THERMAL, alpha=0.75, label=r"$F_{\rm thermal}$ (Thermal pressure)"),
        Patch(facecolor=C_RAD, alpha=0.75, label=r"$F_{\rm rad}$ (Radiation)"),
    ]
    handles.extend(get_marker_legend_handles())
    return handles


# ------------------------------------------------------------------
# Public entry points
# ------------------------------------------------------------------

def plot_from_path(data_input: str, output_dir: str = None):
    """Plot force fractions from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_single_legend,
        file_prefix="paper_forceFraction",
        ylabel=r"$|F_i|/F_{\rm tot}$",
        title_fn=lambda dp: f"Force Fractions: {dp.parent.name}",
        figsize=(8, 5),
        legend_loc="upper right",
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of force fractions from simulations in a folder.

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
        file_prefix="forceFraction",
        ylabel=r"$|F_i|/F_{\rm tot}$",
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
    )


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_forceFraction.py",
        description="Plot TRINITY force fractions",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
