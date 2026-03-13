#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure Components Plot for TRINITY

Shows the three key pressure components as a function of time:

- P_HII(t): HII pressure at the ionization front (Stroemgren-based)
- Pb(t): Hot bubble pressure
- P_ram(t): Ram pressure from freely-streaming wind

Optionally also shows P_drive (total driving pressure) for reference.

Author: TRINITY Team
"""

import numpy as np
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import smooth_1d
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV, CGS

print("...plotting pressure components (P_HII, Pb, P_ram)")

# Unit conversion: code units (Msun/pc/Myr^2) -> K/cm^3 (= P/k_B = n*T)
P_AU_TO_K_CM3 = INV_CONV.Pb_au2cgs / CGS.k_B

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
SHOW_PDRIVE = True  # Show P_drive for reference
USE_LOG_X = False  # Use log scale for x-axis (time)

# --- output
SAVE_PDF = True

# Pressure field styling: (output_key, label, color, linestyle, linewidth)
PRESSURE_FIELDS = [
    ("P_HII", r"$P_{\rm HII}$",  "red",    "-",  1.8),
    ("Pb",    r"$P_b$ (bubble)",  "blue",   "-",  1.8),
    ("P_ram", r"$P_{\rm ram}$",   "green",  "-",  1.8),
]


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

    # Pressure fields - convert from code units to K/cm^3
    P_HII = get_field('P_HII', np.nan) * P_AU_TO_K_CM3
    Pb = get_field('Pb', np.nan) * P_AU_TO_K_CM3
    P_ram = get_field('P_ram', np.nan) * P_AU_TO_K_CM3
    P_drive = get_field('P_drive', np.nan) * P_AU_TO_K_CM3

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        P_HII = P_HII[order]
        Pb = Pb[order]
        P_ram = P_ram[order]
        P_drive = P_drive[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'P_HII': P_HII, 'Pb': Pb, 'P_ram': P_ram, 'P_drive': P_drive,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, show_pdrive=True,
                   use_log_x=False):
    """Plot pressure components on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']

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

    # Plot main pressure components
    for field, label, color, ls, lw in PRESSURE_FIELDS:
        y = data[field]
        if smooth_window:
            y = smooth_1d(y, smooth_window)

        # Skip if all NaN
        if np.all(~np.isfinite(y)):
            continue

        ax.plot(t, y, color=color, ls=ls, lw=lw, label=label, zorder=3)

    # Optionally plot P_drive for reference
    if show_pdrive:
        y = data['P_drive']
        if smooth_window:
            y = smooth_1d(y, smooth_window)
        if not np.all(~np.isfinite(y)):
            ax.plot(t, y, color='black', ls='--', lw=2.2,
                    label=r'$P_{\rm drive}$', alpha=0.8, zorder=4)

    ax.set_yscale('log')

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_pressures = np.concatenate([
        data['P_HII'], data['Pb'], data['P_ram']
    ])
    valid = all_pressures[np.isfinite(all_pressures) & (all_pressures > 0)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        ax.set_ylim(ymin * 0.3, ymax * 3)


def _plot_cell(ax, data):
    """Adapter for grid_template: calls plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, show_pdrive=SHOW_PDRIVE,
                   use_log_x=USE_LOG_X)


def _build_grid_legend():
    """Build legend handles for grid plots."""
    handles = [
        Line2D([0], [0], color="red", ls="-", lw=1.8, label=r"$P_{\rm HII}$"),
        Line2D([0], [0], color="blue", ls="-", lw=1.8, label=r"$P_b$ (bubble)"),
        Line2D([0], [0], color="green", ls="-", lw=1.8, label=r"$P_{\rm ram}$"),
    ]
    if SHOW_PDRIVE:
        handles.append(Line2D([0], [0], color="black", ls="--", lw=2.2,
                              alpha=0.8, label=r"$P_{\rm drive}$"))
    handles.extend(get_marker_legend_handles())
    return handles


def plot_from_path(data_input, output_dir=None):
    """Plot pressure components from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=None,
        file_prefix="PHII_Pb_Pram",
        ylabel=r"$P/k_B$ [K cm$^{-3}$]",
        title_fn=lambda data_path: f"Pressure Components: {data_path.parent.name}",
        legend_loc="upper right",
        legend_ncol=1,
        figsize=(8, 5),
        dpi=150,
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of pressure components from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
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
        file_prefix="PHII_Pb_Pram",
        ylabel=r"$P/k_B$ [K cm$^{-3}$]",
        cell_width=3.0,
        cell_height=2.4,
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


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_PHII_Pb_Pram.py",
        description="Plot TRINITY pressure components (P_HII, Pb, P_ram)",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
