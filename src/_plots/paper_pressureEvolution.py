#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pressure Evolution Plot for TRINITY

Shows how P_drive transitions between P_b (hot bubble) and P_IF (ionization front)
over time. This illustrates the convex blend model:

    P_drive = (1 - w) * P_b + w * P_IF

where w = f_abs_ion * P_IF / (P_IF + P_b)

Plot shows:
- P_b(t): Hot bubble pressure (blue solid)
- P_IF(t): Ionization front pressure (red solid)
- P_drive(t): Effective driving pressure (black dashed)
- Optionally: P_ext (external pressure, gray dotted)

Author: TRINITY Team
"""

import numpy as np
from matplotlib.lines import Line2D

from src._plots.plot_base import smooth_1d
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV, CGS

print("...plotting pressure evolution (P_b, P_IF, P_drive)")

# Unit conversion: code units (Msun/pc/Myr²) → K/cm³ (= P/k_B = n*T)
# P[K/cm³] = P[code] * Pb_au2cgs / k_B
P_AU_TO_K_CM3 = INV_CONV.Pb_au2cgs / CGS.k_B

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
SHOW_PEXT = True  # Show external pressure
USE_LOG_X = False  # Use log scale for x-axis (time)

# --- output - save to project root's fig/ directory
SAVE_PNG = False
SAVE_PDF = True

# Pressure field styling
PRESSURE_FIELDS = [
    ("Pb",      r"$P_b$ (bubble)",      "blue",  "-",  1.8),
    ("P_IF",    r"$P_{\rm IF}$",        "red",   "-",  1.8),
    ("P_drive", r"$P_{\rm drive}$",     "black", "--", 2.2),
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

    # Pressure fields - convert from code units to K/cm³
    Pb = get_field('Pb', np.nan) * P_AU_TO_K_CM3
    P_IF = get_field('P_IF', np.nan) * P_AU_TO_K_CM3
    P_drive = get_field('P_drive', np.nan) * P_AU_TO_K_CM3
    press_HII_in = get_field('press_HII_in', np.nan) * P_AU_TO_K_CM3

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        phase = phase[order]
        Pb = Pb[order]
        P_IF = P_IF[order]
        P_drive = P_drive[order]
        press_HII_in = press_HII_in[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'phase': phase,
        'Pb': Pb, 'P_IF': P_IF, 'P_drive': P_drive,
        'press_HII_in': press_HII_in,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, show_pext=True,
                   use_log_x=False):
    """Plot pressure evolution on given axes."""
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

    # Plot pressures
    for field, label, color, ls, lw in PRESSURE_FIELDS:
        y = data[field]
        if smooth_window:
            y = smooth_1d(y, smooth_window)

        # Skip if all NaN
        if np.all(~np.isfinite(y)):
            continue

        ax.plot(t, y, color=color, ls=ls, lw=lw, label=label, zorder=3)

    # Optionally plot external pressure
    if show_pext and 'press_HII_in' in data:
        y = data['press_HII_in']
        if smooth_window:
            y = smooth_1d(y, smooth_window)
        if not np.all(~np.isfinite(y)):
            ax.plot(t, y, color='gray', ls=':', lw=1.5,
                    label=r'$P_{\rm ext}$', alpha=0.7, zorder=2)

    ax.set_yscale('log')

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        # For log scale, start from first positive time
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_pressures = np.concatenate([
        data['Pb'], data['P_IF'], data['P_drive']
    ])
    valid = all_pressures[np.isfinite(all_pressures) & (all_pressures > 0)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        ax.set_ylim(ymin * 0.3, ymax * 3)


def _plot_cell(ax, data):
    """Adapter for grid_template: calls plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT,
                   use_log_x=USE_LOG_X)


def _build_grid_legend():
    """Build legend handles for grid plots."""
    handles = [
        Line2D([0], [0], color="blue", ls="-", lw=1.8, label=r"$P_b$ (bubble)"),
        Line2D([0], [0], color="red", ls="-", lw=1.8, label=r"$P_{\rm IF}$"),
        Line2D([0], [0], color="black", ls="--", lw=2.2, label=r"$P_{\rm drive}$"),
    ]
    if SHOW_PEXT:
        handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.5,
                              alpha=0.7, label=r"$P_{\rm ext}$"))
    handles.extend(get_marker_legend_handles())
    return handles


def plot_from_path(data_input, output_dir=None):
    """Plot pressure evolution from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=None,
        file_prefix="paper_pressureEvolution",
        ylabel=r"$P/k_B$ [K cm$^{-3}$]",
        title_fn=lambda data_path: f"Pressure Evolution: {data_path.parent.name}",
        post_plot_fn=None,
        legend_loc="upper right",
        legend_ncol=1,
        figsize=(8, 5),
        dpi=150,
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of pressure evolution from simulations in a folder.

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
        file_prefix="pressure",
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
        script_name="paper_pressureEvolution.py",
        description="Plot TRINITY pressure evolution",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
