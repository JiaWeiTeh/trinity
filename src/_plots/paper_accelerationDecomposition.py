#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceleration Decomposition Plot for TRINITY

Shows which physical processes actually control the shell's motion by plotting
acceleration components from the force balance equation:

    M_sh * dv/dt = F_gas + F_rad - F_grav - (dM_sh/dt) * v

Acceleration components (all in km/s/Myr, outward positive):
- a_gas:  Gas pressure acceleration = 4*pi*R2^2*(P_drive - P_ext) / M_sh
- a_rad:  Radiation pressure acceleration = F_rad / M_sh
- a_grav: Gravitational acceleration = -F_grav / M_sh (negative = inward)
- a_acc:  Mass loading acceleration = -dM_sh/dt * v / M_sh (negative when expanding)
- a_net:  Net acceleration = sum of above (thick line)

The sign of a_net indicates:
- a_net > 0: Shell is accelerating outward
- a_net ~ 0: Quasi-equilibrium / coasting
- a_net < 0: Shell is decelerating (may lead to collapse)

Author: TRINITY Team
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, FixedLocator

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import smooth_1d
from src._output.trinity_reader import load_output
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV
from src._plots.grid_template import plot_single, plot_grid as _plot_grid, _mcloud_label_short

print("...plotting acceleration decomposition")

# Unit conversion: pc/Myr² → km/s/Myr (more intuitive unit)
# 1 pc/Myr = 0.978 km/s, so 1 pc/Myr² = 0.978 km/s/Myr
A_AU_TO_KMS_MYR = INV_CONV.v_au2kms  # pc/Myr → km/s, so pc/Myr² → km/s/Myr

# ---------------- configuration ----------------
SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
USE_SYMLOG = True  # Use symmetric log scale for accelerations
USE_LOG_X = False  # Use log scale for x-axis (time)

# Acceleration colors
C_GAS = "blue"       # Thermal/gas pressure
C_RAD = "#9467bd"    # Radiation (purple)
C_GRAV = "black"     # Gravity
C_ACC = "orange"     # Mass loading
C_NET = "gray"       # Net acceleration

# Acceleration fields
ACCEL_FIELDS = [
    ("a_gas",  r"$a_{\rm gas}$",  C_GAS,  "-",  1.5),
    ("a_rad",  r"$a_{\rm rad}$",  C_RAD,  "-",  1.5),
    ("a_grav", r"$a_{\rm grav}$", C_GRAV, "-",  1.5),
    ("a_acc",  r"$a_{\rm acc}$",  C_ACC,  "-",  1.5),
    ("a_net",  r"$a_{\rm net}$",  C_NET,  "--", 2.5),
]

# --- output
SAVE_PDF = True

def load_run(data_path):
    """Load run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Core time series
    t = output.get('t_now')
    R2 = output.get('R2')
    v2 = output.get('v2')
    phase = np.array(output.get('current_phase', as_array=False))

    # Helper to get field with default
    def get_field(field, default=np.nan):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    # Shell properties
    shell_mass = get_field('shell_mass', np.nan)
    shell_massDot = get_field('shell_massDot', np.nan)

    # Forces
    F_grav = get_field('F_grav', 0.0)
    F_rad = get_field('F_rad', 0.0)

    # Pressure fields for gas acceleration
    P_drive = get_field('P_drive', np.nan)
    press_HII_in = get_field('press_HII_in', 0.0)  # External pressure

    # Shell properties for markers
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        R2 = R2[order]
        v2 = v2[order]
        phase = phase[order]
        shell_mass = shell_mass[order]
        shell_massDot = shell_massDot[order]
        F_grav = F_grav[order]
        F_rad = F_rad[order]
        P_drive = P_drive[order]
        press_HII_in = press_HII_in[order]
        isCollapse = isCollapse[order]

    return {
        't': t, 'R2': R2, 'v2': v2, 'phase': phase,
        'shell_mass': shell_mass, 'shell_massDot': shell_massDot,
        'F_grav': F_grav, 'F_rad': F_rad,
        'P_drive': P_drive, 'press_HII_in': press_HII_in,
        'rcloud': rcloud, 'isCollapse': isCollapse,
    }


def compute_accelerations(data):
    """
    Compute acceleration components from simulation data.

    Returns dict with a_gas, a_rad, a_grav, a_acc, a_net (all in km/s/Myr)
    """
    R2 = data['R2']
    v2 = data['v2']
    shell_mass = data['shell_mass']
    shell_massDot = data['shell_massDot']
    F_grav = data['F_grav']
    F_rad = data['F_rad']
    P_drive = data['P_drive']
    press_HII_in = data['press_HII_in']

    # Handle NaN and zeros in shell mass
    shell_mass_safe = np.where(shell_mass > 0, shell_mass, np.nan)

    # Gas pressure acceleration: F_gas = 4*pi*R2^2 * (P_drive - P_ext)
    P_net = np.nan_to_num(P_drive, nan=0.0) - np.nan_to_num(press_HII_in, nan=0.0)
    F_gas = 4 * np.pi * R2**2 * P_net
    a_gas = F_gas / shell_mass_safe

    # Radiation acceleration: a_rad = F_rad / M_sh
    a_rad = np.nan_to_num(F_rad, nan=0.0) / shell_mass_safe

    # Gravity acceleration (negative = inward): a_grav = -F_grav / M_sh
    a_grav = -np.nan_to_num(F_grav, nan=0.0) / shell_mass_safe

    # Mass loading acceleration (negative when expanding with positive massDot)
    # a_acc = -dM/dt * v / M
    shell_massDot_safe = np.nan_to_num(shell_massDot, nan=0.0)
    v2_safe = np.nan_to_num(v2, nan=0.0)
    a_acc = -shell_massDot_safe * v2_safe / shell_mass_safe

    # Net acceleration (should equal dv/dt)
    a_net = a_gas + a_rad + a_grav + a_acc

    # Convert from code units (pc/Myr²) to km/s/Myr
    return {
        'a_gas': a_gas * A_AU_TO_KMS_MYR,
        'a_rad': a_rad * A_AU_TO_KMS_MYR,
        'a_grav': a_grav * A_AU_TO_KMS_MYR,
        'a_acc': a_acc * A_AU_TO_KMS_MYR,
        'a_net': a_net * A_AU_TO_KMS_MYR,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True, use_symlog=True,
                   use_log_x=False):
    """Plot acceleration decomposition on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']

    # Compute accelerations
    accels = compute_accelerations(data)

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

    # Plot each acceleration component
    for field, label, color, ls, lw in ACCEL_FIELDS:
        y = accels[field]
        if smooth_window:
            y = smooth_1d(y, smooth_window)

        # Skip if all NaN
        if np.all(~np.isfinite(y)):
            continue

        ax.plot(t, y, color=color, ls=ls, lw=lw, label=label, zorder=3)

    # Add zero reference line
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5, zorder=1)

    # Set scale
    if use_symlog:
        # Use symmetric log scale (handles positive and negative values)
        ax.set_yscale('symlog', linthresh=1e-3)
        # Reduce tick crowding: show ticks every 3 decades, only non-negative exponents
        # This avoids cramping near 10^0 by excluding 10^-3, 10^-6, etc.
        # Generate tick positions: -10^9, -10^6, -10^3, 0, 10^3, 10^6, 10^9
        tick_positions = [0]
        for exp in range(0, 10, 3):  # every 3 decades from 10^0 to 10^9 (non-negative exp only)
            tick_positions.append(10**exp)
            tick_positions.append(-10**exp)
        ax.yaxis.set_major_locator(FixedLocator(sorted(tick_positions)))
        # Remove minor ticks to reduce clutter
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_yscale('linear')

    # X-axis scale
    if use_log_x:
        # Use symlog: logarithmic for early times, linear for later times
        # linthresh=0.1 means linear above 0.1 Myr, giving more space to late evolution
        ax.set_xscale('symlog', linthresh=0.1)
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_accels = np.concatenate([accels[f] for f, _, _, _, _ in ACCEL_FIELDS])
    valid = all_accels[np.isfinite(all_accels)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        margin = 0.1 * max(abs(ymin), abs(ymax))
        ax.set_ylim(ymin - margin, ymax + margin)


def _plot_cell(ax, data):
    """Adapter: call plot_run_on_ax with module-level config."""
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, use_symlog=USE_SYMLOG,
                   use_log_x=USE_LOG_X)


def _build_single_legend():
    handles = [
        Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
        for _, label, c, ls, lw in ACCEL_FIELDS
    ]
    handles.append(Line2D([0], [0], color='gray', ls='-', lw=0.8,
                          alpha=0.5, label=r'$a=0$'))
    handles.extend(get_marker_legend_handles())
    return handles


def _build_grid_legend():
    handles = [
        Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
        for _, label, c, ls, lw in ACCEL_FIELDS
    ]
    handles.extend(get_marker_legend_handles())
    return handles


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot acceleration decomposition from a direct data path/folder."""
    plot_single(
        data_input, output_dir,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_single_legend,
        file_prefix="paper_accelerationDecomposition",
        ylabel=r"Acceleration [km s$^{-1}$ Myr$^{-1}$]",
        figsize=(8, 5),
        legend_loc="upper right",
        legend_ncol=2,
        title_fn=lambda p: f"Acceleration Decomposition: {p.parent.name}",
    )


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """Plot grid of acceleration decomposition from simulations in a folder."""
    _plot_grid(
        folder_path, output_dir,
        ndens_filter=ndens_filter,
        mCloud_filter=mCloud_filter,
        sfe_filter=sfe_filter,
        load_run_fn=load_run,
        plot_cell_fn=_plot_cell,
        legend_handles_fn=_build_grid_legend,
        file_prefix="acceleration",
        ylabel=r"$a$ [km s$^{-1}$ Myr$^{-1}$]",
        cell_width=3.0,
        cell_height=2.4,
        dpi=300,
        sharey=False,
        legend_ncol=5,
        legend_y=1.0,
        subplots_adjust_top=0.9,
        suptitle_y=1.02,
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
        script_name="paper_accelerationDecomposition.py",
        description="Plot TRINITY acceleration decomposition",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
