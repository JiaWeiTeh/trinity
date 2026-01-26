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

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import SymmetricalLogLocator, NullLocator, FixedLocator

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._functions.unit_conversions import INV_CONV

print("...plotting acceleration decomposition")

# Unit conversion: pc/Myr² → km/s/Myr (more intuitive unit)
# 1 pc/Myr = 0.978 km/s, so 1 pc/Myr² = 0.978 km/s/Myr
A_AU_TO_KMS_MYR = INV_CONV.v_au2kms  # pc/Myr → km/s, so pc/Myr² → km/s/Myr

# ---------------- configuration ----------------
mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
ndens_list = ["1e3"]
sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"

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

# --- optional single-run view (set to None for full grid)
ONLY_M = "1e7"
ONLY_N = "1e4"
ONLY_SFE = "010"

# Comment this out for single mode, leave for grid mode
ONLY_M = ONLY_N = ONLY_SFE = None

# --- output
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


def range_tag(prefix, values, key=float):
    """Create tag string from list of values."""
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"


def smooth_1d(y, window, mode="edge"):
    """Apply 1D smoothing with moving average."""
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")


def load_run(data_path: Path):
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


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot acceleration decomposition from a direct data path/folder."""
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading data from: {data_path}")

    try:
        data = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, use_symlog=USE_SYMLOG,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Acceleration [km s$^{-1}$ Myr$^{-1}$]")
    ax.set_title(f"Acceleration Decomposition: {data_path.parent.name}")

    # Legend
    handles = [
        Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
        for _, label, c, ls, lw in ACCEL_FIELDS
    ]
    handles.append(Line2D([0], [0], color='gray', ls='-', lw=0.8,
                          alpha=0.5, label=r'$a=0$'))
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_accelerationDecomposition_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_single_run(mCloud, ndens, sfe):
    """Plot single run from config."""
    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
    data_path = find_data_file(BASE_DIR, run_name)
    if data_path is None:
        print(f"Missing data for: {run_name}")
        return

    data = load_run(data_path)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=400, constrained_layout=True)
    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                   phase_change=PHASE_CHANGE, use_symlog=USE_SYMLOG,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Acceleration [km s$^{-1}$ Myr$^{-1}$]")
    ax.set_title(f"{run_name}")

    # Legend
    handles = [
        Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
        for _, label, c, ls, lw in ACCEL_FIELDS
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    tag = f"accelerationDecomposition_{mCloud}_sfe{sfe}_n{ndens}"
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of acceleration decomposition."""
    for ndens in ndens_list:
        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.0 * ncols, 2.4 * nrows),
            sharex=False, sharey=False,
            dpi=300,
            constrained_layout=False
        )

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                data_path = find_data_file(BASE_DIR, run_name)

                if data_path is None:
                    print(f"  {run_name}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"  Loading: {data_path}")
                try:
                    data = load_run(data_path)
                    plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                                   phase_change=PHASE_CHANGE, use_symlog=USE_SYMLOG,
                                   use_log_x=USE_LOG_X)
                except Exception as e:
                    print(f"Error in {run_name}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                # X-axis labels only on bottom row
                ax.tick_params(axis="x", which="both", bottom=True)
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

                # Column titles
                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                # Y-axis label on left column
                if j == 0:
                    mval = float(mCloud)
                    mexp = int(np.floor(np.log10(mval)))
                    mcoeff = round(mval / (10 ** mexp))
                    if mcoeff == 10:
                        mcoeff = 1
                        mexp += 1
                    if mcoeff == 1:
                        mlabel = rf"$M_{{\rm cl}}=10^{{{mexp}}}$"
                    else:
                        mlabel = rf"$M_{{\rm cl}}={mcoeff}\times10^{{{mexp}}}$"
                    ax.set_ylabel(mlabel + "\n" + r"$a$ [km s$^{-1}$ Myr$^{-1}$]")
                else:
                    ax.tick_params(labelleft=False)

        # Global legend
        handles = [
            Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
            for _, label, c, ls, lw in ACCEL_FIELDS
        ]
        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Acceleration Decomposition ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

        leg = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=5,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.0)
        )
        leg.set_zorder(10)

        # Save
        m_tag = range_tag("M", mCloud_list, key=float)
        sfe_tag = range_tag("sfe", sfe_list, key=int)
        n_tag = f"n{ndens}"
        tag = f"accelerationDecomposition_grid_{m_tag}_{sfe_tag}_{n_tag}"

        if SAVE_PDF:
            out_pdf = FIG_DIR / f"{tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)


def plot_folder_grid(folder_path, output_dir=None):
    """
    Create grid plot from all simulations found in a folder.

    Searches subfolders for dictionary.jsonl files, parses simulation
    parameters from folder names (e.g., "1e7_sfe020_n1e4"), and arranges
    them in a grid sorted by:
    - Rows: increasing mCloud (top to bottom)
    - Columns: increasing SFE (left to right)

    Saves PDF as {folder_name}_{ndens}.pdf without displaying.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)

    Notes
    -----
    Folder names must follow the pattern: {mCloud}_sfe{sfe}_n{ndens}
    Examples: "1e7_sfe020_n1e4", "5e6_sfe010_n1e3"
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    organized = organize_simulations_for_grid(sim_files)
    mCloud_list_found = organized['mCloud_list']
    sfe_list_found = organized['sfe_list']
    grid = organized['grid']

    if not mCloud_list_found or not sfe_list_found:
        print(f"Could not organize simulations into grid")
        return

    print(f"Found {len(sim_files)} simulations")
    print(f"  mCloud: {mCloud_list_found}")
    print(f"  SFE: {sfe_list_found}")

    nrows, ncols = len(mCloud_list_found), len(sfe_list_found)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.0 * ncols, 2.4 * nrows),
        sharex=False, sharey=False,
        dpi=300,
        squeeze=False,
        constrained_layout=False
    )

    for i, mCloud in enumerate(mCloud_list_found):
        for j, sfe in enumerate(sfe_list_found):
            ax = axes[i, j]
            data_path = grid.get((mCloud, sfe))

            if data_path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            print(f"  Loading: {data_path}")
            try:
                data = load_run(data_path)
                plot_run_on_ax(ax, data, smooth_window=SMOOTH_WINDOW,
                               phase_change=PHASE_CHANGE, use_symlog=USE_SYMLOG,
                               use_log_x=USE_LOG_X)
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            ax.tick_params(axis="x", which="both", bottom=True)
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            if j == 0:
                mval = float(mCloud)
                mexp = int(np.floor(np.log10(mval)))
                mcoeff = round(mval / (10 ** mexp))
                if mcoeff == 10:
                    mcoeff = 1
                    mexp += 1
                if mcoeff == 1:
                    mlabel = rf"$M_{{\rm cl}}=10^{{{mexp}}}$"
                else:
                    mlabel = rf"$M_{{\rm cl}}={mcoeff}\times10^{{{mexp}}}$"
                ax.set_ylabel(mlabel + "\n" + r"$a$ [km s$^{-1}$ Myr$^{-1}$]")
            else:
                ax.tick_params(labelleft=False)

    handles = [
        Line2D([0], [0], color=c, ls=ls, lw=lw, label=label)
        for _, label, c, ls, lw in ACCEL_FIELDS
    ]
    handles.extend(get_marker_legend_handles())

    fig.subplots_adjust(top=0.9)
    fig.suptitle(folder_name, fontsize=14, y=1.02)

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.0)
    )
    leg.set_zorder(10)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    ndens = organized['ndens']
    ndens_tag = f"n{ndens}" if ndens else "nMixed"
    out_pdf = fig_dir / f"{folder_name}_{ndens_tag}_acceleration.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY acceleration decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_accelerationDecomposition.py 1e7_sfe020_n1e4
  python paper_accelerationDecomposition.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_accelerationDecomposition.py /path/to/dictionary.jsonl

  # Folder-based grid (auto-discovers simulations)
  python paper_accelerationDecomposition.py --folder /path/to/my_experiment/
  python paper_accelerationDecomposition.py -F /path/to/simulations/

  # Uses config at top of file
  python paper_accelerationDecomposition.py
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Base directory for output folders'
    )
    parser.add_argument(
        '--linear', action='store_true',
        help='Use linear scale instead of symmetric log scale'
    )
    parser.add_argument(
        '--log-x', action='store_true',
        help='Use log scale for x-axis (time)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Search folder recursively for simulations and create grid plot. '
             'Auto-organizes by mCloud (rows) and SFE (columns). '
             'Saves as {folder}_{ndens}.pdf'
    )

    args = parser.parse_args()

    if args.linear:
        USE_SYMLOG = False
    if args.log_x:
        USE_LOG_X = True

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir)
    elif args.data:
        plot_from_path(args.data, args.output_dir)
    elif (ONLY_M is not None) and (ONLY_N is not None) and (ONLY_SFE is not None):
        plot_single_run(ONLY_M, ONLY_N, ONLY_SFE)
    else:
        plot_grid()
