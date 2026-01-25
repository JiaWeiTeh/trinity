#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cancellation Metric Plot for TRINITY

Shows when forces nearly balance (quasi-equilibrium) by plotting the
cancellation metric:

    C(t) = |a_net| / (|a_gas| + |a_rad| + |a_grav| + |a_acc|)

Interpretation:
- C ~ 1: One term dominates completely, dynamics clearly driven by that term
- C ~ 0: Large cancellation, forces nearly balance, dynamics controlled by small residual
- C ~ 0.1-0.3: Typical quasi-equilibrium expansion

The metric is bounded: C in [0, 1] by construction.

Author: TRINITY Team
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting cancellation metric")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
ndens_list = ["1e3"]
sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"

SMOOTH_WINDOW = 7  # None or 1 disables
PHASE_CHANGE = True
SHOW_EQUILIBRIUM_BAND = True  # Shade quasi-equilibrium region
USE_LOG_X = False  # Use log scale for x-axis (time)

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
    press_HII_in = get_field('press_HII_in', 0.0)

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


def compute_cancellation_metric(data):
    """
    Compute cancellation metric from simulation data.

    C = |a_net| / (|a_gas| + |a_rad| + |a_grav| + |a_acc|)

    Returns dict with cancellation metric C and individual accelerations.
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

    # Gas pressure acceleration
    P_net = np.nan_to_num(P_drive, nan=0.0) - np.nan_to_num(press_HII_in, nan=0.0)
    F_gas = 4 * np.pi * R2**2 * P_net
    a_gas = F_gas / shell_mass_safe

    # Radiation acceleration
    a_rad = np.nan_to_num(F_rad, nan=0.0) / shell_mass_safe

    # Gravity acceleration (negative)
    a_grav = -np.nan_to_num(F_grav, nan=0.0) / shell_mass_safe

    # Mass loading acceleration
    shell_massDot_safe = np.nan_to_num(shell_massDot, nan=0.0)
    v2_safe = np.nan_to_num(v2, nan=0.0)
    a_acc = -shell_massDot_safe * v2_safe / shell_mass_safe

    # Net acceleration
    a_net = a_gas + a_rad + a_grav + a_acc

    # Cancellation metric: C = |a_net| / sum(|a_i|)
    sum_abs = np.abs(a_gas) + np.abs(a_rad) + np.abs(a_grav) + np.abs(a_acc)
    C = np.where(sum_abs > 0, np.abs(a_net) / sum_abs, np.nan)

    # Clip to [0, 1] for numerical safety
    C = np.clip(C, 0, 1)

    return {
        'C': C,
        'a_gas': a_gas,
        'a_rad': a_rad,
        'a_grav': a_grav,
        'a_acc': a_acc,
        'a_net': a_net,
    }


def plot_run_on_ax(ax, data, smooth_window=None, phase_change=True,
                   show_rcloud=True, show_collapse=True,
                   show_equilibrium_band=True, use_log_x=False):
    """Plot cancellation metric on given axes."""
    t = data['t']
    R2 = data['R2']
    phase = data['phase']
    rcloud = data['rcloud']
    isCollapse = data['isCollapse']

    # Compute cancellation metric
    result = compute_cancellation_metric(data)
    C = result['C'].copy()

    # Apply smoothing
    if smooth_window:
        C = smooth_1d(C, smooth_window)

    # Clip again after smoothing
    C = np.clip(C, 0, 1)

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

    # Shade equilibrium region (C < 0.3)
    if show_equilibrium_band:
        ax.fill_between(t, 0, 0.3, color='green', alpha=0.1, zorder=0,
                        label='Quasi-equilibrium')

    # Plot cancellation metric
    ax.plot(t, C, color='black', lw=2, label=r'$\mathcal{C}$', zorder=3)

    # Reference lines
    ax.axhline(0.1, color='green', ls='--', lw=1, alpha=0.5, zorder=1)
    ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5, zorder=1)
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.5, zorder=1)

    # Add regime labels
    ax.text(0.98, 0.05, "Strong cancellation", transform=ax.transAxes,
            fontsize=7, color='green', alpha=0.8, ha='right', va='bottom')
    ax.text(0.98, 0.95, "Single force dominates", transform=ax.transAxes,
            fontsize=7, color='gray', alpha=0.8, ha='right', va='top')

    ax.set_ylim(0, 1.05)

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


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot cancellation metric from a direct data path/folder."""
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
                   phase_change=PHASE_CHANGE,
                   show_equilibrium_band=SHOW_EQUILIBRIUM_BAND,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Cancellation metric $\mathcal{C}$")
    ax.set_title(f"Force Cancellation: {data_path.parent.name}")

    # Legend
    handles = [
        Line2D([0], [0], color='black', lw=2,
               label=r'$\mathcal{C} = |a_{\rm net}|/\Sigma|a_i|$'),
        Patch(facecolor='green', alpha=0.1, edgecolor='none',
              label='Quasi-equilibrium (C<0.3)'),
    ]
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_cancellationMetric_{run_name}.pdf"
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
                   phase_change=PHASE_CHANGE,
                   show_equilibrium_band=SHOW_EQUILIBRIUM_BAND,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$\mathcal{C}$")
    ax.set_title(f"{run_name}")

    tag = f"cancellationMetric_{mCloud}_sfe{sfe}_n{ndens}"
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of cancellation metric."""
    for ndens in ndens_list:
        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.0 * ncols, 2.4 * nrows),
            sharex=False, sharey=True,
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
                                   phase_change=PHASE_CHANGE,
                                   show_equilibrium_band=SHOW_EQUILIBRIUM_BAND,
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
                    ax.set_ylabel(mlabel + "\n" + r"$\mathcal{C}$")
                else:
                    ax.tick_params(labelleft=False)

        # Global legend
        handles = [
            Line2D([0], [0], color='black', lw=2,
                   label=r'$\mathcal{C} = |a_{\rm net}|/\Sigma|a_i|$'),
            Patch(facecolor='green', alpha=0.1, edgecolor='none',
                  label='Quasi-equilibrium'),
            Line2D([0], [0], color='green', ls='--', lw=1, alpha=0.5,
                   label='C=0.1'),
        ]
        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Cancellation Metric ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

        leg = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
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
        tag = f"cancellationMetric_grid_{m_tag}_{sfe_tag}_{n_tag}"

        if SAVE_PDF:
            out_pdf = FIG_DIR / f"{tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)


def plot_folder_grid(folder_path, output_dir=None):
    """
    Create grid plot from all simulations found in a folder.

    Automatically arranges simulations by:
    - Rows: increasing mCloud (top to bottom)
    - Columns: increasing SFE (left to right)

    PDF and title named after the folder.
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
        sharex=False, sharey=True,
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
                               phase_change=PHASE_CHANGE,
                               show_equilibrium_band=SHOW_EQUILIBRIUM_BAND,
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
                ax.set_ylabel(mlabel + "\n" + r"$\mathcal{C}$")
            else:
                ax.tick_params(labelleft=False)

    handles = [
        Line2D([0], [0], color='black', lw=2,
               label=r'$\mathcal{C} = |a_{\rm net}|/\Sigma|a_i|$'),
        Patch(facecolor='green', alpha=0.1, edgecolor='none',
              label='Quasi-equilibrium'),
        Line2D([0], [0], color='green', ls='--', lw=1, alpha=0.5,
               label='C=0.1'),
    ]
    handles.extend(get_marker_legend_handles())

    fig.subplots_adjust(top=0.9)
    fig.suptitle(folder_name, fontsize=14, y=1.02)

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.0)
    )
    leg.set_zorder(10)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"{folder_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY cancellation metric",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_cancellationMetric.py 1e7_sfe020_n1e4
  python paper_cancellationMetric.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_cancellationMetric.py /path/to/dictionary.jsonl
  python paper_cancellationMetric.py  # (uses grid/single config at top of file)
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
        '--no-band', action='store_true',
        help='Do not show equilibrium band shading'
    )
    parser.add_argument(
        '--log-x', action='store_true',
        help='Use log scale for x-axis (time)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Search folder recursively for all simulation .jsonl files'
    )

    args = parser.parse_args()

    if args.no_band:
        SHOW_EQUILIBRIUM_BAND = False
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
