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

print("...plotting thermal regime (w_blend)")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
ndens_list = ["1e3"]
sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"

SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
PLOT_MODE = "line"  # "line" or "stacked"
USE_LOG_X = True  # Use log scale for x-axis (time)

# Colors for stacked mode
C_BUBBLE = "blue"
C_HII = "#d62728"  # red

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


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot thermal regime from a direct data path/folder."""
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
                   phase_change=PHASE_CHANGE, plot_mode=PLOT_MODE,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$w_{\rm blend}$ (HII weight)")
    ax.set_title(f"Thermal Regime: {data_path.parent.name}")

    # Legend
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
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_thermalRegime_{run_name}.pdf"
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
                   phase_change=PHASE_CHANGE, plot_mode=PLOT_MODE,
                   use_log_x=USE_LOG_X)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$w_{\rm blend}$")
    ax.set_title(f"{run_name}")

    tag = f"thermalRegime_{mCloud}_sfe{sfe}_n{ndens}"
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of thermal regime."""
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
                                   phase_change=PHASE_CHANGE, plot_mode=PLOT_MODE,
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
                    ax.set_ylabel(mlabel + "\n" + r"$w_{\rm blend}$")
                else:
                    ax.tick_params(labelleft=False)

        # Global legend
        if PLOT_MODE == "stacked":
            handles = [
                Patch(facecolor=C_BUBBLE, alpha=0.7, label=r"$(1-w)$ Bubble"),
                Patch(facecolor=C_HII, alpha=0.7, label=r"$w$ HII"),
            ]
        else:
            handles = [
                Line2D([0], [0], color='black', lw=2, label=r'$w_{\rm blend}$'),
                Patch(facecolor=C_BUBBLE, alpha=0.1, edgecolor='none',
                      label='Bubble regime (w<0.3)'),
                Patch(facecolor=C_HII, alpha=0.1, edgecolor='none',
                      label='HII regime (w>0.7)'),
            ]
        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Thermal Regime ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

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
        tag = f"thermalRegime_grid_{m_tag}_{sfe_tag}_{n_tag}"

        if SAVE_PDF:
            out_pdf = FIG_DIR / f"{tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY thermal regime (w_blend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_thermalRegime.py 1e7_sfe020_n1e4
  python paper_thermalRegime.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_thermalRegime.py /path/to/dictionary.jsonl
  python paper_thermalRegime.py  # (uses grid/single config at top of file)
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
        '--stacked', action='store_true',
        help='Use stacked area plot instead of line plot'
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

    if args.stacked:
        PLOT_MODE = "stacked"
    if args.log_x:
        USE_LOG_X = True

    if args.folder:
        from src._output.trinity_reader import find_all_simulations
        sim_files = find_all_simulations(args.folder)
        if not sim_files:
            print(f"No simulation files found in {args.folder}")
            sys.exit(1)
        print(f"Found {len(sim_files)} simulations in {args.folder}")
        for data_path in sim_files:
            plot_from_path(str(data_path), args.output_dir)
    elif args.data:
        plot_from_path(args.data, args.output_dir)
    elif (ONLY_M is not None) and (ONLY_N is not None) and (ONLY_SFE is not None):
        plot_single_run(ONLY_M, ONLY_N, ONLY_SFE)
    else:
        plot_grid()
