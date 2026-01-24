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

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting pressure evolution (P_b, P_IF, P_drive)")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
ndens_list = ["1e3"]
sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"

SMOOTH_WINDOW = 5  # None or 1 disables
PHASE_CHANGE = True
SHOW_PEXT = True  # Show external pressure

# --- optional single-run view (set to None for full grid)
ONLY_M = "1e7"
ONLY_N = "1e4"
ONLY_SFE = "010"

# Comment this out for single mode, leave for grid mode
ONLY_M = ONLY_N = ONLY_SFE = None

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

# Pressure field styling
PRESSURE_FIELDS = [
    ("Pb",      r"$P_b$ (bubble)",      "blue",  "-",  1.8),
    ("P_IF",    r"$P_{\rm IF}$",        "red",   "-",  1.8),
    ("P_drive", r"$P_{\rm drive}$",     "black", "--", 2.2),
]

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

    # Pressure fields
    Pb = get_field('Pb', np.nan)
    P_IF = get_field('P_IF', np.nan)
    P_drive = get_field('P_drive', np.nan)
    press_HII_in = get_field('press_HII_in', np.nan)

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
                   show_rcloud=True, show_collapse=True, show_pext=True):
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
    ax.set_xlim(t.min(), t.max())

    # Auto y-limits with some padding
    all_pressures = np.concatenate([
        data['Pb'], data['P_IF'], data['P_drive']
    ])
    valid = all_pressures[np.isfinite(all_pressures) & (all_pressures > 0)]
    if len(valid) > 0:
        ymin, ymax = valid.min(), valid.max()
        ax.set_ylim(ymin * 0.3, ymax * 3)


def plot_from_path(data_input: str, output_dir: str = None):
    """Plot pressure evolution from a direct data path/folder."""
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
                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Pressure [code units]")
    ax.set_title(f"Pressure Evolution: {data_path.parent.name}")

    # Legend
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_pressureEvolution_{run_name}.pdf"
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
                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT)

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Pressure [code units]")
    ax.set_title(f"{run_name}")
    ax.legend(loc="upper right", framealpha=0.9)

    tag = f"pressureEvolution_{mCloud}_sfe{sfe}_n{ndens}"
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of pressure evolution."""
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
                                   phase_change=PHASE_CHANGE, show_pext=SHOW_PEXT)
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
                    ax.set_ylabel(mlabel + "\nP [code]")
                else:
                    ax.tick_params(labelleft=False)

        # Global legend
        handles = [
            Line2D([0], [0], color="blue", ls="-", lw=1.8, label=r"$P_b$ (bubble)"),
            Line2D([0], [0], color="red", ls="-", lw=1.8, label=r"$P_{\rm IF}$"),
            Line2D([0], [0], color="black", ls="--", lw=2.2, label=r"$P_{\rm drive}$"),
        ]
        if SHOW_PEXT:
            handles.append(Line2D([0], [0], color="gray", ls=":", lw=1.5,
                                  alpha=0.7, label=r"$P_{\rm ext}$"))
        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Pressure Evolution ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.02)

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
        tag = f"pressureEvolution_grid_{m_tag}_{sfe_tag}_{n_tag}"

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
        description="Plot TRINITY pressure evolution (P_b, P_IF, P_drive)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_pressureEvolution.py 1e7_sfe020_n1e4
  python paper_pressureEvolution.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_pressureEvolution.py /path/to/dictionary.jsonl
  python paper_pressureEvolution.py  # (uses grid/single config at top of file)
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Base directory for output folders (default: TRINITY_OUTPUT_DIR or "outputs")'
    )

    args = parser.parse_args()

    if args.data:
        plot_from_path(args.data, args.output_dir)
    elif (ONLY_M is not None) and (ONLY_N is not None) and (ONLY_SFE is not None):
        plot_single_run(ONLY_M, ONLY_N, ONLY_SFE)
    else:
        plot_grid()
