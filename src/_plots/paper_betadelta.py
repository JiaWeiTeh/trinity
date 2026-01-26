#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 20:03:19 2025

@author: Jia Wei Teh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input

print("...plotting radius comparison")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
# mCloud_list = ["1e5","1e8"]                 # rows
# ndens_list  = ["1e4", "1e2", "1e3"]                        # one figure per ndens
ndens_list  = ["1e4"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols
# sfe_list    = ["001", "010"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 21

PHASE_CHANGE = True

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"

import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


# ---------- helpers (reuse your smoothing) ----------
def smooth_1d(y, window, mode="edge"):
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")

# ---------- load beta/delta + R2 ----------
def load_cooling_run(data_path: Path):
    """Load cooling run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    R2 = output.get('R2')
    additional_param = output.get('Pb')
    phase = np.array(output.get('current_phase', as_array=False))

    beta = output.get('cool_beta')
    delta = output.get('cool_delta')

    rcloud = float(output[0].get('rCloud', np.nan))

    # ensure increasing time (important for "first crossing" + plotting)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, beta, delta, additional_param = t[order], R2[order], phase[order], beta[order], delta[order], additional_param[order]

    return t, R2, phase, beta, delta, rcloud, additional_param


# ---------- plot on an axis (left: beta/delta, right: R2) ----------
def plot_cooling_on_ax(
    ax, t, R2, phase, beta, delta, rcloud, additional_param,
    smooth_window=None, smooth_mode="edge",
    show_phase_line=True,
    show_cloud_line=True,
    label_pad_points=4
):
    fig = ax.figure

    # optional smoothing (beta/delta often jittery)
    beta_s  = smooth_1d(beta,  smooth_window, mode=smooth_mode)
    delta_s = smooth_1d(delta, smooth_window, mode=smooth_mode)
    R2_s    = smooth_1d(R2,    smooth_window, mode=smooth_mode)

    # --- phase-change line:
    if show_phase_line:
        change_idx = np.flatnonzero(phase[1:] != phase[:-1]) + 1 
        for x in t[change_idx]: 
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)

    # --- cloud breakout line: first time R2 > rcloud (vertical dashed)
    if show_cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(R2_s > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)

            # padded label next to line
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8, color="k", alpha=0.8,
                rotation=90
            )

    # --- left axis: beta + delta
    ax.plot(t, beta_s,  lw=1.6, label=r"$\beta$")
    ax.plot(t, delta_s, lw=1.6, label=r"$\delta$")

    ax.set_xlim(t.min(), t.max())

    # --- right axis: R2
    axr = ax.twinx()
    axr.plot(t, additional_param, lw=1.4, alpha=0.8, c = 'k', label=r"$R_2$")
    axr.set_yscale('log')
    # axr.set_ylabel(r"$R_2$ [pc]")

    # keep the twin axis from hiding things
    ax.patch.set_visible(False)
    ax.set_zorder(2)

    return axr  # in case you want per-panel tweaks


# ---------- plot_from_path for CLI ----------
def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot cooling parameters from a direct data path/folder.

    Parameters
    ----------
    data_input : str
        Can be: folder name, folder path, or file path
    output_dir : str, optional
        Base directory for output folders
    """
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading data from: {data_path}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    try:
        t, R2, phase, beta, delta, rcloud, additional_param = load_cooling_run(data_path)
        axr = plot_cooling_on_ax(
            ax, t, R2, phase, beta, delta, rcloud, additional_param,
            smooth_window=SMOOTH_WINDOW,
            show_phase_line=PHASE_CHANGE,
            show_cloud_line=True,
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        plt.close(fig)
        return

    ax.set_title(f"Cooling Parameters: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"Cooling: $\beta,\delta$")
    axr.set_ylabel(r"$P_b$")

    # Legend
    handles = [
        Line2D([0],[0], lw=1.6, label=r"$\beta$"),
        Line2D([0],[0], lw=1.6, label=r"$\delta$"),
        Line2D([0],[0], lw=1.4, alpha=0.8, c='k', label=r"$P_b$"),
        Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"phase change"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    plt.tight_layout()

    # Save figures
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_betadelta_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ---------- GRID (same layout idea as your previous) ----------
def plot_grid():
    """Plot full grid of cooling parameters."""
    for ndens in ndens_list:
        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=False,
            dpi=200,
            constrained_layout=False
        )

        # leave room for suptitle + global legend
        fig.subplots_adjust(top=0.82)
        fig.suptitle(rf"Cooling parameters with $R_2$ overlay  (n={ndens})", y=0.98)

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                data_path = find_data_file(BASE_DIR, run_name)

                if data_path is None:
                    print(f"  {run_name}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"  Loading: {data_path}")
                try:
                    t, R2, phase, beta, delta, rcloud, additional_param = load_cooling_run(data_path)
                    plot_cooling_on_ax(
                        ax, t, R2, phase, beta, delta, rcloud, additional_param,
                        smooth_window=7,      # set None/1 to disable
                        show_phase_line=True,
                        show_cloud_line=True,
                    )
                except Exception as e:
                    ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    print(f"Error in {run_name}: {e}")
                    continue

                # column titles
                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                # row labels (leftmost) - handle non-power-of-10 masses
                if j == 0:
                    mval = float(mCloud)
                    mexp = int(np.floor(np.log10(mval)))
                    mcoeff = mval / (10 ** mexp)
                    mcoeff = round(mcoeff)
                    if mcoeff == 10:
                        mcoeff = 1
                        mexp += 1
                    if mcoeff == 1:
                        mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}M_\odot$"
                    else:
                        mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}M_\odot$"
                    ax.set_ylabel(mlabel + "\n" + r"Cooling: $\beta,\delta$")

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        # global legend (beta, delta, R2, plus line meanings)
        handles = [
            Line2D([0],[0], lw=1.6, label=r"$\beta$"),
            Line2D([0],[0], lw=1.6, label=r"$\delta$"),
            Line2D([0],[0], lw=1.4, alpha=0.8, label=r"$R_2$"),
            Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
            Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"transition$\to$momentum"),
        ]
        leg = fig.legend(
            handles=handles, loc="upper center", ncol=3,
            frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
            bbox_to_anchor=(0.5, 0.91), bbox_transform=fig.transFigure
        )
        leg.set_zorder(10)

        if SAVE_PDF:
            fig.savefig(FIG_DIR / f"paper_betadelta_n{ndens}.pdf", bbox_inches='tight')
            print(f"Saved: {FIG_DIR / f'paper_betadelta_n{ndens}.pdf'}")
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
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=False,
        dpi=200,
        squeeze=False,
        constrained_layout=False
    )

    fig.subplots_adjust(top=0.82)

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
                t, R2, phase, beta, delta, rcloud, additional_param = load_cooling_run(data_path)
                plot_cooling_on_ax(
                    ax, t, R2, phase, beta, delta, rcloud, additional_param,
                    smooth_window=7,
                    show_phase_line=True,
                    show_cloud_line=True,
                )
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

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
                    mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}M_\odot$"
                else:
                    mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}M_\odot$"
                ax.set_ylabel(mlabel + "\n" + r"Cooling: $\beta,\delta$")

            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    handles = [
        Line2D([0],[0], lw=1.6, label=r"$\beta$"),
        Line2D([0],[0], lw=1.6, label=r"$\delta$"),
        Line2D([0],[0], lw=1.4, alpha=0.8, label=r"$R_2$"),
        Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"transition$\to$momentum"),
    ]
    leg = fig.legend(
        handles=handles, loc="upper center", ncol=3,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        bbox_to_anchor=(0.5, 0.91), bbox_transform=fig.transFigure
    )
    leg.set_zorder(10)

    fig.suptitle(folder_name, fontsize=14, y=0.98)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    ndens = organized['ndens']
    ndens_tag = f"n{ndens}" if ndens else "nMixed"
    out_pdf = fig_dir / f"{folder_name}_{ndens_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY cooling parameters (beta, delta)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_betadelta.py 1e7_sfe020_n1e4
  python paper_betadelta.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_betadelta.py /path/to/dictionary.jsonl

  # Folder-based grid (auto-discovers simulations)
  python paper_betadelta.py --folder /path/to/my_experiment/
  python paper_betadelta.py -F /path/to/simulations/

  # Uses config at top of file
  python paper_betadelta.py
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
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Search folder recursively for simulations and create grid plot. '
             'Auto-organizes by mCloud (rows) and SFE (columns). '
             'Saves as {folder}_{ndens}.pdf'
    )

    args = parser.parse_args()

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir)
    elif args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    else:
        # Config mode: plot grid
        plot_grid()
