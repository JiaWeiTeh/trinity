#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radius comparison plot: TRINITY vs WARPFIELD-like vs Weaver analytical.

Takes two output folders:
  -T / --trinity    : folder with include_PHII = True  (full TRINITY)
  -W / --warpfield  : folder with include_PHII = False (WARPFIELD-like, no P_HII)

For each matched simulation (by subfolder name), plots R2(t) from both
runs on the same axes, together with the Weaver R ∝ t^{3/5} power-law
anchored to the TRINITY curve at early time.

Grid layout: mCloud (rows) × SFE (columns), one PDF per density.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens,
)
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._plots.grid_template import _mcloud_label, _sfe_title

print("...plotting radius comparison (TRINITY vs WARPFIELD vs Weaver)")

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SMOOTH_WINDOW = None       # e.g. 7; None/1 disables
SMOOTH_MODE = "edge"
PHASE_LINE = True
CLOUD_LINE = True
WEAVER_ANCHOR_MYR = 0.01  # anchor Weaver line to TRINITY R2 at this time

# Styling
COLOR_TRINITY  = "C0"      # blue
COLOR_WARPFIELD = "C3"     # red
COLOR_WEAVER   = "0.4"     # grey

SAVE_PDF = True


# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def load_run_R2(data_path):
    """Load a single run, return dict with time, R2, phase, rcloud, isCollapse."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    R2 = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, isCollapse = t[order], R2[order], phase[order], isCollapse[order]

    return dict(t=t, R2=R2, phase=phase, rcloud=rcloud, isCollapse=isCollapse)


# ----------------------------------------------------------------
# Weaver solution
# ----------------------------------------------------------------
def compute_weaver(t, R2, t_anchor=WEAVER_ANCHOR_MYR):
    """R ∝ t^{3/5} anchored to R2 at t_anchor.

    Finds the closest valid time to t_anchor and normalises there.
    """
    valid = np.isfinite(R2) & (R2 > 0) & np.isfinite(t) & (t > 0)
    if not np.any(valid):
        return np.full_like(t, np.nan)

    t_v, R2_v = t[valid], R2[valid]
    idx = np.argmin(np.abs(t_v - t_anchor))
    t_ref, R_ref = t_v[idx], R2_v[idx]

    R_weaver = np.where(t > 0, R_ref * (t / t_ref) ** (3.0 / 5.0), np.nan)
    return R_weaver


# ----------------------------------------------------------------
# Per-cell plotting
# ----------------------------------------------------------------
def plot_cell(ax, data_trinity, data_warpfield):
    """Draw TRINITY, WARPFIELD, and Weaver on one axis."""
    t_T  = data_trinity['t']
    R2_T = smooth_1d(data_trinity['R2'], SMOOTH_WINDOW, mode=SMOOTH_MODE)

    # Phase/cloud markers from TRINITY run
    add_plot_markers(
        ax, t_T,
        phase=data_trinity['phase'] if PHASE_LINE else None,
        R2=R2_T if CLOUD_LINE else None,
        rcloud=data_trinity['rcloud'] if CLOUD_LINE else None,
        isCollapse=data_trinity['isCollapse'],
        show_phase=PHASE_LINE,
        show_rcloud=CLOUD_LINE,
        show_collapse=True,
    )

    # TRINITY R2
    ax.plot(t_T, R2_T, color=COLOR_TRINITY, lw=2.0, ls='-', zorder=4)

    # WARPFIELD R2
    if data_warpfield is not None:
        t_W  = data_warpfield['t']
        R2_W = smooth_1d(data_warpfield['R2'], SMOOTH_WINDOW, mode=SMOOTH_MODE)
        ax.plot(t_W, R2_W, color=COLOR_WARPFIELD, lw=2.0, ls='-', zorder=3)

    # Weaver
    R_weaver = compute_weaver(t_T, R2_T)
    ax.plot(t_T, R_weaver, color=COLOR_WEAVER, lw=1.5, ls='--', zorder=2)

    ax.set_xlim(t_T.min(), t_T.max())


# ----------------------------------------------------------------
# Grid builder (two-folder variant)
# ----------------------------------------------------------------
def build_matched_grid(folder_T, folder_W):
    """Find simulations in both folders and match by subfolder name.

    Returns
    -------
    dict  mapping subfolder-name → (path_T, path_W)
    """
    def _index(folder):
        """Map subfolder name → data-file path."""
        sims = find_all_simulations(folder)
        return {p.parent.name: p for p in sims}

    idx_T = _index(folder_T)
    idx_W = _index(folder_W)

    matched = {}
    for name in idx_T:
        matched[name] = (idx_T[name], idx_W.get(name))

    # Also include WARPFIELD-only runs (rare, but be safe)
    for name in idx_W:
        if name not in matched:
            matched[name] = (None, idx_W[name])

    return matched


def plot_comparison_grid(
    folder_T, folder_W,
    output_dir=None,
    ndens_filter=None,
    mCloud_filter=None,
    sfe_filter=None,
):
    """Create (mCloud × SFE) grid comparing TRINITY vs WARPFIELD."""
    folder_T, folder_W = Path(folder_T), Path(folder_W)

    # Use TRINITY folder to discover the grid structure
    sim_files_T = find_all_simulations(folder_T)
    if not sim_files_T:
        print(f"No simulations found in TRINITY folder: {folder_T}")
        return

    matched = build_matched_grid(folder_T, folder_W)

    ndens_to_plot = [ndens_filter] if ndens_filter else get_unique_ndens(sim_files_T)
    print(f"Found {len(sim_files_T)} TRINITY simulations")
    print(f"  Densities to plot: {ndens_to_plot}")

    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")
        organized = organize_simulations_for_grid(
            sim_files_T, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
        )
        mCloud_list = organized['mCloud_list']
        sfe_list = organized['sfe_list']
        grid_T = organized['grid']          # (mCloud, sfe) → path

        if not mCloud_list or not sfe_list:
            print(f"  No grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.4 * ncols, 2.8 * nrows),
            sharex=False, sharey=False,
            dpi=300, squeeze=False,
        )
        fig.subplots_adjust(top=0.88)

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                path_T = grid_T.get((mCloud, sfe))

                if path_T is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                # Look up matched WARPFIELD run
                sim_name = path_T.parent.name
                _, path_W = matched.get(sim_name, (None, None))

                try:
                    data_T = load_run_R2(path_T)
                    data_W = load_run_R2(path_W) if path_W is not None else None
                    plot_cell(ax, data_T, data_W)
                except Exception as e:
                    print(f"  Error: {sim_name}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                if i == 0:
                    ax.set_title(_sfe_title(sfe))
                if j == 0:
                    ax.set_ylabel(_mcloud_label(mCloud) + "\nRadius [pc]")
                else:
                    ax.tick_params(labelleft=False)
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        # Legend
        handles = [
            Line2D([0], [0], color=COLOR_TRINITY, lw=2.0,
                   label=r"TRINITY ($R_2$, with $P_{\rm HII}$)"),
            Line2D([0], [0], color=COLOR_WARPFIELD, lw=2.0,
                   label=r"WARPFIELD-like ($R_2$, no $P_{\rm HII}$)"),
            Line2D([0], [0], color=COLOR_WEAVER, lw=1.5, ls='--',
                   label=r"Weaver: $R \propto t^{3/5}$"),
        ]
        handles.extend(get_marker_legend_handles())

        leg = fig.legend(
            handles=handles, loc="upper center",
            ncol=3, frameon=True, facecolor="white",
            framealpha=0.9, edgecolor="0.2",
            bbox_to_anchor=(0.5, 0.97),
        )
        leg.set_zorder(10)

        fig.suptitle(
            f"Radius comparison: TRINITY vs WARPFIELD (n{ndens})",
            fontsize=13, y=1.0,
        )

        # Save
        fig_dir = Path(output_dir) if output_dir else FIG_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"radiusComparison_n{ndens}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare TRINITY (with P_HII) vs WARPFIELD-like (no P_HII) radius evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_radiusComparison.py -T /path/to/trinity_runs/ -W /path/to/warpfield_runs/
  python paper_radiusComparison.py -T /path/to/trinity/ -W /path/to/warpfield/ -n 1e4
  python paper_radiusComparison.py -T /path/to/trinity/ -W /path/to/warpfield/ --mCloud 1e6 1e7
        """,
    )
    parser.add_argument(
        '--trinity', '-T', required=True,
        help='Folder with include_PHII=True runs (full TRINITY)',
    )
    parser.add_argument(
        '--warpfield', '-W', required=True,
        help='Folder with include_PHII=False runs (WARPFIELD-like)',
    )
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by density (e.g. "1e4")')
    parser.add_argument('--mCloud', nargs='+', default=None)
    parser.add_argument('--sfe', nargs='+', default=None)

    args = parser.parse_args()

    plot_comparison_grid(
        args.trinity, args.warpfield,
        output_dir=args.output_dir,
        ndens_filter=args.nCore,
        mCloud_filter=args.mCloud,
        sfe_filter=args.sfe,
    )
