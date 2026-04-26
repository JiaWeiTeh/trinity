#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radius comparison plot: TRINITY vs WARPFIELD-like vs Weaver analytical.

Takes a single output folder containing both runs side-by-side, distinguished
by the ``_yesPHII`` / ``_noPHII`` suffix that ``run.py`` appends to each
simulation folder (``include_PHII = True`` → ``_yesPHII``, ``False`` → ``_noPHII``).

For each matched pair (same base name, differing only by suffix), plots R2(t)
from both runs on the same axes, together with the Weaver R ∝ t^{3/5} power-law
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
from src._plots.grid_template import (
    _mcloud_label,
    build_param_tag,
    mark_missing_cell,
    attach_grid_legend,
)

print("...plotting radius comparison (TRINITY vs WARPFIELD vs Weaver)")

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SMOOTH_WINDOW = None       # e.g. 7; None/1 disables
SMOOTH_MODE = "edge"
SHOW_PHASE = False
SHOW_RCLOUD = False
SHOW_RCLOUD_H = False
SHOW_COLLAPSE = False
LOGLOG = False             # log-log axes for the R(t) panels
WEAVER_ANCHOR_MYR = 0.01  # anchor Weaver line to TRINITY R2 at this time

# Styling — TRINITY is the hero curve (black, thick, white halo);
# WARPFIELD is the comparison curve (faded red); analytic scalings are
# subordinate (faint grey).
COLOR_TRINITY   = "k"
COLOR_WARPFIELD = "C3"
ALPHA_WARPFIELD = 0.5
COLOR_WEAVER    = "0.4"
COLOR_MOMENTUM  = "0.4"
COLOR_SPITZER   = "0.4"
ALPHA_SCALING   = 0.5    # alpha for analytic scaling lines
LW_SCALING      = 1.0    # linewidth for analytic scaling lines

# A&A single-column width (\columnwidth ~ 88 mm) used for the 1x1 layout.
COLUMN_WIDTH_INCHES = 3.46
COLUMN_HEIGHT_INCHES = 2.8

SAVE_PDF = True

# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def load_run_R2(data_path):
    """Load a single run, return dict with time, R2, phase, rcloud, isCollapse,
    and the density-profile exponent used to set the Weaver/momentum slopes."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    R2 = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Density profile exponent (0 = uniform)
    densPL_alpha = output[0].get('densPL_alpha', None)

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, isCollapse = t[order], R2[order], phase[order], isCollapse[order]

    return dict(
        t=t, R2=R2, phase=phase, rcloud=rcloud, isCollapse=isCollapse,
        densPL_alpha=densPL_alpha,
    )


# ----------------------------------------------------------------
# Anchored power-law references
# ----------------------------------------------------------------
def compute_weaver_anchored(t, R2, t_anchor=WEAVER_ANCHOR_MYR, exponent=3.0/5.0):
    """R ∝ t^exponent anchored to R2 at t_anchor.

    Default exponent 3/5 is the energy-driven Weaver solution for uniform density.
    For non-uniform density with power-law exponent α_ρ, use
    exponent = 3 / (5 - |α_ρ|).
    """
    valid = np.isfinite(R2) & (R2 > 0) & np.isfinite(t) & (t > 0)
    if not np.any(valid):
        return np.full_like(t, np.nan)

    t_v, R2_v = t[valid], R2[valid]
    idx = np.argmin(np.abs(t_v - t_anchor))
    t_ref, R_ref = t_v[idx], R2_v[idx]

    return np.where(t > 0, R_ref * (t / t_ref) ** exponent, np.nan)


def compute_momentum_driven_anchored(t, R2, t_anchor=WEAVER_ANCHOR_MYR, exponent=0.5):
    """R ∝ t^exponent anchored to R2 at t_anchor.

    Default exponent 1/2 is the momentum-driven solution for uniform density.
    For non-uniform density with power-law exponent α_ρ, use
    exponent = 2 / (4 - |α_ρ|).
    """
    valid = np.isfinite(R2) & (R2 > 0) & np.isfinite(t) & (t > 0)
    if not np.any(valid):
        return np.full_like(t, np.nan)

    t_v, R2_v = t[valid], R2[valid]
    idx = np.argmin(np.abs(t_v - t_anchor))
    t_ref, R_ref = t_v[idx], R2_v[idx]

    return np.where(t > 0, R_ref * (t / t_ref) ** exponent, np.nan)


def compute_spitzer_anchored(t, R2, t_anchor=WEAVER_ANCHOR_MYR, exponent=4.0/7.0):
    """R ∝ t^exponent anchored to R2 at t_anchor.

    Default exponent 4/7 is the Spitzer (1978) D-type HII-region expansion
    in a uniform medium: pressure of the photoionised gas drives the shocked
    shell outward.  For a power-law density profile n ∝ r^{-|α_ρ|}, the
    generalisation is exponent = 4 / (7 - 2|α_ρ|).
    """
    valid = np.isfinite(R2) & (R2 > 0) & np.isfinite(t) & (t > 0)
    if not np.any(valid):
        return np.full_like(t, np.nan)

    t_v, R2_v = t[valid], R2[valid]
    idx = np.argmin(np.abs(t_v - t_anchor))
    t_ref, R_ref = t_v[idx], R2_v[idx]

    return np.where(t > 0, R_ref * (t / t_ref) ** exponent, np.nan)


# ----------------------------------------------------------------
# Per-cell plotting
# ----------------------------------------------------------------
def plot_cell(ax, data_trinity, data_warpfield, inline_label_trinity=False):
    """Draw TRINITY, WARPFIELD, Weaver, and momentum-driven lines on one axis."""
    t_T  = data_trinity['t']
    R2_T = smooth_1d(data_trinity['R2'], SMOOTH_WINDOW, mode=SMOOTH_MODE)

    # Phase/cloud markers from TRINITY run
    add_plot_markers(
        ax, t_T,
        phase=data_trinity['phase'] if SHOW_PHASE else None,
        R2=R2_T if SHOW_RCLOUD else None,
        rcloud=data_trinity['rcloud'] if SHOW_RCLOUD else None,
        isCollapse=data_trinity['isCollapse'],
        show_phase=SHOW_PHASE,
        show_rcloud=SHOW_RCLOUD,
        show_rcloud_horizontal=SHOW_RCLOUD_H,
        show_collapse=SHOW_COLLAPSE,
    )

    # TRINITY R2 (hero curve)
    ax.plot(t_T, R2_T, color=COLOR_TRINITY, lw=2.5, ls='-', zorder=5)

    # WARPFIELD R2 (faded)
    if data_warpfield is not None:
        t_W  = data_warpfield['t']
        R2_W = smooth_1d(data_warpfield['R2'], SMOOTH_WINDOW, mode=SMOOTH_MODE)
        ax.plot(t_W, R2_W, color=COLOR_WARPFIELD, lw=1.6, ls='-',
                alpha=ALPHA_WARPFIELD, zorder=3)

    # Density profile exponent (for scaling exponents)
    alpha_rho = data_trinity.get('densPL_alpha') or 0

    # --- Weaver: pure t^{3/(5-|α|)} power-law anchored to TRINITY at early time
    exp_weaver = 3.0 / (5.0 - abs(alpha_rho))
    R_weaver = compute_weaver_anchored(t_T, R2_T, exponent=exp_weaver)
    ax.plot(t_T, R_weaver, color=COLOR_WEAVER,
            lw=LW_SCALING, ls='--', alpha=ALPHA_SCALING, zorder=2)

    # --- Momentum-driven: diagnostic slope anchored at momentum phase ---
    # Find the start of the momentum phase from the phase array
    phase = data_trinity['phase']
    exp_mom = 2.0 / (4.0 - abs(alpha_rho))
    mom_idx = np.where(phase == 'momentum')[0]

    if len(mom_idx) > 0:
        t_mom_start = t_T[mom_idx[0]]
        R_mom = compute_momentum_driven_anchored(
            t_T, R2_T, t_anchor=t_mom_start, exponent=exp_mom,
        )
        ax.plot(t_T, R_mom, color=COLOR_MOMENTUM,
                lw=LW_SCALING, ls=':', alpha=ALPHA_SCALING, zorder=2)

    # --- Spitzer-like: D-type HII expansion R ∝ t^{4/(7-2|α|)} anchored at early time
    exp_spitzer = 4.0 / (7.0 - 2.0 * abs(alpha_rho))
    R_spitzer = compute_spitzer_anchored(t_T, R2_T, exponent=exp_spitzer)
    ax.plot(t_T, R_spitzer, color=COLOR_SPITZER,
            lw=LW_SCALING, ls='-.', alpha=ALPHA_SCALING, zorder=2)

    # Inline label on the TRINITY curve itself — used in the 1x1 layout
    # where we drop TRINITY from the legend.
    if inline_label_trinity:
        valid = np.isfinite(t_T) & np.isfinite(R2_T) & (R2_T > 0)
        if np.any(valid):
            i_last = np.flatnonzero(valid)[-1]
            ax.annotate(
                "TRINITY",
                xy=(t_T[i_last], R2_T[i_last]),
                xytext=(-4, 4), textcoords="offset points",
                ha='right', va='bottom',
                color=COLOR_TRINITY, fontweight='bold',
                zorder=6,
            )

    if LOGLOG:
        # Anchor x-axis to first positive time so log scale is well-defined.
        t_pos = t_T[t_T > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t_T.max())
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.set_xlim(t_T.min(), t_T.max())


# ----------------------------------------------------------------
# Grid builder (single-folder variant)
# ----------------------------------------------------------------
YES_SUFFIX = "_yesPHII"
NO_SUFFIX = "_noPHII"


def split_by_phii_suffix(folder):
    """Scan one folder and split simulations by ``_yesPHII`` / ``_noPHII`` suffix.

    Returns
    -------
    (sim_files_T, noPHII_by_base) where
      sim_files_T    : list of dictionary paths for the TRINITY (yesPHII) runs,
                       used to discover the mCloud × SFE grid.
      noPHII_by_base : dict mapping base-name (suffix stripped) → noPHII path,
                       used to look up each yesPHII run's WARPFIELD-like partner.
    """
    sim_files_T = []
    noPHII_by_base = {}
    for p in find_all_simulations(folder):
        name = p.parent.name
        if name.endswith(YES_SUFFIX):
            sim_files_T.append(p)
        elif name.endswith(NO_SUFFIX):
            noPHII_by_base[name[: -len(NO_SUFFIX)]] = p
        else:
            print(f"  Skipping (no _yesPHII/_noPHII suffix): {name}")

    return sim_files_T, noPHII_by_base


def plot_comparison_grid(
    folder,
    output_dir=None,
    ndens_filter=None,
    mCloud_filter=None,
    sfe_filter=None,
):
    """Create (mCloud × SFE) grid comparing TRINITY (yesPHII) vs WARPFIELD (noPHII)."""
    folder = Path(folder)

    sim_files_T, noPHII_by_base = split_by_phii_suffix(folder)
    if not sim_files_T:
        print(f"No _yesPHII simulations found in: {folder}")
        return

    ndens_to_plot = [ndens_filter] if ndens_filter else get_unique_ndens(sim_files_T)
    n_paired = sum(
        1 for p in sim_files_T
        if p.parent.name[: -len(YES_SUFFIX)] in noPHII_by_base
    )
    print(f"Found {len(sim_files_T)} yesPHII simulations "
          f"(paired with {n_paired} noPHII)")
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
        is_single = (nrows == 1 and ncols == 1)

        if is_single:
            figsize = (COLUMN_WIDTH_INCHES, COLUMN_HEIGHT_INCHES)
        else:
            figsize = (3.4 * ncols, 2.8 * nrows)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=figsize,
            sharex=False, sharey=False,
            dpi=300, squeeze=False,
        )
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                path_T = grid_T.get((mCloud, sfe))

                if path_T is None:
                    mark_missing_cell(ax, "missing")
                    continue

                # Look up matched WARPFIELD run by stripped base name
                sim_name = path_T.parent.name
                base = sim_name[: -len(YES_SUFFIX)]
                path_W = noPHII_by_base.get(base)

                try:
                    data_T = load_run_R2(path_T)
                    data_W = load_run_R2(path_W) if path_W is not None else None
                    plot_cell(ax, data_T, data_W,
                              inline_label_trinity=is_single)
                except Exception as e:
                    print(f"  Error: {sim_name}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                if is_single:
                    ax.set_ylabel("Radius [pc]")
                    ax.set_xlabel("t [Myr]")
                else:
                    if j == 0:
                        ax.set_ylabel(_mcloud_label(mCloud) + "\nRadius [pc]")
                    else:
                        ax.tick_params(labelleft=False)
                    if i == nrows - 1:
                        ax.set_xlabel("t [Myr]")

        # Legend entries — analytic scalings carry the demoted style; in
        # the 1x1 case TRINITY is shown inline on the curve, not in the
        # legend.
        handles = []
        if not is_single:
            handles.append(
                Line2D([0], [0], color=COLOR_TRINITY, lw=2.5,
                       label=r"TRINITY")
            )
        handles.extend([
            Line2D([0], [0], color=COLOR_WARPFIELD, lw=1.6,
                   alpha=ALPHA_WARPFIELD,
                   label=r"WARPFIELD (no $P_{\rm HII}$)"),
            Line2D([0], [0], color=COLOR_WEAVER, lw=LW_SCALING, ls='--',
                   alpha=ALPHA_SCALING, label=r"Pure energy (wind)"),
            Line2D([0], [0], color=COLOR_SPITZER, lw=LW_SCALING, ls='-.',
                   alpha=ALPHA_SCALING, label=r"Pure photoionised"),
            Line2D([0], [0], color=COLOR_MOMENTUM, lw=LW_SCALING, ls=':',
                   alpha=ALPHA_SCALING, label=r"Pure momentum"),
        ])
        handles.extend(get_marker_legend_handles(include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD, include_rcloud_horizontal=SHOW_RCLOUD_H, include_collapse=SHOW_COLLAPSE))

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)

        if is_single:
            # In-axes legend; no row/col labels, no suptitle.
            axes[0, 0].legend(
                handles=handles,
                loc="best",
                frameon=True, facecolor="white",
                framealpha=0.9, edgecolor="0.2",
                fontsize=8,
            )
            fig.tight_layout()
        else:
            attach_grid_legend(
                fig, handles,
                n_rows_for_layout=nrows,
                cell_height_inches=2.8,
                folder_name="", param_tag=param_tag,
                legend_ncol=4,
                suptitle=False,
            )

        # Save
        if output_dir:
            fig_dir = Path(output_dir)
        else:
            fig_dir = FIG_DIR / folder.name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"radiusComparison_{param_tag}.pdf"
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
  python paper_radiusComparison.py -F /path/to/runs/
  python paper_radiusComparison.py -F /path/to/runs/ -n 1e4
  python paper_radiusComparison.py -F /path/to/runs/ --mCloud 1e6 1e7

The folder should contain sibling simulation subfolders whose names end in
``_yesPHII`` (include_PHII=True, TRINITY) and ``_noPHII`` (include_PHII=False,
WARPFIELD-like). Runs are paired automatically by their base name.
        """,
    )
    parser.add_argument(
        '--folder', '-F', required=True,
        help='Folder containing both _yesPHII and _noPHII simulation subfolders',
    )
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by density (e.g. "1e4")')
    parser.add_argument('--mCloud', nargs='+', default=None)
    parser.add_argument('--sfe', nargs='+', default=None)
    parser.add_argument('--show-phase', action='store_true', default=False)
    parser.add_argument('--show-rcloud', action='store_true', default=False)
    parser.add_argument('--show-rcloud-horizontal', action='store_true', default=False)
    parser.add_argument('--show-collapse', action='store_true', default=False)
    parser.add_argument('--show-all-markers', action='store_true', default=False)
    parser.add_argument('--loglog', action='store_true', default=False,
                        help='Plot R(t) on log-log axes')

    args = parser.parse_args()

    # Apply marker flags to module globals
    from src._plots.cli import get_marker_flags
    _marker_flags = get_marker_flags(args)
    globals()['SHOW_PHASE'] = _marker_flags['show_phase']
    globals()['SHOW_RCLOUD'] = _marker_flags['show_rcloud']
    globals()['SHOW_RCLOUD_H'] = _marker_flags['show_rcloud_horizontal']
    globals()['SHOW_COLLAPSE'] = _marker_flags['show_collapse']
    globals()['LOGLOG'] = args.loglog

    plot_comparison_grid(
        args.folder,
        output_dir=args.output_dir,
        ndens_filter=args.nCore,
        mCloud_filter=args.mCloud,
        sfe_filter=args.sfe,
    )
