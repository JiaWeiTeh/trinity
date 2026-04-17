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
from scipy.integrate import cumulative_trapezoid

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens,
    parse_simulation_params,
)
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles
from src._plots.grid_template import (
    _mcloud_label,
    build_param_tag,
    mark_missing_cell,
    attach_grid_legend,
)
from src._functions.unit_conversions import CONV, INV_CONV, CGS

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
WEAVER_ANCHOR_MYR = 0.01  # anchor Weaver line to TRINITY R2 at this time

# Styling
COLOR_TRINITY   = "C0"      # blue
COLOR_WARPFIELD = "C3"     # red
COLOR_WEAVER    = "0.4"     # grey
COLOR_MOMENTUM  = "0.4"     # grey (subordinate to data curves)

SAVE_PDF = True

# Mean molecular weight for neutral gas
MU_ATOM = 1.4


# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def load_run_R2(data_path):
    """Load a single run, return dict with time, R2, phase, rcloud, isCollapse,
    and feedback fields needed for absolute analytic solutions."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    R2 = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))
    rcloud = float(output[0].get('rCloud', np.nan))
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Feedback fields for absolute analytic solutions (None if missing)
    try:
        Lmech_total = output.get('Lmech_total')
    except Exception:
        Lmech_total = None

    try:
        pdot_total = output.get('pdot_total')
    except Exception:
        pdot_total = None

    # If pdot_total missing but Lmech_total and v_mech_total available, recover it
    if pdot_total is None and Lmech_total is not None:
        try:
            v_mech_total = output.get('v_mech_total')
            if v_mech_total is not None and np.all(v_mech_total > 0):
                pdot_total = 2.0 * Lmech_total / v_mech_total
        except Exception:
            pass

    # Cloud density: try from snapshot, then from folder name.
    # Output nCore is in AU (pc⁻³); convert to cm⁻³ for analytic formulae.
    nCore = output[0].get('nCore', None)
    if nCore is not None:
        nCore = nCore * INV_CONV.ndens_au2cgs          # pc⁻³ → cm⁻³
    else:
        sim_params = parse_simulation_params(data_path.parent.name)
        if sim_params is not None:
            try:
                nCore = float(sim_params['ndens'])      # already cm⁻³
            except (KeyError, ValueError):
                nCore = None

    # Density profile exponent (0 = uniform)
    densPL_alpha = output[0].get('densPL_alpha', None)

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, isCollapse = t[order], R2[order], phase[order], isCollapse[order]
        if Lmech_total is not None:
            Lmech_total = Lmech_total[order]
        if pdot_total is not None:
            pdot_total = pdot_total[order]

    return dict(
        t=t, R2=R2, phase=phase, rcloud=rcloud, isCollapse=isCollapse,
        Lmech_total=Lmech_total, pdot_total=pdot_total,
        nCore=nCore, densPL_alpha=densPL_alpha,
    )


# ----------------------------------------------------------------
# Absolute analytic solutions
# ----------------------------------------------------------------
def compute_weaver_absolute(t_myr, Lmech_total_au, nCore_cm3, mu_atom=MU_ATOM):
    """Absolute Weaver (energy-driven) radius using cumulative energy.

    Parameters
    ----------
    t_myr : array
        Time in Myr.
    Lmech_total_au : array
        Total mechanical luminosity in AU units (same length as t_myr).
    nCore_cm3 : float
        Core number density in cm^{-3}.
    mu_atom : float
        Mean molecular weight of neutral gas.

    Returns
    -------
    R_pc : array
        Weaver radius in parsec.
    """
    # Convert to CGS
    Lw_cgs = Lmech_total_au * INV_CONV.L_au2cgs           # erg/s
    t_s = t_myr * INV_CONV.Myr2s                           # seconds

    # Cumulative energy via trapezoidal integration
    E_w_cum = np.zeros_like(t_s)
    E_w_cum[1:] = cumulative_trapezoid(Lw_cgs, t_s)

    # Ambient mass density
    rho_0 = nCore_cm3 * mu_atom * CGS.m_H                 # g/cm³

    # Prefactor: (125 / (154 π))^{1/5}
    prefactor = (125.0 / (154.0 * np.pi)) ** 0.2           # ≈ 0.7603

    R_cm = np.where(
        (t_s > 0) & (E_w_cum > 0),
        prefactor * (E_w_cum / rho_0) ** 0.2 * t_s ** 0.4,
        np.nan,
    )
    return R_cm * CONV.cm2pc                                # → pc



# ----------------------------------------------------------------
# Anchored power-law fallbacks
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


# ----------------------------------------------------------------
# Per-cell plotting
# ----------------------------------------------------------------
def _can_use_absolute(data):
    """Check whether absolute Weaver solution can be computed."""
    return (
        data.get('Lmech_total') is not None
        and data.get('nCore') is not None
        and (data.get('densPL_alpha') is None
             or data['densPL_alpha'] == 0)
    )


def plot_cell(ax, data_trinity, data_warpfield):
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

    # TRINITY R2
    ax.plot(t_T, R2_T, color=COLOR_TRINITY, lw=2.0, ls='-', zorder=4)

    # WARPFIELD R2
    if data_warpfield is not None:
        t_W  = data_warpfield['t']
        R2_W = smooth_1d(data_warpfield['R2'], SMOOTH_WINDOW, mode=SMOOTH_MODE)
        ax.plot(t_W, R2_W, color=COLOR_WARPFIELD, lw=2.0, ls='-', zorder=3)

    # Density profile exponent (for scaling exponents)
    alpha_rho = data_trinity.get('densPL_alpha') or 0

    # --- Weaver (energy-driven): absolute if possible, else anchored ---
    if _can_use_absolute(data_trinity):
        nCore = data_trinity['nCore']
        R_weaver = compute_weaver_absolute(
            t_T, data_trinity['Lmech_total'], nCore,
        )
    else:
        exp_weaver = 3.0 / (5.0 - abs(alpha_rho))
        R_weaver = compute_weaver_anchored(t_T, R2_T, exponent=exp_weaver)

    ax.plot(t_T, R_weaver, color=COLOR_WEAVER, lw=1.5, ls='--', zorder=2)

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
        ax.plot(t_T, R_mom, color=COLOR_MOMENTUM, lw=1.5, ls=':', zorder=2)

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
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.4 * ncols, 2.8 * nrows),
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
                    plot_cell(ax, data_T, data_W)
                except Exception as e:
                    print(f"  Error: {sim_name}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

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
                   label=r"Weaver (energy-driven)"),
            Line2D([0], [0], color=COLOR_MOMENTUM, lw=1.5, ls=':',
                   label=r"$R \propto t^{1/2}$ (momentum scaling)"),
        ]
        handles.extend(get_marker_legend_handles(include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD, include_rcloud_horizontal=SHOW_RCLOUD_H, include_collapse=SHOW_COLLAPSE))

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
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
  python paper_radiusComparison.py -f /path/to/runs/
  python paper_radiusComparison.py -f /path/to/runs/ -n 1e4
  python paper_radiusComparison.py -f /path/to/runs/ --mCloud 1e6 1e7

The folder should contain sibling simulation subfolders whose names end in
``_yesPHII`` (include_PHII=True, TRINITY) and ``_noPHII`` (include_PHII=False,
WARPFIELD-like). Runs are paired automatically by their base name.
        """,
    )
    parser.add_argument(
        '--folder', '-f', required=True,
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

    args = parser.parse_args()

    # Apply marker flags to module globals
    from src._plots.cli import get_marker_flags
    _marker_flags = get_marker_flags(args)
    SHOW_PHASE = _marker_flags['show_phase']
    SHOW_RCLOUD = _marker_flags['show_rcloud']
    SHOW_RCLOUD_H = _marker_flags['show_rcloud_horizontal']
    SHOW_COLLAPSE = _marker_flags['show_collapse']

    plot_comparison_grid(
        args.folder,
        output_dir=args.output_dir,
        ndens_filter=args.nCore,
        mCloud_filter=args.mCloud,
        sfe_filter=args.sfe,
    )
