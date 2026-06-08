#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radius comparison plot: TRINITY vs WARPFIELD-like vs analytic scaling laws.

Takes a single output folder containing both runs side-by-side, distinguished
by the ``_yesPHII`` / ``_noPHII`` suffix that ``run.py`` appends to each
simulation folder (``include_PHII = True`` → ``_yesPHII``, ``False`` → ``_noPHII``).

For each matched pair (same base name, differing only by suffix), plots R2(t)
from both runs on the same axes, together with three pure-driver scaling lines:
energy-driven (Weaver), photoionised D-type (Spitzer) and momentum-driven.
All three are anchored to the TRINITY curve at the midpoint of its energy
phase, so the comparison is internally consistent.

Grid layout: mCloud (rows) × SFE (columns), one PDF per density.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from paper._lib.plot_base import FIG_DIR, smooth_1d
from trinity._output.trinity_reader import (
    load_output,
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens,
)
from paper._lib.plot_markers import add_plot_markers, get_marker_legend_handles
from paper._lib.grid_template import (
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
LOG_AXIS = "none"          # one of {"none", "x", "y", "both"} — log scale for R(t) panels

# Fallback anchor time used only if the TRINITY run has no 'energy' phase
# snapshots (e.g. the run terminated before any energy-phase output).
ANCHOR_FALLBACK_MYR = 0.01

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

# Column width matches paper_teaser (4.0"), so the rendered fonts/ticks
# scale identically when both figures are placed at A&A \columnwidth.
COLUMN_WIDTH_INCHES = 4.0
COLUMN_HEIGHT_INCHES = 2.8

SAVE_PDF = True

# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def load_run_R2(data_path):
    """Load a single run, return dict with time, R2, phase, rcloud, isCollapse,
    and the density-profile exponent used to set the analytic-scaling slopes."""
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
def energy_phase_midpoint(t, phase, fallback=ANCHOR_FALLBACK_MYR):
    """Return ~half the duration of the TRINITY 'energy' phase.

    All three scaling relations (Weaver, Spitzer, momentum) are anchored
    at this single early time so the comparison is internally consistent
    and lies inside the regime where the energy-driven derivation is valid.
    Falls back to ``fallback`` if no 'energy' snapshots exist.
    """
    energy_idx = np.where(np.asarray(phase) == 'energy')[0]
    if len(energy_idx) == 0:
        return fallback
    t_start = t[energy_idx[0]]
    t_end = t[energy_idx[-1]]
    return 0.5 * (t_start + t_end)


def compute_anchored_power_law(t, R2, t_anchor, exponent):
    """R ∝ t^exponent anchored to R2 at the snapshot closest to t_anchor."""
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
def plot_cell(ax, data_trinity, data_warpfield):
    """Draw TRINITY, WARPFIELD, Weaver, Spitzer, and momentum-driven lines on one axis."""
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

    # All three scalings share a single early-time anchor: half the duration
    # of the TRINITY energy phase.  This keeps the reference lines internally
    # consistent and pins them inside the regime where the energy-driven
    # derivation is valid (rather than projecting forward from a point that
    # has already left the wind-bubble phase).
    t_anchor = energy_phase_midpoint(t_T, data_trinity['phase'])

    # --- Weaver (energy-driven wind): R ∝ t^{3/(5-|α|)}
    exp_weaver = 3.0 / (5.0 - abs(alpha_rho))
    R_weaver = compute_anchored_power_law(t_T, R2_T, t_anchor, exp_weaver)
    ax.plot(t_T, R_weaver, color=COLOR_WEAVER,
            lw=LW_SCALING, ls='--', alpha=ALPHA_SCALING, zorder=2)

    # --- Spitzer (D-type HII expansion): R ∝ t^{4/(7-2|α|)}
    exp_spitzer = 4.0 / (7.0 - 2.0 * abs(alpha_rho))
    R_spitzer = compute_anchored_power_law(t_T, R2_T, t_anchor, exp_spitzer)
    ax.plot(t_T, R_spitzer, color=COLOR_SPITZER,
            lw=LW_SCALING, ls='-.', alpha=ALPHA_SCALING, zorder=2)

    # --- Momentum-driven: R ∝ t^{2/(4-|α|)}
    exp_mom = 2.0 / (4.0 - abs(alpha_rho))
    R_mom = compute_anchored_power_law(t_T, R2_T, t_anchor, exp_mom)
    ax.plot(t_T, R_mom, color=COLOR_MOMENTUM,
            lw=LW_SCALING, ls=':', alpha=ALPHA_SCALING, zorder=2)

    if LOG_AXIS in ("x", "both"):
        # Anchor x-axis to first positive time so log scale is well-defined.
        t_pos = t_T[t_T > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t_T.max())
        ax.set_xscale('log')
    else:
        ax.set_xlim(t_T.min(), t_T.max())

    if LOG_AXIS in ("y", "both"):
        ax.set_yscale('log')


# ----------------------------------------------------------------
# Grid builder (single-folder variant)
# ----------------------------------------------------------------
YES_SUFFIX = "_yesPHII"
NO_SUFFIX = "_noPHII"

# Sentinel for cells whose load raised; rendered with mark_missing_cell("error").
_CELL_ERROR = "error"


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


def _build_cells_for_ndens(sim_files_T, noPHII_by_base, ndens,
                           mCloud_filter=None, sfe_filter=None):
    """Load every cell of one ndens-grid from a folder.

    Returns ``(mCloud_list, sfe_list, cells)`` where ``cells`` is a dict
    keyed by ``(mCloud, sfe)`` with values:

      - ``{"yes": data_T, "no": data_W}`` for successful loads,
      - ``None`` for cells that aren't in the discovered grid,
      - the string ``"error"`` for cells whose load raised.

    Each ``data_*`` dict is what ``load_run_R2`` returns; ``data_W`` may
    be ``None`` if the cell has no _noPHII partner.
    """
    organized = organize_simulations_for_grid(
        sim_files_T, ndens_filter=ndens,
        mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
    )
    mCloud_list = organized['mCloud_list']
    sfe_list = organized['sfe_list']
    grid_T = organized['grid']

    cells = {}
    for mCloud in mCloud_list:
        for sfe in sfe_list:
            path_T = grid_T.get((mCloud, sfe))
            if path_T is None:
                cells[(mCloud, sfe)] = None
                continue
            sim_name = path_T.parent.name
            base = sim_name[: -len(YES_SUFFIX)]
            path_W = noPHII_by_base.get(base)
            try:
                data_T = load_run_R2(path_T)
                data_W = load_run_R2(path_W) if path_W is not None else None
                cells[(mCloud, sfe)] = {"yes": data_T, "no": data_W}
            except Exception as e:
                print(f"  Error: {sim_name}: {e}")
                cells[(mCloud, sfe)] = _CELL_ERROR
    return mCloud_list, sfe_list, cells


def _draw_grid_for_ndens(folder_name, ndens, mCloud_list, sfe_list, cells,
                         output_dir=None):
    """Render+save one (mCloud × SFE) grid for a single ndens."""
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
        squeeze=False,
    )
    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            ax = axes[i, j]
            entry = cells.get((mCloud, sfe))

            if entry is None:
                mark_missing_cell(ax, "missing")
                continue
            if entry == _CELL_ERROR:
                mark_missing_cell(ax, "error")
                continue

            plot_cell(ax, entry["yes"], entry["no"])

            if is_single:
                ax.set_ylabel(r"$R_{\rm b}$ [pc]")
                ax.set_xlabel(r"$t$ [Myr]")
            else:
                if j == 0:
                    ax.set_ylabel(_mcloud_label(mCloud) + "\n" + r"$R_{\rm b}$ [pc]")
                else:
                    ax.tick_params(labelleft=False)
                if i == nrows - 1:
                    ax.set_xlabel(r"$t$ [Myr]")

    handles = [
        Line2D([0], [0], color=COLOR_TRINITY, lw=2.5,
               label=r"TRINITY"),
        Line2D([0], [0], color=COLOR_WARPFIELD, lw=1.6,
               alpha=ALPHA_WARPFIELD,
               label=r"WARPFIELD (no $P_{\rm HII}$)"),
        Line2D([0], [0], color=COLOR_WEAVER, lw=LW_SCALING, ls='--',
               alpha=ALPHA_SCALING, label=r"pure energy (wind)"),
        Line2D([0], [0], color=COLOR_SPITZER, lw=LW_SCALING, ls='-.',
               alpha=ALPHA_SCALING, label=r"pure photoionised"),
        Line2D([0], [0], color=COLOR_MOMENTUM, lw=LW_SCALING, ls=':',
               alpha=ALPHA_SCALING, label=r"pure momentum"),
    ]
    handles.extend(get_marker_legend_handles(
        include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD,
        include_rcloud_horizontal=SHOW_RCLOUD_H,
        include_collapse=SHOW_COLLAPSE,
    ))

    param_tag = build_param_tag(mCloud_list, sfe_list, ndens)

    if is_single:
        ax_single = axes[0, 0]
        y_lo, y_hi = ax_single.get_ylim()
        if ax_single.get_yscale() == "log":
            ax_single.set_ylim(y_lo, y_hi * 3.0)
        else:
            ax_single.set_ylim(y_lo, y_hi * 1.4)
        ax_single.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.04, 0.99),
            frameon=False,
            fontsize=10,
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

    if output_dir:
        fig_dir = Path(output_dir)
    else:
        fig_dir = FIG_DIR / folder_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"radiusComparison_{param_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_pdf}")

    plt.close(fig)


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
        mCloud_list, sfe_list, cells = _build_cells_for_ndens(
            sim_files_T, noPHII_by_base, ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
        )

        if not mCloud_list or not sfe_list:
            print(f"  No grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        _draw_grid_for_ndens(
            folder.name, ndens, mCloud_list, sfe_list, cells,
            output_dir=output_dir,
        )


# ----------------------------------------------------------------
# .npz bundle: read, write, and plot
# ----------------------------------------------------------------
# Bundle layout (one ndens per file)
#   ndens                : scalar string (e.g. "1e4")
#   mCloud_list          : U32 array, length nrows
#   sfe_list             : U32 array, length ncols
#   cell_status          : U16 array, length nrows*ncols, values
#                          {"ok", "missing", "error"}, row-major
#   For each "ok" cell at flat index ``k = i*ncols + j``:
#     cell{k}_yes_t              : float array
#     cell{k}_yes_R2             : float array
#     cell{k}_yes_phase          : U16 array
#     cell{k}_yes_isCollapse     : bool array
#     cell{k}_yes_rcloud         : float scalar
#     cell{k}_yes_densPL_alpha   : float scalar
#     cell{k}_no_t               : float array (omitted if no _noPHII partner)
#     cell{k}_no_R2              : float array (omitted if no _noPHII partner)
def _flat_idx(i, j, ncols):
    return i * ncols + j


def _write_grid_npz(out_path, ndens, mCloud_list, sfe_list, cells):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(sfe_list)

    payload = dict(
        ndens=str(ndens),
        mCloud_list=np.array(mCloud_list, dtype="U32"),
        sfe_list=np.array(sfe_list, dtype="U32"),
    )

    status = []
    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            k = _flat_idx(i, j, ncols)
            entry = cells.get((mCloud, sfe))
            if entry is None:
                status.append("missing")
                continue
            if entry == _CELL_ERROR:
                status.append("error")
                continue
            status.append("ok")

            yes = entry["yes"]
            payload[f"cell{k}_yes_t"]            = np.asarray(yes["t"], dtype=float)
            payload[f"cell{k}_yes_R2"]           = np.asarray(yes["R2"], dtype=float)
            payload[f"cell{k}_yes_phase"]        = np.asarray(yes["phase"], dtype="U32")
            payload[f"cell{k}_yes_isCollapse"]   = np.asarray(yes["isCollapse"], dtype=bool)
            payload[f"cell{k}_yes_rcloud"]       = float(yes["rcloud"])
            payload[f"cell{k}_yes_densPL_alpha"] = float(yes.get("densPL_alpha") or 0)

            no = entry.get("no")
            if no is not None:
                payload[f"cell{k}_no_t"]  = np.asarray(no["t"], dtype=float)
                payload[f"cell{k}_no_R2"] = np.asarray(no["R2"], dtype=float)

    payload["cell_status"] = np.array(status, dtype="U16")
    np.savez(out_path, **payload)
    print(f"Exported: {out_path}")
    return out_path


def _build_cells_from_npz(path):
    """Load (ndens, mCloud_list, sfe_list, cells) from a bundle written by
    :func:`export_radius_comparison_npz`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        ndens = str(z["ndens"])
        mCloud_list = [str(s) for s in z["mCloud_list"]]
        sfe_list = [str(s) for s in z["sfe_list"]]
        cell_status = [str(s) for s in z["cell_status"]]
        ncols = len(sfe_list)

        cells = {}
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                k = _flat_idx(i, j, ncols)
                status = cell_status[k]
                if status == "missing":
                    cells[(mCloud, sfe)] = None
                    continue
                if status == "error":
                    cells[(mCloud, sfe)] = _CELL_ERROR
                    continue

                yes = dict(
                    t=z[f"cell{k}_yes_t"].astype(float),
                    R2=z[f"cell{k}_yes_R2"].astype(float),
                    phase=np.asarray(z[f"cell{k}_yes_phase"]),
                    isCollapse=z[f"cell{k}_yes_isCollapse"].astype(bool),
                    rcloud=float(z[f"cell{k}_yes_rcloud"]),
                    densPL_alpha=float(z[f"cell{k}_yes_densPL_alpha"]),
                )
                no = None
                if f"cell{k}_no_t" in z.files:
                    no = dict(
                        t=z[f"cell{k}_no_t"].astype(float),
                        R2=z[f"cell{k}_no_R2"].astype(float),
                    )
                cells[(mCloud, sfe)] = {"yes": yes, "no": no}
    return ndens, mCloud_list, sfe_list, cells


def export_radius_comparison_npz(
    folder,
    out_path,
    ndens_filter=None,
    mCloud_filter=None,
    sfe_filter=None,
):
    """Reduce a TRINITY run folder to one ``.npz`` per ndens.

    The bundle holds only what ``plot_cell`` consumes: per-cell time / R2 /
    phase / rcloud / isCollapse / densPL_alpha for the yesPHII run, plus
    t / R2 for any noPHII partner. The original run folders can then be
    discarded.

    If more than one ndens needs to be exported, the output filename is
    suffixed with ``_n<ndens>`` so the bundles stay distinguishable.
    """
    folder = Path(folder)
    out_path = Path(out_path)

    sim_files_T, noPHII_by_base = split_by_phii_suffix(folder)
    if not sim_files_T:
        print(f"No _yesPHII simulations found in: {folder}")
        return []

    ndens_to_export = [ndens_filter] if ndens_filter else get_unique_ndens(sim_files_T)
    multi = len(ndens_to_export) > 1

    written = []
    for ndens in ndens_to_export:
        mCloud_list, sfe_list, cells = _build_cells_for_ndens(
            sim_files_T, noPHII_by_base, ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
        )
        if not mCloud_list or not sfe_list:
            print(f"  No grid for n={ndens}, skipping export")
            continue

        this_path = (
            out_path.with_name(f"{out_path.stem}_n{ndens}{out_path.suffix}")
            if multi else out_path
        )
        _write_grid_npz(this_path, ndens, mCloud_list, sfe_list, cells)
        written.append(this_path)
    return written


def plot_comparison_from_npz(npz_path, output_dir=None):
    """Reproduce the grid figure straight from a published ``.npz`` bundle."""
    npz_path = Path(npz_path)
    ndens, mCloud_list, sfe_list, cells = _build_cells_from_npz(npz_path)
    if not mCloud_list or not sfe_list:
        print(f"Empty grid in {npz_path}")
        return
    print(f"Loaded bundle {npz_path.name}: ndens={ndens}, "
          f"mCloud={mCloud_list}, SFE={sfe_list}")
    _draw_grid_for_ndens(
        npz_path.stem, ndens, mCloud_list, sfe_list, cells,
        output_dir=output_dir,
    )


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

  # Collapse a folder into a self-contained .npz bundle for paper-data:
  python paper_radiusComparison.py -F /path/to/runs/ \\
      --export paper/methods/data/radiusComparison.npz

  # Reproduce the figure from a published bundle (no run folder needed):
  python paper_radiusComparison.py --from-npz paper/methods/data/radiusComparison.npz

The folder should contain sibling simulation subfolders whose names end in
``_yesPHII`` (include_PHII=True, TRINITY) and ``_noPHII`` (include_PHII=False,
WARPFIELD-like). Runs are paired automatically by their base name.
        """,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--folder', '-F',
        help='Folder containing both _yesPHII and _noPHII simulation subfolders',
    )
    source.add_argument(
        '--from-npz',
        help='Reproduce the figure from a .npz bundle written by --export',
    )
    parser.add_argument('--output-dir', '-o', default=None)
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by density (e.g. "1e4")')
    parser.add_argument('--mCloud', nargs='+', default=None)
    parser.add_argument('--sfe', nargs='+', default=None)
    parser.add_argument('--export', default=None,
                        help='Export the source folder to this .npz bundle '
                             'and exit (no plot). With multiple ndens, the '
                             'filename is suffixed with _n<ndens> per bundle. '
                             'Recommended location: paper/methods/data/.')
    parser.add_argument('--show-phase', action='store_true', default=False)
    parser.add_argument('--show-rcloud', action='store_true', default=False)
    parser.add_argument('--show-rcloud-horizontal', action='store_true', default=False)
    parser.add_argument('--show-collapse', action='store_true', default=False)
    parser.add_argument('--show-all-markers', action='store_true', default=False)
    parser.add_argument('--log-axis', choices=['x', 'y', 'both', 'none'],
                        default='none',
                        help='Set log scale on the chosen axis (default: none)')

    args = parser.parse_args()

    # Apply marker flags to module globals
    from paper._lib.cli import get_marker_flags
    _marker_flags = get_marker_flags(args)
    globals()['SHOW_PHASE'] = _marker_flags['show_phase']
    globals()['SHOW_RCLOUD'] = _marker_flags['show_rcloud']
    globals()['SHOW_RCLOUD_H'] = _marker_flags['show_rcloud_horizontal']
    globals()['SHOW_COLLAPSE'] = _marker_flags['show_collapse']
    globals()['LOG_AXIS'] = args.log_axis

    if args.export:
        if args.folder is None:
            parser.error("--export requires --folder (cannot re-export a bundle)")
        export_radius_comparison_npz(
            args.folder,
            args.export,
            ndens_filter=args.nCore,
            mCloud_filter=args.mCloud,
            sfe_filter=args.sfe,
        )
    elif args.from_npz:
        plot_comparison_from_npz(args.from_npz, output_dir=args.output_dir)
    else:
        plot_comparison_grid(
            args.folder,
            output_dir=args.output_dir,
            ndens_filter=args.nCore,
            mCloud_filter=args.mCloud,
            sfe_filter=args.sfe,
        )
