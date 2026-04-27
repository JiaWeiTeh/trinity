#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity-radius phase plot: v_2 vs R_2 with rCloud "cliff" annotation.

Designed to verify the hypothesis that LSODA failures in the
``trinity_fiducial_yesno`` sweep correlate with the shell crossing
``R_2 = R_cloud`` at a high expansion velocity.  The "cliff" is the
sharp drop in ambient density at ``R_cloud``; if the shell arrives at
that surface fast (|v_2| above some threshold around 10 pc/Myr), the
solver tends to fail; if it arrives slowly, the crossing is benign.

Per cell we plot the trajectory in (R_2, |v_2|) space:

- x-axis: R_2 in pc (log)
- y-axis: |v_2| in pc/Myr (log)
- vertical band at R_cloud: the cliff
- horizontal reference at the failure-velocity threshold
- start dot (open) and endpoint marker (filled) — the marker shape
  tells you whether the run finished cleanly or was terminated with
  an error code (read from ``simulationEnd.txt``)
- two trajectories per cell when both ``_yesPHII`` and ``_noPHII``
  variants are present (the standard layout produced by
  ``include_PHII = [True, False]`` in a sweep)

Layout: the usual mCloud (rows) × SFE (cols) grid, one PDF per density.

Author: TRINITY Team
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR, smooth_1d
from src._output.trinity_reader import (
    load_output,
    resolve_data_input,
    find_all_simulations,
    organize_simulations_for_grid,
    get_unique_ndens,
)
from src._output.simulation_end import read_simulation_end
from src._functions.unit_conversions import INV_CONV
from src._plots.grid_template import (
    _mcloud_label_short,
    _sfe_title,
    build_param_tag,
    mark_missing_cell,
    attach_grid_legend,
)

print("...plotting v_2 vs R_2 phase trajectory")

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
SMOOTH_WINDOW = None      # None or 1 disables
SMOOTH_MODE = "edge"

# Marker visibility flags — toggled by the standard CLI flags
# (--show-phase, --show-rcloud, --show-collapse, --show-all-markers)
# via marker_pre_dispatch(globals()) in the dispatcher below.
#
# Only SHOW_PHASE is meaningful for this script: it controls the
# energy->implicit and transition->momentum boundary markers along
# the trajectory. (The implicit->transition handoff is *always*
# shown because it carries the LSODA-failure signal that's the
# whole point of this diagnostic plot.)
#
# SHOW_RCLOUD and SHOW_COLLAPSE are accepted but no-ops here:
# - the rCloud cliff is *always* drawn (it's the headline annotation),
# - collapse already manifests in the trajectory itself (dotted
#   segment when v_2 < 0).
SHOW_PHASE    = False
SHOW_RCLOUD   = False     # no-op (rCloud band always drawn)
SHOW_COLLAPSE = False     # no-op (collapse shown via dotted line)

# Failure-velocity threshold (pc/Myr).  ~10 pc/Myr is the empirical
# divider between clean and failed crossings noted in the audit.
V_FAIL_THRESHOLD = 10.0   # pc/Myr

# rCloud band half-width (fraction of rCloud) — visualizes the cliff.
RCLOUD_BAND_FRAC = 0.05

# Convert pc/Myr → km/s for the secondary y-axis label.
V_AU2KMS = INV_CONV.v_au2kms

# Suffixes appended by run.py when include_PHII = [True, False].
YES_SUFFIX = "_yesPHII"
NO_SUFFIX = "_noPHII"

# Per-variant styling.
STYLE_YES = dict(color="#1f77b4", lw=1.6, ls="-",  alpha=0.95,
                 label=r"with $P_{\rm HII}$")
STYLE_NO  = dict(color="#d62728", lw=1.6, ls="--", alpha=0.95,
                 label=r"without $P_{\rm HII}$")

# Endpoint marker: shape tells success vs failure.
END_MARKER_OK   = "o"   # filled circle
END_MARKER_FAIL = "X"   # heavy X
END_MARKER_SIZE = 7

# Implicit->transition handoff marker: color tells whether the implicit
# phase exited via the physical cooling_balance condition (clean) or via
# an LSODA istate failure (truncated). Same shape, two colors so the
# trajectory's eye-line stays uncluttered.
HANDOFF_MARKER       = "D"            # filled diamond
HANDOFF_FACE_OK      = "#f0e442"      # yellow: cooling_balance exit (Wong palette, colorblind-safe vs orange)
HANDOFF_FACE_FAIL    = "#ff7f0e"      # orange: LSODA istate failure
HANDOFF_MARKER_SIZE  = 5.5

# Other phase-boundary markers — only drawn when SHOW_PHASE is True.
# Subdued (white-faced) so they don't compete with the LSODA handoff.
PHASE_E2I_MARKER     = "^"            # energy -> implicit (triangle up)
PHASE_T2M_MARKER     = "v"            # transition -> momentum (triangle down)
PHASE_MARKER_SIZE    = 4.5

SAVE_PDF = True


# ----------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------
def _check_implicit_lsoda_failure(data_path: Path) -> bool:
    """Detect an LSODA ``Unexpected istate`` failure inside the implicit phase.

    Scans ``trinity.log`` (sibling of ``data_path``) for the warning
    emitted by ``run_energy_implicit_phase_modified`` when ``solve_ivp``
    bails. Implicit-phase only — transition-phase solver hiccups are
    intentionally excluded so the marker stays specific to the rCloud
    cliff hypothesis.
    """
    log_path = data_path.parent / "trinity.log"
    if not log_path.exists():
        return False
    try:
        with open(log_path, "r") as fh:
            for line in fh:
                if ("Solver did not succeed" in line
                        and "phase1b_energy_implicit" in line):
                    return True
    except Exception:
        return False
    return False


def load_run_v2R2(data_path: Path) -> dict:
    """Load t, R_2, v_2, rCloud and the simulation-end status.

    Returns NaN-padded arrays sorted by time.  ``end_ok`` is True iff a
    ``simulationEnd.txt`` exit code in [0, 9] is found; otherwise False
    (treated as a failure / numerical termination).  When the file is
    absent we fall back to the in-snapshot ``SimulationEndReason`` field.

    Also reports the implicit->transition handoff so the cliff plot can
    annotate where each run handed off and whether LSODA succeeded:

    - ``handoff_idx``       : first snapshot index with ``current_phase ==
                              'transition'``, or ``None`` if not reached.
    - ``lsoda_failed``      : True if the implicit-phase LSODA bailed
                              (read from ``trinity.log``).
    """
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t  = output.get("t_now")
    R2 = output.get("R2")
    v2 = output.get("v2")  # pc/Myr
    phase = output.get("current_phase", as_array=False)
    phase_arr = np.asarray(phase) if phase is not None else np.array([])
    rcloud = float(output[0].get("rCloud", np.nan))

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, v2 = t[order], R2[order], v2[order]
        if phase_arr.size:
            phase_arr = phase_arr[order]

    # Phase-boundary indices: first sample tagged with each downstream phase.
    # 'implicit' first appearance => energy->implicit boundary
    # 'transition' first appearance => implicit->transition handoff
    # 'momentum' first appearance => transition->momentum boundary
    e2i_idx = None
    handoff_idx = None
    t2m_idx = None
    if phase_arr.size:
        impl_hits  = np.flatnonzero(phase_arr == "implicit")
        trans_hits = np.flatnonzero(phase_arr == "transition")
        mom_hits   = np.flatnonzero(phase_arr == "momentum")
        if impl_hits.size:
            e2i_idx = int(impl_hits[0])
        if trans_hits.size:
            handoff_idx = int(trans_hits[0])
        if mom_hits.size:
            t2m_idx = int(mom_hits[0])

    lsoda_failed = _check_implicit_lsoda_failure(data_path)

    # Status: prefer simulationEnd.txt (canonical), else last snapshot reason.
    end_info = read_simulation_end(str(data_path.parent))
    end_ok = None
    end_reason = None
    if end_info is not None and end_info.get("exit_code") is not None:
        end_ok = 0 <= int(end_info["exit_code"]) <= 9
        end_reason = end_info.get("reason")
    else:
        last_reason = output[-1].get("SimulationEndReason", None)
        end_reason = str(last_reason) if last_reason is not None else None
        if end_reason is not None:
            ok_keywords = ("max_time", "max_radius", "dissolved",
                           "complete", "success")
            end_ok = any(k in end_reason.lower() for k in ok_keywords)

    return dict(
        t=np.asarray(t, dtype=float),
        R2=np.asarray(R2, dtype=float),
        v2=np.asarray(v2, dtype=float),
        rcloud=rcloud,
        end_ok=end_ok,
        end_reason=end_reason,
        e2i_idx=e2i_idx,
        handoff_idx=handoff_idx,
        t2m_idx=t2m_idx,
        lsoda_failed=lsoda_failed,
    )


# ----------------------------------------------------------------
# Per-trajectory drawing
# ----------------------------------------------------------------
def _plot_one_trajectory(ax, data, style, *, smooth_window=None,
                         smooth_mode="edge"):
    """Draw |v_2| vs R_2 for a single run, plus start/end markers."""
    R2 = smooth_1d(data["R2"], smooth_window, mode=smooth_mode)
    v2 = smooth_1d(data["v2"], smooth_window, mode=smooth_mode)

    valid = np.isfinite(R2) & np.isfinite(v2) & (R2 > 0)
    if not np.any(valid):
        return

    R2v = R2[valid]
    v2v = v2[valid]

    # Plot |v_2| but use a dashed segment where v_2 < 0 (recollapse)
    # so the trajectory stays continuous on log-y.
    floor = 1e-3  # pc/Myr — keep log scale finite if v_2 hits 0
    mag = np.maximum(np.abs(v2v), floor)
    sgn = np.sign(v2v)
    sgn[sgn == 0] = 1
    cuts = np.flatnonzero(sgn[1:] != sgn[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(R2v)]
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        ls = style.get("ls", "-") if sgn[a] > 0 else ":"
        ax.plot(R2v[a:b], mag[a:b],
                color=style["color"], lw=style["lw"],
                ls=ls, alpha=style.get("alpha", 0.95),
                solid_capstyle="round", zorder=3)

    # Start marker (open) — first valid sample.
    ax.plot(R2v[0], mag[0],
            marker="o", markerfacecolor="white",
            markeredgecolor=style["color"], markersize=4.5,
            mew=1.2, zorder=4)

    # Phase-boundary markers placed on the trajectory.
    # The handoff (implicit->transition) is always shown because it
    # carries the LSODA-failure signal. The other two boundaries
    # (energy->implicit, transition->momentum) are gated by SHOW_PHASE.
    valid_indices = np.flatnonzero(valid)

    def _plot_at_orig_idx(orig_idx, marker, face, size, mew, zorder):
        """Plot a marker at the snapshot whose original index is orig_idx."""
        if orig_idx is None or len(valid_indices) == 0:
            return
        pos = np.searchsorted(valid_indices, orig_idx, side="left")
        if pos < len(valid_indices):
            ax.plot(R2v[pos], mag[pos],
                    marker=marker, markerfacecolor=face,
                    markeredgecolor="black",
                    markersize=size, mew=mew, zorder=zorder)

    # Implicit -> transition handoff (always-on, LSODA-colored)
    ho_face = (HANDOFF_FACE_FAIL if data.get("lsoda_failed")
               else HANDOFF_FACE_OK)
    _plot_at_orig_idx(data.get("handoff_idx"), HANDOFF_MARKER,
                      ho_face, HANDOFF_MARKER_SIZE, 0.6, 4.5)

    # Other phase boundaries — only with --show-phase
    if SHOW_PHASE:
        _plot_at_orig_idx(data.get("e2i_idx"), PHASE_E2I_MARKER,
                          "white", PHASE_MARKER_SIZE, 0.7, 4.4)
        _plot_at_orig_idx(data.get("t2m_idx"), PHASE_T2M_MARKER,
                          "white", PHASE_MARKER_SIZE, 0.7, 4.4)

    # End marker (filled) — shape encodes success/failure.
    end_marker = END_MARKER_OK if data.get("end_ok") else END_MARKER_FAIL
    ax.plot(R2v[-1], mag[-1],
            marker=end_marker, markerfacecolor=style["color"],
            markeredgecolor="black", markersize=END_MARKER_SIZE,
            mew=0.8, zorder=5)


def plot_cell(ax, data_yes, data_no):
    """Draw both yesPHII and noPHII trajectories on one cell."""
    # Cliff: vertical band at R_cloud (use yesPHII's value; same for both).
    rcloud = None
    for d in (data_yes, data_no):
        if d is not None and np.isfinite(d["rcloud"]):
            rcloud = float(d["rcloud"])
            break
    if rcloud is not None:
        ax.axvspan(
            rcloud * (1 - RCLOUD_BAND_FRAC),
            rcloud * (1 + RCLOUD_BAND_FRAC),
            color="0.55", alpha=0.18, zorder=1,
        )
        ax.axvline(rcloud, color="0.25", lw=1.2, ls="--", alpha=0.7,
                   zorder=2)

    # Failure-velocity threshold — horizontal reference.
    ax.axhline(V_FAIL_THRESHOLD, color="#7f7f7f", lw=1.0, ls=":",
               alpha=0.8, zorder=2)

    if data_yes is not None:
        _plot_one_trajectory(ax, data_yes, STYLE_YES,
                             smooth_window=SMOOTH_WINDOW,
                             smooth_mode=SMOOTH_MODE)
    if data_no is not None:
        _plot_one_trajectory(ax, data_no, STYLE_NO,
                             smooth_window=SMOOTH_WINDOW,
                             smooth_mode=SMOOTH_MODE)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4, zorder=0)


# ----------------------------------------------------------------
# Legend
# ----------------------------------------------------------------
def _build_legend_handles():
    handles = [
        Line2D([0], [0], **{k: STYLE_YES[k] for k in
                            ("color", "lw", "ls", "alpha")},
               label=STYLE_YES["label"]),
        Line2D([0], [0], **{k: STYLE_NO[k] for k in
                            ("color", "lw", "ls", "alpha")},
               label=STYLE_NO["label"]),
        Line2D([0], [0], color="0.25", lw=1.2, ls="--",
               label=r"$R_2 = R_{\rm cloud}$"),
        Line2D([0], [0], color="#7f7f7f", lw=1.0, ls=":",
               label=rf"$|v_2| = {V_FAIL_THRESHOLD:.0f}$ pc/Myr"),
        Line2D([0], [0], marker="o", color="0.3",
               markerfacecolor="white", markeredgecolor="0.3",
               linestyle="", markersize=4.5, label="start"),
        Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
               markerfacecolor=HANDOFF_FACE_OK, markeredgecolor="black",
               linestyle="", markersize=HANDOFF_MARKER_SIZE,
               label="implicit→transition (clean)"),
        Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
               markerfacecolor=HANDOFF_FACE_FAIL, markeredgecolor="black",
               linestyle="", markersize=HANDOFF_MARKER_SIZE,
               label="implicit→transition (LSODA fail)"),
    ]
    if SHOW_PHASE:
        handles += [
            Line2D([0], [0], marker=PHASE_E2I_MARKER, color="0.3",
                   markerfacecolor="white", markeredgecolor="black",
                   linestyle="", markersize=PHASE_MARKER_SIZE,
                   label="energy→implicit"),
            Line2D([0], [0], marker=PHASE_T2M_MARKER, color="0.3",
                   markerfacecolor="white", markeredgecolor="black",
                   linestyle="", markersize=PHASE_MARKER_SIZE,
                   label="transition→momentum"),
        ]
    handles += [
        Line2D([0], [0], marker=END_MARKER_OK, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE,
               label="end (clean)"),
        Line2D([0], [0], marker=END_MARKER_FAIL, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE,
               label="end (failed)"),
        Patch(facecolor="0.55", alpha=0.18, edgecolor="none",
              label=rf"$R_{{\rm cloud}}\,(1\pm{RCLOUD_BAND_FRAC:g})$"),
    ]
    return handles


# ----------------------------------------------------------------
# Sweep folder discovery (paired yes/no PHII)
# ----------------------------------------------------------------
def split_by_phii_suffix(folder):
    """Split ``find_all_simulations`` results into yes/no PHII pairs.

    Returns
    -------
    sim_files_anchor : list[Path]
        Dictionary paths of the runs we use to build the (mCloud, SFE)
        grid.  Prefers ``_yesPHII`` runs; falls back to ``_noPHII``
        when only those are present.
    yes_by_base, no_by_base : dict[str, Path]
        Lookup tables keyed by the suffix-stripped base name so each
        cell can pair its two variants.
    """
    yes_by_base = {}
    no_by_base = {}
    untagged = []
    for p in find_all_simulations(folder):
        name = p.parent.name
        if name.endswith(YES_SUFFIX):
            yes_by_base[name[: -len(YES_SUFFIX)]] = p
        elif name.endswith(NO_SUFFIX):
            no_by_base[name[: -len(NO_SUFFIX)]] = p
        else:
            untagged.append(p)

    # Anchor list: prefer yesPHII (TRINITY) when present.
    if yes_by_base:
        anchor = list(yes_by_base.values())
    elif no_by_base:
        anchor = list(no_by_base.values())
    else:
        anchor = untagged

    return anchor, yes_by_base, no_by_base, untagged


def _strip_suffix(name: str) -> str:
    if name.endswith(YES_SUFFIX):
        return name[: -len(YES_SUFFIX)]
    if name.endswith(NO_SUFFIX):
        return name[: -len(NO_SUFFIX)]
    return name


# ----------------------------------------------------------------
# Single-run plotter (one trajectory; no PHII pairing)
# ----------------------------------------------------------------
def plot_from_path(data_input: str, output_dir: Optional[str] = None):
    """Plot v_2 vs R_2 for a single simulation folder."""
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=150)

    try:
        data = load_run_v2R2(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        plt.close(fig)
        return

    name = data_path.parent.name
    if name.endswith(NO_SUFFIX):
        plot_cell(ax, None, data)
    else:
        plot_cell(ax, data, None)

    ax.set_xlabel(r"$R_2$ [pc]")
    ax.set_ylabel(r"$|v_2|$ [pc Myr$^{-1}$]")
    ax.set_title(f"$v_2$ vs $R_2$: {name}")

    # Secondary y-axis showing km/s for paper readers used to that unit.
    ax_kms = ax.secondary_yaxis(
        "right",
        functions=(lambda v: v * V_AU2KMS,
                   lambda v: v / V_AU2KMS),
    )
    ax_kms.set_ylabel(r"$|v_2|$ [km s$^{-1}$]")

    handles = _build_legend_handles()
    ax.legend(handles=handles, loc="lower left", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()

    out_pdf = FIG_DIR / f"paper_v2R2_{name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------
# Axis harmonisation
# ----------------------------------------------------------------
def _harmonize_axes(populated_axes):
    """Set a common (xmin, xmax, ymin, ymax) on every populated cell.

    Helps direct visual comparison of where each trajectory sits
    relative to its rCloud cliff.  Operates only on cells that were
    actually drawn — passing in cells that received only an axhline
    would bias the limits with the matplotlib log-scale default range.
    """
    if not populated_axes:
        return
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in populated_axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
            xmins.append(x0); xmaxs.append(x1)
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            ymins.append(y0); ymaxs.append(y1)
    if not xmins or not ymins:
        return
    xlo, xhi = min(xmins), max(xmaxs)
    ylo, yhi = min(ymins), max(ymaxs)
    # Clip y from below so we don't expose the v=0 floor band.
    ylo = max(ylo, 1e-2)
    for ax in populated_axes:
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)


# ----------------------------------------------------------------
# Grid plotter (paired yes/no PHII)
# ----------------------------------------------------------------
def plot_grid(folder, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """Build the (mCloud × SFE) grid for a sweep folder.

    Pairs ``_yesPHII`` / ``_noPHII`` runs by stripped base name.
    Untagged runs are plotted as the yes variant.
    """
    folder = Path(folder)
    folder_name = folder.name

    anchor, yes_by_base, no_by_base, untagged = split_by_phii_suffix(folder)
    if not anchor:
        print(f"No simulations found in: {folder}")
        return

    print(f"Found {len(anchor)} anchor runs "
          f"(_yesPHII: {len(yes_by_base)}, "
          f"_noPHII: {len(no_by_base)}, "
          f"untagged: {len(untagged)})")

    ndens_to_plot = [ndens_filter] if ndens_filter else get_unique_ndens(anchor)
    print(f"  Densities to plot: {ndens_to_plot}")

    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")
        organized = organize_simulations_for_grid(
            anchor, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
        )
        mCloud_list = organized["mCloud_list"]
        sfe_list = organized["sfe_list"]
        grid_anchor = organized["grid"]

        if not mCloud_list or not sfe_list:
            print(f"  No grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list}")
        print(f"  SFE: {sfe_list}")

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.4 * ncols, 2.9 * nrows),
            sharex=False, sharey=False,
            dpi=300, squeeze=False,
        )

        populated = []  # cells that received a real trajectory

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                anchor_path = grid_anchor.get((mCloud, sfe))
                if anchor_path is None:
                    mark_missing_cell(ax, "missing")
                    continue

                base = _strip_suffix(anchor_path.parent.name)
                path_yes = yes_by_base.get(base)
                path_no = no_by_base.get(base)

                # If the anchor was untagged, treat it as yesPHII.
                if path_yes is None and path_no is None:
                    path_yes = anchor_path

                data_yes = data_no = None
                try:
                    if path_yes is not None:
                        data_yes = load_run_v2R2(path_yes)
                    if path_no is not None:
                        data_no = load_run_v2R2(path_no)
                    plot_cell(ax, data_yes, data_no)
                except Exception as e:
                    print(f"  Error: {base}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                populated.append(ax)

                # Column title (top row only).
                if i == 0:
                    ax.set_title(_sfe_title(sfe))

                # Row label (left column only): mCloud.
                if j == 0:
                    label = (_mcloud_label_short(mCloud)
                             + "\n" + r"$|v_2|$ [pc Myr$^{-1}$]")
                    ax.set_ylabel(label)
                else:
                    ax.tick_params(labelleft=False)

                # X-axis label: bottom row only.
                if i == nrows - 1:
                    ax.set_xlabel(r"$R_2$ [pc]")
                else:
                    ax.tick_params(labelbottom=False)

        # Tie all cells to a common (R_2, |v_2|) range so the cliff
        # location is visually comparable across the grid.
        _harmonize_axes(populated)

        # Legend + suptitle via shared helper.
        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, _build_legend_handles(),
            n_rows_for_layout=nrows,
            cell_height_inches=2.9,
            folder_name=folder_name, param_tag=param_tag,
            legend_ncol=4,
            legend_fontsize=8,
        )

        # Save: keep the convention <FIG_DIR>/<folder_name>/paper_v2R2_*.pdf
        if not SAVE_PDF:
            plt.close(fig)
            continue

        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"paper_v2R2_{param_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias used by some sweep drivers.
plot_folder_grid = plot_grid


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
if __name__ == "__main__":
    from src._plots.cli import dispatch, marker_pre_dispatch
    dispatch(
        script_name="paper_v2R2.py",
        description="Plot TRINITY v_2 vs R_2 phase trajectory",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
