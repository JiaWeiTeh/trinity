#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after blend trajectory comparison on a v_2 vs R_2 phase plot.

Loads a pair of TRINITY snapshot dumps named ``*_before_blend*.jsonl`` and
``*_after_blend*.jsonl`` from a folder, then overlays both trajectories on a
single (R_2, |v_2|) panel using the same visual conventions as
``paper_v2R2.py`` (log-log axes, rCloud cliff band, failure-velocity
threshold, start/end markers, implicit->transition handoff diamond).

Designed in two layers so the published-paper migration is trivial:

  load_v2R2_pair(source)   accepts either a folder of .jsonl snapshots
                           or a single .npz bundle (forward-compatible)
  plot_v2R2_diff(pair, ...) pure plotting; no I/O assumptions

Plus an ``export_v2R2_npz(folder, out_path)`` helper so any folder of blend
.jsonl files can be reduced to a single self-describing .npz containing
just the arrays the figure needs.

Usage:

    python paper_v2R2_blend.py <folder-or-npz> [-o out.pdf]
    python paper_v2R2_blend.py <folder> --export <out.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR
from src._plots.paper_v2R2 import (
    load_run_v2R2,
    _plot_one_trajectory,
    RCLOUD_BAND_FRAC,
    V_FAIL_THRESHOLD,
    V_AU2KMS,
    HANDOFF_MARKER,
    HANDOFF_FACE_OK,
    HANDOFF_FACE_FAIL,
    HANDOFF_MARKER_SIZE,
    END_MARKER_OK,
    END_MARKER_FAIL,
    END_MARKER_SIZE,
)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
BEFORE_GLOB = "*_before_blend*"
AFTER_GLOB  = "*_after_blend*"

# Per-variant styling. "after" is the hero (drawn on top, solid black);
# "before" is the baseline (dashed red), mirroring the yes/noPHII pairing
# in paper_v2R2.py so the visual grammar transfers directly.
STYLE_AFTER  = dict(color="k",       lw=1.3, ls="-",  alpha=0.95,
                    marker_scale=1.0,  marker_alpha=0.7,
                    label="after blend")
STYLE_BEFORE = dict(color="#d62728", lw=1.6, ls="--", alpha=0.95,
                    marker_scale=0.75, marker_alpha=0.7,
                    label="before blend")


# ----------------------------------------------------------------
# Layer 1: source-agnostic loading
# ----------------------------------------------------------------
def _find_unique(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {folder}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files matching '{pattern}' in {folder}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _load_pair_from_folder(folder: Path) -> dict:
    """Find the before/after blend .jsonl files and load both trajectories."""
    before_path = _find_unique(folder, BEFORE_GLOB)
    after_path  = _find_unique(folder, AFTER_GLOB)
    return dict(
        before=load_run_v2R2(before_path),
        after=load_run_v2R2(after_path),
        meta=dict(
            run_id=folder.name,
            before_path=str(before_path),
            after_path=str(after_path),
        ),
    )


def _phase_indices_from_array(phase_arr: np.ndarray) -> dict:
    """Recover (e2i, handoff, t2m) indices from a saved phase tag array."""
    out = dict(e2i_idx=None, handoff_idx=None, t2m_idx=None)
    if phase_arr is None or phase_arr.size == 0:
        return out
    for tag, key in (("implicit", "e2i_idx"),
                     ("transition", "handoff_idx"),
                     ("momentum", "t2m_idx")):
        hits = np.flatnonzero(phase_arr == tag)
        if hits.size:
            out[key] = int(hits[0])
    return out


def _load_pair_from_npz(path: Path) -> dict:
    """Reconstruct the pair dict from a published .npz bundle."""
    with np.load(path, allow_pickle=False) as z:
        def _side(prefix: str) -> dict:
            phase = z[f"{prefix}_phase"] if f"{prefix}_phase" in z.files else np.array([])
            d = dict(
                t=z[f"{prefix}_t"].astype(float),
                R2=z[f"{prefix}_R2"].astype(float),
                v2=z[f"{prefix}_v2"].astype(float),
                rcloud=float(z["rcloud"]),
                end_ok=bool(z[f"end_ok_{prefix}"]) if f"end_ok_{prefix}" in z.files else None,
                end_reason=str(z[f"end_reason_{prefix}"]) if f"end_reason_{prefix}" in z.files else None,
                lsoda_failed=bool(z[f"lsoda_failed_{prefix}"]) if f"lsoda_failed_{prefix}" in z.files else False,
            )
            d.update(_phase_indices_from_array(np.asarray(phase)))
            return d
        run_id = str(z["run_id"]) if "run_id" in z.files else path.stem
        return dict(
            before=_side("before"),
            after=_side("after"),
            meta=dict(run_id=run_id, source=str(path)),
        )


def load_v2R2_pair(source: Union[str, Path]) -> dict:
    """Load a before/after pair from a folder of .jsonl files or a .npz bundle.

    The returned dict has the shape::

        {'before': {...}, 'after': {...}, 'meta': {...}}

    where each side carries the keys produced by ``load_run_v2R2``: ``t``,
    ``R2``, ``v2``, ``rcloud``, ``end_ok``, ``end_reason``, ``e2i_idx``,
    ``handoff_idx``, ``t2m_idx``, ``lsoda_failed``.
    """
    p = Path(source)
    if p.is_dir():
        return _load_pair_from_folder(p)
    if p.is_file() and p.suffix == ".npz":
        return _load_pair_from_npz(p)
    raise FileNotFoundError(
        f"{source} is neither a folder nor a .npz bundle"
    )


# ----------------------------------------------------------------
# Layer 2: pure plotting
# ----------------------------------------------------------------
def _build_legend_handles() -> list:
    return [
        Line2D([0], [0], **{k: STYLE_AFTER[k]  for k in ("color", "lw", "ls", "alpha")},
               label=STYLE_AFTER["label"]),
        Line2D([0], [0], **{k: STYLE_BEFORE[k] for k in ("color", "lw", "ls", "alpha")},
               label=STYLE_BEFORE["label"]),
        Line2D([0], [0], color="0.3", lw=1.4, ls=":",
               label=r"$v_2 < 0$ (recollapse)"),
        Line2D([0], [0], color="0.25", lw=1.2, ls="--",
               label=r"$R_{\rm cloud}$"),
        Line2D([0], [0], color="#7f7f7f", lw=1.0, ls=":",
               label=rf"$|v_2| = {V_FAIL_THRESHOLD:.0f}$ pc/Myr"),
        Line2D([0], [0], marker="o", color="0.3",
               markerfacecolor="white", markeredgecolor="0.3",
               linestyle="", markersize=4.5, label="start"),
        Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
               markerfacecolor=HANDOFF_FACE_OK, markeredgecolor="black",
               linestyle="", markersize=HANDOFF_MARKER_SIZE,
               label="implicit$\\to$transition (clean)"),
        Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
               markerfacecolor=HANDOFF_FACE_FAIL, markeredgecolor="black",
               linestyle="", markersize=HANDOFF_MARKER_SIZE,
               label="implicit$\\to$transition (LSODA fail)"),
        Line2D([0], [0], marker=END_MARKER_OK, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE, label="end (clean)"),
        Line2D([0], [0], marker=END_MARKER_FAIL, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE, label="end (failed)"),
    ]


def plot_v2R2_diff(pair: dict, out_path: Optional[Path] = None,
                   title: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """Overlay the before/after trajectories on a single v_2 vs R_2 panel."""
    before = pair["before"]
    after  = pair["after"]

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=150)

    # rCloud cliff (use whichever side reports a finite value).
    rcloud = next((float(d["rcloud"]) for d in (after, before)
                   if np.isfinite(d.get("rcloud", np.nan))), None)
    if rcloud is not None:
        ax.axvspan(rcloud * (1 - RCLOUD_BAND_FRAC),
                   rcloud * (1 + RCLOUD_BAND_FRAC),
                   color="0.55", alpha=0.18, zorder=1)
        ax.axvline(rcloud, color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.axhline(V_FAIL_THRESHOLD, color="#7f7f7f", lw=1.0, ls=":",
               alpha=0.8, zorder=2)

    # Draw "before" first so "after" (the hero) renders on top.
    _plot_one_trajectory(ax, before, STYLE_BEFORE)
    _plot_one_trajectory(ax, after,  STYLE_AFTER)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel(r"$R_2$ [pc]")
    ax.set_ylabel(r"$|v_2|$ [pc Myr$^{-1}$]")

    if title is None:
        title = f"$v_2$ vs $R_2$: {pair['meta'].get('run_id', 'blend')}"
    ax.set_title(title)

    ax_kms = ax.secondary_yaxis(
        "right",
        functions=(lambda v: v * V_AU2KMS, lambda v: v / V_AU2KMS),
    )
    ax_kms.set_ylabel(r"$|v_2|$ [km s$^{-1}$]")

    ax.legend(handles=_build_legend_handles(), loc="lower left",
              fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig


# ----------------------------------------------------------------
# Export: folder -> publishable .npz
# ----------------------------------------------------------------
def _phase_array_from_jsonl(jsonl_path: Path) -> np.ndarray:
    """Re-read just the ``current_phase`` column so we can persist it."""
    from src._output.trinity_reader import load_output
    output = load_output(jsonl_path)
    phase = output.get("current_phase", as_array=False)
    return np.asarray(phase) if phase is not None else np.array([])


def export_v2R2_npz(folder: Union[str, Path],
                    out_path: Union[str, Path]) -> Path:
    """Reduce a folder of blend .jsonl files to a single .npz figure bundle.

    Stores only what ``plot_v2R2_diff`` consumes: time/R2/v2/phase arrays,
    rCloud, end status, and LSODA-failure flags.  The resulting file is the
    recommended "ship with the paper" format for this figure.
    """
    folder = Path(folder)
    pair = _load_pair_from_folder(folder)

    # Persist the raw phase tags (rather than just indices) so the file
    # survives unrelated phase-tag changes downstream.
    before_phase = _phase_array_from_jsonl(Path(pair["meta"]["before_path"]))
    after_phase  = _phase_array_from_jsonl(Path(pair["meta"]["after_path"]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        run_id=pair["meta"]["run_id"],
        rcloud=pair["before"]["rcloud"],
        before_t=pair["before"]["t"],
        before_R2=pair["before"]["R2"],
        before_v2=pair["before"]["v2"],
        before_phase=before_phase.astype("U32"),
        after_t=pair["after"]["t"],
        after_R2=pair["after"]["R2"],
        after_v2=pair["after"]["v2"],
        after_phase=after_phase.astype("U32"),
        end_ok_before=bool(pair["before"].get("end_ok") or False),
        end_ok_after=bool(pair["after"].get("end_ok") or False),
        end_reason_before=str(pair["before"].get("end_reason") or ""),
        end_reason_after=str(pair["after"].get("end_reason") or ""),
        lsoda_failed_before=bool(pair["before"].get("lsoda_failed") or False),
        lsoda_failed_after=bool(pair["after"].get("lsoda_failed") or False),
    )
    np.savez(out_path, **payload)
    print(f"Exported: {out_path}")
    return out_path


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Overlay before/after blend trajectories on a v_2 vs R_2 "
                     "phase plot. Source can be a folder of .jsonl snapshots "
                     "or a published .npz bundle."),
    )
    parser.add_argument("source",
                        help="folder containing *_before_blend*/*_after_blend* .jsonl files, "
                             "or a .npz bundle")
    parser.add_argument("-o", "--out", default=None,
                        help="output PDF path (default: <FIG_DIR>/paper_v2R2_blend_<id>.pdf)")
    parser.add_argument("--export", default=None,
                        help="export the pair to this .npz path and exit (no plot)")
    parser.add_argument("--show", action="store_true",
                        help="open the figure window (in addition to saving)")
    args = parser.parse_args(argv)

    if args.export:
        export_v2R2_npz(args.source, args.export)
        return

    pair = load_v2R2_pair(args.source)
    run_id = pair["meta"].get("run_id", "blend")
    out_path = (Path(args.out) if args.out
                else FIG_DIR / f"paper_v2R2_blend_{run_id}.pdf")
    plot_v2R2_diff(pair, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
