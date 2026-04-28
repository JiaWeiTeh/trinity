#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after blend trajectory comparison on a v_2 vs R_2 phase plot.

Expects a *parent* folder containing two TRINITY run subdirectories whose
names end in ``_before_blend`` and ``_after_blend`` (e.g.
``1e6_sfe010_n1e3_PL0_yesPHII_before_blend/`` and the matching
``..._after_blend/``). The standard ``dictionary.jsonl`` inside each is
loaded and both trajectories are overlaid on a single (R_2, |v_2|) panel
using the same visual conventions as ``paper_v2R2.py`` (log-log axes,
rCloud cliff band, failure-velocity threshold, start/end markers,
implicit->transition handoff diamond).

Designed in two layers so the published-paper migration is trivial:

  load_v2R2_pair(source)   accepts either the parent folder described
                           above, or a single .npz bundle (forward-compat)
  plot_v2R2_diff(pair, ...) pure plotting; no I/O assumptions

Plus an ``export_v2R2_npz(folder, out_path)`` helper that reduces such a
parent folder to a single self-describing .npz containing only the arrays
the figure needs.

Usage:

    python paper_v2R2_blend.py <parent-folder-or-npz> [-o out.pdf]
    python paper_v2R2_blend.py <parent-folder> --export <out.npz>
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

import src._functions.unit_conversions as cvt

from src._plots.plot_base import FIG_DIR
from src._plots.paper_v2R2 import (
    load_run_v2R2,
    _plot_one_trajectory,
    HANDOFF_MARKER,
    HANDOFF_FACE_OK,
    HANDOFF_FACE_FAIL,
    HANDOFF_MARKER_SIZE,
    END_MARKER_OK,
    END_MARKER_FAIL,
    END_MARKER_SIZE,
)
from src._output.trinity_reader import find_data_path

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
# These match TRINITY's run-folder naming convention: each side of the
# blend comparison is a full simulation directory whose name ends with
# the corresponding suffix; the .jsonl lives inside (dictionary.jsonl
# by TRINITY convention).
BEFORE_SUFFIX = "_before_blend"
AFTER_SUFFIX  = "_after_blend"

# Per-variant styling. "after" (tanh hyperbolic blend at rCloud — the
# scheme currently in src/cloud_properties/density_profile.py) is the
# hero, drawn on top in solid black. "before" (the original
# discontinuous step at rCloud) is the dashed red baseline. Mirrors the
# yes/noPHII pairing in paper_v2R2.py so the visual grammar transfers.
# marker_alpha = 0.55 (matches paper_v2R2) so the smaller "step" marker
# pokes through the larger "tanh" marker where the trajectories coincide.
# Label convention matches paper_rcloud_smooth.py:126.
STYLE_AFTER  = dict(color="k",       lw=1.3, ls="-",  alpha=0.95,
                    marker_scale=1.0,  marker_alpha=0.55,
                    label="after smoothing")
STYLE_BEFORE = dict(color="#d62728", lw=1.6, ls="--", alpha=0.95,
                    marker_scale=0.75, marker_alpha=0.55,
                    label="before smoothing")


# ----------------------------------------------------------------
# Layer 1: source-agnostic loading
# ----------------------------------------------------------------
def _find_unique_subfolder(parent: Path, suffix: str) -> Path:
    """Return the unique subdirectory of ``parent`` whose name ends with ``suffix``."""
    matches = sorted(p for p in parent.iterdir()
                     if p.is_dir() and p.name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(
            f"No subdirectory ending in '{suffix}' under {parent}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple subdirectories ending in '{suffix}' under {parent}: "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _strip_blend_suffix(name: str) -> str:
    for s in (BEFORE_SUFFIX, AFTER_SUFFIX):
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _load_pair_from_folder(folder: Path) -> dict:
    """Find the before/after blend run folders and load both trajectories.

    Each side is a TRINITY run folder (e.g.
    ``1e6_sfe010_n1e3_PL0_yesPHII_before_blend/``) whose
    ``dictionary.jsonl`` is loaded via ``load_run_v2R2``. Because the two
    sides live in *separate* subfolders, each carries its own
    ``simulationEnd.txt`` / ``trinity.log``, so ``end_ok`` and
    ``lsoda_failed`` are independent per side (unlike the earlier
    same-folder layout this script briefly assumed).
    """
    before_dir = _find_unique_subfolder(folder, BEFORE_SUFFIX)
    after_dir  = _find_unique_subfolder(folder, AFTER_SUFFIX)
    before_path = find_data_path(before_dir)
    after_path  = find_data_path(after_dir)

    # Prefer the stripped run base (e.g. "1e6_sfe010_n1e3_PL0_yesPHII")
    # over the comparison-folder name as the figure's identity.
    base_before = _strip_blend_suffix(before_dir.name)
    base_after  = _strip_blend_suffix(after_dir.name)
    run_id = base_before if base_before == base_after else folder.name

    return dict(
        before=load_run_v2R2(before_path),
        after=load_run_v2R2(after_path),
        meta=dict(
            run_id=run_id,
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
        #Line2D([0], [0], **{k: STYLE_AFTER[k]  for k in ("color", "lw", "ls", "alpha")},
        #       label=STYLE_AFTER["label"]),
        #Line2D([0], [0], **{k: STYLE_BEFORE[k] for k in ("color", "lw", "ls", "alpha")},
        #       label=STYLE_BEFORE["label"]),
        #Line2D([0], [0], color="0.3", lw=1.4, ls=":",
        #       label=r"$v_2 < 0$ (recollapse)"),
        #Line2D([0], [0], color="0.25", lw=1.2, ls="--",
        #       label=r"$R_{\rm cloud}$"),
        #Line2D([0], [0], marker="o", color="0.3",
        #       markerfacecolor="white", markeredgecolor="0.3",
        #       linestyle="", markersize=4.5, label="start"),
        #Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
        #       markerfacecolor=HANDOFF_FACE_OK, markeredgecolor="black",
        #       linestyle="", markersize=HANDOFF_MARKER_SIZE,
        #       label="implicit$\\to$transition (clean)"),
        #Line2D([0], [0], marker=HANDOFF_MARKER, color="0.3",
        #       markerfacecolor=HANDOFF_FACE_FAIL, markeredgecolor="black",
        #       linestyle="", markersize=HANDOFF_MARKER_SIZE,
        #       label="implicit$\\to$transition (LSODA fail)"),
        Line2D([0], [0], marker=END_MARKER_OK, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE, label="end (LSODA succeed)"),
        Line2D([0], [0], marker=END_MARKER_FAIL, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE, label="end (LSODA failed)"),
    ]


def plot_v2R2_diff(pair: dict, out_path: Optional[Path] = None,
                   show: bool = False) -> plt.Figure:
    """Overlay the before/after trajectories on a single v_2 vs R_2 panel."""
    before = pair["before"]
    after  = pair["after"]

    FONTSIZE = 25

    fig, ax = plt.subplots(figsize=[8, 3], dpi=150)
    ax.tick_params(labelsize = FONTSIZE, axis = 'both')
    

    # rCloud cliff: single dashed vertical line, no surrounding band.
    rcloud = next((float(d["rcloud"]) for d in (after, before)
                   if np.isfinite(d.get("rcloud", np.nan))), None)
    if rcloud is not None:
        ax.axvline(rcloud, color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    # Draw "before" first so "after" (the hero) renders on top.
    _plot_one_trajectory(ax, before, STYLE_BEFORE)
    _plot_one_trajectory(ax, after,  STYLE_AFTER)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel(r"$R_b$ [pc]", fontsize = FONTSIZE)
    ax.set_ylabel(r"$v_b$ [km s$^{-1}$]", fontsize = FONTSIZE)

    ax.legend(handles=_build_legend_handles(), loc="lower left",
              fontsize= 17, framealpha=0.9)

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
                        help="parent folder containing *_before_blend/ and *_after_blend/ "
                             "TRINITY run subdirectories, or a .npz bundle")
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
