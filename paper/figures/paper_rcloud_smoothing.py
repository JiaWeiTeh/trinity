#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged figure: rCloud density-smoothing schematic (top) over the
before/after LSODA-failure trajectory comparison (bottom).

The two panels share the same narrative: panel 1 motivates the tanh
hyperbolic blend by showing how it replaces the discontinuous density
step at rCloud, and panel 2 demonstrates the downstream consequence —
the trajectory that previously triggered an LSODA failure now finishes
cleanly.

Self-contained: does NOT import from paper_rcloud_smooth, paper_v2R2,
or paper_v2R2_blend. All loaders, styles, and drawing helpers needed by
either panel are defined locally.

Usage:

    python paper_rcloud_smoothing.py <parent-folder-or-npz> [-o out.pdf]
    python paper_rcloud_smoothing.py <parent-folder> --export <out.npz>

The .npz bundle is the recommended "ship with the paper" format: it
holds the before/after trajectory arrays *and* the top-panel cloud
fiducials, so the figure can be regenerated without the original run
folders.
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

import trinity._functions.unit_conversions as cvt
from paper.figures._lib.plot_base import FIG_DIR
from trinity._output.trinity_reader import load_output, find_data_path
from trinity._output.simulation_end import read_simulation_end
from trinity.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw,
)


# =============================================================================
# Top panel: fiducial cloud parameters and density helpers
# =============================================================================
# These illustrate the tanh blend; they are *not* read from any run folder.
# They are persisted into the .npz bundle by ``export_rcloud_smoothing_npz``
# so the figure stays reproducible if the defaults below ever change.
DEFAULT_FIDUCIALS = dict(
    cloud_mass_Msun=1e6,
    n_core_cgs=1e3,
    n_ism_cgs=1.0,
    alpha=0.0,            # power-law slope; 0 = homogeneous
    r_core_pc=0.1,        # standalone core radius [pc]
    smooth_frac_default=0.01,
    smooth_frac_low=0.005,
    smooth_frac_high=0.02,
)

# Bundle keys for round-tripping the fiducials through the .npz.
_FIDUCIAL_KEYS = tuple(DEFAULT_FIDUCIALS.keys())


def _resolve_fiducials(overrides=None):
    """Merge overrides into the defaults and precompute derived quantities."""
    fid = dict(DEFAULT_FIDUCIALS)
    if overrides:
        fid.update({k: v for k, v in overrides.items()
                    if k in DEFAULT_FIDUCIALS and v is not None})

    mu = 1.4 * cvt.M_H_CGS * cvt.g2Msun
    n_core_au = fid["n_core_cgs"] * cvt.ndens_cgs2au
    n_ism_au = fid["n_ism_cgs"] * cvt.ndens_cgs2au
    if fid["alpha"] == 0:
        r_cloud = compute_rCloud_homogeneous(
            fid["cloud_mass_Msun"], n_core_au, mu=mu,
        )
    else:
        r_cloud, _ = compute_rCloud_powerlaw(
            fid["cloud_mass_Msun"], n_core_au, fid["alpha"],
            rCore=fid["r_core_pc"], mu=mu,
        )
    fid["_n_core_au"] = n_core_au
    fid["_n_ism_au"] = n_ism_au
    fid["_r_cloud_pc"] = r_cloud
    return fid


def _density_inside(r, fid):
    if fid["alpha"] == 0:
        return np.full_like(r, fid["_n_core_au"])
    n = fid["_n_core_au"] * (r / fid["r_core_pc"]) ** fid["alpha"]
    return np.where(r <= fid["r_core_pc"], fid["_n_core_au"], n)


def _density_jump(r, fid):
    return np.where(r <= fid["_r_cloud_pc"],
                    _density_inside(r, fid), fid["_n_ism_au"])


def _density_blend(r, smooth_frac, fid):
    delta = smooth_frac * fid["_r_cloud_pc"]
    w_out = 0.5 * (1.0 + np.tanh((r - fid["_r_cloud_pc"]) / delta))
    n_in = _density_inside(r, fid)
    return n_in * (1.0 - w_out) + fid["_n_ism_au"] * w_out


# =============================================================================
# Bottom panel: styles, markers, and per-trajectory drawing
# =============================================================================
BEFORE_SUFFIX = "_before_blend"
AFTER_SUFFIX  = "_after_blend"

STYLE_AFTER  = dict(color="k",       lw=1.3, ls="-",  alpha=0.95,
                    marker_scale=1.0,  marker_alpha=0.55,
                    label="after smoothing")
STYLE_BEFORE = dict(color="#d62728", lw=1.6, ls="--", alpha=0.95,
                    marker_scale=0.75, marker_alpha=0.55,
                    label="before smoothing")

# Recollapse segments (v_2 < 0) render as dotted lines with reduced
# linewidth/alpha so they read as a secondary echo of the primary curve.
RECOLLAPSE_LW_FACTOR    = 0.7
RECOLLAPSE_ALPHA_FACTOR = 0.55

END_MARKER_OK   = "o"
END_MARKER_FAIL = "X"
END_MARKER_SIZE = 7


def _check_implicit_lsoda_failure(data_path: Path) -> bool:
    """True iff trinity.log shows an LSODA bail in the implicit phase."""
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


def _load_run_v2R2(data_path: Path) -> dict:
    """Load t, R_2, v_2 [km/s], rCloud, and the simulation-end status."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t  = output.get("t_now")
    R2 = output.get("R2")
    v2 = output.get("v2") * cvt.v_au2kms
    rcloud = float(output[0].get("rCloud", np.nan))

    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, v2 = t[order], R2[order], v2[order]

    lsoda_failed = _check_implicit_lsoda_failure(data_path)

    # Termination status:
    #   * v3+ runs   → ``output.termination`` (read from
    #                  ``metadata.json[termination]``).
    #   * legacy runs → ``read_simulation_end`` falls back to text-parsing
    #                  ``simulationEnd.txt`` itself (Phase 2 made it
    #                  JSON-first with text fallback).
    #   * runs missing both → derive from the last snapshot's
    #                  ``SimulationEndReason`` via the legacy keyword
    #                  heuristic below.
    end_ok = None
    end_reason = None
    end_info = output.termination or read_simulation_end(str(data_path.parent))
    if end_info is not None and end_info.get("exit_code") is not None:
        end_ok = 0 <= int(end_info["exit_code"]) <= 9
        # NB: the canonical field is ``detail`` (post-Outcome/Detail
        # rename on main).  ``read_simulation_end`` exposes it under
        # the same name, and so does ``output.termination``.
        end_reason = end_info.get("detail")
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
        lsoda_failed=lsoda_failed,
    )


def _find_unique_subfolder(parent: Path, suffix: str) -> Path:
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
    before_dir = _find_unique_subfolder(folder, BEFORE_SUFFIX)
    after_dir  = _find_unique_subfolder(folder, AFTER_SUFFIX)
    before_path = find_data_path(before_dir)
    after_path  = find_data_path(after_dir)

    base_before = _strip_blend_suffix(before_dir.name)
    base_after  = _strip_blend_suffix(after_dir.name)
    run_id = base_before if base_before == base_after else folder.name

    return dict(
        before=_load_run_v2R2(before_path),
        after=_load_run_v2R2(after_path),
        meta=dict(run_id=run_id),
    )


def _load_pair_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        def _side(prefix: str) -> dict:
            return dict(
                t=z[f"{prefix}_t"].astype(float),
                R2=z[f"{prefix}_R2"].astype(float),
                v2=z[f"{prefix}_v2"].astype(float),
                rcloud=float(z["rcloud"]),
                end_ok=(bool(z[f"end_ok_{prefix}"])
                        if f"end_ok_{prefix}" in z.files else None),
                end_reason=(str(z[f"end_reason_{prefix}"])
                            if f"end_reason_{prefix}" in z.files else None),
                lsoda_failed=(bool(z[f"lsoda_failed_{prefix}"])
                              if f"lsoda_failed_{prefix}" in z.files else False),
            )
        run_id = str(z["run_id"]) if "run_id" in z.files else path.stem
        fiducials = {k: float(z[k]) for k in _FIDUCIAL_KEYS if k in z.files}
        return dict(
            before=_side("before"),
            after=_side("after"),
            meta=dict(run_id=run_id, fiducials=fiducials or None),
        )


def _load_v2R2_pair(source: Union[str, Path]) -> dict:
    """Load a before/after pair from a folder of .jsonl files or a .npz bundle."""
    p = Path(source)
    if p.is_dir():
        return _load_pair_from_folder(p)
    if p.is_file() and p.suffix == ".npz":
        return _load_pair_from_npz(p)
    raise FileNotFoundError(
        f"{source} is neither a folder nor a .npz bundle"
    )


def _plot_one_trajectory(ax, data, style):
    """Draw |v_2| vs R_2 for a single run, plus the success/fail end marker."""
    R2 = np.asarray(data["R2"], dtype=float)
    v2 = np.asarray(data["v2"], dtype=float)

    valid = np.isfinite(R2) & np.isfinite(v2) & (R2 > 0)
    if not np.any(valid):
        return

    R2v = R2[valid]
    v2v = v2[valid]

    # Plot |v_2|, switching to a dotted segment where v_2 < 0 (recollapse)
    # so the trajectory stays continuous on log-y.
    floor = 1e-3  # pc/Myr — keeps log scale finite if v_2 hits 0
    mag = np.maximum(np.abs(v2v), floor)
    sgn = np.sign(v2v)
    sgn[sgn == 0] = 1
    cuts = np.flatnonzero(sgn[1:] != sgn[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(R2v)]
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        is_recollapse = sgn[a] < 0
        ls = ":" if is_recollapse else style.get("ls", "-")
        lw = style["lw"] * (RECOLLAPSE_LW_FACTOR if is_recollapse else 1.0)
        seg_alpha = style.get("alpha", 0.95) * (
            RECOLLAPSE_ALPHA_FACTOR if is_recollapse else 1.0
        )
        ax.plot(R2v[a:b], mag[a:b],
                color=style["color"], lw=lw,
                ls=ls, alpha=seg_alpha,
                solid_capstyle="round", zorder=3)

    m_scale = style.get("marker_scale", 1.0)
    m_alpha = style.get("marker_alpha", 1.0)

    # End marker: shape encodes LSODA success vs failure.
    end_ok = bool(data.get("end_ok")) and not data.get("lsoda_failed")
    end_marker = END_MARKER_OK if end_ok else END_MARKER_FAIL
    ax.plot(R2v[-1], mag[-1],
            marker=end_marker, markerfacecolor=style["color"],
            markeredgecolor="black",
            markersize=END_MARKER_SIZE * m_scale,
            mew=0.8, alpha=m_alpha, zorder=5)


def _build_v2R2_legend_handles() -> list:
    return [
        Line2D([0], [0], marker=END_MARKER_OK, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE,
               label="LSODA succeeded"),
        Line2D([0], [0], marker=END_MARKER_FAIL, color="0.3",
               markerfacecolor="0.3", markeredgecolor="black",
               linestyle="", markersize=END_MARKER_SIZE,
               label="LSODA failed"),
    ]


# =============================================================================
# Panel drawers
# =============================================================================
# Legend fontsize for the small in-panel legends; matches the
# panel-(c) legend in paper_teaser at column width.
_LEGEND_FONTSIZE = 10


def _draw_rcloud_panel(ax, xlim, fid):
    """Top panel: density step vs tanh blends around rCloud.

    Y-axis is configured to mirror the bottom panel exactly (log
    scale, no grid, identical tick params); the x-axis is linear and
    shared with the bottom panel, with its tick labels suppressed.
    """
    xlo, xhi = xlim
    r = np.linspace(max(xlo, 0.0), xhi, 4000)

    n_jump = _density_jump(r, fid)
    ax.plot(r, n_jump * cvt.ndens_au2cgs,
            color='k', ls='-', lw=1.6, label='step (original)')

    blend_specs = [
        (fid["smooth_frac_low"],     '#0072B2', '--', 1.2),
        (fid["smooth_frac_default"], '#009E73', '-',  2.0),
        (fid["smooth_frac_high"],    '#D55E00', '--', 1.2),
    ]
    for sf, color, ls, lw in blend_specs:
        n_b = _density_blend(r, sf, fid)
        is_default = np.isclose(sf, fid["smooth_frac_default"])
        label = (r'$f_{\rm smooth}$' + rf'$={sf:g}$'
                 + (' (default)' if is_default else ''))
        ax.plot(r, n_b * cvt.ndens_au2cgs,
                color=color, ls=ls, lw=lw, label=label)

    ax.axvline(fid["_r_cloud_pc"],
               color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.set_yscale("log")
    ax.set_xscale('log')
    ax.grid(False)
    ax.tick_params(axis='both', labelbottom=False)
    ax.set_ylabel(r'$n(r)$ [cm$^{-3}$]')

    ax.legend(loc='lower left', handlelength=1.6, labelspacing=0.3,
              fontsize=_LEGEND_FONTSIZE, framealpha=0.9)


def _draw_v2R2_panel(ax, pair):
    """Bottom panel: before/after blend trajectories on (R_b, |v_b|)."""
    before = pair["before"]
    after  = pair["after"]

    rcloud = next((float(d["rcloud"]) for d in (after, before)
                   if np.isfinite(d.get("rcloud", np.nan))), None)
    if rcloud is not None:
        ax.axvline(rcloud, color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    # Draw "before" first so "after" (the hero) renders on top.
    _plot_one_trajectory(ax, before, STYLE_BEFORE)
    _plot_one_trajectory(ax, after,  STYLE_AFTER)

    ax.set_yscale("log")
    ax.set_xscale('log')
    ax.grid(False)
    ax.set_xlabel(r"$R_b$ [pc]")
    ax.set_ylabel(r"$v_b$ [km s$^{-1}$]")

    ax.legend(handles=_build_v2R2_legend_handles(), loc="lower left",
              fontsize=_LEGEND_FONTSIZE, framealpha=0.9)


# =============================================================================
# Export: folder -> publishable .npz
# =============================================================================
def export_rcloud_smoothing_npz(folder: Union[str, Path],
                                out_path: Union[str, Path]) -> Path:
    """Reduce a folder of blend .jsonl files to a single .npz figure bundle.

    Stores everything the figure consumes:
      - bottom panel: time / R2 / v2 arrays for the before+after runs,
        the per-run rCloud, end status, and LSODA-failure flags;
      - top panel: the illustrative cloud fiducials in
        ``DEFAULT_FIDUCIALS`` (cloud mass, n_core, n_ISM, alpha, r_core,
        and the three smoothing fractions).

    The resulting file is the recommended "ship with the paper" format
    for this figure — the original run folders can be discarded.
    """
    folder = Path(folder)
    pair = _load_pair_from_folder(folder)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        run_id=pair["meta"]["run_id"],
        rcloud=pair["before"]["rcloud"],
        before_t=pair["before"]["t"],
        before_R2=pair["before"]["R2"],
        before_v2=pair["before"]["v2"],
        after_t=pair["after"]["t"],
        after_R2=pair["after"]["R2"],
        after_v2=pair["after"]["v2"],
        end_ok_before=bool(pair["before"].get("end_ok") or False),
        end_ok_after=bool(pair["after"].get("end_ok") or False),
        end_reason_before=str(pair["before"].get("end_reason") or ""),
        end_reason_after=str(pair["after"].get("end_reason") or ""),
        lsoda_failed_before=bool(pair["before"].get("lsoda_failed") or False),
        lsoda_failed_after=bool(pair["after"].get("lsoda_failed") or False),
    )
    # Top-panel fiducials (one scalar per key).
    payload.update({k: float(DEFAULT_FIDUCIALS[k]) for k in _FIDUCIAL_KEYS})

    np.savez(out_path, **payload)
    print(f"Exported: {out_path}")
    return out_path


# =============================================================================
# Orchestration
# =============================================================================
def plot_merged(pair: dict, out_path: Optional[Path] = None,
                show: bool = False) -> plt.Figure:
    """Stack the rCloud-smoothing schematic over the v_2 vs R_2 comparison.

    Both panels are log-x and share the x-axis, so the rcloud vertical
    lines align by construction.  Canvas width matches paper_teaser
    (4.0") so the rendered fonts/ticks scale identically when both
    figures are placed at A&A \\columnwidth.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=(4.0, 5.5), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1]),
    )

    # Prefer fiducials embedded in the bundle; fall back to defaults.
    fid = _resolve_fiducials(pair.get("meta", {}).get("fiducials"))

    # Draw bottom first so its data-driven xlim drives the top panel.
    _draw_v2R2_panel(ax_bot, pair)
    _draw_rcloud_panel(ax_top, xlim=ax_bot.get_xlim(), fid=fid)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig


# =============================================================================
# CLI
# =============================================================================
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Stacked figure: rCloud smoothing schematic on top, "
                     "before/after blend v_2 vs R_2 trajectory on bottom. "
                     "Source can be a folder of .jsonl snapshots or a "
                     "published .npz bundle."),
    )
    parser.add_argument("source",
                        help="parent folder containing *_before_blend/ and "
                             "*_after_blend/ TRINITY run subdirectories, or "
                             "a .npz bundle")
    parser.add_argument("-o", "--out", default=None,
                        help="output PDF path "
                             "(default: <FIG_DIR>/paper_rcloud_smoothing_<id>.pdf)")
    parser.add_argument("--export", default=None,
                        help="export the source folder to this .npz bundle "
                             "and exit (no plot). Bundle contains both the "
                             "trajectory arrays and the top-panel cloud "
                             "fiducials, so the original folder can be "
                             "discarded.")
    parser.add_argument("--show", action="store_true",
                        help="open the figure window (in addition to saving)")
    args = parser.parse_args(argv)

    if args.export:
        export_rcloud_smoothing_npz(args.source, args.export)
        return

    pair = _load_v2R2_pair(args.source)
    run_id = pair["meta"].get("run_id", "blend")
    out_path = (Path(args.out) if args.out
                else FIG_DIR / f"paper_rcloud_smoothing_{run_id}.pdf")
    plot_merged(pair, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
