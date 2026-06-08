#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barnes 2026 comparison — pressure balance against the ambient ISM.

TRINITY-only (no Barnes overlay yet). Two rows of panels, one column per
stellar age (default 0.5, 1, 3 Myr); every marker is one TRINITY run sampled
at that age. All pressures are on Barnes' ``P/k`` [K cm^-3] basis.

    top row    : P_ISM (x) vs P_tot (y), with the P_tot = P_ISM line
    bottom row : P_tot (x) vs P_tot - P_ISM (y), with the zero line
                 (y > 0 => over-pressured w.r.t. the ambient ISM)

``P_tot = P_thermal + P_radiation`` where P_thermal is TRINITY's HII
ionization-balance pressure ``P_HII`` and P_radiation is the radiation
pressure (native ``F_rad/(4 pi R2^2)`` by default; ``--prad`` selects the
Barnes-formula recompute or both).

Runs whose ``PISM`` is absent or non-positive are skipped (a Barnes
P_ISM = P_DE comparison requires P_ISM > 0).

Usage
-----
  python -m paper.barnes26.paper_PressureBalance -F <runs-folder> [options]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper._lib.plot_base import FIG_DIR  # noqa: E402  applies trinity.mplstyle
from paper.barnes26._barnes_lib import (  # noqa: E402
    DEFAULT_AGES_MYR, load_runs, collect_age_records,
    to_Pk, pism_to_Pk, p_rad_native, p_rad_barnes, project_root,
)

PRAD_STYLES = {
    "native": dict(color="#1f77b4", label=r"$P_{\rm rad}$ native"),
    "barnes": dict(color="#d62728", label=r"$P_{\rm rad}$ Barnes formula"),
}


def _with_positive_pism(recs):
    """Keep records with a finite, positive PISM; return (kept, n_skipped)."""
    kept = [r for r in recs if np.isfinite(r["PISM"]) and r["PISM"] > 0]
    return kept, len(recs) - len(kept)


def _ptot_series(recs, prad_modes):
    """Return (PISM, {mode: P_tot}) [K cm^-3] for the requested P_rad modes."""
    P_th = to_Pk(np.array([r["P_HII"] for r in recs], dtype=float))
    PISM = pism_to_Pk(np.array([r["PISM"] for r in recs], dtype=float))
    F_rad = np.array([r["F_rad"] for r in recs], dtype=float)
    R2 = np.array([r["R2"] for r in recs], dtype=float)
    Lbol = np.array([r["Lbol"] for r in recs], dtype=float)
    Li = np.array([r["Li"] for r in recs], dtype=float)
    f_neu = np.array([r["f_neu"] for r in recs], dtype=float)
    f_ion = np.array([r["f_ion"] for r in recs], dtype=float)

    series = {}
    if "native" in prad_modes:
        series["native"] = P_th + p_rad_native(F_rad, R2)
    if "barnes" in prad_modes:
        series["barnes"] = P_th + p_rad_barnes(Lbol, Li, R2, f_neu, f_ion)
    return PISM, series


def _symlog_y(ax, y):
    """Set a symlog y-scale with a robust linthresh for signed wide-range data."""
    absy = np.abs(y[np.isfinite(y) & (y != 0)])
    linthresh = float(np.percentile(absy, 25)) if absy.size else 1.0
    ax.set_yscale("symlog", linthresh=max(linthresh, 1e-300))


def plot_figure(records_by_age, ages, prad_modes, out_path):
    ncols = len(ages)
    fig, axes = plt.subplots(
        nrows=2, ncols=ncols,
        figsize=(3.4 * ncols, 5.6),
        squeeze=False,
    )

    for j, age in enumerate(ages):
        ax_top, ax_bot = axes[0, j], axes[1, j]
        recs_all = records_by_age.get(age, [])
        recs, n_skipped = _with_positive_pism(recs_all)

        ax_top.set_title(f"t = {age:g} Myr"
                         + (f"  ({n_skipped} skipped: P_ISM<=0)" if n_skipped else ""),
                         fontsize=9)

        if not recs:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, "no runs with P_ISM > 0",
                        ha="center", va="center", transform=ax.transAxes,
                        color="grey", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            continue

        PISM, series = _ptot_series(recs, prad_modes)

        # --- top: P_ISM vs P_tot ---
        all_pos = [PISM]
        for mode, Ptot in series.items():
            st = PRAD_STYLES[mode]
            m = np.isfinite(PISM) & np.isfinite(Ptot) & (PISM > 0) & (Ptot > 0)
            ax_top.scatter(PISM[m], Ptot[m], s=28, color=st["color"],
                           edgecolor="k", linewidth=0.4, alpha=0.85)
            all_pos.append(Ptot[m])
        lims = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in all_pos if a.size])
        if lims.size:
            lo, hi = lims.min(), lims.max()
            ax_top.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0, zorder=1)
        ax_top.set_xscale("log"); ax_top.set_yscale("log")
        ax_top.grid(True, which="both", alpha=0.25, lw=0.5)
        if j == 0:
            ax_top.set_ylabel(r"$P_{\rm tot}/k$ [K cm$^{-3}$]")
        ax_top.set_xlabel(r"$P_{\rm ISM}/k$ [K cm$^{-3}$]")

        # --- bottom: P_tot vs (P_tot - P_ISM) ---
        diff_for_scale = []
        for mode, Ptot in series.items():
            st = PRAD_STYLES[mode]
            diff = Ptot - PISM
            m = np.isfinite(Ptot) & np.isfinite(diff) & (Ptot > 0)
            ax_bot.scatter(Ptot[m], diff[m], s=28, color=st["color"],
                           edgecolor="k", linewidth=0.4, alpha=0.85)
            diff_for_scale.append(diff[m])
        ax_bot.axhline(0.0, color="0.4", ls="--", lw=1.0, zorder=1)
        ax_bot.set_xscale("log")
        if diff_for_scale:
            _symlog_y(ax_bot, np.concatenate(diff_for_scale))
        ax_bot.grid(True, which="both", alpha=0.25, lw=0.5)
        if j == 0:
            ax_bot.set_ylabel(r"$(P_{\rm tot}-P_{\rm ISM})/k$ [K cm$^{-3}$]")
        ax_bot.set_xlabel(r"$P_{\rm tot}/k$ [K cm$^{-3}$]")

    handles = [
        Line2D([0], [0], marker="o", ls="", color=PRAD_STYLES[m]["color"],
               markeredgecolor="k", label=PRAD_STYLES[m]["label"])
        for m in prad_modes
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(r"Pressure balance vs ambient ISM ($P_{\rm tot}=P_{\rm HII}+P_{\rm rad}$)",
                 y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pressure balance vs ambient ISM for Barnes-matched TRINITY runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-F", "--folder", required=True,
                        help="Folder of TRINITY run subfolders (each with dictionary.jsonl)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: paper/plots)")
    parser.add_argument("--ages", nargs="+", type=float, default=list(DEFAULT_AGES_MYR),
                        help="Stellar ages [Myr], one plot column each (default: 0.5 1 3)")
    parser.add_argument("--prad", choices=["native", "barnes", "both"], default="native",
                        help="Radiation-pressure definition used in P_tot (default: native)")
    args = parser.parse_args()

    outputs = load_runs(args.folder)
    if not outputs:
        print(f"No runs found under: {args.folder}")
        return
    print(f"Loaded {len(outputs)} run(s); ages = {args.ages} Myr; prad = {args.prad}")

    records_by_age = collect_age_records(outputs, args.ages)
    prad_modes = ["native", "barnes"] if args.prad == "both" else [args.prad]

    out_dir = Path(args.output_dir) if args.output_dir else project_root() / "paper" / "plots"
    plot_figure(records_by_age, args.ages, prad_modes,
                out_dir / f"barnes26_PressureBalance_{args.prad}.pdf")


if __name__ == "__main__":
    main()
