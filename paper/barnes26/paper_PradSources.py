#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barnes 2026 comparison — radiation pressure vs source properties.

TRINITY-only (no Barnes overlay yet). A grid of P_rad vs source property:

    columns : L_bol, M_star, radius
    rows    : stellar ages (default 0.5, 1, 3 Myr)

Each marker is one TRINITY run sampled at that age (closest snapshot). P_rad
is shown two ways per panel: TRINITY-native ``F_rad/(4 pi R2^2)`` (the force
the model actually exerts, IR-boosted) and the Barnes-formula recompute
``prefactor * L / (4 pi r^2 c)``.

Two figures are produced — the radius column uses ``R2`` in one and ``R_IF``
in the other. P_rad itself is always evaluated at the shell radius ``R2``
(that is where TRINITY's radiation force acts); the radius column is a size
proxy on the x-axis only. Evaluating P_rad at the matched size is deferred to
when Barnes data is overlaid.

Usage
-----
  python -m paper.barnes26.paper_PradSources -F <runs-folder> [options]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper._lib.plot_base import FIG_DIR  # noqa: E402  applies trinity.mplstyle
from trinity._functions.unit_conversions import L_au2cgs  # noqa: E402
from paper.barnes26._barnes_lib import (  # noqa: E402
    DEFAULT_AGES_MYR, load_runs, collect_age_records,
    p_rad_native, p_rad_barnes, project_root,
)

COLOR_NATIVE = "#1f77b4"   # TRINITY-native P_rad
COLOR_BARNES = "#d62728"   # Barnes-formula recompute


def _column_specs(radius_key):
    """(record-getter, x-label) for each column, given the radius variant."""
    r_label = r"$R_2$ [pc]" if radius_key == "R2" else r"$R_{\rm IF}$ [pc]"
    return [
        (lambda r: r["Lbol"] * L_au2cgs, r"$L_{\rm bol}$ [erg/s]"),
        (lambda r: r["mCluster"],        r"$M_\star$ [M$_\odot$]"),
        (lambda r: r[radius_key],        r_label),
    ]


def _finite_positive(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[m], y[m]


def _prad_arrays(recs):
    """Return (P_rad_native, P_rad_barnes) [K cm^-3] for a list of records."""
    F_rad = np.array([r["F_rad"] for r in recs], dtype=float)
    R2 = np.array([r["R2"] for r in recs], dtype=float)
    Lbol = np.array([r["Lbol"] for r in recs], dtype=float)
    Li = np.array([r["Li"] for r in recs], dtype=float)
    f_neu = np.array([r["f_neu"] for r in recs], dtype=float)
    f_ion = np.array([r["f_ion"] for r in recs], dtype=float)
    return (p_rad_native(F_rad, R2),
            p_rad_barnes(Lbol, Li, R2, f_neu, f_ion))


def plot_figure(records_by_age, ages, radius_key, out_path):
    cols = _column_specs(radius_key)
    nrows, ncols = len(ages), len(cols)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.4 * ncols, 2.8 * nrows),
        squeeze=False,
    )

    for i, age in enumerate(ages):
        recs = records_by_age.get(age, [])
        p_native, p_barnes = (_prad_arrays(recs) if recs
                              else (np.array([]), np.array([])))
        for j, (getx, xlabel) in enumerate(cols):
            ax = axes[i, j]
            if not recs:
                ax.text(0.5, 0.5, f"t = {age:g} Myr\n(no runs reach this age)",
                        ha="center", va="center", transform=ax.transAxes,
                        color="grey", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            xvals = np.array([getx(r) for r in recs], dtype=float)
            for p_arr, color in ((p_native, COLOR_NATIVE), (p_barnes, COLOR_BARNES)):
                xf, yf = _finite_positive(xvals, p_arr)
                if xf.size:
                    ax.scatter(xf, yf, s=28, color=color,
                               edgecolor="k", linewidth=0.4, alpha=0.85)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel(f"t = {age:g} Myr\n" r"$P_{\rm rad}/k$ [K cm$^{-3}$]")
            if i == nrows - 1:
                ax.set_xlabel(xlabel)

    handles = [
        Line2D([0], [0], marker="o", ls="", color=COLOR_NATIVE,
               markeredgecolor="k", label=r"TRINITY-native ($F_{\rm rad}/4\pi R_2^2$)"),
        Line2D([0], [0], marker="o", ls="", color=COLOR_BARNES,
               markeredgecolor="k", label=r"Barnes formula ($3L/4\pi r^2 c$)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Radiation pressure vs source properties (radius = {radius_key})",
                 y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="P_rad vs source properties for Barnes-matched TRINITY runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-F", "--folder", required=True,
                        help="Folder of TRINITY run subfolders (each with dictionary.jsonl)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: paper/plots)")
    parser.add_argument("--ages", nargs="+", type=float, default=list(DEFAULT_AGES_MYR),
                        help="Stellar ages [Myr], one plot row each (default: 0.5 1 3)")
    parser.add_argument("--radius", choices=["R2", "R_IF", "both"], default="both",
                        help="Radius variant on the x-axis of the radius column")
    args = parser.parse_args()

    outputs = load_runs(args.folder)
    if not outputs:
        print(f"No runs found under: {args.folder}")
        return
    print(f"Loaded {len(outputs)} run(s); ages = {args.ages} Myr")

    records_by_age = collect_age_records(outputs, args.ages)
    for age in args.ages:
        print(f"  t = {age:g} Myr: {len(records_by_age[age])} run(s) reach this age")

    out_dir = Path(args.output_dir) if args.output_dir else project_root() / "paper" / "plots"
    radii = ["R2", "R_IF"] if args.radius == "both" else [args.radius]
    for radius_key in radii:
        plot_figure(records_by_age, args.ages, radius_key,
                    out_dir / f"barnes26_PradSources_{radius_key}.pdf")


if __name__ == "__main__":
    main()
