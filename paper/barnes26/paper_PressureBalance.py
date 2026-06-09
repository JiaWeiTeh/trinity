#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Barnes 2026 comparison — pressure balance against the ambient ISM.

TRINITY-only (no Barnes overlay yet). Three rows of panels:

    row 0 : P_ISM (x) vs P_tot (y), log-log [K cm^-3], with the P_tot=P_ISM line
    row 1 : P_tot (x) vs log10(P_tot) - log10(P_ISM) (y), with the zero line
    row 2 : Sigma_gas (x) vs log10(P_tot) - log10(P_ISM) (y), with the zero line

The over/under-pressure rows use ``log10(P_tot) - log10(P_ISM)`` (= the dex
ratio log10(P_tot/P_ISM)) on a *linear* y-axis: both pressures are positive so
each log is defined, and the difference crosses zero (y > 0 over-pressured,
y < 0 under-pressured) without dropping any points. ``Sigma_gas`` is the
environmental cloud gas surface density ``mCloud/(pi rCloud^2)`` [Msun/pc^2].

``P_tot = P_thermal + P_radiation`` where P_thermal is TRINITY's HII
ionization-balance pressure ``P_HII`` and P_radiation is the radiation
pressure (native ``F_rad/(4 pi R2^2)``; ``--prad`` selects native, the
Barnes-formula recompute, or both). Runs whose ``PISM`` is absent or
non-positive are skipped (a Barnes P_ISM = P_DE comparison needs P_ISM > 0).

Modes
-----
* ``--population`` (default): synthesize a bubble population at a single
  ``--t-obs`` (see ``paper.barnes26._population``) and show it as a hexbin
  density + median per row (single column); the primary ``--prad`` mode is the
  density, any other mode is a median line. Output:
  ``barnes26_PressureBalance_{prad}_population.pdf``.
* ``--no-population``: one marker per run, one column per ``--ages`` value.
  Output: ``barnes26_PressureBalance_{prad}.pdf``.

The PISM (Barnes' P_DE) sweep is handled automatically: each PISM value is a
separate environment, combined into one population (row 0 then shows a column
of points per P_DE).

Usage
-----
  # both figures via the driver (population mode by default):
  python paper/barnes26/make_figures.py -F outputs/<sweep>

  # this figure directly, with synthesis knobs:
  python -m paper.barnes26.paper_PressureBalance -F outputs/<sweep> --prad both --t-obs 5 --cmf-slope -1.7

  # per-run / fixed-age fallback:
  python -m paper.barnes26.paper_PressureBalance -F outputs/<sweep> --no-population --ages 0.5 1 3

  python -m paper.barnes26.paper_PressureBalance --help   # full option list
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.barnes26._barnes_lib import (  # noqa: E402
    DEFAULT_AGES_MYR, load_runs, collect_age_records,
    to_Pk, pism_to_Pk, p_rad_native, p_rad_barnes, sigma_gas, project_root,
    apply_trinity_style, scatter_median_by_env, pde_env_labels,
)
from paper.barnes26._population import (  # noqa: E402
    synthesize_population, add_population_cli,
)

apply_trinity_style()  # trinity.mplstyle, without plot_base's stray-fig/ side effect

# Okabe-Ito colour-blind-safe palette + redundant marker shapes, so the two
# P_rad series are distinguishable in greyscale / full colour-blindness.
PRAD_STYLES = {
    "native": dict(color="#0072B2", marker="o", label=r"$P_{\rm rad}$ native"),
    "barnes": dict(color="#D55E00", marker="^", label=r"$P_{\rm rad}$ Barnes formula"),
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


def _logratio(Ptot, PISM):
    """log10(P_tot) - log10(P_ISM) = log10(P_tot/P_ISM), NaN where either <= 0."""
    Ptot = np.asarray(Ptot, dtype=float)
    PISM = np.asarray(PISM, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((Ptot > 0) & (PISM > 0),
                        np.log10(Ptot) - np.log10(PISM), np.nan)


# y-axis label shared by the two over/under-pressure rows.
_LOGRATIO_LABEL = r"$\log_{10}P_{\rm tot}-\log_{10}P_{\rm ISM}$"


def plot_figure(records_by_age, ages, prad_modes, out_path):
    ncols = len(ages)
    fig, axes = plt.subplots(
        nrows=3, ncols=ncols,
        figsize=(3.4 * ncols, 8.4),
        squeeze=False,
    )

    for j, age in enumerate(ages):
        ax_abs, ax_pt, ax_sig = axes[0, j], axes[1, j], axes[2, j]
        recs_all = records_by_age.get(age, [])
        recs, n_skipped = _with_positive_pism(recs_all)

        ax_abs.set_title(f"t = {age:g} Myr"
                         + (f"  ({n_skipped} skipped: P_ISM<=0)" if n_skipped else ""),
                         fontsize=9)

        if not recs:
            for ax in (ax_abs, ax_pt, ax_sig):
                ax.text(0.5, 0.5, "no runs with P_ISM > 0",
                        ha="center", va="center", transform=ax.transAxes,
                        color="grey", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            continue

        PISM, series = _ptot_series(recs, prad_modes)
        Sigma = sigma_gas(np.array([r["mCloud"] for r in recs], dtype=float),
                          np.array([r["rCloud"] for r in recs], dtype=float))

        # --- row 0: P_ISM vs P_tot (log-log) ---
        all_pos = [PISM]
        for mode, Ptot in series.items():
            st = PRAD_STYLES[mode]
            m = np.isfinite(PISM) & np.isfinite(Ptot) & (PISM > 0) & (Ptot > 0)
            ax_abs.scatter(PISM[m], Ptot[m], s=28, color=st["color"], marker=st["marker"],
                           edgecolor="k", linewidth=0.4, alpha=0.85)
            all_pos.append(Ptot[m])
        lims = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in all_pos if a.size])
        if lims.size:
            lo, hi = lims.min(), lims.max()
            ax_abs.plot([lo, hi], [lo, hi], color="0.4", ls="--", lw=1.0, zorder=1)
        ax_abs.set_xscale("log"); ax_abs.set_yscale("log")
        ax_abs.grid(True, which="both", alpha=0.25, lw=0.5)
        if j == 0:
            ax_abs.set_ylabel(r"$P_{\rm tot}/k$ [K cm$^{-3}$]")
        ax_abs.set_xlabel(r"$P_{\rm ISM}/k$ [K cm$^{-3}$]")

        # --- rows 1 & 2: log10(P_tot)-log10(P_ISM) (linear y) vs P_tot / Sigma_gas ---
        for ax, xvals, xlabel in (
            (ax_pt,  None,  r"$P_{\rm tot}/k$ [K cm$^{-3}$]"),
            (ax_sig, Sigma, r"$\Sigma_{\rm gas}$ [M$_\odot$ pc$^{-2}$]"),
        ):
            for mode, Ptot in series.items():
                st = PRAD_STYLES[mode]
                y = _logratio(Ptot, PISM)
                x = Ptot if xvals is None else xvals
                m = np.isfinite(x) & np.isfinite(y) & (x > 0)
                ax.scatter(x[m], y[m], s=28, color=st["color"], marker=st["marker"],
                           edgecolor="k", linewidth=0.4, alpha=0.85)
            ax.set_xscale("log")
            # dashed equilibrium line at log-ratio = 0 splits the regimes; keep it
            # in view so over- (y>0) vs under-pressured (y<0) is always visible.
            ax.axhline(0.0, color="0.35", ls="--", lw=1.3, zorder=1)
            ymin, ymax = ax.get_ylim()
            lo, hi = min(ymin, 0.0), max(ymax, 0.0)
            pad = 0.08 * (hi - lo) if hi > lo else 0.5
            ax.set_ylim(lo - pad, hi + pad)
            ax.grid(True, which="both", alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel(_LOGRATIO_LABEL)
                ax.text(0.03, 0.97, "over-pressured", transform=ax.transAxes,
                        va="top", ha="left", fontsize=7, style="italic", color="0.4")
                ax.text(0.03, 0.03, "under-pressured", transform=ax.transAxes,
                        va="bottom", ha="left", fontsize=7, style="italic", color="0.4")
            ax.set_xlabel(xlabel)

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


def _finish_population(fig, handles, info, n_skipped, out_path):
    leg = [Line2D([0], [0], marker="o", ls="", color=c, markersize=7, label=lab)
           for lab, c in handles]
    fig.legend(handles=leg, loc="upper center", ncol=max(2, len(leg)), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    extra = f"  ({n_skipped} skipped: P_ISM<=0)" if n_skipped else ""
    fig.suptitle("Pressure balance — synthetic population "
                 f"(N={info['n_surviving']}, "
                 rf"$t_{{\rm obs}}$={info['t_obs']:g} Myr){extra}",
                 y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_population(records, info, prad_modes, out_path):
    """Population mode: 3 rows at one t_obs, coloured by P_DE environment.

    P_tot uses the primary --prad mode (native by default). Each panel is a
    transparent scatter coloured by ambient pressure (P_DE) with a
    per-environment median line.
    """
    recs, n_skipped = _with_positive_pism(records)
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 10.2), squeeze=False)
    ax_abs, ax_pt, ax_sig = axes[0, 0], axes[1, 0], axes[2, 0]

    if not recs:
        for ax in (ax_abs, ax_pt, ax_sig):
            ax.text(0.5, 0.5, "no bubbles with P_ISM > 0", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        _finish_population(fig, [], info, n_skipped, out_path)
        return

    PISM, series = _ptot_series(recs, prad_modes)
    Ptot = series[prad_modes[0]]
    Sigma = sigma_gas(np.array([r["mCloud"] for r in recs], dtype=float),
                      np.array([r["rCloud"] for r in recs], dtype=float))
    env = np.array([r["PISM"] for r in recs], dtype=float)
    labels = pde_env_labels(recs)

    # --- row 0: P_ISM vs P_tot (log-log), coloured by env; P_tot=P_ISM line ---
    scatter_median_by_env(ax_abs, PISM, Ptot, env, xscale="log", yscale="log",
                          env_labels=labels, median=False)
    pos = np.concatenate([PISM, Ptot])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.size:
        ax_abs.plot([pos.min(), pos.max()], [pos.min(), pos.max()],
                    color="0.4", ls="--", lw=1.0, zorder=1)
    ax_abs.set_xscale("log"); ax_abs.set_yscale("log")
    ax_abs.grid(True, which="both", alpha=0.25, lw=0.5)
    ax_abs.set_ylabel(r"$P_{\rm tot}/k$ [K cm$^{-3}$]")
    ax_abs.set_xlabel(r"$P_{\rm ISM}/k$ [K cm$^{-3}$]")

    # --- rows 1 & 2: log10(P_tot)-log10(P_ISM) (linear y) vs P_tot / Sigma_gas ---
    ylr = _logratio(Ptot, PISM)
    handles = []
    for ax, xvals, xlabel in (
        (ax_pt,  Ptot,  r"$P_{\rm tot}/k$ [K cm$^{-3}$]"),
        (ax_sig, Sigma, r"$\Sigma_{\rm gas}$ [M$_\odot$ pc$^{-2}$]"),
    ):
        handles = scatter_median_by_env(ax, xvals, ylr, env, xscale="log",
                                        yscale="linear", env_labels=labels)
        ax.set_xscale("log")
        ax.axhline(0.0, color="0.35", ls="--", lw=1.3, zorder=1)
        ymin, ymax = ax.get_ylim()
        lo, hi = min(ymin, 0.0), max(ymax, 0.0)
        pad = 0.08 * (hi - lo) if hi > lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.set_ylabel(_LOGRATIO_LABEL)
        ax.text(0.03, 0.97, "over-pressured", transform=ax.transAxes,
                va="top", ha="left", fontsize=7, style="italic", color="0.4")
        ax.text(0.03, 0.03, "under-pressured", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=7, style="italic", color="0.4")
        ax.set_xlabel(xlabel)

    _finish_population(fig, handles, info, n_skipped, out_path)


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
    add_population_cli(parser)
    args = parser.parse_args()

    outputs = load_runs(args.folder)
    if not outputs:
        print(f"No runs found under: {args.folder}")
        return
    prad_modes = ["native", "barnes"] if args.prad == "both" else [args.prad]
    out_dir = Path(args.output_dir) if args.output_dir else project_root() / "paper" / "plots"

    if args.population:
        records, info = synthesize_population(
            outputs, t_obs=args.t_obs, n_bubble=args.n_bubble, cmf_slope=args.cmf_slope,
            sfe_median=args.sfe_median, sfe_sigma_dex=args.sfe_sigma_dex,
            fixed_sfe=args.fixed_sfe, fixed_ncore=args.fixed_ncore, seed=args.seed,
        )
        print(f"Population: {info['n_surviving']}/{info['n_bubble']} bubbles survived "
              f"(t_obs={args.t_obs:g} Myr); prad={args.prad}")
        if not records:
            print("No surviving bubbles — check grid coverage / t_obs.")
            return
        plot_population(records, info, prad_modes,
                        out_dir / f"barnes26_PressureBalance_{args.prad}_population.pdf")
        return

    print(f"Loaded {len(outputs)} run(s); ages = {args.ages} Myr; prad = {args.prad}")
    records_by_age = collect_age_records(outputs, args.ages)
    plot_figure(records_by_age, args.ages, prad_modes,
                out_dir / f"barnes26_PressureBalance_{args.prad}.pdf")


if __name__ == "__main__":
    main()
