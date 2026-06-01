#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pedrini+2026 emergence-timescale: Weaver-scaling diagnostic.

Plots tau_TOT against M_cloud for one or more sweeps where n_core scales as
M_cloud^alpha at fixed sfe. The Weaver+77 wind-driven bubble reaches
R_cloud at

    tau_TOT propto R_cloud^(5/3) (rho/L)^(1/3),

with R_cloud propto (M_cloud / n_core)^(1/3) propto M_cloud^((1-alpha)/3),
rho propto n_core propto M_cloud^alpha, and L propto M_* = sfe M_cloud,
which combines to

    tau_TOT propto M_cloud^(2(1-alpha)/9).

Reference exponents for the three natural scalings:

    alpha = 0   (n constant)        -> tau propto M^(2/9)   ≈ +0.222
    alpha = 1/2 (mild covariance)   -> tau propto M^(1/9)   ≈ +0.111
    alpha = 1   (R_cloud fixed)     -> tau propto M^0       flat

If TRINITY lands on these slopes across all three sweeps, the wind-driven
kernel is intact and the Pedrini gap is a radiation-pressure-physics issue.
Departure from the predicted exponents implicates the cloud setup or another
non-Weaver effect.

Usage
-----
Run from project root:

    python trinity/_plots/pedrini_weaver_scaling.py \\
        --sweep outputs/pedrini_sweep_mcloud_only 0 \\
        --sweep outputs/pedrini_sweep_larson 0.5 \\
        --sweep outputs/pedrini_sweep_density 1.0

Each --sweep takes two arguments: the sweep output directory and the
density-scaling exponent alpha. The script prints, per sweep, the measured
log-log slope alongside the Weaver prediction so the diagnosis is one
scroll away from the run command.

Output (under <FIG_DIR>/pedrini_weaver_scaling/):
    - pedrini_weaver_scaling.pdf
    - pedrini_weaver_scaling_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trinity._plots.plot_base import FIG_DIR
from trinity._plots.pedrini_emergence_timescales import (
    collect_all,
    WONG,
    STYLE_PATH,
)


# ---------------------------------------------------------------------------
# Weaver prediction
# ---------------------------------------------------------------------------

def weaver_exponent(alpha: float) -> float:
    """Predicted log-log slope of tau_TOT vs M_cloud for n_core propto M_cloud^alpha."""
    return 2.0 * (1.0 - alpha) / 9.0


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit log10(y) = a + b log10(x); return (slope b, intercept a)."""
    logx = np.log10(x)
    logy = np.log10(y)
    b, a = np.polyfit(logx, logy, 1)
    return float(b), float(a)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    sweep_data: list[tuple[str, float, list[dict]]],
    out_pdf: Path,
) -> list[dict]:
    """Plot tau_TOT vs M_cloud, one line per sweep, with Weaver overlays.

    Returns a list of per-sweep summary rows for CSV output.
    """
    plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots()

    summary = []

    for i, (label, alpha, rows) in enumerate(sweep_data):
        clean = [r for r in rows
                 if r["breakout"] and not r["recollapse"]]
        if len(clean) < 2:
            print(f"[weaver_scaling] {label}: <2 usable runs, skipping")
            continue
        clean.sort(key=lambda r: r["mCloud"])

        M = np.array([r["mCloud"]  for r in clean], dtype=float)
        tau = np.array([r["tau_TOT"] for r in clean], dtype=float)

        color = WONG[i % len(WONG)]
        b_pred = weaver_exponent(alpha)

        # Fit measured slope
        try:
            b_meas, _ = fit_loglog_slope(M, tau)
        except Exception as e:
            print(f"[weaver_scaling] {label}: slope fit failed ({e})")
            b_meas = float("nan")

        print(f"[weaver_scaling] {label}: "
              f"measured slope = {b_meas:+.3f},  "
              f"Weaver predicts {b_pred:+.3f}  "
              f"(alpha = {alpha:.2g})")

        # Measured curve: solid line + markers
        ax.plot(M, tau, marker="o", linestyle="-", color=color,
                label=(fr"$\alpha = {alpha:.2g}$, "
                       fr"meas $= {b_meas:+.3f}$, "
                       fr"pred $= {b_pred:+.3f}$"))

        # Weaver prediction: dashed line anchored at the lowest-M data point
        tau_pred = tau[0] * (M / M[0]) ** b_pred
        ax.plot(M, tau_pred, linestyle="--", color=color, alpha=0.5)

        summary.append({
            "sweep":         label,
            "alpha":         alpha,
            "n_runs":        len(clean),
            "slope_meas":    b_meas,
            "slope_pred":    b_pred,
            "M_cloud_min":   float(M.min()),
            "M_cloud_max":   float(M.max()),
            "tau_min_Myr":   float(tau.min()),
            "tau_max_Myr":   float(tau.max()),
        })

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm cloud}$ $[M_\odot]$")
    ax.set_ylabel(r"$\tau_{\rm TOT}$ [Myr]")

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", linestyle="--",
                          label="Weaver+77 prediction"))
    ax.legend(handles=handles, loc="best", fontsize="small")

    fig.savefig(out_pdf)
    print(f"Saved: {out_pdf}")
    plt.close(fig)

    return summary


def write_summary_csv(summary: list[dict], out_path: Path) -> None:
    if not summary:
        return
    fieldnames = list(summary[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"Wrote summary: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sweep", action="append", nargs=2,
                    metavar=("SWEEP_DIR", "ALPHA"),
                    required=True,
                    help="Sweep directory and density-scaling exponent "
                         "(n_core propto M_cloud^alpha). Pass once per sweep.")
    ap.add_argument("--output_pdf", type=Path, default=None,
                    help="Output PDF path. "
                         "Default: <FIG_DIR>/pedrini_weaver_scaling/"
                         "pedrini_weaver_scaling.pdf")
    args = ap.parse_args()

    sweep_data = []
    for sweep_dir_str, alpha_str in args.sweep:
        sweep_dir = Path(sweep_dir_str).resolve()
        if not sweep_dir.is_dir():
            ap.error(f"--sweep dir not found: {sweep_dir}")
        try:
            alpha = float(alpha_str)
        except ValueError:
            ap.error(f"alpha must be numeric, got: {alpha_str!r}")
        print(f"\n[weaver_scaling] Loading {sweep_dir.name} (alpha = {alpha})")
        rows = collect_all(sweep_dir, show_tau_pdr=False)
        if not rows:
            print(f"[weaver_scaling] No runs in {sweep_dir}, skipping")
            continue
        sweep_data.append((sweep_dir.name, alpha, rows))

    if not sweep_data:
        ap.error("No sweeps loaded.")

    out_pdf = args.output_pdf
    if out_pdf is None:
        out_dir = FIG_DIR / "pedrini_weaver_scaling"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / "pedrini_weaver_scaling.pdf"
    else:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

    summary = make_plot(sweep_data, out_pdf)
    write_summary_csv(summary, out_pdf.with_suffix("").with_name(
        out_pdf.stem + "_summary.csv"))


if __name__ == "__main__":
    main()
