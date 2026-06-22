#!/usr/bin/env python3
"""New-findings figure: root-fix sizing (PLAN.md S6.6 / mixcool_whatif.py).

The G0 verdict says the stall is under-cooling. Lancaster+2021 / El-Badry+2019
add a turbulent mixing-layer sink L_mix = theta*L_mech. This STATIC what-if on
the committed h0 CSVs (no re-run, ignores dynamical back-reaction) sizes theta by
asking the two mixcool_whatif questions, per config:

  top    : modified retained energy  f_ret_end - theta  -> does it enter 0.01-0.1?
  bottom : modified min cooling ratio  min_t (Lg - Ll - theta*Lg)/Lg
           -> does it cross 0.05 (a cooling transition F0 would now fire)?

Both land near the literature theta ~ 0.25 (dotted). This is a feasibility GATE,
not the validated result: the naive dynamical bulk-sink stalls the solver (it
drives dM/dt<0) and was reverted; the real fix integrates the sink into the
structure solve. Mirrors mixcool_whatif.py exactly.

    python plot_mixcool.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import apply_style, color

apply_style()

HERE = Path(__file__).resolve().parent

BAND = (0.01, 0.10)          # Lancaster+2021 / Geen+2021 observed retained-energy band
THETA_LIT = 0.25             # literature mixing-layer efficiency
THETAS = [i / 100 for i in range(0, 51)]   # 0.00 .. 0.50


def load(path):
    """implicit rows: (Lg=Lmech_total, Ll=bubble_Lloss) and the plateau f_ret_end."""
    rows, fret_end = [], None
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            Lg, Ll = float(r["Lmech_total"]), float(r["bubble_Lloss"])
        except (ValueError, TypeError, KeyError):
            continue
        if Lg > 0:
            rows.append((Lg, Ll))
            try:
                fr = float(r["f_ret"])
                if fr == fr and fr > 0:
                    fret_end = fr
            except (ValueError, TypeError, KeyError):
                pass
    return rows, fret_end


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    configs = []
    for p in paths:
        rows, fret_end = load(p)
        if rows and fret_end is not None:
            name = Path(p).name.replace("c0_", "").replace("_h0.csv", "")
            configs.append((name, rows, fret_end))
    if not configs:
        sys.exit("no implicit-phase data found")

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    # --- top: modified retained energy enters the observed band ---
    axT.axhspan(*BAND, color="#009E73", alpha=0.18, zorder=0)
    axT.axvline(THETA_LIT, ls=":", lw=1.2, color="0.45", zorder=1)
    for name, _rows, fret_end in configs:
        y = [max(fret_end - th, 1e-4) for th in THETAS]
        axT.plot(THETAS, y, color=color(name), lw=1.6, label=name)
    axT.set_yscale("log")
    axT.set_ylim(BAND[0] * 0.5, 0.6)
    axT.set_ylabel(r"retained energy  $f_{\rm ret}-\theta$")
    axT.set_title(r"Root-fix sizing: a $\theta\,L_{\rm mech}$ mixing-layer sink (static what-if)")
    axT.text(0.99, BAND[1] * 1.15, "observed band 0.01-0.1", transform=axT.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color="#006b4f")
    axT.text(THETA_LIT, 0.5, r" literature $\theta\approx0.25$", ha="left", va="top",
             fontsize=8, color="0.4")
    axT.legend(fontsize=7.5, loc="lower left", ncol=2, framealpha=0.9)

    # --- bottom: modified min cooling ratio crosses 0.05 (F0 would now fire) ---
    axB.axhline(0.05, ls="--", lw=1.2, color="#D55E00", zorder=1)
    axB.axvline(THETA_LIT, ls=":", lw=1.2, color="0.45", zorder=1)
    for name, rows, _fret_end in configs:
        ymin = []
        for th in THETAS:
            ymin.append(min((Lg - Ll - th * Lg) / Lg for Lg, Ll in rows))
        axB.plot(THETAS, ymin, color=color(name), lw=1.6)
    axB.set_ylim(-0.15, 1.0)
    axB.set_xlabel(r"mixing-layer efficiency  $\theta$   ($L_{\rm mix}=\theta\,L_{\rm mech}$)")
    axB.set_ylabel(r"min cooling ratio $\frac{L_{\rm mech}-L_{\rm loss}-\theta L_{\rm mech}}{L_{\rm mech}}$")
    axB.text(0.99, 0.055, "F0 fires below 0.05", transform=axB.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color="#a64500")
    axB.text(0.01, 0.02, "naive dynamical bulk-sink reverted (drives dM/dt<0, stalls hybr); "
             "integrate into the structure solve", transform=axB.transAxes, ha="left",
             va="bottom", fontsize=7.5, color="#a33",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdeef0", ec="#f3ccd3"))
    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"mixcool_rootfix.{ext}", dpi=150)
    print(f"wrote {outdir}/mixcool_rootfix.(pdf,png) from {len(configs)} configs")


if __name__ == "__main__":
    main()
