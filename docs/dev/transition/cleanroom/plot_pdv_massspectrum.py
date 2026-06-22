#!/usr/bin/env python3
"""PdV across the mass spectrum: the same term that drives the stall (typical clouds)
drives the large-cloud Eb collapse (user Q, 2026-06-21, linking failed-large-clouds).

The energy budget dEb/dt = Lmech - Lcool - PdV (PdV = 4*pi*R2^2*v2*Pb) has ONE control
parameter, PdV/Lmech, because Lcool is small in both regimes:
  - typical clouds (my six): PdV/Lmech ~0.55-0.70 < 1 -> Eb GROWS, energy-driven; the
    transition would need cooling, which never catches up -> the STALL.
  - massive clusters (5e9): the shell free-expands at ~2000-3700 km/s so PdV/Lmech > 1
    -> Eb PEAKS then COLLAPSES through zero into negative -> R1->R2 -> NaN crash.

Panel A: fail_repro Eb(t) -- the collapse (reliable stored state, from the committed
  failed-large-clouds budget CSV).
Panel B: max PdV/Lmech per config -- the regime sorter, boundary at 1. My six are computed
  from the implicit-phase h0 CSVs (past the IC relaxation, reliable); the two large-cloud
  values are the failed-large-clouds PLAN's reliable post-relaxation numbers (the naive
  per-snapshot recompute is unreliable in the early free-streaming rows -- see that PLAN).

    python plot_pdv_massspectrum.py
"""
from __future__ import annotations

import csv
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False
FOURPI = 4 * math.pi
ROOT = HERE.parents[3]
# reliable post-IC-relaxation values from docs/dev/failed-large-clouds/PLAN.md (cited, not recomputed)
LARGE = [("small_1e6\n(healthy)", 0.95, "large-cloud budget"),
         ("fail_repro 5e9\n(collapses)", 1.56, "large-cloud budget")]


def my_maxpdv(p):
    mx = 0.0
    for r in csv.DictReader(open(p)):
        if r.get("phase") != "implicit":
            continue
        try:
            R2 = float(r["R2"]); v2 = float(r["v2"]); Pb = float(r["Pb"]); Lm = float(r["Lmech_total"])
            if Lm > 0:
                mx = max(mx, FOURPI * R2 * R2 * v2 * Pb / Lm)
        except (ValueError, KeyError, TypeError):
            continue
    return mx


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    # Panel A: fail_repro Eb collapse
    fr = list(csv.DictReader(open(ROOT / "docs/dev/failed-large-clouds/data/budget_fail_repro.csv")))
    t = [float(r["t"]) * 1e3 for r in fr]; Eb = [float(r["Eb"]) / 1e9 for r in fr]
    tpk = t[Eb.index(max(Eb))]
    axA.axhline(0, color="k", lw=1.0)
    axA.plot(t, Eb, color="#D55E00", lw=2.2, marker=".", ms=4)
    axA.fill_between(t, Eb, 0, where=[e < 0 for e in Eb], color="#b30000", alpha=0.25)
    axA.axvline(tpk, color="0.5", ls="--", lw=1.0)
    axA.text(tpk, max(Eb) * 0.5, "  Eb peak\n  (PdV/Lmech→1)", fontsize=8, color="0.3")
    axA.text(t[-1], Eb[-1] - 0.15, "Eb<0\nR1→R2→NaN\ncrash", fontsize=8, color="#b30000",
             ha="right", va="top")
    axA.set_xlabel(r"t  [$10^{-3}$ Myr]"); axA.set_ylabel(r"$E_b$  [$10^9$ code units]")
    axA.set_title("Massive cluster (5e9): PdV>Lmech, Eb peaks then collapses NEGATIVE",
                  fontsize=10)

    # Panel B: max PdV/Lmech per config, regime sorter
    mine = []
    for p in sorted(glob.glob(str(HERE / "data" / "c0_*_h0.csv"))):
        name = Path(p).stem.replace("c0_", "").replace("_h0", "")
        mine.append((name, my_maxpdv(p), "typical (clean-room)"))
    allcfg = sorted(mine, key=lambda x: x[1]) + sorted(LARGE, key=lambda x: x[1])
    y = range(len(allcfg))
    colors = ["#009E73" if v < 1 else "#b30000" for _, v, _ in allcfg]
    axB.barh(list(y), [v for _, v, _ in allcfg], color=colors, alpha=0.85,
             edgecolor="0.3", linewidth=0.5)
    axB.axvline(1.0, color="k", ls="--", lw=1.4)
    axB.text(1.02, 0.3, "PdV/Lmech = 1\n(Eb-peak)", fontsize=8, rotation=0, va="bottom")
    axB.set_yticks(list(y)); axB.set_yticklabels([n for n, _, _ in allcfg], fontsize=8)
    axB.set_xlabel(r"max PdV$/L_{\rm mech}$")
    axB.set_xlim(0, 1.8)
    axB.text(0.42, 2.4, "Eb GROWS\n(energy-driven → stall)", ha="center",
             fontsize=8.5, color="#006b4f")
    axB.text(1.30, len(allcfg) - 1.7, "Eb COLLAPSES\n(momentum from birth\n→ crash)", ha="center",
             fontsize=8.5, color="#b30000")
    axB.set_title("One control parameter sorts the regimes:\nbelow 1 = stall, above 1 = collapse",
                  fontsize=10)

    fig.suptitle("The PdV term at both ends of the mass spectrum — same physics, opposite failure "
                 "(Lcool is small in both: ~1% large, ~50% typical but cooling never catches up)",
                 fontsize=11)
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pdv_massspectrum.{ext}", dpi=150)
    print(f"wrote {out}/pdv_massspectrum.(pdf,png)")


if __name__ == "__main__":
    main()
