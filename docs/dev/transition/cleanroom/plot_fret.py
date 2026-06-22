#!/usr/bin/env python3
"""Headline figure: the f_ret verdict plot (PLAN.md S6.5 #1).

f_ret(t) = Eb / integral(Lmech dt) for every completed full run, against the
literature retained-energy band (~0.01-0.1, Lancaster+2021 / Geen+2021) and the
Weaver energy-conserving value (5/11 ~ 0.45). The shape renders the verdict:
  - curves diving into the green band  -> TRINITY cools efficiently (trigger problem)
  - curves flat ABOVE the band         -> TRINITY under-cools (physics gap, S0.1 #2)

    python plot_fret.py docs/dev/transition/cleanroom/data/c0_*_st6.csv
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import mark

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False  # no LaTeX in this container

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
WEAVER = 5.0 / 11.0          # Weaver energy-conserving retained fraction
BAND = (0.01, 0.10)          # Lancaster+2021 / Geen+2021


def load(path):
    t, f, ph = [], [], []
    for r in csv.DictReader(open(path)):
        try:
            tn, fr = float(r["t_now"]), float(r["f_ret"])
        except (ValueError, TypeError, KeyError):
            continue
        if fr == fr and fr > 0:           # drop NaN / nonpositive
            t.append(tn); f.append(fr); ph.append(r.get("phase"))
    return t, f, ph


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_st6.csv")))
    if not paths:
        sys.exit("no CSVs given/found")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(*BAND, color="#009E73", alpha=0.18, zorder=0)
    ax.axhline(WEAVER, ls="--", lw=1.2, color="0.4", zorder=1)

    n = 0
    for i, p in enumerate(paths):
        name = Path(p).name.replace("c0_", "").replace("_st6.csv", "")
        t, f, ph = load(p)
        if not t:
            continue
        c = WONG[i % len(WONG)]
        ax.plot(t, f, color=c, lw=1.6, label=f"{name}  (end {f[-1]:.2f})")
        # blowout (R2 exits rCloud) as a star ON the f_ret curve; label only the first plotted run
        mark(ax, name, t, f, color=c, label=(n == 0))
        n += 1

    ax.set_yscale("log")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"retained energy  $f_{\rm ret}=E_b/\int L_{\rm mech}\,dt$")
    ax.set_title(f"Hot-bubble energy retention vs. observed band  ({n} full runs)")
    ax.text(0.98, BAND[1] * 1.15, "observed / 3D sims (Lancaster, Geen): 0.01-0.1",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color="#006b4f")
    ax.text(0.98, WEAVER * 1.05, "Weaver 5/11 (energy-conserving)",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8, color="0.4")
    ax.set_ylim(BAND[0] * 0.5, 1.0)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9, ncol=2)
    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fret_verdict.{ext}", dpi=150)
    print(f"wrote {outdir}/fret_verdict.(pdf,png) from {n} runs")


if __name__ == "__main__":
    main()
