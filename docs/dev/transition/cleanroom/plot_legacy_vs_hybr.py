#!/usr/bin/env python3
"""Legacy vs hybr comparison of the dip / cooling structure (user Q, 2026-06-21):
"do the dip diagnostic for the legacy runs too, so I can see what actually changed."

For the early-crossing configs, overlay the legacy (--solver legacy, BEFORE) and hybr
(AFTER) runs through the dip, three quantities per config:
  - cooling ratio (Lg-Ll)/Lg  -- legacy crosses 0.05 (transitions); hybr recovers
  - bubble_Lloss              -- legacy keeps cooling; hybr's collapses out of the dip
  - cool_beta                 -- THE difference: legacy clamped to [0,1]; hybr unbounded
T0 is ~identical between the two (not the difference), so it is annotated, not a panel.

Pure read of committed data/c0_*_legacy.csv (BEFORE) + data/c0_*_h0.csv (AFTER).

    python plot_legacy_vs_hybr.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"  # parents[3]=repo root
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

CONFIGS = ["small_dense_highsfe", "pl2_steep", "simple_cluster"]  # early crossers, clean dips
LEG, HYB = "#0072B2", "#D55E00"  # Wong blue (legacy) / vermillion (hybr)


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); b = float(r["cool_beta"]); T0 = float(r["T0"])
            Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
        except (ValueError, KeyError, TypeError):
            continue
        if Lg > 0:
            out.append(dict(t=t, b=b, T0=T0, Lloss=Ll, ratio=(Lg - Ll) / Lg))
    return out


def main():
    n = len(CONFIGS)
    fig, axes = plt.subplots(n, 3, figsize=(12.5, 2.5 * n + 0.5), squeeze=False)
    for i, cfg in enumerate(CONFIGS):
        leg = load(HERE / "data" / f"c0_{cfg}_legacy.csv")
        hyb = load(HERE / "data" / f"c0_{cfg}_h0.csv")
        if not leg or not hyb:
            continue
        # window to the dip/crossing region (a bit past the legacy crossing)
        tcross = next((d["t"] for d in leg if d["ratio"] < 0.05), None)
        thi = (tcross * 4) if tcross else 0.6
        ar, al, ab = axes[i]

        for d, c, ls, lab in [(leg, LEG, "-", "legacy"), (hyb, HYB, "--", "hybr")]:
            w = [x for x in d if x["t"] <= thi]
            ar.plot([x["t"] for x in w], [x["ratio"] for x in w], color=c, ls=ls, lw=1.8, label=lab)
            al.plot([x["t"] for x in w], [x["Lloss"] for x in w], color=c, ls=ls, lw=1.8)
            ab.plot([x["t"] for x in w], [x["b"] for x in w], color=c, ls=ls, lw=1.8)

        ar.axhline(0.05, color="k", ls=":", lw=1.0)
        if tcross:
            ar.plot([tcross], [0.05], "v", color=LEG, ms=9, zorder=5)
            ar.text(tcross, 0.10, f"legacy\ncrosses\n{tcross:.3f}", color=LEG, fontsize=7,
                    ha="center", va="bottom")
        ar.set_ylabel(f"{cfg}\n\ncooling ratio", fontsize=8.5)
        ar.set_ylim(-0.15, 1.0)
        al.set_ylabel(r"$L_{\rm loss}$", fontsize=9); al.set_yscale("log")
        ab.axhspan(0, 1, color="0.6", alpha=0.18)  # legacy clamp box beta in [0,1]
        ab.axhline(0, color="0.6", lw=0.7)
        ab.set_ylabel(r"$\beta$ (clamp [0,1] shaded)", fontsize=8.5)
        T0l = leg[min(range(len(leg)), key=lambda k: abs(leg[k]["t"] - (tcross or 0.1)))]["T0"]
        T0h = hyb[min(range(len(hyb)), key=lambda k: abs(hyb[k]["t"] - (tcross or 0.1)))]["T0"]
        ab.text(0.97, 0.95, f"T0@cross  leg {T0l:.1e}\n        hybr {T0h:.1e} K  (same order)",
                transform=ab.transAxes, ha="right", va="top", fontsize=6.8, color="0.3")
        for ax in axes[i]:
            ax.set_xscale("log")
        if i == 0:
            ar.legend(fontsize=8, loc="upper right")
            ar.set_title("cooling ratio: legacy crosses 0.05, hybr recovers", fontsize=9)
            al.set_title(r"$L_{\rm loss}$: legacy keeps cooling, hybr collapses", fontsize=9)
            ab.set_title(r"$\beta$: legacy clamped [0,1], hybr unbounded → +", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("t  [Myr]")
    fig.suptitle("What changed, legacy → hybr: comparable T0, but the β-clamp drives the divergence "
                 "(legacy ratio→crossing, hybr ratio→recovery)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"legacy_vs_hybr.{ext}", dpi=150)
    print(f"wrote {out}/legacy_vs_hybr.(pdf,png) for {CONFIGS}")


if __name__ == "__main__":
    main()
