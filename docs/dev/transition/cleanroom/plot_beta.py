#!/usr/bin/env python3
"""beta re-pressurisation plot (PLAN.md S6.5 #6): per config, beta(t) with beta<0
shaded (Pb rising = re-pressurisation) and Lmech_total overlaid -- do the negative-beta
excursions line up with feedback (WR/SN) luminosity surges?

    python plot_beta.py docs/dev/transition/cleanroom/data/c0_*_st6.csv
"""
from __future__ import annotations

import csv, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False


def load(path):
    t, b, lm, ph = [], [], [], []
    for r in csv.DictReader(open(path)):
        try:
            tn = float(r["t_now"]); be = float(r["cool_beta"])
        except (ValueError, TypeError, KeyError):
            continue
        if r.get("phase") != "implicit":
            continue
        t.append(tn); b.append(be)
        try: lm.append(float(r["Lmech_total"]))
        except (ValueError, TypeError): lm.append(float("nan"))
    return t, b, lm


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_st6.csv")))
    paths = [p for p in paths if load(p)[0]]
    n = len(paths)
    if not n:
        sys.exit("no usable CSVs")
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.7 * n + 0.6), sharex=True, squeeze=False)
    for ax, p in zip(axes[:, 0], paths):
        name = Path(p).name.replace("c0_", "").replace("_st6.csv", "")
        t, b, lm = load(p)
        fneg = sum(1 for x in b if x < 0) / len(b)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(t, b, color="#0072B2", lw=1.3, label=r"$\beta$")
        ax.fill_between(t, b, 0, where=[x < 0 for x in b], color="#D55E00", alpha=0.4,
                        interpolate=True)
        ax.set_ylabel(r"$\beta$", color="#0072B2")
        ax.text(0.99, 0.05, f"{name}  (beta<0: {fneg:.0%})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, lm, color="0.45", lw=1.0, ls=":")
        ax2.set_yscale("log"); ax2.set_ylabel(r"$L_{\rm mech}$", color="0.45", fontsize=8)
    axes[0, 0].set_title(r"$\beta(t)$ (red = $\beta<0$, $P_b$ rising) vs mechanical luminosity")
    axes[-1, 0].set_xlabel("time  [Myr]")
    fig.tight_layout()
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"beta_repressurization.{ext}", dpi=150)
    print(f"wrote {out}/beta_repressurization.(pdf,png) from {n} configs")


if __name__ == "__main__":
    main()
