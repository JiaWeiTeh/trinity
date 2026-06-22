#!/usr/bin/env python3
"""F0 pathology figure (PLAN.md S6.5 #2): why the current trigger never fires.

Top:    the cooling ratio (Lgain-Lloss)/Lgain for all configs vs the eps=0.05 threshold.
Bottom: Lmech_total (the Lgain denominator) -- the WR/SN surges that spike it.
Story:  the ratio plateaus at ~0.3-0.5, never reaching 0.05, and BUMPS UP at the
        ~3 Myr SN surge (Lgain spikes) -- the instantaneous metric resets away from
        the threshold exactly when cooling might catch up. Needs the H0-enriched CSVs
        (bubble_Lloss column); run on c0_*_h0.csv.

    python plot_f0path.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import mark

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False
WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def load(path):
    t, ratio, lm = [], [], []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            tn = float(r["t_now"]); Lg = float(r["Lmech_total"]); Ll = float(r["bubble_Lloss"])
        except (ValueError, TypeError, KeyError):
            continue
        if Lg > 0:
            t.append(tn); ratio.append((Lg - Ll) / Lg); lm.append(Lg)
    return t, ratio, lm


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    paths = [p for p in paths if load(p)[0]]
    if not paths:
        sys.exit("no usable H0 CSVs (need bubble_Lloss column -- run the h0 batch first)")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for i, p in enumerate(paths):
        name = Path(p).name.replace("c0_", "").replace("_h0.csv", "")
        c = WONG[i % len(WONG)]
        t, ratio, lm = load(p)
        a1.plot(t, ratio, color=c, lw=1.3, label=name)
        a2.plot(t, lm, color=c, lw=1.0)
        # blowout (R2 exits rCloud) as a star ON each panel's curve; label only the first (top axis)
        mark(a1, name, t, ratio, color=c, label=(i == 0))
        mark(a2, name, t, lm, color=c, label=False)
    a1.axhline(0.05, ls="--", lw=1.3, color="#D55E00")
    a1.text(0.99, 0.06, r"$\epsilon=0.05$ threshold (F0 fires below)", transform=a1.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#D55E00")
    a1.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    a1.set_ylim(-0.05, 1.0)
    a1.set_title("F0 never fires: cooling ratio plateaus ~0.5, never reaches 0.05 (resets up at the SN surge)",
                 fontsize=10)
    a1.legend(fontsize=7, ncol=3, loc="upper right", framealpha=0.9)
    a2.set_yscale("log"); a2.set_ylabel(r"$L_{\rm mech}$")
    a2.set_xlabel("time  [Myr]")
    fig.tight_layout()
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"f0_pathology.{ext}", dpi=150)
    print(f"wrote {out}/f0_pathology.(pdf,png) from {len(paths)} configs")


if __name__ == "__main__":
    main()
