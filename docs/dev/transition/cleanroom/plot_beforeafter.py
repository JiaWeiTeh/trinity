#!/usr/bin/env python3
"""New-findings figure: BEFORE (legacy) vs AFTER (hybr) cooling-balance trigger.

The implicit->momentum hand-off fires when the cooling ratio r=(Lgain-Lloss)/Lgain
drops below 0.05. Two panels, shared y, log t:
  BEFORE (legacy, clamped beta in [0,1]): r drives monotonically down to 0.05 at the
     FIRST cooling episode and the run transitions -> a dot marks each crossing.
  AFTER  (hybr, unbounded root + dMdt>0 gate): the SAME early cooling episode shows
     up as a DIP, but r recovers (Lloss collapses as the bubble expands) and never
     reaches 0.05 -> no crossing, the run stalls in implicit.

Pure read of committed CSVs:
  BEFORE  data/c0_<name>_legacy.csv   (betadelta_solver=legacy; c0_consistency.py --solver legacy)
  AFTER   data/c0_<name>_h0.csv       (betadelta_solver=hybr)
Configs with no legacy CSV yet are simply skipped in the BEFORE panel.

    python plot_beforeafter.py
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
THRESH = 0.05


def ratio(path):
    """(t, r) over rows with a finite ratio; drop momentum-phase rows (post-transition)."""
    t, r = [], []
    for row in csv.DictReader(open(path)):
        if row.get("phase") == "momentum":
            continue
        try:
            tn = float(row["t_now"]); Lg = float(row["bubble_Lgain"]); Ll = float(row["bubble_Lloss"])
        except (ValueError, KeyError, TypeError):
            continue
        if Lg > 0 and tn > 0:
            t.append(tn); r.append((Lg - Ll) / Lg)
    return t, r


def first_crossing(t, r):
    for tt, rr in zip(t, r):
        if rr < THRESH:
            return tt, rr
    return None


def main():
    # config order = the hybr (AFTER) runs we have; match legacy by name
    after = sorted(glob.glob(str(DATA / "c0_*_h0.csv")))
    names = [Path(p).name.replace("c0_", "").replace("_h0.csv", "") for p in after]

    fig, (axB, axA) = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    nlegacy = 0
    for i, name in enumerate(names):
        col = WONG[i % len(WONG)]
        # AFTER (hybr)
        ta, ra = ratio(DATA / f"c0_{name}_h0.csv")
        if ta:
            axA.plot(ta, ra, color=col, lw=1.4)
        # BEFORE (legacy) if present
        leg = DATA / f"c0_{name}_legacy.csv"
        if leg.exists():
            tb, rb = ratio(leg)
            if tb:
                nlegacy += 1
                axB.plot(tb, rb, color=col, lw=1.4, label=name)
                x = first_crossing(tb, rb)
                if x:
                    axB.scatter([x[0]], [THRESH], color=col, s=45, zorder=5,
                                edgecolor="0.2", linewidth=0.6)
        else:
            axA.plot([], [], color=col, label=name)  # keep legend complete

    for ax, title in ((axB, "BEFORE (legacy)"), (axA, "AFTER (hybr)")):
        ax.axhline(THRESH, ls="--", lw=1.1, color="0.5")
        ax.set_xscale("log")
        ax.set_xlabel("t  [Myr]")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.02)
        ax.set_xlim(2e-3, 7)        # common range so BEFORE/AFTER align
    axB.set_ylabel(r"cooling trigger  $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    axB.text(0.97, THRESH + 0.015, "transition threshold 0.05", transform=axB.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color="0.45")
    handles, labels = axB.get_legend_handles_labels()
    if len(labels) < len(names):                       # fall back to AFTER legend
        handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels) or 1,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cooling-balance trigger over the evolution  "
                 "(below 0.05 → implicit transitions to momentum)", y=1.10, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"before_after.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/before_after.(pdf,png)  [{nlegacy}/{len(names)} legacy curves, "
          f"{len(names)} hybr curves]")
    if nlegacy == 0:
        print("WARNING: no c0_*_legacy.csv found yet -- BEFORE panel is empty; "
              "run c0_consistency.py --solver legacy first")


if __name__ == "__main__":
    main()
