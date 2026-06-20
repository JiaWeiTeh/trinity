#!/usr/bin/env python3
"""New-findings figure: the transition is GEOMETRIC, not thermal (PLAN.md S6 / table C).

For each config the only candidate trigger that fires at a physical epoch is F4
(blowout, R2 > rCloud). This plots that firing epoch against the cloud radius,
one point per config: a clean monotonic R2-crossing relation whose epoch is set
purely by cloud size, NOT by any cooling event (F0/F1/F3 never fire). The dashed
guide is t proportional to rCloud (slope 1 in log-log) through the span median.

Pure read of the committed data/c0_*_h0.csv (via harvest_h0); no re-run.

    python plot_blowout.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harvest_h0 import harvest  # noqa: E402

STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    pts = []
    for p in paths:
        a = harvest(p)
        rc, f4 = a.get("rCloud"), a.get("f4")
        if isinstance(rc, float) and isinstance(f4, float) and rc > 0 and f4 > 0:
            pts.append((a["name"], rc, f4))
    if not pts:
        sys.exit("no (rCloud, F4) points found")
    pts.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # per-config label nudges to avoid collisions (be_sphere/pl2_steep nearly coincide)
    NUDGE = {"be_sphere": (-6, 13, "bottom", "right"),
             "pl2_steep": (8, -6, "top", "left")}
    for i, (name, rc, f4) in enumerate(pts):
        ax.scatter(rc, f4, s=95, color=WONG[i % len(WONG)], edgecolor="white",
                   linewidth=0.7, zorder=3)
        dx, dy, va, ha = NUDGE.get(name, (8, -3, "top", "left"))
        ax.annotate(name, (rc, f4), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7.5, color="0.25", va=va, ha=ha)

    # slope-1 (t proportional to rCloud) guide through the geometric-mean anchor
    import math
    gx = math.exp(sum(math.log(rc) for _, rc, _ in pts) / len(pts))
    gy = math.exp(sum(math.log(f4) for _, _, f4 in pts) / len(pts))
    xs = [min(rc for _, rc, _ in pts) * 0.7, max(rc for _, rc, _ in pts) * 1.4]
    ax.plot(xs, [gy * (x / gx) for x in xs], ls="--", lw=1.1, color="0.55", zorder=1,
            label=r"$t_{\rm blowout}\propto r_{\rm cloud}$ (geometric)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"cloud radius  $r_{\rm cloud}$  [pc]")
    ax.set_ylabel(r"F4 blowout epoch  ($R_2>r_{\rm cloud}$)  [Myr]")
    ax.set_title("The only transition is geometric: blowout epoch tracks cloud size, not cooling")
    ax.text(0.02, 0.97, "F0 / F1 / F3 (cooling, force) never fire in any config",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#a33",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdeef0", ec="#f3ccd3"))
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"blowout_geometric.{ext}", dpi=150)
    print(f"wrote {outdir}/blowout_geometric.(pdf,png) from {len(pts)} configs")


if __name__ == "__main__":
    main()
