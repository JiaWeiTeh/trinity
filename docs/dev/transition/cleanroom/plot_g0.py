#!/usr/bin/env python3
"""G0 divergence figure (PLAN.md S6 / table C): per config, a timeline showing where
each candidate transition family WOULD fire, vs the Eb-peak oracle and the implicit
phase span. The story: the cooling/force families (F0,F1,F3) never fire, F2 fires
absurdly early, and only F4 (blowout, R2>rCloud) lands at a physical epoch.

    python plot_g0.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import glob, sys
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

# family -> (marker, color, label)
FAM = {
    "eb_zero": ("*", "#000000", "Eb-peak (oracle)"),
    "f0":      ("o", "#999999", "rate-ratio (current)"),
    "f1":      ("s", "#56B4E9", "cumulative cooling"),
    "f2":      ("v", "#E69F00", "cooling timescale"),
    "f3":      ("D", "#CC79A7", "force balance"),
    "f4":      ("P", "#D55E00", "blowout (R2>rCloud)"),
}


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    res = [harvest(p) for p in paths]
    res = [a for a in res if a["t_end"]]
    n = len(res)
    if not n:
        sys.exit("no data")
    fig, ax = plt.subplots(figsize=(7.6, 0.7 * n + 2.0))
    names = [a["name"] for a in res]
    for y, a in enumerate(res):
        ax.plot([0, a["t_end"]], [y, y], color="0.8", lw=6, solid_capstyle="butt", zorder=0)
        for key in ("eb_zero", "f0", "f1", "f2", "f3", "f4"):
            t = a[key][0.25] if key == "f1" else (a[key][1] if key == "f2" else a[key])
            mk, col, _ = FAM[key]
            if isinstance(t, float):
                ax.scatter(t, y, marker=mk, color=col, s=70, zorder=3,
                           edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(-0.6, n - 0.4); ax.invert_yaxis()
    ax.set_xlabel("time  [Myr]   (grey bar = full run; markers = candidate firing epoch)")
    ax.set_title("Only the blowout trigger fires at a physical epoch", fontsize=10)
    handles = [plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                          markersize=9, label=lab) for k, (mk, col, lab) in FAM.items()]
    ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), framealpha=0.9)
    # note the "never" families per config
    NEVER_LABEL = {"f0": "rate-ratio", "f1": "cumulative", "f3": "force"}
    for y, a in enumerate(res):
        nev = [NEVER_LABEL[k] for k in ("f0", "f1", "f3")
               if not isinstance(a[k][0.25] if k == "f1" else a[k], float)]
        if nev:
            ax.text(a["t_end"] + 0.1, y, "never fire: " + ", ".join(nev), fontsize=6,
                    va="center", color="0.5")
    fig.tight_layout()
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"g0_divergence.{ext}", dpi=150)
    print(f"wrote {out}/g0_divergence.(pdf,png) from {n} configs")


if __name__ == "__main__":
    main()
