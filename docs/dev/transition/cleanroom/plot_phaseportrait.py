#!/usr/bin/env python3
"""New-findings figure: the (delta, beta) phase portrait (user Q follow-up, 2026-06-20).

The surge_coincidence diagnostic showed the F0-ratio surge co-moves with beta
DROPPING (re-pressurisation) but at no fixed beta/delta value. This 2-D view makes
that geometric: every implicit-phase row of all six runs, plotted in (delta, beta)
space and coloured by time. The beta<0 half (re-pressurisation / negative velocity
structure) is shaded; the beta+delta=0 diagonal is drawn so one can read whether
re-pressurisation sits on any fixed beta+delta contour (it does not).

The story by shape: beta<0 points are NOT a tight cluster at one (beta,delta) or
one beta+delta value -- they sweep a band, and they light up at the late-time
(SN-epoch) colours where delta is large/positive. Re-pressurisation is a feedback
event, not a structure threshold.

Pure read of the committed data/c0_*_st6.csv; no re-run.

    python plot_phaseportrait.py docs/dev/transition/cleanroom/data/c0_*_st6.csv
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
STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); b = float(r["cool_beta"]); d = float(r["cool_delta"])
        except (ValueError, KeyError, TypeError):
            continue
        if b == b and d == d:
            out.append((t, d, b))
    return out


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_st6.csv")))
    rows = [pt for p in paths for pt in load(p)]
    if not rows:
        sys.exit("no implicit-phase (beta,delta) rows found")
    t = [x[0] for x in rows]; d = [x[1] for x in rows]; b = [x[2] for x in rows]
    nneg = sum(1 for bb in b if bb < 0)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    dlo, dhi = min(d), max(d)
    ax.axhspan(min(b) - 0.5, 0, color="#D55E00", alpha=0.08, zorder=0)        # beta<0 region
    ax.axhline(0, color="#D55E00", lw=1.1, ls="--", zorder=2)
    ax.plot([dlo, dhi], [-dlo, -dhi], color="0.55", lw=1.0, ls=":", zorder=2,
            label=r"$\beta+\delta=0$")
    sc = ax.scatter(d, b, c=t, cmap="viridis", s=10, alpha=0.65, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("time  [Myr]")

    ax.set_xlabel(r"$\delta \equiv (t/T)\,dT/dt$   (interior heating rate)")
    ax.set_ylabel(r"$\beta \equiv -(t/P_b)\,dP_b/dt$   ($\beta<0$: $P_b$ rising, re-pressurising)")
    ax.set_title("(δ, β) phase portrait: β<0 re-pressurisation has no fixed threshold",
                 fontsize=10.5)
    ax.text(0.015, 0.04, f"β<0: {nneg}/{len(rows)} rows ({100*nneg/len(rows):.0f}%), a wide δ band at\n"
            "late (SN-epoch) times — no fixed β / δ / (β+δ) value",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color="0.3",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff6ec", ec="#f6dcbd"))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_ylim(min(b) - 0.3, max(b) + 0.3)
    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"betadelta_portrait.{ext}", dpi=150)
    print(f"wrote {outdir}/betadelta_portrait.(pdf,png) from {len(rows)} rows "
          f"({nneg} with β<0) across {len(paths)} configs")


if __name__ == "__main__":
    main()
