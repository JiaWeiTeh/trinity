#!/usr/bin/env python3
"""New-findings figure: the (delta, beta) phase portrait (user Q follow-up, 2026-06-20).

CORRECTION (2026-06-21, per docs/dev/archive/betadelta/): the structural quantity
that governs the interior velocity is the dv/dr source (beta+delta)/t = -t*dln(n)/dt,
with an inflow trigger at beta+delta <~ -0.4 -- NOT beta alone, NOT beta+delta=0.
This 2-D view: every implicit-phase row of all six runs in (delta, beta) space,
coloured by time. beta<0 (re-pressurisation, Pb rising) is shaded; the REAL
threshold beta+delta=-0.4 (inflow) is drawn (the beta+delta=0 line is only a faint
reference).

The story by shape: beta dives well below 0 at the SN epoch (re-pressurisation),
but the points stay ABOVE the beta+delta=-0.4 line because delta>0 (T rising)
offsets beta -- so net compression/inflow almost never triggers. Re-pressurisation
(beta<0) is common; compression (beta+delta<-0.4) is not.

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

from blowout_marker import apply_style

apply_style()

HERE = Path(__file__).resolve().parent


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
    ntrig = sum(1 for x in rows if x[1] + x[2] < -0.4)  # beta+delta < -0.4 inflow trigger

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    dlo, dhi = min(d), max(d)
    ax.axhspan(min(b) - 0.5, 0, color="#D55E00", alpha=0.07, zorder=0)        # beta<0 region
    ax.axhline(0, color="#D55E00", lw=1.0, ls="--", zorder=2)
    ax.plot([dlo, dhi], [-dlo, -dhi], color="0.7", lw=0.9, ls=":", zorder=2,
            label=r"$\beta+\delta=0$ (ref)")
    # the REAL structural threshold: beta+delta = -0.4 (interior-inflow trigger); below = compression
    ax.plot([dlo, dhi], [-0.4 - dlo, -0.4 - dhi], color="k", lw=1.4, ls="--", zorder=2,
            label=r"$\beta+\delta=-0.4$ (inflow trigger)")
    ax.fill_between([dlo, dhi], [-0.4 - dlo, -0.4 - dhi], min(b) - 0.5,
                    color="#b30000", alpha=0.10, zorder=1)
    sc = ax.scatter(d, b, c=t, cmap="viridis", s=10, alpha=0.65, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("time  [Myr]")

    ax.set_xlabel(r"$\delta \equiv (t/T)\,dT/dt$   (interior heating rate)")
    ax.set_ylabel(r"$\beta \equiv -(t/P_b)\,dP_b/dt$   ($\beta<0$: $P_b$ rising, re-pressurising)")
    ax.set_title("(δ, β) portrait: β<0 (re-pressurisation) is common; β+δ<−0.4 (inflow) is not",
                 fontsize=10)
    ax.text(0.015, 0.04, f"β<0: {nneg}/{len(rows)} rows ({100*nneg/len(rows):.0f}%) — wide δ band, "
            f"SN-epoch.\nβ+δ<−0.4 (compression/inflow): only {ntrig} rows — δ>0 offsets β.",
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
