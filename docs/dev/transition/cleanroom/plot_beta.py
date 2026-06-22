#!/usr/bin/env python3
"""beta / (beta+delta) re-pressurisation + compression plot.

CORRECTION (2026-06-21, per docs/dev/archive/betadelta/): the structural quantity
that drives the interior velocity is NOT beta alone, and NOT beta+delta=0 -- it is
the dv/dr source term (beta+delta)/t = -t*dln(n)/dt, whose INFLOW trigger sits at
beta+delta <~ -0.4 (steep) .. -0.5. beta<0 alone is only RE-PRESSURISATION (Pb
rising); delta>0 (T rising) partly cancels it, so beta+delta is the physical
compression term. This plot shows both, with the -0.4 inflow trigger marked: beta
dives deep (re-pressurisation) but beta+delta mostly stays ABOVE -0.4 (no real
compression/inflow). The archive's inflow is in any case "real but cosmetic"
(subsonic, ~1e-6 of thermal).

    python plot_beta.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv, glob, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import apply_style, mark

apply_style()

HERE = Path(__file__).resolve().parent

BPD_TRIGGER = -0.4  # beta+delta inflow trigger (archive: -0.4 hunt .. -0.5 steep)


def load(path):
    t, b, d, lm = [], [], [], []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            tn = float(r["t_now"]); be = float(r["cool_beta"]); de = float(r["cool_delta"])
        except (ValueError, TypeError, KeyError):
            continue
        t.append(tn); b.append(be); d.append(de)
        try: lm.append(float(r["Lmech_total"]))
        except (ValueError, TypeError): lm.append(float("nan"))
    return t, b, d, lm


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    paths = [p for p in paths if load(p)[0]]
    n = len(paths)
    if not n:
        sys.exit("no usable CSVs")
    fig, axes = plt.subplots(n, 1, figsize=(7.6, 1.8 * n + 0.6), sharex=True, squeeze=False)
    for i, (ax, p) in enumerate(zip(axes[:, 0], paths)):
        name = Path(p).name.replace("c0_", "").replace("_h0.csv", "").replace("_st6.csv", "")
        t, b, d, lm = load(p)
        bpd = [bi + di for bi, di in zip(b, d)]
        fneg = sum(1 for x in b if x < 0) / len(b)
        ntrig = sum(1 for x in bpd if x < BPD_TRIGGER)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axhline(BPD_TRIGGER, color="k", ls="--", lw=1.0)  # inflow trigger
        ax.plot(t, b, color="#0072B2", lw=1.1, label=r"$\beta$ ($P_b$ rate)")
        ax.plot(t, bpd, color="#D55E00", lw=1.7, label=r"$\beta+\delta$ (compression)")
        # shade only where beta+delta actually dips below the inflow trigger
        ax.fill_between(t, bpd, BPD_TRIGGER, where=[x < BPD_TRIGGER for x in bpd],
                        color="#b30000", alpha=0.35, interpolate=True)
        ax.set_ylabel(r"$\beta,\ \beta+\delta$")
        # blowout (R2 exits rCloud) as a star ON the beta+delta compression curve, in
        # that curve's colour; label only top row
        mark(ax, name, t, bpd, color="#D55E00", label=(i == 0))
        ax.text(0.99, 0.05, f"{name}   β<0: {fneg:.0%}   rows β+δ<−0.4: {ntrig}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, lm, color="0.55", lw=1.0, ls=":")
        ax2.set_yscale("log"); ax2.set_ylabel(r"$L_{\rm mech}$", color="0.55", fontsize=8)
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[0, 0].set_title(r"$\beta$ (re-pressurisation, $P_b\!\uparrow$) vs $\beta+\delta$ "
                         r"(compression; inflow trigger $\beta+\delta\lesssim-0.4$, dashed)")
    axes[-1, 0].set_xlabel("time  [Myr]")
    fig.tight_layout()
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"beta_repressurization.{ext}", dpi=150)
    print(f"wrote {out}/beta_repressurization.(pdf,png) from {n} configs "
          f"(beta vs beta+delta, inflow trigger {BPD_TRIGGER})")


if __name__ == "__main__":
    main()
