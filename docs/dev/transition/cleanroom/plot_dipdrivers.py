#!/usr/bin/env python3
"""Diagnostic figure: what drives the early cooling-trigger dip-then-surge.

The energy-implicit "cooling trigger" ratio r = (Lgain - Lloss)/Lgain shows an
EARLY-TIME DIP (t < ~1 Myr, before any supernovae) followed by a SURGE back up,
in the hybr runs. This figure is a PURE READ of the committed per-config CSVs
that makes the cause visible across three stacked panels:

  1. the trigger r(t) itself — the dip, then the surge back above threshold.
  2. its two ingredients: L_gain (solid) stays ~flat while L_loss (dashed)
     RISES into the dip and COLLAPSES coming out — so the dip is a cooling
     swing, not a feedback swing.
  3. the mechanical budget: wind |L_mech,W| (solid) vs SN |L_mech,SN| (dashed).
     SN onset is ~3 Myr — far later than the early dip — confirming the early
     dip is a pre-SN, pure-cooling phenomenon (the late plateau is SN-sustained).

No physics is re-derived here; every curve is a column read straight off the CSV.

REPRODUCE:
    cd /home/user/trinity && python docs/dev/transition/cleanroom/plot_dipdrivers.py \
        docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv
import glob
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[2] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False  # no LaTeX in this container

# Same Wong palette as plot_fret.py — keep per-config colours consistent across figures.
WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"]
THRESHOLD = 0.05  # transition trigger threshold


def _f(row, key):
    """Parse a float field; return None if missing / unparseable / non-finite."""
    try:
        v = float(row[key])
    except (ValueError, TypeError, KeyError):
        return None
    return v if math.isfinite(v) else None


def load(path):
    """Pure read: all rows with t_now>0. Per row, keep whatever fields are finite.

    Returns parallel lists; an entry is None where that field was nan/missing in
    the CSV (energy-phase rows have no bubble luminosities, etc.).
    """
    t, r, lg, ll, lw, lsn = [], [], [], [], [], []
    for row in csv.DictReader(open(path)):
        tn = _f(row, "t_now")
        if tn is None or tn <= 0:
            continue
        lgain, lloss = _f(row, "bubble_Lgain"), _f(row, "bubble_Lloss")
        ratio = None
        if lgain is not None and lloss is not None and lgain != 0:
            ratio = (lgain - lloss) / lgain
        t.append(tn)
        r.append(ratio)
        lg.append(lgain)
        ll.append(lloss)
        w, sn = _f(row, "Lmech_W"), _f(row, "Lmech_SN")
        lw.append(abs(w) if w is not None else None)
        lsn.append(abs(sn) if sn is not None else None)
    return t, r, lg, ll, lw, lsn


def _xy(t, y):
    """Drop points where y is None, keeping x/y aligned (so log axes don't choke)."""
    xs, ys = [], []
    for ti, yi in zip(t, y):
        if yi is not None and yi > 0:
            xs.append(ti)
            ys.append(yi)
    return xs, ys


def _xy_signed(t, y):
    """Like _xy but keeps any finite y (used for the linear trigger panel)."""
    xs, ys = [], []
    for ti, yi in zip(t, y):
        if yi is not None:
            xs.append(ti)
            ys.append(yi)
    return xs, ys


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    if not paths:
        sys.exit("no CSVs given/found")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7.6, 8.0))

    dip_depths = {}
    n = 0
    for i, p in enumerate(paths):
        name = Path(p).name.replace("c0_", "").replace("_h0.csv", "")
        t, r, lg, ll, lw, lsn = load(p)
        if not t:
            continue
        c = WONG[i % len(WONG)]

        # Panel 1: cooling trigger r(t).
        xr, yr = _xy_signed(t, r)
        if xr:
            ax1.plot(xr, yr, color=c, lw=1.5, label=name)
            dip_depths[name] = min(yr)

        # Panel 2: L_gain (solid) and L_loss (dashed).
        xg, yg = _xy(t, lg)
        xl, yl = _xy(t, ll)
        ax2.plot(xg, yg, color=c, lw=1.5, ls="-")
        ax2.plot(xl, yl, color=c, lw=1.3, ls="--")

        # Panel 3: |L_mech,wind| (solid) and |L_mech,SN| (dashed).
        xw, yw = _xy(t, lw)
        xs, ys = _xy(t, lsn)
        ax3.plot(xw, yw, color=c, lw=1.5, ls="-")
        ax3.plot(xs, ys, color=c, lw=1.3, ls="--")
        n += 1

    # Panel 1 cosmetics.
    ax1.axhline(THRESHOLD, ls=":", lw=1.2, color="0.4",
                label=f"transition threshold {THRESHOLD:g}")
    ax1.set_title("1.  cooling trigger  (it dips early, then surges back up)",
                  pad=8)
    ax1.set_ylabel(r"trigger  $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$")
    ax1.set_ylim(bottom=0.0)  # keep the 0.05 threshold line in frame

    # Panel 2 cosmetics.
    ax2.set_yscale("log")
    ax2.set_title(r"2.  the dip is $L_{\rm loss}$ rising, the surge is "
                  r"$L_{\rm loss}$ collapsing  ($L_{\rm gain}$ ~ flat)", pad=8)
    ax2.set_ylabel(r"$L_{\rm gain}$ (—),  $L_{\rm loss}$ (- -)")

    # Panel 3 cosmetics.
    ax3.set_yscale("log")
    ax3.set_title("3.  mechanical feedback: SN onset (~3 Myr) is late — "
                  "the early dip is pre-SN", pad=8)
    ax3.set_ylabel(r"$|L_{\rm mech}|$:  wind (—),  SN (- -)")
    ax3.set_xlabel("t  [Myr]")
    ax3.set_xscale("log")

    # One shared config legend above the top panel (does not cover data).
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 4),
               framealpha=0.9)

    fig.tight_layout()

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"dip_drivers.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {outdir}/dip_drivers.(pdf,png) from {n} configs")
    for name, depth in sorted(dip_depths.items()):
        print(f"  {name:24s} min trigger r = {depth:+.4f}")


if __name__ == "__main__":
    main()
