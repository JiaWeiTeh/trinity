#!/usr/bin/env python3
"""Diagnostic figure: WHAT MAKES bubble_Lloss RISE-THEN-COLLAPSE in the early dip.

The energy-implicit cooling-ratio dip (t < ~1 Myr, before any SN) is a swing in
bubble_Lloss: Lloss RISES into the dip (2-15x at ~flat Lmech) then COLLAPSES out
of it (the cooling ratio r=(Lmech-Lloss)/Lmech tracks it inversely). This figure
is a PURE READ of the committed per-config CSVs that decomposes the mechanism.

Hypothesis under test:
  Lloss ~ integral of n^2 Lambda(T) dV ~ n^2 V Lambda(T)  (emission measure x Lambda).
  We have only scalars T0, Pb, R2 per row, so use RELATIVE proxies:
    - interior density   n   ∝ Pb / T0     (ideal gas P = n k T, units relative)
    - bubble volume      V   ∝ R2^3
    - emission measure   EM  ∝ n^2 V = (Pb/T0)^2 R2^3   (Lambda ~ const, see below)

Finding the data supports (per config, all 6):
  * T0 falls MONOTONICALLY through the whole window (3-8e6 K) and never crosses
    the Lambda(T) peak (~1e5-1e6 K) -- so the rise is NOT "T entering the cooling
    peak". With T0 >> 1e6 K and varying only ~2x, Lambda(T) is ~flat here, so the
    Lloss shape is set by the emission measure n^2 V, not by Lambda(T).
  * EM = (Pb/T0)^2 R2^3 RISES then COLLAPSES and its peak lands within ~1.3x in
    time of the actual bubble_Lloss peak in every config.
  * The turnover is a competition of log-slopes: EM rises while volume growth
    (dlnV/dlnt ~ +1.6..+1.8) beats density dilution (dln n^2/dlnt ~ -1.3..-1.5);
    at the Lloss peak the two cross (their sum ~ 0); past it dilution wins and EM
    (hence Lloss) collapses as R2 keeps expanding.

So: the RISE is volume growth outrunning dilution; the COLLAPSE is R2 expansion
diluting n^2 faster than V grows. Pure geometry/dilution, not a thermal trigger.

Each row of the figure is one config:
  - left axis  : normalized bubble_Lloss (solid) and EM proxy n^2 V (dashed),
                 each scaled to its own in-window max so the shapes overlay.
  - right axis : T0 [K] (dotted), to show it falls monotonically and stays
                 far above the ~1e6 K Lambda peak (band shaded for reference).
  - markers    : Lloss peak (v) and cooling-ratio minimum (^) -- they coincide.

REPRODUCE:
    cd /home/user/trinity && \
      python docs/dev/transition/cleanroom/plot_dipmechanism.py \
        docs/dev/transition/cleanroom/data/c0_pl2_steep_h0.csv \
        docs/dev/transition/cleanroom/data/c0_simple_cluster_h0.csv \
        docs/dev/transition/cleanroom/data/c0_midrange_pl0_h0.csv
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

from blowout_marker import mark

HERE = Path(__file__).resolve().parent
# repo-root paper/_lib/trinity.mplstyle (parents[3] == /home/user/trinity)
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False  # no LaTeX in this container

# Wong palette (same ordering as the sibling plot_*.py generators).
WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
C_LLOSS = "#D55E00"  # vermilion -- the measured cooling loss
C_EM = "#0072B2"     # blue      -- the n^2 V emission-measure proxy
C_T0 = "#009E73"     # green     -- interior temperature

T_WINDOW = 1.5  # Myr; the early pre-SN dip lives well inside this


def _f(row, key):
    try:
        v = float(row[key])
    except (ValueError, TypeError, KeyError):
        return None
    return v if math.isfinite(v) else None


def load(path):
    """Pure read of implicit-phase rows with finite scalars, t in (0, T_WINDOW).

    Returns dict of parallel lists plus the relative proxies n, V, EM.
    """
    t, ll, lm, T0, Pb, R2, r = [], [], [], [], [], [], []
    for row in csv.DictReader(open(path)):
        tn = _f(row, "t_now")
        if tn is None or tn <= 0 or tn >= T_WINDOW:
            continue
        Lloss, Lmech = _f(row, "bubble_Lloss"), _f(row, "Lmech_total")
        T, P, R = _f(row, "T0"), _f(row, "Pb"), _f(row, "R2")
        if None in (Lloss, Lmech, T, P, R) or Lmech == 0 or T <= 0 or R <= 0:
            continue
        t.append(tn)
        ll.append(Lloss)
        lm.append(Lmech)
        T0.append(T)
        Pb.append(P)
        R2.append(R)
        r.append((Lmech - Lloss) / Lmech)
    n = [P / T for P, T in zip(Pb, T0)]            # relative interior density
    V = [R ** 3 for R in R2]                        # relative volume
    EM = [ni * ni * Vi for ni, Vi in zip(n, V)]     # relative emission measure
    return dict(t=t, ll=ll, lm=lm, T0=T0, Pb=Pb, R2=R2, r=r, n=n, V=V, EM=EM)


def _argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def _argmin(xs):
    return min(range(len(xs)), key=lambda i: xs[i])


def _logslope(t, y, i):
    """Centered d ln y / d ln t at index i (None at the ends)."""
    if i <= 0 or i >= len(y) - 1 or y[i - 1] <= 0 or y[i + 1] <= 0:
        return None
    return (math.log(y[i + 1]) - math.log(y[i - 1])) / (
        math.log(t[i + 1]) - math.log(t[i - 1])
    )


def main():
    paths = sys.argv[1:] or [
        str(HERE / "data" / f"c0_{c}_h0.csv")
        for c in ("pl2_steep", "simple_cluster", "midrange_pl0")
    ]
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        sys.exit("no CSVs given/found")

    nrows = len(paths)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.6, 2.75 * nrows + 0.7))
    if nrows == 1:
        axes = [axes]

    print("config                 t(Lloss pk)  t(EM pk)   T0@pk[K]   r-min t   "
          "EM rise/fall   slopes@pk (n^2 , V , sum)")
    for k, (ax, p) in enumerate(zip(axes, paths)):
        name = Path(p).name.replace("c0_", "").replace("_h0.csv", "")
        d = load(p)
        if not d["t"]:
            ax.set_visible(False)
            continue
        t = d["t"]
        iL = _argmax(d["ll"])
        iE = _argmax(d["EM"])
        irm = _argmin(d["r"])

        # normalize Lloss and EM each to its in-window max for shape overlay
        llmax, emmax = max(d["ll"]), max(d["EM"])
        lln = [v / llmax for v in d["ll"]]
        emn = [v / emmax for v in d["EM"]]

        # left axis: normalized Lloss (solid) and EM proxy (dashed)
        ax.plot(t, lln, color=C_LLOSS, lw=1.8, ls="-",
                label=r"$L_{\rm loss}$ (measured)")
        ax.plot(t, emn, color=C_EM, lw=1.6, ls="--",
                label=r"$n^2V=(P_b/T_0)^2R_2^3$ (proxy)")
        ax.plot(t[iL], lln[iL], marker="v", ms=8, color=C_LLOSS,
                mec="k", mew=0.5, ls="none", zorder=5,
                label=r"$L_{\rm loss}$ peak")
        ax.plot(t[irm], lln[irm], marker="^", ms=8, color="0.25",
                mec="k", mew=0.5, ls="none", zorder=5,
                label="cooling-ratio min")
        # per-config blowout time (shell exits the cloud); grey so it reads as a
        # separate annotation against the Lloss/EM curve colours. Label once.
        mark(ax, name, color="0.35", label=(k == 0))
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.18)
        ax.set_ylabel("normalized\n(own max)")
        ax.set_xlim(t[0], t[-1])

        # right axis: T0 in K, plus the Lambda(T) cooling-peak reference band
        axr = ax.twinx()
        axr.plot(t, d["T0"], color=C_T0, lw=1.5, ls=":",
                 label=r"$T_0$ (right)")
        axr.set_yscale("log")
        axr.set_ylabel(r"$T_0$  [K]", color=C_T0)
        axr.tick_params(axis="y", colors=C_T0)
        axr.axhspan(1e5, 1e6, color="0.6", alpha=0.18, zorder=0)
        axr.set_ylim(8e4, max(d["T0"]) * 1.6)

        ax.set_title(
            f"{name}:  rise = $V$ growth beats dilution,   "
            f"collapse = $R_2$ dilutes $n^2$",
            fontsize=10, pad=5,
        )
        # annotate the two competing log-slopes right at the Lloss-peak marker
        n2 = [ni * ni for ni in d["n"]]
        s_n2, s_V = _logslope(t, n2, iL), _logslope(t, d["V"], iL)
        if s_n2 is not None and s_V is not None:
            ax.annotate(
                f"at peak:\n"
                r"$d\ln V/d\ln t=$" + f"{s_V:+.1f}\n"
                r"$d\ln n^2/d\ln t=$" + f"{s_n2:+.1f}",
                xy=(t[iL], 1.0), xytext=(0.985, 0.62),
                textcoords="axes fraction", ha="right", va="center",
                fontsize=7.5, color="0.25",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="0.6", alpha=0.85),
            )

        # console table with the slope decomposition at the Lloss peak
        sn2, sV = s_n2, s_V
        rise = d["EM"][iE] / d["EM"][0]
        fall = d["EM"][-1] / d["EM"][iE]
        ssum = None if (sn2 is None or sV is None) else sn2 + sV
        print(f"{name:22} {t[iL]:11.4g} {t[iE]:9.4g} {d['T0'][iL]:9.3g} "
              f"{t[irm]:9.4g}   x{rise:4.1f} / x{fall:5.3f}   "
              f"{'' if sn2 is None else f'{sn2:+.2f}':>6} , "
              f"{'' if sV is None else f'{sV:+.2f}':>5} , "
              f"{'' if ssum is None else f'{ssum:+.2f}':>5}")

    axes[-1].set_xlabel("t  [Myr]")

    # one shared legend (left-axis handles + the right-axis T0 + the band)
    h, l = axes[0].get_legend_handles_labels()
    band = plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.18, ec="none")
    t0line = plt.Line2D([], [], color=C_T0, lw=1.5, ls=":")
    h += [t0line, band]
    l += [r"$T_0$ (right axis)", r"$\Lambda(T)$ peak band $10^5$-$10^6$ K"]
    fig.legend(h, l, fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"dip_mechanism.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nwrote {outdir}/dip_mechanism.(png,pdf) from {len(paths)} configs")


if __name__ == "__main__":
    main()
