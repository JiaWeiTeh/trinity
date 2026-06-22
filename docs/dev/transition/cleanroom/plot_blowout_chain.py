#!/usr/bin/env python3
"""Figure: the blowout -> cooling-ratio-RECOVERY causal chain (HYBR, early crossers).

PURE READ of the committed per-config CSVs (`data/c0_<cfg>_h0.csv`). For the early
blowout crossers the cooling ratio r = (Lgain - Lloss)/Lgain dips toward ~0.3 and
then RECOVERS (climbs back up) the moment the shell blows out of the cloud
(R2 > rCloud). This figure makes the four-link chain visible on one time axis per
config, with the blowout epoch marked:

    1. v2 (shell velocity) DECELERATES inside the cloud and hits its MINIMUM at
       blowout, then RE-ACCELERATES as it exits into the low-density ambient.
    2. Pb (bubble pressure) decay STEEPENS just after blowout (free expansion).
    3. Lloss (radiative loss) PEAKS at/just after blowout then COLLAPSES -- the
       emission measure n^2 V = (Pb/T0)^2 R2^3 turns over as R2^3 expansion can no
       longer outrun the n^2 dilution once the shell is in the dilute ambient.
    4. the cooling ratio r therefore bottoms out at blowout and RECOVERS (rises).

So the chain is  v2(min)->Pb(collapse)->Lloss(peak->collapse)->ratio(recovers).

Each row is one config. Left axis: v2 [km/s] (solid) and the cooling ratio r
(thick, normalised-friendly 0..1 scale shared with v2 via twin). Right axis:
Pb and Lloss, each normalised to its own in-window max so the shapes overlay on a
log-friendly linear 0..1 scale. The vertical dash-dot line is blowout
(R2 = rCloud), drawn by blowout_marker.

REPRODUCE (run from the cleanroom dir so blowout_marker import resolves):
    cd /home/user/trinity/docs/dev/transition/cleanroom && python plot_blowout_chain.py
Outputs: figures/blowout_chain.png, figures/blowout_chain.pdf
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blowout_marker import mark, t_blowout, rcloud

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False  # no LaTeX in this container

WONG = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# Early crossers -- where the dip is sharp and blowout-driven (see analysis).
CONFIGS = ["small_dense_highsfe", "simple_cluster", "midrange_pl0"]


def load(cfg):
    rows = []
    for r in csv.DictReader(open(HERE / "data" / f"c0_{cfg}_h0.csv")):
        try:
            t = float(r["t_now"])
        except (ValueError, KeyError):
            continue
        d = {"t": t}
        for k in ("Pb", "R2", "v2", "T0", "bubble_Lloss", "bubble_Lgain"):
            try:
                d[k] = float(r[k])
            except (ValueError, TypeError, KeyError):
                d[k] = float("nan")
        rows.append(d)
    rows.sort(key=lambda x: x["t"])
    return rows


def fin(x):
    return x == x and not math.isinf(x)


def ratio(r):
    g, l = r["bubble_Lgain"], r["bubble_Lloss"]
    return (g - l) / g if fin(g) and fin(l) and g != 0 else float("nan")


# Fixed colours per quantity (consistent across panels, unlike per-config WONG).
C_V2 = "#009E73"   # green   -- shell velocity v2
C_PB = "0.45"      # grey    -- bubble pressure Pb
C_LL = "#D55E00"   # vermilion -- radiative loss Lloss
C_R = "#0072B2"    # blue    -- cooling ratio


def main():
    n = len(CONFIGS)
    fig, axes = plt.subplots(n, 1, figsize=(7.6, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, cfg in enumerate(CONFIGS):
        ax = axes[i]
        rows = load(cfg)
        tb = t_blowout(cfg)
        rc = rcloud(cfg)

        # Window: from a fraction of t_b through the recovery (a few x t_b).
        lo, hi = tb * 0.25, min(tb * 5.0, tb + 0.25)
        win = [r for r in rows if lo <= r["t"] <= hi]
        t = [r["t"] for r in win]

        v2 = [r["v2"] for r in win]
        rr = [ratio(r) for r in win]
        pb = [r["Pb"] for r in win]
        ll = [r["bubble_Lloss"] for r in win]

        def norm(seq):
            vals = [x for x in seq if fin(x)]
            m = max(vals) if vals else 1.0
            return [x / m if fin(x) else float("nan") for x in seq], m

        v2n, v2max = norm(v2)
        pbn, pbmax = norm(pb)
        lln, llmax = norm(ll)

        # All on a shared 0..~1 "normalised" axis so the chain shapes overlay;
        # the cooling ratio is already O(0.3-0.7) so plotted as-is.
        ax.plot(t, v2n, color=C_V2, lw=2.0,
                label=f"v2  (peak {v2max:.0f} km/s)")
        ax.plot(t, pbn, color=C_PB, lw=1.4, ls="--",
                label=f"Pb  (peak {pbmax:.1e})")
        ax.plot(t, lln, color=C_LL, lw=1.6, ls=":",
                label=f"Lloss  (peak {llmax:.1e})")
        ax.plot(t, rr, color=C_R, lw=2.6, alpha=0.95,
                label="ratio r=(Lgain-Lloss)/Lgain")

        # mark v2 minimum, Lloss peak, ratio min
        def amin(seq):
            c = [(j, s) for j, s in enumerate(seq) if fin(s)]
            return min(c, key=lambda x: x[1])[0] if c else None

        def amax(seq):
            c = [(j, s) for j, s in enumerate(seq) if fin(s)]
            return max(c, key=lambda x: x[1])[0] if c else None

        jv = amin(v2n)
        jl = amax(lln)
        jr = amin(rr)
        if jv is not None:
            ax.plot(t[jv], v2n[jv], "v", color=C_V2, ms=8, mec="k", mew=0.6, zorder=5)
        if jl is not None:
            ax.plot(t[jl], lln[jl], "^", color=C_LL, ms=8, mec="k", mew=0.6, zorder=5)
        if jr is not None:
            ax.plot(t[jr], rr[jr], "o", color=C_R, ms=7, mec="k", mew=0.6, zorder=5)

        # blowout vertical line (dash-dot)
        mark(ax, cfg, color="0.2", label=True, lw=1.5)

        ax.set_ylim(0, 1.18)
        ax.set_ylabel("normalised")
        ax.set_xlim(lo, hi)
        ax.set_title(
            f"{cfg}   (t_blowout = {tb:.4f} Myr,  rCloud = {rc:.2f} pc)",
            fontsize=9, loc="left",
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.8, loc="lower center", ncol=2, framealpha=0.92,
                  handlelength=2.2, columnspacing=1.2)

    axes[0].text(0.015, 0.93,
                 "chain:  v2 min (v) at blowout -> Lloss peak (^) -> ratio min (o) -> recovers",
                 transform=axes[0].transAxes, fontsize=7.2, color="0.2", va="top")

    axes[-1].set_xlabel("t  [Myr]")
    fig.suptitle(
        "Blowout (R2 > rCloud) recovers the cooling ratio (HYBR):\n"
        "v2 re-accelerates  ->  Pb collapses  ->  Lloss collapses  ->  ratio rises",
        fontsize=10, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    outdir = HERE / "figures"
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"blowout_chain.{ext}", dpi=150, bbox_inches="tight")
    print("wrote", outdir / "blowout_chain.png", "and .pdf")


if __name__ == "__main__":
    main()
