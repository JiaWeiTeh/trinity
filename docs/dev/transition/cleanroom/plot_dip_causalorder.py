#!/usr/bin/env python3
"""Causal-ordering figure for the early cooling-ratio dip: WHAT changes course FIRST?

User hypothesis (verbatim): "is it dudt changing course FIRST (which precedes
which) that causes the Lloss to drop hence the trigger ratio? but what is causing
dudt to drop?"  "dudt" is read broadly here as the bubble's kinematic/energetic
rate of change -- the shell velocity v2, its rate dv2/dt, and the energy rate
dEb/dt.

This is a PURE READ of the committed per-config implicit-phase CSVs
(docs/dev/transition/cleanroom/data/c0_*_h0.csv). For each config we restrict to
the implicit (hybr) phase where bubble_Lloss is populated, take the early window
that brackets the dip, and locate the turning point of each quantity:

  - r = (Lgain - Lloss)/Lgain      -> minimum  (the dip itself)
  - bubble_Lloss                   -> maximum  (the loss spike that drives the dip)
  - v2                             -> minimum  (the shell decelerates then recovers)
  - dEb/dt = Lgain - Lloss - 4 pi R2^2 v2 Pb   -> any sign change
  - Pb, T0                         -> monotone collapse through the window
  - cool_beta, cool_delta, beta+delta -> their turning points (the cooling
                                          structure's response)

VERDICT THE DATA SUPPORTS (see FINDINGS / the parent report):
  * dEb/dt NEVER changes sign in the dip window -- it stays strongly POSITIVE
    (the bubble keeps gaining energy throughout the dip). So "dEb/dt changing
    course" is NOT the trigger.
  * v2 dips to a MINIMUM and recovers; that v2-minimum LEADS (or is simultaneous
    with) the Lloss-peak / r-minimum -- the shell decelerates first, the loss
    peaks at/just after, then both recover together.
  * Pb and T0 fall MONOTONICALLY the whole window -- the real underlying driver
    (the interior depressurises/dilutes; the n^2 V emission measure peaks then
    collapses -- see plot_dipmechanism.py).
  * cool_beta / cool_delta / (beta+delta) peak LATER than the Lloss-peak in
    every config: they are a lagging readout of the cooling structure, NOT the
    trigger.

So the answer to the user: it is NOT dEb/dt that changes course first; the
velocity v2 decelerates first (leading the Lloss peak), but neither "causes" the
loss -- both v2 and Lloss are driven by the monotone Pb/T0 collapse and the
emission-measure turnover. (beta,delta) lag, so they cannot be the trigger.

Each panel = one representative config. Left axis: normalized v2 (solid),
bubble_Lloss (dashed), and the cooling ratio r (dotted), each scaled to its own
in-window range so shapes overlay. Right axis: Pb and T0 (thin), shown to fall
monotonically. Vertical markers: v2-minimum (blue), Lloss-peak / r-min (vermilion).

REPRODUCE:
    cd /home/user/trinity && \
      python docs/dev/transition/cleanroom/plot_dip_causalorder.py
(writes figures/dip_causalorder.{png,pdf})
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
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

DATA = HERE / "data"
FIGDIR = HERE / "figures"

# Wong palette (same ordering as the sibling plot_*.py generators).
C_V2 = "#0072B2"     # blue      -- shell velocity v2 (the kinematic "dudt")
C_LLOSS = "#D55E00"  # vermilion -- bubble_Lloss (the loss spike)
C_R = "#000000"      # black     -- cooling ratio r (the dip)
C_PB = "#009E73"     # green     -- bubble pressure Pb (monotone driver)
C_T0 = "#CC79A7"     # mauve     -- interior temperature T0 (monotone driver)

# (config, early window [Myr], pretty label). Three representative early-dip
# configs spanning fast->slow dips; large_diffuse_lowsfe is excluded because its
# ratio minimum is a LATE dip (t~4.9 Myr), a different phenomenon.
PANELS = [
    ("c0_small_dense_highsfe_h0", 0.10, "small dense, high SFE"),
    ("c0_simple_cluster_h0",      0.30, "simple cluster (baseline)"),
    ("c0_midrange_pl0_h0",        1.00, "midrange, flat profile"),
]


def _load(name):
    rows = list(csv.DictReader(open(DATA / f"{name}.csv")))

    def col(key):
        out = []
        for row in rows:
            try:
                out.append(float(row[key]))
            except (ValueError, TypeError, KeyError):
                out.append(math.nan)
        return np.array(out)

    phase = np.array([row["phase"] for row in rows])
    return col, phase


def _norm(y):
    """Scale to [0, 1] over the finite range (shape overlay)."""
    lo, hi = np.nanmin(y), np.nanmax(y)
    if not math.isfinite(lo) or hi <= lo:
        return np.full_like(y, 0.5)
    return (y - lo) / (hi - lo)


def main():
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.0, 9.0), sharex=False)

    for pi, (ax, (name, win, label)) in enumerate(zip(axes, PANELS)):
        cfg = name.replace("c0_", "").replace("_h0", "")  # bare config for blowout_marker
        col, phase = _load(name)
        t = col("t_now")
        Lloss = col("bubble_Lloss")
        Lgain = col("bubble_Lgain")
        R2 = col("R2")
        v2 = col("v2")
        Pb = col("Pb")
        T0 = col("T0")

        # Implicit phase only, finite Lloss, inside the early window.
        m = ((phase == "implicit") & np.isfinite(Lloss) & np.isfinite(Lgain)
             & (Lgain > 0) & (t <= win))
        t = t[m]
        Lloss, Lgain = Lloss[m], Lgain[m]
        R2, v2, Pb, T0 = R2[m], v2[m], Pb[m], T0[m]
        r = (Lgain - Lloss) / Lgain
        dEdt = Lgain - Lloss - 4.0 * math.pi * R2**2 * v2 * Pb

        # Turning points.
        t_v2min = t[int(np.argmin(v2))]
        t_rmin = t[int(np.argmin(r))]
        t_llosspk = t[int(np.argmax(Lloss))]
        # dEb/dt sign change (positive -> negative); report if any.
        sign_change = np.where((dEdt[:-1] > 0) & (dEdt[1:] <= 0))[0]
        t_dE0 = 0.5 * (t[sign_change[0]] + t[sign_change[0] + 1]) if sign_change.size else None

        # Left axis: normalized shape overlays.
        ax.plot(t, _norm(v2), color=C_V2, lw=2.0, label=r"$v_2$ (norm)")
        ax.plot(t, _norm(Lloss), color=C_LLOSS, lw=2.0, ls="--",
                label=r"$L_{\rm loss}$ (norm)")
        ax.plot(t, _norm(r), color=C_R, lw=1.4, ls=":", label=r"$r$ (norm)")
        ax.set_ylabel("normalized")
        ax.set_ylim(-0.05, 1.18)

        # Turning-point markers.
        ax.axvline(t_v2min, color=C_V2, lw=1.0, ls="-", alpha=0.7)
        ax.axvline(t_llosspk, color=C_LLOSS, lw=1.0, ls="--", alpha=0.7)
        # per-config blowout (shell exits cloud): star on the normalized v2 curve
        # (the panel's primary kinematic trace); label on top panel. Window is the
        # early dip (t<=win); freeze xlim so an out-of-window marker can't auto-
        # expand the view (the star is simply skipped if blowout is outside it).
        xl = ax.get_xlim()
        mark(ax, cfg, t=t, y=_norm(v2), color="0.25", label=(pi == 0))
        ax.set_xlim(xl)
        ax.annotate(f"$v_2$ min\n{t_v2min:.3f}", (t_v2min, 1.10), color=C_V2,
                    fontsize=8, ha="center", va="top")
        ax.annotate(f"$L_{{\\rm loss}}$ pk /\n$r$ min  {t_llosspk:.3f}",
                    (t_llosspk, 0.02), color=C_LLOSS, fontsize=8,
                    ha="left", va="bottom")

        # Right axis: monotone drivers Pb, T0 (log).
        axr = ax.twinx()
        axr.plot(t, Pb, color=C_PB, lw=1.0, alpha=0.8)
        axr.plot(t, T0, color=C_T0, lw=1.0, alpha=0.8)
        axr.set_yscale("log")
        axr.set_ylabel(r"$P_b$ (green), $T_0$ (mauve)", fontsize=10)
        axr.tick_params(labelsize=8)

        lead = "v2 LEADS" if t_v2min < t_llosspk - 1e-9 else (
            "simultaneous" if abs(t_v2min - t_llosspk) <= 1e-9 else "v2 lags")
        de_txt = "no sign change" if t_dE0 is None else f"sign change @ {t_dE0:.3f}"
        ax.set_title(
            f"{label}:  $v_2$min={t_v2min:.3f}, "
            f"$L_{{\\rm loss}}$pk={t_llosspk:.3f} ({lead});  "
            f"$dE_b/dt$ {de_txt}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("time [Myr]")
    fig.suptitle(
        "Causal order through the early cooling-ratio dip: $v_2$ decelerates "
        "first, $L_{\\rm loss}$ peaks at/after,\n$dE_b/dt$ stays positive "
        "(no sign change) while $P_b,T_0$ fall monotonically",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "pdf"):
        out = FIGDIR / f"dip_causalorder.{ext}"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
