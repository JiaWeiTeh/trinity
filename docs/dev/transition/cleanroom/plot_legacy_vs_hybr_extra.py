#!/usr/bin/env python3
"""Companion to plot_legacy_vs_hybr.py: MORE quantities through the early dip (user Q,
2026-06-21): "show me more of what changed so I understand the cure."

The first figure (plot_legacy_vs_hybr.py) showed cooling ratio / bubble_Lloss / cool_beta
and concluded: T0 is ~identical, the beta-clamp is the difference. This companion overlays
four more quantities for the same three early-crossing configs, through the same dip window:
  - cool_delta      -- the OTHER half of the cooling exponent pair the clamp touches
  - cool_beta+delta -- the sum the solver actually balances; legacy dips below 0 (and the
                       -0.4 reference), hybr stays positive -> repressurization, not collapse
  - Eb              -- bubble energy: hybr retains/builds more out of the dip
  - Pb              -- bubble pressure: legacy holds a floor, hybr's collapses much deeper

NOTE on schema (checked, contrary to the brief): the legacy and h0 CSV headers are
IDENTICAL (30 columns each) and cool_delta, Eb, Pb, v2, cool_beta are all fully populated
in BOTH. So every quantity here is a true legacy-vs-hybr overlay; none is hybr-only.

Pure read of committed data/c0_*_legacy.csv (BEFORE) + data/c0_*_h0.csv (AFTER).

    python plot_legacy_vs_hybr_extra.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = HERE.parents[3] / "paper" / "_lib" / "trinity.mplstyle"  # parents[3]=repo root
if STYLE.exists():
    plt.style.use(str(STYLE))
plt.rcParams["text.usetex"] = False

CONFIGS = ["small_dense_highsfe", "pl2_steep", "simple_cluster"]  # early crossers, clean dips
LEG, HYB = "#0072B2", "#D55E00"  # Wong blue (legacy) / vermillion (hybr)
BPD_LINE = -0.4  # beta+delta reference line (drawn on the beta+delta panel)


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"])
            beta = float(r["cool_beta"]); delta = float(r["cool_delta"])
            Eb = float(r["Eb"]); Pb = float(r["Pb"])
            Lg = float(r["bubble_Lgain"]); Ll = float(r["bubble_Lloss"])
        except (ValueError, KeyError, TypeError):
            continue
        if Lg > 0:
            out.append(dict(t=t, beta=beta, delta=delta, bpd=beta + delta,
                            Eb=Eb, Pb=Pb, ratio=(Lg - Ll) / Lg))
    return out


def main():
    n = len(CONFIGS)
    # 4 columns: cool_delta, beta+delta, Eb, Pb
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 2.6 * n + 0.6), squeeze=False)
    for i, cfg in enumerate(CONFIGS):
        leg = load(HERE / "data" / f"c0_{cfg}_legacy.csv")
        hyb = load(HERE / "data" / f"c0_{cfg}_h0.csv")
        if not leg or not hyb:
            continue
        # window to the dip/crossing region (a bit past the legacy crossing)
        tcross = next((d["t"] for d in leg if d["ratio"] < 0.05), None)
        thi = (tcross * 4) if tcross else 0.6
        ad, abd, ae, ap = axes[i]

        for d, c, ls, lab in [(leg, LEG, "-", "legacy"), (hyb, HYB, "--", "hybr")]:
            w = [x for x in d if x["t"] <= thi]
            tt = [x["t"] for x in w]
            ad.plot(tt, [x["delta"] for x in w], color=c, ls=ls, lw=1.8, label=lab)
            abd.plot(tt, [x["bpd"] for x in w], color=c, ls=ls, lw=1.8)
            ae.plot(tt, [x["Eb"] for x in w], color=c, ls=ls, lw=1.8)
            ap.plot(tt, [x["Pb"] for x in w], color=c, ls=ls, lw=1.8)

        # delta panel
        ad.axhline(0, color="0.6", lw=0.7)
        ad.set_ylabel(f"{cfg}\n\n" + r"$\delta$ (cool_delta)", fontsize=8.5)
        # beta+delta panel: reference lines at 0 and -0.4
        abd.axhline(0, color="0.6", lw=0.7)
        abd.axhline(BPD_LINE, color="k", ls=":", lw=1.0)
        abd.text(0.02, BPD_LINE, f"{BPD_LINE:g}", color="k", fontsize=7, va="center",
                 ha="left", transform=abd.get_yaxis_transform())
        abd.set_ylabel(r"$\beta+\delta$", fontsize=9)
        # Eb / Pb panels (log y)
        ae.set_ylabel(r"$E_b$  [erg]", fontsize=9); ae.set_yscale("log")
        ap.set_ylabel(r"$P_b$", fontsize=9); ap.set_yscale("log")

        # mark the legacy crossing time on every panel for alignment
        for ax in (ad, abd, ae, ap):
            ax.set_xscale("log")
            if tcross:
                ax.axvline(tcross, color=LEG, ls=":", lw=0.8, alpha=0.7)

        if i == 0:
            ad.legend(fontsize=8, loc="lower left")
            ad.set_title(r"$\delta$: cooling exponent partner of $\beta$", fontsize=9)
            abd.set_title(r"$\beta+\delta$: legacy dips $<0$, hybr stays $>0$", fontsize=9)
            ae.set_title(r"$E_b$: hybr retains more bubble energy", fontsize=9)
            ap.set_title(r"$P_b$: hybr pressure collapses deeper", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("t  [Myr]")
    fig.suptitle(r"Legacy $\to$ hybr through the dip (more quantities): $\beta+\delta$ stays "
                 r"positive and $E_b$ holds up under hybr — the cure beyond the $\beta$-clamp "
                 r"(dotted line = legacy 0.05 crossing)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"legacy_vs_hybr_extra.{ext}", dpi=150)
    print(f"wrote {out}/legacy_vs_hybr_extra.(pdf,png) for {CONFIGS}")


if __name__ == "__main__":
    main()
