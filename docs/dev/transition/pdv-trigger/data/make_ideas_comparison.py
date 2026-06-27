#!/usr/bin/env python3
"""One-glance comparison of every transition-fix idea tried in this workstream.

Renders ../ideas_comparison.png from the committed CSVs (no sim re-run):
  * top   — a 6-rung "scoreboard" ladder: each idea, its mechanism, the one number
            that killed/limited it, and a verdict badge. The progression is the
            storyline (constant knobs -> coupled scalar -> the kappa_eff cooling mechanism ->
            optional evaporation-decoupling fidelity bonus).
  * bottom — three real-data evidence panels that back the three key verdicts:
        A  constant f_mix can't span density   (data/fmix_table.csv)
        B  theta_target(Da) saturates / non-monotonic in n   (data/da_replay.csv)
        C  kappa_eff Rung A: cooling up, evaporation rides along   (data/kappa_backreaction.csv)

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_ideas_comparison.py
Deliverable:
    docs/dev/transition/pdv-trigger/ideas_comparison.png
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# Verdict palette.
_RED = "#d9534f"     # refuted
_AMBER = "#f0ad4e"   # partial / magnitude-only
_BLUE = "#5bc0de"    # the cooling mechanism (this work)
_GREEN = "#5cb85c"   # optional fidelity bonus

# The 6-rung ladder. (title, mechanism, killer metric, verdict label, colour, highlight)
_LADDER = [
    ("constant f_mix", "Lleak + f·Lcool", "needs 1.4 (dense)\n→3.8 (diffuse)", "REFUTED", _RED, False),
    ("constant θ", "max(Lcool, θ·Lmech)", "θ=0.95 IS the\ntrigger → degenerate", "REFUTED", _RED, False),
    ("θ_target(Da)", "θmax·Da/(1+Da)", "Da≫1 everywhere\n→ saturates; n-nonmono", "REFUTED", _RED, False),
    ("multiplier f=2\n(live runs)", "scalar boost on Lcool", "dense fires at birth;\ncompact/diffuse miss", "PARTIAL", _AMBER, False),
    ("κ_eff Rung A\n(this work)", "C_thermal → f_κ·C\n(structural)", "Lcool ×1.3:\nthe cooling mechanism", "MECHANISM", _BLUE, True),
    ("κ_eff Rung B", "evaporation\ndecoupling", "optional fidelity;\n1D dMdt resists it", "BONUS", _GREEN, False),
]


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _ladder(ax):
    ax.axis("off")
    ax.set_xlim(0, len(_LADDER))
    ax.set_ylim(0, 1)
    w = 0.86
    for i, (title, mech, killer, verdict, colour, hi) in enumerate(_LADDER):
        x0 = i + (1 - w) / 2
        box = FancyBboxPatch(
            (x0, 0.18), w, 0.66,
            boxstyle="round,pad=0.012,rounding_size=0.04",
            linewidth=2.4 if hi else 1.2,
            edgecolor="black" if hi else "0.4",
            facecolor=colour, alpha=0.92 if hi else 0.78,
            mutation_aspect=0.5,
        )
        ax.add_patch(box)
        cx = i + 0.5
        ax.text(cx, 0.76, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
        ax.text(cx, 0.55, mech, ha="center", va="center", fontsize=7.3, style="italic", color="0.18")
        ax.text(cx, 0.36, killer, ha="center", va="center", fontsize=7.3, color="0.05")
        # verdict chip
        ax.text(cx, 0.235, verdict, ha="center", va="center", fontsize=8.2, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="0.15", edgecolor="none"))
        if i < len(_LADDER) - 1:
            ax.add_patch(FancyArrowPatch(
                (i + 0.5 + w / 2, 0.51), (i + 1 + (1 - w) / 2, 0.51),
                arrowstyle="-|>", mutation_scale=13, color="0.35", lw=1.4))
    ax.text(0.0, 0.96, "Transition-fix ideas — what was tried, and the verdict",
            ha="left", va="top", fontsize=12.5, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.05,
            "scalar knobs (red) can't span density; a constant multiplier (amber) only fixes "
            "magnitude, mistimes by density; κ_eff Rung A (blue, gated) IS the cooling mechanism "
            "— calibrate f_κ(properties) to θ(n_H); evaporation-decoupling (green) is an optional bonus.",
            ha="left", va="bottom", fontsize=7.8, color="0.3", transform=ax.transAxes)


def _panel_fmix(ax):
    rows = _read_csv(os.path.join(_HERE, "fmix_table.csv"))
    # sort diffuse -> dense by the resolved loss fraction at blowout
    rows.sort(key=lambda r: float(r["Lcool_over_Lmech_at_blowout"]))
    labels = [r["config"].replace("_", "\n") for r in rows]
    fmix = [float(r["fmix_no_pdv"]) for r in rows]
    loss = [float(r["Lcool_over_Lmech_at_blowout"]) for r in rows]
    x = range(len(rows))
    bars = ax.bar(x, fmix, color="#6f9bd1", edgecolor="0.3")
    mean = sum(fmix) / len(fmix)
    ax.axhline(mean, color=_RED, ls="--", lw=1.4, label=f"one constant (mean {mean:.1f})\nmisses every bar")
    for xi, fi in zip(x, fmix):
        ax.text(xi, fi + 0.05, f"{fi:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("constant f_mix needed\n(to hit 0.95 at blowout)", fontsize=8)
    ax.set_title("A. No single f_mix spans density (1.4→3.8)", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, loss, "o-", color="0.25", lw=1.2, ms=4)
    ax2.set_ylabel("resolved Lcool/Lmech\nat blowout (0.25→0.70)", fontsize=7.3, color="0.25")
    ax2.tick_params(axis="y", labelsize=7, colors="0.25")
    ax2.set_ylim(0, 1.0)


def _panel_da(ax):
    rows = _read_csv(os.path.join(_HERE, "da_replay.csv"))
    n = [float(r["nCore"]) for r in rows]
    da = [float(r["Da_real_blow"]) for r in rows]
    labels = [r["config"] for r in rows]
    ax.axhspan(1e3, 1e7, color=_AMBER, alpha=0.16)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.scatter(n, da, s=46, color="#b1631f", zorder=3, edgecolor="0.2")
    for ni, di, lab in zip(n, da, labels):
        ax.annotate(lab, (ni, di), fontsize=5.8, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nCore [cm⁻³]", fontsize=8)
    ax.set_ylabel("real Da = t_turb / t_cool,int", fontsize=8)
    ax.set_title("B. θ_target(Da): Da≫1 everywhere → saturates;\nnon-monotonic in n (gate-validated replay)",
                 fontsize=9.0, fontweight="bold")
    ax.text(0.96, 0.10, "θmax·Da/(1+Da) → θmax\n(a constant) in this band",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#8a4d18")


def _panel_kappa(ax):
    rows = _read_csv(os.path.join(_HERE, "kappa_backreaction.csv"))
    t = [float(r["t"]) for r in rows]
    lcool = [float(r["Lcool_ratio"]) for r in rows]
    dmdt = [float(r["dMdt_ratio"]) for r in rows]
    eb = [float(r["Eb_ratio"]) for r in rows]
    ax.axhspan(1.0, 1.62, color="#2ca02c", alpha=0.06)   # "raised" band
    ax.axhspan(0.84, 1.0, color="#d62728", alpha=0.06)   # "lowered" band
    ax.axhline(1.0, color="0.35", lw=1.0, ls=":")
    ax.plot(t, lcool, lw=2, color="#1f77b4", label="Lcool: raised 1.2–1.5×")
    ax.plot(t, dmdt, lw=2, color="#ff7f0e", label="ṁ evap: raised ~1.1× (unwanted)")
    ax.plot(t, eb, lw=1.5, ls="--", color="#2ca02c", label="Eb: drained to ~0.9×")
    ax.set_ylim(0.84, 1.62)
    ax.set_xlabel("t [Myr]", fontsize=8)
    ax.set_ylabel("ratio  f_κ=2 ÷ f_κ=1   (>1 ⇒ raised)", fontsize=7.4)
    ax.set_title("C. lines are RATIOS, not values:\nabove 1 = κ raised it (cooling AND evap)",
                 fontsize=8.4, fontweight="bold")
    ax.legend(fontsize=6.6, loc="lower left")


def main():
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.25], hspace=0.42, wspace=0.32,
                          left=0.06, right=0.94, top=0.95, bottom=0.10)
    _ladder(fig.add_subplot(gs[0, :]))
    _panel_fmix(fig.add_subplot(gs[1, 0]))
    _panel_da(fig.add_subplot(gs[1, 1]))
    _panel_kappa(fig.add_subplot(gs[1, 2]))
    out = os.path.join(_PDV, "ideas_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
