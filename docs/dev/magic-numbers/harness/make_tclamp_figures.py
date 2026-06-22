#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the net_coolingcurve T-floor write-up (audit finding #1).

All figures are PURE READS of committed artifacts (no sim re-run):
  * data/<config>_summary.json   -- the 9.46M-call get_dudt instrument output
  * data/tclamp_dudt_overlay.csv -- old-vs-new dudt(T) (make_tclamp_overlay_data.py)
Numbers for the schematics (per-call 576/576, sha256, 574 passed) come from the
committed TCLAMP_PLAN.md gate table.

    python docs/dev/magic-numbers/harness/make_tclamp_figures.py  ->  ../figs/*.png
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)

# --- the temperature landmarks (log10 K) ---
LOG_EDGE = 3.5                    # nonCIE_Tmin = min(cube.temp): the real table min, 3162 K
LOG_OLD = 4.0                     # the old hard-coded floor, 1e4 K
LOG_BND = np.log10(3e4)           # 4.477: the 3e4 outer boundary == measured min T
LOG_CIE = 5.5                     # non-CIE upper edge / CIE switch
CLAIM_RUN = 3.91                  # the comment's claimed "T ~ 1e3.91" (never observed)
CLAIM_TAB = 3.99                  # the comment's claimed table min (wrong; real edge 3.5)

INK, GREEN, RED, BLUE, ORANGE, PURPLE, GRAY = (
    "#212121", "#2e7d32", "#c62828", "#1565c0", "#e65100", "#6a1b9a", "#9e9e9e")

CONFIGS = [
    ("simple_cluster", "baseline\n(full run)", BLUE),
    ("f1edge_lowdens", "low-$\\rho$ · hi-M · hi-sfe", GREEN),
    ("f1edge_hidens", "hi-$\\rho$ · hi-M · lo-sfe", ORANGE),
    ("conduction_stiff", "stiff LSODA\nflood", PURPLE),
]


def load(tag):
    return json.load(open(os.path.join(DATA, f"{tag}_summary.json")))


SUM = {tag: load(tag) for tag, _, _ in CONFIGS}
TOTAL_CALLS = sum(SUM[t]["calls"] for t, _, _ in CONFIGS)


# ===========================================================================
# FIG 1 — schematic: the temperature axis, the over-floor, where the bubble lives
# ===========================================================================
def fig_schematic():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(3.0, 6.0)
    ax.set_ylim(0, 10)
    ax.axis("off")
    y0, h = 4.2, 1.6  # the bar

    # zones
    ax.axvspan(3.0, LOG_EDGE, ymin=y0/10, ymax=(y0+h)/10, color="#ffebee", ec=RED, lw=1.4, hatch="xx", zorder=1)
    ax.axvspan(LOG_EDGE, LOG_CIE, ymin=y0/10, ymax=(y0+h)/10, color="#e3f2fd", ec=BLUE, lw=1.4, zorder=1)
    ax.axvspan(LOG_CIE, 6.0, ymin=y0/10, ymax=(y0+h)/10, color="#f3e5f5", ec=PURPLE, lw=1.4, zorder=1)
    ax.text((3.0+LOG_EDGE)/2, y0+h/2, "T < table min\n→ raise\nException", ha="center", va="center",
            fontsize=8.5, color=RED, weight="bold", zorder=3)
    ax.text((LOG_EDGE+LOG_CIE)/2, y0+h/2, "non-CIE cooling table   $\\Lambda(n,T,\\phi)$\n[3162 K , 316 000 K]",
            ha="center", va="center", fontsize=10.5, color=BLUE, weight="bold", zorder=3)
    ax.text((LOG_CIE+6.0)/2, y0+h/2, "interp\n+ CIE", ha="center", va="center", fontsize=9, color=PURPLE, zorder=3)

    # over-floored decade highlight (3.5 -> 4.0)
    ax.annotate("", xy=(LOG_OLD, y0-0.55), xytext=(LOG_EDGE, y0-0.55),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    ax.text((LOG_EDGE+LOG_OLD)/2, y0-1.15,
            "the OLD floor lifted this whole valid\ndecade [3162, 10 000) K up to 10⁴ K",
            ha="center", va="top", fontsize=9, color=RED, style="italic")

    # operating range (>= 4.477) shaded green above the bar
    ax.axvspan(LOG_BND, 6.0, ymin=(y0+h+0.15)/10, ymax=(y0+h+0.95)/10, color="#e8f5e9", ec=GREEN, lw=1.2, zorder=1)
    ax.text((LOG_BND+6.0)/2, y0+h+0.55,
            "where the bubble ODE actually lives — measured min T = 30 000 K  (all 4 configs, 9.46M calls)",
            ha="center", va="center", fontsize=9.5, color=GREEN, weight="bold", zorder=3)

    # marker lines
    def marker(x, label, color, ytop, ls="-", lw=2.0):
        ax.plot([x, x], [y0-0.1, ytop], color=color, lw=lw, ls=ls, zorder=4)
        ax.plot([x], [y0+h], marker="v", color=color, ms=9, zorder=5)
        ax.text(x, ytop+0.12, label, ha="center", va="bottom", fontsize=9, color=color, weight="bold")

    marker(LOG_EDGE, "NEW floor\n3162 K (table edge)", GREEN, y0+h+1.7)
    marker(LOG_OLD, "OLD floor\n10⁴ K", RED, y0+h+1.2, ls=(0, (4, 2)))
    marker(LOG_BND, "3·10⁴ K\nboundary", GRAY, y0+h+2.4, ls=(0, (1, 1)), lw=1.5)

    # the comment's two wrong claims
    ax.plot([CLAIM_RUN], [y0-0.1], marker="o", color="#880000", ms=8, zorder=6)
    ax.annotate("comment's claimed\n“T ~ 10$^{3.91}$” — never\nobserved in any run",
                xy=(CLAIM_RUN, y0-0.1), xytext=(3.05, 1.4), fontsize=8.3, color="#880000",
                arrowprops=dict(arrowstyle="->", color="#880000", lw=1.2))
    ax.annotate("comment said “table only to 3.99” —\nwrong: it reaches 3.5 (3162 K)",
                xy=(CLAIM_TAB, y0+h), xytext=(4.15, 9.2), fontsize=8.3, color="#880000",
                arrowprops=dict(arrowstyle="->", color="#880000", lw=1.2))

    # axis ticks along the bottom
    for lx in [3.0, 3.5, 4.0, 4.477, 5.0, 5.5, 6.0]:
        ax.plot([lx, lx], [y0-0.1, y0-0.25], color=INK, lw=1)
        ax.text(lx, y0-0.38, f"{lx:.2f}" if lx != 4.477 else "4.48", ha="center", va="top", fontsize=7.5, color=INK)
    ax.text(6.0, y0-0.78, "log₁₀ T  [K] →", ha="right", va="top", fontsize=9, color=INK)

    ax.set_title("Finding #1 — the cooling-table temperature axis: a hard-coded 10⁴ K floor over a table that "
                 "reaches 3162 K,\nin a regime the bubble never enters (min T = 30 000 K)", fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tclamp_schematic.png"), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ===========================================================================
# FIG 2 — min T reached per config vs the floor / raise / boundary lines
# ===========================================================================
def fig_minT():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(CONFIGS))
    minT = [SUM[t]["min_T"] for t, _, _ in CONFIGS]
    cols = [c for _, _, c in CONFIGS]

    # reference horizontal lines
    ax.axhspan(1e3, 1e4, color="#ffebee", zorder=0)
    ax.text(len(CONFIGS)-0.5, 10**((3+4)/2), "clamp would fire here\n(T < 10⁴) — never reached",
            ha="right", va="center", fontsize=9, color=RED, style="italic")
    ax.axhline(3162, color=GREEN, ls="-", lw=1.8, zorder=1)
    ax.axhline(1e4, color=RED, ls=(0, (5, 3)), lw=1.8, zorder=1)
    ax.axhline(3e4, color=GRAY, ls=(0, (1, 1)), lw=1.6, zorder=1)
    ax.text(-0.45, 3162*1.03, "table edge 3162 K (new floor)", fontsize=8.5, color=GREEN, va="bottom")
    ax.text(-0.45, 1e4*1.03, "old floor 10⁴ K", fontsize=8.5, color=RED, va="bottom")
    ax.text(-0.45, 3e4*1.03, "3·10⁴ boundary", fontsize=8.5, color=GRAY, va="bottom")

    ax.scatter(x, minT, s=260, color=cols, edgecolor="k", zorder=5, marker="v")
    for xi, t in zip(x, [c for c, _, _ in CONFIGS]):
        ax.annotate(f"min T =\n30 000 K", (xi, SUM[t]["min_T"]), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=8, color=INK)
        ax.annotate(f"{SUM[t]['calls']:,}\ncalls", (xi, SUM[t]["min_T"]), textcoords="offset points",
                    xytext=(0, -34), ha="center", fontsize=7.5, color=GRAY)

    ax.set_yscale("log")
    ax.set_ylim(2e3, 1e5)
    ax.set_xlim(-0.7, len(CONFIGS)-0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in CONFIGS], fontsize=9)
    ax.set_ylabel("minimum T the bubble ODE ever passed to get_dudt  [K]")
    ax.set_title("The floor never engages: every regime bottoms out at the 3·10⁴ K boundary — 3× above the old "
                 "floor,\n9.5× above the raise boundary. The 10⁴ K clamp is dead code.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tclamp_minT_by_config.png"), dpi=140, facecolor="white")
    plt.close(fig)


# ===========================================================================
# FIG 3 — the log10(T) histogram of all RHS evals: the "dead-code wall"
# ===========================================================================
def fig_hist():
    fig, ax = plt.subplots(figsize=(12, 6))
    # dead zone (never visited): below the wall at 4.45
    ax.axvspan(3.0, 4.45, color="#fafafa", zorder=0)
    ax.axvspan(LOG_EDGE-0.02, LOG_OLD+0.0, color="#ffebee", alpha=0.5, zorder=0)

    for tag, lab, col in CONFIGS:
        h = SUM[tag]["hist"]
        xs = np.array(sorted(float(k) for k in h))
        ys = np.array([h[f"{x:g}" if f"{x:g}" in h else str(x)] for x in xs], dtype=float)
        # robust key lookup
        ys = np.array([h.get(str(x), h.get(f"{x:g}", 0)) for x in xs], dtype=float)
        ax.step(xs, np.maximum(ys, 0.5), where="mid", color=col, lw=1.7,
                label=f"{tag}  ({SUM[tag]['calls']:,})")

    for x, c, lab in [(LOG_EDGE, GREEN, "table edge 3.5"), (LOG_OLD, RED, "old floor 4.0"),
                      (LOG_BND, GRAY, "3·10⁴ bnd 4.48")]:
        ax.axvline(x, color=c, ls=(0, (5, 3)), lw=1.6)
        ax.text(x, 1.4e5, lab, rotation=90, va="top", ha="right", fontsize=8.5, color=c, weight="bold")

    ax.annotate("hard wall at log T = 4.45 —\nNOTHING below it (0 of 9.46M calls)",
                xy=(4.45, 3e4), xytext=(3.55, 6e3), fontsize=10, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text((3.0+4.45)/2, 1.2, "the clamp's entire operating zone — never visited", ha="center",
            fontsize=9, color=GRAY, style="italic")

    ax.set_yscale("log")
    ax.set_ylim(0.5, 2e5)
    ax.set_xlim(3.0, 8.4)
    ax.set_xlabel("log₁₀ T  passed to get_dudt  [K]")
    ax.set_ylabel("number of RHS evaluations  (per 0.05 dex bin)")
    ax.set_title("Every regime's temperature distribution stops dead at log T = 4.45 — the floor (4.0) and the "
                 "table edge (3.5)\nsit in a region the bubble ODE RHS never visits across 9.46M evaluations",
                 fontsize=10.5)
    ax.legend(title="config (get_dudt calls)", fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tclamp_temperature_histogram.png"), dpi=140, facecolor="white")
    plt.close(fig)


# ===========================================================================
# FIG 4 — calls vs clamp-fires: 9.46M -> 0
# ===========================================================================
def fig_fires():
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(CONFIGS))[::-1]
    calls = [SUM[t]["calls"] for t, _, _ in CONFIGS]
    cols = [c for _, _, c in CONFIGS]
    ax.barh(y, calls, color=cols, edgecolor="k", height=0.6, zorder=3)
    for yi, (tag, _, _) in zip(y, CONFIGS):
        s = SUM[tag]
        ax.text(s["calls"]*1.02, yi, f"  {s['calls']:,} calls", va="center", ha="left", fontsize=9, color=INK)
        ax.text(s["calls"]*0.97, yi, f"T<10⁴ fired: {s['n_below_1e4']}× ", va="center", ha="right",
                fontsize=9, color="white", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}\n{lab}".replace("\n", "  ·  ") for t, lab, _ in
                        [(t, lab.replace(chr(10), " "), c) for t, lab, c in CONFIGS]], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e7)
    ax.set_xlabel("get_dudt calls  (log scale)")
    ax.set_title(f"Measure first: {TOTAL_CALLS:,} get_dudt calls across 4 regimes  →  the T<10⁴ clamp fired 0×  "
                 "(and 0 calls ever\nreached the table min, so the guarded raise was never in play either)", fontsize=10.5)
    ax.text(0.5, -0.22,
            "boundary-transient hair (benign): accepted solves just below 3·10⁴ are all at ≈29999.99 K "
            "(69 / 158 / 136 / 181) — FP dust at the boundary, accepted_below_1e4 = 0 everywhere.",
            transform=ax.transAxes, ha="center", fontsize=8, color=GRAY, style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tclamp_calls_vs_fires.png"), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ===========================================================================
# FIG 5 — dudt(T) old vs new overlay: identical where it matters, fixed below
# ===========================================================================
def fig_overlay():
    rows = list(csv.DictReader(open(os.path.join(DATA, "tclamp_dudt_overlay.csv"))))
    lt = np.array([float(r["log10T"]) for r in rows])
    old = np.array([float(r["dudt_old_1e4floor"]) for r in rows])
    new = np.array([float(r["dudt_new_tableedge"]) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # (a) full picture, symlog
    ax1.axvspan(3.0, LOG_BND, color="#f3f3f3", zorder=0)
    ax1.text((3.0+LOG_BND)/2, 0, "below the measured operating floor\n(T < 30 000 K — never reached by any run)",
             ha="center", va="center", fontsize=8.5, color=GRAY, style="italic")
    ax1.plot(lt, old, "-", color=RED, lw=2.6, label="OLD floor (10⁴ K)")
    ax1.plot(lt, new, "--", color=GREEN, lw=2.0, label="NEW floor (table edge 3162 K)")
    for x, c in [(LOG_EDGE, GREEN), (LOG_OLD, RED), (LOG_BND, GRAY)]:
        ax1.axvline(x, color=c, ls=(0, (1, 1)), lw=1.3)
    ax1.set_yscale("symlog", linthresh=1e6)
    ax1.set_xlim(3.0, 4.7)
    ax1.set_xlabel("log₁₀ T  [K]")
    ax1.set_ylabel("get_dudt  [code units]")
    ax1.set_title("(a) identical for T ≥ 10⁴ (log 4.0); divergent below")
    ax1.legend(fontsize=8.5, loc="lower left")
    ax1.axhline(0, color="k", lw=0.6)

    # (b) zoom into the over-floored decade, linear
    m = (lt >= 3.35) & (lt <= 4.12)
    ax2.axvspan(LOG_EDGE, LOG_OLD, color="#fff3e0", zorder=0)
    ax2.text((LOG_EDGE+LOG_OLD)/2, old.min()*0.0 + new[m].max()*0.85,
             "over-floored\ndecade", ha="center", fontsize=9, color=ORANGE, weight="bold")
    ax2.plot(lt[m], old[m], "-", color=RED, lw=2.6, label="OLD: flat at the 10⁴ value")
    ax2.plot(lt[m], new[m], "--", color=GREEN, lw=2.2, label="NEW: tracks the real rate")
    ax2.axvline(LOG_EDGE, color=GREEN, ls=(0, (1, 1)), lw=1.3)
    ax2.axvline(LOG_OLD, color=RED, ls=(0, (1, 1)), lw=1.3)
    ax2.text(LOG_EDGE, ax2.get_ylim()[1], " table\n edge", fontsize=8, color=GREEN, va="top")
    ax2.text(LOG_OLD, ax2.get_ylim()[1], " curves\n meet", fontsize=8, color=RED, va="top", ha="left")
    ax2.set_xlabel("log₁₀ T  [K]")
    ax2.set_ylabel("get_dudt  [code units]")
    ax2.set_title("(b) zoom: the old floor flatlines the whole decade")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.axhline(0, color="k", lw=0.6)

    fig.suptitle("The fix is bit-identical where every run lives (T ≥ 10⁴) and only changes the never-reached "
                 "sub-10⁴ decade — where it is strictly more correct (table edge, not a 2.5× hotter floor)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGS, "tclamp_dudt_overlay.png"), dpi=140, facecolor="white")
    plt.close(fig)


# ===========================================================================
# FIG 6 — the validation ladder
# ===========================================================================
def fig_ladder():
    fig, ax = plt.subplots(figsize=(13.5, 4.4))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("MEASURE FIRST", "instrument get_dudt\n4 regimes · 9.46M calls", "T<10⁴ fired 0×\nmin T = 30 000 K"),
        ("PER-CALL EQUIV", "vs git show HEAD\nfull T-grid, separate module", "576/576 bit-id (T≥10⁴)\n0 raises  [NOT sufficient]"),
        ("FULL-RUN BYTE-ID", "capped simple_cluster\nseparate processes · matched-t", "identical sha256\n169 snapshots"),
        ("SUITE + LINT", "pytest + ruff F-rules", "574 passed\nruff clean → SHIP"),
    ]
    n = len(stages)
    slot = 1.0 / n
    bw = 0.80 * slot          # leave a clear gap between boxes for the arrows
    for i, (p, desc, gate) in enumerate(stages):
        x0 = 0.01 + i * slot
        fc = "#e8f5e9" if i != 1 else "#fff8e1"
        ec = GREEN if i != 1 else ORANGE
        ax.add_patch(FancyBboxPatch((x0, 0.40), bw, 0.50, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=2.2))
        ax.text(x0 + bw/2, 0.83, p, ha="center", fontweight="bold", fontsize=11, color=INK)
        ax.text(x0 + bw/2, 0.64, desc, ha="center", fontsize=8.0, color=INK)
        ax.text(x0 + bw/2, 0.48, gate, ha="center", fontsize=8.0, color="darkgreen" if i != 1 else "#b35900",
                weight="bold")
        if i < n-1:
            ax.add_patch(FancyArrowPatch((x0+bw, 0.65), (x0+slot, 0.65),
                         arrowstyle="-|>", mutation_scale=22, lw=2.4, color=INK))
    ax.text(0.5, 0.12,
            "Key lesson (CLAUDE.md rule 5): a per-call equivalence is NECESSARY but NOT SUFFICIENT for an "
            "iterative path — only a full-run\nbyte-identity, in separate processes at matched simulation time, "
            "clears a change to the solver's hot loop.",
            ha="center", fontsize=8.8, style="italic", color="#7a0000")
    ax.set_title("Validation ladder — measure before fixing, then gate the (provably inert) fix bit-identical",
                 fontsize=11.5, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tclamp_validation_ladder.png"), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_schematic()
    fig_minT()
    fig_hist()
    fig_fires()
    fig_overlay()
    fig_ladder()
    print("wrote:", ", ".join(sorted(f for f in os.listdir(FIGS) if f.startswith("tclamp"))))
