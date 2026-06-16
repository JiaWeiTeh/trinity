#!/usr/bin/env python3
"""Phase 3 self-consistent hybr-vs-legacy plots from the master runs table.

Unlike analyze_arms.py (which re-reads the per-arm/per-segment shadow jsonl),
the Phase-3 self-consistent runs write per-run ``dictionary.jsonl`` under /tmp,
which is ephemeral and gone on container restart. So the numbers here are
transcribed from the **summary tables** in
``docs/dev/archive/betadelta/PHASE2_ARMS.md`` (section "Phase 3 — self-consistent
validation"): the master runs table + the three headline comparisons.

  ⚠️ Point-in-time values. That doc carries the staleness banner; re-verify
  these against it (or re-run the configs per the doc's "How to regenerate"
  recipe) before relying on any conclusion. Edit the constants below if the
  table moves.

  Re-verified 2026-06-14 against bugfix/beta-delta-solver-pt2: the master table
  + headline comparisons here are UNCHANGED (that branch only *appended* an
  end-to-end robustness-sweep section). NOTE the steep/mock rows below are the
  master-table runs (stop_t = 3.0 / 0.3 Myr); the stop_t = 4 Myr sweep runs used
  by analyze_negvel.py / plot_hunt.py reach higher beta_max (steep 3.43, mock
  4.23) and ratio_end -- so those two configs differ by design across scripts.

Produces:
  - phase3_headline.png : convergence / beta-reach / transition / cost,
                          legacy vs hybr (the four headline findings)
  - phase3_regime.png   : cooling ratio reached per hybr run vs the 0.05
                          transition threshold -- which regimes transition,
                          stall, or stay energy-driven

Usage: python docs/dev/archive/betadelta/diagnostics/analyze_phase3.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
L_COL, H_COL = "0.6", "#2ca02c"  # legacy / hybr
BETA_BOX_TOP = 1.0  # legacy BETA_MAX
RATIO_THRESH = 0.05  # cooling-balance transition trigger
# colourblind-safe (Wong): transition = blue, stall = vermillion
CAT_COL = {"transition": "#0072B2", "energy": "0.5", "stall": "#D55E00", "short": "0.8"}

# apply trinity's house style; usetex off so scratch renders without LaTeX.
_STYLE = HERE.parents[1] / "paper" / "_lib" / "trinity.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))
plt.rcParams["text.usetex"] = False

# --- transcribed from docs/dev/archive/betadelta/PHASE2_ARMS.md (Phase 3 section) ---
# convergence % (< 1e-4): legacy from master table / Phase-0, hybr from master table
CONV = [  # (config, legacy%, hybr%)   None = not run
    ("flat\n1e6 n1e5", 0, 100),
    ("steep\n1e6 a-2", 0, 100),
    ("typical\n1e6 n1e3", None, 100),
    ("mock\n4e3", 0, 100),
    ("simple\n1e5", 50, 100),
]
BETA = [  # (config, legacy beta_max, hybr beta_max)
    ("flat", 0.84, 1.63),
    ("steep", 0.84, 2.82),
    ("cost", 0.76, 0.93),
    ("typical", None, 4.18),
    ("simple", None, 4.20),
    ("mock", None, 1.03),
]
TRANS = [  # (config, legacy t_trans, hybr t_trans, hybr_stalls, factor)
    ("flat\na=0", 0.097, 0.247, False, "2.5x"),
    ("steep\na=-2", 0.098, None, True, ">30x"),
]
COST = [("legacy", 7.7e-6), ("hybr", 1.4e-4)]  # Myr sim / s wall ; ~18x
STOP_T = 3.0  # steep hybr ran to stop_t without transitioning
REGIME = [  # (label, ratio_end, t_trans, category) -- sorted by ratio_end below
    ("flat\n1e6 n1e5", 0.002, 0.247, "transition"),
    ("typical\n1e6 n1e3", 0.009, 2.5, "transition"),
    ("cost\n1e6 short", 0.316, None, "short"),
    ("steep\n1e6 a-2", 0.386, None, "stall"),
    ("mock\n4e3", 0.410, None, "energy"),
    ("simple\n1e5 sfe.3", 0.827, None, "energy"),
]


def _grouped(ax, rows, li, hi, w=0.38):
    """Plot legacy/hybr grouped bars; rows = (label, legacy, hybr, ...)."""
    x = range(len(rows))
    for i, row in enumerate(rows):
        lo, ho = row[li], row[hi]
        if lo is not None:
            ax.bar(i - w / 2, lo, w, color=L_COL, label="legacy" if i == 0 else None)
            ax.text(i - w / 2, lo, f"{lo:g}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(i - w / 2, 0, "n/a", ha="center", va="bottom", fontsize=6.5, color="0.5")
        ax.bar(i + w / 2, ho, w, color=H_COL, label="hybr" if i == 0 else None)
        ax.text(i + w / 2, ho, f"{ho:g}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows])


def plot_headline(path):
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(12, 9.2))

    # A: convergence
    _grouped(axA, CONV, 1, 2)
    axA.set_ylim(0, 118)
    axA.set_ylabel("converged < 1e-4  (%)")
    axA.set_title("A. Convergence — legacy 0% vs hybr 100%")
    axA.legend(loc="center left", fontsize=9)

    # B: beta_max reach
    _grouped(axB, BETA, 1, 2)
    axB.axhline(
        BETA_BOX_TOP, ls="--", c="crimson", lw=1.2, label=f"legacy box top = {BETA_BOX_TOP:g}"
    )
    axB.set_ylim(0, 4.8)
    axB.set_ylabel(r"$\beta_{\max}$ reached")
    axB.set_title(r"B. $\beta$ reach — hybr exceeds the box (most configs); legacy clamped")
    axB.legend(loc="upper left", fontsize=8)

    # C: transition time (log)
    w = 0.38
    for i, (lab, lo, ho, stall, fac) in enumerate(TRANS):
        axC.bar(i - w / 2, lo, w, color=L_COL, label="legacy" if i == 0 else None)
        axC.text(i - w / 2, lo, f"{lo:g}", ha="center", va="bottom", fontsize=7)
        hbar = STOP_T if stall else ho
        axC.bar(
            i + w / 2,
            hbar,
            w,
            color=H_COL,
            hatch="//" if stall else None,
            label="hybr" if i == 0 else None,
        )
        axC.text(
            i + w / 2,
            hbar,
            "stalls\n>3 Myr" if stall else f"{ho:g}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        axC.text(i, STOP_T * 1.25, fac, ha="center", fontsize=9, color="crimson")
    axC.set_yscale("log")
    axC.set_ylim(3e-3, 9)
    axC.set_xticks(range(len(TRANS)))
    axC.set_xticklabels([t[0] for t in TRANS])
    axC.set_ylabel(r"$t_{\rm trans}$  [Myr]  (log)")
    axC.set_title("C. Transition time — legacy profile-blind, hybr physical")
    axC.legend(loc="upper left", fontsize=8)

    # D: throughput (log)
    for i, (lab, rate) in enumerate(COST):
        axD.bar(i, rate, 0.5, color=H_COL if lab == "hybr" else L_COL)
        axD.text(i, rate, f"{rate:.1e}", ha="center", va="bottom", fontsize=8)
    axD.set_yscale("log")
    axD.set_ylim(3e-6, 4e-4)
    axD.set_xticks(range(len(COST)))
    axD.set_xticklabels([c[0] for c in COST])
    axD.set_ylabel("Myr sim / s wall  (log)")
    axD.set_title("D. Throughput — hybr ~18x faster (matched stop_t)")
    axD.annotate(
        "",
        xy=(1, 1.4e-4),
        xytext=(1, 7.7e-6),
        arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.2),
    )
    axD.text(1.08, 3e-5, "~18x", color="crimson", fontsize=10, va="center")

    for ax in (axA, axB, axC, axD):
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Phase 3 — self-consistent hybr vs legacy (from the master runs table)", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_regime(path):
    rows = sorted(REGIME, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for i, (lab, ratio, ttr, cat) in enumerate(rows):
        ax.bar(i, ratio, 0.6, color=CAT_COL[cat], edgecolor="k", lw=0.4)
        note = f"{ratio:g}" + (f"\ntrans @ {ttr:g} Myr" if ttr is not None else "")
        ax.text(i, ratio * 1.08, note, ha="center", va="bottom", fontsize=7)
    ax.axhline(RATIO_THRESH, ls="--", c="k", lw=1.2)
    ax.text(
        len(rows) - 0.5,
        RATIO_THRESH * 1.1,
        f"transition threshold {RATIO_THRESH:g}",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(r"cooling ratio $(L_{\rm gain}-L_{\rm loss})/L_{\rm gain}$ at end")
    ax.set_title(
        "Phase 3 regimes — flat profiles cross 0.05 (transition); " "steep/low-mass stay above it"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CAT_COL[c])
        for c in ("transition", "stall", "energy", "short")
    ]
    ax.legend(
        handles,
        ["transitions", "stalls (steep)", "energy-driven", "short run"],
        fontsize=8,
        loc="upper left",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    plot_headline(HERE / "phase3_headline.png")
    plot_regime(HERE / "phase3_regime.png")
    print("wrote phase3_headline.png, phase3_regime.png")


if __name__ == "__main__":
    main()
