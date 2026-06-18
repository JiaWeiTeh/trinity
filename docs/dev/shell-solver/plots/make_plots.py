#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five diagnostic plots telling the shell-solver investigation end to end, from
"does any variant even reproduce odeint" to the shipped mxstep fix. Each plot
answers one question and hands off to the next.

Source: the committed 100-implicit matrix CSVs
(docs/dev/shell-solver/data/replay_variants_matrix_<config>.csv). No re-run.

REPRODUCE
    cd /home/user/trinity
    python docs/dev/shell-solver/plots/make_plots.py
Outputs PNGs into docs/dev/shell-solver/plots/.
"""
import csv
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

plt.rcParams.update({"text.usetex": False, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "docs" / "dev" / "shell-solver" / "data"
OUT = Path(__file__).resolve().parent

VARIANTS = ["V_lsoda_teval", "V_lsoda_event", "V_lsoda_dense",
            "V_radau_teval", "V_bdf_teval", "V_odeint_hi"]
VLABEL = {"V_lsoda_teval": "LSODA\nt_eval\n(drop-in)", "V_lsoda_event": "LSODA\n+event\n(smart stop)",
          "V_lsoda_dense": "LSODA\ndense", "V_radau_teval": "Radau", "V_bdf_teval": "BDF",
          "V_odeint_hi": "odeint\nmxstep=50k"}
LSODA_FAMILY = {"V_lsoda_teval", "V_lsoda_event", "V_lsoda_dense", "V_odeint_hi"}

CONFIG_ORDER = ["sfe0.3", "sfe0.6", "probe_typical_hybr", "steep", "dense_flat", "mock_hybr"]
CLABEL = {"sfe0.3": "sfe0.3\n(default)", "sfe0.6": "sfe0.6", "probe_typical_hybr": "typical",
          "steep": "steep", "dense_flat": "dense_flat", "mock_hybr": "mock_hybr"}
DEGEN = {"sfe0.3", "sfe0.6"}
C_DEGEN, C_REAL, C_EVENT, C_BASE = "#d1495b", "#3a7ca5", "#e09f3e", "#8d99ae"
FLOW = "#555"


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load():
    rows = []
    for p in glob.glob(str(DATA / "replay_variants_matrix_*.csv")):
        cfg = Path(p).stem.replace("replay_variants_matrix_", "")
        for r in csv.DictReader(open(p)):
            r["_config"] = cfg
            rows.append(r)
    return rows


ROWS = load()


def sel(variant=None, config=None, phase=None, ok=True):
    out = ROWS
    if variant:
        out = [r for r in out if r["variant"] == variant]
    if config:
        out = [r for r in out if r["_config"] == config]
    if phase:
        out = [r for r in out if r["phase"] == phase]
    if ok:
        out = [r for r in out if r["success"] == "1"]
    return out


def median(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    return xs[len(xs) // 2] if xs else math.nan


def footer(fig, text):
    fig.text(0.5, 0.018, text, ha="center", va="bottom", fontsize=8.5,
             color=FLOW, style="italic", linespacing=1.4)


# ----------------------------------------------------------------------------
# 1 - Equivalence gate: accuracy of every variant vs odeint
# ----------------------------------------------------------------------------
def plot1():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    worst, colors, exact = [], [], []
    for v in VARIANTS:
        rels = [f(r["max_rel_diff_n"]) for r in sel(variant=v)
                if not math.isnan(f(r["max_rel_diff_n"]))]
        w = max(rels) if rels else math.nan
        worst.append(w)
        exact.append(w == 0)
        colors.append(C_BASE if v in LSODA_FAMILY else C_DEGEN)
    disp = [5e-13 if (w == 0 or math.isnan(w)) else w for w in worst]
    x = np.arange(len(VARIANTS))
    ax.bar(x, disp, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yscale("log")
    ax.axhline(1.5e-8, ls="--", color="k", lw=1)
    ax.text(len(VARIANTS) - 0.5, 1.7e-8, "odeint's own rtol (~1.5e-8)",
            ha="right", va="bottom", fontsize=8.5)
    for i, w in enumerate(worst):
        if exact[i]:
            ax.text(i, 6e-13, "exact\n(0)", ha="center", va="bottom", fontsize=8)
        elif not math.isnan(w):
            ax.text(i, w * 1.4, f"{w:.0e}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([VLABEL[v] for v in VARIANTS])
    ax.set_ylabel("worst |Δn|/n vs odeint  (over all 12 cells)")
    ax.set_ylim(3e-13, 1e-6)
    ax.set_title("Step 1/5 — Equivalence gate: every variant's worst-case accuracy vs odeint",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_BASE, ec="k", label="LSODA family (incl. odeint)"),
                       Patch(fc=C_DEGEN, ec="k", label="Radau / BDF (different solver)")],
              loc="upper left", fontsize=9)
    footer(fig, "All LSODA-family variants match odeint to the ~1e-8 agreement floor; Radau/BDF drift to ~1e-7.\n"
                "→ Accuracy rules nothing out, so the decision is about SPEED.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "1_accuracy_gate.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2 - Speed: distribution of per-solve speedup vs odeint, per variant
# ----------------------------------------------------------------------------
def plot2():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    data = []
    for v in VARIANTS:
        sp = [f(r["speedup_vs_odeint"]) for r in sel(variant=v)
              if not math.isnan(f(r["speedup_vs_odeint"])) and f(r["speedup_vs_odeint"]) > 0]
        data.append(sp)
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6,
                    medianprops=dict(color="k", lw=1.5))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=C_EVENT if VARIANTS[i] == "V_lsoda_event" else C_BASE,
                edgecolor="k", alpha=0.9)
    ax.set_yscale("log")
    ax.axhline(1.0, ls="--", color=C_DEGEN, lw=1.5)
    ax.text(0.6, 1.08, "1.0× = current default odeint  (above = faster, below = slower)",
            color=C_DEGEN, fontsize=9, va="bottom")
    ax.set_xticks(np.arange(1, len(VARIANTS) + 1))
    ax.set_xticklabels([VLABEL[v] for v in VARIANTS])
    ax.set_ylabel("speedup vs odeint  (per captured solve)")
    ax.set_title("Step 2/5 — Speed: is anything faster than the current default?",
                 fontweight="bold")
    footer(fig, "Everything sits mostly below 1.0× (slower); only LSODA+event pokes above, and only partly.\n"
                "→ Where exactly does the event win?")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "2_speed_distribution.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3 - The event win is energy-phase-only
# ----------------------------------------------------------------------------
def plot3():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(CONFIG_ORDER))
    w = 0.38
    en = [median([f(r["speedup_vs_odeint"]) for r in sel("V_lsoda_event", c, "energy")])
          for c in CONFIG_ORDER]
    im = [median([f(r["speedup_vs_odeint"]) for r in sel("V_lsoda_event", c, "implicit")])
          for c in CONFIG_ORDER]
    ax.bar(x - w / 2, en, w, label="energy phase", color=C_EVENT, edgecolor="k", lw=0.5)
    ax.bar(x + w / 2, im, w, label="implicit phase", color=C_REAL, edgecolor="k", lw=0.5)
    ax.axhline(1.0, ls="--", color=C_DEGEN, lw=1.5)
    ax.text(len(CONFIG_ORDER) - 0.5, 1.05, "1.0× (break-even)", ha="right",
            color=C_DEGEN, fontsize=9)
    for i, (e, m) in enumerate(zip(en, im)):
        if not math.isnan(e):
            ax.text(i - w / 2, e + 0.1, f"{e:.1f}×", ha="center", fontsize=8)
        if not math.isnan(m):
            ax.text(i + w / 2, m + 0.1, f"{m:.2f}×", ha="center", fontsize=8)
    ax.axvspan(-0.5, 1.5, color=C_DEGEN, alpha=0.07)
    ax.text(0.5, ax.get_ylim()[1] * 0.92, "degenerate", ha="center", color=C_DEGEN, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([CLABEL[c] for c in CONFIG_ORDER])
    ax.set_ylabel("LSODA+event median speedup vs odeint")
    ax.set_title("Step 3/5 — The event speedup is ENERGY-phase-only", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    footer(fig, "The 4–5× event win lives only in the degenerate ENERGY phase; in implicit it collapses to ~0.5× (a loss), everywhere.\n"
                "→ Why does it vanish in the implicit phase?")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "3_event_energy_only.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4 - Why: phase composition (ionised / neutral / mass-limited)
# ----------------------------------------------------------------------------
def comp(config, phase):
    base = sel("V_lsoda_teval", config, phase)          # one row per captured call
    n = len(base)
    if n == 0:
        return None
    neu = sum(1 for r in base if r["is_ionised"] == "0") / n
    ml = sum(1 for r in base if r["idx_phi"] == "-1") / n
    return neu, ml


def plot4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    x = np.arange(len(CONFIG_ORDER))
    for ax, phase in zip(axes, ("energy", "implicit")):
        neu = [comp(c, phase)[0] * 100 if comp(c, phase) else 0 for c in CONFIG_ORDER]
        ion = [100 - v for v in neu]
        ml = [comp(c, phase)[1] * 100 if comp(c, phase) else 0 for c in CONFIG_ORDER]
        ax.bar(x, ion, color=C_EVENT, edgecolor="k", lw=0.5, label="ionised")
        ax.bar(x, neu, bottom=ion, color=C_REAL, edgecolor="k", lw=0.5, label="neutral")
        ax.plot(x, ml, "ks", ms=7, label="mass-limited (idx_phi=-1)")
        for i, m in enumerate(ml):
            ax.text(i, m + 3, f"{m:.0f}%", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([CLABEL[c] for c in CONFIG_ORDER], fontsize=8)
        ax.set_title(f"{phase} phase", fontsize=11)
        ax.set_ylim(0, 109)
    axes[0].set_ylabel("% of captured solves")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Step 4/5 — Why: the implicit phase is mixed ionised/neutral + mass-limited",
                 fontweight="bold", y=0.98)
    footer(fig, "Energy solves are ~100% ionised (the event can skip the φ-overflow tail); implicit is mostly neutral / mass-limited, so the event has nothing to skip.\n"
                "→ No speed case anywhere. What about the warning we set out to fix?")
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(OUT / "4_phase_composition.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5 - The free fix: warning is localized; mxstep kills it at ~1.0x, bit-identical
# ----------------------------------------------------------------------------
def plot5():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cells, ew, sp_hi, colors = [], [], [], []
    for c in CONFIG_ORDER:
        for ph in ("energy", "implicit"):
            base = sel("V_lsoda_teval", c, ph)
            if not base:
                continue
            cells.append(f"{CLABEL[c].splitlines()[0]}\n{ph}")
            ew.append(100 * sum(1 for r in base if f(r["baseline_odeint_py_warns"]) > 0) / len(base))
            sp_hi.append(median([f(r["speedup_vs_odeint"]) for r in sel("V_odeint_hi", c, ph)]))
            colors.append(C_DEGEN if c in DEGEN else C_REAL)
    x = np.arange(len(cells))
    ax.bar(x, ew, color=colors, edgecolor="k", lw=0.5, alpha=0.85)
    ax.set_ylabel("excess-work warning  (% of solves)")
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=7, rotation=45, ha="right")
    ax2 = ax.twinx()
    ax2.plot(x, sp_hi, "D", color="k", ms=6, label="odeint(mxstep=50k) speed")
    ax2.axhline(1.0, ls="--", color="k", lw=1)
    ax2.set_ylabel("odeint(mxstep=50k) speedup  (◆)")
    ax2.set_ylim(0, 1.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=C_DEGEN, label="degenerate (warning lives here)"),
                       Patch(fc=C_REAL, label="realistic"),
                       plt.Line2D([], [], marker="D", color="k", ls="", label="mxstep=50k speed (rel_n=0)")],
              loc="center right", fontsize=8.5)
    ax.set_title("Step 5/5 — The fix: warning is localized; mxstep removes it (bit-identical)",
                 fontweight="bold")
    footer(fig, "The warning means odeint TRUNCATED the solve. mxstep=50k completes it: free (◆≈1.0×) in the science configs / implicit cells where the ceiling was barely hit,\n"
                "and ~0.2× in the degenerate energy phase — it now does the heavy overflow work the warning was hiding. rel_n=0 throughout. Shipped; no solver migration needed.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "5_free_fix_mxstep.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plot1(); plot2(); plot3(); plot4(); plot5()
    print("wrote 5 plots ->", OUT)
