#!/usr/bin/env python3
"""P-validate figures: is cooling_or_blowout a better transition trigger?

Reads only COMMITTED CSVs (no sim re-run):
  docs/dev/data/pvalidate_steep_cob_full.csv      steep cob, all 4 phases
  docs/dev/data/pvalidate_dense_flat_cob_full.csv dense-flat cob, all 4 phases
  docs/dev/data/pvalidate_steep_cob.csv           steep cob harvest (implicit)
  docs/dev/data/pvalidate_dense_flat_cob.csv      dense-flat cob harvest (implicit)
  docs/dev/data/transition_steep_long.csv         P0 steep INSTANTANEOUS baseline

Outputs (docs/dev/transition/figures/):
  pvalidate_continuity_handoff.png   Eb/R2/v2/P_drive across the blowout switch
  pvalidate_trigger_compare_steep.png  steep R2(t)/v2(t): new vs current
  pvalidate_steep_F0_broken.png      why F0 can't fire for steep (F4 does)
  pvalidate_retained_scorecard.png   retained-energy at firing vs the eta band
  pvalidate_phase_timeline.png       phase-outcome Gantt per config x trigger

Usage:  python docs/dev/transition/harness/make_pvalidate_figures.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
DATA = ROOT / "docs" / "dev" / "data"
FIGS = ROOT / "docs" / "dev" / "transition" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

WONG = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
        "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
        "vermillion": "#D55E00", "purple": "#CC79A7"}
plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                     "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 130,
                     "savefig.bbox": "tight"})

EPS, RC_LINE = 0.05, 1.0
ETA_LO, ETA_HI = 0.10, 0.25  # Lancaster retained-energy band

steep_full = pd.read_csv(DATA / "pvalidate_steep_cob_full.csv")
dense_full = pd.read_csv(DATA / "pvalidate_dense_flat_cob_full.csv")
steep_h = pd.read_csv(DATA / "pvalidate_steep_cob.csv")
dense_h = pd.read_csv(DATA / "pvalidate_dense_flat_cob.csv")
steep_inst = pd.read_csv(DATA / "transition_steep_long.csv")  # P0 baseline


def phase_bounds(df):
    """{phase: (t_start, t_end)} from a full-trajectory frame."""
    out, cur, t0 = {}, None, None
    t = df["t_now"].to_numpy(float)
    ph = df["current_phase"].astype(str).to_numpy()
    for i in range(len(df)):
        if ph[i] != cur:
            if cur is not None:
                out[cur] = (t0, t[i])
            cur, t0 = ph[i], t[i]
    out[cur] = (t0, t[-1])
    return out


def first_cross_row(df, col, thr, above):
    s = df[col].to_numpy(float)
    m = (s > thr) if above else (s < thr)
    idx = np.flatnonzero(m & np.isfinite(s))
    return int(idx[0]) if idx.size else None


# ── Fig 1 — continuity handoff across the blowout switch ──────────────────
def fig_handoff():
    sb = phase_bounds(steep_full)
    tsw = sb["transition"][0]  # implicit->transition boundary
    w = steep_full[(steep_full["t_now"] > tsw - 0.05) &
                   (steep_full["t_now"] < tsw + 0.05)]
    t = w["t_now"].to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    panels = [("Eb", "Eb (bubble energy)", WONG["black"]),
              ("R2", "R2 (shell radius) [pc]", WONG["blue"]),
              ("v2", "v2 (shell velocity)", WONG["green"]),
              ("P_drive", "P_drive = max(Pb, P_HII)", WONG["vermillion"])]
    for ax, (col, lab, c) in zip(axes.ravel(), panels):
        ax.plot(t, w[col], color=c, lw=2, marker="o", ms=3)
        ax.axvline(tsw, color=WONG["purple"], lw=1.2, ls="--")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("t [Myr]")
    axes[0, 0].annotate("blowout switch\n1b → 1c", (tsw, axes[0, 0].get_ylim()[1]),
                        fontsize=8.5, color=WONG["purple"], ha="center", va="top")
    fig.suptitle("Continuity gate — Eb/R2/v2/P_drive hand off smoothly at blowout "
                 f"(t={tsw:.3f} Myr)\nP_drive moves <0.2%: no injected pressure jump",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / "pvalidate_continuity_handoff.png"
    fig.savefig(out); plt.close(fig); return out


# ── Fig 2 — steep R2(t)/v2(t): new trigger vs current ─────────────────────
def fig_compare_steep():
    blow = steep_h["t_now"].to_numpy(float)[
        first_cross_row(steep_h, "R2_over_rCloud", RC_LINE, True)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
    ax1.plot(steep_inst["t_now"], steep_inst["R2"], color=WONG["orange"], lw=2,
             label="instantaneous (current) — energy-driven to 4 Myr")
    ax1.plot(steep_full["t_now"], steep_full["R2"], color=WONG["blue"], lw=2,
             label="cooling_or_blowout (new) — blowout → 1c → momentum")
    ax1.set_ylabel("R2 [pc]")
    ax2.plot(steep_inst["t_now"], steep_inst["v2"], color=WONG["orange"], lw=2,
             label="instantaneous (current)")
    ax2.plot(steep_full["t_now"], steep_full["v2"], color=WONG["blue"], lw=2,
             label="cooling_or_blowout (new)")
    ax2.set_ylabel("v2 (shell velocity)")
    ax2.set_xlabel("t [Myr]")
    ax2.set_ylim(0, 18)  # clip the t~0 launch transient (~3700) to the relevant band
    ax2.annotate("instantaneous: 14.8", (4.0, 14.8), color=WONG["orange"],
                 fontsize=8, ha="right", va="bottom")
    ax2.annotate("blowout: 6.1", (4.0, 6.06), color=WONG["blue"],
                 fontsize=8, ha="right", va="top")
    for ax in (ax1, ax2):
        ax.axvline(blow, color=WONG["vermillion"], lw=1.3, ls="-.")
    ax1.annotate(f"blowout {blow:.3f} Myr", (blow, ax1.get_ylim()[1] * 0.5),
                 color=WONG["vermillion"], fontsize=9, ha="right")
    ax1.legend(fontsize=8.5, loc="lower right")
    ax1.set_title("Steep cloud: the new trigger transitions at blowout, the current one "
                  "never does\n→ smaller (R2 −12.5%), slower (v2 −59%) by 4 Myr", fontsize=10.5)
    fig.tight_layout()
    out = FIGS / "pvalidate_trigger_compare_steep.png"
    fig.savefig(out); plt.close(fig); return out


# ── Fig 3 — why F0 can't fire for steep, but F4 does ──────────────────────
def fig_F0_broken():
    t = steep_h["t_now"].to_numpy(float)
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(t, steep_h["ratio_F0"], color=WONG["blue"], lw=2,
             label="ratio_F0 (current trigger)")
    ax1.axhline(EPS, color=WONG["blue"], lw=1, ls=":")
    ax1.annotate("F0 threshold 0.05 — ratio_F0 never reaches it (current trigger CAN'T fire)",
                 (t[len(t)//6], EPS), fontsize=8.5, color=WONG["blue"], va="bottom")
    ax1.set_ylabel("ratio_F0 (cooling balance)", color=WONG["blue"])
    ax1.set_ylim(0, 0.8)
    ax2 = ax1.twinx()
    ax2.plot(t, steep_h["R2_over_rCloud"], color=WONG["vermillion"], lw=2,
             label="R2 / rCloud (new trigger)")
    ax2.axhline(RC_LINE, color=WONG["vermillion"], lw=1, ls=":")
    blow = t[first_cross_row(steep_h, "R2_over_rCloud", RC_LINE, True)]
    ax2.axvline(blow, color=WONG["vermillion"], lw=1.2, ls="-.")
    ax2.annotate(f"F4 blowout fires\n{blow:.3f} Myr", (blow, 1.0),
                 color=WONG["vermillion"], fontsize=9, ha="right", va="bottom")
    ax2.set_ylabel("R2 / rCloud (blowout)", color=WONG["vermillion"])
    ax1.set_xlabel("t [Myr]")
    ax1.set_title("Steep cloud: the CURRENT cooling trigger is blind — it never fires;\n"
                  "the NEW blowout criterion fires at the physical escape", fontsize=10.5)
    fig.tight_layout()
    out = FIGS / "pvalidate_steep_F0_broken.png"
    fig.savefig(out); plt.close(fig); return out


# ── Fig 4 — retained-energy scorecard vs the eta band ─────────────────────
def fig_retained():
    # retained-at-firing (Eb/∫Lgain) from the committed harvest CSVs
    iF0_dense = first_cross_row(dense_h, "ratio_F0", EPS, False)
    iF4_steep = first_cross_row(steep_h, "R2_over_rCloud", RC_LINE, True)
    r_dense = float(dense_h["Eb_over_cumLgain"].to_numpy(float)[iF0_dense])
    r_steep = float(steep_h["Eb_over_cumLgain"].to_numpy(float)[iF4_steep])
    f0_steep_fires = first_cross_row(steep_inst, "ratio_F0", EPS, False) is not None

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhspan(ETA_LO, ETA_HI, color=WONG["green"], alpha=0.13, zorder=0)
    ax.text(1.5, (ETA_LO + ETA_HI) / 2, "literature η ≈ 0.1–0.25",
            color=WONG["green"], fontsize=9, ha="center", va="center")
    x = {"dense-flat": 0, "steep": 1}
    # current (instantaneous)
    ax.scatter(x["dense-flat"] - 0.12, r_dense, s=120, color=WONG["orange"],
               edgecolors="white", zorder=3, label="instantaneous (current)")
    ax.scatter(x["dense-flat"] + 0.12, r_dense, s=120, color=WONG["blue"],
               edgecolors="white", zorder=3, label="cooling_or_blowout (new)")
    ax.scatter(x["steep"] + 0.12, r_steep, s=120, color=WONG["blue"],
               edgecolors="white", zorder=3)
    # steep current: F0 never fires -> no transition
    if not f0_steep_fires:
        ax.scatter(x["steep"] - 0.12, 0.42, marker="X", s=130,
                   color=WONG["vermillion"], zorder=3)
        ax.annotate("current trigger\nNEVER FIRES\n(no transition)",
                    (x["steep"] - 0.12, 0.46), color=WONG["vermillion"], fontsize=8.5,
                    ha="center", va="bottom")
    ax.annotate(f"{r_dense:.3f}", (x['dense-flat'] + 0.12, r_dense), fontsize=8,
                ha="left", va="bottom")
    ax.annotate(f"{r_steep:.3f}", (x['steep'] + 0.12, r_steep), fontsize=8,
                ha="left", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["dense-flat", "steep"])
    ax.set_xlim(-0.5, 1.6); ax.set_ylim(0, 0.62)
    ax.set_ylabel("retained-energy Eb/∫Lgain dt  at firing")
    ax.set_title("Retained-energy scorecard — the new trigger lands in the physical band\n"
                 "for BOTH regimes; the current trigger can't fire for steep", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    out = FIGS / "pvalidate_retained_scorecard.png"
    fig.savefig(out); plt.close(fig); return out


# ── Fig 5 — phase-outcome timeline (Gantt) ────────────────────────────────
def fig_timeline():
    PH_COL = {"energy": WONG["skyblue"], "implicit": WONG["blue"],
              "transition": WONG["orange"], "momentum": WONG["green"]}
    sb, db = phase_bounds(steep_full), phase_bounds(dense_full)
    # instantaneous steep: energy then implicit to 4 (never transitions)
    inst = {"energy": (0.0, 0.003),
            "implicit": (0.003, float(steep_inst["t_now"].max()))}
    rows = [("steep · instantaneous (current)", inst),
            ("steep · cooling_or_blowout (new)", sb),
            ("dense-flat · both triggers", db)]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for y, (label, bounds) in enumerate(rows):
        for ph, (a, b) in bounds.items():
            ax.barh(y, b - a, left=a, height=0.55,
                    color=PH_COL.get(ph, "0.6"), edgecolor="white")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("t [Myr]")
    ax.set_xlim(0, 4.05)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PH_COL.values()]
    ax.legend(handles, PH_COL.keys(), ncol=4, fontsize=8.5, loc="lower right",
              framealpha=0.9)
    ax.set_title("Phase outcome per trigger — the current trigger leaves the steep cloud\n"
                 "energy-driven forever; the new trigger routes it through transition→momentum",
                 fontsize=10.5)
    fig.tight_layout()
    out = FIGS / "pvalidate_phase_timeline.png"
    fig.savefig(out); plt.close(fig); return out


def main():
    for f in (fig_handoff, fig_compare_steep, fig_F0_broken, fig_retained, fig_timeline):
        print("wrote", f().relative_to(ROOT))


if __name__ == "__main__":
    main()
