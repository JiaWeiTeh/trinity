#!/usr/bin/env python3
"""Render the P0 / P-sens result figures from the committed harvest CSVs.

Pure offline read of docs/dev/data/transition_*.csv (produced by harvest.py) —
no simulation is run, so a future visit reproduces the figures *without*
re-running the hours-long hybr sims (TRIGGER_PLAN.md 💾 banner).

Every candidate firing epoch is RE-DERIVED from the CSV columns here (not copied
from P0.md), so the figures and the summary table stay self-consistent with the
data. Printed epochs double as a self-check against P0.md.

Outputs (docs/dev/transition/figures/):
  p0_divergence_map.png   — firing epoch of F0/F1/F2/F4/Eb-peak for all configs
  p0_overlay_dense_flat.png — clean cooling transition (F0=F1=Eb-peak)
  p0_overlay_steep_long.png — blowout crux (R2>rCloud before the surge)
  p0_reset_pathology.png  — F0 resets up / F1 drifts down across the WR surge
  p0_divergence_map.csv   — the table behind the divergence-map figure

Usage:  python docs/dev/transition/harness/make_p0_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --- repo-relative paths -------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]  # harness→transition→dev→docs→repo-root
DATA = ROOT / "docs" / "dev" / "data"
FIGS = ROOT / "docs" / "dev" / "transition" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# --- candidate thresholds (TRIGGER_PLAN.md F0–F4) ------------------------
EPS = 0.05        # F0: (Lgain-Lloss)/Lgain < EPS
ETA = 0.30        # F1: frac_cum > 1-ETA  (η≈0.3 is the flat-config anchor)
K2 = 1.0          # F2: t_cool/t_dyn < K2

# --- Wong colour-blind-safe palette + mathtext math rendering ------------
# text.usetex stays False (no system LaTeX in this container); matplotlib's
# built-in mathtext renders the $...$ math (subscripts, integrals, Greek)
# internally, so labels read as real equations without a TeX install.
WONG = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def _halo(ec: str) -> dict:
    """White rounded box behind an on-plot annotation so the text stays
    legible where it would otherwise sit on top of a curve."""
    return dict(boxstyle="round,pad=0.25", fc="white", ec=ec, lw=0.8, alpha=0.85)


# config -> (csv stem, pretty label, fate). Order = low→high duration.
CONFIGS = [
    ("transition_mock4e3", "mock 4e3 (legacy)", "energy-driven"),
    ("transition_mock_hybr", "mock 4e3 (hybr)", "energy-driven"),
    ("transition_dense_flat", "dense-flat n1e5", "transitions"),
    ("transition_steep", "steep α−2 (1 Myr)", "stall"),
    ("transition_steep_long", "steep α−2 (4 Myr)", "blowout"),
]


def load(stem: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{stem}.csv")
    # candidate firing is evaluated on the implicit-phase trajectory
    if "current_phase" in df.columns:
        imp = df[df["current_phase"] == "implicit"].copy()
        if not imp.empty:
            df = imp
    return df.reset_index(drop=True)


def first_cross(t, series, thresh, below: bool):
    """First time `series` crosses `thresh` (below=True → series<thresh)."""
    s = np.asarray(series, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = (s < thresh) if below else (s > thresh)
    idx = np.flatnonzero(mask & np.isfinite(s))
    return float(t[idx[0]]) if idx.size else None


def epochs(df: pd.DataFrame) -> dict:
    t = df["t_now"].to_numpy(float)
    eb = df["Eb"].to_numpy(float)
    return {
        "F0": first_cross(t, df["ratio_F0"], EPS, below=True),
        "F1": first_cross(t, df["frac_cum"], 1 - ETA, below=False),
        "F2": first_cross(t, df["F2_tcool_tdyn"], K2, below=True),
        "F4": first_cross(t, df["R2_over_rCloud"], 1.0, below=False),
        "Ebpeak": float(t[int(np.argmax(eb))]),
        "retained_end": float(1.0 - df["frac_cum"].to_numpy(float)[-1]),
        "tmax": float(t[-1]),
    }


def surge_time(df: pd.DataFrame):
    """First WR/SN feedback reset: beta_plus_delta goes negative (t>1 Myr)."""
    t = df["t_now"].to_numpy(float)
    bpd = df["beta_plus_delta"].to_numpy(float)
    idx = np.flatnonzero((bpd < 0) & (t > 1.0) & np.isfinite(bpd))
    return float(t[idx[0]]) if idx.size else None


# =========================================================================
# Figure 1 — divergence map (headline)
# =========================================================================
def fig_divergence(all_ep):
    style = {
        "F0": (WONG["blue"], "o", r"F0 cooling $(<0.05)$"),
        "F1": (WONG["green"], "s", r"F1 cumulative $(\eta=0.3)$"),
        "F2": (WONG["yellow"], "v", r"F2 $t_\mathrm{cool}/t_\mathrm{dyn}<1$"),
        "F4": (WONG["vermillion"], "D", r"F4 blowout $(R_2>R_\mathrm{cloud})$"),
        "Ebpeak": (WONG["black"], "*", r"$E_b$-peak (reference)"),
    }
    labels = [c[1] for c in CONFIGS]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    never_x = 6.5  # parking column for non-firing candidates (log axis)
    for row, (stem, label, fate) in enumerate(CONFIGS):
        ep = all_ep[stem]
        for key, (col, mk, _) in style.items():
            val = ep[key]
            if val is None or val <= 0:
                ax.scatter(never_x, row, marker=mk, s=70, facecolors="none",
                           edgecolors=col, linewidths=1.3, alpha=0.6, zorder=3)
            else:
                ax.scatter(val, row, marker=mk, s=95 if key != "Ebpeak" else 150,
                           color=col, edgecolors="white", linewidths=0.6, zorder=4)
    ax.axvspan(never_x * 0.78, never_x * 1.3, color="0.92", zorder=0)
    ax.text(never_x, len(CONFIGS) - 0.4, "never\nfires", ha="center",
            va="bottom", fontsize=8.5, color="0.4")
    ax.set_xscale("log")
    ax.set_xlim(2e-3, never_x * 1.4)
    ax.set_yticks(range(len(CONFIGS)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(CONFIGS) - 0.4)
    ax.set_xlabel("firing time  [Myr, log]")
    ax.set_title("P0 divergence map — candidate triggers fire at different epochs\n"
                 r"flat configs $\to$ cooling (F0/F1) at the $E_b$-peak;  "
                 r"steep $\to$ blowout (F4)",
                 fontsize=11)
    handles = [plt.Line2D([], [], marker=mk, color=col, linestyle="none",
                          markersize=9, label=lab) for col, mk, lab in style.values()]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.9,
              ncol=2)
    fig.tight_layout()
    out = FIGS / "p0_divergence_map.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =========================================================================
# Figure 2 — dense-flat overlay (clean cooling transition)
# =========================================================================
def fig_dense_flat():
    df = load("transition_dense_flat")
    ep = epochs(df)
    t = df["t_now"].to_numpy(float)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
    ax1.plot(t, df["Eb"], color=WONG["black"], lw=2, label=r"$E_b$ (bubble energy)")
    ax1.set_xlabel(r"$t$  [Myr]")
    ax1.set_ylabel(r"$E_b$  [code units]")
    ax2 = ax1.twinx()
    ax2.plot(t, df["ratio_F0"], color=WONG["blue"], lw=1.6,
             label=r"F0  $(L_\mathrm{gain}-L_\mathrm{loss})/L_\mathrm{gain}$")
    ax2.plot(t, df["frac_cum"], color=WONG["green"], lw=1.6, ls="--",
             label=r"F1  $\int L_\mathrm{loss}\,dt\,/\!\int L_\mathrm{gain}\,dt$")
    ax2.axhline(EPS, color=WONG["blue"], lw=0.9, ls=":", alpha=0.7)
    ax2.axhline(1 - ETA, color=WONG["green"], lw=0.9, ls=":", alpha=0.7)
    ax2.set_ylabel("retention ratios")
    # F0 and the Eb-peak coincide here (the headline) — draw one merged marker
    if ep["F0"] and ep["Ebpeak"] and abs(ep["F0"] - ep["Ebpeak"]) < 1e-3:
        ax1.axvline(ep["F0"], color=WONG["vermillion"], lw=1.4, ls="-.", alpha=0.85)
        ax1.text(ep["F0"] * 0.97, ax1.get_ylim()[1] * 0.52,
                 f"F0 = F1 = $E_b$-peak\n{ep['F0']:.3f} Myr", ha="right",
                 va="center", color=WONG["vermillion"], fontsize=9,
                 bbox=_halo(WONG["vermillion"]))
    else:
        for key, col, lab in [("Ebpeak", WONG["vermillion"], r"$E_b$-peak"),
                              ("F0", WONG["blue"], "F0 fires")]:
            if ep[key]:
                ax1.axvline(ep[key], color=col, lw=1.3, ls="-.", alpha=0.8)
                ax1.text(ep[key], ax1.get_ylim()[1] * 0.96,
                         f" {lab}\n {ep[key]:.3f}", color=col, fontsize=8.5,
                         va="top", bbox=_halo(col))
    ax1.set_title(r"Dense-flat (n1e5): cooling works — "
                  r"F0 $\approx$ F1$(\eta{=}0.3)$ $\approx$ $E_b$-peak")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    out = FIGS / "p0_overlay_dense_flat.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =========================================================================
# Figure 3 — steep-long overlay (blowout crux)
# =========================================================================
def fig_steep_long():
    df = load("transition_steep_long")
    ep = epochs(df)
    surge = surge_time(df)
    t = df["t_now"].to_numpy(float)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
    ax1.plot(t, df["Eb"], color=WONG["black"], lw=2, label=r"$E_b$ (still rising)")
    ax1.set_xlabel(r"$t$  [Myr]")
    ax1.set_ylabel(r"$E_b$  [code units]")
    ax2 = ax1.twinx()
    ax2.plot(t, df["R2_over_rCloud"], color=WONG["vermillion"], lw=1.8,
             label=r"$R_2/R_\mathrm{cloud}$")
    ax2.axhline(1.0, color=WONG["vermillion"], lw=0.9, ls=":", alpha=0.8)
    ax2.plot(t, df["ratio_F0"], color=WONG["blue"], lw=1.3, alpha=0.7,
             label=r"F0 ratio (never $<0.05$)")
    ax2.set_ylabel(r"$R_2/R_\mathrm{cloud}$   &   F0 ratio")
    if ep["F4"]:
        ax1.axvline(ep["F4"], color=WONG["vermillion"], lw=1.4, ls="-.")
        ax1.text(ep["F4"] * 0.985, ax1.get_ylim()[1] * 0.40,
                 f"blowout\n{ep['F4']:.3f} Myr", color=WONG["vermillion"],
                 fontsize=9, ha="right", va="center",
                 bbox=_halo(WONG["vermillion"]))
    if surge:
        ax1.axvline(surge, color=WONG["purple"], lw=1.2, ls="--")
        ax1.text(surge * 1.012, ax1.get_ylim()[1] * 0.88,
                 f"WR surge\n{surge:.3f}", color=WONG["purple"],
                 fontsize=8.5, ha="left", va="center", bbox=_halo(WONG["purple"]))
    ax1.set_title(r"Steep $\alpha=-2$ (4 Myr): transitions by BLOWOUT before "
                  "the surge —\ncooling (F0 ratio) never reaches 0.05")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    out = FIGS / "p0_overlay_steep_long.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =========================================================================
# Figure 4 — reset pathology (F0 spikes up, F1 drifts down)
# =========================================================================
def fig_reset():
    df = load("transition_steep_long")
    surge = surge_time(df)
    t = df["t_now"].to_numpy(float)
    # zoom on the surge window
    lo = (surge - 0.6) if surge else t[0]
    win = df[df["t_now"] >= lo]
    tt = win["t_now"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(tt, win["ratio_F0"], color=WONG["blue"], lw=2, marker="o", ms=3,
            label=r"F0  $(L_\mathrm{gain}-L_\mathrm{loss})/L_\mathrm{gain}$ — resets UP")
    ax.plot(tt, win["frac_cum"], color=WONG["green"], lw=2, marker="s", ms=3,
            label=r"F1  $\int L_\mathrm{loss}\,dt\,/\!\int L_\mathrm{gain}\,dt$ — drifts DOWN")
    ax.axhline(EPS, color=WONG["blue"], lw=0.9, ls=":", alpha=0.7)
    ax.axhline(1 - ETA, color=WONG["green"], lw=0.9, ls=":", alpha=0.7)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.10)  # headroom so the surge label clears the frame
    if surge:
        ax.axvline(surge, color=WONG["purple"], lw=1.2, ls="--")
        ax.text(surge * 1.005, ax.get_ylim()[1] * 0.95, f"WR surge {surge:.3f}",
                color=WONG["purple"], fontsize=8.5, ha="left", va="top",
                bbox=_halo(WONG["purple"]))
    ax.set_xlabel(r"$t$  [Myr]")
    ax.set_ylabel("retention ratio")
    ax.set_title("Steep reset pathology: neither cooling form rescues the stall\n"
                 r"F0 spikes away from $0.05$ · F1 retreats from $1-\eta$")
    ax.legend(loc="center left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    out = FIGS / "p0_reset_pathology.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# =========================================================================
# Summary table
# =========================================================================
def write_table(all_ep):
    out = FIGS / "p0_divergence_map.csv"
    cols = ["config", "fate", "F0_cooling", "F1_cum_eta0.3", "F2_tcool_tdyn",
            "F4_blowout", "Eb_peak", "retained_at_end", "t_max"]

    def fmt(v):
        return "never" if v is None else f"{v:.3f}"

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for stem, label, fate in CONFIGS:
            ep = all_ep[stem]
            w.writerow([label, fate, fmt(ep["F0"]), fmt(ep["F1"]), fmt(ep["F2"]),
                        fmt(ep["F4"]), fmt(ep["Ebpeak"]),
                        f"{ep['retained_end']:.3f}", f"{ep['tmax']:.3f}"])
    return out


def main():
    all_ep = {stem: epochs(load(stem)) for stem, _, _ in CONFIGS}
    print("Derived firing epochs (Myr) — self-check vs P0.md:")
    for stem, label, _ in CONFIGS:
        e = all_ep[stem]
        print(f"  {label:22s} F0={e['F0']!s:>7} F1={e['F1']!s:>7} "
              f"F2={e['F2']!s:>7} F4={e['F4']!s:>7} Ebpeak={e['Ebpeak']:.3f} "
              f"retained={e['retained_end']:.2f}")
    outs = [
        fig_divergence(all_ep),
        fig_dense_flat(),
        fig_steep_long(),
        fig_reset(),
        write_table(all_ep),
    ]
    print("\nWrote:")
    for o in outs:
        print(f"  {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
