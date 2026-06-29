#!/usr/bin/env python3
"""Figures for the interface-cooling closure shadow test (companion to make_closure_test.py).

NO simulations. Reads the per-step CSVs and the long-form closure_test.csv this script's sibling
emits, and renders the methods-note figures under docs/dev/transition/pdv-trigger/. Plain
matplotlib, NO usetex, dpi=130.

EVERY fire-time plotted is a FROZEN-TRAJECTORY SCREEN, not a prediction: the CSVs were produced
with production cooling, so boosting the effective loss in post does not move the actual
Pb -> PdV -> blowout the boosted run would have taken. Markers/annotations say "screen", not
"forecast".

Run from the repo root (regenerates closure_test.csv first via the sibling, then plots):
  python docs/dev/transition/pdv-trigger/data/make_closure_plots.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Run from the repo root (paths below are repo-root-relative, like make_combined_trigger_table.py)
# but import the sibling regenerator regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_closure_test as ct

OUT = "docs/dev/transition/pdv-trigger"
EPS = ct.EPS_TR
F_GRID = ct.F_GRID
THETA_GRID = ct.THETA_GRID
NORMALS = ["simple_cluster", "small_dense_highsfe", "midrange_pl0",
           "pl2_steep", "be_sphere", "large_diffuse_lowsfe"]


def ratio_series(cfg, closure, value, form):
    """Recompute the (t, r) frozen-trajectory ratio for a config/closure/value/form from SERIES."""
    s = ct.SERIES[cfg]
    fr = s["frame"]
    lm = fr["Lmech"]
    leff = ct.lloss_eff(closure, value, fr["Lcool"], fr["Lleak"], lm)
    if form == "cb":
        r = (lm - leff) / lm
    else:
        r = (lm - leff - fr["PdV"]) / lm
    r = r.replace([np.inf, -np.inf], np.nan)
    return fr["t"], r, s


# ---------------------------------------------------------------- Stage 1: baseline gate audit
def stage1_gate():
    cfgs = [c for c in NORMALS if c in ct.SERIES]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(0.0, EPS, color="tab:red", alpha=0.12, zorder=0)
    ax.axhline(EPS, color="tab:red", lw=0.9, ls="--", zorder=1)
    handles = []
    for i, cfg in enumerate(cfgs):
        c = colors[i % len(colors)]
        t, r_cb, s = ratio_series(cfg, "none", None, "cb")
        _, r_pdv, _ = ratio_series(cfg, "none", None, "pdv")
        ax.plot(t, r_cb, "-", color=c, lw=1.7, zorder=3)
        ax.plot(t, r_pdv, "--", color=c, lw=1.4, zorder=3)
        handles.append(Line2D([0], [0], color=c, lw=1.7, label=cfg))
        bw = s["blowout"]
        if bw is not None:
            bt = bw[0]
            # mark r at blowout on both curves
            idx = int((t - bt).abs().argmin())
            ax.plot(bt, r_cb.iloc[idx], "o", mfc="white", mec=c, mew=1.6, ms=9, zorder=5)
            ax.plot(bt, r_pdv.iloc[idx], "o", mfc="white", mec=c, mew=1.6, ms=9, zorder=5)
    style = [
        Line2D([0], [0], color="0.25", lw=1.7, ls="-", label="cb (note):  (Lmech-Lloss)/Lmech"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="--",
               label="pdv:  (Lmech-Lloss-PdV)/Lmech"),
        Line2D([0], [0], color="0.25", marker="o", mfc="white", mew=1.6, ls="none", ms=9,
               label="blowout: R2 = rCloud"),
        Patch(facecolor="tab:red", alpha=0.12, label=f"trigger band (r < {EPS})"),
    ]
    leg1 = ax.legend(handles=handles, title="config", loc="upper right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=style, title="trigger form", loc="lower left", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("t  [Myr]  (log)")
    ax.set_ylabel("trigger ratio  r")
    ax.set_title("Stage 1 gate audit (closure=none): baseline never trips cb; pdv only the "
                 "diffuse cloud\n[FROZEN-TRAJECTORY SCREEN]")
    fig.tight_layout()
    p = f"{OUT}/closure_stage1_gate.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------- Stage 2: multiplier / theta sweeps
def stage2_sweep_one(cfg, closure, grid, fname, knob_label):
    t0, _, s = ratio_series(cfg, "none", None, "cb")
    bt = s["blowout"][0] if s["blowout"] is not None else None
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, len(grid)))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axhspan(0.0, EPS, color="tab:red", alpha=0.12, zorder=0)
    ax.axhline(EPS, color="tab:red", lw=0.9, ls="--", zorder=1)
    for col, v in zip(cmap, grid):
        t, r, _ = ratio_series(cfg, closure, v, "cb")
        ax.plot(t, r, "-", color=col, lw=1.7, zorder=3, label=f"{knob_label}={v:g}")
    if bt is not None:
        ax.axvline(bt, color="k", lw=1.0, ls=":", zorder=2)
        ax.text(bt, 0.93, " blowout", rotation=90, va="top", ha="left", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, 1.02)
    ax.set_xlabel("t  [Myr]  (log)")
    ax.set_ylabel("trigger ratio  r  (cb, note)")
    ax.set_title(f"Stage 2 sweep: {cfg}, {closure} -- curves descend into the band as {knob_label} "
                 f"rises\n[FROZEN-TRAJECTORY SCREEN]")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    p = f"{OUT}/{fname}"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Stage 2: heatmaps
def stage2_heatmap(df, closure, grid, fname, knob_label):
    cfgs = [c for c in NORMALS if c in ct.SERIES] + ["fail_repro"]
    M = np.full((len(cfgs), len(grid)), np.nan)
    never = np.zeros_like(M, dtype=bool)
    for i, cfg in enumerate(cfgs):
        for j, v in enumerate(grid):
            row = df[(df.config == cfg) & (df.closure == closure) & (df.value == v)
                     & (df.trigger_form == "cb")]
            if row.empty:
                never[i, j] = True
                continue
            r = row.iloc[0]
            if not r.fires or pd.isna(r.fire_minus_blowout):
                never[i, j] = True
            else:
                M[i, j] = r.fire_minus_blowout
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.3 * len(grid) + 2.5, 0.6 * len(cfgs) + 2.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # hatch the never-fires / no-blowout cells
    for i in range(len(cfgs)):
        for j in range(len(grid)):
            if never[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                           facecolor="0.85", edgecolor="0.6", hatch="///",
                                           lw=0.0, zorder=2))
                ax.text(j, i, "x", ha="center", va="center", fontsize=9, color="0.4", zorder=3)
            else:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="k", zorder=3)
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([f"{v:g}" for v in grid])
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(cfgs)
    ax.set_xlabel(f"{knob_label}")
    ax.set_title(f"Stage 2 grid: fire_minus_blowout [Myr] (cb, {closure})\n"
                 "blue = fires before blowout, red = after, hatched = never/no-blowout  "
                 "[FROZEN SCREEN]")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("t_fire - t_blowout  [Myr]")
    fig.tight_layout()
    p = f"{OUT}/{fname}"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Stage 3: double-count diagram
def stage3_doublecount(df):
    th = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(th, th, "-", color="tab:blue", lw=2, label="single count: y = theta  (Lcool/Lmech sits here)")
    ax.plot(th, 2 * th, "-", color="tab:red", lw=2, label="double count: y = 2*theta")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axvspan(0.5, 1.0, color="tab:red", alpha=0.10)
    ax.text(0.74, 1.7, "unphysical\ndouble-count\n(2*theta > 1)", ha="center", va="center",
            fontsize=9, color="tab:red")
    # scatter each config's median Lcool/Lmech (closure none) on the y=theta line
    cfgs = [c for c in NORMALS if c in ct.SERIES] + ["fail_repro", "small_1e6"]
    for cfg in cfgs:
        fr = ct.SERIES[cfg]["frame"]
        m = float((fr["Lcool"] / fr["Lmech"]).median())
        ax.plot(m, m, "o", ms=8, mec="k", mew=0.8, zorder=5)
        ax.annotate(cfg, (m, m), textcoords="offset points", xytext=(6, -2), fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xlabel("theta = (effective loss)/Lmech")
    ax.set_ylabel("counted loss / Lmech")
    ax.set_title("Stage 3 - bookkeeping: the closures ADD only the missing loss,\n"
                 "never rescale Lmech, so they stay on the single-count line", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    p = f"{OUT}/closure_stage3_doublecount.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- Stage 4: summary bars
def needed_value(df, cfg, closure, grid, sustained_only):
    """Smallest grid value whose first (sustained) fire t <= t_blowout. NaN if none in grid."""
    sub = df[(df.config == cfg) & (df.closure == closure) & (df.trigger_form == "cb")]
    if sub.empty or sub.t_blowout.isna().all():
        return np.nan
    tb = sub.t_blowout.iloc[0]
    for v in grid:
        r = sub[sub.value == v]
        if r.empty:
            continue
        r = r.iloc[0]
        if not r.fires or pd.isna(r.t_fire):
            continue
        if sustained_only and not r.sustained:
            continue
        if r.t_fire <= tb:
            return v
    return np.nan


def stage4_summary(df):
    cfgs = [c for c in NORMALS if c in ct.SERIES]
    f_need = [needed_value(df, c, "multiplier", F_GRID, sustained_only=True) for c in cfgs]
    f_trans = [needed_value(df, c, "multiplier", F_GRID, sustained_only=False) for c in cfgs]
    th_need = [needed_value(df, c, "theta_target", THETA_GRID, sustained_only=True) for c in cfgs]
    th_trans = [needed_value(df, c, "theta_target", THETA_GRID, sustained_only=False) for c in cfgs]

    fig, (axf, axt) = plt.subplots(1, 2, figsize=(12, 5.0))
    x = np.arange(len(cfgs))

    axf.axhspan(0, 30, color="tab:green", alpha=0.08)
    axf.text(len(cfgs) - 0.5, 28, "El-Badry/Lancaster\nplausible (f up to ~10-30)",
             ha="right", va="top", fontsize=8, color="tab:green")
    axf.bar(x - 0.2, np.nan_to_num(f_trans, nan=0), 0.38, color="tab:orange", alpha=0.8,
            label="first-fire <= blowout (transient OK)")
    axf.bar(x + 0.2, np.nan_to_num(f_need, nan=0), 0.38, color="tab:blue", alpha=0.85,
            label="sustained fire <= blowout")
    for xi, (a, b) in enumerate(zip(f_trans, f_need)):
        if not np.isnan(a):
            axf.text(xi - 0.2, a + 0.4, f"{a:g}", ha="center", fontsize=8)
        if not np.isnan(b):
            axf.text(xi + 0.2, b + 0.4, f"{b:g}", ha="center", fontsize=8)
    axf.set_xticks(x)
    axf.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
    axf.set_ylabel("f_mix needed to fire at blowout")
    axf.set_title("multiplier (cb)")
    axf.legend(fontsize=8, loc="upper left")

    axt.axhspan(0, 0.95, color="tab:green", alpha=0.08)
    axt.text(len(cfgs) - 0.5, 0.93, "plausible (theta up to ~0.95)", ha="right", va="top",
             fontsize=8, color="tab:green")
    axt.bar(x - 0.2, np.nan_to_num(th_trans, nan=0), 0.38, color="tab:orange", alpha=0.8,
            label="first-fire <= blowout")
    axt.bar(x + 0.2, np.nan_to_num(th_need, nan=0), 0.38, color="tab:blue", alpha=0.85,
            label="sustained <= blowout")
    for xi, (a, b) in enumerate(zip(th_trans, th_need)):
        if not np.isnan(a):
            axt.text(xi - 0.2, a + 0.01, f"{a:g}", ha="center", fontsize=8)
        if not np.isnan(b):
            axt.text(xi + 0.2, b + 0.01, f"{b:g}", ha="center", fontsize=8)
    axt.set_xticks(x)
    axt.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
    axt.set_ylim(0, 1.05)
    axt.set_ylabel("theta_target needed to fire at blowout")
    axt.set_title("theta_target (cb)")
    axt.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage 4: knob needed to fire BY blowout (t_fire <= t_blowout; may be much "
                 "earlier - see heatmap)  [FROZEN SCREEN]", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = f"{OUT}/closure_stage4_summary.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, dict(cfgs=cfgs, f_trans=f_trans, f_need=f_need, th_trans=th_trans, th_need=th_need)


def main():
    df = ct.main()  # regenerate closure_test.csv and populate ct.SERIES
    paths = []
    paths.append(stage1_gate())
    paths.append(stage2_sweep_one("simple_cluster", "multiplier", F_GRID,
                                  "closure_stage2_sweep.png", "f"))
    paths.append(stage2_sweep_one("simple_cluster", "theta_target", THETA_GRID,
                                  "closure_stage2_theta.png", "theta"))
    paths.append(stage2_heatmap(df, "multiplier", F_GRID, "closure_stage2_heatmap.png", "f_mix"))
    paths.append(stage2_heatmap(df, "theta_target", THETA_GRID,
                                "closure_stage2_heatmap_theta.png", "theta_target"))
    paths.append(stage3_doublecount(df))
    p4, knobs = stage4_summary(df)
    paths.append(p4)
    for p in paths:
        print("wrote", p)
    # echo the stage-4 numbers for the report
    print("\nStage 4 knob table (cb):")
    for i, c in enumerate(knobs["cfgs"]):
        print(f"  {c:24s} f_trans={knobs['f_trans'][i]!s:5s} f_sust={knobs['f_need'][i]!s:5s} "
              f"th_trans={knobs['th_trans'][i]!s:5s} th_sust={knobs['th_need'][i]!s:5s}")


if __name__ == "__main__":
    main()
