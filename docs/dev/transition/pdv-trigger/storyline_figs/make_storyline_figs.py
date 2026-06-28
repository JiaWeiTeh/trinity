#!/usr/bin/env python3
"""Publication-style storyline figures for the PdV-trigger dev note.

NO simulations. Reads ONLY the already-committed CSVs under
docs/dev/transition/pdv-trigger/data/ and renders 4 PNGs into
docs/dev/transition/pdv-trigger/storyline_figs/.

Every fire time shown (figs 1 and 4) is a FROZEN-TRAJECTORY SCREEN, not a
prediction: the CSVs were produced with production cooling, so boosting the
effective loss in post does not move the actual blowout the boosted run would
have taken. Captions/titles on those figures carry "FROZEN SCREEN".

Run from the repo root:
  python docs/dev/transition/pdv-trigger/storyline_figs/make_storyline_figs.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = HERE

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# Density-ordered, densest first (matches fmix_table.csv ordering).
DENSITY_ORDER = [
    "small_dense_highsfe", "simple_cluster", "midrange_pl0",
    "be_sphere", "pl2_steep", "large_diffuse_lowsfe",
]
F_GRID = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]


def _load(name):
    return pd.read_csv(os.path.join(DATA, name))


# ----------------------------------------------------------- FIG 1: f_mix convention (headline)
def fig_fmix_convention():
    df = _load("fmix_table.csv")
    # Enforce the density order (densest left).
    df = df.set_index("config").loc[DENSITY_ORDER].reset_index()

    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))

    # Literature target band: lift Lcool/Lmech up to theta ~= 0.95 -> f_mix ~ [1.4, 3.8].
    ax.axhspan(1.4, 3.8, color="tab:green", alpha=0.10, zorder=0,
               label="literature target band (lift Lcool/Lmech up to theta~0.95)")

    ax.bar(x - w / 2, df["fmix_with_pdv"], w, color="tab:gray", alpha=0.9,
           label="PdV inside trigger - note's headline (1.1-1.5)")
    ax.bar(x + w / 2, df["fmix_no_pdv"], w, color="tab:blue", alpha=0.9,
           label="PdV outside trigger - consistent with recommended trigger (1.4-2.8)")

    for xi, (a, b) in enumerate(zip(df["fmix_with_pdv"], df["fmix_no_pdv"])):
        ax.text(xi - w / 2, a + 0.04, f"{a:g}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + w / 2, b + 0.04, f"{b:g}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=20, ha="right")
    ax.set_ylabel(r"$f_{mix}$ to fire at blowout")
    ax.text(0.0, -0.20, "<- denser            ambient density            more diffuse ->",
            transform=ax.transAxes, fontsize=9, color="0.35")
    ax.set_ylim(0, max(df["fmix_no_pdv"].max(), 3.8) * 1.12)
    ax.set_title("f_mix to fire at blowout: the screening trigger (PdV-in) understates the boost\n"
                 "vs the recommended (PdV-out) trigger   [FROZEN-TRAJECTORY SCREEN]")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_fmix_convention.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------- FIG 2: double-count diagram
def fig_doublecount():
    comb = _load("pdv_combined_trigger.csv")
    mc = _load("doublecount_mc.csv").iloc[0]

    th = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(th, th, "-", color="tab:blue", lw=2.2,
            label="single count: y = theta  (where the max() closure sits)")
    ax.plot(th, 2 * th, "-", color="tab:red", lw=2.2,
            label="double count: y = 2*theta  (input-rescale + explicit Lcool)")
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.axvspan(0.5, 1.0, color="tab:red", alpha=0.10)
    ax.text(0.75, 1.62, "unphysical\ndouble-count\n(2*theta > 1)", ha="center", va="center",
            fontsize=10, color="tab:red")

    # Scatter each config's MEDIAN Lcool/Lmech = 1 - med_cool, on the single-count line.
    # Fan label y-offsets out so the low-theta cluster doesn't overlap.
    order = [c for c in DENSITY_ORDER if c in set(comb["config"])]
    extra = [c for c in ["fail_repro", "small_1e6"] if c in set(comb["config"])]
    cfgs = order + extra
    pts = [(cfg, 1.0 - float(comb[comb["config"] == cfg].iloc[0]["med_cool"])) for cfg in cfgs]
    pts.sort(key=lambda kv: kv[1])  # ascending in theta, so labels stack cleanly
    dy = np.linspace(34, -22, len(pts))  # spread vertical label offsets
    for (cfg, m), off in zip(pts, dy):
        ax.plot(m, m, "o", ms=9, mec="k", mew=0.9, color="tab:blue", zorder=5)
        ax.annotate(cfg, (m, m), textcoords="offset points", xytext=(10, off), fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, shrinkA=0, shrinkB=4),
                    ha="left", va="center")

    ax.annotate(
        f"Monte-Carlo: {int(mc['n_enter_double_count'])} / {int(mc['n_draws']):,} draws enter\n"
        "the double-count region (single-count by construction)",
        xy=(0.30, 1.86), fontsize=9.5, color="0.15", ha="left", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.set_xlabel("theta = (effective loss) / Lmech")
    ax.set_ylabel("counted loss / Lmech")
    ax.set_title("The max() closure adds only the missing loss - never double-counts")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_doublecount.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------- FIG 3: regime split
def fig_regime_split():
    df = _load("pdv_regime_budget.csv")

    # normal configs (density-ordered) + heavy fail_repro + small_1e6 ctrl if present.
    order = [c for c in DENSITY_ORDER if c in set(df["config"])]
    if "fail_repro" in set(df["config"]):
        order.append("fail_repro")
    if "small_1e6" in set(df["config"]):
        order.append("small_1e6")
    sub = df.set_index("config").loc[order].reset_index()
    vals = sub["PdV_over_Lmech_median"].to_numpy()

    def color_for(cfg, v):
        if cfg == "fail_repro":
            return "tab:red"
        if cfg == "small_1e6":
            return "tab:orange"
        return "tab:blue"

    colors = [color_for(c, v) for c, v in zip(sub["config"], vals)]

    y = np.arange(len(sub))[::-1]  # densest at top
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, vals, color=colors, alpha=0.9)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:g}", va="center", fontsize=9)

    ax.axvline(1.0, color="k", lw=2.0)
    ax.text(1.0, len(sub) - 0.3, "  super-critical ->", color="k", fontsize=9,
            va="center", ha="left")
    ax.text(1.0, len(sub) - 0.3, "<- sub-critical  ", color="k", fontsize=9,
            va="center", ha="right")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["config"])
    ax.set_xlabel("median PdV / Lmech")
    ax.set_xlim(0, max(1.7, vals.max() * 1.15))

    legend_handles = [
        Patch(facecolor="tab:blue", alpha=0.9, label="normal clouds (~0.45, sub-critical)"),
        Patch(facecolor="tab:red", alpha=0.9, label="heavy 5e9 fail_repro (~1.4, super-critical)"),
    ]
    if "small_1e6" in set(sub["config"]):
        legend_handles.append(
            Patch(facecolor="tab:orange", alpha=0.9, label="small_1e6 ctrl (~0.55)"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_title("PdV/Lmech: normal clouds sub-critical (~0.45), heavy 5e9 super-critical (~1.4)\n"
                 "- the clean regime split")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_regime_split.png")
    fig.savefig(p)
    plt.close(fig)
    return p


# ----------------------------------------------------------- FIG 4: closure heatmap
def fig_closure_heatmap():
    df = _load("closure_test.csv")
    cfgs = [c for c in DENSITY_ORDER if c in set(df["config"])]
    if "fail_repro" in set(df["config"]):
        cfgs.append("fail_repro")

    M = np.full((len(cfgs), len(F_GRID)), np.nan)
    never = np.zeros_like(M, dtype=bool)
    for i, cfg in enumerate(cfgs):
        for j, v in enumerate(F_GRID):
            row = df[(df.config == cfg) & (df.closure == "multiplier")
                     & (df.value == v) & (df.trigger_form == "cb")]
            if row.empty:
                never[i, j] = True
                continue
            r = row.iloc[0]
            if (not bool(r.fires)) or pd.isna(r.fire_minus_blowout):
                never[i, j] = True
            else:
                M[i, j] = r.fire_minus_blowout

    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.25 * len(F_GRID) + 3.0, 0.7 * len(cfgs) + 2.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(cfgs)):
        for j in range(len(F_GRID)):
            if never[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                           facecolor="0.85", edgecolor="0.6", hatch="///",
                                           lw=0.0, zorder=2))
                ax.text(j, i, "no fire", ha="center", va="center", fontsize=8,
                        color="0.35", zorder=3)
            else:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8.5,
                        color="k", zorder=3)

    ax.set_xticks(range(len(F_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in F_GRID])
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(cfgs)
    ax.set_xlabel("f_mix  (constant multiplier on interface loss, cb trigger)")
    ax.set_title("Frozen screen: a CONSTANT f_mix can't fire every config at blowout - the spread\n"
                 "argues for a coupled theta_target(Da)   [FROZEN-TRAJECTORY SCREEN]\n"
                 "blue = fires before blowout, red = after, hatched = never fires / no blowout")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("t_fire - t_blowout  [Myr]")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_closure_heatmap.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    figs = [
        (fig_fmix_convention(),
         "Headline: PdV-in trigger (gray, 1.1-1.5) understates the cooling boost vs the "
         "recommended PdV-out trigger (blue, 1.4-2.8). [FROZEN SCREEN]"),
        (fig_doublecount(),
         "max() closure sits on the single-count line; 0/500,000 MC draws reach the "
         "2*theta>1 double-count region."),
        (fig_regime_split(),
         "PdV/Lmech regime split: normal clouds ~0.45 (sub-critical), heavy 5e9 fail_repro "
         "~1.4 (super-critical), boundary at 1.0."),
        (fig_closure_heatmap(),
         "Frozen-screen heatmap of t_fire - t_blowout over the f_mix grid; no constant f_mix "
         "fires every config at blowout. [FROZEN SCREEN]"),
    ]
    print("\n=== MANIFEST ===")
    for path, cap in figs:
        size = os.path.getsize(path)
        print(f"{os.path.basename(path)}  ({size} bytes)\n    {cap}")
    return figs


if __name__ == "__main__":
    main()
