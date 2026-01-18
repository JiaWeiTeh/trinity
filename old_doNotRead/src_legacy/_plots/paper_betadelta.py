#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 20:03:19 2025

@author: Jia Wei Teh
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pathlib import Path
from matplotlib.lines import Line2D



import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

print("...plotting radius comparison")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
# mCloud_list = ["1e5","1e8"]                 # rows
# ndens_list  = ["1e4", "1e2", "1e3"]                        # one figure per ndens
ndens_list  = ["1e4"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols
# sfe_list    = ["001", "010"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 21

PHASE_CHANGE = True

# --- output
FIG_DIR = Path("./fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"

def set_plot_style(use_tex=True, font_size=12):
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "sans-serif",
        "font.size": font_size,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
    })


set_plot_style(use_tex=True, font_size=12)


# ---------- helpers (reuse your smoothing) ----------
def smooth_1d(y, window, mode="edge"):
    if window is None or window <= 1:
        return y
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode=mode)
    return np.convolve(ypad, kernel, mode="valid")

# ---------- load beta/delta + R2 ----------
def load_cooling_run(json_path: Path):
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]
    
    additional_param = None
    
    t     = np.array([s["t_now"] for s in snaps], dtype=float)
    R2    = np.array([s["R2"] for s in snaps], dtype=float)
    # additional_param    = np.array([s["F_ram"] for s in snaps], dtype=float)
    additional_param    = np.array([s["Pb"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    beta  = np.array([s.get("cool_beta", np.nan) for s in snaps], dtype=float)
    delta = np.array([s.get("cool_delta", np.nan) for s in snaps], dtype=float)

    rcloud = float(snaps[0].get("rCloud", np.nan))

    # ensure increasing time (important for “first crossing” + plotting)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase, beta, delta, additional_param = t[order], R2[order], phase[order], beta[order], delta[order], additional_param[order]

    return t, R2, phase, beta, delta, rcloud, additional_param


# ---------- plot on an axis (left: beta/delta, right: R2) ----------
def plot_cooling_on_ax(
    ax, t, R2, phase, beta, delta, rcloud, additional_param,
    smooth_window=None, smooth_mode="edge",
    show_phase_line=True,
    show_cloud_line=True,
    label_pad_points=4
):
    fig = ax.figure

    # optional smoothing (beta/delta often jittery)
    beta_s  = smooth_1d(beta,  smooth_window, mode=smooth_mode)
    delta_s = smooth_1d(delta, smooth_window, mode=smooth_mode)
    R2_s    = smooth_1d(R2,    smooth_window, mode=smooth_mode)

    # --- phase-change line:
    if show_phase_line:
        change_idx = np.flatnonzero(phase[1:] != phase[:-1]) + 1 
        for x in t[change_idx]: 
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)

    # --- cloud breakout line: first time R2 > rcloud (vertical dashed)
    if show_cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(R2_s > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)

            # padded label next to line
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8, color="k", alpha=0.8,
                rotation=90
            )

    # --- left axis: beta + delta
    ax.plot(t, beta_s,  lw=1.6, label=r"$\beta$")
    ax.plot(t, delta_s, lw=1.6, label=r"$\delta$")

    ax.set_xlim(t.min(), t.max())

    # --- right axis: R2
    axr = ax.twinx()
    axr.plot(t, additional_param, lw=1.4, alpha=0.8, c = 'k', label=r"$R_2$")
    axr.set_yscale('log')
    # axr.set_ylabel(r"$R_2$ [pc]")

    # keep the twin axis from hiding things
    ax.patch.set_visible(False)
    ax.set_zorder(2)

    return axr  # in case you want per-panel tweaks


# ---------- GRID (same layout idea as your previous) ----------
# assumes you already have: mCloud_list, sfe_list, ndens_list, BASE_DIR
# Example:
# mCloud_list = ["1e5","1e7","1e8"]
# sfe_list = ["001","010","030","050","080"]
# ndens_list = ["1e4","1e2"]
# BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

for ndens in ndens_list:
    nrows, ncols = len(mCloud_list), len(sfe_list)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=False,
        dpi=200,
        constrained_layout=False
    )

    # leave room for suptitle + global legend
    fig.subplots_adjust(top=0.82)
    fig.suptitle(rf"Cooling parameters with $R_2$ overlay  (n={ndens})", y=0.98)

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            ax = axes[i, j]
            run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
            json_path = BASE_DIR / run_name / "dictionary.json"

            if not json_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            try:
                t, R2, phase, beta, delta, rcloud, additional_param = load_cooling_run(json_path)
                plot_cooling_on_ax(
                    ax, t, R2, phase, beta, delta, rcloud, additional_param,
                    smooth_window=7,      # set None/1 to disable
                    show_phase_line=True,
                    show_cloud_line=True,
                )
            except Exception as e:
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                print(f"Error in {run_name}: {e}")
                continue

            # column titles
            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            # row labels (leftmost)
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}M_\odot$" + "\n" + r"Cooling: $\beta,\delta$")

            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    # global legend (beta, delta, R2, plus line meanings)
    handles = [
        Line2D([0],[0], lw=1.6, label=r"$\beta$"),
        Line2D([0],[0], lw=1.6, label=r"$\delta$"),
        Line2D([0],[0], lw=1.4, alpha=0.8, label=r"$R_2$"),
        Line2D([0],[0], color="k", ls="--", alpha=0.4, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0],[0], color="r", lw=2, alpha=0.3, label=r"transition$\to$momentum"),
    ]
    leg = fig.legend(
        handles=handles, loc="upper center", ncol=3,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        bbox_to_anchor=(0.5, 0.91), bbox_transform=fig.transFigure
    )
    leg.set_zorder(10)

    plt.show()
    plt.close(fig)
