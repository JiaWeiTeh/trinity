#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

print("...plotting radius evolution grid")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e4", "1e2"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

PHASE_LINE = True          # draw transition->momentum line
CLOUD_LINE = True          # draw breakout line (R2>Rcloud)
SMOOTH_WINDOW = None       # set e.g. 7 to smooth radii; None/1 disables
SMOOTH_MODE = "edge"

# radius line styles/colors
RADIUS_FIELDS = [
    ("R1",     r"$R_1$",     "#9467bd"),   # purple
    ("R2",     r"$R_2$",     "k"),         # black
    ("rShell", r"$r_{\rm shell}$", "#ff7f0e"),  # orange
]


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


def load_run_radii(json_path: Path):
    """Load one run and return arrays sorted by snapshot index."""
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    # radii (use NaN if missing)
    R1 = np.array([s.get("R1", np.nan) for s in snaps], dtype=float)
    R2 = np.array([s.get("R2", np.nan) for s in snaps], dtype=float)
    rShell = np.array([s.get("rShell", np.nan) for s in snaps], dtype=float)

    rcloud = float(snaps[0].get("rCloud", np.nan))

    # ensure increasing time
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        R1, R2, rShell = R1[order], R2[order], rShell[order]

    return t, phase, R1, R2, rShell, rcloud


def plot_radii_on_ax(ax, t, phase, R1, R2, rShell, rcloud,
                     phase_line=True, cloud_line=True,
                     smooth_window=None, smooth_mode="edge",
                     m_label="M", m_alpha=0.6, label_pad_points=4):
    fig = ax.figure

    # optional smoothing
    R1s = smooth_1d(R1, smooth_window, mode=smooth_mode)
    R2s = smooth_1d(R2, smooth_window, mode=smooth_mode)
    rSs = smooth_1d(rShell, smooth_window, mode=smooth_mode)

    # --- phase lines with mini labels
    if phase_line:
        # energy/implicit -> transition  (T)
        idx_T = np.flatnonzero(
            np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")
        ) + 1
        for x in t[idx_T]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "T",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

        # transition -> momentum  (M)
        idx_M = np.flatnonzero((phase[:-1] == "transition") & (phase[1:] == "momentum")) + 1
        for x in t[idx_M]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

    # --- breakout line: first time R2 > rcloud
    if cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(np.isfinite(R2s) & (R2s > rcloud))
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.25, zorder=0)

            # padded label next to the line
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.05, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="bottom",
                fontsize=8, color="k", alpha=0.8,
                rotation=90, zorder=5
            )

    # --- radii lines
    ax.plot(t, R1s, lw=1.6, color=RADIUS_FIELDS[0][2], label=RADIUS_FIELDS[0][1], zorder=3)
    ax.plot(t, R2s, lw=2.0, color=RADIUS_FIELDS[1][2], label=RADIUS_FIELDS[1][1], zorder=4)
    ax.plot(t, rSs, lw=1.6, color=RADIUS_FIELDS[2][2], label=RADIUS_FIELDS[2][1], zorder=3)

    ax.set_xlim(t.min(), t.max())


# ---------------- run plotting ----------------
set_plot_style(use_tex=True, font_size=12)

for ndens in ndens_list:
    nrows, ncols = len(mCloud_list), len(sfe_list)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=False,
        dpi=500,
        constrained_layout=False
    )

    # reserve band for legend + suptitle (prevents overlap with subplot titles)
    fig.subplots_adjust(top=0.90)
    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Radius evolution ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.05)

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
                t, phase, R1, R2, rShell, rcloud = load_run_radii(json_path)
                plot_radii_on_ax(
                    ax, t, phase, R1, R2, rShell, rcloud,
                    phase_line=PHASE_LINE,
                    cloud_line=CLOUD_LINE,
                    smooth_window=SMOOTH_WINDOW,
                    smooth_mode=SMOOTH_MODE
                )
            except Exception as e:
                print(f"Error in {run_name}: {e}")
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # column titles
            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            # row labels + y label only on leftmost column
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"Radius [pc]")
            else:
                ax.tick_params(labelleft=False)

            # x label only on bottom row
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")
            else:
                ax.tick_params(labelbottom=False)

    # global legend
    handles = [
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=1.6, label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=2.0, label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=1.6, label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0], [0], color="r", lw=2, alpha=0.3, label=r"transition$\to$momentum (M)"),
    ]

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.2",
        bbox_to_anchor=(0.5, 0.98),
        bbox_transform=fig.transFigure
    )
    leg.set_zorder(10)

    plt.show()
    plt.close(fig)
