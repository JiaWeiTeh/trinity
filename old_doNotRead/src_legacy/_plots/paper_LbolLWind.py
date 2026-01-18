#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

print("...plotting Qi(or Li) vs LWind with ratio on twin axis")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e2", "1e3", "1e4"]                 # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

PHASE_LINE    = True
CLOUD_LINE    = True
SMOOTH_WINDOW = None      # e.g. 7 or 21; None/1 disables
SMOOTH_MODE   = "edge"

# ---- WHAT TO PLOT ----
PLOT_QI = False            # True: plot Qi vs LWind; False: plot Li vs LWind
QI_PREFER_JSON = True     # if True and JSON has "Qi", use it; else estimate from Li

# If estimating Qi from Li [erg/s], choose mean ionizing photon energy:
MEAN_ION_PHOTON_ENERGY_EV = 20.0   # typical assumption (SED-dependent)

# Scales
LOG_LEFT_AXIS  = True     # log y for Li/LWind or Qi/LWind (if positive)
LOG_RATIO_AXIS = False    # usually keep linear

# colors
C_LWIND = "#1f77b4"   # blue
C_LI    = "#d62728"   # red
C_QI    = "#d62728"   # red (same role)
C_RATIO = "0.2"       # dark gray


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


def ev_to_erg(ev):
    return ev * 1.602176634e-12


def load_run(json_path: Path):
    """Load t, phase, Li, LWind, (optional) Qi, and R2/rCloud for breakout line."""
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    Li    = np.array([s.get("Li", np.nan)    for s in snaps], dtype=float)
    LWind = np.array([s.get("LWind", np.nan) for s in snaps], dtype=float)

    # Qi may or may not exist
    Qi = np.array([s.get("Qi", np.nan) for s in snaps], dtype=float)

    # for breakout marker
    R2 = np.array([s.get("R2", np.nan) for s in snaps], dtype=float)
    rcloud = float(snaps[0].get("rCloud", np.nan))

    # ensure increasing time
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        Li, LWind, Qi, R2 = Li[order], LWind[order], Qi[order], R2[order]

    return t, phase, Li, LWind, Qi, R2, rcloud


def add_phase_and_cloud_markers(ax, t, phase, R2, rcloud, label_pad_points=4):
    """Phase T/M markers + breakout line."""
    fig = ax.figure

    if PHASE_LINE:
        # energy/implicit -> transition (T)
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
                zorder=6
            )

        # transition -> momentum (M)
        idx_M = np.flatnonzero((phase[:-1] == "transition") & (phase[1:] == "momentum")) + 1
        for x in t[idx_M]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=6
            )

    if CLOUD_LINE and np.isfinite(rcloud):
        idx = np.flatnonzero(np.isfinite(R2) & (R2 > rcloud))
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.25, zorder=0)

            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                label_pad_points/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.05, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="bottom",
                fontsize=8, color="k", alpha=0.8,
                rotation=90, zorder=6
            )


def plot_panel(ax, t, phase, Li, LWind, Qi, R2, rcloud):
    # smoothing
    Li_s    = smooth_1d(Li,    SMOOTH_WINDOW, mode=SMOOTH_MODE)
    LWind_s = smooth_1d(LWind, SMOOTH_WINDOW, mode=SMOOTH_MODE)
    Qi_s    = smooth_1d(Qi,    SMOOTH_WINDOW, mode=SMOOTH_MODE)
    R2_s    = smooth_1d(R2,    SMOOTH_WINDOW, mode=SMOOTH_MODE)

    add_phase_and_cloud_markers(ax, t, phase, R2_s, rcloud)

    # decide y-quantity: Li or Qi
    if PLOT_QI:
        # Prefer Qi from JSON, otherwise estimate from Li
        use_Q = Qi_s.copy()
        if (not QI_PREFER_JSON) or np.all(~np.isfinite(use_Q)):
            # estimate Qi = Li / <E_photon>
            Emean = ev_to_erg(MEAN_ION_PHOTON_ENERGY_EV)
            with np.errstate(divide="ignore", invalid="ignore"):
                use_Q = Li_s / Emean

        y_main = use_Q
        y_label = r"$Q_i\ [{\rm s^{-1}}]$"
        main_color = C_QI
        ratio_label = r"$\mathcal{L}=Q_i/L_{\rm Wind}$"
    else:
        y_main = Li_s
        y_label = r"$L_i\ [{\rm erg\ s^{-1}}]$"
        main_color = C_LI
        ratio_label = r"$\mathcal{L}=L_i/L_{\rm Wind}$"

    # left axis lines
    ax.plot(t, y_main,  lw=1.8, color=main_color, label=y_label, zorder=3)
    ax.plot(t, LWind_s, lw=1.8, color=C_LWIND,   label=r"$L_{\rm Wind}$", zorder=3)

    ax.set_xlim(t.min(), t.max())

    if LOG_LEFT_AXIS:
        y_all = np.concatenate([y_main[np.isfinite(y_main)], LWind_s[np.isfinite(LWind_s)]])
        if y_all.size and np.nanmin(y_all) > 0:
            ax.set_yscale("log")

    # right axis: ratio
    axr = ax.twinx()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y_main / LWind_s
    ratio = np.where(np.isfinite(ratio), ratio, np.nan)

    axr.plot(t, ratio, lw=1.5, color=C_RATIO, ls="--", alpha=0.9,
             label=ratio_label, zorder=2)

    if LOG_RATIO_AXIS:
        rr = ratio[np.isfinite(ratio)]
        if rr.size and np.nanmin(rr) > 0:
            axr.set_yscale("log")

    # keep twin axis readable
    axr.patch.set_visible(False)
    return axr


# ---------------- main plotting ----------------
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

    fig.subplots_adjust(top=0.90)
    nlog = int(np.log10(float(ndens)))
    if PLOT_QI:
        fig.suptitle(
            rf"$Q_i$ vs $L_{{\rm Wind}}$ (ratio on right), $n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$",
            y=1.05
        )
    else:
        fig.suptitle(
            rf"$L_i$ vs $L_{{\rm Wind}}$ (ratio on right), $n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$",
            y=1.05
        )

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
                t, phase, Li, LWind, Qi, R2, rcloud = load_run(json_path)
                axr = plot_panel(ax, t, phase, Li, LWind, Qi, R2, rcloud)
            except Exception as e:
                print(f"Error in {run_name}: {e}")
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            # column titles (top row)
            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            # left y label only on left-most
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"left: main + $L_{\rm Wind}$")
            else:
                ax.tick_params(labelleft=False)

            # right y label only on right-most
            if j == ncols - 1:
                if PLOT_QI:
                    axr.set_ylabel(r"$\mathcal{L}=Q_i/L_{\rm Wind}$")
                else:
                    axr.set_ylabel(r"$\mathcal{L}=L_i/L_{\rm Wind}$")
            else:
                axr.tick_params(labelright=False)
                axr.set_ylabel("")

            # x label only on bottom row
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    # -------- global legend --------
    if PLOT_QI:
        main_handle = Line2D([0], [0], color=C_QI, lw=1.8, label=r"$Q_i$ (or estimated from $L_i/\langle h\nu\rangle$)")
        ratio_handle = Line2D([0], [0], color=C_RATIO, lw=1.5, ls="--", label=r"$\mathcal{L}=Q_i/L_{\rm Wind}$")
    else:
        main_handle = Line2D([0], [0], color=C_LI, lw=1.8, label=r"$L_i$")
        ratio_handle = Line2D([0], [0], color=C_RATIO, lw=1.5, ls="--", label=r"$\mathcal{L}=L_i/L_{\rm Wind}$")

    handles = [
        main_handle,
        Line2D([0], [0], color=C_LWIND, lw=1.8, label=r"$L_{\rm Wind}$"),
        ratio_handle,
        Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0], [0], color="r", lw=2, alpha=0.3, label=r"phase: $T$ (→transition), $M$ (→momentum)"),
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
    
    # --------- SAVE FIGURE ---------
    m_tag   = range_tag("M",   mCloud_list, key=float)
    sfe_tag = range_tag("sfe", sfe_list,    key=int)
    n_tag   = f"n{ndens}"
    tag = f"LbolLWind_{m_tag}_{sfe_tag}_{n_tag}"

    if SAVE_PNG:
        out_png = FIG_DIR / f"{tag}.png"
        fig.savefig(out_png, bbox_inches="tight")
        print(f"Saved: {out_png}")
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Saved: {out_pdf}")



    plt.show()
    plt.close(fig)
