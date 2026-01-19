#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 17:45:12 2026

@author: Jia Wei Teh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from load_snapshots import load_snapshots, find_data_file

print("...plotting velocity (v2) + radii (twin axis) grid")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e4", "1e2"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

PHASE_LINE = True
CLOUD_LINE = True
SMOOTH_WINDOW = None        # e.g. 7 or 21; None/1 disables
SMOOTH_MODE = "edge"

# --- output
FIG_DIR = Path("./fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = True
SAVE_PDF = False

# --- unit conversion: pc/Myr -> km/s
PC_PER_MYR_TO_KMS = 0.9777922215250843

# right-axis radii lines
RADIUS_FIELDS = [
    ("R1",     r"$R_1$",                   "#9467bd", "-",  1.3),
    ("R2",     r"$R_2$",                   "0.25",    "-",  1.8),
    ("rShell", r"$r_{\rm shell}$",         "#ff7f0e", "-",  1.3),
    ("r_Tb",   r"$r_{T_b}=R_2\,\xi_{T_b}$","0.45",    ":",  1.5),
]

# left-axis velocity line style
V2_STYLE = dict(color="k", lw=1.8, ls="-", alpha=0.95)


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


def load_run_velocity(data_path: Path):
    """Load t, phase, v2 (pc/Myr), radii, rcloud.

    Supports both JSON and JSONL formats.
    """
    snaps = load_snapshots(data_path)

    if not snaps:
        raise ValueError(f"No snapshots found in {data_path}")

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    v2 = np.array([s.get("v2", np.nan) for s in snaps], dtype=float)          # pc/Myr (per your note)
    R1 = np.array([s.get("R1", np.nan) for s in snaps], dtype=float)
    R2 = np.array([s.get("R2", np.nan) for s in snaps], dtype=float)
    rShell = np.array([s.get("rShell", np.nan) for s in snaps], dtype=float)

    xi_Tb = np.array([s.get("bubble_xi_Tb", np.nan) for s in snaps], dtype=float)
    r_Tb = R2 * xi_Tb

    rcloud = float(snaps[0].get("rCloud", np.nan))

    # enforce increasing time (robust if there are tiny non-monotonicities)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        v2, R1, R2, rShell, r_Tb = v2[order], R1[order], R2[order], rShell[order], r_Tb[order]

    return t, phase, v2, R1, R2, rShell, r_Tb, rcloud


def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"


#--- plot

def plot_signed_logline(ax, t, v, *, color="k", lw=1.8, alpha=0.95, label_pos=None):
    """
    Plot |v| on log-y; solid where v>=0, dashed where v<0.
    Draws contiguous segments so the curve is continuous in time,
    but linestyle changes at sign flips.
    """
    t = np.asarray(t)
    v = np.asarray(v)

    # magnitude for log plotting
    mag = np.abs(v)
    
    floor = 1e-3  # km/s
    mag = np.maximum(mag, floor)


    # sign array (treat exact zeros as positive)
    sgn = np.sign(v)
    sgn[sgn == 0] = 1

    # indices where sign changes
    cuts = np.flatnonzero(sgn[1:] != sgn[:-1]) + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, len(t)]

    first_pos_labeled = False
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue

        ls = "-" if sgn[a] > 0 else "--"
        lab = None
        if (sgn[a] > 0) and (not first_pos_labeled) and (label_pos is not None):
            lab = label_pos
            first_pos_labeled = True

        ax.plot(t[a:b], mag[a:b], color=color, lw=lw, ls=ls, alpha=alpha, label=lab)




def plot_velocity_on_ax(
    ax, t, phase, v2_pcmyr, R1, R2, rShell, r_Tb, rcloud,
    smooth_window=None, smooth_mode="edge",
    phase_line=True, cloud_line=True,
    label_pad_points=4
):
    fig = ax.figure

    # smoothing
    v2s = smooth_1d(v2_pcmyr, smooth_window, mode=smooth_mode)
    R1s = smooth_1d(R1, smooth_window, mode=smooth_mode)
    R2s = smooth_1d(R2, smooth_window, mode=smooth_mode)
    rSs = smooth_1d(rShell, smooth_window, mode=smooth_mode)
    rTbs = smooth_1d(r_Tb, smooth_window, mode=smooth_mode)

    # convert velocity to km/s
    v2_kms = v2s * PC_PER_MYR_TO_KMS

    # --- phase lines with mini labels
    if phase_line:
        # energy/implicit -> transition (T)
        idx_T = np.flatnonzero(np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")) + 1
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

    # --- breakout line: first time R2 > rCloud (using R2)
    if cloud_line and np.isfinite(rcloud):
        idx = np.flatnonzero(np.isfinite(R2s) & (R2s > rcloud))
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
                fontsize=8, color="k", alpha=0.85,
                rotation=90, zorder=6
            )

    # --- left axis: velocity in log space (plot |v2|, dashed if v2<0)
    ax.set_yscale("log")
    plot_signed_logline(
        ax, t, v2_kms,
        color=V2_STYLE.get("color", "k"),
        lw=V2_STYLE.get("lw", 1.8),
        alpha=V2_STYLE.get("alpha", 0.95),
        label_pos=r"$|v_2|$ (solid if $v_2>0$)"
    )
    ax.set_ylabel(r"$|v_2|$ [km s$^{-1}$]")


    ax.set_xlim(t.min(), t.max())
    # --- right axis: radii (pc)
    axr = ax.twinx()
    axr.plot(t, R1s,   color=RADIUS_FIELDS[0][2], ls=RADIUS_FIELDS[0][3], lw=RADIUS_FIELDS[0][4], label=RADIUS_FIELDS[0][1])
    axr.plot(t, R2s,   color=RADIUS_FIELDS[1][2], ls=RADIUS_FIELDS[1][3], lw=RADIUS_FIELDS[1][4], label=RADIUS_FIELDS[1][1])
    axr.plot(t, rSs,   color=RADIUS_FIELDS[2][2], ls=RADIUS_FIELDS[2][3], lw=RADIUS_FIELDS[2][4], label=RADIUS_FIELDS[2][1])
    axr.plot(t, rTbs,  color=RADIUS_FIELDS[3][2], ls=RADIUS_FIELDS[3][3], lw=RADIUS_FIELDS[3][4], label=RADIUS_FIELDS[3][1])
    axr.set_ylabel("Radius [pc]")

    # ensure twin axis is visible (avoid patch covering)
    ax.patch.set_visible(False)
    ax.set_zorder(2)
    axr.set_zorder(1)

    return axr


# ---------------- main plotting ----------------
import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

for ndens in ndens_list:
    nrows, ncols = len(mCloud_list), len(sfe_list)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=False,
        dpi=500,
        constrained_layout=False
    )

    # reserve top band for suptitle + legend
    fig.subplots_adjust(top=0.90)
    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Velocity and radius evolution ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.05)

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            ax = axes[i, j]
            run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
            data_path = find_data_file(BASE_DIR, run_name)

            if data_path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            try:
                t, phase, v2, R1, R2, rShell, r_Tb, rcloud = load_run_velocity(data_path)
                axr = plot_velocity_on_ax(
                    ax, t, phase, v2, R1, R2, rShell, r_Tb, rcloud,
                    smooth_window=SMOOTH_WINDOW,
                    smooth_mode=SMOOTH_MODE,
                    phase_line=PHASE_LINE,
                    cloud_line=CLOUD_LINE
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

            # left y labels only on left-most column
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"$v_2$ [km s$^{-1}$]")
            else:
                ax.tick_params(labelleft=False)

            # right y labels only on right-most column
            if j != ncols - 1:
                axr.set_ylabel("")
                axr.tick_params(labelright=False)

            # x ticks on all, labels only bottom row
            ax.tick_params(axis="x", which="both", bottom=True)
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)

    # -------- global legend --------
    handles = [
        Line2D([0], [0], color="k", lw=1.8, ls="-",  label=r"$v_2>0$ (solid; plotted as $|v_2|$)"),
        Line2D([0], [0], color="k", lw=1.8, ls="--", label=r"$v_2<0$ (dashed; plotted as $|v_2|$)"),
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
        Line2D([0], [0], color="k", ls="--", alpha=0.6, lw=1.6, label=r"$R_2>R_{\rm cloud}$"),
        Line2D([0], [0], color="r", lw=2, alpha=0.3, label=r"phase changes: $T$ (→transition), $M$ (→momentum)"),
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
    tag = f"velocity_grid_{m_tag}_{sfe_tag}_{n_tag}"

    if SAVE_PNG:
        out_png = FIG_DIR / f"{tag}.png"
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved: {out_png}")
    if SAVE_PDF:
        out_pdf = FIG_DIR / f"{tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)
