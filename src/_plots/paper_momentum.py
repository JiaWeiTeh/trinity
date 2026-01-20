#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:26:34 2025

@author: Jia Wei Teh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from load_snapshots import load_snapshots, find_data_file

print("...plotting integrated momentum (line plots)")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e4", "1e2", "1e3"]                               # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True

PHASE_CHANGE = True

SMOOTH_WINDOW = None

DOMINANCE_DT = 0.05          # Myr
DOMINANCE_ALPHA = 0.9
DOMINANCE_STRIP = (0.97, 1)  # (ymin, ymax) in AXES fraction (0..1)


FORCE_FIELDS = [
    ("F_grav",     "Gravity",              "black"),
    # ("F_ram_wind", "Ram (wind)",           "b"),
    # ("F_ram_SN",   "Ram (SN)",             "#2ca02c"),
    ("F_ram",   "Ram",             "b"),
    ("F_ion_out",  "Photoionised gas",     "#d62728"),
    ("F_rad",      "Radiation (dir.+indir.)", "#9467bd"),
]


import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

# --- optional single-run view (set all to None for full grid)
ONLY_MCLOUD = "1e7"   # e.g. "1e8"
ONLY_NDENS  = "1e4"   # e.g. "1e4"
ONLY_SFE    = "001"   # e.g. "030"

# comment this out if want single mode, otherwise leave this be if want grid. 
# ONLY_NDENS = ONLY_MCLOUD = ONLY_SFE = None

# -------- smoothing (optional) --------
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


def smooth_2d(arr, window, mode="edge"):
    if window is None or window <= 1:
        return arr
    return np.vstack([smooth_1d(row, window, mode=mode) for row in arr])


# -------- integration --------
def cumtrapz_2d(Y, x):
    """
    Cumulative trapezoid integral with p[0]=0.
    Y shape: (n_series, n_time)
    """
    Y = np.asarray(Y, dtype=float)
    x = np.asarray(x, dtype=float)

    dx = np.diff(x)  # (n_time-1,)
    incr = 0.5 * (Y[:, 1:] + Y[:, :-1]) * dx  # broadcast dx across rows
    out = np.zeros_like(Y, dtype=float)
    out[:, 1:] = np.cumsum(incr, axis=1)
    return out


def load_run(data_path: Path):
    """Load run data. Supports both JSON and JSONL formats."""
    snaps = load_snapshots(data_path)

    if not snaps:
        raise ValueError(f"No snapshots found in {data_path}")

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    r = np.array([s["R2"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    forces = np.vstack([
        np.array([s.get(field, 0.0) for s in snaps], dtype=float)
        for field, _, _ in FORCE_FIELDS
    ])

    # Ensure time is increasing for integration
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        r = r[order]
        phase = phase[order]
        forces = forces[:, order]

    rcloud = float(snaps[0].get("rCloud", np.nan))
    return t, r, phase, forces, rcloud

# This added functionality solves the problem in which white spaces occur
# when calculating dominanting forces - for small binning values some snapshots
# simply does not exist. This interpolates the value and makes sure that 
# every bin has their own value and colour. 
def _interp_finite(x, y, xnew):
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(xnew, np.nan, dtype=float)
    return np.interp(xnew, x[m], y[m])

def dominant_bins(t, frac, dt=0.05):
    t = np.asarray(t, float)
    frac = np.asarray(frac, float)

    edges = np.arange(t.min(), t.max() + dt, dt)
    centers = 0.5 * (edges[:-1] + edges[1:])  # one value per bin

    frac_c = np.vstack([_interp_finite(t, frac_i, centers) for frac_i in frac])

    # optional: renormalize in case interpolation + NaNs break sum=1
    denom = np.nansum(frac_c, axis=0)
    denom = np.where(denom == 0.0, np.nan, denom)
    frac_c = frac_c / denom

    winner = np.nanargmax(frac_c, axis=0)  # now every bin has a winner (unless all NaN)
    return edges, winner


#--- plots

def plot_single_run(mCloud, ndens, sfe):
    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
    data_path = find_data_file(BASE_DIR, run_name)

    if data_path is None:
        print(f"Missing data for: {run_name}")
        return

    t, r, phase, forces, rcloud = load_run(data_path)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=500, constrained_layout=True)
    plot_momentum_lines_on_ax(
        ax, t, r, phase, forces, rcloud,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE
    )

    ax.set_title(run_name)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$p(t)=\int F\,dt$")
    if SAVE_PDF:
        fig.savefig(FIG_DIR / f"paper_momentum_{run_name}.pdf", bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_momentum_lines_on_ax(
    ax, t, r, phase, forces, rcloud,
    smooth_window=None, smooth_mode="edge",
    lw=1.6, net_lw=4, alpha=0.8, phase_change=PHASE_CHANGE,
):
    # --- phase-change line:
    # --- phase lines with mini labels
    if phase_change:
        # energy/implicit -> transition  (T)
        idx_T = np.flatnonzero(
            np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")
        ) + 1
        for x in t[idx_T]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.95, "T",
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
                x, 0.95, "M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="r", alpha=0.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )

    # --- first time r exceeds rCloud (behind)
    if np.isfinite(rcloud):
        idx = np.flatnonzero(r > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)

            fig = ax.figure  # <-- use the figure owning this axis
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                4/72, 0, fig.dpi_scale_trans
            )

            ax.text(
                x_rc, 0.05, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="bottom",
                fontsize=8,
                color="k",
                alpha=0.8,
                rotation=90
            )


    # --- optional smoothing before integrating
    F = smooth_2d(forces, smooth_window, mode=smooth_mode)

    # === Dominant force every DOMINANCE_DT Myr (based on mean fractional |F|)
    Fabs = np.abs(F)
    denom = Fabs.sum(axis=0)
    denom = np.where(denom == 0.0, np.nan, denom)
    frac = Fabs / denom  # (n_forces, N)

    edges, win = dominant_bins(t, frac, dt=DOMINANCE_DT)
    colors = [c for _, _, c in FORCE_FIELDS]
    y0, y1 = DOMINANCE_STRIP

    for b, k in enumerate(win):
        if k < 0:
            continue
        ax.axvspan(
            edges[b], edges[b + 1],
            ymin=y0, ymax=y1,          # <-- axes-fraction band
            color=colors[k],
            alpha=DOMINANCE_ALPHA,
            lw=0,
            zorder=10
        )

    # --- integrate each force: p_i(t) = ∫ F_i dt  (signed)
    P = cumtrapz_2d(F, t)  # shape (n_forces, n_time)

    def plot_abs_with_sign_linestyle(ax, x, y, *, color, label=None, lw=1.6, alpha=0.95, zorder=3):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    
        yabs = np.abs(y)
        neg = y < 0
    
        xs = [x[0]]
        ys = [yabs[0]]
        current_neg = neg[0]
        first_segment = True
    
        for i in range(len(x) - 1):
            same_sign_next = (neg[i + 1] == current_neg)
            if same_sign_next:
                xs.append(x[i + 1])
                ys.append(yabs[i + 1])
                continue
    
            # sign changes between i and i+1: insert crossing at y=0 if it truly crosses
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]
    
            if y0 * y1 < 0:  # true crossing
                x_cross = x0 + (-y0) * (x1 - x0) / (y1 - y0)
                xs.append(x_cross)
                ys.append(0.0)
                next_start_x, next_start_y = x_cross, 0.0
            else:
                # one of them is exactly 0, no need to interpolate
                next_start_x, next_start_y = x[i + 1], yabs[i + 1]
    
            ls = "--" if current_neg else "-"
            ax.plot(
                xs, ys,
                color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder,
                label=(label if (label is not None and first_segment) else "_nolegend_"),
            )
            first_segment = False
    
            # start new segment
            xs = [next_start_x, x[i + 1]]
            ys = [next_start_y, yabs[i + 1]]
            current_neg = neg[i + 1]
    
        # plot final segment
        ls = "--" if current_neg else "-"
        ax.plot(
            xs, ys,
            color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder,
            label=(label if (label is not None and first_segment) else "_nolegend_"),
        )


    # --- plot components (gravity included) using your FORCE_FIELDS colors
    # P is signed momentum array from cumtrapz_2d(forces, t) with shape (n_forces, n_time)
    for (field, label, color), Pi in zip(FORCE_FIELDS, P):
        plot_abs_with_sign_linestyle(ax, t, Pi, color=color, label=label, lw=lw, alpha=alpha, zorder=3)
    
    # net momentum (signed): integrate F_net = sum(outward) - gravity
    F_net = F[1:].sum(axis=0) - F[0]
    P_net = cumtrapz_2d(F_net[None, :], t)[0]
    plot_abs_with_sign_linestyle(ax, t, P_net, color="darkgrey", label="Net", lw=net_lw, alpha=0.8, zorder=4)


    ax.set_xlim(0, t.max())
    ax.set_yscale('log')
    ax.set_ylim(1e-5*P.max(), 10*P.max())

# --------- MODE SWITCH: single plot or grid ----------
single_mode = (ONLY_MCLOUD is not None) and (ONLY_NDENS is not None) and (ONLY_SFE is not None)

if single_mode:
    plot_single_run(ONLY_MCLOUD, ONLY_NDENS, ONLY_SFE)

else:
    # --- one figure per ndens (grid mode)
    for ndens in ndens_list:
        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=False,
            dpi=500,
            constrained_layout=False
        )

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
                    t, r, phase, forces, rcloud = load_run(data_path)
                    plot_momentum_lines_on_ax(
                        ax, t, r, phase, forces, rcloud,
                        smooth_window=SMOOTH_WINDOW,
                        phase_change=PHASE_CHANGE
                    )
                except Exception as e:
                    print(f"Error in {run_name}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                if j == 0:
                    mlog = int(np.log10(float(mCloud)))
                    ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"$p(t)=\int F\,dt$")

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        # global legend
        handles = []
        handles.append(Line2D([0], [0], color="black", lw=1.6, ls="-", label="Gravity"))
        for _, lab, c in FORCE_FIELDS[1:]:
            handles.append(Line2D([0], [0], color=c, lw=1.6, label=lab))
        handles.append(Line2D([0], [0], color="darkgrey", lw=2.4,
                              label=r"Net: $| \int (\sum F_{\rm out} - F_{\rm grav})\,dt |$"))

        leg = fig.legend(
            handles=handles, loc="upper center", ncol=3,
            frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
            bbox_to_anchor=(0.5, 1.05)
        )
        leg.set_zorder(10)

        fig.subplots_adjust(top=0.91)
        nlog = int(np.log10(float(ndens)))
        fig.suptitle(rf"Momentum injected ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.08)

        if SAVE_PDF:
            fig.savefig(FIG_DIR / f"paper_momentum_n{ndens}.pdf", bbox_inches='tight')
        plt.show()
        plt.close(fig)


