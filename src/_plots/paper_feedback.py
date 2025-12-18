#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:41:07 2025

@author: Jia Wei Teh
"""
        
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

print("...plotting radius comparison")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
# ndens_list  = ["1e4"]                        # one figure per ndens
ndens_list  = ["1e3"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 21

PHASE_CHANGE = True

FORCE_FIELDS = [
    ("F_grav",     "Gravity",              "black"),
    # ("F_ram_wind", "Ram (wind)",           "b"),
    # ("F_ram_SN",   "Ram (SN)",             "#2ca02c"),
    ("F_ram",   "Ram",             "b"),
    ("F_ion_out",  "Photoionised gas",     "#d62728"),
    ("F_rad",      "Radiation (dir.+indir.)", "#9467bd"),
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


set_plot_style(use_tex=True, font_size=12)


def load_run(json_path: Path):
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    r = np.array([s["R2"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    forces = np.vstack([
        np.array([s[field] for s in snaps], dtype=float)
        for field, _, _ in FORCE_FIELDS
    ])

    rcloud = float(snaps[0].get("rCloud", np.nan))
    return t, r, phase, forces, rcloud

def smooth_1d(y, window, mode="edge"):
    """
    Simple boxcar moving-average smoothing.
    window = number of points (will be made odd).
    """
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
    """Smooth along the last axis for a 2D array shaped (n_series, n_time)."""
    if window is None or window <= 1:
        return arr
    return np.vstack([smooth_1d(row, window, mode=mode) for row in arr])

def plot_run_on_ax(ax, t, r, phase, forces, rcloud, alpha=0.75,
                   smooth_window=None, smooth_mode="edge", phase_change=PHASE_CHANGE):

    # --- phase lines with mini labels
    if phase_change:
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
                bbox=dict(facecolor="k", edgecolor="none", alpha=0.7, pad=0.2),
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
                bbox=dict(facecolor="k", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=5
            )



    # --- first time r exceeds rCloud (behind)
    if np.isfinite(rcloud):
        idx = np.flatnonzero(r > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)
    
            # after drawing the axvline at x_rc ...
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                4/72, 0, fig.dpi_scale_trans  # 4 points to the right (increase to pad more)
            )

            # label next to the line
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8,
                color="k",
                alpha=0.8,
                rotation=90
            )

    # --- SMOOTHING (apply to forces, then recompute fractions)
    forces_use = smooth_2d(forces, smooth_window, mode=smooth_mode)

    # stacked force fractions
    ftotal = forces_use.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)

    frac = forces_use / ftotal
    cum  = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    for (_, _, color), y0, y1 in zip(FORCE_FIELDS, prev, cum):
        ax.fill_between(t, y0, y1, color=color, alpha=alpha, lw=0, zorder=2)

    ax.set_ylim(0, 1)
    ax.set_xlim(t.min(), t.max())


# --- one 3x5 figure per ndens
for ndens in ndens_list:
    nrows, ncols = len(mCloud_list), len(sfe_list)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=True,
        dpi=500,
        constrained_layout=True
    )

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            try:
                ax = axes[i, j]
                run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                json_path = BASE_DIR / run_name / "dictionary.json"
    
                if not json_path.exists():
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
    
                t, r, phase, forces, rcloud = load_run(json_path)
                plot_run_on_ax(ax, t, r, phase, forces, rcloud, alpha=0.75, smooth_window=SMOOTH_WINDOW, phase_change=PHASE_CHANGE)
                
            except Exception as e:
                print(f"Error in {run_name}: {e}")
                
            # column titles (top row): SFE
            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            # row labels (left col): Mcloud
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"$F/F_{tot}$")

            # x-labels (bottom row)
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    # one global legend (cleaner than repeating per panel)
    handles = [Patch(facecolor=c, edgecolor="none", label=lab, alpha=0.75)
               for _, lab, c in FORCE_FIELDS]
    leg = fig.legend(
        handles=handles, loc="upper center", ncol=len(handles),
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.05)
    )
    leg.set_zorder(10)

    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Feedback force fractions grid  ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.08)

    plt.show()
    plt.close(fig)

        
        
    #%%
    
# version 2: with SN feedback.
    
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch

print("...plotting radius comparison")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e4"]                        # one figure per ndens
# ndens_list  = ["1e2"]                        # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

# smoothing: number of snapshots in moving average (None or 1 disables)
SMOOTH_WINDOW = 21

PHASE_CHANGE = False

FORCE_FIELDS = [
    ("F_grav",     "Gravity",                    "black"),

    # Ram: use combined in energy/implicit, split in transition/momentum
    ("F_ram",      "Ram (energy/implicit)",      "b"),
    ("F_ram_wind", "Ram (wind; trans/mom)",      "b"),
    ("F_ram_SN",   "Ram (SN; trans/mom)",        "#2ca02c"),

    ("F_ion_out",  "Photoionised gas",           "#d62728"),
    ("F_rad",      "Radiation (dir.+indir.)",    "#9467bd"),
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


set_plot_style(use_tex=True, font_size=12)


def load_run(json_path: Path):
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    r = np.array([s["R2"] for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    # pull force arrays, with fallbacks
    def get_field(field, default=0.0):
        return np.array([s.get(field, default) for s in snaps], dtype=float)

    F_grav     = get_field("F_grav")
    F_ion_out  = get_field("F_ion_out")
    F_rad      = get_field("F_rad")

    F_ram_wind = get_field("F_ram_wind")
    F_ram_SN   = get_field("F_ram_SN")
    # if F_ram not present, build it from wind+SN
    F_ram      = get_field("F_ram", default=np.nan)
    if np.all(np.isnan(F_ram)):
        F_ram = F_ram_wind + F_ram_SN
    else:
        # fill any missing elements with wind+SN
        m = np.isnan(F_ram)
        F_ram[m] = (F_ram_wind + F_ram_SN)[m]

    forces = np.vstack([F_grav, F_ram, F_ram_wind, F_ram_SN, F_ion_out, F_rad])

    rcloud = float(snaps[0].get("rCloud", np.nan))
    return t, r, phase, forces, rcloud




def smooth_1d(y, window, mode="edge"):
    """
    Simple boxcar moving-average smoothing.
    window = number of points (will be made odd).
    """
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
    """Smooth along the last axis for a 2D array shaped (n_series, n_time)."""
    if window is None or window <= 1:
        return arr
    return np.vstack([smooth_1d(row, window, mode=mode) for row in arr])

def plot_run_on_ax(ax, t, r, phase, forces, rcloud, alpha=0.75,
                   smooth_window=None, smooth_mode="edge", phase_change=False):
    # vertical phase-change lines (behind)
    if phase_change:
        # only mark transition -> momentum
        change_idx = np.flatnonzero((phase[:-1] == "transition") & (phase[1:] == "momentum")) + 1
        for x in t[change_idx]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            
    # --- first time r exceeds rCloud (behind)
    if np.isfinite(rcloud):
        idx = np.flatnonzero(r > rcloud)
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)
    
            # after drawing the axvline at x_rc ...
            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                4/72, 0, fig.dpi_scale_trans  # 4 points to the right (increase to pad more)
            )

            # label next to the line
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8,
                color="k",
                alpha=0.8,
                rotation=90
            )

    # --- SMOOTHING (apply to forces, then recompute fractions)
    forces_use = smooth_2d(forces, smooth_window, mode=smooth_mode)

    # phase masks
    use_combined = np.isin(phase, ["energy", "implicit", "transition"])
    use_split    = np.isin(phase, ["momentum"])

    # indices in forces_use (must match stacking order from load_run)
    i_grav, i_ram, i_wind, i_sn, i_ion, i_rad = range(forces_use.shape[0])

    # zero-out inactive ram terms to avoid double counting
    forces_use[i_ram,  ~use_combined] = 0.0
    forces_use[i_wind, ~use_split]    = 0.0
    forces_use[i_sn,   ~use_split]    = 0.0

    # (optional) if you have phases outside these sets, default to combined:
    other = ~(use_combined | use_split)
    forces_use[i_ram,  other] = forces_use[i_ram, other]  # keep combined
    forces_use[i_wind, other] = 0.0
    forces_use[i_sn,   other] = 0.0

    # stacked force fractions
    ftotal = forces_use.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)

    frac = forces_use / ftotal
    cum  = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    for (_, _, color), y0, y1 in zip(FORCE_FIELDS, prev, cum):
        ax.fill_between(t, y0, y1, color=color, alpha=alpha, lw=0, zorder=2)

    ax.set_ylim(0, 1)
    ax.set_xlim(t.min(), t.max())


# --- one 3x5 figure per ndens
for ndens in ndens_list:
    nrows, ncols = len(mCloud_list), len(sfe_list)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=True,
        dpi=500,
        constrained_layout=True
    )

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            try:
                ax = axes[i, j]
                run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
                json_path = BASE_DIR / run_name / "dictionary.json"
    
                if not json_path.exists():
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
    
                t, r, phase, forces, rcloud = load_run(json_path)
                plot_run_on_ax(ax, t, r, phase, forces, rcloud, alpha=0.75, smooth_window=SMOOTH_WINDOW)
                
            except Exception as e:
                print(f"Error in {run_name}: {e}")
                
            # column titles (top row): SFE
            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            # row labels (left col): Mcloud
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"$F/F_{tot}$")

            # x-labels (bottom row)
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    # one global legend (cleaner than repeating per panel)
    handles = [Patch(facecolor=c, edgecolor="none", label=lab, alpha=0.75)
               for _, lab, c in FORCE_FIELDS]
    leg = fig.legend(
        handles=handles, loc="upper center", ncol=len(handles),
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.05)
    )
    leg.set_zorder(10)

    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Feedback force fractions grid  ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.08)

    plt.show()
    plt.close(fig)

        
        
        
        
        
        
        
        