#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay wind+SN contributions inside F_ram while leaving an unhatched residual:
- Base stack uses F_ram (blue) throughout all phases.
- Hatched overlays show wind and SN as fractions of F_ram:
    f_wind = F_ram_wind / F_ram
    f_SN   = F_ram_SN   / F_ram
  The unhatched remainder of the blue band is the residual: 1 - f_wind - f_SN
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

print("...plotting force fractions with ram composition overlay (residual visible)")

# --- configuration
mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
ndens_list  = ["1e2", "1e3", "1e4"]                 # one figure per ndens
# ndens_list  = ["1e4"]                 # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

SMOOTH_WINDOW = 21          # None or 1 disables
PHASE_CHANGE = True
INCLUDE_ALL_FORCE = True    # show wind/SN overlays inside the ram band

# Colors
C_GRAV = "black"
C_RAM  = "b"
C_SN   = "#2ca02c"
C_ION  = "#d62728"
C_RAD  = "#9467bd"


# Base stacked forces (always)
FORCE_FIELDS_BASE = [
    ("F_grav",    "Gravity",                 C_GRAV),
    ("F_ram",     "Ram (total)",             C_RAM),
    ("F_ion_out", "Photoionised gas",        C_ION),
    ("F_rad",     "Radiation (dir.+indir.)", C_RAD),
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


def load_run(json_path: Path):
    with json_path.open("r") as f:
        data = json.load(f)

    snap_keys = sorted((k for k in data.keys() if str(k).isdigit()), key=lambda k: int(k))
    snaps = [data[k] for k in snap_keys]

    t = np.array([s["t_now"] for s in snaps], dtype=float)
    R2 = np.array([s.get("R2", np.nan) for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    def get_field(field, default=np.nan):
        return np.array([s.get(field, default) for s in snaps], dtype=float)

    F_grav = get_field("F_grav", 0.0)
    F_ion  = get_field("F_ion_out", 0.0)
    F_rad  = get_field("F_rad", 0.0)

    # total ram (energy-balance effective)
    F_ram = get_field("F_ram", np.nan)

    # decomposition (SPS output)
    F_wind = get_field("F_ram_wind", np.nan)
    F_sn   = get_field("F_ram_SN", np.nan)

    # If F_ram missing entirely, reconstruct if possible
    if np.all(np.isnan(F_ram)):
        if not (np.all(np.isnan(F_wind)) and np.all(np.isnan(F_sn))):
            F_ram = np.nan_to_num(F_wind, nan=0.0) + np.nan_to_num(F_sn, nan=0.0)
        else:
            F_ram = np.zeros_like(t)

    rcloud = float(snaps[0].get("rCloud", np.nan))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase = t[order], R2[order], phase[order]
        F_grav, F_ram, F_ion, F_rad = F_grav[order], F_ram[order], F_ion[order], F_rad[order]
        F_wind, F_sn = F_wind[order], F_sn[order]

    base_forces = np.vstack([F_grav, F_ram, F_ion, F_rad])
    overlay_forces = np.vstack([F_wind, F_sn])

    return t, R2, phase, base_forces, overlay_forces, rcloud


def plot_run_on_ax(
    ax, t, R2, phase, base_forces, overlay_forces, rcloud,
    alpha=0.75,
    smooth_window=None, smooth_mode="edge",
    phase_change=True,
    overlay_alpha=0.55
):
    fig = ax.figure

    # --- phase markers (T and M)
    if phase_change:
        idx_T = np.flatnonzero(
            np.isin(phase[:-1], ["energy", "implicit"]) & (phase[1:] == "transition")
        ) + 1
        for x in t[idx_T]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "T",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="k", alpha=0.8,
                bbox=dict(facecolor="none", edgecolor="none", pad=0.2),
                zorder=6
            )

        idx_M = np.flatnonzero((phase[:-1] == "transition") & (phase[1:] == "momentum")) + 1
        for x in t[idx_M]:
            ax.axvline(x, color="r", lw=2, alpha=0.2, zorder=0)
            ax.text(
                x, 0.97, "M",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=8, color="k", alpha=0.8,
                bbox=dict(facecolor="none", edgecolor="none", pad=0.2),
                zorder=6
            )

    # --- breakout (first time R2 > rCloud)
    if np.isfinite(rcloud):
        idx = np.flatnonzero(np.isfinite(R2) & (R2 > rcloud))
        if idx.size:
            x_rc = t[idx[0]]
            ax.axvline(x_rc, color="k", ls="--", alpha=0.2, zorder=0)

            text_trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
                4/72, 0, fig.dpi_scale_trans
            )
            ax.text(
                x_rc, 0.95, r"$R_2 = R_{\rm cloud}$",
                transform=text_trans,
                ha="left", va="top",
                fontsize=8, color="k", alpha=0.9,
                rotation=90, zorder=6
            )

    # --- smoothing
    base_use = smooth_2d(base_forces, smooth_window, mode=smooth_mode)

    # --- stacked fractions (base)
    ftotal = base_use.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)

    frac = base_use / ftotal  # order: grav, ram, ion, rad
    cum  = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    for (_, _, color), y0, y1 in zip(FORCE_FIELDS_BASE, prev, cum):
        ax.fill_between(t, y0, y1, color=color, alpha=alpha, lw=0, zorder=2)

    # --- overlay wind/SN inside ram band, leaving residual unhatched
    if INCLUDE_ALL_FORCE:
        Fw_raw, Fsn_raw = overlay_forces[0], overlay_forces[1]

        if not (np.all(np.isnan(Fw_raw)) and np.all(np.isnan(Fsn_raw))):
            # Smooth overlay components too
            Fw  = smooth_1d(np.nan_to_num(Fw_raw,  nan=0.0), smooth_window, mode=smooth_mode)
            Fsn = smooth_1d(np.nan_to_num(Fsn_raw, nan=0.0), smooth_window, mode=smooth_mode)

            # Use smoothed total ram from base stack as denominator
            Fram = base_use[1].copy()

            # Fractions of the TOTAL effective ram force
            eps = 1e-30
            denom = np.where(np.isfinite(Fram) & (Fram > 0), Fram, np.nan)

            f_wind = np.nan_to_num(Fw  / (denom + eps), nan=0.0)
            f_sn   = np.nan_to_num(Fsn / (denom + eps), nan=0.0)

            # Clip and renormalize if wind+SN > 1 (numerical / model mismatch)
            f_wind = np.clip(f_wind, 0.0, 1.0)
            f_sn   = np.clip(f_sn,   0.0, 1.0)
            s = f_wind + f_sn
            mask = s > 1.0
            f_wind[mask] /= s[mask]
            f_sn[mask]   /= s[mask]
            # residual is whatever remains unhatched: 1 - f_wind - f_sn

            # Ram band bounds in the stacked fraction plot
            ram_bottom = prev[1]   # bottom of ram band
            ram_top    = cum[1]    # top of ram band
            ram_h      = ram_top - ram_bottom

            # Wind occupies the lowest fraction of the ram band
            y_wind_top = ram_bottom + f_wind * ram_h
            # SN goes above wind
            y_sn_top   = y_wind_top + f_sn * ram_h
            # Unhatched remainder is y_sn_top -> ram_top (visible blue)

            # Wind hatch (blue)
            ax.fill_between(
                t, ram_bottom, y_wind_top,
                facecolor="none", edgecolor=C_RAM,
                hatch="////", linewidth=0.0, alpha=overlay_alpha, zorder=4
            )
            # SN hatch (green)
            ax.fill_between(
                t, y_wind_top, y_sn_top,
                facecolor="none", edgecolor=C_SN,
                hatch="....", linewidth=0.0, alpha=overlay_alpha, zorder=4
            )
            
            # Outline only the top boundary
            ax.plot(t, y_wind_top, color=C_RAM, lw=1.0, alpha=0.9, zorder=6)
            # ax.plot(t, y_sn_top,   color=C_SN,  lw=1.0, alpha=0.9, zorder=6)

            # Use semi-transparent solid fill
            ax.fill_between(t, ram_bottom, y_wind_top, color=C_RAM, alpha=0.12, lw=0, zorder=4)
            # ax.fill_between(t, y_wind_top, y_sn_top,   color=C_SN,  alpha=0.12, lw=0, zorder=4)
            ax.plot(t, y_wind_top, color=C_RAM, lw=1.0, alpha=0.9, zorder=6)
            # ax.plot(t, y_sn_top,   color=C_SN,  lw=1.0, alpha=0.9, zorder=6)

    ax.set_ylim(0, 1)
    ax.set_xlim(t.min(), t.max())


# ---------------- main loop ----------------
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
            ax = axes[i, j]
            run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
            json_path = BASE_DIR / run_name / "dictionary.json"

            if not json_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            try:
                t, R2, phase, base_forces, overlay_forces, rcloud = load_run(json_path)
                plot_run_on_ax(
                    ax, t, R2, phase, base_forces, overlay_forces, rcloud,
                    alpha=0.75,
                    smooth_window=SMOOTH_WINDOW,
                    phase_change=PHASE_CHANGE
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

            # y label only on left-most
            if j == 0:
                mlog = int(np.log10(float(mCloud)))
                ax.set_ylabel(rf"$M_{{cloud}}=10^{{{mlog}}}\,M_\odot$" + "\n" + r"$F/F_{tot}$")
            else:
                ax.tick_params(labelleft=False)

            # x label only on bottom
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")
            else:
                ax.tick_params(labelbottom=False)

    # -------- global legend --------
    handles = [
        Patch(facecolor=C_GRAV, edgecolor="none", alpha=0.75, label="Gravity"),
        Patch(facecolor=C_RAM,  edgecolor="none", alpha=0.75, label=r"Ram total $F_{\rm ram}$ (blue)"),
        Patch(facecolor=C_ION,  edgecolor="none", alpha=0.75, label="Photoionised gas"),
        Patch(facecolor=C_RAD,  edgecolor="none", alpha=0.75, label="Radiation"),
    ]

    if INCLUDE_ALL_FORCE:
        handles += [
            Patch(facecolor="none", edgecolor=C_RAM, hatch="////", label=r"Wind share of $F_{\rm ram}$"),
            Patch(facecolor="none", edgecolor=C_SN,  hatch="....", label=r"SN share of $F_{\rm ram}$"),
            Line2D([0], [0], color=C_RAM, lw=6, label="Unhatched blue = residual"),
        ]

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.05)
    )
    leg.set_zorder(10)

    nlog = int(np.log10(float(ndens)))
    fig.suptitle(rf"Feedback force fractions grid  ($n=10^{{{nlog}}}\,\mathrm{{cm^{{-3}}}}$)", y=1.08)

    plt.show()
    plt.close(fig)

        
        
        