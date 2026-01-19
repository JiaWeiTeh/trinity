#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force fraction grid with ram composition overlay (wind+SN within F_ram),
PLUS an extra top component: PISM (white band at the top).

- Base stack uses: F_grav, F_ram, F_ion_out, F_rad, PISM
- Hatched overlays show wind/SN as fractions of F_ram, leaving an unhatched residual.
- Phase markers: T (enter transition), M (enter momentum)
- Breakout marker: first time R2 > rCloud (vertical dashed + label)
- X ticks on every subplot; x tick labels only on bottom row.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from load_snapshots import load_snapshots, find_data_file

print("...plotting force fractions with ram composition overlay + PISM")

# ---------------- configuration ----------------
mCloud_list = ["1e5", "1e7", "1e8"]                  # rows
ndens_list  = ["1e2", "1e3", "1e4"]                                # one figure per ndens
sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs"

SMOOTH_WINDOW = 21           # None or 1 disables
PHASE_CHANGE  = True
INCLUDE_ALL_FORCE = True     # show wind/SN overlays inside the ram band

# Colors
C_GRAV = "black"
C_RAM  = "b"
C_SN   = "#2ca02c"
C_ION  = "#d62728"
C_RAD  = "#9467bd"
C_PISM = "white"             # requested: white top band

# Base stacked forces (always) — order matters for stacking + overlay indexing
FORCE_FIELDS_BASE = [
    ("F_grav",    "Gravity",                 C_GRAV),
    ("F_ram",     r"Ram total $F_{\rm ram}$", C_RAM),
    ("F_ion_out", "Photoionised gas",        C_ION),
    ("F_rad",     "Radiation (dir.+indir.)", C_RAD),
    ("F_ion_in",      "PISM",                    C_PISM),
]

# --- optional single-run view (set to None for full grid)
ONLY_M   = '1e7'   # e.g. "1e5" or None
ONLY_N   = '1e4'   # e.g. "1e4" or None
ONLY_SFE = '001'   # e.g. "001" or None

# ONLY_M = ONLY_N = ONLY_SFE = None

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


import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


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

def plot_single_run(mCloud, ndens, sfe):
    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
    data_path = find_data_file(BASE_DIR, run_name)
    if data_path is None:
        print(f"Missing data for: {run_name}")
        return

    t, R2, phase, base_forces, overlay_forces, rcloud = load_run(data_path)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=400, constrained_layout=True)
    plot_run_on_ax(
        ax, t, R2, phase, base_forces, overlay_forces, rcloud,
        alpha=0.75,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE
    )

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$F/F_{tot}$")
    ax.set_title(f"{run_name}")

    tag = f"feedback_{mCloud}_sfe{sfe}_n{ndens}"
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

def load_run(data_path: Path):
    """Load run data. Supports both JSON and JSONL formats."""
    snaps = load_snapshots(data_path)

    if not snaps:
        raise ValueError(f"No snapshots found in {data_path}")

    t     = np.array([s["t_now"] for s in snaps], dtype=float)
    R2    = np.array([s.get("R2", np.nan) for s in snaps], dtype=float)
    phase = np.array([s.get("current_phase", "") for s in snaps])

    def get_field(field, default=np.nan):
        return np.array([s.get(field, default) for s in snaps], dtype=float)

    F_grav = get_field("F_grav", 0.0)
    F_ion  = get_field("F_ion_out", 0.0)
    F_rad  = get_field("F_rad", 0.0)

    # total ram (energy-balance effective)
    F_ram  = get_field("F_ram", np.nan)

    # decomposition (SPS output)
    F_wind = get_field("F_ram_wind", np.nan)
    F_sn   = get_field("F_ram_SN", np.nan)

    # PISM: try press_HII_in first, else PISM, else 0
    F_PISM = get_field("press_HII_in", np.nan)
    F_PISM = np.nan_to_num(F_PISM, nan=0.0)

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
        F_grav, F_ram, F_ion, F_rad, F_PISM = F_grav[order], F_ram[order], F_ion[order], F_rad[order], F_PISM[order]
        F_wind, F_sn = F_wind[order], F_sn[order]

    # base forces order must match FORCE_FIELDS_BASE
    base_forces    = np.vstack([F_grav, F_ram, F_ion, F_rad, F_PISM])
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
                fontsize=8, color="k", alpha=0.85,
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
                fontsize=8, color="k", alpha=0.85,
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

    frac = base_use / ftotal  # order: grav, ram, ion, rad, pism
    cum  = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    # Fill base stack; make PISM white but visible in legend via edgecolor there
    for (field, _, color), y0, y1 in zip(FORCE_FIELDS_BASE, prev, cum):
        a = 1.0 if field == "PISM" else alpha  # keep white crisp
        ax.fill_between(t, y0, y1, color=color, alpha=a, lw=0, zorder=2)

    # --- overlay wind/SN inside ram band, leaving residual unhatched
    if INCLUDE_ALL_FORCE:
        Fw_raw, Fsn_raw = overlay_forces[0], overlay_forces[1]

        if not (np.all(np.isnan(Fw_raw)) and np.all(np.isnan(Fsn_raw))):
            # Smooth overlay components too
            Fw  = smooth_1d(np.nan_to_num(Fw_raw,  nan=0.0), smooth_window, mode=smooth_mode)
            Fsn = smooth_1d(np.nan_to_num(Fsn_raw, nan=0.0), smooth_window, mode=smooth_mode)

            # Use smoothed total ram from base stack as denominator
            Fram = base_use[1].copy()

            eps = 1e-30
            denom = np.where(np.isfinite(Fram) & (Fram > 0), Fram, np.nan)

            f_wind = np.nan_to_num(Fw  / (denom + eps), nan=0.0)
            f_sn   = np.nan_to_num(Fsn / (denom + eps), nan=0.0)

            # Clip and renormalize if wind+SN > 1
            f_wind = np.clip(f_wind, 0.0, 1.0)
            f_sn   = np.clip(f_sn,   0.0, 1.0)
            s = f_wind + f_sn
            mask = s > 1.0
            f_wind[mask] /= s[mask]
            f_sn[mask]   /= s[mask]

            # Ram band bounds in the stacked fraction plot (still index 1)
            ram_bottom = prev[1]
            ram_top    = cum[1]
            ram_h      = ram_top - ram_bottom

            y_wind_top = ram_bottom + f_wind * ram_h
            y_sn_top   = y_wind_top + f_sn * ram_h

            # --- Wind slice: forward slashes, normal hatch density
            ax.fill_between(
                t, ram_bottom, y_wind_top,
                facecolor="none",
                edgecolor=C_RAM,          # blue
                hatch="////",
                linewidth=0.8,            # hatch stroke weight (was 0.0)
                alpha=0.9,
                zorder=5
            )
            
            # --- SN slice: back slashes, "thicker" by overdrawing the hatch twice
            for _ in range(4):  # draw twice -> visually thicker/darker hatch
                ax.fill_between(
                    t, y_wind_top, y_sn_top,
                    facecolor="none",
                    edgecolor=C_RAM,      # still blue
                    hatch="\\\\\\\\",     # opposite direction
                    linewidth=2.5,        # slightly heavier stroke
                    alpha=0.9,
                    zorder=5
                )
            
            # Helpful boundaries (still blue) so the SN region is obvious
            ax.plot(t, y_wind_top, color=C_RAM, lw=1.2, alpha=0.95, zorder=6)
            ax.plot(t, y_sn_top,   color=C_RAM, lw=1.2, alpha=0.95, zorder=6)
            
            # Optional: subtle tint to keep "ram is blue" obvious without overpowering
            ax.fill_between(t, ram_bottom, ram_top, color=C_RAM, alpha=0.10, lw=0, zorder=4)

    ax.set_ylim(0, 1)
    ax.set_xlim(t.min(), t.max())


# ---------------- main loop ----------------

# If any filter is set, do single-run mode
if (ONLY_M is not None) or (ONLY_N is not None) or (ONLY_SFE is not None):
    m = ONLY_M if ONLY_M is not None else mCloud_list[0]
    n = ONLY_N if ONLY_N is not None else ndens_list[0]
    s = ONLY_SFE if ONLY_SFE is not None else sfe_list[0]
    plot_single_run(m, n, s)

else:
    # --- full grid mode (your existing code)
    for ndens in ndens_list:
        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=True,
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
                    t, R2, phase, base_forces, overlay_forces, rcloud = load_run(data_path)
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
    
                # ---- ticks: show tick marks everywhere, labels only on bottom row
                ax.tick_params(axis="x", which="both", bottom=True)  # ticks on all
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")
    
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
    
        # -------- global legend --------
        handles = [
            Patch(facecolor=C_GRAV, edgecolor="none", alpha=0.75, label="Gravity"),
            Patch(facecolor=C_RAM,  edgecolor="none", alpha=0.75, label=r"Ram total $F_{\rm ram}$ (blue)"),
            Patch(facecolor=C_ION,  edgecolor="none", alpha=0.75, label="Photoionised gas"),
            Patch(facecolor=C_RAD,  edgecolor="none", alpha=0.75, label="Radiation"),
            # PISM is white: give it a border so it’s visible in the legend
            Patch(facecolor=C_PISM, edgecolor="0.4",  alpha=1.0,  label="PISM"),
        ]
        
        if INCLUDE_ALL_FORCE:
            handles += [
                Patch(facecolor="none", edgecolor=C_RAM, hatch="////",   label=r"Ram attributed to winds"),
                Patch(facecolor="none", edgecolor=C_RAM, hatch="\\\\\\\\", label=r"Ram attributed to SN (thicker hatch)"),
                Line2D([0], [0], color=C_RAM, lw=6, label="Unhatched blue = residual"),
            ]
            
        # Reserve top space so legend never overlaps subplot titles
        fig.subplots_adjust(top=0.9)           # <-- tune: smaller = more header space

    
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
    
        # --------- SAVE FIGURE ---------
        m_tag   = range_tag("M",   mCloud_list, key=float)
        sfe_tag = range_tag("sfe", sfe_list,    key=int)
        n_tag   = f"n{ndens}"
        tag = f"feedback_grid_{m_tag}_{sfe_tag}_{n_tag}"

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
