#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force fraction grid with phase-aware composition overlays within F_drive.

- Base stack uses: F_grav, F_drive, F_rad, F_ion_in (PISM)
- Energy phase: plain F_drive band + thin driver-indicator strip
  showing which branch of max(Pb, P_HII) is active.
- Transition phase: hatched overlay (P_HII + ram_wind + ram_SN)
  shown only when the non-bubble branch wins max(Pb, P_HII + P_ram).
- Momentum phase: hatched overlay P_HII + ram_wind + ram_SN (always).
- Phase markers: T (enter transition), M (enter momentum)
- Breakout marker: first time R2 > rCloud (vertical dashed + label)
- X ticks on every subplot; x tick labels only on bottom row.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR, smooth_1d, smooth_2d
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting force fractions with ram composition overlay + PISM")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 21           # None or 1 disables smoothing
PHASE_CHANGE  = True         # Show phase transition markers
INCLUDE_ALL_FORCE = True     # Show wind/SN overlays inside the ram band
USE_LOG_X = False            # Use log scale for x-axis (time)

# Colors — centralised ChromaPalette (switch via set_palette or $TRINITY_PALETTE)
from src._plots.force_colors import C, FORCE_FIELDS_BASE  # noqa: E402

C_GRAV  = C.GRAV
C_DRIVE = C.DRIVE
C_SN    = C.SN
C_PHII  = C.PHII
C_RAD   = C.RAD
C_PISM  = C.PISM
C_WIND  = C.WIND

SAVE_PDF = True

def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot feedback force fractions from a direct data path/folder.

    Parameters
    ----------
    data_input : str
        Can be: folder name, folder path, or file path
    output_dir : str, optional
        Base directory for output folders
    """
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


    try:
        t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse, pressures = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_run_on_ax(
        ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse,
        pressures=pressures,
        alpha=0.75,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE,
        use_log_x=USE_LOG_X
    )

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$F/F_{tot}$")
    ax.set_title(f"Feedback Fractions: {data_path.parent.name}")

    # Legend - force colors + markers from helper
    handles = [
        Patch(facecolor=C_GRAV,  edgecolor="none", alpha=0.75, label="Gravity"),
        Patch(facecolor=C_DRIVE, edgecolor="none", alpha=0.75, label=r"$F_{\rm drive}$"),
        Patch(facecolor=C_RAD,   edgecolor="none", alpha=0.75, label="Radiation"),
        Patch(facecolor=C_PISM,  edgecolor="0.3", linewidth=0.8, alpha=1.0,  label="PISM (inner HII)"),
        Patch(facecolor="none", edgecolor=C_PHII, hatch="......",     label=r"$P_{\rm HII}$"),
        Patch(facecolor="none", edgecolor=C_WIND, hatch="\\\\\\\\",   label=r"Ram wind"),
        Patch(facecolor="none", edgecolor=C_SN,   hatch="////",       label=r"Ram SN"),
        Line2D([0], [0], color=C_PHII, lw=3, alpha=0.7, label=r"Driver: $P_{\rm HII}$"),
        Line2D([0], [0], color=C_DRIVE, lw=3, alpha=0.7, label=r"Driver: $P_b$"),
    ]
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    # Save figures to ./fig/{parent_folder}/feedback_{run_name}.pdf
    run_name = data_path.parent.name
    parent_folder = data_path.parent.parent.name
    fig_dir = FIG_DIR / parent_folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"feedback_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def load_run(data_path: Path):
    """Load run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    R2 = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))

    # Helper to get field with default
    def get_field(field, default=np.nan):
        arr = output.get(field)
        if arr is None or (isinstance(arr, np.ndarray) and np.all(arr == None)):
            return np.full(len(output), default)
        return np.where(arr == None, default, arr).astype(float)

    F_grav = get_field("F_grav", 0.0)
    F_rad  = get_field("F_rad", 0.0)

    # F_drive = P_drive * 4πR² — the actual driving pressure force used in the ODE
    P_drive = get_field("P_drive", 0.0)
    R2_safe = np.nan_to_num(R2, nan=0.0)
    F_drive = P_drive * 4.0 * np.pi * R2_safe**2

    # Overlay components for drive-band decomposition
    # F_HII_St = P_HII_St * 4πR² (actual driving contribution from Strömgren)
    F_HII_St = get_field("F_HII_St", 0.0)
    F_wind = get_field("F_ram_wind", np.nan)
    F_sn   = get_field("F_ram_SN", np.nan)

    # Pressure fields for driver-indicator logic
    Pb_arr      = get_field("Pb", 0.0)
    P_HII_St_arr = get_field("P_HII_St", 0.0)

    # PISM: press_HII_in is a pressure — convert to force via F = P * 4πR²
    F_PISM_raw = get_field("press_HII_in", np.nan)
    F_PISM_raw = np.nan_to_num(F_PISM_raw, nan=0.0)
    F_PISM = F_PISM_raw * 4.0 * np.pi * R2_safe**2

    rcloud = float(output[0].get('rCloud', np.nan))

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase = t[order], R2[order], phase[order]
        F_grav, F_drive, F_rad = F_grav[order], F_drive[order], F_rad[order]
        F_PISM = F_PISM[order]
        F_HII_St = F_HII_St[order]
        F_wind, F_sn = F_wind[order], F_sn[order]
        Pb_arr, P_HII_St_arr = Pb_arr[order], P_HII_St_arr[order]
        isCollapse = isCollapse[order]

    # base forces order must match FORCE_FIELDS_BASE
    base_forces    = np.vstack([F_grav, F_drive, F_rad, F_PISM])
    overlay_forces = np.vstack([F_HII_St, F_wind, F_sn])
    pressures      = np.vstack([Pb_arr, P_HII_St_arr])

    return t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse, pressures


def plot_run_on_ax(
    ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse=None,
    pressures=None,
    alpha=0.75,
    smooth_window=None, smooth_mode="edge",
    phase_change=True,
    show_rcloud=True,
    show_collapse=True,
    overlay_alpha=0.55,
    use_log_x=False
):
    # --- Add all time-axis markers using helper module
    add_plot_markers(
        ax, t,
        phase=phase if phase_change else None,
        R2=R2 if show_rcloud else None,
        rcloud=rcloud if show_rcloud else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=phase_change,
        show_rcloud=show_rcloud,
        show_collapse=show_collapse
    )

    # --- normalize first (raw fractions), then smooth
    ftotal = base_forces.sum(axis=0)
    ftotal = np.where(ftotal == 0.0, np.nan, ftotal)
    frac_raw = base_forces / ftotal
    frac = smooth_2d(frac_raw, smooth_window, mode=smooth_mode)

    # --- stacked fractions (base)
    cum  = np.cumsum(frac, axis=0)
    prev = np.vstack([np.zeros_like(t), cum[:-1]])

    # Fill base stack with thin black outlines for visibility
    for (field, _, color), y0, y1 in zip(FORCE_FIELDS_BASE, prev, cum):
        a = 1.0 if field == "F_ion_in" else alpha
        ax.fill_between(t, y0, y1, facecolor=color, alpha=a,
                        edgecolor="black", linewidth=0.4, zorder=4)

    # --- Phase-aware overlay decomposition inside F_drive band ---
    if INCLUDE_ALL_FORCE:
        Fhii_raw = overlay_forces[0]
        Fw_raw   = overlay_forces[1]
        Fsn_raw  = overlay_forces[2]

        # Three phase masks (energy, transition, momentum)
        energy_mask     = np.array([p in ('energy', 'energy_implicit') for p in phase])
        transition_mask = np.array([p == 'transition' for p in phase])
        momentum_mask   = np.array([p == 'momentum' for p in phase])

        # Drive band bounds in the stacked fraction plot (index 1 = F_drive)
        drive_bottom = prev[1]
        drive_top    = cum[1]
        drive_h      = drive_top - drive_bottom

        eps = 1e-30

        # ---- Energy phase: NO hatched ram overlay (ram thermalises at R1) ----
        # Instead, add thin driver-indicator strip at top of F_drive band
        # showing which branch of max(Pb, P_HII_St) is active.
        if np.any(energy_mask) and pressures is not None:
            en_idx = np.where(energy_mask)[0]
            t_en = t[en_idx]
            Pb_en      = pressures[0][en_idx]
            Phii_St_en = pressures[1][en_idx]

            dt_en = drive_top[en_idx]
            strip_h = 0.012  # thin strip in fraction-space
            strip_bot = dt_en - strip_h
            strip_bot = np.maximum(strip_bot, drive_bottom[en_idx])

            # Where P_HII_St > Pb: teal/red strip (HII is driver)
            hii_wins = Phii_St_en > Pb_en
            if np.any(hii_wins):
                idx_hii = en_idx[hii_wins]
                ax.fill_between(
                    t[idx_hii], drive_top[idx_hii] - strip_h, drive_top[idx_hii],
                    facecolor=C_PHII, alpha=0.7, lw=0, zorder=5
                )
            # Where Pb >= P_HII_St: drive colour strip (bubble is driver)
            pb_wins = ~hii_wins
            if np.any(pb_wins):
                idx_pb = en_idx[pb_wins]
                ax.fill_between(
                    t[idx_pb], drive_top[idx_pb] - strip_h, drive_top[idx_pb],
                    facecolor=C_DRIVE, alpha=0.7, lw=0, zorder=5
                )

        # ---- Transition phase: hatched overlay only when non-bubble branch wins ----
        # P_drive = max(Pb, P_HII_St + P_ram). Show decomposition only when
        # P_HII_St + P_ram > Pb (i.e. the HII+ram branch is active).
        if np.any(transition_mask):
            tr_idx = np.where(transition_mask)[0]

            # Determine which branch is active from pressures
            if pressures is not None:
                Pb_tr       = pressures[0][tr_idx]
                Phii_St_tr  = pressures[1][tr_idx]
                Fw_tr   = np.nan_to_num(Fw_raw[tr_idx], nan=0.0)
                Fsn_tr  = np.nan_to_num(Fsn_raw[tr_idx], nan=0.0)
                R2_tr   = R2[tr_idx]
                # P_ram from force: F_ram = P_ram * 4πR²
                P_ram_tr = np.where(R2_tr > 0, (Fw_tr + Fsn_tr) / (4.0 * np.pi * R2_tr**2 + eps), 0.0)
                non_bubble = (Phii_St_tr + P_ram_tr) > Pb_tr
            else:
                non_bubble = np.ones(len(tr_idx), dtype=bool)

            # Only show hatching where non-bubble branch wins
            show_idx = tr_idx[non_bubble]
            if len(show_idx) > 0:
                t_tr = t[show_idx]
                Fhii_clean = np.nan_to_num(Fhii_raw[show_idx], nan=0.0)
                Fw_clean   = np.nan_to_num(Fw_raw[show_idx],   nan=0.0)
                Fsn_clean  = np.nan_to_num(Fsn_raw[show_idx],  nan=0.0)

                Ftotal_tr = Fhii_clean + Fw_clean + Fsn_clean
                denom_tr = np.where(Ftotal_tr > 0, Ftotal_tr, np.nan)

                f_hii  = np.nan_to_num(Fhii_clean / (denom_tr + eps), nan=0.0)
                f_wind = np.nan_to_num(Fw_clean   / (denom_tr + eps), nan=0.0)
                f_sn   = np.nan_to_num(Fsn_clean  / (denom_tr + eps), nan=0.0)

                f_hii  = np.clip(f_hii,  0.0, 1.0)
                f_wind = np.clip(f_wind, 0.0, 1.0)
                f_sn   = np.clip(f_sn,   0.0, 1.0)
                s = f_hii + f_wind + f_sn
                over = s > 1.0
                f_hii[over]  /= s[over]
                f_wind[over] /= s[over]
                f_sn[over]   /= s[over]

                db_tr = drive_bottom[show_idx]
                dh_tr = drive_h[show_idx]

                y_wind_top = db_tr + f_wind * dh_tr
                y_sn_top   = y_wind_top + f_sn * dh_tr
                y_hii_top  = y_sn_top + f_hii * dh_tr

                _draw_hatched_overlay(ax, t_tr, db_tr, y_wind_top, y_sn_top, y_hii_top)
                ax.fill_between(t_tr, db_tr, drive_top[show_idx], color=C_DRIVE, alpha=0.10, lw=0, zorder=4)

            # Driver strip for bubble-dominated transition timesteps
            bubble_idx = tr_idx[~non_bubble]
            if len(bubble_idx) > 0:
                strip_h = 0.012
                ax.fill_between(
                    t[bubble_idx], drive_top[bubble_idx] - strip_h, drive_top[bubble_idx],
                    facecolor=C_DRIVE, alpha=0.7, lw=0, zorder=5
                )

        # ---- Momentum phase: P_HII + wind + SN within F_drive (always) ----
        if np.any(momentum_mask):
            mom_idx = np.where(momentum_mask)[0]
            t_post = t[mom_idx]

            Fhii_clean = np.nan_to_num(Fhii_raw[mom_idx], nan=0.0)
            Fw_clean   = np.nan_to_num(Fw_raw[mom_idx],   nan=0.0)
            Fsn_clean  = np.nan_to_num(Fsn_raw[mom_idx],  nan=0.0)

            Ftotal_post = Fhii_clean + Fw_clean + Fsn_clean
            denom_post = np.where(np.isfinite(Ftotal_post) & (Ftotal_post > 0), Ftotal_post, np.nan)

            f_hii  = np.nan_to_num(Fhii_clean / (denom_post + eps), nan=0.0)
            f_wind = np.nan_to_num(Fw_clean   / (denom_post + eps), nan=0.0)
            f_sn   = np.nan_to_num(Fsn_clean  / (denom_post + eps), nan=0.0)

            f_hii  = np.clip(f_hii,  0.0, 1.0)
            f_wind = np.clip(f_wind, 0.0, 1.0)
            f_sn   = np.clip(f_sn,   0.0, 1.0)
            s = f_hii + f_wind + f_sn
            over = s > 1.0
            f_hii[over]  /= s[over]
            f_wind[over] /= s[over]
            f_sn[over]   /= s[over]

            db_post = drive_bottom[mom_idx]
            dh_post = drive_h[mom_idx]

            y_wind_top = db_post + f_wind * dh_post
            y_sn_top   = y_wind_top + f_sn * dh_post
            y_hii_top  = y_sn_top + f_hii * dh_post

            _draw_hatched_overlay(ax, t_post, db_post, y_wind_top, y_sn_top, y_hii_top)
            ax.fill_between(t_post, db_post, drive_top[mom_idx], color=C_DRIVE, alpha=0.10, lw=0, zorder=4)

    ax.set_ylim(0, 1)

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


def _draw_hatched_overlay(ax, t_seg, db, y_wind_top, y_sn_top, y_hii_top):
    """Draw the hatched wind / SN / P_HII overlay within a drive band segment."""
    # Wind slice: back slashes
    ax.fill_between(
        t_seg, db, y_wind_top,
        facecolor="none", edgecolor=C_WIND,
        hatch="\\\\\\\\", linewidth=0, alpha=0.9, zorder=3
    )
    ax.fill_between(
        t_seg, db, y_wind_top,
        facecolor="none", edgecolor="black", linestyle=":", linewidth=0.4, zorder=6
    )

    # SN slice: forward slashes
    for _ in range(4):
        ax.fill_between(
            t_seg, y_wind_top, y_sn_top,
            facecolor="none", edgecolor=C_SN,
            hatch="////", linewidth=0, alpha=0.9, zorder=3
        )
    ax.fill_between(
        t_seg, y_wind_top, y_sn_top,
        facecolor="none", edgecolor="black", linestyle=":", linewidth=0.4, zorder=6
    )

    # P_HII slice: dots — topmost
    ax.fill_between(
        t_seg, y_sn_top, y_hii_top,
        facecolor="none", edgecolor=C_PHII,
        hatch="......", linewidth=0, alpha=0.9, zorder=3
    )
    ax.fill_between(
        t_seg, y_sn_top, y_hii_top,
        facecolor="none", edgecolor="black", linestyle=":", linewidth=0.4, zorder=6
    )


# ---------------- main loop ----------------

def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of feedback fractions from simulations in a folder.

    Dynamically discovers simulations from the folder, organizes them into
    a grid by mCloud (rows) and SFE (columns).

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
    mCloud_filter : list of str, optional
        Filter simulations by cloud mass (e.g., ["1e6", "1e7"]).
    sfe_filter : list of str, optional
        Filter simulations by SFE (e.g., ["001", "010"]).

    Notes
    -----
    Folder names must follow the pattern: {mCloud}_sfe{sfe}_n{ndens}
    Examples: "1e7_sfe020_n1e4", "5e6_sfe010_n1e3"
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid, get_unique_ndens

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # Find all simulations
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    # Determine which densities to plot
    if ndens_filter:
        ndens_to_plot = [ndens_filter]
    else:
        ndens_to_plot = get_unique_ndens(sim_files)

    print(f"Found {len(sim_files)} simulations")
    print(f"  Densities to plot: {ndens_to_plot}")

    # Create one grid per density
    for ndens in ndens_to_plot:
        print(f"\nProcessing n={ndens}...")

        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
        mCloud_list_use = organized['mCloud_list']
        sfe_list_use = organized['sfe_list']
        grid = organized['grid']

        if not mCloud_list_use or not sfe_list_use:
            print(f"  Could not organize simulations into grid for n={ndens}")
            continue

        print(f"  mCloud: {mCloud_list_use}")
        print(f"  SFE: {sfe_list_use}")

        nrows, ncols = len(mCloud_list_use), len(sfe_list_use)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=True,
            dpi=500,
            squeeze=False,
            constrained_layout=False
        )

        for i, mCloud in enumerate(mCloud_list_use):
            for j, sfe in enumerate(sfe_list_use):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    run_id = f"{mCloud}_sfe{sfe}_n{ndens}"
                    print(f"  {run_id}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                try:
                    t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse, pressures = load_run(data_path)
                    plot_run_on_ax(
                        ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse,
                        pressures=pressures,
                        alpha=0.75,
                        smooth_window=SMOOTH_WINDOW,
                        phase_change=PHASE_CHANGE,
                        use_log_x=USE_LOG_X
                    )
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                # Ticks: show tick marks everywhere, labels only on bottom row
                ax.tick_params(axis="x", which="both", bottom=True)
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

                # Column titles (top row only)
                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                # Row labels (left column only)
                if j == 0:
                    mval = float(mCloud)
                    mexp = int(np.floor(np.log10(mval)))
                    mcoeff = mval / (10 ** mexp)
                    mcoeff = round(mcoeff)
                    if mcoeff == 10:
                        mcoeff = 1
                        mexp += 1
                    if mcoeff == 1:
                        mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}\,M_\odot$"
                    else:
                        mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"
                    ax.set_ylabel(mlabel + "\n" + r"$F/F_{tot}$")
                else:
                    ax.tick_params(labelleft=False)

        # Global legend
        handles = [
            Patch(facecolor=C_GRAV,  edgecolor="none", alpha=0.75, label="Gravity"),
            Patch(facecolor=C_DRIVE, edgecolor="none", alpha=0.75, label=r"$F_{\rm drive}$ (blue)"),
            Patch(facecolor=C_RAD,   edgecolor="none", alpha=0.75, label="Radiation"),
            Patch(facecolor=C_PISM,  edgecolor="0.3", linewidth=0.8, alpha=1.0,  label="PISM (inner HII)"),
        ]

        if INCLUDE_ALL_FORCE:
            handles += [
                Patch(facecolor="none", edgecolor=C_PHII, hatch="......",     label=r"$P_{\rm HII}$"),
                Patch(facecolor="none", edgecolor=C_WIND, hatch="\\\\\\\\",   label=r"Ram wind"),
                Patch(facecolor="none", edgecolor=C_SN,   hatch="////",       label=r"Ram SN"),
                Line2D([0], [0], color=C_PHII, lw=3, alpha=0.7, label=r"Driver: $P_{\rm HII}$"),
                Line2D([0], [0], color=C_DRIVE, lw=3, alpha=0.7, label=r"Driver: $P_b$"),
            ]

        handles.extend(get_marker_legend_handles())

        # fig.subplots_adjust(top=0.88)

        leg = fig.legend(
            handles=handles,
            loc="upper center",
            ncol=4,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="0.2",
            # bbox_to_anchor=(0.5, 1.0),
            bbox_to_anchor=(0.5, 1.2),
            fontsize=7,
        )
        leg.set_zorder(10)

        # Title and filename
        ndens_tag = f"n{ndens}"
        # fig.suptitle(f"{folder_name} ({ndens_tag})", fontsize=14, y=1.03)

        # Save figure to ./fig/{folder_name}/feedback_n{ndens}.pdf
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_PDF:
            out_pdf = fig_dir / f"feedback_{ndens_tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_feedback.py",
        description="Plot TRINITY feedback force fractions",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
