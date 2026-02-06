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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting force fractions with ram composition overlay + PISM")

# ---------------- configuration ----------------
SMOOTH_WINDOW = 21           # None or 1 disables smoothing
PHASE_CHANGE  = True         # Show phase transition markers
INCLUDE_ALL_FORCE = True     # Show wind/SN overlays inside the ram band
USE_LOG_X = False            # Use log scale for x-axis (time)

# Colors
C_GRAV = "black"
C_RAM  = "b"
C_SN   = "#DAA520"  # golden yellow for SN visibility
C_ION  = "#d62728"
C_RAD  = "#9467bd"
C_PISM = "white"

# Base stacked forces — order matters for stacking + overlay indexing
FORCE_FIELDS_BASE = [
    ("F_grav",    "Gravity",                 C_GRAV),
    ("F_ram",     r"Ram total $F_{\rm ram}$", C_RAM),
    ("F_ion_out", "Photoionised gas",        C_ION),
    ("F_rad",     "Radiation (dir.+indir.)", C_RAD),
    ("F_ion_in",      "PISM",                C_PISM),
]

# Output directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True


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

    print(f"Loading data from: {data_path}")

    try:
        t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_run_on_ax(
        ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse,
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
        Patch(facecolor=C_GRAV, edgecolor="none", alpha=0.75, label="Gravity"),
        Patch(facecolor=C_RAM,  edgecolor="none", alpha=0.75, label=r"Ram total"),
        Patch(facecolor=C_ION,  edgecolor="none", alpha=0.75, label="Photoionised gas"),
        Patch(facecolor=C_RAD,  edgecolor="none", alpha=0.75, label="Radiation"),
        Patch(facecolor=C_PISM, edgecolor="0.4",  alpha=1.0,  label="PISM"),
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

    rcloud = float(output[0].get('rCloud', np.nan))

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time increasing
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, R2, phase = t[order], R2[order], phase[order]
        F_grav, F_ram, F_ion, F_rad, F_PISM = F_grav[order], F_ram[order], F_ion[order], F_rad[order], F_PISM[order]
        F_wind, F_sn = F_wind[order], F_sn[order]
        isCollapse = isCollapse[order]

    # base forces order must match FORCE_FIELDS_BASE
    base_forces    = np.vstack([F_grav, F_ram, F_ion, F_rad, F_PISM])
    overlay_forces = np.vstack([F_wind, F_sn])

    return t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse


def plot_run_on_ax(
    ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse=None,
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

    # --- overlay wind/SN inside ram band, ONLY AFTER MOMENTUM PHASE
    if INCLUDE_ALL_FORCE:
        Fw_raw, Fsn_raw = overlay_forces[0], overlay_forces[1]

        if not (np.all(np.isnan(Fw_raw)) and np.all(np.isnan(Fsn_raw))):
            # Find momentum phase start index
            momentum_mask = np.array([p == 'momentum' for p in phase])
            if np.any(momentum_mask):
                momentum_start_idx = np.argmax(momentum_mask)  # first True index
            else:
                momentum_start_idx = len(t)  # no momentum phase = no hatching

            # Only proceed if we have momentum phase data
            if momentum_start_idx < len(t):
                # Slice arrays to momentum phase only
                t_mom = t[momentum_start_idx:]

                # Smooth overlay components too
                Fw  = smooth_1d(np.nan_to_num(Fw_raw,  nan=0.0), smooth_window, mode=smooth_mode)
                Fsn = smooth_1d(np.nan_to_num(Fsn_raw, nan=0.0), smooth_window, mode=smooth_mode)

                # Slice to momentum phase
                Fw_mom = Fw[momentum_start_idx:]
                Fsn_mom = Fsn[momentum_start_idx:]

                # Use smoothed total ram from base stack as denominator
                Fram = base_use[1].copy()
                Fram_mom = Fram[momentum_start_idx:]

                eps = 1e-30
                denom = np.where(np.isfinite(Fram_mom) & (Fram_mom > 0), Fram_mom, np.nan)

                f_wind = np.nan_to_num(Fw_mom  / (denom + eps), nan=0.0)
                f_sn   = np.nan_to_num(Fsn_mom / (denom + eps), nan=0.0)

                # Clip and renormalize if wind+SN > 1
                f_wind = np.clip(f_wind, 0.0, 1.0)
                f_sn   = np.clip(f_sn,   0.0, 1.0)
                s = f_wind + f_sn
                mask = s > 1.0
                f_wind[mask] /= s[mask]
                f_sn[mask]   /= s[mask]

                # Ram band bounds in the stacked fraction plot (still index 1)
                ram_bottom_mom = prev[1][momentum_start_idx:]
                ram_top_mom    = cum[1][momentum_start_idx:]
                ram_h_mom      = ram_top_mom - ram_bottom_mom

                y_wind_top = ram_bottom_mom + f_wind * ram_h_mom
                y_sn_top   = y_wind_top + f_sn * ram_h_mom

                # --- Wind slice: forward slashes, normal hatch density
                ax.fill_between(
                    t_mom, ram_bottom_mom, y_wind_top,
                    facecolor="none",
                    edgecolor=C_RAM,          # blue
                    hatch="////",
                    linewidth=0.8,            # hatch stroke weight
                    alpha=0.9,
                    zorder=5
                )

                # --- SN slice: back slashes, yellow color for visibility
                for _ in range(4):  # draw multiple times for thicker hatch
                    ax.fill_between(
                        t_mom, y_wind_top, y_sn_top,
                        facecolor="none",
                        edgecolor=C_SN,       # yellow for SN
                        hatch="\\\\\\\\",     # opposite direction
                        linewidth=2.5,        # slightly heavier stroke
                        alpha=0.9,
                        zorder=5
                    )

                # Helpful boundaries - blue for wind/ram top, yellow for SN top
                ax.plot(t_mom, y_wind_top, color=C_RAM, lw=1.2, alpha=0.95, zorder=6)
                ax.plot(t_mom, y_sn_top,   color=C_SN,  lw=1.2, alpha=0.95, zorder=6)

                # Optional: subtle tint to keep "ram is blue" obvious without overpowering
                ax.fill_between(t_mom, ram_bottom_mom, ram_top_mom, color=C_RAM, alpha=0.10, lw=0, zorder=4)

    ax.set_ylim(0, 1)

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


# ---------------- main loop ----------------

def plot_grid(folder_path, output_dir=None, ndens_filter=None):
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

        organized = organize_simulations_for_grid(sim_files, ndens_filter=ndens)
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

                print(f"    Loading: {data_path}")
                try:
                    t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse = load_run(data_path)
                    plot_run_on_ax(
                        ax, t, R2, phase, base_forces, overlay_forces, rcloud, isCollapse,
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
            Patch(facecolor=C_GRAV, edgecolor="none", alpha=0.75, label="Gravity"),
            Patch(facecolor=C_RAM,  edgecolor="none", alpha=0.75, label=r"Ram total $F_{\rm ram}$ (blue)"),
            Patch(facecolor=C_ION,  edgecolor="none", alpha=0.75, label="Photoionised gas"),
            Patch(facecolor=C_RAD,  edgecolor="none", alpha=0.75, label="Radiation"),
            Patch(facecolor=C_PISM, edgecolor="0.4",  alpha=1.0,  label="PISM"),
        ]

        if INCLUDE_ALL_FORCE:
            handles += [
                Patch(facecolor="none", edgecolor=C_RAM, hatch="////",   label=r"Ram attributed to winds (blue)"),
                Patch(facecolor="none", edgecolor=C_SN,  hatch="\\\\\\\\", label=r"Ram attributed to SN (yellow)"),
                Line2D([0], [0], color=C_RAM, lw=6, label="Unhatched blue = residual"),
            ]

        handles.extend(get_marker_legend_handles())

        fig.subplots_adjust(top=0.9)

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

        # Title and filename
        ndens_tag = f"n{ndens}"
        fig.suptitle(f"{folder_name} ({ndens_tag})", fontsize=14, y=1.08)

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


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY feedback force fractions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_feedback.py 1e7_sfe020_n1e4
  python paper_feedback.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_feedback.py /path/to/dictionary.jsonl

  # Grid plot from folder (auto-discovers simulations)
  python paper_feedback.py --folder /path/to/my_experiment/
  python paper_feedback.py -F /path/to/simulations/
  python paper_feedback.py -F /path/to/simulations/ -n 1e4  # filter by density
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path (for single simulation)'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Directory to save output figures (default: fig/)'
    )
    parser.add_argument(
        '--log-x', action='store_true',
        help='Use log scale for x-axis (time)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Create grid plot from all simulations in folder. '
             'Auto-organizes by mCloud (rows) and SFE (columns).'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3"). '
             'If not specified, generates one PDF per density found.'
    )

    args = parser.parse_args()

    if args.log_x:
        USE_LOG_X = True

    if args.folder:
        # Grid mode: create grid from all simulations in folder
        plot_grid(args.folder, args.output_dir, ndens_filter=args.nCore)
    elif args.data:
        # Single simulation mode
        plot_from_path(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
