#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 17:45:12 2026

@author: Jia Wei Teh
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

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
USE_LOG_X = False  # Use log scale for x-axis (time)

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = True
SAVE_PDF = True

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

    Uses TrinityOutput reader for clean data access.
    """
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    phase = np.array(output.get('current_phase', as_array=False))

    v2 = output.get('v2')  # pc/Myr
    R1 = output.get('R1')
    R2 = output.get('R2')
    rShell = output.get('rShell')

    xi_Tb = output.get('bubble_xi_Tb')
    r_Tb = R2 * xi_Tb

    rcloud = float(output[0].get('rCloud', np.nan))

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # enforce increasing time (robust if there are tiny non-monotonicities)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        v2, R1, R2, rShell, r_Tb = v2[order], R1[order], R2[order], rShell[order], r_Tb[order]
        isCollapse = isCollapse[order]

    return t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse


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
    ax, t, phase, v2_pcmyr, R1, R2, rShell, r_Tb, rcloud, isCollapse=None,
    smooth_window=None, smooth_mode="edge",
    phase_line=True, cloud_line=True, show_collapse=True,
    label_pad_points=4, use_log_x=False
):
    # smoothing
    v2s = smooth_1d(v2_pcmyr, smooth_window, mode=smooth_mode)
    R1s = smooth_1d(R1, smooth_window, mode=smooth_mode)
    R2s = smooth_1d(R2, smooth_window, mode=smooth_mode)
    rSs = smooth_1d(rShell, smooth_window, mode=smooth_mode)
    rTbs = smooth_1d(r_Tb, smooth_window, mode=smooth_mode)

    # convert velocity to km/s
    v2_kms = v2s * PC_PER_MYR_TO_KMS

    # --- Add all time-axis markers using helper module
    add_plot_markers(
        ax, t,
        phase=phase if phase_line else None,
        R2=R2s if cloud_line else None,
        rcloud=rcloud if cloud_line else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=phase_line,
        show_rcloud=cloud_line,
        show_collapse=show_collapse,
        label_pad_points=label_pad_points
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

    # X-axis scale
    if use_log_x:
        ax.set_xscale('log')
        t_pos = t[t > 0]
        if len(t_pos) > 0:
            ax.set_xlim(t_pos.min(), t.max())
    else:
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


def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot velocity evolution from a direct data path/folder.

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

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    try:
        t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse = load_run_velocity(data_path)
        axr = plot_velocity_on_ax(
            ax, t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse,
            smooth_window=SMOOTH_WINDOW,
            smooth_mode=SMOOTH_MODE,
            phase_line=PHASE_LINE,
            cloud_line=CLOUD_LINE,
            use_log_x=USE_LOG_X
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        plt.close(fig)
        return

    ax.set_title(f"Velocity Evolution: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$|v_2|$ [km s$^{-1}$]")
    axr.set_ylabel("Radius [pc]")

    # Legend - velocity/radii lines + markers from helper
    handles = [
        Line2D([0], [0], color="k", lw=1.8, ls="-",  label=r"$v_2>0$ (solid)"),
        Line2D([0], [0], color="k", lw=1.8, ls="--", label=r"$v_2<0$ (dashed)"),
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
    ]
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    plt.tight_layout()

    # Save figures
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_expansionVelocity_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of velocity evolution."""
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
                    print(f"  {run_name}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"  Loading: {data_path}")
                try:
                    t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse = load_run_velocity(data_path)
                    axr = plot_velocity_on_ax(
                        ax, t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse,
                        smooth_window=SMOOTH_WINDOW,
                        smooth_mode=SMOOTH_MODE,
                        phase_line=PHASE_LINE,
                        cloud_line=CLOUD_LINE,
                        use_log_x=USE_LOG_X
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

                # left y labels only on left-most column - handle non-power-of-10 masses
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
                    ax.set_ylabel(mlabel + "\n" + r"$v_2$ [km s$^{-1}$]")
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
        ]
        handles.extend(get_marker_legend_handles())

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
        if SAVE_PDF:
            out_pdf = FIG_DIR / f"{tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
            print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)


def plot_folder_grid(folder_path, output_dir=None):
    """
    Create grid plot from all simulations found in a folder.

    Searches subfolders for dictionary.jsonl files, parses simulation
    parameters from folder names (e.g., "1e7_sfe020_n1e4"), and arranges
    them in a grid sorted by:
    - Rows: increasing mCloud (top to bottom)
    - Columns: increasing SFE (left to right)

    Saves PDF as {folder_name}_{ndens}.pdf without displaying.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)

    Notes
    -----
    Folder names must follow the pattern: {mCloud}_sfe{sfe}_n{ndens}
    Examples: "1e7_sfe020_n1e4", "5e6_sfe010_n1e3"
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid

    folder_path = Path(folder_path)
    folder_name = folder_path.name

    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return

    organized = organize_simulations_for_grid(sim_files)
    mCloud_list_found = organized['mCloud_list']
    sfe_list_found = organized['sfe_list']
    grid = organized['grid']

    if not mCloud_list_found or not sfe_list_found:
        print(f"Could not organize simulations into grid")
        return

    print(f"Found {len(sim_files)} simulations")
    print(f"  mCloud: {mCloud_list_found}")
    print(f"  SFE: {sfe_list_found}")

    nrows, ncols = len(mCloud_list_found), len(sfe_list_found)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        sharex=False, sharey=False,
        dpi=500,
        squeeze=False,
        constrained_layout=False
    )

    fig.subplots_adjust(top=0.90)

    for i, mCloud in enumerate(mCloud_list_found):
        for j, sfe in enumerate(sfe_list_found):
            ax = axes[i, j]
            data_path = grid.get((mCloud, sfe))

            if data_path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            print(f"  Loading: {data_path}")
            try:
                t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse = load_run_velocity(data_path)
                axr = plot_velocity_on_ax(
                    ax, t, phase, v2, R1, R2, rShell, r_Tb, rcloud, isCollapse,
                    smooth_window=SMOOTH_WINDOW,
                    smooth_mode=SMOOTH_MODE,
                    phase_line=PHASE_LINE,
                    cloud_line=CLOUD_LINE,
                    use_log_x=USE_LOG_X
                )
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            if i == 0:
                eps = int(sfe) / 100.0
                ax.set_title(rf"$\epsilon={eps:.2f}$")

            if j == 0:
                mval = float(mCloud)
                mexp = int(np.floor(np.log10(mval)))
                mcoeff = round(mval / (10 ** mexp))
                if mcoeff == 10:
                    mcoeff = 1
                    mexp += 1
                if mcoeff == 1:
                    mlabel = rf"$M_{{\rm cloud}}=10^{{{mexp}}}\,M_\odot$"
                else:
                    mlabel = rf"$M_{{\rm cloud}}={mcoeff}\times10^{{{mexp}}}\,M_\odot$"
                ax.set_ylabel(mlabel + "\n" + r"$v_2$ [km s$^{-1}$]")
            else:
                ax.tick_params(labelleft=False)

            if j != ncols - 1:
                axr.set_ylabel("")
                axr.tick_params(labelright=False)

            ax.tick_params(axis="x", which="both", bottom=True)
            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")
                ax.tick_params(labelbottom=True)
            else:
                ax.tick_params(labelbottom=False)

    handles = [
        Line2D([0], [0], color="k", lw=1.8, ls="-",  label=r"$v_2>0$ (solid; plotted as $|v_2|$)"),
        Line2D([0], [0], color="k", lw=1.8, ls="--", label=r"$v_2<0$ (dashed; plotted as $|v_2|$)"),
        Line2D([0], [0], color=RADIUS_FIELDS[0][2], lw=RADIUS_FIELDS[0][4], ls=RADIUS_FIELDS[0][3], label=RADIUS_FIELDS[0][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[1][2], lw=RADIUS_FIELDS[1][4], ls=RADIUS_FIELDS[1][3], label=RADIUS_FIELDS[1][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[2][2], lw=RADIUS_FIELDS[2][4], ls=RADIUS_FIELDS[2][3], label=RADIUS_FIELDS[2][1]),
        Line2D([0], [0], color=RADIUS_FIELDS[3][2], lw=RADIUS_FIELDS[3][4], ls=RADIUS_FIELDS[3][3], label=RADIUS_FIELDS[3][1]),
    ]
    handles.extend(get_marker_legend_handles())

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

    fig.suptitle(folder_name, fontsize=14, y=1.05)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    ndens = organized['ndens']
    ndens_tag = f"n{ndens}" if ndens else "nMixed"
    out_pdf = fig_dir / f"{folder_name}_{ndens_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_pdf}")

    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY velocity evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_expansionVelocity.py 1e7_sfe020_n1e4
  python paper_expansionVelocity.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_expansionVelocity.py /path/to/dictionary.jsonl

  # Folder-based grid (auto-discovers simulations)
  python paper_expansionVelocity.py --folder /path/to/my_experiment/
  python paper_expansionVelocity.py -F /path/to/simulations/

  # Uses config at top of file
  python paper_expansionVelocity.py
        """
    )
    parser.add_argument(
        'data', nargs='?', default=None,
        help='Data input: folder name, folder path, or file path'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Base directory for output folders (default: TRINITY_OUTPUT_DIR or "outputs")'
    )
    parser.add_argument(
        '--log-x', action='store_true',
        help='Use log scale for x-axis (time)'
    )
    parser.add_argument(
        '--folder', '-F', default=None,
        help='Search folder recursively for simulations and create grid plot. '
             'Auto-organizes by mCloud (rows) and SFE (columns). '
             'Saves as {folder}_{ndens}.pdf'
    )

    args = parser.parse_args()

    if args.log_x:
        USE_LOG_X = True

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir)
    elif args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    else:
        # Config mode: plot grid
        plot_grid()
