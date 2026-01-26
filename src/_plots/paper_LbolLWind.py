#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting Qi(or Li) vs Lmech_total with ratio on twin axis")

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
PLOT_QI = False            # True: plot Qi vs Lmech_total; False: plot Li vs Lmech_total
QI_PREFER_JSON = True     # if True and JSON has "Qi", use it; else estimate from Li

# If estimating Qi from Li [erg/s], choose mean ionizing photon energy:
MEAN_ION_PHOTON_ENERGY_EV = 20.0   # typical assumption (SED-dependent)

# Scales
LOG_LEFT_AXIS  = True     # log y for Li/Lmech_total or Qi/Lmech_total (if positive)
LOG_RATIO_AXIS = False    # usually keep linear

# colors
C_LWIND = "#1f77b4"   # blue
C_LI    = "#d62728"   # red
C_QI    = "#d62728"   # red (same role)
C_RATIO = "0.2"       # dark gray


# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PNG = False
SAVE_PDF = True

def range_tag(prefix, values, key=float):
    vals = list(values)
    if len(vals) == 1:
        return f"{prefix}{vals[0]}"
    vmin, vmax = min(vals, key=key), max(vals, key=key)
    return f"{prefix}{vmin}-{vmax}"

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


def load_run(data_path: Path):
    """Load t, phase, Li, Lmech_total, (optional) Qi, and R2/rCloud for breakout line.

    Uses TrinityOutput reader.
    """
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t = output.get('t_now')
    phase = np.array(output.get('current_phase', as_array=False))

    Li = output.get('Li')
    Lmech_total = output.get('Lmech_total')

    # Qi may or may not exist
    Qi = output.get('Qi')

    # for breakout marker
    R2 = output.get('R2')
    rcloud = float(output[0].get('rCloud', np.nan))

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # ensure increasing time
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t, phase = t[order], phase[order]
        Li, Lmech_total, Qi, R2 = Li[order], Lmech_total[order], Qi[order], R2[order]
        isCollapse = isCollapse[order]

    return t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse


def add_phase_and_cloud_markers(ax, t, phase, R2, rcloud, isCollapse=None, label_pad_points=4,
                                 show_collapse=True):
    """Phase T/M markers + breakout line + collapse line using helper module."""
    add_plot_markers(
        ax, t,
        phase=phase if PHASE_LINE else None,
        R2=R2 if CLOUD_LINE else None,
        rcloud=rcloud if CLOUD_LINE else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=PHASE_LINE,
        show_rcloud=CLOUD_LINE,
        show_collapse=show_collapse,
        label_pad_points=label_pad_points
    )


def plot_panel(ax, t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse=None):
    # smoothing
    Li_s    = smooth_1d(Li,    SMOOTH_WINDOW, mode=SMOOTH_MODE)
    L_mech_s = smooth_1d(Lmech_total, SMOOTH_WINDOW, mode=SMOOTH_MODE)
    Qi_s    = smooth_1d(Qi,    SMOOTH_WINDOW, mode=SMOOTH_MODE)
    R2_s    = smooth_1d(R2,    SMOOTH_WINDOW, mode=SMOOTH_MODE)

    add_phase_and_cloud_markers(ax, t, phase, R2_s, rcloud, isCollapse)

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
    ax.plot(t, L_mech_s, lw=1.8, color=C_LWIND,   label=r"$L_{\rm Wind}$", zorder=3)

    ax.set_xlim(t.min(), t.max())

    if LOG_LEFT_AXIS:
        y_all = np.concatenate([y_main[np.isfinite(y_main)], L_mech_s[np.isfinite(L_mech_s)]])
        if y_all.size and np.nanmin(y_all) > 0:
            ax.set_yscale("log")

    # right axis: ratio
    axr = ax.twinx()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y_main / L_mech_s
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
import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))


def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot Li/Lmech_total from a direct data path/folder.

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
        t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    axr = plot_panel(ax, t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse)

    ax.set_title(f"Li vs Lmech: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")

    if PLOT_QI:
        ax.set_ylabel(r"$Q_i\ [{\rm s^{-1}}]$, $L_{\rm Wind}$")
        axr.set_ylabel(r"$\mathcal{L}=Q_i/L_{\rm Wind}$")
    else:
        ax.set_ylabel(r"$L_i\ [{\rm erg\ s^{-1}}]$, $L_{\rm Wind}$")
        axr.set_ylabel(r"$\mathcal{L}=L_i/L_{\rm Wind}$")

    plt.tight_layout()

    # Save figures
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_LbolLWind_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_grid():
    """Plot full grid of Li/Lmech_total."""
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
                data_path = find_data_file(BASE_DIR, run_name)

                if data_path is None:
                    print(f"  {run_name}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"  Loading: {data_path}")
                try:
                    t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse = load_run(data_path)
                    axr = plot_panel(ax, t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse)
                except Exception as e:
                    print(f"Error in {run_name}: {e}")
                    ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                # column titles (top row)
                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                # left y label only on left-most - handle non-power-of-10 masses
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
                    ax.set_ylabel(mlabel + "\n" + r"left: main + $L_{\rm Wind}$")
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
        tag = f"LbolLWind_{m_tag}_{sfe_tag}_{n_tag}"

        if SAVE_PNG:
        if SAVE_PDF:
            out_pdf = FIG_DIR / f"{tag}.pdf"
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")

        plt.show()
        plt.close(fig)


def plot_folder_grid(folder_path, output_dir=None, ndens_filter=None):
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
    ndens_filter : str, optional
        Filter simulations by cloud density (e.g., "1e4", "1e3").
        If not specified, generates one PDF per density found.

    Notes
    -----
    Folder names must follow the pattern: {mCloud}_sfe{sfe}_n{ndens}
    Examples: "1e7_sfe020_n1e4", "5e6_sfe010_n1e3"
    """
    from src._output.trinity_reader import find_all_simulations, organize_simulations_for_grid, get_unique_ndens

    folder_path = Path(folder_path)
    folder_name = folder_path.name

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
        mCloud_list_found = organized['mCloud_list']
        sfe_list_found = organized['sfe_list']
        grid = organized['grid']

        if not mCloud_list_found or not sfe_list_found:
            print(f"Could not organize simulations into grid")
            continue

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

                print(f"    Loading: {data_path}")
                try:
                    t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse = load_run(data_path)
                    axr = plot_panel(ax, t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse)
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
                    ax.set_ylabel(mlabel + "\n" + r"left: main + $L_{\rm Wind}$")
                else:
                    ax.tick_params(labelleft=False)

                if j == ncols - 1:
                    if PLOT_QI:
                        axr.set_ylabel(r"$\mathcal{L}=Q_i/L_{\rm Wind}$")
                    else:
                        axr.set_ylabel(r"$\mathcal{L}=L_i/L_{\rm Wind}$")
                else:
                    axr.tick_params(labelright=False)
                    axr.set_ylabel("")

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

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

        fig.suptitle(f"{folder_name} (n{ndens})", fontsize=14, y=1.05)

        fig_dir = Path(output_dir) if output_dir else FIG_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        ndens_tag = f"n{ndens}"
        out_pdf = fig_dir / f"{folder_name}_{ndens_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY Li/Lmech_total comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_LbolLWind.py 1e7_sfe020_n1e4
  python paper_LbolLWind.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_LbolLWind.py /path/to/dictionary.jsonl

  # Folder-based grid (auto-discovers simulations)
  python paper_LbolLWind.py --folder /path/to/my_experiment/
  python paper_LbolLWind.py -F /path/to/simulations/

  # Uses config at top of file
  python paper_LbolLWind.py
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
        '--folder', '-F', default=None,
        help='Search folder recursively for simulations and create grid plot. '
             'Auto-organizes by mCloud (rows) and SFE (columns). '
             'Saves as {folder}_{ndens}.pdf'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3"). '
             'If not specified with --folder, generates one PDF per density found.'
    )

    args = parser.parse_args()

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir, ndens_filter=args.nCore)
    elif args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    else:
        # Config mode: plot grid
        plot_grid()
