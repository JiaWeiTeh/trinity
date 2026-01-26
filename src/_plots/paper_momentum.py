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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import load_output, find_data_file, resolve_data_input
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting integrated momentum (line plots)")

# --- configuration
# mCloud_list = ["1e5", "1e7", "1e8"]                 # rows
# ndens_list  = ["1e4", "1e2", "1e3"]                               # one figure per ndens
# sfe_list    = ["001", "010", "020", "030", "050", "080"]   # cols

mCloud_list = ["1e5", "5e5", "1e6", "5e6", "1e7", "5e7", "1e8"]
ndens_list = ["1e3"]
sfe_list = ["001", "005", "010", "020", "030", "050", "070", "080"]


BASE_DIR = Path.home() / "unsync" / "Code" / "Trinity" / "outputs" / "sweep_test_modified"

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
ONLY_SFE    = "010"   # e.g. "030"

# comment this out if want single mode, otherwise leave this be if want grid. 
ONLY_NDENS = ONLY_MCLOUD = ONLY_SFE = None

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
    """Load run data using TrinityOutput reader."""
    output = load_output(data_path)

    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    # Use TrinityOutput.get() for clean array extraction
    t = output.get('t_now')
    r = output.get('R2')
    phase = np.array(output.get('current_phase', as_array=False))

    # Extract force fields
    forces = np.vstack([
        np.nan_to_num(output.get(field), nan=0.0)
        for field, _, _ in FORCE_FIELDS
    ])

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Ensure time is increasing for integration
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        r = r[order]
        phase = phase[order]
        forces = forces[:, order]
        isCollapse = isCollapse[order]

    rcloud = float(output[0].get('rCloud', np.nan))
    return t, r, phase, forces, rcloud, isCollapse

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

def plot_from_path(data_input: str, output_dir: str = None):
    """
    Plot momentum evolution from a direct data path/folder.

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
        t, r, phase, forces, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_momentum_lines_on_ax(
        ax, t, r, phase, forces, rcloud, isCollapse,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE
    )

    ax.set_title(f"Momentum Evolution: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$p(t)=\int F\,dt$")

    # Legend - force lines + markers from helper
    handles = [Line2D([0], [0], color="black", lw=1.6, ls="-", label="Gravity")]
    for _, lab, c in FORCE_FIELDS[1:]:
        handles.append(Line2D([0], [0], color=c, lw=1.6, label=lab))
    handles.append(Line2D([0], [0], color="darkgrey", lw=2.4, label="Net"))
    handles.extend(get_marker_legend_handles())
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    plt.tight_layout()

    # Save figures
    run_name = data_path.parent.name
    out_pdf = FIG_DIR / f"paper_momentum_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


def plot_single_run(mCloud, ndens, sfe):
    run_name = f"{mCloud}_sfe{sfe}_n{ndens}"
    data_path = find_data_file(BASE_DIR, run_name)

    if data_path is None:
        print(f"Missing data for: {run_name}")
        return

    t, r, phase, forces, rcloud, isCollapse = load_run(data_path)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=500, constrained_layout=True)
    plot_momentum_lines_on_ax(
        ax, t, r, phase, forces, rcloud, isCollapse,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE
    )

    ax.set_title(run_name)
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$p(t)=\int F\,dt$")
    if SAVE_PDF:
        fig.savefig(FIG_DIR / f"paper_momentum_{run_name}.pdf", bbox_inches='tight')
        print(f"Saved: {FIG_DIR / f'paper_momentum_{run_name}.pdf'}")
    plt.show()
    plt.close(fig)


def plot_momentum_lines_on_ax(
    ax, t, r, phase, forces, rcloud, isCollapse=None,
    smooth_window=None, smooth_mode="edge",
    lw=1.6, net_lw=4, alpha=0.8, phase_change=PHASE_CHANGE,
    show_rcloud=True, show_collapse=True,
):
    # --- Add all time-axis markers using helper module
    add_plot_markers(
        ax, t,
        phase=phase if phase_change else None,
        R2=r if show_rcloud else None,
        rcloud=rcloud if show_rcloud else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=phase_change,
        show_rcloud=show_rcloud,
        show_collapse=show_collapse
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
def plot_grid():
    """Plot full grid of momentum evolution."""
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
                    print(f"  {run_name}: missing")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                print(f"  Loading: {data_path}")
                try:
                    t, r, phase, forces, rcloud, isCollapse = load_run(data_path)
                    plot_momentum_lines_on_ax(
                        ax, t, r, phase, forces, rcloud, isCollapse,
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
                    # Handle non-power-of-10 masses (e.g., 5e6)
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
                    ax.set_ylabel(mlabel + "\n" + r"$p(t)=\int F\,dt$")

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        # global legend - force lines + markers from helper
        handles = []
        handles.append(Line2D([0], [0], color="black", lw=1.6, ls="-", label="Gravity"))
        for _, lab, c in FORCE_FIELDS[1:]:
            handles.append(Line2D([0], [0], color=c, lw=1.6, label=lab))
        handles.append(Line2D([0], [0], color="darkgrey", lw=2.4,
                              label=r"Net: $| \int (\sum F_{\rm out} - F_{\rm grav})\,dt |$"))
        handles.extend(get_marker_legend_handles())

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
            print(f"Saved: {FIG_DIR / f'paper_momentum_n{ndens}.pdf'}")
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
                t, r, phase, forces, rcloud, isCollapse = load_run(data_path)
                plot_momentum_lines_on_ax(
                    ax, t, r, phase, forces, rcloud, isCollapse,
                    smooth_window=SMOOTH_WINDOW,
                    phase_change=PHASE_CHANGE
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
                ax.set_ylabel(mlabel + "\n" + r"$p(t)=\int F\,dt$")

            if i == nrows - 1:
                ax.set_xlabel("t [Myr]")

    handles = []
    handles.append(Line2D([0], [0], color="black", lw=1.6, ls="-", label="Gravity"))
    for _, lab, c in FORCE_FIELDS[1:]:
        handles.append(Line2D([0], [0], color=c, lw=1.6, label=lab))
    handles.append(Line2D([0], [0], color="darkgrey", lw=2.4,
                          label=r"Net: $| \int (\sum F_{\rm out} - F_{\rm grav})\,dt |$"))
    handles.extend(get_marker_legend_handles())

    leg = fig.legend(
        handles=handles, loc="upper center", ncol=3,
        frameon=True, facecolor="white", framealpha=0.9, edgecolor="0.2",
        bbox_to_anchor=(0.5, 1.05)
    )
    leg.set_zorder(10)

    fig.subplots_adjust(top=0.91)
    fig.suptitle(folder_name, fontsize=14, y=1.08)

    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    ndens = organized['ndens']
    ndens_tag = f"n{ndens}" if ndens else "nMixed"
    out_pdf = fig_dir / f"{folder_name}_{ndens_tag}_momentum.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.close(fig)


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY momentum evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_momentum.py 1e7_sfe020_n1e4
  python paper_momentum.py /path/to/outputs/1e7_sfe020_n1e4
  python paper_momentum.py /path/to/dictionary.jsonl

  # Folder-based grid (auto-discovers simulations)
  python paper_momentum.py --folder /path/to/my_experiment/
  python paper_momentum.py -F /path/to/simulations/

  # Uses config at top of file
  python paper_momentum.py
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

    args = parser.parse_args()

    if args.folder:
        plot_folder_grid(args.folder, args.output_dir)
    elif args.data:
        # Command-line mode: plot from specified path
        plot_from_path(args.data, args.output_dir)
    elif (ONLY_MCLOUD is not None) and (ONLY_NDENS is not None) and (ONLY_SFE is not None):
        # Config mode: single run
        plot_single_run(ONLY_MCLOUD, ONLY_NDENS, ONLY_SFE)
    else:
        # Config mode: plot grid
        plot_grid()
