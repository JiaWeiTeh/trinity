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
from src._output.trinity_reader import load_output, resolve_data_input, info_simulations
from src._plots.plot_markers import add_plot_markers, get_marker_legend_handles

print("...plotting integrated momentum (line plots)")

# --- output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True

PHASE_CHANGE = True

SMOOTH_WINDOW = None

DOMINANCE_DT = 0.05          # Myr
DOMINANCE_ALPHA = 0.9
DOMINANCE_STRIP = (0.94, 1)  # (ymin, ymax) in AXES fraction (0..1) - doubled thickness

# Main force fields for plotting (solid lines)
# F_ram is plotted continuously; F_ram_wind and F_ram_SN are plotted dashed separately
FORCE_FIELDS = [
    ("F_grav",    "Gravity",                  "black"),
    ("F_ram",     "Ram (thermal/wind)",       "b"),       # Continuous: thermal bubble + wind ram
    ("F_ion_out", "Photoionised gas",         "#d62728"),
    ("F_rad",     "Radiation (dir.+indir.)",  "#9467bd"),
]

# Additional dashed lines for wind/SN breakdown
DASHED_FIELDS = [
    ("F_ram_wind", "Wind",       "b"),       # Blue dashed
    ("F_ram_SN",   "Supernovae", "#FFA500"), # Yellow/orange dashed
]

# Colors for dominant bar (includes wind/SN subclassification)
DOMINANT_COLORS = {
    "F_grav": "black",
    "F_ram": "b",
    "F_ram_wind": "b",
    "F_ram_SN": "#FFA500",
    "F_ion_out": "#d62728",
    "F_rad": "#9467bd",
}


import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

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

    # Load isCollapse for collapse indicator
    isCollapse = np.array(output.get('isCollapse', as_array=False))

    # Extract main force fields
    forces_dict = {}
    for field, _, _ in FORCE_FIELDS:
        forces_dict[field] = np.nan_to_num(output.get(field), nan=0.0)

    # Extract dashed force fields (wind/SN breakdown)
    for field, _, _ in DASHED_FIELDS:
        forces_dict[field] = np.nan_to_num(output.get(field), nan=0.0)

    # Stack main forces for integration
    forces = np.vstack([forces_dict[field] for field, _, _ in FORCE_FIELDS])

    # Ensure time is increasing for integration
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]
        r = r[order]
        phase = phase[order]
        forces = forces[:, order]
        isCollapse = isCollapse[order]
        for key in forces_dict:
            forces_dict[key] = forces_dict[key][order]

    rcloud = float(output[0].get('rCloud', np.nan))
    return t, r, phase, forces, forces_dict, rcloud, isCollapse


def _interp_finite(x, y, xnew):
    """Interpolate y at xnew, handling NaN/inf values."""
    m = np.isfinite(y)
    if m.sum() < 2:
        return np.full_like(xnew, np.nan, dtype=float)
    return np.interp(xnew, x[m], y[m])


def dominant_bins_impulse(t, forces_dict, dt=0.05):
    """
    Compute dominant force in each time bin based on impulse added.

    ΔJ_i = ∫ max(F_i, 0) dt in each bin.

    If F_ram wins, subclassify into F_ram_wind vs F_ram_SN.

    Returns
    -------
    edges : array
        Bin edges
    winners : list of str
        Field name of winner in each bin (for color lookup)
    """
    t = np.asarray(t, float)
    edges = np.arange(t.min(), t.max() + dt, dt)
    n_bins = len(edges) - 1

    # Fields to consider for dominance (main forces)
    main_fields = ["F_grav", "F_ram", "F_ion_out", "F_rad"]

    winners = []

    for b in range(n_bins):
        t0, t1 = edges[b], edges[b + 1]

        # Find indices in this bin
        mask = (t >= t0) & (t < t1)
        if not np.any(mask):
            # No data in bin - use interpolation
            t_bin = np.array([0.5 * (t0 + t1)])
            impulses = {}
            for field in main_fields:
                F_interp = _interp_finite(t, forces_dict[field], t_bin)
                impulses[field] = max(F_interp[0], 0) * dt if np.isfinite(F_interp[0]) else 0.0
        else:
            t_bin = t[mask]
            impulses = {}
            for field in main_fields:
                F_bin = forces_dict[field][mask]
                # ΔJ = ∫ max(F, 0) dt using trapezoidal rule
                F_pos = np.maximum(F_bin, 0)
                if len(F_pos) > 1:
                    impulses[field] = np.trapz(F_pos, t_bin)
                else:
                    impulses[field] = F_pos[0] * dt

        # Find winner
        winner = max(impulses, key=impulses.get)

        # If F_ram wins, subclassify into wind vs SN
        if winner == "F_ram":
            if not np.any(mask):
                t_bin = np.array([0.5 * (t0 + t1)])
                F_wind = _interp_finite(t, forces_dict["F_ram_wind"], t_bin)
                F_SN = _interp_finite(t, forces_dict["F_ram_SN"], t_bin)
                J_wind = max(F_wind[0], 0) * dt if np.isfinite(F_wind[0]) else 0.0
                J_SN = max(F_SN[0], 0) * dt if np.isfinite(F_SN[0]) else 0.0
            else:
                F_wind = forces_dict["F_ram_wind"][mask]
                F_SN = forces_dict["F_ram_SN"][mask]
                F_wind_pos = np.maximum(F_wind, 0)
                F_SN_pos = np.maximum(F_SN, 0)
                if len(F_wind_pos) > 1:
                    J_wind = np.trapz(F_wind_pos, t_bin)
                    J_SN = np.trapz(F_SN_pos, t_bin)
                else:
                    J_wind = F_wind_pos[0] * dt
                    J_SN = F_SN_pos[0] * dt

            # Subclassify
            winner = "F_ram_wind" if J_wind >= J_SN else "F_ram_SN"

        winners.append(winner)

    return edges, winners


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
        t, r, phase, forces, forces_dict, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_momentum_lines_on_ax(
        ax, t, r, phase, forces, forces_dict, rcloud, isCollapse,
        smooth_window=SMOOTH_WINDOW,
        phase_change=PHASE_CHANGE
    )

    ax.set_title(f"Momentum Evolution: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$p(t)=\int F\,dt$")

    # Legend - force lines + markers from helper
    handles = []
    for _, lab, c in FORCE_FIELDS:
        handles.append(Line2D([0], [0], color=c, lw=1.6, ls="-", label=lab))
    for _, lab, c in DASHED_FIELDS:
        handles.append(Line2D([0], [0], color=c, lw=1.2, ls="--", label=lab))
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


def plot_momentum_lines_on_ax(
    ax, t, r, phase, forces, forces_dict, rcloud, isCollapse=None,
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

    # === Dominant force based on impulse in each bin
    edges, winners = dominant_bins_impulse(t, forces_dict, dt=DOMINANCE_DT)
    y0, y1 = DOMINANCE_STRIP

    # Merge consecutive bins with same winner to avoid white lines
    if len(winners) > 0:
        merged_spans = []  # list of (start_edge, end_edge, field_name)
        current_start = 0
        current_winner = winners[0]

        for b in range(1, len(winners)):
            if winners[b] != current_winner:
                # Winner changed - save previous span and start new one
                merged_spans.append((edges[current_start], edges[b], current_winner))
                current_start = b
                current_winner = winners[b]

        # Don't forget the last span
        merged_spans.append((edges[current_start], edges[len(winners)], current_winner))

        # Draw merged spans
        for x0, x1, field in merged_spans:
            color = DOMINANT_COLORS.get(field, "gray")
            ax.axvspan(
                x0, x1,
                ymin=y0, ymax=y1,
                color=color,
                alpha=DOMINANCE_ALPHA,
                lw=0,
                edgecolor='none',
                zorder=10
            )

    # --- integrate each force: p_i(t) = ∫ F_i dt  (signed)
    P = cumtrapz_2d(F, t)  # shape (n_forces, n_time)

    def plot_abs_with_sign_linestyle(ax, x, y, *, color, label=None, lw=1.6, alpha=0.95, zorder=3, base_ls="-"):
        """Plot |y| with solid for positive, dashed for negative."""
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

            # sign changes between i and i+1
            x0_seg, x1_seg = x[i], x[i + 1]
            y0_seg, y1_seg = y[i], y[i + 1]

            if y0_seg * y1_seg < 0:  # true crossing
                x_cross = x0_seg + (-y0_seg) * (x1_seg - x0_seg) / (y1_seg - y0_seg)
                xs.append(x_cross)
                ys.append(0.0)
                next_start_x, next_start_y = x_cross, 0.0
            else:
                next_start_x, next_start_y = x[i + 1], yabs[i + 1]

            # For base_ls="--", use dotted for negative to distinguish
            if base_ls == "--":
                ls = ":" if current_neg else "--"
            else:
                ls = "--" if current_neg else "-"
            ax.plot(
                xs, ys,
                color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder,
                label=(label if (label is not None and first_segment) else "_nolegend_"),
            )
            first_segment = False

            xs = [next_start_x, x[i + 1]]
            ys = [next_start_y, yabs[i + 1]]
            current_neg = neg[i + 1]

        # plot final segment
        if base_ls == "--":
            ls = ":" if current_neg else "--"
        else:
            ls = "--" if current_neg else "-"
        ax.plot(
            xs, ys,
            color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder,
            label=(label if (label is not None and first_segment) else "_nolegend_"),
        )

    # --- plot main force components (solid lines)
    for (field, label, color), Pi in zip(FORCE_FIELDS, P):
        plot_abs_with_sign_linestyle(ax, t, Pi, color=color, label=label, lw=lw, alpha=alpha, zorder=3)

    # --- plot wind/SN breakdown (dashed lines)
    for field, label, color in DASHED_FIELDS:
        F_dashed = forces_dict[field]
        if smooth_window:
            F_dashed = smooth_1d(F_dashed, smooth_window, mode=smooth_mode)
        P_dashed = cumtrapz_2d(F_dashed[None, :], t)[0]
        plot_abs_with_sign_linestyle(ax, t, P_dashed, color=color, label=label,
                                      lw=lw*0.8, alpha=alpha*0.8, zorder=2, base_ls="--")

    # net momentum (signed): integrate F_net = sum(outward) - gravity
    F_net = F[1:].sum(axis=0) - F[0]
    P_net = cumtrapz_2d(F_net[None, :], t)[0]
    plot_abs_with_sign_linestyle(ax, t, P_net, color="darkgrey", label="Net", lw=net_lw, alpha=0.8, zorder=4)

    ax.set_xlim(0, t.max())
    ax.set_yscale('log')
    ax.set_ylim(1e-5*P.max(), 10*P.max())

# ---------------- main loop ----------------

def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None):
    """
    Plot grid of momentum from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
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
        organized = organize_simulations_for_grid(
            sim_files, ndens_filter=ndens,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter
        )
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
                    t, r, phase, forces, forces_dict, rcloud, isCollapse = load_run(data_path)
                    plot_momentum_lines_on_ax(
                        ax, t, r, phase, forces, forces_dict, rcloud, isCollapse,
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
        for _, lab, c in FORCE_FIELDS:
            handles.append(Line2D([0], [0], color=c, lw=1.6, ls="-", label=lab))
        for _, lab, c in DASHED_FIELDS:
            handles.append(Line2D([0], [0], color=c, lw=1.2, ls="--", label=lab))
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
        fig.suptitle(f"{folder_name} (n{ndens})", fontsize=14, y=1.08)

        # Save figure to ./fig/{folder_name}/momentum_{ndens_tag}.pdf
        ndens_tag = f"n{ndens}"
        fig_dir = Path(output_dir) if output_dir else FIG_DIR / folder_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = fig_dir / f"momentum_{ndens_tag}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"  Saved: {out_pdf}")

        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot TRINITY momentum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single simulation
  python paper_momentum.py 1e7_sfe020_n1e4
  python paper_momentum.py /path/to/outputs/1e7_sfe020_n1e4

  # Grid plot from folder (auto-discovers simulations)
  python paper_momentum.py --folder /path/to/my_experiment/
  python paper_momentum.py -F /path/to/simulations/ -n 1e4

  # Filter by cloud mass and/or SFE
  python paper_momentum.py -F /path/to/simulations/ --mCloud 1e6 1e7
  python paper_momentum.py -F /path/to/simulations/ --sfe 001 010 020

  # Scan folder for available parameters
  python paper_momentum.py -F /path/to/simulations/ --info
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
        '--folder', '-F', default=None,
        help='Create grid plot from all simulations in folder.'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3").'
    )
    parser.add_argument(
        '--mCloud', nargs='+', default=None,
        help='Filter simulations by cloud mass (e.g., --mCloud 1e6 1e7).'
    )
    parser.add_argument(
        '--sfe', nargs='+', default=None,
        help='Filter simulations by SFE (e.g., --sfe 001 010).'
    )
    parser.add_argument(
        '--info', action='store_true',
        help='Scan folder and print available mCloud, SFE, and nCore values.'
    )

    args = parser.parse_args()

    if args.info:
        if not args.folder:
            parser.print_help()
            print("\nError: --info requires --folder to be specified.")
        else:
            info = info_simulations(args.folder)
            print("=" * 50)
            print(f"Simulation parameters in: {args.folder}")
            print("=" * 50)
            print(f"  Total simulations: {info['count']}")
            print(f"  mCloud values: {info['mCloud']}")
            print(f"  SFE values: {info['sfe']}")
            print(f"  nCore values: {info['ndens']}")
    elif args.folder:
        plot_grid(args.folder, args.output_dir, ndens_filter=args.nCore,
                  mCloud_filter=args.mCloud, sfe_filter=args.sfe)
    elif args.data:
        plot_from_path(args.data, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Please provide either --folder or a data path.")
