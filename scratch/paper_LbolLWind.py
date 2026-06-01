#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from paper.figures._lib.plot_base import FIG_DIR, smooth_1d

# Add project root to path for imports
from trinity._output.trinity_reader import load_output, resolve_data_input
from paper.figures._lib.plot_markers import add_plot_markers, get_marker_legend_handles
from paper.figures._lib.grid_template import (
    build_param_tag,
    iter_grid_densities,
    mark_missing_cell,
    attach_grid_legend,
    save_grid_figure,
    set_mcloud_ylabel,
    phii_file_prefix,
)

print("...plotting Qi(or Li) vs Lmech_total with ratio on twin axis")

# ---------------- configuration ----------------
SHOW_PHASE    = False
SHOW_RCLOUD    = False
SHOW_COLLAPSE = False
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
SAVE_PDF = True


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
                                 show_collapse=SHOW_COLLAPSE):
    """Phase T/M markers + breakout line + collapse line using helper module."""
    add_plot_markers(
        ax, t,
        phase=phase if SHOW_PHASE else None,
        R2=R2 if SHOW_RCLOUD else None,
        rcloud=rcloud if SHOW_RCLOUD else None,
        isCollapse=isCollapse if show_collapse else None,
        show_phase=SHOW_PHASE,
        show_rcloud=SHOW_RCLOUD,
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


def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None, phii_mode="yes"):
    """
    Plot grid of luminosity comparison from simulations in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing simulation subfolders.
    output_dir : str or Path, optional
        Directory to save figure (default: FIG_DIR)
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4"). If None, creates one
        PDF per unique density found.
    phii_mode : {"yes", "no"}
        PHII suffix variant to plot.  See ``grid_template.filter_sim_files_by_phii``.
    """
    for ndens, mCloud_list, sfe_list, grid, folder_name in iter_grid_densities(
            folder_path, ndens_filter=ndens_filter,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter,
            phii_mode=phii_mode):

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=False,
            dpi=500,
            squeeze=False,
            constrained_layout=False,
        )

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    mark_missing_cell(ax, "missing")
                    continue

                try:
                    t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse = load_run(data_path)
                    axr = plot_panel(ax, t, phase, Li, Lmech_total, Qi, R2, rcloud, isCollapse)
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                if j == 0:
                    set_mcloud_ylabel(ax, mCloud, extra=r"left: main + $L_{\rm Wind}$")
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
        handles.extend(get_marker_legend_handles(include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD, include_collapse=SHOW_COLLAPSE))

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, handles,
            n_rows_for_layout=nrows,
            folder_name=folder_name,
            param_tag=param_tag,
            legend_ncol=3,
            legend_bbox_transform_fig=True,
        )

        save_grid_figure(
            fig, folder_name=folder_name,
            file_prefix=phii_file_prefix("LbolLWind", phii_mode),
            param_tag=param_tag, output_dir=output_dir,
        )
        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


# ---------------- command-line interface ----------------
if __name__ == "__main__":
    from paper.figures._lib.cli import dispatch, marker_pre_dispatch
    dispatch(
        script_name="paper_LbolLWind.py",
        description="Plot TRINITY luminosity comparison (Lbol vs Lwind)",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
