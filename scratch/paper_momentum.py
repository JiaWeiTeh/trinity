#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:26:34 2025

@author: Jia Wei Teh
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent))
from paper.figures._lib.plot_base import FIG_DIR, smooth_1d, smooth_2d
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

print("...plotting integrated momentum (line plots)")

SAVE_PDF = True

SHOW_PHASE = False
SHOW_RCLOUD = False
SHOW_COLLAPSE = False

SMOOTH_WINDOW = 11

DOMINANCE_DT = 0.1          # Myr
DOMINANCE_ALPHA = 0.9
DOMINANCE_STRIP = (0.94, 1)  # (ymin, ymax) in AXES fraction (0..1) - doubled thickness

# Colors — centralised ChromaPalette (switch via set_palette or $TRINITY_PALETTE)
from paper.figures._lib.force_colors import (          # noqa: E402
    FORCE_FIELDS_MOMENTUM as FORCE_FIELDS,
    DOMINANT_COLORS,
)



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

    # Helper: safely get a numeric field (handles None values from missing keys)
    def safe_field(field):
        arr = output.get(field)
        if arr is None:
            return np.zeros(len(output))
        arr = np.where(arr == None, np.nan, arr).astype(float)
        return np.nan_to_num(arr, nan=0.0)

    # Extract main force fields
    forces_dict = {}
    R2_safe = np.nan_to_num(r, nan=0.0)

    for field, _, _ in FORCE_FIELDS:
        if field == "F_PISM":
            continue  # handled separately below
        if field == "F_drive":
            # F_drive = P_drive * 4πR² — the actual driving pressure force
            P_drive = safe_field('P_drive')
            forces_dict["F_drive"] = P_drive * 4.0 * np.pi * R2_safe**2
            continue
        forces_dict[field] = safe_field(field)

    # PISM: press_HII_in is a pressure — convert to force via F = P * 4πR²
    press_HII_in = safe_field('press_HII_in')
    F_PISM = press_HII_in * 4.0 * np.pi * R2_safe**2
    forces_dict["F_PISM"] = F_PISM

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


def dominant_bins_impulse(t, forces_dict, phase=None, dt=0.05):
    """
    Compute dominant force in each time bin based on impulse added.

    ΔJ_i = ∫ max(F_i, 0) dt in each bin.

    F_drive is never sub-classified — the decomposition story is told
    by paper_feedback.py (force fractions) and the pressure evolution plot.

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

    # Fields to consider for dominance (main forces only)
    main_fields = ["F_grav", "F_drive", "F_rad", "F_PISM"]

    winners = []

    for b in range(n_bins):
        t0, t1 = edges[b], edges[b + 1]

        # Find indices in this bin
        mask = (t >= t0) & (t < t1)
        if not np.any(mask):
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
                F_pos = np.maximum(F_bin, 0)
                if len(F_pos) > 1:
                    impulses[field] = np.trapz(F_pos, t_bin)
                else:
                    impulses[field] = F_pos[0] * dt

        winner = max(impulses, key=impulses.get)
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


    try:
        t, r, phase, forces, forces_dict, rcloud, isCollapse = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_momentum_lines_on_ax(
        ax, t, r, phase, forces, forces_dict, rcloud, isCollapse,
        smooth_window=SMOOTH_WINDOW,
        phase_change=SHOW_PHASE
    )

    ax.set_title(f"Momentum Evolution: {data_path.parent.name}")
    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$p(t)=\int F\,\mathrm{d}t$ [$M_\odot\,\mathrm{pc}\,\mathrm{Myr}^{-1}$]")

    # Legend - force lines + markers from helper
    handles = []
    for _, lab, c in FORCE_FIELDS:
        handles.append(Line2D([0], [0], color=c, lw=1.6, ls="-", label=lab))
    handles.append(Line2D([0], [0], color="darkgrey", lw=2.4, label="Net"))
    handles.extend(get_marker_legend_handles(include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD, include_collapse=SHOW_COLLAPSE))
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
    lw=1.6, net_lw=4, alpha=0.8, phase_change=SHOW_PHASE,
    show_rcloud=SHOW_RCLOUD, show_collapse=SHOW_COLLAPSE,
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

    # === Dominant force based on impulse in each bin (uses raw forces_dict)
    edges, winners = dominant_bins_impulse(t, forces_dict, phase=phase, dt=DOMINANCE_DT)
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

    # --- integrate raw forces first, then smooth the resulting momentum curves
    P_raw = cumtrapz_2d(forces, t)  # shape (n_forces, n_time)
    P = smooth_2d(P_raw, smooth_window, mode=smooth_mode)

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

    # net momentum (signed): integrate F_net = sum(outward) - gravity, using raw forces
    F_net = forces[1:].sum(axis=0) - forces[0]
    P_net_raw = cumtrapz_2d(F_net[None, :], t)[0]
    P_net = smooth_1d(P_net_raw, smooth_window, mode=smooth_mode) if smooth_window else P_net_raw
    plot_abs_with_sign_linestyle(ax, t, P_net, color="darkgrey", label="Net", lw=net_lw, alpha=0.8, zorder=4)

    ax.set_xlim(0, t.max())
    ax.set_yscale('log')
    ax.set_ylim(1e-5*P.max(), 10*P.max())

# ---------------- main loop ----------------

def plot_grid(folder_path, output_dir=None, ndens_filter=None,
              mCloud_filter=None, sfe_filter=None, phii_mode="yes"):
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
                    t, r, phase, forces, forces_dict, rcloud, isCollapse = load_run(data_path)
                    plot_momentum_lines_on_ax(
                        ax, t, r, phase, forces, forces_dict, rcloud, isCollapse,
                        smooth_window=SMOOTH_WINDOW,
                        phase_change=SHOW_PHASE,
                    )
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                if i == 0:
                    eps = int(sfe) / 100.0
                    ax.set_title(rf"$\epsilon={eps:.2f}$")

                if j == 0:
                    set_mcloud_ylabel(
                        ax, mCloud,
                        extra=r"$p(t)=\int F\,\mathrm{d}t$ [$M_\odot\,\mathrm{pc}\,\mathrm{Myr}^{-1}$]",
                    )

                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")

        handles = []
        for _, lab, c in FORCE_FIELDS:
            handles.append(Line2D([0], [0], color=c, lw=1.6, ls="-", label=lab))
        handles.append(Line2D([0], [0], color="darkgrey", lw=2.4,
                              label=r"Net: $| \int (\sum F_{\rm out} - F_{\rm grav})\,dt |$"))
        handles.extend(get_marker_legend_handles(include_phase=SHOW_PHASE, include_rcloud=SHOW_RCLOUD, include_collapse=SHOW_COLLAPSE))

        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, handles,
            n_rows_for_layout=nrows,
            folder_name=folder_name,
            param_tag=param_tag,
            legend_ncol=4,
            legend_fontsize=7,
            suptitle=False,
        )

        save_grid_figure(
            fig, folder_name=folder_name,
            file_prefix=phii_file_prefix("momentum", phii_mode),
            param_tag=param_tag, output_dir=output_dir,
        )
        plt.close(fig)


# Backwards compatibility alias
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from paper.figures._lib.cli import dispatch, marker_pre_dispatch
    dispatch(
        script_name="paper_momentum.py",
        description="Plot TRINITY momentum",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
