#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refined feedback force-fraction plot (renewed version).

Stacks force fractions F_i / F_total over time, with a per-phase
band composition that respects the max() driver-selection used in
the model:

  Energy phase:
      F_drive = max(Pb, P_HII) * 4 pi R^2
      Bands (bottom -> top): gravity, F_drive (fused), radiation
  Transition phase:
      F_drive = max(Pb, P_HII + P_ram) * 4 pi R^2
      Bands: gravity, F_drive (fused), radiation
  Momentum phase:
      F_drive = (P_HII + P_ram) * 4 pi R^2 (bubble is gone)
      Bands: gravity, wind ram, P_HII, SN ram, radiation

In all phases F_total = F_grav + F_drive + F_rad, so the gravity and
radiation band heights are continuous across phase boundaries.  Only
the inside of F_drive changes (one fused band -> three components at
transition -> momentum).

A pale phase tint marks each phase region (energy = blue, transition
= amber, momentum = red).

Note (PISM): the inner-HII pressure (`press_HII_in`) is not shown
here.  It could be added in the future as an additional band -- see
the loader for the field name.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Bootstrap project root so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src._plots.plot_base import FIG_DIR, smooth_1d  # noqa: E402
from src._output.trinity_reader import load_output, resolve_data_input  # noqa: E402
from src._plots.plot_markers import (  # noqa: E402
    add_plot_markers, get_marker_legend_handles,
)
from src._plots.grid_template import (  # noqa: E402
    build_param_tag, iter_grid_densities, mark_missing_cell,
    attach_grid_legend, save_grid_figure, set_mcloud_ylabel, _sfe_title,
)


# ---------------- configuration ----------------
SMOOTH_WINDOW = 21      # None or 1 disables smoothing
SHOW_PHASE    = False
SHOW_RCLOUD   = False
SHOW_COLLAPSE = False
USE_LOG_X     = False
SAVE_PDF      = True


# ---------------- palette ----------------
# Local palette for this plot only.  Colours chosen to be physically
# motivated rather than sampled from a continuous colormap (forces are
# categorical, not ordered).
COLOR_GRAV  = "black"
COLOR_RAD   = "mediumpurple"
COLOR_WIND  = "steelblue"
COLOR_PHII  = "crimson"
COLOR_SN    = "darkorange"
COLOR_DRIVE = "#9b8186"   # warm grey-mauve: fused F_drive band (energy + transition)

# Very pale phase background tints.
TINT_ENERGY     = "#dde7f2"
TINT_TRANSITION = "#f4ecd6"
TINT_MOMENTUM   = "#f2d9d9"
TINT_ALPHA      = 0.30    # light enough not to fight the foreground stack


# ---------------- phase helpers ----------------
def _is_energy(p):
    return p in ("energy", "energy_implicit", "implicit")


def _phase_mask(phase, kind):
    """Boolean mask over an array of phase strings for the given kind."""
    if kind == "energy":
        return np.array([_is_energy(p) for p in phase])
    if kind == "transition":
        return np.array([p == "transition" for p in phase])
    if kind == "momentum":
        return np.array([p == "momentum" for p in phase])
    raise ValueError(f"unknown phase kind: {kind}")


def _phase_segments(t, phase):
    """Yield (kind, t0, t1) tuples for each contiguous phase region.

    `kind` is one of 'energy', 'transition', 'momentum', or None for
    timesteps whose phase string falls in none of those categories.
    """
    if len(t) == 0:
        return
    kinds = []
    for p in phase:
        if _is_energy(p):
            kinds.append("energy")
        elif p == "transition":
            kinds.append("transition")
        elif p == "momentum":
            kinds.append("momentum")
        else:
            kinds.append(None)
    start = 0
    for i in range(1, len(kinds)):
        if kinds[i] != kinds[start]:
            yield kinds[start], t[start], t[i]
            start = i
    yield kinds[start], t[start], t[-1]


# ---------------- loader ----------------
def load_run(data_path):
    """Load a run and return a dict with all fields the plotter needs."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots in {data_path}")

    def _f(name, default=0.0):
        arr = output.get(name)
        if arr is None:
            return np.full(len(output), default, dtype=float)
        # The reader returns object arrays when fields are sometimes None;
        # coerce to float, replacing None with the default.
        return np.where(arr == None, default, arr).astype(float)  # noqa: E711

    t     = output.get("t_now").astype(float)
    R2    = output.get("R2").astype(float)
    phase = np.array(output.get("current_phase", as_array=False))

    F_grav = _f("F_grav")
    F_rad  = _f("F_rad")

    # Pressures -> forces via F = P * 4 pi R^2.
    R2_safe = np.nan_to_num(R2, nan=0.0)
    sphere  = 4.0 * np.pi * R2_safe ** 2
    P_drive = _f("P_drive")
    P_HII   = _f("P_HII")
    F_drive = P_drive * sphere
    F_HII   = P_HII * sphere

    F_wind = _f("F_ram_wind")
    F_SN   = _f("F_ram_SN")

    # PISM: not shown in this version (commented in module docstring).
    # F_PISM = _f("press_HII_in") * sphere   # for future use

    rcloud = float(output[0].get("rCloud", np.nan))
    isCollapse = np.array(output.get("isCollapse", as_array=False))

    # Make sure time is monotone increasing.
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t = t[order]; R2 = R2[order]; phase = phase[order]
        F_grav = F_grav[order]; F_rad = F_rad[order]
        F_drive = F_drive[order]; F_HII = F_HII[order]
        F_wind = F_wind[order]; F_SN = F_SN[order]
        isCollapse = isCollapse[order]

    return {
        "t": t, "R2": R2, "phase": phase,
        "F_grav": F_grav, "F_rad": F_rad, "F_drive": F_drive,
        "F_HII": F_HII, "F_wind": F_wind, "F_SN": F_SN,
        "rcloud": rcloud, "isCollapse": isCollapse,
    }


# ---------------- per-phase smoothing ----------------
def _smooth_global(x, window):
    return smooth_1d(x, window) if window and window > 1 else x


def _smooth_within_mask(x, mask, window):
    """Smooth `x` only across the True region of `mask`; leave the rest alone.

    Avoids contaminating the masked region with the (typically zero) values
    outside it that come from a phase where the component does not exist.
    """
    if window is None or window <= 1 or not np.any(mask):
        return x
    out = x.copy()
    out[mask] = smooth_1d(x[mask], window)
    return out


# ---------------- core plotting ----------------
def plot_run_on_ax(ax, data, *, smooth_window=None,
                   phase_change=False, show_rcloud=False, show_collapse=False,
                   use_log_x=False):
    t      = data["t"]
    R2     = data["R2"]
    phase  = data["phase"]
    F_grav = data["F_grav"]
    F_rad  = data["F_rad"]
    F_drv  = data["F_drive"]
    F_HII  = data["F_HII"]
    F_w    = data["F_wind"]
    F_SN   = data["F_SN"]

    # --- Phase tints (drawn first so the stack sits on top) ---
    for kind, t0, t1 in _phase_segments(t, phase):
        if kind == "energy":
            ax.axvspan(t0, t1, color=TINT_ENERGY, alpha=TINT_ALPHA, lw=0, zorder=0)
        elif kind == "transition":
            ax.axvspan(t0, t1, color=TINT_TRANSITION, alpha=TINT_ALPHA, lw=0, zorder=0)
        elif kind == "momentum":
            ax.axvspan(t0, t1, color=TINT_MOMENTUM, alpha=TINT_ALPHA, lw=0, zorder=0)

    # --- Smoothing on raw forces ---
    # F_grav, F_rad, F_drv are defined in every phase -> smooth globally.
    # F_HII, F_w, F_SN are only meaningful in the momentum phase
    # (they are absorbed into F_drv elsewhere) -> smooth only within
    # the momentum mask to avoid bleeding zeros into the momentum region.
    mom_mask = _phase_mask(phase, "momentum")
    F_grav_s = _smooth_global(F_grav, smooth_window)
    F_rad_s  = _smooth_global(F_rad,  smooth_window)
    F_drv_s  = _smooth_global(F_drv,  smooth_window)
    F_HII_s  = _smooth_within_mask(F_HII, mom_mask, smooth_window)
    F_w_s    = _smooth_within_mask(F_w,   mom_mask, smooth_window)
    F_SN_s   = _smooth_within_mask(F_SN,  mom_mask, smooth_window)

    # --- F_total = F_grav + F_drive + F_rad (constant denominator across phases). ---
    F_total = F_grav_s + F_drv_s + F_rad_s
    F_total = np.where(F_total > 0, F_total, np.nan)

    # Fractional contributions of the always-on forces.
    f_grav = F_grav_s / F_total
    f_rad  = F_rad_s  / F_total
    f_drv  = F_drv_s  / F_total    # full F_drive band

    # In momentum, decompose F_drive into wind / P_HII / SN proportionally
    # so the three sub-bands sum to f_drv.  Outside momentum, leave them
    # at zero (the fused f_drv band is drawn instead).
    sub_total = F_w_s + F_HII_s + F_SN_s
    sub_safe  = np.where(sub_total > 0, sub_total, np.nan)
    f_wind   = np.where(mom_mask, F_w_s   / sub_safe * f_drv, 0.0)
    f_phii   = np.where(mom_mask, F_HII_s / sub_safe * f_drv, 0.0)
    f_sn     = np.where(mom_mask, F_SN_s  / sub_safe * f_drv, 0.0)
    f_wind   = np.nan_to_num(f_wind, nan=0.0)
    f_phii   = np.nan_to_num(f_phii, nan=0.0)
    f_sn     = np.nan_to_num(f_sn,   nan=0.0)

    # f_drv_fused: shown only outside momentum.  In momentum, the same height
    # is occupied by f_wind + f_phii + f_sn, so set f_drv_fused = 0 there.
    f_drv_fused = np.where(mom_mask, 0.0, f_drv)

    # --- Stack from bottom -> top in band-fraction space.
    # Order: gravity, [fused F_drive | (wind, P_HII, SN)], radiation.
    y0 = np.zeros_like(t)
    y1 = y0 + f_grav                 # gravity top
    y2 = y1 + f_drv_fused            # fused F_drive top (energy+transition only)
    y3 = y2 + f_wind                 # wind top (momentum only; else == y2)
    y4 = y3 + f_phii                 # P_HII top (momentum only; else == y3)
    y5 = y4 + f_sn                   # SN top (momentum only; else == y4)
    y6 = y5 + f_rad                  # radiation top -> should be ~1.0

    z = 4
    ax.fill_between(t, y0, y1, facecolor=COLOR_GRAV,  alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)
    ax.fill_between(t, y1, y2, facecolor=COLOR_DRIVE, alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)
    ax.fill_between(t, y2, y3, facecolor=COLOR_WIND,  alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)
    ax.fill_between(t, y3, y4, facecolor=COLOR_PHII,  alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)
    ax.fill_between(t, y4, y5, facecolor=COLOR_SN,    alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)
    ax.fill_between(t, y5, y6, facecolor=COLOR_RAD,   alpha=0.85,
                    edgecolor="black", linewidth=0.4, zorder=z)

    # --- Markers ---
    add_plot_markers(
        ax, t,
        phase=phase if phase_change else None,
        R2=R2 if show_rcloud else None,
        rcloud=data["rcloud"] if show_rcloud else None,
        isCollapse=data["isCollapse"] if show_collapse else None,
        show_phase=phase_change,
        show_rcloud=show_rcloud,
        show_collapse=show_collapse,
    )

    ax.set_ylim(0, 1)
    if use_log_x:
        ax.set_xscale("log")
        t_pos = t[t > 0]
        if len(t_pos):
            ax.set_xlim(t_pos.min(), t.max())
    else:
        ax.set_xlim(t.min(), t.max())


# ---------------- legend ----------------
def build_legend_handles(*, include_phase=False, include_rcloud=False,
                         include_collapse=False):
    handles = [
        Patch(facecolor=COLOR_GRAV,  edgecolor="black", lw=0.4, alpha=0.85,
              label="Gravity"),
        Patch(facecolor=COLOR_DRIVE, edgecolor="black", lw=0.4, alpha=0.85,
              label=r"$F_{\rm drive}$ (energy + transition)"),
        Patch(facecolor=COLOR_WIND,  edgecolor="black", lw=0.4, alpha=0.85,
              label="Wind ram (momentum)"),
        Patch(facecolor=COLOR_PHII,  edgecolor="black", lw=0.4, alpha=0.85,
              label=r"$P_{\rm HII}$ (momentum)"),
        Patch(facecolor=COLOR_SN,    edgecolor="black", lw=0.4, alpha=0.85,
              label="SN ram (momentum)"),
        Patch(facecolor=COLOR_RAD,   edgecolor="black", lw=0.4, alpha=0.85,
              label="Radiation"),
        Patch(facecolor=TINT_ENERGY,     edgecolor="none", alpha=TINT_ALPHA,
              label="Energy phase"),
        Patch(facecolor=TINT_TRANSITION, edgecolor="none", alpha=TINT_ALPHA,
              label="Transition phase"),
        Patch(facecolor=TINT_MOMENTUM,   edgecolor="none", alpha=TINT_ALPHA,
              label="Momentum phase"),
    ]
    handles.extend(get_marker_legend_handles(
        include_phase=include_phase,
        include_rcloud=include_rcloud,
        include_collapse=include_collapse,
    ))
    return handles


# ---------------- single-run entry ----------------
def plot_from_path(data_input, output_dir=None):
    try:
        data_path = resolve_data_input(data_input, output_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    try:
        data = load_run(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    plot_run_on_ax(
        ax, data,
        smooth_window=SMOOTH_WINDOW,
        phase_change=SHOW_PHASE,
        show_rcloud=SHOW_RCLOUD,
        show_collapse=SHOW_COLLAPSE,
        use_log_x=USE_LOG_X,
    )

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$F_i / F_{\rm tot}$")
    ax.set_title(f"Feedback fractions: {data_path.parent.name}")

    handles = build_legend_handles(
        include_phase=SHOW_PHASE,
        include_rcloud=SHOW_RCLOUD,
        include_collapse=SHOW_COLLAPSE,
    )
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()

    run_name = data_path.parent.name
    parent_folder = data_path.parent.parent.name
    fig_dir = FIG_DIR / parent_folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"feedback_new_{run_name}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")

    plt.show()
    plt.close(fig)


# ---------------- grid entry ----------------
def plot_grid(folder_path, output_dir=None,
              ndens_filter=None, mCloud_filter=None, sfe_filter=None):
    for ndens, mCloud_list, sfe_list, grid, folder_name in iter_grid_densities(
            folder_path, ndens_filter=ndens_filter,
            mCloud_filter=mCloud_filter, sfe_filter=sfe_filter):

        nrows, ncols = len(mCloud_list), len(sfe_list)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(3.2 * ncols, 2.6 * nrows),
            sharex=False, sharey=True,
            dpi=500, squeeze=False, constrained_layout=False,
        )

        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                ax = axes[i, j]
                data_path = grid.get((mCloud, sfe))

                if data_path is None:
                    mark_missing_cell(ax, "missing")
                    continue

                try:
                    data = load_run(data_path)
                    plot_run_on_ax(
                        ax, data,
                        smooth_window=SMOOTH_WINDOW,
                        phase_change=SHOW_PHASE,
                        show_rcloud=SHOW_RCLOUD,
                        show_collapse=SHOW_COLLAPSE,
                        use_log_x=USE_LOG_X,
                    )
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    mark_missing_cell(ax, "error")
                    continue

                ax.tick_params(axis="x", which="both", bottom=True)
                if i == nrows - 1:
                    ax.set_xlabel("t [Myr]")
                if i == 0:
                    ax.set_title(_sfe_title(sfe))
                if j == 0:
                    set_mcloud_ylabel(ax, mCloud, extra=r"$F_i / F_{\rm tot}$")
                else:
                    ax.tick_params(labelleft=False)

        handles = build_legend_handles(
            include_phase=SHOW_PHASE,
            include_rcloud=SHOW_RCLOUD,
            include_collapse=SHOW_COLLAPSE,
        )
        param_tag = build_param_tag(mCloud_list, sfe_list, ndens)
        attach_grid_legend(
            fig, handles, n_rows_for_layout=nrows,
            folder_name=folder_name, param_tag=param_tag,
            legend_ncol=4, legend_fontsize=7,
            suptitle=False,
        )
        save_grid_figure(
            fig, folder_name=folder_name,
            file_prefix="feedback_new", param_tag=param_tag,
            output_dir=output_dir, save_pdf=SAVE_PDF,
        )
        plt.close(fig)


# Backwards-compatibility alias for older callers.
plot_folder_grid = plot_grid


if __name__ == "__main__":
    from src._plots.cli import dispatch, marker_pre_dispatch
    dispatch(
        script_name="paper_feedback_new.py",
        description="Plot TRINITY feedback force fractions (renewed)",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
        pre_dispatch_fn=marker_pre_dispatch(globals()),
    )
