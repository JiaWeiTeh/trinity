#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper I teaser figure (single fiducial run).

Three panels stacked vertically with a shared linear-time x-axis:

    top    bubble radius R_b (left, linear [pc]) and shell
           velocity v_sh (right, log [km/s]); both rendered in
           black, distinguished by line style and an in-panel
           legend
    middle feedback force-fraction decomposition with phase-aware
           overlays (rendered locally, not delegated to
           paper_feedback)
    bottom ionising-photon Q_i budget — stacked area showing the
           fraction of ionising photons absorbed by gas inside the
           shell, by dust inside the shell, and escaping past the
           shell, summing to unity at every timestep

The implicit phase is treated as part of the energy phase for
display, so only the energy↔transition and transition↔momentum
breaks register.  Vertical dotted grey lines mark these breaks
on the top two panels (panel (c)'s dark stack fill swallows
them, so they are skipped there); the active display-phase name
is printed above the top panel.

Unit handling
-------------
``v2`` is stored in [pc/Myr]; multiply by ``cvt.v_au2kms`` for km/s.

Run input
---------
The fiducial run path is supplied at the command line (positional
arg or ``-F``); this script does not assume a specific simulation
exists on disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR, smooth_2d
from src._output.trinity_reader import load_output, resolve_data_input
import src._functions.unit_conversions as cvt
from src._calc._common.plot_utils import C_BLACK

# paper_feedback supplies the data extraction (load_run); its panel
# renderer is *not* used — we draw the panel locally below so we
# control phase-aware overlays without paper_feedback's conditional
# ``non_bubble`` gating, and so the legend uses the F_*/HII/wind/SN
# nomenclature requested for this figure.
from src._plots import paper_feedback as _pf


# ---------------------------------------------------------------------------
# Colour assignments
# ---------------------------------------------------------------------------
# Panel (a): both R_b and v_sh are drawn black; the line style
# (solid vs dashed) plus an in-panel Line2D legend distinguishes
# them.
_C_TOP = C_BLACK

# Panel (c) sequential purple ramp (darkest = gas absorption)
_SHADE_GAS    = "#6c4a78"
_SHADE_DUST   = "#a98ec0"
_SHADE_ESCAPE = "#dccdec"

# Panel (b) two-tier palette:
#   structural / baseline forces are rendered in a monotonic
#   greyscale ramp (darkest at the bottom of the stack, white at
#   the top) so they read as a neutral backdrop, and the actual
#   feedback channels (HII, wind, SN) sit on top as translucent
#   tinted overlays.
# Base stack (greyscale, bottom→top: dark → light)
_C_GRAV  = "#1a1a1a"   # gravity            (near-black, ~10% lightness)
_C_DRIVE = "#a4a7aa"   # bubble-pressure base inside F_drive (~64%)
_C_RAD   = "#d4d6d8"   # radiation          (~83%)
_C_EXT   = "#ffffff"   # external photoionised (pure white)
# Tinted feedback overlays
_C_HII   = "#c0392b"   # warm red
_C_WIND  = "#1d3557"   # navy
_C_SN    = "#ef6c00"   # vivid orange
_TINT_ALPHA = 0.40

# Vertical phase-boundary lines.  zorder=10 keeps them above the
# stack fills (zorder=4) but below the legends (set_zorder(20)).
_PHASE_LINE_KW = dict(color="0.4", linestyle=":", linewidth=0.9, zorder=10)

_PHASE_LABEL = {
    "energy":     "energy",
    "transition": "transition",
    "momentum":   "momentum",
}

# The implicit phase is a numerical continuation of the energy phase;
# we suppress its internal boundary and label it as "energy" in the
# figure.  Mapping is applied only for boundary / annotation use,
# never to the data masks (which still reference the literal
# ``"energy"`` and ``"implicit"`` strings).
_DISPLAY_PHASE_MAP = {"implicit": "energy"}


def _display_phase(phase):
    """Collapse internal sub-phases for boundary/label placement."""
    p = np.asarray(phase).copy()
    for src, dst in _DISPLAY_PHASE_MAP.items():
        p = np.where(p == src, dst, p)
    return p


# ---------------------------------------------------------------------------
# Data loading (panels (a) and (c) — panel (b) reuses paper_feedback)
# ---------------------------------------------------------------------------
def _to_float_array(values):
    """Cast an iterable of (float | None | NaN) values to a float ndarray.

    The reader returns ``None`` for any snapshot missing a key, which
    breaks a direct ``np.asarray(..., dtype=float)`` cast.  Coerce
    those holes to NaN so downstream masking / gapping logic works.
    """
    return np.asarray(
        [np.nan if v is None else v for v in values],
        dtype=float,
    )


def load_run(data_path):
    """Return a dict of arrays for panels (a) and (c)."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t     = _to_float_array(output.get("t_now", as_array=False))
    phase = np.asarray(output.get("current_phase", as_array=False))
    R2    = _to_float_array(output.get("R2", as_array=False))         # [pc]
    v2_au = _to_float_array(output.get("v2", as_array=False))         # [pc/Myr]
    fAbs  = _to_float_array(output.get("shell_fAbsorbedIon",  as_array=False))
    fDust = _to_float_array(output.get("shell_fIonisedDust",  as_array=False))

    # Restore monotonic time ordering if the reader yielded snapshots
    # out of order — same guard as paper_LgainLloss.py.
    if np.any(np.diff(t) < 0):
        order = np.argsort(t)
        t     = t[order]
        phase = phase[order]
        R2    = R2[order]
        v2_au = v2_au[order]
        fAbs, fDust = fAbs[order], fDust[order]

    return dict(
        t=t, phase=phase,
        R2=R2, v_kms=v2_au * cvt.v_au2kms,
        fAbs=fAbs, fDust=fDust,
    )


# ---------------------------------------------------------------------------
# Phase utilities (rolled locally instead of reusing paper_bubblePhase)
# ---------------------------------------------------------------------------
def _change_points(arr):
    """Indices *i* where ``arr[i] != arr[i-1]``."""
    arr = np.asarray(arr)
    if arr.size <= 1:
        return np.array([], dtype=int)
    return np.where(arr[1:] != arr[:-1])[0] + 1


def _draw_phase_boundaries(axes, t, phase):
    bnd = _change_points(_display_phase(phase))
    for ax in axes:
        for i in bnd:
            ax.axvline(t[i], **_PHASE_LINE_KW)


def _annotate_phase_labels(ax_top, t, phase):
    """Place phase names above *ax_top*, centred over each phase's interval.

    Centre is the arithmetic mean of the segment endpoints; the
    x-axis is linear (see plot_from_path).
    """
    disp   = _display_phase(phase)
    bnd    = _change_points(disp)
    starts = np.concatenate([[0], bnd])
    ends   = np.concatenate([bnd, [len(t)]])
    for i0, i1 in zip(starts, ends):
        if i1 <= i0:
            continue
        seg = disp[i0]
        if seg not in _PHASE_LABEL:
            continue
        t_lo, t_hi = t[i0], t[i1 - 1]
        if not (np.isfinite(t_lo) and np.isfinite(t_hi)):
            continue
        xc = 0.5 * (t_lo + t_hi)
        ax_top.text(
            xc, 1.02, _PHASE_LABEL[seg],
            ha="center", va="bottom",
            transform=ax_top.get_xaxis_transform(),
            fontsize=10,
        )


# ---------------------------------------------------------------------------
# Panel (b) — feedback force-fraction decomposition (local renderer)
# ---------------------------------------------------------------------------
# Reproduces the spirit of paper_feedback's stack but with two
# customisations requested for the teaser figure:
#
#   1.  Base-stack labels are unified as F_grav / F_drive / F_rad /
#       F_ext (paper_feedback's "PISM" band actually plots
#       press_HII_in; see the comment further down).
#   2.  Only the momentum phase carries the wind / SN / HII
#       overlay.  The energy and transition phases render the
#       F_drive band as solid grey, since the bubble-driven and
#       transition regimes aren't usefully decomposed here.
#
# Smoothing window matches paper_feedback's default (21).
_FB_SMOOTH = 21


def _draw_ram_overlay(ax, t_seg, db, y_wind_top, y_sn_top):
    """Wind + SN translucent slices.

    Fills sit at ``zorder=5`` — *above* the base stack at zorder=4
    so the tint is not painted over by the opaque grey F_drive
    band, and below the phase-boundary lines at zorder=10.
    """
    ax.fill_between(t_seg, db, y_wind_top,
                    facecolor=_C_WIND, alpha=_TINT_ALPHA,
                    edgecolor="none", zorder=5)
    ax.fill_between(t_seg, y_wind_top, y_sn_top,
                    facecolor=_C_SN, alpha=_TINT_ALPHA,
                    edgecolor="none", zorder=5)


def _draw_hii_overlay(ax, t_seg, y_sn_top, y_hii_top):
    """HII translucent slice on top of wind+SN (momentum only)."""
    ax.fill_between(t_seg, y_sn_top, y_hii_top,
                    facecolor=_C_HII, alpha=_TINT_ALPHA,
                    edgecolor="none", zorder=5)


def _plot_feedback_panel(ax, t, phase, base_forces, overlay_forces):
    """Draw the feedback force-fraction panel onto *ax*.

    ``base_forces`` is the (4, N) array (F_grav, F_drive, F_rad,
    F_ext) returned by ``paper_feedback.load_run``; ``overlay_forces``
    is (3, N) = (F_HII, F_wind, F_SN).
    """
    F_HII_raw, F_w_raw, F_s_raw = overlay_forces

    # Stack base-fraction (smooth after normalising for stable widths)
    ftotal   = base_forces.sum(axis=0)
    ftotal   = np.where(ftotal == 0.0, np.nan, ftotal)
    frac_raw = base_forces / ftotal
    frac     = smooth_2d(frac_raw, _FB_SMOOTH, mode="edge")
    cum      = np.cumsum(frac, axis=0)
    prev     = np.vstack([np.zeros_like(t), cum[:-1]])

    base_styles = [
        (_C_GRAV,  1.00),
        (_C_DRIVE, 1.00),
        (_C_RAD,   1.00),
        (_C_EXT,   1.00),
    ]
    for (color, alpha), y0, y1 in zip(base_styles, prev, cum):
        ax.fill_between(t, y0, y1, facecolor=color, alpha=alpha,
                        edgecolor="none", zorder=4)

    # F_drive band (the band the overlays live inside)
    drive_bottom = prev[1]
    drive_top    = cum[1]
    drive_h      = drive_top - drive_bottom

    F_HII = np.nan_to_num(F_HII_raw, nan=0.0)
    F_w   = np.nan_to_num(F_w_raw,   nan=0.0)
    F_s   = np.nan_to_num(F_s_raw,   nan=0.0)

    # ---- Momentum phase: wind + SN + HII ----------------------------------
    # Transition phase deliberately renders solid F_drive only — the
    # wind/SN ram overlay is reserved for the momentum phase, where
    # the bubble has been replaced by direct momentum injection.
    momentum = (np.asarray(phase) == "momentum")
    if np.any(momentum):
        idx   = np.where(momentum)[0]
        Ftot  = F_w[idx] + F_s[idx] + F_HII[idx]
        denom = np.where(Ftot > 0, Ftot, np.nan)
        f_w = np.nan_to_num(F_w[idx]   / denom, nan=0.0)
        f_s = np.nan_to_num(F_s[idx]   / denom, nan=0.0)
        f_h = np.nan_to_num(F_HII[idx] / denom, nan=0.0)
        s = f_w + f_s + f_h
        over = s > 1.0
        f_w[over] /= s[over]
        f_s[over] /= s[over]
        f_h[over] /= s[over]

        db = drive_bottom[idx]
        dh = drive_h[idx]
        y_wind_top = db + f_w * dh
        y_sn_top   = y_wind_top + f_s * dh
        y_hii_top  = y_sn_top + f_h * dh
        _draw_ram_overlay(ax, t[idx], db, y_wind_top, y_sn_top)
        _draw_hii_overlay(ax, t[idx], y_sn_top, y_hii_top)

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(t.min(), t.max())


# ---------------------------------------------------------------------------
# Panel (c) decomposition
# ---------------------------------------------------------------------------
def _ionising_components(fAbs, fDust):
    """Return (gas, dust, escape) stack components.

    - Both fields finite: gas = fAbs*(1-fDust); dust = fAbs*fDust;
      escape = 1 - fAbs.
    - fDust NaN, fAbs finite (post-dissolution path in
      shell_structure_modified.py:411): force dust=0 and gas=fAbs.
    - fAbs NaN: every component stays NaN so the stack is gapped.

    Components sum to 1 within float epsilon at every non-gapped step.
    """
    gas    = np.full_like(fAbs, np.nan, dtype=float)
    dust   = np.full_like(fAbs, np.nan, dtype=float)
    escape = np.full_like(fAbs, np.nan, dtype=float)

    abs_finite  = np.isfinite(fAbs)
    dust_finite = abs_finite & np.isfinite(fDust)
    abs_only    = abs_finite & ~np.isfinite(fDust)

    gas   [dust_finite] = fAbs[dust_finite] * (1.0 - fDust[dust_finite])
    dust  [dust_finite] = fAbs[dust_finite] * fDust[dust_finite]
    escape[dust_finite] = 1.0 - fAbs[dust_finite]

    gas   [abs_only] = fAbs[abs_only]
    dust  [abs_only] = 0.0
    escape[abs_only] = 1.0 - fAbs[abs_only]

    return gas, dust, escape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_from_path(data_input, output_dir=None):
    data_path = Path(resolve_data_input(data_input))
    run = load_run(data_path)

    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True,
        figsize=(4.0, 6.5),
        gridspec_kw=dict(hspace=0.05),
    )
    ax_a, ax_b, ax_c = axes

    # ---- panel (a) — R_b solid black (left), v_sh dashed black (right) -----
    # Both curves rendered in black; a Line2D legend in the upper-
    # left disambiguates them (same idiom as paper_densityProfile).
    ax_a.plot(run["t"], run["R2"], color=_C_TOP, lw=1.5, ls="-")
    ax_a.set_ylabel(r"$R_{\rm b}\ [{\rm pc}]$")

    ax_av = ax_a.twinx()
    ax_av.plot(run["t"], run["v_kms"], color=_C_TOP, lw=1.5, ls="--")
    ax_av.set_yscale("log")
    ax_av.set_ylabel(r"$v_{\rm b}\ [{\rm km\ s^{-1}}]$")

    top_handles = [
        Line2D([0], [0], color=_C_TOP, ls="-",  lw=1.5, label=r"$R_{\rm b}$"),
        Line2D([0], [0], color=_C_TOP, ls="--", lw=1.5, label=r"$v_{\rm b}$"),
    ]
    leg_a = ax_a.legend(
        handles=top_handles, loc="upper left", frameon=False,
        fontsize=10, handlelength=1.6, labelspacing=0.3,
    )
    leg_a.set_zorder(20)

    # ---- panel (b) — feedback decomposition (local renderer) --------------
    # paper_feedback.load_run is reused to extract base / overlay
    # arrays; the panel itself is drawn locally so we control the
    # transition-phase logic (no ``non_bubble`` gate; HII suppressed
    # in transition; consistent dotted slice outlines).
    fb_t, _fb_R2, fb_phase, fb_base, fb_overlay, _fb_rc, _fb_ic, _fb_pr = (
        _pf.load_run(data_path)
    )
    _plot_feedback_panel(ax_b, fb_t, fb_phase, fb_base, fb_overlay)
    ax_b.set_ylabel(r"$F/F_{\rm tot}$")

    # The fourth base band ("F_ext") plots the snapshot field
    # ``press_HII_in``, which is dominated by the photo-ionised
    # inflow pressure 2 n_r k_B T_ion (set in run_*_phase_modified
    # once shell_fAbsorbedIon < 1) and only adds an additive
    # ``PISM * k_B`` term once the shell escapes the cloud.
    fb_handles = [
        Patch(facecolor=_C_GRAV,  edgecolor="black", linewidth=0.4,
              label=r"$F_{\rm grav}$"),
        Patch(facecolor=_C_DRIVE, edgecolor="black", linewidth=0.4,
              label=r"$F_{\rm drive}$"),
        Patch(facecolor=_C_RAD,   edgecolor="black", linewidth=0.4,
              label=r"$F_{\rm rad}$"),
        Patch(facecolor=_C_EXT,   edgecolor="black", linewidth=0.4,
              label=r"$F_{\rm ext}$"),
        Patch(facecolor=_C_HII,   alpha=_TINT_ALPHA,
              edgecolor="black", linewidth=0.4, label="HII"),
        Patch(facecolor=_C_WIND,  alpha=_TINT_ALPHA,
              edgecolor="black", linewidth=0.4, label="wind"),
        Patch(facecolor=_C_SN,    alpha=_TINT_ALPHA,
              edgecolor="black", linewidth=0.4, label="SN"),
    ]
    leg_b = ax_b.legend(
        handles=fb_handles, loc="upper right", frameon=True,
        fontsize=8, ncol=2, handlelength=1.2, columnspacing=0.8,
        labelspacing=0.3, framealpha=0.85, edgecolor="0.3",
    )
    # Lift the legend above the dotted phase-boundary lines (zorder=10)
    # so the lines do not show through the box.
    leg_b.set_zorder(20)

    # ---- panel (c) — ionising-photon budget --------------------------------
    gas, dust, escape = _ionising_components(run["fAbs"], run["fDust"])
    ax_c.stackplot(
        run["t"], gas, dust, escape,
        colors=[_SHADE_GAS, _SHADE_DUST, _SHADE_ESCAPE],
        labels=[
            r"$f_{\rm abs}^{\rm gas}$",
            r"$f_{\rm abs}^{\rm dust}$",
            r"$f_{\rm esc}^{\rm LyC}$",
        ],
        edgecolor="none",
    )
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel(r"$Q_{\rm i}$ budget")
    # Top-right boxed legend (the bottom-right of this panel sits in
    # the darkest gas-absorption fill, where text in any colour is
    # hard to read).
    leg_c = ax_c.legend(
        loc="upper right", frameon=True, fontsize=10, ncol=3,
        handlelength=1.0, columnspacing=0.8,
        framealpha=0.95, edgecolor="0.3",
    )
    leg_c.set_zorder(20)

    # ---- shared x-axis, linear t (only bottom panel labels ticks) ----------
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_c.set_xlabel(r"$t\ [{\rm Myr}]$")
    finite_t = run["t"][np.isfinite(run["t"])]
    if finite_t.size > 0:
        ax_c.set_xlim(finite_t.min(), finite_t.max())

    # ---- explicit y-tick placements (avoid boundary-edge labels) -----------
    # Top panel: R_b every 50 pc, skipping 0 (the panel-bottom border)
    # and the axis cap.  v_sh stays on the default log locator.
    ax_a.set_yticks([50, 100, 150, 200, 250])
    # Middle and bottom panels: identical fraction ticks so the eye reads
    # them as a paired stack.
    ax_b.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # ---- phase boundaries + top-panel labels -------------------------------
    # Boundary lines on the top two panels only — panel (c)'s dark
    # stack fill swallows them anyway, and the panel-(a) tints alone
    # don't make the transition obvious enough.
    _draw_phase_boundaries([ax_a, ax_av, ax_b], run["t"], run["phase"])
    _annotate_phase_labels(ax_a, run["t"], run["phase"])

    # ---- save --------------------------------------------------------------
    out_dir = Path(output_dir) if output_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "teaser_fiducial.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    plt.close(fig)


def plot_grid(folder_path, output_dir=None,
              ndens_filter=None, mCloud_filter=None, sfe_filter=None):
    """Single-sim teaser; ``-F`` is treated as a single run directory.

    Filters are accepted for CLI compatibility but ignored.
    """
    plot_from_path(folder_path, output_dir)


if __name__ == "__main__":
    from src._plots.cli import dispatch
    dispatch(
        script_name="paper_teaser.py",
        description="Paper I teaser figure: bubble dynamics, feedback "
                    "decomposition, and ionising-photon budget.",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
