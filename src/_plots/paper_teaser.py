#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper I teaser figure (single fiducial run).

Three panels stacked vertically with a shared linear-time x-axis:

    top    bubble radius R_b (left, linear [pc]) and shell velocity
           v_sh (right, log [km/s])
    middle feedback force-fraction decomposition with phase-aware
           overlays (delegated to paper_feedback.plot_run_on_ax)
    bottom ionising-photon Q_i budget — stacked area showing the
           fraction of ionising photons absorbed by gas inside the
           shell, by dust inside the shell, and escaping past the
           shell, summing to unity at every timestep

Vertical dotted grey lines mark phase boundaries across all panels.
The implicit phase is treated as part of the energy phase for
display, so only the energy↔transition and transition↔momentum
boundaries appear; the active display-phase name is printed above
the top panel.

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
from matplotlib.ticker import MaxNLocator
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR
from src._output.trinity_reader import load_output, resolve_data_input
import src._functions.unit_conversions as cvt
from src._calc._common.plot_utils import C_BLACK, C_VERMILLION
from src._plots.force_colors import C as _FC

# Re-use the feedback-decomposition machinery wholesale; teaser panel
# (b) is the same plot as paper_feedback's, just dropped into our
# multi-panel figure.
from src._plots import paper_feedback as _pf


# ---------------------------------------------------------------------------
# Colour assignments
# ---------------------------------------------------------------------------
_C_R = C_BLACK        # panel (a) bubble radius
_C_V = C_VERMILLION   # panel (a) shell velocity

# Panel (c) sequential purple ramp (darkest = gas absorption)
_SHADE_GAS    = "#6c4a78"
_SHADE_DUST   = "#a98ec0"
_SHADE_ESCAPE = "#dccdec"

_PHASE_LINE_KW = dict(color="0.6", linestyle=":", linewidth=0.8, zorder=0)

_PHASE_LABEL = {
    "energy":     r"\textsc{energy}",
    "transition": r"\textsc{transition}",
    "momentum":   r"\textsc{momentum}",
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

    # ---- panel (a) — R_b linear (left), v_sh log (right) -------------------
    ax_a.plot(run["t"], run["R2"], color=_C_R, lw=1.5)
    ax_a.set_ylabel(r"$R_{\rm b}\ [{\rm pc}]$", color=_C_R)
    ax_a.tick_params(axis="y", colors=_C_R)

    ax_av = ax_a.twinx()
    ax_av.plot(run["t"], run["v_kms"], color=_C_V, lw=1.5)
    ax_av.set_yscale("log")
    ax_av.set_ylabel(r"$v_{\rm sh}\ [{\rm km\ s^{-1}}]$", color=_C_V)
    ax_av.tick_params(axis="y", colors=_C_V)

    # ---- panel (b) — feedback decomposition (delegated) --------------------
    # Uses paper_feedback.load_run + plot_run_on_ax verbatim so the
    # teaser panel stays in lock-step with the standalone figure.
    fb_t, fb_R2, fb_phase, fb_base, fb_overlay, fb_rcloud, fb_iscoll, fb_press = (
        _pf.load_run(data_path)
    )
    _pf.plot_run_on_ax(
        ax_b, fb_t, fb_R2, fb_phase, fb_base, fb_overlay, fb_rcloud, fb_iscoll,
        pressures=fb_press,
        smooth_window=_pf.SMOOTH_WINDOW,
        phase_change=False, show_rcloud=False, show_collapse=False,
        use_log_x=False,
    )
    ax_b.set_ylabel(r"$F/F_{\rm tot}$")
    # The band paper_feedback labels "PISM" actually plots the
    # ``press_HII_in`` field, which is dominated by the photo-ionised
    # gas pressure outside the shell (P_ext = 2 n_r k_B T_ion, set in
    # run_*_phase_modified.py once shell_fAbsorbedIon drops below 1)
    # and only includes the input ``PISM`` parameter as an additive
    # ``PISM * k_B`` once rShell ≥ rCloud.  Relabel here so the teaser
    # legend is unambiguous.
    fb_handles = [
        Patch(facecolor=_FC.GRAV,  edgecolor="none", alpha=0.75, label="Gravity"),
        Patch(facecolor=_FC.DRIVE, edgecolor="none", alpha=0.75,
              label=r"$F_{\rm drive}$"),
        Patch(facecolor=_FC.RAD,   edgecolor="none", alpha=0.75, label="Radiation"),
        Patch(facecolor=_FC.PISM,  edgecolor="0.3",  linewidth=0.8,
              label=r"$P_{\rm ext}$"),
        Patch(facecolor="none", edgecolor=_FC.PHII, hatch="......",
              label=r"$P_{\rm HII}$"),
        Patch(facecolor="none", edgecolor=_FC.WIND, hatch="\\\\\\\\", label="Wind"),
        Patch(facecolor="none", edgecolor=_FC.SN,   hatch="////",     label="SN"),
    ]
    ax_b.legend(
        handles=fb_handles, loc="upper right", frameon=False,
        fontsize=8, ncol=2, handlelength=1.2, columnspacing=0.8,
        labelspacing=0.3,
    )

    # ---- panel (c) — ionising-photon budget --------------------------------
    gas, dust, escape = _ionising_components(run["fAbs"], run["fDust"])
    ax_c.stackplot(
        run["t"], gas, dust, escape,
        colors=[_SHADE_GAS, _SHADE_DUST, _SHADE_ESCAPE],
        labels=["gas", "dust", "escape"],
        edgecolor="none",
    )
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_ylabel(r"$Q_{\rm i}$ budget")
    # Top-right boxed legend (the bottom-right of this panel sits in
    # the darkest gas-absorption fill, where text in any colour is
    # hard to read).
    ax_c.legend(
        loc="upper right", frameon=True, fontsize=10, ncol=3,
        handlelength=1.0, columnspacing=0.8,
        framealpha=0.9, edgecolor="0.3",
    )

    # ---- shared x-axis, linear t (only bottom panel labels ticks) ----------
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_c.set_xlabel(r"$t\ [{\rm Myr}]$")
    finite_t = run["t"][np.isfinite(run["t"])]
    if finite_t.size > 0:
        ax_c.set_xlim(finite_t.min(), finite_t.max())

    # ---- prune top/bottom y-tick labels at panel boundaries ----------------
    # With ``hspace=0.05`` the boundary tick of one panel sits on top
    # of the boundary tick of the next.  Remove the outermost labels
    # on each axis so they do not collide.  ax_av (twinx) and ax_a are
    # both treated.
    ax_a.yaxis.set_major_locator(MaxNLocator(prune="lower"))
    ax_av.yaxis.set_major_locator(MaxNLocator(prune="lower"))
    # Panel (b) is sandwiched between (a) and (c); explicit interior
    # ticks avoid the default ``[0, 0.5, 1.0]`` whose endpoints both
    # land on a panel boundary.
    ax_b.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_c.yaxis.set_major_locator(MaxNLocator(prune="upper"))

    # ---- phase boundaries + top-panel labels -------------------------------
    _draw_phase_boundaries(list(axes) + [ax_av], run["t"], run["phase"])
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
