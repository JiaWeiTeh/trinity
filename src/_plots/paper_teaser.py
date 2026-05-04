#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper I teaser figure (single fiducial run).

Four panels stacked vertically with shared log-time x-axis:

    (a) bubble radius R_b (left, log [pc]) and shell velocity v_sh
        (right, linear [km/s])
    (b) swept-up shell mass [Msun, log] with reference line at the
        gas-reservoir mass M_cloud(1 - sfe)
    (c) bubble energy budget (Lmech, Lgain, Lloss [erg/s, log]),
        masked to the energy + implicit phases where the budget is
        physically meaningful
    (d) ionising-photon budget (gas / dust / escape) as a stacked
        area summing to unity

Vertical dotted grey lines mark phase boundaries across all panels;
the active phase name is printed above the top panel.

Unit handling
-------------
``Lmech_total``, ``bubble_Lgain``, ``bubble_Lloss`` are stored in
TRINITY astronomy units [Msun*pc**2/Myr**3]; here we multiply by
``cvt.L_au2cgs`` to plot in erg/s (matches paper_LgainLloss.py).
``v2`` is stored in [pc/Myr]; multiply by ``cvt.v_au2kms`` for km/s.

Run input
---------
The fiducial run path is supplied at the command line (positional
arg or ``-F``); this script does not assume a specific simulation
exists on disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

from src._plots.plot_base import FIG_DIR
from src._output.trinity_reader import load_output, resolve_data_input
import src._functions.unit_conversions as cvt
from src._calc._common.plot_utils import C_BLACK, C_BLUE, C_GREEN, C_VERMILLION


# ---------------------------------------------------------------------------
# Wong-palette colour assignments
# ---------------------------------------------------------------------------
_C_R     = C_BLACK        # panel (a) bubble radius
_C_V     = C_VERMILLION   # panel (a) shell velocity
_C_LMECH = C_BLUE         # panel (c) Lmech
_C_LGAIN = C_GREEN        # panel (c) Lgain (Wong "bluish green")
_C_LLOSS = C_VERMILLION   # panel (c) Lloss

# Panel (d) sequential purple ramp (darkest = gas absorption)
_SHADE_GAS    = "#6c4a78"
_SHADE_DUST   = "#a98ec0"
_SHADE_ESCAPE = "#dccdec"

_PHASE_LINE_KW = dict(color="0.6", linestyle=":", linewidth=0.8, zorder=0)

# Phases for which the bubble energy budget is well defined.
_BUBBLE_PHASES = ("energy", "implicit")

_PHASE_LABEL = {
    "energy":     r"\textsc{energy}",
    "implicit":   r"\textsc{implicit}",
    "transition": r"\textsc{transition}",
    "momentum":   r"\textsc{momentum}",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run(data_path):
    """Return a dict of arrays needed for the four panels."""
    output = load_output(data_path)
    if len(output) == 0:
        raise ValueError(f"No snapshots found in {data_path}")

    t      = np.asarray(output.get("t_now"), dtype=float)
    phase  = np.asarray(output.get("current_phase", as_array=False))
    R2     = np.asarray(output.get("R2"), dtype=float)              # [pc]
    v2_au  = np.asarray(output.get("v2"), dtype=float)              # [pc/Myr]
    Mshell = np.asarray(output.get("shell_mass"), dtype=float)      # [Msun]

    # Bubble luminosity-budget terms are stored in [Msun*pc^2/Myr^3].
    Lmech = np.asarray(output.get("Lmech_total"),  dtype=float) * cvt.L_au2cgs
    Lgain = np.asarray(output.get("bubble_Lgain"), dtype=float) * cvt.L_au2cgs
    Lloss = np.asarray(output.get("bubble_Lloss"), dtype=float) * cvt.L_au2cgs

    fAbs  = np.asarray(output.get("shell_fAbsorbedIon"), dtype=float)
    fDust = np.asarray(output.get("shell_fIonisedDust"), dtype=float)

    # The snapshot's mCloud is already the post-SF gas mass:
    # see src/_input/read_param.py:307-309, where
    #   mCloud  ←  mCloud_initial * (1 - sfe).
    mCloud_gas = float(output[0].get("mCloud"))

    return dict(
        t=t, phase=phase,
        R2=R2, v_kms=v2_au * cvt.v_au2kms, Mshell=Mshell,
        Lmech=Lmech, Lgain=Lgain, Lloss=Lloss,
        fAbs=fAbs, fDust=fDust, mCloud_gas=mCloud_gas,
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


def _mask_to_phases(values, phase, allowed):
    """Return a copy of *values* with timesteps outside *allowed* set to NaN.

    Inside the kept range, pre-existing NaNs are preserved unchanged so
    matplotlib gaps them rather than interpolating across.
    """
    out = np.array(values, dtype=float, copy=True)
    keep = np.isin(np.asarray(phase), list(allowed))
    out[~keep] = np.nan
    return out


def _draw_phase_boundaries(axes, t, phase):
    bnd = _change_points(phase)
    for ax in axes:
        for i in bnd:
            ax.axvline(t[i], **_PHASE_LINE_KW)


def _annotate_phase_labels(ax_top, t, phase):
    """Place phase names above *ax_top*, centred over each phase's interval.

    Centre is the geometric mean of the segment endpoints since the
    x-axis is log-scaled.
    """
    bnd    = _change_points(phase)
    starts = np.concatenate([[0], bnd])
    ends   = np.concatenate([bnd, [len(t)]])
    for i0, i1 in zip(starts, ends):
        if i1 <= i0:
            continue
        seg = phase[i0]
        if seg not in _PHASE_LABEL:
            continue
        t_lo, t_hi = t[i0], t[i1 - 1]
        if not (np.isfinite(t_lo) and np.isfinite(t_hi) and t_lo > 0 and t_hi > 0):
            continue
        xc = np.sqrt(t_lo * t_hi)
        ax_top.text(
            xc, 1.02, _PHASE_LABEL[seg],
            ha="center", va="bottom",
            transform=ax_top.get_xaxis_transform(),
            fontsize=10,
        )


# ---------------------------------------------------------------------------
# Panel (d) decomposition
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
        nrows=4, ncols=1, sharex=True,
        figsize=(4.0, 8.0),
        gridspec_kw=dict(hspace=0.05),
    )
    ax_a, ax_b, ax_c, ax_d = axes

    # ---- panel (a) ---------------------------------------------------------
    ax_a.plot(run["t"], run["R2"], color=_C_R, lw=1.5)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$R_{\rm b}\ [{\rm pc}]$", color=_C_R)
    ax_a.tick_params(axis="y", colors=_C_R)

    ax_av = ax_a.twinx()
    ax_av.plot(run["t"], run["v_kms"], color=_C_V, lw=1.5)
    ax_av.set_ylabel(r"$v_{\rm sh}\ [{\rm km\ s^{-1}}]$", color=_C_V)
    ax_av.tick_params(axis="y", colors=_C_V)

    # ---- panel (b) ---------------------------------------------------------
    ax_b.plot(run["t"], run["Mshell"], color=_C_R, lw=1.5)
    ax_b.set_yscale("log")
    ax_b.set_ylabel(r"$M_{\rm sh}\ [M_{\odot}]$")
    ax_b.axhline(run["mCloud_gas"], color="0.5", lw=0.8, ls="--", zorder=0)
    ax_b.text(
        0.995, run["mCloud_gas"], r"$M_{\rm cloud}(1-\epsilon)$",
        transform=ax_b.get_yaxis_transform(),
        ha="right", va="bottom", color="0.4", fontsize=9,
    )

    # ---- panel (c) — bubble energy budget ----------------------------------
    Lmech = _mask_to_phases(run["Lmech"], run["phase"], _BUBBLE_PHASES)
    Lgain = _mask_to_phases(run["Lgain"], run["phase"], _BUBBLE_PHASES)
    Lloss = _mask_to_phases(run["Lloss"], run["phase"], _BUBBLE_PHASES)
    ax_c.plot(run["t"], Lmech, color=_C_LMECH, lw=1.5, label=r"$L_{\rm mech}$")
    ax_c.plot(run["t"], Lgain, color=_C_LGAIN, lw=1.5, label=r"$L_{\rm gain}$")
    ax_c.plot(run["t"], Lloss, color=_C_LLOSS, lw=1.5, label=r"$L_{\rm loss}$")
    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"$L\ [{\rm erg\ s^{-1}}]$")
    ax_c.legend(
        loc="lower right", frameon=False, fontsize=10, ncol=3,
        handlelength=1.0, columnspacing=0.8,
    )

    # ---- panel (d) — ionising-photon budget --------------------------------
    gas, dust, escape = _ionising_components(run["fAbs"], run["fDust"])
    ax_d.stackplot(
        run["t"], gas, dust, escape,
        colors=[_SHADE_GAS, _SHADE_DUST, _SHADE_ESCAPE],
        labels=["gas", "dust", "escape"],
        edgecolor="none",
    )
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_ylabel(r"$Q_{\rm i}$ budget")
    ax_d.legend(
        loc="lower right", frameon=False, fontsize=10, ncol=3,
        handlelength=1.0, columnspacing=0.8,
    )

    # ---- shared x-axis (only bottom panel shows tick labels) ---------------
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_d.set_xscale("log")
    ax_d.set_xlabel(r"$t\ [{\rm Myr}]$")
    finite_t = run["t"][np.isfinite(run["t"]) & (run["t"] > 0)]
    if finite_t.size > 0:
        ax_d.set_xlim(finite_t.min(), finite_t.max())

    # ---- phase boundaries + top-panel labels -------------------------------
    _draw_phase_boundaries(list(axes) + [ax_av], run["t"], run["phase"])
    _annotate_phase_labels(ax_a, run["t"], run["phase"])

    # Panel-letter labels in upper-left
    for ax, letter in zip(axes, "abcd"):
        ax.text(
            0.012, 0.93, f"({letter})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11,
        )

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
        description="Paper I teaser figure: bubble dynamics, swept mass, "
                    "energy budget, and ionising-photon budget.",
        plot_from_path_fn=plot_from_path,
        plot_grid_fn=plot_grid,
    )
