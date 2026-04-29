#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged figure: rCloud density smoothing schematic (top) over the
before/after LSODA-failure trajectory comparison (bottom).

The two panels share the same narrative: panel 1 motivates the tanh
hyperbolic blend by showing how it replaces the discontinuous density
step at rCloud, and panel 2 demonstrates the downstream consequence —
the trajectory that previously triggered an LSODA failure now finishes
cleanly.

Top panel logic comes from ``paper_rcloud_smooth.py``; bottom panel
logic comes from ``paper_v2R2_blend.py``. Style follows the lsoda
demonstration (large tick labels, generous panel size).

Usage:

    python paper_rcloud_smoothing.py <parent-folder-or-npz> [-o out.pdf]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))

import src._functions.unit_conversions as cvt

from src._plots.plot_base import FIG_DIR
from src._plots.paper_v2R2_blend import (
    load_v2R2_pair,
    STYLE_AFTER,
    STYLE_BEFORE,
    _build_legend_handles,
)
from src._plots.paper_v2R2 import _plot_one_trajectory
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_homogeneous,
    compute_rCloud_powerlaw,
)


# =============================================================================
# Top-panel: fiducial cloud parameters (mirrors paper_rcloud_smooth.py)
# =============================================================================
M_CLOUD = 1e6           # Msun
N_CORE_CGS = 1e3        # cm^-3
N_ISM_CGS = 1.0         # cm^-3
ALPHA = 0               # power-law slope; 0 = homogeneous
R_CORE = 0.1            # standalone core radius [pc]

SMOOTH_FRAC_DEFAULT = 0.01
SMOOTH_FRAC_LOW = 0.005
SMOOTH_FRAC_HIGH = 0.02

MU_CONVERT = 1.4 * cvt.M_H_CGS * cvt.g2Msun
N_CORE_AU = N_CORE_CGS * cvt.ndens_cgs2au
N_ISM_AU = N_ISM_CGS * cvt.ndens_cgs2au

if ALPHA == 0:
    R_CLOUD = compute_rCloud_homogeneous(M_CLOUD, N_CORE_AU, mu=MU_CONVERT)
else:
    R_CLOUD, _ = compute_rCloud_powerlaw(
        M_CLOUD, N_CORE_AU, ALPHA,
        rCore=R_CORE, mu=MU_CONVERT,
    )


def _density_inside(r):
    if ALPHA == 0:
        return np.full_like(r, N_CORE_AU)
    n = N_CORE_AU * (r / R_CORE) ** ALPHA
    return np.where(r <= R_CORE, N_CORE_AU, n)


def _density_jump(r):
    return np.where(r <= R_CLOUD, _density_inside(r), N_ISM_AU)


def _density_blend(r, smooth_frac):
    delta = smooth_frac * R_CLOUD
    w_out = 0.5 * (1.0 + np.tanh((r - R_CLOUD) / delta))
    n_in = _density_inside(r)
    return n_in * (1.0 - w_out) + N_ISM_AU * w_out


def _draw_rcloud_panel(ax, fontsize):
    """Top panel: density step vs tanh blends around rCloud."""
    r_log = np.geomspace(1e-2 * R_CLOUD, 1.5 * R_CLOUD, 4000)
    r_band = np.linspace(0.7 * R_CLOUD, 1.3 * R_CLOUD, 2000)
    r = np.unique(np.concatenate([r_log, r_band]))

    # Original discontinuous step
    n_jump = _density_jump(r)
    ax.plot(r, n_jump * cvt.ndens_au2cgs,
            color='k', ls='-', lw=1.6, label='step (original)')

    # Three blend widths: below default, default (highlighted), above default
    blend_specs = [
        (SMOOTH_FRAC_LOW,     '#0072B2', '--', 1.2),
        (SMOOTH_FRAC_DEFAULT, '#009E73', '-',  2.0),
        (SMOOTH_FRAC_HIGH,    '#D55E00', '--', 1.2),
    ]
    for sf, color, ls, lw in blend_specs:
        n_b = _density_blend(r, sf)
        is_default = np.isclose(sf, SMOOTH_FRAC_DEFAULT)
        label = (r'$f_{\rm smooth}$' + rf'$={sf:g}$'
                 + (' (default)' if is_default else ''))
        ax.plot(r, n_b * cvt.ndens_au2cgs,
                color=color, ls=ls, lw=lw, label=label)

    ax.axvline(R_CLOUD, color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    ax.set_xlim(0.0, 1.3 * R_CLOUD)
    ax.set_ylim(-0.2 * N_CORE_CGS, 1.25 * N_CORE_CGS)

    ax.tick_params(labelsize=fontsize, axis='both')
    ax.set_xticks([R_CLOUD])
    ax.set_xticklabels([r'$R_\mathrm{cloud}$'])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_ylabel(r'$n(r)$', fontsize=fontsize)

    _label_offset = 0.04 * N_CORE_CGS
    ax.text(0.04 * R_CLOUD, N_CORE_CGS + _label_offset, r'$n_\mathrm{core}$',
            va='bottom', ha='left', color='0.25', fontsize=fontsize - 4)
    ax.text(1.28 * R_CLOUD, N_ISM_CGS + _label_offset, r'$n_\mathrm{ISM}$',
            va='bottom', ha='right', color='0.25', fontsize=fontsize - 4)

    ax.legend(loc='lower left', handlelength=1.6, labelspacing=0.3,
              fontsize=fontsize - 6, framealpha=0.9)


def _draw_v2R2_panel(ax, pair, fontsize):
    """Bottom panel: before/after blend trajectories on (R_b, |v_b|)."""
    before = pair["before"]
    after  = pair["after"]

    rcloud = next((float(d["rcloud"]) for d in (after, before)
                   if np.isfinite(d.get("rcloud", np.nan))), None)
    if rcloud is not None:
        ax.axvline(rcloud, color="0.25", lw=1.2, ls="--", alpha=0.7, zorder=2)

    # Draw "before" first so "after" (the hero) renders on top.
    _plot_one_trajectory(ax, before, STYLE_BEFORE)
    _plot_one_trajectory(ax, after,  STYLE_AFTER)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4, zorder=0)
    ax.tick_params(labelsize=fontsize, axis='both')
    ax.set_xlabel(r"$R_b$ [pc]", fontsize=fontsize)
    ax.set_ylabel(r"$v_b$ [km s$^{-1}$]", fontsize=fontsize)

    ax.legend(handles=_build_legend_handles(), loc="lower left",
              fontsize=fontsize - 6, framealpha=0.9)


def plot_merged(pair: dict, out_path: Optional[Path] = None,
                show: bool = False) -> plt.Figure:
    """Stack the rCloud-smoothing schematic over the v_2 vs R_2 comparison."""
    FONTSIZE = 25

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, figsize=[8, 6],
        gridspec_kw=dict(height_ratios=[1, 1]),
    )

    _draw_rcloud_panel(ax_top, FONTSIZE)
    _draw_v2R2_panel(ax_bot, pair, FONTSIZE)

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()

    return fig


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Stacked figure: rCloud smoothing schematic on top, "
                     "before/after blend v_2 vs R_2 trajectory on bottom. "
                     "Source can be a folder of .jsonl snapshots or a "
                     "published .npz bundle."),
    )
    parser.add_argument("source",
                        help="parent folder containing *_before_blend/ and "
                             "*_after_blend/ TRINITY run subdirectories, or "
                             "a .npz bundle")
    parser.add_argument("-o", "--out", default=None,
                        help="output PDF path "
                             "(default: <FIG_DIR>/paper_rcloud_smoothing_<id>.pdf)")
    parser.add_argument("--show", action="store_true",
                        help="open the figure window (in addition to saving)")
    args = parser.parse_args(argv)

    pair = load_v2R2_pair(args.source)
    run_id = pair["meta"].get("run_id", "blend")
    out_path = (Path(args.out) if args.out
                else FIG_DIR / f"paper_rcloud_smoothing_{run_id}.pdf")
    plot_merged(pair, out_path=out_path, show=args.show)


if __name__ == "__main__":
    main()
