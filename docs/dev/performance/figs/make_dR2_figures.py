#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the dR2 robustness story (idea / safety envelope / test).

Reads the real captured states + the production solver, so the plotted numbers are
the same ones test/test_dR2min_magic_number.py asserts. No LaTeX (Agg backend).

    cd /home/user/trinity
    python docs/dev/performance/figs/make_dR2_figures.py

Writes docs/dev/performance/figs/dR2_{idea,envelope,crosssolver}.png and
docs/dev/performance/data/dR2_crosssolver_residual.csv.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scipy.integrate  # noqa: E402
import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIGS = ROOT / "docs" / "dev" / "performance" / "figs"
DATA = ROOT / "docs" / "dev" / "performance" / "data"

# House style (paper/_lib/trinity.mplstyle) with usetex OFF -- the container has no
# LaTeX, matching docs/dev/shell-solver/plots/make_plots.py. Multi-panel dev figures
# size down the paper defaults and lean on constrained_layout to avoid overlaps.
plt.style.use(str(ROOT / "paper" / "_lib" / "trinity.mplstyle"))
plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.use": True,
})
_BBOX = dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92)

# Reuse the test's loader/helpers so figures and assertions share one source of truth.
_spec = importlib.util.spec_from_file_location(
    "_dr2test", ROOT / "test" / "test_dR2min_magic_number.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

EPS = np.finfo(float).eps

# --- calibrate the analytic layer from the stiff captured state (5e9/sfe0.01) ------
stiff, stiff_params = T._load_state("dR2_stiff_state_fixture.json")
R2_stiff = stiff_params["R2"].value
dR2_stiff = stiff["dR2_over_R2"] * R2_stiff      # pc
MCL_STIFF = 0.01 * 5e9                            # sfe * mCloud = cluster mass [Msun]


def _warpfield_floor(Mclus):
    """WARPFIELD's dR2min: 1e-7, bumped to 1e-14*Mclus+1e-7 for Mclus>1e7 [pc]."""
    return np.where(Mclus > 1e7, 1e-14 * Mclus + 1e-7, 1e-7)


def fig_idea():
    """trinity's exact analytic dR2 vs WARPFIELD's clamped-and-bumped floor."""
    Mclus = np.logspace(4, 10, 400)
    # dR2 ~ 1/dMdt ~ 1/Mclus at fixed bubble radius (illustrative calibration)
    analytic = dR2_stiff * (MCL_STIFF / Mclus)
    floor = _warpfield_floor(Mclus)
    warp_eff = np.maximum(analytic, floor)
    floor_at_stiff = float(_warpfield_floor(np.array(MCL_STIFF)))

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.loglog(Mclus, analytic, "-", lw=2.4, color="#1f77b4",
              label="trinity: exact analytic dR2 (no floor)")
    ax.loglog(Mclus, warp_eff, "--", lw=2.2, color="#d62728",
              label="WARPFIELD: max(analytic, dR2min)")
    ax.loglog(Mclus, floor, ":", lw=1.5, color="#d62728", alpha=0.7,
              label="WARPFIELD dR2min floor (1e-7; +1e-14 Mclus)")
    ax.axvline(1e7, color="0.5", ls=":", lw=1.0)
    ax.text(1.15e7, 4e-12, "Mclus = 1e7\n(floor bump)", fontsize=8, color="0.4")

    ax.plot([MCL_STIFF], [dR2_stiff], "o", ms=10, color="#1f77b4", zorder=6,
            markeredgecolor="k")
    # state info parked in the empty lower-left (legend is up top)
    ax.text(1.7e4, 6e-12,
            f"captured stiff state\nMclus = 5e7 Msun\nexact dR2 = {dR2_stiff:.1e} pc",
            fontsize=8.5, va="bottom", bbox=_BBOX)
    # the gap the floor imposes, annotated beside the double arrow
    ax.annotate("", xy=(MCL_STIFF, dR2_stiff), xytext=(MCL_STIFF, floor_at_stiff),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
    ax.text(7.5e7, 4e-9, f"dR2min is ~{floor_at_stiff/dR2_stiff:.0e}x\n"
            "too thick here\n(-> L_bubble ~8x)", color="green", fontsize=8.5,
            fontweight="bold", va="center", bbox=_BBOX)
    ax.set_xlabel("cluster mass  Mclus = sfe x mCloud  [Msun]")
    ax.set_ylabel("conduction-layer thickness  dR2  [pc]")
    ax.set_title("The idea: trinity uses the exact thin layer; WARPFIELD's magic floor\n"
                 "over-thickens it by ~1000x for massive clusters")
    ax.set_ylim(1e-12, 3e-4)
    ax.legend(loc="upper left")
    fig.savefig(FIGS / "dR2_idea.png")
    plt.close(fig)


def fig_envelope():
    """dR2/R2 vs cluster mass with the float64 cancellation cliff + safety margin."""
    Mclus = np.logspace(4, 16, 500)
    ratio = stiff["dR2_over_R2"] * (MCL_STIFF / Mclus)
    cliff = EPS / 2.0
    M_cliff = MCL_STIFF * stiff["dR2_over_R2"] / cliff   # Mclus where ratio hits cliff

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhspan(cliff, 1.0, color="#2ca02c", alpha=0.07, zorder=0)
    ax.loglog(Mclus, ratio, "-", lw=2.4, color="#1f77b4", label="trinity dR2/R2 (analytic)")
    ax.axhline(cliff, color="#d62728", ls="--", lw=2.0,
               label=f"float64 cancellation cliff (eps/2 = {cliff:.0e})")
    ax.text(2e4, 2e-3, "well-conditioned:  R2 - dR2 < R2", color="#1a7d1a", fontsize=9)

    ax.plot([MCL_STIFF], [stiff["dR2_over_R2"]], "o", ms=10, color="#1f77b4",
            markeredgecolor="k", zorder=6)
    ax.annotate(f"stiffest physical state\n(Mclus = 5e7): {stiff['dR2_over_R2']:.1e}",
                xy=(MCL_STIFF, stiff["dR2_over_R2"]), xytext=(8e8, 2e-6),
                fontsize=8.5, bbox=_BBOX,
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.plot([M_cliff], [cliff], "X", ms=12, color="#d62728", zorder=6)
    ax.annotate(f"would only break at\nMclus ~ {M_cliff:.0e} Msun\n(no real GMC)",
                xy=(M_cliff, cliff), xytext=(2e9, 2e-14), fontsize=8.5, bbox=_BBOX,
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    margin = np.log10(stiff["dR2_over_R2"] / cliff)
    ax.annotate("", xy=(MCL_STIFF, stiff["dR2_over_R2"]), xytext=(MCL_STIFF, cliff),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.6))
    ax.text(1.5e7, 3e-14, f"~{margin:.0f} decades\nof margin", color="green",
            fontsize=9, fontweight="bold", ha="right", bbox=_BBOX)
    ax.set_xlabel("cluster mass  Mclus  [Msun]")
    ax.set_ylabel("relative layer thickness  dR2 / R2")
    ax.set_title("The safety envelope: the unfloored R2 - dR2 only fails near machine\n"
                 "epsilon -- ~5 decades below any real cluster")
    ax.set_ylim(1e-18, 1e-2)
    ax.set_xlim(1e4, 1e16)
    ax.legend(loc="upper right")
    fig.savefig(FIGS / "dR2_envelope.png")
    plt.close(fig)


def _integrate(params, Pb, r2Prime, R1, y0, method, rtol, atol):
    with BL._quiet_lsoda_fortran():
        return scipy.integrate.solve_ivp(
            fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
            t_span=(r2Prime, R1), y0=y0, method=method, rtol=rtol, atol=atol,
            dense_output=True)


def fig_crosssolver():
    """The test: production LSODA vs independent Radau across the stiff thin layer."""
    Pb, R1 = stiff["Pb"], stiff["R1"]
    r2Prime, T0, dTdr0, v0 = T._ic(stiff["dMdt_converged"], stiff_params, Pb, R1)
    y0 = [v0, T0, dTdr0]
    sL = _integrate(stiff_params, Pb, r2Prime, R1, y0, "LSODA",
                    BL._BUBBLE_RTOL, BL._BUBBLE_ATOL)
    sR = _integrate(stiff_params, Pb, r2Prime, R1, y0, "Radau", 1e-10, 1e-12)

    r = np.linspace(r2Prime, R1, 600)
    TL, TR = sL.sol(r)[1], sR.sol(r)[1]
    reldiff = np.abs(TL - TR) / np.abs(TR)
    depth = (r2Prime - r)            # distance INward from the outer boundary [pc]

    with open(DATA / "dR2_crosssolver_residual.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["depth_in_from_R2_pc", "T_LSODA_K", "T_Radau_K", "rel_diff"])
        for d, a, b, c in zip(depth, TL, TR, reldiff):
            w.writerow([f"{d:.6e}", f"{a:.6e}", f"{b:.6e}", f"{c:.3e}"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.3, 1]})
    ax1.semilogy(depth, TL, "-", lw=3.4, color="#1f77b4", alpha=0.5,
                 label="production LSODA (rtol=1e-8)")
    ax1.semilogy(depth, TR, "--", lw=1.4, color="#d62728",
                 label="independent Radau (rtol=1e-10)")
    ax1.set_ylabel("temperature  T  [K]")
    ax1.set_title("The test: the ultra-thin conduction layer is integrated correctly,\n"
                  f"unfloored  (stiff state: dR2/R2 = {stiff['dR2_over_R2']:.1e}, "
                  f"dT/dr|0 = {dTdr0:.1e} K/pc)")
    ax1.legend(loc="lower right")

    ax2.semilogy(depth, np.maximum(reldiff, 1e-12), "-", lw=1.6, color="#6a3d9a",
                 label=f"max ~ {reldiff.max():.0e}")
    ax2.axhline(1e-5, color="k", ls="--", lw=1.2, label="test tolerance (1e-5)")
    ax2.set_ylabel("|T_LSODA - T_Radau| / T")
    ax2.set_xlabel("depth inward from outer shock  R2 - r  [pc]")
    ax2.set_ylim(1e-11, 1e-3)
    ax2.legend(loc="upper right", ncol=2)
    fig.savefig(FIGS / "dR2_crosssolver.png")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig_idea()
    fig_envelope()
    fig_crosssolver()
    print("wrote dR2_idea.png, dR2_envelope.png, dR2_crosssolver.png + residual CSV")
