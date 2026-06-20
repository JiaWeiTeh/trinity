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

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(Mclus, analytic, "-", lw=2.4, color="#1f77b4",
              label="trinity: exact analytic dR2 (no floor)")
    ax.loglog(Mclus, warp_eff, "--", lw=2.2, color="#d62728",
              label="WARPFIELD: max(analytic, dR2min)")
    ax.loglog(Mclus, floor, ":", lw=1.6, color="#d62728", alpha=0.7,
              label="WARPFIELD dR2min floor (1e-7; +1e-14*Mclus)")
    ax.axvline(1e7, color="grey", ls=":", lw=1.0)
    ax.text(1.1e7, analytic.min() * 3, "Mclus = 1e7\n(floor bump)",
            fontsize=8, color="grey")
    ax.plot([MCL_STIFF], [dR2_stiff], "o", ms=10, color="#1f77b4", zorder=5,
            markeredgecolor="k")
    ax.annotate(f"captured stiff state\n5e7 Msun: dR2={dR2_stiff:.1e} pc\n"
                f"(WARPFIELD would force ~{_warpfield_floor(np.array(MCL_STIFF)):.1e} pc)",
                xy=(MCL_STIFF, dR2_stiff), xytext=(2e5, 3e-9),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"))
    # gap annotation
    gap = _warpfield_floor(np.array(MCL_STIFF)) / dR2_stiff
    ax.annotate("", xy=(MCL_STIFF, dR2_stiff),
                xytext=(MCL_STIFF, _warpfield_floor(np.array(MCL_STIFF))),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.8))
    ax.text(MCL_STIFF * 1.3, 1e-8, f"~{gap:.0e}x\ntoo thick", color="green",
            fontsize=9, fontweight="bold")
    ax.set_xlabel("cluster mass  Mclus = sfe x mCloud  [Msun]")
    ax.set_ylabel("conduction-layer thickness  dR2  [pc]")
    ax.set_title("The idea: trinity uses the exact thin layer; WARPFIELD's magic\n"
                 "floor over-thickens it by ~10^3 for massive clusters")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "dR2_idea.png", dpi=130)
    plt.close(fig)


def fig_envelope():
    """dR2/R2 vs cluster mass with the float64 cancellation cliff + safety margin."""
    Mclus = np.logspace(4, 16, 500)
    ratio = stiff["dR2_over_R2"] * (MCL_STIFF / Mclus)
    cliff = EPS / 2.0
    M_cliff = MCL_STIFF * stiff["dR2_over_R2"] / cliff   # Mclus where ratio hits cliff

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.loglog(Mclus, ratio, "-", lw=2.4, color="#1f77b4", label="trinity dR2/R2 (analytic)")
    ax.axhline(cliff, color="#d62728", ls="--", lw=2.0,
               label=f"float64 cancellation cliff (eps/2 = {cliff:.1e})")
    ax.axhspan(cliff, 1.0, color="#2ca02c", alpha=0.08)
    ax.text(2e4, 3e-3, "well-conditioned: R2 - dR2 < R2", color="#2ca02c", fontsize=9)

    ax.plot([MCL_STIFF], [stiff["dR2_over_R2"]], "o", ms=10, color="#1f77b4",
            markeredgecolor="k", zorder=5)
    ax.annotate(f"stiffest physical state\n(5e7 Msun): {stiff['dR2_over_R2']:.1e}",
                xy=(MCL_STIFF, stiff["dR2_over_R2"]), xytext=(3e8, 1e-7),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="k"))
    ax.plot([M_cliff], [cliff], "X", ms=11, color="#d62728", zorder=5)
    ax.annotate(f"would only break at\nMclus ~ {M_cliff:.0e} Msun\n(no real GMC)",
                xy=(M_cliff, cliff), xytext=(1e11, 1e-13), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    margin = np.log10(stiff["dR2_over_R2"] / cliff)
    ax.annotate("", xy=(MCL_STIFF, stiff["dR2_over_R2"]), xytext=(MCL_STIFF, cliff),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.6))
    ax.text(MCL_STIFF * 1.3, 1e-13, f"~{margin:.1f}\ndecades\nof margin",
            color="green", fontsize=9, fontweight="bold")
    ax.set_xlabel("cluster mass  Mclus  [Msun]")
    ax.set_ylabel("relative layer thickness  dR2 / R2")
    ax.set_title("The safety envelope: the unfloored subtraction R2 - dR2 only fails\n"
                 "near machine epsilon -- ~5 decades below any real cluster")
    ax.set_ylim(1e-18, 1e-2)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "dR2_envelope.png", dpi=130)
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.3, 1]})
    ax1.semilogy(depth, TL, "-", lw=3.2, color="#1f77b4", alpha=0.55,
                 label="production LSODA (rtol=1e-8)")
    ax1.semilogy(depth, TR, "--", lw=1.4, color="#d62728",
                 label="independent Radau (rtol=1e-10)")
    ax1.set_ylabel("temperature  T  [K]")
    ax1.set_title("The test: the ultra-thin conduction layer is integrated correctly,\n"
                  f"unfloored  (stiff state: dR2/R2 = {stiff['dR2_over_R2']:.1e}, "
                  f"dT/dr|0 = {dTdr0:.1e} K/pc)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, which="both", alpha=0.25)

    ax2.semilogy(depth, np.maximum(reldiff, 1e-12), "-", lw=1.6, color="#6a3d9a")
    ax2.axhline(1e-5, color="k", ls="--", lw=1.2, label="test tolerance (1e-5)")
    ax2.text(depth.max() * 0.55, 2e-8,
             f"measured max ~ {reldiff.max():.0e}", fontsize=9, color="#6a3d9a")
    ax2.set_ylabel("|T_LSODA - T_Radau| / T")
    ax2.set_xlabel("depth inward from outer shock  R2 - r  [pc]")
    ax2.set_ylim(1e-11, 1e-3)
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "dR2_crosssolver.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig_idea()
    fig_envelope()
    fig_crosssolver()
    print("wrote dR2_idea.png, dR2_envelope.png, dR2_crosssolver.png + residual CSV")
