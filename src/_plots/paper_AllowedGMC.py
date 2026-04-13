#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allowed GMC Parameter Space Visualization

This script creates visualizations of valid parameter combinations for
Giant Molecular Cloud (GMC) simulations in TRINITY.

Produces two PDF files:
1. paper_AllowedGMC_PowerLaw.pdf - rCloud at fixed rCore for various power-law alpha values
2. paper_AllowedGMC_BonnorEbert.pdf - rCloud for various dimensionless radii xi

Constraints checked:
1. rCloud <= 200 pc (typical GMC limit)
2. nEdge >= nISM (edge density above ISM)
3. Mass error <= 0.1% (self-consistent parameters)

@author: TRINITY Team
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import cmasher as cmr

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from src._plots.plot_base import FIG_DIR
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_powerlaw,
    compute_rCloud_homogeneous,
)
from src.cloud_properties.bonnorEbertSphere import (
    solve_lane_emden,
    XI_CRITICAL,
    OMEGA_CRITICAL,
)
from src.cloud_properties.validate_gmc import check_gmc_constraints


# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges
M_CLOUD_RANGE = np.logspace(4, 8, 50)  # 10^4 to 10^8 Msun
N_CORE_RANGE = np.logspace(2, 6, 50)   # 10^2 to 10^6 cm^-3

# Fixed core radius for power-law profiles.
# The mass equation relates (M, n_core, rCore, rCloud); fixing rCore makes
# rCloud a unique smooth function of (M, n_core) and avoids the staircase
# artifact introduced by scanning rCore on a discrete grid.
R_CORE_FIXED = 0.1  # pc

# Bounds used to flag unphysical BE scale lengths a.
A_MIN, A_MAX = 0.01, 5.0  # pc

# Physical constraints
R_CLOUD_MAX = 200.0  # pc - typical GMC limit
N_ISM = 1.0          # cm^-3 - ISM density
MASS_TOLERANCE = 0.001  # 0.1% mass error tolerance
MU = 1.4             # mean molecular weight

# Power-law exponents to test
ALPHA_VALUES = [-0.5, -1, -1.5, -2]

# Bonnor-Ebert dimensionless radii to test (including critical ~6.45)
XI_VALUES = [3.0, 4.0, XI_CRITICAL, 10]

SAVE_PDF = True

# Font sizes
TICK_FONT = 30
LABEL_FONT = 32
PANEL_FONT = 24
CBAR_LABEL_FONT = 34

# Layout (fractions of figure): tight subplot grid with colourbar to the right
SUBPLOT_LEFT = 0.10
SUBPLOT_RIGHT = 0.87
SUBPLOT_BOTTOM = 0.09
SUBPLOT_TOP = 0.98
SUBPLOT_WSPACE = 0.04
SUBPLOT_HSPACE = 0.04
CBAR_LEFT = 0.89
CBAR_WIDTH = 0.035


# =============================================================================
# Power-Law Cloud Functions
# =============================================================================

def compute_mass_powerlaw(rCloud, rCore, nCore, alpha, mu=MU):
    """
    Compute enclosed mass for power-law density profile.

    M = 4πρc [rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]
    """
    rhoCore = nCore * mu

    if alpha == 0:
        # Homogeneous case
        return (4.0/3.0) * np.pi * rCloud**3 * rhoCore
    else:
        return 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0 +
            (rCloud**(3.0 + alpha) - rCore**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore**alpha)
        )


def compute_rCloud_powerlaw_fixed(mCloud, nCore, alpha, rCore=R_CORE_FIXED,
                                   nISM=N_ISM, rCloud_max=R_CLOUD_MAX, mu=MU):
    """
    Compute rCloud for given (mCloud, nCore, alpha) at a fixed rCore.

    Returns rCloud [pc] if all constraints pass, else NaN.
    """
    try:
        if alpha == 0:
            rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu)
        else:
            rCloud, _ = compute_rCloud_powerlaw(mCloud, nCore, alpha,
                                                 rCore=rCore, mu=mu)

        # Derived quantities
        nEdge = nCore if alpha == 0 else nCore * (rCloud / rCore) ** alpha
        M_computed = compute_mass_powerlaw(rCloud, rCore, nCore, alpha, mu)

        issues = check_gmc_constraints(
            rCloud, nEdge, mCloud, M_computed,
            nISM=nISM, r_max=rCloud_max,
            mass_tolerance=MASS_TOLERANCE,
        )
        if issues["errors"]:
            return np.nan

        return rCloud

    except (ValueError, ZeroDivisionError, RuntimeError):
        return np.nan


def compute_rCloud_grid_powerlaw(alpha, rCore=R_CORE_FIXED):
    """
    Compute 2D grid of rCloud values for power-law profile at fixed rCore.
    Invalid (M, n_core) cells are NaN.
    """
    grid_rCloud = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)

    for i, nCore in enumerate(N_CORE_RANGE):
        for j, mCloud in enumerate(M_CLOUD_RANGE):
            grid_rCloud[i, j] = compute_rCloud_powerlaw_fixed(
                mCloud, nCore, alpha, rCore=rCore
            )

    return grid_rCloud


# =============================================================================
# Bonnor-Ebert Cloud Functions
# =============================================================================

# Cache the Lane-Emden solution (computed once)
_LE_SOLUTION = None

def get_lane_emden_solution():
    """Get cached Lane-Emden solution."""
    global _LE_SOLUTION
    if _LE_SOLUTION is None:
        _LE_SOLUTION = solve_lane_emden()
    return _LE_SOLUTION


def find_valid_rCore_BE(mCloud, nCore, xi_out, nISM=N_ISM,
                         rCloud_max=R_CLOUD_MAX, mu=MU):
    """
    Find valid scale length *a* and cloud radius for a BE sphere.

    For BE sphere, "rCore" is approximated as the characteristic length scale a,
    which is where ξ = 1 (where density starts to deviate from central value).

    Returns (a_pc, rCloud), or (NaN, NaN) if invalid.

    Physics:
    - a = √(c_s² / (4πGρc)) is the characteristic length scale
    - ξ = r/a is the dimensionless radius
    - M = 4π × m(ξ) × ρc × a³ where m(ξ) = ξ² du/dξ
    - ρ(ξ)/ρc = exp(-u(ξ)) gives the density profile
    """
    try:
        # Get Lane-Emden solution
        solution = get_lane_emden_solution()

        # Get density ratio at xi_out: Omega = ρc/ρsurf = exp(u(xi_out))
        u_at_xi = np.interp(xi_out, solution.xi, solution.u)
        Omega = np.exp(u_at_xi)

        # Get dimensionless mass at xi_out: m(ξ) = ξ² du/dξ
        m_dim = np.interp(xi_out, solution.xi, solution.m)

        # Physical constants in CGS
        G_cgs = 6.674e-8       # cm³/g/s²
        m_H_cgs = 1.67e-24     # g (hydrogen mass)
        Msun_cgs = 1.989e33    # g
        pc_cgs = 3.086e18      # cm

        # Core density in CGS: ρc = nCore × μ × m_H
        rhoCore_cgs = nCore * mu * m_H_cgs  # g/cm³

        # Cloud mass in CGS
        M_cgs = mCloud * Msun_cgs  # g

        # From M = 4π × m(ξ) × ρc × a³, solve for a:
        # a³ = M / (4π × m(ξ) × ρc)
        # a = (M / (4π × m(ξ) × ρc))^(1/3)
        a_cgs = (M_cgs / (4.0 * np.pi * m_dim * rhoCore_cgs))**(1.0/3.0)
        a_pc = a_cgs / pc_cgs  # convert to pc

        # Physical cloud radius
        rCloud = xi_out * a_pc

        # Derived quantities for constraint check
        nEdge = nCore / Omega  # nEdge = nCore / Omega since ρ_edge/ρc = 1/Omega
        M_computed_cgs = 4.0 * np.pi * m_dim * rhoCore_cgs * a_cgs**3
        M_computed = M_computed_cgs / Msun_cgs  # [Msun]

        # Check all constraints via shared validator (including mass error)
        issues = check_gmc_constraints(
            rCloud, nEdge, mCloud, M_computed,
            nISM=nISM, r_max=rCloud_max,
            mass_tolerance=MASS_TOLERANCE,
        )
        if issues["errors"]:
            return (np.nan, np.nan)

        # Scale length a — clamp to a physically sensible range
        if a_pc < A_MIN or a_pc > A_MAX:
            return (np.nan, np.nan)

        return (a_pc, rCloud)

    except Exception:
        return (np.nan, np.nan)


def compute_valid_rCore_grid_BE(xi_out):
    """
    Compute 2D grids of valid scale-length *a* and rCloud for BE profile.

    Returns (grid_a, grid_rCloud).
    """
    grid_a = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)
    grid_rCloud = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)

    for i, nCore in enumerate(N_CORE_RANGE):
        for j, mCloud in enumerate(M_CLOUD_RANGE):
            a, rcl = find_valid_rCore_BE(mCloud, nCore, xi_out)
            grid_a[i, j] = a
            grid_rCloud[i, j] = rcl

    return grid_a, grid_rCloud


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_powerlaw_grids():
    """
    Figure showing rCloud (colour fill) across (M_cloud, n_core) for each
    power-law alpha, at fixed rCore. Shared colourbar on the right.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Tight subplot grid; leave room on the right for the colourbar
    fig.subplots_adjust(
        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE,
    )

    # Colourmap for rCloud fill
    vmin_rCloud, vmax_rCloud = 0.1, 200.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('white', 1.0)

    im = None  # will hold the last pcolormesh for shared colourbar
    for idx, alpha in enumerate(ALPHA_VALUES):
        ax = axes.flat[idx]
        row, col = divmod(idx, 2)

        print(f"Computing grid for \u03b1 = {alpha}...")
        grid_rCloud = compute_rCloud_grid_powerlaw(alpha)

        n_valid = np.sum(~np.isnan(grid_rCloud))
        n_total = grid_rCloud.size

        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid_rCloud,
            cmap=cmap, norm=LogNorm(vmin=vmin_rCloud, vmax=vmax_rCloud),
            shading='auto',
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=TICK_FONT)

        # Only show tick labels on outer edges
        if row == 0:
            ax.set_xticklabels([])
        if col == 1:
            ax.set_yticklabels([])

        if alpha == 0:
            title = r'$\alpha = 0$ (homogeneous)'
        else:
            title = rf'$\alpha = {alpha}$'
        panel_label = f'{title}\n({n_valid}/{n_total} valid)'
        ax.text(
            0.04, 0.04, panel_label,
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=PANEL_FONT,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='black', alpha=0.9),
        )

    # Shared axis labels
    fig.supxlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]', fontsize=LABEL_FONT)
    fig.supylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]', fontsize=LABEL_FONT)

    # Shared colourbar — height matches the subplot grid
    cbar_height = SUBPLOT_TOP - SUBPLOT_BOTTOM
    cbar_ax = fig.add_axes([CBAR_LEFT, SUBPLOT_BOTTOM, CBAR_WIDTH, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$r_\mathrm{cloud}$ [pc]', fontsize=CBAR_LABEL_FONT)
    cbar.ax.tick_params(labelsize=TICK_FONT)

    if SAVE_PDF:
        out_pdf = FIG_DIR / "paper_AllowedGMC_PowerLaw.pdf"
        fig.savefig(out_pdf)
        print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_BE_grids():
    """
    Figure showing rCloud (colour fill) across (M_cloud, n_core) for each
    Bonnor-Ebert xi_out value. Shared colourbar on the right.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Tight subplot grid; leave room on the right for the colourbar
    fig.subplots_adjust(
        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP,
        wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE,
    )

    # Colourmap for rCloud fill
    vmin_rCloud, vmax_rCloud = 0.1, 200.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('white', 1.0)

    im = None  # will hold the last pcolormesh for shared colourbar
    for idx, xi in enumerate(XI_VALUES):
        ax = axes.flat[idx]
        row, col = divmod(idx, 2)

        print(f"Computing grid for \u03be = {xi:.2f}...")
        _, grid_rCloud = compute_valid_rCore_grid_BE(xi)

        n_valid = np.sum(~np.isnan(grid_rCloud))
        n_total = grid_rCloud.size

        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid_rCloud,
            cmap=cmap, norm=LogNorm(vmin=vmin_rCloud, vmax=vmax_rCloud),
            shading='auto',
        )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=TICK_FONT)

        # Only show tick labels on outer edges
        if row == 0:
            ax.set_xticklabels([])
        if col == 1:
            ax.set_yticklabels([])

        if abs(xi - XI_CRITICAL) < 0.01:
            title = rf'$\xi = {xi:.2f}$ (critical)'
        else:
            title = rf'$\xi = {xi:.1f}$'
        panel_label = f'{title}\n({n_valid}/{n_total} valid)'
        ax.text(
            0.04, 0.04, panel_label,
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=PANEL_FONT,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='black', alpha=0.9),
        )

    # Shared axis labels
    fig.supxlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]', fontsize=LABEL_FONT)
    fig.supylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]', fontsize=LABEL_FONT)

    # Shared colourbar — height matches the subplot grid
    cbar_height = SUBPLOT_TOP - SUBPLOT_BOTTOM
    cbar_ax = fig.add_axes([CBAR_LEFT, SUBPLOT_BOTTOM, CBAR_WIDTH, cbar_height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$r_\mathrm{cloud}$ [pc]', fontsize=CBAR_LABEL_FONT)
    cbar.ax.tick_params(labelsize=TICK_FONT)

    if SAVE_PDF:
        out_pdf = FIG_DIR / "paper_AllowedGMC_BonnorEbert.pdf"
        fig.savefig(out_pdf)
        print(f"Saved: {out_pdf}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate both visualization PDFs."""
    print("=" * 70)
    print("Generating GMC Parameter Space Visualizations")
    print("=" * 70)

    print("\n--- Power-Law Density Profiles ---")
    plot_powerlaw_grids()

    print("\n--- Bonnor-Ebert Density Profiles ---")
    plot_BE_grids()

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
