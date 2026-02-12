#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allowed GMC Parameter Space Visualization

This script creates visualizations of valid parameter combinations for
Giant Molecular Cloud (GMC) simulations in TRINITY.

Produces two PDF files:
1. paper_AllowedGMC_PowerLaw.pdf - Valid rCore for various power-law alpha values
2. paper_AllowedGMC_BonnorEbert.pdf - Valid rCore for various dimensionless radii xi

Constraints checked:
1. rCloud <= 200 pc (typical GMC limit)
2. nEdge >= nISM (edge density above ISM)
3. Mass error <= 0.1% (self-consistent parameters)

@author: TRINITY Team
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
from matplotlib import patheffects
from pathlib import Path
import os
import cmasher as cmr

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import TRINITY modules
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_powerlaw,
    compute_rCloud_homogeneous,
)
from src.cloud_properties.bonnorEbertSphere import (
    solve_lane_emden,
    XI_CRITICAL,
    OMEGA_CRITICAL,
)

# Style
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))

# =============================================================================
# Configuration
# =============================================================================

# Parameter ranges
M_CLOUD_RANGE = np.logspace(4, 8, 50)  # 10^4 to 10^8 Msun
N_CORE_RANGE = np.logspace(2, 6, 50)   # 10^2 to 10^6 cm^-3
R_CORE_RANGE = np.linspace(0.01, 5.0, 100)  # 0.01 to 5 pc

# Physical constraints
R_CLOUD_MAX = 200.0  # pc - typical GMC limit
N_ISM = 1.0          # cm^-3 - ISM density
MASS_TOLERANCE = 0.001  # 0.1% mass error tolerance
MU = 1.4             # mean molecular weight

# Power-law exponents to test
ALPHA_VALUES = [-0.5, -1, -1.5, -2]

# Bonnor-Ebert dimensionless radii to test (including critical ~6.45)
XI_VALUES = [3.0, 4.0, XI_CRITICAL, 10]

# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True

# Contours
CONTOUR_FONT = 17
CONTOUR_N = 5
COLOUR_MAP = cmr.lavender.copy()


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


def find_valid_rCore_powerlaw(mCloud, nCore, alpha, nISM=N_ISM,
                               rCloud_max=R_CLOUD_MAX, mu=MU,
                               rCore_range=R_CORE_RANGE):
    """
    Find valid rCore values for given (mCloud, nCore, alpha).

    Returns (min_rCore, rCloud_at_min_rCore), or (NaN, NaN) if none exist.
    """
    valid_pairs = []

    for rCore in rCore_range:
        try:
            # Compute rCloud for this rCore
            if alpha == 0:
                rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu)
            else:
                rCloud, _ = compute_rCloud_powerlaw(mCloud, nCore, alpha,
                                                     rCore=rCore, mu=mu)

            # Check constraint 1: rCloud <= rCloud_max
            if rCloud > rCloud_max:
                continue

            # Check constraint 2: nEdge >= nISM
            if alpha == 0:
                nEdge = nCore
            else:
                nEdge = nCore * (rCloud / rCore) ** alpha

            if nEdge < nISM:
                continue

            # Check constraint 3: Mass error <= tolerance
            M_computed = compute_mass_powerlaw(rCloud, rCore, nCore, alpha, mu)
            mass_error = abs(M_computed - mCloud) / mCloud if mCloud > 0 else 0

            if mass_error > MASS_TOLERANCE:
                continue

            # All constraints passed
            valid_pairs.append((rCore, rCloud))

        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    if valid_pairs:
        min_pair = min(valid_pairs, key=lambda x: x[0])
        return min_pair  # (min_rCore, rCloud)
    else:
        return (np.nan, np.nan)


def compute_valid_rCore_grid_powerlaw(alpha):
    """
    Compute 2D grids of valid rCore and rCloud values for power-law profile.

    Returns (grid_rCore, grid_rCloud).
    """
    grid_rCore = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)
    grid_rCloud = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)

    for i, nCore in enumerate(N_CORE_RANGE):
        for j, mCloud in enumerate(M_CLOUD_RANGE):
            rc, rcl = find_valid_rCore_powerlaw(mCloud, nCore, alpha)
            grid_rCore[i, j] = rc
            grid_rCloud[i, j] = rcl

    return grid_rCore, grid_rCloud


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

        # Check constraint 1: rCloud <= rCloud_max
        if rCloud > rCloud_max:
            return (np.nan, np.nan)

        # Check constraint 2: nEdge >= nISM
        # nEdge = nCore / Omega (since ρ_edge/ρc = 1/Omega)
        nEdge = nCore / Omega
        if nEdge < nISM:
            return (np.nan, np.nan)

        # Scale length a — clamp to valid range
        if a_pc < R_CORE_RANGE[0] or a_pc > R_CORE_RANGE[-1]:
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
    Create figure showing valid parameter space for power-law alpha values.

    Fill colour = rCloud [pc],  contour lines = rCore [pc].
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle('Valid Parameter Space for Power-Law Density Profiles\n'
                 r'Fill: $r_\mathrm{cloud}$ [pc] \quad '
                 r'Contours: $r_\mathrm{core}$ [pc]',
                 fontsize=12, fontweight='bold')

    # Colourmap for rCloud fill
    vmin_rCloud, vmax_rCloud = 0.1, 200.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('white', 1.0)

    # Contour levels for rCore lines
    contour_levels_rCore = np.logspace(np.log10(0.01), np.log10(5.0), CONTOUR_N)

    for idx, alpha in enumerate(ALPHA_VALUES):
        ax = axes[idx]

        print(f"Computing grid for \u03b1 = {alpha}...")
        grid_rCore, grid_rCloud = compute_valid_rCore_grid_powerlaw(alpha)

        # Count valid cells (either grid; both are NaN at the same places)
        n_valid = np.sum(~np.isnan(grid_rCloud))
        n_total = grid_rCloud.size

        # Fill colour: rCloud
        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid_rCloud,
            cmap=cmap, norm=LogNorm(vmin=vmin_rCloud, vmax=vmax_rCloud),
            shading='auto'
        )

        # Boundary contour (valid / invalid)
        boundary_mask = np.isnan(grid_rCloud).astype(float)
        ax.contour(
            M_CLOUD_RANGE, N_CORE_RANGE, boundary_mask,
            levels=[0.5], colors=['k'], linewidths=1.5, linestyles='-'
        )

        # White contour lines: rCore
        grid_rCore_masked = np.ma.masked_invalid(grid_rCore)
        if n_valid > 0:
            try:
                cs = ax.contour(
                    M_CLOUD_RANGE, N_CORE_RANGE, grid_rCore_masked,
                    levels=contour_levels_rCore,
                    colors='white', linewidths=0.8, alpha=0.8
                )
                texts = ax.clabel(
                    cs, inline=True, fontsize=CONTOUR_FONT, fmt='%.2f',
                    inline_spacing=3, rightside_up=True, use_clabeltext=True
                )
                for t in texts:
                    t.set_rotation_mode("anchor")
                    t.set_path_effects([
                        patheffects.Stroke(linewidth=2, foreground='black'),
                        patheffects.Normal()
                    ])
            except ValueError:
                pass

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]')
        ax.set_ylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]')

        if alpha == 0:
            title = r'$\alpha = 0$ (homogeneous)'
        else:
            title = rf'$\alpha = {alpha}$'
        ax.set_title(f'{title}\n({n_valid}/{n_total} valid cells)', fontsize=10)

        fig.colorbar(im, ax=ax, label=r'$r_\mathrm{cloud}$ [pc]')

    plt.tight_layout()

    if SAVE_PDF:
        out_pdf = FIG_DIR / "paper_AllowedGMC_PowerLaw.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
        print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_BE_grids():
    """
    Create figure showing valid parameter space for Bonnor-Ebert xi values.

    Fill colour = rCloud [pc],  contour lines = scale length *a* [pc].
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle('Valid Parameter Space for Bonnor-Ebert Density Profiles\n'
                 r'Fill: $r_\mathrm{cloud}$ [pc] \quad '
                 r'Contours: scale length $a$ [pc]',
                 fontsize=12, fontweight='bold')

    # Colourmap for rCloud fill
    vmin_rCloud, vmax_rCloud = 0.1, 200.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('white', 1.0)

    # Contour levels for scale-length a
    contour_levels_a = np.logspace(np.log10(0.01), np.log10(5.0), CONTOUR_N)

    for idx, xi in enumerate(XI_VALUES):
        ax = axes[idx]

        print(f"Computing grid for \u03be = {xi:.2f}...")
        grid_a, grid_rCloud = compute_valid_rCore_grid_BE(xi)

        # Count valid cells
        n_valid = np.sum(~np.isnan(grid_rCloud))
        n_total = grid_rCloud.size

        # Fill colour: rCloud
        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid_rCloud,
            cmap=cmap, norm=LogNorm(vmin=vmin_rCloud, vmax=vmax_rCloud),
            shading='auto'
        )

        # Boundary contour (valid / invalid)
        boundary_mask = np.isnan(grid_rCloud).astype(float)
        ax.contour(
            M_CLOUD_RANGE, N_CORE_RANGE, boundary_mask,
            levels=[0.5], colors=['k'], linewidths=1.5, linestyles='-'
        )

        # White contour lines: scale length a
        grid_a_masked = np.ma.masked_invalid(grid_a)
        if n_valid > 0:
            try:
                cs = ax.contour(
                    M_CLOUD_RANGE, N_CORE_RANGE, grid_a_masked,
                    levels=contour_levels_a,
                    colors='white', linewidths=0.8, alpha=0.8
                )
                texts = ax.clabel(
                    cs, inline=True, fontsize=CONTOUR_FONT, fmt='%.2f',
                    inline_spacing=3, rightside_up=True, use_clabeltext=True
                )
                for t in texts:
                    t.set_rotation_mode("anchor")
                    t.set_path_effects([
                        patheffects.Stroke(linewidth=2, foreground='black'),
                        patheffects.Normal()
                    ])
            except ValueError:
                pass

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]')
        ax.set_ylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]')

        if abs(xi - XI_CRITICAL) < 0.01:
            title = rf'$\xi = {xi:.2f}$ (critical)'
        else:
            title = rf'$\xi = {xi:.1f}$'
        ax.set_title(f'{title}\n({n_valid}/{n_total} valid cells)', fontsize=10)

        fig.colorbar(im, ax=ax, label=r'$r_\mathrm{cloud}$ [pc]')

    plt.tight_layout()

    if SAVE_PDF:
        out_pdf = FIG_DIR / "paper_AllowedGMC_BonnorEbert.pdf"
        fig.savefig(out_pdf, bbox_inches='tight')
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
