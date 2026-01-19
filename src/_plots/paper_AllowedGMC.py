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
ALPHA_VALUES = [0, -1, -1.5, -2]

# Bonnor-Ebert dimensionless radii to test (including critical ~6.45)
XI_VALUES = [3.0, 4.0, XI_CRITICAL, 10]

# Output - save to project root's fig/ directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PDF = True


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

    Returns the range of valid rCore values, or NaN if none exist.
    """
    valid_rCores = []

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
            valid_rCores.append(rCore)

        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    if valid_rCores:
        # Return minimum valid rCore
        return np.min(valid_rCores)
    else:
        return np.nan


def compute_valid_rCore_grid_powerlaw(alpha):
    """
    Compute 2D grid of valid rCore values for power-law profile.
    """
    grid = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)

    for i, nCore in enumerate(N_CORE_RANGE):
        for j, mCloud in enumerate(M_CLOUD_RANGE):
            grid[i, j] = find_valid_rCore_powerlaw(mCloud, nCore, alpha)

    return grid


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
    Find valid rCore (core radius where density is ~constant) for BE sphere.

    For BE sphere, "rCore" is approximated as the characteristic length scale a,
    which is where ξ = 1 (where density starts to deviate from central value).

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
            return np.nan

        # Check constraint 2: nEdge >= nISM
        # nEdge = nCore / Omega (since ρ_edge/ρc = 1/Omega)
        nEdge = nCore / Omega
        if nEdge < nISM:
            return np.nan

        # "rCore" for BE sphere is the characteristic length scale a
        # This is where ξ = 1, i.e., where density starts to deviate from core
        rCore = a_pc

        # Clamp to valid range
        if rCore < R_CORE_RANGE[0] or rCore > R_CORE_RANGE[-1]:
            return np.nan

        return rCore

    except Exception:
        return np.nan


def compute_valid_rCore_grid_BE(xi_out):
    """
    Compute 2D grid of valid rCore values for Bonnor-Ebert profile.
    """
    grid = np.full((len(N_CORE_RANGE), len(M_CLOUD_RANGE)), np.nan)

    for i, nCore in enumerate(N_CORE_RANGE):
        for j, mCloud in enumerate(M_CLOUD_RANGE):
            grid[i, j] = find_valid_rCore_BE(mCloud, nCore, xi_out)

    return grid


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_powerlaw_grids():
    """
    Create figure showing valid rCore for various power-law alpha values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle('Valid Core Radius for Power-Law Density Profiles\n'
                 r'Constraints: $r_\mathrm{cloud} \leq 200$ pc, '
                 r'$n_\mathrm{edge} \geq n_\mathrm{ISM}$, mass error $\leq 0.1\%$',
                 fontsize=12, fontweight='bold')

    # Colormap settings
    vmin, vmax = 0.01, 5.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('lightgray', 0.5)  # Forbidden zones are light gray

    # Contour levels evenly spaced in log
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)

    for idx, alpha in enumerate(ALPHA_VALUES):
        ax = axes[idx]

        print(f"Computing grid for α = {alpha}...")
        grid = compute_valid_rCore_grid_powerlaw(alpha)

        # Count valid cells
        n_valid = np.sum(~np.isnan(grid))
        n_total = grid.size

        # Create forbidden zone mask (1 = forbidden, 0 = valid)
        forbidden_mask = np.isnan(grid).astype(float)

        # Shade forbidden zone with hatching
        ax.contourf(
            M_CLOUD_RANGE, N_CORE_RANGE, forbidden_mask,
            levels=[0.5, 1.5], colors=['lightgray'], alpha=0.5
        )

        # Draw boundary line around forbidden zone
        ax.contour(
            M_CLOUD_RANGE, N_CORE_RANGE, forbidden_mask,
            levels=[0.5], colors=['k'], linewidths=1.5, linestyles='-'
        )

        # Add "Forbidden" label in the forbidden zone
        # Find center of forbidden region for label placement
        forbidden_indices = np.where(forbidden_mask > 0.5)
        if len(forbidden_indices[0]) > 0:
            # Get approximate center of forbidden region
            mid_idx = len(forbidden_indices[0]) // 2
            label_n = N_CORE_RANGE[forbidden_indices[0][mid_idx]]
            label_m = M_CLOUD_RANGE[forbidden_indices[1][mid_idx]]
            ax.text(label_m, label_n, 'Forbidden', fontsize=9, ha='center', va='center',
                    color='dimgray', fontstyle='italic', fontweight='bold')

        # Plot pcolormesh for valid regions
        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid,
            cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            shading='auto'
        )

        # Add contour lines (evenly spaced in log)
        grid_masked = np.ma.masked_invalid(grid)
        if n_valid > 0:
            try:
                cs = ax.contour(
                    M_CLOUD_RANGE, N_CORE_RANGE, grid_masked,
                    levels=contour_levels,
                    colors='white', linewidths=0.8, alpha=0.8
                )
                # Labels with proper spacing and path effects for readability
                texts = ax.clabel(
                    cs, inline=True, fontsize=7, fmt='%.2f',
                    inline_spacing=3, rightside_up=True, use_clabeltext=True
                )
                # Add stroke effect for better visibility
                for t in texts:
                    t.set_rotation_mode("anchor")
                    t.set_path_effects([
                        patheffects.Stroke(linewidth=2, foreground='black'),
                        patheffects.Normal()
                    ])
            except ValueError:
                pass  # No contour lines to draw

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]')
        ax.set_ylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]')

        if alpha == 0:
            title = r'$\alpha = 0$ (homogeneous)'
        else:
            title = rf'$\alpha = {alpha}$'
        ax.set_title(f'{title}\n({n_valid}/{n_total} valid cells)', fontsize=10)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label=r'min $r_\mathrm{core}$ [pc]')

    plt.tight_layout()

    if SAVE_PDF:
        out_path = FIG_DIR / "paper_AllowedGMC_PowerLaw.pdf"
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {out_path}")

    plt.close(fig)


def plot_BE_grids():
    """
    Create figure showing valid rCore for various Bonnor-Ebert xi values.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    fig.suptitle('Valid Core Radius for Bonnor-Ebert Density Profiles\n'
                 r'Constraints: $r_\mathrm{cloud} \leq 200$ pc, '
                 r'$n_\mathrm{edge} \geq n_\mathrm{ISM}$',
                 fontsize=12, fontweight='bold')

    # Colormap settings
    vmin, vmax = 0.01, 5.0
    cmap = cmr.rainforest.copy()
    cmap.set_bad('lightgray', 0.5)  # Forbidden zones are light gray

    # Contour levels evenly spaced in log
    contour_levels = np.logspace(np.log10(vmin), np.log10(vmax), 8)

    for idx, xi in enumerate(XI_VALUES):
        ax = axes[idx]

        print(f"Computing grid for ξ = {xi:.2f}...")
        grid = compute_valid_rCore_grid_BE(xi)

        # Count valid cells
        n_valid = np.sum(~np.isnan(grid))
        n_total = grid.size

        # Create forbidden zone mask (1 = forbidden, 0 = valid)
        forbidden_mask = np.isnan(grid).astype(float)

        # Shade forbidden zone
        ax.contourf(
            M_CLOUD_RANGE, N_CORE_RANGE, forbidden_mask,
            levels=[0.5, 1.5], colors=['lightgray'], alpha=0.5
        )

        # Draw boundary line around forbidden zone
        ax.contour(
            M_CLOUD_RANGE, N_CORE_RANGE, forbidden_mask,
            levels=[0.5], colors=['k'], linewidths=1.5, linestyles='-'
        )

        # Add "Forbidden" label in the forbidden zone
        forbidden_indices = np.where(forbidden_mask > 0.5)
        if len(forbidden_indices[0]) > 0:
            mid_idx = len(forbidden_indices[0]) // 2
            label_n = N_CORE_RANGE[forbidden_indices[0][mid_idx]]
            label_m = M_CLOUD_RANGE[forbidden_indices[1][mid_idx]]
            ax.text(label_m, label_n, 'Forbidden', fontsize=9, ha='center', va='center',
                    color='dimgray', fontstyle='italic', fontweight='bold')

        # Plot pcolormesh for valid regions
        im = ax.pcolormesh(
            M_CLOUD_RANGE, N_CORE_RANGE, grid,
            cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
            shading='auto'
        )

        # Add contour lines (evenly spaced in log)
        grid_masked = np.ma.masked_invalid(grid)
        if n_valid > 0:
            try:
                cs = ax.contour(
                    M_CLOUD_RANGE, N_CORE_RANGE, grid_masked,
                    levels=contour_levels,
                    colors='white', linewidths=0.8, alpha=0.8
                )
                # Labels with proper spacing and path effects for readability
                texts = ax.clabel(
                    cs, inline=True, fontsize=7, fmt='%.2f',
                    inline_spacing=3, rightside_up=True, use_clabeltext=True
                )
                # Add stroke effect for better visibility
                for t in texts:
                    t.set_rotation_mode("anchor")
                    t.set_path_effects([
                        patheffects.Stroke(linewidth=2, foreground='black'),
                        patheffects.Normal()
                    ])
            except ValueError:
                pass  # No contour lines to draw

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$M_\mathrm{cloud}$ [M$_\odot$]')
        ax.set_ylabel(r'$n_\mathrm{core}$ [cm$^{-3}$]')

        if abs(xi - XI_CRITICAL) < 0.01:
            title = rf'$\xi = {xi:.2f}$ (critical)'
        else:
            title = rf'$\xi = {xi:.1f}$'
        ax.set_title(f'{title}\n({n_valid}/{n_total} valid cells)', fontsize=10)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, label=r'min $r_\mathrm{core}$ [pc]')

    plt.tight_layout()

    if SAVE_PDF:
        out_path = FIG_DIR / "paper_AllowedGMC_BonnorEbert.pdf"
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {out_path}")

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
