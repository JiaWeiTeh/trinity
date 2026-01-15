#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of Power-Law cloud radius as function of (M_cloud, n_core) for various alpha.

Creates 2D colormap plots showing how cloud radius varies with:
- Cloud mass (x-axis)
- Core density (y-axis)
- Radius shown as color

For power-law density profile:
    n(r) = nCore                    for r ≤ rCore
    n(r) = nCore × (r/rCore)^α      for rCore < r ≤ rCloud
    n(r) = nISM                     for r > rCloud

Physics:
========
For α = 0 (homogeneous): rCloud = (3M/(4πρ))^(1/3)
For α ≠ 0: M = 4πρ[rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]

Author: Claude Code
Date: 2026-01-12
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
import cmasher as cmr
import sys
import os


_style_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', '_plots', 'trinity.mplstyle')
plt.style.use(_style_path)


# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
_functions_dir = os.path.join(_project_root, 'src', '_functions')

if _functions_dir not in sys.path:
    sys.path.insert(0, _functions_dir)

from unit_conversions import CGS, INV_CONV

# Physical constants
M_H_CGS = CGS.m_H                # [g] hydrogen mass
PC_TO_CM = INV_CONV.pc2cm        # [cm/pc]
MSUN_TO_G = INV_CONV.Msun2g      # [g/Msun]

# Conversion: n [cm⁻³] × μ → ρ [Msun/pc³]
DENSITY_CONVERSION = M_H_CGS * PC_TO_CM**3 / MSUN_TO_G



def compute_rCloud_homogeneous(M_cloud, nCore, mu=2.33):
    """
    Compute cloud radius for homogeneous (α=0) profile.

    M = (4/3)πr³ρ → r = (3M/(4πρ))^(1/3)

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm^-3]
    mu : float
        Mean molecular weight

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    """
    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc³
    rCloud = (3 * M_cloud / (4 * np.pi * rhoCore)) ** (1.0/3.0)
    return rCloud


def compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore_fraction=0.1, mu=2.33):
    """
    Compute cloud radius for power-law profile.

    M = 4πρc [rCore³/3 + (rCloud^(3+α) - rCore^(3+α))/((3+α)×rCore^α)]

    Solves for rCloud given M, nCore, alpha, and rCore = rCore_fraction × rCloud.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm^-3]
    alpha : float
        Power-law exponent (typically negative)
    rCore_fraction : float
        Ratio rCore/rCloud, default 0.1
    mu : float
        Mean molecular weight

    Returns
    -------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]
    """
    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(M_cloud, nCore, mu)
        rCore = rCloud * rCore_fraction
        return rCloud, rCore

    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc³

    def mass_residual(rCloud):
        """Residual: M_computed - M_target"""
        rCore = rCloud * rCore_fraction

        # Mass formula for power-law: Eq 25 in Rahner+ 2018
        M_computed = 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0 +
            (rCloud**(3.0 + alpha) - rCore**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore**alpha)
        )
        return M_computed - M_cloud

    # Initial bracket: use homogeneous estimate as starting point
    rCloud_homo = compute_rCloud_homogeneous(M_cloud, nCore, mu)

    # For α < 0, cloud is larger than homogeneous (less mass in outer regions)
    # Search in range [0.1 × rCloud_homo, 10 × rCloud_homo]
    r_min = 0.1 * rCloud_homo
    r_max = 10.0 * rCloud_homo

    try:
        rCloud = brentq(mass_residual, r_min, r_max, xtol=1e-10)
    except ValueError:
        # If brentq fails, use homogeneous approximation
        rCloud = rCloud_homo

    rCore = rCloud * rCore_fraction
    return rCloud, rCore


def compute_forbidden_zones(M_values, n_core_values, alpha, rCore_fraction=0.1,
                            mu=2.33, nISM=1.0, r_max=200.0, mass_tol=0.001):
    """
    Compute boolean mask for forbidden parameter combinations.

    Parameters
    ----------
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    alpha : float
        Power-law exponent
    rCore_fraction : float
        Ratio rCore/rCloud
    mu : float
        Mean molecular weight
    nISM : float
        ISM density for edge constraint [cm^-3] (default 1.0)
    r_max : float
        Maximum cloud radius [pc] (default 200)
    mass_tol : float
        Maximum mass error tolerance (default 0.1% = 0.001)

    Returns
    -------
    dict with 2D boolean arrays (same shape as radius grid):
        'radius': rCloud > r_max
        'density': nEdge < nISM
        'mass': mass error > mass_tol
        'any': union of all constraints
    """
    n_n = len(n_core_values)
    n_m = len(M_values)

    forbidden_radius = np.zeros((n_n, n_m), dtype=bool)
    forbidden_density = np.zeros((n_n, n_m), dtype=bool)
    forbidden_mass = np.zeros((n_n, n_m), dtype=bool)

    for i, nCore in enumerate(n_core_values):
        for j, M_cloud in enumerate(M_values):
            # Compute rCloud
            if alpha == 0:
                rCloud = compute_rCloud_homogeneous(M_cloud, nCore, mu)
                rCore = rCloud * rCore_fraction
                nEdge = nCore
            else:
                rCloud, rCore = compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore_fraction, mu)
                nEdge = nCore * (rCloud / rCore) ** alpha

            # Check constraints
            forbidden_radius[i, j] = rCloud > r_max
            forbidden_density[i, j] = nEdge < nISM

            # Mass validation (compute M at rCloud and compare)
            rhoCore = nCore * mu * DENSITY_CONVERSION
            if alpha == 0:
                M_computed = (4.0/3.0) * np.pi * rCloud**3 * rhoCore
            else:
                M_computed = 4.0 * np.pi * rhoCore * (
                    rCore**3 / 3.0 +
                    (rCloud**(3.0 + alpha) - rCore**(3.0 + alpha)) /
                    ((3.0 + alpha) * rCore**alpha)
                )
            mass_error = abs(M_computed - M_cloud) / M_cloud if M_cloud > 0 else 0
            forbidden_mass[i, j] = mass_error > mass_tol

    forbidden_any = forbidden_radius | forbidden_density | forbidden_mass

    return {
        'radius': forbidden_radius,
        'density': forbidden_density,
        'mass': forbidden_mass,
        'any': forbidden_any
    }


def compute_radius_grid_powerlaw(M_values, n_core_values, alpha, rCore_fraction=0.1, mu=2.33):
    """
    Compute power-law cloud radius for grid of (M_cloud, n_core) values.

    Parameters
    ----------
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    alpha : float
        Power-law exponent
    rCore_fraction : float
        Ratio rCore/rCloud
    mu : float
        Mean molecular weight

    Returns
    -------
    r_out_grid : ndarray
        2D array of shape (len(n_core_values), len(M_values)) containing rCloud [pc]
    """
    r_out_grid = np.zeros((len(n_core_values), len(M_values)))

    for i, n_core in enumerate(n_core_values):
        for j, M_cloud in enumerate(M_values):
            if alpha == 0:
                rCloud = compute_rCloud_homogeneous(M_cloud, n_core, mu)
            else:
                rCloud, _ = compute_rCloud_powerlaw(M_cloud, n_core, alpha, rCore_fraction, mu)
            r_out_grid[i, j] = rCloud

    return r_out_grid


def plot_radius_heatmap_powerlaw(ax, M_values, n_core_values, r_out_grid, alpha,
                                  vmin=None, vmax=None, contour_levels=shared_levels,
                                  show_forbidden=True, nISM=1.0, r_max=200.0,
                                  rCore_fraction=0.1, mu=2.33):
    """
    Create 2D colormap of rCloud vs (M_cloud, n_core) for power-law profile.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to plot on
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    r_out_grid : ndarray
        2D array of radius values [pc]
    alpha : float
        Power-law exponent (for title)
    vmin, vmax : float, optional
        Color scale limits
    contour_levels : array-like, optional
        Contour levels for radius lines. If None, computed from local data.
    show_forbidden : bool, optional
        Whether to overlay forbidden zones (default True)
    nISM : float, optional
        ISM density for edge constraint [cm^-3] (default 1.0)
    r_max : float, optional
        Maximum cloud radius [pc] (default 200)
    rCore_fraction : float, optional
        Ratio rCore/rCloud (default 0.1)
    mu : float, optional
        Mean molecular weight (default 2.33)

    Returns
    -------
    pcm : QuadMesh
        The pcolormesh object (for colorbar)
    """
    # IMPORTANT: set log scales BEFORE drawing contour + labels
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Interpolate to finer grid for smooth visualization
    M_log = np.log10(M_values)
    n_log = np.log10(n_core_values)
    r_log = np.log10(r_out_grid)

    # Create fine grids (100 x 50 points for smooth appearance)
    M_fine_log = np.linspace(M_log.min(), M_log.max(), 100)
    n_fine_log = np.linspace(n_log.min(), n_log.max(), 50)
    M_fine = 10**M_fine_log
    n_fine = 10**n_fine_log

    # Interpolate radius data in log-log space
    interp = RectBivariateSpline(n_log, M_log, r_log, kx=3, ky=3)
    r_fine_log = interp(n_fine_log, M_fine_log)
    r_fine = 10**r_fine_log

    # Use provided limits or compute from data
    if vmin is None:
        vmin = r_fine.min()
    if vmax is None:
        vmax = r_fine.max()

    # Plot colormap with smooth shading
    pcm = ax.pcolormesh(
        M_fine, n_fine, r_fine,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='cmr.rainforest',
        shading='gouraud'
    )

    # Add smooth contour lines on fine grid
    M_fine_grid, n_fine_grid = np.meshgrid(M_fine, n_fine)

    # Use provided contour levels or compute from local range
    if contour_levels is None:
        r_min_local, r_max_local = r_fine.min(), r_fine.max()
        contour_levels = np.logspace(np.log10(r_min_local), np.log10(r_max_local), 6)

    contours = ax.contour(
        M_fine_grid, n_fine_grid, r_fine,
        levels=contour_levels,
        colors='white', linewidths=0.8, alpha=0.8
    )

    texts = ax.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        fontsize=FONTSIZE,
        fmt='%.1f',
        rightside_up=True,
        use_clabeltext=True,
    )

    for t in texts:
        t.set_rotation_mode("anchor")
        t.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

    # Title with alpha value
    if alpha == 0:
        title = r'$\alpha = 0$ (homogeneous)'
    else:
        title = fr'$\alpha = {alpha}$'
    ax.set_title(title, fontsize=FONTSIZE)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Overlay forbidden zones if requested
    if show_forbidden:
        # Compute forbidden zones on original grid
        forbidden = compute_forbidden_zones(
            M_values, n_core_values, alpha,
            rCore_fraction=rCore_fraction, mu=mu,
            nISM=nISM, r_max=r_max
        )

        # Interpolate forbidden mask to fine grid
        # Convert boolean to float for interpolation
        forbidden_float = forbidden['any'].astype(float)

        # Interpolate in log-log space
        interp_forbidden = RectBivariateSpline(
            n_log, M_log, forbidden_float, kx=1, ky=1
        )
        forbidden_fine = interp_forbidden(n_fine_log, M_fine_log)

        # Threshold at 0.5 to get smooth boundary
        forbidden_mask = forbidden_fine > 0.5

        # Only draw if there are forbidden regions
        if np.any(forbidden_mask):
            # Overlay forbidden zones with white fill
            ax.contourf(
                M_fine_grid, n_fine_grid, forbidden_fine,
                levels=[0.5, 1.5],
                colors='white',
                alpha=0.75,
                zorder=5
            )

            # Add dashed boundary line
            ax.contour(
                M_fine_grid, n_fine_grid, forbidden_fine,
                levels=[0.5],
                colors='black',
                linestyles='dashed',
                linewidths=1.5,
                zorder=6
            )

            # Add "Forbidden" label in the center of forbidden region
            # Find centroid of forbidden region
            forbidden_indices = np.where(forbidden_mask)
            if len(forbidden_indices[0]) > 0:
                # Get center in index space
                center_i = int(np.mean(forbidden_indices[0]))
                center_j = int(np.mean(forbidden_indices[1]))
                # Convert to data coordinates
                x_center = M_fine[center_j]
                y_center = n_fine[center_i]
                ax.annotate(
                    'Forbidden',
                    xy=(x_center, y_center),
                    fontsize=FONTSIZE - 1,
                    ha='center', va='center',
                    color='gray',
                    style='italic',
                    zorder=7
                )

    return pcm


def main():
    """Main function to generate power-law radius visualization for multiple alpha values."""
    print("=" * 60)
    print("Power-Law Cloud Radius Visualization")
    print("=" * 60)

    # Define parameter grid
    n_core_values = np.array([1e2, 3e2, 1e3, 3e3, 1e4])
    M_values = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8])

    # Alpha values to plot
    alpha_values = [0, -0.5, -1, -2]

    rCore_fraction = 0.1  # rCore = 10% of rCloud

    print(f"\nComputing radius grids for:")
    print(f"  n_core: {n_core_values} cm^-3")
    print(f"  M_cloud: {M_values} Msun")
    print(f"  alpha values: {alpha_values}")
    print(f"  rCore/rCloud: {rCore_fraction}")
    print(f"  Grid size: {len(n_core_values)} x {len(M_values)} = {len(n_core_values) * len(M_values)} points per alpha")

    # Compute radius grids for each alpha
    grids = {}
    for alpha in alpha_values:
        print(f"\n  Computing alpha = {alpha}...")
        grids[alpha] = compute_radius_grid_powerlaw(M_values, n_core_values, alpha, rCore_fraction)
        print(f"    Radius range: {grids[alpha].min():.2f} - {grids[alpha].max():.2f} pc")

    # Find global min/max for consistent color scale
    all_radii = np.concatenate([g.flatten() for g in grids.values()])
    vmin, vmax = all_radii.min(), all_radii.max()
    print(f"\n  Global radius range: {vmin:.2f} - {vmax:.2f} pc")


    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    print("\nGenerating plots...")

    # Forbidden zone parameters
    nISM = 1.0      # ISM density [cm^-3]
    r_max = 200.0   # Maximum cloud radius [pc]
    mu = 2.33       # Mean molecular weight

    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        pcm = plot_radius_heatmap_powerlaw(
            ax, M_values, n_core_values, grids[alpha], alpha,
            vmin=vmin, vmax=vmax, contour_levels=shared_levels,
            show_forbidden=True, nISM=nISM, r_max=r_max,
            rCore_fraction=rCore_fraction, mu=mu
        )

        # Only add axis labels on edges
        if idx >= 2:  # Bottom row
            ax.set_xlabel(r'Cloud Mass $M_{\rm cloud}$ [M$_\odot$]', fontsize=FONTSIZE)
        if idx % 2 == 0:  # Left column
            ax.set_ylabel(r'Core Density $n_{\rm core}$ [cm$^{-3}$]', fontsize=FONTSIZE)

    # Add single colorbar on the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label(r'Cloud Radius $r_{\rm cloud}$ [pc]', fontsize=FONTSIZE)

    # Main title
    fig.suptitle(r'Power-Law Density Profile: Cloud Radius vs $(M_{\rm cloud}, n_{\rm core})$',
                 fontsize=FONTSIZE, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # Save figure
    output_file = os.path.join(_script_dir, 'powerlaw_radius_map.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Radius at (M=1e6 Msun, nCore=1e3 cm^-3)")
    print("=" * 60)
    print(f"  {'Alpha':<10} {'rCloud [pc]':>15}")
    print(f"  {'-'*10} {'-'*15}")

    # Find index for M=1e6, nCore=1e3
    M_idx = np.argmin(np.abs(M_values - 1e6))
    n_idx = np.argmin(np.abs(n_core_values - 1e3))

    for alpha in alpha_values:
        r = grids[alpha][n_idx, M_idx]
        alpha_str = "0 (homog)" if alpha == 0 else str(alpha)
        print(f"  {alpha_str:<10} {r:>15.2f}")

    print("\nDone!")

    return grids, fig, axes


if __name__ == "__main__":
    grids, fig, axes = main()
