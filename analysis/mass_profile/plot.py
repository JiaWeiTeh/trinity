#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of Bonnor-Ebert sphere radius as function of (M_cloud, n_core).

Creates a 2D colormap plot showing how cloud radius varies with:
- Cloud mass (x-axis)
- Core density (y-axis)
- Radius shown as color

Physics:
========
For fixed Omega, the BE sphere radius scales approximately as:
    r_out ~ M_cloud^(1/3) * n_core^(-1/6)

This follows from the mass-radius-density relation:
    M = (4/3)*pi*r^3 * <rho>  where <rho> ~ n_core (average density ~ core density)

Author: Claude Code
Date: 2026-01-12
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from scipy.interpolate import RectBivariateSpline
import cmasher as cmr
import sys
import os


plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=20)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True  # Show minor ticks
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 6        # Major tick size
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3        # Minor tick size
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1       # Major tick width
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.minor.width"] = 0.8     # Minor tick width
plt.rcParams["ytick.minor.width"] = 0.8


# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
_be_dir = os.path.join(_project_root, 'analysis', 'bonnorEbert')
_functions_dir = os.path.join(_project_root, 'src', '_functions')

for _dir in [_be_dir, _functions_dir]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from bonnorEbertSphere_v2 import create_BE_sphere, solve_lane_emden, OMEGA_CRITICAL


def compute_radius_grid(M_values, n_core_values, Omega=8.0, mu=2.33, gamma=5.0/3.0):
    """
    Compute BE sphere radius for grid of (M_cloud, n_core) values.

    Parameters
    ----------
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    Omega : float
        Density contrast (rho_core / rho_surface), default 8.0
    mu : float
        Mean molecular weight, default 2.33
    gamma : float
        Adiabatic index (unused for isothermal BE, but kept for API), default 5/3

    Returns
    -------
    r_out_grid : ndarray
        2D array of shape (len(n_core_values), len(M_values)) containing r_out [pc]
    """
    # Pre-solve Lane-Emden once for efficiency
    solution = solve_lane_emden()

    # Create output grid: rows = n_core, columns = M_cloud
    r_out_grid = np.zeros((len(n_core_values), len(M_values)))

    for i, n_core in enumerate(n_core_values):
        for j, M_cloud in enumerate(M_values):
            result = create_BE_sphere(M_cloud, n_core, Omega, mu, gamma, solution)
            r_out_grid[i, j] = result.r_out

    return r_out_grid


def plot_radius_heatmap(M_values, n_core_values, r_out_grid, Omega=8.0,
                        output_file=None, show=True):
    """
    Create 2D colormap of r_out vs (M_cloud, n_core).

    Parameters
    ----------
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    r_out_grid : ndarray
        2D array of radius values [pc]
    Omega : float
        Density contrast used (for title)
    output_file : str, optional
        Path to save figure (if None, not saved)
    show : bool
        Whether to display the figure

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    # IMPORTANT: set log scales BEFORE drawing contour + labels
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Interpolate to finer grid for smooth visualization
    # Use log-space interpolation for better results in log-log plot
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

    # Plot colormap with smooth shading
    pcm = ax.pcolormesh(
        M_fine, n_fine, r_fine,
        norm=mcolors.LogNorm(vmin=r_fine.min(), vmax=r_fine.max()),
        cmap='cmr.rainforest',
        shading='gouraud'
    )
    
    # Add smooth contour lines on fine grid
    M_fine_grid, n_fine_grid = np.meshgrid(M_fine, n_fine)
    
    
    # Choose contour levels spanning the radius range
    r_min, r_max = r_fine.min(), r_fine.max()
    contour_levels = np.logspace(np.log10(r_min), np.log10(r_max), 8)
    
    contours = ax.contour(
        M_fine_grid, n_fine_grid, r_fine,
        levels=contour_levels,
        colors='white', linewidths=0.8, alpha=0.8
    )
    
    texts = ax.clabel(
        contours,
        inline=True,
        inline_spacing=3,
        fontsize=20,
        fmt='%.1f pc',
        rightside_up=True,
        use_clabeltext=True,   # keeps rotation consistent on resize
    )
    
    for t in texts:
        t.set_rotation_mode("anchor")
        t.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])

    # Labels and formatting
    ax.set_xlabel(r'Cloud Mass $M_{\rm cloud}$ [M$_\odot$]', fontsize=12)
    ax.set_ylabel(r'Core Density $n_{\rm core}$ [cm$^{-3}$]', fontsize=12)
    ax.set_title(fr'Bonnor-Ebert Sphere Radius ($\Omega$ = {Omega})', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(r'Cloud Radius $r_{\rm out}$ [pc]', fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save if requested
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {output_file}")

    if show:
        plt.show()

    return fig, ax


def main():
    """Main function to generate BE sphere radius visualization."""
    print("=" * 60)
    print("Bonnor-Ebert Sphere Radius Visualization")
    print("=" * 60)

    # Define parameter grid (same as test cases for consistency)
    # nCore: 2 values per decade from 1e2 to 1e4
    n_core_values = np.array([1e2, 3e2, 1e3, 3e3, 1e4])

    # M_cloud: 2 values per decade from 1e4 to 1e8
    M_values = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8])

    Omega = OMEGA_CRITICAL  # Critical density contrast (14.04)

    print(f"\nComputing radius grid for:")
    print(f"  n_core: {n_core_values} cm^-3")
    print(f"  M_cloud: {M_values} Msun")
    print(f"  Omega: {Omega}")
    print(f"  Grid size: {len(n_core_values)} x {len(M_values)} = {len(n_core_values) * len(M_values)} points")

    # Compute radius grid
    print("\nComputing BE sphere radii...")
    r_out_grid = compute_radius_grid(M_values, n_core_values, Omega=Omega)

    # Print summary statistics
    print(f"\nRadius range: {r_out_grid.min():.2f} - {r_out_grid.max():.2f} pc")

    # Print radius grid
    print("\nRadius grid [pc]:")
    print("-" * 80)
    header = "n_core\\M_cloud |" + "".join([f"{M:>9.0e}" for M in M_values])
    print(header)
    print("-" * 80)
    for i, n_core in enumerate(n_core_values):
        row = f"{n_core:>12.0e} |" + "".join([f"{r_out_grid[i,j]:>9.2f}" for j in range(len(M_values))])
        print(row)
    print("-" * 80)

    # Create plot
    print("\nGenerating plot...")
    output_file = os.path.join(_script_dir, 'be_sphere_radius_map.pdf')
    fig, ax = plot_radius_heatmap(M_values, n_core_values, r_out_grid,
                                   Omega=Omega, output_file=output_file, show=False)

    print("\nDone!")
    print(f"Output file: {output_file}")

    return r_out_grid, fig, ax


if __name__ == "__main__":
    r_out_grid, fig, ax = main()