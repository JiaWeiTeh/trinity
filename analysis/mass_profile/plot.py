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
import sys
import os

# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
_be_dir = os.path.join(_project_root, 'analysis', 'bonnorEbert')
_functions_dir = os.path.join(_project_root, 'src', '_functions')

for _dir in [_be_dir, _functions_dir]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from bonnorEbertSphere_v2 import create_BE_sphere, solve_lane_emden


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

    # Create meshgrid for pcolormesh (need cell edges, not centers)
    # For log scale, we compute geometric means for edges
    M_edges = np.zeros(len(M_values) + 1)
    n_edges = np.zeros(len(n_core_values) + 1)

    # Compute edges as geometric means between consecutive values
    M_log = np.log10(M_values)
    n_log = np.log10(n_core_values)

    M_edges[1:-1] = 10**((M_log[:-1] + M_log[1:]) / 2)
    M_edges[0] = 10**(M_log[0] - (M_log[1] - M_log[0]) / 2)
    M_edges[-1] = 10**(M_log[-1] + (M_log[-1] - M_log[-2]) / 2)

    n_edges[1:-1] = 10**((n_log[:-1] + n_log[1:]) / 2)
    n_edges[0] = 10**(n_log[0] - (n_log[1] - n_log[0]) / 2)
    n_edges[-1] = 10**(n_log[-1] + (n_log[-1] - n_log[-2]) / 2)

    # Plot colormap with logarithmic color scale
    pcm = ax.pcolormesh(M_edges, n_edges, r_out_grid,
                        norm=mcolors.LogNorm(vmin=r_out_grid.min(), vmax=r_out_grid.max()),
                        cmap='viridis', shading='flat')

    # Add contour lines at cell centers
    M_grid, n_grid = np.meshgrid(M_values, n_core_values)

    # Choose contour levels spanning the radius range
    r_min, r_max = r_out_grid.min(), r_out_grid.max()
    contour_levels = np.logspace(np.log10(r_min), np.log10(r_max), 8)

    contours = ax.contour(M_grid, n_grid, r_out_grid,
                          levels=contour_levels,
                          colors='white', linewidths=0.8, alpha=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f pc')

    # Labels and formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
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
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
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

    Omega = 8.0  # Standard density contrast

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
    output_file = os.path.join(_script_dir, 'be_sphere_radius_map.png')
    fig, ax = plot_radius_heatmap(M_values, n_core_values, r_out_grid,
                                   Omega=Omega, output_file=output_file, show=False)

    print("\nDone!")
    print(f"Output file: {output_file}")

    return r_out_grid, fig, ax


if __name__ == "__main__":
    r_out_grid, fig, ax = main()
