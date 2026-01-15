#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of valid rCore values as function of (M_cloud, n_core) for various alpha.

Creates 2D colormap plots showing:
- Cloud mass (x-axis)
- Core density (y-axis)
- Valid rCore values shown as color (0.01 - 5 pc range)
- Forbidden zones (white regions) where no valid rCore exists

This visualization helps identify which parameter combinations (mCloud, nCore, rCore)
can produce valid simulations satisfying all constraints:
- rCloud <= 200 pc (typical GMC limit)
- nEdge >= nISM (edge density above ISM)
- Mass error <= 0.1% (self-consistent parameters)

For power-law density profile:
    n(r) = nCore                    for r <= rCore
    n(r) = nCore * (r/rCore)^alpha  for rCore < r <= rCloud
    n(r) = nISM                     for r > rCloud

Date: 2026-01-15
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

# Conversion: n [cm^-3] * mu -> rho [Msun/pc^3]
DENSITY_CONVERSION = M_H_CGS * PC_TO_CM**3 / MSUN_TO_G

# Plot settings
FONTSIZE = 12

# Shared contour levels for rCore (reduced to avoid clutter)
shared_levels = np.array([0.1, 0.5, 1.0, 2.0, 5.0])


def compute_rCloud_homogeneous(M_cloud, nCore, mu=2.33):
    """
    Compute cloud radius for homogeneous (alpha=0) profile.

    M = (4/3)*pi*r^3*rho  ->  r = (3M/(4*pi*rho))^(1/3)

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
    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc^3
    rCloud = (3 * M_cloud / (4 * np.pi * rhoCore)) ** (1.0/3.0)
    return rCloud


def compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore, mu=2.33):
    """
    Compute cloud radius for power-law profile with fixed rCore.

    M = 4*pi*rhoCore * [rCore^3/3 + (rCloud^(3+alpha) - rCore^(3+alpha))/((3+alpha)*rCore^alpha)]

    Solves for rCloud given M, nCore, alpha, and rCore.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm^-3]
    alpha : float
        Power-law exponent (typically negative)
    rCore : float
        Core radius [pc]
    mu : float
        Mean molecular weight

    Returns
    -------
    rCloud : float
        Cloud radius [pc], or NaN if no solution
    """
    if alpha == 0:
        return compute_rCloud_homogeneous(M_cloud, nCore, mu)

    rhoCore = nCore * mu * DENSITY_CONVERSION  # Msun/pc^3

    def mass_residual(rCloud):
        """Residual: M_computed - M_target"""
        if rCloud <= rCore:
            return -M_cloud  # rCloud must be > rCore

        # Mass formula for power-law: Eq 25 in Rahner+ 2018
        M_computed = 4.0 * np.pi * rhoCore * (
            rCore**3 / 3.0 +
            (rCloud**(3.0 + alpha) - rCore**(3.0 + alpha)) /
            ((3.0 + alpha) * rCore**alpha)
        )
        return M_computed - M_cloud

    # Initial bracket: use homogeneous estimate as starting point
    rCloud_homo = compute_rCloud_homogeneous(M_cloud, nCore, mu)

    # Search range depends on alpha
    r_min = rCore * 1.001  # Must be > rCore
    r_max = max(10.0 * rCloud_homo, 500.0)  # Upper bound

    try:
        # Check if solution exists in bracket
        f_min = mass_residual(r_min)
        f_max = mass_residual(r_max)

        if f_min * f_max > 0:
            # No sign change - no solution in bracket
            return np.nan

        rCloud = brentq(mass_residual, r_min, r_max, xtol=1e-10)
        return rCloud
    except (ValueError, RuntimeError):
        return np.nan


def check_constraints(rCloud, rCore, nCore, nISM, alpha, M_cloud, mu,
                      r_max=200.0, mass_tol=0.001):
    """
    Check if parameters satisfy all physical constraints.

    Parameters
    ----------
    rCloud : float
        Cloud radius [pc]
    rCore : float
        Core radius [pc]
    nCore : float
        Core number density [cm^-3]
    nISM : float
        ISM density [cm^-3]
    alpha : float
        Power-law exponent
    M_cloud : float
        Cloud mass [Msun]
    mu : float
        Mean molecular weight
    r_max : float
        Maximum cloud radius [pc]
    mass_tol : float
        Maximum mass error tolerance

    Returns
    -------
    valid : bool
        True if all constraints satisfied
    """
    if np.isnan(rCloud):
        return False

    # 1. Radius constraint
    if rCloud > r_max:
        return False

    # 2. Edge density constraint
    if alpha == 0:
        nEdge = nCore
    else:
        nEdge = nCore * (rCloud / rCore) ** alpha

    if nEdge < nISM:
        return False

    # 3. Mass consistency check
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
    if mass_error > mass_tol:
        return False

    return True


def find_valid_rCore(M_cloud, nCore, alpha, rCore_range, nISM=1.0, r_max=200.0,
                     mu=2.33, mass_tol=0.001):
    """
    Find a valid rCore value for given (M_cloud, nCore, alpha).

    Searches through rCore_range to find the first valid rCore that satisfies
    all constraints.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    nCore : float
        Core number density [cm^-3]
    alpha : float
        Power-law exponent
    rCore_range : array-like
        Array of rCore values to test [pc]
    nISM : float
        ISM density [cm^-3]
    r_max : float
        Maximum cloud radius [pc]
    mu : float
        Mean molecular weight
    mass_tol : float
        Maximum mass error tolerance

    Returns
    -------
    rCore_valid : float
        First valid rCore value, or NaN if none found
    rCloud_valid : float
        Corresponding rCloud value
    """
    for rCore in rCore_range:
        rCloud = compute_rCloud_powerlaw(M_cloud, nCore, alpha, rCore, mu)

        if check_constraints(rCloud, rCore, nCore, nISM, alpha, M_cloud, mu,
                            r_max, mass_tol):
            return rCore, rCloud

    return np.nan, np.nan


def compute_rCore_grid(M_values, n_core_values, alpha, rCore_range,
                       nISM=1.0, r_max=200.0, mu=2.33, mass_tol=0.001):
    """
    Compute valid rCore values for grid of (M_cloud, n_core) values.

    Parameters
    ----------
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    alpha : float
        Power-law exponent
    rCore_range : array-like
        Array of rCore values to search [pc]
    nISM : float
        ISM density [cm^-3]
    r_max : float
        Maximum cloud radius [pc]
    mu : float
        Mean molecular weight
    mass_tol : float
        Maximum mass error tolerance

    Returns
    -------
    rCore_grid : ndarray
        2D array of valid rCore values [pc], NaN where no valid rCore exists
    rCloud_grid : ndarray
        2D array of corresponding rCloud values [pc]
    """
    n_n = len(n_core_values)
    n_m = len(M_values)

    rCore_grid = np.full((n_n, n_m), np.nan)
    rCloud_grid = np.full((n_n, n_m), np.nan)

    for i, nCore in enumerate(n_core_values):
        for j, M_cloud in enumerate(M_values):
            rCore_valid, rCloud_valid = find_valid_rCore(
                M_cloud, nCore, alpha, rCore_range,
                nISM=nISM, r_max=r_max, mu=mu, mass_tol=mass_tol
            )
            rCore_grid[i, j] = rCore_valid
            rCloud_grid[i, j] = rCloud_valid

    return rCore_grid, rCloud_grid


def plot_rCore_heatmap(ax, M_values, n_core_values, rCore_grid, alpha,
                       vmin=0.01, vmax=5.0, contour_levels=None):
    """
    Create 2D colormap of valid rCore vs (M_cloud, n_core) for power-law profile.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes to plot on
    M_values : array-like
        Cloud mass values [Msun]
    n_core_values : array-like
        Core number density values [cm^-3]
    rCore_grid : ndarray
        2D array of rCore values [pc], NaN for forbidden regions
    alpha : float
        Power-law exponent (for title)
    vmin, vmax : float
        Color scale limits [pc]
    contour_levels : array-like, optional
        Contour levels for rCore lines

    Returns
    -------
    pcm : QuadMesh or None
        The pcolormesh object (for colorbar), None if all NaN
    """
    # IMPORTANT: set log scales BEFORE drawing contour + labels
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Check if we have any valid data
    if np.all(np.isnan(rCore_grid)):
        # All forbidden - fill with white and label
        ax.fill_between(
            [M_values.min(), M_values.max()],
            n_core_values.min(), n_core_values.max(),
            color='white', alpha=0.9
        )
        ax.text(
            np.sqrt(M_values.min() * M_values.max()),
            np.sqrt(n_core_values.min() * n_core_values.max()),
            'All Forbidden',
            ha='center', va='center',
            fontsize=FONTSIZE, color='gray', style='italic'
        )
        ax.set_xlim(M_values.min(), M_values.max())
        ax.set_ylim(n_core_values.min(), n_core_values.max())

        if alpha == 0:
            title = r'$\alpha = 0$ (homogeneous)'
        else:
            title = fr'$\alpha = {alpha}$'
        ax.set_title(title, fontsize=FONTSIZE)
        ax.grid(True, alpha=0.3, linestyle='--')
        return None

    # Interpolate to finer grid for smooth visualization
    M_log = np.log10(M_values)
    n_log = np.log10(n_core_values)

    # Create fine grids (higher resolution for smoother visualization)
    M_fine_log = np.linspace(M_log.min(), M_log.max(), 300)
    n_fine_log = np.linspace(n_log.min(), n_log.max(), 150)
    M_fine = 10**M_fine_log
    n_fine = 10**n_fine_log

    # Handle NaN values for interpolation
    # Replace NaN with -999 for interpolation, then mask later
    rCore_for_interp = np.where(np.isnan(rCore_grid), -999, np.log10(rCore_grid + 1e-10))

    # Interpolate in log-log space (cubic for smoother boundaries)
    interp = RectBivariateSpline(n_log, M_log, rCore_for_interp, kx=3, ky=3)
    rCore_fine_log = interp(n_fine_log, M_fine_log)

    # Restore NaN mask
    forbidden_mask = rCore_fine_log < -100
    rCore_fine = 10**rCore_fine_log
    rCore_fine[forbidden_mask] = np.nan

    # Create masked array for plotting
    rCore_masked = np.ma.masked_invalid(rCore_fine)

    # Plot colormap with smooth shading
    pcm = ax.pcolormesh(
        M_fine, n_fine, rCore_masked,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='cmr.ember',  # Different colormap to distinguish from rCloud
        shading='gouraud'
    )

    # Add smooth contour lines on fine grid (only for valid regions)
    M_fine_grid, n_fine_grid = np.meshgrid(M_fine, n_fine)

    # Use provided contour levels or defaults
    if contour_levels is None:
        contour_levels = shared_levels

    # Only plot contours where we have valid data
    if not np.all(np.isnan(rCore_fine)):
        try:
            contours = ax.contour(
                M_fine_grid, n_fine_grid, rCore_masked,
                levels=contour_levels,
                colors='white', linewidths=0.8, alpha=0.8
            )

            texts = ax.clabel(
                contours,
                inline=True,
                inline_spacing=3,
                fontsize=FONTSIZE - 2,
                fmt='%.2f',
                rightside_up=True,
                use_clabeltext=True,
            )

            for t in texts:
                t.set_rotation_mode("anchor")
                t.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal()
                ])
        except ValueError:
            pass  # No contour lines to draw

    # Overlay forbidden zones (where rCore is NaN)
    # Create mask: 1 = forbidden, 0 = valid
    forbidden_float = np.where(forbidden_mask, 1.0, 0.0)

    if np.any(forbidden_mask):
        # White fill for forbidden zones
        ax.contourf(
            M_fine_grid, n_fine_grid, forbidden_float,
            levels=[0.5, 1.5],
            colors='white',
            alpha=0.85,
            zorder=5
        )

        # Dashed boundary line
        ax.contour(
            M_fine_grid, n_fine_grid, forbidden_float,
            levels=[0.5],
            colors='black',
            linestyles='dashed',
            linewidths=1.5,
            zorder=6
        )

        # Add "Forbidden" label in center of forbidden region
        forbidden_indices = np.where(forbidden_mask)
        if len(forbidden_indices[0]) > 10:  # Only label if region is large enough
            center_i = int(np.mean(forbidden_indices[0]))
            center_j = int(np.mean(forbidden_indices[1]))
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

    # Title with alpha value
    if alpha == 0:
        title = r'$\alpha = 0$ (homogeneous)'
    else:
        title = fr'$\alpha = {alpha}$'
    ax.set_title(title, fontsize=FONTSIZE)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    return pcm


def main():
    """Main function to generate rCore validity visualization for multiple alpha values."""
    print("=" * 60)
    print("Power-Law Valid rCore Visualization")
    print("=" * 60)

    # Define parameter grid (higher resolution for smoother visualization)
    n_core_values = np.logspace(2, 4, 50)  # 100 to 10000 cm^-3
    M_values = np.logspace(4, 8, 80)       # 10^4 to 10^8 Msun

    # Alpha values to plot
    alpha_values = [0, -0.5, -1, -2]

    # rCore search range: 0.01 to 5 pc
    rCore_range = np.logspace(np.log10(0.01), np.log10(5.0), 50)

    # Physical constraints
    nISM = 1.0      # ISM density [cm^-3]
    r_max = 200.0   # Maximum cloud radius [pc]
    mu = 2.33       # Mean molecular weight
    mass_tol = 0.001  # 0.1% mass tolerance

    print(f"\nSearching for valid rCore values:")
    print(f"  n_core range: {n_core_values.min():.0f} - {n_core_values.max():.0f} cm^-3 ({len(n_core_values)} points)")
    print(f"  M_cloud range: {M_values.min():.0e} - {M_values.max():.0e} Msun ({len(M_values)} points)")
    print(f"  rCore search: {rCore_range.min():.2f} - {rCore_range.max():.2f} pc ({len(rCore_range)} points)")
    print(f"  alpha values: {alpha_values}")
    print(f"\nConstraints:")
    print(f"  rCloud <= {r_max:.0f} pc")
    print(f"  nEdge >= {nISM:.1f} cm^-3")
    print(f"  Mass error <= {mass_tol*100:.1f}%")

    # Compute rCore grids for each alpha
    grids = {}
    for alpha in alpha_values:
        print(f"\n  Computing alpha = {alpha}...")
        rCore_grid, rCloud_grid = compute_rCore_grid(
            M_values, n_core_values, alpha, rCore_range,
            nISM=nISM, r_max=r_max, mu=mu, mass_tol=mass_tol
        )
        grids[alpha] = rCore_grid

        n_valid = np.sum(~np.isnan(rCore_grid))
        n_total = rCore_grid.size
        print(f"    Valid parameter combinations: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        if n_valid > 0:
            print(f"    rCore range: {np.nanmin(rCore_grid):.3f} - {np.nanmax(rCore_grid):.3f} pc")

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    print("\nGenerating plots...")

    pcm_valid = None  # Track a valid pcolormesh for colorbar
    for idx, alpha in enumerate(alpha_values):
        ax = axes[idx]
        pcm = plot_rCore_heatmap(
            ax, M_values, n_core_values, grids[alpha], alpha,
            vmin=0.01, vmax=5.0, contour_levels=shared_levels
        )
        if pcm is not None:
            pcm_valid = pcm

        # Only add axis labels on edges
        if idx >= 2:  # Bottom row
            ax.set_xlabel(r'Cloud Mass $M_{\rm cloud}$ [M$_\odot$]', fontsize=FONTSIZE)
        if idx % 2 == 0:  # Left column
            ax.set_ylabel(r'Core Density $n_{\rm core}$ [cm$^{-3}$]', fontsize=FONTSIZE)

    # Add single colorbar on the right
    if pcm_valid is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm_valid, cax=cbar_ax)
        cbar.set_label(r'Valid Core Radius $r_{\rm core}$ [pc]', fontsize=FONTSIZE)

    # Main title
    fig.suptitle(r'Power-Law Density Profile: Valid $r_{\rm core}$ vs $(M_{\rm cloud}, n_{\rm core})$' + '\n' +
                 r'White regions: no valid $r_{\rm core}$ in [0.01, 5] pc satisfies constraints',
                 fontsize=FONTSIZE, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.88, 0.93])

    # Save figure
    output_file = os.path.join(_script_dir, 'powerlaw_rCore_validity_map.pdf')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Valid rCore fraction by alpha")
    print("=" * 60)
    print(f"  {'Alpha':<12} {'Valid %':>10} {'rCore range [pc]':>20}")
    print(f"  {'-'*12} {'-'*10} {'-'*20}")

    for alpha in alpha_values:
        grid = grids[alpha]
        n_valid = np.sum(~np.isnan(grid))
        n_total = grid.size
        pct = 100 * n_valid / n_total

        if n_valid > 0:
            rCore_str = f"{np.nanmin(grid):.3f} - {np.nanmax(grid):.3f}"
        else:
            rCore_str = "N/A"

        alpha_str = "0 (homog)" if alpha == 0 else str(alpha)
        print(f"  {alpha_str:<12} {pct:>9.1f}% {rCore_str:>20}")

    print("\nDone!")

    return grids, fig, axes


if __name__ == "__main__":
    grids, fig, axes = main()
