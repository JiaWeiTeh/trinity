#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best-fit analysis for Orion Nebula (M42) parameter sweep.

This script analyzes TRINITY simulation parameter sweeps to find models
that best match M42 observables:
- Shell expansion velocity: v = 13 +/- 2 km/s
- Shell mass: M_shell = 2000 +/- 500 M_sun
- Dynamical age: t = 0.2 +/- 0.05 Myr

Produces:
- Chi-squared heatmap (mCloud x sfe grid)
- Trajectory comparison (v(t), M(t) with observation point marked)
- Residual contours with 1-sigma, 2-sigma, 3-sigma levels
- Ranking table (top 10 best-fit combinations)

@author: Jia Wei Teh
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations, organize_simulations_for_grid,
    get_unique_ndens, parse_simulation_params, resolve_data_input
)

print("...analyzing best-fit models for Orion Nebula (M42)")

# ============================================================================
# M42 Observational Constraints
# ============================================================================
OBS_VELOCITY = 13.0      # km/s
OBS_VELOCITY_ERR = 2.0   # km/s
OBS_SHELL_MASS = 2000.0  # M_sun
OBS_SHELL_MASS_ERR = 500.0  # M_sun
OBS_TIME = 0.2           # Myr
OBS_TIME_ERR = 0.05      # Myr

# Delta chi2 thresholds for confidence regions (2 DOF)
DELTA_CHI2_1SIGMA = 2.30
DELTA_CHI2_2SIGMA = 6.18
DELTA_CHI2_3SIGMA = 11.83

# Unit conversion: 1 km/s ~ 1.0227 pc/Myr
PC_MYR_TO_KM_S = 1.0 / 1.0227

# --- Output directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load matplotlib style if available
try:
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))
except:
    pass


def compute_chi2(v_kms, M_shell):
    """
    Compute chi-squared for a simulation compared to M42 observables.

    Parameters
    ----------
    v_kms : float
        Shell velocity in km/s
    M_shell : float
        Shell mass in M_sun

    Returns
    -------
    float
        Chi-squared value (2 degrees of freedom)
    """
    chi2_v = ((v_kms - OBS_VELOCITY) / OBS_VELOCITY_ERR) ** 2
    chi2_M = ((M_shell - OBS_SHELL_MASS) / OBS_SHELL_MASS_ERR) ** 2
    return chi2_v + chi2_M


def load_simulation_at_time(data_path, t_obs=OBS_TIME):
    """
    Load simulation and extract observables at specified time.

    Parameters
    ----------
    data_path : Path
        Path to simulation data file
    t_obs : float
        Observation time in Myr

    Returns
    -------
    dict
        Dictionary containing extracted observables and chi2
    """
    try:
        output = load_output(data_path)

        if len(output) == 0:
            return None

        # Get snapshot closest to observation time
        snap = output.get_at_time(t_obs, mode='closest')

        if snap is None:
            return None

        t_sim = snap.get('t_now', np.nan)
        v2_pcMyr = snap.get('v2', np.nan)
        shell_mass = snap.get('shell_mass', np.nan)
        R2 = snap.get('R2', np.nan)

        # Convert velocity to km/s
        v_kms = v2_pcMyr * PC_MYR_TO_KM_S if np.isfinite(v2_pcMyr) else np.nan

        # Compute chi2
        chi2 = compute_chi2(v_kms, shell_mass) if (np.isfinite(v_kms) and np.isfinite(shell_mass)) else np.inf

        # Get full time series for trajectory plots
        t_full = output.get('t_now')
        v2_full = output.get('v2')
        shell_mass_full = output.get('shell_mass')

        return {
            't_snap': t_sim,
            'v_kms': v_kms,
            'shell_mass': shell_mass,
            'R2': R2,
            'chi2': chi2,
            't_full': t_full,
            'v_full_kms': v2_full * PC_MYR_TO_KM_S if v2_full is not None else None,
            'shell_mass_full': shell_mass_full,
            'data_path': data_path
        }

    except Exception as e:
        print(f"  Error loading {data_path}: {e}")
        return None


def load_sweep_results(folder_path, ndens_filter=None):
    """
    Load all simulations from a sweep folder and compute chi2 values.

    Parameters
    ----------
    folder_path : str or Path
        Path to sweep folder
    ndens_filter : str, optional
        Filter by density (e.g., "1e4")

    Returns
    -------
    list
        List of result dictionaries sorted by chi2
    """
    folder_path = Path(folder_path)
    sim_files = find_all_simulations(folder_path)

    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return []

    results = []

    for sim_path in sim_files:
        # Parse simulation parameters from folder name
        folder_name = sim_path.parent.name
        params = parse_simulation_params(folder_name)

        if params is None:
            continue

        mCloud = params['mCloud']
        sfe = params['sfe']
        ndens = params['ndens']

        # Apply density filter if specified
        if ndens_filter and ndens != ndens_filter:
            continue

        # Load simulation data
        result = load_simulation_at_time(sim_path)

        if result is not None:
            result['mCloud'] = mCloud
            result['sfe'] = sfe
            result['ndens'] = ndens
            result['folder_name'] = folder_name
            results.append(result)

    # Sort by chi2
    results.sort(key=lambda x: x['chi2'])

    return results


def print_ranking_table(results, top_n=10):
    """
    Print ranking table of best-fit simulations.

    Parameters
    ----------
    results : list
        List of result dictionaries
    top_n : int
        Number of top results to display
    """
    print("\n" + "=" * 80)
    print("BEST-FIT MODELS FOR ORION NEBULA (M42)")
    print("=" * 80)
    print(f"\nObservational constraints:")
    print(f"  Velocity:   v = {OBS_VELOCITY:.1f} +/- {OBS_VELOCITY_ERR:.1f} km/s")
    print(f"  Shell mass: M = {OBS_SHELL_MASS:.0f} +/- {OBS_SHELL_MASS_ERR:.0f} M_sun")
    print(f"  Time:       t = {OBS_TIME:.2f} +/- {OBS_TIME_ERR:.2f} Myr")
    print("\n" + "-" * 80)
    print(f"{'Rank':<6}{'mCloud':<10}{'SFE':<8}{'nCore':<10}{'v [km/s]':<12}{'M [M_sun]':<12}{'chi2':<10}")
    print("-" * 80)

    for i, r in enumerate(results[:top_n]):
        rank = i + 1
        mCloud = r['mCloud']
        sfe = r['sfe']
        ndens = r['ndens']
        v = r['v_kms']
        M = r['shell_mass']
        chi2 = r['chi2']

        # Mark if within confidence regions
        marker = ""
        if chi2 < DELTA_CHI2_1SIGMA:
            marker = " ***"  # Within 1-sigma
        elif chi2 < DELTA_CHI2_2SIGMA:
            marker = " **"   # Within 2-sigma
        elif chi2 < DELTA_CHI2_3SIGMA:
            marker = " *"    # Within 3-sigma

        print(f"{rank:<6}{mCloud:<10}{sfe:<8}{ndens:<10}{v:<12.2f}{M:<12.0f}{chi2:<10.2f}{marker}")

    print("-" * 80)
    print("Legend: *** = 1-sigma, ** = 2-sigma, * = 3-sigma")
    print("=" * 80 + "\n")


def plot_chi2_heatmap(results, output_dir=None, ndens_filter=None):
    """
    Create chi-squared heatmap (mCloud x sfe grid).

    Parameters
    ----------
    results : list
        List of result dictionaries
    output_dir : Path, optional
        Output directory for figures
    ndens_filter : str, optional
        Density filter used (for filename)
    """
    if not results:
        print("No results to plot")
        return

    # Get unique mCloud and sfe values
    mCloud_vals = sorted(set(r['mCloud'] for r in results), key=float)
    sfe_vals = sorted(set(r['sfe'] for r in results), key=lambda x: int(x))

    # Create chi2 grid
    nrows, ncols = len(mCloud_vals), len(sfe_vals)
    chi2_grid = np.full((nrows, ncols), np.nan)

    # Build lookup dict
    lookup = {(r['mCloud'], r['sfe']): r for r in results}

    for i, mCloud in enumerate(mCloud_vals):
        for j, sfe in enumerate(sfe_vals):
            if (mCloud, sfe) in lookup:
                chi2_grid[i, j] = lookup[(mCloud, sfe)]['chi2']

    # Find best fit
    best_result = results[0]
    best_i = mCloud_vals.index(best_result['mCloud'])
    best_j = sfe_vals.index(best_result['sfe'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Use log scale for chi2 with custom normalization
    chi2_min = np.nanmin(chi2_grid)
    chi2_max = np.nanmax(chi2_grid)

    # Create colormap with confidence region boundaries
    cmap = plt.cm.RdYlGn_r  # Red = bad, Green = good

    # Plot heatmap
    im = ax.imshow(chi2_grid, cmap=cmap, aspect='auto',
                   norm=mcolors.LogNorm(vmin=max(0.1, chi2_min), vmax=min(1000, chi2_max)))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$\chi^2$')

    # Add confidence level lines to colorbar
    for chi2_level, label in [(DELTA_CHI2_1SIGMA, r'1$\sigma$'),
                               (DELTA_CHI2_2SIGMA, r'2$\sigma$'),
                               (DELTA_CHI2_3SIGMA, r'3$\sigma$')]:
        if chi2_min < chi2_level < chi2_max:
            cbar.ax.axhline(y=chi2_level, color='k', linestyle='--', linewidth=0.8)

    # Mark best-fit cell with gold star
    ax.plot(best_j, best_i, marker='*', markersize=20, color='gold',
            markeredgecolor='k', markeredgewidth=1.5, zorder=10)

    # Add chi2 values as text in each cell
    for i in range(nrows):
        for j in range(ncols):
            chi2_val = chi2_grid[i, j]
            if np.isfinite(chi2_val):
                text_color = 'white' if chi2_val > 10 else 'black'
                ax.text(j, i, f'{chi2_val:.1f}', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')

    # Labels
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_vals])
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([f'{float(m):.0e}' for m in mCloud_vals])

    ax.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
    ax.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')

    ndens_tag = f"n{ndens_filter}" if ndens_filter else "all"
    ax.set_title(f'M42 Best-Fit Analysis ({ndens_tag})\n'
                 f'Best: mCloud={best_result["mCloud"]}, sfe={best_result["sfe"]}, '
                 rf'$\chi^2$={best_result["chi2"]:.2f}')

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_chi2_heatmap_{ndens_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_trajectory_comparison(results, output_dir=None, ndens_filter=None, top_n=5):
    """
    Plot velocity and mass trajectories for top N simulations.

    Parameters
    ----------
    results : list
        List of result dictionaries
    output_dir : Path, optional
        Output directory
    ndens_filter : str, optional
        Density filter
    top_n : int
        Number of best-fit models to show
    """
    if not results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    ax_v, ax_m = axes

    # Color map for different simulations
    colors = plt.cm.tab10(np.linspace(0, 1, min(top_n, len(results))))

    for i, r in enumerate(results[:top_n]):
        if r['t_full'] is None:
            continue

        t = r['t_full']
        v = r['v_full_kms']
        M = r['shell_mass_full']
        label = f"{r['mCloud']}_sfe{r['sfe']} ($\\chi^2$={r['chi2']:.1f})"

        # Velocity trajectory
        if v is not None:
            ax_v.plot(t, v, color=colors[i], lw=1.5, label=label, alpha=0.8)

        # Mass trajectory
        if M is not None:
            ax_m.plot(t, M, color=colors[i], lw=1.5, label=label, alpha=0.8)

    # Mark observation point with error bars
    ax_v.errorbar(OBS_TIME, OBS_VELOCITY, xerr=OBS_TIME_ERR, yerr=OBS_VELOCITY_ERR,
                  fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                  label='M42 Observed', zorder=10, markeredgecolor='k')

    ax_m.errorbar(OBS_TIME, OBS_SHELL_MASS, xerr=OBS_TIME_ERR, yerr=OBS_SHELL_MASS_ERR,
                  fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                  label='M42 Observed', zorder=10, markeredgecolor='k')

    # Shade observation uncertainty region
    ax_v.axhspan(OBS_VELOCITY - OBS_VELOCITY_ERR, OBS_VELOCITY + OBS_VELOCITY_ERR,
                 alpha=0.2, color='red', zorder=1)
    ax_v.axvspan(OBS_TIME - OBS_TIME_ERR, OBS_TIME + OBS_TIME_ERR,
                 alpha=0.2, color='blue', zorder=1)

    ax_m.axhspan(OBS_SHELL_MASS - OBS_SHELL_MASS_ERR, OBS_SHELL_MASS + OBS_SHELL_MASS_ERR,
                 alpha=0.2, color='red', zorder=1)
    ax_m.axvspan(OBS_TIME - OBS_TIME_ERR, OBS_TIME + OBS_TIME_ERR,
                 alpha=0.2, color='blue', zorder=1)

    # Labels and formatting
    ax_v.set_xlabel('Time [Myr]')
    ax_v.set_ylabel('Shell Velocity [km/s]')
    ax_v.set_title('Velocity Evolution')
    ax_v.legend(loc='upper right', fontsize=8)
    ax_v.set_xlim(0, max(0.5, OBS_TIME * 2))
    ax_v.set_ylim(0, None)
    ax_v.grid(True, alpha=0.3)

    ax_m.set_xlabel('Time [Myr]')
    ax_m.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax_m.set_title('Shell Mass Evolution')
    ax_m.legend(loc='upper right', fontsize=8)
    ax_m.set_xlim(0, max(0.5, OBS_TIME * 2))
    ax_m.set_ylim(0, None)
    ax_m.grid(True, alpha=0.3)

    ndens_tag = f"n{ndens_filter}" if ndens_filter else "all"
    fig.suptitle(f'M42 Trajectory Comparison ({ndens_tag})', fontsize=14, y=1.02)

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_trajectories_{ndens_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_residual_contours(results, output_dir=None, ndens_filter=None):
    """
    Plot residual contours in velocity-mass space with confidence regions.

    Parameters
    ----------
    results : list
        List of result dictionaries
    output_dir : Path, optional
        Output directory
    ndens_filter : str, optional
        Density filter
    """
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot each simulation as a point
    v_vals = [r['v_kms'] for r in results if np.isfinite(r['v_kms'])]
    M_vals = [r['shell_mass'] for r in results if np.isfinite(r['shell_mass'])]
    chi2_vals = [r['chi2'] for r in results if np.isfinite(r['chi2'])]

    if not v_vals:
        print("No valid data points for residual plot")
        return

    # Scatter plot colored by chi2
    scatter = ax.scatter(v_vals, M_vals, c=chi2_vals, cmap='RdYlGn_r',
                        norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                        s=100, edgecolors='k', linewidths=0.5, zorder=5)

    # Draw confidence ellipses centered on observation
    theta = np.linspace(0, 2 * np.pi, 100)

    for delta_chi2, color, label in [(DELTA_CHI2_1SIGMA, 'green', r'1$\sigma$'),
                                      (DELTA_CHI2_2SIGMA, 'orange', r'2$\sigma$'),
                                      (DELTA_CHI2_3SIGMA, 'red', r'3$\sigma$')]:
        # For 2 DOF chi2, the ellipse radius scales as sqrt(delta_chi2)
        scale = np.sqrt(delta_chi2)
        v_ellipse = OBS_VELOCITY + scale * OBS_VELOCITY_ERR * np.cos(theta)
        M_ellipse = OBS_SHELL_MASS + scale * OBS_SHELL_MASS_ERR * np.sin(theta)
        ax.plot(v_ellipse, M_ellipse, color=color, lw=2, linestyle='--',
                label=f'{label} ($\\Delta\\chi^2={delta_chi2:.2f}$)')

    # Mark observation point
    ax.errorbar(OBS_VELOCITY, OBS_SHELL_MASS,
                xerr=OBS_VELOCITY_ERR, yerr=OBS_SHELL_MASS_ERR,
                fmt='s', color='red', markersize=15, capsize=5, capthick=2,
                label='M42 Observed', zorder=10, markeredgecolor='k', markeredgewidth=2)

    # Mark best-fit
    best = results[0]
    if np.isfinite(best['v_kms']) and np.isfinite(best['shell_mass']):
        ax.plot(best['v_kms'], best['shell_mass'], marker='*', markersize=25,
                color='gold', markeredgecolor='k', markeredgewidth=1.5, zorder=15,
                label=f'Best fit ($\\chi^2={best["chi2"]:.2f}$)')

    # Add annotations for top 3
    for i, r in enumerate(results[:3]):
        if np.isfinite(r['v_kms']) and np.isfinite(r['shell_mass']):
            ax.annotate(f"{r['mCloud']}\nsfe{r['sfe']}",
                       (r['v_kms'], r['shell_mass']),
                       textcoords="offset points", xytext=(10, 10),
                       fontsize=8, alpha=0.8)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=r'$\chi^2$')

    # Labels
    ax.set_xlabel('Shell Velocity [km/s]')
    ax.set_ylabel(r'Shell Mass [$M_\odot$]')
    ndens_tag = f"n{ndens_filter}" if ndens_filter else "all"
    ax.set_title(f'M42 Parameter Space ({ndens_tag})\n'
                 f'at t = {OBS_TIME} Myr')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_residuals_{ndens_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def main(folder_path, output_dir=None, ndens_filter=None):
    """
    Main analysis routine.

    Parameters
    ----------
    folder_path : str or Path
        Path to sweep folder containing simulations
    output_dir : str or Path, optional
        Output directory for figures
    ndens_filter : str, optional
        Filter simulations by density (e.g., "1e4")
    """
    print(f"\nAnalyzing sweep: {folder_path}")
    if ndens_filter:
        print(f"  Density filter: {ndens_filter}")

    # Load all results
    results = load_sweep_results(folder_path, ndens_filter)

    if not results:
        print("No valid simulations found!")
        return

    print(f"  Found {len(results)} valid simulations")

    # Print ranking table
    print_ranking_table(results, top_n=10)

    # Generate plots
    plot_chi2_heatmap(results, output_dir, ndens_filter)
    plot_trajectory_comparison(results, output_dir, ndens_filter, top_n=5)
    plot_residual_contours(results, output_dir, ndens_filter)

    print("\nAnalysis complete!")


# ============================================================================
# Command-line interface
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze TRINITY parameter sweep for M42 best-fit models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze sweep folder
  python paper_bestFitOrion.py --folder /path/to/sweep_orion/

  # Filter by density
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ -n 1e4

  # Specify output directory
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ -o /path/to/figures/
        """
    )
    parser.add_argument(
        '--folder', '-F', required=True,
        help='Path to sweep folder containing simulation subfolders'
    )
    parser.add_argument(
        '--nCore', '-n', default=None,
        help='Filter simulations by cloud density (e.g., "1e4", "1e3")'
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Output directory for figures (default: fig/)'
    )

    args = parser.parse_args()

    main(args.folder, args.output_dir, args.nCore)
