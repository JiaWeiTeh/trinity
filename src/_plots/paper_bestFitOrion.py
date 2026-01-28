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


def compute_chi2(v_kms, M_shell, free_param=None):
    """
    Compute chi-squared for a simulation compared to M42 observables.

    Parameters
    ----------
    v_kms : float
        Shell velocity in km/s
    M_shell : float
        Shell mass in M_sun
    free_param : str, optional
        If 'v', exclude velocity from chi2. If 'M', exclude mass from chi2.
        If None or 't', use both velocity and mass.

    Returns
    -------
    float
        Chi-squared value
    """
    chi2 = 0.0

    if free_param != 'v':
        chi2 += ((v_kms - OBS_VELOCITY) / OBS_VELOCITY_ERR) ** 2

    if free_param != 'M':
        chi2 += ((M_shell - OBS_SHELL_MASS) / OBS_SHELL_MASS_ERR) ** 2

    return chi2


def find_best_time(t_arr, v_arr_kms, M_arr, free_param=None):
    """
    Find the time that minimizes chi2 for given trajectories.

    When time is a free parameter, we scan all timesteps to find
    the one that best matches velocity and/or mass constraints.

    Parameters
    ----------
    t_arr : array
        Time array [Myr]
    v_arr_kms : array
        Velocity array [km/s]
    M_arr : array
        Shell mass array [M_sun]
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)

    Returns
    -------
    tuple
        (best_time, best_v, best_M, best_chi2)
    """
    if t_arr is None or len(t_arr) == 0:
        return np.nan, np.nan, np.nan, np.inf

    best_chi2 = np.inf
    best_idx = 0

    for i in range(len(t_arr)):
        v = v_arr_kms[i] if v_arr_kms is not None else np.nan
        M = M_arr[i] if M_arr is not None else np.nan

        if not (np.isfinite(v) and np.isfinite(M)):
            continue

        chi2 = compute_chi2(v, M, free_param)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_idx = i

    best_t = t_arr[best_idx]
    best_v = v_arr_kms[best_idx] if v_arr_kms is not None else np.nan
    best_M = M_arr[best_idx] if M_arr is not None else np.nan

    return best_t, best_v, best_M, best_chi2


def load_simulation_at_time(data_path, t_obs=OBS_TIME, free_param=None):
    """
    Load simulation and extract observables at specified time.

    Parameters
    ----------
    data_path : Path
        Path to simulation data file
    t_obs : float
        Observation time in Myr (ignored if free_param='t')
    free_param : str, optional
        Which parameter is free:
        - None: match all observables at t_obs
        - 't': find best time that matches v and M constraints
        - 'v': match M at t_obs, report v as free
        - 'M': match v at t_obs, report M as free

    Returns
    -------
    dict
        Dictionary containing extracted observables and chi2
    """
    try:
        output = load_output(data_path)

        if len(output) == 0:
            return None

        # Get full time series for trajectory plots
        t_full = output.get('t_now')
        v2_full = output.get('v2')
        shell_mass_full = output.get('shell_mass')
        v_full_kms = v2_full * PC_MYR_TO_KM_S if v2_full is not None else None

        # Handle free time parameter: find best-matching time
        if free_param == 't':
            t_sim, v_kms, shell_mass, chi2 = find_best_time(
                t_full, v_full_kms, shell_mass_full, free_param=None
            )
            # Get R2 at best time
            snap = output.get_at_time(t_sim, mode='closest')
            R2 = snap.get('R2', np.nan) if snap else np.nan
        else:
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

            # Compute chi2 (with free parameter if specified)
            if np.isfinite(v_kms) and np.isfinite(shell_mass):
                chi2 = compute_chi2(v_kms, shell_mass, free_param)
            else:
                chi2 = np.inf

        return {
            't_snap': t_sim,
            'v_kms': v_kms,
            'shell_mass': shell_mass,
            'R2': R2,
            'chi2': chi2,
            't_full': t_full,
            'v_full_kms': v_full_kms,
            'shell_mass_full': shell_mass_full,
            'data_path': data_path
        }

    except Exception as e:
        print(f"  Error loading {data_path}: {e}")
        return None


def load_sweep_results(folder_path, ndens_filter=None, free_param=None):
    """
    Load all simulations from a sweep folder and compute chi2 values.

    Parameters
    ----------
    folder_path : str or Path
        Path to sweep folder
    ndens_filter : str, optional
        Filter by density (e.g., "1e4")
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)

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
        result = load_simulation_at_time(sim_path, free_param=free_param)

        if result is not None:
            result['mCloud'] = mCloud
            result['sfe'] = sfe
            result['ndens'] = ndens
            result['folder_name'] = folder_name
            result['free_param'] = free_param
            results.append(result)

    # Sort by chi2
    results.sort(key=lambda x: x['chi2'])

    return results


def print_ranking_table(results, top_n=10, free_param=None):
    """
    Print ranking table of best-fit simulations.

    Parameters
    ----------
    results : list
        List of result dictionaries
    top_n : int
        Number of top results to display
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)
    """
    print("\n" + "=" * 90)
    print("BEST-FIT MODELS FOR ORION NEBULA (M42)")
    print("=" * 90)

    # Show which parameter is free
    if free_param == 't':
        print("\nMode: FREE TIME - finding best-matching time for each simulation")
    elif free_param == 'v':
        print("\nMode: FREE VELOCITY - matching mass at fixed time, velocity unconstrained")
    elif free_param == 'M':
        print("\nMode: FREE MASS - matching velocity at fixed time, mass unconstrained")
    else:
        print("\nMode: STANDARD - matching all observables at fixed time")

    print(f"\nObservational constraints:")
    v_mark = " [FREE]" if free_param == 'v' else ""
    M_mark = " [FREE]" if free_param == 'M' else ""
    t_mark = " [FREE]" if free_param == 't' else ""
    print(f"  Velocity:   v = {OBS_VELOCITY:.1f} +/- {OBS_VELOCITY_ERR:.1f} km/s{v_mark}")
    print(f"  Shell mass: M = {OBS_SHELL_MASS:.0f} +/- {OBS_SHELL_MASS_ERR:.0f} M_sun{M_mark}")
    print(f"  Time:       t = {OBS_TIME:.2f} +/- {OBS_TIME_ERR:.2f} Myr{t_mark}")

    # Build header based on free parameter
    print("\n" + "-" * 90)
    if free_param == 't':
        print(f"{'Rank':<6}{'mCloud':<10}{'SFE':<8}{'nCore':<10}{'t [Myr]':<12}{'v [km/s]':<12}{'M [M_sun]':<12}{'chi2':<10}")
    else:
        print(f"{'Rank':<6}{'mCloud':<10}{'SFE':<8}{'nCore':<10}{'v [km/s]':<12}{'M [M_sun]':<12}{'chi2':<10}")
    print("-" * 90)

    for i, r in enumerate(results[:top_n]):
        rank = i + 1
        mCloud = r['mCloud']
        sfe = r['sfe']
        ndens = r['ndens']
        t = r['t_snap']
        v = r['v_kms']
        M = r['shell_mass']
        chi2 = r['chi2']

        # Mark if within confidence regions (use appropriate DOF)
        # When a parameter is free, we have 1 DOF instead of 2
        if free_param in ['v', 'M', 't']:
            # 1 DOF thresholds
            thresh_1sig, thresh_2sig, thresh_3sig = 1.0, 4.0, 9.0
        else:
            thresh_1sig = DELTA_CHI2_1SIGMA
            thresh_2sig = DELTA_CHI2_2SIGMA
            thresh_3sig = DELTA_CHI2_3SIGMA

        marker = ""
        if chi2 < thresh_1sig:
            marker = " ***"  # Within 1-sigma
        elif chi2 < thresh_2sig:
            marker = " **"   # Within 2-sigma
        elif chi2 < thresh_3sig:
            marker = " *"    # Within 3-sigma

        if free_param == 't':
            print(f"{rank:<6}{mCloud:<10}{sfe:<8}{ndens:<10}{t:<12.3f}{v:<12.2f}{M:<12.0f}{chi2:<10.2f}{marker}")
        else:
            print(f"{rank:<6}{mCloud:<10}{sfe:<8}{ndens:<10}{v:<12.2f}{M:<12.0f}{chi2:<10.2f}{marker}")

    print("-" * 90)
    print("Legend: *** = 1-sigma, ** = 2-sigma, * = 3-sigma")
    print("=" * 90 + "\n")


def plot_chi2_heatmap(results, output_dir=None, ndens_filter=None, free_param=None):
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
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)
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
    mode_tag = f"free_{free_param}" if free_param else "standard"
    mode_str = {"t": "free time", "v": "free velocity", "M": "free mass"}.get(free_param, "standard")

    title_extra = ""
    if free_param == 't':
        title_extra = f", t={best_result['t_snap']:.3f} Myr"

    ax.set_title(f'M42 Best-Fit Analysis ({ndens_tag}, {mode_str})\n'
                 f'Best: mCloud={best_result["mCloud"]}, sfe={best_result["sfe"]}{title_extra}, '
                 rf'$\chi^2$={best_result["chi2"]:.2f}')

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_chi2_heatmap_{ndens_tag}_{mode_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_trajectory_comparison(results, output_dir=None, ndens_filter=None, top_n=5, free_param=None):
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
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)
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

        if free_param == 't':
            label = f"{r['mCloud']}_sfe{r['sfe']} (t={r['t_snap']:.2f}, $\\chi^2$={r['chi2']:.1f})"
        else:
            label = f"{r['mCloud']}_sfe{r['sfe']} ($\\chi^2$={r['chi2']:.1f})"

        # Velocity trajectory
        if v is not None:
            ax_v.plot(t, v, color=colors[i], lw=1.5, label=label, alpha=0.8)
            # Mark best-fit time point when time is free
            if free_param == 't':
                ax_v.axvline(r['t_snap'], color=colors[i], ls='--', lw=1, alpha=0.5)
                ax_v.plot(r['t_snap'], r['v_kms'], 'o', color=colors[i], markersize=8, zorder=8)

        # Mass trajectory
        if M is not None:
            ax_m.plot(t, M, color=colors[i], lw=1.5, label=label, alpha=0.8)
            # Mark best-fit time point when time is free
            if free_param == 't':
                ax_m.axvline(r['t_snap'], color=colors[i], ls='--', lw=1, alpha=0.5)
                ax_m.plot(r['t_snap'], r['shell_mass'], 'o', color=colors[i], markersize=8, zorder=8)

    # Mark observation point with error bars (only if not free time mode)
    if free_param != 't':
        ax_v.errorbar(OBS_TIME, OBS_VELOCITY, xerr=OBS_TIME_ERR, yerr=OBS_VELOCITY_ERR,
                      fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                      label='M42 Observed', zorder=10, markeredgecolor='k')

        ax_m.errorbar(OBS_TIME, OBS_SHELL_MASS, xerr=OBS_TIME_ERR, yerr=OBS_SHELL_MASS_ERR,
                      fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                      label='M42 Observed', zorder=10, markeredgecolor='k')

        # Shade time uncertainty region
        ax_v.axvspan(OBS_TIME - OBS_TIME_ERR, OBS_TIME + OBS_TIME_ERR,
                     alpha=0.2, color='blue', zorder=1)
        ax_m.axvspan(OBS_TIME - OBS_TIME_ERR, OBS_TIME + OBS_TIME_ERR,
                     alpha=0.2, color='blue', zorder=1)

    # Shade observable uncertainty region (based on which is constrained)
    if free_param != 'v':
        ax_v.axhspan(OBS_VELOCITY - OBS_VELOCITY_ERR, OBS_VELOCITY + OBS_VELOCITY_ERR,
                     alpha=0.2, color='red', zorder=1)

    if free_param != 'M':
        ax_m.axhspan(OBS_SHELL_MASS - OBS_SHELL_MASS_ERR, OBS_SHELL_MASS + OBS_SHELL_MASS_ERR,
                     alpha=0.2, color='red', zorder=1)

    # Determine x-axis limit based on mode
    if free_param == 't':
        # Extend x-axis to show all best-fit times
        max_t = max(r['t_snap'] for r in results[:top_n] if np.isfinite(r['t_snap']))
        x_max = max(0.5, max_t * 1.2)
    else:
        x_max = max(0.5, OBS_TIME * 2)

    # Labels and formatting
    ax_v.set_xlabel('Time [Myr]')
    ax_v.set_ylabel('Shell Velocity [km/s]')
    ax_v.set_title('Velocity Evolution')
    ax_v.legend(loc='upper right', fontsize=8)
    ax_v.set_xlim(0, x_max)
    ax_v.set_ylim(0, None)
    ax_v.grid(True, alpha=0.3)

    ax_m.set_xlabel('Time [Myr]')
    ax_m.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax_m.set_title('Shell Mass Evolution')
    ax_m.legend(loc='upper right', fontsize=8)
    ax_m.set_xlim(0, x_max)
    ax_m.set_ylim(0, None)
    ax_m.grid(True, alpha=0.3)

    ndens_tag = f"n{ndens_filter}" if ndens_filter else "all"
    mode_tag = f"free_{free_param}" if free_param else "standard"
    mode_str = {"t": "free time", "v": "free velocity", "M": "free mass"}.get(free_param, "standard")
    fig.suptitle(f'M42 Trajectory Comparison ({ndens_tag}, {mode_str})', fontsize=14, y=1.02)

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_trajectories_{ndens_tag}_{mode_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def plot_residual_contours(results, output_dir=None, ndens_filter=None, free_param=None):
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
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None)
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
    mode_tag = f"free_{free_param}" if free_param else "standard"
    mode_str = {"t": "free time", "v": "free velocity", "M": "free mass"}.get(free_param, "standard")

    if free_param == 't':
        time_str = "at best-fit time"
    else:
        time_str = f"at t = {OBS_TIME} Myr"

    ax.set_title(f'M42 Parameter Space ({ndens_tag}, {mode_str})\n{time_str}')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    fig_dir = Path(output_dir) if output_dir else FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = fig_dir / f"orion_residuals_{ndens_tag}_{mode_tag}.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_pdf}")

    plt.close(fig)


def main(folder_path, output_dir=None, ndens_filter=None, free_param=None):
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
    free_param : str, optional
        Which parameter is free ('t', 'v', 'M', or None):
        - 't': find best time that matches v and M
        - 'v': match M at fixed time, v unconstrained
        - 'M': match v at fixed time, M unconstrained
    """
    print(f"\nAnalyzing sweep: {folder_path}")
    if ndens_filter:
        print(f"  Density filter: {ndens_filter}")
    if free_param:
        print(f"  Free parameter: {free_param}")

    # Load all results
    results = load_sweep_results(folder_path, ndens_filter, free_param)

    if not results:
        print("No valid simulations found!")
        return

    print(f"  Found {len(results)} valid simulations")

    # Print ranking table
    print_ranking_table(results, top_n=10, free_param=free_param)

    # Generate plots
    plot_chi2_heatmap(results, output_dir, ndens_filter, free_param)
    plot_trajectory_comparison(results, output_dir, ndens_filter, top_n=5, free_param=free_param)
    plot_residual_contours(results, output_dir, ndens_filter, free_param)

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
  # Analyze sweep folder (standard mode - match all observables at t=0.2 Myr)
  python paper_bestFitOrion.py --folder /path/to/sweep_orion/

  # Free time mode - find best time for each simulation
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ --free t

  # Free mass mode - match velocity, report mass as free parameter
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ --free M

  # Free velocity mode - match mass, report velocity as free parameter
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ --free v

  # Filter by density
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ -n 1e4

  # Combine options
  python paper_bestFitOrion.py -F /path/to/sweep_orion/ -n 1e4 --free t
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
    parser.add_argument(
        '--free', choices=['t', 'v', 'M'], default=None,
        help='Make one observable a free parameter: '
             't = find best-fit time (maximize age), '
             'v = free velocity (match mass only), '
             'M = free mass (match velocity only)'
    )

    args = parser.parse_args()

    main(args.folder, args.output_dir, args.nCore, args.free)
