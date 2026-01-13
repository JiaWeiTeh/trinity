#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mass Profile Comparison: Original vs Refactored

Compares outputs from:
- src/cloud_properties/mass_profile.py (get_mass_profile_OLD)
- analysis/mass_profile/REFACTORED_mass_profile.py (get_mass_profile)

Test cases:
- Power-law α = 0 (homogeneous cloud)
- Power-law α = -2 (isothermal sphere)
- Bonnor-Ebert sphere

Author: Claude Code
Date: 2026-01-13
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Setup paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_analysis_dir = _script_dir  # /home/user/trinity/analysis
_mass_profile_dir = os.path.join(_script_dir, 'mass_profile')
_be_dir = os.path.join(_script_dir, 'bonnorEbert')
_functions_dir = os.path.join(_project_root, 'src', '_functions')
_src_cloud_dir = os.path.join(_project_root, 'src', 'cloud_properties')

# Add paths (order matters for module resolution)
# Note: There's a density_profile.py in src/cloud_properties that conflicts with
# the density_profile/ package in analysis/. We need analysis dir to be searched first.
for _dir in [_mass_profile_dir, _be_dir, _functions_dir, _src_cloud_dir, _project_root]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# CRITICAL: Insert analysis dir LAST so it becomes position 0 (highest priority)
# This ensures density_profile/ package is found before density_profile.py module
sys.path.insert(0, _analysis_dir)

# Import implementations
from src.cloud_properties.mass_profile import get_mass_profile_OLD

# Import refactored version
from REFACTORED_mass_profile import get_mass_profile as get_mass_profile_NEW

# Import for Bonnor-Ebert sphere
from bonnorEbertSphere_v2 import create_BE_sphere, solve_lane_emden

print("...comparing mass profile implementations")

plt.style.use('/home/user/trinity/src/_plots/trinity.mplstyle')

# Disable LaTeX if not available (for container environments)
try:
    import subprocess
    subprocess.run(['latex', '--version'], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    plt.rcParams['text.usetex'] = False
    print("  (LaTeX not available - using standard fonts)")

# Output directory
FIG_DIR = Path("./fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Mock Parameter Class
# =============================================================================

class MockParam:
    """Mock parameter object to match TRINITY's expected interface."""
    def __init__(self, value):
        self.value = value


# =============================================================================
# Test Case Parameters
# =============================================================================

def create_powerlaw_params(alpha, n_core=1e3, r_cloud=10.0, m_cloud=1e5):
    """Create parameters for power-law density profile."""
    return {
        'dens_profile': MockParam('densPL'),
        'nCore': MockParam(n_core),
        'nISM': MockParam(1.0),
        'mu_ion': MockParam(1.4),
        'mu_neu': MockParam(2.3),
        'rCore': MockParam(1.0),
        'rCloud': MockParam(r_cloud),
        'mCloud': MockParam(m_cloud),
        'densPL_alpha': MockParam(alpha),
    }


def create_bonnor_ebert_params(M_cloud=1e5, n_core=1e3, Omega=8.0, mu=2.33):
    """Create parameters for Bonnor-Ebert sphere."""
    # Solve Lane-Emden equation
    solution = solve_lane_emden()

    # Create BE sphere
    be_result = create_BE_sphere(
        M_cloud=M_cloud,
        n_core=n_core,
        Omega=Omega,
        mu=mu,
        gamma=5.0/3.0,
        lane_emden_solution=solution
    )

    params = {
        'dens_profile': MockParam('densBE'),
        'nCore': MockParam(n_core),
        'nISM': MockParam(1.0),
        'mu_ion': MockParam(mu),
        'mu_neu': MockParam(2.3),
        'rCore': MockParam(be_result.r_out * 0.1),
        'rCloud': MockParam(be_result.r_out),
        'mCloud': MockParam(M_cloud),
        'densBE_Omega': MockParam(Omega),
        'densBE_Teff': MockParam(be_result.T_eff),
        'densBE_f_rho_rhoc': MockParam(solution.f_rho_rhoc),
        'densBE_xi_arr': MockParam(solution.xi),
        'densBE_u_arr': MockParam(solution.u),
        'densBE_dudxi_arr': MockParam(solution.dudxi),
        'densBE_rho_rhoc_arr': MockParam(solution.rho_rhoc),
        'gamma_adia': MockParam(5.0/3.0),
    }

    return params, be_result


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_mass_profiles(r_arr, params, rdot_arr=None, label="Test"):
    """
    Compare OLD vs NEW mass profile outputs.

    Returns:
        dict with comparison results
    """
    results = {
        'r': r_arr,
        'label': label,
        'M_old': None,
        'M_new': None,
        'M_diff': None,
        'M_rel_diff': None,
        'M_match': False,
        'M_old_error': None,
        'dMdt_old': None,
        'dMdt_new': None,
        'dMdt_diff': None,
        'dMdt_rel_diff': None,
        'dMdt_match': False,
        'dMdt_old_error': None,
    }

    # =========================================================================
    # Mass M(r) comparison
    # =========================================================================

    # Run OLD
    try:
        M_old = get_mass_profile_OLD(r_arr, params, return_mdot=False)
        results['M_old'] = np.asarray(M_old)
    except Exception as e:
        results['M_old_error'] = str(e)
        print(f"  [{label}] OLD M(r) failed: {e}")

    # Run NEW
    try:
        M_new = get_mass_profile_NEW(r_arr, params, return_mdot=False)
        results['M_new'] = np.asarray(M_new)
    except Exception as e:
        print(f"  [{label}] NEW M(r) failed: {e}")
        return results

    # Compute differences
    if results['M_old'] is not None and results['M_new'] is not None:
        results['M_diff'] = np.abs(results['M_new'] - results['M_old'])

        # Relative difference (avoid division by zero)
        denom = np.maximum(np.abs(results['M_old']), 1e-10)
        results['M_rel_diff'] = results['M_diff'] / denom

        # Check if they match (< 1% relative error)
        max_rel_diff = np.max(results['M_rel_diff'])
        results['M_match'] = max_rel_diff < 0.01

        print(f"  [{label}] M(r): max rel diff = {max_rel_diff:.2e} " +
              f"({'✓ MATCH' if results['M_match'] else '✗ MISMATCH'})")

    # =========================================================================
    # Mass accretion rate dM/dt comparison
    # =========================================================================

    if rdot_arr is not None:
        # Run OLD with dM/dt
        try:
            M_old_mdot, dMdt_old = get_mass_profile_OLD(
                r_arr, params, return_mdot=True, rdot_arr=rdot_arr
            )
            results['dMdt_old'] = np.asarray(dMdt_old)
        except Exception as e:
            results['dMdt_old_error'] = str(e)
            print(f"  [{label}] OLD dM/dt failed: {e}")

        # Run NEW with dM/dt
        try:
            M_new_mdot, dMdt_new = get_mass_profile_NEW(
                r_arr, params, return_mdot=True, rdot=rdot_arr
            )
            results['dMdt_new'] = np.asarray(dMdt_new)
        except Exception as e:
            print(f"  [{label}] NEW dM/dt failed: {e}")

        # Compute differences
        if results['dMdt_old'] is not None and results['dMdt_new'] is not None:
            results['dMdt_diff'] = np.abs(results['dMdt_new'] - results['dMdt_old'])

            denom = np.maximum(np.abs(results['dMdt_old']), 1e-10)
            results['dMdt_rel_diff'] = results['dMdt_diff'] / denom

            max_rel_diff = np.max(results['dMdt_rel_diff'])
            results['dMdt_match'] = max_rel_diff < 0.01

            print(f"  [{label}] dM/dt: max rel diff = {max_rel_diff:.2e} " +
                  f"({'✓ MATCH' if results['dMdt_match'] else '✗ MISMATCH'})")

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_mass_comparison_grid(results_dict, save_path=None):
    """
    Create 3×3 grid comparing OLD vs NEW mass profiles.

    Rows: OLD, NEW, Difference
    Cols: α=0, α=-2, Bonnor-Ebert
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)

    cases = ['alpha0', 'alpha2', 'be']
    titles = [r'$\alpha = 0$', r'$\alpha = -2$', 'Bonnor-Ebert']
    row_labels = ['OLD', 'NEW', '|Diff|']

    for j, (case, title) in enumerate(zip(cases, titles)):
        res = results_dict[case]
        r = res['r']

        # Row 0: OLD
        if res['M_old'] is not None:
            axes[0, j].plot(r, res['M_old'], 'b-', lw=1.5)
            axes[0, j].set_yscale('log')
        else:
            axes[0, j].text(0.5, 0.5, 'FAILED', ha='center', va='center',
                           transform=axes[0, j].transAxes, fontsize=12, color='red')

        if j == 0:
            axes[0, j].set_ylabel(r'$M(r)$ [M$_\odot$]' + f'\n{row_labels[0]}')
        axes[0, j].set_title(title)

        # Row 1: NEW
        if res['M_new'] is not None:
            axes[1, j].plot(r, res['M_new'], 'r-', lw=1.5)
            axes[1, j].set_yscale('log')

        if j == 0:
            axes[1, j].set_ylabel(r'$M(r)$ [M$_\odot$]' + f'\n{row_labels[1]}')

        # Row 2: Difference
        if res['M_diff'] is not None:
            axes[2, j].plot(r, res['M_rel_diff'] * 100, 'k-', lw=1.5)
            axes[2, j].set_yscale('log')
            axes[2, j].axhline(1.0, ls='--', color='gray', alpha=0.5, label='1%')

        if j == 0:
            axes[2, j].set_ylabel(f'Rel. Diff [%]\n{row_labels[2]}')
        axes[2, j].set_xlabel(r'$r$ [pc]')

    fig.suptitle(r'Mass Profile $M(r)$ Comparison: OLD vs NEW', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_mdot_comparison_grid(results_dict, save_path=None):
    """
    Create 3×3 grid comparing OLD vs NEW mass accretion rates.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)

    cases = ['alpha0', 'alpha2', 'be']
    titles = [r'$\alpha = 0$', r'$\alpha = -2$', 'Bonnor-Ebert']
    row_labels = ['OLD', 'NEW', '|Diff|']

    for j, (case, title) in enumerate(zip(cases, titles)):
        res = results_dict[case]
        r = res['r']

        # Row 0: OLD
        if res['dMdt_old'] is not None:
            axes[0, j].plot(r, res['dMdt_old'], 'b-', lw=1.5)
            axes[0, j].set_yscale('log')
        else:
            error_msg = res.get('dMdt_old_error', 'FAILED')
            if len(error_msg) > 30:
                error_msg = error_msg[:27] + '...'
            axes[0, j].text(0.5, 0.5, f'FAILED:\n{error_msg}', ha='center', va='center',
                           transform=axes[0, j].transAxes, fontsize=8, color='red',
                           wrap=True)

        if j == 0:
            axes[0, j].set_ylabel(r'$\dot{M}$ [M$_\odot$/Myr]' + f'\n{row_labels[0]}')
        axes[0, j].set_title(title)

        # Row 1: NEW
        if res['dMdt_new'] is not None:
            axes[1, j].plot(r, res['dMdt_new'], 'r-', lw=1.5)
            axes[1, j].set_yscale('log')

        if j == 0:
            axes[1, j].set_ylabel(r'$\dot{M}$ [M$_\odot$/Myr]' + f'\n{row_labels[1]}')

        # Row 2: Difference
        if res['dMdt_diff'] is not None:
            axes[2, j].plot(r, res['dMdt_rel_diff'] * 100, 'k-', lw=1.5)
            axes[2, j].set_yscale('log')
            axes[2, j].axhline(1.0, ls='--', color='gray', alpha=0.5, label='1%')
        else:
            axes[2, j].text(0.5, 0.5, 'N/A', ha='center', va='center',
                           transform=axes[2, j].transAxes, fontsize=12, color='gray')

        if j == 0:
            axes[2, j].set_ylabel(f'Rel. Diff [%]\n{row_labels[2]}')
        axes[2, j].set_xlabel(r'$r$ [pc]')

    fig.suptitle(r'Mass Accretion Rate $\dot{M}$ Comparison: OLD vs NEW', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_overlay_comparison(results_dict, save_path=None):
    """
    Create 2×3 grid with OLD and NEW overlaid on same axes.

    Row 1: M(r) overlay
    Row 2: dM/dt overlay
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    cases = ['alpha0', 'alpha2', 'be']
    titles = [r'$\alpha = 0$', r'$\alpha = -2$', 'Bonnor-Ebert']

    for j, (case, title) in enumerate(zip(cases, titles)):
        res = results_dict[case]
        r = res['r']

        # Row 0: M(r) overlay
        ax = axes[0, j]
        if res['M_old'] is not None:
            ax.plot(r, res['M_old'], 'b-', lw=2, label='OLD', alpha=0.7)
        if res['M_new'] is not None:
            ax.plot(r, res['M_new'], 'r--', lw=2, label='NEW', alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(title)
        if j == 0:
            ax.set_ylabel(r'$M(r)$ [M$_\odot$]')
            ax.legend(loc='lower right', fontsize=8)

        # Row 1: dM/dt overlay
        ax = axes[1, j]
        if res['dMdt_old'] is not None:
            ax.plot(r, res['dMdt_old'], 'b-', lw=2, label='OLD', alpha=0.7)
        if res['dMdt_new'] is not None:
            ax.plot(r, res['dMdt_new'], 'r--', lw=2, label='NEW', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel(r'$r$ [pc]')
        if j == 0:
            ax.set_ylabel(r'$\dot{M}$ [M$_\odot$/Myr]')

    fig.suptitle('Mass Profile Comparison: OLD (blue) vs NEW (red dashed)', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def print_summary_table(results_dict):
    """Print a summary table of comparison results."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    print(f"\n{'Test Case':<20} {'M(r) Match?':<15} {'dM/dt Match?':<15} {'Notes'}")
    print("-" * 70)

    for case, title in [('alpha0', 'α = 0'), ('alpha2', 'α = -2'), ('be', 'Bonnor-Ebert')]:
        res = results_dict[case]

        # M(r) status
        if res['M_old_error']:
            m_status = '✗ OLD FAILED'
        elif res['M_match']:
            m_status = '✓ MATCH'
        else:
            m_status = '✗ MISMATCH'

        # dM/dt status
        if res['dMdt_old_error']:
            mdot_status = '✗ OLD FAILED'
        elif res['dMdt_match']:
            mdot_status = '✓ MATCH'
        elif res['dMdt_new'] is None:
            mdot_status = 'N/A'
        else:
            mdot_status = '✗ MISMATCH'

        # Notes
        notes = []
        if res['M_rel_diff'] is not None:
            notes.append(f"M max err: {np.max(res['M_rel_diff'])*100:.2f}%")
        if res['dMdt_rel_diff'] is not None:
            notes.append(f"dM/dt max err: {np.max(res['dMdt_rel_diff'])*100:.2f}%")

        print(f"{title:<20} {m_status:<15} {mdot_status:<15} {', '.join(notes)}")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run all comparisons."""

    print("\n" + "=" * 70)
    print("MASS PROFILE COMPARISON: OLD vs NEW")
    print("=" * 70)

    results = {}

    # =========================================================================
    # Test Case 1: Power-law α = 0 (homogeneous)
    # =========================================================================
    print("\n[TEST 1] Power-law α = 0 (homogeneous cloud)")
    print("-" * 50)

    params_alpha0 = create_powerlaw_params(alpha=0.0)
    r_arr = np.linspace(0.5, 15.0, 100)
    rdot_arr = np.full_like(r_arr, 10.0)  # 10 pc/Myr velocity

    results['alpha0'] = compare_mass_profiles(
        r_arr, params_alpha0, rdot_arr=rdot_arr, label='α=0'
    )

    # =========================================================================
    # Test Case 2: Power-law α = -2 (isothermal)
    # =========================================================================
    print("\n[TEST 2] Power-law α = -2 (isothermal sphere)")
    print("-" * 50)

    params_alpha2 = create_powerlaw_params(alpha=-2.0)
    r_arr = np.linspace(0.5, 15.0, 100)
    rdot_arr = np.full_like(r_arr, 10.0)

    results['alpha2'] = compare_mass_profiles(
        r_arr, params_alpha2, rdot_arr=rdot_arr, label='α=-2'
    )

    # =========================================================================
    # Test Case 3: Bonnor-Ebert sphere
    # =========================================================================
    print("\n[TEST 3] Bonnor-Ebert sphere")
    print("-" * 50)

    params_be, be_result = create_bonnor_ebert_params()
    r_arr = np.linspace(0.1, be_result.r_out * 0.95, 100)
    rdot_arr = np.full_like(r_arr, 10.0)

    results['be'] = compare_mass_profiles(
        r_arr, params_be, rdot_arr=rdot_arr, label='BE'
    )

    # =========================================================================
    # Generate plots
    # =========================================================================
    print("\n[PLOTTING]")
    print("-" * 50)

    # Mass comparison grid
    plot_mass_comparison_grid(
        results,
        save_path=FIG_DIR / 'mass_profile_M_comparison.pdf'
    )

    # dM/dt comparison grid
    plot_mdot_comparison_grid(
        results,
        save_path=FIG_DIR / 'mass_profile_dMdt_comparison.pdf'
    )

    # Overlay comparison
    plot_overlay_comparison(
        results,
        save_path=FIG_DIR / 'mass_profile_overlay_comparison.pdf'
    )

    # =========================================================================
    # Print summary
    # =========================================================================
    print_summary_table(results)

    plt.show()


if __name__ == "__main__":
    main()
