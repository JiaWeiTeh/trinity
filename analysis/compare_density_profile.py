#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Density Profile Comparison: Original vs Refactored

Compares outputs from:
- src/cloud_properties/density_profile.py (get_density_profile - OLD)
- analysis/density_profile/REFACTORED_density_profile.py (get_density_profile - NEW)

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
_analysis_dir = _script_dir
_density_profile_dir = os.path.join(_script_dir, 'density_profile')
_be_dir = os.path.join(_script_dir, 'bonnorEbert')
_functions_dir = os.path.join(_project_root, 'src', '_functions')
_src_cloud_dir = os.path.join(_project_root, 'src', 'cloud_properties')

# Add paths (order matters for module resolution)
for _dir in [_density_profile_dir, _be_dir, _functions_dir, _src_cloud_dir, _project_root]:
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# CRITICAL: Insert analysis dir LAST so it becomes position 0 (highest priority)
sys.path.insert(0, _analysis_dir)

# Import OLD implementation
from src.cloud_properties.density_profile import get_density_profile as get_density_profile_OLD

# Import REFACTORED version
from density_profile.REFACTORED_density_profile import get_density_profile as get_density_profile_NEW

# Import for Bonnor-Ebert sphere
from bonnorEbertSphere_v2 import create_BE_sphere, solve_lane_emden

# Import utility for computing rCloud from physical parameters
from src.cloud_properties.powerLawSphere import compute_rCloud_powerlaw

print("...comparing density profile implementations")

# Load style file using relative path
_style_file = os.path.join(_project_root, 'src', '_plots', 'trinity.mplstyle')
if os.path.exists(_style_file):
    plt.style.use(_style_file)
else:
    print(f"  (Style file not found: {_style_file})")

# Disable LaTeX if not available
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

def create_powerlaw_params(alpha, n_core=1e3, m_cloud=1e5, mu=1.4, rCore_fraction=0.1):
    """
    Create parameters for power-law density profile.

    Note: rCloud is computed from (m_cloud, n_core, alpha) using the proper
    physical formula, not hardcoded. This ensures self-consistency.
    """
    # Compute rCloud from fundamental inputs
    rCloud, rCore = compute_rCloud_powerlaw(m_cloud, n_core, alpha,
                                             rCore_fraction=rCore_fraction, mu=mu)

    print(f"  [create_powerlaw_params] α={alpha}: computed rCloud={rCloud:.3f} pc, rCore={rCore:.3f} pc")

    return {
        'dens_profile': MockParam('densPL'),
        'nCore': MockParam(n_core),
        'nISM': MockParam(1.0),
        'mu_ion': MockParam(mu),
        'mu_atom': MockParam(2.3),
        'rCore': MockParam(rCore),
        'rCloud': MockParam(rCloud),
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
        'mu_atom': MockParam(2.3),
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

def compare_density_profiles(r_arr, params, label="Test"):
    """
    Compare OLD vs NEW density profile outputs.

    Returns:
        dict with comparison results
    """
    # Extract key parameters for verification
    nCore = params['nCore'].value
    nISM = params['nISM'].value
    rCloud = params['rCloud'].value
    rCore = params['rCore'].value

    results = {
        'r': r_arr,
        'label': label,
        'nCore': nCore,
        'nISM': nISM,
        'rCloud': rCloud,
        'rCore': rCore,
        'n_old': None,
        'n_new': None,
        'n_diff': None,
        'n_rel_diff': None,
        'n_match': False,
        'n_old_error': None,
        'n_new_error': None,
        'scalar_test_old': None,
        'scalar_test_new': None,
    }

    # =========================================================================
    # Density n(r) comparison - Array input
    # =========================================================================

    # Run OLD
    try:
        n_old = get_density_profile_OLD(r_arr, params)
        results['n_old'] = np.asarray(n_old).flatten()
    except Exception as e:
        results['n_old_error'] = str(e)
        print(f"  [{label}] OLD n(r) failed: {e}")

    # Run NEW
    try:
        n_new = get_density_profile_NEW(r_arr, params)
        results['n_new'] = np.asarray(n_new).flatten()
    except Exception as e:
        results['n_new_error'] = str(e)
        print(f"  [{label}] NEW n(r) failed: {e}")
        return results

    # Compute differences
    if results['n_old'] is not None and results['n_new'] is not None:
        results['n_diff'] = np.abs(results['n_new'] - results['n_old'])

        # Relative difference (avoid division by zero)
        denom = np.maximum(np.abs(results['n_new']), 1e-10)
        results['n_rel_diff'] = results['n_diff'] / denom

        # Check if they match (< 1% relative error)
        max_rel_diff = np.max(results['n_rel_diff'])
        results['n_match'] = max_rel_diff < 0.01

        print(f"  [{label}] n(r): max rel diff = {max_rel_diff:.2e} " +
              f"({'✓ MATCH' if results['n_match'] else '✗ MISMATCH'})")

    # =========================================================================
    # Scalar input/output test
    # =========================================================================
    print(f"  [{label}] Testing scalar input/output...")
    test_r = 0.5 * rCloud  # Test at middle of cloud

    # OLD scalar test
    try:
        n_scalar_old = get_density_profile_OLD(test_r, params)
        results['scalar_test_old'] = {
            'input': test_r,
            'output': n_scalar_old,
            'output_type': type(n_scalar_old).__name__,
            'is_scalar': np.ndim(n_scalar_old) == 0,
        }
        scalar_status_old = "scalar" if np.ndim(n_scalar_old) == 0 else f"array({type(n_scalar_old).__name__})"
        print(f"           OLD: n({test_r:.2f}) = {np.asarray(n_scalar_old).flatten()[0]:.2e} [{scalar_status_old}]")
    except Exception as e:
        print(f"           OLD scalar test failed: {e}")

    # NEW scalar test
    try:
        n_scalar_new = get_density_profile_NEW(test_r, params)
        results['scalar_test_new'] = {
            'input': test_r,
            'output': n_scalar_new,
            'output_type': type(n_scalar_new).__name__,
            'is_scalar': np.ndim(n_scalar_new) == 0,
        }
        scalar_status_new = "scalar" if np.ndim(n_scalar_new) == 0 else f"array({type(n_scalar_new).__name__})"
        print(f"           NEW: n({test_r:.2f}) = {n_scalar_new:.2e} [{scalar_status_new}]")
    except Exception as e:
        print(f"           NEW scalar test failed: {e}")

    # =========================================================================
    # Verify boundary conditions
    # =========================================================================
    print(f"  [{label}] Verifying boundary conditions...")
    print(f"           nCore = {nCore:.2e}, nISM = {nISM:.2e}")
    print(f"           rCore = {rCore:.4f} pc, rCloud = {rCloud:.4f} pc")

    # Test at specific locations
    test_points = [
        (0.5 * rCore, "inside core", nCore),
        (rCore, "at rCore", nCore),
        (0.5 * (rCore + rCloud), "in envelope", None),  # Depends on profile
        (rCloud, "at rCloud", None),  # Depends on profile
        (1.5 * rCloud, "outside cloud", nISM),
    ]

    for r_test, location, expected in test_points:
        try:
            n_new_test = get_density_profile_NEW(r_test, params)
            if expected is not None:
                match = "✓" if np.isclose(n_new_test, expected, rtol=0.01) else "✗"
                print(f"           NEW n({r_test:.3f}) = {n_new_test:.2e} {match} (expected {expected:.2e}) [{location}]")
            else:
                print(f"           NEW n({r_test:.3f}) = {n_new_test:.2e} [{location}]")
        except Exception as e:
            print(f"           NEW n({r_test:.3f}) failed: {e}")

    return results


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_density_comparison_grid(results_dict, save_path=None):
    """
    Create 3×3 grid comparing OLD vs NEW density profiles.

    Rows: OLD, NEW, Difference
    Cols: α=0, α=-2, Bonnor-Ebert

    Includes horizontal lines at nCore and nISM,
    vertical lines at rCore and rCloud.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)

    cases = ['alpha0', 'alpha2', 'be']
    titles = [r'$\alpha = 0$', r'$\alpha = -2$', 'Bonnor-Ebert']
    row_labels = ['OLD', 'NEW', '|Diff|']

    for j, (case, title) in enumerate(zip(cases, titles)):
        res = results_dict[case]
        r = res['r']
        nCore = res.get('nCore', None)
        nISM = res.get('nISM', None)
        rCloud = res.get('rCloud', None)
        rCore = res.get('rCore', None)

        # Row 0: OLD
        if res['n_old'] is not None:
            axes[0, j].plot(r, res['n_old'], 'b-', lw=1.5)
            axes[0, j].set_yscale('log')
            # Add reference lines
            if nCore is not None:
                axes[0, j].axhline(nCore, ls='--', color='green', alpha=0.7, lw=1, label=r'$n_{\rm core}$')
            if nISM is not None:
                axes[0, j].axhline(nISM, ls='--', color='red', alpha=0.7, lw=1, label=r'$n_{\rm ISM}$')
            if rCloud is not None:
                axes[0, j].axvline(rCloud, ls=':', color='orange', alpha=0.7, lw=1, label=r'$r_{\rm cloud}$')
            if rCore is not None:
                axes[0, j].axvline(rCore, ls=':', color='purple', alpha=0.7, lw=1, label=r'$r_{\rm core}$')
        else:
            axes[0, j].text(0.5, 0.5, 'FAILED', ha='center', va='center',
                           transform=axes[0, j].transAxes, fontsize=12, color='red')

        if j == 0:
            axes[0, j].set_ylabel(r'$n(r)$ [cm$^{-3}$]' + f'\n{row_labels[0]}')
            axes[0, j].legend(loc='upper right', fontsize=6)
        axes[0, j].set_title(title)

        # Row 1: NEW
        if res['n_new'] is not None:
            axes[1, j].plot(r, res['n_new'], 'r-', lw=1.5)
            axes[1, j].set_yscale('log')
            # Add reference lines
            if nCore is not None:
                axes[1, j].axhline(nCore, ls='--', color='green', alpha=0.7, lw=1)
            if nISM is not None:
                axes[1, j].axhline(nISM, ls='--', color='red', alpha=0.7, lw=1)
            if rCloud is not None:
                axes[1, j].axvline(rCloud, ls=':', color='orange', alpha=0.7, lw=1)
            if rCore is not None:
                axes[1, j].axvline(rCore, ls=':', color='purple', alpha=0.7, lw=1)

        if j == 0:
            axes[1, j].set_ylabel(r'$n(r)$ [cm$^{-3}$]' + f'\n{row_labels[1]}')

        # Row 2: Difference
        if res['n_diff'] is not None:
            axes[2, j].plot(r, res['n_rel_diff'] * 100, 'k-', lw=1.5)
            axes[2, j].set_yscale('log')
            axes[2, j].axhline(1.0, ls='--', color='gray', alpha=0.5, label='1%')
            if rCloud is not None:
                axes[2, j].axvline(rCloud, ls=':', color='orange', alpha=0.7, lw=1)
            if rCore is not None:
                axes[2, j].axvline(rCore, ls=':', color='purple', alpha=0.7, lw=1)

        if j == 0:
            axes[2, j].set_ylabel(f'Rel. Diff [%]\n{row_labels[2]}')
        axes[2, j].set_xlabel(r'$r$ [pc]')

    fig.suptitle(r'Density Profile $n(r)$ Comparison: OLD vs NEW', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_overlay_comparison(results_dict, save_path=None):
    """
    Create 1×3 grid with OLD and NEW overlaid on same axes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    cases = ['alpha0', 'alpha2', 'be']
    titles = [r'$\alpha = 0$ (homogeneous)', r'$\alpha = -2$ (isothermal)', 'Bonnor-Ebert']

    for j, (case, title) in enumerate(zip(cases, titles)):
        res = results_dict[case]
        r = res['r']
        nCore = res.get('nCore', None)
        nISM = res.get('nISM', None)
        rCloud = res.get('rCloud', None)
        rCore = res.get('rCore', None)

        ax = axes[j]

        if res['n_old'] is not None:
            ax.plot(r, res['n_old'], 'b-', lw=2, label='OLD', alpha=0.7)
        if res['n_new'] is not None:
            ax.plot(r, res['n_new'], 'r--', lw=2, label='NEW', alpha=0.7)

        # Add reference lines
        if nCore is not None:
            ax.axhline(nCore, ls='--', color='green', alpha=0.5, lw=1, label=r'$n_{\rm core}$')
        if nISM is not None:
            ax.axhline(nISM, ls='--', color='gray', alpha=0.5, lw=1, label=r'$n_{\rm ISM}$')
        if rCloud is not None:
            ax.axvline(rCloud, ls=':', color='orange', alpha=0.5, lw=1, label=r'$r_{\rm cloud}$')
        if rCore is not None:
            ax.axvline(rCore, ls=':', color='purple', alpha=0.5, lw=1, label=r'$r_{\rm core}$')

        ax.set_yscale('log')
        ax.set_xlabel(r'$r$ [pc]')
        ax.set_title(title)

        if j == 0:
            ax.set_ylabel(r'$n(r)$ [cm$^{-3}$]')
            ax.legend(loc='upper right', fontsize=7)

    fig.suptitle('Density Profile Comparison: OLD (blue) vs NEW (red dashed)', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    return fig


def print_summary_table(results_dict):
    """Print a summary table of comparison results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Test Case':<20} {'n(r) Match?':<15} {'Scalar OK?':<15} {'Notes'}")
    print("-" * 80)

    for case, title in [('alpha0', 'α = 0'), ('alpha2', 'α = -2'), ('be', 'Bonnor-Ebert')]:
        res = results_dict[case]

        # n(r) status
        if res.get('n_old_error'):
            n_status = '✗ OLD FAILED'
        elif res.get('n_new_error'):
            n_status = '✗ NEW FAILED'
        elif res['n_match']:
            n_status = '✓ MATCH'
        else:
            n_status = '✗ MISMATCH'

        # Scalar test status
        scalar_old = res.get('scalar_test_old') or {}
        scalar_new = res.get('scalar_test_new') or {}
        if scalar_old.get('is_scalar') and scalar_new.get('is_scalar'):
            scalar_status = '✓ Both scalar'
        elif scalar_new.get('is_scalar') and not scalar_old:
            scalar_status = '✗ OLD failed'
        elif scalar_new.get('is_scalar'):
            scalar_status = '✗ OLD returns array'
        else:
            scalar_status = '✗ Check needed'

        # Notes
        notes = []
        if res['n_rel_diff'] is not None:
            notes.append(f"max err: {np.max(res['n_rel_diff'])*100:.2f}%")

        print(f"{title:<20} {n_status:<15} {scalar_status:<15} {', '.join(notes)}")

    print("=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run all comparisons."""

    print("\n" + "=" * 80)
    print("DENSITY PROFILE COMPARISON: OLD vs NEW")
    print("=" * 80)

    results = {}

    # =========================================================================
    # Test Case 1: Power-law α = 0 (homogeneous)
    # =========================================================================
    print("\n[TEST 1] Power-law α = 0 (homogeneous cloud)")
    print("-" * 60)

    params_alpha0 = create_powerlaw_params(alpha=0.0)
    rCloud_alpha0 = params_alpha0['rCloud'].value
    rCore_alpha0 = params_alpha0['rCore'].value
    # Test radii from 0.01*rCloud to 2*rCloud
    r_arr = np.linspace(0.01 * rCloud_alpha0, 2.0 * rCloud_alpha0, 200)

    results['alpha0'] = compare_density_profiles(
        r_arr, params_alpha0, label='α=0'
    )

    # =========================================================================
    # Test Case 2: Power-law α = -2 (isothermal)
    # =========================================================================
    print("\n[TEST 2] Power-law α = -2 (isothermal sphere)")
    print("-" * 60)

    params_alpha2 = create_powerlaw_params(alpha=-2.0)
    rCloud_alpha2 = params_alpha2['rCloud'].value
    rCore_alpha2 = params_alpha2['rCore'].value
    # Test radii from 0.01*rCore to 2*rCloud
    r_arr = np.linspace(0.01 * rCore_alpha2, 2.0 * rCloud_alpha2, 200)

    results['alpha2'] = compare_density_profiles(
        r_arr, params_alpha2, label='α=-2'
    )

    # =========================================================================
    # Test Case 3: Bonnor-Ebert sphere
    # =========================================================================
    print("\n[TEST 3] Bonnor-Ebert sphere")
    print("-" * 60)

    params_be, be_result = create_bonnor_ebert_params()
    rCloud_be = params_be['rCloud'].value
    # Test radii from small to 2*rCloud
    r_arr = np.linspace(0.01 * rCloud_be, 2.0 * rCloud_be, 200)

    results['be'] = compare_density_profiles(
        r_arr, params_be, label='BE'
    )

    # =========================================================================
    # Generate plots
    # =========================================================================
    print("\n[PLOTTING]")
    print("-" * 60)

    # Density comparison grid
    plot_density_comparison_grid(
        results,
        save_path=FIG_DIR / 'density_profile_comparison.pdf'
    )

    # Overlay comparison
    plot_overlay_comparison(
        results,
        save_path=FIG_DIR / 'density_profile_overlay_comparison.pdf'
    )

    # =========================================================================
    # Print summary
    # =========================================================================
    print_summary_table(results)

    plt.show()


if __name__ == "__main__":
    main()
