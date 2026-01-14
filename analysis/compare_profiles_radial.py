#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radial Profile Comparison: Mass and Density
============================================

Compare mass M(r) and density n(r) radial profiles for different cloud models
using the REFACTORED implementations.

Cloud models:
- Power-law α = 0 (homogeneous)
- Power-law α = -1
- Power-law α = -2 (isothermal)
- Bonnor-Ebert sphere at critical Ω ≈ 14.04

All with same inputs: mCloud = 1e7 Msun, nCore = 1e3 cm⁻³

@author: Claude Code
@date: 2026-01-13
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_analysis_dir = os.path.dirname(os.path.abspath(__file__))
_functions_dir = os.path.join(_project_root, 'src', '_functions')

for path in [_project_root, _analysis_dir, _functions_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import refactored functions using explicit paths to avoid naming conflicts
# (There's a mass_profile.py file in src/cloud_properties that conflicts with
#  the mass_profile/ directory in analysis/)
import importlib.util

def _import_from_path(module_name, file_path):
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_density_module = _import_from_path(
    "REFACTORED_density_profile",
    os.path.join(_analysis_dir, "density_profile", "REFACTORED_density_profile.py")
)
_mass_module = _import_from_path(
    "REFACTORED_mass_profile",
    os.path.join(_analysis_dir, "mass_profile", "REFACTORED_mass_profile.py")
)

get_density_profile = _density_module.get_density_profile
get_mass_profile = _mass_module.get_mass_profile

# Import utilities
from src.cloud_properties.powerLawSphere import (
    compute_rCloud_powerlaw,
    compute_rCloud_homogeneous,
    DENSITY_CONVERSION
)

# Import BE sphere module
_bonnor_ebert_dir = os.path.join(_analysis_dir, 'bonnorEbert')
if _bonnor_ebert_dir not in sys.path:
    sys.path.insert(0, _bonnor_ebert_dir)

from bonnorEbertSphere_v2 import (
    create_BE_sphere,
    solve_lane_emden,
    OMEGA_CRITICAL
)

# Try to use TRINITY style file
try:
    _style_path = os.path.join(_project_root, 'src', '_plots', 'style_paper.mplstyle')
    if os.path.exists(_style_path):
        plt.style.use(_style_path)
        print(f"  Using style: {_style_path}")
except Exception as e:
    print(f"  (Could not load style: {e})")

# Disable LaTeX (not available in this environment)
plt.rcParams['text.usetex'] = False
print("  (LaTeX disabled - using standard fonts)")


# =============================================================================
# Parameters
# =============================================================================

# Common inputs
M_CLOUD = 1e7     # Msun
N_CORE = 1e3      # cm⁻³
MU = 1.4          # Mean molecular weight
N_ISM = 1.0       # cm⁻³
R_CORE_FRACTION = 0.1  # rCore/rCloud


# =============================================================================
# Helper: Create power-law parameters dict
# =============================================================================

class MockValue:
    """Simple container to mimic params['key'].value pattern."""
    def __init__(self, value):
        self.value = value


def create_powerlaw_params(mCloud, nCore, alpha, mu=1.4, nISM=1.0, rCore_frac=0.1):
    """Create parameter dict for power-law density profile."""

    if alpha == 0:
        rCloud = compute_rCloud_homogeneous(mCloud, nCore, mu=mu)
        rCore = rCloud * rCore_frac
    else:
        rCloud, rCore = compute_rCloud_powerlaw(
            mCloud, nCore, alpha,
            rCore_fraction=rCore_frac, mu=mu
        )

    # Edge density
    if alpha == 0:
        nEdge = nCore
    else:
        nEdge = nCore * (rCloud / rCore) ** alpha

    params = {
        'mCloud': MockValue(mCloud),
        'nCore': MockValue(nCore),
        'rCloud': MockValue(rCloud),
        'rCore': MockValue(rCore),
        'nISM': MockValue(nISM),
        'mu_n': MockValue(mu),
        'mu_ion': MockValue(mu),
        'mu_neu': MockValue(mu),
        'dens_profile': MockValue('densPL'),
        'densPL_alpha': MockValue(alpha),
    }

    print(f"  α={alpha}: rCloud={rCloud:.2f} pc, rCore={rCore:.2f} pc, nEdge={nEdge:.1f} cm⁻³")

    return params, rCloud, rCore, nEdge


def create_BE_params(mCloud, nCore, Omega, mu=1.4, nISM=1.0):
    """Create parameter dict for Bonnor-Ebert sphere."""

    # Pre-solve Lane-Emden for efficiency
    le_solution = solve_lane_emden()

    # Create BE sphere
    be_result = create_BE_sphere(
        M_cloud=mCloud,
        n_core=nCore,
        Omega=Omega,
        mu=mu,
        lane_emden_solution=le_solution
    )

    rCloud = be_result.r_out
    nEdge = be_result.n_out

    # For BE sphere, rCore is defined by the BE profile itself
    # We use the xi_out to scale - typically rCore ~ rCloud / xi_out
    # But for simplicity, use same fraction as power-law
    rCore = rCloud * R_CORE_FRACTION

    params = {
        'mCloud': MockValue(mCloud),
        'nCore': MockValue(nCore),
        'rCloud': MockValue(rCloud),
        'rCore': MockValue(rCore),
        'nISM': MockValue(nISM),
        'mu_n': MockValue(mu),
        'mu_ion': MockValue(mu),
        'mu_neu': MockValue(mu),
        'gamma_adia': MockValue(5.0/3.0),
        'dens_profile': MockValue('densBE'),
        'densBE_f_rho_rhoc': MockValue(le_solution.f_rho_rhoc),
        'densBE_f_m': MockValue(le_solution.f_m),  # Mass function m(ξ) for analytical mass
        'densBE_omega': MockValue(Omega),
        'densBE_xi_out': MockValue(be_result.xi_out),
        'densBE_Teff': MockValue(be_result.T_eff),
    }

    stability = "STABLE" if be_result.is_stable else "UNSTABLE"
    print(f"  BE (Ω={Omega:.2f}): rCloud={rCloud:.2f} pc, nEdge={nEdge:.1f} cm⁻³, T={be_result.T_eff:.1f} K [{stability}]")

    return params, rCloud, rCore, nEdge


# =============================================================================
# Main plotting function
# =============================================================================

def main():
    """Create comparison plot of radial profiles."""

    print("\n" + "=" * 70)
    print("RADIAL PROFILE COMPARISON")
    print(f"mCloud = {M_CLOUD:.0e} Msun, nCore = {N_CORE:.0e} cm⁻³")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Create parameter sets for each profile
    # -------------------------------------------------------------------------
    print("\n[1] Setting up cloud parameters...")

    profiles = {}

    # Power-law α = 0 (homogeneous)
    params_a0, rCloud_a0, rCore_a0, nEdge_a0 = create_powerlaw_params(
        M_CLOUD, N_CORE, alpha=0, mu=MU
    )
    profiles['α = 0'] = {
        'params': params_a0,
        'rCloud': rCloud_a0,
        'color': 'C0',
        'ls': '-'
    }

    # Power-law α = -1
    params_a1, rCloud_a1, rCore_a1, nEdge_a1 = create_powerlaw_params(
        M_CLOUD, N_CORE, alpha=-1, mu=MU
    )
    profiles['α = -1'] = {
        'params': params_a1,
        'rCloud': rCloud_a1,
        'color': 'C1',
        'ls': '-'
    }

    # Power-law α = -2 (isothermal)
    params_a2, rCloud_a2, rCore_a2, nEdge_a2 = create_powerlaw_params(
        M_CLOUD, N_CORE, alpha=-2, mu=MU
    )
    profiles['α = -2'] = {
        'params': params_a2,
        'rCloud': rCloud_a2,
        'color': 'C2',
        'ls': '-'
    }

    # Bonnor-Ebert at critical omega
    # Note: BE sphere uses mu=2.33 (molecular gas)
    params_BE, rCloud_BE, rCore_BE, nEdge_BE = create_BE_params(
        M_CLOUD, N_CORE, Omega=OMEGA_CRITICAL, mu=MU
    )
    profiles['BE (critical)'] = {
        'params': params_BE,
        'rCloud': rCloud_BE,
        'color': 'C3',
        'ls': '--'
    }

    # -------------------------------------------------------------------------
    # Create radial arrays for each profile
    # -------------------------------------------------------------------------
    print("\n[2] Computing profiles...")

    # Determine overall radial range
    r_max = max(p['rCloud'] for p in profiles.values()) * 1.5

    for name, prof in profiles.items():
        rCloud = prof['rCloud']
        rCore = prof['params']['rCore'].value
        # Create radial array from 0.01 pc to 1.5 × rCloud
        # Start from near zero for proper numerical integration (esp. BE sphere)
        r_min = 0.01  # pc - small but nonzero to avoid singularity
        r_arr = np.logspace(np.log10(r_min), np.log10(rCloud * 1.3), 500)
        # Ensure key radii (rCore, rCloud) are included exactly in the array
        r_arr = np.sort(np.unique(np.append(r_arr, [rCore, rCloud])))

        # Compute density profile (element-wise for scalars)
        n_arr = np.array([get_density_profile(r, prof['params']) for r in r_arr])

        # Compute mass profile
        # For BE sphere, numerical integration needs the full sorted array
        profile_type = prof['params']['dens_profile'].value
        if profile_type == 'densBE':
            # Pass full array for proper numerical integration
            M_arr = get_mass_profile(r_arr, prof['params'], return_mdot=False)
        else:
            # Power-law has analytical formula, can compute element-wise
            M_arr = np.array([get_mass_profile(r, prof['params'], return_mdot=False) for r in r_arr])

        prof['r'] = r_arr
        prof['n'] = n_arr
        prof['M'] = M_arr

        # Verify M(rCloud) = mCloud
        if profile_type == 'densBE':
            # For BE sphere, interpolate from computed array
            M_at_rCloud = np.interp(rCloud, r_arr, M_arr)
        else:
            M_at_rCloud = get_mass_profile(rCloud, prof['params'], return_mdot=False)
        rel_err = abs(M_at_rCloud - M_CLOUD) / M_CLOUD * 100
        print(f"  {name}: M(rCloud) = {M_at_rCloud:.3e} Msun (error = {rel_err:.2f}%)")

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    print("\n[3] Creating plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Density profile n(r)
    ax1 = axes[0]
    for name, prof in profiles.items():
        ax1.plot(prof['r'], prof['n'],
                   color=prof['color'], ls=prof['ls'], lw=2,
                   label=name)
        # Mark rCloud
        ax1.axvline(prof['rCloud'], color=prof['color'], ls=':', alpha=0.5, lw=1)

    # Reference lines
    ax1.axhline(N_CORE, color='gray', ls='--', alpha=0.5, lw=1, label=f'$n_{{\\rm core}}$ = {N_CORE:.0e}')
    ax1.axhline(N_ISM, color='gray', ls=':', alpha=0.5, lw=1, label=f'$n_{{\\rm ISM}}$ = {N_ISM:.0f}')

    ax1.set_xlabel(r'Radius $r$ [pc]')
    ax1.set_ylabel(r'Number density $n(r)$ [cm$^{-3}$]')
    ax1.set_title(r'Density Profile')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0.5, r_max)
    ax1.set_ylim(0.5, N_CORE * 2)
    ax1.grid(True, alpha=0.3, which='both')

    # Right panel: Mass profile M(r)
    ax2 = axes[1]
    for name, prof in profiles.items():
        ax2.plot(prof['r'], prof['M'],
                   color=prof['color'], ls=prof['ls'], lw=2,
                   label=name)
        # Mark rCloud
        ax2.axvline(prof['rCloud'], color=prof['color'], ls=':', alpha=0.5, lw=1)

    # Reference line for mCloud
    ax2.axhline(M_CLOUD, color='gray', ls='--', alpha=0.7, lw=1.5,
                label=f'$M_{{\\rm cloud}}$ = {M_CLOUD:.0e}')

    ax2.set_xlabel(r'Radius $r$ [pc]')
    ax2.set_ylabel(r'Enclosed mass $M(r)$ [M$_\odot$]')
    ax2.set_title(r'Mass Profile')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0.5, r_max)
    ax2.set_ylim(1e3, M_CLOUD * 2)
    ax2.grid(True, alpha=0.3, which='both')

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()

    # Save figure
    os.makedirs('fig', exist_ok=True)
    out_path = 'fig/radial_profiles_comparison.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Also save PNG for quick viewing
    # out_path_png = 'fig/radial_profiles_comparison.png'
    # plt.savefig(out_path_png, dpi=150, bbox_inches='tight')
    # print(f"Saved: {out_path_png}")

    # plt.close()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Profile':<20} {'rCloud [pc]':>15} {'nEdge [cm⁻³]':>15} {'M(rCloud)/mCloud':>18}")
    print("-" * 70)

    for name, prof in profiles.items():
        rCloud = prof['rCloud']
        # Get edge density (at rCloud)
        nEdge = get_density_profile(rCloud * 0.999, prof['params'])
        # Get mass at rCloud (interpolate from computed array for BE sphere)
        profile_type = prof['params']['dens_profile'].value
        if profile_type == 'densBE':
            M_at_rCloud = np.interp(rCloud, prof['r'], prof['M'])
        else:
            M_at_rCloud = get_mass_profile(rCloud, prof['params'], return_mdot=False)
        ratio = M_at_rCloud / M_CLOUD
        print(f"{name:<20} {rCloud:>15.2f} {nEdge:>15.1f} {ratio:>18.6f}")

    print("=" * 70)


if __name__ == '__main__':
    main()
