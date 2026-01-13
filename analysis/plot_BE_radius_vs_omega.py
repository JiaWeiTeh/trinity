#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bonnor-Ebert Sphere: Cloud Radius vs Density Contrast (Omega)
==============================================================

This script explores how the cloud radius of a Bonnor-Ebert sphere varies
with the density contrast parameter Omega = n_core / n_edge.

Literature Values:
------------------
- Critical density contrast: Ω_crit ≈ 14.04 (Bonnor 1956, Ebert 1955)
- Critical dimensionless radius: ξ_crit ≈ 6.45
- Spheres with Ω < 14.04 are gravitationally STABLE
- Spheres with Ω > 14.04 are gravitationally UNSTABLE

Observed Example - Barnard 68 (Alves et al. 2001):
--------------------------------------------------
- ξ = 6.9 ± 0.2 (slightly supercritical)
- M = 2.1 M☉
- n_core ≈ 2.4 × 10⁵ cm⁻³
- T = 16 K
- R ≈ 12,500 AU ≈ 0.06 pc

Typical Molecular Cloud Dense Cores:
------------------------------------
- Mass: 0.5 - 10 M☉
- n_core: 10⁴ - 10⁵ cm⁻³
- Temperature: 10-20 K

References:
-----------
- Bonnor (1956), MNRAS 116, 351
- Ebert (1955), Z. Astrophys. 37, 217
- Alves, Lada & Lada (2001), Nature 409, 159 (Barnard 68)
- Sipilä et al. (2011), A&A 535, A49 (non-isothermal BE spheres)

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

# Import BE sphere module
_bonnor_ebert_dir = os.path.join(_analysis_dir, 'bonnorEbert')
if _bonnor_ebert_dir not in sys.path:
    sys.path.insert(0, _bonnor_ebert_dir)

from bonnorEbertSphere_v2 import (
    create_BE_sphere,
    solve_lane_emden,
    OMEGA_CRITICAL,
    BESphereResult
)

# Try to use TRINITY style file
try:
    _style_path = os.path.join(_project_root, 'src', '_plots', 'style_paper.mplstyle')
    if os.path.exists(_style_path):
        plt.style.use(_style_path)
except:
    pass

# Disable LaTeX
plt.rcParams['text.usetex'] = False


# =============================================================================
# Literature values and typical parameters
# =============================================================================

# Critical value for BE sphere stability
OMEGA_CRIT = 14.04  # Bonnor (1956)

# Typical molecular cloud dense core parameters
TYPICAL_PARAMS = {
    'low_mass_core': {
        'M_cloud': 1.0,      # Msun
        'n_core': 1e4,       # cm⁻³
        'T_expected': 10,    # K (typical for cold cores)
        'description': 'Low-mass prestellar core'
    },
    'intermediate_core': {
        'M_cloud': 5.0,      # Msun
        'n_core': 5e4,       # cm⁻³
        'T_expected': 15,    # K
        'description': 'Intermediate prestellar core'
    },
    'barnard_68': {
        'M_cloud': 2.1,      # Msun (Alves et al. 2001)
        'n_core': 2.4e5,     # cm⁻³ (from ρ₀ = 1.0×10⁻¹⁸ g/cm³)
        'T_expected': 16,    # K (observed)
        'Omega_observed': 14.1,  # ξ = 6.9 corresponds to Ω ≈ 14
        'description': 'Barnard 68 (Alves+ 2001)'
    },
    'massive_core': {
        'M_cloud': 10.0,     # Msun
        'n_core': 1e5,       # cm⁻³
        'T_expected': 20,    # K
        'description': 'Massive prestellar core'
    }
}

# Mean molecular weight for molecular gas (H₂ dominated)
MU_MOLECULAR = 2.33


# =============================================================================
# Main functions
# =============================================================================

def compute_BE_properties(M_cloud, n_core, Omega_values, mu=MU_MOLECULAR):
    """
    Compute BE sphere properties for a range of Omega values.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun]
    n_core : float
        Core number density [cm⁻³]
    Omega_values : array
        Density contrast values to compute
    mu : float
        Mean molecular weight

    Returns
    -------
    results : dict
        Dictionary with arrays of r_out, T_eff, xi_out, n_edge, is_stable
    """
    # Pre-solve Lane-Emden for efficiency
    le_solution = solve_lane_emden()

    results = {
        'Omega': [],
        'r_out': [],
        'T_eff': [],
        'xi_out': [],
        'n_edge': [],
        'is_stable': []
    }

    for Omega in Omega_values:
        try:
            be = create_BE_sphere(
                M_cloud=M_cloud,
                n_core=n_core,
                Omega=Omega,
                mu=mu,
                lane_emden_solution=le_solution,
                validate=True
            )
            results['Omega'].append(Omega)
            results['r_out'].append(be.r_out)
            results['T_eff'].append(be.T_eff)
            results['xi_out'].append(be.xi_out)
            results['n_edge'].append(be.n_out)
            results['is_stable'].append(be.is_stable)
        except Exception as e:
            print(f"  Warning: Omega={Omega:.1f} failed: {e}")
            continue

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def main():
    """Create plots showing BE sphere radius dependence on Omega."""

    print("\n" + "=" * 70)
    print("BONNOR-EBERT SPHERE: RADIUS vs DENSITY CONTRAST (Ω)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Print literature information
    # -------------------------------------------------------------------------
    print("\n[LITERATURE VALUES]")
    print("-" * 70)
    print(f"Critical density contrast: Ω_crit = {OMEGA_CRIT:.2f} (Bonnor 1956)")
    print(f"  → Stable if Ω < {OMEGA_CRIT:.2f}")
    print(f"  → Unstable if Ω > {OMEGA_CRIT:.2f}")
    print("\nObserved example - Barnard 68 (Alves et al. 2001):")
    print(f"  M = 2.1 M☉, n_core = 2.4×10⁵ cm⁻³, T = 16 K")
    print(f"  ξ = 6.9 ± 0.2 → Ω ≈ 14 (marginally unstable)")

    # -------------------------------------------------------------------------
    # Compute BE properties for different Omega values
    # -------------------------------------------------------------------------
    print("\n[COMPUTING BE SPHERE PROPERTIES]")
    print("-" * 70)

    # Range of Omega values (from stable to unstable)
    Omega_range = np.concatenate([
        np.linspace(2, 10, 20),      # Stable regime
        np.linspace(10, 14, 10),     # Near-critical
        np.linspace(14, 20, 10),     # Unstable regime
    ])
    Omega_range = np.unique(Omega_range)

    # Compute for typical molecular cloud parameters
    # Using parameters similar to observed dense cores
    M_cloud = 5.0    # Msun - typical prestellar core
    n_core = 1e5     # cm⁻³ - typical dense core

    print(f"\nTypical molecular cloud dense core:")
    print(f"  M_cloud = {M_cloud} M☉")
    print(f"  n_core = {n_core:.0e} cm⁻³")
    print(f"  μ = {MU_MOLECULAR} (molecular gas)")

    results = compute_BE_properties(M_cloud, n_core, Omega_range)

    print(f"\nComputed {len(results['Omega'])} BE sphere configurations")

    # -------------------------------------------------------------------------
    # Also compute for different masses at critical Omega
    # -------------------------------------------------------------------------
    print("\n[MASS DEPENDENCE AT CRITICAL Ω]")
    print("-" * 70)

    mass_range = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])

    le_solution = solve_lane_emden()
    mass_results = {
        'M': [],
        'r_out': [],
        'T_eff': []
    }

    print(f"\n{'M [M☉]':>10} {'r_out [pc]':>12} {'T_eff [K]':>12}")
    print("-" * 36)

    for M in mass_range:
        try:
            be = create_BE_sphere(
                M_cloud=M,
                n_core=n_core,
                Omega=OMEGA_CRIT,
                mu=MU_MOLECULAR,
                lane_emden_solution=le_solution
            )
            mass_results['M'].append(M)
            mass_results['r_out'].append(be.r_out)
            mass_results['T_eff'].append(be.T_eff)
            print(f"{M:>10.1f} {be.r_out:>12.4f} {be.T_eff:>12.1f}")
        except Exception as e:
            print(f"{M:>10.1f} {'FAILED':>12} {str(e)[:20]}")

    for key in mass_results:
        mass_results[key] = np.array(mass_results[key])

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    print("\n[CREATING PLOTS]")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Plot 1: Cloud radius vs Omega ---
    ax1 = axes[0, 0]

    # Separate stable and unstable
    stable = results['is_stable']

    ax1.plot(results['Omega'][stable], results['r_out'][stable],
             'b-', lw=2, label='Stable (Ω < 14.04)')
    ax1.plot(results['Omega'][~stable], results['r_out'][~stable],
             'r--', lw=2, label='Unstable (Ω > 14.04)')

    # Mark critical point
    ax1.axvline(OMEGA_CRIT, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax1.text(OMEGA_CRIT + 0.3, ax1.get_ylim()[1] * 0.9,
             f'Ω$_{{crit}}$ = {OMEGA_CRIT:.2f}', fontsize=10, color='gray')

    ax1.set_xlabel('Density Contrast Ω = n$_{core}$/n$_{edge}$')
    ax1.set_ylabel('Cloud Radius r$_{out}$ [pc]')
    ax1.set_title(f'BE Sphere Radius vs Ω\n(M = {M_cloud} M$_\\odot$, n$_{{core}}$ = {n_core:.0e} cm$^{{-3}}$)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 20)

    # --- Plot 2: Effective temperature vs Omega ---
    ax2 = axes[0, 1]

    ax2.plot(results['Omega'][stable], results['T_eff'][stable],
             'b-', lw=2, label='Stable')
    ax2.plot(results['Omega'][~stable], results['T_eff'][~stable],
             'r--', lw=2, label='Unstable')

    ax2.axvline(OMEGA_CRIT, color='gray', ls=':', lw=1.5, alpha=0.7)

    # Mark typical molecular cloud temperature range
    ax2.axhspan(10, 20, color='green', alpha=0.15, label='Typical T (10-20 K)')

    ax2.set_xlabel('Density Contrast Ω')
    ax2.set_ylabel('Effective Temperature T$_{eff}$ [K]')
    ax2.set_title('BE Sphere Temperature vs Ω')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 20)

    # --- Plot 3: Dimensionless radius ξ vs Omega ---
    ax3 = axes[1, 0]

    ax3.plot(results['Omega'][stable], results['xi_out'][stable],
             'b-', lw=2, label='Stable')
    ax3.plot(results['Omega'][~stable], results['xi_out'][~stable],
             'r--', lw=2, label='Unstable')

    ax3.axvline(OMEGA_CRIT, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax3.axhline(6.45, color='orange', ls='--', lw=1.5, alpha=0.7,
                label='ξ$_{crit}$ = 6.45')

    # Mark Barnard 68
    ax3.axhline(6.9, color='purple', ls=':', lw=1.5, alpha=0.7,
                label='Barnard 68 (ξ = 6.9)')

    ax3.set_xlabel('Density Contrast Ω')
    ax3.set_ylabel('Dimensionless Radius ξ$_{out}$')
    ax3.set_title('Dimensionless Radius vs Ω')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2, 20)

    # --- Plot 4: Cloud radius vs Mass (at critical Omega) ---
    ax4 = axes[1, 1]

    ax4.loglog(mass_results['M'], mass_results['r_out'],
               'ko-', lw=2, markersize=8, label=f'Ω = {OMEGA_CRIT:.2f} (critical)')

    # Power-law fit
    log_M = np.log10(mass_results['M'])
    log_r = np.log10(mass_results['r_out'])
    slope, intercept = np.polyfit(log_M, log_r, 1)
    M_fit = np.logspace(-0.5, 2, 50)
    r_fit = 10**(slope * np.log10(M_fit) + intercept)
    ax4.loglog(M_fit, r_fit, 'g--', lw=1.5, alpha=0.7,
               label=f'Fit: r ∝ M$^{{{slope:.2f}}}$')

    # Mark Barnard 68
    ax4.axvline(2.1, color='purple', ls=':', lw=1.5, alpha=0.7)
    ax4.text(2.3, ax4.get_ylim()[0] * 2, 'B68', fontsize=10, color='purple')

    ax4.set_xlabel('Cloud Mass M [M$_\\odot$]')
    ax4.set_ylabel('Cloud Radius r$_{out}$ [pc]')
    ax4.set_title(f'BE Sphere Radius vs Mass\n(at critical Ω, n$_{{core}}$ = {n_core:.0e} cm$^{{-3}}$)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # Save figure
    os.makedirs('fig', exist_ok=True)
    out_path = 'fig/BE_radius_vs_omega.pdf'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    out_path_png = 'fig/BE_radius_vs_omega.png'
    plt.savefig(out_path_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path_png}")

    plt.close()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: BE SPHERE PROPERTIES FOR SELECTED Ω VALUES")
    print("=" * 70)
    print(f"(M = {M_cloud} M☉, n_core = {n_core:.0e} cm⁻³)")
    print("-" * 70)
    print(f"{'Ω':>8} {'ξ_out':>10} {'r_out [pc]':>12} {'T_eff [K]':>12} {'Stability':>12}")
    print("-" * 70)

    # Select key Omega values for summary
    key_omegas = [2, 5, 10, 13, OMEGA_CRIT, 15, 18]
    for omega_target in key_omegas:
        # Find closest computed value
        idx = np.argmin(np.abs(results['Omega'] - omega_target))
        O = results['Omega'][idx]
        xi = results['xi_out'][idx]
        r = results['r_out'][idx]
        T = results['T_eff'][idx]
        stable_str = "STABLE" if results['is_stable'][idx] else "UNSTABLE"
        print(f"{O:>8.2f} {xi:>10.3f} {r:>12.4f} {T:>12.1f} {stable_str:>12}")

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Print key physics insight
    # -------------------------------------------------------------------------
    print("\n[KEY PHYSICS INSIGHT]")
    print("-" * 70)
    print("""
As Omega (density contrast) increases:
  1. Cloud radius DECREASES - higher central concentration
  2. Effective temperature INCREASES - more pressure support needed
  3. At Ω > 14.04, sphere becomes gravitationally unstable → collapse

For typical molecular cloud cores (T ~ 10-20 K):
  - Low Omega (Ω ~ 2-5): Large, diffuse, very stable
  - Medium Omega (Ω ~ 5-10): Compact, stable
  - Near-critical (Ω ~ 12-14): Compact, marginally stable → star formation
  - Supercritical (Ω > 14): Unstable → gravitational collapse
    """)


if __name__ == '__main__':
    main()
