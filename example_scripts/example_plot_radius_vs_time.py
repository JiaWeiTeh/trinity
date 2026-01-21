#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Output Reader - Plotting Examples

This script demonstrates how to use the TrinityOutput reader to create
publication-quality plots of simulation results.

Run from the trinity root directory:
    python example_scripts/example_plot_radius_vs_time.py

Outputs:
    - example_scripts/plot_radius_vs_time.png
    - example_scripts/plot_multi_panel.png
    - example_scripts/plot_phase_portrait.png

@author: TRINITY Team
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src._output.trinity_reader import TrinityOutput
import numpy as np

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = 'example_scripts/example_dictionary_1e7_sfe001_n1e4.jsonl'
SAVE_DIR = 'example_scripts'

# Plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
}) if HAS_MATPLOTLIB else None


def plot_radius_vs_time(output):
    """
    Simple plot of radius vs time.

    Demonstrates basic usage of output.get() for time series data.
    """
    print("Creating: plot_radius_vs_time.png")

    # Extract data
    t = output.get('t_now')
    R2 = output.get('R2')

    # Get phase information for coloring
    phases = output.get('current_phase', as_array=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot by phase with different colors
    energy_mask = np.array([p == 'energy' for p in phases])
    implicit_mask = np.array([p == 'implicit' for p in phases])

    ax.plot(t[energy_mask], R2[energy_mask], 'b-', lw=2,
            label='Energy phase', alpha=0.8)
    ax.plot(t[implicit_mask], R2[implicit_mask], 'r-', lw=2,
            label='Implicit phase', alpha=0.8)

    # Mark phase transition
    if np.any(energy_mask) and np.any(implicit_mask):
        t_transition = t[energy_mask][-1]
        ax.axvline(t_transition, color='gray', ls='--', alpha=0.5,
                   label=f'Phase transition (t={t_transition:.4f} Myr)')

    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Shell Radius R$_2$ [pc]')
    ax.set_title(f'Shell Expansion: {output.model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add text annotation
    ax.text(0.02, 0.98, f'Final R$_2$ = {R2[-1]:.2f} pc\nFinal t = {t[-1]:.3f} Myr',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot_radius_vs_time.png')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/plot_radius_vs_time.png")


def plot_multi_panel(output):
    """
    Multi-panel plot showing various simulation quantities.

    Demonstrates filtering and accessing multiple parameters.
    """
    print("Creating: plot_multi_panel.png")

    # Extract data
    t = output.get('t_now')
    R2 = output.get('R2')
    v2 = output.get('v2')
    Eb = output.get('Eb')
    T0 = output.get('T0')
    Pb = output.get('Pb')
    shell_mass = output.get('shell_mass')

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Radius
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, R2, 'b-', lw=1.5)
    ax1.set_ylabel('Radius R$_2$ [pc]')
    ax1.set_title('Shell Radius')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    v2_kms = v2 * 0.978  # Convert pc/Myr to km/s
    ax2.plot(t, v2_kms, 'g-', lw=1.5)
    ax2.set_ylabel('Velocity v$_2$ [km/s]')
    ax2.set_title('Shell Velocity')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Energy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(t, Eb, 'r-', lw=1.5)
    ax3.set_ylabel('Energy E$_b$ [erg]')
    ax3.set_title('Bubble Energy')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Temperature
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t, T0, 'orange', lw=1.5)
    ax4.set_ylabel('Temperature T$_0$ [K]')
    ax4.set_title('Bubble Temperature')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Pressure
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(t, Pb, 'm-', lw=1.5)
    ax5.set_xlabel('Time [Myr]')
    ax5.set_ylabel('Pressure P$_b$ [dyn/cm$^2$]')
    ax5.set_title('Bubble Pressure')
    ax5.grid(True, alpha=0.3)

    # Panel 6: Shell Mass
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t, shell_mass, 'c-', lw=1.5)
    ax6.set_xlabel('Time [Myr]')
    ax6.set_ylabel('Shell Mass [M$_\\odot$]')
    ax6.set_title('Swept-up Mass')
    ax6.grid(True, alpha=0.3)

    # Main title
    fig.suptitle(f'TRINITY Simulation: {output.model_name}', fontsize=14, y=1.02)

    plt.savefig(f'{SAVE_DIR}/plot_multi_panel.png')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/plot_multi_panel.png")


def plot_phase_portrait(output):
    """
    Phase portrait and momentum analysis.

    Demonstrates using get_at_time() and calculated quantities.
    """
    print("Creating: plot_phase_portrait.png")

    # Filter to implicit phase only (more interesting dynamics)
    impl = output.filter(phase='implicit')

    t = impl.get('t_now')
    R2 = impl.get('R2')
    v2 = impl.get('v2')
    shell_mass = impl.get('shell_mass')

    # Calculate momentum
    v2_kms = v2 * 0.978
    momentum = shell_mass * v2_kms

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: R vs v (phase portrait)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(R2, v2_kms, c=t, cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlabel('Radius R$_2$ [pc]')
    ax1.set_ylabel('Velocity v$_2$ [km/s]')
    ax1.set_title('Phase Portrait (R$_2$ vs v$_2$)')
    cbar = plt.colorbar(scatter, ax=ax1, label='Time [Myr]')
    ax1.grid(True, alpha=0.3)

    # Add arrows to show direction
    n_arrows = 10
    indices = np.linspace(0, len(R2)-2, n_arrows, dtype=int)
    for i in indices:
        ax1.annotate('', xy=(R2[i+1], v2_kms[i+1]), xytext=(R2[i], v2_kms[i]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))

    # Panel 2: Momentum vs time
    ax2 = axes[0, 1]
    ax2.plot(t, momentum, 'b-', lw=1.5)
    ax2.axhline(momentum.max(), color='r', ls='--', alpha=0.5,
                label=f'Max = {momentum.max():.2e}')
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Momentum [M$_\\odot$ km/s]')
    ax2.set_title('Shell Momentum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Velocity vs time with key points
    ax3 = axes[1, 0]
    ax3.plot(t, v2_kms, 'g-', lw=1.5)

    # Mark specific times (using interpolation for smooth values)
    for t_mark in [0.01, 0.1, 0.5]:
        if t_mark <= t.max():
            # quiet=True suppresses interpolation messages during plotting
            snap = impl.get_at_time(t_mark, quiet=True)
            v_at_t = snap['v2'] * 0.978
            ax3.axvline(t_mark, color='gray', ls=':', alpha=0.5)
            ax3.plot(snap.t_now, v_at_t, 'ro', ms=8)
            ax3.annotate(f't={t_mark}', xy=(snap.t_now, v_at_t),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax3.set_xlabel('Time [Myr]')
    ax3.set_ylabel('Velocity v$_2$ [km/s]')
    ax3.set_title('Velocity Evolution')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Energy partition
    ax4 = axes[1, 1]

    # Kinetic energy: 0.5 * M * v^2
    # Convert units: M in Msun, v in km/s
    # 1 Msun = 1.989e33 g, 1 km/s = 1e5 cm/s
    # E_kin = 0.5 * M * v^2 in erg
    Msun_to_g = 1.989e33
    kms_to_cms = 1e5
    E_kin = 0.5 * shell_mass * Msun_to_g * (v2_kms * kms_to_cms)**2

    E_th = impl.get('Eb')

    ax4.semilogy(t, E_kin, 'b-', lw=1.5, label='Kinetic (shell)')
    ax4.semilogy(t, E_th, 'r-', lw=1.5, label='Thermal (bubble)')
    ax4.set_xlabel('Time [Myr]')
    ax4.set_ylabel('Energy [erg]')
    ax4.set_title('Energy Partition')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'Phase Analysis: {output.model_name} (Implicit Phase)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot_phase_portrait.png')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/plot_phase_portrait.png")


def plot_cooling_parameters(output):
    """
    Plot cooling parameters beta and delta evolution.

    Demonstrates accessing cooling-specific parameters.
    """
    print("Creating: plot_cooling_params.png")

    # Filter to implicit phase where cooling matters
    impl = output.filter(phase='implicit')

    t = impl.get('t_now')
    beta = impl.get('cool_beta')
    delta = impl.get('cool_delta')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Beta
    ax1 = axes[0]
    ax1.plot(t, beta, 'b-', lw=1.5)
    ax1.set_xlabel('Time [Myr]')
    ax1.set_ylabel(r'$\beta$')
    ax1.set_title(r'Pressure Parameter $\beta = -(t/P_b)(dP_b/dt)$')
    ax1.axhline(0, color='gray', ls='--', alpha=0.5)
    ax1.axhline(1, color='gray', ls='--', alpha=0.5)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    # Delta
    ax2 = axes[1]
    ax2.plot(t, delta, 'r-', lw=1.5)
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel(r'$\delta$')
    ax2.set_title(r'Temperature Parameter $\delta$')
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    ax2.axhline(-1, color='gray', ls='--', alpha=0.5)
    ax2.set_ylim(-1.1, 0.1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Cooling Parameters: {output.model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot_cooling_params.png')
    plt.close()
    print(f"  Saved: {SAVE_DIR}/plot_cooling_params.png")


def main():
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required for plotting examples.")
        print("Install with: pip install matplotlib")
        sys.exit(1)

    print("=" * 60)
    print("TRINITY Output Reader - Plotting Examples")
    print("=" * 60)
    print()

    # Open output file
    print(f"Opening: {OUTPUT_FILE}")
    output = TrinityOutput.open(OUTPUT_FILE)
    print(f"  {len(output)} snapshots, t=[{output.t_min:.4e}, {output.t_max:.4e}] Myr")
    print()

    # Generate plots
    plot_radius_vs_time(output)
    plot_multi_panel(output)
    plot_phase_portrait(output)
    plot_cooling_parameters(output)

    print()
    print("=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - {SAVE_DIR}/plot_radius_vs_time.png")
    print(f"  - {SAVE_DIR}/plot_multi_panel.png")
    print(f"  - {SAVE_DIR}/plot_phase_portrait.png")
    print(f"  - {SAVE_DIR}/plot_cooling_params.png")


if __name__ == '__main__':
    main()
