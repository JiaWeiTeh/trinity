#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to analyze parameter changes during TRINITY simulation.

This script reads a .jsonl output file and identifies which parameters
change the most over time, helping to determine which parameters should
be monitored for adaptive stepping.

Usage:
    python diagnostic_parameter_changes.py <path_to_output.jsonl>

@author: TRINITY Team
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def load_jsonl(filepath: str) -> list:
    """Load snapshots from a .jsonl file."""
    snapshots = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    return snapshots


def is_scalar(value) -> bool:
    """Check if a value is a scalar (int or float), not an array."""
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    # Check for numpy scalar types (when loaded from JSON, they become Python types)
    if isinstance(value, (list, dict)):
        return False
    return False


def extract_scalar_params(snapshots: list) -> dict:
    """
    Extract parameters that are scalars across all snapshots.

    Returns a dict of {param_name: [values_over_time]}
    """
    if not snapshots:
        return {}

    # Get all keys from first snapshot
    all_keys = set(snapshots[0].keys())

    # Filter for scalar parameters
    scalar_params = {}

    for key in all_keys:
        values = []
        is_valid = True

        for snap in snapshots:
            val = snap.get(key)
            if not is_scalar(val):
                is_valid = False
                break
            values.append(val)

        if is_valid and len(values) == len(snapshots):
            # Check if parameter actually changes (not constant)
            if len(set(values)) > 1:
                scalar_params[key] = np.array(values)

    return scalar_params


def compute_dex_changes(values: np.ndarray) -> np.ndarray:
    """
    Compute absolute dex (log10) change between consecutive snapshots.

    Returns array of length len(values)-1 with dex changes.
    """
    dex_changes = []

    for i in range(1, len(values)):
        old_val = values[i-1]
        new_val = values[i]

        # Skip if zero or sign change
        if old_val == 0 or new_val == 0:
            dex_changes.append(0.0)
            continue
        if (old_val > 0) != (new_val > 0):
            # Sign change - treat as 1 dex change
            dex_changes.append(1.0)
            continue

        try:
            dex = abs(np.log10(abs(new_val) / abs(old_val)))
            dex_changes.append(dex)
        except (ValueError, ZeroDivisionError):
            dex_changes.append(0.0)

    return np.array(dex_changes)


def compute_percent_changes(values: np.ndarray) -> np.ndarray:
    """
    Compute absolute percentage change between consecutive snapshots.

    Returns array of length len(values)-1 with percent changes.
    """
    pct_changes = []

    for i in range(1, len(values)):
        old_val = values[i-1]
        new_val = values[i]

        # Skip if zero
        if old_val == 0:
            if new_val == 0:
                pct_changes.append(0.0)
            else:
                pct_changes.append(100.0)  # Cap at 100%
            continue

        try:
            pct = abs((new_val - old_val) / old_val) * 100
            pct_changes.append(min(pct, 1000.0))  # Cap at 1000%
        except (ValueError, ZeroDivisionError):
            pct_changes.append(0.0)

    return np.array(pct_changes)


def get_phase_mask(snapshots: list, start_phase: str = 'energy_implicit') -> np.ndarray:
    """
    Get a boolean mask for snapshots starting from a specific phase.

    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries
    start_phase : str
        Phase name to start from (default: 'energy_implicit')

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates snapshot is in or after start_phase
    """
    phase_order = ['energy', 'energy_implicit', 'transition', 'momentum']

    mask = np.zeros(len(snapshots), dtype=bool)
    found_start = False

    for i, snap in enumerate(snapshots):
        phase = snap.get('phase', snap.get('current_phase', ''))

        if start_phase in str(phase):
            found_start = True

        if found_start:
            mask[i] = True

    # If we never found the phase, start from beginning
    if not found_start:
        mask[:] = True

    return mask


def analyze_parameter_changes(filepath: str, start_phase: str = 'energy_implicit'):
    """
    Analyze parameter changes from a .jsonl output file.

    Returns sorted list of (param_name, total_dex_change, max_dex_change, values)
    """
    print(f"Loading {filepath}...")
    snapshots = load_jsonl(filepath)
    print(f"Loaded {len(snapshots)} snapshots")

    # Filter to start from specified phase
    phase_mask = get_phase_mask(snapshots, start_phase)
    filtered_snapshots = [s for s, m in zip(snapshots, phase_mask) if m]
    print(f"Filtered to {len(filtered_snapshots)} snapshots starting from {start_phase}")

    if len(filtered_snapshots) < 2:
        print("Not enough snapshots to compute changes")
        return [], [], []

    # Extract scalar parameters
    print("Extracting scalar parameters...")
    scalar_params = extract_scalar_params(filtered_snapshots)
    print(f"Found {len(scalar_params)} scalar parameters that change over time")

    # Parameters to exclude from analysis
    exclude_prefixes = ['residual_']
    exclude_names = ['t_previousCoolingUpdate']

    # Compute changes for each parameter
    results = []
    for name, values in scalar_params.items():
        # Skip excluded parameters
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if name in exclude_names:
            continue

        dex_changes = compute_dex_changes(values)
        total_dex = np.nansum(dex_changes)  # Use nansum to handle NaN
        max_dex = np.nanmax(dex_changes) if len(dex_changes) > 0 else 0  # Use nanmax
        results.append((name, total_dex, max_dex, values, dex_changes))

    # Sort by total dex change (NaN values go to end)
    results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)

    # Extract time array
    t_values = np.array([s.get('t_now', s.get('t', i)) for i, s in enumerate(filtered_snapshots)])

    return results, t_values, filtered_snapshots


def plot_top_parameters(results: list, t_values: np.ndarray, n_top: int = 16,
                        output_path: str = None):
    """
    Plot grid showing top N parameters with most total dex change.
    """
    if len(results) == 0:
        print("No results to plot")
        return

    n_plot = min(n_top, len(results))

    # Determine grid size
    n_cols = 4
    n_rows = (n_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_plot > 1 else [axes]

    # Time array for dex changes (midpoints)
    if len(t_values) > 1:
        t_changes = (t_values[1:] + t_values[:-1]) / 2
    else:
        t_changes = t_values

    for i in range(n_plot):
        ax = axes[i]
        name, total_dex, max_dex, values, dex_changes = results[i]

        # Plot dex changes over time
        if len(dex_changes) > 0 and len(t_changes) == len(dex_changes):
            # Use scatter with color indicating magnitude
            scatter = ax.scatter(t_changes, dex_changes, c=dex_changes,
                               cmap='viridis', s=10, alpha=0.7)
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='0.1 dex threshold')

        ax.set_yscale('log')
        ax.set_ylim(1e-6, 10)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('dex change')
        ax.set_title(f'{name}\nsum={total_dex:.2f}, max={max_dex:.3f}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    # Hide unused subplots
    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Top {n_plot} Parameters by Total Dex Change\n(Red dashed = 0.1 dex adaptive threshold)',
                 fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    return fig


def print_summary(results: list, n_top: int = 30):
    """Print summary table of top parameters."""
    print("\n" + "="*70)
    print(f"{'Parameter':<35} {'Total Dex':>12} {'Max Dex':>12}")
    print("="*70)

    for i, (name, total_dex, max_dex, _, _) in enumerate(results[:n_top]):
        print(f"{name:<35} {total_dex:>12.4f} {max_dex:>12.4f}")

    print("="*70)

    # Check which ADAPTIVE_MONITOR_KEYS are in top results
    # This list should match ADAPTIVE_MONITOR_KEYS in run_*_phase_modified.py files
    adaptive_keys = [
        # Core state variables
        'R2', 'v2', 'Eb', 'T0', 'Pb', 'R1',
        # Feedback values
        'pdot_SN', 'Lmech_SN', 'pdotdot_total',
        # Cooling parameters
        'cool_delta', 'cool_beta',
        # Bubble properties
        'bubble_mass', 'bubble_r_Tb', 'bubble_LTotal',
        'bubble_L1Bubble', 'bubble_Lloss', 'bubble_dMdt',
        'bubble_L2Conduction', 'bubble_L3Intermediate',
        # Shell parameters
        'shell_mass', 'shell_massDot', 'shell_n0', 'shell_nMax',
        'shell_thickness', 'shell_tauKappaRatio', 'shell_fIonisedDust', 'rShell',
        # Force parameters
        'F_grav', 'F_SN', 'F_ram', 'F_ram_wind', 'F_ram_SN',
        'F_wind', 'F_ion_in', 'F_ion_out', 'F_rad', 'F_ISM',
    ]

    result_names = [r[0] for r in results]

    print("\nADAPTIVE_MONITOR_KEYS coverage:")
    print("-"*70)

    in_top_30 = []
    not_in_top_30 = []
    not_found = []

    for key in adaptive_keys:
        if key in result_names:
            rank = result_names.index(key) + 1
            if rank <= 30:
                in_top_30.append((key, rank))
            else:
                not_in_top_30.append((key, rank))
        else:
            not_found.append(key)

    print(f"\nIn top 30 ({len(in_top_30)}):")
    for key, rank in sorted(in_top_30, key=lambda x: x[1]):
        print(f"  #{rank:2d}: {key}")

    print(f"\nOutside top 30 ({len(not_in_top_30)}):")
    for key, rank in sorted(not_in_top_30, key=lambda x: x[1]):
        print(f"  #{rank:2d}: {key}")

    print(f"\nNot found/constant ({len(not_found)}):")
    for key in not_found:
        print(f"  {key}")

    # Suggest parameters not in ADAPTIVE_MONITOR_KEYS but in top 30
    print("\n" + "-"*70)
    print("Parameters in top 30 NOT in ADAPTIVE_MONITOR_KEYS:")
    for i, (name, total_dex, max_dex, _, _) in enumerate(results[:30]):
        if name not in adaptive_keys:
            print(f"  #{i+1:2d}: {name} (total={total_dex:.4f}, max={max_dex:.4f})")


def main():
    # Default test file path (relative to this script's location)
    script_dir = Path(__file__).parent.parent.parent  # Go up to trinity root
    default_file = script_dir / 'test' / '1e7_sfe001_n1e4_test_dictionary.jsonl'

    parser = argparse.ArgumentParser(
        description='Analyze parameter changes in TRINITY simulation output'
    )
    parser.add_argument('filepath', type=str, nargs='?', default=None,
                        help='Path to .jsonl output file (default: test/1e7_sfe001_n1e4_test_dictionary.jsonl)')
    parser.add_argument('--phase', type=str, default='energy_implicit',
                        help='Starting phase (default: energy_implicit)')
    parser.add_argument('--top', type=int, default=16,
                        help='Number of top parameters to plot (default: 16)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: show interactively)')

    args = parser.parse_args()

    # Use default file if none specified
    filepath = Path(args.filepath) if args.filepath else default_file

    # Check file exists
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        if not args.filepath:
            print(f"(Tried default: {default_file})")
        sys.exit(1)

    filepath = str(filepath)

    # Analyze
    results, t_values, snapshots = analyze_parameter_changes(filepath, args.phase)

    if not results:
        print("No results to display")
        sys.exit(1)

    # Print summary
    print_summary(results, n_top=30)

    # Plot
    plot_top_parameters(results, t_values, n_top=args.top, output_path=args.output)


if __name__ == '__main__':
    main()
