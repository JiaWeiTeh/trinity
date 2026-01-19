#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare TRINITY output folders: original vs modified versions.

This script automatically scans the /outputs/ folder for *_modified directories,
finds their corresponding original versions, and generates comparison plots.

Usage:
    python test_compare_output_folders.py
    python test_compare_output_folders.py --output-dir /path/to/outputs

Can be run from any directory within the trinity project.

Author: TRINITY Team
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import re

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================

def find_project_root() -> Path:
    """
    Find the trinity project root directory.
    Works when run from any directory within the project.
    """
    # First try: relative to this file (when running from test/)
    script_based = Path(__file__).parent.parent.resolve()
    if (script_based / 'src').exists() and (script_based / 'param').exists():
        return script_based

    # Second try: search upward from current working directory
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() and (parent / 'param').exists():
            return parent

    # Fallback: use script-based path
    return script_based


PROJECT_ROOT = find_project_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs'

# Parameters of interest by category
ESSENTIAL_PARAMS = ['R1', 'R2', 'rShell', 'Pb', 'Eb', 'T0']


# =============================================================================
# Data loading functions
# =============================================================================

def load_jsonl(filepath: Path) -> list:
    """
    Load a JSONL file (line-delimited JSON).

    Each line is a JSON object representing one snapshot.
    """
    snapshots = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                snapshots.append(snap)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line {line_num}: {e}")
                continue

    # Sort by snapshot index if available
    if snapshots and 'snap_id' in snapshots[0]:
        snapshots.sort(key=lambda s: s.get('snap_id', 0))

    return snapshots


def extract_time_series(snapshots: list, key: str) -> tuple:
    """
    Extract time and parameter arrays from snapshots.

    Returns (time_array, value_array) as numpy arrays.
    Only returns valid data for scalar (int/float) values.
    """
    t = []
    values = []

    for snap in snapshots:
        t_val = snap.get('t_now', np.nan)
        val = snap.get(key)

        # Skip if value is None, a list/array, or non-numeric
        if val is None:
            val = np.nan
        elif isinstance(val, (list, dict)):
            # Skip arrays - we only want scalars
            val = np.nan
        elif not isinstance(val, (int, float)):
            val = np.nan

        t.append(t_val)
        values.append(val)

    t = np.array(t, dtype=float)
    values = np.array(values, dtype=float)

    # Sort by time
    order = np.argsort(t)
    return t[order], values[order]


def get_scalar_keys(snapshots: list) -> set:
    """
    Get all keys that have scalar (int/float) values.
    """
    scalar_keys = set()

    for snap in snapshots:
        for key, val in snap.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                scalar_keys.add(key)

    return scalar_keys


def is_evolving_parameter(snapshots: list, key: str, rtol: float = 1e-6) -> bool:
    """
    Check if a parameter evolves (changes value) between t.min and t.max.

    Returns True if the value at t_min differs from value at t_max.
    """
    t, values = extract_time_series(snapshots, key)

    # Filter out NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return False

    t_valid = t[valid_mask]
    v_valid = values[valid_mask]

    if len(v_valid) < 2:
        return False

    # Get values at t_min and t_max
    v_min = v_valid[np.argmin(t_valid)]
    v_max = v_valid[np.argmax(t_valid)]

    # Check if they differ (using relative tolerance)
    if v_min == 0 and v_max == 0:
        return False

    denominator = max(abs(v_min), abs(v_max), 1e-300)
    rel_diff = abs(v_max - v_min) / denominator

    return rel_diff > rtol


# =============================================================================
# Folder discovery functions
# =============================================================================

def find_modified_folders(output_dir: Path) -> list:
    """
    Find all *_modified folders in the output directory.

    Returns list of (original_path, modified_path) tuples.
    """
    pairs = []

    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return pairs

    # Find all folders ending with _modified
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.endswith('_modified'):
            # Find corresponding original folder
            original_name = item.name[:-9]  # Remove '_modified' suffix
            original_path = output_dir / original_name

            if original_path.exists() and original_path.is_dir():
                pairs.append((original_path, item))
            else:
                print(f"  Warning: No original folder found for {item.name}")
                print(f"    Expected: {original_path}")

    return pairs


def find_dictionary_jsonl(folder: Path) -> Path:
    """
    Find the dictionary.jsonl file in a folder.
    """
    jsonl_path = folder / 'dictionary.jsonl'
    if jsonl_path.exists():
        return jsonl_path

    # Also check for .json extension
    json_path = folder / 'dictionary.json'
    if json_path.exists():
        return json_path

    return None


# =============================================================================
# Plotting functions
# =============================================================================

def plot_comparison_grid(
    snapshots_orig: list,
    snapshots_mod: list,
    keys: list,
    title: str,
    output_path: Path,
    ncols: int = 3
):
    """
    Plot a grid comparing parameter evolution between original and modified.
    """
    # Filter to valid scalar keys
    valid_keys = []
    for key in keys:
        t_orig, v_orig = extract_time_series(snapshots_orig, key)
        t_mod, v_mod = extract_time_series(snapshots_mod, key)

        # Check if we have any valid data
        if (not np.all(np.isnan(v_orig))) or (not np.all(np.isnan(v_mod))):
            valid_keys.append(key)

    if not valid_keys:
        print(f"  No valid keys to plot for: {title}")
        return

    # Calculate grid dimensions
    n_params = len(valid_keys)
    n_rows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(n_rows, ncols, figsize=(5 * ncols, 4 * n_rows))

    # Handle different axes shapes
    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for idx, key in enumerate(valid_keys):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        t_orig, v_orig = extract_time_series(snapshots_orig, key)
        t_mod, v_mod = extract_time_series(snapshots_mod, key)

        # Plot both (time already in Myr)
        ax.plot(t_orig, v_orig, 'b-', label='Original', linewidth=1.5, alpha=0.8)
        ax.plot(t_mod, v_mod, 'r--', label='Modified', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Use log10 scale if median value > 1e4
        all_values = np.concatenate([v_orig[~np.isnan(v_orig)], v_mod[~np.isnan(v_mod)]])
        if len(all_values) > 0:
            median_val = np.median(np.abs(all_values[all_values != 0])) if np.any(all_values != 0) else 0
            if median_val > 1e4 and np.all(all_values[~np.isnan(all_values)] > 0):
                ax.set_yscale('log')
            else:
                # Use scientific notation for y-axis if values are very large/small
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 3))

    # Hide empty subplots
    for idx in range(n_params, n_rows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_all_comparison_plots(
    snapshots_orig: list,
    snapshots_mod: list,
    output_dir: Path,
    model_name: str
):
    """
    Generate all comparison plots for a model pair.
    """
    # Get all scalar keys from both datasets
    keys_orig = get_scalar_keys(snapshots_orig)
    keys_mod = get_scalar_keys(snapshots_mod)
    all_keys = keys_orig | keys_mod

    print(f"\n  Found {len(all_keys)} scalar parameters")

    # A) Shell parameters (keys starting with shell_)
    shell_keys = sorted([k for k in all_keys if k.startswith('shell_')])
    if shell_keys:
        print(f"  Plotting {len(shell_keys)} shell parameters...")
        plot_comparison_grid(
            snapshots_orig, snapshots_mod,
            shell_keys,
            f"{model_name}: Shell Parameters Comparison",
            output_dir / "comparison_shell_parameters.pdf"
        )

    # B) Bubble parameters (keys starting with bubble_)
    bubble_keys = sorted([k for k in all_keys if k.startswith('bubble_')])
    if bubble_keys:
        print(f"  Plotting {len(bubble_keys)} bubble parameters...")
        plot_comparison_grid(
            snapshots_orig, snapshots_mod,
            bubble_keys,
            f"{model_name}: Bubble Parameters Comparison",
            output_dir / "comparison_bubble_parameters.pdf"
        )

    # C) TRINITY essentials (R1, R2, rShell, Pb, Eb, T0)
    essential_keys = [k for k in ESSENTIAL_PARAMS if k in all_keys]
    if essential_keys:
        print(f"  Plotting {len(essential_keys)} essential parameters...")
        plot_comparison_grid(
            snapshots_orig, snapshots_mod,
            essential_keys,
            f"{model_name}: Essential Parameters Comparison",
            output_dir / "comparison_essential_parameters.pdf"
        )

    # D) Force parameters (keys containing F_)
    force_keys = sorted([k for k in all_keys if 'F_' in k])
    if force_keys:
        print(f"  Plotting {len(force_keys)} force parameters...")
        plot_comparison_grid(
            snapshots_orig, snapshots_mod,
            force_keys,
            f"{model_name}: Force Parameters Comparison",
            output_dir / "comparison_force_parameters.pdf"
        )

    # E) Remaining parameters - not in above categories but evolving
    categorized_keys = set(shell_keys) | set(bubble_keys) | set(essential_keys) | set(force_keys)
    remaining_keys = all_keys - categorized_keys

    # Filter to only evolving parameters (different at t_min vs t_max)
    evolving_remaining = []
    for key in sorted(remaining_keys):
        # Check if evolving in either original or modified
        evolves_orig = is_evolving_parameter(snapshots_orig, key)
        evolves_mod = is_evolving_parameter(snapshots_mod, key)
        if evolves_orig or evolves_mod:
            evolving_remaining.append(key)

    if evolving_remaining:
        print(f"  Plotting {len(evolving_remaining)} remaining evolving parameters...")
        plot_comparison_grid(
            snapshots_orig, snapshots_mod,
            evolving_remaining,
            f"{model_name}: Other Evolving Parameters Comparison",
            output_dir / "comparison_remaining_parameters.pdf"
        )


# =============================================================================
# Main comparison function
# =============================================================================

def compare_output_folders(output_dir: Path = None):
    """
    Main function to compare all original vs modified output folder pairs.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir = Path(output_dir)

    print("=" * 70)
    print("TRINITY Output Comparison: Original vs Modified")
    print("=" * 70)
    print(f"\nSearching for *_modified folders in: {output_dir}")

    # Find folder pairs
    pairs = find_modified_folders(output_dir)

    if not pairs:
        print("\nNo modified/original folder pairs found.")
        print("Make sure you have folders like 'model_name/' and 'model_name_modified/' in the output directory.")
        return False

    print(f"\nFound {len(pairs)} folder pair(s) to compare:")
    for orig, mod in pairs:
        print(f"  {orig.name}  <-->  {mod.name}")

    # Process each pair
    success_count = 0
    for orig_path, mod_path in pairs:
        print(f"\n{'-' * 70}")
        print(f"Comparing: {orig_path.name} vs {mod_path.name}")
        print(f"{'-' * 70}")

        # Find dictionary files
        orig_jsonl = find_dictionary_jsonl(orig_path)
        mod_jsonl = find_dictionary_jsonl(mod_path)

        if orig_jsonl is None:
            print(f"  ERROR: No dictionary.jsonl found in {orig_path}")
            continue
        if mod_jsonl is None:
            print(f"  ERROR: No dictionary.jsonl found in {mod_path}")
            continue

        print(f"  Original: {orig_jsonl}")
        print(f"  Modified: {mod_jsonl}")

        # Load data
        print("  Loading snapshots...")
        snapshots_orig = load_jsonl(orig_jsonl)
        snapshots_mod = load_jsonl(mod_jsonl)

        print(f"  Original: {len(snapshots_orig)} snapshots")
        print(f"  Modified: {len(snapshots_mod)} snapshots")

        if len(snapshots_orig) == 0 or len(snapshots_mod) == 0:
            print("  ERROR: Empty snapshot data")
            continue

        # Generate comparison plots (save to modified folder)
        print("  Generating comparison plots...")
        generate_all_comparison_plots(
            snapshots_orig, snapshots_mod,
            mod_path,  # Save to modified folder
            orig_path.name
        )

        # Print summary statistics
        print("\n  Summary Statistics:")
        for key in ESSENTIAL_PARAMS:
            t_orig, v_orig = extract_time_series(snapshots_orig, key)
            t_mod, v_mod = extract_time_series(snapshots_mod, key)

            if np.all(np.isnan(v_orig)) and np.all(np.isnan(v_mod)):
                continue

            # Get final values
            final_orig = v_orig[~np.isnan(v_orig)][-1] if any(~np.isnan(v_orig)) else np.nan
            final_mod = v_mod[~np.isnan(v_mod)][-1] if any(~np.isnan(v_mod)) else np.nan

            if np.isnan(final_orig) or np.isnan(final_mod):
                continue

            # Calculate relative difference
            rel_diff = abs(final_orig - final_mod) / max(abs(final_orig), 1e-300)
            status = "OK" if rel_diff < 0.01 else "DIFF"

            print(f"    {key:12s}: orig={final_orig:.4e}, mod={final_mod:.4e}, "
                  f"rel_diff={rel_diff:.2e} [{status}]")

        success_count += 1

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"Completed: {success_count}/{len(pairs)} comparisons successful")
    print(f"{'=' * 70}")

    return success_count == len(pairs)


# =============================================================================
# Command-line interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare TRINITY output folders: original vs modified versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --output-dir /path/to/outputs
  %(prog)s -o ./my_outputs

The script searches for folders ending with '_modified' and compares them
with their corresponding original folders (e.g., '1e7_modified/' vs '1e7/').

Comparison plots are saved as PDFs in the _modified folder.
Can be run from any directory within the trinity project.
        """
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help=f'Output directory to search (default: {DEFAULT_OUTPUT_DIR})'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    success = compare_output_folders(output_dir)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
