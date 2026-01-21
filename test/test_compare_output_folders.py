#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare TRINITY output folders: original vs modified versions.

Two modes of operation:

1. Auto mode (default): Scans the outputs folder for *_modified directories,
   finds their corresponding original versions, and generates comparison plots.

2. Manual mode: Compare two specific folders/files directly.

Usage:
    # Auto mode - find and compare all *_modified pairs
    python test_compare_output_folders.py
    python test_compare_output_folders.py --output-dir /path/to/outputs

    # Manual mode - compare two specific folders/files
    python test_compare_output_folders.py folder1 folder2
    python test_compare_output_folders.py /path/to/run1 /path/to/run2

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
ESSENTIAL_PARAMS = ['R1', 'R2', 'rShell', 'Pb', 'Eb', 'T0', 'v2']


# =============================================================================
# Data loading functions
# =============================================================================

def load_jsonl(filepath: Path) -> list:
    """
    Load a JSONL file (line-delimited JSON).

    Each line is a JSON object representing one snapshot.
    Handles corrupted lines with multiple concatenated JSON objects.
    """
    snapshots = []
    corrupted_lines = []
    recovered_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                snapshots.append(snap)
            except json.JSONDecodeError as e:
                # Try to recover concatenated JSON objects (e.g., {...}{...})
                if "Extra data" in str(e):
                    recovered = _try_split_concatenated_json(line)
                    if recovered:
                        snapshots.extend(recovered)
                        recovered_count += len(recovered)
                        continue
                corrupted_lines.append(line_num)

    # Report corrupted lines summary (not individual warnings)
    if corrupted_lines:
        if len(corrupted_lines) <= 5:
            print(f"  Warning: Failed to parse lines: {corrupted_lines}")
        else:
            print(f"  Warning: Failed to parse {len(corrupted_lines)} lines "
                  f"(first 5: {corrupted_lines[:5]})")
    if recovered_count > 0:
        print(f"  Recovered {recovered_count} snapshots from concatenated lines")

    # Sort by snapshot index if available
    if snapshots and 'snap_id' in snapshots[0]:
        snapshots.sort(key=lambda s: s.get('snap_id', 0))

    return snapshots


def _try_split_concatenated_json(line: str) -> list:
    """
    Try to split a line containing multiple concatenated JSON objects.

    E.g., '{"a":1}{"b":2}' -> [{"a":1}, {"b":2}]
    """
    results = []
    decoder = json.JSONDecoder()
    idx = 0
    line = line.strip()

    while idx < len(line):
        # Skip whitespace
        while idx < len(line) and line[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(line):
            break

        try:
            obj, end_idx = decoder.raw_decode(line, idx)
            results.append(obj)
            idx += end_idx
        except json.JSONDecodeError:
            # Can't parse more, stop
            break

    return results if len(results) > 0 else None


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


def get_common_time_range(snapshots_orig: list, snapshots_mod: list) -> tuple:
    """
    Get the common time range between two snapshot sets.

    Returns (t_min, t_max) where t_max is the minimum of the two end times.
    This ensures we only compare overlapping time periods.
    """
    # Get time arrays
    t_orig = np.array([s.get('t_now', np.nan) for s in snapshots_orig], dtype=float)
    t_mod = np.array([s.get('t_now', np.nan) for s in snapshots_mod], dtype=float)

    # Filter NaN values
    t_orig = t_orig[~np.isnan(t_orig)]
    t_mod = t_mod[~np.isnan(t_mod)]

    if len(t_orig) == 0 or len(t_mod) == 0:
        return 0.0, 0.0

    # Common range: start at max of mins, end at min of maxs
    t_min = max(t_orig.min(), t_mod.min())
    t_max = min(t_orig.max(), t_mod.max())

    return t_min, t_max


def filter_snapshots_by_time(snapshots: list, t_min: float, t_max: float) -> list:
    """
    Filter snapshots to only include those within the time range [t_min, t_max].
    """
    filtered = []
    for snap in snapshots:
        t = snap.get('t_now', np.nan)
        if not np.isnan(t) and t_min <= t <= t_max:
            filtered.append(snap)
    return filtered


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
# Phase transition detection functions
# =============================================================================

def find_phase_transition_times(snapshots: list) -> dict:
    """
    Find times when simulation crosses into transition and momentum phases,
    and when R2 exceeds rCloud.

    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries

    Returns
    -------
    dict
        Dictionary with keys:
        - 't_transition': time entering transition phase (or None)
        - 't_momentum': time entering momentum phase (or None)
        - 't_R2_gt_rCloud': time when R2 > rCloud (or None)
    """
    result = {
        't_transition': None,
        't_momentum': None,
        't_R2_gt_rCloud': None
    }

    if not snapshots:
        return result

    # Sort snapshots by time
    sorted_snaps = sorted(snapshots, key=lambda s: s.get('t_now', 0))

    # Track previous phase for transition detection
    prev_phase = None

    for snap in sorted_snaps:
        t = snap.get('t_now')
        if t is None:
            continue

        # Check for phase transitions
        phase = snap.get('current_phase')
        if phase is not None and prev_phase is not None:
            # Detect transition phase entry
            # Phase can be 'transition', '2', '1c', or similar
            phase_str = str(phase).lower()
            prev_str = str(prev_phase).lower()

            if result['t_transition'] is None:
                if ('transition' in phase_str or phase_str == '2' or phase_str == '1c') and \
                   ('transition' not in prev_str and prev_str != '2' and prev_str != '1c'):
                    result['t_transition'] = t

            # Detect momentum phase entry
            if result['t_momentum'] is None:
                if ('momentum' in phase_str or phase_str == '3') and \
                   ('momentum' not in prev_str and prev_str != '3'):
                    result['t_momentum'] = t

        prev_phase = phase

        # Check for R2 > rCloud
        if result['t_R2_gt_rCloud'] is None:
            R2 = snap.get('R2')
            rCloud = snap.get('rCloud')
            if R2 is not None and rCloud is not None:
                if isinstance(R2, (int, float)) and isinstance(rCloud, (int, float)):
                    if R2 > rCloud:
                        result['t_R2_gt_rCloud'] = t

    return result


def merge_transition_times(times_orig: dict, times_mod: dict) -> dict:
    """
    Merge transition times from original and modified runs.

    For plotting, we want to show transitions from both runs if they differ.

    Returns
    -------
    dict
        Dictionary with keys like 't_transition_orig', 't_transition_mod', etc.
    """
    merged = {}

    for key in ['t_transition', 't_momentum', 't_R2_gt_rCloud']:
        t_orig = times_orig.get(key)
        t_mod = times_mod.get(key)

        merged[f'{key}_orig'] = t_orig
        merged[f'{key}_mod'] = t_mod

    return merged


# =============================================================================
# Plotting functions
# =============================================================================

def plot_comparison_grid(
    snapshots_orig: list,
    snapshots_mod: list,
    keys: list,
    title: str,
    output_path: Path,
    ncols: int = 3,
    transition_times: dict = None,
    label1: str = 'Original',
    label2: str = 'Modified'
):
    """
    Plot a grid comparing parameter evolution between original and modified.

    Parameters
    ----------
    snapshots_orig : list
        Original run snapshots
    snapshots_mod : list
        Modified run snapshots
    keys : list
        Parameter keys to plot
    title : str
        Plot title
    output_path : Path
        Output file path
    ncols : int
        Number of columns in grid
    transition_times : dict, optional
        Dictionary with transition times to mark with vertical lines:
        - 't_transition_orig', 't_transition_mod': transition phase entry
        - 't_momentum_orig', 't_momentum_mod': momentum phase entry
        - 't_R2_gt_rCloud_orig', 't_R2_gt_rCloud_mod': R2 > rCloud
    label1 : str
        Label for first dataset (default: 'Original')
    label2 : str
        Label for second dataset (default: 'Modified')
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

    # Define vertical line styles for transitions
    vline_styles = {
        't_transition': {'color': 'green', 'label': 'Transition phase'},
        't_momentum': {'color': 'purple', 'label': 'Momentum phase'},
        't_R2_gt_rCloud': {'color': 'orange', 'label': 'R2 > rCloud'}
    }

    for idx, key in enumerate(valid_keys):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        t_orig, v_orig = extract_time_series(snapshots_orig, key)
        t_mod, v_mod = extract_time_series(snapshots_mod, key)

        # Plot both (time already in Myr)
        ax.plot(t_orig, v_orig, 'b-', label=label1, linewidth=1.5, alpha=0.8)
        ax.plot(t_mod, v_mod, 'r--', label=label2, linewidth=1.5, alpha=0.8)

        # Add vertical lines for phase transitions
        if transition_times:
            added_labels = set()  # Track which labels we've added to avoid duplicates
            for base_key, style in vline_styles.items():
                # Check first dataset
                t_orig_trans = transition_times.get(f'{base_key}_orig')
                if t_orig_trans is not None:
                    vline_label = f"{style['label']} ({label1})" if f"{style['label']} ({label1})" not in added_labels else None
                    ax.axvline(x=t_orig_trans, color=style['color'], linestyle='--',
                               linewidth=1.0, alpha=0.7, label=vline_label)
                    if vline_label:
                        added_labels.add(vline_label)

                # Check second dataset
                t_mod_trans = transition_times.get(f'{base_key}_mod')
                if t_mod_trans is not None:
                    # Only plot if different from first (to avoid clutter)
                    if t_orig_trans is None or abs(t_mod_trans - t_orig_trans) > 1e-10:
                        vline_label = f"{style['label']} ({label2})" if f"{style['label']} ({label2})" not in added_labels else None
                        ax.axvline(x=t_mod_trans, color=style['color'], linestyle=':',
                                   linewidth=1.0, alpha=0.7, label=vline_label)
                        if vline_label:
                            added_labels.add(vline_label)

        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel(key)
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=7, loc='best')
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
    model_name: str,
    label1: str = 'Original',
    label2: str = 'Modified'
):
    """
    Generate all comparison plots for a model pair.

    Only compares data within the common time range (up to the shorter simulation's end time).
    Adds vertical dashed lines for phase transitions and R2 > rCloud events.

    Parameters
    ----------
    label1 : str
        Label for first dataset (default: 'Original')
    label2 : str
        Label for second dataset (default: 'Modified')
    """
    # Get common time range
    t_min, t_max = get_common_time_range(snapshots_orig, snapshots_mod)
    print(f"\n  Common time range: [{t_min:.4e}, {t_max:.4e}] Myr")

    # Filter snapshots to common time range
    snapshots_orig_filtered = filter_snapshots_by_time(snapshots_orig, t_min, t_max)
    snapshots_mod_filtered = filter_snapshots_by_time(snapshots_mod, t_min, t_max)

    print(f"  Using {len(snapshots_orig_filtered)}/{len(snapshots_orig)} {label1} snapshots")
    print(f"  Using {len(snapshots_mod_filtered)}/{len(snapshots_mod)} {label2} snapshots")

    if len(snapshots_orig_filtered) == 0 or len(snapshots_mod_filtered) == 0:
        print("  ERROR: No overlapping time range found")
        return

    # Find phase transition times for both runs
    times_orig = find_phase_transition_times(snapshots_orig_filtered)
    times_mod = find_phase_transition_times(snapshots_mod_filtered)
    transition_times = merge_transition_times(times_orig, times_mod)

    # Print detected transitions
    print("  Phase transitions detected:")
    if times_orig['t_transition'] is not None:
        print(f"    {label1} -> Transition phase at t = {times_orig['t_transition']:.4e} Myr")
    if times_mod['t_transition'] is not None:
        print(f"    {label2} -> Transition phase at t = {times_mod['t_transition']:.4e} Myr")
    if times_orig['t_momentum'] is not None:
        print(f"    {label1} -> Momentum phase at t = {times_orig['t_momentum']:.4e} Myr")
    if times_mod['t_momentum'] is not None:
        print(f"    {label2} -> Momentum phase at t = {times_mod['t_momentum']:.4e} Myr")
    if times_orig['t_R2_gt_rCloud'] is not None:
        print(f"    {label1} -> R2 > rCloud at t = {times_orig['t_R2_gt_rCloud']:.4e} Myr")
    if times_mod['t_R2_gt_rCloud'] is not None:
        print(f"    {label2} -> R2 > rCloud at t = {times_mod['t_R2_gt_rCloud']:.4e} Myr")

    # Get all scalar keys from both datasets (use filtered data)
    keys_orig = get_scalar_keys(snapshots_orig_filtered)
    keys_mod = get_scalar_keys(snapshots_mod_filtered)
    all_keys = keys_orig | keys_mod

    print(f"  Found {len(all_keys)} scalar parameters")

    # A) Shell parameters (keys starting with shell_)
    shell_keys = sorted([k for k in all_keys if k.startswith('shell_')])
    if shell_keys:
        print(f"  Plotting {len(shell_keys)} shell parameters...")
        plot_comparison_grid(
            snapshots_orig_filtered, snapshots_mod_filtered,
            shell_keys,
            f"{model_name}: Shell Parameters Comparison",
            output_dir / "comparison_shell_parameters.pdf",
            transition_times=transition_times,
            label1=label1, label2=label2
        )

    # B) Bubble parameters (keys starting with bubble_)
    bubble_keys = sorted([k for k in all_keys if k.startswith('bubble_')])
    if bubble_keys:
        print(f"  Plotting {len(bubble_keys)} bubble parameters...")
        plot_comparison_grid(
            snapshots_orig_filtered, snapshots_mod_filtered,
            bubble_keys,
            f"{model_name}: Bubble Parameters Comparison",
            output_dir / "comparison_bubble_parameters.pdf",
            transition_times=transition_times,
            label1=label1, label2=label2
        )

    # C) TRINITY essentials (R1, R2, rShell, Pb, Eb, T0)
    essential_keys = [k for k in ESSENTIAL_PARAMS if k in all_keys]
    if essential_keys:
        print(f"  Plotting {len(essential_keys)} essential parameters...")
        plot_comparison_grid(
            snapshots_orig_filtered, snapshots_mod_filtered,
            essential_keys,
            f"{model_name}: Essential Parameters Comparison",
            output_dir / "comparison_essential_parameters.pdf",
            transition_times=transition_times,
            label1=label1, label2=label2
        )

    # D) Force parameters (keys containing F_)
    force_keys = sorted([k for k in all_keys if 'F_' in k])
    if force_keys:
        print(f"  Plotting {len(force_keys)} force parameters...")
        plot_comparison_grid(
            snapshots_orig_filtered, snapshots_mod_filtered,
            force_keys,
            f"{model_name}: Force Parameters Comparison",
            output_dir / "comparison_force_parameters.pdf",
            transition_times=transition_times,
            label1=label1, label2=label2
        )

    # E) Remaining parameters - not in above categories but evolving
    categorized_keys = set(shell_keys) | set(bubble_keys) | set(essential_keys) | set(force_keys)
    remaining_keys = all_keys - categorized_keys

    # Filter to only evolving parameters (different at t_min vs t_max)
    evolving_remaining = []
    for key in sorted(remaining_keys):
        # Check if evolving in either original or modified (using filtered data)
        evolves_orig = is_evolving_parameter(snapshots_orig_filtered, key)
        evolves_mod = is_evolving_parameter(snapshots_mod_filtered, key)
        if evolves_orig or evolves_mod:
            evolving_remaining.append(key)

    if evolving_remaining:
        print(f"  Plotting {len(evolving_remaining)} remaining evolving parameters...")
        plot_comparison_grid(
            snapshots_orig_filtered, snapshots_mod_filtered,
            evolving_remaining,
            f"{model_name}: Other Evolving Parameters Comparison",
            output_dir / "comparison_remaining_parameters.pdf",
            transition_times=transition_times,
            label1=label1, label2=label2
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

        # Print summary statistics (using common time range)
        t_min, t_max = get_common_time_range(snapshots_orig, snapshots_mod)
        snapshots_orig_filtered = filter_snapshots_by_time(snapshots_orig, t_min, t_max)
        snapshots_mod_filtered = filter_snapshots_by_time(snapshots_mod, t_min, t_max)

        print(f"\n  Summary Statistics (at t={t_max:.4e} Myr, common end time):")
        for key in ESSENTIAL_PARAMS:
            t_orig, v_orig = extract_time_series(snapshots_orig_filtered, key)
            t_mod, v_mod = extract_time_series(snapshots_mod_filtered, key)

            if np.all(np.isnan(v_orig)) and np.all(np.isnan(v_mod)):
                continue

            # Get final values (at common end time)
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
# Manual comparison function
# =============================================================================

def compare_two_folders(folder1: Path, folder2: Path, output_dir: Path = None):
    """
    Compare two specific folders/files directly.

    Parameters
    ----------
    folder1 : Path
        First folder or file path
    folder2 : Path
        Second folder or file path
    output_dir : Path, optional
        Where to save comparison plots (default: folder2's directory)
    """
    folder1 = Path(folder1)
    folder2 = Path(folder2)

    print("=" * 70)
    print("TRINITY Output Comparison: Manual Mode")
    print("=" * 70)

    # Find dictionary files
    if folder1.is_file():
        jsonl1 = folder1
    else:
        jsonl1 = find_dictionary_jsonl(folder1)

    if folder2.is_file():
        jsonl2 = folder2
    else:
        jsonl2 = find_dictionary_jsonl(folder2)

    if jsonl1 is None:
        print(f"ERROR: No dictionary.jsonl found in {folder1}")
        return False
    if jsonl2 is None:
        print(f"ERROR: No dictionary.jsonl found in {folder2}")
        return False

    print(f"\nFile 1: {jsonl1}")
    print(f"File 2: {jsonl2}")

    # Determine output directory
    if output_dir is None:
        output_dir = jsonl2.parent if jsonl2.is_file() else folder2
    output_dir = Path(output_dir)

    # Load data
    print("\nLoading snapshots...")
    snapshots1 = load_jsonl(jsonl1)
    snapshots2 = load_jsonl(jsonl2)

    print(f"  File 1: {len(snapshots1)} snapshots")
    print(f"  File 2: {len(snapshots2)} snapshots")

    if len(snapshots1) == 0 or len(snapshots2) == 0:
        print("ERROR: Empty snapshot data")
        return False

    # Generate comparison name from folder names
    name1 = jsonl1.parent.name if jsonl1.parent.name != '' else 'file1'
    name2 = jsonl2.parent.name if jsonl2.parent.name != '' else 'file2'
    model_name = f"{name1} vs {name2}"

    # Generate comparison plots with folder names as labels
    print("\nGenerating comparison plots...")
    generate_all_comparison_plots(
        snapshots1, snapshots2,
        output_dir,
        model_name,
        label1=name1,
        label2=name2
    )

    # Print summary statistics
    t_min, t_max = get_common_time_range(snapshots1, snapshots2)
    snapshots1_filtered = filter_snapshots_by_time(snapshots1, t_min, t_max)
    snapshots2_filtered = filter_snapshots_by_time(snapshots2, t_min, t_max)

    print(f"\nSummary Statistics (at t={t_max:.4e} Myr, common end time):")
    for key in ESSENTIAL_PARAMS:
        t1, v1 = extract_time_series(snapshots1_filtered, key)
        t2, v2 = extract_time_series(snapshots2_filtered, key)

        if np.all(np.isnan(v1)) and np.all(np.isnan(v2)):
            continue

        final1 = v1[~np.isnan(v1)][-1] if any(~np.isnan(v1)) else np.nan
        final2 = v2[~np.isnan(v2)][-1] if any(~np.isnan(v2)) else np.nan

        if np.isnan(final1) or np.isnan(final2):
            continue

        rel_diff = abs(final1 - final2) / max(abs(final1), 1e-300)
        status = "OK" if rel_diff < 0.01 else "DIFF"

        print(f"  {key:12s}: {name1}={final1:.4e}, {name2}={final2:.4e}, "
              f"rel_diff={rel_diff:.2e} [{status}]")

    print(f"\n{'=' * 70}")
    print(f"Comparison complete. Plots saved to: {output_dir}")
    print(f"{'=' * 70}")

    return True


# =============================================================================
# Command-line interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare TRINITY output folders: original vs modified versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode - find and compare all *_modified pairs
  %(prog)s
  %(prog)s --output-dir /path/to/outputs
  %(prog)s -o ./my_outputs

  # Manual mode - compare two specific folders
  %(prog)s folder1 folder2
  %(prog)s /path/to/run1 /path/to/run2
  %(prog)s run1/dictionary.jsonl run2/dictionary.jsonl

In auto mode, the script searches for folders ending with '_modified' and
compares them with their corresponding original folders.

In manual mode, you provide two folder paths (or file paths) to compare directly.

Comparison plots are saved as PDFs.
        """
    )

    parser.add_argument(
        'folders',
        nargs='*',
        help='Two folders/files to compare (manual mode). If not provided, runs in auto mode.'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help=f'Output directory to search (auto mode) or save plots (manual mode). Default: {DEFAULT_OUTPUT_DIR}'
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if len(args.folders) == 2:
        # Manual mode: compare two specific folders
        folder1, folder2 = args.folders
        output_dir = Path(args.output_dir) if args.output_dir else None
        success = compare_two_folders(folder1, folder2, output_dir)
    elif len(args.folders) == 0:
        # Auto mode: find and compare *_modified pairs
        output_dir = Path(args.output_dir) if args.output_dir else None
        success = compare_output_folders(output_dir)
    else:
        parser.error("Manual mode requires exactly two folder/file arguments.")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
