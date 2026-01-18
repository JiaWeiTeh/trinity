#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare original vs modified energy phase implementations.

This script runs both the original (run_energy_phase.py) and modified
(run_energy_phase_modified.py) versions of the energy phase solver on
parameter file(s), then produces comparison plots grouped by category.

Output PDFs (saved to both output directories):
    - comparison_shell_*.pdf: Shell parameters (shell_* prefix)
    - comparison_bubble_*.pdf: Bubble parameters (bubble_* prefix)
    - comparison_force_*.pdf: Force parameters (F_* prefix)
    - comparison_sb99_*.pdf: Starburst99 parameters
    - comparison_main_*.pdf: Main bubble parameters (v2, R2, R1, etc.)

Usage (from project root):
    python comparison/compare_energy_phase.py param/1e7_sfe030_n1e4.param

    # Multiple parameter files (each generates its own set of PDFs):
    python comparison/compare_energy_phase.py param/1e7_sfe001_n1e4.param param/1e7_sfe030_n1e4.param

    # Run in nohup/headless environment:
    nohup python comparison/compare_energy_phase.py param/*.param &

Author: TRINITY Team
"""

import os
import sys
import copy
import json
import logging
import argparse
import datetime
import numpy as np
import scipy
import matplotlib
from pathlib import Path

# Always use non-interactive backend (no display needed)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =============================================================================
# Setup paths and imports
# =============================================================================

# Add project root to path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src._input import read_param
from src._input.dictionary import DescribedItem, DescribedDict
from src.sb99 import read_SB99
from src.phase0_init import get_InitCloudProp, get_InitPhaseParam
import src._functions.unit_conversions as cvt

# Import both versions of the energy phase
from src.phase1_energy import run_energy_phase
from src.phase1_energy import run_energy_phase_modified

# =============================================================================
# Parameter categories for plotting
# =============================================================================

# Category 1: Shell parameters (prefix shell_)
# Will be auto-detected from snapshots

# Category 2: Bubble parameters (prefix bubble_)
# Will be auto-detected from snapshots

# Category 3: Force parameters (prefix F_)
# Will be auto-detected from snapshots

# Category 4: Starburst99 parameters
SB99_PARAMS = [
    'Qi', 'Li', 'Lmech_W', 'Lmech_total',
    'pdot_W', 'pdot_total', 'pdotdot_total', 'v_mech_total'
]

# Category 5: Main bubble/shell parameters
MAIN_PARAMS = [
    'v2', 'R2', 'R1', 'rShell', 'T0', 'Eb', 'Pb', 'c_sound'
]

# =============================================================================
# Helper functions
# =============================================================================

def setup_simple_logging(level='INFO'):
    """Setup basic logging for the comparison script."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt='%H:%M:%S',
    )
    # Suppress noisy libraries
    for lib in ['matplotlib', 'PIL', 'urllib3', 'fontTools', 'numba']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def deep_copy_params(params):
    """
    Create a deep copy of the params DescribedDict.

    This ensures complete isolation between runs.
    """
    # Use copy.deepcopy for the entire structure
    params_copy = copy.deepcopy(params)

    # Reset snapshot-related state for fresh run
    params_copy.save_count = 0
    params_copy.flush_count = 0
    params_copy.previous_snapshot = {}

    return params_copy


def create_isolated_params(base_params, suffix):
    """
    Create an isolated copy of params with modified output path.

    Parameters
    ----------
    base_params : DescribedDict
        The base parameters dictionary
    suffix : str
        Suffix to append to output path (e.g., 'original', 'modified')

    Returns
    -------
    DescribedDict
        Isolated copy with modified output path
    """
    params_copy = deep_copy_params(base_params)

    # Modify output path
    original_path = base_params['path2output'].value
    new_path = f"{original_path}_{suffix}"
    params_copy['path2output'].value = new_path

    # Create the output directory
    Path(new_path).mkdir(parents=True, exist_ok=True)

    return params_copy


def initialize_simulation(params, logger):
    """
    Run the initialization steps (cloud properties, SB99, cooling).

    This mimics what main.start_expansion() does before calling run_energy().
    """
    logger.info("Initializing cloud properties...")
    get_InitCloudProp.get_InitCloudProp(params)

    logger.info("Loading Starburst99 data...")
    f_mass = params['mCluster'] / params['SB99_mass']
    SB99_data = read_SB99.read_SB99(f_mass, params)
    SB99f = read_SB99.get_interpolation(SB99_data)
    params['SB99_data'].value = SB99_data
    params['SB99f'].value = SB99f

    # Extract SB99 arrays for the modified version
    # SB99_data = [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
    #              pdot_W, pdot_SN, pdot_total]
    t_arr = SB99_data[0]           # Time array
    Lmech_arr = SB99_data[7]       # Lmech_total array
    pdot_arr = SB99_data[10]       # pdot_total array

    # Compute v_mech = 2 * Lmech / pdot (wind terminal velocity formula)
    v_mech_arr = np.where(pdot_arr > 0, 2.0 * Lmech_arr / pdot_arr, 0.0)

    # Store arrays in params for modified version
    params['SB99_t'] = DescribedItem(t_arr, info="SB99 time array")
    params['SB99_Lmech'] = DescribedItem(Lmech_arr, info="SB99 Lmech_total array")
    params['SB99_vmech'] = DescribedItem(v_mech_arr, info="SB99 v_mech_total array")

    logger.info("Loading CIE cooling curve...")
    cooling_path = params['path_cooling_CIE'].value
    logT, logLambda = np.loadtxt(cooling_path, unpack=True)
    cooling_CIE_interpolation = scipy.interpolate.interp1d(logT, logLambda, kind='linear')
    params['cStruc_cooling_CIE_logT'].value = logT
    params['cStruc_cooling_CIE_logLambda'].value = logLambda
    params['cStruc_cooling_CIE_interpolation'].value = cooling_CIE_interpolation

    logger.info("Computing initial phase parameters...")
    t0, r0, v0, E0, T0 = get_InitPhaseParam.get_y0(params)
    params['t_now'].value = t0
    params['R2'].value = r0
    params['v2'].value = v0
    params['Eb'].value = E0
    params['T0'].value = T0
    params['current_phase'].value = 'energy'

    logger.info(f"  Initial: t={t0:.6e} Myr, R2={r0:.6e} pc, v2={v0:.6e} pc/Myr")


def load_jsonl(filepath):
    """Load snapshots from a JSONL file."""
    snapshots = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    return snapshots


def is_scalar_param(snapshots, key):
    """
    Check if a parameter is scalar (int/float) and not an array.

    Parameters
    ----------
    snapshots : list
        List of snapshot dictionaries
    key : str
        Parameter key to check

    Returns
    -------
    bool
        True if scalar, False if array or not present
    """
    for snap in snapshots:
        val = snap.get(key)
        if val is None:
            continue
        # Check if it's a list/array
        if isinstance(val, (list, np.ndarray)):
            return False
        # Check if it's a scalar number
        if isinstance(val, (int, float, np.integer, np.floating)):
            return True
    return False


def extract_time_series(snapshots, key):
    """Extract time and value arrays from snapshots."""
    t = []
    values = []
    for snap in snapshots:
        t.append(snap.get('t_now', np.nan))
        val = snap.get(key, np.nan)
        values.append(val if val is not None else np.nan)

    t = np.array(t, dtype=float)
    values = np.array(values, dtype=float)

    # Sort by time
    order = np.argsort(t)
    return t[order], values[order]


def should_use_log_scale(values_orig, values_mod):
    """
    Determine if log scale should be used based on value ranges.

    Criteria:
    - Values span more than 2 dex (orders of magnitude)
    - OR average value is > 1e4
    - AND all values are positive

    Parameters
    ----------
    values_orig, values_mod : ndarray
        Value arrays from original and modified runs

    Returns
    -------
    bool
        True if log scale should be used
    """
    # Combine and filter finite positive values
    all_vals = np.concatenate([values_orig, values_mod])
    finite_vals = all_vals[np.isfinite(all_vals)]

    if len(finite_vals) == 0:
        return False

    # Must all be positive for log scale
    if np.any(finite_vals <= 0):
        return False

    # Check if average > 1e4
    avg_val = np.mean(finite_vals)
    if avg_val > 1e4:
        return True

    # Check if range spans > 2 dex
    min_val = np.min(finite_vals)
    max_val = np.max(finite_vals)
    if min_val > 0 and max_val / min_val > 100:  # 2 dex = factor of 100
        return True

    return False


def get_params_by_category(snapshots_orig, snapshots_mod):
    """
    Categorize parameters from snapshots.

    Returns dict mapping category name to list of valid scalar parameters.
    """
    # Get all available keys
    available_orig = set()
    available_mod = set()
    for snap in snapshots_orig:
        available_orig.update(snap.keys())
    for snap in snapshots_mod:
        available_mod.update(snap.keys())

    # Only keep parameters present in both
    common_keys = available_orig & available_mod

    # Filter to scalar parameters only
    scalar_keys = set()
    for key in common_keys:
        if is_scalar_param(snapshots_orig, key) and is_scalar_param(snapshots_mod, key):
            scalar_keys.add(key)

    # Categorize
    categories = {}

    # Shell parameters (shell_* prefix)
    shell_params = sorted([k for k in scalar_keys if k.startswith('shell_')])
    if shell_params:
        categories['shell'] = shell_params

    # Bubble parameters (bubble_* prefix)
    bubble_params = sorted([k for k in scalar_keys if k.startswith('bubble_')])
    if bubble_params:
        categories['bubble'] = bubble_params

    # Force parameters (F_* prefix)
    force_params = sorted([k for k in scalar_keys if k.startswith('F_')])
    if force_params:
        categories['force'] = force_params

    # SB99 parameters
    sb99_params = [p for p in SB99_PARAMS if p in scalar_keys]
    if sb99_params:
        categories['sb99'] = sb99_params

    # Main parameters
    main_params = [p for p in MAIN_PARAMS if p in scalar_keys]
    if main_params:
        categories['main'] = main_params

    return categories


def plot_category(snaps_orig, snaps_mod, params_to_plot, category_name,
                  output_paths, base_name):
    """
    Create comparison plot for a single category.

    Parameters
    ----------
    snaps_orig : list
        Snapshots from original run
    snaps_mod : list
        Snapshots from modified run
    params_to_plot : list
        List of parameter keys to plot
    category_name : str
        Name of the category (for title)
    output_paths : list of Path
        Directories to save figures to
    base_name : str
        Base filename for saved plots
    """
    if not params_to_plot:
        return

    print(f"  Plotting {category_name}: {len(params_to_plot)} parameters")

    # Calculate grid dimensions
    ncols = min(3, len(params_to_plot))
    nrows = int(np.ceil(len(params_to_plot) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        dpi=150,
        constrained_layout=True
    )

    # Flatten axes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    axes_flat = axes.flatten()

    for idx, key in enumerate(params_to_plot):
        ax = axes_flat[idx]

        t_orig, v_orig = extract_time_series(snaps_orig, key)
        t_mod, v_mod = extract_time_series(snaps_mod, key)

        # Plot both
        ax.plot(t_orig, v_orig, 'b-', lw=1.5, label='Original', alpha=0.8)
        ax.plot(t_mod, v_mod, 'r--', lw=1.5, label='Modified', alpha=0.8)

        ax.set_xlabel('t [Myr]')
        ax.set_ylabel(key)
        ax.set_title(key, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Auto-detect log scale
        if should_use_log_scale(v_orig, v_mod):
            ax.set_yscale('log')

    # Hide unused axes
    for idx in range(len(params_to_plot), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Category title mapping
    title_map = {
        'shell': 'Shell Parameters',
        'bubble': 'Bubble Parameters',
        'force': 'Force Parameters',
        'sb99': 'Starburst99 Parameters',
        'main': 'Main Parameters',
    }
    fig.suptitle(f'Energy Phase Comparison: {title_map.get(category_name, category_name)}',
                 fontsize=14, fontweight='bold')

    # Save figures to each output path
    for output_path in output_paths:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        pdf_path = output_path / f"{base_name}_{category_name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")

    plt.close(fig)


# =============================================================================
# Main comparison routine
# =============================================================================

def run_comparison(param_file):
    """
    Run both original and modified energy phase and compare results.

    Parameters
    ----------
    param_file : str or Path
        Path to parameter file
    """
    logger = setup_simple_logging('INFO')

    param_file = Path(param_file)
    if not param_file.exists():
        logger.error(f"Parameter file not found: {param_file}")
        return

    # =========================================================================
    # Step 1: Read parameter file
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ENERGY PHASE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Parameter file: {param_file}")

    base_params = read_param.read_param(str(param_file), write_summary=False)
    base_output_path = base_params['path2output'].value

    # =========================================================================
    # Step 2: Run MODIFIED version (first, for easier debugging)
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Running MODIFIED energy phase...")
    logger.info("-" * 60)

    params_mod = create_isolated_params(base_params, "modified")
    initialize_simulation(params_mod, logger)

    start_mod = datetime.datetime.now()
    run_energy_phase_modified.run_energy(params_mod)
    params_mod.flush()  # Ensure all snapshots written
    elapsed_mod = datetime.datetime.now() - start_mod

    logger.info(f"Modified completed in {elapsed_mod}")
    logger.info(f"  Output: {params_mod['path2output'].value}")

    # =========================================================================
    # Step 3: Run ORIGINAL version
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Running ORIGINAL energy phase...")
    logger.info("-" * 60)

    params_orig = create_isolated_params(base_params, "original")
    initialize_simulation(params_orig, logger)

    start_orig = datetime.datetime.now()
    run_energy_phase.run_energy(params_orig)
    params_orig.flush()  # Ensure all snapshots written
    elapsed_orig = datetime.datetime.now() - start_orig

    logger.info(f"Original completed in {elapsed_orig}")
    logger.info(f"  Output: {params_orig['path2output'].value}")


    # =========================================================================
    # Step 4: Load results and compare
    # =========================================================================
    logger.info("-" * 60)
    logger.info("Loading and comparing results...")
    logger.info("-" * 60)

    jsonl_orig = Path(params_orig['path2output'].value) / "dictionary.jsonl"
    jsonl_mod = Path(params_mod['path2output'].value) / "dictionary.jsonl"

    if not jsonl_orig.exists():
        logger.error(f"Original output not found: {jsonl_orig}")
        return
    if not jsonl_mod.exists():
        logger.error(f"Modified output not found: {jsonl_mod}")
        return

    snaps_orig = load_jsonl(jsonl_orig)
    snaps_mod = load_jsonl(jsonl_mod)

    logger.info(f"Original: {len(snaps_orig)} snapshots")
    logger.info(f"Modified: {len(snaps_mod)} snapshots")

    # =========================================================================
    # Step 5: Categorize and plot
    # =========================================================================
    output_paths = [
        Path(params_orig['path2output'].value),
        Path(params_mod['path2output'].value),
    ]

    # Base name from param file
    base_name = f"comparison_{param_file.stem}"

    # Get parameters by category
    categories = get_params_by_category(snaps_orig, snaps_mod)

    logger.info("Creating comparison plots by category...")
    for cat_name, cat_params in categories.items():
        plot_category(
            snaps_orig, snaps_mod, cat_params, cat_name,
            output_paths, base_name
        )

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Original runtime: {elapsed_orig}")
    logger.info(f"Modified runtime: {elapsed_mod}")
    logger.info(f"Speedup: {elapsed_orig.total_seconds() / max(elapsed_mod.total_seconds(), 0.001):.2f}x")
    logger.info(f"PDFs saved to: {output_paths[0]} and {output_paths[1]}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs modified energy phase implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output PDFs (saved to both *_original and *_modified directories):
  - comparison_<param>_shell.pdf   : Shell parameters (shell_* prefix)
  - comparison_<param>_bubble.pdf  : Bubble parameters (bubble_* prefix)
  - comparison_<param>_force.pdf   : Force parameters (F_* prefix)
  - comparison_<param>_sb99.pdf    : Starburst99 parameters
  - comparison_<param>_main.pdf    : Main parameters (v2, R2, R1, etc.)

Examples:
  %(prog)s param/1e7_sfe030_n1e4.param

  # Multiple parameter files (each generates its own set of PDFs):
  %(prog)s param/1e7_sfe001_n1e4.param param/1e7_sfe030_n1e4.param

  # Run in nohup/background:
  nohup %(prog)s param/*.param &
        """
    )

    parser.add_argument(
        'param_files',
        type=str,
        nargs='+',
        help='Path(s) to the parameter file(s) (.param)'
    )

    args = parser.parse_args()

    # Run comparison for each parameter file
    for i, param_file in enumerate(args.param_files):
        if len(args.param_files) > 1:
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(args.param_files)}: {param_file}")
            print(f"{'='*60}\n")

        run_comparison(param_file=param_file)


if __name__ == "__main__":
    main()
