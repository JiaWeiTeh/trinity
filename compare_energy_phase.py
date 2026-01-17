#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare original vs modified energy phase implementations.

This script runs both the original (run_energy_phase.py) and modified
(run_energy_phase_modified.py) versions of the energy phase solver on
parameter file(s), then produces comparison plots.

Each version runs with its own isolated output directory to avoid conflicts.
When multiple parameter files are provided, each generates its own separate
PDF comparison plot.

Usage:
    python compare_energy_phase.py param/1e7_sfe030_n1e4.param
    python compare_energy_phase.py param/test.param --save-pdf
    python compare_energy_phase.py param/test.param --params R2,Eb,v2

    # Multiple parameter files (each generates its own PDF):
    python compare_energy_phase.py param/1e7_sfe001_n1e4.param param/1e7_sfe030_n1e4.param --save-pdf

    # Run in nohup/headless environment (no display):
    nohup python compare_energy_phase.py param/*.param --save-pdf --no-display &

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

# Set non-interactive backend if --no-display flag is present (for nohup/headless)
if '--no-display' in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =============================================================================
# Setup paths and imports
# =============================================================================

# Add project root to path if needed
project_root = Path(__file__).parent
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
# Configuration: Default parameters to compare
# =============================================================================

DEFAULT_COMPARE_PARAMS = [
    'R2', 'v2', 'Eb', 'R1',
    'F_ram', 'F_grav', 'F_ion_out', 'F_rad',
    'shell_mass', 'Pb', 'L_mech_wind', 'shell_massDot', 'pdotdot_total',
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


def plot_comparison(snaps_orig, snaps_mod, params_to_plot, output_paths=None,
                    save_pdf=False, save_png=False, base_name="comparison_energy_phase"):
    """
    Create comparison plots for original vs modified runs.

    Parameters
    ----------
    snaps_orig : list
        Snapshots from original run
    snaps_mod : list
        Snapshots from modified run
    params_to_plot : list
        List of parameter keys to plot
    output_paths : list of Path, optional
        List of directories to save figures to (saves to each)
    save_pdf : bool
        Save as PDF
    save_png : bool
        Save as PNG
    base_name : str
        Base filename for saved plots (without extension)
    """
    # Filter to available parameters
    available_orig = set()
    available_mod = set()
    for snap in snaps_orig:
        available_orig.update(snap.keys())
    for snap in snaps_mod:
        available_mod.update(snap.keys())

    valid_params = [p for p in params_to_plot if p in available_orig and p in available_mod]

    if not valid_params:
        print("No valid parameters to plot!")
        print(f"Available in original: {sorted(available_orig)}")
        print(f"Available in modified: {sorted(available_mod)}")
        return

    print(f"Plotting {len(valid_params)} parameters: {valid_params}")

    # Calculate grid dimensions
    ncols = min(3, len(valid_params))
    nrows = int(np.ceil(len(valid_params) / ncols))

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

    # Log-scale parameters
    log_params = {'Eb', 'F_ram', 'F_grav', 'F_ion_out', 'F_ion_in', 'F_rad',
                  'shell_mass', 'Pb', 'Qi', 'Lbol', 'L_mech_total'}

    for idx, key in enumerate(valid_params):
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

        # Log scale for certain parameters
        if key in log_params:
            finite_orig = v_orig[np.isfinite(v_orig)]
            finite_mod = v_mod[np.isfinite(v_mod)]
            if len(finite_orig) > 0 and len(finite_mod) > 0:
                if np.all(finite_orig > 0) and np.all(finite_mod > 0):
                    ax.set_yscale('log')

    # Hide unused axes
    for idx in range(len(valid_params), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Energy Phase Comparison: Original vs Modified', fontsize=14, fontweight='bold')

    # Save figures to each output path
    if output_paths and (save_pdf or save_png):
        for output_path in output_paths:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            if save_pdf:
                pdf_path = output_path / f"{base_name}.pdf"
                fig.savefig(pdf_path, bbox_inches='tight')
                print(f"Saved: {pdf_path}")

            if save_png:
                png_path = output_path / f"{base_name}.png"
                fig.savefig(png_path, bbox_inches='tight', dpi=300)
                print(f"Saved: {png_path}")

    plt.close(fig)


# =============================================================================
# Main comparison routine
# =============================================================================

def run_comparison(param_file, params_to_compare=None, save_pdf=False, save_png=False,
                   output_name=None):
    """
    Run both original and modified energy phase and compare results.

    Parameters
    ----------
    param_file : str or Path
        Path to parameter file
    params_to_compare : list, optional
        Parameters to compare. If None, uses DEFAULT_COMPARE_PARAMS.
    save_pdf : bool
        Save comparison plot as PDF
    save_png : bool
        Save comparison plot as PNG
    output_name : str, optional
        Base name for output file. If None, derived from param_file name.
    """
    logger = setup_simple_logging('INFO')

    if params_to_compare is None:
        params_to_compare = DEFAULT_COMPARE_PARAMS

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
    # Step 5: Plot comparison
    # =========================================================================
    # Save to both output directories
    output_paths = [
        Path(params_orig['path2output'].value),
        Path(params_mod['path2output'].value),
    ]

    # Derive output name from param file if not provided
    if output_name is None:
        output_name = f"comparison_{param_file.stem}"

    plot_comparison(
        snaps_orig, snaps_mod, params_to_compare,
        output_paths=output_paths,
        save_pdf=save_pdf,
        save_png=save_png,
        base_name=output_name,
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


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs modified energy phase implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s param/1e7_sfe030_n1e4.param
  %(prog)s param/test.param --save-pdf
  %(prog)s param/test.param --params R2,Eb,v2,F_ram

  # Multiple parameter files (each generates its own PDF):
  %(prog)s param/1e7_sfe001_n1e4.param param/1e7_sfe030_n1e4.param --save-pdf

  # Run in nohup/headless environment (no display):
  nohup %(prog)s param/*.param --save-pdf --no-display &
        """
    )

    parser.add_argument(
        'param_files',
        type=str,
        nargs='+',
        help='Path(s) to the parameter file(s) (.param)'
    )

    parser.add_argument(
        '--params', '-p',
        type=str,
        default=None,
        help='Comma-separated list of parameters to compare (default: R2,v2,Eb,F_ram,...)'
    )

    parser.add_argument(
        '--save-pdf',
        action='store_true',
        help='Save comparison plot as PDF'
    )

    parser.add_argument(
        '--save-png',
        action='store_true',
        help='Save comparison plot as PNG'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable interactive plot display (for nohup/headless environments)'
    )

    args = parser.parse_args()

    # Parse custom parameters
    params_to_compare = None
    if args.params:
        params_to_compare = [p.strip() for p in args.params.split(',') if p.strip()]

    # Run comparison for each parameter file
    for i, param_file in enumerate(args.param_files):
        if len(args.param_files) > 1:
            print(f"\n{'='*60}")
            print(f"Processing file {i+1}/{len(args.param_files)}: {param_file}")
            print(f"{'='*60}\n")

        run_comparison(
            param_file=param_file,
            params_to_compare=params_to_compare,
            save_pdf=args.save_pdf,
            save_png=args.save_png,
        )


if __name__ == "__main__":
    main()
