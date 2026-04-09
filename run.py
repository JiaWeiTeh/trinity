#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:27:38 2022

@author: Jia Wei Teh

This script contains the main file to run WARPFIELD.

Unified entry point for single runs and parameter sweeps.
Auto-detects sweep mode when parameters contain list syntax or tuple definitions.

Usage (single run):
    python run.py param/example.param

Usage (sweep - auto-detected):
    python run.py param/sweep.param
    python run.py param/sweep.param --workers 4
    python run.py param/sweep.param --dry-run
    python run.py param/sweep.param --yes
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event


# =============================================================================
# Configure EARLY logging so read_param messages are captured
# =============================================================================
# This is a minimal setup - will be reconfigured after params are loaded
logging.basicConfig(
    level=logging.DEBUG,  # Start with DEBUG to capture everything
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Suppress noisy third-party libraries during early logging
for lib in ['matplotlib', 'PIL', 'urllib3', 'asyncio', 'parso', 'fontTools', 'numba', 'h5py']:
    logging.getLogger(lib).setLevel(logging.INFO)

early_logger = logging.getLogger(__name__)
early_logger.debug("Early logging configured (pre-params)")

# Global flag for graceful shutdown on Ctrl+C (sweep mode)
_shutdown_requested = Event()

TRINITY_ROOT = Path(__file__).parent.resolve()


# =============================================================================
# Auto-detection
# =============================================================================

def is_sweep_param_file(path2file):
    """
    Quick scan to detect if a parameter file contains sweep/tuple syntax.

    Returns True if the file contains:
    - List syntax with multiple elements: key [val1, val2, ...]
    - Tuple syntax: tuple(param1, param2, ...) [val1, val2] ...
    """
    with open(path2file, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip comments
            if '#' in line:
                line = line[:line.find('#')]
            line = line.strip()
            if not line:
                continue
            # Check for tuple syntax
            if line.lower().startswith('tuple('):
                return True
            # Check for list syntax in value
            parts = line.split(None, 1)
            if len(parts) == 2:
                val = parts[1].strip()
                if val.startswith('[') and val.endswith(']'):
                    # Multi-element list = sweep
                    inner = val[1:-1].strip()
                    if ',' in inner:
                        return True
    return False


# =============================================================================
# Single-run mode
# =============================================================================

def run_single(args):
    """Run a single TRINITY simulation."""
    from src._input import read_param

    from src._output import header
    header.display()

    # Get class and write summary file
    # Note: read_param logging is now captured with early config above
    params = read_param.read_param(args.path2param, write_summary=True)

    header.show_param(params)


    from src import main

    # main_dict = create_dictionary.create()


    # =================================================================
    # Reconfigure logging with params settings
    # =================================================================
    # Now reconfigure with user's preferred log level from params
    from src._functions.logging_setup import setup_logging

    # Get log_level from params (default to INFO if not set)
    log_level = 'INFO'
    if 'log_level' in params:
        log_level = params['log_level'].value if hasattr(params['log_level'], 'value') else params['log_level']

    # Get log_console from params (default to True if not set)
    log_console = True
    if 'log_console' in params:
        log_console = params['log_console'].value if hasattr(params['log_console'], 'value') else params['log_console']

    # Get log_file from params (default to True if not set)
    log_file = True
    if 'log_file' in params:
        log_file = params['log_file'].value if hasattr(params['log_file'], 'value') else params['log_file']

    logger = setup_logging(
        log_level=log_level,
        console_output=log_console,
        file_output=log_file,
        log_file_path=params['path2output'].value,
        log_file_name='trinity.log',
        use_colors=True,
    )
    logger.info(f"Parameter file loaded: {args.path2param}")
    logger.info(f"Log level set to: {log_level}")
    logger.info(f"Console output: {log_console}, File output: {log_file}")


    # =================================================================
    # Validate GMC parameters before running
    # =================================================================
    from src.cloud_properties.validate_gmc import validate_gmc_from_params

    gmc_check = validate_gmc_from_params(params)
    for w in gmc_check.warnings:
        logger.warning(w)
    if not gmc_check.valid:
        logger.error("GMC parameter validation failed:")
        for e in gmc_check.errors:
            logger.error(f"  {e}")
        if gmc_check.suggestions:
            logger.info("Suggested valid alternatives:")
            for i, s in enumerate(gmc_check.suggestions, 1):
                parts = ", ".join(f"{k}={v}" for k, v in s.items())
                logger.info(f"  {i}. {parts}")
        sys.exit("Simulation stopped: implausible GMC parameters. See errors above.")


    main.start_expansion(params)


# =============================================================================
# Sweep mode helpers
# =============================================================================

def get_optimal_workers():
    """
    Determine optimal number of worker processes.

    Strategy:
    - Use (CPU_count - 1) to leave one core for system/monitoring
    - Cap at 8 to avoid overwhelming I/O
    - Minimum of 1
    """
    cpus = multiprocessing.cpu_count()
    optimal = max(1, min(cpus - 1, 8))
    return optimal


def _validate_sweep_combination(params_dict):
    """
    Validate a single sweep combination's GMC parameters.

    Parameters
    ----------
    params_dict : dict
        Plain parameter dictionary (no .value wrappers).

    Returns
    -------
    result : GMCValidationResult or None
        Validation result, or None if validation cannot be performed
        (e.g. missing keys).
    """
    from src.cloud_properties.validate_gmc import validate_gmc_params

    dens_profile = params_dict.get('dens_profile')
    if dens_profile not in ('densPL', 'densBE'):
        return None

    mCloud = params_dict.get('mCloud')
    nCore = params_dict.get('nCore')
    if mCloud is None or nCore is None:
        return None

    mu = float(params_dict.get('mu_convert', 1.4))
    nISM = float(params_dict.get('nISM', 1.0))

    kwargs = dict(
        mCloud=float(mCloud), nCore=float(nCore),
        mu=mu, nISM=nISM, dens_profile=dens_profile,
    )

    if dens_profile == 'densPL':
        alpha = params_dict.get('densPL_alpha')
        rCore = params_dict.get('rCore')
        if alpha is None:
            return None
        kwargs['alpha'] = float(alpha)
        if rCore is not None:
            kwargs['rCore'] = float(rCore)
    elif dens_profile == 'densBE':
        Omega = params_dict.get('densBE_Omega')
        if Omega is None:
            return None
        kwargs['Omega'] = float(Omega)
        kwargs['gamma'] = float(params_dict.get('gamma_adia', 5.0 / 3.0))

    try:
        return validate_gmc_params(**kwargs)
    except Exception:
        return None


# =============================================================================
# Sweep mode
# =============================================================================

def run_sweep(args):
    """Run a TRINITY parameter sweep."""
    from src._input.sweep_parser import (
        read_sweep_config,
        generate_combinations_from_config,
        count_combinations_from_config,
    )
    from src._input.sweep_runner import (
        run_single_simulation,
        ProgressBar,
        SweepReport,
        SimulationResult,
    )

    # Reconfigure logging for sweep mode
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    # Validate input file
    param_path = Path(args.path2param).resolve()
    if not param_path.exists():
        print(f"Error: Parameter file not found: {param_path}")
        sys.exit(1)

    # =================================================================
    # Parse sweep parameters
    # =================================================================

    print(f"\nReading sweep parameters from: {param_path}")
    try:
        config = read_sweep_config(str(param_path))
    except Exception as e:
        print(f"Error parsing parameter file: {e}")
        sys.exit(1)

    # Count combinations
    n_combinations = count_combinations_from_config(config)

    # =================================================================
    # Display sweep summary
    # =================================================================

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 60)

    print(f"\nBase parameters (constant across all runs):")
    for k, v in sorted(config.base_params.items()):
        print(f"  {k}: {v}")

    if config.is_hybrid_mode:
        print(f"\nHybrid mode:")
        print(f"  Tuple parameters: {config.tuple_params}")
        print(f"    {len(config.tuple_values)} explicit combinations:")
        for i, values in enumerate(config.tuple_values[:5]):  # Show first 5
            combo_str = ", ".join(f"{p}={v}" for p, v in zip(config.tuple_params, values))
            print(f"      [{i+1}] {combo_str}")
        if len(config.tuple_values) > 5:
            print(f"      ... and {len(config.tuple_values) - 5} more")
        print(f"  Sweep parameters (Cartesian with tuples):")
        for k, v in sorted(config.sweep_params.items()):
            print(f"    {k}: {v} ({len(v)} values)")
    elif config.is_tuple_mode:
        print(f"\nTuple mode: {config.tuple_params}")
        print(f"  {len(config.tuple_values)} explicit combinations:")
        for i, values in enumerate(config.tuple_values[:5]):  # Show first 5
            combo_str = ", ".join(f"{p}={v}" for p, v in zip(config.tuple_params, values))
            print(f"    [{i+1}] {combo_str}")
        if len(config.tuple_values) > 5:
            print(f"    ... and {len(config.tuple_values) - 5} more")
    else:
        print(f"\nSweep parameters (varying):")
        for k, v in sorted(config.sweep_params.items()):
            print(f"  {k}: {v} ({len(v)} values)")

    print(f"\nTotal combinations: {n_combinations}")

    # =================================================================
    # Determine workers
    # =================================================================

    cpus = multiprocessing.cpu_count()
    suggested = get_optimal_workers()

    if args.workers is None:
        workers = suggested
        print(f"\nAuto-detected {cpus} CPUs")
        print(f"Suggested parallel workers: {suggested}")
    else:
        workers = args.workers
        print(f"\nUsing {workers} workers (user-specified)")

    print(f"Will use {workers} parallel worker(s)")

    # Estimate time (rough estimate: 5 minutes per simulation)
    estimated_time_per_sim = 1000  # seconds
    estimated_total_seconds = (n_combinations / workers) * estimated_time_per_sim
    estimated_total_minutes = estimated_total_seconds / 60

    print(f"\nEstimated total time: ~{estimated_total_minutes:.0f} minutes")
    print(f"  (assuming ~{estimated_time_per_sim} s/sim with {workers} workers)")

    # =================================================================
    # Determine output directory
    # =================================================================

    # Use path2output from params if specified, else use 'outputs/'
    base_output_dir = config.base_params.get('path2output', 'outputs')
    if base_output_dir == 'def_dir':
        base_output_dir = os.path.join(os.getcwd(), 'outputs')

    # Make it absolute
    base_output_dir = str(Path(base_output_dir).resolve())

    print(f"\nOutput directory: {base_output_dir}")

    # =================================================================
    # Dry run mode
    # =================================================================

    if args.dry_run:
        print("\n" + "-" * 60)
        print("DRY RUN - Combinations to be generated:")
        print("-" * 60)

        combinations = list(generate_combinations_from_config(config))
        # Determine which keys to show
        if config.is_hybrid_mode:
            # Show both tuple params and sweep params
            varying_keys = config.tuple_params + sorted(config.sweep_params.keys())
        elif config.is_tuple_mode:
            varying_keys = config.tuple_params
        else:
            varying_keys = sorted(config.sweep_params.keys())

        n_invalid = 0
        for params, name in combinations:
            gmc_result = _validate_sweep_combination(params)
            if gmc_result is not None and not gmc_result.valid:
                status = " [INVALID GMC]"
                n_invalid += 1
            else:
                status = ""

            print(f"\n{name}:{status}")
            for k in varying_keys:
                print(f"  {k}: {params.get(k)}")
            if gmc_result is not None and not gmc_result.valid:
                for e in gmc_result.errors:
                    print(f"    >> {e}")

        print(f"\n(Would run {n_combinations} simulations)")
        if n_invalid > 0:
            print(f"WARNING: {n_invalid}/{n_combinations} combinations have "
                  f"implausible GMC parameters and will fail.")
        print("(No simulations were run - dry run mode)")
        sys.exit(0)

    # =================================================================
    # Pre-flight GMC validation
    # =================================================================

    pre_combinations = list(generate_combinations_from_config(config))
    invalid_combos = []
    for params_check, name_check in pre_combinations:
        gmc_result = _validate_sweep_combination(params_check)
        if gmc_result is not None and not gmc_result.valid:
            invalid_combos.append((name_check, gmc_result))

    if invalid_combos:
        print(f"\nWARNING: {len(invalid_combos)}/{n_combinations} combinations "
              f"have implausible GMC parameters:")
        for name_inv, res_inv in invalid_combos:
            print(f"  {name_inv}:")
            for e in res_inv.errors:
                print(f"    - {e}")
        print()

    # =================================================================
    # Confirmation
    # =================================================================

    if not args.yes:
        print("\n" + "-" * 60)
        prompt_msg = f"Run {n_combinations} simulations with {workers} workers?"
        if invalid_combos:
            prompt_msg += (f" ({len(invalid_combos)} will fail due to invalid "
                          f"GMC parameters)")
        response = input(f"{prompt_msg} [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            sys.exit(0)

    # =================================================================
    # Setup output directory
    # =================================================================

    output_base = Path(base_output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Run sweep
    # =================================================================

    print("\n" + "=" * 60)
    print("STARTING PARAMETER SWEEP")
    print("=" * 60)
    print("(Press Ctrl+C to cancel all remaining simulations)\n")

    start_time = datetime.now()

    # Generate all combinations
    combinations = list(generate_combinations_from_config(config))
    n_combinations = len(combinations)

    progress = ProgressBar(n_combinations, desc="Sweep")

    successful = []
    failed = []
    cancelled = []

    # Setup Ctrl+C handler
    def signal_handler(signum, frame):
        _shutdown_requested.set()
        print("\n\n*** Ctrl+C received - cancelling remaining simulations... ***\n")

    original_handler = signal.signal(signal.SIGINT, signal_handler)

    # Execute with process pool
    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=workers)

        # Submit all jobs
        futures = {}
        for params, name in combinations:
            if _shutdown_requested.is_set():
                # Don't submit new jobs if shutdown requested
                cancelled.append(SimulationResult(
                    name=name,
                    params=params,
                    success=False,
                    return_code=-2,
                    duration=0,
                    error_message="Cancelled by user (Ctrl+C)"
                ))
                continue

            future = executor.submit(
                run_single_simulation,
                params,
                name,
                TRINITY_ROOT,
                base_output_dir
            )
            futures[future] = (params, name)

        # Show what's running
        running_names = [name for _, name in combinations[:workers]]
        progress.set_running(running_names)

        # Process results as they complete
        for future in as_completed(futures):
            if _shutdown_requested.is_set():
                # Cancel remaining futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

            params, name = futures[future]
            try:
                result = future.result(timeout=0.1)
            except Exception as e:
                result = SimulationResult(
                    name=name,
                    params=params,
                    success=False,
                    return_code=-1,
                    duration=0,
                    error_message=str(e)
                )

            progress.update(result.name, result.success)

            if result.success:
                successful.append(result)
            else:
                failed.append(result)
                # Log failed simulation immediately
                logger.warning(f"FAILED: {result.name} - {result.error_message[:100] if result.error_message else 'Unknown error'}...")

        # If shutdown was requested, mark remaining as cancelled
        if _shutdown_requested.is_set():
            for future, (params, name) in futures.items():
                if not future.done():
                    cancelled.append(SimulationResult(
                        name=name,
                        params=params,
                        success=False,
                        return_code=-2,
                        duration=0,
                        error_message="Cancelled by user (Ctrl+C)"
                    ))

    except KeyboardInterrupt:
        print("\n\n*** Sweep cancelled by user ***\n")
        _shutdown_requested.set()
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

        # Shutdown executor
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)

    progress.close()
    end_time = datetime.now()

    # Add cancelled to failed for reporting
    failed.extend(cancelled)

    # =================================================================
    # Generate report
    # =================================================================

    report = SweepReport(
        sweep_file=str(param_path),
        start_time=start_time,
        end_time=end_time,
        total_combinations=n_combinations,
        successful=successful,
        failed=failed
    )

    # Write reports
    txt_report = report.write_report(output_base)
    json_report = report.write_json(output_base)

    # =================================================================
    # Print summary
    # =================================================================

    print(progress.summary())
    print(f"\nReports written to:")
    print(f"  {txt_report}")
    print(f"  {json_report}")

    if _shutdown_requested.is_set():
        n_cancelled = len(cancelled)
        n_actual_failed = len(failed) - n_cancelled
        print(f"\n*** SWEEP CANCELLED ***")
        print(f"Completed: {len(successful)} | Failed: {n_actual_failed} | Cancelled: {n_cancelled}")
        if failed:
            print(f"\nSee {txt_report} for details.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    elif failed:
        print(f"\nWARNING: {len(failed)} simulation(s) failed!")
        print("Failed simulations:")
        for result in failed:
            print(f"  - {result.name}")
        print(f"\nSee {txt_report} for details.")
        sys.exit(1)
    else:
        print("\nAll simulations completed successfully!")
        sys.exit(0)


# =============================================================================
# Main entry point
# =============================================================================

# parser
parser = argparse.ArgumentParser(
    description="Run TRINITY simulation or parameter sweep (auto-detected)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
# Positional: path to param file
parser.add_argument(
    'path2param',
    help="Path to .param file (single run or sweep with list syntax)"
)
# Sweep-specific options (ignored for single runs)
parser.add_argument(
    '--workers', '-w',
    type=int,
    default=None,
    help="Number of parallel workers for sweep mode (default: auto-detect CPUs)"
)
parser.add_argument(
    '--dry-run', '-n',
    action='store_true',
    help="Show sweep combinations without running simulations"
)
parser.add_argument(
    '--yes', '-y',
    action='store_true',
    help="Skip sweep confirmation prompt"
)
parser.add_argument(
    '--verbose', '-v',
    action='store_true',
    help="Enable verbose output"
)
# grab argument
args = parser.parse_args()

# Auto-detect mode from parameter file content
if is_sweep_param_file(args.path2param):
    early_logger.info("Sweep syntax detected in parameter file — entering sweep mode")
    run_sweep(args)
else:
    run_single(args)
