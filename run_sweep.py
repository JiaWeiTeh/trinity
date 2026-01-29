#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRINITY Parameter Sweep Runner
==============================

Execute multiple TRINITY simulations in parallel from a single parameter file
with list-valued parameters.

Usage:
    python run_sweep.py sweep.param [options]

Options:
    --workers N     Number of parallel workers (default: auto-detect CPUs)
    --dry-run       Show combinations without running simulations
    --yes           Skip confirmation prompt

Example sweep.param:
    mCloud    [1e5, 1e7, 1e8]
    sfe       [0.01, 0.10]
    nCore     [1e3, 1e4]
    dens_profile    densPL
    densPL_alpha    0

This generates 3 x 2 x 2 = 12 simulations.

Date: 2026-01-14
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

# Global flag for graceful shutdown on Ctrl+C
_shutdown_requested = Event()

# Add trinity root to path
TRINITY_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(TRINITY_ROOT))

from src._input.sweep_parser import (
    read_sweep_param,
    read_sweep_config,
    generate_combinations,
    generate_combinations_from_config,
    count_combinations,
    count_combinations_from_config,
    SweepConfig,
)
from src._input.sweep_runner import (
    run_single_simulation,
    ProgressBar,
    SweepReport,
    SimulationResult,
)


def get_optimal_workers() -> int:
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


def main():
    parser = argparse.ArgumentParser(
        description="Run TRINITY parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'param_file',
        help="Sweep-enabled .param file with list values"
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f"Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help="Show combinations without running simulations"
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate input file
    param_path = Path(args.param_file).resolve()
    if not param_path.exists():
        print(f"Error: Parameter file not found: {param_path}")
        sys.exit(1)

    # =============================================================================
    # Parse sweep parameters
    # =============================================================================

    print(f"\nReading sweep parameters from: {param_path}")
    try:
        config = read_sweep_config(str(param_path))
    except Exception as e:
        print(f"Error parsing parameter file: {e}")
        sys.exit(1)

    # Count combinations
    n_combinations = count_combinations_from_config(config)

    # =============================================================================
    # Display sweep summary
    # =============================================================================

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

    # =============================================================================
    # Determine workers
    # =============================================================================

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

    # =============================================================================
    # Determine output directory
    # =============================================================================

    # Use path2output from params if specified, else use 'outputs/'
    base_output_dir = config.base_params.get('path2output', 'outputs')
    if base_output_dir == 'def_dir':
        base_output_dir = os.path.join(os.getcwd(), 'outputs')

    # Make it absolute
    base_output_dir = str(Path(base_output_dir).resolve())

    print(f"\nOutput directory: {base_output_dir}")

    # =============================================================================
    # Dry run mode
    # =============================================================================

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

        for params, name in combinations:
            print(f"\n{name}:")
            for k in varying_keys:
                print(f"  {k}: {params.get(k)}")

        print(f"\n(Would run {n_combinations} simulations)")
        print("(No simulations were run - dry run mode)")
        sys.exit(0)

    # =============================================================================
    # Confirmation
    # =============================================================================

    if not args.yes:
        print("\n" + "-" * 60)
        response = input(f"Run {n_combinations} simulations with {workers} workers? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Aborted.")
            sys.exit(0)

    # =============================================================================
    # Setup output directory
    # =============================================================================

    output_base = Path(base_output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Run sweep
    # =============================================================================

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

    # =============================================================================
    # Generate report
    # =============================================================================

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

    # =============================================================================
    # Print summary
    # =============================================================================

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


if __name__ == "__main__":
    main()
