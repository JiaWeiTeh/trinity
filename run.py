#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:27:38 2022

@author: Jia Wei Teh

This script contains the main file to run TRINITY.

Unified entry point for single runs and parameter sweeps.
Auto-detects sweep mode when parameters contain list syntax or tuple definitions.

Usage (single run):
    python run.py param/example.param

Usage (sweep - auto-detected):
    python run.py param/sweep.param
    python run.py param/sweep.param --workers 4
    python run.py param/sweep.param --dry-run
    python run.py param/sweep.param --yes

Usage (cluster job array - SLURM, e.g. bwForCluster Helix):
    python run.py param/sweep.param --emit-jobs jobs/
    sbatch jobs/submit_sweep.sbatch
    python run.py --collect-report jobs/
"""

import argparse
import logging
import os
import sys
from pathlib import Path


# Module-level logger handle. The actual logging.basicConfig() call lives
# inside __main__ so that ProcessPoolExecutor workers re-importing this
# module (under 'spawn' on macOS/Windows) don't reconfigure logging and
# leak messages into the parent terminal.
early_logger = logging.getLogger(__name__)

TRINITY_ROOT = Path(__file__).parent.resolve()


# =============================================================================
# Dependency version advisory
# =============================================================================
# Exclusive upper major for each core dependency (matches requirements.txt).
# TRINITY is tested below these; a newer major may have breaking changes.
_DEP_MAX_MAJOR = {
    'numpy': 2, 'scipy': 2, 'astropy': 8, 'matplotlib': 4, 'pandas': 3,
}


def warn_if_unsupported_deps():
    """Warn (without failing) when an installed core dependency is newer than
    the tested range, and point the user at the supported set."""
    from importlib.metadata import version, PackageNotFoundError

    for pkg, max_major in _DEP_MAX_MAJOR.items():
        try:
            installed = version(pkg)
        except PackageNotFoundError:
            continue
        try:
            major = int(installed.split('.')[0])
        except ValueError:
            continue
        if major >= max_major:
            early_logger.warning(
                "%s %s is newer than the tested range (<%d). If you hit "
                "errors, install the supported versions with: "
                "pip install -r requirements.txt",
                pkg, installed, max_major,
            )


# =============================================================================
# Auto-detection
# =============================================================================

def is_sweep_param_file(path2file):
    """
    Quick scan to detect if a parameter file contains sweep/tuple syntax.

    Returns True if the file contains:
    - List syntax with multiple elements: key [val1, val2, ...]
    - Tuple syntax: tuple(param1, param2, ...) [val1, val2] ...

    Exits with a clean error message if the file cannot be opened,
    so callers get a friendly message instead of a stack trace.
    """
    if not os.path.exists(path2file):
        print(f"Error: Parameter file not found: {path2file}", file=sys.stderr)
        sys.exit(1)
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
# Worker / output-path helpers
# =============================================================================

def positive_int(value):
    """argparse type: a strictly positive integer (for --workers).

    Rejecting <= 0 here yields a clean argparse error (exit 2) instead of a
    ProcessPoolExecutor ``ValueError`` traceback later in the run.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def resolve_base_output_dir(config):
    """Absolute base output directory for a sweep.

    Mirrors the single-run sentinel handling: 'def_dir' (or an unset
    path2output) -> ``<cwd>/outputs``; a user path is taken as given. Always
    returns an absolute path so emitted job-array param files and the
    in-process runner agree regardless of the launch directory.
    """
    base = config.base_params.get('path2output', 'outputs')
    if base == 'def_dir':
        base = os.path.join(os.getcwd(), 'outputs')
    return str(Path(base).resolve())


# =============================================================================
# Single-run mode
# =============================================================================

def run_single(args):
    """Run a single TRINITY simulation."""
    from trinity._input import read_param
    from trinity._output import header

    params = read_param.read_param(args.path2param)

    header.show_param(params)


    from trinity import main

    # main_dict = create_dictionary.create()


    # =================================================================
    # Reconfigure logging with params settings
    # =================================================================
    # Now reconfigure with user's preferred log level from params
    from trinity._functions.logging_setup import setup_logging

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
    logger.info(f"Output directory: {os.path.abspath(params['path2output'].value)}")


    # =================================================================
    # Validate GMC parameters before running
    # =================================================================
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params

    gmc_check = validate_gmc_from_params(params)
    for w in gmc_check.warnings:
        logger.warning(w)
    if not gmc_check.valid:
        logger.error("GMC parameter validation failed:")
        for e in gmc_check.errors:
            logger.error(f"  {e}")
        if gmc_check.suggestions:
            from trinity.cloud_properties.validate_gmc import format_suggestion
            logger.info("Suggested valid alternatives:")
            for i, s in enumerate(gmc_check.suggestions, 1):
                logger.info(f"  {i}. {format_suggestion(s)}")
        # Record the real cause so the atexit-driven termination_debug block
        # reports it instead of the generic "Normal exit / atexit".
        params.set_termination_reason(
            "GMC validation failed: implausible cloud parameters "
            "(simulation not started)"
        )
        sys.exit("Simulation stopped: implausible GMC parameters. See errors above.")


    main.start_expansion(params)


# =============================================================================
# Sweep mode
# =============================================================================

def run_sweep(args):
    """Run a TRINITY parameter sweep."""
    import shutil
    import signal
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime
    from threading import Event

    from trinity._functions.cluster import detect_allocated_cpus, get_optimal_workers
    from trinity._input.sweep_parser import (
        read_sweep_config,
        generate_combinations_from_config,
        count_combinations_from_config,
    )
    from trinity._input.sweep_runner import (
        run_single_simulation,
        ProgressBar,
        SweepReport,
        SimulationResult,
        _validate_sweep_combination,
    )

    # Module-level shutdown flag for signal handler access
    global _shutdown_requested
    _shutdown_requested = Event()

    # -----------------------------------------------------------------
    # Sweep helper functions
    # -----------------------------------------------------------------

    def _build_parent_map():
        """
        Return ``{ppid: [child_pid, ...]}`` for all processes visible to us.

        Cross-platform best-effort. Uses ``ps`` on POSIX (Linux, macOS,
        Ubuntu, BSD) and ``wmic`` on Windows (falls back silently if
        unavailable — e.g. sandboxed environments).
        """
        import subprocess as _sp
        parent_map = {}
        is_windows = sys.platform.startswith('win')

        if is_windows:
            try:
                result = _sp.run(
                    ['wmic', 'process', 'get', 'ParentProcessId,ProcessId'],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines()[1:]:  # skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            ppid = int(parts[0])
                            pid = int(parts[1])
                        except ValueError:
                            continue
                        parent_map.setdefault(ppid, []).append(pid)
            except (OSError, _sp.SubprocessError, FileNotFoundError):
                pass
        else:
            try:
                result = _sp.run(
                    ['ps', '-eo', 'pid=,ppid='],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            pid = int(parts[0])
                            ppid = int(parts[1])
                        except ValueError:
                            continue
                        parent_map.setdefault(ppid, []).append(pid)
            except (OSError, _sp.SubprocessError, FileNotFoundError):
                pass

        return parent_map

    def _kill_sweep_children(worker_pids, sig_module):
        """
        Kill only the given pool workers and any of their descendants.

        Safety guarantees (important for shared/HPC environments):

        * Never uses ``os.killpg()`` or any process-group signalling,
          so unrelated processes sharing our shell/IDE/login-node
          session group are untouched.
        * Walks *only downward* from our known worker PIDs through a
          parent→child map. It is structurally impossible to traverse
          into another user's (or another job's) process tree because
          PPID chains can only be followed from parent to child, and
          our workers were spawned by us.
        * Every kill goes through ``os.kill(pid, sig)``, which the
          OS enforces via UID-based permissions. On a shared HPC node,
          ``os.kill`` on another user's PID returns EPERM and does
          nothing — even if a PID were (impossibly) misidentified.
        * Self-guard: never signals our own PID.
        * Works on Linux, macOS, Ubuntu, BSD, and Windows.
        """
        import time as _time

        # Defensive filter: drop empties and our own PID.
        worker_pids = {p for p in worker_pids if p and p != os.getpid()}
        if not worker_pids:
            return

        to_kill = set(worker_pids)

        # BFS downward through the parent→child map to find descendants
        # (e.g. each worker's `python run.py <param>` subprocess).
        parent_map = _build_parent_map()
        queue = list(worker_pids)
        while queue:
            ppid = queue.pop(0)
            for child_pid in parent_map.get(ppid, []):
                if child_pid == os.getpid():
                    continue  # paranoia: never target ourselves
                if child_pid not in to_kill:
                    to_kill.add(child_pid)
                    queue.append(child_pid)

        # Signal constants differ across platforms (Windows has no SIGKILL).
        sigterm = getattr(sig_module, 'SIGTERM', None)
        sigkill = getattr(sig_module, 'SIGKILL', None)

        # Phase 1: polite SIGTERM (on Windows this calls TerminateProcess).
        if sigterm is not None:
            for pid in to_kill:
                try:
                    os.kill(pid, sigterm)
                except OSError:
                    # Process already gone or owned by another user (EPERM) —
                    # either way, safe to ignore.
                    pass
            _time.sleep(0.5)

        # Phase 2: SIGKILL for stragglers (POSIX only).
        if sigkill is not None:
            for pid in to_kill:
                try:
                    os.kill(pid, sigkill)
                except OSError:
                    pass

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

    print("\nSweep summary")
    print("-" * 50)

    # Show base params as a compact count (use --verbose for full list)
    n_base = len(config.base_params)
    if args.verbose:
        print(f"\nBase parameters ({n_base} constant):")
        for k, v in sorted(config.base_params.items()):
            print(f"  {k}: {v}")
    else:
        print(f"\nBase parameters: {n_base} constant across all runs")

    if config.is_hybrid_mode:
        print(f"\nHybrid mode:")
        print(f"  Tuple: {config.tuple_params} "
              f"({len(config.tuple_values)} combinations)")
        for i, values in enumerate(config.tuple_values[:5]):
            combo_str = ", ".join(f"{p}={v}" for p, v in zip(config.tuple_params, values))
            print(f"    [{i+1}] {combo_str}")
        if len(config.tuple_values) > 5:
            print(f"    ... and {len(config.tuple_values) - 5} more")
        print(f"  Sweep (Cartesian with tuples):")
        for k, v in sorted(config.sweep_params.items()):
            print(f"    {k}: {v}")
    elif config.is_tuple_mode:
        print(f"\nTuple mode: {config.tuple_params}")
        for i, values in enumerate(config.tuple_values[:5]):
            combo_str = ", ".join(f"{p}={v}" for p, v in zip(config.tuple_params, values))
            print(f"  [{i+1}] {combo_str}")
        if len(config.tuple_values) > 5:
            print(f"  ... and {len(config.tuple_values) - 5} more")
    else:
        print(f"\nSweep parameters:")
        for k, v in sorted(config.sweep_params.items()):
            print(f"  {k}: {v}")

    # =================================================================
    # Determine workers and output directory
    # =================================================================

    if args.workers is not None:
        workers = args.workers
        worker_source = "explicit --workers"
    else:
        workers = get_optimal_workers()
        worker_source = ("SLURM allocation" if os.environ.get("SLURM_JOB_ID")
                         else "laptop default")

    # Resolve to an absolute base output directory (see resolve_base_output_dir).
    base_output_dir = resolve_base_output_dir(config)

    print(f"\nTotal: {n_combinations} simulations | "
          f"Workers: {workers} ({worker_source}) | Output: {base_output_dir}")

    # =================================================================
    # Dry run mode
    # =================================================================

    if args.dry_run:
        print("\nDry run — combinations to be generated:")
        print("-" * 50)

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
    # Worker budget + environment advisories (real runs only)
    # =================================================================
    # An explicit --workers larger than the cores actually available to this
    # process oversubscribes the machine (or the SLURM allocation), so we
    # refuse rather than thrash. The default path is never capped here:
    # get_optimal_workers() already respects the allocation.
    if args.workers is not None:
        n_alloc, alloc_source = detect_allocated_cpus()
        if args.workers > n_alloc:
            sys.exit(
                f"Error: --workers {args.workers} exceeds the {n_alloc} core(s) "
                f"available to this process (detected via {alloc_source}). "
                f"Re-run with --workers <= {n_alloc}."
            )

    # Nudge toward conventional cluster usage (advisory only).
    if os.environ.get('SLURM_JOB_ID'):
        print("\nNote: the in-process pool inside a SLURM job uses only this "
              "node's allocation. For multi-node scaling, generate a job array:\n"
              "  python run.py <param> --emit-jobs jobs/ && sbatch jobs/submit_sweep.sbatch")
    elif shutil.which('sbatch') is not None:
        print("\nNote: SLURM detected but no active job - you may be on a login "
              "node. Running a sweep here is discouraged; submit a batch job or "
              "use --emit-jobs to generate a job array.")

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
        print()
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

    print("\nStarting sweep (Ctrl+C to cancel)")
    print("-" * 50)

    start_time = datetime.now()

    # Generate all combinations
    combinations = list(generate_combinations_from_config(config))
    n_combinations = len(combinations)

    progress = ProgressBar(n_combinations, desc="Sweep")

    successful = []
    failed = []
    cancelled = []

    # Setup shutdown handlers. We trap both SIGINT (Ctrl+C) and SIGTERM
    # (sent by HPC schedulers like SLURM `scancel` or PBS on timeout)
    # so subprocess simulations don't get orphaned when a cluster job
    # is cancelled or hits its walltime grace period.
    def signal_handler(signum, frame):
        _shutdown_requested.set()
        label = "Ctrl+C" if signum == signal.SIGINT else f"signal {signum}"
        print(f"\n\n*** {label} received - cancelling remaining simulations... ***\n")

    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = None
    try:
        # SIGTERM exists on POSIX and Windows; signal.signal() for SIGTERM
        # may fail on non-main threads or restricted environments.
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError, OSError):
        pass

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
                    error_message="Cancelled (Ctrl+C or SIGTERM)"
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
                if result.error_message:
                    msg = result.error_message[:100]
                    if len(result.error_message) > 100:
                        msg += '...'
                else:
                    msg = 'Unknown error'
                logger.warning(f"FAILED: {result.name} - {msg}")

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
                        error_message="Cancelled (Ctrl+C or SIGTERM)"
                    ))

    except KeyboardInterrupt:
        print("\n\n*** Sweep cancelled by user ***\n")
        _shutdown_requested.set()
    finally:
        # Restore original signal handlers.
        signal.signal(signal.SIGINT, original_sigint)
        if original_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, original_sigterm)
            except (AttributeError, ValueError, OSError):
                pass

        if executor:
            if _shutdown_requested.is_set():
                # Ctrl+C / SIGTERM path: capture worker PIDs BEFORE shutdown
                # (shutdown(wait=False) may start clearing _processes), then
                # kill only our own process tree.
                worker_pids = set(getattr(executor, '_processes', {}).keys())
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python 3.8: no cancel_futures kwarg.
                    executor.shutdown(wait=False)
                _kill_sweep_children(worker_pids, signal)
            else:
                # Normal completion: all futures already resolved, workers
                # are idle. wait=True simply joins them — no kill, no signal,
                # no effect on any process outside our pool.
                executor.shutdown(wait=True)

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

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser(
        description="Run TRINITY simulation or parameter sweep (auto-detected)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Positional: path to param file
    parser.add_argument(
        'path2param',
        nargs='?',
        default=None,
        help="Path to .param file (single run or sweep with list syntax). "
             "Optional only with --collect-report."
    )
    # Sweep-specific options (ignored for single runs)
    parser.add_argument(
        '--workers', '-w',
        type=positive_int,
        default=None,
        help="Number of parallel workers for sweep mode (must be >= 1; "
             "default: full SLURM allocation in a job, else "
             "max(1, cpu_count // 2 - 1)). Refused if it exceeds the cores "
             "available to this process."
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
    # Cluster job-array options (mutually exclusive).
    cluster_group = parser.add_mutually_exclusive_group()
    cluster_group.add_argument(
        '--emit-jobs', metavar='DIR', default=None,
        help="Generate a SLURM job-array bundle (one task per sweep "
             "combination) in DIR instead of running locally. Requires a "
             "sweep param file."
    )
    cluster_group.add_argument(
        '--collect-report', metavar='DIR', default=None,
        help="Aggregate results from an --emit-jobs DIR into "
             "sweep_report.txt/.json (run after the array finishes)."
    )
    # grab argument
    args = parser.parse_args()

    # Configure logging now that we know --verbose. Kept inside __main__ so
    # spawn-based ProcessPoolExecutor workers don't reconfigure on re-import.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    for lib in ['matplotlib', 'PIL', 'urllib3', 'asyncio', 'parso', 'fontTools', 'numba', 'h5py']:
        logging.getLogger(lib).setLevel(logging.INFO)

    warn_if_unsupported_deps()

    # Banner first so it shows in both single and sweep modes.
    from trinity._output import header
    header.display()

    # Cluster job-array collection: self-contained from the manifest, so it
    # needs no param file.
    if args.collect_report is not None:
        from trinity._input.sweep_jobs import collect_report
        collect_report(args.collect_report)
        sys.exit(0)

    if args.path2param is None:
        parser.error("path2param is required unless using --collect-report")

    # Cluster job-array generation (opt-in; requires a sweep param file).
    if args.emit_jobs is not None:
        if not is_sweep_param_file(args.path2param):
            sys.exit("--emit-jobs requires a sweep param file (list/tuple syntax).")
        from trinity._input.sweep_parser import read_sweep_config
        from trinity._input.sweep_jobs import emit_jobs
        config = read_sweep_config(args.path2param)
        emit_jobs(
            config,
            resolve_base_output_dir(config),
            args.emit_jobs,
            TRINITY_ROOT,
            concurrency=args.workers,
            dry_run=args.dry_run,
            sweep_file=args.path2param,
        )
        sys.exit(0)

    # Auto-detect mode from parameter file content
    if is_sweep_param_file(args.path2param):
        run_sweep(args)
    else:
        # Single-run mode. The sweep-only flags have no effect here. Rather
        # than silently ignore them (a foot-gun, especially for --dry-run,
        # which a user may expect to suppress the run), handle them explicitly.
        if args.dry_run:
            print("Single run - dry run, nothing executed.")
            print(f"Parameter file: {args.path2param}")
            print("-" * 50)
            with open(args.path2param, 'r', encoding='utf-8') as f:
                print(f.read())
            sys.exit(0)
        ignored = [flag for flag, used in
                   (('--workers', args.workers is not None), ('--yes', args.yes))
                   if used]
        if ignored:
            print(f"Note: {', '.join(ignored)} only apply to sweeps; "
                  f"ignoring for single run.")
        run_single(args)
