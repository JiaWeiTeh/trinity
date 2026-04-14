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
import os
import sys
from pathlib import Path


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
# Sweep mode
# =============================================================================

def run_sweep(args):
    """Run a TRINITY parameter sweep."""
    import multiprocessing
    import signal
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime
    from threading import Event

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
    from src.cloud_properties.validate_gmc import validate_gmc_params

    # Module-level shutdown flag for signal handler access
    global _shutdown_requested
    _shutdown_requested = Event()

    # -----------------------------------------------------------------
    # Sweep helper functions
    # -----------------------------------------------------------------

    def get_optimal_workers():
        """
        Determine a conservative default number of worker processes.

        Formula: ``max(1, cpu_count // 2 - 1)``.

        - Uses less than half the CPUs so the machine stays responsive
          for other work (browser, editor, meetings) while the sweep
          runs. On an 8-core laptop this gives 3 workers, leaving 5
          cores free; on a 4-core laptop it gives 1 worker.
        - Minimum of 1 (for 1-2 core machines where the formula would
          otherwise go to 0 or negative).
        - On HPC / workstations with many cores, pass ``--workers N``
          explicitly if you want more parallelism than this default.

        Rationale: each worker spawns a full simulation subprocess that
        uses BLAS/LAPACK; even with ``OMP_NUM_THREADS=1`` per worker,
        too many concurrent Python interpreters will saturate memory
        bandwidth and overheat laptops.
        """
        cpus = multiprocessing.cpu_count()
        return max(1, cpus // 2 - 1)

    def _validate_sweep_combination(params_dict):
        """
        Validate a single sweep combination's GMC parameters.

        Sweep parameter values come directly from the .param file *without*
        going through ``read_param``, so they are still in their input
        units (nCore/nISM in cm⁻³, mu_convert in m_H, mCloud in Msun,
        rCore in pc).  The GMC validator, however, expects values in
        TRINITY's astronomy code units (pc⁻³, Msun, pc).  Apply the same
        ``convert2au`` conversions that ``read_param`` applies on the
        single-run path so the preflight check matches what the actual
        simulation will see.

        Returns GMCValidationResult or None if validation cannot be performed.
        """
        from src._functions.unit_conversions import convert2au

        dens_profile = params_dict.get('dens_profile')
        if dens_profile not in ('densPL', 'densBE'):
            return None

        mCloud = params_dict.get('mCloud')
        nCore_cgs = params_dict.get('nCore')
        if mCloud is None or nCore_cgs is None:
            return None

        # Unit conversions matching param/default.param unit annotations:
        #   mCloud:       [Msun]      -> Msun           (factor 1)
        #   nCore, nISM:  [cm**-3]    -> pc⁻³           (factor ~2.94e+55)
        #   rCore:        [pc]        -> pc             (factor 1)
        #   mu_convert:   [m_H]       -> Msun           (factor ~9.42e-58)
        ndens_factor = convert2au('cm**-3')
        mu_factor = convert2au('m_H')

        mu = float(params_dict.get('mu_convert', 1.4)) * mu_factor
        nISM = float(params_dict.get('nISM', 1.0)) * ndens_factor

        kwargs = dict(
            mCloud=float(mCloud),
            nCore=float(nCore_cgs) * ndens_factor,
            mu=mu,
            nISM=nISM,
            dens_profile=dens_profile,
        )

        if dens_profile == 'densPL':
            alpha = params_dict.get('densPL_alpha')
            rCore = params_dict.get('rCore')
            if alpha is None:
                return None
            kwargs['alpha'] = float(alpha)
            if rCore is not None:
                kwargs['rCore'] = float(rCore)  # already in pc
        elif dens_profile == 'densBE':
            Omega = params_dict.get('densBE_Omega')
            if Omega is None:
                return None
            kwargs['Omega'] = float(Omega)  # dimensionless ratio
            kwargs['gamma'] = float(params_dict.get('gamma_adia', 5.0 / 3.0))

        try:
            return validate_gmc_params(**kwargs)
        except Exception:
            return None

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

    # Reconfigure logging for sweep mode — suppress parser noise
    log_level = logging.DEBUG if args.verbose else logging.WARNING
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

    workers = args.workers if args.workers is not None else get_optimal_workers()

    # Use path2output from params if specified, else use 'outputs/'
    base_output_dir = config.base_params.get('path2output', 'outputs')
    if base_output_dir == 'def_dir':
        base_output_dir = os.path.join(os.getcwd(), 'outputs')
    base_output_dir = str(Path(base_output_dir).resolve())

    print(f"\nTotal: {n_combinations} simulations | "
          f"Workers: {workers} | Output: {base_output_dir}")

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
        help="Path to .param file (single run or sweep with list syntax)"
    )
    # Sweep-specific options (ignored for single runs)
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help="Number of parallel workers for sweep mode "
             "(default: max(1, cpu_count // 2 - 1))"
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
