#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLURM job-array generation and result collection for TRINITY sweeps.

The in-process sweep runner (run.py:run_sweep) parallelises a sweep across
the cores of a *single* machine. On an HPC cluster the conventional pattern
is instead a scheduler job array: one array task per combination, so the
scheduler packs them across many nodes, handles fair-share, and restarts
failed tasks independently.

``emit_jobs`` writes a submittable bundle (one .param per combination, a
manifest, and an sbatch script); ``collect_report`` aggregates the per-task
exit-code sentinels into the same SweepReport the in-process runner produces.

Each array task simply runs ``python run.py <combo>.param`` -- the emitted
files contain only scalar values, so they route through the single-run path
(no nested sweep). Inputs are located relative to the package
(_REPO_ROOT-anchored) and each combo's path2output is absolute, so tasks are
independent of the working directory the array runs in.
"""

import json
import os
import stat
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from trinity._input.sweep_parser import generate_combinations_from_config
from trinity._input.sweep_runner import (
    generate_param_file,
    _validate_sweep_combination,
    SimulationResult,
    SweepReport,
)


MANIFEST_NAME = 'manifest.json'
RUNS_TSV_NAME = 'runs.tsv'
SBATCH_NAME = 'submit_sweep.sbatch'
EXIT_CODE_FILE = '.exit_code'
DURATION_FILE = '.duration'


# One simulation per array task. Math libraries are pinned to a single thread
# (parallelism comes from running many tasks, not from threading one sim),
# mirroring the in-process runner's per-worker environment. Paths are absolute
# so the task is independent of the directory the array runs in.
_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=trinity_sweep
#SBATCH --array=1-{n}{throttle}
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --output={logs_dir}/%A_%a.out
# --- EDIT for your cluster (e.g. bwForCluster Helix / bwUniCluster): ---
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --partition=cpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# Optional OFFSET (set via --export=OFFSET=N) shifts the runs.tsv line, so a
# grid larger than a MaxSubmitJobs cap can be submitted in chunks (each chunk a
# separate --array with its own OFFSET). Defaults to 0 (plain single submit).
N=$(( SLURM_ARRAY_TASK_ID + ${{OFFSET:-0}} ))
LINE=$(sed -n "${{N}}p" "{runs_tsv}")
if [ -z "$LINE" ]; then
    echo "ERROR: no line $N in runs.tsv (TASK=$SLURM_ARRAY_TASK_ID OFFSET=${{OFFSET:-0}})" >&2
    exit 1
fi
PARAM=$(printf '%s' "$LINE" | cut -f1)
OUTDIR=$(printf '%s' "$LINE" | cut -f2)
mkdir -p "$OUTDIR"
SECONDS=0
python "{run_py}" "$PARAM"
code=$?
echo "$code" > "$OUTDIR/{exit_code_file}"
echo "$SECONDS" > "$OUTDIR/{duration_file}"
exit $code
"""


def emit_jobs(config, base_output_dir, jobs_dir, trinity_root,
              concurrency=None, dry_run=False, sweep_file=None):
    """Generate a SLURM job-array bundle for a sweep.

    Writes ``<jobs_dir>/params/<name>.param`` per combination, ``runs.tsv``
    (``param_path<TAB>output_dir``, one per line, index == SLURM array id),
    a self-describing ``manifest.json``, and ``submit_sweep.sbatch``.

    Parameters
    ----------
    config : SweepConfig
    base_output_dir : str
        Absolute base directory; each run lands in ``<base>/<name>``.
    jobs_dir : str or Path
        Where the bundle is written (resolved to absolute).
    trinity_root : Path
        TRINITY root, used to locate ``run.py`` in the sbatch script.
    concurrency : int or None
        Array throttle ``%K`` (from ``--workers``); None means no limit.
    dry_run : bool
        If True, validate and print a summary but write nothing.
    sweep_file : str or None
        Original sweep param path, recorded in the manifest for the report.

    Returns
    -------
    (n_jobs, n_invalid) : tuple[int, int]
    """
    jobs_dir = Path(jobs_dir).resolve()
    base_output_dir = str(Path(base_output_dir).resolve())

    combinations = list(generate_combinations_from_config(config))
    n_jobs = len(combinations)
    if n_jobs == 0:
        sys.exit("Error: sweep produced no combinations to emit.")

    # Validate up front so the operator sees implausible combos before
    # queueing doomed tasks. Warn but still emit: a few bad combos shouldn't
    # block the rest (they fail fast and surface in the collected report),
    # matching run_sweep's behaviour.
    invalid = []
    for params, name in combinations:
        result = _validate_sweep_combination(params)
        if result is not None and not result.valid:
            invalid.append((name, result))
    n_invalid = len(invalid)

    throttle = f"%{concurrency}" if concurrency else ""

    if dry_run:
        print(f"\n[dry-run] Would emit {n_jobs} job(s) to {jobs_dir}")
        print(f"          base output: {base_output_dir}")
        print(f"          array: 1-{n_jobs}{throttle}")
        if n_invalid:
            print(f"          WARNING: {n_invalid}/{n_jobs} combos have "
                  f"implausible GMC parameters and will likely fail.")
        print("          (nothing written)")
        return n_jobs, n_invalid

    # Overwrite guard: never clobber an existing bundle. A submitted array
    # reads runs.tsv by index, so regenerating in place would desync a
    # running job. Require a fresh directory.
    manifest_path = jobs_dir / MANIFEST_NAME
    if manifest_path.exists():
        sys.exit(
            f"Error: {manifest_path} already exists. Refusing to overwrite a "
            f"job bundle (a running array reads runs.tsv by index). Use a "
            f"fresh --emit-jobs directory."
        )

    params_dir = jobs_dir / 'params'
    logs_dir = jobs_dir / 'logs'
    params_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    tsv_lines = []
    for index, (params, name) in enumerate(combinations, start=1):
        out_dir = os.path.join(base_output_dir, name)
        param_path = params_dir / f"{name}.param"
        param_path.write_text(
            generate_param_file(params, name, out_dir), encoding='utf-8'
        )
        runs.append({
            'index': index,
            'name': name,
            'param_path': str(param_path),
            'output_dir': out_dir,
            'params': params,
        })
        tsv_lines.append(f"{param_path}\t{out_dir}")

    (jobs_dir / RUNS_TSV_NAME).write_text(
        "\n".join(tsv_lines) + "\n", encoding='utf-8'
    )

    manifest = {
        'sweep_file': sweep_file or '',
        'base_output_dir': base_output_dir,
        'generated': datetime.now().isoformat(),
        'concurrency': concurrency,
        'n_jobs': n_jobs,
        'runs': runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    sbatch_path = jobs_dir / SBATCH_NAME
    sbatch_path.write_text(_SBATCH_TEMPLATE.format(
        n=n_jobs,
        throttle=throttle,
        logs_dir=logs_dir,
        runs_tsv=jobs_dir / RUNS_TSV_NAME,
        run_py=Path(trinity_root) / 'run.py',
        exit_code_file=EXIT_CODE_FILE,
        duration_file=DURATION_FILE,
    ), encoding='utf-8')
    sbatch_path.chmod(sbatch_path.stat().st_mode
                      | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"\nEmitted {n_jobs} job(s) to {jobs_dir}")
    print(f"  params:   {params_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"  sbatch:   {sbatch_path}")
    print(f"  output:   {base_output_dir}/<run_name>")
    if n_invalid:
        print(f"\nWARNING: {n_invalid}/{n_jobs} combinations have implausible "
              f"GMC parameters and will likely fail:")
        for name, result in invalid:
            print(f"  {name}:")
            for e in result.errors:
                print(f"    - {e}")
    print(f"\nSubmit with:\n  sbatch {sbatch_path}")
    if not concurrency:
        print("Tip: cap concurrent tasks with --workers K "
              "(adds %K to the array) or edit the sbatch.")
    return n_jobs, n_invalid


def _fmt(v):
    """Float-with-no-fraction -> int for tidy printing (100000.0 -> 100000); else as-is."""
    return int(v) if isinstance(v, float) and v.is_integer() else v


def failure_breakdown(failed, manifest_runs):
    """Tally failed runs by each *swept* parameter (and by return code) so a regime-shaped
    failure -- e.g. 'small clouds + high sfe + high cooling_boost_kappa' -- is visible at a
    glance instead of only a flat list of array indices. Returns the printable block ('' if none).

    A 'swept' axis is a param that takes >1 but <n_runs distinct values (so per-run identifiers
    like path2output/model_name are excluded). Return code -2 = no sentinel (the task was killed,
    e.g. wall-time/OOM, or was still running at collect time), not a sim-level crash.
    """
    if not failed:
        return ""
    n = len(manifest_runs)
    valuesets = {}
    for run in manifest_runs:
        for k, v in (run.get('params') or {}).items():
            valuesets.setdefault(k, set()).add(tuple(v) if isinstance(v, list) else v)
    axes = [k for k, vs in valuesets.items() if 1 < len(vs) < n]

    rc_note = {-2: "no sentinel: wall-time/OOM kill, or still running",
               -1: "unreadable sentinel"}
    rc = Counter(r.return_code for r in failed)
    out = ["\nFailed runs by parameter (look for a regime, not just indices):",
           "  return code: " + ", ".join(
               f"{code}x{cnt}" + (f" [{rc_note[code]}]" if code in rc_note else " [sim exited nonzero]")
               for code, cnt in sorted(rc.items()))]
    for k in axes:
        c = Counter(r.params.get(k) for r in failed)
        body = ", ".join(f"{_fmt(val)}: {cnt}" for val, cnt
                         in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))
        out.append(f"  {k}: {body}")
    return "\n".join(out)


def collect_report(jobs_dir):
    """Aggregate per-task results into a SweepReport.

    Reads ``<jobs_dir>/manifest.json`` and each run's ``.exit_code`` /
    ``.duration`` sentinels, then writes ``sweep_report.txt`` /
    ``sweep_report.json`` into the sweep's base output directory -- identical
    to what the in-process runner produces.
    """
    jobs_dir = Path(jobs_dir).resolve()
    manifest_path = jobs_dir / MANIFEST_NAME
    if not manifest_path.exists():
        sys.exit(f"Error: no {MANIFEST_NAME} in {jobs_dir}. "
                 f"Pass the directory created by --emit-jobs.")

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    base_output_dir = Path(manifest['base_output_dir'])

    try:
        start_time = datetime.fromisoformat(manifest['generated'])
    except (KeyError, ValueError):
        start_time = datetime.now()
    end_time = datetime.now()

    successful = []
    failed = []
    failed_indices = []
    n_runs = len(manifest['runs'])
    t0 = time.monotonic()
    print(f"Scanning {n_runs} run sentinels (.exit_code/.duration)...", flush=True)
    for i, run in enumerate(manifest['runs'], start=1):
        if i % 250 == 0 or i == n_runs:
            print(f"  {i}/{n_runs}  ({time.monotonic() - t0:.0f}s)", flush=True)
        out_dir = Path(run['output_dir'])
        exit_file = out_dir / EXIT_CODE_FILE
        dur_file = out_dir / DURATION_FILE

        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                code = -1
        else:
            code = -2  # no sentinel: task did not run or is still running

        try:
            duration = (float(dur_file.read_text().strip())
                        if dur_file.exists() else 0.0)
        except ValueError:
            duration = 0.0

        success = (code == 0)
        if success:
            error_message = None
        elif code == -2:
            error_message = ("No exit-code sentinel "
                             "(task did not run or is still running).")
        else:
            error_message = f"Simulation exited with code {code}. See {out_dir}."

        result = SimulationResult(
            name=run['name'],
            params=run.get('params', {}),
            success=success,
            return_code=code,
            duration=duration,
            error_message=error_message,
            output_path=str(out_dir),
        )
        if success:
            successful.append(result)
        else:
            failed.append(result)
            failed_indices.append(run['index'])

    report = SweepReport(
        sweep_file=manifest.get('sweep_file', '') or str(manifest_path),
        start_time=start_time,
        end_time=end_time,
        total_combinations=manifest['n_jobs'],
        successful=successful,
        failed=failed,
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    # JSON first: it needs no extra I/O, so the machine-readable report is
    # guaranteed even if the (slower) text report's per-run metadata reads stall
    # or are interrupted on a large sweep.
    js = report.write_json(base_output_dir)
    print(f"Wrote {js}", flush=True)
    print("Building text report (reads each run's metadata.json — slower)...",
          flush=True)
    txt = report.write_report(base_output_dir)

    print(f"\nCollected {len(successful) + len(failed)} task(s): "
          f"{len(successful)} succeeded, {len(failed)} failed.")
    print(f"Reports written to:\n  {txt}\n  {js}")
    if failed_indices:
        print(failure_breakdown(failed, manifest['runs']))
        ids = ",".join(str(i) for i in sorted(failed_indices))
        print(f"\nRe-run only the failed tasks:\n"
              f"  sbatch --array={ids} {jobs_dir / SBATCH_NAME}")
    return report
