#!/usr/bin/env python3
"""Gate check for one weak-winds batch — run this before starting the next one.

Two modes (see ../RUNBOOK.md):

  Batch health + fate table (default):
      python docs/dev/weak-winds/harness/check_batch.py outputs/weak_winds_study/c0p3

  Matched-t trajectory comparison of two runs (the H0 plumbing gate, and any
  A/B you want to bound):
      python docs/dev/weak-winds/harness/check_batch.py --compare RUN_A RUN_B

Exit code is the gate: 0 = every run completed, non-zero = something needs a
look before descending a rung. Reads run outputs only; writes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _rows(run_dir: Path):
    jsonl = run_dir / "dictionary.jsonl"
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]


def _coeff(run_dir: Path) -> str:
    """Knob value actually used, preferring the run's own metadata record."""
    meta = run_dir / "metadata.json"
    if meta.exists():
        value = json.loads(meta.read_text()).get("FB_thermCoeffWind")
        if value is not None:
            return str(value)
    for param in run_dir.glob("*.param"):
        for line in param.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "FB_thermCoeffWind":
                return parts[1]
    return "1 (default)"


def _fate(run_dir: Path, rows) -> str:
    """Termination fate, or '' if the run stopped for no recorded reason.

    Only runs that end on an *event* (collapse, dissolution, ...) stamp
    SimulationEndReason on their final snapshot; a run that simply reaches
    stop_t leaves the field empty (verified 2026-08-08 on the smoke pair), so
    a bare 'no reason' check would flag every healthy full-horizon run. Infer
    that case from the run's own stop_t; anything else really is unexplained.
    """
    reason = rows[-1].get("SimulationEndReason")
    if reason:
        return str(reason)
    meta = run_dir / "metadata.json"
    if meta.exists():
        stop_t = json.loads(meta.read_text()).get("stop_t")
        if stop_t and rows[-1]["t_now"] >= 0.999 * float(stop_t):
            return "stop_t reached"
    return ""


def _phase_path(rows) -> str:
    phases = [r.get("current_phase", "?") for r in rows]
    out = [phases[0]] if phases else []
    for prev, cur in zip(phases, phases[1:]):
        if cur != prev:
            out.append(cur)
    return ">".join(p[:4] for p in out)


def _wall_times(batch_dir: Path):
    report = batch_dir / "sweep_report.json"
    if not report.exists():
        return {}
    data = json.loads(report.read_text())
    return {r["name"]: r.get("duration") for r in data.get("results", [])}


def check_batch(batch_dir: Path) -> int:
    run_dirs = sorted(p.parent for p in batch_dir.rglob("dictionary.jsonl"))
    if not run_dirs:
        print(f"NO RUNS FOUND under {batch_dir}", file=sys.stderr)
        return 2

    walls = _wall_times(batch_dir)
    width = max(len(p.name) for p in run_dirs) + 2
    print(f"\n{batch_dir}  —  {len(run_dirs)} run(s)\n")
    header = (
        f"{'run':<{width}}{'coeff':>7}  {'fate':<22}{'t_end':>8}"
        f"{'R2_end':>9}{'R2_max':>9}{'wall':>8}  phases"
    )
    print(header)
    print("-" * len(header))

    # sweep_report.json is written only when the sweep finishes, so its absence
    # means "not confirmed complete" — either still running or launched as
    # single runs. Without this distinction, checking a batch mid-flight reports
    # a hard FAIL for runs that are merely still integrating.
    complete = (batch_dir / "sweep_report.json").exists()

    failures, unconfirmed = [], []
    for run_dir in run_dirs:
        rows = _rows(run_dir)
        last = rows[-1]
        R2 = [r["R2"] for r in rows]
        fate = _fate(run_dir, rows)
        wall = walls.get(run_dir.name)
        finite = all(np.isfinite(r["R2"]) and np.isfinite(r["v2"]) for r in rows)
        print(
            f"{run_dir.name:<{width}}{_coeff(run_dir):>7}  {(fate or '??')[:21]:<22}"
            f"{last['t_now']:>8.3f}{last['R2']:>9.3f}{max(R2):>9.3f}"
            f"{(f'{wall / 60:.1f}m' if wall else '-'):>8}  {_phase_path(rows)}"
        )
        if not finite:
            failures.append(f"{run_dir.name}: non-finite R2/v2 in trajectory")
        if not fate:
            (failures if complete else unconfirmed).append(
                f"{run_dir.name}: at t={last['t_now']:.4g} with no recorded "
                "reason and short of stop_t"
            )

    report = batch_dir / "sweep_report.json"
    if complete:
        data = json.loads(report.read_text())
        print(f"\nsweep_report: {data.get('succeeded')}/{data.get('total')} succeeded")
        for result in data.get("results", []):
            if not result.get("success"):
                failures.append(f"{result['name']}: run failed ({result.get('error')})")
        # A run that died before writing its first snapshot leaves no
        # dictionary.jsonl at all, so the table above cannot show it. Without
        # this the gate would pass a batch that is quietly short a run.
        found = {p.name for p in run_dirs}
        for result in data.get("results", []):
            if result["name"] not in found:
                failures.append(f"{result['name']}: no dictionary.jsonl written")

    if failures:
        print("\nGATE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if unconfirmed:
        print("\nGATE: INCOMPLETE — no sweep_report.json, so the batch is not")
        print("confirmed finished. Still-integrating runs:")
        for item in unconfirmed:
            print(f"  - {item}")
        print("Re-check once the sweep prints its report.")
        return 3
    print("\nGATE: PASS — all runs completed with a recorded fate. Descend a rung.")
    return 0


def compare(run_a: Path, run_b: Path, tol: float) -> int:
    """Max relative R2 deviation over the overlapping time range."""
    rows_a, rows_b = _rows(run_a), _rows(run_b)
    if not rows_a or not rows_b:
        print("missing dictionary.jsonl in one of the runs", file=sys.stderr)
        return 2

    ta, Ra = np.array([r["t_now"] for r in rows_a]), np.array([r["R2"] for r in rows_a])
    tb, Rb = np.array([r["t_now"] for r in rows_b]), np.array([r["R2"] for r in rows_b])

    # Matched simulation time only: runs truncate at different t (CLAUDE.md rule 5).
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    if not hi > lo:
        print(f"no overlapping time range ({lo:.4g}, {hi:.4g})", file=sys.stderr)
        return 2
    grid = np.linspace(lo, hi, 200)
    dev = np.abs(np.interp(grid, ta, Ra) - np.interp(grid, tb, Rb)) / np.abs(
        np.interp(grid, ta, Ra)
    )
    worst = float(np.max(dev))

    print(f"A: {run_a}  (coeff {_coeff(run_a)}, t -> {ta[-1]:.4f})")
    print(f"B: {run_b}  (coeff {_coeff(run_b)}, t -> {tb[-1]:.4f})")
    print(f"overlap: t = {lo:.6f} .. {hi:.6f} Myr over {len(grid)} samples")
    print(f"max |dR2/R2| = {worst:.3e}   (tolerance {tol:.1e})")
    if worst < tol:
        print("\nGATE: PASS — equivalent at matched t.")
        return 0
    print("\nGATE: FAIL — trajectories differ beyond tolerance.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir", type=Path, nargs="?", help="batch output directory")
    ap.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--tol", type=float, default=1e-9, help="--compare tolerance")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.tol)
    if not args.batch_dir:
        ap.error("give a batch directory, or --compare RUN_A RUN_B")
    return check_batch(args.batch_dir)


if __name__ == "__main__":
    sys.exit(main())
