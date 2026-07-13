#!/usr/bin/env python3
"""In-container / no-HPC runner for the Rosette Cf scan (72 arms, PISM=1e5).

Adapted from docs/dev/transition/pdv-trigger/runs/run_bench5_local.py (the proven resumable,
restart-surviving pattern), but driving an ``--emit-jobs`` bundle instead of a hand-rolled params
dir: ``python run.py param/rosette_cf_survey_PISM1e5_fmix.param --emit-jobs <dir>`` writes one
.param per combo (path2output already absolute), a manifest.json carrying each run's full axes
(coverFraction, cooling_boost_fmix, nCore, mass-pair, include_PHII), and uses the same
.exit_code/.duration sentinel convention this pool writes — maximal reuse, no name parsing.

Resumable: an arm is skipped if its output dir already has .exit_code (same container window) or
its row in the committed summary CSV is quotable, exit_code==0 (survives a container restart that
wipes the raw outputs). Each arm runs in its OWN subprocess (trinity leaks module-level globals;
in-process reuse would violate the separate-processes gate).

Decision-first ordering: dense nCore first, then Cf descending (the sealed Cf=1.0 baselines carry
the paper/rosette PLAN.md §0.3 adjudication), then fmix ascending — if the container dies
mid-campaign, the committed partial set is maximally decision-relevant.

📏 compliance: an arm killed at --per-arm-timeout gets exit_code 124 — NON-COMPLIANT, its t_final
is a wall-clock artifact, not a physics end. Never quote a Cf from it; re-run with a longer
timeout instead (match_cf_scan.py excludes exit_code!=0 rows from all minima).

    python docs/dev/rosette-cf/harness/run_cf_scan_local.py --jobs "$WS/cf_jobs" \
        --workers 3 --per-arm-timeout 3600 \
        --summary docs/dev/rosette-cf/data/cf_scan_PISM1e5_summary.csv
    # timing probes first: add  --only '*n1e5*coverFraction1p0*' --limit 2 --workers 1
"""

import argparse
import csv
import fnmatch
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def order_key(run):
    """Dense nCore first, then Cf descending (sealed Cf=1.0 §0.3 baselines first), fmix ascending."""
    p = run.get("params", {})
    return (
        -float(p.get("nCore", 0)),
        -float(p.get("coverFraction", 0)),
        float(p.get("cooling_boost_fmix", 1)),
        run["name"],
    )


def done_in_summary(summary_path):
    """run_names already quotable (exit_code==0) in the committed summary — skip on restart."""
    done = set()
    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as fh:
            for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#")):
                if r.get("exit_code") == "0":
                    done.add(r["run_name"])
    return done


def run_arm(run, timeout, env):
    name, outdir = run["name"], Path(run["output_dir"])
    if (outdir / ".exit_code").exists():
        return name, "skip", 0
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, str(REPO / "run.py"), run["param_path"]],
            cwd=REPO,
            env=env,
            timeout=timeout,
            stdout=open(outdir / "run.log", "w"),
            stderr=subprocess.STDOUT,
        )
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = 124  # wall-killed -> non-compliant; never quote its Cf (📏)
    dur = int(time.time() - t0)
    (outdir / ".exit_code").write_text(f"{code}\n")
    (outdir / ".duration").write_text(f"{dur}\n")
    return name, ("ok" if code == 0 else f"exit{code}"), dur


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="--emit-jobs bundle dir (contains manifest.json)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--per-arm-timeout", type=int, default=3600, help="seconds (default 60 min)")
    ap.add_argument("--summary", help="committed summary CSV; skip runs already quotable there")
    ap.add_argument("--only", action="append", default=[], help="fnmatch on run name (repeatable)")
    ap.add_argument("--limit", type=int, help="run at most N arms (after ordering) — timing probes")
    ap.add_argument("--dry-run", action="store_true", help="list the ordered todo, run nothing")
    args = ap.parse_args(argv)

    manifest = json.loads((Path(args.jobs) / "manifest.json").read_text())
    runs = sorted(manifest["runs"], key=order_key)
    if args.only:
        runs = [r for r in runs if any(fnmatch.fnmatch(r["name"], pat) for pat in args.only)]

    harvested = done_in_summary(args.summary)
    todo = [
        r
        for r in runs
        if not (Path(r["output_dir"]) / ".exit_code").exists() and r["name"] not in harvested
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"[cf-scan] {len(manifest['runs'])} arms in bundle, {len(runs)} selected, "
        f"{len(todo)} to run ({len(runs) - len(todo)} already done), {args.workers} workers",
        flush=True,
    )
    if args.dry_run:
        for r in todo:
            print(f"  {r['name']}")
        return

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_arm, r, args.per_arm_timeout, env): r for r in todo}
        for fut in as_completed(futs):
            name, status, dur = fut.result()
            done += 1
            print(f"[cf-scan] {done}/{len(todo)}  {name:64s} {status:8s} {dur}s", flush=True)
    print(f"[cf-scan] finished: {done} arms ran this pass", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
