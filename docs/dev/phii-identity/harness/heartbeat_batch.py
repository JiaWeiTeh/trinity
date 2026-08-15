#!/usr/bin/env python3
"""Keep a batch of runs alive: poll, reap, restart the dead, cold-start the unstarted.

The container running these batches restarts without warning, and a run killed
mid-solve leaves a partial `dictionary.jsonl` with no `metadata.json`. This loop
is the recovery layer:

  * COMPLETE   dictionary.jsonl non-empty AND metadata.json present -> never touched
  * ALIVE      a tracked child process is still running               -> left alone
  * DEAD       tracked child exited without completing the run        -> relaunched
  * UNSTARTED  no run directory / no materialised .param at all       -> cold-started

Restart replays the run's OWN materialised `.param` (its `path2output` is baked
in at materialise time), so a restart can never scatter output to a second root.
Cold start materialises through run_batch, so the base param + override dict stay
the single source of truth (PLAN C-4).

It is safe to run this script again after the container itself dies: it re-derives
every state from disk, so completed runs are skipped and only the unfinished are
relaunched. Nothing is resumed mid-solve -- run.py has no checkpoint -- a dead run
restarts from t=0.

Usage:
    python docs/dev/phii-identity/harness/heartbeat_batch.py \
        --root outputs/phii/b7 --configs SC,GMC,BE --jobs 3 --stop-t 1.5
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_batch  # noqa: E402

REPO = Path(__file__).resolve().parents[4]


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated ids")
    ap.add_argument("--jobs", type=int, default=3, help="max concurrent runs")
    ap.add_argument("--stop-t")
    ap.add_argument("--poll", type=int, default=360, help="seconds between beats")
    ap.add_argument("--max-restarts", type=int, default=4,
                    help="per-config relaunch budget before it is declared FAILED")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    names = [c for c in args.configs.split(",") if c]
    unknown = [n for n in names if n not in run_batch.MATRIX]
    if unknown:
        sys.exit(f"unknown config id(s): {unknown}")
    extra = {"stop_t": args.stop_t} if args.stop_t else {}

    alive = {}          # cfg -> Popen
    restarts = {c: 0 for c in names}
    failed = set()

    def complete(cfg):
        return run_batch.done(root / cfg)

    def launch(cfg):
        run_dir = root / cfg
        param = run_dir / f"{cfg}.param"
        if not param.exists():
            # Cold start: no materialised param, so this config never ran at all.
            param, _ = run_batch.materialise(cfg, run_dir, extra)
            log(f"  {cfg}: cold-started (materialised {param.name})")
        else:
            log(f"  {cfg}: restarting from its own {param.name}")
        logf = (run_dir / "_heartbeat_run.log").open("ab")
        return subprocess.Popen(
            [sys.executable, "run.py", str(param), "--yes"],
            cwd=REPO, stdout=logf, stderr=subprocess.STDOUT,
        )

    log(f"heartbeat: root={root} configs={names} jobs={args.jobs} poll={args.poll}s")

    while True:
        # --- reap ------------------------------------------------------------
        for cfg, proc in list(alive.items()):
            if proc.poll() is None:
                continue
            del alive[cfg]
            if complete(cfg):
                log(f"  {cfg}: COMPLETE (rc={proc.returncode})")
            else:
                restarts[cfg] += 1
                if restarts[cfg] > args.max_restarts:
                    failed.add(cfg)
                    log(f"  {cfg}: FAILED — {restarts[cfg]} restarts exhausted")
                else:
                    log(f"  {cfg}: died rc={proc.returncode}, "
                        f"restart {restarts[cfg]}/{args.max_restarts}")

        # --- status ----------------------------------------------------------
        done = [c for c in names if complete(c)]
        pending = [c for c in names
                   if c not in done and c not in alive and c not in failed]
        if len(done) + len(failed) == len(names):
            log(f"ALL SETTLED — complete={len(done)} failed={sorted(failed)}")
            return 0 if not failed else 1

        # --- fill free slots -------------------------------------------------
        for cfg in pending:
            if len(alive) >= args.jobs:
                break
            alive[cfg] = launch(cfg)

        rows = {}
        for cfg in alive:
            d = root / cfg / "dictionary.jsonl"
            rows[cfg] = sum(1 for _ in d.open()) if d.exists() else 0
        log(f"done={len(done)}/{len(names)} running={rows} "
            f"queued={len([c for c in names if c not in done and c not in alive and c not in failed])}")

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
