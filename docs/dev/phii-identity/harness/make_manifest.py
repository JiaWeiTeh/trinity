#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One row per array task: what ran, whether it finished, and how far it got.

The failure mode this exists to prevent: a 1000-task array WILL have partial failures, and
without a manifest a short reduced CSV reads as "the effect is small" rather than "300 tasks
died". Every reduced artifact in `data-new/` should be read next to this file.

Reads each run directory's `.provenance`, `.exit_code`, `.duration` (written by
`helix_array.sbatch`) plus the tail of `dictionary.jsonl` for the phases actually reached.

    python3 docs/dev/phii-identity/harness/make_manifest.py <arm_out_dir> \
        --arm b13_grid --sha abc1234 --out docs/dev/phii-identity/data-new/b13_grid_manifest.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402


def read_kv(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def read_one(path, cast=str, default=None):
    try:
        return cast(path.read_text().strip())
    except (OSError, ValueError):
        return default


def scan(run_dir):
    prov = read_kv(run_dir / ".provenance")
    phases, rows, t_last, fate = {}, 0, None, None
    dj = run_dir / "dictionary.jsonl"
    if dj.exists():
        with dj.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except (ValueError, TypeError):
                    continue
                rows += 1
                p = r.get("current_phase")
                if p:
                    phases[p] = phases.get(p, 0) + 1
                if isinstance(r.get("t_now"), (int, float)):
                    t_last = r["t_now"]
                if r.get("SimulationEndReason"):
                    fate = r["SimulationEndReason"]
    exit_code = read_one(run_dir / ".exit_code", int)
    return dict(
        name=prov.get("name", run_dir.name),
        param=prov.get("param"),
        code=prov.get("code"),
        slurm_job=prov.get("slurm_job"),
        node=prov.get("node"),
        started=prov.get("started"),
        exit_code=exit_code,
        # The three states that matter, kept separate on purpose.
        ok=(exit_code == 0 and rows > 0),
        duration_s=read_one(run_dir / ".duration", int),
        snapshots=rows,
        t_last=t_last,
        fate=fate,
        reached_transition="transition" in phases,
        reached_momentum="momentum" in phases,
        phases=",".join(f"{k}:{v}" for k, v in sorted(phases.items())) or None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_dir", type=Path)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--sha", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dirs = sorted(d for d in args.arm_dir.iterdir() if d.is_dir())
    if not dirs:
        sys.exit(f"no run directories under {args.arm_dir}")
    rows = [scan(d) for d in dirs]

    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    mom = sum(1 for r in rows if r["reached_momentum"])
    tra = sum(1 for r in rows if r["reached_transition"])
    print(f"arm {args.arm or args.arm_dir.name}  code {args.sha or '?'}")
    print(f"  tasks           {n}")
    print(f"  completed ok    {ok}   ({n - ok} FAILED OR EMPTY)")
    print(f"  reached 1c      {tra}")
    print(f"  reached phase 2 {mom}")
    if ok < n:
        print("\n  failures:")
        for r in rows:
            if not r["ok"]:
                print(f"    {r['name']:40s} exit={r['exit_code']} snapshots={r['snapshots']}")
    if mom < n:
        print(f"\n  ⚠️  {n - mom} tasks never reached the momentum phase. Any driving-branch")
        print("      gate is VOID on those, never a confirming null (PLAN.md, Batch 11/12 rule).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write(f"# arm {args.arm} | code {args.sha} | source {args.arm_dir}\n")
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out}  ({n} rows)")


if __name__ == "__main__":
    main()
