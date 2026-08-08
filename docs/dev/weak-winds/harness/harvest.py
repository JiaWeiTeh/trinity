#!/usr/bin/env python3
"""Harvest a weak-winds sweep into one committed long-format CSV.

Usage (from repo root):
    python docs/dev/weak-winds/harness/harvest.py outputs/weak_winds_smoke \
        --out docs/dev/weak-winds/data/smoke_pair.csv

Reads every ``<run>/dictionary.jsonl`` under the sweep dir, tags rows with the
run name and its FB_thermCoeffWind (parsed from the run's .param copy), and
writes a single CSV with a provenance header (commit, command, date). Reads
only run outputs; writes only the --out file.
"""

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

FIELDS = [
    "run",
    "FB_thermCoeffWind",
    "t_now",
    "R2",
    "v2",
    "Eb",
    "T0",
    "current_phase",
    "F_ram_wind",
    "F_ram_SN",
    "F_HII",
    "F_rad",
    "F_grav",
    "P_HII",
    "P_ram",
]


def coeff_from_run(run_dir: Path) -> str:
    """Knob value actually used, preferring the run's own metadata record.

    metadata.json carries the resolved value even when the run never named
    the knob (schema default) or set it as a fixed base parameter, which the
    run-folder name does not encode.
    """
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
    return "1"  # default when the run never set the knob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # Recursive: a batched study nests one level deeper
    # (weak_winds_study/<batch>/<run>/) than a single flat sweep.
    jsonls = sorted(args.sweep_dir.rglob("dictionary.jsonl"))
    if not jsonls:
        print(f"no dictionary.jsonl under {args.sweep_dir}", file=sys.stderr)
        return 1

    commit = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(
            f"# weak-winds harvest | commit {commit} | {datetime.date.today()}\n"
            f"# command: python docs/dev/weak-winds/harness/harvest.py "
            f"{args.sweep_dir} --out {args.out}\n"
            f"# source runs: "
            f"{', '.join(p.parent.relative_to(args.sweep_dir).as_posix() for p in jsonls)}\n"
        )
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for jsonl in jsonls:
            # Relative path, so batch subdirectories stay distinguishable
            # (run folder names repeat across batches).
            run = jsonl.parent.relative_to(args.sweep_dir).as_posix()
            coeff = coeff_from_run(jsonl.parent)
            for line in jsonl.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["run"] = run
                row["FB_thermCoeffWind"] = coeff
                writer.writerow(row)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
