#!/usr/bin/env python3
"""Merge-checkpoint the theta5s summary across container restarts.

The in-container runner writes outputs to /tmp, which is WIPED on every container restart, while
the committed runs/data/theta5s_summary.csv survives in git. This helper MERGES the arms completed
in the current /tmp output dir into the committed cumulative summary (union by run_name, preferring
a COMPLIANT row over a non-compliant one), so repeated restarts accumulate results instead of
overwriting them. Print the arm count so a caller can decide whether to commit.

    python runs/checkpoint_theta5s.py --out $SP/t5s_out --summary runs/data/theta5s_summary.csv
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = ["run_name", "theta_max", "t_at_theta_max", "theta_first", "n_impl", "t_final",
        "phase_final", "reached_momentum", "fired_cooling_balance", "outcome", "detail"]


def _compliant(row):
    try:
        return float(row.get("t_final") or 0) >= 5.0 or row.get("phase_final") not in (None, "", "implicit")
    except (ValueError, TypeError):
        return False


def _read(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r["run_name"]: r for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#"))}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args(argv)

    merged = _read(args.summary)  # committed rows survive restarts
    # harvest the current container's completed arms to a temp CSV, then merge
    dirs = [str(d) for d in Path(args.out).glob("*/") if (d / ".exit_code").exists()]
    if dirs:
        tmp = Path(args.out) / "_harvest.csv"
        subprocess.run([sys.executable, str(HERE / "harvest_theta_max.py"), *dirs, "--csv", str(tmp)],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for name, row in _read(str(tmp)).items():
            # prefer a compliant row; otherwise keep whichever exists
            if name not in merged or (_compliant(row) and not _compliant(merged[name])):
                merged[name] = row

    rows = [merged[k] for k in sorted(merged)]
    with open(args.summary, "w", newline="") as fh:
        fh.write("# theta5s cumulative summary (merged across container restarts by "
                 "checkpoint_theta5s.py). theta from bubble_Lloss/Lmech_total on accepted rows.\n")
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_comp = sum(1 for r in rows if _compliant(r))
    print(f"{len(rows)} arms in summary ({n_comp} compliant)")


if __name__ == "__main__":
    main(sys.argv[1:])
