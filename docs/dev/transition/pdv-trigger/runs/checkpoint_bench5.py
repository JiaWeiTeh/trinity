#!/usr/bin/env python3
"""Merge-checkpoint the bench5 summary + trajectory dir across container restarts.

Same role as checkpoint_theta5s.py: the in-container runner writes to /tmp (wiped on restart) while
the committed runs/data/bench5_summary.csv + runs/data/bench5_traj/ survive in git. This MERGES the
current container's completed arms into both (summary: union by run_name, prefer compliant; traj:
refresh each arm's committed CSV). Prints the arm count so the autocommitter can decide to commit.

    python runs/checkpoint_bench5.py --out $WS/bench5_out \
        --summary runs/data/bench5_summary.csv --traj-dir runs/data/bench5_traj
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _compliant(row):
    try:
        return (float(row.get("t_final") or 0) >= 5.0
                or row.get("phase_final") not in (None, "", "implicit", "energy"))
    except (ValueError, TypeError):
        return False


def _read(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return {r["run_name"]: r
                for r in csv.DictReader(x for x in fh if not x.lstrip().startswith("#"))}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--traj-dir", required=True)
    args = ap.parse_args(argv)

    merged = _read(args.summary)
    dirs = [str(d) for d in Path(args.out).glob("*/") if (d / ".exit_code").exists()]
    if dirs:
        tmp = Path(args.out) / "_harvest.csv"
        # harvest current arms straight into the committed traj-dir + a temp summary, then merge.
        subprocess.run([sys.executable, str(HERE / "harvest_bench5.py"), *dirs,
                        "--csv", str(tmp), "--traj-dir", args.traj_dir],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for name, row in _read(str(tmp)).items():
            if name not in merged or (_compliant(row) and not _compliant(merged[name])):
                merged[name] = row

    from harvest_theta_max import COLUMNS
    rows = [merged[k] for k in sorted(merged)]
    with open(args.summary, "w", newline="") as fh:
        fh.write("# bench5 cumulative summary (Phase 5, Lancaster 2021b calibration), merged across "
                 "container restarts by checkpoint_bench5.py. theta = bubble_Lloss/Lmech_total on "
                 "accepted rows; per-arm trajectory in runs/data/bench5_traj/.\n")
        fh.write("# ⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC (HPC was down 2026-07-12). In-container-"
                 "vs-HPC numerical fidelity unverified; re-confirm on HPC before any paper number. "
                 "FIRE = actually fired cooling_balance (see fired_cooling_balance), stricter than "
                 "theta_max>=0.95. Production arms fire/fate; *_diag arms (transition_trigger=blowout) "
                 "give uncensored theta(t).\n")
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    n_comp = sum(1 for r in rows if _compliant(r))
    print(f"{len(rows)} arms in summary ({n_comp} compliant)")


if __name__ == "__main__":
    main(sys.argv[1:])
