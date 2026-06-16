#!/usr/bin/env python3
"""Extract per-run expansion trajectories from the committed run outputs.

Reads outputs/probe_steep_{legacy,hybr}/dictionary.jsonl and writes the two
committed CSVs make_rootmap_gif.py reads for panels C and F:
    steep_legacy_traj.csv   the real CAGED run (cage clamps beta,delta)
    steep_hybr_traj.csv     the real HYBR run (free beta,delta)
Columns: t,R2,v2,rShell,R_IF.

The first records are the ultra-fast free-expansion transient (v2 ~ 1e3 pc/Myr at
R2 ~ 1e-3 pc); they're trimmed (R2 < R2_MIN) so the v2 axis isn't dominated by the
spike -- both runs agree there anyway. The kept range covers the deceleration the
two runs share AND the divergence: the caged run peaks at R2~2.1 pc (t~0.26 Myr)
then recollapses (v2<0) to R2~1.5 pc by t~0.44 Myr, while hybr expands to R2~37 pc.

Needs only the committed outputs + numpy (no pinned venv). Run once after a run;
the CSVs it writes are force-added to git so the gif reproduces without the outputs.

  python docs/dev/scratch/betadelta-diagnostics/extract_traj.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R2_MIN = 0.2  # pc -- trim the free-expansion spike below this
COLS = ("t", "R2", "v2", "rShell", "R_IF")
RUNS = {
    "steep_legacy_traj.csv": "probe_steep_legacy",
    "steep_hybr_traj.csv": "probe_steep_hybr",
}


def extract(run_dir):
    rows = []
    with open(run_dir / "dictionary.jsonl") as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["t_now"], d["R2"], d["v2"], d["rShell"], d["R_IF"]))
    a = np.array(rows, float)
    return a[a[:, 1] >= R2_MIN]


def main():
    for csv_name, run in RUNS.items():
        run_dir = ROOT / "outputs" / run
        if not (run_dir / "dictionary.jsonl").exists():
            raise SystemExit(f"missing {run_dir}/dictionary.jsonl -- run the sim first")
        a = extract(run_dir)
        out = HERE / csv_name
        np.savetxt(out, a, delimiter=",", header=",".join(COLS), comments="", fmt="%.8g")
        print(
            f"wrote {out.name}: {len(a)} rows  "
            f"t->{a[:,0].max():.3g}  R2 peak={a[:,1].max():.3g}  final R2={a[-1,1]:.3g}"
        )


if __name__ == "__main__":
    main()
