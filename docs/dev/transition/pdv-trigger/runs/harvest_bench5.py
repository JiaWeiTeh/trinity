#!/usr/bin/env python3
"""bench5 harvest — fire-map summary + a compact, restart-durable θ(t) trajectory per arm.

Two outputs, both committed so the whole Phase-5 analysis (Θ_cum band, 1−θ dex metric, Lcool/Lleak
channel split, El-Badry overlay) runs OFFLINE from git — never re-running the sims (the theta5s
lesson: its raw arms were lost to a /tmp wipe and dMdt had to be salvaged in a scramble).

1. --csv <summary>: the fire map, via harvest_theta_max.harvest (θ_max, fired?, fate, t_final) —
   the same sanctioned θ = bubble_Lloss/Lmech_total on accepted rows.
2. --traj-dir <dir>: per arm, <arm>.csv with the accepted-implicit trajectory
   (t_now, theta, Lcool=bubble_LTotal, Lleak=bubble_Leak, Lmech=Lmech_total, R2). ALL accepted rows
   are kept (trapezoid Θ_cum needs them) up to a 4000-row cap; beyond that, log-t downsample keeping
   endpoints. θ numerator uses bubble_Lloss (the effective/boosted loss the trigger sees) =
   Lcool + Lleak; committing the split lets the Rogers & Pittard channel check run offline.

    python harvest_bench5.py <arm_dirs...> --csv runs/data/bench5_summary.csv \
        --traj-dir runs/data/bench5_traj
    # extra dictionary.jsonl columns (state-coupled f_A screen, FA_STATE_COUPLED.md SC-0):
    python harvest_bench5.py <arm_dirs...> --traj-dir <dir> \
        --extra-cols Pb,bubble_L2Conduction,bubble_L3Intermediate,bubble_dMdt
"""
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from harvest_theta_max import COLUMNS, harvest  # noqa: E402
from _stamp import stamp  # noqa: E402

TRAJ_COLS = ["t_now", "theta", "Lcool", "Lleak", "Lmech", "R2"]
TRAJ_CAP = 4000


def _finite(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def trajectory(run_dir, extra=()):
    """Accepted implicit rows as [t_now, theta, Lcool, Lleak, Lmech, R2] + any ``extra`` keys.

    ``extra`` names dictionary.jsonl keys appended verbatim (finite-filtered) after R2 — used by
    the state-coupled-f_A screen, which needs Pb and the L2/L3 split that the six default columns
    do not carry (FA_STATE_COUPLED.md SC-0). Empty ``extra`` reproduces the original 6 columns.
    """
    out = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("current_phase") != "implicit":
                continue
            t = _finite(d.get("t_now"))
            Lloss = _finite(d.get("bubble_Lloss"))
            if Lloss is None:
                Lloss = _finite(d.get("bubble_LTotal"))
            Lmech = _finite(d.get("Lmech_total"))
            if t is None or Lloss is None or not Lmech:
                continue
            Lcool = _finite(d.get("bubble_LTotal"))
            Lleak = _finite(d.get("bubble_Leak")) or 0.0
            row = [t, Lloss / Lmech, Lcool, Lleak, Lmech, _finite(d.get("R2"))]
            row += [_finite(d.get(k)) for k in extra]
            out.append(row)
    out.sort(key=lambda r: r[0])
    if len(out) > TRAJ_CAP:                     # log-t downsample, keep endpoints
        import numpy as np
        ts = np.array([r[0] for r in out])
        lo = max(ts[0], 1e-9)
        grid = np.unique(np.geomspace(lo, ts[-1], TRAJ_CAP))
        idx = sorted({0, len(out) - 1} | {int(np.searchsorted(ts, g)) for g in grid})
        out = [out[min(i, len(out) - 1)] for i in idx]
    return out


def write_traj(run_dir, traj_dir, extra=()):
    rows = trajectory(run_dir, extra)
    if not rows:
        return 0
    traj_dir.mkdir(parents=True, exist_ok=True)
    with (traj_dir / f"{run_dir.name}.csv").open("w", newline="") as fh:
        # Stamped like every other artifact (maintainer ALL-FRESH ruling 2026-07-29): the trajectory
        # CSVs ARE the data the Theta_cum metric reads, so "when was this measured" has to be legible
        # on the file itself, not inferred from the summary next to it. Readers already skip leading
        # '#' lines (make_bench5_analysis._read_csv), and the campaign hash is taken over the
        # non-comment lines (sync_bench.sh), so the stamp cannot make two identical runs differ.
        fh.write(stamp(__file__) + "\n")
        w = csv.writer(fh)
        w.writerow(TRAJ_COLS + list(extra))
        w.writerows(rows)
    return len(rows)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    csv_out = traj_dir = None
    if "--csv" in argv:
        csv_out = Path(argv[argv.index("--csv") + 1])
        args = [a for a in args if str(csv_out) != a]
    if "--traj-dir" in argv:
        traj_dir = Path(argv[argv.index("--traj-dir") + 1])
        args = [a for a in args if str(traj_dir) != a]
    extra = ()
    if "--extra-cols" in argv:
        spec = argv[argv.index("--extra-cols") + 1]
        extra = tuple(c.strip() for c in spec.split(",") if c.strip())
        args = [a for a in args if spec != a]

    rows = []
    for a in args:
        run_dir = Path(a)
        if not (run_dir / "dictionary.jsonl").exists():
            continue
        rows.append(harvest(run_dir))
        if traj_dir is not None:
            write_traj(run_dir, traj_dir, extra)
    rows.sort(key=lambda r: r["run_name"])
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        stamp_line = stamp(str(HERE / "harvest_bench5.py"))
        with csv_out.open("w", newline="") as fh:
            fh.write(stamp_line + "\n")
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} summary rows -> {csv_out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
