#!/usr/bin/env python3
"""Harvest per-phase P_HII identity metrics from finished TRINITY runs.

Implements the measurement half of `docs/dev/phii-identity/PLAN.md` batches 0
and 1. Reads each run's `dictionary.jsonl` + `metadata.json` and emits one CSV
row per (run x phase), reporting how close `P_HII` sits to the phase's confining
pressure and — once the Batch-1 shadow diagnostic exists — how hard the
Strömgren cap is biting.

The confining pressure is `Pb` in every phase. In the momentum phase that IS the
wind ram pressure: `run_momentum_phase.py:585` sets `params['Pb'] = pRam(...)`.
So one column (`relmax_PHII_vs_Pb`) covers every phase, and `P_ram` is carried
separately for the momentum cross-check.

Cap-binding is reported two ways, because the two answer different questions:
  * `frac_PHII_eq_Pb`  — how often the identity actually holds (<= IDENTITY_BAR).
    Available at base SHA. This is Batch 0's product.
  * `frac_cap_binding` / `blowup_*` — how often the raw Strömgren density exceeded
    the cap, and by how much. Needs `n_IF_Str_raw`, which only exists once the
    Batch-1 diagnostic lands; columns read `NA` before that. This is Batch 1's
    product and the input to the pre-registered C2a kill bar
    (p99 blowup > 1e2 in phase 1a/1b of any core config => C2a dead on arrival).

Usage (from the repo root):
    python docs/dev/phii-identity/harness/harvest_identity.py \
        --out docs/dev/phii-identity/data/b0_identity_grid.csv \
        outputs/b0/<run> [<run> ...]

Add --trajectories <path.csv> to also dump a thinned matched-grid trajectory per
run (Batch 0's `b0_trajectories.csv`).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

# relΔ bar for "the identity holds" — measured ceiling 3.6e-16 (PLAN §5) + margin.
IDENTITY_BAR = 5e-16
PHASES = ["energy", "implicit", "transition", "momentum"]

GRID_COLS = [
    "run",
    "phase",
    "n_rows",
    "t_first",
    "t_last",
    "Pb_max_over_min",
    "relmax_PHII_vs_Pb",
    "frac_PHII_eq_Pb",
    "relmax_PHII_vs_Pram",
    "frac_cap_binding",
    "blowup_max",
    "blowup_p99",
    "fate",
    "wall_s",
]
TRAJ_COLS = [
    "run",
    "t_now",
    "current_phase",
    "R2",
    "v2",
    "Eb",
    "Pb",
    "P_HII",
    "P_ram",
    "P_drive",
    "F_grav",
    "F_rad",
    "F_HII",
    "shell_mass",
    "n_IF_Str",
    "n_IF_Str_raw",
]


def _num(v):
    """Finite float, or None. Guards against bools, nulls and NaN/inf in the jsonl."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def read_rows(run_dir):
    rows = []
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    rows.sort(key=lambda d: d.get("t_now") or 0.0)
    return rows


def read_meta(run_dir):
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def _pct(sorted_vals, q):
    """Nearest-rank percentile; sorted_vals must be non-empty and sorted."""
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def analyse_phase(run_name, rows, phase, fate, wall_s):
    rows = [d for d in rows if d.get("current_phase") == phase]
    if not rows:
        return None
    pbs, rel_pb, rel_pram, blowups = [], [], [], []
    n_eq = n_cap = n_cap_seen = 0
    for d in rows:
        P_HII, Pb = _num(d.get("P_HII")), _num(d.get("Pb"))
        if P_HII is not None and Pb:
            pbs.append(Pb)
            rel = abs(P_HII - Pb) / abs(Pb)
            rel_pb.append(rel)
            n_eq += rel <= IDENTITY_BAR
        P_ram = _num(d.get("P_ram"))
        if P_HII is not None and P_ram:
            rel_pram.append(abs(P_HII - P_ram) / abs(P_ram))
        # Batch-1 shadow diagnostic: present only once the fix lands.
        raw, capped = _num(d.get("n_IF_Str_raw")), _num(d.get("n_IF_Str"))
        n0 = _num(d.get("shell_n0"))
        if raw is not None and n0:
            n_cap_seen += 1
            # The cap bit iff the raw value exceeded shell_n0 (i.e. the stored
            # n_IF_Str came from the min, not from the Strömgren balance).
            if raw > n0:
                n_cap += 1
                blowups.append(raw / n0)
            elif capped is not None and capped:
                blowups.append(raw / n0)
    if not rel_pb:
        return None
    blow_sorted = sorted(b for b in blowups if b > 0)
    na = "NA"
    return {
        "run": run_name,
        "phase": phase,
        "n_rows": len(rows),
        "t_first": f"{rows[0].get('t_now', 0.0):.6g}",
        "t_last": f"{rows[-1].get('t_now', 0.0):.6g}",
        "Pb_max_over_min": f"{max(pbs) / min(pbs):.6g}" if pbs and min(pbs) > 0 else na,
        "relmax_PHII_vs_Pb": f"{max(rel_pb):.3g}",
        "frac_PHII_eq_Pb": f"{n_eq / len(rel_pb):.4f}",
        "relmax_PHII_vs_Pram": f"{max(rel_pram):.3g}" if rel_pram else na,
        "frac_cap_binding": f"{n_cap / n_cap_seen:.4f}" if n_cap_seen else na,
        "blowup_max": f"{max(blow_sorted):.4g}" if blow_sorted else na,
        "blowup_p99": f"{_pct(blow_sorted, 0.99):.4g}" if blow_sorted else na,
        "fate": fate,
        "wall_s": wall_s,
    }


def thin(rows, n_max):
    if len(rows) <= n_max:
        return rows
    step = len(rows) / n_max
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n_max)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path)
    ap.add_argument(
        "--traj-max",
        type=int,
        default=400,
        help="max snapshots kept per run in the trajectory dump",
    )
    ap.add_argument(
        "--walltimes", type=Path, help="optional 'name,seconds' CSV to join into the wall_s column"
    )
    args = ap.parse_args()

    walls = {}
    if args.walltimes and args.walltimes.exists():
        for line in args.walltimes.read_text().splitlines():
            if "," in line and not line.startswith("#"):
                k, _, v = line.partition(",")
                walls[k.strip()] = v.strip()

    grid, traj = [], []
    for run_dir in args.runs:
        if not (run_dir / "dictionary.jsonl").exists():
            print(f"skip {run_dir} — no dictionary.jsonl")
            continue
        rows = read_rows(run_dir)
        meta = read_meta(run_dir)
        term = meta.get("termination") or {}
        fate = term.get("outcome") or term.get("reason") or "NA"
        wall = walls.get(run_dir.name, "NA")
        for phase in PHASES:
            r = analyse_phase(run_dir.name, rows, phase, fate, wall)
            if r:
                grid.append(r)
        if args.trajectories:
            for d in thin(rows, args.traj_max):
                traj.append({c: d.get(c, "") if c != "run" else run_dir.name for c in TRAJ_COLS})

    if not grid:
        print("no usable rows found")
        return 1

    w = max(len(r["run"]) for r in grid)
    print(
        f"{'run':{w}} {'phase':>11} {'rows':>5} {'relmax P_HII/Pb':>16} "
        f"{'frac==':>7} {'capbind':>8} {'blowup p99':>11}"
    )
    for r in grid:
        print(
            f"{r['run']:{w}} {r['phase']:>11} {r['n_rows']:>5} "
            f"{r['relmax_PHII_vs_Pb']:>16} {r['frac_PHII_eq_Pb']:>7} "
            f"{r['frac_cap_binding']:>8} {r['blowup_p99']:>11}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Per-(run,phase) P_HII identity metrics. See docs/dev/phii-identity/PLAN.md.\n")
        fh.write(f"# frac_PHII_eq_Pb: fraction of rows with |P_HII-Pb|/Pb <= {IDENTITY_BAR:g}.\n")
        fh.write(
            "# frac_cap_binding/blowup_*: need n_IF_Str_raw (Batch 1); 'NA' before it lands.\n"
        )
        wr = csv.DictWriter(fh, fieldnames=GRID_COLS)
        wr.writeheader()
        wr.writerows(grid)
    print(f"\nwrote {args.out}")

    if args.trajectories and traj:
        args.trajectories.parent.mkdir(parents=True, exist_ok=True)
        with args.trajectories.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(f"# Thinned trajectories (<={args.traj_max} snapshots/run).\n")
            wr = csv.DictWriter(fh, fieldnames=TRAJ_COLS)
            wr.writeheader()
            wr.writerows(traj)
        print(f"wrote {args.trajectories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
