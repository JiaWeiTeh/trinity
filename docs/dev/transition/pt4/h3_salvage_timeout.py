#!/usr/bin/env python3
"""On a TIMEOUT (grind), the run was killed before h3_run_variant.py could write
its CSV row / trajectory. Salvage both from the partial dictionary.jsonl that
trinity streamed to disk during the run, and append an eval row marked
outcome=timeout so the grind is recorded (not silently dropped).

Usage: python h3_salvage_timeout.py --config C --variant V --out DIR \
           --csv h3_eval.csv --traj traj.csv --floor F --timeout_s N
"""
import argparse
import csv
import json
import os
from pathlib import Path

TRIGGER = 0.05


def _ratio(r):
    lg, ll = r.get("bubble_Lgain"), r.get("bubble_Lloss")
    try:
        if lg is None or ll is None or lg != lg or ll != ll or lg <= 0:
            return None
        return (lg - ll) / lg
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--timeout_s", type=float, default=None)
    args = ap.parse_args()

    rows = []
    hits = list(Path(args.out).rglob("dictionary.jsonl"))
    if hits:
        jsonl = str(max(hits, key=lambda p: p.stat().st_mtime))
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda d: d.get("t_now", 0.0))

    phases = {r.get("current_phase") for r in rows}
    order = {"energy": 1, "implicit": 2, "transition": 3, "momentum": 4}
    drank = max([order.get(p, 0) for p in phases], default=0)
    reached = {0: "", 1: "1a", 2: "1b", 3: "1c", 4: "momentum"}[drank]

    last = rows[-1] if rows else {}
    E0 = rows[0].get("Eb") if rows else None
    ratios = [x for x in (_ratio(r) for r in rows) if x is not None]
    ratio_min = min(ratios) if ratios else None
    trigger_fired = any(x < TRIGGER for x in ratios) if ratios else False
    n_impl = sum(1 for r in rows if r.get("current_phase") == "implicit")

    row = {
        "config": args.config, "variant": args.variant,
        "crashed": False, "crash_phase": "", "crash_excpt": "",
        "end_reason": f"TIMEOUT after {args.timeout_s}s (grind; {n_impl} implicit rows)",
        "end_code": "timeout", "reached_phase": reached,
        "final_t": last.get("t_now"), "final_R2": last.get("R2"),
        "final_v2": last.get("v2"), "final_Eb": last.get("Eb"),
        "E0": E0, "floor": args.floor, "floor_activated": "unknown(killed)",
        "act_drive": "", "act_state": "", "min_Eb_seen": "",
        "trigger_fired": trigger_fired, "ratio_min": ratio_min,
        "n_rows": len(rows), "runtime_s": args.timeout_s,
    }
    write_header = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    if args.traj and rows:
        cols = ["t_now", "phase", "R2", "v2", "Eb", "Pb", "R1", "T0",
                "bubble_Lgain", "bubble_Lloss", "ratio", "rCloud"]
        os.makedirs(os.path.dirname(os.path.abspath(args.traj)), exist_ok=True)
        with open(args.traj, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({
                    "t_now": r.get("t_now"), "phase": r.get("current_phase"),
                    "R2": r.get("R2"), "v2": r.get("v2"), "Eb": r.get("Eb"),
                    "Pb": r.get("Pb"), "R1": r.get("R1"), "T0": r.get("T0"),
                    "bubble_Lgain": r.get("bubble_Lgain"),
                    "bubble_Lloss": r.get("bubble_Lloss"),
                    "ratio": _ratio(r), "rCloud": r.get("rCloud"),
                })
    print(f"[salvage {args.variant} {args.config}] timeout rows={len(rows)} "
          f"impl={n_impl} reached={reached} final_t={row['final_t']}")


if __name__ == "__main__":
    main()
