#!/usr/bin/env python3
"""📏 Standard-protocol harvest: theta_max over the whole run, from dictionary.jsonl.

This is THE sanctioned theta measurement (PLAN.md 📏 rule 2 + retraction R6):
  theta(t) = bubble_Lloss / Lmech_total on ACCEPTED implicit-phase snapshot rows
  (bubble_Lloss = the effective/boosted loss the transition trigger itself sees;
  falls back to bubble_LTotal for runs predating that key). Reported per run:
  theta_max, t at theta_max, theta_1 proxy (theta at the first implicit row),
  final time/phase, and the metadata termination — NEVER theta-at-blowout, and
  NEVER a call-level observer (solver trial points contaminate those; R6).

Usage:
    python harvest_theta_max.py outputs/theta5/* [--csv runs/data/theta5_summary.csv]
Each argument is a run dir containing dictionary.jsonl (+ metadata.json).
"""

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _stamp import stamp  # noqa: E402  (workstream provenance stamp)

COLUMNS = [
    "run_name",
    "theta_max",
    "t_at_theta_max",
    "theta_first",
    "n_impl",
    "t_final",
    "phase_final",
    "reached_momentum",
    "fired_cooling_balance",
    "outcome",
    "detail",
]


def num(d, k):
    v = d.get(k)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    # dictionary.jsonl can carry NaN on solver-breakdown rows (e.g. the dense edge
    # under boost) — a NaN theta would poison max(); treat as missing
    return v if math.isfinite(v) else None


def harvest(run_dir: Path) -> dict:
    row = {"run_name": run_dir.name}
    theta_max = t_at_max = theta_first = t_final = None
    phase_final = None
    n_impl = 0
    reached_momentum = False
    with (run_dir / "dictionary.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            phase_final = d.get("current_phase", phase_final)
            t_final = num(d, "t_now") if num(d, "t_now") is not None else t_final
            if d.get("current_phase") in ("transition", "momentum"):
                reached_momentum = True
            if d.get("current_phase") != "implicit":
                continue
            Lloss = num(d, "bubble_Lloss")
            if Lloss is None:
                Lloss = num(d, "bubble_LTotal")
            Lmech = num(d, "Lmech_total")
            if Lloss is None or not Lmech:
                continue
            n_impl += 1
            theta = Lloss / Lmech
            if theta_first is None:
                theta_first = theta
            if theta_max is None or theta > theta_max:
                theta_max, t_at_max = theta, num(d, "t_now")

    outcome = detail = None
    meta_fired = False
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        term = json.loads(meta_path.read_text()).get("termination") or {}
        outcome, detail = term.get("outcome"), term.get("detail")
        meta_fired = "cooling_balance" in str(term)
    # metadata only carries the FINAL termination — a run that fired and then ran on in the
    # momentum phase to stop_t ends as 'stopping_time'. Infer firing the way the proven sweep
    # reducer does: it left the energy phase AND theta crossed the trigger (default trigger is
    # cooling_balance-only, so nothing else exits the phase upward; Eb<=0 handoffs have theta<0.95).
    fired = meta_fired or (reached_momentum and theta_max is not None and theta_max >= 0.95)
    row.update(
        {
            "theta_max": theta_max,
            "t_at_theta_max": t_at_max,
            "theta_first": theta_first,
            "n_impl": n_impl,
            "t_final": t_final,
            "phase_final": phase_final,
            "reached_momentum": reached_momentum,
            "fired_cooling_balance": fired,
            "outcome": outcome,
            "detail": detail,
        }
    )
    return row


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    csv_out = None
    if "--csv" in argv:
        csv_out = Path(argv[argv.index("--csv") + 1])
        args = [a for a in args if str(csv_out) != a]
    rows = []
    for a in args:
        run_dir = Path(a)
        if not (run_dir / "dictionary.jsonl").exists():
            print(f"skip (no dictionary.jsonl): {run_dir}", file=sys.stderr)
            continue
        rows.append(harvest(run_dir))
    rows.sort(key=lambda r: r["run_name"])
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        stamp_line = stamp(__file__)  # BEFORE opening: writing the output dirties the tree
        with csv_out.open("w", newline="") as fh:
            fh.write(stamp_line + "\n")
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {csv_out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
