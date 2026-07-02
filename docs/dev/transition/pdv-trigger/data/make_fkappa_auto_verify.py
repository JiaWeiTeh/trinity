#!/usr/bin/env python3
"""Acceptance check for cooling_boost_kappa='auto' (pt3) -> fkappa_auto_verify.csv.

Reduces the fkauto_verify run (runs/params/fkauto_verify.param: a Lancaster-like
1e5 Msun GMC at nCore=1e3, sfe=0.03, with cooling_boost_kappa=auto) using the SAME
streaming reducer as the 819-run sweep, and checks the pt3 contract:

  1. 'auto' resolved to the sweep-measured f_kappa_fire = 12 (metadata.json);
  2. the cooling_balance trigger fired (theta = Lloss/Lgain crossed 0.95);
  3. the run left the energy phase and reached the momentum chain.

REPRODUCE:
    python run.py docs/dev/transition/pdv-trigger/runs/params/fkauto_verify.param
    python docs/dev/transition/pdv-trigger/data/make_fkappa_auto_verify.py [run_dir]
"""

import csv
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, *[".."] * 5))
sys.path.insert(0, _HERE)
from reduce_fkappa_sweep import reduce_run  # noqa: E402  (sibling module)

_EXPECT_FK = 12.0  # fkappa_nH_sweep.csv, cell (1e5, 0.03, 1e3)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_REPO, "outputs", "fkauto_verify")
    row = reduce_run(Path(run_dir) / "dictionary.jsonl")
    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    row["cooling_boost_kappa"] = meta.get("cooling_boost_kappa")
    row["mCloud"], row["sfe"], row["nCore"] = 1e5, 0.03, 1e3  # folder name carries no axes

    checks = {
        "auto_resolved_to_measured": abs(row["cooling_boost_kappa"] - _EXPECT_FK) < 1e-6,
        "cooling_fired": bool(row["cooling_fired"]),
        "reached_momentum": bool(row["reached_momentum"]),
        "theta_crossed_trigger": row["theta_max"] is not None and row["theta_max"] >= 0.95,
    }
    for k, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {k}")
    print(
        f"theta_max={row['theta_max']:.3f}  phase_final={row['phase_final']}  "
        f"t_final={row['t_final']:.3f}  f_kappa={row['cooling_boost_kappa']:.3f}"
    )

    out = os.path.join(_HERE, "fkappa_auto_verify.csv")
    cols = [
        "run_name",
        "mCloud",
        "sfe",
        "nCore",
        "cooling_boost_kappa",
        "rCloud",
        "n_impl",
        "t_final",
        "phase_final",
        "theta_blowout",
        "theta_max",
        "blowout_t",
        "reached_momentum",
        "cooling_fired",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerow({k: row.get(k) for k in cols})
    print(f"wrote {out}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
