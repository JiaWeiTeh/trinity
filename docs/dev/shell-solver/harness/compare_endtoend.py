#!/usr/bin/env python3
"""Compare a variant run's dictionary.jsonl to the baseline run's: max relative
diff of the FINAL row and of the key science TRAJECTORY columns.

Usage:
    python docs/dev/shell-solver/harness/compare_endtoend.py <baseline_jsonl> <variant_jsonl>
Prints a JSON line: endtoend_final_maxrel, endtoend_traj_maxrel, plus a per-column
breakdown (final + trajectory) so a worst-offending column is identifiable.

Science columns compared (the quantities the science question cares about):
  t_now   -- simulation time
  R2, rShell, R_IF -- shell radii
  v2, v_mech_total -- velocities
  Eb      -- bubble energy
  Lbol, Li, Ln -- luminosities
  n_IF    -- ionization-front density
  shell_mass -- swept shell mass
Trajectory diff is over the common-length prefix (min of the two row counts),
element-wise; final diff is the last row of each (independent of length).
"""
import sys
import json

SCI_COLS = [
    "t_now", "R2", "rShell", "R_IF", "v2", "v_mech_total",
    "Eb", "Lbol", "Li", "Ln", "n_IF", "shell_mass",
]


def _load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _relemax(a, b):
    """Max relative diff over two equal-length scalar sequences; robust to 0/None/nan."""
    worst = 0.0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        try:
            x = float(x); y = float(y)
        except (TypeError, ValueError):
            continue
        if x != x or y != y:  # nan
            continue
        denom = max(abs(x), 1e-300)
        if abs(x) < 1e-300 and abs(y) < 1e-300:
            rel = 0.0
        else:
            rel = abs(x - y) / denom
        worst = max(worst, rel)
    return worst


def main():
    base = _load(sys.argv[1])
    var = _load(sys.argv[2])
    n = min(len(base), len(var))

    final_per = {}
    traj_per = {}
    bf, vf = base[-1], var[-1]
    for c in SCI_COLS:
        if c not in bf or c not in vf:
            continue
        final_per[c] = _relemax([bf[c]], [vf[c]])
        bcol = [r.get(c) for r in base[:n]]
        vcol = [r.get(c) for r in var[:n]]
        traj_per[c] = _relemax(bcol, vcol)

    out = {
        "n_rows_baseline": len(base),
        "n_rows_variant": len(var),
        "endtoend_final_maxrel": max(final_per.values()) if final_per else None,
        "endtoend_traj_maxrel": max(traj_per.values()) if traj_per else None,
        "final_worst_col": max(final_per, key=final_per.get) if final_per else None,
        "traj_worst_col": max(traj_per, key=traj_per.get) if traj_per else None,
        "final_per_col": {k: f"{v:.3e}" for k, v in final_per.items()},
        "traj_per_col": {k: f"{v:.3e}" for k, v in traj_per.items()},
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
