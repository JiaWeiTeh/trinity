#!/usr/bin/env python3
"""Kappa stability map — resolves the FINDINGS §8e ⇄ §9 tension from the committed sweep.

§8e (pt2) found `cooling_boost_kappa=8` breaks the β-δ solver on simple_cluster (θ frozen ~0.53,
never fires). §9 (pt3) reported 57/57 sweep cells firing at their measured f_κ_fire ≤ 64. Both are
true, and BOTH are in the committed `summary.csv`: the sweep's simple_cluster-analog cell
(mCloud 1e5, sfe 0.3, nCore 1e5) fires at f_κ=4–6, then at f_κ=8/12 the run freezes mid-implicit
with θ_max = 0.5331 — §8e's ~0.53, reproduced on Helix. The knob's failure is NON-MONOTONIC in
f_κ: firing bands interleave with breakdown windows.

This script reads ONLY `summary.csv` (819 rows) and writes the per-cell stability map:
f_κ_fire, every no-fire f above it, and which of those froze early (t_final < stop_t, still
implicit — the §8e failure signature). Headline stats are printed and embedded in the CSV header.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_kappa_stability_map.py
Deliverable:
    docs/dev/transition/pdv-trigger/data/kappa_stability_map.csv
"""

import csv
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _stamp import stamp  # noqa: E402  (workstream provenance stamp)

STOP_T = 2.0  # the sweep's stop_t; a no-fire run ending well short of it died/froze mid-implicit
EARLY = 0.95 * STOP_T


def main():
    with open(os.path.join(_HERE, "summary.csv")) as fh:
        rows = list(csv.DictReader(fh))

    cells = defaultdict(list)
    for r in rows:
        key = (float(r["mCloud"]), float(r["sfe"]), float(r["nCore"]))
        t_final = float(r["t_final"]) if r["t_final"] else None
        cells[key].append(
            (
                float(r["cooling_boost_kappa"]),
                r["cooling_fired"] == "True",
                t_final,
                r["phase_final"],
                float(r["theta_max"]) if r["theta_max"] else None,
            )
        )

    out_rows = []
    n_fired = n_nonmono = 0
    for key, fs in sorted(cells.items()):
        fs.sort()
        fired = [f for f, ok, t, p, th in fs if ok]
        if not fired:
            continue
        n_fired += 1
        f_fire = min(fired)
        dead_above = [(f, t, p, th) for f, ok, t, p, th in fs if f > f_fire and not ok]
        froze_above = [
            f for f, t, p, th in dead_above if t is not None and t < EARLY and p == "implicit"
        ]
        if dead_above:
            n_nonmono += 1
        mC, sfe, nC = key
        out_rows.append(
            {
                "mCloud": mC,
                "sfe": sfe,
                "nCore": nC,
                "f_kappa_fire": f_fire,
                "f_fired_all": ";".join(str(f) for f in fired),
                "f_nofire_above_fire": ";".join(str(f) for f, t, p, th in dead_above),
                "f_froze_early": ";".join(str(f) for f in froze_above),
                "nonmonotonic": bool(dead_above),
            }
        )

    froze_runs = [
        r
        for r in rows
        if r["cooling_fired"] != "True"
        and r["phase_final"] == "implicit"
        and r["t_final"]
        and float(r["t_final"]) < EARLY
    ]

    header = (
        f"# kappa stability map from summary.csv ({len(rows)} runs): "
        f"{n_fired} cells fired; {n_nonmono} NON-MONOTONIC (some f > f_fire did not fire); "
        f"{len(froze_runs)} runs froze mid-implicit without firing (the sec8e signature). "
        f"sec8e reproduced on Helix: cell (1e5, 0.3, 1e5) fires at 4-6, freezes at 8/12 "
        f"(theta_max 0.5331 = sec8e's ~0.53)."
    )
    out_path = os.path.join(_HERE, "kappa_stability_map.csv")
    stamp_line = stamp(__file__)  # BEFORE opening: writing the output dirties the tree
    with open(out_path, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        fh.write(header + "\n")
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(header)
    print(f"wrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
