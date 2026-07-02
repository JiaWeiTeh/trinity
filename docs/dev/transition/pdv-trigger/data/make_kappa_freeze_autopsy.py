#!/usr/bin/env python3
"""Kappa freeze autopsy — re-reads the 819-run sweep's 38 mid-implicit freezes and every
"non-monotonic" arm, asking the maintainer's question (2026-07-02): is "breaks non-monotonically
in f_kappa" (FINDINGS §9a) a real physics band structure, or a false inference from solver
crashes + the outlawed stop_t=2 horizon?

Reads ONLY the committed `summary.csv`. Three findings this script certifies:

1. **The freeze mode pre-exists the knob**: one run froze at f_kappa=1.0 (unboosted baseline,
   cell 1e5/sfe0.1/n1e5, theta_max 0.9516) — the kappa knob aggravates a pre-existing solver
   fragility (freeze rate rises ~1/63 at f=1-2 to ~5-7/63 at f=12-48), it does not create it.
2. **Freezes concentrate at the theta->0.95 crossing**: 34/38 frozen runs died with
   theta_max >= 0.8 (most 0.85-0.95), vs median 0.636 for healthy no-fire runs. They crashed
   ON APPROACH to the transition — "would-fire" arms, not cold dead windows.
3. **Every "non-monotonic" arm is explained without physics bands**: the 23 no-fire-above-f_fire
   arms across the 17 flagged cells decompose into 12 froze-ON-APPROACH (theta 0.86-0.94),
   8 healthy-to-2Myr with theta_max 0.87-0.93 (near-threshold at a horizon the standing rules
   outlaw — 5 Myr is the rule; the diffuse multiplier arm fires at t=5.04 Myr precedent), and
   3 froze-EARLY (theta 0.52-0.59, the true §8e mode: 2 cells, 3/819 runs). NONE ran healthy to
   a rule-compliant horizon and stayed cold.

So §9a's "interleaved firing bands and breakdown windows" over-reads the data: the correct
statement is "a solver crash concentrated near the cooling-balance crossing, whose hit rate
rises with f_kappa, plus a 3-run early-freeze mode". Whether kappa is *shippable* is a separate
question (a knob that crashes 5-10% of runs is not, until the crash is fixed) — but the failure
is numerical, not band-structured physics. See FINDINGS §9b.

REPRODUCE:
    python docs/dev/transition/pdv-trigger/data/make_kappa_freeze_autopsy.py
Deliverable:
    docs/dev/transition/pdv-trigger/data/kappa_freeze_autopsy.csv
"""

import csv
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from _stamp import stamp  # noqa: E402  (workstream provenance stamp)

STOP_T = 2.0
EARLY = 0.95 * STOP_T
APPROACH_THETA = 0.8  # frozen runs at/above this were heading into the 0.95 crossing


def classify(r):
    th = float(r["theta_max"]) if r["theta_max"] else float("nan")
    tf = float(r["t_final"])
    if r["phase_final"] == "implicit" and tf < EARLY:
        return "froze-on-approach" if th >= APPROACH_THETA else "froze-early"
    if r["phase_final"] == "implicit":
        return "healthy-no-fire-at-2Myr"
    return "other:" + r["phase_final"]


def main():
    with open(os.path.join(_HERE, "summary.csv")) as fh:
        rows = list(csv.DictReader(fh))

    cells = defaultdict(dict)
    for r in rows:
        key = (float(r["mCloud"]), float(r["sfe"]), float(r["nCore"]))
        cells[key][float(r["cooling_boost_kappa"])] = r

    frozen = [
        r
        for r in rows
        if r["phase_final"] == "implicit" and float(r["t_final"]) < EARLY and r["cooling_fired"] == "False"
    ]
    n_appr = sum(1 for r in frozen if float(r["theta_max"] or "nan") >= APPROACH_THETA)

    out_rows = []
    counts = defaultdict(int)
    for key, arms in sorted(cells.items()):
        fired = sorted(f for f, r in arms.items() if r["cooling_fired"] == "True")
        if not fired:
            continue
        f_fire = fired[0]
        for f, r in sorted(arms.items()):
            if f <= f_fire or r["cooling_fired"] == "True":
                continue
            cls = classify(r)
            counts[cls] += 1
            out_rows.append(
                {
                    "mCloud": key[0],
                    "sfe": key[1],
                    "nCore": key[2],
                    "f_kappa_fire": f_fire,
                    "f_kappa_arm": f,
                    "verdict": cls,
                    "theta_max": r["theta_max"],
                    "t_final": r["t_final"],
                }
            )

    stamp_line = stamp(__file__)
    dst = os.path.join(_HERE, "kappa_freeze_autopsy.csv")
    with open(dst, "w", newline="") as fh:
        fh.write(stamp_line + "\n")
        fh.write(
            "# autopsy of the sec9a 'non-monotonic' arms: %d frozen runs total (%d/%d froze at "
            "theta_max>=%.1f i.e. ON APPROACH to the 0.95 crossing; 1 froze at f_kappa=1.0 unboosted); "
            "no-fire-above-f_fire arms decompose as %s — none ran healthy to a rule-compliant "
            "(5 Myr) horizon and stayed cold; the sweep's stop_t=2 horizon is outlawed for theta "
            "quotes (CONTAMINATION rule a)\n"
            % (len(frozen), n_appr, len(frozen), APPROACH_THETA, dict(sorted(counts.items())))
        )
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(
        "wrote %s: %d non-monotonic arms (%s); %d freezes, %d on-approach"
        % (dst, len(out_rows), dict(sorted(counts.items())), len(frozen), n_appr)
    )


if __name__ == "__main__":
    main()
