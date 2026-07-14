#!/usr/bin/env python3
"""bench5 fidelity check — in-container vs Helix, per arm.

The Phase-5 in-container result (FINDINGS §15h) is PROVISIONAL because in-container-vs-HPC
numerical fidelity was never measured. Once the HPC confirmation lands
(`./sync_bench.sh bench5 run && down` -> runs/data/bench5_summary_hpc.csv), this prints the
per-arm deltas and the verdict: HPC wins any disagreement (project rule).

    python docs/dev/transition/pdv-trigger/data/compare_bench5_hpc.py

Reads runs/data/bench5_summary.csv (in-container) + runs/data/bench5_summary_hpc.csv (Helix).
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "runs" / "data"


def _read(path):
    with open(path) as fh:
        return {r["run_name"]: r for r in csv.DictReader(x for x in fh if not x.startswith("#"))}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    hpc_path = DATA / "bench5_summary_hpc.csv"
    if not hpc_path.exists():
        sys.exit(
            "no runs/data/bench5_summary_hpc.csv yet — run './sync_bench.sh bench5 run' "
            "then 'down' first."
        )
    ic, hpc = _read(DATA / "bench5_summary.csv"), _read(hpc_path)
    common = sorted(set(ic) & set(hpc))
    only_ic, only_hpc = sorted(set(ic) - set(hpc)), sorted(set(hpc) - set(ic))

    worst_th, worst_arm, fire_flips = 0.0, None, []
    print(f"{'arm':34s} {'θmax(ic)':>9s} {'θmax(hpc)':>9s} {'Δθmax':>8s} {'fired ic→hpc':>13s}")
    for a in common:
        t0, t1 = _f(ic[a]["theta_max"]), _f(hpc[a]["theta_max"])
        f0, f1 = ic[a]["fired_cooling_balance"], hpc[a]["fired_cooling_balance"]
        d = abs(t1 - t0) if (t0 is not None and t1 is not None) else None
        if d is not None and d > worst_th:
            worst_th, worst_arm = d, a
        if f0 != f1:
            fire_flips.append(a)
        flag = " <-- FIRE FLIP" if f0 != f1 else ""
        print(
            f"{a:34s} {t0 if t0 is not None else '-':>9} {t1 if t1 is not None else '-':>9} "
            f"{f'{d:.4f}' if d is not None else '-':>8} {f0:>5s}→{f1:<5s}{flag}"
        )

    print(
        f"\n{len(common)} arms compared; only-in-container: {only_ic or '—'}; "
        f"only-HPC: {only_hpc or '—'}"
    )
    print(f"max |Δθ_max| = {worst_th:.4f}  ({worst_arm})")
    print(f"fire-map flips: {fire_flips or 'NONE'}")
    print(
        "verdict: "
        + (
            "FIDELITY OK — fire map identical; quote the HPC numbers as "
            "authoritative and drop the PROVISIONAL banner (§15h)."
            if not fire_flips
            else "DISAGREEMENT — HPC wins; update §15h fire map from the HPC summary "
            "and record the flip(s), dated."
        )
    )


if __name__ == "__main__":
    main()
