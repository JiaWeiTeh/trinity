#!/usr/bin/env python3
"""Gate G0 for the bench7 f_kappa re-open — the baseline check that must clear BEFORE HPC time.

KAPPA_REOPEN_PLAN.md section 5 defines G0 as: *"data/bench6_analysis.csv Theta_0 = 0.462/0.341/0.221
and section 18's band-entry table are reproduced by re-running the two analysis scripts on the
committed trajectories; fail => the baseline moved, stop and reconcile before spending HPC time."*
This harness is that gate, made machine-checkable and durable, so a later visit re-clears it in a
second instead of re-reading three docs and eyeballing a console dump.

It recomputes every G0 quantity FROM THE TRAJECTORY CSVs through make_bench6_analysis's own
functions — not by re-reading bench6_analysis.csv, which would only prove the CSV still says what it
said. Nothing is written outside data/bench7_gate_g0.csv, so the gate has no side effects on the
tracked analysis outputs.

RUN IT TWICE — it answers a different question each time, and the SOURCES line in the CSV says which:
  * BEFORE the all-fresh re-run it reads the 2026-07-19 harvest and is a self-check ("the published
    numbers still fall out of the trajectories they were computed from" — cleared 11/11 on 07-29);
  * AFTER `./sync_bench.sh bench5r|bench6r down` it prefers bench5r/bench6r automatically and checks
    the SAME pre-registered targets against arms run today, so a PASS means the 07-19 result
    REPRODUCED and a FAIL means it did not. That is the whole point of the ALL-FRESH ruling
    (maintainer, 2026-07-29), and the targets are not relaxed in either direction.

Two tables:
  G0   the pre-registered baseline targets vs measured: Theta_0 per bench, the f_A and f_mix
       band-entry doses, and the two uniformity spreads. Every row carries PASS/FAIL.
  P1   the pre-registered f_kappa band-entry prediction the campaign is about to test, computed
       from the SAME measured Theta_0 the gate just checked: entry = (0.90/Theta_0)^(1/q) over
       q in [0.55, 0.70] (the K0.Q1 fixed-state L_cool exponents 0.586/0.669, assumed to carry to
       the integrated metric). Frozen here BEFORE any arm runs, per the planning protocol -- if the
       measurement misses, the miss is recorded against these numbers (the SC-0 pattern, section 15k).

    python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py
Deliverable: data/bench7_gate_g0.csv (+ console verdict; exit 1 if any G0 row FAILs).
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RDATA = HERE.parent / "runs" / "data"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _stamp import stamp  # noqa: E402
from make_bench6_analysis import (  # noqa: E402
    CLEAN_BLOWOUT,
    _knob_dose,
    _load,
    band_entry,
    band_entry_extrapolated,
)

# (label, target, tol) — tol is half the last digit the source quotes, so the gate is exactly as
# tight as the published number and no tighter. Sources: FINDINGS section 18 (band-entry table,
# spreads) and section 15h / bench6_analysis.csv (Theta_0). KAPPA_REOPEN_PLAN section 3 P1 restates
# the Theta_0 triple.
TARGETS = [
    ("theta0_bench3_m1e5_r5", 0.462, 0.0005),
    ("theta0_bench2_m1e5_r10", 0.341, 0.0005),
    ("theta0_bench1_m5e4_r20", 0.221, 0.0005),
    ("band_entry_fA_bench3_m1e5_r5", 13.9, 0.05),
    ("band_entry_fA_bench2_m1e5_r10", 53.5, 0.05),
    ("band_entry_fA_bench1_m5e4_r20", 74.8, 0.05),
    ("spread_fA", 5.39, 0.005),
    ("band_entry_fmix_bench3_m1e5_r5", 4.0, 0.05),
    ("band_entry_fmix_bench2_m1e5_r10", 8.16, 0.005),
    ("band_entry_fmix_bench1_m5e4_r20", 11.9, 0.05),
    ("spread_fmix", 2.96, 0.005),
]
Q_GRID = [0.55, 0.60, 0.70]  # P1's exponent bracket; 0.60 is the pre-registered central value
BAND_LO = 0.90


def _pick(fresh_summary, fresh_traj, old_summary, old_traj):
    """Prefer the ALL-FRESH re-run harvest when it has landed; fall back to the 2026-07-19 one.

    This is what makes G0 do double duty. Before the re-run it is a self-check ("the published
    numbers still fall out of the trajectories they were computed from"). After `sync_bench.sh
    bench5r|bench6r down`, the SAME pre-registered targets are checked against TODAY's arms, so a
    PASS means the 07-19 result reproduced and a FAIL means it did not — which is the question the
    all-fresh re-run was ordered to answer. The targets are never relaxed either way.
    """
    if (RDATA / fresh_summary).exists():
        return RDATA / fresh_summary, RDATA / fresh_traj, "FRESH"
    return RDATA / old_summary, RDATA / old_traj, "2026-07-19"


def measured():
    """Every G0 quantity, recomputed from the committed trajectories."""
    p5, t5, src5 = _pick(
        "bench5r_summary.csv", "bench5r_traj", "bench5_summary_hpc.csv", "bench5_traj_hpc"
    )
    p6, t6, src6 = _pick("bench6r_summary.csv", "bench6r_traj", "bench6_summary.csv", "bench6_traj")
    b5, b6 = _load(p5, t5), _load(p6, t6)
    if not b5 or not b6:
        sys.exit(
            f"ABORT: no usable bench5/bench6 harvest ({p5.name} / {p6.name}) — nothing to gate on."
        )
    print(f"  sources: bench5 <- {p5.name} [{src5}]   bench6 <- {p6.name} [{src6}]\n")

    series = {}
    for name, r in {**b5, **b6}.items():
        if not name.endswith("_diag") or r.get("_theta_cum") is None:
            continue
        bench = name.split("__")[0]
        if bench not in CLEAN_BLOWOUT:
            continue
        knob, dose = _knob_dose(name)
        series.setdefault((bench, knob), []).append((dose, r["_theta_cum"]))

    out = {}
    for bench in CLEAN_BLOWOUT:
        fa = sorted(series[(bench, "fA")])
        out[f"theta0_{bench}"] = next(t for d, t in fa if d == 1.0)
        # The fm ladder shares the unboosted dose-1 point (the __none arm), as bench6 does.
        fm = [(1.0, out[f"theta0_{bench}"])] + sorted(series[(bench, "fmix")])
        for knob, pts in (("fA", fa), ("fmix", fm)):
            e = band_entry(pts)
            out[f"band_entry_{knob}_{bench}"] = e if e else band_entry_extrapolated(pts)
            out[f"measured_in_grid_{knob}_{bench}"] = e is not None
    for knob in ("fA", "fmix"):
        vals = [out[f"band_entry_{knob}_{b}"] for b in CLEAN_BLOWOUT]
        out[f"spread_{knob}"] = max(vals) / min(vals)
    out["_sources"] = f"bench5={p5.name} [{src5}], bench6={p6.name} [{src6}]"
    return out


def main():
    got = measured()
    rows, failed = [], 0
    for label, target, tol in TARGETS:
        val = got[label]
        ok = abs(val - target) <= tol
        failed += not ok
        rows.append(
            {
                "table": "G0",
                "quantity": label,
                "pre_registered": f"{target:g}",
                "measured": f"{val:.6g}",
                "abs_tol": f"{tol:g}",
                "verdict": "PASS" if ok else "FAIL",
                "note": (
                    ""
                    if label.startswith(("theta0", "spread"))
                    else (
                        "MEASURED in-grid"
                        if got[label.replace("band_entry", "measured_in_grid")]
                        else "EXTRAPOLATED past the grid (section 18) — an estimate, not a measurement"
                    )
                ),
            }
        )

    for bench in sorted(CLEAN_BLOWOUT):
        t0 = got[f"theta0_{bench}"]
        for q in Q_GRID:
            rows.append(
                {
                    "table": "P1",
                    "quantity": f"predicted_fkappa_band_entry_{bench}_q{q:g}",
                    "pre_registered": f"{(BAND_LO / t0) ** (1 / q):.4g}",
                    "measured": "",
                    "abs_tol": "",
                    "verdict": "PENDING",
                    "note": f"(0.90/{t0:.3f})^(1/{q:g}) — P1, frozen before any bench7 arm runs",
                }
            )
    for q in Q_GRID:
        preds = [(BAND_LO / got[f"theta0_{b}"]) ** (1 / q) for b in CLEAN_BLOWOUT]
        rows.append(
            {
                "table": "P1",
                "quantity": f"predicted_fkappa_spread_q{q:g}",
                "pre_registered": f"{max(preds) / min(preds):.3f}",
                "measured": "",
                "abs_tol": "",
                "verdict": "PENDING",
                "note": "P1: predicted f_kappa spread, to be compared against f_A 5.39x and f_mix 2.96x",
            }
        )

    out = HERE / "bench7_gate_g0.csv"
    with out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write(
            f"# SOURCES READ: {got['_sources']}  <- this is the line that says whether the numbers "
            "below are today's re-run or the 2026-07-19 harvest.\n"
            "# bench7 gate G0 — the baseline check for the f_kappa re-open "
            "(KAPPA_REOPEN_PLAN.md section 5). Recomputed FROM the trajectory CSVs via "
            "make_bench6_analysis's own functions, NOT re-read from bench6_analysis.csv. Prefers the "
            "ALL-FRESH bench5r/bench6r harvest when present, else the 2026-07-19 one; either way the "
            "targets are the SAME pre-registered numbers, so against fresh arms a PASS means the "
            "07-19 result reproduced and a FAIL means it did not.\n"
            "# Targets: FINDINGS section 18 (band-entry table + spreads) and section 15h / "
            "bench6_analysis.csv (Theta_0). abs_tol = half the last digit the source quotes.\n"
            "# Table P1 rows are the pre-registered f_kappa band-entry PREDICTION "
            "(entry = (0.90/Theta_0)^(1/q), q in 0.55-0.70 from the K0.Q1 fixed-state L_cool "
            "exponents), frozen BEFORE any bench7 arm runs. A miss is recorded, not re-negotiated "
            "(the SC-0 pattern, FINDINGS section 15k).\n"
            "# Regenerate: python docs/dev/transition/pdv-trigger/data/make_bench7_gate_g0.py\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        if r["table"] == "G0":
            print(
                f"  {r['verdict']:4s} {r['quantity']:34s} {r['measured']:>10s}  "
                f"(target {r['pre_registered']} ± {r['abs_tol']})  {r['note']}"
            )
    print(
        "\nP1 predicted f_kappa band-entry spread: "
        + ", ".join(
            f"q={q:g} -> {next(r['pre_registered'] for r in rows if r['quantity'] == f'predicted_fkappa_spread_q{q:g}')}x"
            for q in Q_GRID
        )
    )
    print(f"\nwrote {len(rows)} rows -> {out}")
    print(
        f"G0: {'PASS — the baseline is intact; bench7 may be submitted' if not failed else f'FAIL on {failed} row(s) — STOP, reconcile before spending HPC time'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
