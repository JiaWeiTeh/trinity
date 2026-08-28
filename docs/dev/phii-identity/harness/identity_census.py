#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-phase census of the P_HII identity and the drive composition, old vs new.

Maintainer question (2026-08-28): "if P_HII == P_conf -- is that always the case in
energy/implicit/momentum/transition? -- then we are essentially just double counting."

A pressure double-count needs BOTH halves, and they are separate facts:
  (i)  the IDENTITY   P_HII == P_conf (the relabelling), and
  (ii) an ADDITIVE composition that then adds P_HII to something it already equals.

Half (ii) is fixed by the source and differs by phase (verified at cce8c924):
    energy / implicit   energy_phase_ODEs.py:256   P_drive = max(Pb_eff, P_HII)
    transition          energy_phase_ODEs.py:253   P_drive = max(Pb_eff, P_HII + P_ram)
    momentum            run_momentum_phase.py:445  P_drive = P_HII + P_ram
So a max phase cannot double-count no matter what the identity does -- max(Pb, Pb) = Pb --
while the two summing phases can. This script measures half (i) per phase per arm, and
CHECKS half (ii) by recomputing P_drive from the stored components and comparing it to
the run's own stored P_drive (drive_recompute_max_relerr) rather than asserting it.

Reads the committed both-arm trajectories; no solver run, no run dirs needed:
    b7_regime_trajectory.csv   B3M     stock (pre-C3c worktree fca7d88) vs c3c
    b12_lowwind_trajectory.csv B3MW01  stock (fca7d88e) vs c3c (bac9547e)

Columns per (config, arm, phase):
    frac_identity        |P_HII/Pb - 1| <= 1e-12          -- the relabelling
    frac_PHII_zero       P_HII == 0.0 exactly             -- C3c's confined branch
    med_PHII_over_Pb     median over DRIVING rows (P_HII > 0); nan if none
    med_Pdrive_over_Pb   the drive anatomy (Batch 0's Pdrive_over_Fram_med analogue)
    frac_Pb_eq_Pram      |Pb/P_ram - 1| <= 1e-12 -- checks the momentum convention
                         (run_momentum_phase assigns Pb := P_ram) instead of assuming it

    python docs/dev/phii-identity/harness/identity_census.py \
        --out docs/dev/phii-identity/data/b14_identity_census.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
SOURCES = [("B3M", "b7_regime_trajectory.csv"), ("B3MW01", "b12_lowwind_trajectory.csv")]
PHASES = ("energy", "implicit", "transition", "momentum")
TOL = 1e-12

FIELDS = [
    "config", "arm", "phase", "n_rows", "composition",
    "frac_identity", "frac_PHII_zero", "med_PHII_over_Pb",
    "med_Pdrive_over_Pb", "frac_Pb_eq_Pram", "drive_recompute_max_relerr",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def composition(phase):
    """The live P_drive expression for this phase, verified against source at cce8c924."""
    if phase == "momentum":
        return "P_HII + P_ram (SUM)"
    if phase == "transition":
        return "max(Pb, P_HII + P_ram) (sum in max)"
    return "max(Pb, P_HII) (MAX)"


def recompute(phase, Pb, PH, Pram):
    if phase == "momentum":
        return PH + Pram
    if phase == "transition":
        return max(Pb, PH + Pram)
    return max(Pb, PH)


def census(config, path):
    rows = list(csv.DictReader(l for l in open(DATA / path) if not l.startswith("#")))
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        for phase in PHASES:
            sel = [r for r in rows if r["arm"] == arm and r["current_phase"] == phase]
            if not sel:
                continue
            ident = zero = n_pb = pb_eq_pram = 0
            ratios, drives, errs = [], [], []
            for r in sel:
                Pb, PH = fnum(r, "Pb"), fnum(r, "P_HII")
                Pram, Pd = fnum(r, "P_ram") or 0.0, fnum(r, "P_drive")
                if PH is None or Pb is None:
                    continue
                if PH == 0.0:
                    zero += 1
                if Pb > 0:
                    n_pb += 1
                    if abs(PH / Pb - 1.0) <= TOL:
                        ident += 1
                    if PH > 0:
                        ratios.append(PH / Pb)
                    if Pd is not None:
                        drives.append(Pd / Pb)
                    if Pram > 0 and abs(Pb / Pram - 1.0) <= TOL:
                        pb_eq_pram += 1
                if Pd is not None and Pd > 0:
                    errs.append(abs(recompute(phase, Pb, PH, Pram) / Pd - 1.0))
            out.append(dict(
                config=config, arm=arm, phase=phase, n_rows=len(sel),
                composition=composition(phase),
                frac_identity=(ident / n_pb) if n_pb else float("nan"),
                frac_PHII_zero=zero / len(sel),
                med_PHII_over_Pb=med(ratios),
                med_Pdrive_over_Pb=med(drives),
                frac_Pb_eq_Pram=(pb_eq_pram / n_pb) if n_pb else float("nan"),
                drive_recompute_max_relerr=(max(errs) if errs else float("nan")),
            ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b14_identity_census.csv")
    args = ap.parse_args()

    rows = [r for cfg, path in SOURCES for r in census(cfg, path)]

    hdr = (f"{'config':8}{'arm':7}{'phase':11}{'n':>4}  {'ident':>7}{'P=0':>7}"
           f"{'PH/Pb':>9}{'Pdrv/Pb':>9}{'Pb=Pram':>9}  composition")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:8}{r['arm']:7}{r['phase']:11}{r['n_rows']:>4}  "
              f"{r['frac_identity']:>7.4f}{r['frac_PHII_zero']:>7.4f}"
              f"{r['med_PHII_over_Pb']:>9.3f}{r['med_Pdrive_over_Pb']:>9.3f}"
              f"{r['frac_Pb_eq_Pram']:>9.4f}  {r['composition']}")

    worst = max((r["drive_recompute_max_relerr"] for r in rows
                 if not math.isnan(r["drive_recompute_max_relerr"])), default=float("nan"))
    print(f"\ncomposition check: recomputed P_drive vs the runs' own stored P_drive, "
          f"worst rel err {worst:.2e}")
    if not math.isnan(worst) and worst > 1e-9:
        print("  ^ the composition read from source does NOT reproduce the stored drive "
              "on some phase — treat the composition column as unverified until diagnosed")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write("# Per-phase P_HII identity + drive-composition census, stock "
                     "(pre-C3c) vs c3c (production), from committed both-arm trajectories.\n")
            fh.write("# Answers the maintainer's 2026-08-28 double-counting question; "
                     "see PLAN.md SBatch-14 / S9.\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
