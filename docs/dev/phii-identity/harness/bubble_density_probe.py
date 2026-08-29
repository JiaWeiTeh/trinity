#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory — can the bubble's REAL density (mass/volume) inform P_HII? — 2026-08-29.

Maintainer: *"can we try out the bubble profile method (since we have bubble mass and
volume we can easily get density) and see how that goes? keep in mind that in transition
and momentum phase there will be no bubble since it slowly gets compressed into R2 because
Pb -> 0."*

EXPLORATORY, not a gated candidate: this measures whether the idea is viable before anyone
writes gates for it. The hot bubble is the shocked wind between R1 and R2, so

    V_bub = (4/3) pi (R2**3 - R1**3)
    n_bub = bubble_mass / (mu_convert * V_bub)
    T_implied = Pb * mu_i / (mu_c * n_bub * k_B)     [the bubble's own P = n k T]

Three questions:
  Q1  Is `bubble_mass` usable at all? B11.0 called it frozen; this localises WHERE.
  Q2  Does the cavity collapse as the maintainer predicts, and where does n_bub diverge?
  Q3  Where the density IS well defined, is it self-consistent (does n_bub k T reproduce
      Pb at a sane temperature), and can the cavity absorb any ionising photons?

Inputs, all committed: data/b11_mass_ledger.csv (bubble_mass, Pb, R2, Qi, phase, t) and
data/b7_regime_trajectory.csv (R1, Eb) joined on t.

    python docs/dev/phii-identity/harness/bubble_density_probe.py \
        --out docs/dev/phii-identity/data/b22_bubble_density.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
BENCH = REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/bench3_m1e5_r5__none_diag.param"

FIELDS = ["phase", "t", "R1", "R2", "R1_over_R2", "V_bub_over_sphere", "bubble_mass",
          "n_bub", "Pb", "T_implied_K", "recomb_rate_over_Qi", "status"]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b22_bubble_density.csv")
    args = ap.parse_args()

    p = read_param(str(BENCH))
    mu_c, mu_i = p["mu_convert"].value, p["mu_ion_shell"].value
    kB, chi, aB = p["k_B"].value, p["chi_e_shell"].value, p["caseB_alpha"].value

    def rd(n):
        return list(csv.DictReader(l for l in open(DATA / n) if not l.startswith("#")))

    traj = {}
    for r in rd("b7_regime_trajectory.csv"):
        if r.get("arm") == "c3c" and fnum(r, "t_now") is not None:
            traj[round(fnum(r, "t_now"), 9)] = r

    rows = []
    for r in rd("b11_mass_ledger.csv"):
        if r.get("status") != "ok":
            continue
        t = fnum(r, "t")
        tr = traj.get(round(t, 9))
        if tr is None:
            continue
        R1, R2 = fnum(tr, "R1"), fnum(r, "R2")
        mb, Pb, Qi = fnum(r, "bubble_mass"), fnum(r, "Pb"), fnum(r, "Qi")
        if None in (R1, R2, mb, Pb) or R2 <= 0:
            continue
        vfrac = (R2**3 - R1**3) / R2**3          # bubble volume as a fraction of the sphere
        V = 4.0 / 3.0 * math.pi * (R2**3 - R1**3)
        if V <= 0 or mb <= 0:
            rows.append(dict(phase=r["phase"], t=t, R1=R1, R2=R2, R1_over_R2=R1 / R2,
                             V_bub_over_sphere=vfrac, bubble_mass=mb, n_bub=None, Pb=Pb,
                             T_implied_K=None, recomb_rate_over_Qi=None,
                             status="UNDEFINED: bubble volume is zero"))
            continue
        n = mb / (mu_c * V)
        T = Pb * mu_i / (mu_c * n * kB)
        # what fraction of Qi could this gas actually consume by recombination?
        rec = chi * aB * n * n * V / Qi if Qi else None
        rows.append(dict(phase=r["phase"], t=t, R1=R1, R2=R2, R1_over_R2=R1 / R2,
                         V_bub_over_sphere=vfrac, bubble_mass=mb, n_bub=n, Pb=Pb,
                         T_implied_K=T, recomb_rate_over_Qi=rec, status="ok"))

    print(f"{len(rows)} rows\n")

    print("Q1 — is bubble_mass usable? (distinct values per phase)")
    for ph in ("energy", "implicit", "transition", "momentum"):
        v = [r["bubble_mass"] for r in rows if r["phase"] == ph]
        if v:
            flag = "  <-- FROZEN" if len(set(v)) == 1 and len(v) > 1 else ""
            print(f"    {ph:11} n={len(v):3d}  {min(v):9.4f} .. {max(v):9.4f}   "
                  f"distinct {len(set(v)):3d}{flag}")

    print("\nQ2 — does the cavity collapse? (the maintainer's prediction)")
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r for r in rows if r["phase"] == ph]
        if sel:
            print(f"    {ph:11} R1/R2 {min(r['R1_over_R2'] for r in sel):.4f} .. "
                  f"{max(r['R1_over_R2'] for r in sel):.4f}   "
                  f"V_bub/sphere min {min(r['V_bub_over_sphere'] for r in sel):.3e}")
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"    rows where the density is UNDEFINED (zero volume): {len(bad)}/{len(rows)}"
          f"   phases {sorted({r['phase'] for r in bad})}")

    print("\nQ3 — where it IS defined, is it self-consistent?")
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r for r in rows if r["phase"] == ph and r["status"] == "ok"]
        if sel:
            print(f"    {ph:11} n={len(sel):3d}  n_bub median {med([r['n_bub'] for r in sel]):.3e}"
                  f"   T_implied median {med([r['T_implied_K'] for r in sel]):.3e} K"
                  f"   recomb/Qi median {med([r['recomb_rate_over_Qi'] for r in sel]):.3e}")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# EXPLORATORY (not a gated candidate): can the bubble's real density inform\n")
        fh.write("# P_HII? Answers the maintainer's 2026-08-29 question, including the cavity\n")
        fh.write("# collapse they predicted for transition/momentum.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
