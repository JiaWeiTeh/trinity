#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is Batch 13's "the two errors partly cancel" claim true? — maintainer check, 2026-08-29.

THE CLAIM UNDER TEST (PLAN.md §Batch 13 RESULT, repeated in §7.1's K10 row):
    "evidence for a cancellation in the shipped scheme: C3a's cavity-volume error
     (which inflates, per K5) and its missing dust sink (which deflates) are of
     similar size and opposite sign in this regime."

It is checkable by algebra on committed data, so it should not have stood on
plausibility. C3a's density is

    n_C3a = sqrt( Qi_abs / (chi_e alpha_B V) )      with V = V_cavity = (4/3) pi R2^3

and P_HII is linear in n, so each correction contributes a multiplicative factor:

    volume-only  (use the cavity-EXCLUDED layer, K5)   f_V = sqrt(V_cav / V_layer)
    dust-only    (spend only the RECOMBINING photons)  f_D = sqrt(recomb / Qi_abs)
    both                                               f_VD = f_V * f_D

FALSIFIER of the cancellation claim: if f_V < 1 AND f_D < 1 on the same rows, the two
corrections push the SAME way and compound — they cannot cancel, whatever their sizes.
Cancellation requires one factor above 1 and one below.

Second question, since the claim was offered to explain why K10-with-dust lands near the
shipped drive: is corrected-C3a (f_VD applied) anywhere near K10? If not, K10's agreement
with shipped is not an error-cancellation story at all — K10's density comes from pressure
equilibrium, not from a photon balance, so it is not "C3a with two fixes".

Inputs, all committed: data/b9_layer_density.csv (R2, dR_ion -> R_IF, n_cavity,
recomb_over_Qiabs, n_rms_profile) and data/b17_dust_closure.csv (K10's rho, for the
second question). Exact spherical volumes throughout (B11.0 S1).

    python docs/dev/phii-identity/harness/cancellation_check.py \
        --out docs/dev/phii-identity/data/b19_cancellation.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"

FIELDS = [
    "phase", "t", "R2", "R_IF", "R_IF_over_R2", "V_cav_over_V_layer",
    "recomb_frac", "f_volume", "f_dust", "f_both",
    "same_direction", "pdrive_cavity", "pdrive_corrected", "pdrive_profile_form",
]


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
    ap.add_argument("--out", type=Path, default=DATA / "b19_cancellation.csv")
    args = ap.parse_args()

    src = [r for r in csv.DictReader(
        l for l in open(DATA / "b9_layer_density.csv") if not l.startswith("#"))
        if r.get("status") == "ok"]
    # b9_layer_density.csv predates layer_density_check.py's pdrive_* columns, so the
    # shipped P_HII/Pb is joined from the mass ledger on row_idx -- the same join Batch
    # 14 validated (worst |dt| 7.97e-08, |dR2| 1.45e-07 across the two B3M realisations).
    led = {int(r["row_idx"]): r for r in csv.DictReader(
        l for l in open(DATA / "b11_mass_ledger.csv") if not l.startswith("#"))
        if r.get("status") == "ok"}

    rows = []
    for r in src:
        R2, dR = fnum(r, "R2"), fnum(r, "dR_ion")
        rec = fnum(r, "recomb_over_Qiabs")
        n_cav, n_rms = fnum(r, "n_cavity"), fnum(r, "n_rms_profile")
        lr = led.get(int(r["row_idx"]))
        pd_cav = None
        if lr:
            _ph, _pb = fnum(lr, "P_HII"), fnum(lr, "Pb")
            if _ph and _pb and _pb > 0 and _ph > 0:
                # sanity: the joined row must be the same instant
                if abs(fnum(lr, "t") / fnum(r, "t") - 1.0) < 1e-4:
                    pd_cav = _ph / _pb
        if None in (R2, dR, rec) or not (R2 > 0 and dR > 0 and rec > 0):
            continue
        R_IF = R2 + dR
        # exact spherical volumes; the (4/3)pi cancels in the ratio
        v_ratio = R2**3 / (R_IF**3 - R2**3)      # V_cavity / V_layer
        f_V = math.sqrt(v_ratio)
        f_D = math.sqrt(rec)
        rows.append(dict(
            phase=r["phase"], t=fnum(r, "t"), R2=R2, R_IF=R_IF,
            R_IF_over_R2=R_IF / R2, V_cav_over_V_layer=v_ratio,
            recomb_frac=rec, f_volume=f_V, f_dust=f_D, f_both=f_V * f_D,
            same_direction=((f_V < 1.0) == (f_D < 1.0)),
            pdrive_cavity=pd_cav,
            pdrive_corrected=(pd_cav * f_V * f_D) if pd_cav else None,
            pdrive_profile_form=(pd_cav * n_rms / n_cav) if (pd_cav and n_rms and n_cav) else None,
        ))

    if not rows:
        sys.exit("no usable rows")

    print(f"{len(rows)} rows from b9_layer_density.csv (B3M)\n")
    hdr = (f"{'phase':11}{'n':>4}{'R_IF/R2':>10}{'f_volume':>11}{'f_dust':>10}"
           f"{'f_both':>10}{'same dir':>10}")
    print(hdr)
    print("-" * len(hdr))
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r for r in rows if r["phase"] == ph]
        if not sel:
            continue
        same = sum(1 for r in sel if r["same_direction"])
        print(f"{ph:11}{len(sel):>4}{med([r['R_IF_over_R2'] for r in sel]):>10.3f}"
              f"{med([r['f_volume'] for r in sel]):>11.4f}"
              f"{med([r['f_dust'] for r in sel]):>10.4f}"
              f"{med([r['f_both'] for r in sel]):>10.4f}"
              f"{same:>6}/{len(sel):<4}")

    n_same = sum(1 for r in rows if r["same_direction"])
    both_below = sum(1 for r in rows if r["f_volume"] < 1 and r["f_dust"] < 1)
    print(f"\nVERDICT on the cancellation claim")
    print(f"  rows where both corrections push the SAME way: {n_same}/{len(rows)}")
    print(f"  rows where BOTH are < 1 (both DEFLATE C3a):    {both_below}/{len(rows)}")
    if n_same == len(rows):
        print("  => the two corrections COMPOUND on every row. They cannot cancel.")
        print("     Batch 13's 'opposite sign' clause is FALSE as written.")
    else:
        print("  => mixed; the claim survives on some rows. Report per phase.")

    print("\nSize of each, momentum phase (the regime the claim was made about):")
    mom = [r for r in rows if r["phase"] == "momentum"]
    if mom:
        print(f"  volume correction alone : x{med([r['f_volume'] for r in mom]):.4f}")
        print(f"  dust correction alone   : x{med([r['f_dust'] for r in mom]):.4f}")
        print(f"  both together           : x{med([r['f_both'] for r in mom]):.4f}")
        print(f"  shipped P_HII/Pb        : {med([r['pdrive_cavity'] for r in mom]):.3f}")
        print(f"  corrected-C3a P_HII/Pb  : {med([r['pdrive_corrected'] for r in mom]):.3f}")
        print(f"  profile-form P_HII/Pb   : {med([r['pdrive_profile_form'] for r in mom]):.3f}"
              "   (K5b, independent route)")

    # Second question: is K10 anywhere near corrected-C3a?
    k10 = [r for r in csv.DictReader(
        l for l in open(DATA / "b17_dust_closure.csv") if not l.startswith("#"))
        if r.get("status") == "ok" and r.get("config") == "B3M"
        and r.get("phase") == "momentum"]
    if k10 and mom:
        k10_ratio = med([fnum(r, "drive_selfconsistent") / fnum(r, "P_conf") for r in k10])
        corr = med([r["pdrive_corrected"] for r in mom])
        print(f"\nIs K10 'C3a with the two fixes'?  K10 drive/P_conf = {k10_ratio:.3f} "
              f"vs corrected-C3a {corr:.3f}  -> ratio {k10_ratio/corr:.2f}x")
        print("  K10's density comes from PRESSURE EQUILIBRIUM, not a photon balance, so it")
        print("  is a different closure -- not C3a with corrections applied.")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Factorial test of Batch 13's 'the two errors partly cancel' claim.\n")
        fh.write("# f_volume = sqrt(V_cav/V_layer), f_dust = sqrt(recomb/Qi_abs); both\n")
        fh.write("# multiply C3a's density (and hence P_HII) directly.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
