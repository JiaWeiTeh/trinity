#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 16 — K10's composition mapping, gated THROUGH the real P_drive expressions.

Gates G16.0–G16.4 are pre-registered in PLAN.md (§Batch 16) and committed BEFORE this
script existed. This script only measures; no bar is moved.

Batch 13 screened K10 by computing `P_conf·(R_i/R2)²` directly. That is not what the code
would do: the helper returns a value which then flows through each phase's own `P_drive`
expression, and Batch 14 found the "excess rides the existing compositions" claim holds in
momentum ONLY. This harness closes that hole — it routes a candidate helper return through
the ACTUAL composition and asks whether the CEM drive comes back out.

Mapping under test (PLAN §Batch 16), with rho = (R_i/R2)^2 >= 1:
    energy / implicit   composition max(P_conf, P_HII)          return  P_conf*rho
    transition          composition max(P_conf, P_HII + P_ram)  return  P_conf*rho - P_ram
    momentum            composition P_HII + P_ram               return  P_ram*(rho - 1)
i.e. return = P_conf*rho - (P_ram if this phase's composition adds it else 0).

P_conf, and the D-ramp (G16.3): in energy/implicit the LIVE drive uses the RAMPED
press_bubble, which is not a stored column -- but on CONFINED energy/implicit rows the
shipped helper returns exactly 0.0, so `P_drive = max(press_bubble, 0) = press_bubble` and
the ramped pressure is recoverable from the run's own P_drive. Both choices are evaluated
and compared, because using un-ramped Pb re-admits the defect class C3c removed.

Q_eff is run for BOTH Batch 13 variants (A = Qi*f_abs, B = with dust from the photon
ledgers). The mapping algebra does not depend on Q_eff, so a gate passing for only one
variant would indicate a bug in the gate rather than a property of K10.

    python docs/dev/phii-identity/harness/k10_composition_check.py \
        --out docs/dev/phii-identity/data/b16_composition.csv
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
FOUR_PI = 4.0 * math.pi
RAMP_T = 1e-3          # dt_switchon window (Batch 13's convention, reused deliberately)
TOL_G160 = 1e-12       # the pre-registered G16.0 bar

SOURCES = [
    ("B3M", "b7_regime_trajectory.csv", "b11_photon_ledger.csv"),
    ("B3MW01", "b12_lowwind_trajectory.csv", "b12_lowwind_photon_ledger.csv"),
]

FIELDS = [
    "config", "variant", "phase", "t", "R2", "in_ramp_window",
    "P_conf_used", "P_conf_ramped", "P_conf_unramped", "P_ram", "rho",
    "mapped_return", "composed_drive", "target_drive", "g160_relerr",
    "g161_nonneg", "g162_exceeds_conf", "shipped_drive", "composed_over_shipped",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def load_dust(path):
    rows = [r for r in csv.DictReader(l for l in open(DATA / path) if not l.startswith("#"))]
    return sorted((float(r["t"]), float(r["dust_Pb"])) for r in rows
                  if r.get("dust_Pb") not in (None, "", "None"))


def nearest_dust(pts, t, tol=0.02):
    if not pts:
        return None
    best = min(pts, key=lambda p: abs(p[0] - t))
    return best[1] if abs(best[0] - t) <= tol else None


def compose(phase, P_conf, P_HII, P_ram):
    """The phase's REAL P_drive expression (verified at cce8c924) -- the point of this harness."""
    if phase == "momentum":
        return P_HII + P_ram
    if phase == "transition":
        return max(P_conf, P_HII + P_ram)
    return max(P_conf, P_HII)


def mapped(phase, P_conf, rho, P_ram):
    """PLAN §Batch 16: P_conf*rho minus whatever this phase's composition already adds."""
    if phase == "momentum":
        return P_ram * (rho - 1.0)
    if phase == "transition":
        return P_conf * rho - P_ram
    return P_conf * rho


def screen(config, traj, dust_pts, consts):
    mu_c, mu_i, kB, T, chi, aB = consts
    rows = [r for r in csv.DictReader(l for l in open(DATA / traj) if not l.startswith("#"))
            if r.get("arm") == "c3c"]
    out = []
    for r in rows:
        t, ph = fnum(r, "t_now"), r.get("current_phase")
        R2, Qi, fa = fnum(r, "R2"), fnum(r, "Qi"), fnum(r, "shell_fAbsorbedIon")
        Pb, Pram = fnum(r, "Pb"), fnum(r, "P_ram") or 0.0
        PH, Pd = fnum(r, "P_HII"), fnum(r, "P_drive")
        if None in (t, ph, R2, Qi, fa, Pb, PH, Pd) or not (R2 > 0 and Qi * fa > 0):
            continue

        # G16.3: recover the RAMPED confining pressure where the run reveals it.
        # On confined energy/implicit rows the shipped helper returns exactly 0.0, so the
        # stored P_drive IS press_bubble. Elsewhere the ramp is over and Pb is the value.
        if ph in ("energy", "implicit"):
            conf_ramped = Pd if PH == 0.0 else Pb
            conf_unramped = Pb
        elif ph == "transition":
            conf_ramped = conf_unramped = max(Pb, Pram)
        else:
            conf_ramped = conf_unramped = Pram
        if not (conf_ramped > 0 and conf_unramped > 0):
            continue

        fd = nearest_dust(dust_pts, t)
        for variant, q_eff in (("A", Qi * fa),
                               ("B", Qi * fa * (1.0 - fd) if (fd is not None and 0 <= fd < 1) else None)):
            if q_eff is None or q_eff <= 0:
                continue
            for label, P_conf in (("ramped", conf_ramped), ("unramped", conf_unramped)):
                if label == "unramped" and ph not in ("energy", "implicit"):
                    continue  # identical to ramped outside the ED phases; don't double-count rows
                n0 = (mu_i / mu_c) * P_conf / (kB * T)
                ri3 = R2**3 + 3.0 * q_eff / (FOUR_PI * chi * aB * n0**2)
                rho = (ri3 / R2**3) ** (2.0 / 3.0)
                ret = mapped(ph, P_conf, rho, Pram)
                drive = compose(ph, P_conf, ret, Pram)
                target = P_conf * rho
                out.append(dict(
                    config=config, variant=f"{variant}/{label}", phase=ph, t=t, R2=R2,
                    in_ramp_window=(t <= RAMP_T),
                    P_conf_used=P_conf, P_conf_ramped=conf_ramped,
                    P_conf_unramped=conf_unramped, P_ram=Pram, rho=rho,
                    mapped_return=ret, composed_drive=drive, target_drive=target,
                    g160_relerr=abs(drive / target - 1.0) if target > 0 else None,
                    g161_nonneg=(ret >= 0.0),
                    g162_exceeds_conf=(drive > P_conf) if rho > 1.0 else None,
                    shipped_drive=Pd,
                    composed_over_shipped=(drive / Pd) if Pd > 0 else None,
                ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b16_composition.csv")
    args = ap.parse_args()

    params = read_param(str(
        REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
        "bench3_m1e5_r5__none_diag.param"))
    consts = (params["mu_convert"].value, params["mu_ion_shell"].value,
              params["k_B"].value, params["TShell_ion"].value,
              params["chi_e_shell"].value, params["caseB_alpha"].value)

    rows = []
    for cfg, traj, ledger in SOURCES:
        rows += screen(cfg, traj, load_dust(ledger), consts)
    print(f"{len(rows)} (row × variant) evaluations over {len({r['config'] for r in rows})} configs")

    # ---- G16.0 (BLOCKING) ----
    errs = [r["g160_relerr"] for r in rows if r["g160_relerr"] is not None]
    worst = max(errs) if errs else float("nan")
    print(f"\nG16.0 mapping reproduces P_conf*rho through the REAL compositions: "
          f"worst rel err {worst:.2e} vs {TOL_G160:.0e} bar -> "
          f"{'PASS' if worst <= TOL_G160 else 'FAIL'}")
    for ph in ("energy", "implicit", "transition", "momentum"):
        e = [r["g160_relerr"] for r in rows if r["phase"] == ph and r["g160_relerr"] is not None]
        if e:
            print(f"    {ph:11} n={len(e):4d}  worst {max(e):.2e}")
    for v in sorted({r["variant"] for r in rows}):
        e = [r["g160_relerr"] for r in rows if r["variant"] == v and r["g160_relerr"] is not None]
        print(f"    variant {v:12} n={len(e):4d}  worst {max(e):.2e}")

    # ---- G16.1 ----
    neg = [r for r in rows if not r["g161_nonneg"]]
    print(f"\nG16.1 admissibility (mapped return >= 0): {len(rows)-len(neg)}/{len(rows)} ok -> "
          f"{'PASS' if not neg else 'FAIL'}")
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r["mapped_return"] for r in rows if r["phase"] == ph]
        if sel:
            print(f"    {ph:11} min return {min(sel):+.4e}")
    if neg:
        print(f"    ⛔ negative on {len(neg)} rows, phases "
              f"{sorted({r['phase'] for r in neg})} — needs a floor DECISION (physics)")

    # ---- G16.2 ----
    checked = [r for r in rows if r["g162_exceeds_conf"] is not None]
    bad = [r for r in checked if not r["g162_exceeds_conf"]]
    print(f"\nG16.2 confined-limit term delivered (drive > P_conf where rho>1): "
          f"{len(checked)-len(bad)}/{len(checked)} -> {'PASS' if not bad else 'FAIL'}")
    conf_rows = [r for r in rows if r["phase"] in ("energy", "implicit") and "ramped" in r["variant"]]
    if conf_rows:
        exc = [r["composed_drive"] / r["P_conf_used"] - 1.0 for r in conf_rows]
        print(f"    ED-phase excess over P_conf: median {med(exc)*100:.2f}%  "
              f"(shipped returns exactly 0.0 there, so this is the 'better than 0.0' term)")

    # ---- G16.3 ----
    print("\nG16.3 D-ramp — ramped vs un-ramped P_conf in energy/implicit:")
    ed = [r for r in rows if r["phase"] in ("energy", "implicit")]
    ramp = [r for r in ed if r["in_ramp_window"]]
    for tag, sel in (("inside dt_switchon", ramp), ("outside", [r for r in ed if not r["in_ramp_window"]])):
        d = [r["P_conf_ramped"] / r["P_conf_unramped"] for r in sel if r["P_conf_unramped"] > 0]
        if d:
            print(f"    {tag:20} n={len(d):4d}  P_conf_ramped/unramped "
                  f"min {min(d):.4f} median {med(d):.4f} max {max(d):.4f}")
    print("    params carries current_phase but NOT press_bubble (verified) => K10 needs the "
          "ramped pressure PASSED IN: a signature change, as pre-registered.")

    # ---- G16.4 ----
    print("\nG16.4 magnitude — composed drive / shipped drive (median):")
    for cfg in sorted({r["config"] for r in rows}):
        for ph in ("energy", "implicit", "transition", "momentum"):
            for v in ("A/ramped", "B/ramped"):
                sel = [r["composed_over_shipped"] for r in rows
                       if r["config"] == cfg and r["phase"] == ph and r["variant"] == v
                       and r["composed_over_shipped"]]
                if sel:
                    print(f"    {cfg:8}{ph:11}{v:10} n={len(sel):3d}  {med(sel):7.3f}")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 16: K10 composition mapping routed through the REAL P_drive\n")
        fh.write("# expressions. Gates pre-registered in PLAN.md SBatch-16 before this ran.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
