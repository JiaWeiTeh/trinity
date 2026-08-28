#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 13 — K10 offline screen: the smooth CEM drive vs the shipped C3c drive.

Gates G13.1–G13.5 and the input/mapping contract are pre-registered in PLAN.md
(§Batch 13), committed BEFORE this script first ran. This script only measures.

K10, phase-agnostic form (see §7.1):
    n_H0     = (mu_i/mu_c) * P_conf / (k_B T)            pressure-equilibrium skin density
    R_i^3    = R2^3 + 3 Qi_eff / (4 pi chi_e alpha_B n_H0^2)
    P_drive  = P_conf * (R_i/R2)^2
with P_conf = P_ram (momentum), max(Pb, P_ram) (transition), Pb (energy/implicit),
and Qi_eff = Qi*f_abs (variant A) or Qi*f_abs*(1 - f_dust,ion) (variant B, where the
photon-ledger CSVs carry dust for the row's nearest t).

Shipped comparator, from stored columns: P_HII + P_ram (momentum),
max(Pb, P_HII + P_ram) (transition), max(Pb, P_HII) (energy/implicit).

    python docs/dev/phii-identity/harness/k10_cem_drive_screen.py \
        docs/dev/phii-identity/data/b7_regime_trajectory.csv \
        docs/dev/phii-identity/data/b12_lowwind_trajectory.csv \
        --dust docs/dev/phii-identity/data/b11_photon_ledger.csv \
        --dust docs/dev/phii-identity/data/b12_lowwind_photon_ledger.csv \
        --out docs/dev/phii-identity/data/b13_k10_screen.csv
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

FOUR_PI = 4.0 * math.pi
RAMP_T = 1e-3  # dt_switchon window, excluded by the pre-registered mapping

FIELDS = [
    "src",
    "arm",
    "t",
    "phase",
    "R2",
    "P_conf",
    "P_ram",
    "Pb",
    "P_HII_shipped",
    "drive_shipped",
    "Ri_over_R2_A",
    "drive_K10_A",
    "excess_A_over_conf",
    "f_dust_ion",
    "Ri_over_R2_B",
    "drive_K10_B",
    "K10A_over_shipped",
    "K10B_over_shipped",
    "md_identity_relerr",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def load_dust(paths):
    """t -> f_dust,ion per source label, from the photon-ledger CSVs (driving rows only)."""
    out = {}
    for path in paths:
        rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
        pts = [
            (float(r["t"]), float(r["dust_Pb"]))
            for r in rows
            if r.get("dust_Pb") not in (None, "", "None")
        ]
        out[Path(path).name] = sorted(pts)
    return out


def nearest_dust(pts, t, tol=0.02):
    if not pts:
        return None
    best = min(pts, key=lambda p: abs(p[0] - t))
    return best[1] if abs(best[0] - t) <= tol else None


def screen(traj_path, consts, dust_pts, arm="c3c"):
    mu_c, mu_i, kB, T, chi, aB = consts
    rows = [r for r in csv.DictReader(l for l in open(traj_path) if not l.startswith("#"))]
    rows = [r for r in rows if r.get("arm") == arm]
    out, skipped = [], 0
    for r in rows:
        t = fnum(r, "t_now")
        R2, Qi, fa = fnum(r, "R2"), fnum(r, "Qi"), fnum(r, "shell_fAbsorbedIon")
        Pb, Pram, PH = fnum(r, "Pb"), fnum(r, "P_ram"), fnum(r, "P_HII")
        ph = r.get("current_phase")
        if None in (t, R2, Qi, fa, Pb, PH) or t <= RAMP_T:
            skipped += 1
            continue
        Pram = Pram or 0.0
        if ph == "momentum":
            P_conf, shipped = Pram, PH + Pram
        elif ph == "transition":
            P_conf, shipped = max(Pb, Pram), max(Pb, PH + Pram)
        else:
            P_conf, shipped = Pb, max(Pb, PH)
        if not (P_conf > 0 and Qi * fa > 0):
            skipped += 1
            continue

        def k10(q_eff):
            n0 = (mu_i / mu_c) * P_conf / (kB * T)
            ri3 = R2**3 + 3.0 * q_eff / (FOUR_PI * chi * aB * n0**2)
            rat2 = (ri3 / R2**3) ** (2.0 / 3.0)
            return rat2**0.5 * R2 / R2, P_conf * rat2  # (R_i/R2, drive)

        riA, driveA = k10(Qi * fa)
        riA = (driveA / P_conf) ** 0.5

        # G13.3 — MD identity: P_conf*(1 + R2/R_ch)^(2/3) with R_ch from pdot = 4 pi R2^2 P_ram
        md_err = None
        if ph == "momentum" and Pram > 0:
            pdot = FOUR_PI * R2**2 * Pram
            Rch = aB * pdot**2 / (12.0 * math.pi * ((mu_c / mu_i) * kB * T) ** 2 * (Qi * fa))
            alt = Pram * (1.0 + R2 / Rch) ** (2.0 / 3.0)
            md_err = abs(alt / driveA - 1.0)

        fd = nearest_dust(dust_pts, t)
        riB = driveB = None
        if fd is not None and 0.0 <= fd < 1.0:
            _, driveB = k10(Qi * fa * (1.0 - fd))
            riB = (driveB / P_conf) ** 0.5

        out.append(
            dict(
                src=Path(traj_path).name,
                arm=arm,
                t=t,
                phase=ph,
                R2=R2,
                P_conf=P_conf,
                P_ram=Pram,
                Pb=Pb,
                P_HII_shipped=PH,
                drive_shipped=shipped,
                Ri_over_R2_A=riA,
                drive_K10_A=driveA,
                excess_A_over_conf=driveA / P_conf - 1.0,
                f_dust_ion=fd,
                Ri_over_R2_B=riB,
                drive_K10_B=driveB,
                K10A_over_shipped=driveA / shipped if shipped > 0 else None,
                K10B_over_shipped=(driveB / shipped) if (driveB and shipped > 0) else None,
                md_identity_relerr=md_err,
            )
        )
    return out, skipped


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajs", nargs="+")
    ap.add_argument("--dust", action="append", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    params = read_param(
        str(
            REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
            "bench3_m1e5_r5__none_diag.param"
        )
    )
    consts = (
        params["mu_convert"].value,
        params["mu_ion_shell"].value,
        params["k_B"].value,
        params["TShell_ion"].value,
        params["chi_e_shell"].value,
        params["caseB_alpha"].value,
    )
    dust = load_dust(args.dust)

    allrows = []
    for tp in args.trajs:
        # pair each trajectory with its own run's ledger (b7<->b11, b12<->b12)
        key = "b11_photon_ledger.csv" if "b7_" in Path(tp).name else "b12_lowwind_photon_ledger.csv"
        rows, skipped = screen(tp, consts, dust.get(key, []))
        print(
            f"{Path(tp).name}: {len(rows)} rows screened, {skipped} skipped "
            f"(missing cols / ramp window / P_conf<=0)"
        )
        allrows += rows

    # ---- G13.1 continuity at the shipped switch ----
    print("\nG13.1 — drive step at rows where shipped P_HII crosses 0 -> positive:")
    worst_k10 = 0.0
    for src in sorted({r["src"] for r in allrows}):
        v = sorted((r for r in allrows if r["src"] == src), key=lambda r: r["t"])
        for a, b in zip(v, v[1:]):
            if a["P_HII_shipped"] == 0.0 and b["P_HII_shipped"] > 0.0:
                s_ship = abs(b["drive_shipped"] / a["drive_shipped"] - 1.0)
                s_k10 = abs(b["drive_K10_A"] / a["drive_K10_A"] - 1.0)
                worst_k10 = max(worst_k10, s_k10)
                print(
                    f"  {src} t={b['t']:.4f} ({b['phase']}): shipped step {s_ship*100:6.2f}%"
                    f"   K10 step {s_k10*100:6.2f}%"
                )
    print(
        f"  G13.1 {'PASS' if worst_k10 < 0.05 else 'FAIL'} (worst K10 step "
        f"{worst_k10*100:.2f}% vs 5% bar)"
    )

    # ---- G13.2 healthy branch ----
    b3m_ei = [r for r in allrows if "b7_" in r["src"] and r["phase"] in ("energy", "implicit")]
    m = med([r["drive_K10_A"] / r["drive_shipped"] for r in b3m_ei]) - 1.0
    print(
        f"\nG13.2 — B3M energy+implicit median K10A excess over shipped: {m*100:+.2f}%"
        f"   {'PASS' if m <= 0.15 else 'FAIL'} (bar 15%)"
    )

    # ---- G13.3 MD identity ----
    errs = [r["md_identity_relerr"] for r in allrows if r["md_identity_relerr"] is not None]
    print(
        f"\nG13.3 — MD identity worst relerr: {max(errs):.2e}   "
        f"{'PASS' if max(errs) < 1e-10 else 'FAIL'}"
        if errs
        else "G13.3 — VOID (no rows)"
    )

    # ---- G13.4 dust sensitivity + G13.5 magnitudes ----
    print(f"\nG13.4/G13.5 — per config per phase (medians):")
    print(
        f"  {'src':32s}{'phase':11s}{'n':>4}{'K10A/ship':>10}{'K10B/ship':>10}"
        f"{'A/B':>7}{'Ri/R2 A':>9}{'Ri/R2 B':>9}"
    )
    worst_ab = 0.0
    for src in sorted({r["src"] for r in allrows}):
        for ph in ("energy", "implicit", "transition", "momentum"):
            v = [r for r in allrows if r["src"] == src and r["phase"] == ph]
            if not v:
                continue
            a, b = med([r["K10A_over_shipped"] for r in v]), med(
                [r["K10B_over_shipped"] for r in v]
            )
            ab = a / b if (b and b == b) else float("nan")
            if ph == "momentum" and ab == ab:
                worst_ab = max(worst_ab, ab)
            print(
                f"  {src:32s}{ph:11s}{len(v):>4d}{a:>10.3f}{b:>10.3f}{ab:>7.2f}"
                f"{med([r['Ri_over_R2_A'] for r in v]):>9.2f}"
                f"{med([r['Ri_over_R2_B'] for r in v]):>9.2f}"
            )
    if worst_ab == worst_ab and worst_ab > 0:
        verdict = (
            "K10 CANNOT SHIP WITHOUT A DUST MODEL"
            if worst_ab > 2.0
            else "dust matters but is below the 2x bar"
        )
        print(f"\nG13.4 — momentum-phase A/B sensitivity: {worst_ab:.2f}x  ->  {verdict}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(
                "# Batch 13 K10 screen; gates + mapping pre-registered in PLAN.md "
                "before first run. Variant A: Q_eff = Qi*f_abs; B: *(1 - f_dust,ion)\n"
            )
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {args.out} ({len(allrows)} rows)")


if __name__ == "__main__":
    main()
