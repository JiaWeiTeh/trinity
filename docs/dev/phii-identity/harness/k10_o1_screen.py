#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 21 — K10-O1 screened offline: read the shell solve's own front instead of solving one.

Gates G21.0-G21.5 are pre-registered in PLAN.md (§Batch 21) and committed BEFORE this
script existed. This script only measures.

O1: rho = (shell_props.R_IF / R2)**2, everything else unchanged (ramped P_conf, Batch 16
composition mapping). `_k10_front_radius` is deleted.

The committed data carries the shell solve's front directly:
  B3M     data/b9_layer_density.csv          R_IF = R2 + dR_ion, shell outer = R2 + dR_full
  B3MW01  data/b12_lowwind_photon_ledger.csv R_IF = R2 + dR_ion_Pb (driving rows only)
joined to the trajectories for P_conf / P_ram / phase / shipped drive.

Comparators come from the SAME rows so the O1-vs-K10 difference is the front alone:
K10's own front is b17_dust_closure.csv's `Ri_dust`.

    python docs/dev/phii-identity/harness/k10_o1_screen.py \
        --out docs/dev/phii-identity/data/b21_o1_screen.csv
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

FIELDS = ["config", "phase", "t", "R2", "R_IF_shell", "shell_outer", "Ri_k10",
          "rho_o1", "rho_k10", "P_conf", "P_ram", "drive_o1", "drive_k10", "shipped_drive",
          "o1_over_shipped", "o1_over_k10", "m_implied", "shell_mass", "m_over_shell",
          "beyond_shell", "excess_over_conf"]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    return v[len(v) // 2] if v else float("nan")


def read(name):
    return list(csv.DictReader(l for l in open(DATA / name) if not l.startswith("#")))


def compose(phase, P_conf, P_HII, P_ram):
    if phase == "momentum":
        return P_HII + P_ram
    if phase == "transition":
        return max(P_conf, P_HII + P_ram)
    return max(P_conf, P_HII)


def mapped(phase, P_conf, rho, P_ram):
    if phase == "momentum":
        return P_ram * (rho - 1.0)
    if phase == "transition":
        return P_conf * rho - P_ram
    return P_conf * rho


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b21_o1_screen.csv")
    args = ap.parse_args()

    params = read_param(str(BENCH))
    mu = params["mu_convert"].value
    mu_i, mu_c = params["mu_ion_shell"].value, params["mu_convert"].value
    kB, T = params["k_B"].value, params["TShell_ion"].value
    coef = 4.0 / 3.0 * math.pi * params["nCore"].value * mu   # shell_mass(R2) slope, uniform cloud
    rCloud = (params["mCloud"].value / coef) ** (1.0 / 3.0)

    def shell_mass(R2):
        """Analytic for this uniform cloud; slice 2 validated the form to 4.4e-5."""
        return coef * R2**3 if R2 <= rCloud else params["mCloud"].value

    # K10's own front + state, keyed by (config, t)
    k10 = {(r["config"], round(fnum(r, "t"), 9)): r for r in read("b17_dust_closure.csv")
           if r.get("status") == "ok"}

    # the shell solve's front
    fronts = {}
    for r in read("b9_layer_density.csv"):
        if r.get("status") == "ok" and fnum(r, "dR_ion"):
            fronts[("B3M", round(fnum(r, "t"), 9))] = (
                fnum(r, "R2") + fnum(r, "dR_ion"),
                fnum(r, "R2") + fnum(r, "dR_full") if fnum(r, "dR_full") else None)
    for r in read("b12_lowwind_photon_ledger.csv"):
        d = fnum(r, "dR_ion_Pb")
        if d and fnum(r, "R2"):
            fronts[("B3MW01", round(fnum(r, "t"), 9))] = (fnum(r, "R2") + d, None)

    rows = []
    for key, kr in k10.items():
        f = fronts.get(key)
        if not f:
            continue
        R_IF, outer = f
        cfg, ph = kr["config"], kr["phase"]
        R2, Pc, Pr = fnum(kr, "R2"), fnum(kr, "P_conf"), fnum(kr, "P_ram") or 0.0
        Ri_k10, ship = fnum(kr, "Ri_dust"), fnum(kr, "shipped_drive")
        if None in (R2, Pc, R_IF) or R2 <= 0 or R_IF < R2:
            continue
        rho1, rhok = (R_IF / R2) ** 2, (Ri_k10 / R2) ** 2
        ret1 = mapped(ph, Pc, rho1, Pr)
        comp1 = compose(ph, Pc, ret1, Pr)
        n0 = (mu_i / mu_c) / (kB * T) * Pc
        m_imp = 4.0 / 3.0 * math.pi * (R_IF**3 - R2**3) * n0 * mu
        sm = shell_mass(R2)
        rows.append(dict(
            config=cfg, phase=ph, t=fnum(kr, "t"), R2=R2, R_IF_shell=R_IF,
            shell_outer=outer, Ri_k10=Ri_k10, rho_o1=rho1, rho_k10=rhok,
            P_conf=Pc, P_ram=Pr, drive_o1=Pc * rho1, drive_k10=Pc * rhok,
            shipped_drive=ship, o1_over_shipped=(comp1 / ship) if ship else None,
            o1_over_k10=rho1 / rhok, m_implied=m_imp, shell_mass=sm,
            m_over_shell=m_imp / sm if sm else None,
            beyond_shell=(R_IF > outer) if outer else None,
            excess_over_conf=rho1 - 1.0,
        ))

    print(f"{len(rows)} rows screened\n")

    print("G21.1 — is the front still outside the shell?  (K10 was 18/18 momentum, 43/44 energy)")
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r for r in rows if r["config"] == "B3M" and r["phase"] == ph
               and r["beyond_shell"] is not None]
        if sel:
            n = sum(1 for r in sel if r["beyond_shell"])
            print(f"    B3M {ph:11} {n:3d}/{len(sel):<3d} beyond the shell   "
                  f"-> {'PASS' if n == 0 else 'FAIL'}")
    allb = [r for r in rows if r["beyond_shell"] is not None]
    nb = sum(1 for r in allb if r["beyond_shell"])
    print(f"    overall {len(allb)-nb}/{len(allb)} inside -> "
          f"{'G21.1 PASS' if nb == 0 else 'G21.1 FAIL'}")

    print("\nG21.2 — seam C: implied layer mass vs the shell's own "
          "(K10 was median 0.4835 / max 2.4892 in B3M momentum)")
    for cfg in ("B3M", "B3MW01"):
        for ph in ("energy", "implicit", "transition", "momentum"):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph]
            if sel:
                over = sum(1 for r in sel if r["m_over_shell"] and r["m_over_shell"] > 1)
                print(f"    {cfg:8}{ph:11} n={len(sel):3d}  median "
                      f"{med([r['m_over_shell'] for r in sel]):8.4f}  max "
                      f"{max(r['m_over_shell'] for r in sel):8.4f}  rows>1: {over}")

    print("\nG21.3 — magnitude: drive/P_conf, and O1 vs K10 vs shipped")
    for cfg in ("B3M", "B3MW01"):
        for ph in ("energy", "implicit", "transition", "momentum"):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph]
            if sel:
                print(f"    {cfg:8}{ph:11} n={len(sel):3d}  O1 rho {med([r['rho_o1'] for r in sel]):7.3f}"
                      f"   K10 rho {med([r['rho_k10'] for r in sel]):7.3f}"
                      f"   O1/K10 {med([r['o1_over_k10'] for r in sel]):6.3f}"
                      f"   O1/shipped {med([r['o1_over_shipped'] for r in sel]):6.3f}")

    print("\nG21.4 — the confined branch: does the excess survive?")
    for cfg in ("B3M", "B3MW01"):
        sel = [r for r in rows if r["config"] == cfg and r["phase"] in ("energy", "implicit")]
        if sel:
            print(f"    {cfg:8} ED excess over P_conf: median "
                  f"{med([r['excess_over_conf'] for r in sel])*100:8.4f}%   "
                  f"max {max(r['excess_over_conf'] for r in sel)*100:.4f}%"
                  f"   (K10 gave +0.96% no-dust / +0.67% dusty)")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 21 (K10-O1): rho from the shell solve's own R_IF. Gates pre-registered\n")
        fh.write("# in PLAN.md SBatch-21 before this ran. Comparators are same-row K10 values.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
