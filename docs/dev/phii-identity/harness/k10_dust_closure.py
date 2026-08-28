#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 17 — dust INSIDE the K10 closure, as a reduction of trinity's own shell ODE.

Gates G17.0–G17.4 are pre-registered in PLAN.md (§Batch 17) and committed BEFORE this
script existed. This script only measures; no bar is moved.

Batch 13 fired G13.4 at 2.05x and ruled "K10 cannot ship without a dust model". Its dust
was a post-hoc JOIN (`f_dust` looked up from the photon ledgers at nearest t). This batch
puts dust in the closure itself, using the code's own ionised-region photon equation
(`get_shellODE.py:120`) at the closure's uniform density:

    dphi/dr = -(4 pi r^2 chi_e alpha_B n0^2 / Qi) - n0 sigma_d phi ,   phi(R2) = 1
    R_i := r where phi = 0
    n0   = (mu_i/mu_c) * P_conf / (k_B T)      == shell_structure.py:125's nShell0
    drive = P_conf * (R_i/R2)^2, composed through Batch 16's mapping

so it is a REDUCTION of trinity's dust treatment, not a new model. Qi is used WHOLE (not
Qi*f_abs): the shell solve starts at phi = 1 with the full budget, and the
recombination/dust/escape split is an output of the solve, not an input. A third ODE
variable D accumulates the dust sink, dD/dr = n0 sigma_d phi, so the closure's own
predicted dust fraction is D(R_i) -- that is what G17.0 tests against the run's measured
value.

Integration bracket: dust only makes phi reach 0 SOONER, so the no-dust closed-form radius
R_i_nodust is a guaranteed upper bound and [R2, R_i_nodust] always contains the root. The
event is therefore guaranteed to fire, which is why G17.1 is a real check rather than a
formality -- a `no_front` row means phi never reached 0 even without dust.

    python docs/dev/phii-identity/harness/k10_dust_closure.py \
        --out docs/dev/phii-identity/data/b17_dust_closure.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
FOUR_PI = 4.0 * math.pi
RTOL, ATOL = 1e-12, 1e-14

SOURCES = [
    ("B3M", "b7_regime_trajectory.csv", "b11_photon_ledger.csv"),
    ("B3MW01", "b12_lowwind_trajectory.csv", "b12_lowwind_photon_ledger.csv"),
]

FIELDS = [
    "config", "phase", "t", "R2", "P_conf", "P_ram", "n0", "tau_dust_layer",
    "Ri_nodust", "Ri_dust", "rho_nodust", "rho_dust",
    "f_dust_closure", "f_dust_measured", "f_dust_ratio",
    "drive_nodust", "drive_posthoc_B", "drive_selfconsistent",
    "composed_selfconsistent", "shipped_drive", "composed_over_shipped",
    "g172_sigma0_relerr", "status",
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
    rows = list(csv.DictReader(l for l in open(DATA / path) if not l.startswith("#")))
    return sorted((float(r["t"]), float(r["dust_Pb"])) for r in rows
                  if r.get("dust_Pb") not in (None, "", "None"))


def nearest(pts, t, tol=0.02):
    if not pts:
        return None
    best = min(pts, key=lambda p: abs(p[0] - t))
    return best[1] if abs(best[0] - t) <= tol else None


def ri_nodust(R2, Qi, n0, chi, aB):
    """Closed form: R_i^3 = R2^3 + 3 Qi / (4 pi chi_e alpha_B n0^2)."""
    return (R2**3 + 3.0 * Qi / (FOUR_PI * chi * aB * n0**2)) ** (1.0 / 3.0)


def solve_front(R2, Qi, n0, chi, aB, sigma_d):
    """Integrate the code's own ionised photon ODE at uniform n0; return (R_i, f_dust).

    y = [phi, D]  with  dphi/dr = -A r^2 - k phi,  dD/dr = k phi  (the dust sink).
    """
    A = FOUR_PI * chi * aB * n0**2 / Qi
    k = n0 * sigma_d
    hi = ri_nodust(R2, Qi, n0, chi, aB)

    def rhs(r, y):
        phi = max(0.0, y[0])
        return [-A * r * r - k * phi, k * phi]

    def hit_zero(r, y):
        return y[0]
    hit_zero.terminal = True
    hit_zero.direction = -1

    sol = solve_ivp(rhs, (R2, hi), [1.0, 0.0], events=hit_zero,
                    rtol=RTOL, atol=ATOL, dense_output=True, method="LSODA")
    if sol.t_events[0].size:
        r_i = float(sol.t_events[0][0])
        f_d = float(sol.y_events[0][0][1])
        return r_i, f_d, "ok"
    # phi never reached 0 inside a bracket that guarantees it without dust
    return float(sol.t[-1]), float(sol.y[1][-1]), "no_front"


def compose(phase, P_conf, P_HII, P_ram):
    """Batch 16's verified compositions (the real P_drive expressions at cce8c924)."""
    if phase == "momentum":
        return P_HII + P_ram
    if phase == "transition":
        return max(P_conf, P_HII + P_ram)
    return max(P_conf, P_HII)


def mapped(phase, P_conf, rho, P_ram):
    """Batch 16's mapping: P_conf*rho minus what the composition already adds."""
    if phase == "momentum":
        return P_ram * (rho - 1.0)
    if phase == "transition":
        return P_conf * rho - P_ram
    return P_conf * rho


def screen(config, traj, dust_pts, consts):
    mu_c, mu_i, kB, T, chi, aB, sigma_d = consts
    rows = [r for r in csv.DictReader(l for l in open(DATA / traj) if not l.startswith("#"))
            if r.get("arm") == "c3c"]
    out = []
    for r in rows:
        t, ph = fnum(r, "t_now"), r.get("current_phase")
        R2, Qi, fa = fnum(r, "R2"), fnum(r, "Qi"), fnum(r, "shell_fAbsorbedIon")
        Pb, Pram = fnum(r, "Pb"), fnum(r, "P_ram") or 0.0
        PH, Pd = fnum(r, "P_HII"), fnum(r, "P_drive")
        if None in (t, ph, R2, Qi, fa, Pb, PH, Pd) or not (R2 > 0 and Qi > 0):
            continue
        # G16.3: the RAMPED confining pressure, recovered where the run reveals it
        if ph in ("energy", "implicit"):
            P_conf = Pd if PH == 0.0 else Pb
        elif ph == "transition":
            P_conf = max(Pb, Pram)
        else:
            P_conf = Pram
        if not (P_conf > 0):
            continue

        n0 = (mu_i / mu_c) * P_conf / (kB * T)
        hi = ri_nodust(R2, Qi, n0, chi, aB)
        r_i, f_d, status = solve_front(R2, Qi, n0, chi, aB, sigma_d)

        # G17.2: sigma_d = 0 must reproduce the closed form
        r0, _, _ = solve_front(R2, Qi, n0, chi, aB, 0.0)
        g172 = abs(r0 / hi - 1.0)

        rho_nd, rho_d = (hi / R2) ** 2, (r_i / R2) ** 2
        f_meas = nearest(dust_pts, t)

        # Batch 13's post-hoc variant B, for the G17.3 comparison
        drive_B = None
        if f_meas is not None and 0.0 <= f_meas < 1.0:
            q_b = Qi * fa * (1.0 - f_meas)
            drive_B = P_conf * (ri_nodust(R2, q_b, n0, chi, aB) / R2) ** 2

        ret = mapped(ph, P_conf, rho_d, Pram)
        out.append(dict(
            config=config, phase=ph, t=t, R2=R2, P_conf=P_conf, P_ram=Pram, n0=n0,
            tau_dust_layer=n0 * sigma_d * (r_i - R2),
            Ri_nodust=hi, Ri_dust=r_i, rho_nodust=rho_nd, rho_dust=rho_d,
            f_dust_closure=f_d, f_dust_measured=f_meas,
            f_dust_ratio=(f_d / f_meas) if (f_meas and f_meas > 0) else None,
            drive_nodust=P_conf * rho_nd, drive_posthoc_B=drive_B,
            drive_selfconsistent=P_conf * rho_d,
            composed_selfconsistent=compose(ph, P_conf, ret, Pram),
            shipped_drive=Pd,
            composed_over_shipped=(compose(ph, P_conf, ret, Pram) / Pd) if Pd > 0 else None,
            g172_sigma0_relerr=g172, status=status,
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b17_dust_closure.csv")
    args = ap.parse_args()

    params = read_param(str(
        REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
        "bench3_m1e5_r5__none_diag.param"))
    consts = (params["mu_convert"].value, params["mu_ion_shell"].value,
              params["k_B"].value, params["TShell_ion"].value,
              params["chi_e_shell"].value, params["caseB_alpha"].value,
              params["dust_sigma"].value)
    print(f"sigma_dust = {consts[-1]:.4e} pc^2   (code units)")

    rows = []
    for cfg, traj, ledger in SOURCES:
        rows += screen(cfg, traj, load_dust(ledger), consts)
    print(f"{len(rows)} rows screened over {len({r['config'] for r in rows})} configs")

    # ---- G17.1 convergence ----
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\nG17.1 front convergence: {len(rows)-len(bad)}/{len(rows)} -> "
          f"{'PASS' if not bad else 'FAIL'}")
    if bad:
        print(f"    no_front rows: {len(bad)}, phases {sorted({r['phase'] for r in bad})}")

    # ---- G17.2 sigma_d -> 0 ----
    errs = [r["g172_sigma0_relerr"] for r in rows if r["g172_sigma0_relerr"] is not None]
    worst = max(errs) if errs else float("nan")
    print(f"\nG17.2 sigma_d=0 recovers the closed form: worst rel err {worst:.2e} vs 1e-10 -> "
          f"{'PASS' if worst <= 1e-10 else 'FAIL'}")

    # ---- G17.0 dust fraction vs the shell solve's own (BLOCKING) ----
    cmp_rows = [r for r in rows if r["f_dust_ratio"] is not None and r["status"] == "ok"]
    print(f"\nG17.0 closure dust vs measured (n={len(cmp_rows)} rows with a measured value)")
    if cmp_rows:
        ratios = [r["f_dust_ratio"] for r in cmp_rows]
        m = med(ratios)
        within25 = sum(1 for x in ratios if abs(x - 1.0) <= 0.25) / len(ratios)
        print(f"    predicted/measured: min {min(ratios):.3f} median {m:.3f} max {max(ratios):.3f}")
        print(f"    within 25%: {within25*100:.1f}% of rows")
        print(f"    closure f_dust median {med([r['f_dust_closure'] for r in cmp_rows]):.4f} "
              f"vs measured {med([r['f_dust_measured'] for r in cmp_rows]):.4f}")
        print(f"    G17.0 {'PASS' if 0.5 <= m <= 2.0 else 'FAIL'} "
              f"(median {m:.3f} vs the [0.5, 2.0] bar)")
        for cfg in sorted({r["config"] for r in cmp_rows}):
            sub = [r["f_dust_ratio"] for r in cmp_rows if r["config"] == cfg]
            print(f"      {cfg:8} n={len(sub):3d} median {med(sub):.3f}")

    # ---- G17.3 sensitivity now internal ----
    print("\nG17.3 drive under (a) no dust, (b) Batch 13 post-hoc, (c) self-consistent:")
    for cfg in sorted({r["config"] for r in rows}):
        for ph in ("energy", "implicit", "transition", "momentum"):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph
                   and r["drive_posthoc_B"] and r["status"] == "ok"]
            if not sel:
                continue
            a = med([r["drive_nodust"] / r["P_conf"] for r in sel])
            b = med([r["drive_posthoc_B"] / r["P_conf"] for r in sel])
            c = med([r["drive_selfconsistent"] / r["P_conf"] for r in sel])
            between = min(a, b) <= c <= max(a, b)
            print(f"    {cfg:8}{ph:11} n={len(sel):3d}  a {a:8.3f}  b {b:8.3f}  c {c:8.3f}"
                  f"   c between a,b: {between}   a/c = {a/c:.3f}")

    # ---- G17.4 end-to-end magnitude ----
    print("\nG17.4 composed drive (Batch 16 mapping) / shipped, median:")
    for cfg in sorted({r["config"] for r in rows}):
        for ph in ("energy", "implicit", "transition", "momentum"):
            sel = [r["composed_over_shipped"] for r in rows if r["config"] == cfg
                   and r["phase"] == ph and r["composed_over_shipped"] and r["status"] == "ok"]
            if sel:
                print(f"    {cfg:8}{ph:11} n={len(sel):3d}  {med(sel):7.3f}")

    tau = [r["tau_dust_layer"] for r in rows if r["status"] == "ok"]
    print(f"\ndust optical depth across the closure layer: min {min(tau):.3f} "
          f"median {med(tau):.3f} max {max(tau):.3f}")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 17: dust inside the K10 closure, as a uniform-density reduction of\n")
        fh.write("# get_shellODE.py:120. Gates pre-registered in PLAN.md SBatch-17 before this ran.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
