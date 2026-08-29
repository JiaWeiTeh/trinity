#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does K10's confinement requirement bite INSIDE trinity's own regime? — 2026-08-29.

Maintainer's challenge to Batch 20's CRITICAL finding: *"what if a photoionised limit
cannot be reached in our Weaver-like wind-driven bubble?"* — i.e. `P_conf = 0` never
happens in trinity (the thermal `Pb` in energy/implicit, `max(P_thermal, P_ram)` in
transition, `P_ram` in momentum are all strictly positive), so a test fixture that turns
the wind off is artificial and the limit failure may be irrelevant.

That is correct about the LIMIT. This harness asks the question it leaves open: the CEM
closure assumes the photoionised gas is CONFINED by the wind bubble, and its drive
diverges as `P_conf^(-1/3)`. Even if the endpoint is unreachable, **how close does trinity
get, and is the model already outside its own domain of validity at wind strengths the
matrix actually runs?**

Three domain tests, all on committed data, no runs:
  D1  R_i vs the shell's own outer edge (R2 + dR_full). If K10's ionisation front sits
      beyond the shell the ODE produced, the "ionised inner layer of the shell" picture
      has failed -- there is no shell out there to be the layer of.
  D2  R_i vs rCloud. Beyond the cloud there is no cloud gas to ionise, only ISM.
  D3  the P_conf^(-1/3) trend across the wind ladder, i.e. how fast the model runs toward
      its own singularity as the wind weakens.

Inputs: data/b17_dust_closure.csv (R_i, R2, P_conf, phase, config) joined on t to
data/b9_layer_density.csv (dR_full, B3M only). rCloud from each config's param.

    python docs/dev/phii-identity/harness/k10_domain_check.py \
        --out docs/dev/phii-identity/data/b20_domain.csv
"""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
BENCH = REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/bench3_m1e5_r5__none_diag.param"

FIELDS = ["config", "phase", "t", "R2", "R_i", "Ri_over_R2", "shell_outer", "Ri_over_shell",
          "rCloud", "Ri_over_rCloud", "P_conf", "drive_over_Pconf",
          "beyond_shell", "beyond_cloud"]


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
    ap.add_argument("--out", type=Path, default=DATA / "b20_domain.csv")
    args = ap.parse_args()

    params = read_param(str(BENCH))
    # rCloud is a derived_init quantity and is 0 in the raw param, so derive it from the
    # code's own unit-converted values. densPL_alpha = 0 => uniform cloud, so
    # mCloud = (4/3) pi rCloud^3 nCore mu_convert. The coefficient reproduces the
    # shell_mass(R2) slope Batch 20 slice 2 measured independently (800.465 vs 800.5).
    import math as _m
    coef = 4.0 / 3.0 * _m.pi * params["nCore"].value * params["mu_convert"].value
    rCloud = (params["mCloud"].value / coef) ** (1.0 / 3.0)
    print(f"rCloud (bench3_m1e5_r5, shared by B3M and B3MW01) = {rCloud:.4f} pc "
          f"[derived; coef {coef:.3f} vs slice 2's independently measured 800.5]\n")

    # B3M shell thickness, joined on t (nearest, tight tol) from the layer-density replay
    lay = [r for r in csv.DictReader(l for l in open(DATA / "b9_layer_density.csv")
                                     if not l.startswith("#")) if r.get("status") == "ok"]
    lay_pts = sorted((fnum(r, "t"), fnum(r, "R2"), fnum(r, "dR_full")) for r in lay
                     if fnum(r, "dR_full") is not None)

    def shell_outer(t, R2):
        if not lay_pts:
            return None
        best = min(lay_pts, key=lambda p: abs(p[0] - t))
        if abs(best[0] - t) > 1e-4 * max(t, 1e-12):
            return None
        return best[1] + best[2]

    rows = []
    for r in csv.DictReader(l for l in open(DATA / "b17_dust_closure.csv")
                            if not l.startswith("#")):
        if r.get("status") != "ok":
            continue
        cfg, ph, t = r["config"], r["phase"], fnum(r, "t")
        R2, Ri, Pc = fnum(r, "R2"), fnum(r, "Ri_dust"), fnum(r, "P_conf")
        if None in (R2, Ri, Pc) or R2 <= 0:
            continue
        so = shell_outer(t, R2) if cfg == "B3M" else None
        rows.append(dict(
            config=cfg, phase=ph, t=t, R2=R2, R_i=Ri, Ri_over_R2=Ri / R2,
            shell_outer=so, Ri_over_shell=(Ri / so) if so else None,
            rCloud=rCloud, Ri_over_rCloud=Ri / rCloud, P_conf=Pc,
            drive_over_Pconf=(Ri / R2) ** 2,
            beyond_shell=(Ri > so) if so else None,
            beyond_cloud=(Ri > rCloud),
        ))

    print("D1 — is K10's ionisation front beyond the shell the ODE actually produced?")
    for ph in ("energy", "implicit", "transition", "momentum"):
        sel = [r for r in rows if r["config"] == "B3M" and r["phase"] == ph
               and r["beyond_shell"] is not None]
        if sel:
            n = sum(1 for r in sel if r["beyond_shell"])
            print(f"    B3M {ph:11} {n:3d}/{len(sel):<3d} rows beyond the shell   "
                  f"median R_i/shell_outer {med([r['Ri_over_shell'] for r in sel]):.3f}")

    print("\nD2 — is it beyond the cloud itself?")
    for cfg in ("B3M", "B3MW01"):
        for ph in ("transition", "momentum"):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph]
            if sel:
                n = sum(1 for r in sel if r["beyond_cloud"])
                print(f"    {cfg:8}{ph:11} {n:3d}/{len(sel):<3d} beyond rCloud   "
                      f"median R_i/rCloud {med([r['Ri_over_rCloud'] for r in sel]):.2f}"
                      f"   max R_i {max(r['R_i'] for r in sel):.1f} pc")

    print("\nD3 — how fast does the model run toward its own singularity?")
    for cfg in ("B3M", "B3MW01"):
        sel = [r for r in rows if r["config"] == cfg and r["phase"] == "momentum"]
        if sel:
            print(f"    {cfg:8} median drive/P_conf {med([r['drive_over_Pconf'] for r in sel]):7.3f}"
                  f"   median R_i/R2 {med([r['Ri_over_R2'] for r in sel]):.3f}")
    b = med([r["drive_over_Pconf"] for r in rows if r["config"] == "B3M"
             and r["phase"] == "momentum"])
    w = med([r["drive_over_Pconf"] for r in rows if r["config"] == "B3MW01"
             and r["phase"] == "momentum"])
    if b and w:
        print(f"    ratio across one decade of wind: {w/b:.3f}  vs P_conf^(-1/3) prediction "
              f"{10**(1/3):.3f}")
        print(f"    EXTRAPOLATED to B3MW001 (Lw x0.01, a REGISTERED config, never run for K10):")
        print(f"      drive/P_conf ~ {b*10**(2/3):.1f},  R_i/R2 ~ {(b*10**(2/3))**0.5:.2f}, "
              f"i.e. R_i ~ {(b*10**(2/3))**0.5 * 7.7:.0f} pc against rCloud {rCloud:.1f} pc")

    # ---- What would each candidate FIX cost? All on B3M momentum, committed data. ----
    # The unbounded front comes from integrating outward at a fixed low n0 with no
    # knowledge of where gas ends. Four distinct remedies, priced here so the choice is
    # made on numbers rather than on adjectives.
    print("\nOPTIONS — drive/P_conf on B3M momentum under each candidate remedy:")
    lay_m = [r for r in lay if r["phase"] == "momentum"]
    k10_m = [r for r in rows if r["config"] == "B3M" and r["phase"] == "momentum"]
    if lay_m and k10_m:
        rif = med([1.0 + fnum(r, "dR_ion") / fnum(r, "R2") for r in lay_m])       # shell's own front
        rsh = med([1.0 + fnum(r, "dR_full") / fnum(r, "R2") for r in lay_m])      # shell outer edge
        nrat = med([fnum(r, "n_rms_profile") / fnum(r, "n_cavity") for r in lay_m
                    if fnum(r, "n_rms_profile") and fnum(r, "n_cavity")])
        k10rho = med([r["drive_over_Pconf"] for r in k10_m])
        # 7.0953 is the TRUE median of C3c's composed drive / P_ram over B3M's 34 momentum
        # rows in b7_regime_trajectory.csv. NOTE b14_identity_census.csv reports 7.164582 for
        # the same quantity, because every harness here uses med(v) = sorted(v)[len(v)//2] --
        # an upper order statistic, not a statistical median. On even-length row sets the two
        # differ by ~1%. Both numbers circulate in PLAN.md for this quantity; see the
        # 2026-08-29 housekeeping entry.
        shipped = 7.095
        print(f"    shipped C3c (for reference)                      {shipped:7.3f}")
        print(f"    K10 as implemented                               {k10rho:7.3f}"
              f"   (R_i/R2 {k10rho**0.5:.3f})")
        print(f"    O1  use the shell solve's OWN front R_IF         {rif**2:7.3f}"
              f"   (R_IF/R2 {rif:.3f})  cut x{k10rho/rif**2:.2f}")
        print(f"    O2  clip at the shell's outer edge               {rsh**2:7.3f}"
              f"   (edge/R2 {rsh:.3f})  cut x{k10rho/rsh**2:.2f}")
        print(f"    O3  cap by available mass (slice 2)                 5.490"
              f"   (R_i/R2 2.343)  cut x{k10rho/5.490:.2f} at the median,"
              f" x1.78 at t=1.5")
        print(f"    O4  keep the solve, fix the DENSITY               (no separate number)")
        print(f"        K10's uniform n0 sits at n0/n_rms = 0.611 (slice 2), and R_i shrinks")
        print(f"        as n^(-2/3), so using the real profile IS O1 computed exactly.")
        print(f"        O4 is the MECHANISM behind O1, not an independent option.")
        print(f"\n    O1 and O2 agree to 0.5% because the momentum shell is 99.5% ionised")
        print(f"    (B11.0: dR_ion/dR_full = 0.9954), so 'the front' and 'the shell edge'")
        print(f"    are nearly the same place. The O1/O2 choice is principle, not magnitude.")

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 20 follow-up: is K10 outside its own domain of validity INSIDE\n")
        fh.write("# trinity's wind-driven regime? Answers the maintainer's challenge that\n")
        fh.write("# P_conf = 0 is unreachable, so the photo-only limit may not matter.\n")
        w_ = csv.DictWriter(fh, fieldnames=FIELDS)
        w_.writeheader()
        w_.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
