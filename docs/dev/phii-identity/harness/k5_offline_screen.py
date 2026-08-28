#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 14 — K5 offline screen: the layer-volume denominator on committed data.

Gates G14.0–G14.5 are pre-registered in PLAN.md (§Batch 14), committed 2026-08-27
(`d234e945`) BEFORE this script existed. This script only measures; no bar is moved.
Offline scope: G14.0 (decoupling regression, BLOCKING) and G14.3 (magnitude, no bar)
are computable from committed CSVs; G14.1/G14.2/G14.4/G14.5 need the live helper or
an arm and are deferred to implementation time.

K5 swaps `get_phii_c3c`'s recombination volume from the wind cavity (4/3)πR2³ to the
cavity-excluded ionised layer (Batch 14, conventions pinned there):
    K5a (analytic layer):  n = sqrt(3 Qi_abs / (4π χ_e α_B (R_IF³ − R2³)))
    K5b (profile):         n = n_rms over the ionised layer from the shipped solve
Volumes are EXACT spherical shells (B11.0 S1: the thin-shell 4πR²dR form overstates
the analytic density 1.34–1.70× in momentum, so `b9_layer_density.csv`'s
`n_layer_analytic` is NOT reused; K5a is recomputed as n_cavity·sqrt(R2³/(R_IF³−R2³))
with R_IF = R2 + dR_ion, both columns from the same replay row). χ_e is carried
explicitly by the consts; R_IF is rShell_arr_ion[-1] (what dR_ion measures from).

Inputs (all committed; no run dirs needed):
  B3M leg   — data/b9_layer_density.csv   (dR_ion, n_cavity, n_rms_profile; stride-2
              replay of the b9 B3M run) joined on row_idx to
              data/b11_mass_ledger.csv    (Pb, P_HII, Qi, f_abs; stride-2 replay of
              B11.0's fresh B3M run @ ef624195). DISCLOSED: two different B3M
              realisations — B11.0 measured their agreement at ≤3.3e-6 rel on the
              shared columns; this join re-verifies |Δt| and |ΔR2| per row and stops
              at 1e-4.
  B3MW01 leg — data/b12_lowwind_photon_ledger.csv alone (R2, Qi, Pb, P_HII,
              dR_ion_Pb on the replayed driving rows). K5a only: no committed n_rms
              exists for B3MW01, so the K5b leg is B3M-only — recorded as a coverage
              cap, not silently.

Pressure reconstruction: P = (mu_convert/mu_ion_shell)·n·k_B·TShell_ion, consts read
from the B3M param (shared by B3MW01 — only Lw differs). Route check: reconstructed
P_C3a must match the stored P_HII on every shipped-driving row.

G14.0 (BLOCKING, from the gate text): regress P_HII on Pb over driving rows;
FAIL if slope ∈ [0.95, 1.05] with r² > 0.99 — the old P_HII ≡ P_conf identity
returning. Reported per variant; log-log slope alongside as the Batch 3b diagnostic.

    python docs/dev/phii-identity/harness/k5_offline_screen.py \
        --out docs/dev/phii-identity/data/b14_k5_screen.csv
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
JOIN_TOL = 1e-4  # rel; B11.0 measured the two B3M realisations agree to <=3.3e-6

FIELDS = [
    "config", "row_idx", "phase", "t", "R2", "Pb", "P_HII_shipped",
    "dR_ion", "vol_cav_over_layer", "n_cavity", "n_k5a", "n_rms_profile",
    "P_c3a", "P_k5a", "P_k5b",
    "c3a_over_pb", "k5a_over_pb", "k5b_over_pb", "branch_change",
]


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def read_csv(name):
    with open(DATA / name) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def ols(x, y):
    """Plain OLS y = a·x + b; returns (slope, intercept, r2). The gate's bar is on this."""
    n = len(x)
    if n < 3:
        return None, None, None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    syy = sum((yi - my) ** 2 for yi in y)
    if not (sxx > 0 and syy > 0):
        return None, None, None
    a = sxy / sxx
    return a, my - a * mx, (sxy * sxy) / (sxx * syy)


def med(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return float("nan")
    return v[len(v) // 2]


def vol_ratio(R2, dR_ion):
    """V_cavity / V_layer with EXACT spherical volumes: R2^3 / ((R2+dR)^3 - R2^3)."""
    if not (R2 > 0 and dR_ion > 0):
        return None
    lay = (R2 + dR_ion) ** 3 - R2**3
    return (R2**3 / lay) if lay > 0 else None


def b3m_rows(pref):
    lay = {int(r["row_idx"]): r for r in read_csv("b9_layer_density.csv")
           if r.get("status") == "ok"}
    led = {int(r["row_idx"]): r for r in read_csv("b11_mass_ledger.csv")
           if r.get("status") == "ok"}
    out, worst_dt, worst_dr = [], 0.0, 0.0
    for idx in sorted(set(lay) & set(led)):
        L, M = lay[idx], led[idx]
        t_l, t_m = fnum(L, "t"), fnum(M, "t")
        r_l, r_m = fnum(L, "R2"), fnum(M, "R2")
        if None in (t_l, t_m, r_l, r_m):
            continue
        dt, dr = abs(t_l / t_m - 1.0), abs(r_l / r_m - 1.0)
        worst_dt, worst_dr = max(worst_dt, dt), max(worst_dr, dr)
        if dt > JOIN_TOL or dr > JOIN_TOL or L["phase"] != M["phase"]:
            sys.exit(f"join mismatch at row_idx {idx}: dt={dt:.2e} dr={dr:.2e} "
                     f"phases {L['phase']}/{M['phase']} — the two B3M realisations "
                     "have diverged; re-derive the join before trusting anything")
        n_cav, n_rms = fnum(L, "n_cavity"), fnum(L, "n_rms_profile")
        dR_ion = fnum(L, "dR_ion")
        Pb, PH = fnum(M, "Pb"), fnum(M, "P_HII")
        if None in (n_cav, dR_ion, Pb, PH):
            continue
        vr = vol_ratio(r_l, dR_ion)
        n_k5a = n_cav * math.sqrt(vr) if vr else None
        out.append(dict(
            config="B3M", row_idx=idx, phase=L["phase"], t=t_l, R2=r_l, Pb=Pb,
            P_HII_shipped=PH, dR_ion=dR_ion, vol_cav_over_layer=vr,
            n_cavity=n_cav, n_k5a=n_k5a, n_rms_profile=n_rms,
            P_c3a=pref * n_cav,
            P_k5a=(pref * n_k5a) if n_k5a else None,
            P_k5b=(pref * n_rms) if n_rms else None,
        ))
    print(f"B3M join: {len(out)} rows, worst |dt| {worst_dt:.2e}, "
          f"worst |dR2| {worst_dr:.2e} (tol {JOIN_TOL:.0e})")
    return out


def b3mw01_rows():
    """K5a only, driving rows only — the photon ledger replays no confined rows and
    commits no n_rms, so this leg cannot see confined-branch flips or K5b."""
    out = []
    for r in read_csv("b12_lowwind_photon_ledger.csv"):
        Pb, PH = fnum(r, "Pb"), fnum(r, "P_HII")
        R2, dR = fnum(r, "R2"), fnum(r, "dR_ion_Pb")
        t = fnum(r, "t")
        if None in (Pb, PH, R2, dR, t) or PH <= 0:
            continue
        vr = vol_ratio(R2, dR)
        if vr is None:
            continue
        out.append(dict(
            config="B3MW01", row_idx=int(r["row_idx"]), phase=r["phase"], t=t,
            R2=R2, Pb=Pb, P_HII_shipped=PH, dR_ion=dR, vol_cav_over_layer=vr,
            n_cavity=None, n_k5a=None, n_rms_profile=None,
            P_c3a=PH, P_k5a=PH * math.sqrt(vr), P_k5b=None,
        ))
    print(f"B3MW01 leg: {len(out)} driving rows (K5a only; no committed n_rms)")
    return out


def g14_0(rows, config, variant, key):
    """The pre-registered regression. Driving rows = shipped P_HII > 0."""
    pts = [(r["Pb"], r[key]) for r in rows
           if r["config"] == config and r["P_HII_shipped"] > 0 and r.get(key)]
    if len(pts) < 3:
        print(f"  {config:7s} {variant:4s}: <3 driving rows with data — VOID")
        return
    x, y = [p[0] for p in pts], [p[1] for p in pts]
    a, b, r2 = ols(x, y)
    la, _, lr2 = ols([math.log10(v) for v in x], [math.log10(v) for v in y])
    fail = a is not None and 0.95 <= a <= 1.05 and r2 > 0.99
    print(f"  {config:7s} {variant:4s}: N={len(pts):3d}  slope {a:+.4f}  r2 {r2:.4f}"
          f"  (log-log slope {la:+.3f}, r2 {lr2:.3f})  -> "
          f"{'FAIL — identity returned' if fail else 'pass'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=DATA / "b14_k5_screen.csv")
    args = ap.parse_args()

    params = read_param(str(
        REPO / "docs/dev/transition/pdv-trigger/runs/params/bench5/"
        "bench3_m1e5_r5__none_diag.param"))
    pref = (params["mu_convert"].value / params["mu_ion_shell"].value
            * params["k_B"].value * params["TShell_ion"].value)

    rows = b3m_rows(pref)

    # Route check: reconstructed P_c3a must be the stored P_HII on driving rows.
    drv = [r for r in rows if r["P_HII_shipped"] > 0]
    worst = max(abs(r["P_c3a"] / r["P_HII_shipped"] - 1.0) for r in drv)
    print(f"route check: reconstructed P_c3a vs stored P_HII on {len(drv)} driving "
          f"rows, worst rel err {worst:.2e}")
    if worst > 1e-6:
        sys.exit("route check failed — the consts do not reproduce the shipped "
                 "P_HII; nothing downstream is trustworthy")

    rows += b3mw01_rows()

    for r in rows:
        pb = r["Pb"]
        r["c3a_over_pb"] = r["P_c3a"] / pb if (r["P_c3a"] and pb > 0) else None
        r["k5a_over_pb"] = r["P_k5a"] / pb if (r["P_k5a"] and pb > 0) else None
        r["k5b_over_pb"] = r["P_k5b"] / pb if (r["P_k5b"] and pb > 0) else None
        ship = r["P_HII_shipped"] > 0
        flips = []
        for tag, key in (("k5a", "k5a_over_pb"), ("k5b", "k5b_over_pb")):
            v = r.get(key)
            if v is None:
                continue
            if ship and v <= 1.0:
                flips.append(f"{tag}:driving->confined")
            if not ship and v > 1.0:
                flips.append(f"{tag}:confined->DRIVING")
        r["branch_change"] = ";".join(flips)

    print("\nG14.0 — decoupling regression, P on Pb over shipped-driving rows")
    print("  (FAIL bar, pre-registered: slope in [0.95, 1.05] AND r2 > 0.99)")
    for config in ("B3M", "B3MW01"):
        for variant, key in (("c3a", "P_c3a"), ("K5a", "P_k5a"), ("K5b", "P_k5b")):
            g14_0(rows, config, variant, key)

    print("\nG14.3 — magnitude, median P_X/Pb per phase (Pb IS P_ram in momentum):")
    for config in ("B3M", "B3MW01"):
        for ph in ("energy", "implicit", "transition", "momentum"):
            sel = [r for r in rows if r["config"] == config and r["phase"] == ph]
            if not sel:
                continue
            print(f"  {config:7s} {ph:10s} N={len(sel):3d}  "
                  f"shipped {med([r['c3a_over_pb'] for r in sel]):8.3f}  "
                  f"K5a {med([r['k5a_over_pb'] for r in sel]):8.3f}  "
                  f"K5b {med([r['k5b_over_pb'] for r in sel]):8.3f}")

    print("\nbranch-structure census (K5 changes which rows drive):")
    for config in ("B3M", "B3MW01"):
        sel = [r for r in rows if r["config"] == config]
        for tag in ("k5a", "k5b"):
            dc = sum(1 for r in sel if f"{tag}:driving->confined" in r["branch_change"])
            cd = sum(1 for r in sel if f"{tag}:confined->DRIVING" in r["branch_change"])
            n_d = sum(1 for r in sel if r["P_HII_shipped"] > 0)
            n_c = len(sel) - n_d
            print(f"  {config:7s} {tag}: {dc}/{n_d} driving rows flip confined, "
                  f"{cd}/{n_c} confined rows flip DRIVING")
    cd_rows = [r for r in rows if "DRIVING" in r["branch_change"]]
    if cd_rows:
        ph_t = sorted((r["phase"], r["t"]) for r in cd_rows)
        print(f"  confined->DRIVING rows sit at: {ph_t[0][0]} t={ph_t[0][1]:.3e} "
              f"... {ph_t[-1][0]} t={ph_t[-1][1]:.3e}")

    print("\nG14.1 (denominator == shell_structure's _vol_ion to 1e-12): needs the "
          "live helper — deferred to the arm. G14.2/G14.4/G14.5: arm-time gates.")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write("# Batch 14 K5 offline screen; gates pre-registered in PLAN.md "
                     "SBatch-14 (d234e945) before this ran.\n")
            fh.write("# B3M: b9_layer_density x b11_mass_ledger join on row_idx "
                     "(two realisations, agreement <=3.3e-6 per B11.0). "
                     "B3MW01: b12_lowwind_photon_ledger driving rows, K5a only.\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
