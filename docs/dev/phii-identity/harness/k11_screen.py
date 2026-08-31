#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 22 stage 1 — K11, the Geen 2019 additive closure, screened offline.

Gates G22.1-G22.5 are pre-registered in PLAN.md (SBatch-22) and were committed BEFORE
this script existed. This script only measures; no bar is moved here.

THE CLOSURE (Geen 2019, translated to trinity coordinates; r_i -> the shell solve's R_IF)

    recombination   (4 pi/3) chi_e alpha_B n^2 (r_i^3 - r_w^3) = Q_eff
    balance         pdot_w / (4 pi r_w^2) = pref * n
    eliminate r_w   f(n) = n^2 R_IF^3 - C_W n^(1/2) - C_Q = 0
                    C_W = (pdot_w / (4 pi pref))^(3/2)    C_Q = 3 Q_eff / (4 pi chi_e alpha_B)
    drive at R2     P_K11 = pref * n * (R_IF/R2)^2        (Lancaster force-at-front, O1's convention)

The elimination was re-derived symbolically this session (sympy, exact zero) rather than
taken from the stage-0 note.

BRACKET, proved rather than tuned. With R3 = R_IF^3, n_Str = sqrt(C_Q/R3) (photon-only
root) and n_w = (C_W/R3)^(2/3) (wind-only root):
  * f(n_Str) = -C_W sqrt(n_Str) <= 0 and f(n_w) = -C_Q <= 0, so max(n_Str, n_w) <= n*.
  * f(n_Str + n_w) >= 0, because with sqrt(a+b) <= sqrt(a)+sqrt(b) the residual reduces to
    n_w + 2 n_Str >= sqrt(n_w) sqrt(n_Str + n_w), whose squares differ by 3 n_Str n_w +
    4 n_Str^2 >= 0.
So [max(n_Str,n_w), n_Str+n_w] always brackets, and a small multiplicative pad keeps the
sign change strict in the degenerate limits G22.1 exercises (one term zeroed, where the
bracket collapses onto the analytic answer).

IDENTITY WORTH KNOWING (used by G22.4). At the root, C_W n^(-3/2) = R_IF^3 - C_Q/n^2, i.e.
    r_w^3 = R_IF^3 - C_Q / n^2 ,
so r_w < R_IF *identically* whenever Q_eff > 0, and the closure's own layer mass is
    m_layer = (4 pi/3)(R_IF^3 - r_w^3) n mu  ==  mu Q_eff / (chi_e alpha_B n)   [exact]
i.e. the recombination-balanced mass. The right-hand form is what this harness uses, because
the left-hand one CANCELS CATASTROPHICALLY exactly where G22.4 is barred: measured this
session, `(R_IF^3 - r_w^3)` computed directly is wrong by up to 7.2e-4 relative on 2000
random draws (worst at r_w/R_IF = 1.000000000000, layer volume 5.5e-13 of R_IF^3) while the
root itself is exact there (scaled |f(n)| = 2.1e-16). The same cancellation makes the
"r_w >= r_i" flag FLOAT-REACHABLE in the wind-dominated corner -- 377 of 2000 draws with
pdot_w in [1e0,1e6] and Q_eff in [1e-12,1e-6] -- which is almost certainly what stage 0's
"34/2000 draws are r_w >= r_i" was measuring. It is a floating-point artefact, not a
physical out-of-domain condition, and the report says so rather than passing a zero (or a
non-zero) count off as a clean measurement.

INPUTS (all committed; C-6 stamps checked by eye, no run performed)
  b17_dust_closure.csv    anchor: config/phase/t/R2 + the RAMPED P_conf and P_ram (G16.3).
                          Batch 21's O1 screen used the same anchor, so the O1 comparator in
                          G22.3 comes from the SAME rows and the K11-vs-O1 difference is the
                          closure alone. Carries G18.0's caveat: P_conf is RECOVERED from
                          stored columns, exact in implicit/momentum, <=0.59% transition,
                          <=6.8% on 2 of 156 energy rows.
  B3M     b9_layer_density x b11_mass_ledger x b11_photon_ledger on row_idx (guarded)
          -> R_IF = R2 + dR_ion, Qi, f_ionised_dust, shell_mass, Pb, shipped P_HII
  B3MW01  b12_lowwind_photon_ledger x b12_lowwind_mass_ledger on row_idx
          -> R_IF = R2 + dR_ion_Pb (DRIVING ROWS ONLY -- 27 rows, transition+momentum; the
             low-wind ledger replayed no confined rows, so this leg cannot see the confined
             branch at all. Same coverage cap Batch 21 hit.)
  consts  read_param on the bench param: pref = (mu_c/mu_i) k_B T, explicit chi_e, alpha_B.

Median convention: statistics.median, declared in the batch registration -- NOT the med()
upper order statistic every other harness here uses (2026-08-29 housekeeping finding).

    python docs/dev/phii-identity/harness/k11_screen.py \
        --out docs/dev/phii-identity/data/b22_k11_screen.csv
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trinity._input.read_param import read_param  # noqa: E402

from _stamp import stamp  # noqa: E402

DATA = REPO / "docs/dev/phii-identity/data"
BENCH = REPO / ("docs/dev/transition/pdv-trigger/runs/params/bench5/"
                "bench3_m1e5_r5__none_diag.param")

FOUR_PI = 4.0 * math.pi
JOIN_TOL = 1e-4     # rel; the B3M three-way join measures 1.5e-7 (k5_offline_screen's bar)
ANCHOR_TOL = 2e-2   # rel in t, for the nearest-t join onto b17 (k10_o1_screen's tolerance)
PAD = 1e-9          # multiplicative bracket pad, so the degenerate limits still sign-change
G221_BAR = 1e-10    # the pre-registered blocking bar

VARIANTS = (("q", "Qi"), ("qd", "Qi*(1-f_dust)"))

FIELDS = [
    "config", "phase", "t", "R2", "R_IF", "rho_o1", "Qi", "f_dust", "shell_mass",
    "Pb", "P_conf", "P_ram", "pdot_w", "P_HII_shipped", "shipped_drive", "n0",
    "drive_o1", "g221_stromgren_rel", "g221_wind_rel", "g221_wind_rel_astext",
]
for _s, _ in VARIANTS:
    FIELDS += [f"Q_eff_{_s}", f"n_k11_{_s}", f"n_over_n0_{_s}", f"r_w_{_s}",
               f"rw_over_R2_{_s}", f"drive_k11_{_s}", f"k11_over_conf_{_s}",
               f"k11_over_o1_{_s}", f"k11_over_pb_{_s}", f"recomb_resid_{_s}",
               f"m_layer_{_s}", f"m_over_shell_{_s}", f"m_layer_fromR2_{_s}",
               f"oob_{_s}"]
FIELDS.append("branch_change")


# ----------------------------------------------------------------- io helpers


def fnum(row, key):
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return None
    return float(v)


def read_csv(name):
    with open(DATA / name) as fh:
        return list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))


def med(vals):
    """statistics.median -- the declared convention for this batch."""
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else float("nan")


def ols(x, y):
    """Plain OLS y = a*x + b; returns (slope, intercept, r2). G22.2's bar is on this."""
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


# ----------------------------------------------------------------- the closure


def k11_solve(R_IF, pdot_w, Q_eff, pref, chi, aB):
    """The unique positive root of f(n) = n^2 R_IF^3 - C_W n^(1/2) - C_Q.

    Returns (n, r_w, C_W, C_Q). r_w comes from the pressure balance itself, not from the
    eliminated form, so `recomb_resid` downstream is a real check on the pair of source
    equations rather than a restatement of the algebra.
    """
    C_W = (pdot_w / (FOUR_PI * pref)) ** 1.5 if pdot_w > 0 else 0.0
    C_Q = 3.0 * Q_eff / (FOUR_PI * chi * aB) if Q_eff > 0 else 0.0
    R3 = R_IF ** 3
    n_str = math.sqrt(C_Q / R3) if C_Q > 0 else 0.0
    n_w = (C_W / R3) ** (2.0 / 3.0) if C_W > 0 else 0.0
    if n_str <= 0 and n_w <= 0:
        return 0.0, 0.0, C_W, C_Q

    def f(n):
        return n * n * R3 - C_W * math.sqrt(n) - C_Q

    lo, hi = max(n_str, n_w) * (1.0 - PAD), (n_str + n_w) * (1.0 + PAD)
    n = brentq(f, lo, hi, xtol=1e-300, rtol=8.9e-16, maxiter=300)
    r_w = math.sqrt(pdot_w / (FOUR_PI * pref * n)) if pdot_w > 0 else 0.0
    return n, r_w, C_W, C_Q


def pdot_wind(phase, P_conf, P_ram, R2):
    """Registered mapping: momentum -> 4 pi R2^2 P_ram; the rest -> 4 pi R2^2 P_conf (ramped)."""
    return FOUR_PI * R2 * R2 * (P_ram if phase == "momentum" else P_conf)


# ----------------------------------------------------------------- the joins


def b3m_front():
    """b9 x b11_mass x b11_photon on row_idx, guarded. Returns t-sorted front records."""
    lay = {int(r["row_idx"]): r for r in read_csv("b9_layer_density.csv")
           if r.get("status") == "ok"}
    mas = {int(r["row_idx"]): r for r in read_csv("b11_mass_ledger.csv")
           if r.get("status") == "ok"}
    pho = {int(r["row_idx"]): r for r in read_csv("b11_photon_ledger.csv")
           if r.get("status") == "ok"}
    out, worst_dt, worst_dr, worst_dust = [], 0.0, 0.0, 0.0
    for idx in sorted(set(lay) & set(mas) & set(pho)):
        L, M, P = lay[idx], mas[idx], pho[idx]
        t, R2, dR = fnum(L, "t"), fnum(L, "R2"), fnum(L, "dR_ion")
        if None in (t, R2, dR):
            continue
        for other in (M, P):
            worst_dt = max(worst_dt, abs(t / fnum(other, "t") - 1.0))
            worst_dr = max(worst_dr, abs(R2 / fnum(other, "R2") - 1.0))
            if L["phase"] != other["phase"]:
                sys.exit(f"join phase mismatch at row_idx {idx} — re-derive the join")
        fd, dp = fnum(L, "f_ionised_dust"), fnum(P, "dust_Pb")
        if fd is not None and dp:
            worst_dust = max(worst_dust, abs(fd / dp - 1.0))
        out.append(dict(config="B3M", t=t, R2=R2, R_IF=R2 + dR, Qi=fnum(M, "Qi"),
                        f_dust=fd, shell_mass=fnum(M, "shell_mass"), Pb=fnum(M, "Pb"),
                        P_HII_shipped=fnum(M, "P_HII")))
    if worst_dt > JOIN_TOL or worst_dr > JOIN_TOL:
        sys.exit(f"B3M join drifted: dt {worst_dt:.2e} dR2 {worst_dr:.2e} > {JOIN_TOL:.0e}")
    print(f"B3M front join: {len(out)} rows, worst |dt| {worst_dt:.2e}, "
          f"|dR2| {worst_dr:.2e} (tol {JOIN_TOL:.0e})")
    print(f"    b9 f_ionised_dust vs b11 dust_Pb agree to {worst_dust:.2e} "
          "— the same quantity under two names")
    return out


def b3mw01_front():
    """b12 photon x b12 mass on row_idx. dR_ion_Pb exists on DRIVING rows only."""
    pho = {int(r["row_idx"]): r for r in read_csv("b12_lowwind_photon_ledger.csv")
           if r.get("status") == "ok"}
    mas = {int(r["row_idx"]): r for r in read_csv("b12_lowwind_mass_ledger.csv")
           if r.get("status") == "ok"}
    out = []
    for idx in sorted(set(pho) & set(mas)):
        P, M = pho[idx], mas[idx]
        t, R2, dR = fnum(P, "t"), fnum(P, "R2"), fnum(P, "dR_ion_Pb")
        if None in (t, R2, dR) or P["phase"] != M["phase"]:
            continue
        out.append(dict(config="B3MW01", t=t, R2=R2, R_IF=R2 + dR, Qi=fnum(P, "Qi"),
                        f_dust=fnum(P, "dust_Pb"), shell_mass=fnum(M, "shell_mass"),
                        Pb=fnum(P, "Pb"), P_HII_shipped=fnum(P, "P_HII")))
    print(f"B3MW01 front join: {len(out)} rows (driving only — the low-wind ledger "
          "replayed no confined rows)")
    return out


def anchor_state():
    """b17's per-row ramped P_conf / P_ram / shipped drive, keyed by config, t-sorted."""
    by_cfg = {}
    for r in read_csv("b17_dust_closure.csv"):
        if r.get("status") != "ok":
            continue
        t, Pc = fnum(r, "t"), fnum(r, "P_conf")
        if t is None or not Pc:
            continue
        by_cfg.setdefault(r["config"], []).append(
            (t, r["phase"], Pc, fnum(r, "P_ram") or 0.0,
             fnum(r, "shipped_drive"), fnum(r, "n0")))
    for v in by_cfg.values():
        v.sort()
    return by_cfg


# ----------------------------------------------------------------- screen


def screen(consts):
    pref, chi, aB, mu = consts
    state = anchor_state()
    rows, dropped, pram_mismatch = [], 0, 0.0
    for fr in b3m_front() + b3mw01_front():
        pts = state.get(fr["config"]) or []
        if not pts or None in (fr["Qi"], fr["R_IF"], fr["shell_mass"]):
            dropped += 1
            continue
        best = min(pts, key=lambda p: abs(p[0] - fr["t"]))
        if abs(best[0] - fr["t"]) > ANCHOR_TOL * max(fr["t"], 1e-12):
            dropped += 1
            continue
        _, phase, P_conf, P_ram, ship, n0 = best
        R2, R_IF, Qi = fr["R2"], fr["R_IF"], fr["Qi"]
        if not (R2 > 0 and R_IF >= R2 and Qi > 0 and P_conf > 0):
            dropped += 1
            continue
        if phase == "momentum" and P_conf > 0:
            pram_mismatch = max(pram_mismatch, abs(P_ram / P_conf - 1.0))

        pdw = pdot_wind(phase, P_conf, P_ram, R2)
        amp = (R_IF / R2) ** 2
        row = dict(config=fr["config"], phase=phase, t=fr["t"], R2=R2, R_IF=R_IF,
                   rho_o1=amp, Qi=Qi, f_dust=fr["f_dust"], shell_mass=fr["shell_mass"],
                   Pb=fr["Pb"], P_conf=P_conf, P_ram=P_ram, pdot_w=pdw,
                   P_HII_shipped=fr["P_HII_shipped"], shipped_drive=ship, n0=n0,
                   drive_o1=P_conf * amp)

        # ---- G22.1, per row, on the SAME machinery the value uses ----
        n_s, _, _, cq = k11_solve(R_IF, 0.0, Qi, pref, chi, aB)
        n_s_closed = math.sqrt(cq / R_IF ** 3)
        row["g221_stromgren_rel"] = abs(pref * n_s * amp / (pref * n_s_closed * amp) - 1.0)
        n_p, _, _, _ = k11_solve(R_IF, pdw, 0.0, pref, chi, aB)
        drive_p = pref * n_p * amp
        target = pdw / (FOUR_PI * R2 * R2)          # == P_ram in momentum, P_conf elsewhere
        row["g221_wind_rel"] = abs(drive_p / target - 1.0)
        # the clause as literally typed in PLAN.md: P_K11*(R2/R_IF)^2 == P_ram
        row["g221_wind_rel_astext"] = (abs(drive_p / amp / P_ram - 1.0)
                                       if P_ram > 0 else None)

        # ---- the two Q_eff variants ----
        for suf, _ in VARIANTS:
            fd = fr["f_dust"]
            if suf == "qd":
                if fd is None or not (0.0 <= fd < 1.0):
                    for k in ("Q_eff", "n_k11", "n_over_n0", "r_w", "rw_over_R2",
                              "drive_k11", "k11_over_conf", "k11_over_o1", "k11_over_pb",
                              "recomb_resid", "m_layer", "m_over_shell",
                              "m_layer_fromR2", "oob"):
                        row[f"{k}_{suf}"] = None
                    continue
                Q_eff = Qi * (1.0 - fd)
            else:
                Q_eff = Qi
            n, r_w, cw_v, cq_v = k11_solve(R_IF, pdw, Q_eff, pref, chi, aB)
            drive = pref * n * amp
            # residual of the eliminated equation, SCALED -- the well-conditioned check.
            # (n^2 (R_IF^3 - r_w^3)/C_Q - 1 is the same statement but cancels; see docstring.)
            resid = (abs(n * n * R_IF ** 3 - cw_v * math.sqrt(n) - cq_v)
                     / (n * n * R_IF ** 3))
            # exact, cancellation-free: (4pi/3)(R_IF^3 - r_w^3) n mu == mu Q_eff/(chi aB n)
            m_layer = mu * Q_eff / (chi * aB * n)
            dR = R_IF - R2
            m_fromR2 = ((FOUR_PI / 3.0) * dR * (R_IF ** 2 + R_IF * R2 + R2 ** 2) * n * mu)
            oob = []
            if r_w >= R_IF:
                oob.append("rw>=ri")
            if fr["shell_mass"] and m_layer / fr["shell_mass"] >= 1.0:
                oob.append("overflow")
            row.update({
                f"Q_eff_{suf}": Q_eff, f"n_k11_{suf}": n,
                f"n_over_n0_{suf}": (n / n0) if n0 else None,
                f"r_w_{suf}": r_w, f"rw_over_R2_{suf}": r_w / R2,
                f"drive_k11_{suf}": drive,
                f"k11_over_conf_{suf}": drive / P_conf,
                f"k11_over_o1_{suf}": drive / row["drive_o1"],
                f"k11_over_pb_{suf}": (drive / fr["Pb"]) if fr["Pb"] else None,
                f"recomb_resid_{suf}": resid,
                f"m_layer_{suf}": m_layer,
                f"m_over_shell_{suf}": (m_layer / fr["shell_mass"]) if fr["shell_mass"] else None,
                f"m_layer_fromR2_{suf}": m_fromR2,
                f"oob_{suf}": ";".join(oob),
            })

        # ---- G22.5 census: k5_offline_screen's convention, driving = shipped P_HII > 0 ----
        ship_drv = (fr["P_HII_shipped"] or 0.0) > 0
        flips = []
        for suf, _ in VARIANTS:
            v = row.get(f"k11_over_pb_{suf}")
            if v is None:
                continue
            if ship_drv and v <= 1.0:
                flips.append(f"{suf}:driving->confined")
            if not ship_drv and v > 1.0:
                flips.append(f"{suf}:confined->DRIVING")
        row["branch_change"] = ";".join(flips)
        rows.append(row)

    print(f"{len(rows)} rows screened, {dropped} dropped (no anchor within "
          f"{ANCHOR_TOL:.0e} rel in t, or missing input)")
    if pram_mismatch:
        print(f"    momentum P_ram vs P_conf: worst rel {pram_mismatch:.2e} "
              "(the registered pdot_w mapping assumes they coincide there)")
    return rows


# ----------------------------------------------------------------- gates


def phases(rows, cfg):
    order = ("energy", "implicit", "transition", "momentum")
    return [p for p in order if any(r["config"] == cfg and r["phase"] == p for r in rows)]


def report(rows):
    cfgs = ("B3M", "B3MW01")

    print("\n" + "=" * 78)
    print("G22.1 — LIMITS (BLOCKING, first). Bar: 1e-10 relative on EVERY row.")
    print("=" * 78)
    s_worst = max(r["g221_stromgren_rel"] for r in rows)
    w_worst = max(r["g221_wind_rel"] for r in rows)
    s_bad = sum(1 for r in rows if r["g221_stromgren_rel"] > G221_BAR)
    w_bad = sum(1 for r in rows if r["g221_wind_rel"] > G221_BAR)
    print(f"  wind term zeroed  -> closed-form Stromgren : worst {s_worst:.3e}  "
          f"rows over bar {s_bad}/{len(rows)}  -> {'PASS' if not s_bad else 'FAIL'}")
    print(f"  photon term zeroed -> pdot_w/(4 pi R2^2)   : worst {w_worst:.3e}  "
          f"rows over bar {w_bad}/{len(rows)}  -> {'PASS' if not w_bad else 'FAIL'}")
    print(f"  G22.1 VERDICT: {'PASS' if not (s_bad or w_bad) else 'FAIL — STOP'}")
    lit = [r["g221_wind_rel_astext"] for r in rows if r["g221_wind_rel_astext"] is not None]
    if lit:
        print("\n  ⚠ The second clause AS TYPED in PLAN.md reads `P_K11*(R2/R_IF)^2 == P_ram`.")
        print(f"    Measured that way: worst {max(lit):.3e}, median {med(lit):.3e} — it fails,")
        print( "    and it must: the batch's own stage-0 [D] result says the wind-only drive AT R2")
        print( "    is pdot_w/(4 pi R2^2), i.e. P_K11 itself == P_ram with no ratio applied. The")
        print( "    typed ratio is a transcription slip in the gate text, not a property of K11.")
        print( "    Reported both ways; the physics test is the one barred above.")

    print("\n" + "=" * 78)
    print("G22.2 — DECOUPLING REGRESSION (G14.0's bars verbatim)")
    print("  FAIL bar: slope in [0.95, 1.05] AND r2 > 0.99 over shipped-driving rows")
    print("=" * 78)
    for cfg in cfgs:
        for suf, lbl in VARIANTS:
            pts = [(r["Pb"], r[f"drive_k11_{suf}"]) for r in rows
                   if r["config"] == cfg and (r["P_HII_shipped"] or 0) > 0
                   and r["Pb"] and r.get(f"drive_k11_{suf}")]
            if len(pts) < 3:
                print(f"  {cfg:8}{lbl:16}: <3 driving rows — VOID")
                continue
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            a, _, r2 = ols(x, y)
            la, _, lr2 = ols([math.log10(v) for v in x], [math.log10(v) for v in y])
            fail = a is not None and 0.95 <= a <= 1.05 and r2 > 0.99
            print(f"  {cfg:8}{lbl:16}: N={len(pts):3d}  slope {a:+.4f}  r2 {r2:.4f}"
                  f"  (log-log {la:+.3f}, r2 {lr2:.3f})  -> "
                  f"{'FAIL — identity returned' if fail else 'pass'}")
    print("  [E, registered] strong coupling was EXPECTED on confined rows; the confined-row")
    print("      regression is disclosed below, outside the gate's own driving-row bar:")
    for cfg in cfgs:
        pts = [(r["Pb"], r["drive_k11_q"]) for r in rows
               if r["config"] == cfg and not (r["P_HII_shipped"] or 0) > 0
               and r["Pb"] and r.get("drive_k11_q")]
        if len(pts) < 3:
            print(f"      {cfg:8} confined: <3 rows — VOID")
            continue
        a, _, r2 = ols([p[0] for p in pts], [p[1] for p in pts])
        la, _, lr2 = ols([math.log10(p[0]) for p in pts], [math.log10(p[1]) for p in pts])
        print(f"      {cfg:8} confined Qi: N={len(pts):3d}  slope {a:+.4f}  r2 {r2:.4f}"
              f"  (log-log {la:+.3f}, r2 {lr2:.3f})")

    print("\n" + "=" * 78)
    print("G22.3 — MAGNITUDE + THE SEAM (no bar). medians, statistics.median")
    print("=" * 78)
    print(f"  {'cfg':8}{'phase':11}{'N':>4}  {'K11/P_conf':>11}{'K11/P_O1':>10}"
          f"{'n_K11/n0':>10}{'r_w/R2':>9}{'K11/P_ram':>11}")
    for cfg in cfgs:
        for ph in phases(rows, cfg):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph]
            pr = [r["drive_k11_q"] / r["P_ram"] for r in sel if r["P_ram"]]
            print(f"  {cfg:8}{ph:11}{len(sel):4d}  "
                  f"{med([r['k11_over_conf_q'] for r in sel]):11.4f}"
                  f"{med([r['k11_over_o1_q'] for r in sel]):10.4f}"
                  f"{med([r['n_over_n0_q'] for r in sel]):10.4f}"
                  f"{med([r['rw_over_R2_q'] for r in sel]):9.4f}"
                  f"{(med(pr) if pr else float('nan')):11.4f}")
    print("  dust variant Qi*(1-f_dust), K11/P_conf and K11/P_O1:")
    for cfg in cfgs:
        for ph in phases(rows, cfg):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph
                   and r.get("k11_over_conf_qd") is not None]
            if sel:
                print(f"      {cfg:8}{ph:11}{len(sel):4d}  "
                      f"{med([r['k11_over_conf_qd'] for r in sel]):11.4f}"
                      f"{med([r['k11_over_o1_qd'] for r in sel]):10.4f}")

    print("\n" + "=" * 78)
    print("G22.4 — DOMAIN + OVERFLOW.  bar: implied layer mass <= shell mass (G21.2)")
    print("=" * 78)
    for cfg in cfgs:
        for ph in phases(rows, cfg):
            sel = [r for r in rows if r["config"] == cfg and r["phase"] == ph]
            mo = [r["m_over_shell_q"] for r in sel if r["m_over_shell_q"] is not None]
            if not mo:
                continue
            over = sum(1 for v in mo if v >= 1.0)
            rwri = sum(1 for r in sel if "rw>=ri" in (r["oob_q"] or ""))
            print(f"  {cfg:8}{ph:11}N={len(sel):3d}  m_layer/m_shell median {med(mo):8.5f}"
                  f"  max {max(mo):8.5f}  overflow rows {over}  rw>=ri {rwri}")
    allmo = [r["m_over_shell_q"] for r in rows if r["m_over_shell_q"] is not None]
    nover = sum(1 for v in allmo if v >= 1.0)
    print(f"  G22.4 bar: {len(allmo)-nover}/{len(allmo)} rows within the shell's own mass"
          f"  -> {'PASS' if not nover else 'FAIL'}")
    print("  ⚠ `r_w >= r_i` is UNREACHABLE by construction: at the root r_w^3 = R_IF^3 - C_Q/n^2,")
    print("    so Q_eff > 0 forces r_w < R_IF. A zero count here is an identity, not evidence.")
    print("    It IS float-reachable when the wind dominates (377/2000 draws at pdot_w 1e0-1e6,")
    print("    Q_eff 1e-12..1e-6) — the likely source of stage 0's reported 34/2000, which is")
    print("    therefore a cancellation artefact rather than a physical out-of-domain count.")
    rr = [r["recomb_resid_q"] for r in rows if r["recomb_resid_q"] is not None]
    print(f"  root quality (scaled residual of the eliminated equation): worst {max(rr):.3e}")

    print("\n" + "=" * 78)
    print("G22.5 — BRANCH CENSUS vs C3c's driving set (driving := shipped P_HII > 0;")
    print("        K11 'drives' when its value exceeds Pb, k5_offline_screen's convention)")
    print("=" * 78)
    for cfg in cfgs:
        sel = [r for r in rows if r["config"] == cfg]
        n_d = sum(1 for r in sel if (r["P_HII_shipped"] or 0) > 0)
        n_c = len(sel) - n_d
        for suf, lbl in VARIANTS:
            dc = sum(1 for r in sel if f"{suf}:driving->confined" in r["branch_change"])
            cd = sum(1 for r in sel if f"{suf}:confined->DRIVING" in r["branch_change"])
            print(f"  {cfg:8}{lbl:16}: {dc}/{n_d} driving rows flip confined, "
                  f"{cd}/{n_c} confined rows flip DRIVING")
    print("  ⚠ COVERAGE: B3MW01 contributes DRIVING ROWS ONLY (no committed confined front),")
    print("    so its confined census is structurally empty — not a clean result.")


def selfcheck(draws=2000, seed=20260831):
    """Re-run stage 0's derivation check independently, on random draws.

    Not a substitute for G22.1 -- this exercises the solver over 9 decades of each input
    where the real rows occupy a narrow corner. Asserts, so it fails loudly.
    """
    import random
    rnd = random.Random(seed)
    worst_bal = worst_rec = worst_str = worst_wind = 0.0
    rw_ge_ri = 0
    for _ in range(draws):
        R = 10 ** rnd.uniform(-2, 2)
        pdw = 10 ** rnd.uniform(-6, 3)
        Q = 10 ** rnd.uniform(-6, 3)
        pref, chi, aB = 10 ** rnd.uniform(-3, 1), rnd.uniform(1.0, 1.2), 10 ** rnd.uniform(-6, -2)
        n, r_w, C_W, C_Q = k11_solve(R, pdw, Q, pref, chi, aB)
        # the pressure balance, and the eliminated equation scaled (see the docstring on why
        # the direct (R^3 - r_w^3) form of the recombination check is not usable here)
        worst_bal = max(worst_bal, abs(pdw / (FOUR_PI * r_w ** 2) / (pref * n) - 1.0))
        worst_rec = max(worst_rec,
                        abs(n * n * R ** 3 - C_W * math.sqrt(n) - C_Q) / (n * n * R ** 3))
        if r_w >= R:
            rw_ge_ri += 1
        # limits
        n_s, _, _, cq = k11_solve(R, 0.0, Q, pref, chi, aB)
        worst_str = max(worst_str, abs(n_s / math.sqrt(cq / R ** 3) - 1.0))
        n_p, _, cw, _ = k11_solve(R, pdw, 0.0, pref, chi, aB)
        worst_wind = max(worst_wind, abs(pref * n_p / (pdw / (FOUR_PI * R ** 2)) - 1.0))
    print(f"selfcheck on {draws} draws (seed {seed}):")
    print(f"  pressure balance recovered   worst {worst_bal:.3e}")
    print(f"  eliminated eq (scaled) resid worst {worst_rec:.3e}")
    print(f"  wind->0  == Stromgren(R_IF)  worst {worst_str:.3e}")
    print(f"  photons->0 == pdot/(4 pi R^2) worst {worst_wind:.3e}")
    print(f"  draws with r_w >= r_i        {rw_ge_ri}/{draws}   "
          "(0 in this range; float-reachable when the wind dominates — see the docstring)")
    assert max(worst_bal, worst_rec, worst_str, worst_wind) < 1e-10
    assert rw_ge_ri == 0
    print("  selfcheck OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "b22_k11_screen.csv")
    ap.add_argument("--selfcheck", action="store_true",
                    help="re-verify the closure on random draws (stage 0's check) and exit")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    p = read_param(str(BENCH))
    pref = (p["mu_convert"].value / p["mu_ion_shell"].value
            * p["k_B"].value * p["TShell_ion"].value)
    consts = (pref, p["chi_e_shell"].value, p["caseB_alpha"].value, p["mu_convert"].value)
    print(f"pref = (mu_c/mu_i) k_B T = {pref:.6e}   chi_e = {consts[1]}   "
          f"alpha_B = {consts[2]:.4e}")

    rows = screen(consts)
    if not rows:
        sys.exit("no rows screened")
    report(rows)

    with open(args.out, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Batch 22 stage 1 (K11, Geen 2019 additive closure). Gates G22.1-G22.5\n")
        fh.write("# pre-registered in PLAN.md SBatch-22 and committed before this ran.\n")
        fh.write("# Anchor b17_dust_closure.csv (ramped P_conf, G16.3); fronts from the B3M\n")
        fh.write("# b9 x b11 x b11-photon row_idx join and the B3MW01 b12 ledgers (driving only).\n")
        fh.write("# Medians in the report are statistics.median, NOT the med() order statistic.\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
