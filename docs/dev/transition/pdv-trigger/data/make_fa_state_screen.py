#!/usr/bin/env python3
"""SC-0 — the offline falsification screen for the state-coupled f_A (FA_STATE_COUPLED.md SC-0).

Scores THREE candidate f_A laws against MEASURED targets, offline, from committed trajectories.
No sims are run. If neither derived candidate holds one constant across the suite, SC-0 FAILS and
no production code is written (the fitted scalar table stands as the honest result).

CANDIDATES (FA_STATE_COUPLED.md §1)
  C1  El-Badry mixing:   f_A = theta_EB(lambda_dv, n_amb) * Lmech / (L2+L3)
      theta_EB from the VALIDATED closed form in make_elbadry_theta.py (theta(1,1)=0.61 vs his Fig 7).
      n_amb = local ambient density at R2 (= nCore for densPL_alpha=0; the power law otherwise).
      Free constant: lambda_dv (literature ~3-3.5).
  C2  Lancaster fractal area (L21b Eq 11, PARAMETER-FREE):  f_A = alpha_A * (R2/l_cool)^d
      l_cool = [v_t * t_cool]^2 / L      (p=1/2 cascade; LANCASTER_REFERENCE §7c — p is [D]-grade)
      v_t    = sqrt(6 G M_cloud / (5 R_cloud))   (L21b Eq 23; reproduces Table-1 v_t to <=0.4% [V])
      L      = L_box = 2 R_cloud [V];  alpha_A ~ 1 [V, order-unity];  d in [0.4,0.7] [V]
      t_cool = P / (n^2 Lambda(T_pk)) evaluated with TRINITY'S OWN non-CIE table: the table returns the
      VOLUMETRIC rate n^2*Lambda, so t_cool = P_cgs / cool_rate_cgs directly. That is algebraically
      identical to Eq 13's other form (k_B T_pk)^2/(P Lambda) once n = P/(k_B T_pk) -- the two forms
      agreeing IS the unit cross-check the plan demands (units are this repo's declared bug class).
  C3  the fitted scalar baseline to beat:  f_A(n) = 315 * n^-0.335   (FINDINGS §15j, un-derived)

SCORE.  Per arm: the luminosity-weighted mean of the candidate over the accepted-implicit window
(int f_A Lmech dt / int Lmech dt -- the same weighting Theta_cum uses), plus the median as a
robustness check. Then ratio = predicted / measured-target. The DISCRIMINATOR is the max/min SPREAD
of that ratio across arms: it is calibration-invariant, so a law with the right SHAPE gives a
constant ratio even if offset (absorbable by alpha_A or lambda_dv), while a wrong shape scatters.
⚠️ REPORT IT PER TARGET TYPE. 'band' (the f_A that lands Theta_cum in [0.90,0.99]) and 'fire' (the
f_A whose theta_max crosses 0.95) are DIFFERENT criteria and are known to disagree (FINDINGS §15j:
bench3 needs ~16 for the band but fires at 12). A combined spread conflates them and inflates the
number -- e.g. C1 is 3.3x within 'band' and 4.5x within 'fire' but 30x combined, because the two
criteria sit ~10x apart. The per-type spreads are the meaningful test of a law's shape.

TARGETS.  Band-entry dose for the clean-blowout benches (FINDINGS §15j) and the measured fire
threshold f_fire for the theta5s configs (data/theta5s_fire_map.csv, HPC-confirmed §15e). The two
CONTROLS (fail_repro, small_1e6) never fire at any dose -- a candidate predicting a modest f_A for
them is falsified.

⚠️ FIRST-ORDER BY CONSTRUCTION: candidates are evaluated along the UNBOOSTED (f_A=1) trajectory, so
there is no back-reaction. SC-0 can therefore FALSIFY a law cheaply, but a PASS must still be
confirmed live (SC-2 per-call equivalence, SC-4 the L21b matrix).

    python docs/dev/transition/pdv-trigger/data/make_fa_state_screen.py
Deliverable: data/fa_state_screen.csv (+ console verdict).
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
REPO = HERE.parents[3]
TRAJ = PDV / "runs" / "data" / "bench_state_traj"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from make_elbadry_theta import theta as theta_EB  # noqa: E402  (validated closed form)

G_PC = 4.301e-3          # pc Msun^-1 (km/s)^2
T_PK = 2.0e4             # K, temperature of peak cooling (El-Badry / L21b)
KM_PER_PC = 3.0857e13
D_GRID = (0.4, 0.5, 0.6, 0.7)
LDV_GRID = (1.0, 2.0, 3.0, 3.5, 5.0)

# arm -> measured target. 'band' = f_A that lands Theta_cum in the L21b band (FINDINGS §15j);
# 'fire' = measured fire threshold (data/theta5s_fire_map.csv); None = never fires (control).
TARGETS = {
    "bench1_m5e4_r20__none_diag": ("band", 74.8),
    "bench2_m1e5_r10__none_diag": ("band", 53.5),
    "bench3_m1e5_r5__none_diag": ("band", 13.9),
    "bench4_m1e5_r2p5__none_diag": ("fire", 4.0),
    "bench5_m5e5_r2p5__none_diag": ("fire", 1.0),
    "normal_n1e3__none": ("fire", 1.0),
    "small_dense_highsfe__none": ("fire", 4.0),
    "simple_cluster__none": ("fire", 4.0),
    "midrange_pl0__none": ("fire", 6.0),
    "large_diffuse_lowsfe__none": ("fire", 8.0),
    "be_sphere__none": ("fire", 12.0),
    "pl2_steep__none": ("fire", 12.0),
    "small_1e6__none": ("control", None),
    "fail_repro__none": ("control", None),
}


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _meta(arm, outroot):
    """rCloud / gas mass / nCore(cgs) / profile straight from the run's metadata.json."""
    import json
    for sub in ("theta5s", "bench5"):
        p = outroot / "outputs" / sub / arm / "metadata.json"
        if p.exists():
            m = json.load(p.open())
            ncore_au = m.get("nCore")
            return {
                "rCloud": m.get("rCloud"), "mCloud_gas": m.get("mCloud"),
                "nCore_cgs": (ncore_au / 2.938e55) if ncore_au else None,
                "alpha": m.get("densPL_alpha"), "rCore": m.get("rCore", 1.0),
                "profile": m.get("dens_profile"),
            }
    return None


def _n_amb(md, R2):
    """Local ambient density [cm^-3] at R2 for the C1 option-A mapping."""
    n0, rc, a = md["nCore_cgs"], md["rCore"] or 1.0, md["alpha"]
    if n0 is None:
        return None
    if not a:                       # uniform (densPL_alpha = 0)
        return n0
    return n0 * (R2 / rc) ** a if R2 > rc else n0


def _cool_rate_fn():
    """TRINITY's own non-CIE volumetric cooling rate n^2*Lambda [erg/cm^3/s] at (n, T_pk, phi)."""
    from trinity._input import read_param
    import trinity._functions.unit_conversions as cvt
    import numpy as np
    import trinity.cooling.non_CIE.read_cloudy as non_CIE
    pp = PDV / "runs" / "params" / "bench5" / "bench3_m1e5_r5__none.param"
    params = read_param.read_param(str(pp))
    tab, _heat, _ni = non_CIE.get_coolingStructure(params)   # read_param alone leaves it unloaded
    # the cube's axes are already log10: ndens[-4,12], temp[3.5,5.5], phi[0,21] -> clamp into range
    ax = {a: (float(np.min(getattr(tab, a))), float(np.max(getattr(tab, a))))
          for a in ("ndens", "temp", "phi")}

    def _clip(x, lo_hi):
        return min(max(x, lo_hi[0]), lo_hi[1])

    def rate(n_cgs, phi_cgs):
        q = np.array([[_clip(np.log10(max(n_cgs, 1e-30)), ax["ndens"]),
                       _clip(np.log10(T_PK), ax["temp"]),
                       _clip(np.log10(max(phi_cgs, 1e-30)), ax["phi"])]])
        return float(10 ** tab.interp(q)[0])
    return rate, cvt


def candidates(arm, rows, md, rate, cvt):
    """Per-row candidate f_A values + the weights for the luminosity-weighted mean."""
    out = []
    for r in rows:
        t, R2, Pb = _f(r["t_now"]), _f(r["R2"]), _f(r.get("Pb"))
        Lm = _f(r["Lmech"])
        L2 = _f(r.get("bubble_L2Conduction")) or 0.0
        L3 = _f(r.get("bubble_L3Intermediate")) or 0.0
        Qi = _f(r.get("Qi"))
        if None in (t, R2, Pb, Lm) or not Lm or (L2 + L3) <= 0 or R2 <= 0:
            continue
        n_amb = _n_amb(md, R2)
        # --- C1: El-Badry (validated closed form; Lmech/(L2+L3) already in au, ratio is unitless)
        c1 = {ldv: theta_EB(ldv, n_amb) * Lm / (L2 + L3) for ldv in LDV_GRID} if n_amb else {}
        # --- C2: Lancaster fractal area (parameter-free)
        n_pk = Pb * cvt.Pb_au2_KcmInv / T_PK                     # cm^-3 at T_pk (pressure balance)
        phi = (Qi / (4 * math.pi * R2 ** 2)) if Qi else 0.0       # au; table wants cgs
        phi_cgs = phi / cvt.phi_cgs2au if Qi else 0.0
        cr = rate(n_pk, phi_cgs)                                  # erg/cm^3/s  (= n^2 Lambda)
        t_cool_s = (Pb * cvt.Pb_au2cgs) / cr if cr > 0 else None  # == (k_B T_pk)^2/(P Lambda)
        c2 = {}
        if t_cool_s and md["rCloud"] and md["mCloud_gas"]:
            L_box = 2.0 * md["rCloud"]
            v_t = math.sqrt(6 * G_PC * md["mCloud_gas"] / (5 * md["rCloud"]))   # km/s
            l_cool = (v_t * t_cool_s / KM_PER_PC) ** 2 / L_box                  # pc
            if l_cool > 0:
                c2 = {d: (R2 / l_cool) ** d for d in D_GRID}                    # alpha_A = 1
        out.append((t, Lm, c1, c2))
    return out


def _lw_mean(vals, weights, times):
    """int v*w dt / int w dt (trapezoid) -- the Theta_cum weighting."""
    num = den = 0.0
    for (t0, v0, w0), (t1, v1, w1) in zip(zip(times, vals, weights), list(zip(times, vals, weights))[1:]):
        dt = t1 - t0
        num += 0.5 * (v0 * w0 + v1 * w1) * dt
        den += 0.5 * (w0 + w1) * dt
    return num / den if den else None


def main():
    if not TRAJ.is_dir():
        sys.exit(f"no {TRAJ} -- run the SC-0 arms + harvest first (FA_STATE_COUPLED SC-0 step 1).")
    outroot = Path("/tmp/claude-0/-home-user-trinity/986d7831-e333-5601-8bf9-13b33d1615f0/"
                   "scratchpad/sc0_out")
    rate, cvt = _cool_rate_fn()
    rows_out = []
    for f in sorted(TRAJ.glob("*.csv")):
        arm = f.stem
        md = _meta(arm, outroot)
        if md is None:
            print(f"  [skip] {arm}: no metadata.json (raw run dir gone) ")
            continue
        tr = _read_csv(f)
        pts = candidates(arm, tr, md, rate, cvt)
        if len(pts) < 3:
            print(f"  [skip] {arm}: {len(pts)} usable rows")
            continue
        ts = [p[0] for p in pts]
        ws = [p[1] for p in pts]
        kind, tgt = TARGETS.get(arm, ("?", None))
        rec = {"arm": arm, "target_kind": kind, "target": tgt, "n_rows": len(pts),
               "nCore_cgs": f"{md['nCore_cgs']:.4g}" if md["nCore_cgs"] else "",
               "rCloud_pc": f"{md['rCloud']:.3f}" if md["rCloud"] else "", "alpha": md["alpha"]}
        for ldv in LDV_GRID:
            v = [p[2].get(ldv) for p in pts]
            if all(x is not None for x in v):
                rec[f"C1_ldv{ldv:g}"] = f"{_lw_mean(v, ws, ts):.4g}"
        for d in D_GRID:
            v = [p[3].get(d) for p in pts]
            if all(x is not None for x in v):
                rec[f"C2_d{d:g}"] = f"{_lw_mean(v, ws, ts):.4g}"
        if md["nCore_cgs"]:
            rec["C3_fitted"] = f"{315 * md['nCore_cgs'] ** -0.335:.4g}"
        rows_out.append(rec)

    if not rows_out:
        sys.exit("no arms scored.")
    cols = sorted({k for r in rows_out for k in r}, key=lambda k: (k != "arm", k))
    out = HERE / "fa_state_screen.csv"
    with out.open("w", newline="") as fh:
        fh.write("# SC-0 state-coupled-f_A screen (FA_STATE_COUPLED.md SC-0). Candidates evaluated on the "
                 "UNBOOSTED fa1 trajectory => FIRST-ORDER, no back-reaction; falsification is sound, a PASS "
                 "needs SC-2/SC-4. C1 = theta_EB*Lmech/(L2+L3) [validated closed form]; C2 = (R2/l_cool)^d "
                 "[L21b Eq 11, parameter-free, alpha_A=1]; C3 = 315*n^-0.335 [fitted baseline]. Values are "
                 "luminosity-weighted means over the accepted-implicit window.\n")
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} arms -> {out}\n")

    # ---- verdict: spread of predicted/target across the arms that HAVE a numeric target ----
    print("SPREAD OF ratio = predicted / measured-target  (calibration-invariant; 1.0x = perfect shape)")
    scored = [r for r in rows_out if isinstance(r.get("target"), float) and r["target"]]
    best = []
    for key in [k for k in cols if k.startswith(("C1_", "C2_", "C3"))]:
        rs = [(float(r[key]) / r["target"], r["arm"]) for r in scored if r.get(key)]
        if len(rs) < 3:
            continue
        lo, hi = min(rs), max(rs)
        spread = hi[0] / lo[0] if lo[0] > 0 else float("inf")
        best.append((spread, key, len(rs), lo, hi))
    for spread, key, n, lo, hi in sorted(best):
        # per-target-type spread: 'band' (Theta_cum in-band dose) and 'fire' (theta_max threshold)
        # are DIFFERENT criteria, so a combined spread conflates them -- report both.
        per = []
        for kind in ("band", "fire"):
            rr = [float(r[key]) / r["target"] for r in scored
                  if r.get(key) and r.get("target_kind") == kind]
            if len(rr) >= 2:
                per.append(f"{kind} {max(rr)/min(rr):.1f}x")
        print(f"  {key:14s} n={n:2d}  combined={spread:8.2f}x  [{', '.join(per) or '-'}]"
              f"   min {lo[0]:.3g} ({lo[1][:20]})  max {hi[0]:.3g} ({hi[1][:20]})")
    if best:
        s, key, *_ = sorted(best)[0]
        print(f"\nbest-shape candidate: {key} (spread {s:.2f}x). C3 baseline to beat is the C3_fitted row.")
    print("\nCONTROLS (must come out un-firable; a modest predicted f_A here falsifies the law):")
    for r in rows_out:
        if r["target_kind"] == "control":
            vals = {k: r[k] for k in r if k.startswith(("C1_ldv3", "C2_d0.7", "C3"))}
            print(f"  {r['arm']:26s} {vals}")


if __name__ == "__main__":
    main()
