#!/usr/bin/env python3
"""Leakage cooling-trigger analysis (scratch; leaktest only).

Per implicit row compute r = (Lmech_total - bubble_Lloss)/Lmech_total.
NOTE: in trinity, the snapshot column `bubble_Lloss` (= solver L_loss) already
includes bubble_Leak (get_betadelta.py:427-432), so this r matches the actual
cooling-balance trigger test (run_energy_implicit_phase.py:1095). Report MIN r,
whether it crosses 0.05 (trigger fires), max bubble_Lloss/Lmech_total, and whether
the run reached transition/momentum phase. Also report solver-health hints.
"""
import csv
import sys
from collections import Counter


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def analyze(path, label, cf):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    phases = Counter(r.get("phase") for r in rows)
    impl = [r for r in rows if r.get("phase") == "implicit"]
    rs, loss_ratio = [], []
    for r in impl:
        Lm = fnum(r.get("Lmech_total"))
        Ll = fnum(r.get("bubble_Lloss"))
        if Lm and Lm > 0 and Ll is not None:
            rs.append((Lm - Ll) / Lm)
            loss_ratio.append(Ll / Lm)
    rmin = min(rs) if rs else None
    lmax = max(loss_ratio) if loss_ratio else None
    fires = rmin is not None and rmin < 0.05
    transitioned = any(r.get("phase") in ("transition", "momentum") for r in rows)
    last_t = fnum(rows[-1].get("t_now")) if rows else None
    # solver health: betadelta_converged fraction over implicit rows
    conv = sum(1 for r in impl if str(r.get("betadelta_converged")).lower() == "true")
    print(f"=== {label}  (coverFraction={cf}) ===")
    print(f"  phases={dict(phases)}  last_t={last_t}")
    print(f"  implicit rows w/ valid r: {len(rs)}  betadelta_converged={conv}/{len(impl)}")
    print(f"  r_min={rmin!r}")
    print(f"  max(bubble_Lloss/Lmech_total)={lmax!r}")
    print(f"  fires(r<0.05)={fires}  transitioned={transitioned}")
    return dict(cf=cf, rmin=rmin, lmax=lmax, fires=fires,
               transitioned=transitioned, n_impl=len(rs), conv=conv,
               n_total_impl=len(impl), last_t=last_t, phases=dict(phases))


if __name__ == "__main__":
    base = "docs/dev/transition/cleanroom/data"
    jobs = [
        (f"{base}/c0_simple_cluster_h0.csv", "BASELINE Cf=1.0", 1.0),
        (f"{base}/leaktest/c0_sc_cf099.csv", "cf099", 0.99),
        (f"{base}/leaktest/c0_sc_cf095.csv", "cf095", 0.95),
        (f"{base}/leaktest/c0_sc_cf090.csv", "cf090", 0.90),
    ]
    results = []
    for path, label, cf in jobs:
        try:
            results.append(analyze(path, label, cf))
        except FileNotFoundError:
            print(f"=== {label} (coverFraction={cf}) === MISSING: {path}")
        print()
    print("coverFraction | r_min | fires? | transitioned? | conv/impl | last_t")
    for r in results:
        rm = f"{r['rmin']:.4f}" if r["rmin"] is not None else "NA"
        print(f"  {r['cf']:.2f} | {rm} | {r['fires']} | {r['transitioned']} | "
              f"{r['conv']}/{r['n_total_impl']} | {r['last_t']}")
