#!/usr/bin/env python3
"""Registered, SIM-FREE El-Badry θ prediction per bench5 config — the non-circular Phase-5 support.

Calibrating f_A on Lancaster 2021b and then "agreeing" with L21b is a fit, not a validation
(SOURCE_TERM_DESIGN.md §3 Phase 5, "Honesty (circularity)"). The independent support is El-Badry's
√n closed form: it predicts θ per bench from density ALONE, with NO TRINITY sim and NO L21b fit — so
registering it BEFORE the calibration arms land makes any later agreement a real cross-check.

θ_EB(λδv, n) = X/(11/5 + X), X = A_mix·√(λδv·n)  [El-Badry+2019 Eq 37/38; A_mix=3.5 fit / 1.7 analytic]
— reused verbatim from make_elbadry_theta.py (do not re-derive). n = the bench ambient density n̄_H
(flat cloud ⇒ nCore = n̄, LANCASTER_REFERENCE §7b); λδv=3 pc·km/s is the adopted value (El-Badry's own
A_mix=3.5 was fit at λδv=3; doubly-anchored to Lancaster's GMC band, LANCASTER_REFERENCE §7a).

Caveat (from make_elbadry_theta.py): El-Badry tested only n∈[0.1,10]; the bench n̄∈[43, 2.3e5] is
EXTRAPOLATED (supported by El-Badry's own high-n note + Lancaster's 3D θ≈0.9–0.99 at GMC density).
θ_EB is a LATE-TIME (t≳3 Myr) equilibrium — a shape/asymptote check, flag-only, never a pass/fail bar.

    python docs/dev/transition/pdv-trigger/data/make_bench5_elbadry_prediction.py
Deliverable: data/bench5_elbadry_prediction.csv
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PDV / "runs"))

from make_elbadry_theta import theta  # noqa: E402  (the one calibrated closed form)
from make_bench5_params import BENCHES  # noqa: E402  (single-source the bench densities)

LDV_ADOPTED = 3.0
L21B_BAND = (0.90, 0.99)
_MH_G, _MSUN_G, _PC_CM, _G = 1.6726e-24, 1.989e33, 3.086e18, 6.674e-8
_MYR = 3.156e13


def t_ff_myr(n_h):
    rho = 1.4 * _MH_G * n_h
    return math.sqrt(3 * math.pi / (32 * _G * rho)) / _MYR


def main():
    out = HERE / "bench5_elbadry_prediction.csv"
    rows = []
    for name, m_cl, r_cl, n_h, eps in BENCHES:
        rows.append({
            "bench": name,
            "mCloud_Msun": f"{m_cl:.3g}",
            "R_cloud_pc": r_cl,
            "n_bar_H": f"{n_h:.3g}",
            "eps_star": eps,
            "t_ff_Myr": f"{t_ff_myr(n_h):.3g}",
            "theta_EB_ldv3_Amix3p5": f"{theta(LDV_ADOPTED, n_h, 3.5):.4f}",
            "theta_EB_ldv1_Amix3p5": f"{theta(1.0, n_h, 3.5):.4f}",
            "theta_EB_ldv3_Amix1p7": f"{theta(LDV_ADOPTED, n_h, 1.7):.4f}",
            "in_L21b_band": L21B_BAND[0] <= theta(LDV_ADOPTED, n_h, 3.5) <= L21B_BAND[1],
        })
    cols = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        fh.write("# REGISTERED (sim-free, pre-calibration) El-Badry theta prediction per bench5 config "
                 "— the non-circular Phase-5 support (SOURCE_TERM_DESIGN §3 Phase 5). "
                 "theta_EB = X/(2.2+X), X=A_mix*sqrt(ldv*n), El-Badry Eq 37/38 via make_elbadry_theta.py. "
                 "n = bench n_bar_H (LANCASTER_REFERENCE §7b); ldv=3 adopted. Flag-only late-time check, "
                 "NOT a pass/fail bar. L21b band Theta in [0.90, 0.99].\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")
    for r in rows:
        print(f"  {r['bench']:20s} n={r['n_bar_H']:>8}  theta_EB(ldv3)={r['theta_EB_ldv3_Amix3p5']}  "
              f"in_band={r['in_L21b_band']}")


if __name__ == "__main__":
    main()
