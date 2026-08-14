#!/usr/bin/env python3
"""Finding #3 per-call study: the ``dt = 1e-9`` Myr central-difference step in
``get_current_sps_feedback`` (trinity/sps/update_feedback.py) vs the exact
derivative of the same cubic interpolant.

What this measures (no trinity/ source is modified):

  1. IDENTITY -- ``scipy.interpolate.interp1d(t, y, kind='cubic')`` (what
     ``read_sps.get_interpolation`` builds) evaluates identically to
     ``make_interp_spline(t, y, k=3)`` (both are the not-a-knot cubic B-spline),
     so the latter's ``.derivative()`` is THE exact derivative of the
     interpolant the production FD is differencing. Verified numerically here,
     not assumed.
  2. FD ERROR -- relative error of the h=1e-9 central difference against that
     exact derivative, over a dense log grid covering every phase window
     (1a: t0~1e-8..3e-3 Myr; 1b/1c/2 beyond), plus the knot neighbourhoods.
  3. h-SWEEP -- the same max/median error for h from 1e-5 down to 1e-13, to
     locate the truncation/roundoff trade-off basin and where 1e-9 sits in it.
  4. EDGE WINDOW -- the in-function range check accepts t_min <= t <= t_max,
     but the FD then evaluates t +/- h, so any t within 1e-9 Myr of either
     table edge raises interp1d's bounds error. Demonstrated, not speculated.
  5. LEVER -- pdotdot_total's only physics consumer is the phase-1b beta-delta
     chain, a_coeff = 1.5*pdotdot/pdot (get_betadelta.py, get_bubbleParams.py),
     so the a_coeff noise floor implied by (2) is reported alongside.

f_mass scales pdot_total linearly, so every RELATIVE number here is
mass/sfe-independent; one table covers all configs.

Usage (from repo root; writes docs/dev/magic-numbers/data/pdotdot_percall.csv):
    python docs/dev/magic-numbers/harness/pdotdot_study.py <param file>
"""
import csv
import os
import sys

import numpy as np
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.getcwd())

from trinity._input import read_param  # noqa: E402
from trinity.sps import read_sps  # noqa: E402
from trinity.sps.update_feedback import get_current_sps_feedback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "pdotdot_percall.csv")
H_PROD = 1e-9  # the constant under audit


def main():
    params = read_param.read_param(sys.argv[1])
    f_mass = params['mCluster'] / params['sps_refmass']
    sps = read_sps.read_sps(f_mass, params)
    sps_f = read_sps.get_interpolation(sps)
    params['sps_f'].value = sps_f

    t_knots = np.asarray(sps[0], dtype=float)
    pdot = np.asarray(sps[10], dtype=float)
    fpdot = sps_f['fpdot_total']
    spline = make_interp_spline(t_knots, pdot, k=3)
    dspline = spline.derivative()

    rows = []

    # --- 1. identity: interp1d(kind='cubic') vs make_interp_spline(k=3) ---
    dense = np.linspace(t_knots[0], t_knots[-1], 200001)
    ident = np.max(np.abs(fpdot(dense) - spline(dense)))
    scale = np.max(np.abs(pdot))
    rows.append(("identity_maxabsdiff_interp1d_vs_bspline", f"{ident:.3e}",
                 f"rel {ident / scale:.3e} of max|pdot|"))

    # --- 2. FD(h=1e-9) vs exact derivative, on the t ranges runs visit ---
    # log grid over the full table (avoiding the +/-h edge windows), plus
    # every knot neighbourhood at +/- {0.3h, 3h, 100h}.
    tgrid = np.geomspace(max(t_knots[0] + 2 * H_PROD, 1e-8),
                         t_knots[-1] - 2 * H_PROD, 20000)
    near = []
    for tk in t_knots[1:-1]:
        for off in (-100 * H_PROD, -3 * H_PROD, -0.3 * H_PROD,
                    0.3 * H_PROD, 3 * H_PROD, 100 * H_PROD):
            tt = tk + off
            if t_knots[0] + 2 * H_PROD < tt < t_knots[-1] - 2 * H_PROD:
                near.append(tt)
    tall = np.sort(np.concatenate([tgrid, np.array(near)]))

    exact = dspline(tall)
    fd = (fpdot(tall + H_PROD) - fpdot(tall - H_PROD)) / (2.0 * H_PROD)
    # relative to a floor that keeps flat-derivative regions honest:
    # |exact| where it is meaningful, else the global derivative scale.
    dscale = np.max(np.abs(exact))
    denom = np.maximum(np.abs(exact), 1e-6 * dscale)
    rel = np.abs(fd - exact) / denom
    iworst = int(np.argmax(rel))
    rows.append(("fd_h1e-9_relerr_max", f"{rel.max():.3e}",
                 f"at t={tall[iworst]:.6e} Myr (denom floored at 1e-6*max|d|)"))
    rows.append(("fd_h1e-9_relerr_median", f"{np.median(rel):.3e}", ""))
    rows.append(("fd_h1e-9_relerr_p99", f"{np.percentile(rel, 99):.3e}", ""))
    # absolute-vs-scale view (immune to denominator choice)
    rows.append(("fd_h1e-9_abserr_max_over_dscale",
                 f"{np.max(np.abs(fd - exact)) / dscale:.3e}",
                 f"dscale=max|dpdot/dt|={dscale:.3e} au"))

    # --- 3. h sweep ---
    for h in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        ok = (tall - h > t_knots[0]) & (tall + h < t_knots[-1])
        fdh = (fpdot(tall[ok] + h) - fpdot(tall[ok] - h)) / (2.0 * h)
        r = np.abs(fdh - dspline(tall[ok])) / np.maximum(
            np.abs(dspline(tall[ok])), 1e-6 * dscale)
        rows.append((f"hsweep_relerr_max_h{h:.0e}", f"{r.max():.3e}", ""))

    # --- 4. edge windows crash (latent; needs full params dict) ---
    for name, t_edge in (("tmin", float(t_knots[0])), ("tmax", float(t_knots[-1]))):
        t_in = t_edge + (0.5 * H_PROD if name == "tmin" else -0.5 * H_PROD)
        try:
            get_current_sps_feedback(t_in, params)
            rows.append((f"edge_{name}_plus_halfh", "NO-CRASH",
                         f"t={t_in!r} inside range check but t-/+h outside table"))
        except ValueError as e:
            rows.append((f"edge_{name}_plus_halfh", "CRASH",
                         f"ValueError: {str(e)[:80]}"))

    # --- 5. lever: a_coeff noise implied by the FD error ---
    # a_coeff = 1.5*pdotdot/pdot; da/a = d(pdotdot)/pdotdot where pdotdot != 0.
    a_exact = 1.5 * exact / spline(tall)
    a_fd = 1.5 * fd / spline(tall)
    rows.append(("acoeff_absdiff_max", f"{np.max(np.abs(a_fd - a_exact)):.3e}",
                 f"a_coeff spans [{a_exact.min():.3e}, {a_exact.max():.3e}] /Myr"))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        fh.write("# finding #3 per-call study: FD h=1e-9 vs exact not-a-knot cubic "
                 "spline derivative of fpdot_total\n")
        fh.write(f"# param={sys.argv[1]} f_mass={float(f_mass):.6g} "
                 f"knots={len(t_knots)} t_table=[{t_knots[0]:.6g},{t_knots[-1]:.6g}] Myr; "
                 "relative numbers are f_mass-independent\n")
        fh.write("# command: python docs/dev/magic-numbers/harness/pdotdot_study.py "
                 f"{sys.argv[1]}\n")
        w = csv.writer(fh)
        w.writerow(["quantity", "value", "note"])
        w.writerows(rows)
    for r in rows:
        print(f"{r[0]:<45} {r[1]:>12}  {r[2]}")
    print(f"\nwrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
