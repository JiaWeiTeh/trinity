"""Phase 2: validate the reference solver on its OWN terms (no production ref).

Establishes the reference as converged, method-independent ground truth -- the
prerequisite for it to adjudicate the Phase-3 audit. Four legs, each reported
per state with a pass/fail gate:

  1. Convergence   -- tighten rtol/atol/epsrel/limit; do the 7 quantities stop
                      moving? (report value + last relative change.)
  2. Method-indep. -- the peaked L_bubble (and mBubble) via adaptive quad vs an
                      independent clustered-Simpson; they must agree. A value
                      that is method-independent is the TRUE integral, not a
                      quadrature artifact (this is how the uniform-Simpson bug
                      was caught: it was method-DEPENDENT).
  3. Invariants    -- inner BC v(R1)~0 (the production residual), the ODE
                      residual d(sol)/dr - RHS ~ 0, event vs independent brentq,
                      and structural (L_total=L1+L2+L3, >=0, R1<rc<r2Prime).
  4. Cross-solver  -- LSODA vs Radau vs production odeint agree on T(r).

None of these references production values, so the verdict cannot be circular.
No production change.
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import scipy.integrate
import scipy.optimize

sys.path.insert(0, os.path.dirname(__file__))
from audit import load_state  # noqa: E402
from reference import reference_bubble_luminosity, _CIEswitch  # noqa: E402
import trinity.bubble_structure.bubble_luminosity as bl  # noqa: E402
import trinity._functions.unit_conversions as cvt  # noqa: E402

QTYS = ['L1Bubble', 'L2Conduction', 'L3Intermediate', 'L_total',
        'mBubble', 'Tavg', 'T_rgoal']

# gate thresholds (report actuals; actuals are typically far tighter)
TOL_CONV = 1e-3      # last relative change as tolerances tighten
TOL_METHOD = 1e-3    # quad vs clustered-Simpson
TOL_BC = 1e-3        # |v(R1)/v(r2P)|  (fsolve solved dMdt to xtol=1e-4)
TOL_ODE = 1e-3       # d(sol)/dr vs RHS
TOL_EVENT = 1e-6     # event vs brentq
TOL_SOLVER = 1e-4    # LSODA vs Radau / odeint


def _solve(params, R1, Pb, r2P, ic, rtol=1e-10, atol=1e-13, method='LSODA'):
    def g(r, y):
        return y[1] - _CIEswitch
    g.direction = 0.0
    return scipy.integrate.solve_ivp(
        lambda r, y: bl._get_bubble_ODE(r, y, params, Pb),
        (r2P, R1), [float(x) for x in ic], method=method,
        dense_output=True, events=[g], rtol=rtol, atol=atol)


def _clustered_simpson(integrand_vec, a, b, n=40001):
    # cluster sample points near b (where both L_bubble and mBubble peak)
    u = np.linspace(0.0, 1.0, n)
    r = a + (b - a) * (1.0 - (1.0 - u) ** 3)
    return float(np.abs(scipy.integrate.simpson(integrand_vec(r), x=r)))


def convergence(params, inputs):
    a = inputs
    levels = [dict(rtol=1e-6, atol=1e-9, quad_limit=100, epsrel=1e-6),
              dict(rtol=1e-8, atol=1e-11, quad_limit=200, epsrel=1e-8),
              dict(rtol=1e-10, atol=1e-13, quad_limit=400, epsrel=1e-10)]
    res = [reference_bubble_luminosity(params, a['R1'], a['Pb'], a['r2Prime'],
                                       a['initial_conditions'], a['bubble_r_Tb'], **lv)
           for lv in levels]
    rows = {}
    worst = 0.0
    for q in QTYS:
        vals = [r[q] for r in res]
        drel = abs(vals[-1] - vals[-2]) / max(abs(vals[-1]), 1e-300)
        rows[q] = (vals[-1], drel)
        worst = max(worst, drel)
    return rows, worst, (worst < TOL_CONV)


def method_independence(params, inputs):
    a = inputs
    R1, Pb, r2P = a['R1'], a['Pb'], a['r2Prime']
    sol = _solve(params, R1, Pb, r2P, a['initial_conditions'])
    rc = float(sol.t_events[0][0])
    kB = params['k_B'].value
    mu = params['mu_ion'].value
    cieI = params['cStruc_cooling_CIE_interpolation'].value

    def f_L1(rv):
        T = sol.sol(rv)[1]
        n = Pb / (2 * kB * T)
        Lam = 10 ** (cieI(np.log10(T))) * cvt.Lambda_cgs2au
        return n ** 2 * Lam * 4 * np.pi * rv ** 2

    def f_mass(rv):
        T = sol.sol(rv)[1]
        return 4 * np.pi * (Pb / (2 * kB * T)) * mu * rv ** 2

    out = {}
    for name, fv, lo, hi in (('L1Bubble', f_L1, R1, rc), ('mBubble', f_mass, R1, r2P)):
        # adaptive quad (scalar)
        def fs(x, _fv=fv):
            return float(_fv(np.asarray(x, float)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
            q, _ = scipy.integrate.quad(fs, lo, hi, limit=400, epsrel=1e-10)
        cs = _clustered_simpson(fv, lo, hi)
        rel = abs(abs(q) - cs) / max(abs(q), 1e-300)
        out[name] = (abs(q), cs, rel)
    worst = max(v[2] for v in out.values())
    return out, worst, (worst < TOL_METHOD)


def invariants(params, inputs):
    a = inputs
    R1, Pb, r2P, ic = a['R1'], a['Pb'], a['r2Prime'], [float(x) for x in a['initial_conditions']]
    sol = _solve(params, R1, Pb, r2P, ic)
    rc = float(sol.t_events[0][0])
    res = reference_bubble_luminosity(params, R1, Pb, r2P, ic, a['bubble_r_Tb'])

    bc = abs(float(sol.sol(R1)[0]) / (ic[0] + 1e-4))
    # ODE residual
    rs = np.linspace(R1 * 1.0001, r2P * 0.9999, 50)
    ode = 0.0
    for rr in rs:
        h = rr * 1e-6
        dydr = (sol.sol(rr + h) - sol.sol(rr - h)) / (2 * h)
        rhs = np.array(bl._get_bubble_ODE(rr, sol.sol(rr), params, Pb))
        ode = max(ode, float(np.max(np.abs(dydr - rhs) / np.maximum(np.abs(rhs), 1e-300))))
    # event vs brentq
    rb = scipy.optimize.brentq(lambda r: float(sol.sol(r)[1]) - _CIEswitch, R1, r2P)
    ev = abs(rc - rb) / rb
    # structural
    struct = (abs(res['L_total'] - (res['L1Bubble'] + res['L2Conduction'] + res['L3Intermediate']))
              / max(res['L_total'], 1e-300))
    nonneg = all(res[q] >= 0 for q in ('L1Bubble', 'L2Conduction', 'L3Intermediate'))
    ordered = R1 < rc < r2P
    ok = (bc < TOL_BC and ode < TOL_ODE and ev < TOL_EVENT
          and struct < 1e-12 and nonneg and ordered)
    return dict(bc=bc, ode=ode, event=ev, struct=struct, nonneg=nonneg, ordered=ordered), ok


def cross_solver(params, inputs, ref_arrays):
    a = inputs
    R1, Pb, r2P, ic = a['R1'], a['Pb'], a['r2Prime'], [float(x) for x in a['initial_conditions']]
    lsoda = _solve(params, R1, Pb, r2P, ic, method='LSODA')
    rD, TD = ref_arrays['r_array'], ref_arrays['T_array']
    m = (rD >= min(R1, r2P)) & (rD <= max(R1, r2P))
    odeint_rel = float(np.max(np.abs(lsoda.sol(rD[m])[1] - TD[m]) / np.abs(TD[m])))
    try:
        radau = _solve(params, R1, Pb, r2P, ic, method='Radau')
        rr = np.linspace(R1 * 1.0001, r2P * 0.9999, 200)
        radau_rel = float(np.max(np.abs(lsoda.sol(rr)[1] - radau.sol(rr)[1]) / np.abs(lsoda.sol(rr)[1])))
        radau_ok = radau.success
    except Exception as e:
        radau_rel, radau_ok = float('nan'), False
        print(f"    (Radau failed: {e})")
    ok = (odeint_rel < TOL_SOLVER and radau_ok and radau_rel < TOL_SOLVER)
    return dict(odeint=odeint_rel, radau=radau_rel, radau_ok=radau_ok), ok


def main(argv):
    if not argv:
        print("usage: python validate.py <states_dir|state.pkl> [base.param]")
        return 2
    target, base = argv[0], (argv[1] if len(argv) > 1 else None)
    files = sorted(glob.glob(os.path.join(target, "*.pkl"))) if os.path.isdir(target) else [target]
    all_ok = True
    for f in files:
        kwargs = {} if base is None else {'base_param': base}
        params, inputs, ref_arrays, meta = load_state(f, **kwargs)
        print(f"\n================ {os.path.basename(f)} ================")
        crows, cworst, cok = convergence(params, inputs)
        print(f"[1] convergence (last Δrel, gate<{TOL_CONV:.0e}): worst={cworst:.2e} {'PASS' if cok else 'FAIL'}")
        for q in QTYS:
            v, d = crows[q]
            print(f"      {q:16} = {v:.6e}   Δrel={d:.2e}")
        mrows, mworst, mok = method_independence(params, inputs)
        print(f"[2] method-independence (gate<{TOL_METHOD:.0e}): worst={mworst:.2e} {'PASS' if mok else 'FAIL'}")
        for k, (q, cs, rel) in mrows.items():
            print(f"      {k:16} quad={q:.6e} clustSimpson={cs:.6e} rel={rel:.2e}")
        irows, iok = invariants(params, inputs)
        print(f"[3] invariants: {'PASS' if iok else 'FAIL'}  "
              f"BC={irows['bc']:.2e} ODE={irows['ode']:.2e} event={irows['event']:.2e} "
              f"struct={irows['struct']:.1e} nonneg={irows['nonneg']} ordered={irows['ordered']}")
        srows, sok = cross_solver(params, inputs, ref_arrays)
        print(f"[4] cross-solver: {'PASS' if sok else 'FAIL'}  "
              f"vs_odeint={srows['odeint']:.2e} vs_Radau={srows['radau']:.2e} (Radau_ok={srows['radau_ok']})")
        state_ok = cok and mok and iok and sok
        all_ok &= state_ok
        print(f"  --> state {'VALIDATED' if state_ok else 'FAILED'}")
    print(f"\n=== Phase-2 reference validation: {'PASS (ground truth established)' if all_ok else 'FAIL'} ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
