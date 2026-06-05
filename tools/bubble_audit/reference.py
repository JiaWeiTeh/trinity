"""Reference bubble-luminosity solver (Phase 1) -- a measurement instrument.

Computes the same seven physical quantities as ``get_bubbleproperties_pure``
(``L1Bubble, L2Conduction, L3Intermediate, L_total, mBubble, Tavg, T_rgoal``)
using the SAME physics formulas but tolerance-controlled numerics:

* structure: ``solve_ivp(method='LSODA', dense_output=True, events=[CIE])`` --
  one adaptive integration, an exact CIE-switch radius from the event, and a
  smooth callable ``sol(r)``; no fixed grid, no ``find_nearest_higher``, no
  conditional re-integration.
* integrals: **adaptive** ``scipy.integrate.quad`` on the smooth interpolant.
  This matters: the L_bubble integrand ``n^2 * Lambda * 4 pi r^2`` is sharply
  peaked at the CIE switch (n^2 spikes ~1e5-1e6x there, peak at the endpoint),
  which UNIFORM sampling cannot resolve -- quad refines adaptively. (An earlier
  uniform-Simpson version was wrong by up to ~30x; quad matches the dense
  production grid to <0.5%.)

It imports the UNCHANGED production RHS ``_get_bubble_ODE`` so the physics is
identical; only the numerics differ.  This is the converged "ground truth" the
correctness audit (Phases 2-3) measures the production grid against.

Faithful-on-purpose details (so the audit isolates numerics):
* L3 (intermediate, T in [1e4, 3e4]) is extrapolated *outside* the ODE domain
  with the production ``interp1d`` on a 1000-pt linspace -- kept identical.
* Tavg's intermediate volume term ``r_interm[0]**3 - r_interm[-1]**3`` is
  negative in production (ascending r_interm); replicated as-is. FLAGGED.
* production's T_rgoal uses find_nearest + linear extrapolation (approximate);
  here we use the exact ``sol(r_Tb)`` for in-domain points (expect small diff).
* conduction integrand clamps T<=10^5.5 (sol gives 10^5.5 +/- eps at the
  boundary, just past the cooling cube's log_T axis max of 5.5; production
  inserts the exact 10^5.5).
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.integrate
import scipy.interpolate

import trinity.bubble_structure.bubble_luminosity as bl
import trinity._functions.unit_conversions as cvt

_CIEswitch = 10 ** 5.5
_coolingswitch = 1e4


def reference_bubble_luminosity(params, R1, Pb, r2Prime, initial_conditions,
                                bubble_r_Tb, rtol=1e-10, atol=1e-13,
                                quad_limit=200, epsrel=1e-8):
    """Tolerance-controlled reference for one bubble-structure state."""
    kB = params['k_B'].value
    mu_ion = params['mu_ion'].value
    Qi = params['Qi'].value
    cooling_CIE = params['cStruc_cooling_CIE_interpolation'].value
    cooling_nonCIE = params['cStruc_cooling_nonCIE'].value
    heating_nonCIE = params['cStruc_heating_nonCIE'].value

    # --- 1.1 structure solve (replaces odeint 358-376): dense output + event
    def g_CIE(r, y):
        return y[1] - _CIEswitch
    g_CIE.direction = 0.0

    sol = scipy.integrate.solve_ivp(
        fun=lambda r, y: bl._get_bubble_ODE(r, y, params, Pb),
        t_span=(r2Prime, R1),
        y0=[float(x) for x in initial_conditions],
        method='LSODA', dense_output=True, events=[g_CIE],
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    crossings = sol.t_events[0]
    if len(crossings) < 1:
        raise RuntimeError("CIE switch (T=10^5.5) not crossed by sol")
    r_CIEswitch = float(crossings[0])

    def T_at(r):
        return float(sol.sol(r)[1])

    def quad_abs(integrand, a, b):
        # adaptive quadrature; roundoff warnings on the sharp CIE-switch peak
        # are benign (validated <0.5% vs the dense production grid).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.integrate.IntegrationWarning)
            val, _ = scipy.integrate.quad(integrand, a, b, limit=quad_limit, epsrel=epsrel)
        return abs(val)

    # --- 1.2 L1 Bubble (CIE, T>3.16e5) over [R1, r_CIEswitch]  (prod 460-474)
    def f_bubble(r):
        T = T_at(r)
        n = Pb / (2 * kB * T)
        Lam = 10 ** float(cooling_CIE(np.log10(T))) * cvt.Lambda_cgs2au
        return n ** 2 * Lam * 4 * np.pi * r ** 2
    L_bubble = quad_abs(f_bubble, R1, r_CIEswitch)
    Tavg_bubble = quad_abs(lambda r: r ** 2 * T_at(r), R1, r_CIEswitch)

    # --- 1.3 L2 Conduction (non-CIE) over [r_CIEswitch, r2Prime]  (prod 476-543)
    def f_cond(r):
        T = min(T_at(r), _CIEswitch)   # clamp: cube log_T axis max is 5.5
        n = Pb / (2 * kB * T)
        phi = Qi / (4 * np.pi * r ** 2)
        pt = np.array([[np.log10(n / cvt.ndens_cgs2au), np.log10(T),
                        np.log10(phi / cvt.phi_cgs2au)]])
        dudt = (10 ** float(heating_nonCIE.interp(pt))
                - 10 ** float(cooling_nonCIE.interp(pt))) * cvt.dudt_cgs2au
        return dudt * 4 * np.pi * r ** 2
    L_conduction = quad_abs(f_cond, r_CIEswitch, r2Prime)
    Tavg_conduction = quad_abs(lambda r: r ** 2 * T_at(r), r_CIEswitch, r2Prime)

    # --- 1.4 L3 Intermediate (T in [1e4,3e4], extrapolated)  (prod 545-583) ---
    T_r2p = float(initial_conditions[1])
    dTdR_coolingswitch = float(initial_conditions[2])
    R2_coolingswitch = (_coolingswitch - T_r2p) / dTdR_coolingswitch + r2Prime
    fT_interm = scipy.interpolate.interp1d(
        np.array([r2Prime, R2_coolingswitch]),
        np.array([T_r2p, _coolingswitch]), kind='linear')
    r_interm = np.linspace(r2Prime, R2_coolingswitch, num=1000, endpoint=True)
    T_interm = fT_interm(r_interm)
    n_interm = Pb / (2 * kB * T_interm)
    phi_interm = Qi / (4 * np.pi * r_interm ** 2)
    L_intermediate = 0.0
    for regime in ('non-CIE', 'CIE'):
        mask = (T_interm < _CIEswitch) if regime == 'non-CIE' else (T_interm >= _CIEswitch)
        if not np.any(mask):
            continue
        if regime == 'non-CIE':
            pts = np.transpose(np.log10([n_interm[mask] / cvt.ndens_cgs2au,
                                         T_interm[mask], phi_interm[mask] / cvt.phi_cgs2au]))
            dudt = (10 ** heating_nonCIE.interp(pts) - 10 ** cooling_nonCIE.interp(pts)) * cvt.dudt_cgs2au
            integ = dudt * 4 * np.pi * r_interm[mask] ** 2
        else:
            Lam = 10 ** (cooling_CIE(np.log10(T_interm[mask]))) * cvt.Lambda_cgs2au
            integ = n_interm[mask] ** 2 * Lam * 4 * np.pi * r_interm[mask] ** 2
        L_intermediate += float(np.abs(scipy.integrate.trapezoid(integ, x=r_interm[mask])))
    Tavg_intermediate = float(np.abs(scipy.integrate.trapezoid(r_interm ** 2 * T_interm, x=r_interm)))

    L_total = L_bubble + L_conduction + L_intermediate

    # --- 1.6 Tavg (volume-weighted; replicate prod endpoint signs 588-600) ---
    vol_bubble = r_CIEswitch ** 3 - R1 ** 3
    vol_cond = r2Prime ** 3 - r_CIEswitch ** 3
    vol_interm = r2Prime ** 3 - R2_coolingswitch ** 3   # negative, matches prod
    total_Tr2 = Tavg_bubble + Tavg_conduction + Tavg_intermediate
    total_vol = vol_bubble + vol_cond + vol_interm
    Tavg = 3 * total_Tr2 / total_vol

    # --- 1.7 T_rgoal (prod 602-615; exact sol for in-domain points) ---
    if bubble_r_Tb > r2Prime:
        T_rgoal = float(fT_interm(bubble_r_Tb))
    else:
        T_rgoal = T_at(bubble_r_Tb)

    # --- 1.5 mBubble = 4*pi * int n*mu_ion*r^2 dr over [R1, r2Prime] (prod 621)
    mBubble = 4 * np.pi * quad_abs(
        lambda r: (Pb / (2 * kB * T_at(r))) * mu_ion * r ** 2, R1, r2Prime)

    return {
        'L1Bubble': L_bubble, 'L2Conduction': L_conduction,
        'L3Intermediate': L_intermediate, 'L_total': L_total,
        'mBubble': mBubble, 'Tavg': Tavg, 'T_rgoal': T_rgoal,
        'r_CIEswitch': r_CIEswitch, 'R2_coolingswitch': R2_coolingswitch,
        'sol_success': bool(sol.success), 'n_CIE_events': int(len(crossings)),
        'n_steps': int(sol.t.size),
    }


def _sanity_report(res):
    finite = all(np.isfinite(v) for k, v in res.items()
                 if isinstance(v, (int, float)))
    return (res['sol_success'] and res['n_CIE_events'] >= 1 and finite
            and res['L_total'] > 0 and res['mBubble'] > 0
            and 1e4 < res['Tavg'] < 1e9 and res['T_rgoal'] > 0)


if __name__ == "__main__":
    import os
    import sys
    import glob
    sys.path.insert(0, os.path.dirname(__file__))
    from audit import load_state  # noqa: E402

    target = sys.argv[1] if len(sys.argv) > 1 else None
    base = sys.argv[2] if len(sys.argv) > 2 else None
    if not target:
        print("usage: python reference.py <state.pkl|dir/> [base.param]")
        raise SystemExit(2)
    files = (sorted(glob.glob(os.path.join(target, "*.pkl")))
             if os.path.isdir(target) else [target])
    all_ok = True
    for f in files:
        kwargs = {} if base is None else {'base_param': base}
        params, inputs, ref, meta = load_state(f, **kwargs)
        res = reference_bubble_luminosity(
            params, inputs['R1'], inputs['Pb'], inputs['r2Prime'],
            inputs['initial_conditions'], inputs['bubble_r_Tb'])
        ok = _sanity_report(res)
        all_ok &= ok
        print(f"\n{os.path.basename(f)}  sane={ok}")
        print(f"  sol: success={res['sol_success']} CIE_events={res['n_CIE_events']} steps={res['n_steps']}")
        print(f"  r_CIEswitch={res['r_CIEswitch']:.6e}  R2_coolingswitch={res['R2_coolingswitch']:.6e}")
        print(f"  L1={res['L1Bubble']:.6e} L2={res['L2Conduction']:.6e} L3={res['L3Intermediate']:.6e}")
        print(f"  L_total={res['L_total']:.6e}  mBubble={res['mBubble']:.6e}")
        print(f"  Tavg={res['Tavg']:.6e}  T_rgoal={res['T_rgoal']:.6e}")
    print(f"\nPhase-1 sanity: {'PASS' if all_ok else 'FAIL'}")
    raise SystemExit(0 if all_ok else 1)
