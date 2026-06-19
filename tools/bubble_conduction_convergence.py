"""Conduction-zone luminosity convergence audit (analysis only; no behavior change).

Quantifies how under-resolved the production ``odeint`` conduction-zone integral
is, and what the *converged* value is, at real Phase-1a bubble states.

Background: the bubble structure is integrated with ``scipy.integrate.odeint``
on a ~60k-point legacy grid. That grid fuses two concerns the physics wants
separate -- the *integration* step control and the *output* sampling used for
the conduction-zone trapezoid integral. Refining the output sampling means
asking LSODA for near-duplicate output radii, which both (a) stresses the
integrator (the intermittent crash; see docs/dev/archive/bubble/integrator-robustness.md)
and (b) still under-resolves the trapezoid. This tool measures (b): it
re-integrates each state with ``solve_ivp(dense_output=True)`` -- whose accuracy
is set by ``rtol`` independently of output sampling -- and converges the
conduction integral by refining the (cheap) dense-output sampling ``K`` alone.

For each Phase-1a state it reports:
  * the production odeint ``bubble_L2Conduction``,
  * the converged ``L_conduction`` (solve_ivp + dense-output K-sweep, using the
    same ``T < 10**5.5`` non-CIE mask the production code applies),
  * the relative bias of the production value vs the converged value,
  * solve_ivp success/step-count across ``rtol`` (mechanism + efficiency).

Single-threaded, so the numbers are deterministic. This does NOT change any
runtime behavior; it is a measurement harness for model-author sign-off.

Usage::

    python tools/bubble_conduction_convergence.py [--param PARAM] [--stride N]
        [--max-states M] [--param-text "mCloud 1e5\\nsfe 0.3\\n..."]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# Pin BLAS threads BEFORE numpy import so each solve is deterministic.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np  # noqa: E402
import scipy.integrate  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from trinity._input import read_param  # noqa: E402
from trinity import main  # noqa: E402
from trinity.bubble_structure import bubble_luminosity as BL  # noqa: E402
from trinity._functions import unit_conversions as cvt  # noqa: E402

_trap = getattr(np, "trapezoid", None) or np.trapz
_CIE = 10 ** 5.5  # upper edge of the non-CIE (conduction) cooling table
_K_SWEEP = (500, 2000, 10000, 50000, 200000)
_RTOLS = (1e-6, 1e-8, 1e-10)


def _L_conduction(sol_fn, r_lo, r_hi, params, Pb, K):
    """Conduction-zone luminosity from a continuous solution, sampled at K
    points. Mirrors bubble_luminosity._bubble_luminosity's non-CIE
    integral, including the production ``T < 10**5.5`` mask (line ~568)."""
    r = np.linspace(r_hi, r_lo, K)
    T = sol_fn(r)[1]
    m = T < _CIE
    r, T = r[m], T[m]
    if r.size < 2:
        return np.nan
    mu_ratio = params["mu_convert"].value / params["mu_ion"].value
    n = Pb / (mu_ratio * params["k_B"].value * T)
    phi = params["Qi"].value / (4 * np.pi * r ** 2)
    xi = np.transpose(np.log10([n / cvt.ndens_cgs2au, T, phi / cvt.phi_cgs2au]))
    cool = 10 ** params["cStruc_cooling_nonCIE"].value.interp(xi)
    heat = 10 ** params["cStruc_heating_nonCIE"].value.interp(xi)
    dudt = (heat - cool) * cvt.dudt_cgs2au
    return abs(_trap(dudt * 4 * np.pi * r ** 2, x=r))


def _audit_state(params):
    """Return a dict of convergence metrics for the current bubble state, or
    None if the state has no conduction zone / cannot be integrated."""
    bp = BL.get_bubbleproperties_pure(params)  # production (odeint) reference

    R1 = scipy.optimize.brentq(
        BL.get_bubbleParams.get_r1, 1e-3 * params["R2"].value, params["R2"].value,
        args=([params["Lmech_total"].value, params["Eb"].value,
               params["v_mech_total"].value, params["R2"].value]))
    Pb = BL.get_bubbleParams.bubble_E2P(
        params["Eb"].value, params["R2"].value, R1, params["gamma_adia"].value)
    r2P, T0, dTdr0, v0 = BL._get_bubble_ODE_initial_conditions(
        bp.bubble_dMdt, params, Pb, R1)

    # solve_ivp across rtol: success + step count (mechanism + efficiency)
    sols, steps = {}, {}
    for rtol in _RTOLS:
        s = scipy.integrate.solve_ivp(
            fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
            t_span=(r2P, R1), y0=[v0, T0, dTdr0], method="LSODA",
            dense_output=True, rtol=rtol, atol=1e-12)
        sols[rtol], steps[rtol] = s, (s.success, int(s.t.size))
    s = sols[1e-10]
    if not s.success:
        return None

    # inner edge of the conduction band: where T first reaches 10**5.5
    rr = np.linspace(r2P, R1, 200000)
    TT = s.sol(rr)[1]
    if not np.any(TT >= _CIE):
        return None
    r_CIE = rr[int(np.argmax(TT >= _CIE))]

    # converged L_conduction via K-sweep (accuracy from K, not the integrator)
    Lc = {K: _L_conduction(s.sol, r_CIE, r2P, params, Pb, K) for K in _K_SWEEP}
    converged = Lc[_K_SWEEP[-1]]
    prod = bp.bubble_L2Conduction
    bias = (prod - converged) / converged if converged else np.nan

    # rtol-independence of the converged integral
    rtol_indep = {rtol: _L_conduction(sols[rtol].sol, r_CIE, r2P, params, Pb,
                                      _K_SWEEP[-1]) for rtol in _RTOLS}

    return {
        "R1": R1, "r2P": r2P, "T0": T0, "dMdt": bp.bubble_dMdt,
        "prod_L2Conduction": prod, "converged_Lc": converged, "bias": bias,
        "Lc_sweep": Lc, "steps": steps, "rtol_indep": rtol_indep,
        "prod_LTotal": bp.bubble_LTotal,
    }


def main_audit(param_path, stride, max_states):
    rows = []
    real_gbp = BL.get_bubbleproperties_pure
    counter = {"i": 0}

    def hook(params):
        i = counter["i"]
        counter["i"] += 1
        if (i % stride == 0) and len(rows) < max_states:
            BL.get_bubbleproperties_pure = real_gbp  # avoid recursion in audit
            try:
                rec = _audit_state(params)
                if rec is not None:
                    rec["state"] = i
                    rows.append(rec)
                    r = rows[-1]
                    print(f"  state {i:>3}: prod={r['prod_L2Conduction']:.4e}  "
                          f"converged={r['converged_Lc']:.4e}  "
                          f"bias={r['bias']*100:+.2f}%  "
                          f"steps(rtol1e-8)={r['steps'][1e-8][1]}")
                    sys.stdout.flush()
            finally:
                BL.get_bubbleproperties_pure = hook
        return real_gbp(params)

    BL.get_bubbleproperties_pure = hook

    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        params = read_param.read_param(param_path)
        print(f"Driving run; auditing every {stride}th Phase-1a state "
              f"(max {max_states})...")
        try:
            main.start_expansion(params)
        except SystemExit:
            pass
        except BaseException as e:  # a live crash is itself a data point
            print(f"  [live run ended: {type(e).__name__}: {str(e)[:80]}]")

    _report(rows)
    return rows


def _report(rows):
    if not rows:
        print("\nNo auditable states (no conduction zone reached).")
        return
    print("\n================ CONDUCTION-ZONE CONVERGENCE SUMMARY ================")
    print(f"{'state':>5} {'R1':>9} {'r2Prime':>9} {'prod_L2Cond':>13} "
          f"{'converged':>13} {'bias%':>8} {'steps@1e-8':>10}")
    for r in rows:
        print(f"{r['state']:>5} {r['R1']:>9.5f} {r['r2P']:>9.5f} "
              f"{r['prod_L2Conduction']:>13.5e} {r['converged_Lc']:>13.5e} "
              f"{r['bias']*100:>+8.2f} {r['steps'][1e-8][1]:>10}")
    biases = np.array([r["bias"] for r in rows])
    print(f"\nproduction odeint@20k vs converged L_conduction: "
          f"mean bias {np.mean(biases)*100:+.2f}%, "
          f"max |bias| {np.max(np.abs(biases))*100:.2f}% over {len(rows)} states")
    # mechanism: did solve_ivp ever fail?
    fails = [(r["state"], rt) for r in rows for rt, (ok, _) in r["steps"].items() if not ok]
    print(f"solve_ivp failures across all states/rtols: {len(fails)}  "
          f"(efficiency: typical step count "
          f"{int(np.median([r['steps'][1e-8][1] for r in rows]))} @ rtol=1e-8)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--param", default=None, help="path to a .param file")
    ap.add_argument("--param-text", default=None,
                    help="inline param body (used if --param is omitted)")
    ap.add_argument("--stride", type=int, default=8,
                    help="audit every Nth Phase-1a state (default 8)")
    ap.add_argument("--max-states", type=int, default=12,
                    help="max states to audit (default 12)")
    args = ap.parse_args()

    if args.param:
        param_path = args.param
    else:
        text = args.param_text or (
            "mCloud      1e5\nsfe         0.3\nstop_t      1e-4\nmodel_name  conv\n")
        tf = tempfile.NamedTemporaryFile("w", suffix=".param", delete=False)
        tf.write(text.replace("\\n", "\n"))
        tf.close()
        param_path = tf.name

    main_audit(param_path, args.stride, args.max_states)
