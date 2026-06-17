#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot diagnostic: grab the FIRST real in-run shell-solve problem
(func, y0, t, args) via monkeypatch, then probe several solve_ivp(LSODA)
configurations on it to localise WHY solve_ivp fails where odeint succeeds.

Run:
    cd /home/user/trinity
    python docs/dev/shell-solver/harness/diagnose_first_call.py
"""
import sys
import logging
from pathlib import Path

import numpy as np
import scipy.integrate

TRINITY_ROOT = Path(__file__).resolve().parents[4]
if str(TRINITY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRINITY_ROOT))

_REAL = scipy.integrate.odeint
_grab = {}


class _Stop(Exception):
    pass


def _patched(func, y0, t, args=(), **kw):
    _grab["func"] = func
    _grab["y0"] = np.asarray(y0, float)
    _grab["t"] = np.asarray(t, float)
    _grab["args"] = args
    # args = (f_cover, is_ionised, params)
    _grab["params"] = args[2] if len(args) >= 3 else None
    _grab["sol_odeint"] = _REAL(func, y0, t, args=args, **kw)
    raise _Stop()


def main():
    logging.disable(logging.CRITICAL)
    scipy.integrate.odeint = _patched
    try:
        from trinity._input import read_param
        from trinity import main as trinity_main
        params = read_param.read_param(str(TRINITY_ROOT / "param" / "simple_cluster.param"))
        trinity_main.start_expansion(params)
    except _Stop:
        pass
    finally:
        scipy.integrate.odeint = _REAL
    logging.disable(logging.NOTSET)

    func = _grab["func"]; y0 = _grab["y0"]; t = _grab["t"]; args = _grab["args"]
    sol_odeint = _grab["sol_odeint"]

    print("=== first real shell-solve problem ===")
    print(f"y0 = {y0}")
    print(f"len(t) = {len(t)}   t[0]={t[0]!r}   t[-1]={t[-1]!r}")
    print(f"t span = {t[-1] - t[0]:.6e} pc   nominal step = {(t[-1]-t[0])/(len(t)-1):.6e} pc")
    dt = np.diff(t)
    print(f"min dt = {dt.min():.6e}  max dt = {dt.max():.6e}  "
          f"strictly increasing? {bool(np.all(dt > 0))}")
    print(f"unique t values: {len(np.unique(t))} / {len(t)}")
    # float spacing near these radii
    print(f"np.spacing(t[0]) = {np.spacing(t[0]):.3e}  "
          f"(step/spacing ratio = {((t[-1]-t[0])/(len(t)-1))/np.spacing(t[0]):.1f})")
    print(f"odeint endpoint = {sol_odeint[-1]}")

    def fun(r, y):
        return np.asarray(func(y, r, *args))

    rtol = atol = 1.49012e-8

    def try_variant(label, **kwargs):
        try:
            sol = scipy.integrate.solve_ivp(
                fun, (float(t[0]), float(t[-1])), y0, method="LSODA",
                rtol=rtol, atol=atol, **kwargs)
            end = sol.y.T[-1] if sol.y.size else None
            print(f"[{label:38s}] success={sol.success} status={sol.status} "
                  f"nfev={sol.nfev} msg={sol.message!r}")
            if end is not None and sol.success:
                rel = np.abs(end - sol_odeint[-1]) / np.maximum(np.abs(sol_odeint[-1]), 1e-300)
                print(f"{'':40s} endpoint={end}  rel_vs_odeint={rel}")
            return sol
        except Exception as exc:  # noqa: BLE001
            print(f"[{label:38s}] EXCEPTION {type(exc).__name__}: {exc}")
            return None

    print("\n=== solve_ivp(LSODA) variants ===")
    try_variant("t_eval + dense_output (harness cfg)", t_eval=t, dense_output=True)
    try_variant("t_eval only (no dense_output)", t_eval=t)
    try_variant("dense_output only (no t_eval)", dense_output=True)
    try_variant("bare (no t_eval, no dense)")
    # also LSODA with min_step / first_step nudges
    try_variant("bare + first_step=span/1000",
                first_step=(t[-1] - t[0]) / 1000.0)

    # Try odeint-equivalent with solve_ivp on a COARSER but realistic span:
    # integrate the same RHS over the full slice but only ask for 50 eval pts.
    t_coarse = np.linspace(t[0], t[-1], 50)
    try_variant("t_eval=linspace(50) + dense", t_eval=t_coarse, dense_output=True)

    # RK45 sanity (different method) on the harness cfg
    try:
        sol = scipy.integrate.solve_ivp(
            fun, (float(t[0]), float(t[-1])), y0, method="RK45",
            rtol=rtol, atol=atol, t_eval=t, dense_output=True)
        print(f"[{'RK45 t_eval+dense':38s}] success={sol.success} status={sol.status} "
              f"msg={sol.message!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{'RK45 t_eval+dense':38s}] EXCEPTION {type(exc).__name__}: {exc}")

    # --- profile-vs-profile: where do odeint and solve_ivp(t_eval) diverge? ---
    print("\n=== profile comparison: odeint vs solve_ivp(t_eval-only) ===")
    sol_te = scipy.integrate.solve_ivp(
        fun, (float(t[0]), float(t[-1])), y0, method="LSODA",
        rtol=rtol, atol=atol, t_eval=t)
    iv = sol_te.y.T
    od = np.asarray(sol_odeint)
    print(f"odeint    finite all? {np.all(np.isfinite(od))}   "
          f"first nonfinite row: "
          f"{int(np.argmax(~np.all(np.isfinite(od), axis=1))) if not np.all(np.isfinite(od)) else 'none'}")
    print(f"solve_ivp finite all? {np.all(np.isfinite(iv))}   "
          f"first nonfinite row: "
          f"{int(np.argmax(~np.all(np.isfinite(iv), axis=1))) if not np.all(np.isfinite(iv)) else 'none'}")
    # find first row index where either goes non-finite or they diverge >1e-3
    both_finite = np.all(np.isfinite(od), axis=1) & np.all(np.isfinite(iv), axis=1)
    n_both = int(np.sum(both_finite))
    print(f"rows where BOTH finite: {n_both} / {len(t)}")
    if n_both > 0:
        idxs = np.where(both_finite)[0]
        last_common = idxs[-1]
        print(f"last common finite index = {last_common} (r={t[last_common]:.6e})")
        for nm, j in (("n", 0), ("phi", 1), ("tau", 2)):
            a = od[idxs, j]; b = iv[idxs, j]
            denom = np.maximum(np.abs(a), 1e-300)
            rel = np.abs(a - b) / denom
            print(f"  {nm:3s}: max rel diff over common-finite rows = {np.max(rel):.3e}  "
                  f"(odeint[{last_common}]={a[-1]:.4e}, ivp={b[-1]:.4e})")
    # show the first 5 rows side by side
    print("  first 5 rows (n column):  r | odeint_n | ivp_n")
    for k in range(min(5, len(t))):
        print(f"    {t[k]:.6e}  {od[k,0]:.6e}  {iv[k,0]:.6e}")

    # --- where does the PHYSICAL truncation land? --------------------------
    # Replicate shell_structure's termination logic for the ionised region on
    # the odeint profile to see whether the cutoff (idx) is reached BEFORE the
    # float-overflow rows (so the physically-used part is identical to 1e-9).
    print("\n=== physical truncation index (ionised region) ===")
    params = _grab.get("params")
    nShell = od[:, 0]; phi = od[:, 1]
    rstep = t[1] - t[0]
    mu_H = params['mu_convert'].value
    mShell_end = params['shell_mass'].value
    mShell = np.empty_like(t)
    mShell[0] = 0.0
    mShell[1:] = nShell[1:] * mu_H * 4 * np.pi * t[1:]**2 * rstep
    mcum = np.cumsum(mShell)
    massCondition = mcum >= mShell_end
    phiCondition = phi <= 1e-9
    idx_arr = np.nonzero(massCondition | phiCondition)[0]
    idx = int(idx_arr[0]) if len(idx_arr) else len(t) - 1
    print(f"shell_mass(end) = {mShell_end:.4e}  | mShell_arr_cum[-1] = {mcum[-1]:.4e}")
    print(f"first mass-condition idx = "
          f"{int(np.argmax(massCondition)) if massCondition.any() else 'never'}")
    print(f"first phi-condition  idx = "
          f"{int(np.argmax(phiCondition)) if phiCondition.any() else 'never'}")
    print(f">>> PHYSICAL truncation idx used = {idx} (r={t[idx]:.6e})")
    print(f">>> rows where both solvers finite = 26; "
          f"truncation {'<' if idx < 26 else '>='} 26  -> "
          f"{'physically-used part IS in the agreeing region' if idx < 26 else 'truncation is PAST the overflow point'}")


if __name__ == "__main__":
    main()
