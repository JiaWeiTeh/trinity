#!/usr/bin/env python3
"""Verify the shell-ODE float64 overflow is real and lives in the discarded tail.

Monkeypatches scipy.integrate.odeint, drives a real `simple_cluster` run, captures
the FIRST ionised shell solve, then bails (we only need one). Proves:
  (1) nShell reaches ~1e65 code units (== ~1e10 cm^-3 physical),
  (2) the n^2 recombination pole overflows float64 a few steps past the front,
  (3) the overflow row index > the phi<=1e-9 truncation index (discarded tail),
so the consumed shell properties are clean while odeint still floods warnings.

Run from repo root:  python docs/dev/shell-solver/harness/verify_overflow.py
~30 s. Recorded output is committed in OVERFLOW_FIX_PLAN.md §9 (2026-06-18).
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import runpy
import warnings

import numpy as np
import scipy.integrate as si

np.seterr(over="warn")
_orig = si.odeint
cap = {}


class _Stop(Exception):
    pass


def spy(func, y0, t, args=(), **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = _orig(func, y0, t, args=args, **kw)
    is_ionised = args[1] if len(args) >= 2 else None
    if is_ionised and "first" not in cap:
        n, phi = sol[:, 0], sol[:, 1]
        finite = np.isfinite(n)
        ovf_idx = int(np.argmax(~finite)) if (~finite).any() else -1
        pc = np.nonzero(phi <= 1e-9)[0]
        front_idx = int(pc[0]) if len(pc) else -1
        cap["first"] = dict(
            y0_n=float(y0[0]),
            n_front=(float(n[front_idx]) if front_idx >= 0 else None),
            front_idx=front_idx,
            ovf_idx=ovf_idx,
            npts=len(n),
            n_max_finite=(float(np.nanmax(n[finite])) if finite.any() else None),
            overflow_warns=sum("overflow" in str(x.message) for x in w),
        )
        raise _Stop
    return sol


si.odeint = spy
sys.argv = ["run.py", "param/simple_cluster.param"]
try:
    runpy.run_path("run.py", run_name="__main__")
except (_Stop, SystemExit):
    pass

c = cap.get("first")
print("\n=== FIRST IONISED SOLVE CAPTURE (real simple_cluster) ===")
if not c:
    print("no ionised solve captured")
    sys.exit(1)
for k, v in c.items():
    print(f"  {k:16s} = {v}")
nf, nm = c["n_front"], c["n_max_finite"]
print(f"\n  nShell(front) ~ {nf:.3e} code-units == {nf * 3.40e-56:.3e} cm^-3 physical")
print(f"  nShell(front)**2 = {nf**2:.3e}  -> overflows float64 (1.8e308)? {nf**2 > 1.7e308}")
print(f"  max finite nShell before inf = {nm:.3e}  (float64 n^2 overflow needs n > 1.34e154)")
print(
    f"  overflow row PAST front (discarded tail)? "
    f"ovf_idx={c['ovf_idx']} > front_idx={c['front_idx']} -> {c['ovf_idx'] > c['front_idx']}"
)
