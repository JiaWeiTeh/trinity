#!/usr/bin/env python3
"""HOTPATH F2.3/F2.4 — prove the cooling-cutoff caching + Lambda_CIE move are
bit-identical, and measure the speedup.

Loads the PRE-CHANGE `get_dudt` from `git show HEAD:...` as a reference, builds
the real cooling structures from `param/simple_cluster.param`, then compares the
new (working-tree) `get_dudt` against the reference over a (T, ndens, phi) grid
spanning all three branches (non-CIE / CIE / interpolation). Asserts EXACT
float equality (no tolerance) and reports per-call timing.

Run from repo root:  python docs/dev/performance/harness/verify_getdudt_equiv.py
~20 s. Recorded output committed alongside in OUTPUT.md / HOTPATH_PLAN.md F2.
"""
import os
import sys
import time
import types
import subprocess

import numpy as np

sys.path.insert(0, os.getcwd())

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity.cooling.net_coolingcurve as new_mod  # working-tree (new) version
import scipy.interpolate


def _load_reference():
    """Exec the committed (pre-change) net_coolingcurve.py as a separate module."""
    src = subprocess.check_output(
        ["git", "show", "HEAD:trinity/cooling/net_coolingcurve.py"], text=True
    )
    ref = types.ModuleType("net_coolingcurve_ref")
    exec(compile(src, "<HEAD:net_coolingcurve.py>", "exec"), ref.__dict__)
    return ref


def _setup_params():
    """Load params + build CIE and non-CIE cooling structures (as the runner does)."""
    params = read_param("param/simple_cluster.param")
    params["t_now"].value = 0.1  # Myr; any in-table age works

    # CIE (mirrors main.py)
    logT, logLambda = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    params["cStruc_cooling_CIE_logLambda"].value = logLambda
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logLambda, kind="linear"
    )
    # non-CIE cube (mirrors run_energy_phase.py)
    cooling_nonCIE, heating_nonCIE, netcool = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = netcool
    return params


def main():
    ref = _load_reference()
    params = _setup_params()

    # Grid: T spans all branches; ndens/phi a few physical cgs values -> AU inputs.
    T_grid = np.logspace(3.6, 8.0, 60)          # K (below 1e4 floored; through 1e5.5)
    ndens_cgs = [1e0, 1e2, 1e4]                  # cm^-3
    phi_cgs = [1e8, 1e10, 1e12]                  # 1/cm^2/s

    n_compared = 0
    n_mismatch = 0
    worst = 0.0
    branch_hits = {"noncie": 0, "cie": 0, "interp": 0, "other": 0}
    for T in T_grid:
        for nc in ndens_cgs:
            for pc in phi_cgs:
                nd_au = nc * cvt.ndens_cgs2au
                ph_au = pc * cvt.phi_cgs2au
                # get_dudt mutates its ndens/phi args in place (/=), so pass fresh
                # scalars to each call.
                try:
                    a = new_mod.get_dudt(0.1, nd_au, T, ph_au, params)
                except Exception as ea:  # noqa: BLE001
                    a = ("EXC", type(ea).__name__)
                try:
                    b = ref.get_dudt(0.1, nd_au, T, ph_au, params)
                except Exception as eb:  # noqa: BLE001
                    b = ("EXC", type(eb).__name__)
                n_compared += 1
                lo = np.log10(min(T, max(T, 1e4)) if T >= 1e4 else 1e4)
                if isinstance(a, tuple) or isinstance(b, tuple):
                    same = a == b
                else:
                    same = (a == b) or (np.isnan(a) and np.isnan(b))
                    if a != 0:
                        worst = max(worst, abs(a - b) / abs(a))
                if not same:
                    n_mismatch += 1
                    if n_mismatch <= 5:
                        print(f"  MISMATCH T={T:.4e} n={nc:.0e} phi={pc:.0e}: new={a!r} ref={b!r}")

    print(f"\ncompared={n_compared}  mismatches={n_mismatch}  "
          f"worst_rel_diff={worst:.3e}")

    # Timing: many calls in the non-CIE branch (the per-call-constant recompute
    # this change removes lives on every branch; non-CIE exercises the cube
    # cutoffs). Use a mid-range in-table point.
    nd_au = 1e2 * cvt.ndens_cgs2au
    ph_au = 1e10 * cvt.phi_cgs2au
    T = 10 ** 4.8
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        new_mod.get_dudt(0.1, nd_au, T, ph_au, params)
    t_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(N):
        ref.get_dudt(0.1, nd_au, T, ph_au, params)
    t_ref = time.perf_counter() - t0
    print(f"\nper-call get_dudt (non-CIE branch, N={N}):")
    print(f"  ref (old): {t_ref / N * 1e6:7.3f} us/call")
    print(f"  new      : {t_new / N * 1e6:7.3f} us/call")
    print(f"  speedup  : {t_ref / t_new:5.2f}x  ({(1 - t_new / t_ref) * 100:.1f}% faster)")

    assert n_mismatch == 0, f"{n_mismatch} bit-identity mismatches!"
    print("\nRESULT: bit-identical (0 mismatches across all branches). PASS.")


if __name__ == "__main__":
    main()
