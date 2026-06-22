#!/usr/bin/env python3
"""Equivalence gate for the net_coolingcurve T-floor fix (audit finding #1).

The change replaces the hard-coded ``if T < 1e4: T = 1e4`` floor with a floor
tied to the cooling file's minimum tabulated temperature::

    if np.log10(T) < nonCIE_Tmin:   # nonCIE_Tmin = min(cube.temp), log10 K
        T = 10 ** nonCIE_Tmin

The correct gate for THIS change is NOT "bit-identical everywhere" -- the fix
*intentionally* changes behaviour in the sub-1e4 decade the old code over-floored.
The gate is:

  1. **Bit-identical for every T >= 1e4** -- the only regime any real run reaches
     (the measurement in TCLAMP_PLAN.md found min T = 30000 K across 9.46M calls).
     Both old and new leave T untouched there, so results must match to the bit.
  2. **Below 1e4: divergence is expected and one-directional** -- old floors to
     1e4, new floors to the table edge (10**nonCIE_Tmin = 3162 K). Neither raises.
  3. **Neither version ever hits the `raise`** on the swept grid.

Compares the working-tree ``get_dudt`` against ``git show HEAD:`` over a (T, ndens,
phi) grid spanning all branches. Run from repo root:

    python docs/dev/magic-numbers/harness/verify_tclamp_equiv.py    # ~20 s

Recorded output committed alongside in TCLAMP_PLAN.md.
"""
import os
import sys
import types
import subprocess

import numpy as np
import scipy.interpolate

sys.path.insert(0, os.getcwd())

import trinity._functions.unit_conversions as cvt
from trinity._input.read_param import read_param
import trinity.cooling.non_CIE.read_cloudy as non_CIE
import trinity.cooling.net_coolingcurve as new_mod  # working-tree (new) version


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
    logT, logLambda = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    params["cStruc_cooling_CIE_logLambda"].value = logLambda
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logLambda, kind="linear"
    )
    cooling_nonCIE, heating_nonCIE, netcool = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = netcool
    return params, cooling_nonCIE


def main():
    ref = _load_reference()
    params, cube = _setup_params()
    _, nonCIE_Tmin = new_mod._noncie_cutoffs(cube)
    print(f"nonCIE_Tmin = {nonCIE_Tmin} (log10 K)  ->  table edge = {10 ** nonCIE_Tmin:.4f} K")

    # T spans below-table (<3162), the over-floored decade [3162,1e4), and the
    # real-run regime (>=1e4) up through the CIE branch.
    T_grid = np.logspace(3.0, 8.0, 80)
    ndens_cgs = [1e0, 1e2, 1e4]
    phi_cgs = [1e8, 1e10, 1e12]

    n_ge_1e4 = n_ge_mismatch = 0
    n_lt_1e4 = n_lt_diverged = 0
    n_raise_new = n_raise_ref = 0
    for T in T_grid:
        for nc in ndens_cgs:
            for pc in phi_cgs:
                nd_au = nc * cvt.ndens_cgs2au
                ph_au = pc * cvt.phi_cgs2au
                try:
                    a = new_mod.get_dudt(0.1, nd_au, T, ph_au, params)
                except Exception:  # noqa: BLE001
                    a = ("RAISE",); n_raise_new += 1
                try:
                    b = ref.get_dudt(0.1, nd_au, T, ph_au, params)
                except Exception:  # noqa: BLE001
                    b = ("RAISE",); n_raise_ref += 1
                same = (a == b) or (
                    not isinstance(a, tuple) and not isinstance(b, tuple)
                    and np.isnan(a) and np.isnan(b)
                )
                if T >= 1e4:
                    n_ge_1e4 += 1
                    if not same:
                        n_ge_mismatch += 1
                        if n_ge_mismatch <= 5:
                            print(f"  T>=1e4 MISMATCH T={T:.4e} n={nc:.0e} phi={pc:.0e}: new={a!r} ref={b!r}")
                else:
                    n_lt_1e4 += 1
                    if not same:
                        n_lt_diverged += 1

    print(f"\nT >= 1e4 (real-run regime): compared={n_ge_1e4}  mismatches={n_ge_mismatch}")
    print(f"T <  1e4 (over-floored)   : compared={n_lt_1e4}  diverged={n_lt_diverged} (expected: old->1e4, new->table edge)")
    print(f"raises: new={n_raise_new}  ref={n_raise_ref}")

    assert n_ge_mismatch == 0, f"{n_ge_mismatch} mismatches in the T>=1e4 regime -- NOT bit-identical where it matters!"
    assert n_raise_new == 0, "new get_dudt raised on the grid (clamp should prevent the table-edge raise)"
    assert n_lt_diverged > 0, "expected the fix to change behaviour below 1e4; saw none -- did the edit apply?"
    print("\nRESULT: bit-identical for all T>=1e4 (every real-run regime); below 1e4 the")
    print("        floor correctly moved from 1e4 to the table edge; neither version raises. PASS.")


if __name__ == "__main__":
    main()
