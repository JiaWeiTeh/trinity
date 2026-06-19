#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression guard for the F1 "residual resample" hot-path optimization.

`_get_velocity_residuals` is the velocity-residual that the dMdt fsolve drives
thousands of times per run. Today it builds the ~60k-point `_create_radius_grid`
and resamples a dense `solve_ivp` solution onto it; the staged P3 patch
(docs/dev/performance/P3_PRODUCTION_PATCH.md) integrates ONCE on a coarse
`t_eval=linspace(r2Prime, R1, _RESIDUAL_NPTS=500)` and drops the resample.

This test pins the residual CONTRACT so applying P3 is a guarded drop-in:

  * `_reference_residual` reproduces the residual formula + all gate branches at
    a high resolution (npts=20000) using the SAME production helpers
    (`_get_bubble_ODE_initial_conditions`, `_get_bubble_ODE`) and the SAME
    solver/tolerances. It is independent of how production samples the path.
  * The primary test asserts `_get_velocity_residuals` matches that reference
    within 1e-3 (abs-or-rel) for several trial dMdt. This passes PRE-patch
    (60k dense vs 20k dense) AND POST-patch (500 coarse vs 20k dense) -- so it
    guards specifically against a too-small `_RESIDUAL_NPTS`. P0 saw ~1e-6, so
    1e-3 has huge margin while still catching a coarse-grid regression.
  * A sanity check that at the converged dMdt the residual is ~0 (the fsolve
    root), and a deterministic non-finite-IC branch check (-> _SOLVER_FAIL_RESIDUAL).

The fixture (test/data/residual_resample_fixture.json) is a tiny scalar-only
distillation of one captured mock_hybr state; the cooling cubes are rebuilt
deterministically here, mirroring
docs/dev/performance/harness/replay_from_dump.py:load_state.
"""

import json
import os
import warnings

import numpy as np
import pytest
import scipy.integrate
import scipy.interpolate

import trinity.bubble_structure.bubble_luminosity as BL
import trinity._functions.operations as operations
import trinity.cooling.non_CIE.read_cloudy as non_CIE
from trinity._input.read_param import read_param

# Repo root = two levels up from this file (test/ -> repo).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE_PATH = os.path.join(_REPO_ROOT, "test", "data", "residual_resample_fixture.json")

# Keys whose values are filesystem paths baked into the captured state; they
# point at the ORIGINAL checkout, which need not be this worktree. Let
# read_param(base) supply the worktree-local path instead (the bundled tables
# are byte-identical), so the test is self-contained.
_PATH_OVERRIDE_SKIP = {"path_cooling_CIE"}


def _load_fixture():
    with open(_FIXTURE_PATH) as fh:
        return json.load(fh)


def _build_params(fixture):
    """Reconstruct a full ``params`` from the distilled fixture.

    Mirrors docs/dev/performance/harness/replay_from_dump.py:load_state:
    read_param(base) -> apply scalar overrides -> rebuild the CIE table
    (loadtxt + interp1d) and the non-CIE cooling cubes (get_coolingStructure).
    The 5 large arrays dropped from the fixture are not read by the residual
    path; the cooling cubes are rebuilt, not stored.
    """
    base = os.path.join(_REPO_ROOT, fixture["base_param"])
    params = read_param(base)

    for k, v in fixture["param_values"].items():
        if k in _PATH_OVERRIDE_SKIP:
            continue
        if k in params:
            params[k].value = v

    # Rebuild CIE cooling from the (worktree-local) path read_param supplied.
    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear"
    )

    # Rebuild non-CIE cubes (depend only on t_now / Z / rotation / path).
    cooling_nonCIE, heating_nonCIE, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = net
    return params


def _reference_residual(dMdt, params, Pb, R1, npts=20000):
    """High-resolution reference for ``_get_velocity_residuals``.

    Replicates production's residual formula and EVERY return branch
    (non-finite ICs / solver-fail -> _SOLVER_FAIL_RESIDUAL, min_T<floor penalty,
    nan -> -1e3, non-monotonic -> 1e2) using the production helpers and the same
    LSODA / _RESIDUAL_RTOL / _BUBBLE_ATOL, but on a dense `t_eval` grid that is
    independent of production's sampling choice.
    """
    r2Prime, T_r2Prime, dTdr_r2Prime, v_r2Prime = BL._get_bubble_ODE_initial_conditions(
        dMdt, params, Pb, R1
    )
    r2Prime_val = np.asarray(r2Prime).item()
    v_init = np.asarray(v_r2Prime).item()
    T_init = np.asarray(T_r2Prime).item()
    dTdr_init = np.asarray(dTdr_r2Prime).item()

    if not np.all(np.isfinite([v_init, T_init, dTdr_init])):
        return BL._SOLVER_FAIL_RESIDUAL
    try:
        sol = scipy.integrate.solve_ivp(
            fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
            t_span=(r2Prime_val, R1),
            y0=[v_init, T_init, dTdr_init],
            method="LSODA",
            t_eval=np.linspace(r2Prime_val, R1, npts),
            rtol=BL._RESIDUAL_RTOL,
            atol=BL._BUBBLE_ATOL,
        )
    except BL.BubbleSolverError:
        return BL._SOLVER_FAIL_RESIDUAL
    if not sol.success:
        return BL._SOLVER_FAIL_RESIDUAL

    v_array = sol.y[0]
    T_array = sol.y[1]

    residual = (v_array[-1] - 0) / (v_array[0] + 1e-4)

    min_T = np.min(T_array)
    if min_T < BL._T_INIT_BOUNDARY:
        return residual * (BL._T_INIT_BOUNDARY / (min_T + 1e-1)) ** 2
    if np.isnan(min_T):
        return -1e3
    if not operations.monotonic(T_array):
        return 1e2
    return residual


def _close(a, b, tol=1e-3):
    """Absolute-or-relative closeness with a clear margin."""
    return abs(a - b) <= tol * max(1.0, abs(b))


@pytest.fixture(scope="module")
def fixture_state():
    fixture = _load_fixture()
    params = _build_params(fixture)
    return fixture, params


def test_coarse_matches_highres_reference(fixture_state):
    """Production residual ~= 20k-dense reference for dMdt around the root.

    Guards the resample: pre-patch (60k) and post-patch (500) must both track
    the dense reference within 1e-3. A too-small _RESIDUAL_NPTS breaks this.
    """
    fixture, params = fixture_state
    Pb = fixture["Pb"]
    R1 = fixture["R1"]
    dMdt0 = fixture["dMdt_converged"]

    for factor in (0.9, 1.0, 1.1):
        dMdt = dMdt0 * factor
        prod = BL._get_velocity_residuals(dMdt, params, Pb, R1)
        ref = _reference_residual(dMdt, params, Pb, R1, npts=20000)
        assert _close(prod, ref), (
            f"residual mismatch at dMdt={dMdt:.6e} (x{factor}): "
            f"production={prod:.6e} reference(20k)={ref:.6e} "
            f"absdiff={abs(prod - ref):.3e} -- _RESIDUAL_NPTS may be too small "
            f"or the resample changed behaviour."
        )


def test_residual_small_at_converged_dMdt(fixture_state):
    """At the captured fsolve root the residual is ~0."""
    fixture, params = fixture_state
    Pb = fixture["Pb"]
    R1 = fixture["R1"]
    dMdt0 = fixture["dMdt_converged"]

    resid = BL._get_velocity_residuals(dMdt0, params, Pb, R1)
    assert abs(resid) < 1e-3, (
        f"residual at converged dMdt={dMdt0:.6e} is {resid:.6e}, expected ~0 "
        f"(fixture dMdt_converged should be an fsolve root)."
    )


def test_nonfinite_ics_return_solver_fail(fixture_state):
    """dMdt=0 -> dR2 denominator 0 -> non-finite ICs -> _SOLVER_FAIL_RESIDUAL.

    Verified against _get_bubble_ODE_initial_conditions: dMdt=0 drives the dR2
    denominator ``constant * dMdt / (4*pi*R2**2)`` to 0, so dR2, then T/dTdr/v,
    go non-finite. Production's dMdt always arrives from scipy fsolve as a numpy
    float, for which ``x / 0.0`` yields ``inf``/``nan`` (a RuntimeWarning, not an
    exception) -- so the residual takes the ``isfinite`` guard and returns
    _SOLVER_FAIL_RESIDUAL. A bare Python ``0.0`` would instead raise a
    ZeroDivisionError (all-Python-float operands), which is NOT the production
    path; we therefore probe with ``np.float64(0.0)`` to match how fsolve calls it.

    This is the deterministic, cheap branch. The min_T / monotonic gates are not
    reproducible without an expensive collapsing-T search and are left to the
    P3 patch's own equivalence gate.
    """
    fixture, params = fixture_state
    Pb = fixture["Pb"]
    R1 = fixture["R1"]
    dMdt_zero = np.float64(0.0)  # fsolve passes numpy floats, not Python floats

    # Confirm the IC path is genuinely non-finite for dMdt=0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic = BL._get_bubble_ODE_initial_conditions(dMdt_zero, params, Pb, R1)
        ic_vals = [np.asarray(x).item() for x in (ic[3], ic[1], ic[2])]  # v, T, dTdr
        assert not np.all(np.isfinite(ic_vals)), (
            f"expected non-finite ICs at dMdt=0, got {ic_vals}"
        )
        resid = BL._get_velocity_residuals(dMdt_zero, params, Pb, R1)

    assert resid == BL._SOLVER_FAIL_RESIDUAL, (
        f"non-finite-IC residual was {resid}, expected "
        f"_SOLVER_FAIL_RESIDUAL={BL._SOLVER_FAIL_RESIDUAL}."
    )
