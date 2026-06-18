#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Velocity-residual method variants for the HOTPATH F1 resample study.

``baseline`` is the unchanged production ``_get_velocity_residuals`` (the 60k
dense-output resample). The ``M*`` variants are the Option-(b) coarse-``t_eval``
rewrite from RESAMPLE_PLAN.md "## The fix": one ``solve_ivp(LSODA)`` with no
``dense_output`` and no 60k grid build, numerator off the integrator's own end
node, denominator off the exact IC ``v_init``, ``min_T``/monotonic off either a
fixed ``t_eval`` grid (M2000..M200) or the adaptive nodes only (Mnodes).

Each variant keeps the production signature ``(dMdt_init, params, Pb, R1)`` and
the same ``_RESIDUAL_RTOL`` / ``_BUBBLE_ATOL`` tolerances, the
``_T_INIT_BOUNDARY`` rejection, and the ``_SOLVER_FAIL_RESIDUAL`` failure
contract, so the only thing that changes vs baseline is the resample cost.
"""

import numpy as np
import scipy.integrate

import trinity.bubble_structure.bubble_luminosity as BL


def make_variant(npts):  # npts=None -> Mnodes (no t_eval)
    def _resid(dMdt_init, params, Pb, R1):
        r2Prime, T0, dTdr0, v0 = BL._get_bubble_ODE_initial_conditions(
            dMdt_init, params, Pb, R1)
        r2 = np.asarray(r2Prime).item()
        v_init = np.asarray(v0).item()
        T_init = np.asarray(T0).item()
        dTdr_init = np.asarray(dTdr0).item()
        if not np.all(np.isfinite([v_init, T_init, dTdr_init])):
            return BL._SOLVER_FAIL_RESIDUAL
        kw = {} if npts is None else {'t_eval': np.linspace(r2, R1, npts)}
        try:
            sol = scipy.integrate.solve_ivp(
                fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
                t_span=(r2, R1), y0=[v_init, T_init, dTdr_init], method='LSODA',
                rtol=BL._RESIDUAL_RTOL, atol=BL._BUBBLE_ATOL, **kw)
        except BL.BubbleSolverError:
            return BL._SOLVER_FAIL_RESIDUAL
        if not sol.success:
            return BL._SOLVER_FAIL_RESIDUAL
        v_last = sol.y[0, -1]
        T_array = sol.y[1, :]
        residual = (v_last - 0) / (v_init + 1e-4)
        min_T = np.min(T_array)
        if min_T < BL._T_INIT_BOUNDARY:
            return residual * (BL._T_INIT_BOUNDARY / (min_T + 1e-1))**2
        if np.isnan(min_T):
            return -1e3
        if not BL.operations.monotonic(T_array):
            return 1e2
        return residual
    return _resid


VARIANTS = {"baseline": BL._get_velocity_residuals, "M2000": make_variant(2000),
            "M1000": make_variant(1000), "M500": make_variant(500),
            "M200": make_variant(200), "Mnodes": make_variant(None)}

# npts per variant id (None / baseline -> "" in the CSV); used by the harness.
VARIANT_NPTS = {"baseline": "", "M2000": 2000, "M1000": 1000, "M500": 500,
                "M200": 200, "Mnodes": ""}
