"""B7, S8 (docs/dev/turnaround-stall/PLAN.md, ruled 2026-11-21): phase_events.solve_segment under its RHS budget.
Runs on an engine with apply_b7.py's s8 group (stage_b7b.sh task 'unit'); copied into test/ when B7 ships."""
import numpy as np
import pytest
import scipy.integrate

import trinity.phase_general.phase_events as events
from trinity._output.simulation_end import SimulationEndCode


class _P:
    def __init__(self, v):
        self.value = v


def _params():
    return {"solver_flags": _P(""), "EndSimulationDirectly": _P(False), "SimulationEndCode": _P(0),
            "SimulationEndReason": _P("")}


def _kw(fun, span=1e-3):
    return dict(fun=fun, t_span=(0.0, span), y0=np.array([1.0]), method="LSODA", rtol=1e-6, atol=1e-8,
                max_step=2e-5, min_step=0.0)


def test_a_solve_within_its_budget_is_solve_ivp_bit_for_bit():
    p = _params()
    a = events.solve_segment(p, _kw(lambda t, y: -1e3 * y))
    b = scipy.integrate.solve_ivp(**_kw(lambda t, y: -1e3 * y))
    assert np.array_equal(a.t, b.t) and np.array_equal(a.y, b.y) and a.success
    assert p["solver_flags"].value == "" and p["EndSimulationDirectly"].value is False


def test_a_solve_past_its_budget_ends_the_run_flagged():
    # y' = w cos(w t) with w = 1e9 /Myr needs steps near 1e-10 Myr: ~1e6 calls over 1e-3 Myr, far past the budget
    # 5e4 + 1e7 * 1e-3 = 6e4, for LSODA and for its Radau retry alike
    calls = [0]

    def fun(t, y):
        calls[0] += 1
        return np.array([1e9 * np.cos(1e9 * t)])
    p = _params()
    sol = events.solve_segment(p, _kw(fun))
    assert not sol.success and len(sol.t) == 0
    assert p["solver_flags"].value == "rhs_budget,rhs_budget_end"
    assert p["EndSimulationDirectly"].value is True
    assert p["SimulationEndCode"].value == SimulationEndCode.ERROR_SOLVER.code
    assert calls[0] <= 2 * 60_000            # each solve stopped at its budget


def test_an_error_that_is_not_the_budget_propagates():
    def fun(t, y):
        raise ValueError("shell_mass_net: refused")
    with pytest.raises(ValueError, match="shell_mass_net"):
        events.solve_segment(_params(), _kw(fun))
