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


def _decel(a):
    # a shell (r, u) decelerating at about a, so the event u = 0 lies at the start of a segment begun at u0 ~ 0+
    return lambda t, y: np.array([y[1], -a * (1 + 0.3 * np.sin(40 * t))])


def _u_event():
    ev = lambda t, y: y[1]  # noqa: E731
    ev.terminal, ev.direction = True, -1
    return ev


def test_an_event_lsoda_cannot_locate_is_redone_with_radau():
    """2026-11-23 (paper/II-survey CHANGELOG): 28 Paper II step-6 runs entered swept momentum at a turnaround with
    u = +3e-16..+2e-13 pc/Myr, in SWEEP, whose terminal event is u itself. scipy locates an event by brentq on the
    step's dense output; LSODA's misses the step start by its local error, so brentq raised 'f(a) and f(b) must have
    different signs' and the run ended. solve_segment redoes such a segment with Radau (exact at the step start)."""
    raised = 0
    for u0 in (2.9e-16, 9.5e-16, 3e-15, 2e-13):
        for a in (1.0, 30.0, 100.0, 1e3, 1e4):
            kw = dict(_kw(_decel(a), span=2e-3), t_span=(0.27, 0.272), y0=np.array([2.85, u0]), events=[_u_event()])
            try:
                scipy.integrate.solve_ivp(**kw)
            except ValueError as exc:
                assert "different signs" in str(exc)
                raised += 1
                p = _params()
                sol = events.solve_segment(p, kw)
                assert sol.success and sol.t_events[0].size == 1 and sol.t_events[0][0] - 0.27 < 1e-6
                assert p["solver_flags"].value == "event_retry" and p["EndSimulationDirectly"].value is False
    if not raised:
        pytest.skip("this scipy's LSODA locates every event here; the retry path is not exercised")


def test_an_error_that_is_not_the_budget_propagates():
    def fun(t, y):
        raise ValueError("shell_mass_net: refused")
    with pytest.raises(ValueError, match="shell_mass_net"):
        events.solve_segment(_params(), _kw(fun))
