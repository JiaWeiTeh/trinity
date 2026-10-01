"""Unit tests for phase event factories and event-result handling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trinity.phase_general import phase_events as events


def _y(R2=2.0, v2=3.0, Eb=4.0):
    return np.array([R2, v2, Eb], dtype=float)


def _cooling_balance(Lloss):
    return events.make_cooling_balance_event(threshold=0.05)(100.0, Lloss)


@pytest.mark.parametrize(
    ("event", "negative_y", "zero_y", "positive_y", "direction", "terminal", "ends_run"),
    [
        (events.make_min_radius_event(2.0), _y(R2=1.99), _y(R2=2.0), _y(R2=2.01), -1, True, True),
        (events.make_max_radius_event(2.0), _y(R2=1.99), _y(R2=2.0), _y(R2=2.01), 1, True, True),
        (
            events.make_velocity_runaway_event(5.0, direction="collapse"),
            _y(v2=-5.01),
            _y(v2=-5.0),
            _y(v2=-4.99),
            -1,
            True,
            True,
        ),
        (
            events.make_velocity_runaway_event(5.0, direction="expansion"),
            _y(v2=5.01),
            _y(v2=5.0),
            _y(v2=4.99),
            -1,
            True,
            True,
        ),
        (
            events.make_velocity_runaway_event(5.0, direction="both"),
            _y(v2=-5.01),
            _y(v2=-5.0),
            _y(v2=-4.99),
            -1,
            True,
            True,
        ),
        (
            events.make_cloud_boundary_event(2.0),
            _y(R2=1.99),
            _y(R2=2.0),
            _y(R2=2.01),
            1,
            True,
            False,
        ),
        (
            events.make_energy_floor_event(4.0),
            _y(Eb=3.99),
            _y(Eb=4.0),
            _y(Eb=4.01),
            -1,
            True,
            False,
        ),
        (
            events.make_velocity_sign_event(),
            _y(v2=-0.01),
            _y(v2=0.0),
            _y(v2=0.01),
            -1,
            False,
            False,
        ),
        (
            _cooling_balance(95.1),
            _y(),
            _y(),
            _y(),
            -1,
            True,
            False,
        ),
    ],
)
def test_event_factories_cross_threshold(
    event,
    negative_y,
    zero_y,
    positive_y,
    direction,
    terminal,
    ends_run,
):
    if event.name == "cooling_balance":
        event = _cooling_balance(95.0)
        negative_value = _cooling_balance(95.1)(0.0, negative_y)
        positive_value = _cooling_balance(94.9)(0.0, positive_y)
    else:
        negative_value = event(0.0, negative_y)
        positive_value = event(0.0, positive_y)

    assert negative_value < 0
    assert event(0.0, zero_y) == pytest.approx(0.0)
    assert positive_value > 0
    assert event.direction == direction
    assert event.terminal is terminal
    assert event.is_simulation_ending is ends_run


def _param(value=None):
    return SimpleNamespace(value=value)


def _collapse_params(coll_r=1.0, frac=0.25):
    return {"coll_r": _param(coll_r), "coll_r_frac": _param(frac), "R2_max": _param(0.0),
            "collapse_radius": _param(float("nan")), "collapse_rule": _param("")}


def test_check_and_apply_event_result_classify_run_vs_phase_end():
    run_end_event = events.make_min_radius_event(1.0)
    phase_end_event = events.make_cloud_boundary_event(5.0)

    sol = SimpleNamespace(
        t_events=[np.array([]), np.array([0.25])],
        y_events=[np.empty((0, 2)), np.array([[1.0, -2.0]])],
    )
    result = events.check_event_termination(sol, [phase_end_event, run_end_event])
    assert result.triggered is True
    assert result.index == 1
    assert result.name == "min_radius"
    assert result.is_simulation_ending is True

    params = {
        "t_now": _param(),
        "R2": _param(),
        "v2": _param(),
        "SimulationEndReason": _param(""),
        "SimulationEndCode": _param(),
        "EndSimulationDirectly": _param(False),
        "isCollapse": _param(False),
        **_collapse_params(),
    }
    events.update_collapse_radius(params, 40.0)
    events.apply_event_result(params, result, result.t, result.y)
    assert params["t_now"].value == pytest.approx(0.25)
    assert params["R2"].value == pytest.approx(1.0)
    assert params["v2"].value == pytest.approx(-2.0)
    assert params["SimulationEndReason"].value == "Small radius reached (event): R2 < 1 pc (coll_r)"
    assert params["SimulationEndCode"].value == run_end_event.end_code.code
    assert params["EndSimulationDirectly"].value is True
    assert params["isCollapse"].value is True

    phase_sol = SimpleNamespace(
        t_events=[np.array([0.5])],
        y_events=[np.array([[5.0, 1.0]])],
    )
    phase_result = events.check_event_termination(phase_sol, [phase_end_event])
    phase_params = {
        "t_now": _param(),
        "R2": _param(),
        "v2": _param(),
        "SimulationEndReason": _param(""),
        "EndSimulationDirectly": _param(False),
    }
    events.apply_event_result(phase_params, phase_result, phase_result.t, phase_result.y)
    assert phase_result.is_simulation_ending is False
    assert phase_params["t_now"].value == pytest.approx(0.5)
    assert phase_params["R2"].value == pytest.approx(5.0)
    assert phase_params["EndSimulationDirectly"].value is False
    assert phase_params["SimulationEndReason"].value == ""


def test_monitoring_event_neither_ends_nor_preempts_the_phase():
    """Phase 1b's list starts with velocity_sign (terminal=False). Until 2026-09-30 the
    checker returned it, so 1b ended at the shell's first turnaround and a min_radius
    later in the same segment was replaced by a rewind to the turnaround."""
    sign = events.make_velocity_sign_event()
    min_r = events.make_min_radius_event(1.5)

    both = SimpleNamespace(
        t_events=[np.array([0.1]), np.array([0.3])],
        y_events=[np.array([[4.0, 0.0]]), np.array([[1.5, -3.0]])],
    )
    result = events.check_event_termination(both, [sign, min_r])
    assert result.name == "min_radius"
    assert result.t == pytest.approx(0.3)
    assert result.is_simulation_ending is True

    only_sign = SimpleNamespace(
        t_events=[np.array([0.1]), np.array([])],
        y_events=[np.array([[4.0, 0.0]]), np.empty((0, 2))],
    )
    assert events.check_event_termination(only_sign, [sign, min_r]).triggered is False


def test_collapse_radius_is_min_of_cap_and_fraction_of_largest_radius():
    """collapse_radius = min(coll_r, coll_r_frac * R2_max), with the term that set it
    recorded, and R2_max a running maximum (a shrinking shell keeps its old max)."""
    p = _collapse_params()
    assert events.update_collapse_radius(p, 0.8) == pytest.approx(0.2)
    assert p["collapse_rule"].value == "frac_R2max"
    assert events.collapse_reason(p) == (
        "Small radius reached: R2 < 0.2 pc (0.25 x R2_max 0.8 pc; coll_r 1 pc not used)")
    assert events.update_collapse_radius(p, 10.0) == pytest.approx(1.0)
    assert p["collapse_rule"].value == "coll_r"
    assert events.update_collapse_radius(p, 3.0) == pytest.approx(1.0)
    assert p["R2_max"].value == pytest.approx(10.0)
    # exactly at the switch the cap wins
    q = _collapse_params()
    events.update_collapse_radius(q, 4.0)
    assert q["collapse_rule"].value == "coll_r"
    # a shell that never passed 0.04 pc: the solver floor binds, and is what is recorded
    f = _collapse_params()
    assert events.update_collapse_radius(f, 0.03) == pytest.approx(events.MIN_RADIUS_SAFETY)
    assert f["collapse_rule"].value == "floor"
    assert events.collapse_reason(f) == "Small radius reached: R2 < 0.01 pc (the 0.01 pc floor; R2_max 0.03 pc)"


def test_min_radius_event_follows_the_live_collapse_radius():
    """The phase builders' min_radius event reads collapse_radius when called, so it
    moves with R2_max inside one phase; MIN_RADIUS_SAFETY floors it."""
    p = {"rCloud": _param(20.0), "stop_r": _param(500.0), **_collapse_params()}
    evs, _ = events.build_implicit_phase_events(p)
    min_r = next(e for e in evs if e.name == "min_radius")
    events.update_collapse_radius(p, 0.8)
    assert min_r(0.0, _y(R2=0.3)) == pytest.approx(0.1)
    events.update_collapse_radius(p, 40.0)
    assert min_r(0.0, _y(R2=0.3)) == pytest.approx(-0.7)
    p["collapse_radius"].value = 0.001
    assert min_r(0.0, _y(R2=0.3)) == pytest.approx(0.3 - events.MIN_RADIUS_SAFETY)


@pytest.mark.parametrize("build", ["energy", "implicit", "transition"])
def test_turnaround_ends_every_energy_carrying_phase(build):
    """Ruled 2026-10-01: v2 < 0 goes straight to momentum, so 1a/1b/1c carry a
    TERMINAL velocity_sign event (phase-ending, not run-ending); phase 2 has none."""
    p = {"rCloud": _param(20.0), "stop_r": _param(500.0), **_collapse_params()}
    evs = {"energy": lambda: events.build_energy_phase_events(p),
           "implicit": lambda: events.build_implicit_phase_events(p)[0],
           "transition": lambda: events.build_transition_phase_events(p)}[build]()
    sign = [e for e in evs if e.name == "velocity_sign"]
    assert len(sign) == 1
    assert sign[0].terminal is True and sign[0].is_simulation_ending is False
    assert sign[0].reason_code == "velocity_sign_change"
    assert not any(e.name == "velocity_sign" for e in events.build_momentum_phase_events(p))


def test_solver_flags_accumulate_once():
    p = {"solver_flags": _param("")}
    events.add_solver_flag(p, "cost_cap")
    events.add_solver_flag(p, "no_physical_root_handoff")
    events.add_solver_flag(p, "cost_cap")
    assert p["solver_flags"].value == "cost_cap,no_physical_root_handoff"
