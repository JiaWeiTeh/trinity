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
    }
    events.apply_event_result(params, result, result.t, result.y)
    assert params["t_now"].value == pytest.approx(0.25)
    assert params["R2"].value == pytest.approx(1.0)
    assert params["v2"].value == pytest.approx(-2.0)
    assert params["SimulationEndReason"].value == "Small radius reached (event)"
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


# =============================================================================
# Regression: a non-terminal monitoring event must not mask a terminal one,
# and `isCollapse` must mean "the shell was contracting", not "the reason_code
# happened to contain the substring 'radius'".
#
# Both defects live in phase_events.py and were found by the code audit
# (NUM-02 / S11-R-01 / DD-001 / ST-002, and S11-R-02).
# =============================================================================


def _sol(t_events, y_events):
    return SimpleNamespace(
        t_events=[np.asarray(t, dtype=float) for t in t_events],
        y_events=[np.asarray(y, dtype=float).reshape(-1, 2) for y in y_events],
    )


def test_nonterminal_event_does_not_mask_a_terminal_one():
    """`velocity_sign` is non-terminal, so solve_ivp records it and keeps going.

    It sits at index 0 of the implicit-phase event list, so a selection that
    returns the first *by list index* hands back a monitoring event and loses the
    run-ending one that fired later in the same segment -- along with its
    SimulationEndCode and EndSimulationDirectly.
    """
    monitoring = events.make_velocity_sign_event()      # terminal=False, index 0
    terminal = events.make_min_radius_event(1.0)        # terminal=True,  index 1

    # Both fire in one segment: the sign change first, the collapse after.
    sol = _sol(
        t_events=[[0.10], [0.40]],
        y_events=[[[5.0, 0.0]], [[1.0, -2.0]]],
    )

    result = events.check_event_termination(sol, [monitoring, terminal])

    assert result.is_simulation_ending is True, (
        "a non-terminal monitoring event masked the run-ending event"
    )
    assert result.name == "min_radius"
    assert result.end_code is not None


def test_lone_nonterminal_event_is_still_reported():
    """With nothing terminal firing, the monitoring event is still the answer."""
    monitoring = events.make_velocity_sign_event()
    terminal = events.make_min_radius_event(1.0)

    sol = _sol(t_events=[[0.10], []], y_events=[[[5.0, 0.0]], np.empty((0, 2))])

    result = events.check_event_termination(sol, [monitoring, terminal])

    assert result.triggered is True
    assert result.name == "velocity_sign"
    assert result.is_simulation_ending is False


def test_earliest_terminal_event_wins_regardless_of_list_order():
    """Two terminal events in one segment: the one that physically happened
    first ends the run, not whichever sits earlier in the list."""
    late = events.make_max_radius_event(50.0)      # index 0, fires at t=0.9
    early = events.make_min_radius_event(1.0)      # index 1, fires at t=0.2

    sol = _sol(
        t_events=[[0.90], [0.20]],
        y_events=[[[50.0, 4.0]], [[1.0, -2.0]]],
    )

    result = events.check_event_termination(sol, [late, early])

    assert result.name == "min_radius"
    assert result.t == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("factory", "y", "expected", "why"),
    [
        (lambda: events.make_min_radius_event(1.0), [1.0, -2.0], True,
         "contracting through the floor - a real collapse"),
        (lambda: events.make_max_radius_event(50.0), [50.0, +12.0], False,
         "EXPANDING out through stop_r - a clean LARGE_RADIUS success"),
        (lambda: events.make_velocity_runaway_event(500.0, direction="collapse"),
         [3.0, -500.0], True,
         "runaway INFALL - the definition of collapse"),
        (lambda: events.make_velocity_runaway_event(500.0, direction="expansion"),
         [30.0, +500.0], False,
         "runaway EXPANSION"),
    ],
)
def test_iscollapse_tracks_the_sign_of_v2_not_the_reason_code_spelling(
    factory, y, expected, why
):
    """`isCollapse` must mean v2 < 0 at exit.

    show_run.py documents exactly this: "isCollapse alone only means the shell
    was *contracting* (v2 < 0 and R2 falling) at exit", and the phase 1b/1c/2
    segment loops implement it as `if v2 < 0 and R2 < R2_prev`. The substring
    test on reason_code is the only place that departs from it, and it departs
    in both directions: 'large_radius_event' contains "radius" (false positive
    on an expanding shell) while 'velocity_runaway_event' contains neither
    "radius" nor "collapse" (false negative on runaway infall).
    """
    event = factory()
    sol = _sol(t_events=[[0.25]], y_events=[[y]])
    result = events.check_event_termination(sol, [event])

    params = {
        "t_now": _param(),
        "R2": _param(),
        "v2": _param(),
        "SimulationEndReason": _param(""),
        "SimulationEndCode": _param(),
        "EndSimulationDirectly": _param(False),
        "isCollapse": _param(False),
    }
    events.apply_event_result(params, result, result.t, result.y)

    assert params["isCollapse"].value is expected, f"v2={y[1]:+g}: {why}"
