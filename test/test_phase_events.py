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
