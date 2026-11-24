"""R0 (docs/dev/turnaround-stall/PLAN_R0.md, ruled 2026-11-24): turnaround_rule sets what a turnaround (v2 falling
through 0) does in phases 1a, 1b and 1c. 'continue', the default: velocity_sign only records the crossing and the phase
goes on with its energy; under phii_scheme swept a SWEEP segment then ends at its own mode switch (u = 0) and the next
one coasts. 'momentum', the rule of 2026-10-01: velocity_sign is terminal (test_phase_events.py pins that side)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from scipy.integrate import solve_ivp

from trinity._input.errors import ParameterFileError
from trinity._input.registry import SPECS, _validate_turnaround_rule
from trinity.phase2_momentum.swept import mode_event
from trinity.phase_general import phase_events as events


def _params(rule):
    """A phase builder's params: rCloud 20 pc, stop_r 500 pc, collapse radius 1 pc (as test_phase_events.py)."""
    p = {k: SimpleNamespace(value=v) for k, v in (("rCloud", 20.0), ("stop_r", 500.0), ("coll_r", 1.0),
                                                  ("coll_r_frac", 0.25), ("R2_max", 0.0),
                                                  ("collapse_radius", 1.0), ("collapse_rule", "coll_r"))}
    p["turnaround_rule"] = SimpleNamespace(value=rule)
    return p


def test_default_is_continue_and_only_the_two_names_pass():
    spec = next(s for s in SPECS if s.name == "turnaround_rule")
    assert spec.default == "continue" and spec.validator is _validate_turnaround_rule
    for ok in ("continue", "momentum"):
        _validate_turnaround_rule(ok, {})
    for bad in ("Momentum", "momentun", "", None, 1.0):
        with pytest.raises(ParameterFileError):
            _validate_turnaround_rule(bad, {})


@pytest.mark.parametrize("build", ["energy", "implicit", "transition"])
def test_continue_keeps_velocity_sign_as_a_monitor(build):
    """Each of 1a/1b/1c carries one velocity_sign event: non-terminal under continue, terminal under momentum."""
    for rule, terminal in (("continue", False), ("momentum", True)):
        p = _params(rule)
        evs = {"energy": lambda: events.build_energy_phase_events(p),
               "implicit": lambda: events.build_implicit_phase_events(p)[0],
               "transition": lambda: events.build_transition_phase_events(p)}[build]()
        sign = [e for e in evs if e.name == "velocity_sign"]
        assert len(sign) == 1 and sign[0].terminal is terminal
        assert events.turnaround_ends_phase(p) is terminal


def test_continue_sweep_segment_ends_at_u0_without_ending_the_phase():
    """The contract the swept 1b/1c runners rely on under continue: velocity_sign non-terminal, the SWEEP mode switch
    appended past the phase's events. A solve whose u falls through 0 stops there (the segment ends, t_events[-1] holds
    the crossing) and check_event_termination reports no phase end. A shell at 10 pc moving out at 2 pc/Myr and
    decelerating at 1 pc/Myr^2 turns at t = 2 Myr (state R2, v2, Eb, T0 as in 1b)."""
    p = _params("continue")
    ode_events = events.build_implicit_phase_events(p)[0]
    sol = solve_ivp(lambda t, y: [y[1], -1.0, 0.0, 0.0], (0.0, 5.0), [10.0, 2.0, 1.0e6, 1.0e6],
                    events=ode_events + [mode_event(True, 1.0, None)], rtol=1e-10, atol=1e-12)
    assert sol.status == 1 and sol.t[-1] == pytest.approx(2.0, rel=1e-6)
    assert sol.t_events[-1].size == 1
    assert events.check_event_termination(sol, ode_events).triggered is False
