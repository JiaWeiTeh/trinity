"""Phase-1a spent-bubble handoff (2026-09-30,
docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md).

A phase-1a energy collapse used to end the run as ENERGY_COLLAPSED. It now goes
to momentum via 1c, skipping 1b, as 1b already does for Eb <= 0. These pin the
state the handoff leaves behind and the event name run_energy_phase keys on.
The full-run path is covered by test_phase1a_collapse_terminates.py.
"""
from types import SimpleNamespace

from trinity._output.simulation_end import SimulationEndCode
from trinity.phase1_energy.run_energy_phase import _handoff_spent_bubble
from trinity.phase1b_energy_implicit.run_energy_implicit_phase import ENERGY_HANDOFF_FLOOR
from trinity.phase_general.phase_events import make_energy_collapse_event


def _params():
    # the state apply_event_result leaves after the energy_collapse event
    return {k: SimpleNamespace(value=v) for k, v in {
        't_now': 0.0, 'R2': 0.0, 'v2': 0.0, 'Eb': 1e-3,
        'energy_handoff_1a': False,
        'Eb_handoff': float('nan'),
        'transition_channel': '',
        'EndSimulationDirectly': True,
        'SimulationEndReason': 'Energy-driven bubble collapsed (Eb fell to a fraction of segment start)',
        'SimulationEndCode': SimulationEndCode.ENERGY_COLLAPSED.code,
        'isCollapse': True,
    }.items()}


def test_handoff_clears_the_collapse_ending_and_marks_the_run():
    p = _params()
    Eb = _handoff_spent_bubble(p, 2.57e-3, 0.8647, 245.1, "test")
    assert Eb == ENERGY_HANDOFF_FLOOR == p['Eb'].value
    assert (p['t_now'].value, p['R2'].value, p['v2'].value) == (2.57e-3, 0.8647, 245.1)
    assert p['energy_handoff_1a'].value is True
    assert p['EndSimulationDirectly'].value is False
    assert p['SimulationEndReason'].value == ''
    assert p['SimulationEndCode'].value is None
    assert p['isCollapse'].value is False   # the event sets it; the shell is still expanding
    assert p['Eb_handoff'].value == 1e-3     # what was dropped, kept for the record
    assert p['transition_channel'].value == 'energy_to_momentum'


def test_turnaround_uses_the_same_handoff_under_its_own_channel():
    """Ruled 2026-10-01: v2 < 0 in phase 1a goes straight to momentum too."""
    p = _params()
    p['Eb'].value = 3.4e10
    _handoff_spent_bubble(p, 1.7e-3, 1.09, 0.0, "shell turned around",
                          channel="velocity_sign_change")
    assert p['Eb'].value == ENERGY_HANDOFF_FLOOR and p['Eb_handoff'].value == 3.4e10
    assert p['transition_channel'].value == 'velocity_sign_change'


def test_run_energy_keys_on_the_collapse_event_name():
    """run_energy_phase routes on event_result.name == 'energy_collapse'."""
    assert make_energy_collapse_event(1.0).name == 'energy_collapse'
