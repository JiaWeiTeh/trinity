"""Tests for the reader's units, phase ordering, and final-state summary.

These cover three additions to ``TrinityOutput``:

- ``units(key)`` / ``quantity(key)``, so a caller can find out what internal
  units a stored quantity is in without reading the source.
- ``phases`` returning phases in the order the simulation runs them. It used to
  come back from a ``set``, and ``info()`` sorted it alphabetically, which
  printed ``transition`` after ``momentum``.
- ``info()`` reporting how a run ended and the state of the bubble when it did.

They run against the committed example runs, so they are skipped in a checkout
that does not have them.
"""
from pathlib import Path

import pytest

from trinity._output.trinity_reader import TrinityOutput

EXAMPLE = Path(__file__).parent.parent / 'examples' / 'runs' / 'homogeneous'

pytestmark = pytest.mark.skipif(
    not (EXAMPLE / 'dictionary.jsonl').exists(),
    reason='example runs not present',
)


@pytest.fixture(scope='module')
def output():
    return TrinityOutput.open(EXAMPLE / 'dictionary.jsonl')


# --- units ------------------------------------------------------------------

@pytest.mark.parametrize('key, expected', [
    ('R2', 'pc'),
    ('v2', 'pc/Myr'),
    ('t_now', 'Myr'),
    ('mCloud', 'Msun'),
])
def test_units_reports_the_internal_unit(output, key, expected):
    assert output.units(key) == expected


@pytest.mark.parametrize('key', ['isCollapse', 'current_phase', 'cool_alpha'])
def test_units_is_none_for_unitless_quantities(output, key):
    """Flags, strings and dimensionless numbers have no unit to report."""
    assert output.units(key) is None


def test_units_is_none_for_an_unknown_key(output):
    assert output.units('not_a_real_key') is None


# --- astropy quantities -----------------------------------------------------

def test_quantity_matches_get_and_carries_the_unit(output):
    """quantity() must be get() with units attached — same numbers, not rescaled."""
    import astropy.units as u

    plain = output.get('v2')
    with_units = output.quantity('v2')

    assert with_units.unit == u.Unit('pc/Myr')
    assert (with_units.value == plain).all()


def test_quantity_converts_through_astropy(output):
    """The point of the method: let astropy do the conversion."""
    import astropy.units as u

    v_kms = output.quantity('v2').to('km/s')
    assert v_kms.unit == u.Unit('km/s')
    # 1 pc/Myr is about 0.978 km/s, so the numbers must shrink slightly.
    assert (v_kms.value < output.get('v2')).all()


def test_quantity_refuses_a_unitless_key(output):
    """Better a clear error than a silently dimensionless array."""
    with pytest.raises(ValueError, match='no units recorded'):
        output.quantity('current_phase')


# --- phase ordering ---------------------------------------------------------

def test_phases_are_in_simulation_order_not_alphabetical(output):
    """Alphabetical order would put 'transition' after 'momentum'."""
    phases = output.phases
    assert phases == [p for p in TrinityOutput.PHASE_ORDER if p in phases]

    if 'transition' in phases and 'momentum' in phases:
        assert phases.index('transition') < phases.index('momentum')


def test_unknown_phase_labels_are_kept_not_dropped(output):
    """A phase this version has never heard of must still be reported."""
    out = TrinityOutput.open(EXAMPLE / 'dictionary.jsonl')
    out._snapshots[0]['current_phase'] = 'some_future_phase'
    assert 'some_future_phase' in out.phases


# --- final-state summary ----------------------------------------------------

def test_info_reports_the_ending_and_final_state(output, capsys):
    output.info()
    printed = capsys.readouterr().out

    assert 'Final state' in printed
    for label in ('age', 'shell radius', 'expansion velocity',
                  'shell mass swept', 'fate'):
        assert label in printed, f'info() no longer reports {label!r}'


def test_info_survives_a_run_with_no_metadata(tmp_path, capsys):
    """A bare .jsonl has no final state to report; info() must still work."""
    import shutil

    shutil.copy(EXAMPLE / 'dictionary.jsonl', tmp_path / 'dictionary.jsonl')
    TrinityOutput.open(tmp_path / 'dictionary.jsonl').info()

    printed = capsys.readouterr().out
    assert 'Snapshots:' in printed
    assert 'Final state' not in printed
