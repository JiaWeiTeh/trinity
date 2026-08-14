"""The phase-1a in-band energy-collapse event
(`docs/dev/phase1a-stiffness/PLAN.md`, candidate C2).

Phase 1a stops cleanly on a collapsed energy-driven bubble, but that check runs
*between* segments and tests ``Eb <= 0``. Measured, both parts miss the real
failure: the collapse happens inside one segment and ``Eb`` never reaches zero —
it pins at ~1.6e-6 au on a stiff slow manifold where the explicit solver needs
~1e9 steps to cross the segment (~7 days). These tests pin the event that puts
the same decision in-band, and — more importantly — pin the *margins* that
justify its threshold, so a future edit to ``ENERGY_COLLAPSE_FRAC`` has to
confront the measurements that bound it.

Measured bounds (docs/dev/phase1a-stiffness/data/seg_stepcount.csv, 437 healthy
segments over five configs spanning four decades of mass and density):

* healthy segments never lose energy at all — ``Eb`` grows every segment, worst
  ratio 1.027 — so the event must be unreachable for any ratio >= 1;
* a segment losing 59% of ``Eb`` still integrates (the ablated control clears
  segments at ratios 0.67, 0.58, 0.41), so the event must NOT fire there;
* at 5.5e-8 of segment start the integrator is already dead, so the event must
  fire well above that.
"""
import pytest

from trinity._output.simulation_end import SimulationEndCode
from trinity.phase_general.phase_events import (
    ENERGY_COLLAPSE_FRAC,
    make_energy_collapse_event,
)

# Segment-start Eb values measured across the screen configs [au]: the compact
# probe through a heavy cloud. The threshold is relative, so it must behave
# identically across this whole span.
EB_STARTS = (90.0, 180.3, 2.35e5, 8.86e7)

# Healthy per-segment ratios (min/median/max measured) plus the deepest drop
# that still integrated, from the ablated control.
HEALTHY_RATIOS = (1.027, 1.098, 1.115)
SURVIVABLE_DROPS = (0.673, 0.584, 0.412)


def _value(Eb_start, ratio):
    """Event function value at Eb = ratio * Eb_start."""
    event = make_energy_collapse_event(Eb_start)
    return event(0.0, [1.0, 100.0, ratio * Eb_start])


@pytest.mark.parametrize("Eb_start", EB_STARTS)
@pytest.mark.parametrize("ratio", HEALTHY_RATIOS + SURVIVABLE_DROPS)
def test_event_is_inert_for_every_segment_that_still_integrates(Eb_start, ratio):
    """No healthy segment, and no measured survivable drop, may trip the event —
    this is what makes the change inert on production configs."""
    assert _value(Eb_start, ratio) > 0


@pytest.mark.parametrize("Eb_start", EB_STARTS)
@pytest.mark.parametrize("ratio", [1e-4, 5.5e-8])
def test_event_fires_once_the_bubble_has_collapsed(Eb_start, ratio):
    """At and below the regime where the solver stalls, the event must fire."""
    assert _value(Eb_start, ratio) < 0


@pytest.mark.parametrize("Eb_start", EB_STARTS)
def test_crossing_is_exactly_the_relative_floor(Eb_start):
    assert _value(Eb_start, ENERGY_COLLAPSE_FRAC) == pytest.approx(0.0, abs=1e-12 * Eb_start)


def test_threshold_keeps_its_measured_margins():
    """The value is bounded on both sides by measurement; keep it that way.

    Fails if someone raises the fraction into the survivable band or drops it
    into the dead zone — the two failure modes the derivation rules out.
    """
    assert ENERGY_COLLAPSE_FRAC < 0.412 / 100, "must sit well below the deepest survivable drop"
    assert ENERGY_COLLAPSE_FRAC > 5.5e-8 * 100, "must fire well before the solver stalls"


def test_event_is_terminal_and_ends_the_run_as_an_energy_collapse():
    """It must route to the same fate phase 1a already uses for a dead bubble,
    so the change adds no new stopping outcome."""
    event = make_energy_collapse_event(180.3)
    assert event.terminal is True
    assert event.direction == -1                      # only the downward crossing
    assert event.is_simulation_ending is True
    assert event.end_code is SimulationEndCode.ENERGY_COLLAPSED
    assert event.name == "energy_collapse"
