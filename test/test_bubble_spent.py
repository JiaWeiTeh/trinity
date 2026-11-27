"""The spent-bubble test, get_bubbleParams.bubble_spent (docs/dev/main-code-change/PLAN_SPENT_BUBBLE.md).

A strong wind: 3e8 code units (1.8e38 erg/s) at 2e3 pc/Myr, R2 = 20 pc, gamma 5/3.
"""
import pytest

from trinity.bubble_structure import get_bubbleParams as gb

L, V, R2, G = 3e8, 2e3, 20.0, 5 / 3


@pytest.mark.parametrize("Eb", [1e9, 1e7, 1e5, 1e3])
def test_pb_is_the_ram_pressure_at_r1(Eb):
    """Why the old ram fraction > 0.9 exit never fired: solve_R1 makes Pb = P_ram(R1) >= P_ram(R2) at gamma 5/3, so
    P_ram(R2) / (Pb + P_ram(R2)) <= 1/2."""
    R1 = gb.solve_R1(R2, Eb, L, V)
    Pb = gb.bubble_E2P(Eb, R2, R1, G)
    assert Pb == pytest.approx(gb.pRam(R1, L, V), rel=1e-9)
    assert gb.pRam(R2, L, V) / (Pb + gb.pRam(R2, L, V)) <= 0.5


def test_spent_once_pb_is_within_the_threshold_of_p_ram():
    assert not gb.bubble_spent(1e8, R2, L, V, G, 1.01)   # Pb = 33.5 P_ram(R2)
    assert gb.bubble_spent(1e3, R2, L, V, G, 1.01)       # Pb = 1.0002 P_ram(R2)
    assert not gb.bubble_spent(1e3, R2, L, V, G, 1.0001)


def test_no_hot_gas_is_spent_with_or_without_wind():
    assert gb.bubble_spent(0.0, R2, L, V, G, 1.01)
    assert gb.bubble_spent(-5.0, R2, 0.0, 0.0, G, 1.01)
    assert not gb.bubble_spent(5.0, R2, 0.0, 0.0, G, 1.01)   # no wind but hot gas left: still a bubble


def test_a_given_pb_is_used_as_is():
    assert gb.bubble_spent(1e8, R2, L, V, G, 1.01, P_b=1.005 * gb.pRam(R2, L, V))


@pytest.mark.parametrize("bad", [1.0, 0.5, 1.1, 2.0, float("inf"), float("nan"), True, "1.01"])
def test_a_threshold_outside_one_to_one_point_one_is_refused(bad):
    from trinity._input.errors import ParameterFileError
    from trinity._input.registry import _validate_phaseSwitch_PbPram
    with pytest.raises(ParameterFileError):
        _validate_phaseSwitch_PbPram(bad, {})
    _validate_phaseSwitch_PbPram(1.01, {})
