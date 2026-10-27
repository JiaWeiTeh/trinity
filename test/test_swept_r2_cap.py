"""swept_r2_cap (PLAN.md §9, 2026-10-27, gate G-R2cap): the inward speed cap on the swept bubble radius R2."""
import pytest

from trinity._input.errors import ParameterFileError
from trinity._input.registry import _validate_swept_r2_cap
from trinity.phase2_momentum.swept import capped_r2


def test_off_or_unseeded_returns_the_root():
    assert capped_r2(3.0, 10.0, 0.1, 10.4, 0.0) == 3.0          # k = 0: the cap is off
    assert capped_r2(3.0, None, 0.1, 10.4, 2.0) == 3.0          # no previous segment yet


def test_inward_motion_is_capped_outward_is_free():
    c_i, dt, k = 10.4, 0.05, 2.0                                 # pc/Myr (~10.2 km/s), Myr
    assert capped_r2(3.0, 10.0, dt, c_i, k) == pytest.approx(10.0 - k * c_i * dt)   # too fast inward: held
    assert capped_r2(9.5, 10.0, dt, c_i, k) == 9.5               # slower than the cap: the root
    assert capped_r2(12.0, 10.0, dt, c_i, k) == 12.0             # outward: never capped


@pytest.mark.parametrize("bad", [-1.0, "fast", True])
def test_validator_rejects_bad_values(bad):
    with pytest.raises(ParameterFileError):
        _validate_swept_r2_cap(bad, None)


@pytest.mark.parametrize("good", [0, 0.0, 2, 4.5])
def test_validator_accepts_non_negative_numbers(good):
    _validate_swept_r2_cap(good, None)
