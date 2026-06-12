"""
Tests for the beta-delta non-convergence mitigation in the implicit runner
(``run_energy_implicit_phase``): the unconverged-streak counter, the WARNING
at the streak threshold, the dt_segment shrink-and-suppress-growth policy,
and the snapshot registration of the persisted convergence keys.
"""

from __future__ import annotations

import logging

import pytest

import trinity.phase1b_energy_implicit.run_energy_implicit_phase as RIP
from trinity._input.dictionary import COOLING_PHASE_KEYS
from trinity._input.registry import REGISTRY

# =============================================================================
# update_unconverged_streak
# =============================================================================


def test_streak_resets_on_converged():
    assert RIP.update_unconverged_streak(0, True, 1.0, 1e-6) == 0
    assert RIP.update_unconverged_streak(5, True, 1.0, 1e-6) == 0


def test_streak_increments_and_warns_once_at_threshold(caplog):
    streak = 0
    with caplog.at_level(logging.WARNING, logger=RIP.logger.name):
        for expected in (1, 2):
            streak = RIP.update_unconverged_streak(streak, False, 1.0, 0.5)
            assert streak == expected
        assert not caplog.records

        streak = RIP.update_unconverged_streak(streak, False, 1.0, 0.5)
        assert streak == RIP.BETADELTA_UNCONVERGED_WARN_STREAK
        assert len(caplog.records) == 1
        assert "unconverged for 3 consecutive segments" in caplog.text

        # No re-warn past the threshold
        streak = RIP.update_unconverged_streak(streak, False, 1.0, 0.5)
        assert streak == 4
        assert len(caplog.records) == 1


# =============================================================================
# next_dt_segment
# =============================================================================

DT = 1e-3


def test_dt_shrinks_on_large_change():
    out = RIP.next_dt_segment(DT, max_dex_change=0.1, unconverged_streak=0)
    assert out == pytest.approx(DT / RIP.ADAPTIVE_FACTOR)


def test_dt_grows_on_small_change_when_converged():
    out = RIP.next_dt_segment(DT, max_dex_change=0.01, unconverged_streak=0)
    assert out == pytest.approx(DT * RIP.ADAPTIVE_FACTOR)


def test_growth_suppressed_and_shrunk_while_unconverged():
    # Small change would normally grow dt; an active streak must instead shrink
    out = RIP.next_dt_segment(DT, max_dex_change=0.01, unconverged_streak=2)
    assert out == pytest.approx(DT / RIP.ADAPTIVE_FACTOR)


def test_large_change_and_unconverged_shrink_compound():
    out = RIP.next_dt_segment(DT, max_dex_change=0.1, unconverged_streak=1)
    assert out == pytest.approx(DT / RIP.ADAPTIVE_FACTOR**2)


def test_dt_floor_and_cap_respected():
    floor = RIP.next_dt_segment(RIP.DT_SEGMENT_MIN, 0.5, unconverged_streak=5)
    assert floor == RIP.DT_SEGMENT_MIN
    cap = RIP.next_dt_segment(RIP.DT_SEGMENT_MAX, 0.0, unconverged_streak=0)
    assert cap == RIP.DT_SEGMENT_MAX


# =============================================================================
# Persistence registration (mirrors residual_deltaT end-to-end)
# =============================================================================


@pytest.mark.parametrize("key", ["betadelta_converged", "betadelta_total_residual"])
def test_convergence_keys_registered_like_residual_deltaT(key):
    spec = REGISTRY[key]
    reference = REGISTRY["residual_deltaT"]
    assert spec.category == reference.category == "runtime_residuals"
    assert not spec.exclude_from_snapshot  # saved into snapshots
    assert key in COOLING_PHASE_KEYS  # reset with the other solver residuals
