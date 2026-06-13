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


def test_mitigation_disengages_beyond_max_streak():
    # A long streak means the root is unreachable; dt policy reverts to
    # standard adaptive stepping (growth allowed, no forced shrink).
    over = RIP.BETADELTA_DT_SHRINK_MAX_STREAK + 1
    grown = RIP.next_dt_segment(DT, max_dex_change=0.01, unconverged_streak=over)
    assert grown == pytest.approx(DT * RIP.ADAPTIVE_FACTOR)
    shrunk = RIP.next_dt_segment(DT, max_dex_change=0.1, unconverged_streak=over)
    assert shrunk == pytest.approx(DT / RIP.ADAPTIVE_FACTOR)  # single, not compound


def test_mitigation_still_active_at_exactly_max_streak():
    at_cap = RIP.BETADELTA_DT_SHRINK_MAX_STREAK
    out = RIP.next_dt_segment(DT, max_dex_change=0.01, unconverged_streak=at_cap)
    assert out == pytest.approx(DT / RIP.ADAPTIVE_FACTOR)


def test_warns_once_when_mitigation_disengages(caplog):
    cap = RIP.BETADELTA_DT_SHRINK_MAX_STREAK
    with caplog.at_level(logging.WARNING, logger=RIP.logger.name):
        streak = RIP.update_unconverged_streak(cap, False, 1.0, 0.5)
        assert streak == cap + 1
        assert "disengaged" in caplog.text
        n_warn = len(caplog.records)
        streak = RIP.update_unconverged_streak(streak, False, 1.0, 0.5)
        assert len(caplog.records) == n_warn  # no re-warn past the transition


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


# =============================================================================
# betadelta_phase_summary (end-of-phase solver summary)
# =============================================================================


def test_summary_clean_when_all_converged_no_no_root():
    clean, msg = RIP.betadelta_phase_summary(
        solve_count=40, converged_count=40, no_root_count=0)
    assert clean is True
    assert "40/40" in msg and "100%" in msg and "0 with no physical root" in msg


def test_summary_dirty_when_some_unconverged():
    clean, msg = RIP.betadelta_phase_summary(
        solve_count=40, converged_count=10, no_root_count=0)
    assert clean is False
    assert "10/40" in msg and "25%" in msg


def test_summary_dirty_when_any_no_root():
    clean, msg = RIP.betadelta_phase_summary(
        solve_count=40, converged_count=40, no_root_count=3)
    assert clean is False
    assert "3 with no physical root" in msg


def test_summary_handles_zero_solves():
    clean, msg = RIP.betadelta_phase_summary(
        solve_count=0, converged_count=0, no_root_count=0)
    assert clean is True  # vacuously: nothing failed
    assert "0/0" in msg and "0%" in msg
