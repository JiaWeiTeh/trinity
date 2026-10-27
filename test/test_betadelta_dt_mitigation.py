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


# =============================================================================
# cost cap and the windowed unsolvable hand-off (2026-10-01)
# =============================================================================


def test_rescue_ladder_runs_until_unreachable_then_every_tenth_segment():
    cap = RIP.BETADELTA_DT_SHRINK_MAX_STREAK
    assert all(RIP.rescue_allowed(k) for k in range(cap + 1))
    assert not any(RIP.rescue_allowed(k) for k in range(cap + 1, 2 * RIP.RESCUE_EVERY))
    assert RIP.rescue_allowed(2 * RIP.RESCUE_EVERY)


def test_unsolvable_handoff_counts_a_window_not_a_streak():
    """Two pilot_v5 runs never handed off because a rescue that succeeded now and
    then reset the old consecutive count. The window ignores the interruptions."""
    from collections import deque
    w = deque(maxlen=RIP.UNSOLVABLE_WINDOW)
    for i in range(RIP.UNSOLVABLE_WINDOW):
        w.append(i % 6 != 0)          # every 6th segment solves: 50 of 60 unsolvable
    assert RIP.unsolvable_handoff(w)
    w.append(False)
    w.append(False)
    assert not RIP.unsolvable_handoff(w)


def test_unconverged_handoff_after_the_mitigation_streak():
    """Step 7b-4: an unconverged solve hands off once BETADELTA_DT_SHRINK_MAX_STREAK unconverged segments came before it,
    under the production 'hybr' solver only."""
    M = RIP.BETADELTA_DT_SHRINK_MAX_STREAK
    assert not RIP.unconverged_handoff(M - 1, False, "hybr")       # still inside the mitigation
    assert RIP.unconverged_handoff(M, False, "hybr")               # the (M+1)-th unconverged segment in a row
    assert RIP.unconverged_handoff(5 * M, False, "hybr")
    assert not RIP.unconverged_handoff(M, True, "hybr")            # a converged solve never hands off here
    assert not RIP.unconverged_handoff(0, False, "hybr")           # an isolated miss does not
    assert not RIP.unconverged_handoff(5 * M, False, "legacy")     # legacy: non-convergence may be its bounded box


# The runner's call site (Step 7b-4, /xcheck 2026-10-27): a run whose beta-delta solves are reported unconverged from the third
# implicit segment on must hand off on the 11th such segment, and only there. c3c fixture, in its own process (trinity keeps
# module-level state). The forced results keep the solver's real (beta, delta); only the convergence flag is overridden.
_UNCONV_DRIVER = r"""
import atexit, dataclasses, json, runpy, sys
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as RIP
n, rec = [0], {"calls": []}
_solve = RIP.solve_betadelta_pure
def solve(b, d, params, *a, **kw):
    n[0] += 1
    res = _solve(b, d, params, *a, **kw)
    return dataclasses.replace(res, converged=False) if n[0] >= 3 else res
RIP.solve_betadelta_pure = solve
_uh = RIP.unconverged_handoff
def uh(streak, converged, solver):
    out = _uh(streak, converged, solver)
    rec["calls"].append([streak, bool(converged), bool(out)])
    return out
RIP.unconverged_handoff = uh
atexit.register(lambda: json.dump(rec, open("unconv.json", "w")))
sys.argv = ["run.py", "fixture.param"]
runpy.run_path(%r, run_name="__main__")
"""


def test_runner_hands_off_on_the_eleventh_unconverged_segment(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "fixture.param").write_text("mCloud 1e4\nsfe 0.1\nnCore 5e2\nstop_t 0.008\nmodel_name unconv\nlog_console False\n")
    (tmp_path / "driver.py").write_text(_UNCONV_DRIVER % str(repo / "run.py"))
    res = subprocess.run([sys.executable, "driver.py"], cwd=tmp_path, capture_output=True, text=True, timeout=1800,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(repo), os.environ.get("PYTHONPATH")]))})
    assert res.returncode == 0, res.stderr[-3000:]
    calls = json.loads((tmp_path / "unconv.json").read_text())["calls"]
    M = RIP.BETADELTA_DT_SHRINK_MAX_STREAK
    forced = [c for c in calls if not c[1]]
    assert len(forced) == M + 1                                       # 1b ended on the (M+1)-th forced segment
    assert [c[0] for c in forced] == list(range(M + 1))               # streak 0..M before each
    assert [c[2] for c in calls] == [False] * (len(calls) - 1) + [True]   # handed off once, on the last call
    rows = [json.loads(ln) for ln in (tmp_path / "outputs" / "unconv" / "dictionary.jsonl").read_text().splitlines() if ln.strip()]
    assert any(r["current_phase"] == "transition" for r in rows)       # and the run went on into 1c
    final = json.loads((tmp_path / "outputs" / "unconv" / "metadata.json").read_text())["final_state"]
    assert final["transition_channel"] == "no_physical_root_handoff"     # the rows do not carry the channel; metadata does
