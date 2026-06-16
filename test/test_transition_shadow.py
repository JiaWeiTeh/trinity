"""Tests for the transition-trigger criteria (F0/F4) and shadow diagnostics.

Covers the shared pure predicates and decision function used by **both** the live
implicit-phase terminator and the log-only ``ShadowTransitionLog`` (so the shadow
F0 epoch equals the live break epoch by construction), the F0/F4 first-fire
serialization, and that the ``transition_trigger`` param is registered as a
run-constant, snapshot-excluded knob defaulting to ``instantaneous``.
See ``docs/dev/transition/pshadow-design.md`` (P-shadow / P-promote).
"""
from __future__ import annotations

import json

import pytest

from trinity.phase_general.transition_shadow import (
    SHADOW_FILENAME,
    VALID_TRANSITION_TRIGGERS,
    ShadowTransitionLog,
    blowout_fires,
    cooling_balance_fires,
    implicit_termination_reason,
    validate_transition_trigger,
)
from trinity._input.registry import SPECS

THRESHOLD = 0.05  # phaseSwitch_LlossLgain default
RCLOUD = 10.0     # pc


def _flat_like_segments():
    """Cooling fires (F0) before the shell reaches rCloud — the flat-profile fate.

    ratio = (Lgain - Lloss)/Lgain drops below 0.05 on the 3rd segment while
    R2 is still well inside rCloud.
    """
    # (t, R2, Lgain, Lloss) -> ratios 0.50, 0.20, 0.04 (fires at t=0.2)
    return [
        (0.0, 1.0, 100.0, 50.0),
        (0.1, 2.0, 100.0, 80.0),
        (0.2, 3.0, 100.0, 96.0),
        (0.3, 4.0, 100.0, 97.0),
    ]


def _steep_like_segments():
    """No cooling family fires; the shell crosses rCloud (F4) — the steep fate.

    Lloss collapses as the bubble expands into ever-lower density, so the ratio
    stays far above 0.05; R2 crosses rCloud=10 on the 3rd segment.
    """
    # ratios 0.90, 0.95, 0.99 (never < 0.05); R2 crosses 10 at t=0.2
    return [
        (0.0, 4.0, 100.0, 10.0),
        (0.1, 8.0, 100.0, 5.0),
        (0.2, 12.0, 100.0, 1.0),
        (0.3, 20.0, 100.0, 0.5),
    ]


def _run(log, segments):
    for t, R2, Lgain, Lloss in segments:
        log.update(t, R2, RCLOUD, Lgain, Lloss, THRESHOLD)
    return log


def test_flat_fires_f0_not_f4():
    log = _run(ShadowTransitionLog(), _flat_like_segments())
    assert log.F0 is not None
    assert log.F0["which"] == "F0"
    assert log.F0["t"] == 0.2  # first segment with ratio < 0.05
    assert log.F0["ratio_F0"] == (100.0 - 96.0) / 100.0
    assert log.F4 is None  # never reached rCloud before cooling fired


def test_steep_fires_f4_not_f0():
    log = _run(ShadowTransitionLog(), _steep_like_segments())
    assert log.F0 is None  # cooling ratio never crosses 0.05
    assert log.F4 is not None
    assert log.F4["which"] == "F4"
    assert log.F4["t"] == 0.2  # first segment with R2 > rCloud
    assert log.F4["R2"] == 12.0


def test_no_criterion_fires():
    log = ShadowTransitionLog()
    # ratio stays at 0.5, R2 stays below rCloud
    log.update(0.0, 1.0, RCLOUD, 100.0, 50.0, THRESHOLD)
    assert log.F0 is None and log.F4 is None
    assert log.records() == []


def test_first_epoch_is_kept_idempotent():
    """Only the first segment where a criterion holds is recorded."""
    log = ShadowTransitionLog()
    log.update(0.2, 3.0, RCLOUD, 100.0, 96.0, THRESHOLD)  # F0 fires
    log.update(0.3, 4.0, RCLOUD, 100.0, 98.0, THRESHOLD)  # would also fire
    assert log.F0["t"] == 0.2  # unchanged


def test_lgain_nonpositive_does_not_fire_f0():
    """A zero/negative Lgain must not spuriously trip the cooling ratio."""
    log = ShadowTransitionLog()
    log.update(0.0, 1.0, RCLOUD, 0.0, 0.0, THRESHOLD)
    assert log.F0 is None


def test_write_emits_jsonl(tmp_path):
    log = _run(ShadowTransitionLog(), _steep_like_segments())
    log.write(tmp_path)
    path = tmp_path / SHADOW_FILENAME
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(lines) == 1
    rec = lines[0]
    assert set(rec) == {"which", "t", "R2", "rCloud", "ratio_F0"}
    assert rec["which"] == "F4"


def test_write_skips_when_nothing_fired(tmp_path):
    log = ShadowTransitionLog()
    log.update(0.0, 1.0, RCLOUD, 100.0, 50.0, THRESHOLD)
    log.write(tmp_path)
    assert not (tmp_path / SHADOW_FILENAME).exists()


def test_f0_before_f4_ordering(tmp_path):
    """When both fire, records() lists F0 then F4."""
    log = ShadowTransitionLog()
    log.update(0.2, 3.0, RCLOUD, 100.0, 96.0, THRESHOLD)   # F0
    log.update(0.3, 12.0, RCLOUD, 100.0, 1.0, THRESHOLD)   # F4
    recs = log.records()
    assert [r["which"] for r in recs] == ["F0", "F4"]


# ---------------------------------------------------------------------------
# Shared criteria predicates (used by both the live terminator and the shadow log)
# ---------------------------------------------------------------------------
def test_cooling_balance_fires_matches_live_expression():
    # ratio = (100-96)/100 = 0.04 < 0.05 -> fires
    assert cooling_balance_fires(100.0, 96.0, THRESHOLD) is True
    # ratio = (100-90)/100 = 0.10 -> does not fire
    assert cooling_balance_fires(100.0, 90.0, THRESHOLD) is False


def test_cooling_balance_nonpositive_lgain_never_fires():
    assert cooling_balance_fires(0.0, 0.0, THRESHOLD) is False
    assert cooling_balance_fires(-5.0, 100.0, THRESHOLD) is False


def test_blowout_fires_on_crossing():
    assert blowout_fires(11.0, RCLOUD) is True
    assert blowout_fires(10.0, RCLOUD) is False  # strict >, equality does not fire
    assert blowout_fires(9.0, RCLOUD) is False
    assert blowout_fires(1e9, None) is False  # no cloud radius -> never


# ---------------------------------------------------------------------------
# implicit_termination_reason — the live F0 v F4 decision (P-promote)
# ---------------------------------------------------------------------------
def test_flat_returns_cooling_balance_in_both_modes():
    # cooling fires, shell still inside cloud
    for mode in VALID_TRANSITION_TRIGGERS:
        assert implicit_termination_reason(
            mode, 100.0, 96.0, THRESHOLD, R2=3.0, rCloud=RCLOUD) == "cooling_balance"


def test_steep_returns_blowout_only_when_promoted():
    # cooling never fires (ratio 0.99); shell escaped the cloud
    assert implicit_termination_reason(
        "cooling_or_blowout", 100.0, 1.0, THRESHOLD, R2=12.0, rCloud=RCLOUD) == "blowout"
    # default mode: F4 is shadow-only -> no termination (would run to stop_t)
    assert implicit_termination_reason(
        "instantaneous", 100.0, 1.0, THRESHOLD, R2=12.0, rCloud=RCLOUD) is None


def test_no_criterion_returns_none():
    assert implicit_termination_reason(
        "cooling_or_blowout", 100.0, 50.0, THRESHOLD, R2=3.0, rCloud=RCLOUD) is None


def test_cooling_takes_precedence_when_both_hold():
    # both F0 and F4 true in the same segment -> cooling_balance wins
    # (preserves the byte-identical pre-promote cooling path)
    assert implicit_termination_reason(
        "cooling_or_blowout", 100.0, 99.0, THRESHOLD, R2=12.0, rCloud=RCLOUD) == "cooling_balance"


# ---------------------------------------------------------------------------
# Trigger validation
# ---------------------------------------------------------------------------
def test_validate_accepts_known_triggers():
    for mode in ("instantaneous", "cooling_or_blowout"):
        assert validate_transition_trigger(mode) == mode


def test_validate_rejects_unknown_trigger():
    with pytest.raises(ValueError):
        validate_transition_trigger("blowout_only")


# ---------------------------------------------------------------------------
# Param registration
# ---------------------------------------------------------------------------
def test_transition_trigger_registered():
    spec = next((s for s in SPECS if s.name == "transition_trigger"), None)
    assert spec is not None, "transition_trigger not registered in SPECS"
    assert spec.default == "instantaneous"
    assert spec.run_const is True
    assert spec.exclude_from_snapshot is True
    assert spec.category == "input_solver"
