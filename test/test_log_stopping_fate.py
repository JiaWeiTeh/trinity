"""
Tests for the INFO-level physical-state and stopping-fate log helpers in
``trinity._output.terminal_prints``.

These lock in two things a researcher relies on when reading ``trinity.log``:

1. The stopping fate is spelled out in words (enum name + code + reason),
   not just a bare exit code — this is the headline result of a run.
2. The phase-boundary state block is tolerant of missing/zero fields and
   shows conventional display units (km/s, erg), not raw internal units.

The helpers only need ``params.get(key).value``, so we feed them a lightweight
stub dict (avoids DescribedDict's atexit snapshot machinery).
"""

from __future__ import annotations

import logging

import trinity._functions.unit_conversions as cvt
import trinity._output.terminal_prints as terminal_prints


class _Item:
    """Minimal stand-in for DescribedItem: just carries ``.value``."""

    def __init__(self, value):
        self.value = value


def _params(**kw):
    """Build a params-like dict mapping keys to objects with ``.value``."""
    return {k: _Item(v) for k, v in kw.items()}


# Representative internal-unit state (numbers are non-round so a wrong/missing
# unit conversion cannot reproduce the expected string by coincidence).
def _full_state(**overrides):
    state = dict(
        t_now=0.00291, R2=0.5324, v2=104.4, Eb=2.63e4,
        Pb=7.75e-3, T0=6.93e6, R1=0.4981, shell_mass=2.188,
    )
    state.update(overrides)
    return _params(**state)


# ---------------------------------------------------------------------------
# Stopping fate spelled out in words
# ---------------------------------------------------------------------------

def test_end_report_spells_out_clean_fate():
    params = _full_state()
    params["SimulationEndCode"] = _Item(1)  # STOPPING_TIME
    params["SimulationEndReason"] = _Item("reached stop_t=10.0 Myr")

    out = terminal_prints.format_end_report(params)

    assert "Simulation ended" in out
    assert "STOPPING_TIME" in out
    assert "code 1" in out
    assert "reached stop_t=10.0 Myr" in out
    # the final-state block is appended
    assert "final state" in out


def test_end_report_error_code_says_failed():
    params = _full_state()
    params["SimulationEndCode"] = _Item(20)  # ERROR_NUMERICAL
    params["SimulationEndReason"] = _Item("numerical blowup")

    out = terminal_prints.format_end_report(params)

    assert "Simulation FAILED" in out
    assert "ERROR_NUMERICAL" in out
    assert "code 20" in out


def test_end_report_missing_code_is_unknown_and_flagged():
    params = _full_state()  # no SimulationEndCode / Reason set
    out = terminal_prints.format_end_report(params)

    assert "UNKNOWN" in out
    assert "inspection required" in out
    assert "unknown" in out  # the default reason string


# ---------------------------------------------------------------------------
# State block: tolerance + correct display units
# ---------------------------------------------------------------------------

def test_state_missing_field_renders_na():
    params = _full_state()
    del params["R1"]
    out = terminal_prints.format_state(params)
    assert "R1 = n/a" in out


def test_state_eb_zero_renders_not_na():
    # Momentum phase sets Eb=0; it must render as a number, not 'n/a'.
    params = _full_state(Eb=0.0)
    out = terminal_prints.format_state(params)
    assert "Eb = 0.0000e+00 erg" in out


def test_state_velocity_shown_in_kms_not_internal():
    # v2 = 100 pc/Myr must be converted to km/s (~97.78), not printed as 100.
    params = _full_state(v2=100.0)
    out = terminal_prints.format_state(params, oneline=True)
    expected = format(100.0 * cvt.v_au2kms, ".4f")
    assert f"v2 = {expected} km/s" in out
    assert "v2 = 100.0000 km/s" not in out


def test_state_block_is_multiline_oneline_is_single_line():
    params = _full_state()
    block = terminal_prints.format_state(params, label="entry")
    line = terminal_prints.format_state(params, label="entry", oneline=True)
    assert "\n" in block
    assert "\n" not in line


# ---------------------------------------------------------------------------
# Heartbeat: self-throttling
# ---------------------------------------------------------------------------

def test_heartbeat_only_fires_on_multiples(caplog):
    params = _full_state(t_now=1.83)
    every = terminal_prints.HEARTBEAT_EVERY
    with caplog.at_level(logging.INFO, logger="trinity._output.terminal_prints"):
        terminal_prints.heartbeat(params, "1b implicit", every - 1, 0.0, 10.0)
        assert "1b implicit" not in caplog.text  # off-cadence: silent
        terminal_prints.heartbeat(params, "1b implicit", every, 0.0, 10.0)
    assert "1b implicit" in caplog.text
    assert f"seg {every}" in caplog.text
    assert "18.3% of t 0->10 Myr" in caplog.text  # progress bar from t_now=1.83
