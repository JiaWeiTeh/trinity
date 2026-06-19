"""
Failure-contract tests for the bubble-structure solver.

Pins the deterministic failure handling around ``_solve_bubble_structure``:
a bubble solve must never kill the process (the former ``sys.exit`` paths) --
it either returns ``ok=False`` (converted to the fsolve penalty in
``_get_velocity_residuals``) or raises the catchable ``BubbleSolverError``
(penalised by the Phase-1b ``except Exception`` handler). ``SystemExit`` is
NOT an ``Exception``, so the old exits bypassed every handler and took down
whole runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from trinity.bubble_structure import bubble_luminosity as BL


def test_rhs_collapse_returns_ok_false(monkeypatch):
    """A BubbleSolverError raised inside the ODE RHS (T -> 0 collapse) is
    converted by _solve_bubble_structure into its ok=False contract instead
    of escaping (or, formerly, sys.exit-ing the process)."""
    def collapsing_rhs(r, y, params, Pb):
        raise BL.BubbleSolverError("temperature reached zero in bubble ODE RHS")

    monkeypatch.setattr(BL, "_get_bubble_ODE", collapsing_rhs)
    r = np.linspace(1.0, 0.5, 40)
    psoln, ok, info, sol = BL._solve_bubble_structure([1.0, 3e4, -1.0], r, None, None)
    assert ok is False
    assert sol is None
    assert psoln.shape == (40, 3) and np.isnan(psoln).all()
    assert "temperature reached zero" in info["message"]


def test_nonfinite_initial_conditions_return_ok_false():
    """Non-finite y0 must come back as ok=False (solve_ivp would raise raw
    ValueError on it), matching the documented contract."""
    r = np.linspace(1.0, 0.5, 5)
    psoln, ok, info, sol = BL._solve_bubble_structure([np.nan, 3e4, -1.0], r, None, None)
    assert ok is False
    assert sol is None
    assert np.isnan(psoln).all()
    assert "non-finite" in info["message"]


def test_failed_solve_raises_with_message(monkeypatch):
    """_bubble_luminosity turns ok=False into BubbleSolverError carrying
    the infodict message (the message-only failure dict is sufficient)."""
    def failing_solve(initial_conditions, r_array, params, Pb, rtol=None):
        return np.full((len(r_array), 3), np.nan), False, {"message": "xyz-solver-failed"}, None

    monkeypatch.setattr(BL, "_solve_bubble_structure", failing_solve)
    monkeypatch.delenv("TRINITY_BUBBLE_DIAG", raising=False)
    with pytest.raises(BL.BubbleSolverError, match="xyz-solver-failed"):
        BL._bubble_luminosity(object(), 0.5, 1.0, 1.0, [1.0, 3e4, -1.0], 0.7, 1.0)


def test_negative_temperature_raises(monkeypatch):
    """A 'successful' solve whose T profile contains negative (unphysical)
    values raises BubbleSolverError -- it must not be consumed, and must not
    sys.exit. The profile is monotonic so the check itself (not the
    find_nearest_higher monotonic guard) is what fires."""
    def rigged_solve(initial_conditions, r_array, params, Pb, rtol=None):
        n = len(r_array)
        psoln = np.column_stack([
            np.ones(n),                    # v
            np.linspace(-5.0, 1e7, n),     # T: negative start, monotonic rise
            np.full(n, -1.0),              # dTdr
        ])
        return psoln, True, {"message": "ok"}, None

    monkeypatch.setattr(BL, "_solve_bubble_structure", rigged_solve)
    monkeypatch.delenv("TRINITY_BUBBLE_DIAG", raising=False)
    with pytest.raises(BL.BubbleSolverError, match="negative temperature"):
        BL._bubble_luminosity(object(), 0.5, 1.0, 1.0, [1.0, 3e4, -1.0], 0.7, 1.0)
