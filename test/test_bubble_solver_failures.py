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


def _fake_solve_ivp(fail_methods, calls):
    """solve_ivp stand-in: raises the scipy duplicate-step error for the listed
    methods and returns a trivial dense solution otherwise."""
    from types import SimpleNamespace

    def fake(fun, t_span, y0, method, **kw):
        calls.append(method)
        if method in fail_methods:
            raise ValueError("`ts` must be strictly increasing or decreasing.")
        return SimpleNamespace(sol=lambda r: np.ones((3, np.size(r))), success=True,
                               message="ok", status=0, nfev=1, t=np.array([1.0, 0.5]))
    return fake


def test_lsoda_duplicate_step_is_retried_with_radau(monkeypatch):
    """pilot_v5 (2026-09-30): LSODA's dense output raised `ts must be strictly
    increasing` on growing 5e9 Msun bubbles and phase 1a ended the run. The solve is
    now retried once with Radau."""
    calls = []
    monkeypatch.setattr(BL.scipy.integrate, "solve_ivp", _fake_solve_ivp({"LSODA"}, calls))
    r = np.linspace(1.0, 0.5, 7)
    psoln, ok, info, sol = BL._solve_bubble_structure([1.0, 3e4, -1.0], r, None, None)
    assert calls == ["LSODA", "Radau"]
    assert ok is True and psoln.shape == (7, 3) and np.isfinite(psoln).all()


def test_other_value_errors_are_not_retried(monkeypatch):
    """Only the duplicate-step error is retried; e.g. a cooling-table bounds error
    propagates exactly as before (the caller turns it into a penalty)."""
    calls = []

    def fake(fun, t_span, y0, method, **kw):
        calls.append(method)
        raise ValueError("One of the requested xi is out of bounds in dimension 0")
    monkeypatch.setattr(BL.scipy.integrate, "solve_ivp", fake)
    with pytest.raises(ValueError, match="out of bounds"):
        BL._solve_bubble_structure([1.0, 3e4, -1.0], np.linspace(1.0, 0.5, 7), None, None)
    assert calls == ["LSODA"]


def test_second_duplicate_step_failure_is_ok_false(monkeypatch):
    calls = []
    monkeypatch.setattr(BL.scipy.integrate, "solve_ivp",
                        _fake_solve_ivp({"LSODA", "Radau"}, calls))
    r = np.linspace(1.0, 0.5, 7)
    psoln, ok, info, sol = BL._solve_bubble_structure([1.0, 3e4, -1.0], r, None, None)
    assert calls == ["LSODA", "Radau"]
    assert ok is False and sol is None and np.isnan(psoln).all()
    assert "Radau retry" in info["message"]
