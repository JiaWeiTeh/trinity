"""Phase-3 scaffolding: the ``betadelta_solver`` param switch.

Commit 1 wires a solver-selection param (``legacy`` default, ``hybr``
reserved) into ``solve_betadelta_pure`` without changing legacy behaviour.
These tests pin the switch contract: the validator's allowed set, the
dispatch routing (legacy default, missing key -> legacy, hybr stubbed),
and the byte-identical guarantee that the default path is exactly the
pre-switch legacy implementation.
"""

from types import SimpleNamespace

import pytest

import trinity.phase1b_energy_implicit.get_betadelta as GBD
from trinity._input.errors import ParameterFileError
from trinity._input.registry import SPECS

# =============================================================================
# Validator
# =============================================================================


def test_validator_accepts_legacy_and_hybr():
    from trinity._input.registry import _validate_betadelta_solver

    # Should not raise.
    _validate_betadelta_solver("legacy", None)
    _validate_betadelta_solver("hybr", None)


@pytest.mark.parametrize("bad", ["LEGACY", "grid", "scipy", "", "hybrid"])
def test_validator_rejects_unknown(bad):
    from trinity._input.registry import _validate_betadelta_solver

    with pytest.raises(ParameterFileError):
        _validate_betadelta_solver(bad, None)


# =============================================================================
# Registry / default
# =============================================================================


def test_registry_default_is_legacy():
    spec = next(s for s in SPECS if s.name == "betadelta_solver")
    assert spec.default == "legacy"
    assert spec.validator is not None
    assert spec.run_const is True


# =============================================================================
# Solver-choice reader
# =============================================================================


def test_solver_choice_defaults_to_legacy_when_missing():
    assert GBD._get_betadelta_solver({}) == "legacy"


def test_solver_choice_reads_described_item():
    params = {"betadelta_solver": SimpleNamespace(value="hybr")}
    assert GBD._get_betadelta_solver(params) == "hybr"


def test_solver_choice_empty_value_falls_back_to_legacy():
    params = {"betadelta_solver": SimpleNamespace(value="")}
    assert GBD._get_betadelta_solver(params) == "legacy"


# =============================================================================
# Dispatch routing
# =============================================================================


def test_dispatch_legacy_calls_legacy_impl(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_legacy(beta, delta, params, method):
        seen["args"] = (beta, delta, params, method)
        return sentinel

    monkeypatch.setattr(GBD, "_solve_betadelta_legacy", fake_legacy)
    params = {"betadelta_solver": SimpleNamespace(value="legacy")}
    out = GBD.solve_betadelta_pure(0.5, -0.5, params, method="grid")
    assert out is sentinel
    assert seen["args"] == (0.5, -0.5, params, "grid")


def test_dispatch_missing_key_routes_to_legacy(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(GBD, "_solve_betadelta_legacy", lambda *a, **k: sentinel)
    # No 'betadelta_solver' key (mirrors the older unit-test fixtures).
    assert GBD.solve_betadelta_pure(0.5, -0.5, {}) is sentinel


def test_dispatch_hybr_calls_hybr_impl(monkeypatch):
    sentinel = object()
    seen = {}

    def fake_hybr(beta, delta, params, method):
        seen["args"] = (beta, delta, params, method)
        return sentinel

    monkeypatch.setattr(GBD, "_solve_betadelta_hybr", fake_hybr)
    params = {"betadelta_solver": SimpleNamespace(value="hybr")}
    out = GBD.solve_betadelta_pure(0.5, -0.5, params, method="grid")
    assert out is sentinel
    assert seen["args"] == (0.5, -0.5, params, "grid")
