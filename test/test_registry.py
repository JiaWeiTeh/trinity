"""Guardrails on the ParamSpec registry.

Phase 1 added the construction-time invariants (frozen, identifier
names, deprecated-needs-note).  Phase 2 populated ``SPECS`` with all
187 parameters, so the reconciliation tests below now have teeth:
they pin the registry against a live ``read_param`` run and against
the legacy ``run_constants`` hand-lists it will replace in Phase 5.
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path

import pytest

from src._input.dictionary import DescribedDict
from src._input.param_spec import SENTINEL_PREFIX, ParamSpec
from src._input.read_param import read_param
from src._input.registry import (
    KNOWN_STALE_METADATA_EXCLUDE,
    KNOWN_STALE_RUN_CONST,
    REGISTRY,
    SPECS,
    metadata_exclude_keys,
    run_const_keys,
)
from src._output.run_constants import METADATA_EXCLUDE, RUN_CONST_KEYS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _disable_crash_handlers(monkeypatch):
    """Stop ``DescribedDict.__init__`` registering atexit/signal hooks
    so repeated ``read_param`` calls don't leak handlers."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _parse_value(val_str: str):
    """Mirror of ``read_param``'s nested ``parse_value`` (None → bool →
    float → fraction → str).  Duplicated here because the original is a
    closure; Phase 6 will lift it to module scope and this can import it.
    """
    s = val_str.strip()
    if s.lower() == "none":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return s


def _parse_default_param() -> dict:
    """Return {key: (raw_value, unit_or_None)} parsed from default.param,
    using the same INFO/UNIT grammar as read_param."""
    path = Path(__file__).resolve().parents[1] / "src" / "_input" / "default.param"
    out: dict[str, tuple] = {}
    unit = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("# UNIT:"):
            unit = s[len("# UNIT:"):].strip().strip("[]").strip()
            continue
        if s.startswith("#"):
            continue
        if not s:
            unit = None
            continue
        parts = s.split(None, 1)
        if len(parts) == 2:
            key, val = parts[0], parts[1].split("#")[0].strip()
            out[key] = (val, unit)
            unit = None
    return out


def _write_param(tmp_path: Path, profile: str) -> Path:
    """Write a minimal user .param selecting the given density profile,
    routing output into tmp_path so no real outputs/ dir is created."""
    body = (
        f"model_name    reg_test_{profile}\n"
        f"path2output    {tmp_path / profile}\n"
        f"dens_profile    {profile}\n"
    )
    if profile == "densBE":
        body += "densBE_Omega    14.1\n"
    p = tmp_path / f"{profile}.param"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture(scope="function")
def live_keys(tmp_path) -> dict:
    """Union of params produced by a densPL run and a densBE run, plus
    the per-key live DescribedItem for value/unit/exclude assertions."""
    pl = read_param(_write_param(tmp_path, "densPL"))
    be = read_param(_write_param(tmp_path, "densBE"))
    merged = dict(pl)
    for k, v in be.items():
        merged.setdefault(k, v)
    return merged


# ---------------------------------------------------------------------------
# Structural invariants (Phase 1, still hold)
# ---------------------------------------------------------------------------
def test_registry_has_unique_names() -> None:
    names = [s.name for s in SPECS]
    assert len(names) == len(set(names)), "Duplicate spec names in SPECS"


def test_registry_dict_matches_spec_tuple() -> None:
    assert list(REGISTRY.keys()) == [s.name for s in SPECS]
    assert all(REGISTRY[s.name] is s for s in SPECS)


def test_run_const_keys_returns_tuple_of_strings() -> None:
    result = run_const_keys()
    assert isinstance(result, tuple)
    assert all(isinstance(k, str) for k in result)


def test_metadata_exclude_keys_returns_frozenset_of_strings() -> None:
    result = metadata_exclude_keys()
    assert isinstance(result, frozenset)
    assert all(isinstance(k, str) for k in result)


def test_run_const_and_metadata_exclude_disjoint() -> None:
    assert set(run_const_keys()) & metadata_exclude_keys() == set()


# ---------------------------------------------------------------------------
# Construction-time guards
# ---------------------------------------------------------------------------
def test_deprecated_category_requires_note() -> None:
    with pytest.raises(ValueError, match="deprecated_note"):
        ParamSpec(name="probe_dep", default=False, info="x", category="deprecated")


def test_run_const_and_metadata_exclude_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually"):
        ParamSpec(
            name="probe_both", default=0, info="x", category="input_physical",
            run_const=True, metadata_exclude=True,
        )


def test_paramspec_name_must_be_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        ParamSpec(name="not valid", default=0, info="x", category="input_physical")


def test_paramspec_is_frozen() -> None:
    spec = ParamSpec(name="probe_frozen", default=0, info="x", category="input_physical")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_sentinel_prefix_constant() -> None:
    assert SENTINEL_PREFIX == "def_"


@pytest.mark.xfail(reason="resolvers wired in Phase 7", strict=True)
def test_every_sentinel_default_has_resolver() -> None:
    """Every ``def_*`` sentinel default must carry a resolver.  Wired in
    Phase 7; ``strict=True`` forces removal of this marker once it does."""
    offenders = [
        s.name for s in SPECS
        if isinstance(s.default, str)
        and s.default.startswith(SENTINEL_PREFIX)
        and s.resolver is None
    ]
    assert not offenders, f"sentinel specs missing resolver: {offenders}"


# ---------------------------------------------------------------------------
# Reconciliation against a live read_param run
# ---------------------------------------------------------------------------
def test_registry_covers_all_param_keys(live_keys) -> None:
    """Every key produced by a real run has a spec, and no spec names a
    key the loader never creates."""
    registry_names = set(REGISTRY)
    live = set(live_keys)
    assert live - registry_names == set(), f"params missing a spec: {sorted(live - registry_names)}"
    assert registry_names - live == set(), f"specs with no live param: {sorted(registry_names - live)}"


def test_input_defaults_match_default_param() -> None:
    """Parsed value of each input spec's default equals the parsed value
    of default.param's raw entry (tolerates 14/11 vs 1.2727… encoding)."""
    dp = _parse_default_param()
    for s in SPECS:
        if not s.category.startswith("input_") and s.category != "deprecated":
            continue
        assert s.name in dp, f"{s.name} not in default.param"
        raw, _ = dp[s.name]
        assert _parse_value(str(s.default)) == _parse_value(raw), (
            f"{s.name}: registry default {s.default!r} != default.param {raw!r}"
        )


def test_input_units_match_default_param() -> None:
    """Input spec units match default.param's # UNIT: annotation."""
    dp = _parse_default_param()
    for s in SPECS:
        if not s.category.startswith("input_") and s.category != "deprecated":
            continue
        _, unit = dp[s.name]
        assert s.unit == unit, f"{s.name}: unit {s.unit!r} != annotation {unit!r}"


def test_runtime_units_match_live(live_keys) -> None:
    """Runtime/derived spec units match the live DescribedItem.ori_units."""
    for s in SPECS:
        if s.category.startswith("input_") or s.category == "deprecated":
            continue
        assert s.unit == live_keys[s.name].ori_units, (
            f"{s.name}: unit {s.unit!r} != live {live_keys[s.name].ori_units!r}"
        )


def test_exclude_from_snapshot_matches_live(live_keys) -> None:
    for s in SPECS:
        assert s.exclude_from_snapshot == live_keys[s.name].exclude_from_snapshot, (
            f"{s.name}: exclude_from_snapshot {s.exclude_from_snapshot} != live "
            f"{live_keys[s.name].exclude_from_snapshot}"
        )


# ---------------------------------------------------------------------------
# Reconciliation against the legacy run_constants hand-lists (Phase-5 target)
# ---------------------------------------------------------------------------
def test_run_const_keys_matches_legacy() -> None:
    assert set(run_const_keys()) == set(RUN_CONST_KEYS) - KNOWN_STALE_RUN_CONST


def test_metadata_exclude_matches_legacy() -> None:
    assert metadata_exclude_keys() == set(METADATA_EXCLUDE) - KNOWN_STALE_METADATA_EXCLUDE


def test_known_stale_keys_are_actually_stale(live_keys) -> None:
    """The carve-out sets must only contain keys that genuinely no longer
    exist — otherwise we'd be masking a real spec gap."""
    live = set(live_keys)
    for k in KNOWN_STALE_RUN_CONST | KNOWN_STALE_METADATA_EXCLUDE:
        assert k not in live, f"{k} is marked stale but a live param exists"


# ---------------------------------------------------------------------------
# Per-field consistency
# ---------------------------------------------------------------------------
def test_deprecated_specs_have_notes() -> None:
    dep = [s for s in SPECS if s.category == "deprecated"]
    assert {s.name for s in dep} == {
        "stop_v", "adiabaticOnlyInCore", "immediate_leak", "use_adaptive_solver"
    }
    assert all(s.deprecated_note for s in dep)


def test_active_when_only_on_conditional_specs() -> None:
    """Only the densBE/densPL profile-conditional keys carry active_when."""
    with_active = {s.name for s in SPECS if s.active_when is not None}
    expected = {
        "densBE_Omega", "densPL_alpha",
        "densBE_Teff", "densBE_xi_arr", "densBE_u_arr", "densBE_dudxi_arr",
        "densBE_rho_rhoc_arr", "densBE_f_rho_rhoc", "densBE_f_m", "densBE_xi_out",
    }
    assert with_active == expected
