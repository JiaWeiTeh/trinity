"""Phase-1 guardrails on the ParamSpec registry.

These tests pass on an empty ``SPECS``.  Phase 2's population is
the first commit that gives the iteration-based tests teeth; the
construction-time tests already exercise ``ParamSpec.__post_init__``
today.
"""
from __future__ import annotations

import dataclasses

import pytest

from src._input.param_spec import SENTINEL_PREFIX, ParamSpec
from src._input.registry import (
    REGISTRY,
    SPECS,
    metadata_exclude_keys,
    run_const_keys,
)


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


def test_run_const_and_exclude_are_disjoint() -> None:
    """A spec is either serializable (→ run_const) or not (→ exclude).
    Phase-5's swap into ``run_constants.py`` relies on this."""
    assert set(run_const_keys()) & metadata_exclude_keys() == set()


def test_sentinel_default_requires_resolver_at_construction() -> None:
    """Sentinel default without a resolver must fail at __post_init__."""
    with pytest.raises(ValueError, match="sentinel"):
        ParamSpec(
            name="probe_sentinel",
            default="def_dir",
            info="x",
            category="input_admin",
        )


def test_deprecated_category_requires_note() -> None:
    with pytest.raises(ValueError, match="deprecated_note"):
        ParamSpec(
            name="probe_deprecated",
            default=False,
            info="x",
            category="deprecated",
        )


def test_paramspec_name_must_be_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        ParamSpec(
            name="not a valid name",
            default=0,
            info="x",
            category="input_physical",
        )


def test_paramspec_is_frozen() -> None:
    spec = ParamSpec(
        name="probe_frozen",
        default=0,
        info="x",
        category="input_physical",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_sentinel_prefix_constant() -> None:
    """``SENTINEL_PREFIX`` is part of the public contract — readers
    (Phase 7 resolver) detect sentinels by checking this prefix."""
    assert SENTINEL_PREFIX == "def_"
