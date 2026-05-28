"""Module-level registry of ParamSpec entries.

Phase 1: ``SPECS`` is empty; ``run_const_keys`` and
``metadata_exclude_keys`` are dormant drop-in replacements for the
hand-curated lists in ``src._output.run_constants``.  Phase 2
populates ``SPECS``; Phase 5 wires the swap.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from src._input.param_spec import Category, ParamSpec

SPECS: tuple[ParamSpec, ...] = ()

REGISTRY: "OrderedDict[str, ParamSpec]" = OrderedDict(
    (s.name, s) for s in SPECS
)

# Categories whose values are established at startup and stay
# constant for the rest of the run — eligible for
# ``RUN_CONST_KEYS``.  Runtime-only categories
# (``runtime_state`` / ``runtime_loaded``) are excluded.
_INPUT_LIKE_CATEGORIES: frozenset[Category] = frozenset({
    "input_admin",
    "input_physical",
    "input_profile",
    "input_termination",
    "input_sps",
    "input_cooling",
    "input_constants",
    "input_solver",
    "derived_init",
    "deprecated",
})


def specs_by_category(*categories: Category) -> Iterable[ParamSpec]:
    cat_set = set(categories)
    return (s for s in SPECS if s.category in cat_set)


def run_const_keys() -> tuple[str, ...]:
    """Keys constant once phase 0 finishes — written to metadata.json.

    Phase-5 drop-in replacement for
    ``src._output.run_constants.RUN_CONST_KEYS``.
    """
    return tuple(
        s.name for s in SPECS
        if s.category in _INPUT_LIKE_CATEGORIES and s.serializable
    )


def metadata_exclude_keys() -> frozenset[str]:
    """Keys that look constant but must NOT land in metadata.json
    (paths, interpolators, function tables).

    Phase-5 drop-in replacement for
    ``src._output.run_constants.METADATA_EXCLUDE``.
    """
    return frozenset(s.name for s in SPECS if not s.serializable)
