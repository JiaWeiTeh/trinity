"""Codegen gate: the committed ``default.param`` must stay in sync with
the ParamSpec registry.

Phase 3 pins *semantic* equivalence (same keys → parsed values, units,
and INFO/DEPRECATED text).  The committed file still carries decorative
prose and multi-line INFO that the generator collapses; those are
compared by parsed meaning, not bytes.  Phase 4 relocates the prose to
``docs/`` and upgrades this gate to byte-exact.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from tools._param_text import parse_param_text
from tools.gen_default_param import (
    DEFAULT_PARAM_PATH,
    is_file_backed,
    render,
)
from src._input.registry import SPECS


def _parse_value(s: str):
    """read_param's value precedence: None → bool → float → fraction → str."""
    s = s.strip()
    if s.lower() == "none":
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return float(s)
    except ValueError:
        pass
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return s


def _committed() -> dict:
    return parse_param_text(DEFAULT_PARAM_PATH.read_text(encoding="utf-8"))


def _generated() -> dict:
    return parse_param_text(render())


# ---------------------------------------------------------------------------
# Core drift gate
# ---------------------------------------------------------------------------
def test_generated_and_committed_have_same_keys() -> None:
    c, g = _committed(), _generated()
    assert set(c) == set(g), (
        f"key mismatch — only in file: {sorted(set(c) - set(g))}; "
        f"only in generated: {sorted(set(g) - set(c))}"
    )


def test_parsed_values_match() -> None:
    c, g = _committed(), _generated()
    for k in c:
        assert _parse_value(c[k]["value"]) == _parse_value(g[k]["value"]), (
            f"{k}: file value {c[k]['value']!r} parses != generated {g[k]['value']!r}"
        )


def test_units_match() -> None:
    c, g = _committed(), _generated()
    for k in c:
        assert c[k]["unit"] == g[k]["unit"], (
            f"{k}: file unit {c[k]['unit']!r} != generated {g[k]['unit']!r}"
        )


def test_info_text_matches() -> None:
    """Joined INFO text (lossless) must match between file and registry."""
    c, g = _committed(), _generated()
    for k in c:
        assert c[k]["info"] == g[k]["info"], (
            f"{k}: INFO differs\n   file={c[k]['info']!r}\n   gen ={g[k]['info']!r}"
        )


def test_deprecated_text_matches() -> None:
    c, g = _committed(), _generated()
    for k in c:
        assert c[k]["deprecated"] == g[k]["deprecated"], f"{k}: DEPRECATED text differs"


# ---------------------------------------------------------------------------
# Generator scope / sanity
# ---------------------------------------------------------------------------
def test_only_file_backed_specs_emitted() -> None:
    """The generated keys are exactly the input_* + deprecated specs —
    no runtime/derived specs leak into the file."""
    g = _generated()
    expected = {s.name for s in SPECS if is_file_backed(s)}
    assert set(g) == expected
    # And none of those are runtime/derived
    by_name = {s.name: s for s in SPECS}
    for k in g:
        cat = by_name[k].category
        assert cat.startswith("input_") or cat == "deprecated", f"{k}: {cat} leaked"


def test_render_is_idempotent() -> None:
    """Parsing the render and re-parsing a second render agree (stable)."""
    assert parse_param_text(render()) == parse_param_text(render())


def test_generated_text_loads_via_grammar() -> None:
    """Every file-backed spec round-trips: appears in the generated text
    with its registry default/unit."""
    g = _generated()
    for s in SPECS:
        if not is_file_backed(s):
            continue
        assert s.name in g
        assert _parse_value(g[s.name]["value"]) == _parse_value(str(s.default))
        assert g[s.name]["unit"] == s.unit
