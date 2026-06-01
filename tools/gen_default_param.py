#!/usr/bin/env python3
"""Generate ``trinity/_input/default.param`` from the ParamSpec registry.

The registry (``trinity._input.registry.SPECS``) is the single source of
truth for TRINITY's parameters.  ``default.param`` is, as of Phase 4,
a *generated artifact*: the schema + defaults file that ``read_param``
reads.  This module renders it.

Phase 3 (this commit) adds the renderer + a CI gate
(``test/test_gen_default_param.py``) that asserts the committed
``default.param`` is *semantically* equivalent to ``render(SPECS)``
(same keys → parsed values, units, and INFO/DEPRECATED text).  The
source-of-truth flip — making the committed file byte-identical to the
render and relocating the decorative prose to ``docs/`` — happens in
Phase 4.  Until then nothing here is wired into production.

Only *file-backed* specs are emitted: those whose ``category`` starts
with ``input_`` or equals ``deprecated`` (the 68 keys that live in
``default.param``).  Runtime / derived specs are created in
``read_param`` Steps 6/8/10 and never appear in the file.

CLI
---
    python -m tools.gen_default_param            # print to stdout
    python -m tools.gen_default_param --check    # exit 1 if committed file drifted
    python -m tools.gen_default_param --write     # overwrite default.param
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from trinity._input.param_spec import ParamSpec
from trinity._input.registry import SPECS

# Path to the file this module renders.
DEFAULT_PARAM_PATH = (
    Path(__file__).resolve().parents[1] / "trinity" / "_input" / "default.param"
)

# Generated-file banner.  ``read_param`` ignores all of these comment
# lines; they orient a human who opens the file.
_HEADER = """\
# ==============================================================================
# TRINITY PARAMETER SCHEMA + DEFAULTS  —  AUTO-GENERATED, DO NOT EDIT BY HAND
# ------------------------------------------------------------------------------
# This file is generated from the ParamSpec registry:
#     trinity/_input/registry.py
# Regenerate after changing a spec:
#     python -m tools.gen_default_param --write
#
# Roles (unchanged): (1) SCHEMA — the complete set of valid parameter names;
# any key not listed here is rejected by read_param.py. (2) DEFAULTS — every
# value here is used when a user .param omits the key.
#
# To configure a run, create a .param file under param/ and override only the
# keys you need. Human-facing parameter docs (units, allowed values, physics):
# see docs/source/parameters.rst.
# ==============================================================================
"""

# Category -> human section title, in the order sections are emitted.
# Every file-backed category must appear here (asserted in render()).
_SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("input_admin",        "Administrative & Logging"),
    ("input_physical",     "Physical Parameters"),
    ("input_profile",      "Density Profile"),
    ("input_termination",  "Termination Conditions"),
    ("input_sps",          "Stellar Feedback (SPS)"),
    ("input_cooling",      "Cooling Tables"),
    ("input_constants",    "Physical Constants & Microphysics"),
    ("input_solver",       "Solver / Phase Control"),
    ("deprecated",         "Deprecated (parsed for back-compat, not consumed)"),
)


def is_file_backed(spec: ParamSpec) -> bool:
    """True iff this spec is declared in ``default.param``."""
    return spec.category.startswith("input_") or spec.category == "deprecated"


def _render_spec(spec: ParamSpec) -> str:
    """Render one spec's comment annotations + ``key value`` line."""
    lines: list[str] = []
    # INFO (single line — the registry stores the full joined paragraph;
    # read_param only surfaces the last # INFO: line, but the loader is
    # whitespace-insensitive so one long line is faithful).
    if spec.info:
        lines.append(f"# INFO: {spec.info}")
    # DEPRECATED note (read_param ignores these lines; they document the key).
    if spec.deprecated_note:
        lines.append(f"# DEPRECATED: {spec.deprecated_note}")
    # UNIT (bracketed, matching the historical convention; read_param strips []).
    if spec.unit:
        lines.append(f"# UNIT: [{spec.unit}]")
    # The parameter line itself.  Input specs store ``default`` as the raw
    # source string, so it is emitted verbatim.
    lines.append(f"{spec.name}    {spec.default}")
    return "\n".join(lines)


def render(specs: Iterable[ParamSpec] = SPECS) -> str:
    """Return the full ``default.param`` text for the given specs."""
    specs = list(specs)
    file_backed = [s for s in specs if is_file_backed(s)]

    known_sections = {cat for cat, _ in _SECTION_ORDER}
    seen_cats = {s.category for s in file_backed}
    missing = seen_cats - known_sections
    if missing:
        raise ValueError(
            f"file-backed categories with no section title: {sorted(missing)} "
            f"(add them to _SECTION_ORDER in tools/gen_default_param.py)"
        )

    out: list[str] = [_HEADER]
    for cat, title in _SECTION_ORDER:
        group = [s for s in file_backed if s.category == cat]
        if not group:
            continue
        out.append(
            "\n# "
            + "=" * 78
            + f"\n# {title}\n# "
            + "=" * 78
        )
        for spec in group:
            out.append(_render_spec(spec))
    # Trailing newline so the file ends cleanly.
    return "\n\n".join(out) + "\n"


def _semantic_diff() -> list[str]:
    """Compare the committed file to ``render(SPECS)`` semantically.

    Returns a list of human-readable drift messages (empty if in sync).
    Imported lazily by ``--check`` and by the test module.
    """
    from tools._param_text import parse_param_text  # local helper

    committed = parse_param_text(DEFAULT_PARAM_PATH.read_text(encoding="utf-8"))
    generated = parse_param_text(render())
    return _diff_parsed(committed, generated)


def _diff_parsed(committed: dict, generated: dict) -> list[str]:
    from fractions import Fraction

    def pv(s):
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

    msgs: list[str] = []
    only_c = set(committed) - set(generated)
    only_g = set(generated) - set(committed)
    if only_c:
        msgs.append(f"keys only in committed default.param: {sorted(only_c)}")
    if only_g:
        msgs.append(f"keys only in generated output: {sorted(only_g)}")
    for k in sorted(set(committed) & set(generated)):
        c, g = committed[k], generated[k]
        if pv(c["value"]) != pv(g["value"]):
            msgs.append(f"{k}: value {c['value']!r} (file) != {g['value']!r} (gen)")
        if c["unit"] != g["unit"]:
            msgs.append(f"{k}: unit {c['unit']!r} (file) != {g['unit']!r} (gen)")
        if c["info"] != g["info"]:
            msgs.append(f"{k}: INFO differs\n   file={c['info']!r}\n   gen ={g['info']!r}")
        if c["deprecated"] != g["deprecated"]:
            msgs.append(f"{k}: DEPRECATED differs")
    return msgs


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--write" in argv:
        DEFAULT_PARAM_PATH.write_text(render(), encoding="utf-8")
        print(f"wrote {DEFAULT_PARAM_PATH}")
        return 0
    if "--check" in argv:
        drift = _semantic_diff()
        if drift:
            print("default.param is OUT OF SYNC with the registry:")
            for m in drift:
                print("  -", m)
            return 1
        print("default.param is in sync with the registry (semantic).")
        return 0
    sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
