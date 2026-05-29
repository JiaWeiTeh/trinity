"""Lossless parser for the ``default.param`` grammar.

Shared by ``tools.gen_default_param`` (``--check``) and the codegen
test, so both compare the committed file and the generated text the
same way.

Differs from ``read_param``'s inline parser in one deliberate way:
``read_param`` keeps only the *last* ``# INFO:`` line for a key (it
overwrites), whereas this parser joins *all* ``# INFO:`` lines.  The
registry stores the full joined paragraph, so the lossless join is the
correct basis for a registry↔file comparison.  ``# DEPRECATED:`` lines
(which ``read_param`` ignores entirely) are likewise collected here.
"""
from __future__ import annotations


def parse_param_text(text: str) -> dict[str, dict]:
    """Parse default.param-format text.

    Returns ``{key: {"value": raw_str, "unit": str|None,
    "info": joined_str, "deprecated": joined_str}}``.

    ``# INFO:`` / ``# UNIT:`` / ``# DEPRECATED:`` annotations attach to
    the next ``key value`` line and reset after it (matching how
    ``read_param`` consumes pending metadata at each parameter).  Plain
    ``#`` comment lines and blank lines are ignored and do NOT reset
    pending annotations (also matching ``read_param``).
    """
    out: dict[str, dict] = {}
    info: list[str] = []
    deprecated: list[str] = []
    unit: str | None = None

    for raw_line in text.splitlines():
        s = raw_line.strip()
        if s.startswith("# INFO:"):
            info.append(s[len("# INFO:"):].strip())
            continue
        if s.startswith("# UNIT:"):
            unit = s[len("# UNIT:"):].strip().strip("[]").strip()
            continue
        if s.startswith("# DEPRECATED:"):
            deprecated.append(s[len("# DEPRECATED:"):].strip())
            continue
        if s.startswith("#") or not s:
            # decorative comment / blank — ignored, does not reset pending meta
            continue
        # parameter line: "key value..."
        parts = s.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts[0], parts[1].split("#")[0].strip()
        out[key] = {
            "value": value,
            "unit": unit,
            "info": " ".join(info),
            "deprecated": " ".join(deprecated),
        }
        info, deprecated, unit = [], [], None
    return out
