#!/usr/bin/env python3
"""No bare `np.trapezoid` anywhere the cluster can reach it.

numpy 2.0 renamed np.trapz -> np.trapezoid. The repo PINS numpy>=1.20,<2, so the new
spelling is an AttributeError on Helix while running green on a numpy-2 laptop. That
asymmetry killed all 8100 tasks of the 2026-09-07 rosette sweep at 12 s, and PLAN.md
W76 then found three more copies of it. This is the third recurrence; hence a test.

`np.trapz` needs no guard: it still exists on numpy 2 (DeprecationWarning only).
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", ".git", "node_modules", "__pycache__",
              "pre_w69_backup",      # frozen pre-patch bytes, kept byte-exact on purpose
              "to-be-deleted"}       # retired, not on any path
GUARDS = ("getattr(np", "hasattr(np")
_CALL = "np.trapezoid" + "("      # a CALL, not prose mentioning the name
# `getattr(np, "trapezoid", np.trapz)` LOOKS guarded but evaluates np.trapz eagerly as the
# default, so it raises the day numpy drops trapz. Only the short-circuiting form is safe.
_EAGER = re.compile(r"getattr\(\s*np\s*,\s*['\"]trapezoid['\"]\s*,\s*np\.trapz")


SELF = Path(__file__).resolve()


def _files():
    return [p for p in REPO.rglob("*.py")
            if not SKIP_PARTS & set(p.parts) and p.resolve() != SELF]


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_bare_trapezoid(path):
    bad = [
        (i, line.rstrip())
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if _CALL in line
        and not line.lstrip().startswith("#")
        and not any(g in line for g in GUARDS)
    ]
    assert not bad, (
        f"{path.relative_to(REPO)}: bare np.trapezoid dies on the pinned numpy<2. "
        f"Use the shim `_trapz = getattr(np, 'trapezoid', None) or np.trapz`. Lines: {bad}"
    )
    eager = [(i, line.rstrip())
             for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
             if _EAGER.search(line) and not line.lstrip().startswith("#")]
    assert not eager, (
        f"{path.relative_to(REPO)}: eager default -- np.trapz is evaluated before the lookup. "
        f"Use `getattr(np, 'trapezoid', None) or np.trapz`. Lines: {eager}"
    )
