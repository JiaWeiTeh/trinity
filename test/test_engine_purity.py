"""Engine-purity guard (Phase C invariant).

The installed ``trinity`` package is the engine; the plotting/personal
trees (``paper/``, ``paper/methods/figures/``, ``docs/dev/scratch/``) are downstream
consumers. The dependency must stay strictly one-way: everything may
import ``trinity``; ``trinity`` imports nothing downstream. Without this
guard the separation erodes the next time a plot helper is imported back
into the engine.
"""
from __future__ import annotations

import ast
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1] / "trinity"
FORBIDDEN_TOP_LEVEL = {"paper", "scratch", "figures"}


def _imported_top_level_modules(py: Path):
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # absolute imports only (relative ImportFrom stays inside trinity)
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def test_engine_imports_nothing_downstream():
    offenders = []
    for py in ENGINE_ROOT.rglob("*.py"):
        for top in _imported_top_level_modules(py):
            if top in FORBIDDEN_TOP_LEVEL:
                offenders.append(f"{py.relative_to(ENGINE_ROOT.parent)} imports '{top}'")
    assert not offenders, (
        "trinity/ (engine) must not import downstream plotting/personal code:\n  "
        + "\n  ".join(offenders)
    )
