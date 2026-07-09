"""
Guard: phase helpers that exist as verbatim copies must stay in sync.

``compute_max_dex_change`` is duplicated in the phase 1b / 1c / 2 runners
(docs/dev/roadmap/solver-audit.md finding F5; consolidation is
docs/dev/roadmap/REORG.md item R1). Until consolidated, a fix applied to one
copy and not the others is a latent phase-dependent bug — this test fails the
moment the copies' logic diverges (docstrings excluded).

The force-balance functions (``compute_forces_pure`` in 1b/1c,
``compute_forces_momentum_pure`` in 2) are NOT checked here: they have
already diverged and the divergences are unclassified (solver-audit.md F5) —
do not add them until each difference is either documented as intentional
phase physics or consolidated away.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

COPIES = [
    "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "trinity/phase1c_transition/run_transition_phase.py",
    "trinity/phase2_momentum/run_momentum_phase.py",
]


def _logic_dump(path, name):
    """AST dump of the named function's body with its docstring stripped."""
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # docstrings may differ; only logic must match
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"{name} not found in {path}")


def test_compute_max_dex_change_copies_identical():
    dumps = {p: _logic_dump(p, "compute_max_dex_change") for p in COPIES}
    ref = dumps[COPIES[0]]
    for p in COPIES[1:]:
        assert dumps[p] == ref, (
            f"compute_max_dex_change in {p} has diverged from {COPIES[0]}: "
            "sync the copies or consolidate them (docs/dev/roadmap/REORG.md R1) — "
            "a one-copy fix is a phase-dependent bug."
        )
