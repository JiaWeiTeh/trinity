"""Behavioural guard for phase 1a: a segment that drives ``Eb`` to collapse must
**terminate**, not grind (`docs/dev/phase1a-stiffness/PLAN.md`, gate P4).

This asserts the behaviour, deliberately not the identity of the solver or the
event: any future scheme that ends such a segment cleanly keeps this test green,
and any change that lets the integrator crawl again turns it red.

The collapse regime is reached by disabling the ``dt_switchon`` R1 ramp, which
is exactly what the committed harness does (it forwards ``t=None``, changing
nothing else). Without the in-band energy-collapse guard this configuration does
not finish at all: measured, one segment needs ~1e9 explicit steps, about seven
days (docs/dev/phase1a-stiffness/data/stall_anatomy.csv). With it, the run ends
in ~22 s on the pre-existing ENERGY_COLLAPSED fate.

Run in a subprocess on purpose — trinity leaks module-level global state, so an
in-process full run would contaminate the rest of the suite.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "docs" / "dev" / "phase1a-stiffness" / "harness" / "seg_stepcount_runner.py"

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`: absent in a
# fresh clone and in CI. The single test here drives that runner, so skip the module.
if not RUNNER.is_file():
    pytest.skip(
        "docs/dev is untracked (local-only); seg_stepcount_runner.py unavailable",
        allow_module_level=True,
    )

# Measured 22 s with the guard in place. The budget is deliberately an order of
# magnitude looser: the failure this pins is "does not terminate at all", not a
# few seconds of drift on a contended container.
WALL_BUDGET_S = 300


def test_collapsing_phase1a_segment_terminates_instead_of_grinding(tmp_path):
    assert RUNNER.is_file(), f"harness missing: {RUNNER}"

    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--config", "f1edge_hidens",
         "--stop-t", "0.02", "--ablate-ramp", "--workdir", str(tmp_path)],
        capture_output=True, text=True, timeout=WALL_BUDGET_S,
    )
    wall = time.monotonic() - start

    assert proc.returncode == 0, (
        f"run exited {proc.returncode}\n---stderr (tail)---\n{proc.stderr[-2000:]}"
    )

    metadata = tmp_path / "outputs" / "screen" / "metadata.json"
    assert metadata.is_file(), "run wrote no metadata.json"
    termination = json.loads(metadata.read_text()).get("termination") or {}

    # It must stop, and stop as the collapse phase 1a already knows how to name —
    # not as a new outcome, and not by exhausting a wall clock.
    assert termination.get("outcome") == "energy_collapsed", (
        f"expected an energy_collapsed fate, got {termination!r}"
    )
    assert wall < WALL_BUDGET_S
