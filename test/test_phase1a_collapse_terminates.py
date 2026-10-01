"""Behavioural guard for phase 1a: a segment that drives ``Eb`` to collapse must
**terminate**, not grind (`docs/dev/phase1a-stiffness/PLAN.md`, gate P4).

This asserts the behaviour, deliberately not the identity of the solver or the
event: any future scheme that ends such a segment cleanly keeps this test green,
and any change that lets the integrator crawl again turns it red.

The collapse regime is reached by disabling the ``dt_switchon`` R1 ramp, which
is exactly what the committed harness does (it forwards ``t=None``, changing
nothing else). Without the in-band energy-collapse guard this configuration does
not finish at all: measured, one segment needs ~1e9 explicit steps, about seven
days (docs/dev/phase1a-stiffness/data/stall_anatomy.csv).

Since 2026-09-30 a phase-1a collapse is handed to momentum (via 1c, phase 1b
skipped) instead of ending the run as ENERGY_COLLAPSED
(docs/dev/transition/pdv-trigger/HIMASS_HANDOFF_PLAN.md). The test now pins the
handoff as well as termination. In this ablated configuration the handoff comes
at t ~ 0.3 yr, R2 ~ 1e-3 pc, and the transition and momentum solvers fail at
once (LSODA istate), so the run ends UNKNOWN in ~4 s; that fate is an artefact
of the ablation and is deliberately not pinned.

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

    # It must stop without exhausting a wall clock, and the collapse must have been
    # handed on rather than ended as ENERGY_COLLAPSED.
    assert termination.get("outcome") != "energy_collapsed", (
        f"phase-1a collapse ended the run instead of handing off: {termination!r}"
    )
    log = (tmp_path / "trinity.log").read_text(errors="ignore")
    assert "routing to momentum via 1c, phase 1b skipped" in log, "no phase-1a handoff logged"
    # Since 2026-10-01 a turnaround also hands off from 1a (whichever comes first).
    assert ("Implicit phase completed: energy_to_momentum" in log
            or "Implicit phase completed: velocity_sign_change" in log), "phase 1b did not report the handoff"
    assert wall < WALL_BUDGET_S
