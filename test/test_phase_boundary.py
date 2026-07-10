"""Default-suite phase-boundary golden for the ``run.py`` pipeline."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PARAM = (
    "mCloud      1e5\n"
    "sfe         0.3\n"
    "stop_t      0.004\n"
    "betadelta_solver  hybr\n"
    "model_name  phaseboundary\n"
    "log_level   WARNING\n"
)

# Matches test_betadelta_hybr_stress._GOLDEN[:2] for the shortened stop_t=0.004 run.
_GOLDEN = [
    (0.759260, -0.035387),  # t=0.00341 Myr
    (0.759260, -0.035387),  # t=0.00381 Myr
]
_TOL = 2e-3


def _run(cwd: Path):
    """One full ``run.py`` in ``cwd``; returns (CompletedProcess, run_dir)."""
    param = cwd / "phase_boundary.param"
    param.write_text(_PARAM)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result, cwd / "outputs" / "phaseboundary"


def _snapshot_rows(run_dir: Path):
    return [
        json.loads(line)
        for line in (run_dir / "dictionary.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _deduped_phases(rows):
    phases = []
    for row in rows:
        phase = row["current_phase"]
        if not phases or phases[-1] != phase:
            phases.append(phase)
    return phases


def _implicit_rows(rows):
    implicit = []
    for row in rows:
        res = row.get("betadelta_total_residual")
        if res is None or (isinstance(res, float) and math.isnan(res)):
            continue
        implicit.append(row)
    return implicit


def test_default_run_crosses_energy_to_implicit_boundary(tmp_path):
    result, run_dir = _run(tmp_path)
    assert (
        result.returncode == 0
    ), f"run.py exited {result.returncode}\n---stderr (tail)---\n{result.stderr[-2000:]}"

    rows = _snapshot_rows(run_dir)
    assert _deduped_phases(rows) == ["energy", "implicit"]

    implicit = _implicit_rows(rows)
    assert len(implicit) >= 2
    for row in implicit:
        assert row["betadelta_converged"] is True
        assert math.isfinite(row["cool_beta"])
        assert math.isfinite(row["cool_delta"])

    for (golden_beta, golden_delta), row in zip(_GOLDEN, implicit):
        assert row["cool_beta"] == pytest.approx(golden_beta, abs=_TOL)
        assert row["cool_delta"] == pytest.approx(golden_delta, abs=_TOL)
