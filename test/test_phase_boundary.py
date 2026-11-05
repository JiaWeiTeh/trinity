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

# The stored (cool_beta, cool_delta) of the first two implicit rows live in
# test/goldens/phase_boundary_implicit_rows.json; `pytest --regen-goldens` rewrites them.
#
# Matched test_betadelta_hybr_stress._GOLDEN[:2] for the shortened stop_t=0.004 run
# until 2026-11-04; that stress golden was not re-checked at the re-baseline below.
# Verified 2026-08-14 that the two runs agree to all printed digits on both rows,
# so stop_t does not perturb the trajectory ahead of the truncation.
#
# Re-baselined 2026-08-14 for C3c (`c43a50e`).  cool_alpha = t*v2/R2 is carried into
# phase 1b FROM the phase-1a exit state, and C3c moves that exit state: the
# capped-Strömgren `P_HII` was `Pb` un-ramped and won `max(press_bubble, P_HII)`, so
# the dt_switchon ramp never reached the momentum equation; C3c returns exactly 0.0
# while confined, so it does.  Before/after table + reproduce commands (the G3.4
# condition on D4's re-baselining authority):
# docs/dev/phii-identity/data/g34_golden_rebaseline.csv.  Was (0.888197, -0.046294).
#
# Earlier re-baseline 2026-08-05 with the phase-1a segment schedule (was
# (0.759260, -0.035387) twice, at t=0.00341/0.00381): see
# docs/dev/phase1a-init/PLAN.md §8, last block.
#
# Re-baselined 2026-09-06 under D4's authority, for THREE changes landing together
# (before/after table: docs/dev/phii-identity/shell-structure-test/golden_three_way.csv):
#   W32  the shell solver's ionised-loop termination flags were read with any() over the
#        whole integrated slice instead of AT the termination index, so `has_neutral` came
#        back False on rows that have a neutral rind and the neutral integration was
#        SKIPPED (329/895 archived rows). Restoring it saturates shell_fAbsorbedNeu at 1.0
#        and multiplies tau_kappa_IR, hence F_rad (+69.3% on the first implicit row here) --
#        the trajectory moves through RADIATION PRESSURE, not through P_HII, which is
#        exactly 0.0 on these confined-branch rows. This is the whole of the shift.
#   W52  adaptive slice sizing + terminal refinement in the shell march. Shifts cool_delta
#        by 0.002x tolerance here; it matters far more elsewhere (the shipped grid was
#        >10% wrong in tau_kappa_IR on 194/895 rows).
#   W63  TShell_ion 1e4 -> 8000 K with caseB_alpha scaled to match. Invisible here
#        (0.002x tol): P_HII is 0 on this config's rows either way.
# cool_beta was inside tolerance throughout (0.06x tol against the old golden) and moved
# only in the sixth decimal; cool_delta is what broke the gate, at 1.63x tolerance.
# Was (0.878396, -0.038973).
#
# Re-baselined 2026-11-04 for `eded76a5` (2026-10-19): the implicit phase now starts
# 0.05 kyr past TFINAL_ENERGY_PHASE and caps each segment at 0.02 t/|delta|, so the
# first two implicit rows sit at t = 0.00305 and 0.00311 Myr (were 0.00350 and
# 0.00390) and carry the values solved there. At matched t = 0.004 Myr the runs agree
# closely (main `c126ba10` against `d544aa62`): cool_beta 0.8559 vs 0.8539, cool_delta
# -0.1289 vs -0.1187, R2 -4.6e-5 and v2 -4.6e-4 relative. This test failed on every
# push from `eded76a5` to `c126ba10`. Was (0.878279, -0.042223) on both rows.
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


def test_default_run_crosses_energy_to_implicit_boundary(tmp_path, golden):
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

    expected = golden(
        "phase_boundary_implicit_rows",
        [[row["cool_beta"], row["cool_delta"]] for row in implicit[:2]],
    )
    for (golden_beta, golden_delta), row in zip(expected, implicit):
        assert row["cool_beta"] == pytest.approx(golden_beta, abs=_TOL)
        assert row["cool_delta"] == pytest.approx(golden_delta, abs=_TOL)
