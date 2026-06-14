"""Phase-3 Commit 4: end-to-end stress + regression guard for ``betadelta_solver='hybr'``.

Opt-in (``@pytest.mark.stress``, deselected by default via ``pyproject.toml``'s
``-m 'not stress'``): each test runs a full ``run.py`` pipeline to a ``stop_t``
past the energy->implicit boundary (~0.003 Myr), so the hybr root-finder
actually runs. Slow (~minutes/run); not in the default CI suite.

Run explicitly::

    pytest -m stress test/test_betadelta_hybr_stress.py
    TRINITY_STRESS_N=10 pytest -m stress test/test_betadelta_hybr_stress.py

These cover what the synthetic-landscape unit tests (``test_betadelta_hybr.py``)
cannot: the hybr solver driving the real bubble-structure physics through the
integrated pipeline. No-root never fires on a self-consistent trajectory (see
the 2x2 validation matrix in ``docs/dev/BETADELTA_HYBR_PLAN.md``), so these
assert the healthy path: no crashes, 100% convergence, dMdt>0, and a
(beta, delta) trajectory that matches a recorded golden within tolerance.

The golden uses a *loose* tolerance, not exact values: on the pinned numpy<2 /
scipy<2 stack the run is deterministic, but a scipy patch can shift the last
bits. The tolerance absorbs that while still catching a real solver regression
(which shifts beta by O(0.1+)). The exact solver logic is pinned separately by
``test_betadelta_hybr.py``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_N = 5
_STRESS_N = int(os.environ.get("TRINITY_STRESS_N", _DEFAULT_N))

# A run reaching a few implicit-phase segments (energy->implicit boundary is
# ~0.0029 Myr for this config). Same mCloud/sfe as the smoke / bubble-solver
# stress tests; stop_t raised past the boundary so the hybr solver runs.
_PARAM = (
    "mCloud      1e5\n"
    "sfe         0.3\n"
    "stop_t      0.008\n"
    "betadelta_solver  hybr\n"
    "model_name  hybrstress\n"
    "log_level   WARNING\n"
)

# Golden accepted (beta, delta) at the first implicit-phase segments, recorded
# on the pinned numpy<2 / scipy<2 stack. FILLED FROM A RECORDING RUN.
_GOLDEN: list = [
    (0.759260, -0.035387),  # t=0.00341 Myr
    (0.759260, -0.035387),  # t=0.00381
    (0.757865, -0.122600),  # t=0.00412
    (0.757865, -0.122600),  # t=0.00437
]
_TOL = 2e-3

_CRASH_SIGNATURES = (
    "BubbleSolverError",
    "MonotonicError",
    "NoPhysicalRoot",
    "out of bounds",
    "Traceback",
)


def _run(cwd: Path):
    """One full ``run.py`` in ``cwd``; returns (CompletedProcess, run_dir)."""
    param = cwd / "hybr.param"
    param.write_text(_PARAM)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result, cwd / "outputs" / "hybrstress"


def _implicit_rows(run_dir: Path):
    """Snapshots where the beta-delta solver actually ran (implicit phase)."""
    rows = []
    for line in (run_dir / "dictionary.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        res = r.get("betadelta_total_residual")
        if res is None or (isinstance(res, float) and math.isnan(res)):
            continue
        rows.append(r)
    return rows


@pytest.mark.stress
def test_hybr_endtoend_no_crashes(tmp_path):
    """Run the hybr pipeline ``TRINITY_STRESS_N`` times; assert 0 crashes."""
    failures = []
    for i in range(_STRESS_N):
        run_cwd = tmp_path / f"run_{i}"
        run_cwd.mkdir()
        result, _ = _run(run_cwd)
        if result.returncode != 0:
            sig = next((s for s in _CRASH_SIGNATURES if s in result.stderr), "other")
            failures.append((i, result.returncode, sig, result.stderr[-1500:]))
    if failures:
        counts: dict = {}
        for _, _, sig, _ in failures:
            counts[sig] = counts.get(sig, 0) + 1
        i, rc, sig, tail = failures[0]
        pytest.fail(
            f"{len(failures)}/{_STRESS_N} hybr runs crashed (signatures: {counts}).\n"
            f"First: run {i} exited {rc} ({sig}).\n---stderr (tail)---\n{tail}"
        )


@pytest.mark.stress
def test_hybr_implicit_converges_and_matches_golden(tmp_path):
    """One hybr run reaches the implicit phase, converges every segment with a
    physical dMdt, and reproduces the recorded (beta, delta) golden."""
    result, run_dir = _run(tmp_path)
    assert (
        result.returncode == 0
    ), f"run.py exited {result.returncode}\n---stderr (tail)---\n{result.stderr[-2000:]}"
    rows = _implicit_rows(run_dir)
    assert len(rows) >= len(_GOLDEN), (
        f"only {len(rows)} implicit-phase segments; expected >= {len(_GOLDEN)} "
        f"(did the run reach the energy->implicit boundary?)"
    )
    # Health: hybr converges every segment with a finite, positive dMdt.
    for r in rows:
        assert r["betadelta_converged"] is True, (
            f"segment t={r['t_now']:.5f} did not converge "
            f"(residual={r['betadelta_total_residual']:.2e})"
        )
        dm = r["bubble_dMdt"]
        assert math.isfinite(dm) and dm > 0, f"non-physical dMdt={dm} at t={r['t_now']:.5f}"
        assert math.isfinite(r["cool_beta"]) and math.isfinite(r["cool_delta"])
    # Regression: the accepted (beta, delta) trajectory has not drifted.
    for (gbeta, gdelta), r in zip(_GOLDEN, rows):
        assert r["cool_beta"] == pytest.approx(
            gbeta, abs=_TOL
        ), f"beta drift at t={r['t_now']:.5f}: {r['cool_beta']} vs golden {gbeta}"
        assert r["cool_delta"] == pytest.approx(
            gdelta, abs=_TOL
        ), f"delta drift at t={r['t_now']:.5f}: {r['cool_delta']} vs golden {gdelta}"
