"""
Opt-in stress test for the nondeterministic bubble-solver failure.

This is the *statistical* gate for the LSODA bubble-structure crash (the
intermittent ``BubbleSolverError`` / ``MonotonicError`` / cooling-table
``ValueError`` documented in ``docs/dev/bubble-integrator-robustness.md``).
The failure is floating-point / thread / scipy-version sensitive: the *same*
inputs pass on one run and crash on the next, so a single run proves nothing.
This test runs the failing scenario ``N`` times and asserts that none crash.

Why a separate, opt-in test (``@pytest.mark.stress``, deselected by default):
running the smoke scenario N times is slow (~30 s/run). It must not slow or
flake the default ``pytest test/`` suite (which CI runs), so it is excluded by
the ``-m 'not stress'`` default in ``pyproject.toml``. Run it explicitly with::

    pytest -m stress
    TRINITY_STRESS_N=50 pytest -m stress      # heavier sweep

Measured baseline (negative control — proves this harness *detects* the bug):

    * CI: ~100% red on py3.11/3.12 (scipy >= 1.16) on the unfixed code
      (commits #657-#659 all failed the smoke scenario).
    * Local A/B (12 runs each, default threads): the first-red commit #657
      failed 1/12; the prior-green commit #656 failed 0/12. The local rate is
      far below CI's (this multi-core runner is more robust than GitHub's), so
      a clean local sweep is necessary but not sufficient -- CI is the gate.

A single green run therefore proves nothing; this test exists to drive the
rate to 0 across many runs. CI's py3.11/3.12 matrix is the ultimate gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Number of repetitions. Overridable so CI/nightly can run a heavier sweep
# without editing the test. Default is a compromise between catching the flake
# and keeping a manual ``pytest -m stress`` invocation tractable.
_DEFAULT_N = 15
_STRESS_N = int(os.environ.get("TRINITY_STRESS_N", _DEFAULT_N))

# Failure signatures we classify (all are faces of the same LSODA failure).
_SIGNATURES = ("BubbleSolverError", "MonotonicError", "out of bounds")


def _write_smoke_param(path: Path) -> None:
    """Same fast scenario as test_run_smoke (stop_t=1e-4 -> ~10 snapshots)."""
    path.write_text(
        "mCloud      1e5\n"
        "sfe         0.3\n"
        "stop_t      1e-4\n"
        "model_name  smoke\n"
    )


@pytest.mark.stress
def test_smoke_no_bubble_solver_failures(tmp_path):
    """Run the smoke scenario ``TRINITY_STRESS_N`` times; assert 0 crashes.

    Each repetition is a fresh ``run.py`` subprocess in its own cwd (matching
    test_run_smoke's isolation). A non-zero exit is a failure; we capture and
    classify stderr so a failure report names the LSODA mode that fired.
    """
    param = tmp_path / "smoke.param"
    _write_smoke_param(param)

    failures = []  # (run_index, returncode, classified_signature, stderr_tail)
    for i in range(_STRESS_N):
        run_cwd = tmp_path / f"run_{i}"
        run_cwd.mkdir()
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
            cwd=run_cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            sig = next((s for s in _SIGNATURES if s in result.stderr), "other")
            failures.append((i, result.returncode, sig, result.stderr[-1500:]))

    if failures:
        sig_counts = {}
        for _, _, sig, _ in failures:
            sig_counts[sig] = sig_counts.get(sig, 0) + 1
        first = failures[0]
        pytest.fail(
            f"{len(failures)}/{_STRESS_N} smoke runs crashed "
            f"(signatures: {sig_counts}).\n"
            f"First failure: run {first[0]} exited {first[1]} ({first[2]}).\n"
            f"---stderr (tail)---\n{first[3]}"
        )
