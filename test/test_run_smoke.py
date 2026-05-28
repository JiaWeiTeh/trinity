"""
End-to-end smoke test for the ``run.py`` CLI entry point.

This is a slow integration test (~2.5 min on a typical machine) that
exercises the full read-param → integrate → write-outputs pipeline.
Heavier than the unit tests but catches install-time regressions that
unit tests cannot: missing bundled defaults, broken imports, scipy /
numpy API drift, path-resolution bugs, etc.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_completes_cleanly(tmp_path):
    """``python run.py <fast param>`` exits 0 and writes the expected outputs.

    Uses ``stop_t = 1e-4`` Myr so the run terminates after ~10 snapshots —
    enough to enter Phase 1 and exercise the bubble-structure code path
    (where past regressions have lived) without running for minutes.
    ``stop_t_diss`` is left at its default. Runs from a fresh CWD
    (``tmp_path``) — outputs land under ``tmp_path/outputs/smoke/`` and
    are cleaned by pytest.
    """
    param = tmp_path / "smoke.param"
    param.write_text(
        "mCloud      1e5\n"
        "sfe         0.3\n"
        "stop_t      1e-4\n"
        "model_name  smoke\n"
    )

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"run.py exited {result.returncode}\n"
        f"---stdout (tail)---\n{result.stdout[-4000:]}\n"
        f"---stderr (tail)---\n{result.stderr[-4000:]}"
    )

    run_dir = tmp_path / "outputs" / "smoke"
    # Only assert on artifacts written unconditionally during setup/integration.
    # summary.txt and simulationEnd.txt are emitted on "normal termination"
    # paths a very short stop_t does not reach.
    for fname in ("metadata.json", "dictionary.jsonl"):
        assert (run_dir / fname).exists(), (
            f"expected output {fname} missing from {run_dir}; "
            f"got: {sorted(p.name for p in run_dir.iterdir())}"
        )
    # dictionary.jsonl should be non-trivial — proves the integrator ran,
    # not just that setup wrote a placeholder.
    snapshots = (run_dir / "dictionary.jsonl").read_text().count("\n")
    assert snapshots >= 2, (
        f"dictionary.jsonl has only {snapshots} snapshot lines; "
        f"integrator likely never ran"
    )
