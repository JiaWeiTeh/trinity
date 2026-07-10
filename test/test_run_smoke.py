"""
End-to-end smoke test for the ``run.py`` CLI entry point.

This is an integration test that exercises the full read-param → integrate →
write-outputs pipeline in about a minute on a typical development machine.
Heavier than the unit tests but catches install-time regressions that
unit tests cannot: missing bundled defaults, broken imports, scipy /
numpy API drift, path-resolution bugs, etc.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_FINAL_GOLDENS = {
    # Captured 2026-07-10 on Python 3.9.6, numpy 1.26.4, scipy 1.13.1,
    # astropy 6.0.1, pandas 2.3.3, matplotlib 3.9.4, pytest 8.4.2.
    "R2": 0.2857315185200479,
    "v2": 44.73918438203256,
    "Eb": 778236.3470566473,
}


def test_quickstart_completes_cleanly(tmp_path):
    """``python run.py <fast param>`` exits 0 and writes the expected outputs.

    Uses ``stop_t = 1e-4`` Myr so the run stays short while writing enough
    snapshots to enter Phase 1 and exercise the bubble-structure code path
    (where past regressions have lived) without running for minutes.
    ``stop_t_diss`` is left at its default. Runs from a fresh CWD
    (``tmp_path``) — outputs land under ``tmp_path/outputs/smoke/`` and
    are cleaned by pytest.
    """
    param = tmp_path / "smoke.param"
    param.write_text(
        "mCloud      1e5\n" "sfe         0.3\n" "stop_t      1e-4\n" "model_name  smoke\n"
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
    rows = [
        json.loads(line)
        for line in (run_dir / "dictionary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 2, (
        f"dictionary.jsonl has only {len(rows)} snapshot lines; " f"integrator likely never ran"
    )

    final = rows[-1]
    for key, expected in _FINAL_GOLDENS.items():
        value = final.get(key)
        assert isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        assert value == pytest.approx(expected, rel=1e-6)
