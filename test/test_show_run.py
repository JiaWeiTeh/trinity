"""
Tests for ``src/_output/show_run.py`` — the human-readable run summary CLI.

Covers:
- ``format_run_summary()`` on synthetic v3 runs (full Phase 1+2 path)
- ``format_run_summary()`` on legacy v1 runs with only simulationEnd.txt
  (legacy text-parse fallback)
- ``format_run_summary()`` when a run terminated unsuccessfully
- ``format_run_summary()`` when no termination block exists (aborted)
- ``--json`` flag prints metadata.json verbatim
- ``--quiet`` flag returns 0 on success, non-zero otherwise
- error handling when the run directory doesn't exist
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src._input.dictionary import DescribedDict, DescribedItem
from src._output.run_constants import METADATA_FILENAME, METADATA_VERSION
from src._output.show_run import format_run_summary, main


@pytest.fixture
def disable_crash_handlers(monkeypatch):
    """Stop DescribedDict from registering atexit/signal handlers."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _write_v3_run(tmp_path: Path, *, exit_code: int = 1,
                  detail: str = "Stopping time reached") -> Path:
    """Produce a complete v3-format run directory the CLI can read.

    The ``outcome`` field of the termination block is set automatically
    by ``write_simulation_end`` from ``SimulationEndCode.from_code(exit_code)``
    — callers pick an ``exit_code`` whose mapped outcome they want to
    test.
    """
    from src._output.simulation_end import write_simulation_end

    d = DescribedDict()
    d["path2output"] = DescribedItem(str(tmp_path))
    # Minimal run-constants
    for k, v in [
        ("model_name", "show_run_test"), ("mCloud", 1.0e6),
        ("sfe", 0.01), ("ZCloud", 1.0), ("dens_profile", "densPL"),
        ("densPL_alpha", 0.0), ("nCore", 1.0e3), ("nISM", 1.0),
        ("rCore", 0.01), ("rCloud", 20.0), ("nEdge", 2.0), ("tSF", 0),
        ("mu_convert", 1.18e-57),
    ]:
        d[k] = DescribedItem(v)
    # Varying scalars
    d["t_now"] = DescribedItem(0.0)
    d["R2"] = DescribedItem(0.1)
    d["v2"] = DescribedItem(100.0)
    d["current_phase"] = DescribedItem("energy")
    d["isCollapse"] = DescribedItem(False)
    d.save_snapshot()
    # End-of-run scalars
    d["t_now"] = DescribedItem(0.300)
    d["R2"] = DescribedItem(2.51)
    d["v2"] = DescribedItem(2.45e-3)
    d["current_phase"] = DescribedItem("momentum")
    d["shell_mass"] = DescribedItem(1143.84)
    d["shell_nMax"] = DescribedItem(3.31e55)
    d["Eb"] = DescribedItem(0.0)
    d["Pb"] = DescribedItem(4.91)
    d.save_snapshot()
    d.flush()

    # Termination
    d["SimulationEndReason"] = DescribedItem(detail)
    d["SimulationEndCode"] = DescribedItem(exit_code)
    write_simulation_end(d, str(tmp_path))
    return tmp_path


def _write_legacy_v1_run(tmp_path: Path) -> Path:
    """Produce a v1-style run (no termination block in metadata,
    only simulationEnd.txt). Mimics pre-Phase-2 output layout."""
    v1_md = {
        "_metadata_version": 1,
        "model_name": "legacy_show_run",
        "mCloud": 4.0e3,
        "nCore": 5.0e2,
        "nISM": 0.1,
        "rCore": 0.01,
        "rCloud": 3.80,
        "dens_profile": "densPL",
        "densPL_alpha": 0.0,
        "nEdge": 0.1,
        "tSF": 0,
        "initial_cloud_r_arr": [0.001, 0.1, 1.0, 3.8],
        "initial_cloud_n_arr": [500.0, 500.0, 500.0, 0.1],
        "initial_cloud_m_arr": [0.0, 1.0, 100.0, 3966.0],
    }
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(v1_md))
    text = (
        "==================================================\n"
        "TRINITY Simulation End Report\n"
        "==================================================\n"
        "Timestamp: 2026-01-01 00:00:00\n"
        "Model: legacy_show_run\n"
        "\n"
        "Outcome: stopping_time\n"
        "Detail: Legacy text-parsed reason\n"
        "Exit Code: 1\n"
    )
    (tmp_path / "simulationEnd.txt").write_text(text)
    (tmp_path / "dictionary.jsonl").write_text(
        json.dumps({"t_now": 0.0, "R2": 0.1, "v2": 100.0}) + "\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# format_run_summary()
# ---------------------------------------------------------------------------

class TestFormatRunSummary:

    def test_v3_run_full_summary(self, tmp_path, disable_crash_handlers):
        _write_v3_run(tmp_path)
        out = format_run_summary(tmp_path)
        # Header
        assert "TRINITY run: show_run_test" in out
        assert "SUCCESS" in out
        assert "stopping_time" in out
        assert "Stopping time reached" in out
        # Cloud section uses run-constants
        assert "mCloud" in out and "1.00e+06" in out
        assert "sfe" in out
        assert "mCluster" in out  # derived from mCloud * sfe
        # Final state
        assert "Final state" in out
        assert "[t = 0.300 Myr]" in out
        assert "2.510 pc" in out         # R2
        assert "km/s" in out             # v2 unit-converted
        assert "shell_mass" in out
        assert "phase" in out and "momentum" in out
        assert "collapsed" in out and "no" in out

    def test_legacy_v1_run_falls_back_to_text(self, tmp_path):
        _write_legacy_v1_run(tmp_path)
        out = format_run_summary(tmp_path)
        # Status pulled from text simulationEnd.txt
        assert "SUCCESS" in out
        assert "stopping_time" in out
        assert "Legacy text-parsed reason" in out
        # Cloud section reads from v1 metadata (top-level scalars)
        assert "mCloud" in out and "4.00e+03" in out
        # No final_state block in v1 — placeholder text shows
        assert "no final_state block" in out.lower()

    def test_unsuccessful_run_shows_error_glyph(
        self, tmp_path, disable_crash_handlers,
    ):
        # Exit code 20 → outcome 'error_numerical' (outside [0, 9] → ✗ ERROR)
        _write_v3_run(tmp_path, exit_code=20, detail="LSODA gave up")
        out = format_run_summary(tmp_path)
        assert "ERROR" in out
        assert "error_numerical" in out
        assert "LSODA gave up" in out
        # Glyph indicates failure
        assert "✗" in out

    def test_missing_termination_shows_unknown(
        self, tmp_path, disable_crash_handlers,
    ):
        # Build a v2-style run (run-constants only, no termination block)
        v2_md = {
            "_metadata_version": 2,
            "model_name": "aborted_run",
            "mCloud": 1.0e6, "nCore": 1.0e3, "nISM": 1.0, "rCore": 0.5,
            "rCloud": 20.0, "dens_profile": "densPL", "densPL_alpha": 0.0,
            "mu_convert": 1.18e-57, "nEdge": 2.0,
        }
        (tmp_path / METADATA_FILENAME).write_text(json.dumps(v2_md))
        (tmp_path / "dictionary.jsonl").write_text(
            json.dumps({"t_now": 0.0, "R2": 0.1}) + "\n"
        )
        out = format_run_summary(tmp_path)
        assert "TRINITY run: aborted_run" in out
        assert "UNKNOWN" in out
        assert "?" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:

    def test_pretty_print_default(self, tmp_path, capsys,
                                  disable_crash_handlers):
        _write_v3_run(tmp_path)
        rc = main([str(tmp_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "TRINITY run: show_run_test" in captured.out

    def test_json_flag_passes_through_metadata_file(
        self, tmp_path, capsys, disable_crash_handlers,
    ):
        _write_v3_run(tmp_path)
        # Drop noise from write_simulation_end (it prints a status line)
        capsys.readouterr()
        rc = main(["--json", str(tmp_path)])
        assert rc == 0
        captured = capsys.readouterr()
        # --json output is parseable JSON
        parsed = json.loads(captured.out)
        assert parsed["_metadata_version"] == METADATA_VERSION
        assert parsed["model_name"] == "show_run_test"
        assert "termination" in parsed
        assert "final_state" in parsed

    def test_quiet_returns_0_on_success(
        self, tmp_path, capsys, disable_crash_handlers,
    ):
        _write_v3_run(tmp_path, exit_code=1)  # 1 ∈ [0, 9] → success
        rc = main(["--quiet", str(tmp_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out
        # --quiet prints ONLY the status line (no Detail / Cloud / Final)
        assert "Detail" not in captured.out
        assert "Cloud" not in captured.out

    def test_quiet_returns_nonzero_on_error(
        self, tmp_path, capsys, disable_crash_handlers,
    ):
        _write_v3_run(tmp_path, exit_code=42)
        rc = main(["--quiet", str(tmp_path)])
        assert rc != 0
        assert 1 <= rc <= 9  # capped at 9 per the contract
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_directory_not_found_returns_1(self, capsys, tmp_path):
        rc = main([str(tmp_path / "definitely_does_not_exist")])
        assert rc == 1
        captured = capsys.readouterr()
        assert "not a directory" in captured.err

    def test_json_flag_with_missing_metadata_returns_1(
        self, tmp_path, capsys,
    ):
        # Empty run dir — no metadata.json
        rc = main(["--json", str(tmp_path)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err
