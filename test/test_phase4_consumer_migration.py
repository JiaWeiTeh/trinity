"""
Tests for Phase 4 of the metadata-source-of-truth migration:
the two plot-script consumers that previously parsed
``simulationEnd.txt`` directly are now routed through
``read_simulation_end()`` (which is itself JSON-first since Phase 2).

These tests don't exercise the matplotlib pipeline — they call the
small helper functions in isolation against synthetic v3 / legacy v1
fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trinity._input.dictionary import DescribedDict, DescribedItem
from trinity._output.run_constants import METADATA_FILENAME


@pytest.fixture
def disable_crash_handlers(monkeypatch):
    """Stop DescribedDict from registering atexit/signal handlers."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _build_v3_run(tmp_path: Path, *, exit_code: int = 1,
                  detail: str = "Stopping time reached") -> Path:
    """Synth a v3-format run directory the consumer scripts can read.

    Mirrors the simpler fixture in ``test_show_run.py``.
    """
    from trinity._output.simulation_end import write_simulation_end
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(tmp_path))
    for k, v in [
        ("model_name", "phase4_test"), ("mCloud", 1.0e6),
        ("sfe", 0.01), ("ZCloud", 1.0), ("dens_profile", "densPL"),
        ("densPL_alpha", 0.0), ("nCore", 1.0e3), ("nISM", 1.0),
        ("rCore", 0.01), ("rCloud", 20.0), ("nEdge", 2.0), ("tSF", 0),
        ("mu_convert", 1.18e-57),
    ]:
        d[k] = DescribedItem(v)
    d["t_now"] = DescribedItem(0.3); d["R2"] = DescribedItem(2.51)
    d["v2"] = DescribedItem(2.45e-3)
    d["current_phase"] = DescribedItem("momentum")
    d.save_snapshot(); d.flush()
    d["SimulationEndReason"] = DescribedItem(detail)
    d["SimulationEndCode"] = DescribedItem(exit_code)
    write_simulation_end(d, str(tmp_path))
    return tmp_path


def _build_legacy_v1_run(tmp_path: Path) -> Path:
    """Synth a legacy run: v1 metadata.json (no termination block) +
    text-only simulationEnd.txt with the new Outcome/Detail schema."""
    (tmp_path / METADATA_FILENAME).write_text(json.dumps({
        "_metadata_version": 1,
        "model_name": "phase4_legacy",
    }))
    (tmp_path / "simulationEnd.txt").write_text(
        "Timestamp: 2026-01-01 00:00:00\n"
        "Model: phase4_legacy\n"
        "Outcome: stopping_time\n"
        "Detail: Legacy text-only reason\n"
        "Exit Code: 1\n"
    )
    (tmp_path / "dictionary.jsonl").write_text(
        json.dumps({"t_now": 0.0, "R2": 0.1}) + "\n"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# paper_rcloud_smoothing: fixed end_info.get("reason") (was always None)
# ---------------------------------------------------------------------------

class TestRcloudSmoothingMigration:
    """``paper_rcloud_smoothing`` now reads ``output.termination`` for
    v3+ runs and falls back to ``read_simulation_end()`` for legacy.

    Previously it called ``end_info.get("reason")`` — silent bug, the
    actual key is ``detail``, so ``end_reason`` was None for every run.
    """

    def test_end_reason_resolved_from_v3_termination(
        self, tmp_path, disable_crash_handlers,
    ):
        """v3 run: ``end_reason`` populates from ``termination.detail``."""
        _build_v3_run(tmp_path, detail="Stopping time reached")
        # Emulate the field-name resolution paper_rcloud_smoothing does
        from trinity._output.trinity_reader import TrinityOutput
        from trinity._output.simulation_end import read_simulation_end
        output = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        end_info = output.termination or read_simulation_end(str(tmp_path))
        assert end_info is not None
        assert end_info.get("exit_code") == 1
        # The migration fix: use 'detail', not 'reason'
        assert end_info.get("detail") == "Stopping time reached"
        # Sanity: the old buggy key returns nothing
        assert end_info.get("reason") is None

    def test_end_reason_resolved_from_legacy_text(
        self, tmp_path, disable_crash_handlers,
    ):
        """Legacy run: the text-parse path still feeds the same key."""
        _build_legacy_v1_run(tmp_path)
        from trinity._output.trinity_reader import TrinityOutput
        from trinity._output.simulation_end import read_simulation_end
        output = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        # v1 → output.termination is None → falls back to text-parse
        end_info = output.termination or read_simulation_end(str(tmp_path))
        assert end_info is not None
        assert end_info["exit_code"] == 1
        assert end_info["detail"] == "Legacy text-only reason"
