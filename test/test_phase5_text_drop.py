"""
Phase 5 of the metadata-source-of-truth migration: stop writing the
three legacy text artefacts (``simulationEnd.txt``,
``termination_debug.txt``, ``<model>_summary.txt``).

These tests exercise the writer side (nothing produces the text files
anymore) and the reader side (the back-compat text-parse paths now
emit ``DeprecationWarning``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from src._input.dictionary import DescribedDict, DescribedItem
from src._output.run_constants import METADATA_FILENAME


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def disable_crash_handlers(monkeypatch):
    """Stop DescribedDict from registering atexit/signal handlers."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _make_minimal_v4_run(tmp_path: Path) -> DescribedDict:
    """Synth a small but-complete v4 run by driving the real writers.

    Calls ``write_simulation_end`` + ``write_termination_debug_report``
    so we hit every text-writing site that Phase 5 is meant to drop.
    """
    from src._output.simulation_end import (
        write_simulation_end, write_termination_debug_report,
    )
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(tmp_path))
    for k, v in [
        ("model_name", "phase5_test"), ("mCloud", 1.0e6),
        ("sfe", 0.01), ("ZCloud", 1.0), ("dens_profile", "densPL"),
        ("densPL_alpha", 0.0), ("nCore", 1.0e3), ("nISM", 1.0),
        ("rCore", 0.01), ("rCloud", 20.0), ("nEdge", 2.0), ("tSF", 0),
        ("mu_convert", 1.18e-57),
    ]:
        d[k] = DescribedItem(v)
    d["t_now"] = DescribedItem(0.2); d["R2"] = DescribedItem(1.5)
    d.save_snapshot()
    d["t_now"] = DescribedItem(0.3); d["R2"] = DescribedItem(2.0)
    d.save_snapshot()
    d.flush()
    d["SimulationEndReason"] = DescribedItem("Stopping time reached")
    d["SimulationEndCode"] = DescribedItem(1)
    write_simulation_end(d, str(tmp_path))
    write_termination_debug_report(str(tmp_path), reason="Stopping time reached")
    return d


# ---------------------------------------------------------------------------
# No legacy text artefacts are written
# ---------------------------------------------------------------------------

class TestNoLegacyTextWrites:
    """A full Phase-5 run must produce zero of the three legacy text
    files.  Only metadata.json + dictionary.jsonl land on disk."""

    def test_no_simulation_end_txt(self, tmp_path, disable_crash_handlers):
        _make_minimal_v4_run(tmp_path)
        assert not (tmp_path / "simulationEnd.txt").exists()

    def test_no_termination_debug_txt(self, tmp_path, disable_crash_handlers):
        _make_minimal_v4_run(tmp_path)
        assert not (tmp_path / "termination_debug.txt").exists()

    def test_no_summary_txt(self, tmp_path, disable_crash_handlers):
        """``read_param`` no longer writes ``<model>_summary.txt``.

        We can't drive ``read_param`` from a synthetic dict (it needs a
        real .param file on disk), so this test just asserts the
        code path is gone by inspecting the public function's
        signature: the ``write_summary`` kwarg has been removed.
        """
        from inspect import signature
        from src._input.read_param import read_param
        params = signature(read_param).parameters
        assert "write_summary" not in params, (
            "Phase 5 should have removed the write_summary kwarg "
            "from read_param; the text writer was its only consumer."
        )

    def test_only_v4_artefacts_remain(self, tmp_path, disable_crash_handlers):
        """The Phase 5 directory layout: metadata.json + dictionary.jsonl
        (logs are written by the logging subsystem in production, not
        by the unit-test path)."""
        _make_minimal_v4_run(tmp_path)
        names = {p.name for p in tmp_path.iterdir() if p.is_file()}
        # The two canonical artefacts are present
        assert "metadata.json" in names
        assert "dictionary.jsonl" in names
        # None of the three legacy text artefacts are
        assert "simulationEnd.txt" not in names
        assert "termination_debug.txt" not in names
        assert not any(n.endswith("_summary.txt") for n in names)


# ---------------------------------------------------------------------------
# Deprecation warnings on the back-compat reader paths
# ---------------------------------------------------------------------------

class TestDeprecationWarnings:
    """Reader fallbacks for legacy text files now emit
    ``DeprecationWarning``; removed in Phase 6."""

    def test_read_simulation_end_warns_on_text_fallback(self, tmp_path):
        """``read_simulation_end`` falls back to text-parse when no
        metadata.json[termination] is present; this fallback path
        warns."""
        from src._output.simulation_end import read_simulation_end
        (tmp_path / "simulationEnd.txt").write_text(
            "Outcome: stopping_time\nDetail: bye\nExit Code: 1\nModel: legacy\n"
        )
        with pytest.warns(DeprecationWarning, match="simulationEnd.txt"):
            result = read_simulation_end(str(tmp_path))
        assert result["exit_code"] == 1
        assert result["detail"] == "bye"

    def test_cloudy_parse_summary_txt_warns(self):
        """``_parse_summary_txt`` warns on call (it only runs for
        legacy v1 fixtures now)."""
        from src._output.cloudy.run_loader import _parse_summary_txt
        with pytest.warns(DeprecationWarning, match="<model>_summary.txt"):
            out = _parse_summary_txt("ZCloud 1.0\nmCloud 1e6\n")
        assert out["ZCloud"] == 1.0
        assert out["mCloud"] == 1e6

    def test_cloudy_parse_simulation_end_warns(self):
        """``_parse_simulation_end`` warns on call (legacy text)."""
        from src._output.cloudy.run_loader import _parse_simulation_end
        with pytest.warns(DeprecationWarning, match="simulationEnd.txt"):
            out = _parse_simulation_end(
                "Model: legacy\nOutcome: stopping_time\nExit Code: 1\n"
            )
        assert out["exit_code"] == 1
        assert out["outcome"] == "stopping_time"

    def test_cloudy_load_run_v4_emits_no_warnings(
        self, tmp_path, disable_crash_handlers,
    ):
        """A clean v4 run drives the JSON-only path and emits zero
        DeprecationWarnings (no text files ever read)."""
        from src._output.cloudy.run_loader import load_run
        _make_minimal_v4_run(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            bundle = load_run(tmp_path)
        assert bundle.model_name == "phase5_test"
        # The summary IS the metadata (minus reserved keys)
        assert bundle.summary["ZCloud"] == 1.0
        assert bundle.summary["dens_profile"] == "densPL"
        assert "termination" not in bundle.summary  # reserved key stripped
        # end_state comes from metadata.json[termination]
        assert bundle.end_state["exit_code"] == 1
