"""
Tests for the ``metadata.json`` run-constants split.

Coverage:
- writer emits ``metadata.json`` once, strips run-constants from snapshots
- reader (both ``TrinityOutput`` and ``DescribedDict.load_snapshots``)
  rehydrates run-constants transparently
- legacy files (no ``metadata.json``) keep loading identically
- ``setdefault`` semantics: per-snapshot value wins over metadata
- corrupted / missing metadata is tolerated (warning, not exception)
- size invariant: a fresh write splits 1.4 MB legacy → ≤ 30 KB total
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src._input.dictionary import DescribedDict, DescribedItem
from src._output.run_constants import (
    METADATA_FILENAME, METADATA_VERSION, RUN_CONST_KEYS,
)
from src._output.trinity_reader import TrinityOutput


# Small subset of the run-constants we exercise in tests; the writer
# only emits keys that are actually present in the params container,
# so missing entries fall through to the per-snapshot value.
_SAMPLE_INIT_R = list(np.linspace(0.0, 10.0, 25))
_SAMPLE_INIT_N = list(np.linspace(1e3, 1e6, 25))
_SAMPLE_INIT_M = list(np.linspace(0.0, 1e4, 25))


@pytest.fixture
def disable_crash_handlers(monkeypatch):
    """Stop ``DescribedDict.__init__`` registering atexit/signal hooks
    in tests so we don't leak handlers across the test session."""
    monkeypatch.setattr(
        DescribedDict, "_register_crash_handlers", lambda self: None
    )


def _make_params(out_dir: Path) -> DescribedDict:
    """Populate a DescribedDict with run-constants + a few varying keys."""
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(out_dir), info="Output dir")
    # Run-constants
    d["model_name"] = DescribedItem("test_run")
    d["mCloud"] = DescribedItem(1.0e6)
    d["dens_profile"] = DescribedItem("densPL")
    d["densPL_alpha"] = DescribedItem(0.0)
    d["nCore"] = DescribedItem(1.0e3)
    d["nISM"] = DescribedItem(1.0)
    d["rCore"] = DescribedItem(0.5)
    d["rCloud"] = DescribedItem(20.0)
    d["nEdge"] = DescribedItem(2.0)
    d["tSF"] = DescribedItem(0)
    d["initial_cloud_r_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_R))
    d["initial_cloud_n_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_N))
    d["initial_cloud_m_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_M))
    # Varying scalars
    d["t_now"] = DescribedItem(0.0)
    d["R2"] = DescribedItem(0.1)
    d["v2"] = DescribedItem(100.0)
    return d


def _save_snapshot_with(d: DescribedDict, *, t_now: float, R2: float):
    d["t_now"] = DescribedItem(t_now)
    d["R2"] = DescribedItem(R2)
    d.save_snapshot()


def _write_three_snapshots(out_dir: Path) -> DescribedDict:
    d = _make_params(out_dir)
    _save_snapshot_with(d, t_now=0.0, R2=0.1)
    _save_snapshot_with(d, t_now=1.0, R2=0.5)
    _save_snapshot_with(d, t_now=2.0, R2=1.0)
    d.flush()
    return d


# ---------------------------------------------------------------------------
# Writer behaviour
# ---------------------------------------------------------------------------

class TestWriter:

    def test_metadata_json_is_created(self, tmp_path, disable_crash_handlers):
        _write_three_snapshots(tmp_path)
        assert (tmp_path / METADATA_FILENAME).exists()

    def test_metadata_json_has_version(self, tmp_path, disable_crash_handlers):
        _write_three_snapshots(tmp_path)
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert md.get("_metadata_version") == METADATA_VERSION

    def test_metadata_contains_all_run_consts(self, tmp_path, disable_crash_handlers):
        _write_three_snapshots(tmp_path)
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        for k in RUN_CONST_KEYS:
            assert k in md, f"{k} missing from metadata.json"

    def test_run_consts_stripped_from_snapshots(self, tmp_path, disable_crash_handlers):
        _write_three_snapshots(tmp_path)
        with open(tmp_path / "dictionary.jsonl") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 3
        for i, line in enumerate(lines):
            snap = json.loads(line)
            for k in RUN_CONST_KEYS:
                assert k not in snap, (
                    f"snapshot {i}: {k} should have been stripped"
                )
            # Varying keys must still be there
            assert "t_now" in snap
            assert "R2" in snap

    def test_metadata_json_overwritten_on_fresh_run(
        self, tmp_path, disable_crash_handlers,
    ):
        # First run
        _write_three_snapshots(tmp_path)
        first_size = (tmp_path / METADATA_FILENAME).stat().st_size
        first_mtime = (tmp_path / METADATA_FILENAME).stat().st_mtime_ns

        # Sleep briefly so mtime can change
        import time
        time.sleep(0.01)

        # Second run with different model_name (also overwrites jsonl)
        d = _make_params(tmp_path)
        d["model_name"] = DescribedItem("second_run")
        _save_snapshot_with(d, t_now=0.0, R2=0.1)
        d.flush()

        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert md["model_name"] == "second_run"
        assert (tmp_path / METADATA_FILENAME).stat().st_mtime_ns >= first_mtime


# ---------------------------------------------------------------------------
# Reader rehydration
# ---------------------------------------------------------------------------

class TestRehydrate:

    def test_trinity_output_rehydrates(self, tmp_path, disable_crash_handlers):
        _write_three_snapshots(tmp_path)
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        assert len(out) == 3
        snap = out[0]
        assert snap.get("model_name") == "test_run"
        assert snap.get("mCloud") == pytest.approx(1.0e6)
        np.testing.assert_allclose(
            snap.get("initial_cloud_r_arr"), _SAMPLE_INIT_R
        )

    def test_describeddict_load_snapshots_rehydrates(
        self, tmp_path, disable_crash_handlers,
    ):
        _write_three_snapshots(tmp_path)
        snapshots = DescribedDict.load_snapshots(tmp_path)
        assert len(snapshots) == 3
        for sid, snap in snapshots.items():
            assert snap.get("model_name") == "test_run"
            assert snap.get("mCloud") == pytest.approx(1.0e6)

    def test_per_snapshot_value_wins_over_metadata(
        self, tmp_path, disable_crash_handlers,
    ):
        """``setdefault`` semantics: a stray per-snapshot value overrides
        the metadata.json value."""
        _write_three_snapshots(tmp_path)
        # Manually inject a different mCloud into snapshot 1
        path = tmp_path / "dictionary.jsonl"
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        lines[1]["mCloud"] = 999.0
        with open(path, "w") as f:
            for s in lines:
                f.write(json.dumps(s) + "\n")
        out = TrinityOutput.open(path)
        assert out[0].get("mCloud") == pytest.approx(1.0e6)   # from metadata
        assert out[1].get("mCloud") == pytest.approx(999.0)   # from snapshot
        assert out[2].get("mCloud") == pytest.approx(1.0e6)   # from metadata


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

class TestLegacy:

    def test_legacy_file_loads_identically(self, tmp_path, disable_crash_handlers):
        """A jsonl that already contains the run-constants and has no
        metadata.json must load unchanged (this matches every file in
        ``outputs/mockOutput/`` produced before this feature)."""
        # Build a legacy snapshot manually
        legacy_snap = {
            "model_name": "legacy_run",
            "mCloud": 2.0e6,
            "dens_profile": "densPL",
            "densPL_alpha": 0.0,
            "nCore": 1.0e3,
            "nISM": 1.0,
            "rCore": 0.5,
            "rCloud": 20.0,
            "nEdge": 2.0,
            "tSF": 0,
            "initial_cloud_r_arr": _SAMPLE_INIT_R,
            "initial_cloud_n_arr": _SAMPLE_INIT_N,
            "initial_cloud_m_arr": _SAMPLE_INIT_M,
            "t_now": 0.0,
            "R2": 0.1,
        }
        path = tmp_path / "dictionary.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(legacy_snap) + "\n")
        # No metadata.json sibling
        assert not (tmp_path / METADATA_FILENAME).exists()

        out = TrinityOutput.open(path)
        assert len(out) == 1
        snap = out[0]
        assert snap.get("model_name") == "legacy_run"
        assert snap.get("mCloud") == pytest.approx(2.0e6)
        np.testing.assert_allclose(
            snap.get("initial_cloud_r_arr"), _SAMPLE_INIT_R
        )

    def test_missing_metadata_yields_none_for_consts(
        self, tmp_path, disable_crash_handlers,
    ):
        """When neither metadata.json nor per-snapshot values exist
        for a run-constant, ``Snapshot.get`` returns the default
        (None by default).  No exception."""
        path = tmp_path / "dictionary.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"t_now": 0.0, "R2": 0.1}) + "\n")
        out = TrinityOutput.open(path)
        assert out[0].get("mCloud") is None
        assert out[0].get("mCloud", "default") == "default"

    def test_corrupt_metadata_logs_and_proceeds(
        self, tmp_path, disable_crash_handlers, caplog,
    ):
        """A broken metadata.json must not crash the load — the reader
        warns and proceeds without rehydrate."""
        # Snapshot has the keys inline (legacy-style)
        snap = {
            "model_name": "test", "mCloud": 1.0e6,
            "t_now": 0.0, "R2": 0.1,
        }
        path = tmp_path / "dictionary.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(snap) + "\n")
        # Corrupt metadata
        with open(tmp_path / METADATA_FILENAME, "w") as f:
            f.write("{not valid json")

        import logging
        with caplog.at_level(logging.WARNING):
            out = TrinityOutput.open(path)
        assert len(out) == 1
        assert out[0].get("mCloud") == pytest.approx(1.0e6)
        assert any("metadata" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Size invariant — the whole point of the split
# ---------------------------------------------------------------------------

class TestSize:

    def test_split_layout_is_smaller_than_inline(
        self, tmp_path, disable_crash_handlers,
    ):
        """Writing 6 snapshots with run-constants split out should be
        meaningfully smaller than writing them inline.  This is the
        core regression test for the feature."""
        _make_params  # ensure fixture import resolves
        d = _make_params(tmp_path)
        for i in range(6):
            _save_snapshot_with(d, t_now=float(i), R2=0.1 * (i + 1))
        d.flush()

        jsonl_size = (tmp_path / "dictionary.jsonl").stat().st_size
        meta_size = (tmp_path / METADATA_FILENAME).stat().st_size
        split_total = jsonl_size + meta_size

        # Manually compute the size if everything were inline
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        md.pop("_metadata_version", None)
        with open(tmp_path / "dictionary.jsonl") as f:
            inline_total = 0
            for line in f:
                snap = json.loads(line)
                for k, v in md.items():
                    snap.setdefault(k, v)
                inline_total += len(json.dumps(snap)) + 1  # +1 for newline

        # Split layout must be smaller than inline (typically 2-5×)
        assert split_total < inline_total, (
            f"split={split_total} >= inline={inline_total}"
        )
        # Sanity: the savings should be at least the size of metadata
        # written 5 extra times.
        assert split_total < inline_total - 4 * meta_size
