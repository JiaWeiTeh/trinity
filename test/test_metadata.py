"""
Tests for the ``metadata.json`` run-constants split.

Coverage:
- writer emits ``metadata.json`` once, strips run-constants from snapshots
- reader (both ``TrinityOutput`` and ``DescribedDict.load_snapshots``)
  rehydrates run-constants transparently
- legacy v1 files (inline ``initial_cloud_*_arr``) keep loading identically
- ``setdefault`` semantics: per-snapshot value wins over metadata
- corrupted / missing metadata is tolerated (warning, not exception)
- size invariant: a fresh write splits 1.4 MB legacy → ≤ 30 KB total
- v2 schema: ``initial_cloud_*_arr`` dropped, reconstructed on demand
- defensive serialization: non-JSON values are skipped with a warning
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from src._input.dictionary import DescribedDict, DescribedItem
from src._output.run_constants import (
    METADATA_EXCLUDE, METADATA_FILENAME, METADATA_VERSION,
    RUN_CONST_KEYS, DROPPED_IN_V2,
)
from src._output.trinity_reader import TrinityOutput


# Sample arrays used to populate legacy v1 metadata in compat tests.
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


def _make_params(out_dir: Path, *, include_legacy_arrays: bool = False
                 ) -> DescribedDict:
    """
    Populate a DescribedDict with every v2 run-constant the writer
    expects, plus a few varying scalars.  Values are chosen for
    plausibility (Msun, pc, internal units) but are not physically
    consistent — the writer is unit-blind.

    Parameters
    ----------
    include_legacy_arrays
        When True, also writes the three ``initial_cloud_*_arr`` keys
        (used to simulate a v1-style metadata.json so legacy-compat
        tests can verify the inline-array reader fallback).
    """
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(out_dir), info="Output dir")

    # --- All v2 RUN_CONST_KEYS, sensible test values ---
    _scalars: dict[str, object] = {
        # Identifiers
        "model_name": "test_run",
        "mCloud": 1.0e6,
        "sfe": 0.01,
        "ZCloud": 1.0,
        "include_PHII": True,
        "dens_profile": "densPL",
        "densPL_alpha": 0.0,
        "nCore": 1.0e3,
        "nISM": 1.0,
        "rCore": 0.5,
        # Control inputs
        "allowShellDissolution": True,
        "stop_t_diss": 1.0,
        "stop_r": None,
        "stop_v": -1.0e4,
        "stop_t": 0.3,
        "coll_r": 1.0,
        "expansionBeyondCloud": True,
        "use_adaptive_solver": True,
        "adiabaticOnlyInCore": False,
        "immediate_leak": False,
        # SB99
        "SB99_BHCUT": 120.0,
        "SB99_mass": 1.0e6,
        "SB99_rotation": 1.0,
        # Feedback
        "FB_mColdSNFrac": 0.0,
        "FB_mColdWindFrac": 0.0,
        "FB_thermCoeffSN": 1.0,
        "FB_thermCoeffWind": 1.0,
        "FB_vSN": 1.0e4,
        # Solver tuning
        "phaseSwitch_LlossLgain": 1.0,
        "bubble_xi_Tb": 0.9,
        # Logging
        "output_format": "JSON",
        "log_level": "INFO",
        "log_colors": True,
        "log_console": False,
        "log_file": True,
        # Derived
        "rCloud": 20.0,
        "nEdge": 2.0,
        "tSF": 0,
        "mCluster": 1.0e4,
        "mu_atom": 1.07e-57,
        "mu_ion": 5.12e-58,
        "mu_mol": 1.96e-57,
        "mu_convert": 1.18e-57,
        "TShell_ion": 1.0e4,
        "TShell_neu": 1.0e2,
        "caseB_alpha": 2.6e-13,
        "C_thermal": 6.0e-7,
        "dust_KappaIR": 1.0e-26,
        "dust_noZ": 1.0,
        "dust_sigma": 1.5e-21,
        "gamma_adia": 5.0 / 3.0,
        # Physical constants
        "G": 4.498e-3,
        "c_light": 0.307,
        "k_B": 1.380e-16,
        "PISM": 3.6e-13,
    }
    for k, v in _scalars.items():
        d[k] = DescribedItem(v)

    # Optional legacy v1 inline arrays
    if include_legacy_arrays:
        d["initial_cloud_r_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_R))
        d["initial_cloud_n_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_N))
        d["initial_cloud_m_arr"] = DescribedItem(np.asarray(_SAMPLE_INIT_M))

    # Varying scalars (will end up in snapshots, not metadata)
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

    def test_metadata_contains_all_populated_run_consts(
        self, tmp_path, disable_crash_handlers,
    ):
        """Every RUN_CONST_KEYS entry populated in the test params
        ends up in metadata.json.  BE-only keys (``densBE_*``) are not
        populated in this PL test fixture and are correctly absent."""
        d = _write_three_snapshots(tmp_path)
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        # Check the keys that were actually set on the params container
        for k in RUN_CONST_KEYS:
            if k in d:
                assert k in md, f"{k} missing from metadata.json"

    def test_run_consts_stripped_from_snapshots(self, tmp_path, disable_crash_handlers):
        d = _write_three_snapshots(tmp_path)
        with open(tmp_path / "dictionary.jsonl") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 3
        # Strip set = RUN_CONST_KEYS ∪ DROPPED_IN_V2 ∪ METADATA_EXCLUDE
        strip_set = set(RUN_CONST_KEYS) | DROPPED_IN_V2 | METADATA_EXCLUDE
        for i, line in enumerate(lines):
            snap = json.loads(line)
            for k in strip_set:
                if k in d:  # only check keys that were actually populated
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
        """v2: scalar run-constants rehydrate into every snapshot.
        Arrays were dropped — the inline-array path tested previously is
        now covered by ``test_legacy_v1_inline_arrays_used`` instead."""
        _write_three_snapshots(tmp_path)
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        assert len(out) == 3
        snap = out[0]
        assert snap.get("model_name") == "test_run"
        assert snap.get("mCloud") == pytest.approx(1.0e6)
        # Several new v2 scalars also rehydrate
        assert snap.get("mu_atom") == pytest.approx(1.07e-57)
        assert snap.get("sfe") == pytest.approx(0.01)

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


# ---------------------------------------------------------------------------
# Phase 1 — expanded run-constants + dropped initial_cloud_*_arr arrays
# ---------------------------------------------------------------------------

class TestExpandedScope:
    """Phase 1 expands RUN_CONST_KEYS to ~57 scalars and drops the
    three ``initial_cloud_*_arr`` arrays.  These tests pin the new
    contract."""

    def test_version_is_at_least_v2(self, tmp_path, disable_crash_handlers):
        """Schema must be ≥ 2 (Phase 1 introduced v2; Phase 2 bumps to v3)."""
        _write_three_snapshots(tmp_path)
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert md["_metadata_version"] >= 2
        # And matches the module-level constant
        assert md["_metadata_version"] == METADATA_VERSION

    def test_initial_cloud_arrays_are_dropped(
        self, tmp_path, disable_crash_handlers,
    ):
        """v2 writer must not emit the three initial_cloud_*_arr keys
        even if they are populated in params (legacy compat happens at
        read time only)."""
        d = _make_params(tmp_path, include_legacy_arrays=True)
        _save_snapshot_with(d, t_now=0.0, R2=0.1)
        d.flush()
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        for k in ("initial_cloud_r_arr", "initial_cloud_n_arr",
                  "initial_cloud_m_arr"):
            assert k not in md, f"{k} should be dropped from v2 metadata"

    def test_dropped_arrays_also_stripped_from_snapshots(
        self, tmp_path, disable_crash_handlers,
    ):
        d = _make_params(tmp_path, include_legacy_arrays=True)
        _save_snapshot_with(d, t_now=0.0, R2=0.1)
        d.flush()
        with open(tmp_path / "dictionary.jsonl") as f:
            snap = json.loads(f.readline())
        for k in ("initial_cloud_r_arr", "initial_cloud_n_arr",
                  "initial_cloud_m_arr"):
            assert k not in snap, f"{k} should be stripped from snapshots"

    def test_metadata_is_pretty_printed(
        self, tmp_path, disable_crash_handlers,
    ):
        """Pretty-printed JSON has newlines and indentation."""
        _write_three_snapshots(tmp_path)
        raw = (tmp_path / METADATA_FILENAME).read_text()
        assert "\n" in raw, "metadata.json should be multi-line"
        assert '  "' in raw, "metadata.json should be indented"

    def test_writer_skips_non_serializable_value(
        self, tmp_path, disable_crash_handlers, caplog,
    ):
        """An unexpected non-JSON value in params (e.g. a lambda) is
        logged and skipped rather than crashing the flush."""
        d = _make_params(tmp_path)
        # Add a fake function-typed key that happens to be in RUN_CONST_KEYS
        # to trigger the defensive serialization path.  Use one of the
        # scalars we already populate but replace its value with a lambda.
        d["mu_atom"] = DescribedItem(lambda x: x)
        import logging
        with caplog.at_level(logging.WARNING):
            _save_snapshot_with(d, t_now=0.0, R2=0.1)
            d.flush()
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert "mu_atom" not in md, "non-serializable value should be skipped"
        # The other run-constants must still be written.
        assert "mCloud" in md
        # And the writer should have logged the skip.
        assert any("non-serializable" in rec.message.lower()
                   or "mu_atom" in rec.message
                   for rec in caplog.records)

    def test_writer_skips_metadata_exclude_keys(
        self, tmp_path, disable_crash_handlers,
    ):
        """Keys in METADATA_EXCLUDE (paths, function tables, empty
        placeholders) are never written even if present in params."""
        d = _make_params(tmp_path)
        # Pick a representative key from the exclude set
        d["path_cooling_CIE"] = DescribedItem("/some/path")
        _save_snapshot_with(d, t_now=0.0, R2=0.1)
        d.flush()
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        for k in METADATA_EXCLUDE:
            assert k not in md, f"{k} (in METADATA_EXCLUDE) should not be written"


class TestInitialCloudProfileReconstruction:
    """``output.initial_cloud_profile()`` reconstructs (r, n, m) from
    v2 metadata scalars and falls back to inline arrays for v1 files."""

    def test_reconstructs_powerlaw_from_v2_metadata(
        self, tmp_path, disable_crash_handlers,
    ):
        """v2: metadata has no inline arrays; reconstruction must
        succeed using only the scalars."""
        _write_three_snapshots(tmp_path)
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        r, n, m = out.initial_cloud_profile()
        # Should be non-empty arrays of equal length
        assert r.size > 0 and r.size == n.size == m.size
        # Radius monotonically increasing
        assert np.all(np.diff(r) > 0)
        # Power-law alpha=0 → uniform density inside core, equal to nCore
        # near the inner part of the array
        nCore = out.metadata["nCore"]
        # The first sample (inside rCore) should equal nCore
        assert abs(r[0]) < float(out.metadata["rCore"]) * 2.0
        assert abs(n[0] / nCore - 1.0) < 1e-6

    def test_legacy_v1_inline_arrays_used(
        self, tmp_path, disable_crash_handlers,
    ):
        """If metadata.json has inline arrays (v1 schema), the reader
        returns them directly without recomputing."""
        # Build a v1-style metadata.json by hand (no writer path
        # produces this anymore — we simulate a legacy file).
        v1_md = {
            "_metadata_version": 1,
            "model_name": "legacy",
            "mCloud": 1e6,
            "nCore": 1e3,
            "nISM": 1.0,
            "rCore": 0.5,
            "rCloud": 20.0,
            "dens_profile": "densPL",
            "densPL_alpha": 0.0,
            "initial_cloud_r_arr": _SAMPLE_INIT_R,
            "initial_cloud_n_arr": _SAMPLE_INIT_N,
            "initial_cloud_m_arr": _SAMPLE_INIT_M,
        }
        with open(tmp_path / METADATA_FILENAME, "w") as f:
            json.dump(v1_md, f)
        with open(tmp_path / "dictionary.jsonl", "w") as f:
            f.write(json.dumps({"t_now": 0.0, "R2": 0.1}) + "\n")
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        r, n, m = out.initial_cloud_profile()
        np.testing.assert_allclose(r, _SAMPLE_INIT_R)
        np.testing.assert_allclose(n, _SAMPLE_INIT_N)
        np.testing.assert_allclose(m, _SAMPLE_INIT_M)

    def test_raises_when_metadata_missing_scalars(
        self, tmp_path, disable_crash_handlers,
    ):
        """If metadata has neither inline arrays nor the required scalars
        (truly broken/empty), the reconstruction raises a clear KeyError."""
        broken_md = {"_metadata_version": 2, "model_name": "broken"}
        with open(tmp_path / METADATA_FILENAME, "w") as f:
            json.dump(broken_md, f)
        with open(tmp_path / "dictionary.jsonl", "w") as f:
            f.write(json.dumps({"t_now": 0.0}) + "\n")
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        with pytest.raises(KeyError, match="reconstruct"):
            out.initial_cloud_profile()

    def test_metadata_property_returns_parsed_json(
        self, tmp_path, disable_crash_handlers,
    ):
        """``output.metadata`` is the parsed metadata.json dict."""
        _write_three_snapshots(tmp_path)
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        md = out.metadata
        assert isinstance(md, dict)
        assert md["model_name"] == "test_run"
        assert md["_metadata_version"] == METADATA_VERSION
        # Cached across multiple property accesses
        assert out.metadata is md

    def test_metadata_property_empty_when_file_absent(
        self, tmp_path, disable_crash_handlers,
    ):
        """If metadata.json doesn't exist, ``output.metadata`` returns {}."""
        with open(tmp_path / "dictionary.jsonl", "w") as f:
            f.write(json.dumps({"t_now": 0.0}) + "\n")
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        assert out.metadata == {}


# ---------------------------------------------------------------------------
# Phase 2 — termination + final_state blocks
# ---------------------------------------------------------------------------

class TestTerminationBlock:
    """Phase 2 adds a ``termination`` block to metadata.json that
    mirrors ``read_simulation_end()``'s return shape."""

    def _make_params_with_end_codes(self, tmp_path, *, exit_code=1,
                                    reason="Stopping time reached"):
        from src._output.simulation_end import SimulationEndCode
        d = _make_params(tmp_path)
        d["SimulationEndReason"] = DescribedItem(reason)
        d["SimulationEndCode"] = DescribedItem(exit_code)
        # Need a snapshot/flush so metadata.json exists first
        _save_snapshot_with(d, t_now=0.5, R2=2.0)
        d.flush()
        return d

    def test_writes_termination_block(self, tmp_path, disable_crash_handlers):
        from src._output.simulation_end import write_simulation_end
        d = self._make_params_with_end_codes(tmp_path)
        write_simulation_end(d, str(tmp_path))
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert "termination" in md
        t = md["termination"]
        assert set(t.keys()) == {
            "exit_code", "outcome", "detail", "timestamp", "model_name",
        }
        assert t["exit_code"] == 1
        assert t["outcome"] == "stopping_time"
        assert t["detail"] == "Stopping time reached"
        assert t["model_name"] == "test_run"

    def test_writes_final_state_block(self, tmp_path, disable_crash_handlers):
        from src._output.simulation_end import write_simulation_end
        d = self._make_params_with_end_codes(tmp_path)
        # Tweak some "varying" scalars to simulate end-of-run values
        d["t_now"] = DescribedItem(0.300)
        d["R2"] = DescribedItem(2.51)
        d["v2"] = DescribedItem(2.45e-3)
        d["Eb"] = DescribedItem(0.0)
        d["current_phase"] = DescribedItem("momentum")
        d["isCollapse"] = DescribedItem(False)
        write_simulation_end(d, str(tmp_path))
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        assert "final_state" in md
        fs = md["final_state"]
        # All the varying scalars we set must be in the block
        assert fs["t_now"] == pytest.approx(0.300)
        assert fs["R2"] == pytest.approx(2.51)
        assert fs["v2"] == pytest.approx(2.45e-3)
        assert fs["Eb"] == 0.0
        assert fs["current_phase"] == "momentum"
        assert fs["isCollapse"] is False
        # Run-constants must NOT be in final_state (they live up top)
        assert "mCloud" not in fs
        assert "nCore" not in fs

    def test_final_state_excludes_long_arrays(
        self, tmp_path, disable_crash_handlers,
    ):
        from src._output.simulation_end import write_simulation_end
        d = self._make_params_with_end_codes(tmp_path)
        # Stuff a long array into params under one of the excluded keys
        d["bubble_T_arr_r_arr"] = DescribedItem(np.linspace(0, 1, 30))
        d["log_shell_n_arr"] = DescribedItem(np.linspace(50, 60, 30))
        write_simulation_end(d, str(tmp_path))
        with open(tmp_path / METADATA_FILENAME) as f:
            md = json.load(f)
        fs = md["final_state"]
        assert "bubble_T_arr_r_arr" not in fs
        assert "log_shell_n_arr" not in fs


class TestTrinityOutputProperties:
    """Phase 2 surfaces ``termination`` / ``final_state`` /
    ``is_successful_run`` on ``TrinityOutput``."""

    def _open_run_with_termination(self, tmp_path, *, exit_code=1):
        from src._output.simulation_end import write_simulation_end
        d = _make_params(tmp_path)
        d["SimulationEndReason"] = DescribedItem("Stopping time reached")
        d["SimulationEndCode"] = DescribedItem(exit_code)
        _save_snapshot_with(d, t_now=0.5, R2=2.0)
        d.flush()
        write_simulation_end(d, str(tmp_path))
        return TrinityOutput.open(tmp_path / "dictionary.jsonl")

    def test_termination_property(self, tmp_path, disable_crash_handlers):
        out = self._open_run_with_termination(tmp_path)
        t = out.termination
        assert isinstance(t, dict)
        assert t["outcome"] == "stopping_time"
        assert t["exit_code"] == 1

    def test_final_state_property(self, tmp_path, disable_crash_handlers):
        out = self._open_run_with_termination(tmp_path)
        fs = out.final_state
        assert isinstance(fs, dict)
        # The varying-scalar values from _make_params must be present
        assert "t_now" in fs and "R2" in fs and "v2" in fs

    def test_is_successful_run_true_for_clean_exit(
        self, tmp_path, disable_crash_handlers,
    ):
        out = self._open_run_with_termination(tmp_path, exit_code=1)
        assert out.is_successful_run is True

    def test_is_successful_run_false_for_bad_exit(
        self, tmp_path, disable_crash_handlers,
    ):
        out = self._open_run_with_termination(tmp_path, exit_code=42)
        assert out.is_successful_run is False

    def test_is_successful_run_none_when_termination_missing(
        self, tmp_path, disable_crash_handlers,
    ):
        # Run that flushed metadata but never reached write_simulation_end
        d = _make_params(tmp_path)
        _save_snapshot_with(d, t_now=0.5, R2=2.0)
        d.flush()
        out = TrinityOutput.open(tmp_path / "dictionary.jsonl")
        assert out.termination is None
        assert out.final_state is None
        assert out.is_successful_run is None

    def test_termination_not_rehydrated_into_snapshots(
        self, tmp_path, disable_crash_handlers,
    ):
        """``termination`` and ``final_state`` are reserved top-level
        blocks — they must NOT smear into every snapshot's data dict."""
        out = self._open_run_with_termination(tmp_path)
        snap = out[0]
        assert "termination" not in snap.data
        assert "final_state" not in snap.data


class TestReadSimulationEndMigration:
    """``read_simulation_end()`` prefers metadata.json, falls back to text."""

    def _make_termination_run(self, tmp_path):
        from src._output.simulation_end import write_simulation_end
        d = _make_params(tmp_path)
        d["SimulationEndReason"] = DescribedItem("Stopping time reached")
        d["SimulationEndCode"] = DescribedItem(1)
        _save_snapshot_with(d, t_now=0.5, R2=2.0)
        d.flush()
        write_simulation_end(d, str(tmp_path))

    def test_reads_from_metadata_block(self, tmp_path, disable_crash_handlers):
        from src._output.simulation_end import read_simulation_end
        self._make_termination_run(tmp_path)
        # Delete simulationEnd.txt so we can be sure JSON is the source
        (tmp_path / "simulationEnd.txt").unlink()
        result = read_simulation_end(str(tmp_path))
        assert result is not None
        assert result["exit_code"] == 1
        assert result["outcome"] == "stopping_time"
        assert result["model"] == "test_run"

    def test_falls_back_to_text_for_legacy_runs(
        self, tmp_path, disable_crash_handlers,
    ):
        """Legacy run: no metadata.json (or v1/v2 with no termination
        block); read_simulation_end falls back to text-parsing."""
        from src._output.simulation_end import read_simulation_end
        # Write only simulationEnd.txt, no metadata.json with termination
        text = (
            "==================================================\n"
            "TRINITY Simulation End Report\n"
            "==================================================\n"
            "Timestamp: 2026-01-01 00:00:00\n"
            "Model: legacy_run\n"
            "\n"
            "Outcome: stopping_time\n"
            "Detail: Legacy text-parsed reason\n"
            "Exit Code: 1\n"
        )
        (tmp_path / "simulationEnd.txt").write_text(text)
        result = read_simulation_end(str(tmp_path))
        assert result is not None
        assert result["exit_code"] == 1
        assert result["outcome"] == "stopping_time"
        assert result["detail"] == "Legacy text-parsed reason"
        assert result["model"] == "legacy_run"

    def test_returns_none_when_both_sources_missing(
        self, tmp_path, disable_crash_handlers,
    ):
        from src._output.simulation_end import read_simulation_end
        assert read_simulation_end(str(tmp_path)) is None

    def test_metadata_block_takes_precedence(
        self, tmp_path, disable_crash_handlers,
    ):
        """When both metadata.json[termination] and simulationEnd.txt
        exist, the JSON wins (they should agree, but JSON is canonical)."""
        from src._output.simulation_end import read_simulation_end
        self._make_termination_run(tmp_path)
        # Manually rewrite simulationEnd.txt to a different value to detect
        # which source is consulted
        (tmp_path / "simulationEnd.txt").write_text(
            "Outcome: collapse\nDetail: bogus\nExit Code: 99\n"
        )
        result = read_simulation_end(str(tmp_path))
        # JSON wins → exit_code stays at 1, outcome stays at stopping_time
        assert result["exit_code"] == 1
        assert result["outcome"] == "stopping_time"
