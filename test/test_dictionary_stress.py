"""Characterization tests for the ``DescribedDict`` snapshot machinery.

Batteries A-G of ``docs/dev/dictionary-robustness/PLAN.md``. Every test here
pins **current** behavior of ``trinity/_input/dictionary.py`` — including the
places where that behavior is a defect. Cases that pin a defect carry a
``CANDIDATE-BUG F<n>`` comment naming the finding in that plan; the fix
decisions are the maintainer's (plan §6), so nothing here asserts the
*desired* behavior.

Reading these as a spec would be a mistake: a red test in this file after a
deliberate fix means "the pin is stale, re-baseline it", not "regression".

Batteries C (crash handlers / signals) and H (full-run field checks) live in
``test_dictionary_stress_process.py`` — they need real interpreters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from trinity._input.dictionary import (
    DescribedDict,
    DescribedItem,
    save_debug_snapshot,
    updateDict,
)


@pytest.fixture
def no_handlers(monkeypatch):
    """Stop ``__init__`` registering atexit/signal hooks (as test_metadata.py:39).

    Without this every constructed dict leaks an atexit handler that rewrites
    a run directory at interpreter exit — finding F7, which battery C pins
    deliberately in a subprocess.
    """
    monkeypatch.setattr(DescribedDict, "_register_crash_handlers", lambda self: None)


def _params(out_dir: Path, **extra) -> DescribedDict:
    """Minimal snapshot-capable dict: t_now + R2 are the duplicate-guard key.

    Machinery-only — no physics is exercised, so plausible-value discipline
    does not apply here (see plan §4.3); the values only need to be distinct.
    """
    d = DescribedDict()
    d["path2output"] = DescribedItem(str(out_dir))
    d["t_now"] = DescribedItem(0.0)
    d["R2"] = DescribedItem(1.0)
    for k, v in extra.items():
        d[k] = DescribedItem(v)
    return d


def _lines(out_dir: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out_dir / "dictionary.jsonl").read_text().splitlines()]


# =============================================================================
# Battery A — duplicate-guard semantics (F1-F4)
# =============================================================================
class TestDuplicateGuard:
    def test_guard_is_disarmed_at_the_flush_boundary(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F1 — the motivating question: yes, skipped at every boundary.

        ``flush()`` empties ``previous_snapshot``; the guard requires a
        non-empty buffer, so the first save after a flush is unconditional.
        """
        d = _params(tmp_path)
        for i in range(10):  # the 10th save flushes and clears the buffer
            d["t_now"].value = 0.1 * i
            d["R2"].value = 1.0 + i
            d.save_snapshot()
        assert d.save_count == 10
        assert d.previous_snapshot == {}, "flush() should have cleared the buffer"

        d.save_snapshot()  # identical (t_now, R2) to snapshot 9
        assert d.save_count == 11, "duplicate slipped through at the boundary"

        d.flush()
        lines = _lines(tmp_path)
        assert len(lines) == 11
        assert (lines[9]["t_now"], lines[9]["R2"]) == (lines[10]["t_now"], lines[10]["R2"])

    def test_guard_works_inside_the_window(self, tmp_path, no_handlers):
        """The guard's positive control: in-window duplicates are dropped."""
        d = _params(tmp_path)
        d.save_snapshot()
        d.save_snapshot()
        assert d.save_count == 1

    def test_guard_disarmed_after_any_manual_flush(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F2 — mid-window flushes disarm it too."""
        d = _params(tmp_path)
        for i in range(3):
            d["t_now"].value = 0.1 * i
            d.save_snapshot()
        d.flush()
        d.save_snapshot()  # identical to snapshot 2
        assert d.save_count == 4

    def test_nan_t_now_defeats_the_guard(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F3 — NaN != NaN, so equality never matches."""
        d = _params(tmp_path)
        d["t_now"].value = float("nan")
        d.save_snapshot()
        d.save_snapshot()
        assert d.save_count == 2

    def test_guard_key_ignores_every_other_field(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F4 — only (t_now, R2) is compared.

        Load-bearing: run_energy_phase.py's reconciliation snapshot exists
        *because* this blocks the next phase's stale first save. A fix must
        re-read that call site (plan §6.2).
        """
        d = _params(tmp_path, current_phase="energy", Eb=1.0e5)
        d.save_snapshot()
        d["current_phase"].value = "implicit"  # different phase, same (t_now, R2)
        d["Eb"].value = 9.9e9
        d.save_snapshot()
        assert d.save_count == 1, "phase change did not survive the guard"

    def test_record_content_depends_on_flush_alignment(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F1 — the same save sequence yields different records.

        Two identical-state saves are dropped mid-window but both kept when
        they straddle a boundary, so the on-disk record is not a pure function
        of the trajectory.
        """

        def count_lines(offset: int, out: Path) -> int:
            d = _params(out)
            for i in range(offset):  # shift where the identical pair lands
                d["t_now"].value = 0.1 * (i + 1)
                d.save_snapshot()
            d["t_now"].value = 99.0
            d.save_snapshot()
            d.save_snapshot()  # identical to the one above
            d.flush()
            return len(_lines(out))

        mid = count_lines(3, tmp_path / "mid")  # pair sits inside a window
        straddle = count_lines(9, tmp_path / "straddle")  # pair straddles the flush
        assert mid == 4, "in-window duplicate should have been dropped"
        assert straddle == 11, "boundary duplicate should have been kept"

    def test_snapshot_interval_of_one_never_dedupes(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F1/F16 — interval=1 flushes every save, so the guard is dead."""
        d = _params(tmp_path)
        d.snapshot_interval = 1
        d.save_snapshot()
        d.save_snapshot()  # byte-identical state
        assert d.save_count == 2

    def test_snapshot_interval_of_zero_raises(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F16 — no validation on a plain public attribute."""
        d = _params(tmp_path)
        d.snapshot_interval = 0
        with pytest.raises(ZeroDivisionError):
            d.save_snapshot()

    def test_missing_r2_skips_duplicate_detection(self, tmp_path, no_handlers):
        """The guard's ``except KeyError`` path: no R2 ⇒ no dedup at all."""
        d = DescribedDict()
        d["path2output"] = DescribedItem(str(tmp_path))
        d["t_now"] = DescribedItem(0.0)  # no R2
        d.save_snapshot()
        d.save_snapshot()
        assert d.save_count == 2


# =============================================================================
# Battery B — flush atomicity, retry, fresh-run semantics (F6, O1, O2)
# =============================================================================
class TestFlushAtomicity:
    def test_poisoned_flush_writes_partially_and_keeps_buffer(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F6 — json.dumps is inside the write loop, so the
        file is already partly written when the bad line raises."""
        d = _params(tmp_path)
        d["t_now"].value = 0.5
        d.save_snapshot()
        d["bad"] = DescribedItem(object())  # not JSON-serializable
        d["t_now"].value = 0.6
        d.save_snapshot()

        with pytest.raises(TypeError):
            d.flush()

        assert len(_lines(tmp_path)) == 1, "clean line 0 was written before the failure"
        assert set(d.previous_snapshot) == {"0", "1"}, "buffer retained, so a retry re-writes"
        assert d.flush_count == 0

    def test_first_flush_retry_self_heals(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F6 — accidentally correct: flush_count is still 0, so
        the fresh-run branch deletes the partial file and rewrites it."""
        d = _params(tmp_path)
        d["t_now"].value = 0.5
        d.save_snapshot()
        d["bad"] = DescribedItem(object())
        d["t_now"].value = 0.6
        d.save_snapshot()
        with pytest.raises(TypeError):
            d.flush()

        del d["bad"]
        d.previous_snapshot["1"] = {"t_now": 0.6, "R2": 1.0}  # caller repairs the record
        d.flush()

        assert [ln["t_now"] for ln in _lines(tmp_path)] == [0.5, 0.6]

    def test_later_flush_retry_duplicates_written_lines(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F6 — the real corruption: from flush #2 on, a retry
        appends the lines it already wrote, shifting every later snapshot id."""
        d = _params(tmp_path)
        d.save_snapshot()
        d.flush()  # flush_count -> 1, so no fresh-run rescue from here on
        d["t_now"].value = 1.0
        d.save_snapshot()
        d["bad"] = DescribedItem(object())
        d["t_now"].value = 2.0
        d.save_snapshot()
        with pytest.raises(TypeError):
            d.flush()

        del d["bad"]
        d.previous_snapshot["2"] = {"t_now": 2.0, "R2": 1.0}
        d.flush()

        assert [ln["t_now"] for ln in _lines(tmp_path)] == [0.0, 1.0, 1.0, 2.0]

    def test_empty_flush_deletes_a_previous_runs_output(self, tmp_path, no_handlers):
        """CANDIDATE-BUG O1 — the fresh-run branch fires on flush_count == 0
        even with nothing pending, so a bare flush() is destructive."""
        (tmp_path / "dictionary.jsonl").write_text('{"t_now": 42.0}\n')
        (tmp_path / "metadata.json").write_text('{"_metadata_version": 4}')

        d = _params(tmp_path)
        d.flush()  # nothing saved

        assert (tmp_path / "dictionary.jsonl").read_text() == "", "previous run's data survived?"
        assert json.loads((tmp_path / "metadata.json").read_text())["_metadata_version"] != 42

    def test_second_dict_clobbers_the_first(self, tmp_path, no_handlers):
        """CANDIDATE-BUG O1 — two writers on one directory: last one wins."""
        first = _params(tmp_path)
        for i in range(3):
            first["t_now"].value = float(i)
            first.save_snapshot()
        first.flush()
        assert len(_lines(tmp_path)) == 3

        second = _params(tmp_path)
        second["t_now"].value = 99.0
        second.save_snapshot()
        second.flush()
        assert [ln["t_now"] for ln in _lines(tmp_path)] == [99.0]

    def test_torn_trailing_line_is_skipped_on_load(self, tmp_path, no_handlers, capsys):
        """O2 — no fsync/temp+rename, so a hard crash can tear the last line.
        The loader tolerates it (warn + skip) and keeps the intact prefix."""
        d = _params(tmp_path)
        for i in range(3):
            d["t_now"].value = float(i)
            d.save_snapshot()
        d.flush()

        jl = tmp_path / "dictionary.jsonl"
        jl.write_text(jl.read_text()[:-12])  # truncate mid-line

        snaps = DescribedDict.load_snapshots(tmp_path)
        assert {"0", "1"} <= set(snaps)
        assert "Could not parse line" in capsys.readouterr().out

    def test_safe_flush_swallows_a_poisoned_buffer(self, tmp_path, no_handlers):
        """_safe_flush must never raise — it runs from atexit/signal handlers."""
        d = _params(tmp_path)
        d["bad"] = DescribedItem(object())
        d.save_snapshot()

        d._safe_flush(termination_reason="test")  # must not raise

        assert d.previous_snapshot, "buffer is only cleared on a successful flush"


# =============================================================================
# Battery D — profile-array special cases (F5)
# =============================================================================
class TestProfileArrays:
    def test_empty_bubble_pair_records_empty_arrays(self, tmp_path, no_handlers):
        """F5a/F20 — **fixed**: an empty pair now records empty arrays.

        Was ``ValueError: array of sample points is empty`` from the R²
        diagnostic (``_simplify_error`` → ``np.interp``) — never from the
        downsampler, which handles empty input (test_simplify.py::test_empty).
        Re-baselined when F20's guard landed.
        """
        d = _params(tmp_path, bubble_r_arr=np.array([]), bubble_T_arr=np.array([]))
        d.save_snapshot()
        snap = d.previous_snapshot["0"]
        assert snap["log_bubble_T_arr"] == []
        assert snap["bubble_T_arr_r_arr"] == []

    def test_missing_companion_r_array_crashes(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F5b — derived array without its x-grid."""
        d = _params(tmp_path, bubble_T_arr=np.linspace(1.0e6, 1.0e4, 50))
        with pytest.raises(KeyError, match="bubble_r_arr"):
            d.save_snapshot()

    def test_orphan_r_array_is_dropped_silently(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F5b-reverse (audit #4) — the *silent* half of the
        asymmetry: an x-grid with no derived partner hits an unconditional
        ``continue`` and vanishes from the record with no exception."""
        d = _params(tmp_path, bubble_r_arr=np.linspace(0.1, 1.0, 20))
        d.save_snapshot()
        snap = d.previous_snapshot["0"]
        assert "bubble_r_arr" not in snap
        assert not [k for k in snap if "bubble" in k], "no bubble data recorded at all"

    def test_scalar_nan_profile_arrays_crash(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F5c — exactly what ``reset_keys()`` writes by default.
        The commented-out bubble entries in COOLING_PHASE_KEYS are this bug's
        fossil (dictionary.py:1217-1222)."""
        d = _params(tmp_path, bubble_r_arr=np.nan, bubble_T_arr=np.nan)
        with pytest.raises(IndexError):
            d.save_snapshot()

    def test_reset_keys_then_snapshot_crashes(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F5c via the public API that produces the state."""
        d = _params(tmp_path, bubble_r_arr=np.linspace(0.1, 1.0, 20),
                    bubble_T_arr=np.linspace(1.0e6, 1.0e4, 20))
        d.reset_keys(["bubble_r_arr", "bubble_T_arr"])
        with pytest.raises(IndexError):
            d.save_snapshot()

    def test_empty_shell_grav_pair_records_empty_arrays(self, tmp_path, no_handlers):
        """F5d/F20 — **fixed** alongside the bubble pair (same code path).

        Note the surviving schema asymmetry (plan §6.5): an empty bubble or
        shell-grav pair now emits ``[]`` for both keys, while ``shell_n_arr``'s
        older guard omits its keys from the line entirely. Unifying the two is
        a maintainer decision, not part of F20.
        """
        d = _params(tmp_path, shell_grav_r=np.array([]), shell_grav_force_m=np.array([]))
        d.save_snapshot()
        snap = d.previous_snapshot["0"]
        assert snap["shell_grav_force_m"] == []
        assert snap["shell_grav_r"] == []

    def test_shell_n_arr_guard_drops_keys_instead(self, tmp_path, no_handlers):
        """F5 asymmetry — shell_n_arr is the *only* guarded pair, and when the
        guard trips the keys are silently absent, so the per-line schema
        varies across a run (feeds invariant I4)."""
        d = _params(tmp_path, shell_r_arr=np.array([]), shell_n_arr=np.array([]))
        d.save_snapshot()  # no crash, unlike every other pair
        snap = d.previous_snapshot["0"]
        assert "log_shell_n_arr" not in snap
        assert "shell_r_arr" not in snap

    def test_mismatched_pair_lengths_name_the_key(self, tmp_path, no_handlers):
        """simplify()'s error contract carries the keyname for debuggability."""
        d = _params(tmp_path, bubble_r_arr=np.linspace(0.1, 1.0, 20),
                    bubble_T_arr=np.linspace(1.0e6, 1.0e4, 19))
        with pytest.raises(ValueError, match="bubble_T_arr"):
            d.save_snapshot()

    def test_nonpositive_values_clamp_to_minus_300(self, tmp_path, no_handlers):
        """The log10(max(., 1e-300)) clamp silently masks zero/negative input.

        bubble_T_arr takes no abs() (a negative temperature lands at -300),
        while bubble_dTdr_arr does — an inconsistency worth knowing about.
        """
        n = np.zeros(30)
        d = _params(tmp_path, bubble_r_arr=np.linspace(0.1, 1.0, 30), bubble_n_arr=n)
        d.save_snapshot()
        assert min(d.previous_snapshot["0"]["log_bubble_n_arr"]) == -300.0

    def test_freshly_read_params_can_snapshot(self, tmp_path, no_handlers, monkeypatch):
        """F19 — a real ``read_param`` dict must survive ``save_snapshot()``.

        The registry defaults every profile array to ``np.array([])`` with
        ``exclude_from_snapshot=False``, so this is the genuine phase-0 state.
        Production only dodges it because phase-0 init populates the arrays
        before the first save fires; anything that snapshots earlier (an
        initial-condition record, an early-termination record) lands here.
        """
        from trinity._input import read_param

        monkeypatch.chdir(tmp_path)  # read_param resolves path2output under cwd
        param = tmp_path / "f19.param"
        param.write_text(
            "mCloud      1e5\nsfe         0.3\nstop_t      1e-4\nmodel_name  f19probe\n"
        )
        params = read_param.read_param(str(param))
        monkeypatch.setattr(type(params), "_register_crash_handlers", lambda self: None)

        assert np.asarray(params["bubble_r_arr"].value).size == 0, "expected the empty default"
        assert params["bubble_r_arr"].exclude_from_snapshot is False

        params.save_snapshot()  # F20's guard is what makes this survive

        assert params.save_count == 1

    def test_inf_in_profile_suppresses_the_r2_warning(self, tmp_path, no_handlers, caplog):
        """CANDIDATE-BUG — an inf makes R² NaN, and ``NaN < 0.9`` is False, so
        the low-fidelity warning never fires: silent degradation."""
        y = np.linspace(1.0e6, 1.0e4, 60)
        y[10] = np.inf
        d = _params(tmp_path, bubble_r_arr=np.linspace(0.1, 1.0, 60), bubble_T_arr=y)
        with caplog.at_level("WARNING"):
            d.save_snapshot()
        assert not [r for r in caplog.records if "R²" in r.message]


# =============================================================================
# Battery E — serialization round-trip (F11, F12, I8)
# =============================================================================
class TestSerializationRoundTrip:
    def test_type_morphing_table(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F12 — every JSON list returns as an ndarray, so
        strings and tuples change type across a round-trip."""
        d = _params(tmp_path, names=["alpha", "beta"], pair=(1, 2), flag=True,
                    count=7, ratio=0.5, nothing=None, npint=np.int64(3))
        d.save_snapshot()
        d.flush()
        loaded = DescribedDict.load_snapshot(tmp_path, 0)

        assert isinstance(loaded["names"].value, np.ndarray)  # list[str] -> ndarray[<U]
        assert loaded["names"].value.dtype.kind == "U"
        assert isinstance(loaded["pair"].value, np.ndarray)  # tuple -> ndarray
        assert isinstance(loaded["flag"].value, bool)  # scalars survive
        assert isinstance(loaded["count"].value, int)
        assert isinstance(loaded["ratio"].value, float)
        assert loaded["nothing"].value is None
        assert loaded["npint"].value == 3

    def test_nan_and_inf_are_written_as_bare_literals(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F11 — the file is not strict RFC-8259 JSON. Python
        reads it back; jq and other strict parsers reject it."""
        d = _params(tmp_path, weird=np.array([1.0, np.inf, -np.inf, np.nan]))
        d.save_snapshot()
        d.flush()

        raw = (tmp_path / "dictionary.jsonl").read_text()
        assert "Infinity" in raw and "NaN" in raw

        json.loads(raw)  # permissive parser: fine
        with pytest.raises(ValueError):  # strict parser: rejects
            json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

    def test_ragged_list_written_then_fails_to_load(self, tmp_path, no_handlers):
        """CANDIDATE-BUG — the writer accepts a ragged nested list that the
        loader's unconditional np.asarray cannot rebuild."""
        d = _params(tmp_path, ragged=[[1, 2], [3]])
        d.save_snapshot()
        d.flush()
        with pytest.raises(ValueError):
            DescribedDict.load_snapshot(tmp_path, 0)

    def test_save_load_save_is_value_stable(self, tmp_path, no_handlers):
        """I8 — a reloaded state re-serializes to the same line."""
        d = _params(tmp_path, arr=np.linspace(0.0, 1.0, 8), label="phase-1a", n=3)
        d.save_snapshot()
        d.flush()
        first = (tmp_path / "dictionary.jsonl").read_text().splitlines()[0]

        loaded = DescribedDict.load_snapshot(tmp_path, 0)
        out2 = tmp_path / "again"
        loaded["path2output"].value = str(out2)
        loaded.save_snapshot()
        loaded.flush()
        second = (tmp_path / "again" / "dictionary.jsonl").read_text().splitlines()[0]

        assert json.loads(first) == json.loads(second)

    def test_dict_valued_param_survives_as_a_dict(self, tmp_path, no_handlers):
        """_to_json_ready_value falls through for dicts; NpEncoder rescues the
        nested numpy at dump time and the loader leaves dicts alone."""
        d = _params(tmp_path, mapping={"a": np.float64(1.5), "b": 2})
        d.save_snapshot()
        d.flush()
        loaded = DescribedDict.load_snapshot(tmp_path, 0)
        assert loaded["mapping"].value == {"a": 1.5, "b": 2}


# =============================================================================
# Battery F — loader robustness (F13)
# =============================================================================
class TestLoaderRobustness:
    def _three_snapshots(self, out: Path) -> None:
        d = _params(out)
        for i in range(3):
            d["t_now"].value = float(i)
            d.save_snapshot()
        d.flush()

    def test_blank_line_shifts_every_later_id(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F13 — ids come from enumerate(), and a skipped line
        still consumes an index, so ids go {0, 2, 3} and snapshot 1's data
        now answers to id 2."""
        self._three_snapshots(tmp_path)
        jl = tmp_path / "dictionary.jsonl"
        a, b, c = jl.read_text().splitlines()
        jl.write_text("\n".join([a, "", b, c]) + "\n")

        snaps = DescribedDict.load_snapshots(tmp_path)
        assert sorted(snaps) == ["0", "2", "3"]
        assert snaps["2"]["t_now"] == 1.0  # was written as snapshot 1

    def test_corrupt_line_shifts_ids_and_warns_on_stdout(self, tmp_path, no_handlers, capsys):
        """CANDIDATE-BUG F13 + the warning goes to ``print``, not ``logging``,
        so it never reaches the run's structured log."""
        self._three_snapshots(tmp_path)
        jl = tmp_path / "dictionary.jsonl"
        a, b, c = jl.read_text().splitlines()
        jl.write_text("\n".join([a, "{not json", b, c]) + "\n")

        snaps = DescribedDict.load_snapshots(tmp_path)
        assert sorted(snaps) == ["0", "2", "3"]
        assert "Could not parse line 1" in capsys.readouterr().out

    def test_duplicated_line_shifts_content_off_by_one(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F6 aftermath — ids stay contiguous, so nothing looks
        wrong, but every id past the duplicate points at the wrong state."""
        self._three_snapshots(tmp_path)
        jl = tmp_path / "dictionary.jsonl"
        a, b, c = jl.read_text().splitlines()
        jl.write_text("\n".join([a, b, b, c]) + "\n")

        snaps = DescribedDict.load_snapshots(tmp_path)
        assert sorted(snaps) == ["0", "1", "2", "3"]
        assert snaps["2"]["t_now"] == 1.0  # writer's snapshot 2 had t_now == 2.0
        assert snaps["3"]["t_now"] == 2.0

    def test_unknown_snapshot_id_lists_what_exists(self, tmp_path, no_handlers):
        self._three_snapshots(tmp_path)
        with pytest.raises(KeyError, match="Snapshot 99 not found"):
            DescribedDict.load_snapshot(tmp_path, 99)

    def test_missing_file_raises_filenotfound(self, tmp_path, no_handlers):
        with pytest.raises(FileNotFoundError, match="No dictionary.jsonl"):
            DescribedDict.load_snapshots(tmp_path)

    def test_latest_snapshot_on_empty_file_raises(self, tmp_path, no_handlers):
        (tmp_path / "dictionary.jsonl").write_text("")
        with pytest.raises(ValueError, match="No snapshots found"):
            DescribedDict.load_latest_snapshot(tmp_path)


# =============================================================================
# Battery G — API footguns (F8, F9, F10, F18)
# =============================================================================
class TestApiEdges:
    def test_none_t_now_crashes_in_the_log_fstring(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F8 — the guard's ``except KeyError`` is the wrong net,
        and the ``:.6e`` f-string is built regardless of log level."""
        d = _params(tmp_path)
        d["t_now"].value = None
        with pytest.raises(TypeError, match="format string"):
            d.save_snapshot()

    def test_print_crashes_on_zero_d_array(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F9 — hasattr(__len__) is True for a 0-d ndarray but
        len() raises, so printing the whole dict dies."""
        d = _params(tmp_path, zero_d=np.array(3.0))
        with pytest.raises(TypeError, match="unsized object"):
            str(d)

    def test_exclusion_is_sticky(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F10 — _excluded_keys only ever grows, so replacing an
        excluded item with an included one keeps it out of every snapshot."""
        d = _params(tmp_path)
        d["secret"] = DescribedItem(42.0, exclude_from_snapshot=True)
        d["secret"] = DescribedItem(43.0, exclude_from_snapshot=False)
        d.save_snapshot()
        assert "secret" not in d.previous_snapshot["0"]

    def test_exclusion_positive_control(self, tmp_path, no_handlers):
        d = _params(tmp_path)
        d["kept"] = DescribedItem(1.0)
        d["dropped"] = DescribedItem(2.0, exclude_from_snapshot=True)
        d.save_snapshot()
        snap = d.previous_snapshot["0"]
        assert "kept" in snap and "dropped" not in snap

    def test_non_describeditem_assignment_is_rejected(self, tmp_path, no_handlers):
        d = _params(tmp_path)
        with pytest.raises(TypeError, match="must be a DescribedItem"):
            d["oops"] = 1.0

    def test_copy_loses_the_machinery(self, tmp_path, no_handlers):
        """dict.copy() is inherited, so the copy is a plain dict with no
        snapshot state — a footgun for anything that copies params."""
        d = _params(tmp_path)
        assert type(d.copy()) is dict

    def test_item_equality_on_arrays_is_ambiguous(self, tmp_path, no_handlers):
        """DescribedItem.__eq__ delegates to numpy, so ``if item == arr``
        raises; defining __eq__ without __hash__ also makes items unhashable."""
        item = DescribedItem(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="ambiguous"):
            bool(item == np.array([1.0, 2.0]))
        with pytest.raises(TypeError):
            hash(item)

    def test_update_dict_contracts(self, tmp_path, no_handlers):
        d = _params(tmp_path, alpha=1.0)
        updateDict(d, ["alpha"], [2.0])
        assert d["alpha"].value == 2.0

        with pytest.raises(ValueError, match="must match"):
            updateDict(d, ["alpha", "t_now"], [1.0])
        with pytest.raises(ValueError, match="values must be provided"):
            updateDict(d, ["alpha"])

    def test_update_dict_dataclass_skips_unknown_fields(self, tmp_path, no_handlers):
        """CANDIDATE-BUG F18 (audit #13) — a field absent from params is
        dropped with no warning: a typo or a missed registration is silent."""
        import dataclasses

        @dataclasses.dataclass
        class Feedback:
            alpha: float
            not_registered: float

        d = _params(tmp_path, alpha=1.0)
        updateDict(d, Feedback(alpha=5.0, not_registered=7.0))
        assert d["alpha"].value == 5.0
        assert "not_registered" not in d

        with pytest.raises(ValueError, match="values must be None"):
            updateDict(d, Feedback(1.0, 2.0), [1.0])

    def test_debug_snapshot_skips_unserializable_and_overwrites(self, tmp_path, no_handlers):
        """save_debug_snapshot's contract — it has no other coverage."""
        d = _params(tmp_path, arr=np.linspace(0.0, 1.0, 4), fn=lambda x: x)
        path = save_debug_snapshot(d, tmp_path)
        payload = json.loads(path.read_text())

        assert payload["_meta"]["type"] == "debug_snapshot"
        assert any("fn" in s for s in payload["_meta"]["skipped_keys"])
        assert payload["arr"] == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]

        d["t_now"].value = 123.0
        again = save_debug_snapshot(d, tmp_path)
        assert again == path  # always overwrites, never appends
        assert json.loads(path.read_text())["t_now"] == 123.0

    def test_reset_keys_ignores_absent_keys(self, tmp_path, no_handlers):
        d = _params(tmp_path, present=1.0)
        d.reset_keys(["present", "never_registered"])
        assert math.isnan(d["present"].value)
        assert "never_registered" not in d
