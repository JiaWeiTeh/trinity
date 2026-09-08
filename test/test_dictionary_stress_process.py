"""Process-level characterization tests for the snapshot machinery.

Batteries C (crash handlers / signals) and H (full-run field checks) of
``docs/dev/dictionary-robustness/PLAN.md``. These need **real interpreters**:
battery C is about the atexit and signal handlers that
``DescribedDict.__init__`` registers, which the in-process suite
(``test_dictionary_stress.py``) deliberately disables.

As in that file, every test pins **current** behavior — ``CANDIDATE-BUG F<n>``
marks a pin of a defect, not a desired outcome.

Battery H's real-simulation scans are ``@pytest.mark.stress`` (opt-in, minutes);
the scanner itself is checked in the default set against a synthetic
multi-phase record, so a scanner regression is caught without paying for a run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "docs" / "dev" / "dictionary-robustness" / "harness"

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`, so this
# harness is absent in a fresh clone and in CI. Skip rather than fail at import
# -- a module-scope loader raising here aborts collection for the WHOLE suite.
if not (HARNESS / "scan_field_record.py").is_file():
    pytest.skip(
        "docs/dev is untracked (local-only); dictionary-robustness harness unavailable",
        allow_module_level=True,
    )


def _load_harness(name: str):
    """Import a harness script by path (it is stdlib-only, not a package)."""
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_PREAMBLE = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from trinity._input.dictionary import DescribedDict, DescribedItem
"""


def _run(code: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + code],
        capture_output=True, text=True, timeout=120, **kwargs,
    )


def _writer_code(out: Path, n: int = 1, reason: str | None = None, var: str = "p") -> str:
    """Source for a subprocess that writes ``n`` snapshots into ``out``.

    ``var`` names the dict so one process can build several (battery C's
    handler-stacking case).
    """
    body = f"""
{var} = DescribedDict()
{var}["path2output"] = DescribedItem({str(out)!r})
{var}["t_now"] = DescribedItem(0.0)
{var}["R2"] = DescribedItem(1.0)
for i in range({n}):
    {var}["t_now"].value = 0.1 * i
    {var}["R2"].value = 1.0 + i
    {var}.save_snapshot()
{var}.flush()
"""
    if reason is not None:
        body += f"{var}.set_termination_reason({reason!r})\n"
    return body


def _reason(out: Path) -> str | None:
    meta = json.loads((out / "metadata.json").read_text())
    return meta.get("termination_debug", {}).get("reason")


# =============================================================================
# Battery C — crash handlers and process lifecycle (F7, O5, O6)
# =============================================================================
class TestCrashHandlers:
    def test_loading_a_snapshot_leaves_the_source_run_untouched(self, tmp_path):
        """F7 — **fixed**: loading is side-effect-free (invariant I6).

        Was: ``load_snapshot`` built a dict that registered an atexit handler
        pointed at the *loaded* directory, so an analysis script that only read
        a run rewrote its termination report — clobbering a real crash reason
        with "Normal exit / atexit". Loader-built dicts now skip handler
        registration.
        """
        _run(_writer_code(tmp_path, reason="ODE solver failed")).check_returncode()
        assert _reason(tmp_path) == "ODE solver failed"

        before = {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir()) if p.is_file()}
        _run(f"loaded = DescribedDict.load_snapshot({str(tmp_path)!r}, 0)").check_returncode()
        after = {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir()) if p.is_file()}

        assert after == before, "a read-only load mutated the run directory"
        assert _reason(tmp_path) == "ODE solver failed", "the real reason must survive a load"

    def test_read_only_load_writes_no_humanreadable_file(self, tmp_path):
        """F7 — **fixed**: the same handler used to (re)create this artifact in
        a directory the caller only meant to read."""
        _run(_writer_code(tmp_path)).check_returncode()
        (tmp_path / "metadata_humanreadable.txt").unlink(missing_ok=True)

        _run(f"loaded = DescribedDict.load_snapshot({str(tmp_path)!r}, 0)").check_returncode()

        assert not (tmp_path / "metadata_humanreadable.txt").exists()

    def test_loading_does_not_hijack_signal_handlers(self, tmp_path):
        """F7/O5 — a load also used to take over the *process's* SIGINT and
        SIGTERM handlers, so importing-and-loading inside a larger tool
        silently changed how that tool responded to Ctrl+C."""
        _run(_writer_code(tmp_path)).check_returncode()
        code = f"""
import signal
before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
loaded = DescribedDict.load_snapshot({str(tmp_path)!r}, 0)
after = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
print("UNCHANGED" if before == after else "HIJACKED")
"""
        assert _run(code).stdout.strip() == "UNCHANGED"

    def test_explicit_save_on_a_loaded_dict_still_works(self, tmp_path):
        """Scope guard for the F7 fix: skipping *handler registration* must not
        disable the snapshot machinery itself. A caller that deliberately
        loads, mutates and saves still can — including the destructive
        fresh-run delete of the source, which is finding O1 and stays open
        (plan §6.4); this test pins that the fix did not silently change it.
        """
        _run(_writer_code(tmp_path, n=3)).check_returncode()
        out = tmp_path / "resaved"
        code = f"""
loaded = DescribedDict.load_snapshot({str(tmp_path)!r}, 0)
loaded["path2output"].value = {str(out)!r}
loaded.save_snapshot()
loaded.flush()
"""
        _run(code).check_returncode()
        assert len((out / "dictionary.jsonl").read_text().splitlines()) == 1

    def test_atexit_flushes_pending_snapshots(self, tmp_path):
        """The positive control for the handler's *intended* job: snapshots
        still in the buffer at exit are not lost."""
        code = _writer_code(tmp_path, n=10)  # 10 saves -> flushed
        code += """
p["t_now"].value = 99.0   # an 11th save stays pending in the buffer
p.save_snapshot()
"""
        _run(code).check_returncode()
        lines = (tmp_path / "dictionary.jsonl").read_text().splitlines()
        assert len(lines) == 11, "pending snapshot was not flushed at exit"

    @pytest.mark.parametrize(
        "signame, expected_code",
        [("SIGINT", 130), ("SIGTERM", 143)],
    )
    def test_signal_flushes_and_exits_with_128_plus_signum(
        self, tmp_path, signame, expected_code
    ):
        """O5 — signals flush the buffer and exit 128+signum.

        Also resolves the open question of *which* reason survives: the
        handler writes "Signal <NAME>", then ``sys.exit`` runs the atexit
        handler, which overwrites it with the generic reason. Pinned below.
        """
        ready = tmp_path / "ready"
        code = _writer_code(tmp_path, n=3)
        code += f"""
p["t_now"].value = 42.0
p.save_snapshot()          # pending, must survive the signal
open({str(ready)!r}, "w").close()
import time
time.sleep(60)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", _PREAMBLE + code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.time() + 60
            while not ready.exists() and time.time() < deadline:
                if proc.poll() is not None:
                    pytest.fail(f"writer died early: {proc.communicate()[1]}")
                time.sleep(0.05)
            assert ready.exists(), "writer never became ready"

            proc.send_signal(getattr(signal, signame))
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()

        assert proc.returncode == expected_code
        assert [json.loads(ln)["t_now"] for ln in
                (tmp_path / "dictionary.jsonl").read_text().splitlines()][-1] == 42.0

        # CANDIDATE-BUG O5 — atexit runs after sys.exit and overwrites the
        # signal-specific reason with the generic one.
        assert _reason(tmp_path) == "Normal exit / atexit"

    def test_sigkill_loses_at_most_the_pending_window(self, tmp_path):
        """O2 — SIGKILL cannot be caught, so the buffer is lost; the already
        flushed prefix must still load cleanly (no torn line on a clean
        flush boundary)."""
        ready = tmp_path / "ready"
        code = _writer_code(tmp_path, n=10)  # flushed at 10
        code += f"""
for i in range(4):         # 4 saves left pending (interval is 10)
    p["t_now"].value = 50.0 + i
    p.save_snapshot()
open({str(ready)!r}, "w").close()
import time
time.sleep(60)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", _PREAMBLE + code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.time() + 60
            while not ready.exists() and time.time() < deadline:
                time.sleep(0.05)
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()

        from trinity._input.dictionary import DescribedDict

        snaps = DescribedDict.load_snapshots(tmp_path)
        assert len(snaps) == 10, "flushed prefix should be intact and complete"

    def test_every_dict_registers_its_own_handler(self, tmp_path):
        """O5 — handlers accumulate: two dicts in one process both write a
        termination report at exit, into two different directories."""
        a, b = tmp_path / "a", tmp_path / "b"
        _run(_writer_code(a, var="p") + _writer_code(b, var="q")).check_returncode()

        assert _reason(a) == "Normal exit / atexit"
        assert _reason(b) == "Normal exit / atexit"

    def test_construction_off_the_main_thread_raises(self, tmp_path):
        """O6 — ``signal.signal`` is main-thread only, so a threaded driver
        could not construct one. ``run.py --workers`` uses processes, so this
        is a latent constraint, not a live bug."""
        code = """
import threading
err = []
def make():
    try:
        DescribedDict()
    except Exception as e:
        err.append(type(e).__name__)
t = threading.Thread(target=make)
t.start(); t.join()
print(err[0] if err else "NO-ERROR")
"""
        assert _run(code).stdout.strip() == "ValueError"


# =============================================================================
# Battery H — full-run field checks (invariants I1-I4)
# =============================================================================
# The invariant scanner lives in the workstream harness so the CSV artifact and
# these tests share one implementation (pattern: test_rosette_cf_harness.py).
scan_run_record = _load_harness("scan_field_record").scan_run_record


class TestFieldScanner:
    """Default-set coverage for the scanner itself (no simulation)."""

    def _write_record(self, out: Path, rows: list[dict]) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "dictionary.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )

    def test_clean_record_passes_every_invariant(self, tmp_path):
        rows = [{"t_now": 0.1 * i, "R2": 1.0 + i, "current_phase": "energy"}
                for i in range(12)]
        self._write_record(tmp_path, rows)
        res = scan_run_record(tmp_path)
        assert res["snapshots"] == 12
        assert res["unparsable_lines"] == 0
        assert res["distinct_keysets"] == 1
        assert res["t_non_decreasing"] is True
        assert res["adjacent_dup_indices"] == []
        assert res["nan_bearing_lines"] == 0

    def test_scanner_catches_a_boundary_duplicate(self, tmp_path):
        """The F1 signature the real-run scans look for: a duplicated guard
        key at a flush-cycle boundary."""
        rows = [{"t_now": 0.1 * i, "R2": 1.0 + i, "current_phase": "energy"}
                for i in range(10)]
        rows.append(dict(rows[-1]))  # the boundary duplicate: index 10
        self._write_record(tmp_path, rows)
        res = scan_run_record(tmp_path)
        assert res["adjacent_dup_indices"] == [10]
        assert res["adjacent_dups_off_boundary"] == []

    def test_scanner_catches_schema_drift_and_corruption(self, tmp_path):
        rows = [{"t_now": 0.0, "R2": 1.0}, {"t_now": 1.0, "R2": 2.0, "extra": 1}]
        self._write_record(tmp_path, rows)
        jl = tmp_path / "dictionary.jsonl"
        jl.write_text(jl.read_text() + "{not json\n")
        res = scan_run_record(tmp_path)
        assert res["unparsable_lines"] == 1
        assert res["distinct_keysets"] == 2

    def test_scanner_finds_phase_boundaries(self, tmp_path):
        rows = [{"t_now": float(i), "R2": 1.0, "current_phase": p}
                for i, p in enumerate(["energy"] * 3 + ["implicit"] * 2)]
        self._write_record(tmp_path, rows)
        assert scan_run_record(tmp_path)["phase_boundary_indices"] == [3]


@pytest.mark.stress
class TestRealRunRecord:
    """Battery H proper — scans the record of an actual simulation.

    Opt-in (``pytest -m stress``): each case runs ``run.py`` end to end.
    Committed results live in
    ``docs/dev/dictionary-robustness/data/field_scan.csv``.
    """

    def _simulate(self, tmp_path: Path, param_text: str, model: str) -> Path:
        param = tmp_path / f"{model}.param"
        param.write_text(param_text)
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
            cwd=tmp_path, capture_output=True, text=True, timeout=3600,
        )
        assert result.returncode == 0, f"run failed:\n{result.stdout[-3000:]}"
        run_dir = tmp_path / "outputs" / model
        assert (run_dir / "dictionary.jsonl").exists()
        return run_dir

    def test_smoke_record_holds_the_invariants(self, tmp_path):
        """I1-I4 on the fast config. Known: every line carries NaN (F11),
        because keys the energy phase never populates serialize as NaN."""
        run_dir = self._simulate(
            tmp_path,
            "mCloud      1e5\nsfe         0.3\nstop_t      1e-4\nmodel_name  scanfast\n",
            "scanfast",
        )
        res = scan_run_record(run_dir)

        assert res["unparsable_lines"] == 0                       # I1
        assert res["distinct_keysets"] == 1                       # I4
        assert res["t_non_decreasing"] is True                    # I3
        assert res["adjacent_dups_off_boundary"] == [], (
            "a duplicate away from a flush boundary is a NEW finding — "
            "the guard should catch those in-window"
        )
        assert res["nan_bearing_lines"] == res["snapshots"]       # F11, field-confirmed

    def test_multiphase_record_holds_the_invariants(self, tmp_path):
        """The same scan on a config that crosses phase boundaries — where
        F4 (guard drops each new phase's first save) is observable."""
        param = (REPO_ROOT / "param" / "simple_cluster.param").read_text()
        param = param.replace("simple_cluster", "scanmulti")
        run_dir = self._simulate(tmp_path, param, "scanmulti")
        res = scan_run_record(run_dir)

        assert res["unparsable_lines"] == 0
        assert res["t_non_decreasing"] is True
        assert res["adjacent_dups_off_boundary"] == []
        assert len(res["phases"]) > 1, "config did not cross a phase boundary"
