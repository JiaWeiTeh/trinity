"""HPC submit orchestration — chunking + feeder + auto-collect (workstream C).

Fully scheduler-free: a fake ``sbatch`` runner returns canned stdout/stderr and
``sleep`` is injected, so the retry/feed logic is exercised without SLURM.
"""
from __future__ import annotations

import types

import pytest

from trinity._input import cluster_submit as cs


def _proc(stdout="", stderr="", rc=0):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=rc)


class _FakeSbatch:
    """Pops a canned proc per call; records the commands it was given."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,chunk,expected", [
    (4, None, [(0, 4)]),
    (4, 0, [(0, 4)]),
    (4, 10, [(0, 4)]),                                    # chunk >= n -> single
    (2640, 880, [(0, 880), (880, 880), (1760, 880)]),    # the real paperII case
    (10, 4, [(0, 4), (4, 4), (8, 2)]),                   # ragged last chunk
    (880, 880, [(0, 880)]),                              # exact multiple, one chunk
    (0, 5, []),
])
def test_compute_chunks(n, chunk, expected):
    assert cs.compute_chunks(n, chunk) == expected


def test_parse_job_id_and_queue_full():
    assert cs.parse_job_id("Submitted batch job 12345") == "12345"
    assert cs.parse_job_id("garbage") is None
    assert cs.is_queue_full("sbatch: error: QOSMaxSubmitJobPerUserLimit")
    assert not cs.is_queue_full("Submitted batch job 1")


def test_build_commands():
    assert cs.build_array_command("/x.sbatch", 880, 880, 150) == [
        "sbatch", "--export=OFFSET=880", "--array=1-880%150", "/x.sbatch"]
    assert cs.build_array_command("/x.sbatch", 0, 24, None) == [
        "sbatch", "--export=OFFSET=0", "--array=1-24", "/x.sbatch"]
    assert cs.build_collect_command("99", "python run.py --collect /b") == [
        "sbatch", "--dependency=afterany:99", "--wrap", "python run.py --collect /b"]


# ---------------------------------------------------------------------------
# feed_and_collect
# ---------------------------------------------------------------------------
def test_single_chunk_with_autocollect():
    runner = _FakeSbatch([
        _proc("Submitted batch job 100"),   # the array
        _proc("Submitted batch job 101"),   # the collect
    ])
    waits = []
    submitted, collect = cs.feed_and_collect(
        sbatch_path="/b/submit.sbatch", n_jobs=24, throttle=150, chunk=880,
        collect_cmd="python run.py --collect /b", runner=runner,
        sleep=waits.append, log=lambda *_: None,
    )
    assert submitted == [(0, 24, "100")]
    assert collect == "101"
    assert waits == []                                   # no retries
    assert runner.calls[0] == [
        "sbatch", "--export=OFFSET=0", "--array=1-24%150", "/b/submit.sbatch"]
    assert runner.calls[1][:2] == ["sbatch", "--dependency=afterany:100"]


def test_chunks_with_queue_full_retry():
    runner = _FakeSbatch([
        _proc("Submitted batch job 1"),                       # chunk 0
        _proc(stderr="error: QOSMaxSubmitJobPerUserLimit"),   # chunk 1 full
        _proc(stderr="error: QOSMaxSubmitJobPerUserLimit"),   # chunk 1 full again
        _proc("Submitted batch job 2"),                       # chunk 1 ok
        _proc("Submitted batch job 3"),                       # chunk 2
        _proc("Submitted batch job 9"),                       # collect
    ])
    waits = []
    submitted, collect = cs.feed_and_collect(
        sbatch_path="/b.sbatch", n_jobs=2640, throttle=150, chunk=880,
        collect_cmd="collect", runner=runner, sleep=waits.append,
        retry_wait=300, log=lambda *_: None,
    )
    assert [s[0] for s in submitted] == [0, 880, 1760]   # auto-computed offsets
    assert [s[2] for s in submitted] == ["1", "2", "3"]
    assert collect == "9"
    assert waits == [300, 300]                           # two retry waits, chunk 1


def test_on_submitted_called_per_chunk_for_resume_progress():
    runner = _FakeSbatch([
        _proc("Submitted batch job 1"),
        _proc("Submitted batch job 2"),
        _proc("Submitted batch job 9"),   # collect
    ])
    recorded = []
    cs.feed_and_collect(
        sbatch_path="/b", n_jobs=10, throttle=None, chunk=5, collect_cmd="c",
        on_submitted=lambda off, size, jid: recorded.append((off, size, jid)),
        runner=runner, sleep=lambda *_: None, log=lambda *_: None,
    )
    assert recorded == [(0, 5, "1"), (5, 5, "2")]


def test_no_autocollect_when_disabled():
    runner = _FakeSbatch([_proc("Submitted batch job 5")])
    submitted, collect = cs.feed_and_collect(
        sbatch_path="/b", n_jobs=4, throttle=None, chunk=None, collect_cmd=None,
        runner=runner, sleep=lambda *_: None, log=lambda *_: None,
    )
    assert submitted == [(0, 4, "5")] and collect is None
    assert len(runner.calls) == 1                        # no collect submission


def test_skip_offsets_resume():
    runner = _FakeSbatch([
        _proc("Submitted batch job 2"),    # only chunk at offset 1760
        _proc("Submitted batch job 9"),    # collect
    ])
    submitted, collect = cs.feed_and_collect(
        sbatch_path="/b", n_jobs=2640, throttle=150, chunk=880, collect_cmd="collect",
        skip_offsets=(0, 880), runner=runner, sleep=lambda *_: None, log=lambda *_: None,
    )
    assert submitted == [(1760, 880, "2")]
    assert collect == "9"


def test_hard_error_raises():
    runner = _FakeSbatch([_proc(stderr="error: Invalid partition specified", rc=1)])
    with pytest.raises(RuntimeError, match="sbatch failed"):
        cs.feed_and_collect(
            sbatch_path="/b", n_jobs=4, throttle=None, chunk=None, collect_cmd=None,
            runner=runner, sleep=lambda *_: None, log=lambda *_: None,
        )


def test_max_retries_exceeded_raises():
    runner = _FakeSbatch([_proc(stderr="QOSMaxSubmitJobPerUserLimit")] * 6)
    with pytest.raises(RuntimeError, match="queue still full"):
        cs.feed_and_collect(
            sbatch_path="/b", n_jobs=4, throttle=None, chunk=None, collect_cmd=None,
            runner=runner, sleep=lambda *_: None, max_retries=3, log=lambda *_: None,
        )
