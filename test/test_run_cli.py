"""run.py CLI restructure — the mode flags + bundle helpers (workstream C).

Exercises build_parser() (mode mutual-exclusion, no-param modes) and
_feed_bundle (reads the submit plan + submitted.tsv, skips done chunks), plus a
subprocess check that the bare form now errors (the hard cutover).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("trinity_run_cli", _REPO / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------
def test_parser_accepts_each_mode():
    p = _load_run().build_parser()
    assert p.parse_args(["x.param", "--local"]).local is True
    a = p.parse_args(["x.param", "--submit", "--throttle", "150", "--chunk", "880"])
    assert a.submit and a.throttle == 150 and a.chunk == 880
    assert p.parse_args(["x.param", "--emit", "/d"]).emit == "/d"


def test_parser_collect_and_resume_need_no_param():
    p = _load_run().build_parser()
    assert p.parse_args(["--collect", "/b"]).collect == "/b"
    assert p.parse_args(["--resume", "/b"]).resume == "/b"


def test_parser_modes_mutually_exclusive():
    p = _load_run().build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["x.param", "--local", "--submit"])


def test_bare_form_errors_hard_cutover(tmp_path):
    """The cutover guarantee: `run.py x.param` with no mode exits non-zero and
    tells the user to choose."""
    pf = tmp_path / "x.param"
    pf.write_text("mCloud 1e6\nsfe 0.01\n")
    r = subprocess.run(
        [sys.executable, str(_REPO / "run.py"), str(pf)],
        capture_output=True, text=True, cwd=str(_REPO),
    )
    assert r.returncode != 0
    assert "choose how to run" in r.stderr


# ---------------------------------------------------------------------------
# _feed_bundle
# ---------------------------------------------------------------------------
def test_feed_bundle_reads_plan_and_skips_done(tmp_path, monkeypatch):
    run = _load_run()
    import trinity._input.cluster_submit as cs

    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "submit_plan.json").write_text(json.dumps({
        "n_jobs": 2640, "throttle": 150, "chunk": 880,
        "sbatch": str(bundle / "submit_sweep.sbatch"), "collect_cmd": "collect",
    }))
    # two chunks already landed -> their offsets must be skipped
    (bundle / "submitted.tsv").write_text("0\t880\t1\n880\t880\t2\n")

    captured = {}

    def fake(**kw):
        captured.update(kw)
        return ([], None)

    monkeypatch.setattr(cs, "feed_and_collect", fake)
    run._feed_bundle(str(bundle))

    assert captured["n_jobs"] == 2640
    assert captured["throttle"] == 150 and captured["chunk"] == 880
    assert set(captured["skip_offsets"]) == {0, 880}
    assert captured["collect_cmd"] == "collect"


def test_feed_bundle_no_progress_file(tmp_path, monkeypatch):
    run = _load_run()
    import trinity._input.cluster_submit as cs

    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "submit_plan.json").write_text(json.dumps({
        "n_jobs": 24, "throttle": None, "chunk": None,
        "sbatch": str(bundle / "s.sbatch"), "collect_cmd": None,
    }))
    captured = {}
    monkeypatch.setattr(cs, "feed_and_collect",
                        lambda **kw: (captured.update(kw) or ([], None)))
    run._feed_bundle(str(bundle))
    assert set(captured["skip_offsets"]) == set()      # nothing done yet
