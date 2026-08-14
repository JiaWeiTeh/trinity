"""Harness tests for ``docs/dev/screen/screen.py`` (the multi-config scheme screen).

Two tiers.

**Default suite — logic only, no simulations.** These pin the parts of the screen
that can silently rot: that every config in the screen set still points at a
`.param` that exists, that the interpolation refuses to extrapolate, and that
the fate check actually fires. A screen whose comparison logic is quietly wrong
is worse than no screen, because it reports PASS.

**Opt-in (``-m stress``) — structural pass over the whole config set.** One
short run per config, asserting only invariants that must hold for *any* healthy
run: it completes, snapshots are finite and time-ordered, and it records a
stopping fate. The expensive matched-`t` trajectory comparison is what
`screen.py` itself is for, and is not run here (~5 min per config per arm).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREEN_PY = REPO_ROOT / "docs" / "dev" / "screen" / "screen.py"


def _screen():
    """Import screen.py by path — docs/dev is not a package."""
    spec = importlib.util.spec_from_file_location("screen", SCREEN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


screen = _screen()


# ---------------------------------------------------------------------------
# the screen set points at real files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(screen.CONFIGS))
def test_screen_config_param_exists(name: str) -> None:
    """A moved or renamed .param must fail here, not 40 minutes into a screen."""
    assert (REPO_ROOT / screen.CONFIGS[name]).is_file(), (
        f"screen config {name!r} -> {screen.CONFIGS[name]} does not exist"
    )


def test_screen_set_spans_more_than_the_suite_default() -> None:
    """The whole point: the screen must not be five copies of the one config
    every end-to-end test already uses."""
    assert len(screen.CONFIGS) >= 3
    assert len(set(screen.CONFIGS.values())) == len(screen.CONFIGS)


# ---------------------------------------------------------------------------
# interpolation — the matched-t machinery
# ---------------------------------------------------------------------------
def test_interp_hits_sample_points_exactly() -> None:
    xs, ys = [1.0, 2.0, 4.0], [10.0, 20.0, 40.0]
    assert screen.interp(xs, ys, 1.0) == pytest.approx(10.0)
    assert screen.interp(xs, ys, 4.0) == pytest.approx(40.0)


def test_interp_is_linear_between_samples() -> None:
    assert screen.interp([0.0, 10.0], [0.0, 100.0], 2.5) == pytest.approx(25.0)


@pytest.mark.parametrize("x", [0.5, 5.0])
def test_interp_refuses_to_extrapolate(x: float) -> None:
    """Outside the sampled range returns None. Extrapolating past the end of a
    truncated arm would invent agreement (or disagreement) that was not run."""
    assert screen.interp([1.0, 2.0], [1.0, 2.0], x) is None


def test_interp_handles_empty_series() -> None:
    assert screen.interp([], [], 1.0) is None


# ---------------------------------------------------------------------------
# compare() — the verdict
# ---------------------------------------------------------------------------
def _rows(times_myr, r2s, end_code=1, end_reason="STOPPING_TIME"):
    return [
        {"t_now": t, "R2": r, "SimulationEndCode": end_code, "SimulationEndReason": end_reason}
        for t, r in zip(times_myr, r2s)
    ]


def test_compare_identical_arms_pass_with_zero_shift() -> None:
    rows = _rows([0.001, 0.01, 0.05], [0.5, 1.0, 2.0])
    ledger, ok, worst, last, fb, fa = screen.compare("cfg", rows, rows, bar=5.0)
    assert ok and worst == pytest.approx(0.0)
    assert fb == fa
    assert all(r[-1] == "PASS" for r in ledger)


def test_compare_flags_a_breach_of_the_bar() -> None:
    before = _rows([0.001, 0.01, 0.05], [0.5, 1.0, 2.0])
    after = _rows([0.001, 0.01, 0.05], [0.5, 1.2, 2.0])  # +20% at 1e4 yr
    ledger, ok, worst, *_ = screen.compare("cfg", before, after, bar=5.0)
    assert not ok and worst > 5.0
    assert any(r[2].startswith("dR2_at_1e+04") and r[-1] == "FAIL" for r in ledger)


def test_compare_flags_a_fate_change_even_when_radii_agree() -> None:
    """The load-bearing case: a loose radius bar alone would pass a run that
    collapses when it should not, by comparing at its own truncated endpoint."""
    before = _rows([0.001, 0.01, 0.05], [0.5, 1.0, 2.0], 1, "STOPPING_TIME")
    after = _rows([0.001, 0.01, 0.05], [0.5, 1.0, 2.0], 4, "SHELL_COLLAPSED")
    ledger, ok, worst, *_ = screen.compare("cfg", before, after, bar=5.0)
    assert worst == pytest.approx(0.0)
    assert not ok, "identical radii but a different stopping fate must not pass"
    assert any(r[2] == "stopping_fate" and r[-1] == "FAIL" for r in ledger)


def test_compare_uses_the_last_shared_time_when_arms_truncate_apart() -> None:
    """Arms that stop at different t are compared where both have data."""
    before = _rows([0.001, 0.01, 0.10], [0.5, 1.0, 3.0])
    after = _rows([0.001, 0.01, 0.04], [0.5, 1.0, 2.0])
    _, _, _, last, _, _ = screen.compare("cfg", before, after, bar=5.0)
    assert last == pytest.approx(0.04 * 1e6)


# ---------------------------------------------------------------------------
# fate — where the end record actually lives
# ---------------------------------------------------------------------------
def test_fate_reads_metadata_termination_block(tmp_path: Path) -> None:
    """Real runs stamp the end record in metadata.json[termination], NOT in the
    snapshot rows: a STOPPING_TIME run flushes its last snapshot before main.py
    sets the code, so the jsonl tail carries SimulationEndCode: None (verified
    2026-08-06 on f1edge_hidens). A fate check that only reads the rows is
    vacuous on every such run."""
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"termination": {"exit_code": 1, "outcome": "stopping_time",
                         "detail": "Stopping time reached"}}))
    rows = _rows([0.001, 0.01], [0.5, 1.0], end_code=None, end_reason=None)
    assert screen.fate(rows, str(tmp_path)) == "1 stopping_time"


def test_fate_falls_back_to_rows_then_reports_vacuous(tmp_path: Path) -> None:
    rows_with = _rows([0.001], [0.5], 4, "SHELL_COLLAPSED")
    assert screen.fate(rows_with, str(tmp_path)) == "4 SHELL_COLLAPSED"
    rows_without = _rows([0.001], [0.5], end_code=None, end_reason=None)
    assert screen.fate(rows_without, str(tmp_path)) == "(no stop condition reached)"


# ---------------------------------------------------------------------------
# param rewriting
# ---------------------------------------------------------------------------
def test_write_param_overrides_only_what_the_screen_controls(tmp_path: Path) -> None:
    dst = tmp_path / "p.param"
    screen.write_param(screen.CONFIGS["m43_probe"], str(dst), 0.004, "screen")
    text = dst.read_text()
    keys = [line.split()[0] for line in text.splitlines()
            if line.strip() and not line.startswith("#")]

    assert keys.count("stop_t") == 1 and "stop_t 0.004" in text
    assert "model_name screen" in text
    assert "log_console False" in text
    assert "path2output" not in keys, "run output must land in the screen's own cwd"
    # Prefix collision: stop_t_diss starts with 'stop_t' and is a different
    # setting (shell-dissolution timeout). Dropping it would silently change the
    # physics of every config that sets it.
    assert "stop_t_diss" in keys
    # the rest of the config survives the rewrite
    assert "mCloud" in keys and "nCore" in keys


# ---------------------------------------------------------------------------
# stress tier: every config in the set actually runs
# ---------------------------------------------------------------------------
@pytest.mark.stress
@pytest.mark.parametrize("name", sorted(screen.CONFIGS))
def test_config_runs_and_is_structurally_sound(name: str, tmp_path: Path) -> None:
    """Structural invariants only — no golden values, no cross-arm comparison."""
    param = tmp_path / "p.param"
    screen.write_param(screen.CONFIGS[name], str(param), 0.002, "screen")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(param)],
        cwd=tmp_path, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, (
        f"{name} exited {result.returncode}\n---stderr (tail)---\n{result.stderr[-3000:]}"
    )

    jsonl = tmp_path / "outputs" / "screen" / "dictionary.jsonl"
    assert jsonl.exists(), f"{name} wrote no dictionary.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) >= 2, f"{name} produced {len(rows)} snapshots; integrator likely never ran"

    times = [r["t_now"] for r in rows]
    assert times == sorted(times), f"{name} snapshots are not time-ordered"
    for key in ("R2", "v2", "Eb", "t_now"):
        assert all(math.isfinite(r[key]) for r in rows), f"{name} has non-finite {key}"
    assert all(r["R2"] > 0 for r in rows), f"{name} has a non-positive radius"
    # The end record lives in metadata.json[termination], not the jsonl tail
    # (see test_fate_reads_metadata_termination_block).
    meta = json.loads((tmp_path / "outputs" / "screen" / "metadata.json").read_text())
    term = meta.get("termination") or {}
    assert term.get("exit_code") is not None or term.get("outcome"), (
        f"{name} recorded no stopping fate in metadata.json[termination]"
    )
