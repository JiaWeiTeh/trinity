"""Cluster-aware worker resolution and --workers edge-case tests (Phase 1).

Covers trinity/_functions/cpu_allocation.py (allocation detection + the default
worker count) and run.py's --workers guardrails / single-run flag handling,
on both laptop and SLURM-shaped environments.

Unit tests are dependency-free; the CLI subprocess tests exercise the real
``run.py`` entry point (which imports the TRINITY stack) the same way
test_run_smoke.py does.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run
from trinity._functions.cpu_allocation import detect_allocated_cpus, get_optimal_workers

REPO_ROOT = Path(__file__).resolve().parents[1]

_SLURM_VARS = ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE', 'SLURM_JOB_ID')


def _clear_slurm(monkeypatch) -> None:
    for var in _SLURM_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# detect_allocated_cpus: priority tiers
# ---------------------------------------------------------------------------
def test_detect_prefers_slurm_cpus_per_task(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '64')  # lower priority, ignored
    assert detect_allocated_cpus() == (4, 'SLURM_CPUS_PER_TASK')


def test_detect_falls_back_to_cpus_on_node(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setenv('SLURM_CPUS_ON_NODE', '6')
    assert detect_allocated_cpus() == (6, 'SLURM_CPUS_ON_NODE')


def test_detect_uses_affinity_when_no_slurm(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0, 1, 2},
                        raising=False)
    assert detect_allocated_cpus() == (3, 'sched_getaffinity')


def test_detect_falls_back_to_cpu_count(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.delattr(os, 'sched_getaffinity', raising=False)
    monkeypatch.setattr(os, 'cpu_count', lambda: 7)
    assert detect_allocated_cpus() == (7, 'cpu_count')


def test_detect_ignores_blank_slurm_var(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '')  # present but empty
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _pid: {0, 1},
                        raising=False)
    assert detect_allocated_cpus() == (2, 'sched_getaffinity')


# ---------------------------------------------------------------------------
# get_optimal_workers: SLURM job vs laptop
# ---------------------------------------------------------------------------
def test_optimal_workers_uses_full_slurm_allocation(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '8')
    assert get_optimal_workers() == 8


def test_optimal_workers_laptop_is_conservative(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    assert get_optimal_workers() == 3  # max(1, 8 // 2 - 1)


def test_optimal_workers_never_below_one(monkeypatch) -> None:
    _clear_slurm(monkeypatch)
    monkeypatch.setattr(os, 'cpu_count', lambda: 1)
    assert get_optimal_workers() == 1


# ---------------------------------------------------------------------------
# positive_int argparse type
# ---------------------------------------------------------------------------
def test_positive_int_accepts_positive() -> None:
    assert run.positive_int('4') == 4


@pytest.mark.parametrize('bad', ['0', '-1', 'x', '1.5'])
def test_positive_int_rejects(bad) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        run.positive_int(bad)


# ---------------------------------------------------------------------------
# CLI behaviour (subprocess, mirrors test_run_smoke.py)
# ---------------------------------------------------------------------------
def _run_cli(args, *, cwd, env_extra=None, timeout=180):
    env = dict(os.environ)
    for var in _SLURM_VARS:
        env.pop(var, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / 'run.py'), *args],
        cwd=str(cwd), capture_output=True, text=True, input='',
        timeout=timeout, env=env,
    )


def _write_sweep(tmp_path):
    sweep = tmp_path / 'sweep.param'
    sweep.write_text('mCloud [1e5, 1e7]\nsfe 0.01\n')
    return sweep


def test_cli_workers_zero_exits_2(tmp_path) -> None:
    r = _run_cli([str(_write_sweep(tmp_path)), '--local', '--workers', '0'], cwd=tmp_path)
    assert r.returncode == 2
    assert 'must be >= 1' in r.stderr


def test_cli_workers_noninteger_exits_2(tmp_path) -> None:
    r = _run_cli([str(_write_sweep(tmp_path)), '--local', '--workers', 'abc'], cwd=tmp_path)
    assert r.returncode == 2
    assert 'expected an integer' in r.stderr


def test_cli_over_request_refused_before_prompt(tmp_path) -> None:
    r = _run_cli([str(_write_sweep(tmp_path)), '--local', '--workers', '2'], cwd=tmp_path,
                 env_extra={'SLURM_CPUS_PER_TASK': '1'})
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert 'exceeds' in out
    assert 'simulations with' not in out  # never reached the y/N prompt


def test_cli_dry_run_unaffected_by_over_request(tmp_path) -> None:
    r = _run_cli([str(_write_sweep(tmp_path)), '--local', '--dry-run', '--workers', '9999'],
                 cwd=tmp_path, env_extra={'SLURM_CPUS_PER_TASK': '1'})
    out = r.stdout + r.stderr
    assert r.returncode == 0
    assert 'Would run' in out
    assert 'exceeds' not in out


def test_cli_single_file_dry_run_runs_nothing(tmp_path) -> None:
    single = tmp_path / 'single.param'
    single.write_text('mCloud 1e5\nsfe 0.3\n')
    r = _run_cli([str(single), '--local', '--dry-run'], cwd=tmp_path)
    assert r.returncode == 0
    assert 'dry run, nothing executed' in r.stdout
    assert not (tmp_path / 'outputs').exists()
