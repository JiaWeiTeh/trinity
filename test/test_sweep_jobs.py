"""Job-array generation and result collection tests (Phase 2).

Covers trinity/_input/sweep_jobs.py: emit_jobs() (bundle layout, manifest,
sbatch script, throttle, dry-run, overwrite guard) and collect_report()
(sentinel aggregation, resubmit hint).

Uses a no-dens_profile sweep so combinations are cheap. ``emit_jobs`` still
runs the GMC pre-flight (which pulls numpy via the validator -- a TRINITY core
dependency); ``collect_report`` is dependency-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trinity._input.sweep_parser import (
    read_sweep_config,
    generate_combinations_from_config,
)
from trinity._input.sweep_jobs import emit_jobs, collect_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_sweep(tmp_path):
    """A 2x2 = 4-combo sweep with no density profile."""
    out = tmp_path / 'out'
    sweep = tmp_path / 'sweep.param'
    sweep.write_text(f"mCloud [1e5, 1e7]\nsfe [0.01, 0.10]\npath2output {out}\n")
    return sweep, out


def _emit(tmp_path, **kw):
    sweep, out = _make_sweep(tmp_path)
    cfg = read_sweep_config(str(sweep))
    jobs = tmp_path / 'jobs'
    n_jobs, n_invalid = emit_jobs(cfg, str(out), str(jobs), REPO_ROOT,
                                  sweep_file=str(sweep), **kw)
    return sweep, out.resolve(), jobs, n_jobs, n_invalid


# ---------------------------------------------------------------------------
# emit_jobs
# ---------------------------------------------------------------------------
def test_emit_writes_full_bundle(tmp_path) -> None:
    _sweep, out, jobs, n_jobs, n_invalid = _emit(tmp_path)
    assert (n_jobs, n_invalid) == (4, 0)
    assert len(list((jobs / 'params').glob('*.param'))) == 4
    assert len((jobs / 'runs.tsv').read_text().strip().splitlines()) == 4

    manifest = json.loads((jobs / 'manifest.json').read_text())
    assert manifest['n_jobs'] == 4
    assert len(manifest['runs']) == 4
    assert manifest['base_output_dir'] == str(out)

    sbatch = (jobs / 'submit_sweep.sbatch').read_text()
    assert '#SBATCH --array=1-4' in sbatch
    assert 'export OMP_NUM_THREADS=1' in sbatch
    assert 'export MPLBACKEND=Agg' in sbatch
    assert str(REPO_ROOT / 'run.py') in sbatch


def test_emit_manifest_matches_combinations(tmp_path) -> None:
    sweep, out, jobs, _n, _i = _emit(tmp_path)
    cfg = read_sweep_config(str(sweep))
    expected = [name for _params, name in generate_combinations_from_config(cfg)]

    manifest = json.loads((jobs / 'manifest.json').read_text())
    assert [r['name'] for r in manifest['runs']] == expected
    for run_entry in manifest['runs']:
        param_path = Path(run_entry['param_path'])
        assert param_path.exists()
        assert run_entry['output_dir'] == str(out / run_entry['name'])
        # emitted .param carries the absolute, per-run path2output
        assert (f"path2output    {out / run_entry['name']}"
                in param_path.read_text())


def test_emit_concurrency_sets_array_throttle(tmp_path) -> None:
    _s, _o, jobs, _n, _i = _emit(tmp_path, concurrency=4)
    assert '#SBATCH --array=1-4%4' in (jobs / 'submit_sweep.sbatch').read_text()


def test_emit_sbatch_is_offset_aware(tmp_path) -> None:
    """Emitted sbatch shifts the runs.tsv line by $OFFSET (chunked submission
    under a MaxSubmitJobs cap) and fails loud on an out-of-range line."""
    _s, _o, jobs, _n, _i = _emit(tmp_path)
    sbatch = (jobs / 'submit_sweep.sbatch').read_text()
    assert 'SLURM_ARRAY_TASK_ID + ${OFFSET:-0}' in sbatch  # offset arithmetic
    assert 'no line $N in runs.tsv' in sbatch              # empty-line guard


def test_emit_dry_run_writes_nothing(tmp_path) -> None:
    sweep, out = _make_sweep(tmp_path)
    cfg = read_sweep_config(str(sweep))
    jobs = tmp_path / 'jobs'
    n_jobs, n_invalid = emit_jobs(cfg, str(out), str(jobs), REPO_ROOT,
                                  dry_run=True)
    assert (n_jobs, n_invalid) == (4, 0)
    assert not (jobs / 'manifest.json').exists()


def test_emit_refuses_to_clobber(tmp_path) -> None:
    sweep, out, jobs, _n, _i = _emit(tmp_path)
    cfg = read_sweep_config(str(sweep))
    with pytest.raises(SystemExit):
        emit_jobs(cfg, str(out), str(jobs), REPO_ROOT)


# ---------------------------------------------------------------------------
# collect_report
# ---------------------------------------------------------------------------
def _seed(jobs, codes):
    """Seed sentinels for runs in manifest order; None leaves a run with no
    sentinel ('did not run')."""
    manifest = json.loads((jobs / 'manifest.json').read_text())
    for run_entry, code in zip(manifest['runs'], codes):
        out_dir = Path(run_entry['output_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        if code is not None:
            (out_dir / '.exit_code').write_text(f"{code}\n")
            (out_dir / '.duration').write_text("12\n")


def test_collect_report_aggregates(tmp_path, capsys) -> None:
    _s, out, jobs, _n, _i = _emit(tmp_path)
    _seed(jobs, [0, 0, 1, None])  # 2 ok, 1 failed (code 1), 1 not-run
    collect_report(str(jobs))

    data = json.loads((out / 'sweep_report.json').read_text())
    assert (data['total'], data['succeeded'], data['failed']) == (4, 2, 2)
    assert (out / 'sweep_report.txt').exists()
    assert 'sbatch --array=3,4' in capsys.readouterr().out  # 1-based indices


def test_collect_report_missing_sentinel_is_failure(tmp_path) -> None:
    _s, out, jobs, _n, _i = _emit(tmp_path)
    _seed(jobs, [0, None, None, None])
    collect_report(str(jobs))

    data = json.loads((out / 'sweep_report.json').read_text())
    assert (data['succeeded'], data['failed']) == (1, 3)
    assert sorted(r['return_code'] for r in data['results']) == [-2, -2, -2, 0]


def test_collect_report_requires_manifest(tmp_path) -> None:
    with pytest.raises(SystemExit):
        collect_report(str(tmp_path / 'empty'))


def test_collect_report_breakdown_surfaces_failing_regime(tmp_path, capsys) -> None:
    """Failures should be reported by swept-parameter value (not just array indices), so a
    regime stands out. Grid = mCloud[1e5,1e7] x sfe[0.01,0.10] (manifest order: sfe fastest).
    Fail exactly the two sfe=0.10 runs -> the breakdown must show sfe 0.1: 2 and the -2 note."""
    _s, out, jobs, _n, _i = _emit(tmp_path)
    manifest = json.loads((jobs / 'manifest.json').read_text())
    codes = [None if r['params'].get('sfe') == 0.10 else 0 for r in manifest['runs']]
    _seed(jobs, codes)
    collect_report(str(jobs))

    o = capsys.readouterr().out
    assert 'Failed runs by parameter' in o
    assert 'sfe: 0.1: 2' in o                       # both failures share sfe=0.10
    assert 'no sentinel' in o                       # return-code -2 is explained, not raw
    assert 'mCloud: ' in o                          # mCloud (the other swept axis) is broken out too
    # path2output is per-run (n distinct) -> must NOT be treated as a swept axis
    assert 'path2output:' not in o
