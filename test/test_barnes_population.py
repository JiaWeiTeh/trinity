"""Unit tests for the Barnes 2026 synthetic bubble-population core
(``paper.barnes26._population``).

Pure-synthesis tests: the samplers and the 2D-grid interpolation are exercised
with hand-built grids and tiny on-disk fixtures, so nothing here runs a TRINITY
simulation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from paper.barnes26._population import (
    sample_powerlaw,
    sample_lognormal_truncated,
    build_grid,
    interpolate_bubble,
    synthesize_population,
)
from paper.barnes26._barnes_lib import ir_fraction
from trinity._output.trinity_reader import load_output, find_all_simulations
from trinity._output.run_constants import METADATA_FILENAME

# Keys every per-bubble record must carry (matches _barnes_lib.sample_run_at_age).
_RECORD_KEYS = {
    "name", "t", "R2", "R_IF", "F_rad", "P_HII", "Lbol", "Li",
    "f_neu", "f_ion", "mCluster", "PISM", "mCloud", "rCloud", "tau_IR",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cell(t, R2, Lbol, rCloud, PISM):
    t = np.asarray(t, dtype=float)
    return dict(
        t=t,
        R2=np.asarray(R2, dtype=float),
        R_IF=np.asarray(R2, dtype=float) * 0.5,
        Lbol=np.full(t.size, float(Lbol)),
        Li=np.full(t.size, float(Lbol) * 0.1),
        F_rad=np.full(t.size, float(Lbol) * 1e-3),
        P_HII=np.full(t.size, 1e3),
        f_neu=np.full(t.size, 0.5),
        f_ion=np.full(t.size, 1.0),
        tau_IR=np.full(t.size, 0.1),
        rCloud=float(rCloud),
        PISM=float(PISM),
        t_max=float(t[-1]),
    )


def _grid_2x2():
    logM = np.array([6.0, 7.0])
    sfe = np.array([0.01, 0.1])
    t = [0.0, 1.0, 2.0]
    cells = {
        (0, 0): _cell(t, [1, 10, 20], 1e6, 18.0, 1e4),
        (1, 0): _cell(t, [1, 100, 200], 1e7, 40.0, 1e4),
        (0, 1): _cell(t, [1, 30, 60], 1e6, 18.0, 1e4),
        (1, 1): _cell(t, [1, 300, 600], 1e7, 40.0, 1e4),
    }
    return dict(logM=logM, sfe=sfe, log_sfe=np.log10(sfe),
                cells=cells, single_sfe=False)


def _snaps(R2vals):
    t = [0.0, 0.5, 1.0]
    return [
        dict(t_now=t[i], R2=R2vals[i], R_IF=R2vals[i] * 0.5, Lbol=1e6, Li=1e5,
             F_rad=1e3, P_HII=1e3, shell_fAbsorbedNeu=0.5, shell_fAbsorbedIon=1.0,
             shell_tauKappaRatio=0.1)
        for i in range(3)
    ]


def _write_run(dirpath: Path, snaps, meta):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / "dictionary.jsonl", "w") as fh:
        for s in snaps:
            fh.write(json.dumps(s) + "\n")
    with open(dirpath / METADATA_FILENAME, "w") as fh:
        json.dump(meta, fh)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def test_powerlaw_recovers_slope():
    rng = np.random.default_rng(0)
    M = sample_powerlaw(200_000, 1e4, 1e7, -2.0, rng)
    assert M.min() >= 1e4 - 1 and M.max() <= 1e7 + 1
    edges = np.logspace(4, 7, 25)
    counts, _ = np.histogram(M, bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    dM = np.diff(edges)
    good = counts > 0
    slope = np.polyfit(np.log10(centres[good]),
                       np.log10((counts / dM)[good]), 1)[0]
    assert abs(slope - (-2.0)) < 0.1


def test_powerlaw_alpha_minus_one_loguniform():
    rng = np.random.default_rng(0)
    M = sample_powerlaw(50_000, 1e4, 1e7, -1.0, rng)
    # log-uniform => log10(M) roughly uniform across [4, 7]
    assert M.min() >= 1e4 - 1 and M.max() <= 1e7 + 1
    assert 5.3 < np.mean(np.log10(M)) < 5.7


def test_lognormal_range_terminates_and_degenerate():
    rng = np.random.default_rng(1)
    x = sample_lognormal_truncated(50_000, 0.03, 0.4, 0.001, 0.5, rng)
    assert x.min() >= 0.001 and x.max() <= 0.5
    assert 0.015 < np.median(x) < 0.06
    # A very narrow window must not hang and stays in range.
    y = sample_lognormal_truncated(1000, 0.03, 0.4, 0.0299, 0.0301, rng)
    assert y.min() >= 0.0299 and y.max() <= 0.0301
    # Degenerate window -> constant (single-SFE grid case).
    z = sample_lognormal_truncated(10, 0.03, 0.4, 0.05, 0.05, rng)
    assert np.allclose(z, 0.05)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def test_node_recovery():
    grid = _grid_2x2()
    rec = interpolate_bubble(1.0, 6.0, 0.01, grid)  # exact node (0, 0), age in range
    assert rec is not None
    assert set(rec) == _RECORD_KEYS
    assert abs(rec["R2"] - 10.0) < 1e-6
    assert abs(rec["Lbol"] - 1e6) < 1e-3
    assert abs(rec["rCloud"] - 18.0) < 1e-6
    assert abs(rec["PISM"] - 1e4) < 1e-6
    assert abs(rec["mCluster"] - 0.01 * 1e6) < 1.0       # epsilon * M_input
    assert abs(rec["mCloud"] - 0.99 * 1e6) < 1.0         # (1-epsilon) * M_input


def test_log_midpoint_is_geometric_mean():
    grid = _grid_2x2()
    # midpoint in mass (logM=6.5), sfe at node 0.01: log-space combine of
    # R2 from cells (0,0)=10 and (1,0)=100 -> geometric mean.
    rec = interpolate_bubble(1.0, 6.5, 0.01, grid)
    assert abs(rec["R2"] - math.sqrt(10.0 * 100.0)) < 1e-6


def test_interior_bilinear_is_geometric_mean():
    grid = _grid_2x2()
    # interior point: logM=6.5 (w_m=0.5) and sfe=10**-1.5 (w_s=0.5) -> all four
    # corners weight 0.25 -> log-space combine is the geometric mean of the
    # corner R2 values (10, 100, 30, 300).
    rec = interpolate_bubble(1.0, 6.5, 10.0 ** -1.5, grid)
    assert abs(rec["R2"] - (10.0 * 100.0 * 30.0 * 300.0) ** 0.25) < 1e-6
    assert abs(rec["f_neu"] - 0.5) < 1e-9   # linear combine of equal corners


def test_exclude_age_beyond_lifetime():
    grid = _grid_2x2()
    # all cells have t_max = 2.0
    assert interpolate_bubble(3.0, 6.0, 0.01, grid) is None


def test_no_coverage_returns_none():
    grid = _grid_2x2()
    grid["cells"] = {}  # nothing to interpolate from
    assert interpolate_bubble(1.0, 6.0, 0.01, grid) is None


def test_single_sfe_path():
    logM = np.array([6.0, 7.0])
    sfe = np.array([0.05])
    t = [0.0, 1.0, 2.0]
    cells = {
        (0, 0): _cell(t, [1, 10, 20], 1e6, 18.0, 1e4),
        (1, 0): _cell(t, [1, 100, 200], 1e7, 40.0, 1e4),
    }
    grid = dict(logM=logM, sfe=sfe, log_sfe=np.log10(sfe),
                cells=cells, single_sfe=True)
    rec = interpolate_bubble(1.0, 6.0, 0.05, grid)
    assert rec is not None and abs(rec["R2"] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# build_grid + synthesize_population
# ---------------------------------------------------------------------------
def test_build_grid_from_fixtures(tmp_path):
    base = tmp_path / "sweep"
    _write_run(base / "1e6_sfe010", _snaps([1, 5, 10]),
               dict(mCloud_input=1e6, sfe=0.1, nCore=1e3, rCloud=18.0,
                    PISM=2.9e59, mCloud=9e5, mCluster=1e5))
    _write_run(base / "1e7_sfe010", _snaps([1, 8, 16]),
               dict(mCloud_input=1e7, sfe=0.1, nCore=1e3, rCloud=40.0,
                    PISM=2.9e59, mCloud=9e6, mCluster=1e6))
    outs = [load_output(p) for p in find_all_simulations(base)]
    grid = build_grid(outs)
    assert len(grid["logM"]) == 2
    assert grid["single_sfe"] is True
    assert len(grid["cells"]) == 2
    # cell time-series came through
    any_cell = next(iter(grid["cells"].values()))
    assert any_cell["t"].size == 3 and any_cell["t_max"] == 1.0


def test_build_grid_derives_axes_from_gas_and_cluster(tmp_path):
    base = tmp_path / "sweep2"
    # metadata lacking mCloud_input/sfe -> must derive from mCloud + mCluster
    _write_run(base / "runA", _snaps([1, 5, 10]),
               dict(nCore=1e3, rCloud=18.0, PISM=2.9e59, mCloud=9e5, mCluster=1e5))
    outs = [load_output(p) for p in find_all_simulations(base)]
    grid = build_grid(outs)
    assert len(grid["cells"]) == 1
    assert abs(grid["logM"][0] - 6.0) < 0.01          # mCloud_input = 1e6
    assert abs(grid["sfe"][0] - 0.1) < 1e-4           # 1e5 / 1e6


def test_synthesize_population_smoke(tmp_path):
    base = tmp_path / "sweep3"
    _write_run(base / "1e6_sfe010", _snaps([1, 5, 10]),
               dict(mCloud_input=1e6, sfe=0.1, nCore=1e3, rCloud=18.0,
                    PISM=2.9e59, mCloud=9e5, mCluster=1e5, dust_KappaIR=8.35e-4))
    _write_run(base / "1e7_sfe010", _snaps([1, 8, 16]),
               dict(mCloud_input=1e7, sfe=0.1, nCore=1e3, rCloud=40.0,
                    PISM=2.9e59, mCloud=9e6, mCluster=1e6, dust_KappaIR=8.35e-4))
    outs = [load_output(p) for p in find_all_simulations(base)]
    records, info = synthesize_population(
        outs, t_obs=0.8, n_bubble=500, cmf_slope=-1.7, seed=7)
    assert info["n_surviving"] == len(records)
    assert len(records) > 400          # ages all < 0.8 < t_max=1.0 -> few/none excluded
    assert set(records[0]) == _RECORD_KEYS
    M_in = np.array([r["mCluster"] / 0.1 for r in records])  # epsilon=0.1 (single-SFE)
    assert M_in.min() >= 1e6 - 1 and M_in.max() <= 1e7 + 1
    # tau_IR plumbed through (dust_KappaIR present) -> finite IR fraction in [0, 1)
    f = ir_fraction(records)
    assert np.all(np.isfinite(f)) and f.min() >= 0.0 and f.max() < 1.0


def test_synthesize_multi_pism_environments(tmp_path):
    base = tmp_path / "sweep_pism"
    # 2 mCloud x 2 PISM (single SFE) -> two environments, each a 2-mass grid.
    for M, mcl, mgas in [(1e6, 1e5, 9e5), (1e7, 1e6, 9e6)]:
        for pism in (2.9e59, 2.9e61):
            _write_run(base / f"{M:.0e}_p{pism:.0e}", _snaps([1, 5, 10]),
                       dict(mCloud_input=M, sfe=0.1, nCore=1e3, rCloud=20.0,
                            PISM=pism, mCloud=mgas, mCluster=mcl))
    outs = [load_output(p) for p in find_all_simulations(base)]
    records, info = synthesize_population(outs, t_obs=0.8, n_bubble=400, seed=3)
    assert info["n_environments"] == 2
    # the combined population spans BOTH ambient pressures; cluster the values
    # by relative closeness (the log-space combine leaves only ~1e-15 round-off).
    uniq = []
    for p in sorted(r["PISM"] for r in records):
        if not uniq or not np.isclose(p, uniq[-1], rtol=1e-6):
            uniq.append(p)
    assert len(uniq) == 2
    assert np.isclose(uniq[0], 2.9e59, rtol=1e-6)
    assert np.isclose(uniq[1], 2.9e61, rtol=1e-6)
    # split roughly evenly across environments
    assert len(info["per_env_counts"]) == 2
    assert min(info["per_env_counts"]) > 0


def test_ir_fraction():
    # f_IR = tau_IR / (1 + tau_IR)
    f = ir_fraction([{"tau_IR": 0.0}, {"tau_IR": 1.0}, {"tau_IR": 400.0}])
    assert np.isclose(f[0], 0.0)
    assert np.isclose(f[1], 0.5)
    assert f[2] > 0.99
    # NaN tau_IR (e.g. dust_KappaIR absent) -> NaN, so it drops out of the plot
    assert np.isnan(ir_fraction([{"tau_IR": float("nan")}])[0])

