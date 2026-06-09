#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic bubble-population synthesis for the Barnes 2026 comparison.

Builds a synthetic *population* of feedback bubbles from a grid of TRINITY
runs, so the Barnes figures can show a realistic spread instead of one marker
per run. The idea (after Watkins+2023-style population synthesis):

  * sample a cloud mass M_cloud from a cloud mass function dN/dM ~ M^slope;
  * sample a star-formation efficiency epsilon (log-normal) — or hold it fixed;
  * give each bubble a random age in [0, t_obs];
  * interpolate that bubble's state on the 2D (M_cloud, epsilon) TRINITY grid
    at its age;
  * exclude bubbles whose age exceeds the simulated lifetime of the runs they
    interpolate from (conservative: never extrapolate past what was run).

The per-bubble records carry the **same keys** as
:func:`paper.barnes26._barnes_lib.sample_run_at_age`, so the existing plotting
code consumes a population and a per-run set identically.

This module is pure synthesis — no plotting, no CLI (those live in the figure
scripts). ``build_grid`` consumes ``TrinityOutput`` objects (e.g. from
``_barnes_lib.load_runs``); everything below it operates on plain arrays and is
unit-testable without TRINITY runs.

Mass convention
---------------
The cloud mass function samples ``mCloud_input`` — the *total* (pre-SFE) cloud
mass — so the cluster mass M_star = epsilon * mCloud_input emerges as a
prediction. The post-SFE gas mass (used for Sigma_gas) is (1-epsilon) *
mCloud_input. Grid axes are read from run metadata (``mCloud_input``/``sfe``),
falling back to ``mCloud + mCluster`` for older metadata that predates those
keys.
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def sample_powerlaw(N, M_min, M_max, alpha, rng):
    """Draw ``N`` samples from a truncated power law dN/dM ~ M^alpha.

    Inverse-CDF sampling on ``[M_min, M_max]``. Handles the ``alpha == -1``
    (log-uniform) singular case separately.
    """
    u = rng.uniform(0.0, 1.0, size=N)
    ap1 = alpha + 1.0
    if abs(ap1) < 1e-12:
        lo, hi = math.log(M_min), math.log(M_max)
        return np.exp(lo + u * (hi - lo))
    return (M_min ** ap1 + u * (M_max ** ap1 - M_min ** ap1)) ** (1.0 / ap1)


def sample_lognormal_truncated(N, median, sigma_dex, lo, hi, rng, max_iter=1000):
    """Draw ``N`` samples from a log-normal truncated to ``[lo, hi]``.

    ``sigma_dex`` is the width in dex (std of log10). Rejection sampling with a
    hard iteration cap; any shortfall after ``max_iter`` is filled by clipping
    fresh draws into range (so a very narrow window can never hang). A
    degenerate window (``lo >= hi``) returns the clipped median for every
    sample — the right behaviour for a single-SFE grid.
    """
    if not (hi > lo):
        return np.full(N, float(np.clip(median, lo, hi)))
    mu_ln = math.log(median)
    sigma_ln = sigma_dex * math.log(10.0)
    out = np.empty(N)
    n = 0
    for _ in range(max_iter):
        if n >= N:
            break
        cand = rng.lognormal(mu_ln, sigma_ln, size=2 * (N - n))
        ok = cand[(cand >= lo) & (cand <= hi)]
        take = min(len(ok), N - n)
        out[n:n + take] = ok[:take]
        n += take
    if n < N:
        out[n:] = np.clip(rng.lognormal(mu_ln, sigma_ln, size=N - n), lo, hi)
    return out


# ---------------------------------------------------------------------------
# 2D grid interpolation
# ---------------------------------------------------------------------------
# Quantities interpolated in age then combined across grid corners.
_TS_LOG = ("R2", "R_IF", "Lbol", "Li", "F_rad", "P_HII")   # positive -> log space
_TS_LIN = ("f_neu", "f_ion")                               # in [0,1] -> linear
_SCALAR_LOG = ("rCloud", "PISM")                           # per-run constants


def _bracket(x, axis):
    """Bracket ``x`` in sorted ``axis``; return (idx_lo, idx_hi, weight in [0,1]).

    Clamps to the endpoints for ``x`` outside the grid (no extrapolation).
    """
    n = len(axis)
    idx = int(np.searchsorted(axis, x))
    if idx <= 0:
        return 0, 0, 0.0
    if idx >= n:
        return n - 1, n - 1, 0.0
    span = axis[idx] - axis[idx - 1]
    if span <= 0:
        return idx - 1, idx - 1, 0.0
    return idx - 1, idx, (x - axis[idx - 1]) / span


def _combine(vals, weights, log):
    """Weighted combine of corner values; log-space for positive quantities.

    Non-finite (and, in log mode, non-positive) corners drop out and their
    weight is renormalised away. Returns NaN if nothing is usable (0.0 in log
    mode when every corner is non-positive).
    """
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if log:
        m = np.isfinite(vals) & (vals > 0)
        if not m.any():
            return 0.0
        w = weights[m]
        return float(10.0 ** (np.sum(w * np.log10(vals[m])) / np.sum(w)))
    m = np.isfinite(vals)
    if not m.any():
        return float("nan")
    w = weights[m]
    return float(np.sum(w * vals[m]) / np.sum(w))


def _age_value(cell, q, age):
    """Value of time-series ``q`` in ``cell`` at ``age`` (clamped to its range)."""
    t = cell["t"]
    if t.size == 0:
        return float("nan")
    age_c = min(max(age, float(t[0])), float(t[-1]))
    return float(np.interp(age_c, t, cell[q]))


def interpolate_bubble(age, log_Mcl, sfe, grid):
    """Interpolate one bubble's state at ``age`` on the (logM, sfe) grid.

    Returns a record with the ``sample_run_at_age`` key set, or ``None`` if the
    bubble can't be placed: no covering grid cell, or its ``age`` exceeds the
    simulated ``t_max`` of a contributing cell (the exclude rule).
    """
    cells = grid["cells"]
    i_lo, i_hi, w_m = _bracket(log_Mcl, grid["logM"])
    if grid["single_sfe"]:
        j_lo, j_hi, w_s = 0, 0, 0.0
    else:
        j_lo, j_hi, w_s = _bracket(math.log10(sfe), grid["log_sfe"])

    corners = [
        ((i_lo, j_lo), (1.0 - w_m) * (1.0 - w_s)),
        ((i_hi, j_lo), w_m * (1.0 - w_s)),
        ((i_lo, j_hi), (1.0 - w_m) * w_s),
        ((i_hi, j_hi), w_m * w_s),
    ]
    present = [(ij, w) for ij, w in corners if w > 1e-12 and ij in cells]
    if not present:
        return None
    if age > min(cells[ij]["t_max"] for ij, _ in present):
        return None

    weights = [w for _, w in present]
    out = {}
    for q in _TS_LOG:
        out[q] = _combine([_age_value(cells[ij], q, age) for ij, _ in present],
                          weights, log=True)
    for q in _TS_LIN:
        out[q] = _combine([_age_value(cells[ij], q, age) for ij, _ in present],
                          weights, log=False)
    for q in _SCALAR_LOG:
        out[q] = _combine([cells[ij][q] for ij, _ in present], weights, log=True)

    m_input = 10.0 ** log_Mcl
    return dict(
        name="synthetic", t=float(age),
        R2=out["R2"], R_IF=out["R_IF"], F_rad=out["F_rad"], P_HII=out["P_HII"],
        Lbol=out["Lbol"], Li=out["Li"], f_neu=out["f_neu"], f_ion=out["f_ion"],
        mCluster=sfe * m_input, PISM=out["PISM"],
        mCloud=(1.0 - sfe) * m_input, rCloud=out["rCloud"],
    )


# ---------------------------------------------------------------------------
# Grid construction from TRINITY runs
# ---------------------------------------------------------------------------
def _grid_axes(m_input, sfe_val, md):
    """Return (mCloud_input, sfe) from metadata, deriving from gas+cluster if
    the explicit keys are absent (older metadata). Returns (None, None) on
    insufficient metadata."""
    if m_input is not None and sfe_val is not None:
        return float(m_input), float(sfe_val)
    m_gas = md.get("mCloud")
    m_cl = md.get("mCluster")
    if m_gas is None or m_cl is None:
        return None, None
    total = float(m_gas) + float(m_cl)
    if total <= 0:
        return None, None
    return total, float(m_cl) / total


def build_grid(outputs, fixed_ncore=None, logM_round=2, sfe_round=4):
    """Build a 2D (log10 mCloud_input, sfe) grid of per-quantity time-series.

    Parameters
    ----------
    outputs : list
        ``TrinityOutput`` objects (need ``.metadata``, ``.get``, ``.filepath``).
    fixed_ncore : float, optional
        Keep only runs whose ``nCore`` matches within 10% (sweeps should be a
        single core density).

    Returns
    -------
    dict
        ``{"logM", "sfe", "log_sfe", "cells", "single_sfe"}`` where ``cells``
        maps ``(i, j)`` -> per-cell dict of time-series arrays + scalars.
    """
    rows = []
    for o in outputs:
        md = o.metadata
        m_input, sfe_val = _grid_axes(md.get("mCloud_input"), md.get("sfe"), md)
        name = o.filepath.parent.name
        if m_input is None or sfe_val is None:
            logger.warning("skip %s: insufficient mass metadata", name)
            continue
        nCore = md.get("nCore")
        if (fixed_ncore is not None and nCore is not None
                and abs(float(nCore) - fixed_ncore) / fixed_ncore > 0.10):
            continue

        t = np.asarray(o.get("t_now"), dtype=float)
        if t.size == 0:
            logger.warning("skip %s: no snapshots", name)
            continue
        order = np.argsort(t)
        t_sorted = t[order]
        _, uidx = np.unique(t_sorted, return_index=True)

        def ser(key):
            return np.asarray(o.get(key), dtype=float)[order][uidx]

        cell = dict(
            t=t_sorted[uidx],
            R2=ser("R2"), R_IF=ser("R_IF"), Lbol=ser("Lbol"), Li=ser("Li"),
            F_rad=ser("F_rad"), P_HII=ser("P_HII"),
            f_neu=ser("shell_fAbsorbedNeu"), f_ion=ser("shell_fAbsorbedIon"),
            rCloud=float(md.get("rCloud") if md.get("rCloud") is not None else np.nan),
            PISM=float(md.get("PISM") if md.get("PISM") is not None else np.nan),
        )
        cell["t_max"] = float(cell["t"][-1])
        rows.append((round(math.log10(m_input), logM_round),
                     round(sfe_val, sfe_round), cell, name))

    if not rows:
        logger.warning("build_grid: no usable runs")
        return dict(logM=np.array([]), sfe=np.array([]),
                    log_sfe=np.array([]), cells={}, single_sfe=True)

    logM = sorted({r[0] for r in rows})
    sfe_axis = sorted({r[1] for r in rows})
    logM_idx = {v: i for i, v in enumerate(logM)}
    sfe_idx = {v: j for j, v in enumerate(sfe_axis)}

    cells = {}
    for logm, sfe_v, cell, name in rows:
        key = (logM_idx[logm], sfe_idx[sfe_v])
        if key in cells:
            logger.warning("duplicate grid cell for %s (logM=%.2f, sfe=%.4f) "
                           "— keeping the first", name, logm, sfe_v)
            continue
        cells[key] = cell

    n_total = len(logM) * len(sfe_axis)
    logger.info("grid: %d logM x %d sfe = %d cells, %d filled",
                len(logM), len(sfe_axis), n_total, len(cells))
    if len(logM) < 3:
        logger.warning("grid spans only %d mass value(s); interpolation will be "
                        "coarse — run a denser (M_cloud x SFE) sweep", len(logM))

    return dict(
        logM=np.array(logM, dtype=float),
        sfe=np.array(sfe_axis, dtype=float),
        log_sfe=np.log10(np.array(sfe_axis, dtype=float)),
        cells=cells,
        single_sfe=(len(sfe_axis) == 1),
    )


# ---------------------------------------------------------------------------
# Population synthesis
# ---------------------------------------------------------------------------
def synthesize_population(
    outputs,
    *,
    t_obs: float = 5.0,
    n_bubble: int = 20000,
    cmf_slope: float = -1.7,
    m_range: Optional[Tuple[float, float]] = None,
    sfe_median: float = 0.03,
    sfe_sigma_dex: float = 0.4,
    fixed_sfe: Optional[float] = None,
    fixed_ncore: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    """Synthesize a bubble population from a grid of TRINITY runs.

    Returns ``(records, info)``: ``records`` is a list of per-bubble dicts with
    the ``sample_run_at_age`` key set (so the figure scripts plot them as they
    would per-run records); ``info`` carries the sampling parameters and counts.
    """
    grid = build_grid(outputs, fixed_ncore=fixed_ncore)
    if not grid["cells"]:
        return [], dict(t_obs=t_obs, n_bubble=n_bubble, n_surviving=0,
                        n_excluded=0, cmf_slope=cmf_slope, reason="empty grid")

    rng = np.random.default_rng(seed)
    logM = grid["logM"]
    if m_range is None:
        m_min, m_max = 10.0 ** float(logM[0]), 10.0 ** float(logM[-1])
    else:
        m_min, m_max = m_range

    M = sample_powerlaw(n_bubble, m_min, m_max, cmf_slope, rng)
    if fixed_sfe is not None:
        sfe = np.full(n_bubble, float(fixed_sfe))
    else:
        s_lo, s_hi = float(grid["sfe"][0]), float(grid["sfe"][-1])
        if grid["single_sfe"]:
            logger.info("single-SFE grid: holding epsilon = %.4g for all bubbles", s_lo)
        sfe = sample_lognormal_truncated(n_bubble, sfe_median, sfe_sigma_dex,
                                         s_lo, s_hi, rng)
    ages = rng.uniform(0.0, t_obs, size=n_bubble)
    log_M = np.log10(M)

    records: List[Dict] = []
    n_excluded = 0
    for k in range(n_bubble):
        rec = interpolate_bubble(float(ages[k]), float(log_M[k]), float(sfe[k]), grid)
        if rec is None:
            n_excluded += 1
            continue
        records.append(rec)

    info = dict(
        t_obs=t_obs, n_bubble=n_bubble, n_surviving=len(records),
        n_excluded=n_excluded, cmf_slope=cmf_slope, m_range=(m_min, m_max),
        sfe_median=sfe_median, sfe_sigma_dex=sfe_sigma_dex,
        fixed_sfe=fixed_sfe, seed=seed,
    )
    logger.info("population: drew %d, excluded %d (age > run t_max / no coverage), "
                "%d surviving", n_bubble, n_excluded, len(records))
    return records, info


def add_population_cli(parser):
    """Register the shared ``--population`` synthesis options on an argparse parser.

    Co-located with the synthesis so both figure scripts share one definition
    (defaults match :func:`synthesize_population`).
    """
    parser.add_argument("--population", action=argparse.BooleanOptionalAction, default=True,
                        help="Synthesize a bubble population at a single --t-obs (default). "
                             "--no-population reverts to per-run markers at fixed --ages.")
    parser.add_argument("--t-obs", type=float, default=5.0,
                        help="Observation time [Myr] for the population (default: 5).")
    parser.add_argument("--n-bubble", type=int, default=20000,
                        help="Number of synthetic bubbles (default: 20000).")
    parser.add_argument("--cmf-slope", type=float, default=-1.7,
                        help="Cloud mass function slope dN/dM ~ M^slope (default: -1.7).")
    parser.add_argument("--sfe-median", type=float, default=0.03,
                        help="SFE log-normal median (default: 0.03).")
    parser.add_argument("--sfe-sigma-dex", type=float, default=0.4,
                        help="SFE log-normal width [dex] (default: 0.4).")
    parser.add_argument("--fixed-sfe", type=float, default=None,
                        help="Hold SFE fixed at this value (overrides the SFE distribution).")
    parser.add_argument("--fixed-ncore", type=float, default=None,
                        help="Use only grid runs with this core density.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Population RNG seed (default: 42).")
