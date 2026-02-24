#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Population-convolved bubble size distribution from TRINITY outputs.

Physics background
------------------
Watkins et al. (2023) measured the bubble size distribution in NGC 628
and found dN/dR proportional to R^{-2.2}.  A single decelerating shell
always gives dN/dR proportional to 1/v(R) with *positive* slope (v
decreases with R so 1/v increases).  The observed *negative* slope
arises from a **population** of bubbles driven by clusters spanning a
luminosity / mass function: many low-mass clusters make small bubbles,
few high-mass clusters make large ones.

This script tests the population picture by convolving a grid of TRINITY
trajectories with a cluster mass function (CMF):

    dN/dR = integral over M_cloud [ n(M_cloud) * (dt/dR)|_{M_cloud} ] dM_cloud

In practice we use Monte Carlo population synthesis on the discrete
TRINITY grid.

**Part 1 — Single-trajectory residence time (diagnostic):**
    For each TRINITY run, compute 1/v(R) — the time a *single* shell
    lingers at each radius.  Positive slopes are expected; this is a
    diagnostic, not a prediction of the observed size distribution.

**Part 2 — Population synthesis:**
    Sample N_bubble cluster masses from dN/dM_star proportional to
    M_star^alpha (Lada & Lada 2003, default alpha = -2).  Match each
    to the closest TRINITY run in M_star = sfe * M_cloud.  Assign a
    random birth time, evaluate R(age), histogram the surviving R
    values, and fit a power-law slope to compare with -2.2.

**Part 3 — Parameter sensitivity:**
    Vary the CMF slope and t_obs to show how the synthetic slope depends
    on assumptions.

References
----------
* Watkins, E. J. et al. (2023), ApJS, 264, 16 — NGC 628 bubble catalogue.
* Lada, C. J. & Lada, E. A. (2003), ARA&A, 41, 57 — embedded clusters.
* Weaver, R. et al. (1977), ApJ, 218, 377 — wind-bubble expansion.
* Lancaster, L. et al. (2021), ApJ, 914, 89 — size distribution theory.

CLI usage
---------
    python bubble_distribution.py -F /path/to/sweep_output
    python bubble_distribution.py -F /path/to/sweep_output --N-bubble 5000 --fmt png
"""

import sys
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    parse_simulation_params,
)
from src._functions.unit_conversions import INV_CONV

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root, matching other _calc scripts
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Constants
# ======================================================================

V_AU2KMS = INV_CONV.v_au2kms   # pc/Myr -> km/s  (~0.978)

# Observed power-law index from Watkins+2023
OBSERVED_ALPHA = -2.2

# Alternative completeness turnover from Nath+2020
R_COMPLETE_NATH20 = 100.0

# Lower completeness cut at 10 pc
R_COMPLETE_10PC = 10.0

# Minimum expanding points required for analysis
MIN_PTS = 10

# Colourblind-safe palette (Wong 2011)
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_BLACK = "#000000"

# Line styles for redundant encoding
LINESTYLES = ["-", "--", "-.", ":"]


# ======================================================================
# Data extraction
# ======================================================================

def extract_bubble_data(data_path: Path, t_end: float = None) -> Optional[Dict]:
    """
    Load one TRINITY run and extract R(t) and v(R) for the expanding phase.

    Parameters
    ----------
    data_path : Path
        Path to dictionary.jsonl.
    t_end : float, optional
        If given, truncate time series at this value [Myr].

    Returns
    -------
    dict or None
        Keys: t, R, v_au, v_kms, phase, outcome, folder.
        All arrays are ordered by time and restricted to expanding portions.
    """
    try:
        output = load_output(data_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", data_path, e)
        return None

    if len(output) < MIN_PTS:
        logger.warning("Too few snapshots (%d) in %s — skipping",
                        len(output), data_path)
        return None

    t = output.get("t_now")
    R = output.get("R2")
    v_au = output.get("v2")                 # pc/Myr
    phase = np.array(output.get("current_phase", as_array=False))

    # Replace NaN
    v_au = np.nan_to_num(v_au, nan=0.0)
    R = np.nan_to_num(R, nan=0.0)

    # Truncate at t_end if requested
    _truncated = False
    if t_end is not None and t[-1] > t_end:
        mask_t = t <= t_end
        if mask_t.sum() < MIN_PTS:
            logger.info("Fewer than %d snapshots within t_end=%.3f in %s — skip",
                        MIN_PTS, t_end, data_path.parent.name)
            return None
        t = t[mask_t]
        R = R[mask_t]
        v_au = v_au[mask_t]
        phase = phase[mask_t]
        _truncated = True

    # Outcome
    if _truncated:
        outcome = "expand" if v_au[-1] > 0 else "collapse"
    else:
        last = output[-1]
        is_collapse = last.get("isCollapse", False)
        is_dissolved = last.get("isDissolved", False)
        end_reason = str(last.get("SimulationEndReason", "")).lower()
        if is_dissolved or "dissolved" in end_reason or "large radius" in end_reason:
            outcome = "expand"
        elif is_collapse or "small radius" in end_reason:
            outcome = "collapse"
        else:
            outcome = "stalled"

    # Only keep expanding portions (v > 0, R > 0) for bubble analysis
    expanding = (v_au > 0) & (R > 0)
    if expanding.sum() < MIN_PTS:
        logger.debug("Too few expanding points in %s — skipping",
                      data_path.parent.name)
        return None

    return {
        "t": t,
        "R": R,
        "v_au": v_au,
        "v_kms": v_au * V_AU2KMS,
        "phase": phase,
        "expanding": expanding,
        "outcome": outcome,
        "folder": data_path.parent.name,
        # Full-run t and R for interpolation in synthesis
        "t_full": t,
        "R_full": R,
    }


def collect_data(folder_path: Path, t_end: float = None) -> List[Dict]:
    """
    Walk sweep directory and extract bubble data from each run.

    Parameters
    ----------
    folder_path : Path
        Top-level sweep output directory.
    t_end : float, optional
        Maximum time [Myr] to consider.

    Returns
    -------
    list of dict
        One record per valid simulation run.
    """
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        logger.error("No simulation files under %s", folder_path)
        return []

    logger.info("Found %d simulation(s) in %s", len(sim_files), folder_path)

    records: List[Dict] = []
    for data_path in sim_files:
        folder_name = data_path.parent.name
        parsed = parse_simulation_params(folder_name)
        if parsed is None:
            logger.warning("Cannot parse '%s' — skipping", folder_name)
            continue

        nCore = float(parsed["ndens"])
        mCloud = float(parsed["mCloud"])
        sfe = int(parsed["sfe"]) / 100.0

        info = extract_bubble_data(data_path, t_end=t_end)
        if info is None:
            continue

        # Compute derived quantities for this run
        t = info["t"]
        R = info["R"]
        v_au = info["v_au"]
        expanding = info["expanding"]

        # Extract expanding-only arrays (sorted by R for analysis)
        R_exp = R[expanding]
        v_exp = v_au[expanding]
        t_exp = t[expanding]
        sort_idx = np.argsort(R_exp)
        R_exp = R_exp[sort_idx]
        v_exp = v_exp[sort_idx]
        t_exp = t_exp[sort_idx]

        # Residence time: dN/dR proportional to 1/v(R)
        dNdR_pred = 1.0 / v_exp
        dNdR_norm = dNdR_pred / dNdR_pred[0] if dNdR_pred[0] != 0 else dNdR_pred

        # Fit power law: log(dN/dR) = a + slope * log(R)
        valid = (R_exp > 0) & (v_exp > 0) & np.isfinite(dNdR_pred)
        residence_slope = np.nan
        residence_R2 = np.nan
        if valid.sum() >= MIN_PTS:
            logR = np.log10(R_exp[valid])
            log_dNdR = np.log10(dNdR_pred[valid])
            X = np.column_stack([np.ones_like(logR), logR])
            try:
                beta = np.linalg.lstsq(X, log_dNdR, rcond=None)[0]
                residence_slope = beta[1]
                resid = log_dNdR - X @ beta
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((log_dNdR - log_dNdR.mean()) ** 2)
                residence_R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except np.linalg.LinAlgError:
                pass

        # Inferred N_dot(R) = (dN/dR) * v(R) assuming Watkins+2023 dN/dR
        ndot_unnorm = R_exp ** OBSERVED_ALPHA * v_exp
        if ndot_unnorm[0] != 0:
            ndot_norm = ndot_unnorm / ndot_unnorm[0]
        else:
            ndot_norm = np.full_like(ndot_unnorm, np.nan)

        # Derived stellar mass
        M_star = sfe * mCloud

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "M_star": M_star,
            "folder": info["folder"],
            "outcome": info["outcome"],
            # Expanding arrays sorted by R
            "R_exp": R_exp,
            "v_exp": v_exp,
            "t_exp": t_exp,
            # Part 1 results
            "ndot_norm": ndot_norm,
            "dNdR_norm": dNdR_norm,
            "residence_slope": residence_slope,
            "residence_R2": residence_R2,
            # Full run for synthesis
            "t_full": info["t_full"],
            "R_full": info["R_full"],
        }
        records.append(rec)

    logger.info("Collected %d valid runs", len(records))
    return records


# ======================================================================
# Part 2: Population synthesis (CMF-convolved)
# ======================================================================

def _sample_powerlaw(
    N: int,
    M_min: float,
    M_max: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw N samples from a truncated power-law dN/dM proportional to M^alpha.

    Uses inverse-CDF sampling.

    Parameters
    ----------
    N : int
        Number of samples.
    M_min, M_max : float
        Mass range.
    alpha : float
        Power-law index (e.g. -2.0).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray
        Array of sampled masses.
    """
    u = rng.uniform(0.0, 1.0, size=N)
    ap1 = alpha + 1.0
    if abs(ap1) < 1e-12:
        # alpha == -1: dN/dM proportional to 1/M => log-uniform
        log_min = np.log(M_min)
        log_max = np.log(M_max)
        return np.exp(log_min + u * (log_max - log_min))
    else:
        return (M_min**ap1 + u * (M_max**ap1 - M_min**ap1)) ** (1.0 / ap1)


def _build_interp_grid(
    filtered_records: List[Dict],
) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
    """
    Build a sorted grid of unique M_star values with associated R(t) tracks.

    For duplicate M_star values (from different M_cloud/sfe combinations),
    pick the run with the HIGHEST sfe.  Rationale: higher sfe means more of
    the cloud mass is in the cluster, so the feedback-to-gravity ratio is
    most favourable and the trajectory best represents the "typical" bubble
    from that cluster mass.

    Parameters
    ----------
    filtered_records : list of dict
        Records from collect_data(), already filtered by sfe/ncore.

    Returns
    -------
    grid_logMstar : 1D array, sorted
        log10(M_star) for each unique grid point.
    grid_interp_data : list of dict
        Each dict has keys 't' and 'R' (deduplicated, sorted by t).
    grid_records : list of dict
        Corresponding record dicts (for run-usage tracking).
    """
    from collections import defaultdict

    # Group by rounded log10(M_star) to handle float comparison issues
    groups = defaultdict(list)
    for rec in filtered_records:
        key = round(np.log10(rec["M_star"]), 2)
        groups[key].append(rec)

    # Within each group, pick the record with the highest sfe
    selected = []
    for key in sorted(groups.keys()):
        group = groups[key]
        if len(group) > 1:
            best = max(group, key=lambda r: r["sfe"])
            logger.info(
                "Duplicates resolved: M_star=%.2e had %d runs, picked sfe=%.3f (%s)",
                10.0 ** key, len(group), best["sfe"], best["folder"],
            )
            selected.append(best)
        else:
            selected.append(group[0])

    # Sort by M_star
    selected.sort(key=lambda r: r["M_star"])

    # Build deduplicated interp arrays (t, R) for each
    grid_logMstar = np.array([np.log10(r["M_star"]) for r in selected])
    grid_interp_data = []
    for rec in selected:
        t_full = rec["t_full"]
        R_full = rec["R_full"]
        _, unique_idx = np.unique(t_full, return_index=True)
        grid_interp_data.append({
            "t": t_full[unique_idx],
            "R": R_full[unique_idx],
        })

    # Log the grid
    for rec in selected:
        logger.info(
            "  M_star = %.2e -> %s (sfe=%.3f)",
            rec["M_star"], rec["folder"], rec["sfe"],
        )

    return grid_logMstar, grid_interp_data, selected


def _bracket_on_grid(
    log_Mstar_sampled: float,
    grid_logMstar: np.ndarray,
) -> Tuple[int, int, float]:
    """
    Find the two bracketing grid points for a sampled log10(M_star) and
    compute the interpolation weight.

    Parameters
    ----------
    log_Mstar_sampled : float
        log10 of the sampled cluster mass.
    grid_logMstar : ndarray
        Sorted array of log10(M_star) on the grid.

    Returns
    -------
    idx_lo : int
        Index of the lower bracketing grid point.
    idx_hi : int
        Index of the upper bracketing grid point.
    w : float
        Interpolation weight in [0, 1].
        w = 0 means use idx_lo entirely, w = 1 means use idx_hi entirely.
        For values below/above the grid, clamp to the nearest endpoint.
    """
    n = len(grid_logMstar)
    idx = np.searchsorted(grid_logMstar, log_Mstar_sampled)

    if idx <= 0:
        return 0, 0, 0.0
    elif idx >= n:
        return n - 1, n - 1, 0.0
    else:
        idx_lo = idx - 1
        idx_hi = idx
        span = grid_logMstar[idx_hi] - grid_logMstar[idx_lo]
        if span <= 0:
            return idx_lo, idx_lo, 0.0
        w = (log_Mstar_sampled - grid_logMstar[idx_lo]) / span
        return idx_lo, idx_hi, w


def _interpolate_R(
    age: float,
    interp_lo: Dict,
    interp_hi: Dict,
    w: float,
) -> float:
    """
    Evaluate R(age) by log-interpolating between two grid tracks.

    Parameters
    ----------
    age : float
        Bubble age [Myr].
    interp_lo, interp_hi : dict
        Each has 't' and 'R' arrays for the lower/upper bracketing run.
    w : float
        Interpolation weight (0 = all lo, 1 = all hi).

    Returns
    -------
    R : float
        Interpolated bubble radius [pc].  Returns NaN if R <= 0 on either track.
    """
    # Clamp age to the overlapping time range of both tracks
    t_min = max(interp_lo["t"].min(), interp_hi["t"].min())
    t_max = min(interp_lo["t"].max(), interp_hi["t"].max())
    age_c = np.clip(age, t_min, t_max)

    # Evaluate R on each track
    R_lo = np.interp(age_c, interp_lo["t"], interp_lo["R"])
    R_hi = np.interp(age_c, interp_hi["t"], interp_hi["R"])

    # If either track has R <= 0 at this age (collapsed), return NaN
    if R_lo <= 0 or R_hi <= 0:
        return np.nan

    # Log-space interpolation
    log_R = (1.0 - w) * np.log10(R_lo) + w * np.log10(R_hi)
    return 10.0 ** log_R


def _fit_powerlaw_mle(R_values: np.ndarray, R_min: float
                      ) -> Tuple[float, float, int]:
    """
    Maximum-likelihood estimate for a power-law dN/dR proportional to R^gamma.

    For a Pareto distribution P(R) proportional to R^gamma with R >= R_min,
    the MLE estimator is:
        gamma_hat = -1 - N / sum(ln(R_i / R_min))

    This is equivalent to the Hill estimator (Alstott et al. 2014).

    Parameters
    ----------
    R_values : array
        Bubble radii (only those >= R_min will be used).
    R_min : float
        Minimum radius for the fit (= R_complete).

    Returns
    -------
    gamma : float
        Fitted power-law index.
    gamma_err : float
        Standard error = (gamma + 1) / sqrt(N).
    N_fit : int
        Number of points used.
    """
    R_fit = R_values[R_values >= R_min]
    N = len(R_fit)
    if N < 10:
        return np.nan, np.nan, 0

    # MLE for Pareto: gamma = -1 - N / sum(ln(R/R_min))
    log_ratio = np.log(R_fit / R_min)
    gamma = -1.0 - N / np.sum(log_ratio)
    gamma_err = (gamma + 1.0) / np.sqrt(N)

    return gamma, gamma_err, N


def _fit_powerlaw_mle_upper(R_values: np.ndarray,
                            R_max: float) -> Tuple[float, float, int]:
    """
    MLE power-law fit for R <= R_max (upper-truncated Pareto).

    Transforms S = R_max / R (so S >= 1), applies the Hill estimator to
    obtain alpha_S, then converts back:  gamma = alpha_S - 2.

    Parameters
    ----------
    R_values : ndarray
        Array of radii.
    R_max : float
        Upper cutoff — only R <= R_max are used.

    Returns
    -------
    gamma : float
        Fitted dN/dR power-law index for R <= R_max.
    gamma_err : float
        Standard error.
    N_fit : int
        Number of points used.
    """
    R_fit = R_values[(R_values > 0) & (R_values <= R_max)]
    N = len(R_fit)
    if N < 10:
        return np.nan, np.nan, 0

    # S = R_max / R  =>  S >= 1.   dN/dS ~ S^{-(gamma+2)}.
    # Hill estimator on S:  alpha_S = 1 + N / sum(ln(S_i))
    log_S = np.log(R_max / R_fit)
    alpha_S = 1.0 + N / np.sum(log_S)
    gamma = alpha_S - 2.0            # convert back to dN/dR index
    gamma_err = (alpha_S - 1.0) / np.sqrt(N)

    return gamma, gamma_err, N


def _fit_powerlaw_binned(R_values: np.ndarray, R_min: float,
                         n_bins: int = 60) -> Tuple[float, float, int]:
    """
    Binned log-space least-squares power-law fit for R >= R_min.

    Histograms R in *n_bins* log-spaced bins, converts to dN/dR, then
    fits log(dN/dR) = gamma * log(R) + const via OLS on non-empty bins.

    Parameters
    ----------
    R_values : ndarray
        Array of radii.
    R_min : float
        Lower cutoff — only R >= R_min are used.
    n_bins : int
        Number of log-spaced bins.

    Returns
    -------
    gamma : float
        Fitted power-law index.
    gamma_err : float
        Standard error of the slope from OLS.
    N_fit : int
        Number of non-empty bins used in the fit.
    """
    R_fit = R_values[R_values >= R_min]
    if len(R_fit) < 10:
        return np.nan, np.nan, 0

    log_edges = np.linspace(np.log10(R_fit.min()), np.log10(R_fit.max()),
                            n_bins + 1)
    counts, _ = np.histogram(np.log10(R_fit), bins=log_edges)
    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_c = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    good = dNdR > 0
    N_good = int(good.sum())
    if N_good < 3:
        return np.nan, np.nan, 0

    logR = np.log10(R_c[good])
    logdN = np.log10(dNdR[good])

    # OLS: logdN = gamma * logR + b
    A = np.vstack([logR, np.ones(N_good)]).T
    result = np.linalg.lstsq(A, logdN, rcond=None)
    gamma = result[0][0]
    residuals = logdN - A @ result[0]
    if N_good > 2:
        se2 = np.sum(residuals ** 2) / (N_good - 2)
        var_gamma = se2 / np.sum((logR - logR.mean()) ** 2)
        gamma_err = np.sqrt(var_gamma)
    else:
        gamma_err = np.nan

    return gamma, gamma_err, N_good


def _fit_powerlaw_binned_upper(R_values: np.ndarray, R_max: float,
                               n_bins: int = 60) -> Tuple[float, float, int]:
    """
    Binned log-space least-squares power-law fit for R <= R_max.

    Parameters
    ----------
    R_values : ndarray
        Array of radii.
    R_max : float
        Upper cutoff — only R <= R_max (and R > 0) are used.
    n_bins : int
        Number of log-spaced bins.

    Returns
    -------
    gamma : float
        Fitted power-law index.
    gamma_err : float
        Standard error of the slope from OLS.
    N_fit : int
        Number of non-empty bins used in the fit.
    """
    R_fit = R_values[(R_values > 0) & (R_values <= R_max)]
    if len(R_fit) < 10:
        return np.nan, np.nan, 0

    log_edges = np.linspace(np.log10(R_fit.min()), np.log10(R_max),
                            n_bins + 1)
    counts, _ = np.histogram(np.log10(R_fit), bins=log_edges)
    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_c = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    good = dNdR > 0
    N_good = int(good.sum())
    if N_good < 3:
        return np.nan, np.nan, 0

    logR = np.log10(R_c[good])
    logdN = np.log10(dNdR[good])

    A = np.vstack([logR, np.ones(N_good)]).T
    result = np.linalg.lstsq(A, logdN, rcond=None)
    gamma = result[0][0]
    residuals = logdN - A @ result[0]
    if N_good > 2:
        se2 = np.sum(residuals ** 2) / (N_good - 2)
        var_gamma = se2 / np.sum((logR - logR.mean()) ** 2)
        gamma_err = np.sqrt(var_gamma)
    else:
        gamma_err = np.nan

    return gamma, gamma_err, N_good


def run_population_synthesis(
    records: List[Dict],
    N_bubble: int = 20000,
    t_obs: float = 5.0,
    cmf_slope: float = -2.0,
    R_complete: float = 30.0,
    seed: int = 42,
    fixed_sfe: float = None,
    fixed_ncore: float = None,
) -> Optional[Dict]:
    """
    Population synthesis: sample clusters from a CMF, match to TRINITY
    runs, assign random birth times, histogram surviving radii.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    N_bubble : int
        Number of synthetic bubbles to draw.
    t_obs : float
        Observation time [Myr].
    cmf_slope : float
        CMF power-law index dN/dM_star proportional to M_star^alpha.
    R_complete : float
        Completeness radius [pc] below which bubbles are discarded.
    seed : int
        Random seed.
    fixed_sfe : float, optional
        If set, only use runs with this SFE.
    fixed_ncore : float, optional
        If set, only use runs with this core density.

    Returns
    -------
    dict or None
        Synthesis results including R_synth, fitted slope, etc.
    """
    if not records:
        return None

    # Filter runs if requested
    filtered = records
    if fixed_sfe is not None:
        filtered = [r for r in filtered
                    if abs(r["sfe"] - fixed_sfe) < 1e-6]
    if fixed_ncore is not None:
        filtered = [r for r in filtered
                    if abs(r["nCore"] - fixed_ncore) / fixed_ncore < 0.1]

    if not filtered:
        logger.warning("No runs match fixed_sfe=%s, fixed_ncore=%s",
                        fixed_sfe, fixed_ncore)
        return None

    # Build unique M_star interpolation grid (resolves duplicates)
    grid_logMstar, grid_interp, grid_recs = _build_interp_grid(filtered)

    if len(grid_logMstar) == 0:
        logger.warning("Empty M_star grid after deduplication")
        return None

    M_star_min = 10.0 ** grid_logMstar[0]
    M_star_max = 10.0 ** grid_logMstar[-1]

    if M_star_min <= 0 or M_star_max <= 0:
        logger.warning("Invalid M_star range [%.2e, %.2e]", M_star_min, M_star_max)
        return None

    logger.info("Unique M_star grid: %d points from %.2e to %.2e Msun",
                len(grid_logMstar), M_star_min, M_star_max)

    rng = np.random.default_rng(seed)

    # Step 1: sample cluster masses
    sampled_Mstar = _sample_powerlaw(N_bubble, M_star_min, M_star_max,
                                     cmf_slope, rng)

    # Step 2: assign random birth times
    t_form = rng.uniform(0.0, t_obs, size=N_bubble)
    ages = t_obs - t_form

    # Step 3: evaluate R(age) with log-interpolation between grid tracks
    R_synth = np.full(N_bubble, np.nan)
    run_usage = np.zeros(len(grid_recs), dtype=int)

    log_sampled = np.log10(sampled_Mstar)

    for i in range(N_bubble):
        idx_lo, idx_hi, w = _bracket_on_grid(log_sampled[i], grid_logMstar)
        run_usage[idx_lo] += 1
        if idx_hi != idx_lo:
            run_usage[idx_hi] += 1

        if idx_lo == idx_hi:
            # At grid edge — single track
            td = grid_interp[idx_lo]
            age_c = np.clip(ages[i], td["t"].min(), td["t"].max())
            R_synth[i] = np.interp(age_c, td["t"], td["R"])
        else:
            R_synth[i] = _interpolate_R(ages[i], grid_interp[idx_lo],
                                         grid_interp[idx_hi], w)

    # Step 4: apply cuts — keep ALL R > 0 for histogramming
    valid_mask = np.isfinite(R_synth) & (R_synth > 0)
    recollapsed = np.sum(~valid_mask)
    R_valid = R_synth[valid_mask]

    below_complete = np.sum(R_valid < R_complete)
    N_surviving = int(np.sum(R_valid >= R_complete))
    logger.info("Synthesis: %d drawn, %d recollapsed, %d below R_complete=%.0f pc, "
                "%d surviving",
                N_bubble, recollapsed, below_complete, R_complete, N_surviving)

    if len(R_valid) < MIN_PTS:
        logger.warning("Too few valid bubbles (%d) — skipping synthesis",
                        len(R_valid))
        return None

    # Step 5: display histogram in log-space over the FULL R range
    log_R_all = np.log10(R_valid)
    n_bins_display = 40
    log_edges = np.linspace(log_R_all.min(), log_R_all.max(), n_bins_display + 1)
    counts, _ = np.histogram(log_R_all, bins=log_edges)

    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    # Step 6: fit power law via MLE on raw radii (no binning needed)
    synth_slope, synth_slope_err, N_fit = _fit_powerlaw_mle(R_valid, R_complete)

    # MLE fit for R <= 10 pc (upper-truncated Pareto)
    slope_le10, slope_err_le10, N_fit_le10 = _fit_powerlaw_mle_upper(
        R_valid, R_COMPLETE_10PC)

    # MLE fit with Nath+2020 turnover at 100 pc
    slope_nath, slope_err_nath, N_fit_nath = _fit_powerlaw_mle(
        R_valid, R_COMPLETE_NATH20)

    # Step 7: binned log-space LSQ fits (finer bins, for comparison)
    bin_slope_le10, bin_slope_err_le10, bin_N_le10 = _fit_powerlaw_binned_upper(
        R_valid, R_COMPLETE_10PC)
    bin_slope_30, bin_slope_err_30, bin_N_30 = _fit_powerlaw_binned(
        R_valid, R_complete)
    bin_slope_100, bin_slope_err_100, bin_N_100 = _fit_powerlaw_binned(
        R_valid, R_COMPLETE_NATH20)

    # Build run-usage summary (references unique grid records)
    runs_used_summary = []
    for j, rec in enumerate(grid_recs):
        if run_usage[j] > 0:
            runs_used_summary.append(
                f"{rec['folder']}({run_usage[j]})")

    return {
        "method": "interp",
        "R_synth": R_valid,
        "R_centres": R_centres,
        "dNdR": dNdR,
        "synth_slope": synth_slope,
        "synth_slope_err": synth_slope_err,
        "N_fit": N_fit,
        "synth_slope_le10": slope_le10,
        "synth_slope_err_le10": slope_err_le10,
        "N_fit_le10": N_fit_le10,
        "synth_slope_nath": slope_nath,
        "synth_slope_err_nath": slope_err_nath,
        "N_fit_nath": N_fit_nath,
        "bin_slope_le10": bin_slope_le10,
        "bin_slope_err_le10": bin_slope_err_le10,
        "bin_N_le10": bin_N_le10,
        "bin_slope_30": bin_slope_30,
        "bin_slope_err_30": bin_slope_err_30,
        "bin_N_30": bin_N_30,
        "bin_slope_100": bin_slope_100,
        "bin_slope_err_100": bin_slope_err_100,
        "bin_N_100": bin_N_100,
        "N_bubble": N_bubble,
        "N_surviving": N_surviving,
        "N_recollapsed": recollapsed,
        "N_below_complete": below_complete,
        "t_obs": t_obs,
        "cmf_slope": cmf_slope,
        "R_complete": R_complete,
        "runs_used": ", ".join(runs_used_summary),
        "run_usage": run_usage,
        "filtered_records": filtered,
    }


def run_population_synthesis_cmf(
    records: List[Dict],
    N_bubble: int = 20000,
    t_obs: float = 5.0,
    cloud_mf_slope: float = -1.7,
    fixed_sfe: float = 0.01,
    R_complete: float = 30.0,
    seed: int = 42,
    fixed_ncore: float = None,
) -> Optional[Dict]:
    """
    Option A: Sample M_cloud from a cloud mass function with fixed SFE.

    Each (M_cloud, SFE) pair maps to exactly one TRINITY run, eliminating
    the SFE degeneracy present in M_star-based sampling.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    N_bubble : int
        Number of synthetic bubbles to draw.
    t_obs : float
        Observation time [Myr].
    cloud_mf_slope : float
        Cloud mass function slope dN/dM_cloud ~ M_cloud^beta (default: -1.7,
        Solomon+1987, Heyer+2001).
    fixed_sfe : float
        Fixed star formation efficiency (default: 0.01, Utomo+2018).
    R_complete : float
        Completeness radius [pc].
    seed : int
        Random seed.
    fixed_ncore : float, optional
        If set, only use runs with this core density.

    Returns
    -------
    dict or None
        Same structure as run_population_synthesis(), plus method-specific keys.
    """
    if not records:
        return None

    # Filter by nCore
    filtered = records
    if fixed_ncore is not None:
        filtered = [r for r in filtered
                    if abs(r["nCore"] - fixed_ncore) / fixed_ncore < 0.1]

    if not filtered:
        logger.warning("Option A: no runs match fixed_ncore=%s", fixed_ncore)
        return None

    # Filter to runs matching fixed_sfe (tolerance 0.005)
    sfe_filtered = [r for r in filtered
                    if abs(r["sfe"] - fixed_sfe) < 0.005]

    if not sfe_filtered:
        logger.warning("Option A: no runs with sfe≈%.3f (available: %s)",
                        fixed_sfe,
                        sorted(set(f"{r['sfe']:.3f}" for r in filtered)))
        return None

    logger.info("Option A: %d runs with sfe≈%.3f", len(sfe_filtered), fixed_sfe)

    # Build sorted grid in log10(M_cloud) with interp data
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in sfe_filtered:
        key = round(np.log10(rec["mCloud"]), 1)
        groups[key].append(rec)

    # Within each group, pick first (should be unique for a given M_cloud/sfe)
    selected = []
    for key in sorted(groups.keys()):
        selected.append(groups[key][0])

    selected.sort(key=lambda r: r["mCloud"])

    grid_logMcloud = np.array([np.log10(r["mCloud"]) for r in selected])
    grid_interp_data = []
    for rec in selected:
        t_full = rec["t_full"]
        R_full = rec["R_full"]
        _, unique_idx = np.unique(t_full, return_index=True)
        grid_interp_data.append({
            "t": t_full[unique_idx],
            "R": R_full[unique_idx],
        })

    for rec in selected:
        logger.info("  Option A grid: M_cloud=%.2e, sfe=%.3f -> %s",
                     rec["mCloud"], rec["sfe"], rec["folder"])

    if len(grid_logMcloud) == 0:
        logger.warning("Option A: empty M_cloud grid")
        return None

    M_cloud_min = 10.0 ** grid_logMcloud[0]
    M_cloud_max = 10.0 ** grid_logMcloud[-1]

    rng = np.random.default_rng(seed)

    # Step 1: sample M_cloud from cloud mass function
    sampled_Mcloud = _sample_powerlaw(N_bubble, M_cloud_min, M_cloud_max,
                                       cloud_mf_slope, rng)

    # Step 2: random birth times
    t_form = rng.uniform(0.0, t_obs, size=N_bubble)
    ages = t_obs - t_form

    # Step 3: log-interpolate between bracketing M_cloud tracks
    R_synth = np.full(N_bubble, np.nan)
    run_usage = np.zeros(len(selected), dtype=int)
    log_sampled = np.log10(sampled_Mcloud)

    for i in range(N_bubble):
        idx_lo, idx_hi, w = _bracket_on_grid(log_sampled[i], grid_logMcloud)
        run_usage[idx_lo] += 1
        if idx_hi != idx_lo:
            run_usage[idx_hi] += 1

        if idx_lo == idx_hi:
            td = grid_interp_data[idx_lo]
            age_c = np.clip(ages[i], td["t"].min(), td["t"].max())
            R_synth[i] = np.interp(age_c, td["t"], td["R"])
        else:
            R_synth[i] = _interpolate_R(ages[i], grid_interp_data[idx_lo],
                                          grid_interp_data[idx_hi], w)

    # Step 4: apply cuts
    valid_mask = np.isfinite(R_synth) & (R_synth > 0)
    recollapsed = int(np.sum(~valid_mask))
    R_valid = R_synth[valid_mask]

    below_complete = int(np.sum(R_valid < R_complete))
    N_surviving = int(np.sum(R_valid >= R_complete))
    logger.info("Option A: %d drawn, %d recollapsed, %d below R_complete=%.0f pc, "
                "%d surviving",
                N_bubble, recollapsed, below_complete, R_complete, N_surviving)

    if len(R_valid) < MIN_PTS:
        logger.warning("Option A: too few valid bubbles (%d)", len(R_valid))
        return None

    # Step 5: histogram
    log_R_all = np.log10(R_valid)
    n_bins_display = 40
    log_edges = np.linspace(log_R_all.min(), log_R_all.max(), n_bins_display + 1)
    counts, _ = np.histogram(log_R_all, bins=log_edges)
    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    # Step 6: fit power laws (same as Option C)
    synth_slope, synth_slope_err, N_fit = _fit_powerlaw_mle(R_valid, R_complete)
    slope_le10, slope_err_le10, N_fit_le10 = _fit_powerlaw_mle_upper(
        R_valid, R_COMPLETE_10PC)
    slope_nath, slope_err_nath, N_fit_nath = _fit_powerlaw_mle(
        R_valid, R_COMPLETE_NATH20)
    bin_slope_le10, bin_slope_err_le10, bin_N_le10 = _fit_powerlaw_binned_upper(
        R_valid, R_COMPLETE_10PC)
    bin_slope_30, bin_slope_err_30, bin_N_30 = _fit_powerlaw_binned(
        R_valid, R_complete)
    bin_slope_100, bin_slope_err_100, bin_N_100 = _fit_powerlaw_binned(
        R_valid, R_COMPLETE_NATH20)

    runs_used_summary = []
    for j, rec in enumerate(selected):
        if run_usage[j] > 0:
            runs_used_summary.append(f"{rec['folder']}({run_usage[j]})")

    return {
        "method": "cmf",
        "cloud_mf_slope": cloud_mf_slope,
        "fixed_sfe_used": fixed_sfe,
        "R_synth": R_valid,
        "R_centres": R_centres,
        "dNdR": dNdR,
        "synth_slope": synth_slope,
        "synth_slope_err": synth_slope_err,
        "N_fit": N_fit,
        "synth_slope_le10": slope_le10,
        "synth_slope_err_le10": slope_err_le10,
        "N_fit_le10": N_fit_le10,
        "synth_slope_nath": slope_nath,
        "synth_slope_err_nath": slope_err_nath,
        "N_fit_nath": N_fit_nath,
        "bin_slope_le10": bin_slope_le10,
        "bin_slope_err_le10": bin_slope_err_le10,
        "bin_N_le10": bin_N_le10,
        "bin_slope_30": bin_slope_30,
        "bin_slope_err_30": bin_slope_err_30,
        "bin_N_30": bin_N_30,
        "bin_slope_100": bin_slope_100,
        "bin_slope_err_100": bin_slope_err_100,
        "bin_N_100": bin_N_100,
        "N_bubble": N_bubble,
        "N_surviving": N_surviving,
        "N_recollapsed": recollapsed,
        "N_below_complete": below_complete,
        "t_obs": t_obs,
        "cmf_slope": cloud_mf_slope,
        "R_complete": R_complete,
        "runs_used": ", ".join(runs_used_summary),
        "run_usage": run_usage,
        "filtered_records": sfe_filtered,
    }


def run_population_synthesis_avg(
    records: List[Dict],
    N_bubble: int = 20000,
    t_obs: float = 5.0,
    cmf_slope: float = -2.0,
    R_complete: float = 30.0,
    seed: int = 42,
    fixed_ncore: float = None,
    sfe_prior_median: float = 0.05,
    sfe_prior_sigma: float = 0.5,
) -> Optional[Dict]:
    """
    Option B: Sample M_star from CMF, then randomly select among degenerate
    runs weighted by a log-normal prior on SFE.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    N_bubble : int
        Number of synthetic bubbles to draw.
    t_obs : float
        Observation time [Myr].
    cmf_slope : float
        CMF slope dN/dM_star ~ M_star^alpha (default: -2.0).
    R_complete : float
        Completeness radius [pc].
    seed : int
        Random seed.
    fixed_ncore : float, optional
        If set, only use runs with this core density.
    sfe_prior_median : float
        Median of log-normal SFE prior (default: 0.05, Murray 2011).
    sfe_prior_sigma : float
        Sigma in dex for log-normal SFE prior (default: 0.5, Chevance+2020).

    Returns
    -------
    dict or None
        Same structure as run_population_synthesis(), plus method-specific keys.
    """
    if not records:
        return None

    # Filter by nCore
    filtered = records
    if fixed_ncore is not None:
        filtered = [r for r in filtered
                    if abs(r["nCore"] - fixed_ncore) / fixed_ncore < 0.1]

    if not filtered:
        logger.warning("Option B: no runs match fixed_ncore=%s", fixed_ncore)
        return None

    # Group by M_star — keep ALL runs (no deduplication)
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in filtered:
        key = round(np.log10(rec["M_star"]), 2)
        groups[key].append(rec)

    # Compute log-normal weights for each group
    log_median = np.log10(sfe_prior_median)
    grid_keys = sorted(groups.keys())

    grid_logMstar = np.array(grid_keys)
    grid_groups = []  # list of lists of (rec, interp_data, weight)

    for key in grid_keys:
        group_recs = groups[key]
        entries = []
        for rec in group_recs:
            log_sfe = np.log10(rec["sfe"]) if rec["sfe"] > 0 else -10.0
            weight = np.exp(-0.5 * ((log_sfe - log_median) / sfe_prior_sigma) ** 2)
            t_full = rec["t_full"]
            R_full = rec["R_full"]
            _, unique_idx = np.unique(t_full, return_index=True)
            interp_data = {"t": t_full[unique_idx], "R": R_full[unique_idx]}
            entries.append((rec, interp_data, weight))
        # Normalize weights within group
        total_w = sum(e[2] for e in entries)
        if total_w > 0:
            entries = [(r, d, w / total_w) for r, d, w in entries]
        grid_groups.append(entries)

    if len(grid_logMstar) == 0:
        logger.warning("Option B: empty M_star grid")
        return None

    M_star_min = 10.0 ** grid_logMstar[0]
    M_star_max = 10.0 ** grid_logMstar[-1]

    logger.info("Option B: %d unique M_star groups from %.2e to %.2e, "
                "SFE prior median=%.3f sigma=%.1f dex",
                len(grid_logMstar), M_star_min, M_star_max,
                sfe_prior_median, sfe_prior_sigma)

    rng = np.random.default_rng(seed)

    # Step 1: sample cluster masses from CMF
    sampled_Mstar = _sample_powerlaw(N_bubble, M_star_min, M_star_max,
                                      cmf_slope, rng)

    # Step 2: random birth times
    t_form = rng.uniform(0.0, t_obs, size=N_bubble)
    ages = t_obs - t_form

    # Step 3: for each sampled M_star, find nearest group, weighted selection
    R_synth = np.full(N_bubble, np.nan)
    log_sampled = np.log10(sampled_Mstar)

    for i in range(N_bubble):
        # Find nearest M_star group
        idx = int(np.argmin(np.abs(grid_logMstar - log_sampled[i])))
        group = grid_groups[idx]

        # Weighted random selection among group members
        if len(group) == 1:
            rec, interp_data, _ = group[0]
        else:
            weights = np.array([e[2] for e in group])
            chosen = rng.choice(len(group), p=weights)
            rec, interp_data, _ = group[chosen]

        # Evaluate R(age) from the selected run (single track, no interpolation)
        age_c = np.clip(ages[i], interp_data["t"].min(),
                        interp_data["t"].max())
        R_synth[i] = np.interp(age_c, interp_data["t"], interp_data["R"])

    # Step 4: apply cuts
    valid_mask = np.isfinite(R_synth) & (R_synth > 0)
    recollapsed = int(np.sum(~valid_mask))
    R_valid = R_synth[valid_mask]

    below_complete = int(np.sum(R_valid < R_complete))
    N_surviving = int(np.sum(R_valid >= R_complete))
    logger.info("Option B: %d drawn, %d recollapsed, %d below R_complete=%.0f pc, "
                "%d surviving",
                N_bubble, recollapsed, below_complete, R_complete, N_surviving)

    if len(R_valid) < MIN_PTS:
        logger.warning("Option B: too few valid bubbles (%d)", len(R_valid))
        return None

    # Step 5: histogram
    log_R_all = np.log10(R_valid)
    n_bins_display = 40
    log_edges = np.linspace(log_R_all.min(), log_R_all.max(), n_bins_display + 1)
    counts, _ = np.histogram(log_R_all, bins=log_edges)
    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    # Step 6: fit power laws (same as Option C)
    synth_slope, synth_slope_err, N_fit = _fit_powerlaw_mle(R_valid, R_complete)
    slope_le10, slope_err_le10, N_fit_le10 = _fit_powerlaw_mle_upper(
        R_valid, R_COMPLETE_10PC)
    slope_nath, slope_err_nath, N_fit_nath = _fit_powerlaw_mle(
        R_valid, R_COMPLETE_NATH20)
    bin_slope_le10, bin_slope_err_le10, bin_N_le10 = _fit_powerlaw_binned_upper(
        R_valid, R_COMPLETE_10PC)
    bin_slope_30, bin_slope_err_30, bin_N_30 = _fit_powerlaw_binned(
        R_valid, R_complete)
    bin_slope_100, bin_slope_err_100, bin_N_100 = _fit_powerlaw_binned(
        R_valid, R_COMPLETE_NATH20)

    return {
        "method": "avg",
        "sfe_prior_median": sfe_prior_median,
        "sfe_prior_sigma": sfe_prior_sigma,
        "R_synth": R_valid,
        "R_centres": R_centres,
        "dNdR": dNdR,
        "synth_slope": synth_slope,
        "synth_slope_err": synth_slope_err,
        "N_fit": N_fit,
        "synth_slope_le10": slope_le10,
        "synth_slope_err_le10": slope_err_le10,
        "N_fit_le10": N_fit_le10,
        "synth_slope_nath": slope_nath,
        "synth_slope_err_nath": slope_err_nath,
        "N_fit_nath": N_fit_nath,
        "bin_slope_le10": bin_slope_le10,
        "bin_slope_err_le10": bin_slope_err_le10,
        "bin_N_le10": bin_N_le10,
        "bin_slope_30": bin_slope_30,
        "bin_slope_err_30": bin_slope_err_30,
        "bin_N_30": bin_N_30,
        "bin_slope_100": bin_slope_100,
        "bin_slope_err_100": bin_slope_err_100,
        "bin_N_100": bin_N_100,
        "N_bubble": N_bubble,
        "N_surviving": N_surviving,
        "N_recollapsed": recollapsed,
        "N_below_complete": below_complete,
        "t_obs": t_obs,
        "cmf_slope": cmf_slope,
        "R_complete": R_complete,
        "runs_used": "",
        "run_usage": np.array([]),
        "filtered_records": filtered,
    }


# ======================================================================
# Part 3: Parameter sensitivity
# ======================================================================

def run_sensitivity(
    records: List[Dict],
    N_bubble: int = 20000,
    seed: int = 42,
    R_complete: float = 30.0,
    cmf_slopes: List[float] = None,
    t_obs_values: List[float] = None,
    fixed_sfe: float = None,
    fixed_ncore: float = None,
    n_bootstrap: int = 5,
    method: str = "interp",
    cloud_mf_slope_default: float = -1.7,
    cmf_fixed_sfe: float = 0.01,
    sfe_prior_median: float = 0.05,
    sfe_prior_sigma: float = 0.5,
) -> List[Dict]:
    """
    Run synthesis for a grid of slopes and t_obs values.

    Each (slope, t_obs) point is repeated n_bootstrap times with
    independent seeds to estimate mean +/- std of the fitted slope.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    N_bubble : int
        Number of synthetic bubbles per run.
    seed : int
        Base random seed.
    R_complete : float
        Completeness radius [pc].
    cmf_slopes : list of float, optional
        Slopes to test (CMF slopes for interp/avg, cloud MF slopes for cmf).
    t_obs_values : list of float, optional
        Observation times to test [Myr] (default: [3, 5, 10]).
    fixed_sfe : float, optional
        If set, only use runs with this SFE (for interp method).
    fixed_ncore : float, optional
        If set, only use runs with this core density.
    n_bootstrap : int
        Number of independent runs per point for error bars.
    method : str
        ``"interp"`` (Option C), ``"cmf"`` (Option A), or ``"avg"`` (Option B).
    cloud_mf_slope_default : float
        Default cloud MF slope for Option A (used when cmf_slopes is None).
    cmf_fixed_sfe : float
        Fixed SFE for Option A.
    sfe_prior_median : float
        SFE prior median for Option B.
    sfe_prior_sigma : float
        SFE prior sigma for Option B.

    Returns
    -------
    list of dict
        Each entry has: cmf_slope, t_obs, synth_slope, synth_slope_std, etc.
    """
    if cmf_slopes is None:
        if method == "cmf":
            cmf_slopes = [-1.3, -1.5, -1.7, -1.9, -2.1]
        else:
            cmf_slopes = [-1.5, -2.0, -2.5]
    if t_obs_values is None:
        t_obs_values = [3.0, 5.0, 10.0]

    results = []
    for cmf_sl in cmf_slopes:
        for t_obs in t_obs_values:
            run_seed = seed + hash((cmf_sl, t_obs)) % (2**31)

            slopes = []
            last_N_surviving = 0
            last_slope_err = np.nan
            last_N_fit = 0
            for k in range(n_bootstrap):
                if method == "cmf":
                    synth = run_population_synthesis_cmf(
                        records,
                        N_bubble=N_bubble,
                        t_obs=t_obs,
                        cloud_mf_slope=cmf_sl,
                        fixed_sfe=cmf_fixed_sfe,
                        R_complete=R_complete,
                        seed=run_seed + k,
                        fixed_ncore=fixed_ncore,
                    )
                elif method == "avg":
                    synth = run_population_synthesis_avg(
                        records,
                        N_bubble=N_bubble,
                        t_obs=t_obs,
                        cmf_slope=cmf_sl,
                        R_complete=R_complete,
                        seed=run_seed + k,
                        fixed_ncore=fixed_ncore,
                        sfe_prior_median=sfe_prior_median,
                        sfe_prior_sigma=sfe_prior_sigma,
                    )
                else:
                    synth = run_population_synthesis(
                        records,
                        N_bubble=N_bubble,
                        t_obs=t_obs,
                        cmf_slope=cmf_sl,
                        R_complete=R_complete,
                        seed=run_seed + k,
                        fixed_sfe=fixed_sfe,
                        fixed_ncore=fixed_ncore,
                    )
                if synth is not None:
                    last_N_surviving = synth["N_surviving"]
                    last_slope_err = synth["synth_slope_err"]
                    last_N_fit = synth["N_fit"]
                    if np.isfinite(synth["synth_slope"]):
                        slopes.append(synth["synth_slope"])

            if slopes:
                results.append({
                    "cmf_slope": cmf_sl,
                    "t_obs": t_obs,
                    "synth_slope": np.mean(slopes),
                    "synth_slope_std": np.std(slopes) if len(slopes) > 1 else 0.0,
                    "synth_slope_err": last_slope_err,
                    "N_surviving": last_N_surviving,
                    "N_fit": last_N_fit,
                    "n_runs": len(slopes),
                })
            else:
                results.append({
                    "cmf_slope": cmf_sl,
                    "t_obs": t_obs,
                    "synth_slope": np.nan,
                    "synth_slope_std": np.nan,
                    "synth_slope_err": np.nan,
                    "N_surviving": 0,
                    "N_fit": 0,
                    "n_runs": 0,
                })

    return results


# ======================================================================
# Figures
# ======================================================================

def _model_label(rec: Dict) -> str:
    """Short label for a simulation run."""
    return (f"$M={rec['mCloud']:.0e}$, "
            f"$\\epsilon={rec['sfe']:.0e}$, "
            f"$n={rec['nCore']:.0e}$")


def _assign_styles(records: List[Dict]) -> List[Dict]:
    """Assign colourblind-safe colors and linestyles to each record."""
    colors = [C_BLUE, C_VERMILLION, C_GREEN, C_PURPLE, C_ORANGE, C_SKY, C_BLACK]
    for i, rec in enumerate(records):
        rec["_color"] = colors[i % len(colors)]
        rec["_ls"] = LINESTYLES[i % len(LINESTYLES)]
    return records


def _color_by_Mstar(records: List[Dict]) -> List[Dict]:
    """Assign colors based on M_star using a log-scale colormap."""
    Mstar_vals = np.array([r["M_star"] for r in records])
    if Mstar_vals.max() > Mstar_vals.min() > 0:
        log_Mstar = np.log10(Mstar_vals)
        norm = plt.Normalize(vmin=log_Mstar.min(), vmax=log_Mstar.max())
        cmap = plt.cm.viridis
        for i, rec in enumerate(records):
            rec["_color"] = cmap(norm(log_Mstar[i]))
    else:
        _assign_styles(records)
    return records


def plot_residence_time(records: List[Dict], output_dir: Path,
                        fmt: str = "pdf") -> None:
    """
    Figure 1: Single-trajectory residence-time distribution 1/v(R) vs R.

    This shows how long a single shell lingers at each radius.  Positive
    slopes are expected — this is a diagnostic, not a prediction.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    """
    records = _color_by_Mstar(records)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    R_range = [np.inf, -np.inf]
    for rec in records:
        R = rec["R_exp"]
        dNdR = rec["dNdR_norm"]
        if len(R) < 2:
            continue
        ax.plot(R, dNdR, color=rec["_color"], ls="-",
                lw=1.2, alpha=0.85,
                label=(f"{_model_label(rec)}"
                       f" ($\\gamma={rec['residence_slope']:.2f}$)"))
        R_range[0] = min(R_range[0], R.min())
        R_range[1] = max(R_range[1], R.max())

    # Reference slopes: energy-driven (R^{+2/3}) and momentum-driven (R^{+3})
    if R_range[0] < R_range[1]:
        R_ref = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), 100)
        R_mid = np.sqrt(R_range[0] * R_range[1])
        # Energy-driven: v ~ R^{-2/3} => 1/v ~ R^{+2/3}
        ref_energy = (R_ref / R_mid) ** (2.0 / 3.0)
        ax.plot(R_ref, ref_energy, color="grey", ls="--", lw=1.5, alpha=0.5,
                label=r"$R^{+2/3}$ (energy-driven)")
        # Momentum-driven: v ~ R^{-3} => 1/v ~ R^{+3}
        ref_momentum = (R_ref / R_mid) ** 3.0
        ax.plot(R_ref, ref_momentum, color="grey", ls=":", lw=1.5, alpha=0.5,
                label=r"$R^{+3}$ (momentum-driven)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$1/v(R)$ (normalized)")

    if len(records) <= 8:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_residence_time.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_synth_row(axes_row: tuple, synth: Dict, row_label: str,
                    fit_method: str = "mle") -> None:
    """
    Plot one row (histogram + dN/dR) of the synthesis figure.

    Parameters
    ----------
    axes_row : tuple of two Axes
        (left, right) axes for histogram and dN/dR panels.
    synth : dict
        Output of run_population_synthesis().
    row_label : str
        Row label, e.g. "(a)" or "(b)", prepended to panel titles.
    fit_method : str
        ``"mle"`` for MLE fit lines, ``"binned"`` for binned-LSQ fit lines.
    """
    R_synth = synth["R_synth"]
    R_centres = synth["R_centres"]
    dNdR = synth["dNdR"]
    R_complete = synth["R_complete"]

    # Left panel: histogram of full population (log-spaced bins)
    ax_hist = axes_row[0]
    R_pos = R_synth[R_synth > 0]
    log_bins = np.logspace(np.log10(R_pos.min()), np.log10(R_pos.max()), 31)
    hist_counts, hist_edges = np.histogram(R_pos, bins=log_bins)
    hist_centres = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    ax_hist.bar(hist_edges[:-1], hist_counts, width=np.diff(hist_edges),
                align="edge", color=C_SKY, edgecolor=C_BLUE, alpha=0.7)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.axvline(R_complete, color=C_VERMILLION, ls=":", lw=1.2, alpha=0.8)
    ax_hist.axvspan(ax_hist.get_xlim()[0], R_complete, color="grey",
                    alpha=0.15, zorder=0)
    ax_hist.set_xlabel(r"$R$ [pc]")
    ax_hist.set_ylabel(r"$N$")

    # Helper: overlay a power-law fit on the histogram panel.
    # Log-spaced bins have dR ∝ R, so counts N ∝ dN/dR * R ∝ R^{gamma+1}.
    def _add_hist_fit(ax, slope_val, R_range, color, label):
        if not np.isfinite(slope_val):
            return
        R_line = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), 50)
        # Find histogram bins within range for normalization
        in_range = (hist_centres >= R_range[0]) & (hist_centres <= R_range[1])
        counts_in = hist_counts[in_range]
        if counts_in.sum() == 0:
            return
        R_norm = hist_centres[in_range][np.argmax(counts_in)]
        N_norm = counts_in.max()
        N_line = N_norm * (R_line / R_norm) ** (slope_val + 1.0)
        ax.plot(R_line, N_line, color=color, ls="--", lw=1.5, alpha=0.8,
                label=label)

    # Overlay fit lines on histogram
    R_min_all = R_pos.min()
    R_max_all = R_pos.max()
    if fit_method == "mle":
        _add_hist_fit(ax_hist, synth.get("synth_slope_le10", np.nan),
                      (R_min_all, R_COMPLETE_10PC), C_PURPLE,
                      f"$R\\leq{R_COMPLETE_10PC:.0f}$")
        _add_hist_fit(ax_hist, synth["synth_slope"],
                      (R_complete, R_max_all), C_VERMILLION,
                      f"$R\\geq{R_complete:.0f}$")
        _add_hist_fit(ax_hist, synth.get("synth_slope_nath", np.nan),
                      (R_COMPLETE_NATH20, R_max_all), C_ORANGE,
                      f"$R\\geq{R_COMPLETE_NATH20:.0f}$")
    else:
        _add_hist_fit(ax_hist, synth.get("bin_slope_le10", np.nan),
                      (R_min_all, R_COMPLETE_10PC), C_PURPLE,
                      f"$R\\leq{R_COMPLETE_10PC:.0f}$")
        _add_hist_fit(ax_hist, synth.get("bin_slope_30", np.nan),
                      (R_complete, R_max_all), C_VERMILLION,
                      f"$R\\geq{R_complete:.0f}$")
        _add_hist_fit(ax_hist, synth.get("bin_slope_100", np.nan),
                      (R_COMPLETE_NATH20, R_max_all), C_ORANGE,
                      f"$R\\geq{R_COMPLETE_NATH20:.0f}$")
    method_tag = "MLE" if fit_method == "mle" else "Binned LSQ"
    ax_hist.legend(
        fontsize=6, framealpha=0.7,
        title=(f"{row_label} [{method_tag}] $N={synth['N_bubble']}$, "
               f"$t_{{\\rm obs}}={synth['t_obs']:.1f}$ Myr, "
               f"CMF $\\alpha={synth['cmf_slope']:.1f}$"),
        title_fontsize=7,
    )

    # Right panel: dN/dR vs R
    ax = axes_row[1]
    valid = dNdR > 0
    above = valid & (R_centres >= R_complete)
    below = valid & (R_centres < R_complete)
    ax.scatter(R_centres[above], dNdR[above], color=C_BLUE, s=25,
               zorder=3, label="Synthesis")
    if below.any():
        ax.scatter(R_centres[below], dNdR[below], color=C_BLUE, s=25,
                   zorder=3, alpha=0.25)

    # Helper: plot a fit line normalized at the first histogram bin >= R_cut
    def _add_fit_line(ax, slope_val, slope_err_val, R_min_cut, color,
                      tag, valid_mask, R_centres, dNdR):
        above_cut = valid_mask & (R_centres >= R_min_cut)
        if not np.isfinite(slope_val) or not above_cut.any():
            return
        idx0 = np.where(above_cut)[0][0]
        R_norm = R_centres[idx0]
        dNdR_norm = dNdR[idx0]
        R_fit = np.logspace(np.log10(R_min_cut),
                            np.log10(R_centres[valid_mask].max()), 50)
        dNdR_fit = dNdR_norm * (R_fit / R_norm) ** slope_val
        if np.isfinite(slope_err_val):
            label = (f"{tag} $R\\geq{R_min_cut:.0f}$ pc: "
                     f"$\\gamma={slope_val:.2f} \\pm {slope_err_val:.2f}$")
        else:
            label = (f"{tag} $R\\geq{R_min_cut:.0f}$ pc: "
                     f"$\\gamma={slope_val:.2f}$")
        ax.plot(R_fit, dNdR_fit, color=color, ls="--", lw=1.5, label=label)

    # Helper: plot fit line for R <= R_max (upper-truncated region)
    def _add_fit_line_upper(ax, slope_val, slope_err_val, R_max_cut, color,
                            tag, valid_mask, R_centres, dNdR):
        below_cut = valid_mask & (R_centres <= R_max_cut)
        if not np.isfinite(slope_val) or not below_cut.any():
            return
        idx0 = np.where(below_cut)[0][-1]  # normalize at last bin <= R_max
        R_norm = R_centres[idx0]
        dNdR_norm = dNdR[idx0]
        R_fit = np.logspace(np.log10(R_centres[valid_mask].min()),
                            np.log10(R_max_cut), 50)
        dNdR_fit = dNdR_norm * (R_fit / R_norm) ** slope_val
        if np.isfinite(slope_err_val):
            label = (f"{tag} $R\\leq{R_max_cut:.0f}$ pc: "
                     f"$\\gamma={slope_val:.2f} \\pm {slope_err_val:.2f}$")
        else:
            label = (f"{tag} $R\\leq{R_max_cut:.0f}$ pc: "
                     f"$\\gamma={slope_val:.2f}$")
        ax.plot(R_fit, dNdR_fit, color=color, ls="--", lw=1.5, label=label)

    if fit_method == "mle":
        tag = "MLE"
        # R <= 10 pc fit
        _add_fit_line_upper(ax, synth.get("synth_slope_le10", np.nan),
                            synth.get("synth_slope_err_le10", np.nan),
                            R_COMPLETE_10PC, C_PURPLE, tag,
                            valid, R_centres, dNdR)
        # R >= 30 pc (Watkins+23)
        _add_fit_line(ax, synth["synth_slope"],
                      synth.get("synth_slope_err", np.nan),
                      R_complete, C_VERMILLION, tag,
                      valid, R_centres, dNdR)
        # R >= 100 pc (Nath+20)
        _add_fit_line(ax, synth.get("synth_slope_nath", np.nan),
                      synth.get("synth_slope_err_nath", np.nan),
                      R_COMPLETE_NATH20, C_ORANGE, tag,
                      valid, R_centres, dNdR)
    else:  # binned
        tag = "Bin"
        # R <= 10 pc
        _add_fit_line_upper(ax, synth.get("bin_slope_le10", np.nan),
                            synth.get("bin_slope_err_le10", np.nan),
                            R_COMPLETE_10PC, C_PURPLE, tag,
                            valid, R_centres, dNdR)
        # R >= 30 pc
        _add_fit_line(ax, synth.get("bin_slope_30", np.nan),
                      synth.get("bin_slope_err_30", np.nan),
                      R_complete, C_VERMILLION, tag,
                      valid, R_centres, dNdR)
        # R >= 100 pc
        _add_fit_line(ax, synth.get("bin_slope_100", np.nan),
                      synth.get("bin_slope_err_100", np.nan),
                      R_COMPLETE_NATH20, C_ORANGE, tag,
                      valid, R_centres, dNdR)

    # Observed reference — solid line, transparent to reduce clutter
    if valid.sum() >= 2:
        R_ref = np.logspace(np.log10(R_centres[valid].min()),
                            np.log10(R_centres[valid].max()), 50)
        R_mid = np.sqrt(R_centres[valid].min() * R_centres[valid].max())
        dN_mid = np.interp(np.log10(R_mid), np.log10(R_centres[valid]),
                           np.log10(dNdR[valid]))
        dNdR_obs = 10.0 ** (dN_mid + OBSERVED_ALPHA * (np.log10(R_ref) - np.log10(R_mid)))
        ax.plot(R_ref, dNdR_obs, color=C_GREEN, ls="-", lw=1.5, alpha=0.35,
                label=r"$R^{-2.2}$ (Watkins+2023)")

    # Completeness lines
    ax.axvline(R_COMPLETE_10PC, color=C_PURPLE, ls=":", lw=1.0, alpha=0.6)
    ax.axvline(R_complete, color=C_VERMILLION, ls=":", lw=1.0, alpha=0.6)
    ax.axvline(R_COMPLETE_NATH20, color=C_ORANGE, ls=":", lw=1.0, alpha=0.6)
    ax.axvspan(ax.get_xlim()[0], R_complete, color="grey", alpha=0.10,
               zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}R$")
    ax.legend(fontsize=6, framealpha=0.7)


def _save_synth_figure(synth_list: list, output_dir: Path,
                       fmt: str, fit_method: str,
                       suffix: str,
                       filename_prefix: str = "bubble_synthesis_population") -> None:
    """Build and save one synthesis figure (MLE or binned)."""
    n_rows = len(synth_list)
    labels = [chr(ord("a") + i) for i in range(n_rows)]

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i, synth in enumerate(synth_list):
        _plot_synth_row((axes[i, 0], axes[i, 1]), synth,
                        f"({labels[i]})", fit_method=fit_method)

    fig.tight_layout()
    path = output_dir / f"{filename_prefix}_{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_synthesis_population(synth_list: list, output_dir: Path,
                              fmt: str = "pdf",
                              filename_prefix: str = "bubble_synthesis_population"
                              ) -> None:
    """
    Figure 2: Population synthesis results — two separate figures.

    Saves ``{prefix}_mle.{fmt}`` (MLE fits) and
    ``{prefix}_binned.{fmt}`` (binned-LSQ fits).

    Parameters
    ----------
    synth_list : list of dict
        List of synthesis outputs (one per t_obs).
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    filename_prefix : str
        Base filename (default: ``"bubble_synthesis_population"``).
    """
    _save_synth_figure(synth_list, output_dir, fmt,
                       fit_method="mle", suffix="mle",
                       filename_prefix=filename_prefix)
    _save_synth_figure(synth_list, output_dir, fmt,
                       fit_method="binned", suffix="binned",
                       filename_prefix=filename_prefix)


def plot_slope_vs_cmf(sensitivity: List[Dict], output_dir: Path,
                      fmt: str = "pdf",
                      filename: str = "bubble_slope_vs_cmf",
                      xlabel: str = None) -> None:
    """
    Figure 3: Predicted dN/dR slope vs input MF slope for multiple t_obs.

    Parameters
    ----------
    sensitivity : list of dict
        Output of run_sensitivity().
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    filename : str
        Output filename (without extension).
    xlabel : str, optional
        Custom x-axis label. Defaults to CMF slope label.
    """
    if not sensitivity:
        logger.warning("No sensitivity data — skipping slope-vs-CMF plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Group by t_obs
    t_obs_vals = sorted(set(s["t_obs"] for s in sensitivity))
    colors = [C_BLUE, C_VERMILLION, C_GREEN, C_PURPLE, C_ORANGE]
    markers = ["o", "s", "D", "^", "v"]

    for j, t_obs in enumerate(t_obs_vals):
        subset = [s for s in sensitivity if s["t_obs"] == t_obs]
        cmf_sl = np.array([s["cmf_slope"] for s in subset])
        synth_sl = np.array([s["synth_slope"] for s in subset])
        synth_std = np.array([s.get("synth_slope_std", 0.0) for s in subset])
        synth_std = np.nan_to_num(synth_std, nan=0.0)
        valid = np.isfinite(synth_sl)
        if valid.sum() == 0:
            continue
        c = colors[j % len(colors)]
        m = markers[j % len(markers)]
        ax.errorbar(cmf_sl[valid], synth_sl[valid], yerr=synth_std[valid],
                     color=c, marker=m, ms=8, lw=1.5, ls="-", capsize=3,
                     label=f"$t_{{\\rm obs}}={t_obs:.0f}$ Myr")

    # Observed reference with Watkins+2023 uncertainty band
    ax.axhspan(OBSERVED_ALPHA - 0.1, OBSERVED_ALPHA + 0.1,
               color="grey", alpha=0.15, zorder=0)
    ax.axhline(OBSERVED_ALPHA, color="grey", ls="--", lw=1.5, alpha=0.7,
               label=f"Observed $= {OBSERVED_ALPHA} \\pm 0.1$")

    if xlabel is None:
        xlabel = (r"CMF slope $\alpha$ "
                  r"($\mathrm{d}N/\mathrm{d}M_\star \propto M_\star^\alpha$)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Synthetic $\mathrm{d}N/\mathrm{d}R$ slope $\gamma$")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"{filename}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _plot_dNdR_panel(ax, synth: Dict, title: str = "") -> None:
    """
    Draw a single dN/dR panel (MLE fits) for the comparison figure.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    synth : dict
        Synthesis output dict.
    title : str
        Panel title.
    """
    R_centres = synth["R_centres"]
    dNdR = synth["dNdR"]
    R_complete = synth["R_complete"]

    valid = dNdR > 0
    above = valid & (R_centres >= R_complete)
    below = valid & (R_centres < R_complete)

    ax.scatter(R_centres[above], dNdR[above], color=C_BLUE, s=20, zorder=3)
    if below.any():
        ax.scatter(R_centres[below], dNdR[below], color=C_BLUE, s=20,
                   zorder=3, alpha=0.25)

    # MLE fit lines
    def _add_line(slope_val, slope_err, R_cut, color, is_upper=False):
        if not np.isfinite(slope_val):
            return
        if is_upper:
            mask = valid & (R_centres <= R_cut)
            if not mask.any():
                return
            idx0 = np.where(mask)[0][-1]
            R_fit = np.logspace(np.log10(R_centres[valid].min()),
                                np.log10(R_cut), 50)
        else:
            mask = valid & (R_centres >= R_cut)
            if not mask.any():
                return
            idx0 = np.where(mask)[0][0]
            R_fit = np.logspace(np.log10(R_cut),
                                np.log10(R_centres[valid].max()), 50)
        R_norm = R_centres[idx0]
        dNdR_norm = dNdR[idx0]
        dNdR_fit = dNdR_norm * (R_fit / R_norm) ** slope_val
        if np.isfinite(slope_err):
            label = f"$\\gamma={slope_val:.2f} \\pm {slope_err:.2f}$"
        else:
            label = f"$\\gamma={slope_val:.2f}$"
        ax.plot(R_fit, dNdR_fit, color=color, ls="--", lw=1.5, label=label)

    _add_line(synth["synth_slope"], synth.get("synth_slope_err", np.nan),
              R_complete, C_VERMILLION)
    _add_line(synth.get("synth_slope_nath", np.nan),
              synth.get("synth_slope_err_nath", np.nan),
              R_COMPLETE_NATH20, C_ORANGE)

    # Observed reference
    if valid.sum() >= 2:
        R_ref = np.logspace(np.log10(R_centres[valid].min()),
                            np.log10(R_centres[valid].max()), 50)
        R_mid = np.sqrt(R_centres[valid].min() * R_centres[valid].max())
        dN_mid = np.interp(np.log10(R_mid), np.log10(R_centres[valid]),
                           np.log10(dNdR[valid]))
        dNdR_obs = 10.0 ** (dN_mid + OBSERVED_ALPHA *
                            (np.log10(R_ref) - np.log10(R_mid)))
        ax.plot(R_ref, dNdR_obs, color=C_GREEN, ls="-", lw=1.5, alpha=0.35,
                label=r"$R^{-2.2}$ (Watkins+23)")

    ax.axvline(R_complete, color=C_VERMILLION, ls=":", lw=1.0, alpha=0.6)
    ax.axvline(R_COMPLETE_NATH20, color=C_ORANGE, ls=":", lw=1.0, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}R$")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, framealpha=0.7)


def plot_synthesis_comparison(
    synth_interp_list: List[Dict],
    synth_cmf_list: List[Dict],
    synth_avg_list: List[Dict],
    output_dir: Path,
    fmt: str = "pdf",
) -> None:
    """
    3x2 comparison figure: one row per method, one column per t_obs.

    Row 0: Option C (M_star interpolation)
    Row 1: Option A (cloud MF sampling)
    Row 2: Option B (SFE averaging)

    Each panel shows dN/dR scatter + MLE fit lines.

    Parameters
    ----------
    synth_interp_list : list of dict
        Option C synthesis outputs (one per t_obs).
    synth_cmf_list : list of dict
        Option A synthesis outputs (one per t_obs).
    synth_avg_list : list of dict
        Option B synthesis outputs (one per t_obs).
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    """
    method_lists = [
        ("Option C: $M_\\star$ interp", synth_interp_list),
        ("Option A: CMF sampling", synth_cmf_list),
        ("Option B: $\\varepsilon$ averaging", synth_avg_list),
    ]

    # Determine number of columns from the longest list
    n_cols = max(len(lst) for _, lst in method_lists if lst)
    n_rows = sum(1 for _, lst in method_lists if lst)

    if n_rows == 0 or n_cols == 0:
        logger.warning("No synthesis data for comparison plot")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    row = 0
    for method_label, synth_list in method_lists:
        if not synth_list:
            continue
        for col, synth in enumerate(synth_list):
            if col >= n_cols:
                break
            title = f"{method_label}, $t_{{\\rm obs}}={synth['t_obs']:.0f}$ Myr"
            _plot_dNdR_panel(axes[row, col], synth, title=title)
        # Blank out unused columns
        for col in range(len(synth_list), n_cols):
            axes[row, col].set_visible(False)
        row += 1

    fig.tight_layout()
    path = output_dir / f"bubble_synthesis_comparison.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_ndot_inferred(records: List[Dict], output_dir: Path,
                       fmt: str = "pdf") -> None:
    """
    Figure 4: Inferred N_dot(R)/N_dot(R_0) vs R.

    Tests single-trajectory self-consistency only: if N_dot is constant,
    lines should be horizontal.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    """
    records = _assign_styles(records)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for rec in records:
        R = rec["R_exp"]
        ndot = rec["ndot_norm"]
        if len(R) < 2:
            continue
        ax.plot(R, ndot, color=rec["_color"], ls=rec["_ls"],
                lw=1.5, label=_model_label(rec), alpha=0.85)

    ax.axhline(1.0, color="grey", ls=":", lw=1.0, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\dot{N}(R)\;/\;\dot{N}(R_0)$")

    if len(records) <= 8:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    ax.annotate("Single-trajectory self-consistency test",
                xy=(0.05, 0.95), xycoords="axes fraction",
                ha="left", va="top", fontsize=8, style="italic",
                color="grey")

    fig.tight_layout()
    path = output_dir / f"bubble_ndot_inferred.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    """
    Write per-run summary CSV.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    csv_path = output_dir / "bubble_distribution_results.csv"
    header = ["folder", "mCloud", "sfe", "nCore", "M_star", "outcome",
              "R_min_pc", "R_max_pc", "residence_slope", "residence_R2"]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rec in records:
            writer.writerow([
                rec["folder"],
                f"{rec['mCloud']:.0e}",
                f"{rec['sfe']:.3f}",
                f"{rec['nCore']:.0e}",
                f"{rec['M_star']:.2e}",
                rec["outcome"],
                f"{rec['R_exp'].min():.3f}" if len(rec["R_exp"]) > 0 else "N/A",
                f"{rec['R_exp'].max():.3f}" if len(rec["R_exp"]) > 0 else "N/A",
                f"{rec['residence_slope']:.4f}" if np.isfinite(rec["residence_slope"]) else "N/A",
                f"{rec['residence_R2']:.4f}" if np.isfinite(rec["residence_R2"]) else "N/A",
            ])

    logger.info("Saved: %s", csv_path)
    return csv_path


def write_synthesis_csv(synth_input, sensitivity: List[Dict],
                        output_dir: Path) -> Path:
    """
    Write synthesis summary CSV.

    Parameters
    ----------
    synth_input : dict, list of dict, or None
        Output(s) of run_population_synthesis().
    sensitivity : list of dict
        Output of run_sensitivity().
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    # Normalise: accept single dict or list
    if synth_input is None:
        synth_all = []
    elif isinstance(synth_input, dict):
        synth_all = [synth_input]
    else:
        synth_all = list(synth_input)

    csv_path = output_dir / "bubble_synthesis_summary.csv"
    header = ["method", "cmf_slope", "t_obs_Myr", "R_complete_pc", "N_bubble",
              "N_surviving",
              "N_fit_mle30", "mle_slope_30", "mle_slope_err_30",
              "mle_slope_std",
              "N_fit_mle_le10", "mle_slope_le10", "mle_slope_err_le10",
              "N_fit_mle100", "mle_slope_100", "mle_slope_err_100",
              "bin_N_le10", "bin_slope_le10", "bin_slope_err_le10",
              "bin_N_30", "bin_slope_30", "bin_slope_err_30",
              "bin_N_100", "bin_slope_100", "bin_slope_err_100",
              "cloud_mf_slope", "fixed_sfe_used",
              "sfe_prior_median", "sfe_prior_sigma",
              "runs_used"]

    def _fmt(v):
        return f"{v:.4f}" if np.isfinite(v) else "N/A"

    rows = []
    for synth in synth_all:
        rows.append([
            synth.get("method", "interp"),
            f"{synth['cmf_slope']:.2f}",
            f"{synth['t_obs']:.1f}",
            f"{synth['R_complete']:.1f}",
            synth["N_bubble"],
            synth["N_surviving"],
            synth.get("N_fit", ""),
            _fmt(synth["synth_slope"]),
            _fmt(synth.get("synth_slope_err", np.nan)),
            "",
            synth.get("N_fit_le10", ""),
            _fmt(synth.get("synth_slope_le10", np.nan)),
            _fmt(synth.get("synth_slope_err_le10", np.nan)),
            synth.get("N_fit_nath", ""),
            _fmt(synth.get("synth_slope_nath", np.nan)),
            _fmt(synth.get("synth_slope_err_nath", np.nan)),
            synth.get("bin_N_le10", ""),
            _fmt(synth.get("bin_slope_le10", np.nan)),
            _fmt(synth.get("bin_slope_err_le10", np.nan)),
            synth.get("bin_N_30", ""),
            _fmt(synth.get("bin_slope_30", np.nan)),
            _fmt(synth.get("bin_slope_err_30", np.nan)),
            synth.get("bin_N_100", ""),
            _fmt(synth.get("bin_slope_100", np.nan)),
            _fmt(synth.get("bin_slope_err_100", np.nan)),
            _fmt(synth.get("cloud_mf_slope", np.nan)),
            _fmt(synth.get("fixed_sfe_used", np.nan)),
            _fmt(synth.get("sfe_prior_median", np.nan)),
            _fmt(synth.get("sfe_prior_sigma", np.nan)),
            synth["runs_used"],
        ])

    for s in sensitivity:
        slope_err = s.get("synth_slope_err", np.nan)
        slope_std = s.get("synth_slope_std", np.nan)
        rows.append([
            "sensitivity",
            f"{s['cmf_slope']:.2f}",
            f"{s['t_obs']:.1f}",
            "",
            "",
            s["N_surviving"],
            s.get("N_fit", ""),
            _fmt(s["synth_slope"]),
            _fmt(slope_err),
            _fmt(slope_std),
            "", "", "",
            "", "", "",
            "", "", "",
            "", "", "",
            "", "", "",
            "", "",
            "", "",
            "",
        ])

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(records: List[Dict],
                  synth_list: Optional[list] = None,
                  sensitivity: List[Dict] = None) -> None:
    """
    Print summary table to stdout.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    synth_list : list of dict, optional
        Outputs of run_population_synthesis() (one per t_obs).
        Also accepts a single dict for backwards compatibility.
    sensitivity : list of dict, optional
        Output of run_sensitivity().
    """
    # Normalise input: accept a single dict or a list
    if synth_list is not None and isinstance(synth_list, dict):
        synth_list = [synth_list]

    print()
    print("=" * 80)
    print("BUBBLE SIZE DISTRIBUTION SUMMARY")
    print("=" * 80)

    # Per-run table
    print()
    print("-" * 80)
    print(f"  {'Folder':<35s} {'Outcome':<10s} {'M_star':>10s} "
          f"{'Res.Slope':>10s} {'R2':>8s} {'R_min':>8s} {'R_max':>8s}")
    print("-" * 80)
    for rec in records:
        sl = f"{rec['residence_slope']:.3f}" if np.isfinite(rec["residence_slope"]) else "N/A"
        r2 = f"{rec['residence_R2']:.3f}" if np.isfinite(rec["residence_R2"]) else "N/A"
        rmin = f"{rec['R_exp'].min():.2f}" if len(rec["R_exp"]) > 0 else "N/A"
        rmax = f"{rec['R_exp'].max():.2f}" if len(rec["R_exp"]) > 0 else "N/A"
        mstar = f"{rec['M_star']:.2e}"
        print(f"  {rec['folder']:<35s} {rec['outcome']:<10s} {mstar:>10s} "
              f"{sl:>10s} {r2:>8s} {rmin:>8s} {rmax:>8s}")

    def _slope_str(val, err):
        if not np.isfinite(val):
            return "N/A"
        e = f" +/- {err:.3f}" if np.isfinite(err) else ""
        return f"{val:.3f}{e}"

    def _print_synth_block(synth):
        method = synth.get("method", "interp")
        method_labels = {
            "interp": "Option C: M_star interpolation",
            "cmf": "Option A: Cloud MF sampling",
            "avg": "Option B: SFE averaging",
        }
        print()
        print("-" * 80)
        header = method_labels.get(method, method)
        print(f"  Population synthesis — {header} "
              f"(t_obs = {synth['t_obs']:.1f} Myr):")
        print(f"    MF slope     = {synth['cmf_slope']:.1f}")
        if method == "cmf":
            print(f"    Fixed SFE    = {synth.get('fixed_sfe_used', 'N/A')}")
        elif method == "avg":
            print(f"    SFE prior    = median {synth.get('sfe_prior_median', 'N/A')}, "
                  f"sigma {synth.get('sfe_prior_sigma', 'N/A')} dex")
        print(f"    N_bubble     = {synth['N_bubble']}")
        print(f"    R_complete   = {synth['R_complete']:.0f} pc")
        print(f"    N_surviving  = {synth['N_surviving']}")
        print(f"    N_recollapsed = {synth['N_recollapsed']}")
        print(f"    N_below_cut  = {synth['N_below_complete']}")

        print(f"    --- MLE fits ---")
        print(f"    R<={R_COMPLETE_10PC:.0f} pc : "
              f"slope = {_slope_str(synth.get('synth_slope_le10', np.nan), synth.get('synth_slope_err_le10', np.nan))}"
              f"  (N={synth.get('N_fit_le10', 'N/A')})")
        print(f"    R>={synth['R_complete']:.0f} pc (Watkins+23): "
              f"slope = {_slope_str(synth['synth_slope'], synth.get('synth_slope_err', np.nan))}"
              f"  (N={synth.get('N_fit', 'N/A')})  "
              f"[observed = {OBSERVED_ALPHA}]")
        print(f"    R>={R_COMPLETE_NATH20:.0f} pc (Nath+20)  : "
              f"slope = {_slope_str(synth.get('synth_slope_nath', np.nan), synth.get('synth_slope_err_nath', np.nan))}"
              f"  (N={synth.get('N_fit_nath', 'N/A')})")
        print(f"    --- Binned LSQ fits (60 log-bins) ---")
        print(f"    R<={R_COMPLETE_10PC:.0f} pc : "
              f"slope = {_slope_str(synth.get('bin_slope_le10', np.nan), synth.get('bin_slope_err_le10', np.nan))}"
              f"  (N_bins={synth.get('bin_N_le10', 'N/A')})")
        print(f"    R>={synth['R_complete']:.0f} pc : "
              f"slope = {_slope_str(synth.get('bin_slope_30', np.nan), synth.get('bin_slope_err_30', np.nan))}"
              f"  (N_bins={synth.get('bin_N_30', 'N/A')})")
        print(f"    R>={R_COMPLETE_NATH20:.0f} pc: "
              f"slope = {_slope_str(synth.get('bin_slope_100', np.nan), synth.get('bin_slope_err_100', np.nan))}"
              f"  (N_bins={synth.get('bin_N_100', 'N/A')})")
        if synth.get("runs_used"):
            print(f"    Runs used    : {synth['runs_used']}")

    if synth_list:
        for synth in synth_list:
            _print_synth_block(synth)

    if sensitivity:
        print()
        print("-" * 80)
        print("  Parameter sensitivity:")
        print(f"    {'CMF slope':>10s} {'t_obs':>8s} {'Synth slope':>18s} "
              f"{'N_surv':>8s} {'n_runs':>7s}")
        print("    " + "-" * 55)
        for s in sensitivity:
            slope_std = s.get("synth_slope_std", np.nan)
            if np.isfinite(s["synth_slope"]):
                if np.isfinite(slope_std) and slope_std > 0:
                    sl = f"{s['synth_slope']:.3f} +/- {slope_std:.3f}"
                else:
                    sl = f"{s['synth_slope']:.3f}"
            else:
                sl = "N/A"
            n_runs = s.get("n_runs", "")
            print(f"    {s['cmf_slope']:>10.1f} {s['t_obs']:>8.1f} "
                  f"{sl:>18s} {s['N_surviving']:>8d} {str(n_runs):>7s}")

    print()
    print("=" * 80)


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Population-convolved bubble size distribution from TRINITY outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bubble_distribution.py -F /path/to/sweep_output
  python bubble_distribution.py -F /path/to/sweep_output --N-bubble 5000 --fmt png
  python bubble_distribution.py -F /path/to/sweep_output --cmf-slope -1.8 --R-complete 20
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--N-bubble", type=int, default=20000,
        help="Number of synthetic bubbles for population synthesis (default: 20000).",
    )
    parser.add_argument(
        "--t-obs", type=float, default=5.0,
        help="Observation time for synthesis [Myr] (default: 5.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for synthesis (default: 42).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--no-synthesis", action="store_true",
        help="Skip population synthesis (Parts 2 and 3).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Maximum time [Myr] to consider in calculations.",
    )
    parser.add_argument(
        "--cmf-slope", type=float, default=-2.0,
        help="Cluster mass function slope dN/dM proportional to M^alpha (default: -2.0).",
    )
    parser.add_argument(
        "--R-complete", type=float, default=30.0,
        help="Completeness radius [pc] (Watkins+2023 turnover, default: 30.0).",
    )
    parser.add_argument(
        "--fixed-sfe", type=float, default=None,
        help="If set, only use runs with this SFE for synthesis.",
    )
    parser.add_argument(
        "--fixed-ncore", type=float, default=None,
        help="If set, only use runs with this core density for synthesis.",
    )
    parser.add_argument(
        "--density-profile", type=str, default="all",
        choices=["PL0", "uniform", "all"],
        help=("Filter runs by density profile: 'PL0' keeps only folders "
              "containing '_PL0', 'uniform' keeps only folders without '_PL0', "
              "'all' keeps everything (default: all)."),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory override (default: fig/<folder>/).",
    )
    parser.add_argument(
        "--synthesis-method", type=str, default="all",
        choices=["interp", "cmf", "avg", "all"],
        help=("Synthesis method: 'interp' (Option C, current), "
              "'cmf' (Option A, cloud MF sampling), "
              "'avg' (Option B, degenerate averaging), "
              "'all' (run all three). Default: all."),
    )
    parser.add_argument(
        "--cloud-mf-slope", type=float, default=-1.7,
        help="Cloud mass function slope for Option A (default: -1.7).",
    )
    parser.add_argument(
        "--cmf-fixed-sfe", type=float, default=0.01,
        help="Fixed SFE for Option A cloud MF sampling (default: 0.01).",
    )
    parser.add_argument(
        "--sfe-prior-median", type=float, default=0.05,
        help="Median SFE for Option B log-normal prior (default: 0.05).",
    )
    parser.add_argument(
        "--sfe-prior-sigma", type=float, default=0.5,
        help="Sigma [dex] for Option B log-normal SFE prior (default: 0.5).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        logger.error("Folder does not exist: %s", folder_path)
        return 1

    folder_name = folder_path.name
    output_dir = Path(args.output_dir) if args.output_dir else FIG_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: collect & analyse
    records = collect_data(folder_path, t_end=args.t_end)
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Filter by density profile if requested
    density_profile = args.density_profile
    if density_profile == "PL0":
        before = len(records)
        records = [r for r in records if "_PL0" in r["folder"]]
        logger.info("Density profile filter 'PL0': %d -> %d runs",
                     before, len(records))
    elif density_profile == "uniform":
        before = len(records)
        records = [r for r in records if "_PL0" not in r["folder"]]
        logger.info("Density profile filter 'uniform': %d -> %d runs",
                     before, len(records))

    if not records:
        logger.error("No runs remain after density-profile filter — aborting.")
        return 1

    # Warn if mixing different n_core values without --fixed-ncore
    unique_ncores = sorted(set(r["nCore"] for r in records))
    if len(unique_ncores) > 1 and args.fixed_ncore is None:
        logger.warning(
            "Mixing %d different nCore values %s without --fixed-ncore. "
            "Consider setting --fixed-ncore to avoid incoherent grid matching.",
            len(unique_ncores),
            [f"{n:.0e}" for n in unique_ncores],
        )

    # Step 2: Part 1 figures (diagnostics)
    plot_residence_time(records, output_dir, args.fmt)
    plot_ndot_inferred(records, output_dir, args.fmt)

    # Step 3: population synthesis (Parts 2 & 3)
    synth_list = []
    synth_cmf_list = []
    synth_avg_list = []
    all_synths_for_csv = []
    sensitivity = []
    T_OBS_SECOND = 10.0  # second t_obs for comparison
    method = args.synthesis_method

    if not args.no_synthesis:
        t_obs_values = [args.t_obs]
        if args.t_obs != T_OBS_SECOND:
            t_obs_values.append(T_OBS_SECOND)

        # --- Option C: M_star interpolation (existing) ---
        if method in ("interp", "all"):
            for t_obs_run in t_obs_values:
                logger.info("Option C (interp): N=%d, t_obs=%.1f Myr, "
                             "CMF slope=%.1f, R_complete=%.0f pc",
                             args.N_bubble, t_obs_run, args.cmf_slope,
                             args.R_complete)
                s = run_population_synthesis(
                    records,
                    N_bubble=args.N_bubble,
                    t_obs=t_obs_run,
                    cmf_slope=args.cmf_slope,
                    R_complete=args.R_complete,
                    seed=args.seed,
                    fixed_sfe=args.fixed_sfe,
                    fixed_ncore=args.fixed_ncore,
                )
                if s is not None:
                    synth_list.append(s)
                    all_synths_for_csv.append(s)
                else:
                    logger.warning("Option C returned no results for "
                                   "t_obs=%.1f Myr.", t_obs_run)

            if synth_list:
                plot_synthesis_population(synth_list, output_dir, args.fmt)

        # --- Option A: Cloud MF sampling ---
        if method in ("cmf", "all"):
            for t_obs_run in t_obs_values:
                logger.info("Option A (CMF): N=%d, t_obs=%.1f Myr, "
                             "cloud MF slope=%.1f, fixed_sfe=%.3f",
                             args.N_bubble, t_obs_run,
                             args.cloud_mf_slope, args.cmf_fixed_sfe)
                s_cmf = run_population_synthesis_cmf(
                    records,
                    N_bubble=args.N_bubble,
                    t_obs=t_obs_run,
                    cloud_mf_slope=args.cloud_mf_slope,
                    fixed_sfe=args.cmf_fixed_sfe,
                    R_complete=args.R_complete,
                    seed=args.seed,
                    fixed_ncore=args.fixed_ncore,
                )
                if s_cmf is not None:
                    synth_cmf_list.append(s_cmf)
                    all_synths_for_csv.append(s_cmf)
                else:
                    logger.warning("Option A returned no results for "
                                   "t_obs=%.1f Myr.", t_obs_run)

            if synth_cmf_list:
                plot_synthesis_population(synth_cmf_list, output_dir, args.fmt,
                                          filename_prefix="bubble_synthesis_cmf")

        # --- Option B: SFE averaging ---
        if method in ("avg", "all"):
            for t_obs_run in t_obs_values:
                logger.info("Option B (avg): N=%d, t_obs=%.1f Myr, "
                             "CMF slope=%.1f, SFE prior median=%.3f "
                             "sigma=%.1f dex",
                             args.N_bubble, t_obs_run, args.cmf_slope,
                             args.sfe_prior_median, args.sfe_prior_sigma)
                s_avg = run_population_synthesis_avg(
                    records,
                    N_bubble=args.N_bubble,
                    t_obs=t_obs_run,
                    cmf_slope=args.cmf_slope,
                    R_complete=args.R_complete,
                    seed=args.seed,
                    fixed_ncore=args.fixed_ncore,
                    sfe_prior_median=args.sfe_prior_median,
                    sfe_prior_sigma=args.sfe_prior_sigma,
                )
                if s_avg is not None:
                    synth_avg_list.append(s_avg)
                    all_synths_for_csv.append(s_avg)
                else:
                    logger.warning("Option B returned no results for "
                                   "t_obs=%.1f Myr.", t_obs_run)

            if synth_avg_list:
                plot_synthesis_population(synth_avg_list, output_dir, args.fmt,
                                          filename_prefix="bubble_synthesis_avg")

        # --- Comparison figure (all three methods) ---
        if method == "all" and (synth_list or synth_cmf_list or synth_avg_list):
            plot_synthesis_comparison(synth_list, synth_cmf_list,
                                      synth_avg_list, output_dir, args.fmt)

        # --- Part 3: sensitivity analysis ---
        logger.info("Running sensitivity analysis (Option C: interp)...")
        sensitivity = run_sensitivity(
            records,
            N_bubble=args.N_bubble,
            seed=args.seed,
            R_complete=args.R_complete,
            fixed_sfe=args.fixed_sfe,
            fixed_ncore=args.fixed_ncore,
            method="interp",
        )
        if sensitivity:
            plot_slope_vs_cmf(sensitivity, output_dir, args.fmt)

        if method in ("cmf", "all"):
            logger.info("Running sensitivity analysis (Option A: CMF)...")
            sens_cmf = run_sensitivity(
                records,
                N_bubble=args.N_bubble,
                seed=args.seed,
                R_complete=args.R_complete,
                fixed_ncore=args.fixed_ncore,
                method="cmf",
                cmf_fixed_sfe=args.cmf_fixed_sfe,
            )
            if sens_cmf:
                plot_slope_vs_cmf(
                    sens_cmf, output_dir, args.fmt,
                    filename="bubble_slope_vs_cmf_cmf",
                    xlabel=(r"Cloud MF slope $\beta$ "
                            r"($\mathrm{d}N/\mathrm{d}M_{\rm cl}"
                            r" \propto M_{\rm cl}^\beta$)"),
                )
                sensitivity.extend(sens_cmf)

        if method in ("avg", "all"):
            logger.info("Running sensitivity analysis (Option B: avg)...")
            sens_avg = run_sensitivity(
                records,
                N_bubble=args.N_bubble,
                seed=args.seed,
                R_complete=args.R_complete,
                fixed_ncore=args.fixed_ncore,
                method="avg",
                sfe_prior_median=args.sfe_prior_median,
                sfe_prior_sigma=args.sfe_prior_sigma,
            )
            if sens_avg:
                plot_slope_vs_cmf(
                    sens_avg, output_dir, args.fmt,
                    filename="bubble_slope_vs_cmf_avg",
                )
                sensitivity.extend(sens_avg)

    # Step 4: output
    write_results_csv(records, output_dir)
    # Combine all synth lists for summary
    all_synth_for_summary = synth_list + synth_cmf_list + synth_avg_list
    if not args.no_synthesis:
        write_synthesis_csv(all_synths_for_csv if all_synths_for_csv else None,
                            sensitivity, output_dir)
    print_summary(records,
                  all_synth_for_summary if all_synth_for_summary else None,
                  sensitivity if sensitivity else None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
