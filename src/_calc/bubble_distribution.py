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


def _match_to_grid(
    sampled_masses: np.ndarray,
    grid_masses: np.ndarray,
) -> np.ndarray:
    """
    For each sampled mass, find the index of the closest grid mass in log-space.

    Parameters
    ----------
    sampled_masses : ndarray
        Sampled M_star values.
    grid_masses : ndarray
        M_star values from the TRINITY grid.

    Returns
    -------
    ndarray of int
        Index into grid_masses for each sample.
    """
    log_grid = np.log10(grid_masses)
    log_samp = np.log10(sampled_masses)
    # For each sample, find nearest grid point
    indices = np.argmin(np.abs(log_samp[:, None] - log_grid[None, :]), axis=1)
    return indices


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

    # Build grid of M_star values
    grid_Mstar = np.array([r["M_star"] for r in filtered])
    M_star_min = grid_Mstar.min()
    M_star_max = grid_Mstar.max()

    if M_star_min <= 0 or M_star_max <= 0:
        logger.warning("Invalid M_star range [%.2e, %.2e]", M_star_min, M_star_max)
        return None

    logger.info("M_star grid: [%.2e, %.2e] Msun (%d runs)",
                M_star_min, M_star_max, len(filtered))

    # Prepare interpolation arrays for each run (deduplicate times)
    interp_data = []
    for rec in filtered:
        t_full = rec["t_full"]
        R_full = rec["R_full"]
        _, unique_idx = np.unique(t_full, return_index=True)
        interp_data.append({
            "t": t_full[unique_idx],
            "R": R_full[unique_idx],
        })

    rng = np.random.default_rng(seed)

    # Step 1: sample cluster masses
    sampled_Mstar = _sample_powerlaw(N_bubble, M_star_min, M_star_max,
                                     cmf_slope, rng)

    # Step 2: match to nearest grid run
    match_idx = _match_to_grid(sampled_Mstar, grid_Mstar)

    # Step 3: assign random birth times and evaluate R(age)
    t_form = rng.uniform(0.0, t_obs, size=N_bubble)
    ages = t_obs - t_form

    R_synth = np.full(N_bubble, np.nan)
    run_usage = np.zeros(len(filtered), dtype=int)

    for i in range(N_bubble):
        idx = match_idx[i]
        run_usage[idx] += 1
        td = interp_data[idx]
        t_run = td["t"]
        R_run = td["R"]
        age = ages[i]

        if age < t_run.min() or age > t_run.max():
            # Clamp to simulation range
            age_c = np.clip(age, t_run.min(), t_run.max())
        else:
            age_c = age

        R_synth[i] = np.interp(age_c, t_run, R_run)

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

    # Step 5: histogram in log-space over the FULL R range (no R_complete cut)
    log_R = np.log10(R_valid)
    n_bins = 25
    log_edges = np.linspace(log_R.min(), log_R.max(), n_bins + 1)
    counts, _ = np.histogram(log_R, bins=log_edges)

    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])
    dNdR = counts / dR

    # Step 6: fit power law to non-zero bins above R_complete
    fit_mask = (dNdR > 0) & (R_centres >= R_complete)
    synth_slope = np.nan
    synth_R2 = np.nan
    if fit_mask.sum() >= 5:
        logR_c = np.log10(R_centres[fit_mask])
        log_dN = np.log10(dNdR[fit_mask])
        X = np.column_stack([np.ones_like(logR_c), logR_c])
        try:
            beta = np.linalg.lstsq(X, log_dN, rcond=None)[0]
            synth_slope = beta[1]
            resid = log_dN - X @ beta
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((log_dN - log_dN.mean()) ** 2)
            synth_R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except np.linalg.LinAlgError:
            pass

    # Build run-usage summary
    runs_used_summary = []
    for j, rec in enumerate(filtered):
        if run_usage[j] > 0:
            runs_used_summary.append(
                f"{rec['folder']}({run_usage[j]})")

    return {
        "R_synth": R_valid,
        "R_centres": R_centres,
        "dNdR": dNdR,
        "synth_slope": synth_slope,
        "synth_R2": synth_R2,
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
) -> List[Dict]:
    """
    Run synthesis for a grid of CMF slopes and t_obs values.

    Each (cmf_slope, t_obs) point is repeated n_bootstrap times with
    independent seeds to estimate mean ± std of the fitted slope.

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
        CMF slopes to test (default: [-1.5, -2.0, -2.5]).
    t_obs_values : list of float, optional
        Observation times to test [Myr] (default: [3, 5, 10]).
    fixed_sfe : float, optional
        If set, only use runs with this SFE.
    fixed_ncore : float, optional
        If set, only use runs with this core density.
    n_bootstrap : int
        Number of independent runs per (cmf_slope, t_obs) for error bars.

    Returns
    -------
    list of dict
        Each entry has: cmf_slope, t_obs, synth_slope, synth_slope_std,
        synth_R2, N_surviving, n_runs.
    """
    if cmf_slopes is None:
        cmf_slopes = [-1.5, -2.0, -2.5]
    if t_obs_values is None:
        t_obs_values = [3.0, 5.0, 10.0]

    results = []
    for cmf_sl in cmf_slopes:
        for t_obs in t_obs_values:
            # Deterministic but independent base seed for each configuration
            run_seed = seed + hash((cmf_sl, t_obs)) % (2**31)

            slopes = []
            last_N_surviving = 0
            for k in range(n_bootstrap):
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
                    if np.isfinite(synth["synth_slope"]):
                        slopes.append(synth["synth_slope"])

            if slopes:
                results.append({
                    "cmf_slope": cmf_sl,
                    "t_obs": t_obs,
                    "synth_slope": np.mean(slopes),
                    "synth_slope_std": np.std(slopes) if len(slopes) > 1 else 0.0,
                    "synth_R2": np.nan,
                    "N_surviving": last_N_surviving,
                    "n_runs": len(slopes),
                })
            else:
                results.append({
                    "cmf_slope": cmf_sl,
                    "t_obs": t_obs,
                    "synth_slope": np.nan,
                    "synth_slope_std": np.nan,
                    "synth_R2": np.nan,
                    "N_surviving": 0,
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


def plot_synthesis_population(synth: Dict, output_dir: Path,
                              fmt: str = "pdf") -> None:
    """
    Figure 2: Population synthesis results.

    Two panels: (a) histogram of synthetic bubble radii, (b) dN/dR vs R
    with power-law fit and Watkins+2023 reference.

    Parameters
    ----------
    synth : dict
        Output of run_population_synthesis().
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    R_synth = synth["R_synth"]
    R_centres = synth["R_centres"]
    dNdR = synth["dNdR"]
    R_complete = synth["R_complete"]

    # Panel (a): histogram of full population, fade region below R_complete
    ax = axes[0]
    ax.hist(R_synth, bins=30, color=C_SKY, edgecolor=C_BLUE,
            alpha=0.7, density=False)
    ax.axvline(R_complete, color=C_VERMILLION, ls=":", lw=1.2, alpha=0.8,
               label=f"$R_{{\\rm complete}}={R_complete:.0f}$ pc")
    # Shade the incomplete region
    ax.axvspan(ax.get_xlim()[0], R_complete, color="grey", alpha=0.15,
               zorder=0)
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$N$")
    # Combine annotation info into the legend title to avoid overlap
    ax.legend(
        fontsize=8, framealpha=0.7,
        title=(f"$N={synth['N_bubble']}$, "
               f"$t_{{\\rm obs}}={synth['t_obs']:.1f}$ Myr, "
               f"CMF $\\alpha={synth['cmf_slope']:.1f}$"),
        title_fontsize=7,
    )

    # Panel (b): dN/dR vs R — show all bins, fade below R_complete
    ax = axes[1]
    valid = dNdR > 0
    above = valid & (R_centres >= R_complete)
    below = valid & (R_centres < R_complete)
    ax.scatter(R_centres[above], dNdR[above], color=C_BLUE, s=25,
               zorder=3, label="Synthesis")
    if below.any():
        ax.scatter(R_centres[below], dNdR[below], color=C_BLUE, s=25,
                   zorder=3, alpha=0.25)

    # Power-law fit (only above R_complete)
    slope = synth["synth_slope"]
    if np.isfinite(slope):
        fit_mask = valid & (R_centres >= R_complete)
        if fit_mask.sum() >= 2:
            logR_c = np.log10(R_centres[fit_mask])
            log_dN = np.log10(dNdR[fit_mask])
            X = np.column_stack([np.ones_like(logR_c), logR_c])
            beta = np.linalg.lstsq(X, log_dN, rcond=None)[0]
            R_fit = np.logspace(np.log10(R_centres[fit_mask].min()),
                                np.log10(R_centres[fit_mask].max()), 50)
            dNdR_fit = 10.0 ** (beta[0] + beta[1] * np.log10(R_fit))
            ax.plot(R_fit, dNdR_fit, color=C_VERMILLION, ls="--", lw=1.5,
                    label=f"Fit: $\\gamma={slope:.2f}$")

    # Observed reference
    if valid.sum() >= 2:
        R_ref = np.logspace(np.log10(R_centres[valid].min()),
                            np.log10(R_centres[valid].max()), 50)
        R_mid = np.sqrt(R_centres[valid].min() * R_centres[valid].max())
        dN_mid = np.interp(np.log10(R_mid), np.log10(R_centres[valid]),
                           np.log10(dNdR[valid]))
        dNdR_obs = 10.0 ** (dN_mid + OBSERVED_ALPHA * (np.log10(R_ref) - np.log10(R_mid)))
        ax.plot(R_ref, dNdR_obs, color=C_GREEN, ls="-.", lw=1.5,
                label=r"$R^{-2.2}$ (Watkins+2023)")

    # Completeness line and shaded incomplete region
    ax.axvline(R_complete, color=C_VERMILLION, ls=":", lw=1.0, alpha=0.6)
    ax.axvspan(ax.get_xlim()[0], R_complete, color="grey", alpha=0.10,
               zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}R$")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_synthesis_population.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_slope_vs_cmf(sensitivity: List[Dict], output_dir: Path,
                      fmt: str = "pdf") -> None:
    """
    Figure 3: Predicted dN/dR slope vs CMF slope for multiple t_obs.

    Parameters
    ----------
    sensitivity : list of dict
        Output of run_sensitivity().
    output_dir : Path
        Figure output directory.
    fmt : str
        Figure format.
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

    ax.set_xlabel(r"CMF slope $\alpha$ ($\mathrm{d}N/\mathrm{d}M_\star \propto M_\star^\alpha$)")
    ax.set_ylabel(r"Synthetic $\mathrm{d}N/\mathrm{d}R$ slope $\gamma$")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_slope_vs_cmf.{fmt}"
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


def write_synthesis_csv(synth: Optional[Dict], sensitivity: List[Dict],
                        output_dir: Path) -> Path:
    """
    Write synthesis summary CSV.

    Parameters
    ----------
    synth : dict or None
        Output of run_population_synthesis() (primary run).
    sensitivity : list of dict
        Output of run_sensitivity().
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the written CSV.
    """
    csv_path = output_dir / "bubble_synthesis_summary.csv"
    header = ["cmf_slope", "t_obs_Myr", "R_complete_pc", "N_bubble",
              "N_surviving", "synth_slope", "synth_slope_std", "synth_R2",
              "runs_used"]

    rows = []
    if synth is not None:
        rows.append([
            f"{synth['cmf_slope']:.2f}",
            f"{synth['t_obs']:.1f}",
            f"{synth['R_complete']:.1f}",
            synth["N_bubble"],
            synth["N_surviving"],
            f"{synth['synth_slope']:.4f}" if np.isfinite(synth["synth_slope"]) else "N/A",
            "",
            f"{synth['synth_R2']:.4f}" if np.isfinite(synth["synth_R2"]) else "N/A",
            synth["runs_used"],
        ])

    for s in sensitivity:
        slope_std = s.get("synth_slope_std", np.nan)
        rows.append([
            f"{s['cmf_slope']:.2f}",
            f"{s['t_obs']:.1f}",
            "",
            "",
            s["N_surviving"],
            f"{s['synth_slope']:.4f}" if np.isfinite(s["synth_slope"]) else "N/A",
            f"{slope_std:.4f}" if np.isfinite(slope_std) else "N/A",
            f"{s['synth_R2']:.4f}" if np.isfinite(s["synth_R2"]) else "N/A",
            "",
        ])

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(records: List[Dict], synth: Optional[Dict] = None,
                  sensitivity: List[Dict] = None) -> None:
    """
    Print summary table to stdout.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    synth : dict, optional
        Output of run_population_synthesis().
    sensitivity : list of dict, optional
        Output of run_sensitivity().
    """
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

    if synth is not None:
        print()
        print("-" * 80)
        print("  Population synthesis (primary run):")
        print(f"    CMF slope    = {synth['cmf_slope']:.1f}")
        print(f"    N_bubble     = {synth['N_bubble']}")
        print(f"    t_obs        = {synth['t_obs']:.1f} Myr")
        print(f"    R_complete   = {synth['R_complete']:.0f} pc")
        print(f"    N_surviving  = {synth['N_surviving']}")
        print(f"    N_recollapsed = {synth['N_recollapsed']}")
        print(f"    N_below_cut  = {synth['N_below_complete']}")
        if np.isfinite(synth["synth_slope"]):
            print(f"    Fitted slope = {synth['synth_slope']:.3f}  "
                  f"(observed = {OBSERVED_ALPHA})")
        else:
            print(f"    Fitted slope = N/A")
        if np.isfinite(synth["synth_R2"]):
            print(f"    R2           = {synth['synth_R2']:.3f}")
        print(f"    Runs used    : {synth['runs_used']}")

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
    output_dir = FIG_DIR / folder_name
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
    synth = None
    sensitivity = []
    if not args.no_synthesis:
        logger.info("Running population synthesis (N=%d, t_obs=%.1f Myr, "
                     "CMF slope=%.1f, R_complete=%.0f pc)...",
                     args.N_bubble, args.t_obs, args.cmf_slope, args.R_complete)

        synth = run_population_synthesis(
            records,
            N_bubble=args.N_bubble,
            t_obs=args.t_obs,
            cmf_slope=args.cmf_slope,
            R_complete=args.R_complete,
            seed=args.seed,
            fixed_sfe=args.fixed_sfe,
            fixed_ncore=args.fixed_ncore,
        )
        if synth is not None:
            plot_synthesis_population(synth, output_dir, args.fmt)
        else:
            logger.warning("Population synthesis returned no results.")

        # Part 3: sensitivity
        logger.info("Running parameter sensitivity analysis...")
        sensitivity = run_sensitivity(
            records,
            N_bubble=args.N_bubble,
            seed=args.seed,
            R_complete=args.R_complete,
            fixed_sfe=args.fixed_sfe,
            fixed_ncore=args.fixed_ncore,
        )
        if sensitivity:
            plot_slope_vs_cmf(sensitivity, output_dir, args.fmt)

    # Step 4: output
    write_results_csv(records, output_dir)
    if not args.no_synthesis:
        write_synthesis_csv(synth, sensitivity, output_dir)
    print_summary(records, synth, sensitivity if sensitivity else None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
