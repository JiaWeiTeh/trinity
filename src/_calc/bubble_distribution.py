#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bubble size distribution analysis from TRINITY outputs.

Physics background
------------------
Watkins et al. (2023) measured the bubble size distribution in NGC 628
and found dN/dR proportional to R^{-2.2}.  From a simple steady-state
model, the observed size distribution is

    dN/dR = N_dot / v(R),

where N_dot is the bubble formation rate (number of new bubbles per
unit time) and v(R) = dR/dt is the expansion velocity at radius R.
If the formation rate is constant and the expansion follows v ~ R^beta,
then dN/dR ~ R^{-beta} = R^{-(1-beta)}, giving a power-law index
of -(1 - beta) on the size distribution.

This script tests two complementary questions using TRINITY outputs:

**Part 1 — Inferred N_dot(R):**
    Assuming the *observed* dN/dR = A R^{-2.2} (Watkins+2023), compute
    the implied formation rate  N_dot(R) = (dN/dR) * v(R).  If N_dot(R)
    is constant, the model is self-consistent with a constant birth rate.
    Deviations indicate that a non-constant formation history or
    additional physics (e.g. stalling, gravity) is needed.

**Part 2 — Predicted dN/dR:**
    Assuming constant N_dot, compute dN/dR proportional to 1/v(R) for each
    TRINITY model.  Fit a power law over the expanding portion and compare
    the predicted slope to the observed -2.2.

**Part 3 — Forward population synthesis:**
    Draw N_bubble bubbles with uniformly distributed formation times
    t_form in [0, t_obs].  Evolve each with TRINITY (interpolated from
    pre-run output) to obtain R(t_obs - t_form).  Histogram the resulting
    R values to construct a synthetic dN/dR.  Compare to R^{-2.2}.

References
----------
* Watkins, E. J. et al. (2023), ApJS, 264, 16 — NGC 628 bubble catalogue.
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
from scipy.signal import savgol_filter
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

def extract_bubble_data(data_path: Path) -> Optional[Dict]:
    """
    Load one TRINITY run and extract R(t) and v(R) for the expanding phase.

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

    # Outcome
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
        # Full-run t and R for interpolation in Part 3
        "t_full": t,
        "R_full": R,
    }


def collect_data(folder_path: Path) -> List[Dict]:
    """Walk sweep directory and extract bubble data from each run."""
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

        info = extract_bubble_data(data_path)
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

        # Part 1: Inferred N_dot(R) = (dN/dR) * v(R)
        # dN/dR = A * R^{-2.2}, so N_dot(R) = A * R^{-2.2} * v(R)
        # Normalize: N_dot(R) / N_dot(R_0)
        ndot_unnorm = R_exp ** OBSERVED_ALPHA * v_exp
        R0_idx = 0  # normalize to smallest R in expanding range
        if ndot_unnorm[R0_idx] != 0:
            ndot_norm = ndot_unnorm / ndot_unnorm[R0_idx]
        else:
            ndot_norm = np.full_like(ndot_unnorm, np.nan)

        # Part 2: Predicted dN/dR proportional to 1/v(R)
        # dN/dR = C / v(R), fit power law
        dNdR_pred = 1.0 / v_exp
        # Normalize to first point for plotting
        dNdR_norm = dNdR_pred / dNdR_pred[0] if dNdR_pred[0] != 0 else dNdR_pred

        # Fit power law: log(dN/dR) = a + slope * log(R)
        # over the expanding range
        valid = (R_exp > 0) & (v_exp > 0) & np.isfinite(dNdR_pred)
        pred_slope = np.nan
        pred_R2 = np.nan
        if valid.sum() >= MIN_PTS:
            logR = np.log10(R_exp[valid])
            log_dNdR = np.log10(dNdR_pred[valid])
            # OLS fit
            X = np.column_stack([np.ones_like(logR), logR])
            try:
                beta = np.linalg.lstsq(X, log_dNdR, rcond=None)[0]
                pred_slope = beta[1]
                resid = log_dNdR - X @ beta
                ss_res = np.sum(resid ** 2)
                ss_tot = np.sum((log_dNdR - log_dNdR.mean()) ** 2)
                pred_R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except np.linalg.LinAlgError:
                pass

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "folder": info["folder"],
            "outcome": info["outcome"],
            # Expanding arrays sorted by R
            "R_exp": R_exp,
            "v_exp": v_exp,
            "t_exp": t_exp,
            # Part 1 results
            "ndot_norm": ndot_norm,
            # Part 2 results
            "dNdR_norm": dNdR_norm,
            "pred_slope": pred_slope,
            "pred_R2": pred_R2,
            # Full run for Part 3
            "t_full": info["t_full"],
            "R_full": info["R_full"],
        }
        records.append(rec)

    logger.info("Collected %d valid runs", len(records))
    return records


# ======================================================================
# Part 3: Forward population synthesis
# ======================================================================

def run_population_synthesis(
    records: List[Dict],
    N_bubble: int = 2000,
    t_obs: float = 5.0,
    seed: int = 42,
) -> Optional[Dict]:
    """
    Forward population synthesis: draw N_bubble bubbles with uniformly
    distributed formation times, evolve each with TRINITY, histogram R.

    Uses fiducial (first available) run for the R(t) relation.  If multiple
    runs exist, picks the one that reaches the largest radius.

    Parameters
    ----------
    records : list of dict
        Output of collect_data().
    N_bubble : int
        Number of synthetic bubbles to draw.
    t_obs : float
        Observation time [Myr].  Bubbles form at random t_form in [0, t_obs].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict or None
        Keys: R_synth, R_bins, dNdR_synth, synth_slope, synth_R2.
    """
    if not records:
        return None

    # Pick the run with the longest expanding trajectory
    best = max(records, key=lambda r: r["R_exp"].max() if len(r["R_exp"]) > 0 else 0)
    t_full = best["t_full"]
    R_full = best["R_full"]

    # Need a monotonically increasing t for interpolation
    # Remove duplicate times
    _, unique_idx = np.unique(t_full, return_index=True)
    t_interp = t_full[unique_idx]
    R_interp = R_full[unique_idx]

    # Ensure t_obs is within the simulation range
    t_max_sim = t_interp.max()
    if t_obs > t_max_sim:
        logger.warning("t_obs=%.2f Myr > simulation max=%.2f Myr; "
                        "clamping to simulation max", t_obs, t_max_sim)
        t_obs = t_max_sim

    rng = np.random.default_rng(seed)
    t_form = rng.uniform(0.0, t_obs, size=N_bubble)
    ages = t_obs - t_form  # age of each bubble at observation time

    # Interpolate R(age) from TRINITY trajectory
    # Clamp ages to simulation range
    ages_clamped = np.clip(ages, t_interp.min(), t_interp.max())
    R_synth = np.interp(ages_clamped, t_interp, R_interp)

    # Remove non-positive R
    R_synth = R_synth[R_synth > 0]

    if len(R_synth) < MIN_PTS:
        logger.warning("Too few valid synthetic bubbles — skipping synthesis")
        return None

    # Build histogram in log-space
    log_R = np.log10(R_synth)
    n_bins = 25
    log_edges = np.linspace(log_R.min(), log_R.max(), n_bins + 1)
    counts, _ = np.histogram(log_R, bins=log_edges)

    # Convert to dN/dR: counts / dR, where dR = 10^edge[i+1] - 10^edge[i]
    R_edges = 10.0 ** log_edges
    dR = np.diff(R_edges)
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])

    dNdR = counts / dR
    # Normalize to max for comparison
    dNdR_max = dNdR.max() if dNdR.max() > 0 else 1.0
    dNdR_norm = dNdR / dNdR_max

    # Fit power law to non-zero bins
    valid = dNdR > 0
    synth_slope = np.nan
    synth_R2 = np.nan
    if valid.sum() >= 5:
        logR_c = np.log10(R_centres[valid])
        log_dN = np.log10(dNdR[valid])
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

    return {
        "R_synth": R_synth,
        "R_centres": R_centres,
        "dNdR": dNdR,
        "dNdR_norm": dNdR_norm,
        "synth_slope": synth_slope,
        "synth_R2": synth_R2,
        "N_bubble": N_bubble,
        "t_obs": t_obs,
        "run_used": best["folder"],
    }


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


def plot_ndot_inferred(records: List[Dict], output_dir: Path,
                       fmt: str = "pdf") -> None:
    """
    Part 1: Inferred N_dot(R)/N_dot(R_0) vs R.

    If N_dot is constant, lines should be horizontal.  Deviations show
    where the constant-formation-rate assumption breaks down.
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

    # Horizontal reference (constant N_dot)
    ax.axhline(1.0, color="grey", ls=":", lw=1.0, alpha=0.6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\dot{N}(R)\;/\;\dot{N}(R_0)$")
    ax.set_title(r"Inferred $\dot{N}(R)$ assuming $\mathrm{d}N/\mathrm{d}R \propto R^{-2.2}$"
                 " (Watkins+2023)")

    if len(records) <= 8:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_ndot_inferred.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_dNdR_predicted(records: List[Dict], output_dir: Path,
                        fmt: str = "pdf") -> None:
    """
    Part 2: Predicted dN/dR proportional to 1/v(R) vs R for each model.

    Overplots the observed R^{-2.2} slope for comparison.
    """
    records = _assign_styles(records)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    R_range = [np.inf, -np.inf]
    for rec in records:
        R = rec["R_exp"]
        dNdR = rec["dNdR_norm"]
        if len(R) < 2:
            continue
        ax.plot(R, dNdR, color=rec["_color"], ls=rec["_ls"],
                lw=1.5, alpha=0.85,
                label=(f"{_model_label(rec)}"
                       f" ($\\gamma={rec['pred_slope']:.2f}$)"))
        R_range[0] = min(R_range[0], R.min())
        R_range[1] = max(R_range[1], R.max())

    # Observed R^{-2.2} reference slope (arbitrary normalization)
    if R_range[0] < R_range[1]:
        R_ref = np.logspace(np.log10(R_range[0]), np.log10(R_range[1]), 100)
        # Normalize to match predicted curves at geometric mean
        R_mid = np.sqrt(R_range[0] * R_range[1])
        dNdR_ref = (R_ref / R_mid) ** OBSERVED_ALPHA
        # Scale to roughly match the median of predicted curves at R_mid
        ax.plot(R_ref, dNdR_ref, color="grey", ls="--", lw=2.0,
                alpha=0.7, label=r"$R^{-2.2}$ (Watkins+2023)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}R$ (normalized)")
    ax.set_title(r"Predicted $\mathrm{d}N/\mathrm{d}R \propto 1/v(R)$"
                 " (constant $\\dot{N}$)")

    ax.legend(fontsize=7, loc="best", framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_dNdR_predicted.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_slope_comparison(records: List[Dict], output_dir: Path,
                          fmt: str = "pdf") -> None:
    """
    Summary figure: predicted dN/dR slope vs model parameters.

    Horizontal dashed line at -2.2 (observed).
    """
    slopes = [rec["pred_slope"] for rec in records if np.isfinite(rec["pred_slope"])]
    labels = [rec["folder"] for rec in records if np.isfinite(rec["pred_slope"])]
    nCores = [rec["nCore"] for rec in records if np.isfinite(rec["pred_slope"])]

    if not slopes:
        logger.warning("No valid slopes for comparison plot — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(slopes))
    ax.scatter(x, slopes, color=C_BLUE, s=50, zorder=3, edgecolors="white", lw=0.5)
    ax.axhline(OBSERVED_ALPHA, color=C_VERMILLION, ls="--", lw=1.5,
               label=f"Observed = {OBSERVED_ALPHA}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"Predicted $\mathrm{d}N/\mathrm{d}R$ slope $\gamma$")
    ax.set_title("Predicted vs. observed size distribution slope")
    ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_dir / f"bubble_slope_comparison.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_synthesis(synth: Dict, output_dir: Path,
                   fmt: str = "pdf") -> None:
    """
    Part 3: Forward population synthesis histogram.

    Two panels: (a) histogram of R values, (b) dN/dR vs R with power-law fit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    R_synth = synth["R_synth"]
    R_centres = synth["R_centres"]
    dNdR = synth["dNdR"]
    dNdR_norm = synth["dNdR_norm"]

    # Panel (a): histogram
    ax = axes[0]
    ax.hist(R_synth, bins=30, color=C_SKY, edgecolor=C_BLUE,
            alpha=0.7, density=False)
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$N$")
    ax.set_title(f"Synthetic bubble radii ($N={synth['N_bubble']}$, "
                 f"$t_{{\\rm obs}}={synth['t_obs']:.1f}$ Myr)")

    # Panel (b): dN/dR vs R
    ax = axes[1]
    valid = dNdR > 0
    ax.scatter(R_centres[valid], dNdR[valid], color=C_BLUE, s=25,
               zorder=3, label="Synthesis")

    # Plot power-law fit
    slope = synth["synth_slope"]
    if np.isfinite(slope):
        R_fit = np.logspace(np.log10(R_centres[valid].min()),
                            np.log10(R_centres[valid].max()), 50)
        # Use the fit to compute reference line
        logR_c = np.log10(R_centres[valid])
        log_dN = np.log10(dNdR[valid])
        X = np.column_stack([np.ones_like(logR_c), logR_c])
        beta = np.linalg.lstsq(X, log_dN, rcond=None)[0]
        dNdR_fit = 10.0 ** (beta[0] + beta[1] * np.log10(R_fit))
        ax.plot(R_fit, dNdR_fit, color=C_VERMILLION, ls="--", lw=1.5,
                label=f"Fit: $\\gamma={slope:.2f}$")

    # Observed reference
    if valid.sum() >= 2:
        R_ref = np.logspace(np.log10(R_centres[valid].min()),
                            np.log10(R_centres[valid].max()), 50)
        # Normalize to match data at midpoint
        R_mid = np.sqrt(R_centres[valid].min() * R_centres[valid].max())
        dN_mid = np.interp(np.log10(R_mid), np.log10(R_centres[valid]),
                           np.log10(dNdR[valid]))
        dNdR_obs = 10.0 ** (dN_mid + OBSERVED_ALPHA * (np.log10(R_ref) - np.log10(R_mid)))
        ax.plot(R_ref, dNdR_obs, color=C_GREEN, ls="-.", lw=1.5,
                label=r"$R^{-2.2}$ (Watkins+2023)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}R$")
    ax.set_title("Synthetic size distribution")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    path = output_dir / f"bubble_synthesis.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    """Write per-run summary CSV."""
    csv_path = output_dir / "bubble_distribution_results.csv"
    header = ["folder", "mCloud", "sfe", "nCore", "outcome",
              "R_min_pc", "R_max_pc", "pred_slope", "pred_R2"]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for rec in records:
            writer.writerow([
                rec["folder"],
                f"{rec['mCloud']:.0e}",
                f"{rec['sfe']:.3f}",
                f"{rec['nCore']:.0e}",
                rec["outcome"],
                f"{rec['R_exp'].min():.3f}" if len(rec["R_exp"]) > 0 else "N/A",
                f"{rec['R_exp'].max():.3f}" if len(rec["R_exp"]) > 0 else "N/A",
                f"{rec['pred_slope']:.4f}" if np.isfinite(rec["pred_slope"]) else "N/A",
                f"{rec['pred_R2']:.4f}" if np.isfinite(rec["pred_R2"]) else "N/A",
            ])

    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(records: List[Dict], synth: Optional[Dict] = None) -> None:
    """Print summary table to stdout."""
    print()
    print("=" * 80)
    print("BUBBLE SIZE DISTRIBUTION SUMMARY")
    print("=" * 80)

    slopes = [rec["pred_slope"] for rec in records
              if np.isfinite(rec["pred_slope"])]
    if slopes:
        arr = np.array(slopes)
        print(f"\n  Predicted dN/dR slope (constant N_dot):")
        print(f"    Mean  = {arr.mean():.3f}")
        print(f"    Std   = {arr.std():.3f}")
        print(f"    Range = [{arr.min():.3f}, {arr.max():.3f}]")
        print(f"    Observed (Watkins+2023) = {OBSERVED_ALPHA}")
        print(f"    N_runs = {len(slopes)}")

    # Per-run table
    print()
    print("-" * 80)
    print(f"  {'Folder':<35s} {'Outcome':<10s} {'Slope':>8s} {'R2':>8s} "
          f"{'R_min':>8s} {'R_max':>8s}")
    print("-" * 80)
    for rec in records:
        sl = f"{rec['pred_slope']:.3f}" if np.isfinite(rec["pred_slope"]) else "N/A"
        r2 = f"{rec['pred_R2']:.3f}" if np.isfinite(rec["pred_R2"]) else "N/A"
        rmin = f"{rec['R_exp'].min():.2f}" if len(rec["R_exp"]) > 0 else "N/A"
        rmax = f"{rec['R_exp'].max():.2f}" if len(rec["R_exp"]) > 0 else "N/A"
        print(f"  {rec['folder']:<35s} {rec['outcome']:<10s} {sl:>8s} {r2:>8s} "
              f"{rmin:>8s} {rmax:>8s}")

    # N_dot constancy check
    print()
    print("-" * 80)
    print("  N_dot constancy check (ratio of N_dot at R_max vs R_min):")
    for rec in records:
        ndot = rec["ndot_norm"]
        if len(ndot) < 2 or not np.isfinite(ndot).any():
            continue
        finite = ndot[np.isfinite(ndot)]
        if len(finite) < 2:
            continue
        ratio = finite[-1] / finite[0] if finite[0] != 0 else np.nan
        if np.isfinite(ratio):
            print(f"    {rec['folder']:<35s}  "
                  f"N_dot(R_max)/N_dot(R_min) = {ratio:.3f}")

    if synth is not None:
        print()
        print("-" * 80)
        print(f"  Population synthesis (run: {synth['run_used']}):")
        print(f"    N_bubble = {synth['N_bubble']}")
        print(f"    t_obs    = {synth['t_obs']:.1f} Myr")
        print(f"    Fitted slope = {synth['synth_slope']:.3f}"
              if np.isfinite(synth["synth_slope"]) else "    Fitted slope = N/A")
        print(f"    R2       = {synth['synth_R2']:.3f}"
              if np.isfinite(synth["synth_R2"]) else "    R2       = N/A")

    print()
    print("=" * 80)


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bubble size distribution analysis from TRINITY outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bubble_distribution.py -F /path/to/sweep_output
  python bubble_distribution.py -F /path/to/sweep_output --N-bubble 5000 --fmt png
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--N-bubble", type=int, default=2000,
        help="Number of synthetic bubbles for population synthesis (default: 2000).",
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
        help="Skip Part 3 (population synthesis).",
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
    records = collect_data(folder_path)
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Step 2: figures
    plot_ndot_inferred(records, output_dir, args.fmt)
    plot_dNdR_predicted(records, output_dir, args.fmt)
    plot_slope_comparison(records, output_dir, args.fmt)

    # Step 3: population synthesis (optional)
    synth = None
    if not args.no_synthesis:
        logger.info("Running population synthesis (N=%d, t_obs=%.1f Myr)...",
                     args.N_bubble, args.t_obs)
        synth = run_population_synthesis(
            records,
            N_bubble=args.N_bubble,
            t_obs=args.t_obs,
            seed=args.seed,
        )
        if synth is not None:
            plot_synthesis(synth, output_dir, args.fmt)
        else:
            logger.warning("Population synthesis returned no results.")

    # Step 4: output
    write_results_csv(records, output_dir)
    print_summary(records, synth)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
