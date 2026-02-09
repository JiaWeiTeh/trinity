#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity–radius relation and self-similar expansion index from TRINITY.

Physics background
------------------
The relationship v(R) between the shell expansion velocity and its
radius encodes the underlying driving mechanism.  For self-similar
solutions where R ∝ t^n, the velocity scales as v ∝ R^α with
α = (1 − n)/n, and the dimensionless ratio η = v t / R equals n.

Classical predictions for α (i.e. d log v / d log R):

========================  ======  ======  ============================
Solution                  n       α       Reference
========================  ======  ======  ============================
Weaver energy-driven      3/5     −2/3    Weaver et al. (1977)
Spitzer HII expansion     4/7     −3/2    Spitzer (1978)
Momentum-driven           1/2     −1      snowplough limit
Radiation-pressure-driven 1/2     −1/2    Krumholz & Matzner (2009)
========================  ======  ======  ============================

In practice, TRINITY shells do not follow a single power law because:

* The driving mechanism changes between phases (energy → transition →
  momentum), each with its own effective α.
* Mass loading (sweeping up ambient material) steepens deceleration.
* Gravity can flatten or reverse the velocity trend, especially at
  high Σ.
* Supernovae at t ≈ 3–4 Myr re-accelerate the shell.

The **self-similar constant** η = v t / R is particularly useful
observationally.  An observer measuring v and R of an HII region can
infer its dynamical age as t_dyn = η R / v.  The "correct" η depends
on the expansion phase; using the wrong value introduces a systematic
age bias.  This script quantifies η(phase) and η(R) across the
parameter grid, providing look-up tables for observational studies.

Computed quantities
-------------------
* **α_local(R)** — instantaneous power-law index via centred finite
  differences in log-space, smoothed with Savitzky–Golay filtering.
* **α_phase** — least-squares fit of log v = const + α log R within
  each phase (energy, transition, momentum).
* **η(t)** — time-resolved self-similar constant.
* **η at fixed R** — interpolated η at observer-relevant radii
  (10, 50, 100 pc).
* **η at fixed t** — interpolated η at 1, 3, 5, 10 Myr.

References
----------
* Weaver, R. et al. (1977), ApJ, 218, 377 — wind-bubble expansion.
* Spitzer, L. (1978), *Physical Processes in the ISM* — HII expansion.
* Krumholz, M. R. & Matzner, C. D. (2009), ApJ, 703, 1352 — radiation.

CLI usage
---------
    python velocity_radius.py -F /path/to/sweep_output
    python velocity_radius.py -F /path/to/sweep_output --R-bins 30 --fmt png
"""

import sys
import logging
import argparse
import csv
import json
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
from src._plots.plot_markers import find_phase_transitions
from src._functions.unit_conversions import INV_CONV

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root, matching paper_* scripts
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Constants
# ======================================================================

V_AU2KMS = INV_CONV.v_au2kms   # pc/Myr → km/s  (~0.978)

# Theoretical reference slopes  (v ∝ R^alpha)
THEORY_SLOPES = {
    "Weaver (energy)":   -2 / 3,    # R ∝ t^{3/5}
    "Momentum-driven":   -1.0,       # R ∝ t^{1/2}
    "Spitzer HII":       -3 / 2,    # R ∝ t^{4/7}
    "Rad. pressure":     -1 / 2,    # Krumholz & Matzner 09
}

# Theoretical reference eta  (eta = v*t/R = n  for R ∝ t^n)
THEORY_ETA = {
    "Energy-driven":     3 / 5,
    "Momentum-driven":   1 / 2,
    "SN snowplow":       1 / 4,
}

# Observable radii at which to evaluate eta
ETA_RADII = [10.0, 50.0, 100.0]       # pc
ETA_TIMES = [1.0, 3.0, 5.0, 10.0]     # Myr

# Phase name mapping (for grouping energy+implicit together)
ENERGY_PHASES = {"energy", "implicit"}
PHASE_GROUP = {
    "energy":     "energy",
    "implicit":   "energy",
    "transition": "transition",
    "momentum":   "momentum",
}

# Minimum points in a phase to attempt a fit
MIN_PHASE_PTS = 5

# Exclude early transient before velocity stabilises [Myr]
T_MIN_STABLE = 3e-3


# ======================================================================
# Data extraction
# ======================================================================

def extract_run(data_path: Path) -> Optional[Dict]:
    """
    Load one TRINITY run and extract v(R) trajectory + phase info.

    Returns
    -------
    dict or None
        Keys: t, R, v_kms, phase, rCloud, outcome, alpha_local,
        eta, phase_fits, eta_at_R, eta_at_t.
    """
    try:
        output = load_output(data_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", data_path, e)
        return None

    if len(output) < MIN_PHASE_PTS:
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

    rCloud = output[0].get("rCloud", None)

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

    v_kms = v_au * V_AU2KMS

    # Separate expanding and contracting portions;
    # exclude early transient before velocity has stabilised
    stable = t >= T_MIN_STABLE
    expanding = (v_au > 0) & (R > 0) & stable
    contracting = (v_au < 0) & (R > 0) & stable

    if expanding.sum() < MIN_PHASE_PTS and contracting.sum() < MIN_PHASE_PTS:
        logger.debug("Too few valid points in %s — skipping",
                      data_path.parent.name)
        return None

    return {
        "t": t,
        "R": R,
        "v_au": v_au,
        "v_kms": v_kms,
        "phase": phase,
        "rCloud": rCloud if rCloud and rCloud > 0 else None,
        "outcome": outcome,
        "expanding": expanding,
        "contracting": contracting,
    }


def compute_alpha_local(R: np.ndarray, v_au: np.ndarray,
                        expanding: np.ndarray) -> np.ndarray:
    """
    Instantaneous power-law index alpha = d(log v)/d(log R).

    Uses centred finite differences on log-space, smoothed with
    Savitzky-Golay if enough points exist.
    """
    alpha = np.full_like(R, np.nan)
    sel = expanding & (R > 0) & (v_au > 0)
    idx = np.where(sel)[0]

    if len(idx) < 3:
        return alpha

    logR = np.log10(R[idx])
    logv = np.log10(v_au[idx])

    # Centred finite differences
    dlogv = np.gradient(logv, logR)

    # Smooth with Savitzky-Golay if enough points
    n = len(dlogv)
    if n >= 11:
        win = min(n // 3 * 2 + 1, 51)  # odd window, at most 51
        if win >= 5:
            dlogv = savgol_filter(dlogv, win, min(3, win - 1))
    elif n >= 5:
        dlogv = savgol_filter(dlogv, 5, 2)

    alpha[idx] = dlogv
    return alpha


def compute_eta(t: np.ndarray, R: np.ndarray,
                v_au: np.ndarray) -> np.ndarray:
    """Self-similar constant eta(t) = v*t / R."""
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = v_au * t / R
    eta[~np.isfinite(eta)] = np.nan
    return eta


def fit_phase_power_law(
    R: np.ndarray, v_au: np.ndarray,
    phase: np.ndarray, expanding: np.ndarray,
) -> Dict[str, Dict]:
    """
    For each phase group, fit log(v) = log(B) + alpha * log(R).

    Returns dict  phase_name → {alpha, alpha_unc, B, n_pts}.
    """
    results: Dict[str, Dict] = {}

    for grp_name in ["energy", "transition", "momentum"]:
        if grp_name == "energy":
            mask = np.isin(phase, list(ENERGY_PHASES))
        else:
            mask = phase == grp_name
        mask = mask & expanding & (R > 0) & (v_au > 0)

        idx = np.where(mask)[0]
        if len(idx) < MIN_PHASE_PTS:
            continue

        logR = np.log10(R[idx])
        logv = np.log10(v_au[idx])

        # OLS: logv = a + alpha * logR
        X = np.column_stack([np.ones(len(idx)), logR])
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue
        beta = XtX_inv @ (X.T @ logv)

        resid = logv - X @ beta
        ss_res = float(np.sum(resid ** 2))
        s2 = ss_res / max(len(idx) - 2, 1)
        unc = np.sqrt(np.diag(s2 * XtX_inv))

        results[grp_name] = {
            "alpha": float(beta[1]),
            "alpha_unc": float(unc[1]),
            "logB": float(beta[0]),
            "n_pts": len(idx),
        }

    return results


def evaluate_eta_at_radii(
    t: np.ndarray, R: np.ndarray, eta: np.ndarray,
    expanding: np.ndarray,
) -> Dict[float, float]:
    """Interpolate eta at specific radii (if shell reaches them)."""
    sel = expanding & np.isfinite(eta) & (R > 0)
    idx = np.where(sel)[0]
    if len(idx) < 2:
        return {}

    R_sel = R[idx]
    eta_sel = eta[idx]

    result = {}
    for R_target in ETA_RADII:
        if R_sel.min() <= R_target <= R_sel.max():
            result[R_target] = float(np.interp(R_target, R_sel, eta_sel))
    return result


def evaluate_eta_at_times(
    t: np.ndarray, eta: np.ndarray,
    expanding: np.ndarray,
) -> Dict[float, float]:
    """Interpolate eta at specific times."""
    sel = expanding & np.isfinite(eta) & (t > 0)
    idx = np.where(sel)[0]
    if len(idx) < 2:
        return {}

    t_sel = t[idx]
    eta_sel = eta[idx]

    result = {}
    for t_target in ETA_TIMES:
        if t_sel.min() <= t_target <= t_sel.max():
            result[t_target] = float(np.interp(t_target, t_sel, eta_sel))
    return result


def phase_averaged_eta(
    t: np.ndarray, eta: np.ndarray,
    phase: np.ndarray, expanding: np.ndarray,
) -> Dict[str, float]:
    """Mean eta within each phase."""
    result = {}
    for grp_name in ["energy", "transition", "momentum"]:
        if grp_name == "energy":
            mask = np.isin(phase, list(ENERGY_PHASES))
        else:
            mask = phase == grp_name
        mask = mask & expanding & np.isfinite(eta)
        if mask.sum() >= MIN_PHASE_PTS:
            result[grp_name] = float(np.nanmean(eta[mask]))
    return result


# ======================================================================
# Collect all runs
# ======================================================================

def collect_data(folder_path: Path) -> List[Dict]:
    """Walk sweep and analyse each run."""
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

        info = extract_run(data_path)
        if info is None:
            continue

        t = info["t"]
        R = info["R"]
        v_au = info["v_au"]
        v_kms = info["v_kms"]
        phase = info["phase"]
        expanding = info["expanding"]

        # --- Expanding portion ---
        alpha_local = compute_alpha_local(R, v_au, expanding)
        eta = compute_eta(t, R, v_au)
        phase_fits = fit_phase_power_law(R, v_au, phase, expanding)
        eta_at_R = evaluate_eta_at_radii(t, R, eta, expanding)
        eta_at_t = evaluate_eta_at_times(t, eta, expanding)
        eta_phase = phase_averaged_eta(t, eta, phase, expanding)

        # --- Contracting portion (use |v| for log-space analysis) ---
        contracting = info["contracting"]
        alpha_local_contract = compute_alpha_local(
            R, np.abs(v_au), contracting)
        phase_fits_contract = fit_phase_power_law(
            R, np.abs(v_au), phase, contracting)

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "folder": folder_name,
            "outcome": info["outcome"],
            "rCloud": info["rCloud"],
            # Time series (for plotting)
            "t": t,
            "R": R,
            "v_au": v_au,
            "v_kms": v_kms,
            "phase": phase,
            "expanding": expanding,
            "contracting": contracting,
            "alpha_local": alpha_local,
            "alpha_local_contract": alpha_local_contract,
            "eta": eta,
            # Per-phase fits (expanding)
            "phase_fits": phase_fits,
            "phase_fits_contract": phase_fits_contract,
            "eta_phase": eta_phase,
            # Observable-point evaluations
            "eta_at_R": eta_at_R,
            "eta_at_t": eta_at_t,
        }
        records.append(rec)

    logger.info("Analysed %d valid run(s)", len(records))
    return records


# ======================================================================
# Fitting alpha / eta dependence on cloud parameters
# ======================================================================

def _ols_sigma_clip(X, y, sigma_clip, max_iter=10):
    """OLS with iterative sigma-clipping."""
    n = len(y)
    mask = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        X_u, y_u = X[mask], y[mask]
        if mask.sum() < X.shape[1]:
            return None
        XtX = X_u.T @ X_u
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return None
        beta = XtX_inv @ (X_u.T @ y_u)
        resid = y - X @ beta
        rms = np.std(resid[mask], ddof=X.shape[1])
        if rms == 0:
            break
        new_mask = np.abs(resid) <= sigma_clip * rms
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    n_used = int(mask.sum())
    y_pred = X @ beta
    ss_res = np.sum((y[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms_dex = np.sqrt(ss_res / max(n_used - X.shape[1], 1))
    s2 = ss_res / max(n_used - X.shape[1], 1)
    unc = np.sqrt(np.diag(s2 * XtX_inv))
    return {
        "beta": beta, "unc": unc, "R2": R2, "rms_dex": rms_dex,
        "n_used": n_used, "n_rejected": n - n_used, "mask": mask,
    }


def fit_alpha_vs_params(
    records: List[Dict],
    phase_name: str,
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
) -> Optional[Dict]:
    """
    Fit alpha_phase = a0 + a1*log(nCore/n0) + a2*log(mCloud/M0) + ...

    Returns None if too few points or if alpha is approximately constant.
    """
    pts = [
        r for r in records
        if phase_name in r["phase_fits"]
        and r["phase_fits"][phase_name]["n_pts"] >= MIN_PHASE_PTS
    ]
    if len(pts) < 3:
        return None

    alpha_arr = np.array([r["phase_fits"][phase_name]["alpha"] for r in pts])
    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])

    cols = [np.ones(len(pts))]
    names = ["intercept"]
    refs = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}
    for pname, arr, ref in [("nCore", nC, nCore_ref),
                            ("mCloud", mC, mCloud_ref),
                            ("sfe", sfe, sfe_ref)]:
        if len(np.unique(arr)) >= 2:
            cols.append(np.log10(arr / ref))
            names.append(pname)

    X = np.column_stack(cols)
    result = _ols_sigma_clip(X, alpha_arr, sigma_clip)
    if result is None:
        return None

    result["param_names"] = names
    result["alpha_values"] = alpha_arr
    result["mean"] = float(np.mean(alpha_arr))
    result["std"] = float(np.std(alpha_arr))
    result["phase"] = phase_name
    result["refs"] = refs
    return result


# ======================================================================
# Velocity scaling relation: v = A · R^a · (n/n0)^b · (M/M0)^c · (ε/ε0)^d
# ======================================================================

def fit_velocity_scaling(
    records: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    population: str = "expanding",
) -> Optional[Dict]:
    """
    Fit a multi-parameter velocity scaling relation in log-space.

    Fits  log₁₀(v) = log₁₀(A) + a·log₁₀(R) + b·log₁₀(n/n₀)
                                + c·log₁₀(M/M₀) + d·log₁₀(ε/ε₀)

    using data pooled from all runs, separately for expanding (dispersing)
    or collapsing clouds.

    Parameters
    ----------
    records : list of dict
        Output of :func:`collect_data`.
    nCore_ref, mCloud_ref, sfe_ref : float
        Reference normalizations.
    sigma_clip : float
        Sigma-clipping threshold for outlier rejection.
    population : str
        ``"expanding"`` for dispersing clouds (v > 0),
        ``"collapsing"`` for collapsing clouds (|v| vs R).

    Returns
    -------
    dict or None
        Fit results with coefficients, uncertainties, R², equations, etc.
    """
    # Pool (R, v, params) data across all runs
    all_logR, all_logv = [], []
    all_nCore, all_mCloud, all_sfe = [], [], []
    all_folders = []

    for rec in records:
        R = rec["R"]
        v_au = rec["v_au"]

        if population == "expanding":
            # Only dispersing clouds
            if rec["outcome"] == "collapse":
                continue
            sel = rec["expanding"] & (R > 0) & (v_au > 0)
        else:
            # Only collapsing clouds
            if rec["outcome"] != "collapse":
                continue
            sel = rec.get("contracting", np.zeros(len(R), dtype=bool))
            sel = sel & (R > 0) & (np.abs(v_au) > 0)

        idx = np.where(sel)[0]
        if len(idx) < MIN_PHASE_PTS:
            continue

        R_sel = R[idx]
        if population == "expanding":
            v_sel = v_au[idx] * V_AU2KMS            # km/s
        else:
            v_sel = np.abs(v_au[idx]) * V_AU2KMS     # |v| in km/s

        # Subsample to avoid dominating with one long run
        # Keep at most 50 evenly-spaced points per run
        if len(idx) > 50:
            step = len(idx) // 50
            R_sel = R_sel[::step]
            v_sel = v_sel[::step]

        n_pts = len(R_sel)
        all_logR.append(np.log10(R_sel))
        all_logv.append(np.log10(v_sel))
        all_nCore.append(np.full(n_pts, rec["nCore"]))
        all_mCloud.append(np.full(n_pts, rec["mCloud"]))
        all_sfe.append(np.full(n_pts, rec["sfe"]))
        all_folders.extend([rec["folder"]] * n_pts)

    if not all_logR:
        logger.warning("No valid %s data for velocity scaling fit", population)
        return None

    logR = np.concatenate(all_logR)
    logv = np.concatenate(all_logv)
    nCore_arr = np.concatenate(all_nCore)
    mCloud_arr = np.concatenate(all_mCloud)
    sfe_arr = np.concatenate(all_sfe)
    n_total = len(logR)

    if n_total < 5:
        logger.warning("Too few points (%d) for %s velocity scaling",
                        n_total, population)
        return None

    # Build design matrix: intercept + log R + log(param/ref) for each varying param
    param_info = [
        ("R", logR, None),   # R column always included (no ref normalization)
        ("nCore", nCore_arr, nCore_ref),
        ("mCloud", mCloud_arr, mCloud_ref),
        ("sfe", sfe_arr, sfe_ref),
    ]

    active_names = ["intercept"]
    cols = [np.ones(n_total)]

    # R always included
    active_names.append("R")
    cols.append(logR)

    excluded = []
    ref_map = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}
    for pname, arr, ref in param_info[1:]:  # skip R
        if len(np.unique(arr)) >= 2:
            active_names.append(pname)
            cols.append(np.log10(arr / ref))
        else:
            excluded.append(pname)

    X = np.column_stack(cols)

    # Iterative sigma-clipping OLS
    result = _ols_sigma_clip(X, logv, sigma_clip)
    if result is None:
        logger.warning("OLS failed for %s velocity scaling", population)
        return None

    beta = result["beta"]
    unc = result["unc"]
    mask = result["mask"]
    n_used = result["n_used"]
    n_rejected = result["n_rejected"]
    R2 = result["R2"]
    rms_dex = result["rms_dex"]

    # Unpack coefficients
    log_A = beta[0]
    A = 10.0 ** log_A

    exponents = {}
    exponent_unc = {}
    for i, name in enumerate(active_names[1:], 1):
        exponents[name] = float(beta[i])
        exponent_unc[name] = float(unc[i])
    for name in excluded:
        exponents[name] = 0.0
        exponent_unc[name] = 0.0

    # Human-readable equation string
    label_map = {
        "R": "R",
        "nCore": "n_c",
        "mCloud": r"M_{\rm cloud}",
        "sfe": r"\varepsilon",
    }
    eq_parts = [f"{A:.3g} km/s"]
    eq_latex_parts = [f"{A:.3g}" + r"\;\mathrm{km\,s^{-1}}"]

    # R exponent
    exp_R = exponents["R"]
    eq_parts.append(f"R^{{{exp_R:+.2f}}}")
    eq_latex_parts.append(rf"R^{{{exp_R:+.2f}}}")

    # Cloud parameter exponents
    for pname in ["nCore", "mCloud", "sfe"]:
        exp = exponents.get(pname, 0.0)
        if exp == 0.0 and pname in excluded:
            continue
        ref = ref_map[pname]
        lab = label_map[pname]
        eq_parts.append(f"({pname}/{ref:.0e})^{{{exp:+.2f}}}")
        eq_latex_parts.append(
            rf"\left(\frac{{{lab}}}{{{ref:.0e}}}\right)^{{{exp:+.2f}}}"
        )

    equation_str = " * ".join(eq_parts)
    equation_latex = r" \cdot ".join(eq_latex_parts)

    y_pred = X @ beta

    logger.info(
        "[%s] v scaling: %s  (R2=%.3f, rms=%.3f dex, n=%d/%d)",
        population, equation_str, R2, rms_dex, n_used, n_total,
    )

    return {
        "population": population,
        "A": A,
        "log_A": log_A,
        "sigma_logA": float(unc[0]),
        "exponents": exponents,
        "exponent_unc": exponent_unc,
        "active_params": active_names[1:],   # exclude intercept
        "excluded_params": excluded,
        "R2": R2,
        "rms_dex": rms_dex,
        "n_used": n_used,
        "n_rejected": n_rejected,
        "n_total": n_total,
        "equation_str": equation_str,
        "equation_latex": equation_latex,
        # Arrays for parity plot
        "logR": logR,
        "logv_actual": logv,
        "logv_predicted": y_pred,
        "nCore": nCore_arr,
        "mCloud": mCloud_arr,
        "sfe": sfe_arr,
        "mask": mask,
        "folders": all_folders,
        "refs": ref_map,
    }


# ======================================================================
# Radius scaling relation: R = A · t^a · (n/n0)^b · (M/M0)^c · (ε/ε0)^d
# ======================================================================

# Theoretical time exponents  (R ∝ t^n)
THEORY_TIME_EXP = {
    "Weaver (energy)":      3 / 5,   # wind bubble
    "Momentum-driven":      1 / 2,   # snowplough
    "Spitzer HII":          4 / 7,   # classical HII
    "Rad. pressure":        1 / 2,   # Krumholz & Matzner 09
}


def fit_radius_scaling(
    records: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    phase_name: str = "energy",
    population: str = "expanding",
    cloud_region: str = "all",
) -> Optional[Dict]:
    """
    Fit a multi-parameter radius scaling relation in log-space.

    Fits  log₁₀(R) = log₁₀(A) + a·log₁₀(t) + b·log₁₀(n/n₀)
                                + c·log₁₀(M/M₀) + d·log₁₀(ε/ε₀)

    restricted to one expansion phase (energy or momentum) and one
    population (dispersing or collapsing).

    Parameters
    ----------
    records : list of dict
        Output of :func:`collect_data`.
    nCore_ref, mCloud_ref, sfe_ref : float
        Reference normalizations.
    sigma_clip : float
        Sigma-clipping threshold for outlier rejection.
    phase_name : str
        ``"energy"`` (includes implicit) or ``"momentum"``.
    population : str
        ``"expanding"`` for dispersing clouds,
        ``"collapsing"`` for collapsing clouds.
    cloud_region : str
        ``"all"`` — use all radii (default).
        ``"within"`` — only R <= rCloud (expanding within the GMC).
        ``"beyond"`` — only R > rCloud (plowing into the ISM).
        Runs without a valid rCloud are skipped for "within"/"beyond".

    Returns
    -------
    dict or None
        Fit results with coefficients, uncertainties, R², equations, etc.
    """
    all_logt, all_logR = [], []
    all_nCore, all_mCloud, all_sfe = [], [], []
    all_folders = []

    for rec in records:
        t = rec["t"]
        R = rec["R"]
        v_au = rec["v_au"]
        phase = rec["phase"]

        # Population filter
        if population == "expanding":
            if rec["outcome"] == "collapse":
                continue
            direction_mask = rec["expanding"]
        else:
            if rec["outcome"] != "collapse":
                continue
            direction_mask = rec.get("contracting",
                                     np.zeros(len(R), dtype=bool))

        # Phase filter
        if phase_name == "energy":
            phase_mask = np.isin(phase, list(ENERGY_PHASES))
        else:
            phase_mask = phase == phase_name

        # Cloud-region filter (for momentum sub-classification)
        if cloud_region == "within":
            rCloud = rec.get("rCloud")
            if rCloud is None or rCloud <= 0:
                continue
            region_mask = R <= rCloud
        elif cloud_region == "beyond":
            rCloud = rec.get("rCloud")
            if rCloud is None or rCloud <= 0:
                continue
            region_mask = R > rCloud
        else:
            region_mask = np.ones(len(R), dtype=bool)

        sel = direction_mask & phase_mask & region_mask & (R > 0) & (t > 0)
        idx = np.where(sel)[0]
        if len(idx) < MIN_PHASE_PTS:
            continue

        t_sel = t[idx]
        R_sel = R[idx]

        # Subsample to at most 50 evenly-spaced points per run
        if len(idx) > 50:
            step = len(idx) // 50
            t_sel = t_sel[::step]
            R_sel = R_sel[::step]

        n_pts = len(t_sel)
        all_logt.append(np.log10(t_sel))
        all_logR.append(np.log10(R_sel))
        all_nCore.append(np.full(n_pts, rec["nCore"]))
        all_mCloud.append(np.full(n_pts, rec["mCloud"]))
        all_sfe.append(np.full(n_pts, rec["sfe"]))
        all_folders.extend([rec["folder"]] * n_pts)

    if not all_logt:
        logger.warning("No valid %s/%s data for R(t) scaling fit",
                        population, phase_name)
        return None

    logt = np.concatenate(all_logt)
    logR = np.concatenate(all_logR)
    nCore_arr = np.concatenate(all_nCore)
    mCloud_arr = np.concatenate(all_mCloud)
    sfe_arr = np.concatenate(all_sfe)
    n_total = len(logt)

    if n_total < 5:
        logger.warning("Too few points (%d) for %s/%s R(t) scaling",
                        n_total, population, phase_name)
        return None

    # Build design matrix: intercept + log t + log(param/ref)
    active_names = ["intercept"]
    cols = [np.ones(n_total)]

    # t always included
    active_names.append("t")
    cols.append(logt)

    excluded = []
    ref_map = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}
    for pname, arr, ref in [("nCore", nCore_arr, nCore_ref),
                            ("mCloud", mCloud_arr, mCloud_ref),
                            ("sfe", sfe_arr, sfe_ref)]:
        if len(np.unique(arr)) >= 2:
            active_names.append(pname)
            cols.append(np.log10(arr / ref))
        else:
            excluded.append(pname)

    X = np.column_stack(cols)

    result = _ols_sigma_clip(X, logR, sigma_clip)
    if result is None:
        logger.warning("OLS failed for %s/%s R(t) scaling",
                        population, phase_name)
        return None

    beta = result["beta"]
    unc = result["unc"]
    mask = result["mask"]
    n_used = result["n_used"]
    n_rejected = result["n_rejected"]
    R2 = result["R2"]
    rms_dex = result["rms_dex"]

    # Unpack coefficients
    log_A = beta[0]
    A = 10.0 ** log_A

    exponents = {}
    exponent_unc = {}
    for i, name in enumerate(active_names[1:], 1):
        exponents[name] = float(beta[i])
        exponent_unc[name] = float(unc[i])
    for name in excluded:
        exponents[name] = 0.0
        exponent_unc[name] = 0.0

    # Human-readable equation string
    label_map = {
        "t": "t",
        "nCore": "n_c",
        "mCloud": r"M_{\rm cloud}",
        "sfe": r"\varepsilon",
    }
    eq_parts = [f"{A:.3g} pc"]
    eq_latex_parts = [f"{A:.3g}" + r"\;\mathrm{pc}"]

    # t exponent
    exp_t = exponents["t"]
    eq_parts.append(f"t^{{{exp_t:+.2f}}}")
    eq_latex_parts.append(rf"t^{{{exp_t:+.2f}}}")

    for pname in ["nCore", "mCloud", "sfe"]:
        exp = exponents.get(pname, 0.0)
        if exp == 0.0 and pname in excluded:
            continue
        ref = ref_map[pname]
        lab = label_map[pname]
        eq_parts.append(f"({pname}/{ref:.0e})^{{{exp:+.2f}}}")
        eq_latex_parts.append(
            rf"\left(\frac{{{lab}}}{{{ref:.0e}}}\right)^{{{exp:+.2f}}}"
        )

    equation_str = " * ".join(eq_parts)
    equation_latex = r" \cdot ".join(eq_latex_parts)

    y_pred = X @ beta

    pop_label = "dispersing" if population == "expanding" else "collapsing"
    region_tag = f"/{cloud_region}" if cloud_region != "all" else ""
    logger.info(
        "[%s/%s%s] R(t) scaling: %s  (R2=%.3f, rms=%.3f dex, n=%d/%d)",
        pop_label, phase_name, region_tag, equation_str,
        R2, rms_dex, n_used, n_total,
    )

    return {
        "population": population,
        "phase": phase_name,
        "cloud_region": cloud_region,
        "A": A,
        "log_A": log_A,
        "sigma_logA": float(unc[0]),
        "exponents": exponents,
        "exponent_unc": exponent_unc,
        "active_params": active_names[1:],
        "excluded_params": excluded,
        "R2": R2,
        "rms_dex": rms_dex,
        "n_used": n_used,
        "n_rejected": n_rejected,
        "n_total": n_total,
        "equation_str": equation_str,
        "equation_latex": equation_latex,
        # Arrays for parity plot
        "logt": logt,
        "logR_actual": logR,
        "logR_predicted": y_pred,
        "nCore": nCore_arr,
        "mCloud": mCloud_arr,
        "sfe": sfe_arr,
        "mask": mask,
        "folders": all_folders,
        "refs": ref_map,
    }


# ======================================================================
# Plotting helpers
# ======================================================================

_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

PHASE_LS = {"energy": "-", "implicit": "-", "transition": "--", "momentum": ":"}


def _phase_segments(R, v_kms, phase, expanding):
    """Yield (R_seg, v_seg, phase_group, ls) for contiguous phase blocks."""
    grp = np.array([PHASE_GROUP.get(str(p), "energy") for p in phase])
    prev = None
    seg_idx = []
    for i in range(len(R)):
        g = grp[i]
        if g != prev:
            if seg_idx:
                yield seg_idx, prev
            seg_idx = [i]
            prev = g
        else:
            seg_idx.append(i)
    if seg_idx:
        yield seg_idx, prev


# ======================================================================
# Plotting
# ======================================================================

def plot_trajectories(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 1: v(R) trajectory gallery."""
    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = max(len(unique_nCore), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             squeeze=False, dpi=150)
    axes = axes.ravel()

    for pi, nc in enumerate(unique_nCore):
        ax = axes[pi]
        subset = [r for r in records if r["nCore"] == nc]

        # Colour by log10(mCloud)
        all_mc = [r["mCloud"] for r in subset]
        if len(set(all_mc)) > 1:
            norm = matplotlib.colors.LogNorm(
                vmin=min(all_mc) * 0.8, vmax=max(all_mc) * 1.2)
        else:
            norm = matplotlib.colors.LogNorm(
                vmin=all_mc[0] * 0.5, vmax=all_mc[0] * 2)
        cmap = plt.cm.viridis

        for rec in subset:
            R = rec["R"]
            v = rec["v_kms"]
            phase = rec["phase"]
            expanding = rec["expanding"]
            contracting = rec.get("contracting",
                                  np.zeros(len(R), dtype=bool))
            color = cmap(norm(rec["mCloud"]))

            # Expanding segments (solid/dashed by phase)
            for idx_seg, grp in _phase_segments(R, v, phase, expanding):
                sel = np.array(idx_seg)
                sel = sel[expanding[sel] & (R[sel] > 0) & (v[sel] > 0)]
                if len(sel) < 2:
                    continue
                ls = PHASE_LS.get(grp, "-")
                ax.plot(R[sel], v[sel], color=color, ls=ls, lw=0.9,
                        alpha=0.7, zorder=3)

            # Contracting: plot |v| vs R with thin dotted lines
            sel_c = contracting & (R > 0) & (np.abs(v) > 0)
            if sel_c.sum() >= 2:
                ax.plot(R[sel_c], np.abs(v[sel_c]), color=color, ls=':',
                        lw=0.7, alpha=0.4, zorder=2)

        # Reference power-law slopes
        R_ref = np.geomspace(0.1, 500, 100)
        v_anchor = 30.0   # km/s at R=10 pc
        for label, alpha in THEORY_SLOPES.items():
            v_ref = v_anchor * (R_ref / 10.0) ** alpha
            ax.plot(R_ref, v_ref, color="grey", ls=":", lw=0.6, alpha=0.5)
            # Label near right end
            idx_lab = 70
            ax.text(R_ref[idx_lab], v_ref[idx_lab] * 1.15, label,
                    fontsize=5.5, color="0.5", ha="center", va="bottom",
                    rotation=np.degrees(np.arctan(alpha * 0.3)))

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02,
                     label=r"$M_{\rm cloud}$ [$M_\odot$]")

        # Legend for expanding vs contracting
        leg_handles = [
            Line2D([0], [0], color='0.4', ls='-', lw=1,
                   label='Expanding'),
            Line2D([0], [0], color='0.4', ls=':', lw=1,
                   label=r'Contracting ($|v|$)'),
        ]
        ax.legend(handles=leg_handles, loc='lower left', fontsize=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("R [pc]")
        if pi == 0:
            ax.set_ylabel("v [km/s]")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)

    for j in range(len(unique_nCore), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"velocity_radius_trajectories.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_alpha_local(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 2: alpha_local(R) diagnostic."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    cmap = plt.cm.viridis
    all_mc = sorted(set(r["mCloud"] for r in records))
    if len(all_mc) > 1:
        norm = matplotlib.colors.LogNorm(vmin=min(all_mc) * 0.8,
                                          vmax=max(all_mc) * 1.2)
    else:
        norm = matplotlib.colors.LogNorm(vmin=all_mc[0] * 0.5,
                                          vmax=all_mc[0] * 2)

    for rec in records:
        R = rec["R"]
        alpha = rec["alpha_local"]
        expanding = rec["expanding"]
        sel = expanding & np.isfinite(alpha) & (R > 0)
        if sel.sum() < 3:
            continue
        color = cmap(norm(rec["mCloud"]))
        ax.plot(R[sel], alpha[sel], color=color, lw=0.7, alpha=0.6, zorder=3)

    # Reference lines
    for label, val in THEORY_SLOPES.items():
        ax.axhline(val, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 100,
                val + 0.02, label, fontsize=6, color="0.5",
                ha="right", va="bottom")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02,
                 label=r"$M_{\rm cloud}$ [$M_\odot$]")

    ax.set_xscale("log")
    ax.set_xlabel("R [pc]")
    ax.set_ylabel(r"$\alpha_{\rm local} = d\log v / d\log R$")
    ax.set_title("Instantaneous v-R power-law index", fontsize=10)
    ax.set_ylim(-3, 1)

    fig.tight_layout()
    out = output_dir / f"velocity_radius_alpha_local.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_eta_evolution(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 3: eta(t) = v*t/R evolution."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    cmap = plt.cm.viridis
    all_mc = sorted(set(r["mCloud"] for r in records))
    if len(all_mc) > 1:
        norm = matplotlib.colors.LogNorm(vmin=min(all_mc) * 0.8,
                                          vmax=max(all_mc) * 1.2)
    else:
        norm = matplotlib.colors.LogNorm(vmin=all_mc[0] * 0.5,
                                          vmax=all_mc[0] * 2)

    for rec in records:
        t = rec["t"]
        eta = rec["eta"]
        expanding = rec["expanding"]
        sel = expanding & np.isfinite(eta) & (t > 0)
        if sel.sum() < 3:
            continue
        color = cmap(norm(rec["mCloud"]))
        ax.plot(t[sel], eta[sel], color=color, lw=0.7, alpha=0.6, zorder=3)

    for label, val in THEORY_ETA.items():
        ax.axhline(val, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0.1 else 10,
                val + 0.01, label, fontsize=6, color="0.5",
                ha="right", va="bottom")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02,
                 label=r"$M_{\rm cloud}$ [$M_\odot$]")

    ax.set_xlabel("t [Myr]")
    ax.set_ylabel(r"$\eta = v\,t\,/\,R$")
    ax.set_title("Self-similar expansion index", fontsize=10)
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    out = output_dir / f"velocity_radius_eta_evolution.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_alpha_phase(
    records: List[Dict],
    alpha_fits: Dict[str, Optional[Dict]],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 4: Phase-averaged alpha distribution / parameter dependence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)

    for i, phase_name in enumerate(["energy", "transition", "momentum"]):
        ax = axes[i]
        vals = [
            r["phase_fits"][phase_name]["alpha"]
            for r in records if phase_name in r["phase_fits"]
        ]
        if not vals:
            ax.set_visible(False)
            continue

        vals = np.array(vals)
        ax.hist(vals, bins=max(5, len(vals) // 3), color="C0", alpha=0.7,
                edgecolor="k", linewidth=0.4)

        mean = np.mean(vals)
        std = np.std(vals)
        ax.axvline(mean, color="C3", lw=1.5, label=rf"mean = {mean:.3f}")
        ax.axvspan(mean - std, mean + std, color="C3", alpha=0.1)

        # Theoretical reference
        if phase_name == "energy":
            ax.axvline(-2 / 3, color="grey", ls="--", lw=1,
                        label=r"Weaver ($-2/3$)")
        elif phase_name == "momentum":
            ax.axvline(-1, color="grey", ls="--", lw=1,
                        label=r"Momentum ($-1$)")

        ax.set_xlabel(rf"$\alpha_{{\rm {phase_name}}}$")
        ax.set_ylabel("Count")
        ax.set_title(f"{phase_name} phase", fontsize=10)
        ax.legend(fontsize=7, loc="best")

        # Annotate fit info if available
        fit = alpha_fits.get(phase_name)
        if fit is not None:
            ax.text(
                0.96, 0.96,
                rf"$R^2 = {fit['R2']:.3f}$, "
                + rf"$\sigma = {fit['std']:.3f}$",
                transform=ax.transAxes, va="top", ha="right", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                          boxstyle="round,pad=0.3"),
            )

    fig.tight_layout()
    out = output_dir / f"velocity_radius_alpha_phase.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_eta_at_radii(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 5: eta at observable radii (box plots)."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    # Collect eta values at each target radius
    data_by_R = {R_t: [] for R_t in ETA_RADII}
    for rec in records:
        for R_t in ETA_RADII:
            if R_t in rec["eta_at_R"]:
                data_by_R[R_t].append(rec["eta_at_R"][R_t])

    positions = []
    box_data = []
    labels = []
    for R_t in ETA_RADII:
        vals = data_by_R[R_t]
        if vals:
            positions.append(R_t)
            box_data.append(vals)
            labels.append(f"{R_t:.0f} pc")

    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=range(len(box_data)),
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="C0", alpha=0.3),
            medianprops=dict(color="C3", lw=1.5),
        )
        ax.set_xticks(range(len(box_data)))
        ax.set_xticklabels(labels)

        # Also scatter individual points
        for i, vals in enumerate(box_data):
            jitter = np.random.default_rng(42).normal(0, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=15, c="C0", alpha=0.5, edgecolors="none", zorder=4,
            )

    # Reference lines
    for label, val in THEORY_ETA.items():
        ax.axhline(val, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(len(box_data) - 0.5 if box_data else 0, val + 0.01,
                label, fontsize=6, color="0.5", ha="right", va="bottom")

    ax.set_xlabel("Observed bubble radius")
    ax.set_ylabel(r"$\eta = v\,t\,/\,R$")
    ax.set_title(r"$\eta$ at observable radii (for observer age estimates)",
                 fontsize=10)
    ax.set_ylim(0, 1.2)

    fig.tight_layout()
    out = output_dir / f"velocity_radius_eta_at_radii.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Figure 6: v ∝ R^β power-law test
# ======================================================================

def plot_vR_powerlaw(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
    nCore_ref: float = 1e3,
    mCloud_ref: float = 1e5,
    sfe_ref: float = 0.01,
) -> Optional[Path]:
    """
    Test whether v ∝ R^β holds as a power law.

    Fits log(v) = A + β·log(R) separately for expanding and
    contracting portions at fiducial parameters.  If no runs
    match the fiducial, all available runs are used.
    """
    logger.info("Figure 6: v ∝ R^β power-law test")

    # Find fiducial runs
    fiducial = [
        r for r in records
        if (abs(np.log10(r["nCore"] / nCore_ref)) < 0.15
            and abs(np.log10(r["mCloud"] / mCloud_ref)) < 0.15
            and abs(r["sfe"] - sfe_ref) / max(sfe_ref, 1e-10) < 0.2)
    ]
    if not fiducial:
        logger.info("No fiducial match — using all %d runs", len(records))
        fiducial = records

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # --- Panel (a): v(R) with power-law fits ---
    ax = axes[0]

    all_logR_exp, all_logv_exp = [], []
    all_logR_con, all_logv_con = [], []

    for rec in fiducial:
        R = rec["R"]
        v = rec["v_kms"]
        expanding = rec["expanding"]
        contracting = rec.get("contracting",
                              np.zeros(len(R), dtype=bool))

        # Expanding data
        sel = expanding & (R > 0) & (v > 0)
        if sel.sum() >= 2:
            ax.plot(R[sel], v[sel], '-', color='#0072B2', lw=1.0,
                    alpha=0.6, zorder=3)
            all_logR_exp.append(np.log10(R[sel]))
            all_logv_exp.append(np.log10(v[sel]))

        # Contracting data
        sel_c = contracting & (R > 0) & (np.abs(v) > 0)
        if sel_c.sum() >= 2:
            ax.plot(R[sel_c], np.abs(v[sel_c]), '--', color='#D55E00',
                    lw=1.0, alpha=0.6, zorder=3)
            all_logR_con.append(np.log10(R[sel_c]))
            all_logv_con.append(np.log10(np.abs(v[sel_c])))

    # Fit expanding
    beta_exp, R2_exp = None, None
    if all_logR_exp:
        logR_cat = np.concatenate(all_logR_exp)
        logv_cat = np.concatenate(all_logv_exp)
        X = np.column_stack([np.ones(len(logR_cat)), logR_cat])
        coeff = np.linalg.lstsq(X, logv_cat, rcond=None)[0]
        A_exp, beta_exp = coeff
        ss_res = np.sum((logv_cat - X @ coeff) ** 2)
        ss_tot = np.sum((logv_cat - logv_cat.mean()) ** 2)
        R2_exp = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        R_fit = np.geomspace(10 ** logR_cat.min(),
                             10 ** logR_cat.max(), 100)
        v_fit = 10 ** A_exp * R_fit ** beta_exp
        ax.plot(R_fit, v_fit, '-', color='#0072B2', lw=2.5, alpha=0.9,
                zorder=5,
                label=rf'Expand: $\beta={beta_exp:.2f}$, '
                      rf'$R^2={R2_exp:.3f}$')

    # Fit contracting
    beta_con, R2_con = None, None
    if all_logR_con:
        logR_cat = np.concatenate(all_logR_con)
        logv_cat = np.concatenate(all_logv_con)
        X = np.column_stack([np.ones(len(logR_cat)), logR_cat])
        coeff = np.linalg.lstsq(X, logv_cat, rcond=None)[0]
        A_con, beta_con = coeff
        ss_res = np.sum((logv_cat - X @ coeff) ** 2)
        ss_tot = np.sum((logv_cat - logv_cat.mean()) ** 2)
        R2_con = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        R_fit = np.geomspace(10 ** logR_cat.min(),
                             10 ** logR_cat.max(), 100)
        v_fit = 10 ** A_con * R_fit ** beta_con
        ax.plot(R_fit, v_fit, '--', color='#D55E00', lw=2.5, alpha=0.9,
                zorder=5,
                label=rf'Contract: $\beta={beta_con:.2f}$, '
                      rf'$R^2={R2_con:.3f}$')

    # Reference slopes
    R_ref = np.geomspace(0.1, 500, 100)
    v_anchor = 30.0
    for label, alpha in THEORY_SLOPES.items():
        v_ref = v_anchor * (R_ref / 10.0) ** alpha
        ax.plot(R_ref, v_ref, color="grey", ls=":", lw=0.6, alpha=0.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R [pc]")
    ax.set_ylabel(r"$v$ [km\,s$^{-1}$]")
    ax.set_title(r"$v \propto R^{\beta}$ power-law test", fontsize=10)
    ax.legend(fontsize=7, loc="best")

    # --- Panel (b): alpha_local(R) for fiducial ---
    ax2 = axes[1]

    for rec in fiducial:
        R = rec["R"]
        expanding = rec["expanding"]
        contracting = rec.get("contracting",
                              np.zeros(len(R), dtype=bool))

        # Expanding alpha_local
        alpha = rec["alpha_local"]
        sel = expanding & np.isfinite(alpha) & (R > 0)
        if sel.sum() >= 3:
            ax2.plot(R[sel], alpha[sel], '-', color='#0072B2', lw=0.8,
                     alpha=0.6, zorder=3)

        # Contracting alpha_local
        alpha_c = rec.get("alpha_local_contract")
        if alpha_c is not None:
            sel_c = contracting & np.isfinite(alpha_c) & (R > 0)
            if sel_c.sum() >= 3:
                ax2.plot(R[sel_c], alpha_c[sel_c], '--', color='#D55E00',
                         lw=0.8, alpha=0.6, zorder=3)

    # Reference slope lines
    for label, val in THEORY_SLOPES.items():
        ax2.axhline(val, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax2.text(0.98, val + 0.04, label, fontsize=6, color="0.5",
                 ha="right", va="bottom", transform=ax2.get_yaxis_transform())

    # Fitted beta markers
    leg2 = []
    if beta_exp is not None:
        ax2.axhline(beta_exp, color='#0072B2', ls='-', lw=1.5, alpha=0.6)
        leg2.append(Line2D([0], [0], color='#0072B2', ls='-', lw=1.5,
                           label=rf'Expand $\beta = {beta_exp:.2f}$'))
    if beta_con is not None:
        ax2.axhline(beta_con, color='#D55E00', ls='--', lw=1.5, alpha=0.6)
        leg2.append(Line2D([0], [0], color='#D55E00', ls='--', lw=1.5,
                           label=rf'Contract $\beta = {beta_con:.2f}$'))
    if leg2:
        ax2.legend(handles=leg2, fontsize=7, loc="best")

    ax2.set_xscale("log")
    ax2.set_xlabel("R [pc]")
    ax2.set_ylabel(r"$\alpha_{\rm local} = d\log v / d\log R$")
    ax2.set_title("Instantaneous power-law index", fontsize=10)
    ax2.set_ylim(-3, 2)

    fig.tight_layout()
    out = output_dir / f"velocity_radius_powerlaw.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Figure 7: Velocity scaling parity plot
# ======================================================================

def plot_velocity_scaling_parity(
    vscale_fits: Dict[str, Optional[Dict]],
    output_dir: Path,
    fmt: str,
) -> Optional[Path]:
    """
    Parity plot (actual vs predicted v) for the multi-parameter
    velocity scaling relation, one panel per population
    (expanding / collapsing).
    """
    fit_list = [(pop, fit) for pop, fit in vscale_fits.items()
                if fit is not None]
    if not fit_list:
        logger.warning("No velocity scaling fits — skipping parity plot")
        return None

    n_panels = len(fit_list)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * n_panels, 5), dpi=150,
                             squeeze=False)
    axes = axes.ravel()

    for pi, (pop, fit) in enumerate(fit_list):
        ax = axes[pi]
        v_act = 10.0 ** fit["logv_actual"]       # km/s
        v_pred = 10.0 ** fit["logv_predicted"]    # km/s
        mask = fit["mask"]
        mCloud = fit["mCloud"]
        nCore = fit["nCore"]

        log_mCloud = np.log10(mCloud)
        vmin, vmax = log_mCloud.min(), log_mCloud.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # Marker shape by unique nCore
        unique_nCore = sorted(set(nCore))
        nCore_to_marker = {
            nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
        }

        # 1:1 line
        all_vals = np.concatenate([v_act, v_pred])
        lo = all_vals[all_vals > 0].min() * 0.5
        hi = all_vals.max() * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

        # Plot each nCore group
        for nc in unique_nCore:
            nc_mask = nCore == nc
            marker = nCore_to_marker[nc]

            # Inliers
            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    v_act[sel], v_pred[sel],
                    c=log_mCloud[sel],
                    cmap="viridis", norm=norm,
                    marker=marker, s=20, alpha=0.7,
                    edgecolors="k", linewidths=0.3,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$" + " cm$^{-3}$",
                )

            # Outliers
            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    v_act[sel_out], v_pred[sel_out],
                    c=log_mCloud[sel_out],
                    cmap="viridis", norm=norm,
                    marker=marker, s=20, alpha=0.2,
                    edgecolors="grey", linewidths=0.5,
                    zorder=3,
                )

        # Colourbar
        sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

        # Annotation
        ann = (
            rf"$R^2 = {fit['R2']:.3f}$"
            + "\n"
            + rf"RMS $= {fit['rms_dex']:.3f}$ dex"
            + "\n"
            + rf"$N = {fit['n_used']}$ (rejected {fit['n_rejected']})"
        )
        ax.text(
            0.04, 0.96, ann,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      alpha=0.85, boxstyle="round,pad=0.3"),
        )

        # Equation
        eq = fit["equation_latex"]
        pop_label = "Dispersing" if pop == "expanding" else "Collapsing"
        ax.set_title(f"{pop_label}: velocity scaling", fontsize=10)
        ax.text(
            0.96, 0.04,
            rf"$v \approx {eq}$",
            transform=ax.transAxes,
            va="bottom", ha="right", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      alpha=0.85, boxstyle="round,pad=0.3"),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$v$ from TRINITY [km\,s$^{-1}$]")
        ax.set_ylabel(r"$v$ from scaling fit [km\,s$^{-1}$]")
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"velocity_scaling_parity.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Figure 8: R(t) scaling parity plot
# ======================================================================

def plot_radius_scaling_parity(
    rscale_fits: Dict[str, Optional[Dict]],
    output_dir: Path,
    fmt: str,
) -> Optional[Path]:
    """
    Parity plot (actual vs predicted R) for the multi-parameter
    R(t) scaling relation.  One panel per (population, phase) combination.
    """
    fit_list = [(key, fit) for key, fit in rscale_fits.items()
                if fit is not None]
    if not fit_list:
        logger.warning("No R(t) scaling fits — skipping parity plot")
        return None

    n_panels = len(fit_list)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows), dpi=150,
                             squeeze=False)
    axes_flat = axes.ravel()

    for pi, (key, fit) in enumerate(fit_list):
        ax = axes_flat[pi]
        R_act = 10.0 ** fit["logR_actual"]       # pc
        R_pred = 10.0 ** fit["logR_predicted"]    # pc
        mask = fit["mask"]
        mCloud = fit["mCloud"]
        nCore = fit["nCore"]

        log_mCloud = np.log10(mCloud)
        vmin, vmax = log_mCloud.min(), log_mCloud.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        norm_c = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        unique_nCore = sorted(set(nCore))
        nCore_to_marker = {
            nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
        }

        # 1:1 line
        all_vals = np.concatenate([R_act, R_pred])
        lo = all_vals[all_vals > 0].min() * 0.5
        hi = all_vals.max() * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

        for nc in unique_nCore:
            nc_mask = nCore == nc
            marker = nCore_to_marker[nc]

            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    R_act[sel], R_pred[sel],
                    c=log_mCloud[sel],
                    cmap="viridis", norm=norm_c,
                    marker=marker, s=20, alpha=0.7,
                    edgecolors="k", linewidths=0.3,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$" + " cm$^{-3}$",
                )

            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    R_act[sel_out], R_pred[sel_out],
                    c=log_mCloud[sel_out],
                    cmap="viridis", norm=norm_c,
                    marker=marker, s=20, alpha=0.2,
                    edgecolors="grey", linewidths=0.5,
                    zorder=3,
                )

        sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm_c)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

        ann = (
            rf"$R^2 = {fit['R2']:.3f}$"
            + "\n"
            + rf"RMS $= {fit['rms_dex']:.3f}$ dex"
            + "\n"
            + rf"$N = {fit['n_used']}$ (rejected {fit['n_rejected']})"
        )
        ax.text(
            0.04, 0.96, ann,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      alpha=0.85, boxstyle="round,pad=0.3"),
        )

        eq = fit["equation_latex"]
        pop = fit["population"]
        phase = fit["phase"]
        region = fit.get("cloud_region", "all")
        pop_label = "Dispersing" if pop == "expanding" else "Collapsing"
        region_suffix = ""
        if region == "within":
            region_suffix = r" ($R \leq R_{\rm cloud}$)"
        elif region == "beyond":
            region_suffix = r" ($R > R_{\rm cloud}$)"
        ax.set_title(f"{pop_label} / {phase}{region_suffix}: R(t) scaling",
                     fontsize=10)
        ax.text(
            0.96, 0.04,
            rf"$R \approx {eq}$",
            transform=ax.transAxes,
            va="bottom", ha="right", fontsize=6,
            bbox=dict(facecolor="white", edgecolor="0.7",
                      alpha=0.85, boxstyle="round,pad=0.3"),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$R$ from TRINITY [pc]")
        ax.set_ylabel(r"$R$ from scaling fit [pc]")
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

    # Hide unused panels
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"radius_scaling_parity.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    csv_path = output_dir / "velocity_radius_results.csv"
    header = [
        "nCore", "mCloud", "SFE", "outcome",
        "alpha_energy", "alpha_energy_unc",
        "alpha_transition", "alpha_transition_unc",
        "alpha_momentum", "alpha_momentum_unc",
        "alpha_energy_contract", "alpha_transition_contract",
        "alpha_momentum_contract",
        "eta_energy", "eta_transition", "eta_momentum",
    ]
    for R_t in ETA_RADII:
        header.append(f"eta_at_R{R_t:.0f}")
    for t_t in ETA_TIMES:
        header.append(f"eta_at_t{t_t:.0f}")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            row = [
                f"{r['nCore']:.4e}", f"{r['mCloud']:.4e}",
                f"{r['sfe']:.4f}", r["outcome"],
            ]
            for ph in ["energy", "transition", "momentum"]:
                pf = r["phase_fits"].get(ph)
                if pf:
                    row.extend([f"{pf['alpha']:.4f}", f"{pf['alpha_unc']:.4f}"])
                else:
                    row.extend(["N/A", "N/A"])
            for ph in ["energy", "transition", "momentum"]:
                pf_c = r.get("phase_fits_contract", {}).get(ph)
                row.append(f"{pf_c['alpha']:.4f}" if pf_c else "N/A")
            for ph in ["energy", "transition", "momentum"]:
                val = r["eta_phase"].get(ph)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            for R_t in ETA_RADII:
                val = r["eta_at_R"].get(R_t)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            for t_t in ETA_TIMES:
                val = r["eta_at_t"].get(t_t)
                row.append(f"{val:.4f}" if val is not None else "N/A")
            writer.writerow(row)

    logger.info("Saved: %s", csv_path)
    return csv_path


def write_fits_csv(
    alpha_fits: Dict[str, Optional[Dict]],
    output_dir: Path,
) -> Path:
    csv_path = output_dir / "velocity_radius_fits.csv"
    header = ["phase", "param", "coefficient", "uncertainty",
              "R2", "RMS", "N_used", "N_rejected", "mean_alpha", "std_alpha"]

    rows = []
    refs = None
    for phase_name, fit in alpha_fits.items():
        if fit is None:
            rows.append([phase_name, "N/A"] + [""] * 8)
            continue
        if refs is None and "refs" in fit:
            refs = fit["refs"]
        for i, name in enumerate(fit["param_names"]):
            rows.append([
                phase_name, name,
                f"{fit['beta'][i]:.6f}", f"{fit['unc'][i]:.6f}",
                f"{fit['R2']:.4f}" if i == 0 else "",
                f"{fit['rms_dex']:.4f}" if i == 0 else "",
                str(fit["n_used"]) if i == 0 else "",
                str(fit["n_rejected"]) if i == 0 else "",
                f"{fit['mean']:.4f}" if i == 0 else "",
                f"{fit['std']:.4f}" if i == 0 else "",
            ])

    with open(csv_path, "w", newline="") as fh:
        if refs:
            fh.write(f"# Normalizations: nCore_ref={refs.get('nCore',0):.0e} cm^-3"
                     f", mCloud_ref={refs.get('mCloud',0):.0e} Msun"
                     f", sfe_ref={refs.get('sfe',0):.0e}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(
    records: List[Dict],
    alpha_fits: Dict[str, Optional[Dict]],
    vscale_fits: Optional[Dict[str, Optional[Dict]]] = None,
    rscale_fits: Optional[Dict[str, Optional[Dict]]] = None,
) -> None:
    print()
    print("=" * 90)
    print("VELOCITY-RADIUS RELATION SUMMARY")
    print("=" * 90)

    for phase_name in ["energy", "transition", "momentum"]:
        vals = [
            r["phase_fits"][phase_name]["alpha"]
            for r in records if phase_name in r["phase_fits"]
        ]
        if not vals:
            continue
        arr = np.array(vals)
        print(f"\n  alpha_{phase_name}: mean = {arr.mean():.3f} +/- {arr.std():.3f}"
              f"  (N = {len(vals)})")

        fit = alpha_fits.get(phase_name)
        if fit is not None and fit["R2"] > 0.1:
            print(f"    R^2 = {fit['R2']:.3f} → varies with parameters")
            for i, name in enumerate(fit["param_names"][1:], 1):
                print(f"    d(alpha)/d(log {name}) = "
                      f"{fit['beta'][i]:+.4f} +/- {fit['unc'][i]:.4f}")
        else:
            print("    Approximately constant across grid")

    # Contracting alpha summary
    has_contract = False
    for phase_name in ["energy", "transition", "momentum"]:
        vals = [
            r["phase_fits_contract"][phase_name]["alpha"]
            for r in records
            if phase_name in r.get("phase_fits_contract", {})
        ]
        if vals:
            if not has_contract:
                print()
                print("-" * 90)
                print("CONTRACTING PORTIONS (|v| vs R)")
                print("-" * 90)
                has_contract = True
            arr = np.array(vals)
            print(f"  alpha_{phase_name} (contract): "
                  f"mean = {arr.mean():.3f} +/- {arr.std():.3f}"
                  f"  (N = {len(vals)})")

    # Outcome breakdown
    outcomes = [r["outcome"] for r in records]
    print()
    print("-" * 90)
    n_exp = outcomes.count("expand")
    n_col = outcomes.count("collapse")
    n_stl = outcomes.count("stalled")
    print(f"Outcomes: {n_exp} expand, {n_col} collapse, {n_stl} stalled "
          f"(total {len(records)})")

    # eta summary
    print()
    print("-" * 90)
    print("Self-similar constant eta = v*t/R")
    print("-" * 90)
    for phase_name in ["energy", "transition", "momentum"]:
        vals = [r["eta_phase"][phase_name]
                for r in records if phase_name in r["eta_phase"]]
        if vals:
            arr = np.array(vals)
            print(f"  eta_{phase_name}: mean = {arr.mean():.3f} +/- {arr.std():.3f}")

    # eta at radii
    print()
    for R_t in ETA_RADII:
        vals = [r["eta_at_R"][R_t] for r in records if R_t in r["eta_at_R"]]
        if vals:
            arr = np.array(vals)
            print(f"  eta at R = {R_t:.0f} pc: {arr.mean():.3f} +/- {arr.std():.3f}"
                  f"  (N = {len(vals)})")

    # Velocity scaling relations
    if vscale_fits:
        print()
        print("-" * 90)
        print("VELOCITY SCALING RELATIONS:  v = A * R^a * (n/n0)^b * (M/M0)^c * (sfe/sfe0)^d")
        print("-" * 90)
        for pop, fit in vscale_fits.items():
            if fit is None:
                continue
            pop_label = "Dispersing" if pop == "expanding" else "Collapsing"
            print(f"\n  [{pop_label}]")
            print(f"    {fit['equation_str']}")
            print(f"    R^2 = {fit['R2']:.3f},  "
                  f"RMS = {fit['rms_dex']:.3f} dex,  "
                  f"N = {fit['n_used']} (rejected {fit['n_rejected']})")
            for pname in ["R", "nCore", "mCloud", "sfe"]:
                exp = fit["exponents"].get(pname)
                unc = fit["exponent_unc"].get(pname)
                if exp is not None:
                    status = ""
                    if pname in fit.get("excluded_params", []):
                        status = "  (constant — excluded)"
                    print(f"    {pname:>8s}: {exp:+.4f} +/- {unc:.4f}{status}")

    # Radius scaling relations
    if rscale_fits:
        print()
        print("-" * 90)
        print("RADIUS SCALING RELATIONS:  R = A * t^a * (n/n0)^b * (M/M0)^c * (sfe/sfe0)^d")
        print("-" * 90)
        for key, fit in rscale_fits.items():
            if fit is None:
                continue
            pop = fit["population"]
            phase = fit["phase"]
            region = fit.get("cloud_region", "all")
            pop_label = "Dispersing" if pop == "expanding" else "Collapsing"
            region_tag = ""
            if region == "within":
                region_tag = " / R <= R_cloud"
            elif region == "beyond":
                region_tag = " / R > R_cloud"
            print(f"\n  [{pop_label} / {phase} phase{region_tag}]")
            print(f"    {fit['equation_str']}")
            print(f"    R^2 = {fit['R2']:.3f},  "
                  f"RMS = {fit['rms_dex']:.3f} dex,  "
                  f"N = {fit['n_used']} (rejected {fit['n_rejected']})")
            # Theoretical comparison for t exponent
            exp_t = fit["exponents"].get("t")
            if exp_t is not None:
                theory_str = ", ".join(
                    f"{name}={val:.3f}"
                    for name, val in THEORY_TIME_EXP.items()
                )
                print(f"    t exponent = {exp_t:+.4f}  "
                      f"(theory: {theory_str})")
            for pname in ["t", "nCore", "mCloud", "sfe"]:
                exp = fit["exponents"].get(pname)
                unc = fit["exponent_unc"].get(pname)
                if exp is not None:
                    status = ""
                    if pname in fit.get("excluded_params", []):
                        status = "  (constant — excluded)"
                    print(f"    {pname:>8s}: {exp:+.4f} +/- {unc:.4f}{status}")

    print()
    print("=" * 90)


# ======================================================================
# Equation JSON (for run_all summary)
# ======================================================================

def _write_equation_json(
    alpha_fits: Dict[str, Optional[Dict]],
    output_dir: Path,
    vscale_fits: Optional[Dict[str, Optional[Dict]]] = None,
    rscale_fits: Optional[Dict[str, Optional[Dict]]] = None,
) -> Path:
    """Write equation data for the run_all summary PDF.

    Note: velocity_radius fits alpha (linear), not log-space.
    We store the mean alpha and the fit R2 for the summary.
    """
    entries = []
    for phase_name, fit in alpha_fits.items():
        if fit is None:
            continue
        refs = fit.get("refs", {})
        names = fit["param_names"]
        exponents = {}
        exponent_unc = {}
        for i, name in enumerate(names[1:], 1):
            exponents[name] = float(fit["beta"][i])
            exponent_unc[name] = float(fit["unc"][i])
        entries.append({
            "script": "velocity_radius",
            "label": f"alpha_{phase_name} (mean={fit['mean']:.3f})",
            "A": float(fit["beta"][0]),
            "exponents": exponents,
            "exponent_unc": exponent_unc,
            "refs": {k: float(v) for k, v in refs.items()},
            "R2": float(fit["R2"]),
            "rms_dex": float(fit["rms_dex"]),
            "n_used": int(fit["n_used"]),
            "linear_fit": True,
        })

    # Velocity scaling fits
    if vscale_fits:
        for pop, fit in vscale_fits.items():
            if fit is None:
                continue
            entries.append({
                "script": "velocity_radius",
                "label": f"v_scaling_{pop}",
                "A": float(fit["A"]),
                "exponents": {k: float(v) for k, v in fit["exponents"].items()},
                "exponent_unc": {k: float(v) for k, v in fit["exponent_unc"].items()},
                "refs": {k: float(v) for k, v in fit["refs"].items()},
                "R2": float(fit["R2"]),
                "rms_dex": float(fit["rms_dex"]),
                "n_used": int(fit["n_used"]),
                "linear_fit": False,
                "equation": fit["equation_str"],
            })

    # Radius scaling fits
    if rscale_fits:
        for key, fit in rscale_fits.items():
            if fit is None:
                continue
            entries.append({
                "script": "velocity_radius",
                "label": f"R_scaling_{fit['population']}_{fit['phase']}"
                         f"_{fit.get('cloud_region', 'all')}",
                "A": float(fit["A"]),
                "exponents": {k: float(v) for k, v in fit["exponents"].items()},
                "exponent_unc": {k: float(v) for k, v in fit["exponent_unc"].items()},
                "refs": {k: float(v) for k, v in fit["refs"].items()},
                "R2": float(fit["R2"]),
                "rms_dex": float(fit["rms_dex"]),
                "n_used": int(fit["n_used"]),
                "linear_fit": False,
                "equation": fit["equation_str"],
            })

    path = output_dir / "velocity_radius_equations.json"
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Velocity-radius relation and self-similar expansion index from TRINITY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python velocity_radius.py -F /path/to/sweep_output
  python velocity_radius.py -F /path/to/sweep_output --R-bins 30 --fmt png
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--nCore-ref", type=float, default=1e3,
        help="Reference nCore normalization [cm^-3] (default: 1e3).",
    )
    parser.add_argument(
        "--mCloud-ref", type=float, default=1e5,
        help="Reference mCloud normalization [Msun] (default: 1e5).",
    )
    parser.add_argument(
        "--sfe-ref", type=float, default=0.01,
        help="Reference SFE normalization (default: 0.01).",
    )
    parser.add_argument(
        "--sigma-clip", type=float, default=3.0,
        help="Sigma-clipping threshold (default: 3.0).",
    )
    parser.add_argument(
        "--R-bins", type=int, default=20,
        help="Number of radius bins for binned analysis (default: 20).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
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

    # Step 2: fit alpha(params) for each phase
    fit_kwargs = dict(
        nCore_ref=args.nCore_ref,
        mCloud_ref=args.mCloud_ref,
        sfe_ref=args.sfe_ref,
        sigma_clip=args.sigma_clip,
    )
    alpha_fits: Dict[str, Optional[Dict]] = {}
    for phase_name in ["energy", "transition", "momentum"]:
        logger.info("--- Fitting alpha_%s vs parameters ---", phase_name)
        alpha_fits[phase_name] = fit_alpha_vs_params(
            records, phase_name, **fit_kwargs)

    # Step 2b: velocity scaling relation  v = A * R^a * n^b * M^c * sfe^d
    vscale_fits: Dict[str, Optional[Dict]] = {}
    for pop in ["expanding", "collapsing"]:
        logger.info("--- Fitting velocity scaling (%s) ---", pop)
        vscale_fits[pop] = fit_velocity_scaling(
            records, population=pop, **fit_kwargs)

    # Step 2c: radius scaling relation  R = A * t^a * n^b * M^c * sfe^d
    #   Energy phase: all radii.
    #   Momentum phase: total, within cloud (R <= rCloud), beyond cloud (R > rCloud).
    rscale_fits: Dict[str, Optional[Dict]] = {}
    for pop in ["expanding", "collapsing"]:
        # Energy phase (single fit, all radii)
        key = f"{pop}_energy"
        logger.info("--- Fitting R(t) scaling (%s / energy) ---", pop)
        rscale_fits[key] = fit_radius_scaling(
            records, phase_name="energy", population=pop, **fit_kwargs)

        # Momentum phase — total, within cloud, beyond cloud
        for region in ["all", "within", "beyond"]:
            suffix = {"all": "", "within": "_within", "beyond": "_beyond"}[region]
            key = f"{pop}_momentum{suffix}"
            logger.info("--- Fitting R(t) scaling (%s / momentum / %s) ---",
                        pop, region)
            rscale_fits[key] = fit_radius_scaling(
                records, phase_name="momentum", population=pop,
                cloud_region=region, **fit_kwargs)

    # Step 3: figures
    plot_trajectories(records, output_dir, args.fmt)
    plot_alpha_local(records, output_dir, args.fmt)
    plot_eta_evolution(records, output_dir, args.fmt)
    plot_alpha_phase(records, alpha_fits, output_dir, args.fmt)
    plot_eta_at_radii(records, output_dir, args.fmt)
    plot_vR_powerlaw(records, output_dir, args.fmt,
                     args.nCore_ref, args.mCloud_ref, args.sfe_ref)
    plot_velocity_scaling_parity(vscale_fits, output_dir, args.fmt)
    plot_radius_scaling_parity(rscale_fits, output_dir, args.fmt)

    # Step 4: output
    write_results_csv(records, output_dir)
    write_fits_csv(alpha_fits, output_dir)
    print_summary(records, alpha_fits, vscale_fits, rscale_fits)

    # Equation JSON for run_all summary
    _write_equation_json(alpha_fits, output_dir, vscale_fits, rscale_fits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
