#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-transition timescale scaling relations for TRINITY.

Physics background
------------------
TRINITY evolves expanding shells through distinct dynamical phases:

1. **Energy-driven phase** — a hot, shocked stellar-wind bubble provides
   the driving pressure.  The bubble interior is approximately adiabatic
   (Weaver et al. 1977) and the shell radius grows as R ∝ t^{3/5}.

2. **Transition phase** — radiative cooling of the bubble becomes important,
   the internal energy drops, and driving shifts from thermal pressure to
   direct momentum deposition and warm-ionised-gas pressure.

3. **Momentum-driven phase** — the bubble has cooled completely; the shell
   coasts under its accumulated momentum with R ∝ t^{1/2} (snowplough).

The *times* at which these transitions occur encode how quickly feedback
couples to the ambient cloud.  They depend on the cloud parameters
(core density n_c, cloud mass M_cloud, star-formation efficiency ε)
through the competition between energy injection, radiative cooling,
and the column density of swept-up material.

Fitted quantities
-----------------
* **t_trans** — onset of the transition phase [Myr].  Marks where
  the bubble's cooling time becomes comparable to the dynamical time.
  Scales roughly as t_trans ∝ n^{-a} M^{b} ε^{g}, reflecting the
  trade-off between luminosity (∝ ε M) and cooling rate (∝ n²).

* **t_trans_dur** — duration of the transition phase [Myr], i.e.
  t_mom − t_trans.  A short transition indicates an abrupt switch to
  momentum-driving; a long one signals a gradual handoff.

* **t_mom** — onset of the momentum phase [Myr].  After this time
  the bubble contributes negligible thermal energy and expansion is
  purely momentum-conserving.

Method
------
For each quantity X ∈ {t_trans, t_trans_dur, t_mom} the script fits:

    log₁₀(X) = log₁₀(A)
                + α log₁₀(n_c / n₀)
                + β log₁₀(M_cloud / M₀)
                + γ log₁₀(ε / ε₀)

using ordinary least-squares (OLS) with iterative sigma-clipping to
reject outliers.  Only non-collapsing (expanding) runs contribute to
the fit.  Parameters that are constant across the sweep are
automatically excluded from the design matrix.

References
----------
* Weaver, R. et al. (1977), ApJ, 218, 377 — adiabatic wind-bubble model.
* Mac Low, M.-M. & McCray, R. (1988), ApJ, 324, 776 — bubble cooling time.
* Rahner, D. et al. (2017), MNRAS, 470, 4453 — WARPFIELD phase evolution.

CLI usage
---------
    python scaling_phases.py -F /path/to/sweep_output
    python scaling_phases.py -F /path/to/sweep_output --quantities t_trans,t_mom
    python scaling_phases.py -F /path/to/sweep_output --sigma-clip 2.5 --fmt png
"""

import sys
import logging
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    parse_simulation_params,
)
from src._plots.plot_markers import find_phase_transitions

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root, matching other paper_* scripts
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Data extraction
# ======================================================================

# Recognised timescale quantities
QUANTITY_DEFS = {
    "t_trans":     "Time at which the transition phase begins [Myr]",
    "t_trans_dur": "Duration of the transition phase [Myr]",
    "t_mom":       "Time at which the momentum phase begins [Myr]",
}


def extract_timescales(data_path: Path, t_end: float = None) -> Optional[Dict[str, float]]:
    """
    Load a single TRINITY run and return its phase-transition timescales.

    Parameters
    ----------
    data_path : Path
        Path to dictionary.jsonl (or .json) for the run.
    t_end : float, optional
        If given, ignore any timescale exceeding this value [Myr].

    Returns
    -------
    dict or None
        Keys are quantity names from QUANTITY_DEFS; values are floats (Myr).
        Returns None if the run cannot be loaded or has no useful phase data.
    """
    try:
        output = load_output(data_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", data_path, e)
        return None

    if len(output) < 2:
        logger.warning("Fewer than 2 snapshots in %s — skipping", data_path)
        return None

    t = output.get("t_now")
    phase = np.array(output.get("current_phase", as_array=False))

    # Check completion: last snapshot should not have EndSimulationDirectly
    # with an error reason (we still accept max-time / dissolved / max-radius).
    last = output[-1]
    is_collapse = last.get("isCollapse", False)
    if is_collapse:
        logger.info("Run %s ended in collapse — skip", data_path.parent.name)
        return None

    # Truncate at t_end if requested
    if t_end is not None:
        mask = t <= t_end
        t = t[mask]
        phase = phase[mask]
        if len(t) < 2:
            logger.info("Fewer than 2 snapshots within t_end=%.3f in %s — skip",
                        t_end, data_path.parent.name)
            return None

    # Use the existing helper to find phase transitions
    trans = find_phase_transitions(t, phase)

    result: Dict[str, float] = {}

    # t_trans: first entry into transition phase
    if trans["t_transition"]:
        t_tr = trans["t_transition"][0]
        result["t_trans"] = t_tr
    else:
        logger.debug("No transition phase in %s", data_path.parent.name)

    # t_mom: first entry into momentum phase
    if trans["t_momentum"]:
        t_mom = trans["t_momentum"][0]
        result["t_mom"] = t_mom

        # t_trans_dur: transition duration = t_mom - t_trans
        if "t_trans" in result:
            result["t_trans_dur"] = t_mom - result["t_trans"]
    else:
        logger.debug("No momentum phase in %s", data_path.parent.name)

    if not result:
        return None

    return result


def collect_data(
    folder_path: Path,
    t_end: float = None,
) -> List[Dict]:
    """
    Walk a sweep output folder and collect (params, timescales) for every run.

    Parameters
    ----------
    folder_path : Path
        Root of the sweep output tree.

    Returns
    -------
    list of dict
        Each dict has keys ``nCore``, ``mCloud``, ``sfe`` (floats) plus
        any timescale keys that were successfully extracted.
    """
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        logger.error("No simulation files found under %s", folder_path)
        return []

    logger.info("Found %d simulation(s) in %s", len(sim_files), folder_path)

    records: List[Dict] = []
    for data_path in sim_files:
        folder_name = data_path.parent.name
        parsed = parse_simulation_params(folder_name)
        if parsed is None:
            logger.warning("Cannot parse folder name '%s' — skipping", folder_name)
            continue

        # Convert string tokens to physical values
        nCore_val = float(parsed["ndens"])       # cm^-3
        mCloud_val = float(parsed["mCloud"])      # Msun
        sfe_val = int(parsed["sfe"]) / 100.0      # fraction

        timescales = extract_timescales(data_path, t_end=t_end)
        if timescales is None:
            continue

        rec = {
            "nCore": nCore_val,
            "mCloud": mCloud_val,
            "sfe": sfe_val,
            "folder": folder_name,
        }
        rec.update(timescales)
        records.append(rec)

    logger.info("Collected %d valid run(s)", len(records))
    return records


# ======================================================================
# Fitting
# ======================================================================

def _ols_sigma_clip(
    X: np.ndarray,
    y: np.ndarray,
    sigma_clip: float,
    max_iter: int = 10,
) -> Optional[Dict]:
    """OLS with iterative sigma-clipping (reusable helper)."""
    n_total = len(y)
    mask = np.ones(n_total, dtype=bool)

    for _ in range(max_iter):
        X_use, y_use = X[mask], y[mask]
        n_use = mask.sum()
        if n_use < X.shape[1]:
            return None
        XtX = X_use.T @ X_use
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return None
        beta = XtX_inv @ (X_use.T @ y_use)

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
        "n_used": n_used, "n_rejected": n_total - n_used,
        "mask": mask, "y_pred": y_pred,
    }


def fit_scaling(
    records: List[Dict],
    quantity: str,
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float = 3.0,
    max_iter: int = 10,
) -> Optional[Dict]:
    """
    Fit a power-law scaling relation for one timescale quantity.

    Parameters
    ----------
    records : list of dict
        Output of :func:`collect_data`.
    quantity : str
        Name of the timescale to fit (e.g. ``"t_trans"``).
    nCore_ref, mCloud_ref, sfe_ref : float
        Reference normalizations (n_0, M_0, eps_0).
    sigma_clip : float
        Number of sigma for iterative outlier rejection.
    max_iter : int
        Maximum sigma-clipping iterations.

    Returns
    -------
    dict or None
        Fit results including coefficients, uncertainties, R^2, etc.
        None if too few data points.
    """
    # Gather valid points
    nCore_arr, mCloud_arr, sfe_arr, tX_arr = [], [], [], []
    folders = []
    for rec in records:
        if quantity not in rec:
            continue
        val = rec[quantity]
        if val is None or not np.isfinite(val) or val <= 0:
            continue
        nCore_arr.append(rec["nCore"])
        mCloud_arr.append(rec["mCloud"])
        sfe_arr.append(rec["sfe"])
        tX_arr.append(val)
        folders.append(rec["folder"])

    n_total = len(tX_arr)
    if n_total < 2:
        logger.warning(
            "Only %d valid point(s) for '%s' — cannot fit", n_total, quantity
        )
        return None

    nCore_arr = np.array(nCore_arr)
    mCloud_arr = np.array(mCloud_arr)
    sfe_arr = np.array(sfe_arr)
    tX_arr = np.array(tX_arr)

    # Identify which parameters vary (>= 2 unique values)
    param_info = [
        ("nCore", nCore_arr, nCore_ref),
        ("mCloud", mCloud_arr, mCloud_ref),
        ("sfe", sfe_arr, sfe_ref),
    ]
    active_names: List[str] = []
    active_cols: List[np.ndarray] = []
    excluded: List[str] = []
    for name, arr, ref in param_info:
        n_unique = len(np.unique(arr))
        if n_unique >= 2:
            active_names.append(name)
            active_cols.append(np.log10(arr / ref))
        else:
            excluded.append(name)

    if not active_names:
        logger.warning(
            "All parameters constant for '%s'; only fitting A (intercept)",
            quantity,
        )

    logger.info(
        "[%s] %d points, active params: %s, excluded (constant): %s",
        quantity, n_total, active_names or ["(none)"], excluded or ["(none)"],
    )

    # Build design matrix: column 0 = intercept, remaining = log10(param/ref)
    log_tX = np.log10(tX_arr)
    X = np.column_stack([np.ones(n_total)] + active_cols)

    # Iterative sigma-clipping OLS
    mask = np.ones(n_total, dtype=bool)
    for iteration in range(max_iter):
        X_use = X[mask]
        y_use = log_tX[mask]
        n_use = mask.sum()

        if n_use < X.shape[1]:
            logger.warning("Too few points (%d) to fit %d coefficients", n_use, X.shape[1])
            return None

        # OLS: beta = (X^T X)^{-1} X^T y
        XtX = X_use.T @ X_use
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix for '%s'", quantity)
            return None
        beta = XtX_inv @ (X_use.T @ y_use)

        residuals_all = log_tX - X @ beta
        residuals_used = residuals_all[mask]
        rms = np.std(residuals_used, ddof=X.shape[1])

        if rms == 0:
            break

        outlier = np.abs(residuals_all) > sigma_clip * rms
        new_mask = ~outlier
        n_rejected_new = n_total - new_mask.sum()
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
        logger.debug(
            "  iter %d: n_used=%d, rms=%.4f dex, n_rejected=%d",
            iteration, mask.sum(), rms, n_rejected_new,
        )

    # Final fit statistics
    n_used = int(mask.sum())
    n_rejected = n_total - n_used
    y_pred = X @ beta
    ss_res = np.sum((log_tX[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((log_tX[mask] - np.mean(log_tX[mask])) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms_dex = np.sqrt(ss_res / max(n_used - X.shape[1], 1))

    # Covariance matrix for uncertainties
    s2 = ss_res / max(n_used - X.shape[1], 1)
    cov = s2 * XtX_inv
    uncertainties = np.sqrt(np.diag(cov))

    # Unpack coefficients
    log_A = beta[0]
    A = 10.0 ** log_A
    sigma_logA = uncertainties[0]

    exponents = {}
    exponent_unc = {}
    idx = 1
    for name in active_names:
        exponents[name] = beta[idx]
        exponent_unc[name] = uncertainties[idx]
        idx += 1
    for name in excluded:
        exponents[name] = 0.0
        exponent_unc[name] = 0.0

    # Human-readable equation string
    eq_parts = [f"{A:.3g} Myr"]
    eq_latex_parts = [f"{A:.3g}" + r"\;\mathrm{Myr}"]
    ref_map = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}
    label_map = {"nCore": "n_c", "mCloud": r"M_{\rm cloud}", "sfe": r"\varepsilon"}
    for pname in ["nCore", "mCloud", "sfe"]:
        exp = exponents[pname]
        if exp == 0.0 and pname in excluded:
            continue
        ref = ref_map[pname]
        eq_parts.append(f"({pname}/{ref:.0e})^{{{exp:+.2f}}}")
        lab = label_map[pname]
        eq_latex_parts.append(
            rf"\left(\frac{{{lab}}}{{{ref:.0e}}}\right)^{{{exp:+.2f}}}"
        )

    equation_str = " * ".join(eq_parts)
    equation_latex = r" \cdot ".join(eq_latex_parts)

    return {
        "quantity": quantity,
        "A": A,
        "log_A": log_A,
        "sigma_logA": sigma_logA,
        "exponents": exponents,           # dict  pname -> float
        "exponent_unc": exponent_unc,     # dict  pname -> float
        "active_params": active_names,
        "excluded_params": excluded,
        "R2": R2,
        "rms_dex": rms_dex,
        "n_used": n_used,
        "n_rejected": n_rejected,
        "n_total": n_total,
        "equation_str": equation_str,
        "equation_latex": equation_latex,
        # Arrays for plotting
        "nCore": nCore_arr,
        "mCloud": mCloud_arr,
        "sfe": sfe_arr,
        "tX_actual": tX_arr,
        "tX_predicted": 10.0 ** y_pred,
        "mask": mask,
        "folders": folders,
        "refs": ref_map,
    }


def fit_scaling_piecewise(
    records: List[Dict],
    quantity: str,
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float = 3.0,
    n_break_candidates: int = 50,
) -> Optional[Dict]:
    """
    Piecewise power-law for a timescale with automatic break in log10(M_cl).

    Scans candidate break values in log10(M_cl) from the 10th to the 90th
    percentile and selects the break that minimises total BIC.
    """
    # Gather valid points
    nC_list, mC_list, sfe_list, tX_list, fld_list = [], [], [], [], []
    for rec in records:
        if quantity not in rec:
            continue
        val = rec[quantity]
        if val is None or not np.isfinite(val) or val <= 0:
            continue
        nC_list.append(rec["nCore"])
        mC_list.append(rec["mCloud"])
        sfe_list.append(rec["sfe"])
        tX_list.append(val)
        fld_list.append(rec.get("folder", ""))

    if len(tX_list) < 10:
        logger.info("Piecewise '%s': too few points (%d < 10) — skipping",
                     quantity, len(tX_list))
        return None

    nC = np.array(nC_list)
    mC = np.array(mC_list)
    sfe = np.array(sfe_list)
    tX = np.array(tX_list)
    folders = np.array(fld_list)
    log_tX = np.log10(tX)
    log_Mcl = np.log10(sfe * mC)

    n_unique_Mcl = len(np.unique(np.round(log_Mcl, decimals=6)))
    logger.info("Piecewise '%s': %d points, %d unique log10(sfe*mCloud) values",
                quantity, len(tX_list), n_unique_Mcl)
    if n_unique_Mcl < 3:
        logger.info("Piecewise '%s': fewer than 3 unique sfe*mCloud values — "
                     "break search will likely fail", quantity)

    refs = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}

    def _build_X(indices):
        """Build design matrix for a subset."""
        cols = [np.ones(len(indices))]
        names = ["intercept"]
        for pname, arr, ref in [("nCore", nC[indices], nCore_ref),
                                ("mCloud", mC[indices], mCloud_ref),
                                ("sfe", sfe[indices], sfe_ref)]:
            if len(np.unique(arr)) >= 2:
                cols.append(np.log10(arr / ref))
                names.append(pname)
        return np.column_stack(cols), names

    def _bic(n, k, rss):
        if n <= k or rss <= 0:
            return np.inf
        return n * np.log(rss / n) + k * np.log(n)

    # Single-fit BIC
    X_all, names_all = _build_X(np.arange(len(tX)))
    fit_single = _ols_sigma_clip(X_all, log_tX, sigma_clip)
    if fit_single is None:
        logger.info("Piecewise '%s': single-fit OLS failed — skipping", quantity)
        return None
    rss_single = np.sum((log_tX[fit_single["mask"]] -
                         fit_single["y_pred"][fit_single["mask"]]) ** 2)
    bic_single = _bic(fit_single["n_used"], X_all.shape[1], rss_single)

    # Scan break candidates
    lo_pct = np.percentile(log_Mcl, 10)
    hi_pct = np.percentile(log_Mcl, 90)
    candidates = np.linspace(lo_pct, hi_pct, n_break_candidates)
    logger.info("Piecewise '%s': break scan range log10(sfe*mCloud) = [%.3f, %.3f]",
                quantity, lo_pct, hi_pct)

    best_bic = np.inf
    best_break = None
    best_fit_lo = None
    best_fit_hi = None
    best_idx_lo = None
    best_idx_hi = None
    best_names_lo = None
    best_names_hi = None
    bic_scan = []

    n_too_few = 0     # candidates rejected for too few points per side
    n_ols_fail = 0    # candidates where OLS failed on one side
    n_valid = 0       # candidates with valid BIC

    for brk in candidates:
        idx_lo = np.where(log_Mcl <= brk)[0]
        idx_hi = np.where(log_Mcl > brk)[0]
        if len(idx_lo) < 5 or len(idx_hi) < 5:
            bic_scan.append((brk, np.inf))
            n_too_few += 1
            continue

        X_lo, names_lo = _build_X(idx_lo)
        X_hi, names_hi = _build_X(idx_hi)
        fl = _ols_sigma_clip(X_lo, log_tX[idx_lo], sigma_clip)
        fh = _ols_sigma_clip(X_hi, log_tX[idx_hi], sigma_clip)
        if fl is None or fh is None:
            bic_scan.append((brk, np.inf))
            n_ols_fail += 1
            continue

        rss_lo = np.sum((log_tX[idx_lo][fl["mask"]] -
                         fl["y_pred"][fl["mask"]]) ** 2)
        rss_hi = np.sum((log_tX[idx_hi][fh["mask"]] -
                         fh["y_pred"][fh["mask"]]) ** 2)
        bic_total = _bic(fl["n_used"], X_lo.shape[1], rss_lo) + \
                    _bic(fh["n_used"], X_hi.shape[1], rss_hi)
        bic_scan.append((brk, bic_total))
        n_valid += 1

        if bic_total < best_bic:
            best_bic = bic_total
            best_break = brk
            best_fit_lo = fl
            best_fit_hi = fh
            best_idx_lo = idx_lo
            best_idx_hi = idx_hi
            best_names_lo = names_lo
            best_names_hi = names_hi

    logger.info("Piecewise '%s': break scan results — %d valid, %d too-few-points, "
                "%d OLS-failed (out of %d candidates)",
                quantity, n_valid, n_too_few, n_ols_fail, len(candidates))

    if best_fit_lo is None or best_fit_hi is None:
        logger.info("Piecewise '%s': no valid break found — skipping piecewise",
                     quantity)
        return None

    # Combine arrays
    n_lo = len(best_idx_lo)
    combined_mask = np.concatenate([best_fit_lo["mask"], best_fit_hi["mask"]])
    combined_actual = np.concatenate([tX[best_idx_lo], tX[best_idx_hi]])
    combined_pred = np.concatenate([
        10.0 ** best_fit_lo["y_pred"],
        10.0 ** best_fit_hi["y_pred"],
    ])
    combined_nCore = np.concatenate([nC[best_idx_lo], nC[best_idx_hi]])
    combined_mCloud = np.concatenate([mC[best_idx_lo], mC[best_idx_hi]])
    combined_sfe = np.concatenate([sfe[best_idx_lo], sfe[best_idx_hi]])
    combined_folders = np.concatenate([folders[best_idx_lo], folders[best_idx_hi]])
    side_labels = np.array(["low"] * n_lo + ["high"] * (len(combined_mask) - n_lo))

    # Add metadata to sub-fits for _write_equation_json compatibility
    for subfit, idx, names in [
        (best_fit_lo, best_idx_lo, best_names_lo),
        (best_fit_hi, best_idx_hi, best_names_hi),
    ]:
        subfit["param_names"] = names
        subfit["nCore"] = nC[idx]
        subfit["mCloud"] = mC[idx]
        subfit["sfe"] = sfe[idx]
        subfit["folders"] = folders[idx]
        subfit["tX_actual"] = tX[idx]
        subfit["tX_predicted"] = 10.0 ** subfit["y_pred"]
        subfit["refs"] = refs
        subfit["quantity"] = quantity
        # Build standard exponents/A for summary compatibility
        A = 10.0 ** subfit["beta"][0]
        subfit["A"] = A
        subfit["log_A"] = subfit["beta"][0]
        exponents = {}
        exponent_unc = {}
        for i, nm in enumerate(names[1:], 1):
            exponents[nm] = float(subfit["beta"][i])
            exponent_unc[nm] = float(subfit["unc"][i])
        # Zero-fill excluded params
        for pname in ["nCore", "mCloud", "sfe"]:
            if pname not in exponents:
                exponents[pname] = 0.0
                exponent_unc[pname] = 0.0
        subfit["exponents"] = exponents
        subfit["exponent_unc"] = exponent_unc

    return {
        "quantity": quantity,
        "break_log_Mcl": float(best_break),
        "fit_low": best_fit_lo,
        "fit_high": best_fit_hi,
        "BIC_piecewise": float(best_bic),
        "BIC_single": float(bic_single),
        "bic_scan": bic_scan,
        "tX_actual": combined_actual,
        "tX_predicted": combined_pred,
        "mask": combined_mask,
        "nCore": combined_nCore,
        "mCloud": combined_mCloud,
        "sfe": combined_sfe,
        "folders": combined_folders,
        "side": side_labels,
        "refs": refs,
    }


# ======================================================================
# Plotting
# ======================================================================

# Marker shapes cycled by nCore
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def plot_parity(fit: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    """
    Create a parity plot (predicted vs actual) for one timescale.

    Parameters
    ----------
    fit : dict
        Result from :func:`fit_scaling`.
    output_dir : Path
        Directory to save the figure.
    fmt : str
        Figure file extension.

    Returns
    -------
    Path
        Path to saved figure.
    """
    quantity = fit["quantity"]
    tX_act = fit["tX_actual"]
    tX_pred = fit["tX_predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    nCore = fit["nCore"]

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)

    # Colour by log10(mCloud)
    log_mCloud = np.log10(mCloud)

    # Marker shape by unique nCore
    unique_nCore = sorted(set(nCore))
    nCore_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    # 1:1 line
    all_vals = np.concatenate([tX_act, tX_pred])
    lo, hi = all_vals[all_vals > 0].min() * 0.5, all_vals.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    # Plot each nCore group
    for nc in unique_nCore:
        nc_mask = nCore == nc
        marker = nCore_to_marker[nc]

        # Inliers
        sel = nc_mask & mask
        if sel.any():
            sc = ax.scatter(
                tX_act[sel], tX_pred[sel],
                c=log_mCloud[sel],
                marker=marker,
                s=50,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label=rf"$n_c = {nc:.0e}$" + " cm$^{-3}$",
            )

        # Outliers (faded, open)
        sel_out = nc_mask & ~mask
        if sel_out.any():
            ax.scatter(
                tX_act[sel_out], tX_pred[sel_out],
                c=log_mCloud[sel_out],
                marker=marker,
                s=50,
                alpha=0.3,
                edgecolors="grey",
                linewidths=0.8,
                zorder=3,
            )

    # Colourbar
    # Use a ScalarMappable so the bar reflects all data
    vmin, vmax = log_mCloud.min(), log_mCloud.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

    # Annotation
    eq = fit["equation_latex"]
    R2 = fit["R2"]
    rms = fit["rms_dex"]
    ann = (
        rf"$R^2 = {R2:.3f}$"
        + "\n"
        + rf"RMS $= {rms:.3f}$ dex"
        + "\n"
        + rf"$N = {fit['n_used']}$ (rejected {fit['n_rejected']})"
    )
    ax.text(
        0.04, 0.96, ann,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    # Equation as title
    desc = QUANTITY_DEFS.get(quantity, quantity)
    ax.set_title(f"{quantity}: {desc}", fontsize=10)

    # Equation annotation in lower-right
    ax.text(
        0.96, 0.04,
        rf"${quantity} \approx {eq}$",
        transform=ax.transAxes,
        va="bottom", ha="right",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{quantity} from TRINITY [Myr]")
    ax.set_ylabel(f"{quantity} from power-law fit [Myr]")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.tight_layout()

    out_path = output_dir / f"scaling_{quantity}.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_parity_diagnostic(fit: Dict, output_dir: Path, fmt: str = "pdf") -> List[Path]:
    """
    Diagnostic parity plots colored by SFE, cluster mass, and residual.

    Produces three figures per quantity to help identify two-population
    bifurcation in the parity plot.
    """
    quantity = fit["quantity"]
    tX_act = fit["tX_actual"]
    tX_pred = fit["tX_predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    nCore = fit["nCore"]
    sfe = fit["sfe"]

    unique_nCore = sorted(set(nCore))
    nCore_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    # 1:1 line bounds
    all_vals = np.concatenate([tX_act, tX_pred])
    lo, hi = all_vals[all_vals > 0].min() * 0.5, all_vals.max() * 2.0

    # Signed residual in dex
    residual = np.log10(tX_act) - np.log10(tX_pred)

    # Three diagnostic colorings:
    #   1. log10(epsilon)
    #   2. log10(M_cl) = log10(epsilon * M_cloud)
    #   3. residual (signed)
    log_sfe = np.log10(sfe)
    log_Mcl = np.log10(sfe * mCloud)

    diag_configs = [
        ("sfe",      log_sfe,   r"$\log_{10}(\varepsilon)$",                        "coolwarm"),
        ("Mcl",      log_Mcl,   r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$",            "viridis"),
        ("residual", residual,  r"$\log_{10}(t_{\rm TRINITY}) - \log_{10}(t_{\rm fit})$", "RdBu"),
    ]

    saved = []
    for var_name, color_arr, cbar_label, cmap_name in diag_configs:
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)

        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

        vmin, vmax = np.nanmin(color_arr), np.nanmax(color_arr)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        # For residual, centre on zero
        if var_name == "residual":
            vlim = max(abs(vmin), abs(vmax), 0.1)
            vmin, vmax = -vlim, vlim
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for nc in unique_nCore:
            nc_mask = nCore == nc
            marker = nCore_to_marker[nc]

            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    tX_act[sel], tX_pred[sel],
                    c=color_arr[sel], cmap=cmap_name, norm=norm,
                    marker=marker, s=50, edgecolors="k", linewidths=0.4,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$" + " cm$^{-3}$",
                )

            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    tX_act[sel_out], tX_pred[sel_out],
                    c=color_arr[sel_out], cmap=cmap_name, norm=norm,
                    marker=marker, s=50, alpha=0.3,
                    edgecolors="grey", linewidths=0.8, zorder=3,
                )

        sm = matplotlib.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{quantity} from TRINITY [Myr]")
        ax.set_ylabel(f"{quantity} from power-law fit [Myr]")
        ax.set_aspect("equal")
        ax.set_title(f"{quantity} — colored by {var_name}", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

        fig.tight_layout()
        out_path = output_dir / f"scaling_{quantity}_diag_by_{var_name}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved diagnostic: %s", out_path)
        saved.append(out_path)

    return saved


def plot_residuals(fit: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    """
    Residual-vs-parameter panel figure (1x3) for one timescale.

    Panels: residual vs log10(M_cloud), log10(epsilon), log10(M_cl).
    """
    quantity = fit["quantity"]
    tX_act = fit["tX_actual"]
    tX_pred = fit["tX_predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    nCore = fit["nCore"]
    sfe = fit["sfe"]

    residual = np.log10(tX_act) - np.log10(tX_pred)

    unique_nCore = sorted(set(nCore))
    nCore_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    log_mCloud = np.log10(mCloud)
    log_sfe = np.log10(sfe)
    log_Mcl = np.log10(sfe * mCloud)

    panels = [
        (log_mCloud, r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$"),
        (log_sfe,    r"$\log_{10}(\varepsilon)$"),
        (log_Mcl,    r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    for ax, (x_arr, xlabel) in zip(axes, panels):
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.6)

        for nc in unique_nCore:
            nc_mask = nCore == nc
            marker = nCore_to_marker[nc]

            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    x_arr[sel], residual[sel],
                    marker=marker, s=40, edgecolors="k", linewidths=0.3,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$" + " cm$^{-3}$",
                )

            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    x_arr[sel_out], residual[sel_out],
                    marker=marker, s=40, alpha=0.3,
                    edgecolors="grey", linewidths=0.6, zorder=3,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\log_{10}(t_{\rm TRINITY}) - \log_{10}(t_{\rm fit})$")
        ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.suptitle(f"{quantity} — fit residuals", fontsize=11)
    fig.tight_layout()

    out_path = output_dir / f"scaling_{quantity}_residuals.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residuals: %s", out_path)
    return out_path


def plot_parity_piecewise(
    pw: Optional[Dict],
    output_dir: Path,
    fmt: str = "pdf",
) -> Optional[Path]:
    """
    Parity plot for a piecewise power-law fit, colored by log10(M_cl).

    Points below the break get thin black edges; points above get thick
    vermillion edges.
    """
    if pw is None:
        return None

    quantity = pw["quantity"]
    actual = pw["tX_actual"]
    predicted = pw["tX_predicted"]
    mask = pw["mask"]
    nCore = pw["nCore"]
    mCloud = pw["mCloud"]
    sfe = pw["sfe"]
    side = pw["side"]
    brk = pw["break_log_Mcl"]
    fit_lo = pw["fit_low"]
    fit_hi = pw["fit_high"]

    log_Mcl = np.log10(sfe * mCloud)

    unique_nCore = sorted(set(nCore))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    vals = np.concatenate([actual, predicted])
    lo, hi = vals[vals > 0].min() * 0.5, vals.max() * 2.0

    vmin, vmax = log_Mcl.min(), log_Mcl.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    for nc in unique_nCore:
        nc_mask = nCore == nc
        marker = nc_to_marker[nc]

        for side_val, edgecolor, lw_edge in [("low", "k", 0.4),
                                              ("high", "#D55E00", 1.2)]:
            sel = nc_mask & mask & (side == side_val)
            if sel.any():
                label_parts = [rf"$n_c = {nc:.0e}$ cm$^{{-3}}$"]
                if side_val == "low":
                    label_parts.append(r"($<$ break)")
                else:
                    label_parts.append(r"($>$ break)")
                ax.scatter(
                    actual[sel], predicted[sel],
                    c=log_Mcl[sel], cmap="viridis", norm=norm,
                    marker=marker, s=50,
                    edgecolors=edgecolor, linewidths=lw_edge,
                    zorder=5,
                    label=" ".join(label_parts),
                )

        sel_out = nc_mask & ~mask
        if sel_out.any():
            ax.scatter(
                actual[sel_out], predicted[sel_out],
                c=log_Mcl[sel_out], cmap="viridis", norm=norm,
                marker=marker, s=50, alpha=0.3,
                edgecolors="grey", linewidths=0.8, zorder=3,
            )

    sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$")

    delta_bic = pw["BIC_single"] - pw["BIC_piecewise"]
    ann = (
        rf"$R^2_{{\rm low}} = {fit_lo['R2']:.3f}$,"
        rf"  $R^2_{{\rm high}} = {fit_hi['R2']:.3f}$" + "\n"
        + rf"Break: $\log_{{10}} M_{{\rm cl}} = {brk:.2f}$" + "\n"
        + rf"$\Delta$BIC $= {delta_bic:+.1f}$ (single $-$ piecewise)"
    )
    ax.text(0.04, 0.96, ann, transform=ax.transAxes, va="top", ha="left",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{quantity} from TRINITY [Myr]")
    ax.set_ylabel(f"{quantity} from piecewise fit [Myr]")
    ax.set_title(f"Piecewise: {quantity}", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / f"scaling_{quantity}_piecewise.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved piecewise parity: %s", out_path)
    return out_path


def plot_bic_scan(
    pw: Optional[Dict],
    output_dir: Path,
    fmt: str = "pdf",
) -> Optional[Path]:
    """BIC scan: total BIC vs break-point location."""
    if pw is None or "bic_scan" not in pw:
        return None

    quantity = pw["quantity"]
    scan = np.array(pw["bic_scan"])
    brk_vals = scan[:, 0]
    bic_vals = scan[:, 1]

    finite = np.isfinite(bic_vals)
    if finite.sum() < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)
    ax.plot(brk_vals[finite], bic_vals[finite], "-", color="#0072B2", lw=1.5)

    opt_brk = pw["break_log_Mcl"]
    opt_bic = pw["BIC_piecewise"]
    ax.axvline(opt_brk, color="#D55E00", ls="--", lw=1.2,
               label=rf"Optimum: $\log_{{10}} M_{{\rm cl}} = {opt_brk:.2f}$")
    ax.plot(opt_brk, opt_bic, "o", color="#D55E00", ms=8, zorder=6)

    ax.axhline(pw["BIC_single"], color="0.5", ls=":", lw=1.0,
               label=f"Single-fit BIC = {pw['BIC_single']:.1f}")

    ax.set_xlabel(r"Break-point $\log_{10}(M_{\rm cl}\;/\;M_\odot)$")
    ax.set_ylabel("Total BIC (piecewise)")
    ax.set_title(f"BIC scan: {quantity}", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / f"scaling_{quantity}_bic_scan.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved BIC scan: %s", out_path)
    return out_path


# ======================================================================
# Summary table
# ======================================================================

def write_summary(fits: List[Dict], output_dir: Path) -> Path:
    """
    Write and print a summary CSV of all fitted scaling relations.

    Parameters
    ----------
    fits : list of dict
        Results from :func:`fit_scaling` (one per quantity).
    output_dir : Path
        Directory to save the CSV.

    Returns
    -------
    Path
        Path to saved CSV.
    """
    csv_path = output_dir / "scaling_summary.csv"
    refs = fits[0]["refs"]
    nC_ref = refs["nCore"]
    mC_ref = refs["mCloud"]
    sfe_ref = refs["sfe"]
    header = [
        "quantity", "A",
        f"alpha (nCore/{nC_ref:.0e} cm^-3)", "sigma_alpha",
        f"beta (mCloud/{mC_ref:.0e} Msun)", "sigma_beta",
        f"gamma (SFE/{sfe_ref:.0e})", "sigma_gamma",
        "R2", "RMS [dex]", "N_used", "N_rejected",
    ]

    rows = []
    for f in fits:
        row = [
            f["quantity"],
            f"{f['A']:.6g}",
            f"{f['exponents']['nCore']:.4f}",
            f"{f['exponent_unc']['nCore']:.4f}",
            f"{f['exponents']['mCloud']:.4f}",
            f"{f['exponent_unc']['mCloud']:.4f}",
            f"{f['exponents']['sfe']:.4f}",
            f"{f['exponent_unc']['sfe']:.4f}",
            f"{f['R2']:.4f}",
            f"{f['rms_dex']:.4f}",
            str(f["n_used"]),
            str(f["n_rejected"]),
        ]
        rows.append(row)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    # Also print to stdout
    print()
    print("=" * 100)
    print("SCALING RELATION SUMMARY")
    print("=" * 100)

    # Formatted table
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2
                  for i, h in enumerate(header)]
    fmt_row = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt_row.format(*header))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt_row.format(*row))
    print()

    for f in fits:
        print(f"  {f['quantity']}  ~  {f['equation_str']}")
    print()

    logger.info("Saved summary CSV: %s", csv_path)
    return csv_path


# ======================================================================
# Equation JSON (for run_all summary)
# ======================================================================

def _regenerate_summary_pdf(output_dir: Path, fmt: str = "pdf") -> None:
    """Re-generate the cross-script scaling-relations summary PDF.

    Loads ``run_all.generate_summary_pdf`` via importlib so that this
    script can update the PDF when run individually (without run_all).
    """
    try:
        import importlib.util
        _mod_path = str(Path(__file__).resolve().parent / "run_all.py")
        spec = importlib.util.spec_from_file_location("_run_all", _mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # run_all was designed without usetex; honour that inside rc_context
        with plt.rc_context({"text.usetex": False}):
            mod.generate_summary_pdf(output_dir, fmt=fmt)
    except Exception as exc:
        logger.warning("Could not regenerate summary PDF: %s", exc)


def _extract_rejected(fit):
    """Extract identifying info for sigma-clipped (rejected) points."""
    mask = fit.get("mask")
    if mask is None:
        return []
    rejected = []
    for i, m in enumerate(mask):
        if not m:
            info = {}
            for k in ("nCore", "mCloud", "sfe"):
                arr = fit.get(k)
                if arr is not None and i < len(arr):
                    info[k] = float(arr[i])
            flds = fit.get("folders")
            if flds is not None and i < len(flds):
                info["folder"] = flds[i]
            if info:
                rejected.append(info)
    return rejected


def _write_equation_json(
    fits: List[Dict],
    output_dir: Path,
    pw_entries: Optional[List[Tuple[str, Dict]]] = None,
) -> Path:
    """Write equation data for the run_all summary PDF."""
    entries = []
    for f in fits:
        entries.append({
            "script": "scaling_phases",
            "label": f"{f['quantity']} [Myr]",
            "A": float(f["A"]),
            "exponents": {k: float(v) for k, v in f["exponents"].items()},
            "exponent_unc": {k: float(v) for k, v in f["exponent_unc"].items()},
            "refs": {k: float(v) for k, v in f["refs"].items()},
            "R2": float(f["R2"]),
            "rms_dex": float(f["rms_dex"]),
            "n_used": int(f["n_used"]),
            "n_rejected": int(f.get("n_rejected", 0)),
            "rejected": _extract_rejected(f),
        })
    # Piecewise sub-fit entries (when BIC is better)
    if pw_entries:
        for label, sf in pw_entries:
            entries.append({
                "script": "scaling_phases",
                "label": label,
                "A": float(sf["A"]),
                "exponents": {k: float(v) for k, v in sf["exponents"].items()},
                "exponent_unc": {k: float(v) for k, v in sf["exponent_unc"].items()},
                "refs": {k: float(v) for k, v in sf["refs"].items()},
                "R2": float(sf["R2"]),
                "rms_dex": float(sf["rms_dex"]),
                "n_used": int(sf["n_used"]),
                "n_rejected": int(sf.get("n_rejected", 0)),
                "rejected": _extract_rejected(sf),
            })
    path = output_dir / "scaling_phases_equations.json"
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit power-law scaling relations for TRINITY phase-transition timescales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scaling_phases.py -F /path/to/sweep_output
  python scaling_phases.py -F /path/to/sweep_output --quantities t_trans,t_mom
  python scaling_phases.py -F /path/to/sweep_output --sigma-clip 2.5 --fmt png
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--nCore-ref", type=float, default=1e3,
        help="Reference normalization for core density [cm^-3] (default: 1e3).",
    )
    parser.add_argument(
        "--mCloud-ref", type=float, default=1e5,
        help="Reference normalization for cloud mass [Msun] (default: 1e5).",
    )
    parser.add_argument(
        "--sfe-ref", type=float, default=0.01,
        help="Reference normalization for star formation efficiency (default: 0.01).",
    )
    parser.add_argument(
        "--sigma-clip", type=float, default=3.0,
        help="Number of sigma for outlier rejection (default: 3.0).",
    )
    parser.add_argument(
        "--quantities", type=str, default="t_trans,t_trans_dur,t_mom",
        help="Comma-separated list of timescales to fit (default: all three).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Maximum time [Myr] to consider in calculations.",
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Enable piecewise power-law fits with automatic break-point detection.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    folder_path = Path(args.folder)
    if not folder_path.is_dir():
        logger.error("Folder does not exist: %s", folder_path)
        return 1

    # Output into ./fig/{folder_name}/ matching other paper_* scripts
    folder_name = folder_path.name
    output_dir = FIG_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    quantities = [q.strip() for q in args.quantities.split(",")]
    for q in quantities:
        if q not in QUANTITY_DEFS:
            logger.error(
                "Unknown quantity '%s'. Valid: %s",
                q, ", ".join(QUANTITY_DEFS),
            )
            return 1

    # Step 1: collect data
    records = collect_data(folder_path, t_end=args.t_end)
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Step 2: fit each quantity
    fits: List[Dict] = []
    for q in quantities:
        logger.info("--- Fitting: %s ---", q)
        result = fit_scaling(
            records, q,
            nCore_ref=args.nCore_ref,
            mCloud_ref=args.mCloud_ref,
            sfe_ref=args.sfe_ref,
            sigma_clip=args.sigma_clip,
        )
        if result is None:
            logger.warning("Skipping '%s' (insufficient data or fit failure)", q)
            continue
        fits.append(result)

        # Step 3: plot
        plot_parity(result, output_dir, fmt=args.fmt)
        plot_parity_diagnostic(result, output_dir, fmt=args.fmt)
        plot_residuals(result, output_dir, fmt=args.fmt)

    if not fits:
        logger.error("No quantities could be fitted.")
        return 1

    # Step 3b: piecewise fits (always run; diagnostic plots gated)
    pw_entries: List[Tuple[str, Dict]] = []
    for q in quantities:
        print(f"\n--- Piecewise fitting: {q} ---")
        pw = fit_scaling_piecewise(
            records, q,
            nCore_ref=args.nCore_ref,
            mCloud_ref=args.mCloud_ref,
            sfe_ref=args.sfe_ref,
            sigma_clip=args.sigma_clip,
        )
        if pw is None:
            print(f"  Piecewise fit returned None for '{q}' "
                  f"(see INFO log above for details)")
            continue

        if args.diagnostics:
            plot_parity_piecewise(pw, output_dir, fmt=args.fmt)
            plot_bic_scan(pw, output_dir, fmt=args.fmt)

        delta_bic = pw["BIC_single"] - pw["BIC_piecewise"]
        brk_val = pw["break_log_Mcl"]
        print(f"\n--- Piecewise: {q} ---")
        print(f"  Break: log10(M_cl) = {brk_val:.3f}"
              f"  (M_cl = {10**brk_val:.1f} Msun)")
        print(f"  BIC single   = {pw['BIC_single']:.1f}")
        print(f"  BIC piecewise = {pw['BIC_piecewise']:.1f}"
              f"  (Delta = {delta_bic:+.1f})")
        print(f"  R2 low  = {pw['fit_low']['R2']:.4f}"
              f"  (N = {pw['fit_low']['n_used']})")
        print(f"  R2 high = {pw['fit_high']['R2']:.4f}"
              f"  (N = {pw['fit_high']['n_used']})")

        # If piecewise is better, include sub-fits in summary
        if delta_bic > 0:
            brk_str = f"Mcl<{10**brk_val:.0f}"
            pw_entries.append((
                f"{q} pw-low [{brk_str}] [Myr]",
                pw["fit_low"],
            ))
            brk_str = f"Mcl>{10**brk_val:.0f}"
            pw_entries.append((
                f"{q} pw-high [{brk_str}] [Myr]",
                pw["fit_high"],
            ))

    # Step 4: summary
    write_summary(fits, output_dir)

    # Step 5: equation JSON for run_all summary
    _write_equation_json(fits, output_dir, pw_entries=pw_entries or None)

    # Step 6: regenerate the cross-script summary PDF
    _regenerate_summary_pdf(output_dir, fmt=args.fmt or "pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
