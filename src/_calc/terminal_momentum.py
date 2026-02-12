#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terminal radial momentum per stellar mass from TRINITY parameter sweeps.

Physics background
------------------
The **terminal momentum** p_fin is the total radial momentum carried by
the swept-up shell when it reaches the cloud boundary (for expanding
runs) or at peak expansion (for collapsing runs).  Normalised by the
stellar mass M_* = ε M_cloud, the ratio p_fin/M_* [km/s] quantifies
the net momentum yield of stellar feedback per unit star formation —
a key input for sub-grid feedback models in galaxy simulations.

For a single supernova exploding into a uniform medium the momentum
yield has been calibrated numerically:

    p_fin/M_* ≈ 1000–3000 km/s   (Kim & Ostriker 2015; Martizzi+ 2015)

with a weak dependence on ambient density (∝ n^{-0.16}).  In reality,
multiple feedback mechanisms operate *simultaneously*:

* **Stellar winds** inject momentum continuously before the first SN
  (t < 3–4 Myr), pre-evacuating the cavity and reducing the density
  into which the SN blast propagates.

* **Radiation pressure** (direct UV + reprocessed IR) acts on dust
  grains in the shell and can dominate in high-Σ environments.

* **Photoionised-gas (HII) pressure** pushes outward as ionising
  photons heat the inner shell surface to ≈ 10⁴ K.

* **Supernovae** deliver impulsive momentum injection after ≈ 3 Myr.

The *momentum boost factor* η = p_fin / p_input measures how much
extra momentum the shell gains beyond the direct mechanical input,
through work done by thermal pressure in the hot bubble (PdV
boosting).  In the adiabatic Weaver limit η ≈ 10–30; with cooling,
η ≈ 2–10.

Fitted quantities
-----------------
* **p_fin / M_*** [km/s] — terminal specific momentum, fitted
  separately for expanding and collapsing subsets.
* **p_boost** = p_fin / p_input — momentum boost factor (requires
  ``--decompose``).
* **Component momenta** — impulse from each force (F_grav, F_ram,
  F_ion_out, F_rad, F_ram_wind, F_ram_SN) integrated to t_fin.

Method
------
Shell momentum at each snapshot is p(t) = M_shell(t) × v₂(t)
[M☉ pc/Myr].  For expanding runs t_fin is the last snapshot; for
collapsing runs t_fin is the time of peak p (before deceleration
brings p → 0).  Power-law fits use sigma-clipping OLS.

References
----------
* Kim, C.-G. & Ostriker, E. C. (2015), ApJ, 802, 99 — SN momentum.
* Martizzi, D. et al. (2015), MNRAS, 450, 504 — SN momentum vs density.
* Lancaster, L. et al. (2021), ApJ, 914, 89 — wind momentum boost.
* Grudić, M. Y. et al. (2018), MNRAS, 475, 3511 — multi-mechanism feedback.

CLI usage
---------
    python terminal_momentum.py -F /path/to/sweep_output
    python terminal_momentum.py -F /path/to/sweep_output --decompose
    python terminal_momentum.py -F /path/to/sweep_output --fmt png
"""

import sys
import logging
import argparse
import csv
import json
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
from src._functions.unit_conversions import CGS, CONV, INV_CONV

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root, matching paper_* scripts
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Physical constants and helpers
# ======================================================================

MU_MOL = 1.4            # molecular gas mean molecular weight per H nucleus
V_WIND_REF = 2000.0     # reference wind velocity [km/s] for loading factor

# Velocity conversion: internal (pc/Myr) → km/s
V_AU2KMS = INV_CONV.v_au2kms    # ~0.978 km/s per pc/Myr

# Reference p/m* from Martizzi+ 2015 (SN-only, n=100 cm^-3, solar Z)
P_M_MARTIZZI = 1420.0   # km/s


def _cumtrapz_1d(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with p[0]=0."""
    dx = np.diff(x)
    incr = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(incr)
    return out


def _surface_density(mCloud: float, rCloud: float) -> float:
    """Sigma = M / (pi R^2) [Msun/pc^2]."""
    return mCloud / (np.pi * rCloud ** 2)


def _cloud_radius_pc(mCloud_Msun: float, nCore_cm3: float) -> float:
    """Cloud radius for a uniform sphere [pc]."""
    rho_cgs = MU_MOL * CGS.m_H * nCore_cm3
    M_g = mCloud_Msun / CONV.g2Msun
    R_cm = (3.0 * M_g / (4.0 * np.pi * rho_cgs)) ** (1.0 / 3.0)
    return R_cm * CONV.cm2pc


# ======================================================================
# Data extraction
# ======================================================================

# Force fields whose impulses we integrate (same order as paper_momentum.py)
# F_grav is inward (negative contribution); all others are outward (positive).
MAIN_FORCES = ["F_grav", "F_ram", "F_ion_out", "F_rad"]
SUB_FORCES = ["F_ram_wind", "F_ram_SN"]

EXPAND = "expand"
COLLAPSE = "collapse"
STALLED = "stalled"


def extract_run(data_path: Path, decompose: bool = False,
                t_end: float = None) -> Optional[Dict]:
    """
    Load one TRINITY run and extract terminal momentum + metadata.

    Parameters
    ----------
    data_path : Path
        Path to dictionary.jsonl.
    decompose : bool
        If True, also compute cumulative momentum per force component.
    t_end : float, optional
        If given, truncate time series at this value [Myr].

    Returns
    -------
    dict or None
    """
    try:
        output = load_output(data_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", data_path, e)
        return None

    if len(output) < 3:
        logger.warning("Fewer than 3 snapshots in %s — skipping", data_path)
        return None

    # Time series
    t = output.get("t_now")
    v2 = output.get("v2")
    shell_mass = output.get("shell_mass")
    R2 = output.get("R2")

    # Replace NaN
    v2 = np.nan_to_num(v2, nan=0.0)
    shell_mass = np.nan_to_num(shell_mass, nan=0.0)

    # Truncate at t_end if requested
    _truncated = False
    if t_end is not None and t[-1] > t_end:
        mask_t = t <= t_end
        if mask_t.sum() < 3:
            logger.info("Fewer than 3 snapshots within t_end=%.3f in %s — skip",
                        t_end, data_path.parent.name)
            return None
        t = t[mask_t]
        v2 = v2[mask_t]
        shell_mass = shell_mass[mask_t]
        R2 = R2[mask_t]
        _truncated = True

    # Outcome
    first = output[0]
    if _truncated:
        # Classify by velocity at truncation point
        outcome = EXPAND if v2[-1] > 0 else COLLAPSE
    else:
        last = output[-1]
        is_collapse = last.get("isCollapse", False)
        is_dissolved = last.get("isDissolved", False)
        end_reason = str(last.get("SimulationEndReason", "")).lower()

        if is_dissolved or "dissolved" in end_reason or "large radius" in end_reason:
            outcome = EXPAND
        elif is_collapse or "small radius" in end_reason or "collapsed" in end_reason:
            outcome = COLLAPSE
        elif "stopping time" in end_reason or "max time" in end_reason:
            outcome = STALLED
        else:
            outcome = EXPAND if v2[-1] > 0 else COLLAPSE

    # Cloud / cluster params
    rCloud = first.get("rCloud", None)
    mCloud_snap = first.get("mCloud", None)

    # Shell momentum time series: p(t) = shell_mass * v2  [Msun * pc/Myr]
    p_shell = shell_mass * v2

    # Terminal momentum
    if outcome == EXPAND:
        # At dispersal (last snapshot)
        i_term = len(t) - 1
    else:
        # For collapse/stalled: peak momentum (maximum p > 0) before re-collapse
        # (At peak R2, v2≈0 so momentum≈0; the physically useful quantity is
        # the peak momentum achieved before deceleration.)
        i_term = int(np.argmax(p_shell))

    p_fin = p_shell[i_term]                   # Msun * pc/Myr
    v_fin = v2[i_term]                         # pc/Myr
    m_shell_fin = shell_mass[i_term]           # Msun
    t_fin = t[i_term]                          # Myr

    rec: Dict = {
        "outcome": outcome,
        "rCloud": rCloud,
        "mCloud_snap": mCloud_snap,
        "p_fin_au": float(p_fin),              # Msun * pc/Myr
        "v_fin_au": float(v_fin),
        "m_shell_fin": float(m_shell_fin),
        "t_fin": float(t_fin),
    }

    # Momentum decomposition via force integration
    if decompose:
        components: Dict[str, float] = {}
        for field in MAIN_FORCES + SUB_FORCES:
            F_arr = np.nan_to_num(output.get(field), nan=0.0)
            if _truncated:
                F_arr = F_arr[mask_t]
            P_arr = _cumtrapz_1d(F_arr, t)        # Msun * pc/Myr
            components[field] = float(P_arr[i_term])
        rec["components"] = components

        # Input momentum (wind + SN injection)
        F_wind = np.nan_to_num(output.get("F_ram_wind"), nan=0.0)
        F_SN = np.nan_to_num(output.get("F_ram_SN"), nan=0.0)
        if _truncated:
            F_wind = F_wind[mask_t]
            F_SN = F_SN[mask_t]
        P_input = _cumtrapz_1d(F_wind + F_SN, t)
        rec["p_input_au"] = float(P_input[i_term])  # Msun * pc/Myr

    return rec


def collect_data(folder_path: Path, decompose: bool = False,
                 t_end: float = None) -> List[Dict]:
    """
    Walk sweep output and collect terminal momentum for every run.
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

        nCore = float(parsed["ndens"])
        mCloud = float(parsed["mCloud"])
        sfe = int(parsed["sfe"]) / 100.0

        info = extract_run(data_path, decompose=decompose, t_end=t_end)
        if info is None:
            continue

        M_star = sfe * mCloud                     # Msun

        rCloud = info["rCloud"]
        if rCloud is None or rCloud <= 0:
            rCloud = _cloud_radius_pc(mCloud, nCore)

        Sigma = _surface_density(mCloud, rCloud)

        # p_fin / M_* in km/s
        p_per_mstar_kms = (info["p_fin_au"] / M_star) * V_AU2KMS if M_star > 0 else 0.0

        # Kinetic energy at dispersal [Msun (pc/Myr)^2] → erg
        E_kin_au = 0.5 * info["m_shell_fin"] * info["v_fin_au"] ** 2
        E_kin_erg = E_kin_au * INV_CONV.E_au2cgs

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "M_star": M_star,
            "rCloud": rCloud,
            "Sigma": Sigma,
            "outcome": info["outcome"],
            "t_fin": info["t_fin"],
            "p_fin_au": info["p_fin_au"],
            "p_per_mstar_kms": p_per_mstar_kms,
            "v_fin_kms": info["v_fin_au"] * V_AU2KMS,
            "m_shell_fin": info["m_shell_fin"],
            "E_kin_erg": E_kin_erg,
            "folder": folder_name,
        }

        # Momentum loading factor: eta_p = (p/m*) / v_wind_ref
        rec["eta_p"] = p_per_mstar_kms / V_WIND_REF if V_WIND_REF > 0 else 0.0

        if decompose and "components" in info:
            rec["components"] = info["components"]
            p_input = info.get("p_input_au", 0.0)
            rec["p_input_au"] = p_input
            # Boost = p_fin / p_input
            rec["p_boost"] = (info["p_fin_au"] / p_input) if p_input > 0 else float("nan")
        else:
            rec["p_input_au"] = float("nan")
            rec["p_boost"] = float("nan")

        records.append(rec)

    n_exp = sum(1 for r in records if r["outcome"] == EXPAND)
    n_col = sum(1 for r in records if r["outcome"] == COLLAPSE)
    n_sta = sum(1 for r in records if r["outcome"] == STALLED)
    logger.info(
        "Collected %d runs: %d expand, %d collapse, %d stalled",
        len(records), n_exp, n_col, n_sta,
    )
    return records


# ======================================================================
# Fitting (reuses sigma-clipping OLS from scaling_phases.py pattern)
# ======================================================================

def _ols_sigma_clip(
    X: np.ndarray,
    y: np.ndarray,
    sigma_clip: float,
    max_iter: int = 10,
) -> Optional[Dict]:
    """OLS with iterative sigma-clipping."""
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


def fit_p_mstar(
    records: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    outcome_filter: Optional[str] = None,
    quantity_key: str = "p_per_mstar_kms",
) -> Optional[Dict]:
    """
    Fit power-law scaling for p_fin/M_* (or any per-M_* quantity).

    Parameters
    ----------
    outcome_filter : str or None
        If set, only include runs with this outcome ("expand" / "collapse").
    """
    pts = [
        r for r in records
        if (outcome_filter is None or r["outcome"] == outcome_filter)
        and np.isfinite(r[quantity_key]) and r[quantity_key] > 0
    ]
    if len(pts) < 2:
        logger.warning("Too few points (%d) for fit (filter=%s, key=%s)",
                        len(pts), outcome_filter, quantity_key)
        return None

    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])
    val = np.array([r[quantity_key] for r in pts])
    log_val = np.log10(val)

    # Build design matrix (only include varying parameters)
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
    result = _ols_sigma_clip(X, log_val, sigma_clip)
    if result is None:
        return None

    result["param_names"] = names
    result["nCore"] = nC
    result["mCloud"] = mC
    result["sfe"] = sfe
    result["actual"] = val
    result["predicted"] = 10.0 ** result["y_pred"]
    result["refs"] = refs
    result["outcome_filter"] = outcome_filter
    result["quantity_key"] = quantity_key

    # Human-readable equation
    A = 10.0 ** result["beta"][0]
    parts = [f"{A:.2f} km/s" if "kms" in quantity_key else f"{A:.3g}"]
    label_map = {"nCore": "n_c", "mCloud": r"M_{\rm cl}", "sfe": r"\varepsilon"}
    for i, name in enumerate(names[1:], 1):
        b = result["beta"][i]
        r = refs.get(name, 1.0)
        parts.append(f"({name}/{r:.0e})^{{{b:+.2f}}}")
    result["equation_str"] = " * ".join(parts)

    return result


def fit_p_mstar_quad(
    records: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    outcome_filter: Optional[str] = None,
    quantity_key: str = "p_per_mstar_kms",
) -> Optional[Dict]:
    """
    Fit power-law with quadratic log-correction in M_cloud.

    Model:
        log10(p/M*) = log10(A) + alpha*log10(n_c/n0)
                     + beta*log10(M/M0) + gamma*log10(eps/eps0)
                     + delta*[log10(M/M0)]^2
    """
    pts = [
        r for r in records
        if (outcome_filter is None or r["outcome"] == outcome_filter)
        and np.isfinite(r[quantity_key]) and r[quantity_key] > 0
    ]
    if len(pts) < 2:
        logger.warning("Too few points (%d) for quad fit (filter=%s)",
                        len(pts), outcome_filter)
        return None

    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])
    val = np.array([r[quantity_key] for r in pts])
    log_val = np.log10(val)

    # Build design matrix (standard columns + quadratic mCloud term)
    cols = [np.ones(len(pts))]
    names = ["intercept"]
    refs = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}
    log_mC_norm = np.log10(mC / mCloud_ref)

    for pname, arr, ref in [("nCore", nC, nCore_ref),
                            ("mCloud", mC, mCloud_ref),
                            ("sfe", sfe, sfe_ref)]:
        if len(np.unique(arr)) >= 2:
            cols.append(np.log10(arr / ref))
            names.append(pname)

    # Append quadratic term for mCloud
    cols.append(log_mC_norm ** 2)
    names.append("mCloud_sq")

    X = np.column_stack(cols)
    result = _ols_sigma_clip(X, log_val, sigma_clip)
    if result is None:
        return None

    result["param_names"] = names
    result["nCore"] = nC
    result["mCloud"] = mC
    result["sfe"] = sfe
    result["actual"] = val
    result["predicted"] = 10.0 ** result["y_pred"]
    result["refs"] = refs
    result["outcome_filter"] = outcome_filter
    result["quantity_key"] = quantity_key

    # Human-readable equation
    A = 10.0 ** result["beta"][0]
    parts = [f"{A:.2f} km/s" if "kms" in quantity_key else f"{A:.3g}"]
    label_map = {"nCore": "n_c", "mCloud": r"M_{\rm cl}", "sfe": r"\varepsilon"}
    for i, name in enumerate(names[1:], 1):
        b = result["beta"][i]
        if name == "mCloud_sq":
            parts.append(
                f"10^{{{b:+.3f}[log(M/{mCloud_ref:.0e})]^2}}")
        else:
            r_val = refs.get(name, 1.0)
            parts.append(f"({name}/{r_val:.0e})^{{{b:+.2f}}}")
    result["equation_str"] = " * ".join(parts)

    return result


def fit_p_mstar_piecewise(
    records: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    outcome_filter: Optional[str] = None,
    quantity_key: str = "p_per_mstar_kms",
    n_break_candidates: int = 50,
) -> Optional[Dict]:
    """
    Piecewise power-law with automatic break-point detection in log10(M_cl).

    Scans candidate break values in log10(M_cl) from the 10th to the 90th
    percentile and selects the break that minimises total BIC.
    """
    pts = [
        r for r in records
        if (outcome_filter is None or r["outcome"] == outcome_filter)
        and np.isfinite(r[quantity_key]) and r[quantity_key] > 0
    ]
    if len(pts) < 10:
        logger.warning("Too few points (%d) for piecewise fit (filter=%s)",
                        len(pts), outcome_filter)
        return None

    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])
    val = np.array([r[quantity_key] for r in pts])
    log_val = np.log10(val)
    log_Mcl = np.log10(sfe * mC)
    folders = np.array([r.get("folder", "") for r in pts])

    refs = {"nCore": nCore_ref, "mCloud": mCloud_ref, "sfe": sfe_ref}

    def _build_X(indices):
        """Build standard design matrix for a subset."""
        cols = [np.ones(len(indices))]
        names = ["intercept"]
        nC_sub = nC[indices]
        mC_sub = mC[indices]
        sfe_sub = sfe[indices]
        for pname, arr, ref in [("nCore", nC_sub, nCore_ref),
                                ("mCloud", mC_sub, mCloud_ref),
                                ("sfe", sfe_sub, sfe_ref)]:
            if len(np.unique(arr)) >= 2:
                cols.append(np.log10(arr / ref))
                names.append(pname)
        return np.column_stack(cols), names

    def _bic(n, k, rss):
        """BIC = n*ln(RSS/n) + k*ln(n)."""
        if n <= k or rss <= 0:
            return np.inf
        return n * np.log(rss / n) + k * np.log(n)

    # Single-fit BIC for comparison
    X_all, names_all = _build_X(np.arange(len(pts)))
    fit_single = _ols_sigma_clip(X_all, log_val, sigma_clip)
    if fit_single is None:
        return None
    rss_single = np.sum((log_val[fit_single["mask"]] -
                         fit_single["y_pred"][fit_single["mask"]]) ** 2)
    bic_single = _bic(fit_single["n_used"], X_all.shape[1], rss_single)

    # Scan break candidates
    lo_pct = np.percentile(log_Mcl, 10)
    hi_pct = np.percentile(log_Mcl, 90)
    candidates = np.linspace(lo_pct, hi_pct, n_break_candidates)

    best_bic = np.inf
    best_break = None
    best_fit_lo = None
    best_fit_hi = None
    best_idx_lo = None
    best_idx_hi = None
    best_names_lo = None
    best_names_hi = None
    bic_scan = []

    for brk in candidates:
        idx_lo = np.where(log_Mcl <= brk)[0]
        idx_hi = np.where(log_Mcl > brk)[0]

        if len(idx_lo) < 5 or len(idx_hi) < 5:
            bic_scan.append((brk, np.inf))
            continue

        X_lo, names_lo = _build_X(idx_lo)
        X_hi, names_hi = _build_X(idx_hi)

        fit_lo = _ols_sigma_clip(X_lo, log_val[idx_lo], sigma_clip)
        fit_hi = _ols_sigma_clip(X_hi, log_val[idx_hi], sigma_clip)

        if fit_lo is None or fit_hi is None:
            bic_scan.append((brk, np.inf))
            continue

        rss_lo = np.sum(
            (log_val[idx_lo][fit_lo["mask"]] -
             fit_lo["y_pred"][fit_lo["mask"]]) ** 2)
        rss_hi = np.sum(
            (log_val[idx_hi][fit_hi["mask"]] -
             fit_hi["y_pred"][fit_hi["mask"]]) ** 2)

        bic_lo = _bic(fit_lo["n_used"], X_lo.shape[1], rss_lo)
        bic_hi = _bic(fit_hi["n_used"], X_hi.shape[1], rss_hi)
        bic_total = bic_lo + bic_hi

        bic_scan.append((brk, bic_total))

        if bic_total < best_bic:
            best_bic = bic_total
            best_break = brk
            best_fit_lo = fit_lo
            best_fit_hi = fit_hi
            best_idx_lo = idx_lo
            best_idx_hi = idx_hi
            best_names_lo = names_lo
            best_names_hi = names_hi

    if best_fit_lo is None or best_fit_hi is None:
        logger.warning("Piecewise fit failed — no valid break found")
        return None

    # Package results so plot_parity can consume them
    # Combine arrays from both subsets
    n_lo = len(best_idx_lo)
    combined_mask = np.concatenate([best_fit_lo["mask"], best_fit_hi["mask"]])
    combined_actual = np.concatenate([val[best_idx_lo], val[best_idx_hi]])
    combined_pred = np.concatenate([
        10.0 ** best_fit_lo["y_pred"],
        10.0 ** best_fit_hi["y_pred"],
    ])
    combined_nCore = np.concatenate([nC[best_idx_lo], nC[best_idx_hi]])
    combined_mCloud = np.concatenate([mC[best_idx_lo], mC[best_idx_hi]])
    combined_sfe = np.concatenate([sfe[best_idx_lo], sfe[best_idx_hi]])
    combined_folders = np.concatenate([folders[best_idx_lo], folders[best_idx_hi]])
    side_labels = np.array(["low"] * n_lo + ["high"] * (len(combined_mask) - n_lo))

    # Add standard fields to sub-fits
    for subfit, idx, names in [
        (best_fit_lo, best_idx_lo, best_names_lo),
        (best_fit_hi, best_idx_hi, best_names_hi),
    ]:
        subfit["param_names"] = names
        subfit["nCore"] = nC[idx]
        subfit["mCloud"] = mC[idx]
        subfit["sfe"] = sfe[idx]
        subfit["actual"] = val[idx]
        subfit["predicted"] = 10.0 ** subfit["y_pred"]
        subfit["refs"] = refs
        subfit["outcome_filter"] = outcome_filter
        subfit["quantity_key"] = quantity_key

    return {
        "break_log_Mcl": float(best_break),
        "fit_low": best_fit_lo,
        "fit_high": best_fit_hi,
        "BIC_piecewise": float(best_bic),
        "BIC_single": float(bic_single),
        "bic_scan": bic_scan,
        # Combined arrays for plotting
        "actual": combined_actual,
        "predicted": combined_pred,
        "mask": combined_mask,
        "nCore": combined_nCore,
        "mCloud": combined_mCloud,
        "sfe": combined_sfe,
        "folders": combined_folders,
        "side": side_labels,
        "outcome_filter": outcome_filter,
        "quantity_key": quantity_key,
        "refs": refs,
    }


# ======================================================================
# Plotting
# ======================================================================

_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

OUTCOME_COLORS = {EXPAND: "C0", COLLAPSE: "C3", STALLED: "0.55"}
OUTCOME_LABELS = {EXPAND: "Dispersal", COLLAPSE: "Collapse", STALLED: "Stalled"}


def plot_p_vs_sfe(
    records: List[Dict],
    fit_exp: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 1: p_fin/M_* vs SFE."""
    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = max(len(unique_nCore), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5),
                             squeeze=False, dpi=150)
    axes = axes.ravel()

    for i, nc in enumerate(unique_nCore):
        ax = axes[i]
        subset = [r for r in records if r["nCore"] == nc]

        unique_mCloud = sorted(set(r["mCloud"] for r in subset))
        cmap = plt.cm.viridis
        if len(unique_mCloud) > 1:
            norm = matplotlib.colors.LogNorm(
                vmin=min(unique_mCloud), vmax=max(unique_mCloud))
        else:
            norm = matplotlib.colors.LogNorm(
                vmin=unique_mCloud[0] * 0.5, vmax=unique_mCloud[0] * 2.0)

        for outcome in [COLLAPSE, STALLED, EXPAND]:
            pts = [r for r in subset if r["outcome"] == outcome]
            if not pts:
                continue
            facecolors = [cmap(norm(r["mCloud"])) for r in pts]
            marker = "o" if outcome == EXPAND else ("x" if outcome == COLLAPSE else "+")
            ax.scatter(
                [r["sfe"] for r in pts],
                [r["p_per_mstar_kms"] for r in pts],
                c=facecolors,
                marker=marker,
                s=40,
                edgecolors="k" if outcome == EXPAND else "none",
                linewidths=0.3,
                label=OUTCOME_LABELS[outcome],
                zorder=5,
            )

        # Reference lines
        ax.axhline(P_M_MARTIZZI, color="grey", ls="--", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1e-3,
                P_M_MARTIZZI * 1.1,
                "Martizzi+ 2015 (SN-only)", fontsize=6, color="grey")
        ax.axhline(3000, color="grey", ls=":", lw=0.8, alpha=0.4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"SFE ($\varepsilon$)")
        if i == 0:
            ax.set_ylabel(r"$p_{\rm fin}/M_*$ [km/s]")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)
        ax.legend(fontsize=7, loc="best", framealpha=0.8)

    for j in range(len(unique_nCore), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"terminal_momentum_vs_sfe.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_momentum_budget(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Optional[Path]:
    """Figure 2: Stacked bar chart of fractional momentum by mechanism."""
    pts = [r for r in records if "components" in r and r["outcome"] == EXPAND
           and r["p_fin_au"] > 0]
    if not pts:
        logger.warning("No decomposed expanding runs — skipping budget plot")
        return None

    # Sort by SFE then mCloud
    pts.sort(key=lambda r: (r["nCore"], r["mCloud"], r["sfe"]))

    labels = [f"{r['sfe']:.0e}\n{r['mCloud']:.0e}" for r in pts]

    comp_names = [
        ("F_ram_wind", "Wind", "C0"),
        ("F_ram_SN", "SN", "C3"),
        ("F_ion_out", "HII", "C2"),
        ("F_rad", "Radiation", "C4"),
    ]
    grav_name = ("F_grav", "Gravity", "0.5")

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(pts)), 5), dpi=150)
    x = np.arange(len(pts))

    # Compute positive fractional contributions
    for ri, rec in enumerate(pts):
        comp = rec["components"]
        total_pos = sum(max(comp.get(f, 0), 0) for f, _, _ in comp_names)
        if total_pos == 0:
            total_pos = 1.0
        bottom = 0.0
        for field, label, color in comp_names:
            val = max(comp.get(field, 0), 0) / total_pos
            ax.bar(ri, val, bottom=bottom, color=color,
                   label=label if ri == 0 else None, width=0.8)
            bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_xlabel(r"SFE / $M_{\rm cloud}$")
    ax.set_ylabel("Fractional momentum contribution")
    ax.set_title("Momentum budget (expanding runs)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = output_dir / f"terminal_momentum_budget.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_boost(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Optional[Path]:
    """Figure 3: Momentum boost factor p_fin / p_input."""
    pts = [r for r in records
           if np.isfinite(r["p_boost"]) and r["p_boost"] > 0
           and np.isfinite(r["p_input_au"]) and r["p_input_au"] > 0]
    if len(pts) < 2:
        logger.warning("Not enough runs with p_boost — skipping boost plot")
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    unique_nCore = sorted(set(r["nCore"] for r in pts))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    for nc in unique_nCore:
        sub = [r for r in pts if r["nCore"] == nc]
        for outcome in [EXPAND, COLLAPSE, STALLED]:
            osub = [r for r in sub if r["outcome"] == outcome]
            if not osub:
                continue
            p_inp = np.array([r["p_input_au"] * V_AU2KMS for r in osub])  # Msun km/s
            boost = np.array([r["p_boost"] for r in osub])
            marker = nc_to_marker[nc]
            facecolor = OUTCOME_COLORS[outcome]
            ax.scatter(
                p_inp, boost,
                marker=marker, s=40, c=facecolor,
                edgecolors="k", linewidths=0.3, zorder=5,
                label=f"$n_c={nc:.0e}$ ({OUTCOME_LABELS[outcome]})",
            )

    ax.axhline(1, color="grey", ls="--", lw=0.8, alpha=0.5, label="Boost = 1")
    ax.axhline(10, color="grey", ls=":", lw=0.8, alpha=0.4, label="Boost = 10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_{\rm input}$ [$M_\odot\,$km/s]")
    ax.set_ylabel(r"$p_{\rm fin} / p_{\rm input}$")
    ax.set_title("Momentum boost factor", fontsize=10)
    ax.legend(fontsize=6, loc="best", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"terminal_momentum_boost.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_parity(
    fit: Optional[Dict],
    output_dir: Path,
    fmt: str,
    suffix: str = "",
) -> Optional[Path]:
    """Figure 4: Parity plot for p_fin/M_* fit."""
    if fit is None:
        return None

    actual = fit["actual"]
    predicted = fit["predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    nCore = fit["nCore"]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    log_mCloud = np.log10(mCloud)
    unique_nCore = sorted(set(nCore))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    vals = np.concatenate([actual, predicted])
    lo, hi = vals[vals > 0].min() * 0.5, vals.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    for nc in unique_nCore:
        nc_mask = nCore == nc
        marker = nc_to_marker[nc]
        sel = nc_mask & mask
        if sel.any():
            ax.scatter(
                actual[sel], predicted[sel],
                c=log_mCloud[sel], marker=marker, s=50,
                edgecolors="k", linewidths=0.4, zorder=5,
                label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
            )
        sel_out = nc_mask & ~mask
        if sel_out.any():
            ax.scatter(
                actual[sel_out], predicted[sel_out],
                c=log_mCloud[sel_out], marker=marker, s=50,
                alpha=0.3, edgecolors="grey", linewidths=0.8, zorder=3,
            )

    vmin, vmax = log_mCloud.min(), log_mCloud.max()
    if vmin == vmax:
        vmin -= 0.5; vmax += 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

    filt = fit.get("outcome_filter", "all")
    ann = (
        rf"$R^2 = {fit['R2']:.3f}$" + "\n"
        + rf"RMS $= {fit['rms_dex']:.3f}$ dex" + "\n"
        + rf"$N = {fit['n_used']}$ (rejected {fit['n_rejected']})" + "\n"
        + f"Filter: {filt}"
    )
    ax.text(0.04, 0.96, ann, transform=ax.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$p_{\rm fin}/M_*$ from TRINITY [km/s]")
    ax.set_ylabel(r"$p_{\rm fin}/M_*$ from power-law fit [km/s]")
    ax.set_title(r"Parity: $p_{\rm fin}/M_*$" + (f" ({filt})" if filt else ""),
                 fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    tag = f"_{filt}" if filt else ""
    out = output_dir / f"terminal_momentum_parity{tag}{suffix}.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_p_vs_sigma(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 5: p_fin/M_* vs surface density."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    unique_nCore = sorted(set(r["nCore"] for r in records))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    for nc in unique_nCore:
        for outcome in [EXPAND, COLLAPSE]:
            pts = [r for r in records
                   if r["nCore"] == nc and r["outcome"] == outcome
                   and r["p_per_mstar_kms"] > 0]
            if not pts:
                continue
            marker = nc_to_marker[nc]
            facecolor = OUTCOME_COLORS[outcome]
            ax.scatter(
                [r["Sigma"] for r in pts],
                [r["p_per_mstar_kms"] for r in pts],
                marker=marker, s=40, c=facecolor,
                edgecolors="k", linewidths=0.3, zorder=5,
                label=rf"$n_c={nc:.0e}$ ({OUTCOME_LABELS[outcome]})",
            )

    # Martizzi+ 2015 reference: p/m* ~ 1420 (n/100)^{-0.16} km/s
    Sig_plot = np.geomspace(10, 1e4, 200)
    ax.axhline(P_M_MARTIZZI, color="grey", ls="--", lw=0.8, alpha=0.5,
               label=f"Martizzi+ 2015 SN-only ({P_M_MARTIZZI:.0f} km/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Sigma_{\rm cl}$ [$M_\odot\,$pc$^{-2}$]")
    ax.set_ylabel(r"$p_{\rm fin}/M_*$ [km/s]")
    ax.set_title("Terminal momentum vs. surface density", fontsize=10)
    ax.legend(fontsize=6, loc="best", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"terminal_momentum_vs_sigma.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Diagnostic plots (enabled by --diagnostics)
# ======================================================================

def plot_parity_diagnostic(
    fit: Optional[Dict],
    records: List[Dict],
    output_dir: Path,
    fmt: str,
    suffix: str = "",
) -> List[Path]:
    """
    Diagnostic parity plots colored by SFE, M_cl, and residual,
    plus a residual-vs-parameter panel.

    Produces four figures per fit to help identify curvature or
    two-population bifurcation.
    """
    if fit is None:
        return []

    actual = fit["actual"]
    predicted = fit["predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    nCore = fit["nCore"]
    sfe = fit["sfe"]
    filt = fit.get("outcome_filter", "all")

    unique_nCore = sorted(set(nCore))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    # 1:1 line bounds
    vals = np.concatenate([actual, predicted])
    lo, hi = vals[vals > 0].min() * 0.5, vals.max() * 2.0

    # Derived arrays
    residual = np.log10(actual) - np.log10(predicted)
    log_sfe = np.log10(sfe)
    log_Mcl = np.log10(sfe * mCloud)
    log_mCloud = np.log10(mCloud)

    # --- Figures 1-3: parity plots colored by different variables ---
    diag_configs = [
        ("sfe",      log_sfe,   r"$\log_{10}(\varepsilon)$",   "coolwarm"),
        ("Mcl",      log_Mcl,   r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$", "viridis"),
        ("residual", residual,
         r"$\log_{10}({\rm actual}) - \log_{10}({\rm fit})$",  "RdBu_r"),
    ]

    saved = []
    for var_name, color_arr, cbar_label, cmap_name in diag_configs:
        fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

        vmin, vmax = np.nanmin(color_arr), np.nanmax(color_arr)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        if var_name == "residual":
            vlim = max(abs(vmin), abs(vmax), 0.1)
            vmin, vmax = -vlim, vlim
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for nc in unique_nCore:
            nc_mask = nCore == nc
            marker = nc_to_marker[nc]

            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    actual[sel], predicted[sel],
                    c=color_arr[sel], cmap=cmap_name, norm=norm,
                    marker=marker, s=50, edgecolors="k", linewidths=0.4,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
                )
            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    actual[sel_out], predicted[sel_out],
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
        ax.set_xlabel(r"$p_{\rm fin}/M_*$ from TRINITY [km/s]")
        ax.set_ylabel(r"$p_{\rm fin}/M_*$ from fit [km/s]")
        ax.set_aspect("equal")
        tag = f"_{filt}" if filt else ""
        ax.set_title(
            rf"$p_{{\rm fin}}/M_*$ ({filt}) — by {var_name}", fontsize=10)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

        fig.tight_layout()
        out = output_dir / (
            f"terminal_momentum_parity{tag}_diag_by_{var_name}"
            f"{suffix}.{fmt}")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved diagnostic: %s", out)
        saved.append(out)

    # --- Figure 4: residual-vs-parameter panel (1x3) ---
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
            marker = nc_to_marker[nc]
            sel = nc_mask & mask
            if sel.any():
                ax.scatter(
                    x_arr[sel], residual[sel],
                    marker=marker, s=40, edgecolors="k", linewidths=0.3,
                    zorder=5,
                    label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
                )
            sel_out = nc_mask & ~mask
            if sel_out.any():
                ax.scatter(
                    x_arr[sel_out], residual[sel_out],
                    marker=marker, s=40, alpha=0.3,
                    edgecolors="grey", linewidths=0.6, zorder=3,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            r"$\log_{10}({\rm actual}) - \log_{10}({\rm fit})$")
        ax.legend(fontsize=7, loc="best", framealpha=0.8)

    tag = f"_{filt}" if filt else ""
    fig.suptitle(
        rf"$p_{{\rm fin}}/M_*$ ({filt}) — fit residuals", fontsize=11)
    fig.tight_layout()
    out = output_dir / (
        f"terminal_momentum_residuals{tag}{suffix}.{fmt}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residuals: %s", out)
    saved.append(out)

    return saved


def plot_parity_piecewise(
    pw: Optional[Dict],
    output_dir: Path,
    fmt: str,
    suffix: str = "",
) -> Optional[Path]:
    """
    Parity plot for a piecewise power-law fit, colored by log10(M_cl).

    Points below the break get thin black edges; points above get thick
    red edges.
    """
    if pw is None:
        return None

    actual = pw["actual"]
    predicted = pw["predicted"]
    mask = pw["mask"]
    nCore = pw["nCore"]
    mCloud = pw["mCloud"]
    sfe = pw["sfe"]
    side = pw["side"]
    filt = pw.get("outcome_filter", "all")
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
    ax.set_xlabel(r"$p_{\rm fin}/M_*$ from TRINITY [km/s]")
    ax.set_ylabel(r"$p_{\rm fin}/M_*$ from piecewise fit [km/s]")
    tag = f"_{filt}" if filt else ""
    ax.set_title(
        rf"Piecewise: $p_{{\rm fin}}/M_*$ ({filt})", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / (
        f"terminal_momentum_parity{tag}_piecewise{suffix}.{fmt}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_bic_scan(
    pw: Optional[Dict],
    output_dir: Path,
    fmt: str,
    suffix: str = "",
) -> Optional[Path]:
    """
    BIC scan plot: total BIC vs break-point location.
    """
    if pw is None or "bic_scan" not in pw:
        return None

    filt = pw.get("outcome_filter", "all")
    scan = np.array(pw["bic_scan"])
    brk_vals = scan[:, 0]
    bic_vals = scan[:, 1]

    # Filter out infinite values for plotting
    finite = np.isfinite(bic_vals)
    if finite.sum() < 2:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=150)
    ax.plot(brk_vals[finite], bic_vals[finite], "-", color="#0072B2", lw=1.5)

    # Mark optimum
    opt_brk = pw["break_log_Mcl"]
    opt_bic = pw["BIC_piecewise"]
    ax.axvline(opt_brk, color="#D55E00", ls="--", lw=1.2,
               label=rf"Optimum: $\log_{{10}} M_{{\rm cl}} = {opt_brk:.2f}$")
    ax.plot(opt_brk, opt_bic, "o", color="#D55E00", ms=8, zorder=6)

    # Single-fit BIC reference
    ax.axhline(pw["BIC_single"], color="0.5", ls=":", lw=1.0,
               label=f"Single-fit BIC = {pw['BIC_single']:.1f}")

    ax.set_xlabel(r"Break-point $\log_{10}(M_{\rm cl}\;/\;M_\odot)$")
    ax.set_ylabel("Total BIC (piecewise)")
    tag = f"_{filt}" if filt else ""
    ax.set_title(
        rf"BIC scan: $p_{{\rm fin}}/M_*$ ({filt})", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / (
        f"terminal_momentum_bic_scan{tag}{suffix}.{fmt}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    csv_path = output_dir / "terminal_momentum_results.csv"
    has_decompose = any("components" in r for r in records)

    header = [
        "nCore", "mCloud", "SFE", "M_star", "Sigma", "outcome",
        "p_fin_per_Mstar_kms", "v_fin_kms", "m_shell_fin",
        "E_kin_erg", "eta_p", "p_boost",
    ]
    if has_decompose:
        for f in MAIN_FORCES + SUB_FORCES:
            header.append(f"P_{f}")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            row = [
                f"{r['nCore']:.4e}", f"{r['mCloud']:.4e}",
                f"{r['sfe']:.4f}", f"{r['M_star']:.4e}",
                f"{r['Sigma']:.2f}", r["outcome"],
                f"{r['p_per_mstar_kms']:.2f}",
                f"{r['v_fin_kms']:.2f}", f"{r['m_shell_fin']:.4e}",
                f"{r['E_kin_erg']:.4e}", f"{r['eta_p']:.4f}",
                f"{r['p_boost']:.4f}" if np.isfinite(r["p_boost"]) else "N/A",
            ]
            if has_decompose:
                comp = r.get("components", {})
                for f in MAIN_FORCES + SUB_FORCES:
                    row.append(f"{comp.get(f, 0.0):.6e}")
            writer.writerow(row)

    logger.info("Saved: %s", csv_path)
    return csv_path


def write_fits_csv(fits: List[Tuple[str, Dict]], output_dir: Path) -> Path:
    csv_path = output_dir / "terminal_momentum_fits.csv"
    header = ["fit_label", "param", "coefficient", "uncertainty",
              "R2", "RMS_dex", "N_used", "N_rejected"]
    rows = []
    refs = None
    for label, fit in fits:
        if fit is None:
            continue
        if refs is None and "refs" in fit:
            refs = fit["refs"]
        for i, name in enumerate(fit["param_names"]):
            rows.append([
                label, name,
                f"{fit['beta'][i]:.6f}", f"{fit['unc'][i]:.6f}",
                f"{fit['R2']:.4f}" if i == 0 else "",
                f"{fit['rms_dex']:.4f}" if i == 0 else "",
                str(fit["n_used"]) if i == 0 else "",
                str(fit["n_rejected"]) if i == 0 else "",
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
    fits: List[Tuple[str, Optional[Dict]]],
) -> None:
    print()
    print("=" * 90)
    print("TERMINAL MOMENTUM SUMMARY")
    print("=" * 90)

    # Compact table
    exp = [r for r in records if r["outcome"] == EXPAND and r["p_per_mstar_kms"] > 0]
    col = [r for r in records if r["outcome"] == COLLAPSE and r["p_per_mstar_kms"] > 0]

    if exp:
        vals = np.array([r["p_per_mstar_kms"] for r in exp])
        print(f"\nExpanding runs ({len(exp)}):")
        print(f"  p_fin/M_* : median = {np.median(vals):.0f} km/s, "
              f"range = [{vals.min():.0f}, {vals.max():.0f}] km/s")
    if col:
        vals = np.array([r["p_per_mstar_kms"] for r in col])
        print(f"\nCollapsing runs ({len(col)}):")
        print(f"  p_peak/M_* : median = {np.median(vals):.0f} km/s, "
              f"range = [{vals.min():.0f}, {vals.max():.0f}] km/s")

    for label, fit in fits:
        if fit is None:
            continue
        print(f"\n--- {label} ---")
        A = 10.0 ** fit["beta"][0]
        print(f"  A = {A:.2f}")
        for i, name in enumerate(fit["param_names"][1:], 1):
            print(f"  {name}: {fit['beta'][i]:+.4f} +/- {fit['unc'][i]:.4f}")
        print(f"  R^2 = {fit['R2']:.4f}, RMS = {fit['rms_dex']:.4f} dex")
        print(f"  Scaling: {fit['equation_str']}")

    print()
    print("=" * 90)


# ======================================================================
# Equation JSON (for run_all summary)
# ======================================================================

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
    fits: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
    script_name: str,
) -> Path:
    """Write equation data for the run_all summary PDF."""
    entries = []
    for label, fit in fits:
        if fit is None:
            continue
        A = 10.0 ** fit["beta"][0]
        refs = fit.get("refs", {})
        names = fit["param_names"]
        exponents = {}
        exponent_unc = {}
        for i, name in enumerate(names[1:], 1):
            exponents[name] = float(fit["beta"][i])
            exponent_unc[name] = float(fit["unc"][i])
        entries.append({
            "script": script_name,
            "label": label,
            "A": float(A),
            "exponents": exponents,
            "exponent_unc": exponent_unc,
            "refs": {k: float(v) for k, v in refs.items()},
            "R2": float(fit["R2"]),
            "rms_dex": float(fit["rms_dex"]),
            "n_used": int(fit["n_used"]),
            "n_rejected": int(fit.get("n_rejected", 0)),
            "rejected": _extract_rejected(fit),
        })
    path = output_dir / f"{script_name}_equations.json"
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Terminal momentum per stellar mass from TRINITY parameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python terminal_momentum.py -F /path/to/sweep_output
  python terminal_momentum.py -F /path/to/sweep_output --decompose
  python terminal_momentum.py -F /path/to/sweep_output --fmt png
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--nCore-ref", type=float, default=1e3,
        help="Reference normalization for nCore [cm^-3] (default: 1e3).",
    )
    parser.add_argument(
        "--mCloud-ref", type=float, default=1e5,
        help="Reference normalization for mCloud [Msun] (default: 1e5).",
    )
    parser.add_argument(
        "--sfe-ref", type=float, default=0.01,
        help="Reference normalization for SFE (default: 0.01).",
    )
    parser.add_argument(
        "--sigma-clip", type=float, default=3.0,
        help="Number of sigma for outlier rejection (default: 3.0).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--decompose", action="store_true",
        help="Also fit and plot momentum components (wind, rad, HII, SN).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Maximum time [Myr] to consider in calculations.",
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Enable diagnostic plots and alternative fits "
             "(quadratic log-correction, piecewise power-law).",
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

    # Step 1: collect
    records = collect_data(folder_path, decompose=args.decompose, t_end=args.t_end)
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Step 2: fits
    fit_kwargs = dict(
        nCore_ref=args.nCore_ref,
        mCloud_ref=args.mCloud_ref,
        sfe_ref=args.sfe_ref,
        sigma_clip=args.sigma_clip,
    )

    fits: List[Tuple[str, Optional[Dict]]] = []

    # Fit 1a: p/M_* for expanding runs only
    logger.info("--- Fit: p_fin/M_* (expanding) ---")
    fit_exp = fit_p_mstar(records, outcome_filter=EXPAND, **fit_kwargs)
    fits.append(("p_fin/M_* (expand)", fit_exp))

    # Fit 1b: p_peak/M_* for collapsing runs
    logger.info("--- Fit: p_peak/M_* (collapse) ---")
    fit_col = fit_p_mstar(records, outcome_filter=COLLAPSE, **fit_kwargs)
    fits.append(("p_peak/M_* (collapse)", fit_col))

    # Fit 1c: all runs together
    logger.info("--- Fit: p/M_* (all) ---")
    fit_all = fit_p_mstar(records, outcome_filter=None, **fit_kwargs)
    fits.append(("p/M_* (all)", fit_all))

    # Fit 3: boost factor (if decompose)
    if args.decompose:
        logger.info("--- Fit: momentum boost (expanding) ---")
        fit_boost = fit_p_mstar(
            records, outcome_filter=EXPAND, quantity_key="p_boost", **fit_kwargs)
        fits.append(("p_boost (expand)", fit_boost))

        # Component fits
        # Map from force field to per-M_* key — need to add these to records
        for field, label in [("F_ram_wind", "wind"), ("F_ram_SN", "SN"),
                             ("F_ion_out", "HII"), ("F_rad", "rad")]:
            key = f"p_{label}_per_mstar_kms"
            for r in records:
                comp = r.get("components", {})
                val = comp.get(field, 0.0)
                r[key] = (val / r["M_star"]) * V_AU2KMS if r["M_star"] > 0 else 0.0

            logger.info("--- Fit: p_%s/M_* (expanding) ---", label)
            fit_comp = fit_p_mstar(
                records, outcome_filter=EXPAND, quantity_key=key, **fit_kwargs)
            fits.append((f"p_{label}/M_* (expand)", fit_comp))

    # Step 3: plots
    plot_p_vs_sfe(records, fit_exp, output_dir, args.fmt)
    plot_parity(fit_exp, output_dir, args.fmt)
    plot_parity(fit_col, output_dir, args.fmt, suffix="_collapse")
    plot_p_vs_sigma(records, output_dir, args.fmt)

    if args.decompose:
        plot_momentum_budget(records, output_dir, args.fmt)
        plot_boost(records, output_dir, args.fmt)

    # Step 3b: diagnostic plots and alternative fits (gated by --diagnostics)
    if args.diagnostics:
        # Part A — diagnostic parity plots for each standard fit
        logger.info("--- Diagnostic plots (Part A) ---")
        plot_parity_diagnostic(fit_exp, records, output_dir, args.fmt)
        plot_parity_diagnostic(fit_col, records, output_dir, args.fmt)

        # Part B — quadratic log-correction fit
        logger.info("--- Fit: p_fin/M_* QUAD (expanding) ---")
        fit_exp_quad = fit_p_mstar_quad(
            records, outcome_filter=EXPAND, **fit_kwargs)
        fits.append(("p_fin/M_* quad (expand)", fit_exp_quad))
        plot_parity(fit_exp_quad, output_dir, args.fmt, suffix="_quad")

        logger.info("--- Fit: p_peak/M_* QUAD (collapse) ---")
        fit_col_quad = fit_p_mstar_quad(
            records, outcome_filter=COLLAPSE, **fit_kwargs)
        fits.append(("p_peak/M_* quad (collapse)", fit_col_quad))
        plot_parity(fit_col_quad, output_dir, args.fmt,
                    suffix="_collapse_quad")

        # Part C — piecewise power-law with automatic break-point
        logger.info("--- Fit: p_fin/M_* PIECEWISE (expanding) ---")
        pw_exp = fit_p_mstar_piecewise(
            records, outcome_filter=EXPAND, **fit_kwargs)
        plot_parity_piecewise(pw_exp, output_dir, args.fmt)
        plot_bic_scan(pw_exp, output_dir, args.fmt)

        logger.info("--- Fit: p_peak/M_* PIECEWISE (collapse) ---")
        pw_col = fit_p_mstar_piecewise(
            records, outcome_filter=COLLAPSE, **fit_kwargs)
        plot_parity_piecewise(pw_col, output_dir, args.fmt)
        plot_bic_scan(pw_col, output_dir, args.fmt)

        # Print piecewise summary and add to fits if BIC improves
        for label, pw in [("expand", pw_exp), ("collapse", pw_col)]:
            if pw is None:
                continue
            delta_bic = pw["BIC_single"] - pw["BIC_piecewise"]
            brk_val = pw["break_log_Mcl"]
            print(f"\n--- Piecewise fit ({label}) ---")
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
                fits.append((
                    f"p/M_* pw-low [{brk_str}] ({label})",
                    pw["fit_low"],
                ))
                brk_str = f"Mcl>{10**brk_val:.0f}"
                fits.append((
                    f"p/M_* pw-high [{brk_str}] ({label})",
                    pw["fit_high"],
                ))

    # Step 4: output
    write_results_csv(records, output_dir)
    write_fits_csv(fits, output_dir)
    print_summary(records, fits)

    # Equation JSON for run_all summary
    _write_equation_json(fits, output_dir, "terminal_momentum")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
