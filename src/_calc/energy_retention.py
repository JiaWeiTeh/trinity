#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy retention fraction (heating efficiency) from TRINITY parameter sweeps.

Computes the energy retention fraction:

    xi(t) = E_b(t) / integral_0^t L_w dt'

where E_b is the bubble thermal energy and L_w is the cumulative
mechanical luminosity input (winds + SNe).  Also tracks the full energy
budget: cooling losses, PdV work on the shell, and leakage.

Compares to the Weaver (1977) adiabatic limit (xi ~ 0.77) and the
momentum-driven limit (xi -> 0).  Fits power-law dependences of xi
on cloud parameters for use as sub-grid coupling efficiencies.

CLI usage
---------
    python energy_retention.py -F /path/to/sweep_output
    python energy_retention.py -F /path/to/sweep_output --fmt png

Author: Claude Code
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

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    parse_simulation_params,
)
from src._plots.plot_markers import find_phase_transitions
from src._functions.unit_conversions import CGS, CONV, INV_CONV

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Constants
# ======================================================================

MU_MOL = 1.4                   # mean molecular weight
V_AU2KMS = INV_CONV.v_au2kms   # pc/Myr -> km/s

# Weaver (1977) adiabatic retention for gamma=5/3
XI_WEAVER = 5.0 / 13.0 * (11.0 / 5.0)  # = 11/13 ≈ 0.846 for full
# More commonly quoted value from detailed models
XI_ADIABATIC = 0.77

# Floor value for log-space operations
XI_FLOOR = 1e-10

# Outcome labels
EXPAND = "expand"
COLLAPSE = "collapse"
STALLED = "stalled"

# Phase grouping
ENERGY_PHASES = {"energy", "implicit"}


# ======================================================================
# Helpers
# ======================================================================

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with result[0]=0."""
    dx = np.diff(x)
    incr = 0.5 * (y[1:] + y[:-1]) * dx
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(incr)
    return out


def _cloud_radius_pc(mCloud_Msun: float, nCore_cm3: float) -> float:
    """Cloud radius for a uniform sphere [pc]."""
    rho_cgs = MU_MOL * CGS.m_H * nCore_cm3
    M_g = mCloud_Msun / CONV.g2Msun
    R_cm = (3.0 * M_g / (4.0 * np.pi * rho_cgs)) ** (1.0 / 3.0)
    return R_cm * CONV.cm2pc


def _surface_density(mCloud: float, rCloud: float) -> float:
    """Sigma = M / (pi R^2) [Msun/pc^2]."""
    return mCloud / (np.pi * rCloud ** 2)


def _freefall_time_Myr(nCore_cm3: float) -> float:
    """Free-fall time [Myr]."""
    rho = MU_MOL * CGS.m_H * nCore_cm3
    t_ff_s = np.sqrt(3.0 * np.pi / (32.0 * CGS.G * rho))
    return t_ff_s * CONV.s2Myr


def _analytic_tcool_Myr(nCore_cm3: float, Lmech_avg_cgs: float,
                        Z_Zsun: float = 1.0) -> float:
    """
    Mac Low & McCray (1988) / Rahner+ (2019) analytic cooling time [Myr].

    t_cool = 16 Myr * (Z/Z_sun)^{-35/22} * n^{-8/11} * L_38^{3/11}

    Parameters
    ----------
    nCore_cm3 : float
        Core number density [cm^-3].
    Lmech_avg_cgs : float
        Time-averaged mechanical luminosity [erg/s].
    Z_Zsun : float
        Metallicity in solar units (default: 1.0).
    """
    L_38 = Lmech_avg_cgs / 1e38
    if L_38 <= 0:
        return np.nan
    return 16.0 * Z_Zsun ** (-35.0 / 22.0) * nCore_cm3 ** (-8.0 / 11.0) * L_38 ** (3.0 / 11.0)


# ======================================================================
# Data extraction
# ======================================================================

def extract_run(data_path: Path) -> Optional[Dict]:
    """
    Load one TRINITY run and extract energy retention data.

    Returns
    -------
    dict or None
        Keys: t, Eb, E_w_cum, xi, xi_peak, xi_1Myr, xi_3Myr, xi_trans,
        xi_disp, xi_avg, t_half, E_cool_cum, E_leak_cum, E_pdV_cum,
        f_cool_disp, f_leak_disp, f_pdV_disp, f_xi_disp, outcome,
        rCloud, mCloud_snap, phase, R2, Lmech_avg_cgs, ...
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
    Eb = output.get("Eb")                        # Msun pc^2 Myr^-2
    R2 = output.get("R2")                        # pc
    v2 = output.get("v2")                        # pc/Myr
    Pb = output.get("Pb")                        # Msun/pc/Myr^2
    Lmech_total = output.get("Lmech_total")      # Msun pc^2 Myr^-3
    phase = np.array(output.get("current_phase", as_array=False))

    # Cooling and leakage (may not be tracked in all runs)
    try:
        bubble_LTotal = output.get("bubble_LTotal")
    except Exception:
        bubble_LTotal = None
    try:
        bubble_Leak = output.get("bubble_Leak")
    except Exception:
        bubble_Leak = None

    # Replace NaN
    Eb = np.nan_to_num(Eb, nan=0.0)
    R2 = np.nan_to_num(R2, nan=0.0)
    v2 = np.nan_to_num(v2, nan=0.0)
    Pb = np.nan_to_num(Pb, nan=0.0)
    Lmech_total = np.nan_to_num(Lmech_total, nan=0.0)

    has_cooling = bubble_LTotal is not None
    has_leak = bubble_Leak is not None

    if has_cooling:
        bubble_LTotal = np.nan_to_num(bubble_LTotal, nan=0.0)
    if has_leak:
        bubble_Leak = np.nan_to_num(bubble_Leak, nan=0.0)

    first = output[0]
    last = output[-1]

    rCloud = first.get("rCloud", None)
    mCloud_snap = first.get("mCloud", None)

    # Outcome classification
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

    # --- Cumulative integrals ---
    E_w_cum = _cumtrapz(Lmech_total, t)                     # cumulative wind energy
    E_cool_cum = _cumtrapz(bubble_LTotal, t) if has_cooling else np.zeros_like(t)
    E_leak_cum = _cumtrapz(bubble_Leak, t) if has_leak else np.zeros_like(t)

    # PdV work: dE_pdV/dt = 4 pi R^2 Pb v
    L_pdV = 4.0 * np.pi * R2 ** 2 * Pb * v2                 # Msun pc^2 Myr^-3
    L_pdV = np.maximum(L_pdV, 0.0)  # only count expansion work
    E_pdV_cum = _cumtrapz(L_pdV, t)

    # --- Energy retention fraction ---
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(E_w_cum > 0, Eb / E_w_cum, np.nan)
    xi = np.clip(xi, 0, None)

    # --- Phase transitions ---
    try:
        transitions = find_phase_transitions(t, phase)
        t_transition_list = transitions.get("t_transition", [])
        t_transition = t_transition_list[0] if t_transition_list else np.nan
    except Exception:
        t_transition = np.nan

    # --- Characteristic xi values ---

    # xi_peak: max during energy phase
    energy_mask = np.isin(phase, list(ENERGY_PHASES))
    valid = energy_mask & np.isfinite(xi) & (E_w_cum > 0)
    xi_peak = float(np.nanmax(xi[valid])) if valid.any() else np.nan

    # xi at specific times
    def _xi_at_time(t_target):
        if t.min() > t_target or t.max() < t_target:
            return np.nan
        return float(np.interp(t_target, t, xi))

    xi_1Myr = _xi_at_time(1.0)
    xi_3Myr = _xi_at_time(3.0)

    # xi_trans: at start of transition phase
    if np.isfinite(t_transition) and t_transition > 0:
        xi_trans = _xi_at_time(t_transition)
    else:
        xi_trans = np.nan

    # xi_disp: at dispersal (expanding) or peak R (collapsing)
    if outcome == EXPAND:
        xi_disp = float(xi[-1]) if np.isfinite(xi[-1]) else np.nan
    else:
        i_peak = int(np.argmax(R2))
        xi_disp = float(xi[i_peak]) if np.isfinite(xi[i_peak]) else np.nan

    # xi_avg: time-weighted average
    valid_xi = np.isfinite(xi) & (t > 0)
    if valid_xi.sum() >= 2:
        dt = np.diff(t[valid_xi])
        xi_mid = 0.5 * (xi[valid_xi][1:] + xi[valid_xi][:-1])
        xi_avg = float(np.sum(xi_mid * dt) / np.sum(dt))
    else:
        xi_avg = np.nan

    # t_half: when xi drops below 0.5 * xi_peak
    t_half = np.nan
    if np.isfinite(xi_peak) and xi_peak > 0:
        threshold = 0.5 * xi_peak
        below = xi < threshold
        # Only look after xi_peak is reached
        i_peak_xi = int(np.nanargmax(xi)) if np.any(np.isfinite(xi)) else 0
        below[:i_peak_xi + 1] = False
        if below.any():
            t_half = float(t[np.argmax(below)])

    # --- Energy budget fractions at dispersal ---
    if outcome == EXPAND:
        i_disp = len(t) - 1
    else:
        i_disp = int(np.argmax(R2))

    E_w_at_disp = E_w_cum[i_disp] if E_w_cum[i_disp] > 0 else np.nan

    if np.isfinite(E_w_at_disp) and E_w_at_disp > 0:
        f_xi_disp = float(Eb[i_disp] / E_w_at_disp)
        f_cool_disp = float(E_cool_cum[i_disp] / E_w_at_disp) if has_cooling else np.nan
        f_leak_disp = float(E_leak_cum[i_disp] / E_w_at_disp) if has_leak else np.nan
        f_pdV_disp = float(E_pdV_cum[i_disp] / E_w_at_disp)
    else:
        f_xi_disp = np.nan
        f_cool_disp = np.nan
        f_leak_disp = np.nan
        f_pdV_disp = np.nan

    # Budget closure check
    budget_sum = 0.0
    for f in [f_xi_disp, f_cool_disp, f_leak_disp, f_pdV_disp]:
        if np.isfinite(f):
            budget_sum += f
    budget_residual = abs(1.0 - budget_sum) if budget_sum > 0 else np.nan

    # Average mechanical luminosity in CGS for t_cool comparison
    dt_total = t[-1] - t[0]
    if dt_total > 0:
        Lmech_avg_au = E_w_cum[-1] / dt_total
        Lmech_avg_cgs = Lmech_avg_au * INV_CONV.L_au2cgs
    else:
        Lmech_avg_cgs = np.nan

    return {
        "outcome": outcome,
        "rCloud": rCloud,
        "mCloud_snap": mCloud_snap,
        # Time series (for plotting)
        "t": t,
        "R2": R2,
        "xi": xi,
        "Eb": Eb,
        "E_w_cum": E_w_cum,
        "E_cool_cum": E_cool_cum,
        "E_leak_cum": E_leak_cum,
        "E_pdV_cum": E_pdV_cum,
        "phase": phase,
        "has_cooling": has_cooling,
        "has_leak": has_leak,
        # Characteristic values
        "xi_peak": xi_peak,
        "xi_1Myr": xi_1Myr,
        "xi_3Myr": xi_3Myr,
        "xi_trans": xi_trans,
        "xi_disp": xi_disp,
        "xi_avg": xi_avg,
        "t_half": t_half,
        "t_transition": t_transition,
        # Budget fractions at dispersal
        "f_xi_disp": f_xi_disp,
        "f_cool_disp": f_cool_disp,
        "f_leak_disp": f_leak_disp,
        "f_pdV_disp": f_pdV_disp,
        "budget_residual": budget_residual,
        # For t_cool comparison
        "Lmech_avg_cgs": Lmech_avg_cgs,
    }


def collect_data(folder_path: Path) -> List[Dict]:
    """Walk sweep output and collect energy retention data."""
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

        rCloud = info["rCloud"]
        if rCloud is None or rCloud <= 0:
            rCloud = _cloud_radius_pc(mCloud, nCore)

        Sigma = _surface_density(mCloud, rCloud)
        t_ff = _freefall_time_Myr(nCore)

        # Analytic cooling time
        t_cool = _analytic_tcool_Myr(nCore, info["Lmech_avg_cgs"])

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "rCloud": rCloud,
            "Sigma": Sigma,
            "t_ff": t_ff,
            "outcome": info["outcome"],
            "folder": folder_name,
            # Time series (for plotting)
            "t": info["t"],
            "R2": info["R2"],
            "xi": info["xi"],
            "Eb": info["Eb"],
            "E_w_cum": info["E_w_cum"],
            "E_cool_cum": info["E_cool_cum"],
            "E_leak_cum": info["E_leak_cum"],
            "E_pdV_cum": info["E_pdV_cum"],
            "phase": info["phase"],
            "has_cooling": info["has_cooling"],
            "has_leak": info["has_leak"],
            # Characteristic values
            "xi_peak": info["xi_peak"],
            "xi_1Myr": info["xi_1Myr"],
            "xi_3Myr": info["xi_3Myr"],
            "xi_trans": info["xi_trans"],
            "xi_disp": info["xi_disp"],
            "xi_avg": info["xi_avg"],
            "t_half": info["t_half"],
            "t_transition": info["t_transition"],
            # Budget fractions
            "f_xi_disp": info["f_xi_disp"],
            "f_cool_disp": info["f_cool_disp"],
            "f_leak_disp": info["f_leak_disp"],
            "f_pdV_disp": info["f_pdV_disp"],
            "budget_residual": info["budget_residual"],
            # Analytic comparison
            "t_cool_analytic": t_cool,
            "Lmech_avg_cgs": info["Lmech_avg_cgs"],
        }
        records.append(rec)

    n_exp = sum(1 for r in records if r["outcome"] == EXPAND)
    n_col = sum(1 for r in records if r["outcome"] == COLLAPSE)
    logger.info("Collected %d runs: %d expand, %d collapse",
                len(records), n_exp, n_col)
    return records


# ======================================================================
# Fitting
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
        "y_pred": y_pred,
    }


def fit_scaling(
    records: List[Dict],
    quantity_key: str,
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    outcome_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Fit log10(quantity) = intercept + sum(beta_i * log10(param_i/ref_i))."""
    pts = [
        r for r in records
        if (outcome_filter is None or r["outcome"] == outcome_filter)
        and np.isfinite(r[quantity_key]) and r[quantity_key] > XI_FLOOR
    ]
    if len(pts) < 2:
        logger.warning("Too few points (%d) for fit key=%s filter=%s",
                        len(pts), quantity_key, outcome_filter)
        return None

    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])
    val = np.array([r[quantity_key] for r in pts])
    y = np.log10(np.maximum(val, XI_FLOOR))

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
    result = _ols_sigma_clip(X, y, sigma_clip)
    if result is None:
        return None

    result["param_names"] = names
    result["nCore"] = nC
    result["mCloud"] = mC
    result["sfe"] = sfe
    result["actual"] = val
    result["predicted"] = 10.0 ** result["y_pred"]
    result["refs"] = refs
    result["quantity_key"] = quantity_key
    result["outcome_filter"] = outcome_filter

    A = 10.0 ** result["beta"][0]
    parts = [f"{A:.3g}"]
    for i, name in enumerate(names[1:], 1):
        b = result["beta"][i]
        u = result["unc"][i]
        parts.append(f"({name}/{refs[name]:.0e})^{{{b:+.2f}+/-{u:.2f}}}")
    result["equation_str"] = " * ".join(parts)
    return result


# ======================================================================
# Plotting helpers
# ======================================================================

_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

PHASE_LS = {"energy": "-", "implicit": "-", "transition": "--", "momentum": ":"}
PHASE_GROUP = {
    "energy": "energy", "implicit": "energy",
    "transition": "transition", "momentum": "momentum",
}

OUTCOME_COLORS = {EXPAND: "C0", COLLAPSE: "C3", STALLED: "0.55"}


def _phase_segments(phase):
    """Yield (start, end, group_name) for contiguous phase blocks."""
    grp = np.array([PHASE_GROUP.get(str(p), "energy") for p in phase])
    prev = grp[0]
    start = 0
    for i in range(1, len(grp)):
        if grp[i] != prev:
            yield start, i, prev
            start = i
            prev = grp[i]
    yield start, len(grp), prev


# ======================================================================
# Plotting
# ======================================================================

def plot_xi_evolution(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 1: xi(t) evolution gallery."""
    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = max(len(unique_nCore), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             squeeze=False, dpi=150)
    axes = axes.ravel()

    for pi, nc in enumerate(unique_nCore):
        ax = axes[pi]
        subset = [r for r in records if r["nCore"] == nc]

        all_sfe = sorted(set(r["sfe"] for r in subset))
        if len(all_sfe) > 1:
            sfe_norm = matplotlib.colors.LogNorm(
                vmin=min(all_sfe), vmax=max(all_sfe))
        else:
            sfe_norm = matplotlib.colors.LogNorm(
                vmin=all_sfe[0] * 0.5, vmax=all_sfe[0] * 2.0)
        cmap = plt.cm.viridis

        for rec in subset:
            t = rec["t"]
            xi = rec["xi"]
            phase = rec["phase"]
            color = cmap(sfe_norm(rec["sfe"]))

            for s, e, grp in _phase_segments(phase):
                sel = slice(s, e)
                t_seg = t[sel]
                xi_seg = xi[sel]
                valid = (t_seg > 0) & np.isfinite(xi_seg) & (xi_seg > 0)
                if valid.sum() < 2:
                    continue
                ls = PHASE_LS.get(grp, "-")
                ax.plot(t_seg[valid], xi_seg[valid], color=color,
                        ls=ls, lw=0.8, alpha=0.7, zorder=3)

        # Reference lines
        ax.axhline(XI_ADIABATIC, color="grey", ls="--", lw=1, alpha=0.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0.01 else 10,
                XI_ADIABATIC * 1.05, r"Weaver ($\xi = 0.77$)",
                fontsize=6, color="0.5", ha="right", va="bottom")
        ax.axhline(0.01, color="grey", ls=":", lw=0.8, alpha=0.4)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0.01 else 10,
                0.012, "Momentum-driven", fontsize=6, color="0.5",
                ha="right", va="bottom")

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=sfe_norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02, label=r"SFE ($\varepsilon$)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t [Myr]")
        if pi == 0:
            ax.set_ylabel(r"$\xi(t) = E_b / E_{w,\mathrm{cum}}$")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)
        ax.set_ylim(1e-4, 2)

    for j in range(len(unique_nCore), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"energy_retention_evolution.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_energy_budget(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 2: Energy budget evolution (stacked fractions)."""
    # Select a representative subset: pick up to 9 runs with cooling data
    budget_runs = [r for r in records if r["has_cooling"]
                   and r["outcome"] == EXPAND]
    if not budget_runs:
        budget_runs = [r for r in records if r["has_cooling"]]
    if not budget_runs:
        budget_runs = records[:min(9, len(records))]

    # Subsample if too many
    if len(budget_runs) > 9:
        step = max(1, len(budget_runs) // 9)
        budget_runs = budget_runs[::step][:9]

    n = len(budget_runs)
    ncols = min(n, 3)
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False, dpi=150)
    axes_flat = axes.ravel()

    for idx, rec in enumerate(budget_runs):
        ax = axes_flat[idx]
        t = rec["t"]
        E_w = rec["E_w_cum"]

        # Normalize to E_w_cum
        mask = E_w > 0
        if mask.sum() < 3:
            ax.set_visible(False)
            continue

        t_m = t[mask]
        f_thermal = np.clip(rec["Eb"][mask] / E_w[mask], 0, 1)
        f_cool = np.clip(rec["E_cool_cum"][mask] / E_w[mask], 0, 1) if rec["has_cooling"] else np.zeros_like(t_m)
        f_pdV = np.clip(rec["E_pdV_cum"][mask] / E_w[mask], 0, 1)
        f_leak = np.clip(rec["E_leak_cum"][mask] / E_w[mask], 0, 1) if rec["has_leak"] else np.zeros_like(t_m)

        ax.fill_between(t_m, 0, f_thermal, color="C0", alpha=0.7,
                         label=r"$E_b$ (thermal)")
        ax.fill_between(t_m, f_thermal, f_thermal + f_pdV,
                         color="C2", alpha=0.7, label=r"$PdV$ work")
        ax.fill_between(t_m, f_thermal + f_pdV,
                         f_thermal + f_pdV + f_cool,
                         color="C3", alpha=0.7, label="Cooling")
        if rec["has_leak"]:
            ax.fill_between(t_m, f_thermal + f_pdV + f_cool,
                             f_thermal + f_pdV + f_cool + f_leak,
                             color="C1", alpha=0.7, label="Leakage")

        ax.set_xlim(t_m[1], t_m[-1])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("t [Myr]", fontsize=8)
        ax.set_ylabel("Energy fraction", fontsize=8)
        ax.set_title(
            rf"$M={rec['mCloud']:.0e},\;\varepsilon={rec['sfe']:.2f}$",
            fontsize=8)
        if idx == 0:
            ax.legend(fontsize=6, loc="center right", framealpha=0.8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"energy_retention_budget.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_xi_vs_params(
    records: List[Dict],
    fits: Dict[str, Optional[Dict]],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 3: xi at characteristic times vs parameters."""
    keys = ["xi_1Myr", "xi_3Myr", "xi_trans", "xi_disp"]
    labels = [r"$\xi$ at 1 Myr", r"$\xi$ at 3 Myr",
              r"$\xi$ at transition", r"$\xi$ at dispersal"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    axes = axes.ravel()

    all_sfe = sorted(set(r["sfe"] for r in records))
    if len(all_sfe) > 1:
        sfe_norm = matplotlib.colors.LogNorm(
            vmin=min(all_sfe), vmax=max(all_sfe))
    else:
        sfe_norm = matplotlib.colors.LogNorm(
            vmin=all_sfe[0] * 0.5, vmax=all_sfe[0] * 2.0)
    cmap = plt.cm.viridis

    for i, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[i]
        pts = [r for r in records
               if np.isfinite(r[key]) and r[key] > XI_FLOOR]
        if not pts:
            ax.text(0.5, 0.5, f"No data for {label}",
                    transform=ax.transAxes, ha="center")
            continue

        colors = [cmap(sfe_norm(r["sfe"])) for r in pts]
        ax.scatter(
            [r["Sigma"] for r in pts],
            [r[key] for r in pts],
            c=colors, s=30, edgecolors="k", linewidths=0.3, zorder=5,
        )

        # Reference
        ax.axhline(XI_ADIABATIC, color="grey", ls="--", lw=0.8, alpha=0.5)

        fit = fits.get(key)
        if fit is not None:
            ax.text(
                0.04, 0.04,
                rf"$R^2 = {fit['R2']:.3f}$, "
                + f"RMS = {fit['rms_dex']:.3f} dex",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                          boxstyle="round,pad=0.3"),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\Sigma_{\rm cl}$ [$M_\odot\,$pc$^{-2}$]")
        ax.set_ylabel(r"$\xi$")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(1e-4, 2)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=sfe_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, pad=0.02, shrink=0.6,
                 label=r"SFE ($\varepsilon$)")

    fig.tight_layout()
    out = output_dir / f"energy_retention_vs_params.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_thalf_vs_tcool(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 4: t_half vs analytic t_cool."""
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)

    pts = [r for r in records
           if np.isfinite(r["t_half"]) and r["t_half"] > 0
           and np.isfinite(r["t_cool_analytic"]) and r["t_cool_analytic"] > 0]

    if not pts:
        ax.text(0.5, 0.5, "No runs with valid t_half and t_cool",
                transform=ax.transAxes, ha="center")
        fig.tight_layout()
        out = output_dir / f"energy_retention_thalf_vs_tcool.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    unique_nCore = sorted(set(r["nCore"] for r in pts))
    nc_to_marker = {nc: _MARKERS[i % len(_MARKERS)]
                    for i, nc in enumerate(unique_nCore)}
    cmap = plt.cm.viridis
    all_mc = sorted(set(r["mCloud"] for r in pts))
    if len(all_mc) > 1:
        mc_norm = matplotlib.colors.LogNorm(vmin=min(all_mc), vmax=max(all_mc))
    else:
        mc_norm = matplotlib.colors.LogNorm(
            vmin=all_mc[0] * 0.5, vmax=all_mc[0] * 2.0)

    for nc in unique_nCore:
        sub = [r for r in pts if r["nCore"] == nc]
        colors = [cmap(mc_norm(r["mCloud"])) for r in sub]
        ax.scatter(
            [r["t_cool_analytic"] for r in sub],
            [r["t_half"] for r in sub],
            c=colors, marker=nc_to_marker[nc], s=40,
            edgecolors="k", linewidths=0.3, zorder=5,
            label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
        )

    # 1:1 line
    all_vals = ([r["t_cool_analytic"] for r in pts]
                + [r["t_half"] for r in pts])
    lo, hi = min(all_vals) * 0.3, max(all_vals) * 3.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=mc_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02,
                 label=r"$M_{\rm cloud}$ [$M_\odot$]")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\rm cool}$ (Mac Low \& McCray 1988) [Myr]")
    ax.set_ylabel(r"$t_{1/2}$ (TRINITY) [Myr]")
    ax.set_title(r"Cooling dominance timescale", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.set_aspect("equal")

    fig.tight_layout()
    out = output_dir / f"energy_retention_thalf_vs_tcool.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_xi_vs_radius(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 5: xi vs bubble radius."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    all_sfe = sorted(set(r["sfe"] for r in records))
    if len(all_sfe) > 1:
        sfe_norm = matplotlib.colors.LogNorm(
            vmin=min(all_sfe), vmax=max(all_sfe))
    else:
        sfe_norm = matplotlib.colors.LogNorm(
            vmin=all_sfe[0] * 0.5, vmax=all_sfe[0] * 2.0)
    cmap = plt.cm.viridis

    for rec in records:
        R = rec["R2"]
        xi = rec["xi"]
        sel = (R > 0) & np.isfinite(xi) & (xi > 0)
        if sel.sum() < 3:
            continue
        color = cmap(sfe_norm(rec["sfe"]))
        ax.plot(R[sel], xi[sel], color=color, lw=0.7, alpha=0.6, zorder=3)

    ax.axhline(XI_ADIABATIC, color="grey", ls="--", lw=1, alpha=0.5)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 100,
            XI_ADIABATIC * 1.05, r"Weaver ($\xi = 0.77$)",
            fontsize=6, color="0.5", ha="right", va="bottom")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=sfe_norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label=r"SFE ($\varepsilon$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R [pc]")
    ax.set_ylabel(r"$\xi = E_b / E_{w,\mathrm{cum}}$")
    ax.set_title("Energy retention vs. bubble radius", fontsize=10)
    ax.set_ylim(1e-4, 2)

    fig.tight_layout()
    out = output_dir / f"energy_retention_vs_radius.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_parity(
    fit: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 6: Parity plot for xi_disp fit."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    if fit is None:
        ax.text(0.5, 0.5, "No valid fit for xi_disp",
                transform=ax.transAxes, ha="center")
        fig.tight_layout()
        out = output_dir / f"energy_retention_parity.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    actual = fit["actual"]
    predicted = fit["predicted"]
    mask = fit["mask"]
    mCloud = fit["mCloud"]
    log_mC = np.log10(mCloud)
    vmin, vmax = log_mC.min(), log_mC.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    vals = np.concatenate([actual, predicted])
    lo = max(vals[vals > 0].min() * 0.3, XI_FLOOR)
    hi = vals.max() * 3.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    sel = mask
    if sel.any():
        ax.scatter(
            actual[sel], predicted[sel],
            c=log_mC[sel], s=40, edgecolors="k", linewidths=0.3,
            zorder=5, cmap="viridis", vmin=vmin, vmax=vmax,
        )
    sel_out = ~mask
    if sel_out.any():
        ax.scatter(
            actual[sel_out], predicted[sel_out],
            c=log_mC[sel_out], s=40, alpha=0.3,
            edgecolors="grey", linewidths=0.6, zorder=3,
            cmap="viridis", vmin=vmin, vmax=vmax,
        )

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

    ax.text(
        0.04, 0.96,
        rf"$R^2 = {fit['R2']:.3f}$" + "\n"
        + rf"RMS $= {fit['rms_dex']:.3f}$ dex" + "\n"
        + rf"$N = {fit['n_used']}$ (rejected {fit['n_rejected']})",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\xi_{\rm disp}$ from TRINITY")
    ax.set_ylabel(r"$\xi_{\rm disp}$ from power-law fit")
    ax.set_title(r"Parity: $\xi$ at dispersal", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"energy_retention_parity.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    csv_path = output_dir / "energy_retention_results.csv"
    header = [
        "nCore", "mCloud", "SFE", "Sigma", "outcome",
        "xi_peak", "xi_1Myr", "xi_3Myr", "xi_trans", "xi_disp", "xi_avg",
        "t_half_Myr", "t_cool_analytic_Myr",
        "f_thermal_disp", "f_cool_disp", "f_leak_disp", "f_pdV_disp",
        "budget_residual",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            def fv(v, f=".6f"):
                return f"{v:{f}}" if np.isfinite(v) else "N/A"
            writer.writerow([
                f"{r['nCore']:.4e}", f"{r['mCloud']:.4e}",
                f"{r['sfe']:.4f}", f"{r['Sigma']:.2f}", r["outcome"],
                fv(r["xi_peak"]), fv(r["xi_1Myr"]), fv(r["xi_3Myr"]),
                fv(r["xi_trans"]), fv(r["xi_disp"]), fv(r["xi_avg"]),
                fv(r["t_half"], ".4f"), fv(r["t_cool_analytic"], ".4f"),
                fv(r["f_xi_disp"]), fv(r["f_cool_disp"]),
                fv(r["f_leak_disp"]), fv(r["f_pdV_disp"]),
                fv(r["budget_residual"], ".4f"),
            ])
    logger.info("Saved: %s", csv_path)
    return csv_path


def write_fits_csv(
    fits: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
) -> Path:
    csv_path = output_dir / "energy_retention_fits.csv"
    header = ["fit_label", "param", "coefficient", "uncertainty",
              "R2", "RMS_dex", "N_used", "N_rejected"]
    rows = []
    for label, fit in fits:
        if fit is None:
            continue
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
    print("ENERGY RETENTION SUMMARY")
    print("=" * 90)

    # Overall statistics
    for key, label in [("xi_peak", "xi_peak (energy phase max)"),
                       ("xi_1Myr", "xi at 1 Myr"),
                       ("xi_3Myr", "xi at 3 Myr"),
                       ("xi_trans", "xi at transition"),
                       ("xi_disp", "xi at dispersal/peak-R"),
                       ("xi_avg", "xi time-averaged")]:
        vals = [r[key] for r in records
                if np.isfinite(r[key]) and r[key] > XI_FLOOR]
        if vals:
            arr = np.array(vals)
            print(f"\n  {label}:")
            print(f"    median = {np.median(arr):.4f}, "
                  f"range = [{arr.min():.4f}, {arr.max():.4f}], "
                  f"N = {len(arr)}")

    # t_half
    t_half_vals = [r["t_half"] for r in records
                   if np.isfinite(r["t_half"]) and r["t_half"] > 0]
    if t_half_vals:
        arr = np.array(t_half_vals)
        print(f"\n  t_half (cooling dominance):")
        print(f"    median = {np.median(arr):.3f} Myr, "
              f"range = [{arr.min():.3f}, {arr.max():.3f}] Myr")

    # Energy budget at dispersal
    has_budget = [r for r in records
                  if np.isfinite(r["f_cool_disp"])]
    if has_budget:
        print(f"\n  Energy budget at dispersal ({len(has_budget)} runs):")
        for key, label in [("f_xi_disp", "Thermal (xi)"),
                           ("f_cool_disp", "Cooling"),
                           ("f_pdV_disp", "PdV work"),
                           ("f_leak_disp", "Leakage")]:
            vals = [r[key] for r in has_budget
                    if np.isfinite(r[key])]
            if vals:
                arr = np.array(vals)
                print(f"    {label}: median = {np.median(arr):.3f}, "
                      f"range = [{arr.min():.3f}, {arr.max():.3f}]")

    # Fit results
    for label, fit in fits:
        if fit is None:
            continue
        print(f"\n--- {label} ---")
        A = 10.0 ** fit["beta"][0]
        print(f"  A = {A:.4g}")
        for i, name in enumerate(fit["param_names"][1:], 1):
            print(f"  {name}: {fit['beta'][i]:+.4f} +/- {fit['unc'][i]:.4f}")
        print(f"  R^2 = {fit['R2']:.4f}, RMS = {fit['rms_dex']:.4f} dex")
        print(f"  Scaling: {fit['equation_str']}")

    print()
    print("=" * 90)


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Energy retention fraction from TRINITY parameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python energy_retention.py -F /path/to/sweep_output
  python energy_retention.py -F /path/to/sweep_output --fmt png
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

    # Step 1: collect data
    records = collect_data(folder_path)
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

    for key, label in [("xi_1Myr", "xi at 1 Myr"),
                       ("xi_3Myr", "xi at 3 Myr"),
                       ("xi_trans", "xi at transition"),
                       ("xi_disp", "xi at dispersal")]:
        logger.info("--- Fit: %s ---", label)
        f = fit_scaling(records, key, **fit_kwargs)
        fits.append((label, f))

    logger.info("--- Fit: t_half ---")
    fit_thalf = fit_scaling(records, "t_half", **fit_kwargs)
    fits.append(("t_half [Myr]", fit_thalf))

    # Budget fractions at dispersal
    for key, label in [("f_cool_disp", "f_cool at dispersal"),
                       ("f_pdV_disp", "f_pdV at dispersal")]:
        logger.info("--- Fit: %s ---", label)
        f = fit_scaling(records, key, **fit_kwargs)
        fits.append((label, f))

    # Step 3: figures
    plot_xi_evolution(records, output_dir, args.fmt)
    plot_energy_budget(records, output_dir, args.fmt)

    xi_fits = {key: f for (label, f), key in
               zip(fits[:4], ["xi_1Myr", "xi_3Myr", "xi_trans", "xi_disp"])}
    plot_xi_vs_params(records, xi_fits, output_dir, args.fmt)
    plot_thalf_vs_tcool(records, output_dir, args.fmt)
    plot_xi_vs_radius(records, output_dir, args.fmt)

    # Parity for xi_disp
    fit_xi_disp = dict(fits).get("xi at dispersal")
    plot_parity(fit_xi_disp, output_dir, args.fmt)

    # Step 4: output
    write_results_csv(records, output_dir)
    write_fits_csv(fits, output_dir)
    print_summary(records, fits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
