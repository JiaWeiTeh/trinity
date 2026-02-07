#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity-radius relation and self-similar expansion index from TRINITY.

Analyses the v(R) trajectory of each TRINITY run to extract:
    - Instantaneous power-law index  alpha_local = d(log v)/d(log R)
    - Phase-averaged power-law index alpha_phase for each phase
    - Self-similar constant  eta(t) = v*t / R   (for age estimation)

Produces trajectory galleries, diagnostic plots, and scaling fits.

CLI usage
---------
    python velocity_radius.py -F /path/to/sweep_output
    python velocity_radius.py -F /path/to/sweep_output --R-bins 30 --fmt png

Author: Claude Code
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

    # Only keep expansion portion (R increasing, v > 0)
    expanding = (v_au > 0) & (R > 0)
    if expanding.sum() < MIN_PHASE_PTS:
        logger.debug("Too few expanding points in %s — skipping",
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

        alpha_local = compute_alpha_local(R, v_au, expanding)
        eta = compute_eta(t, R, v_au)
        phase_fits = fit_phase_power_law(R, v_au, phase, expanding)
        eta_at_R = evaluate_eta_at_radii(t, R, eta, expanding)
        eta_at_t = evaluate_eta_at_times(t, eta, expanding)
        eta_phase = phase_averaged_eta(t, eta, phase, expanding)

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
            "v_kms": v_kms,
            "phase": phase,
            "expanding": expanding,
            "alpha_local": alpha_local,
            "eta": eta,
            # Per-phase fits
            "phase_fits": phase_fits,
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
    return result


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
            color = cmap(norm(rec["mCloud"]))

            for idx_seg, grp in _phase_segments(R, v, phase, expanding):
                sel = np.array(idx_seg)
                sel = sel[expanding[sel] & (R[sel] > 0) & (v[sel] > 0)]
                if len(sel) < 2:
                    continue
                ls = PHASE_LS.get(grp, "-")
                ax.plot(R[sel], v[sel], color=color, ls=ls, lw=0.9,
                        alpha=0.7, zorder=3)

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
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    csv_path = output_dir / "velocity_radius_results.csv"
    header = [
        "nCore", "mCloud", "SFE", "outcome",
        "alpha_energy", "alpha_energy_unc",
        "alpha_transition", "alpha_transition_unc",
        "alpha_momentum", "alpha_momentum_unc",
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
    for phase_name, fit in alpha_fits.items():
        if fit is None:
            rows.append([phase_name, "N/A"] + [""] * 8)
            continue
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
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(
    records: List[Dict],
    alpha_fits: Dict[str, Optional[Dict]],
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

    print()
    print("=" * 90)


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

    # Step 3: figures
    plot_trajectories(records, output_dir, args.fmt)
    plot_alpha_local(records, output_dir, args.fmt)
    plot_eta_evolution(records, output_dir, args.fmt)
    plot_alpha_phase(records, alpha_fits, output_dir, args.fmt)
    plot_eta_at_radii(records, output_dir, args.fmt)

    # Step 4: output
    write_results_csv(records, output_dir)
    write_fits_csv(alpha_fits, output_dir)
    print_summary(records, alpha_fits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
