#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimum star-formation efficiency for cloud dispersal (collapse criterion).

Physics background
------------------
Whether stellar feedback can disperse a molecular cloud or whether the
cloud re-collapses depends on the balance between outward feedback forces
(thermal pressure, radiation, ionised-gas pressure) and the inward pull
of gravity.  For a given cloud density n_c and mass M_cloud, there exists
a **minimum SFE** ε_min below which feedback is too weak to unbind the
cloud, and the swept-up shell re-collapses.

This threshold is controlled by:

* **Gravitational binding energy**  E_bind = (3/5) G M²/R, which
  increases with cloud mass and surface density Σ = M/(πR²).

* **Feedback energy budget**, set by the stellar mass M_* = ε M_cloud
  and hence the mechanical luminosity L_w ∝ M_* and the ionising
  photon rate Q_i ∝ M_*.

* **Radiative cooling**, which removes thermal energy from the bubble
  and reduces the effective coupling of winds/SNe.

Denser, more massive clouds require a higher SFE to overcome their
deeper gravitational potential.  This motivates the power-law ansatz:

    ε_min ∝ (n_c / n₀)^α (M_cloud / M₀)^β

and the complementary one-parameter form:

    ε_min ∝ (Σ / Σ₀)^δ

where Σ encapsulates the combined dependence on mass and density
through the cloud radius–density relation.

Method
------
1. Each TRINITY run is classified as **expand** (shell escapes the
   cloud), **collapse** (shell re-collapses), or **stalled** (reaches
   maximum time without clear outcome; treated as expand).

2. For every unique (n_c, M_cloud) pair the script identifies the
   SFE boundary between collapse and expansion by bracketing.  The
   midpoint of the highest-collapsing and lowest-expanding SFE is
   taken as ε_min (labelled "exact"); if all runs collapse or all
   expand, the point becomes a lower/upper limit.

3. Power-law fits are performed with sigma-clipping OLS on the
   exact-boundary points.

Connection to observations
--------------------------
Observed giant molecular clouds have integrated SFEs of ε ≈ 1–10 %
and surface densities Σ ≈ 30–300 M☉ pc⁻².  Comparing TRINITY's
ε_min(Σ) to these ranges constrains which clouds can be disrupted by
internal stellar feedback alone, and which require additional
mechanisms (e.g. cloud–cloud collisions, galactic shear).

This analysis is the TRINITY equivalent of the WARPFIELD headline
result presented in Rahner et al. (2019, MNRAS, 483, 2547, Fig. 5).

References
----------
* Rahner, D. et al. (2019), MNRAS, 483, 2547 — WARPFIELD collapse criterion.
* Fall, S. M. et al. (2010), ApJ, 710, L142 — ε_min vs Σ framework.
* Grudić, M. Y. et al. (2018), MNRAS, 475, 3511 — simulated ε_min.

CLI usage
---------
    python collapse_criterion.py -F /path/to/sweep_output
    python collapse_criterion.py -F /path/to/sweep_output --sigma-clip 2.5
    python collapse_criterion.py -F /path/to/sweep_output --fmt png
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
from src._functions.unit_conversions import CGS, CONV

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root, matching other paper_* scripts
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Physical constants and helpers
# ======================================================================

MU_MOL = 1.4          # mean molecular weight per H nucleus (molecular gas)
SIGMA_REF = 100.0      # reference surface density [Msun pc^-2]


def cloud_radius_cgs(mCloud_Msun: float, nCore_cm3: float) -> float:
    """
    Cloud radius for a uniform sphere [cm].

    R_cl = (3 M_cl / (4 pi rho))^{1/3}
    """
    rho = MU_MOL * CGS.m_H * nCore_cm3          # g cm^-3
    M_g = mCloud_Msun / CONV.g2Msun              # g
    return (3.0 * M_g / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def surface_density(mCloud_Msun: float, rCloud_pc: float) -> float:
    """Mean surface density Sigma = M / (pi R^2) [Msun pc^-2]."""
    return mCloud_Msun / (np.pi * rCloud_pc ** 2)


def freefall_time_Myr(nCore_cm3: float) -> float:
    """Free-fall time t_ff = sqrt(3 pi / (32 G rho)) [Myr]."""
    rho = MU_MOL * CGS.m_H * nCore_cm3
    t_ff_s = np.sqrt(3.0 * np.pi / (32.0 * CGS.G * rho))
    return t_ff_s * CONV.s2Myr


def binding_energy(mCloud_Msun: float, rCloud_pc: float) -> float:
    """
    Gravitational binding energy E_bind = (3/5) G M^2 / R [Msun pc^2 Myr^-2].

    Uses TRINITY internal units (Msun, pc, Myr).
    """
    G_au = CONV.G_cgs2au  # pc^3 Msun^-1 Myr^-2
    return 0.6 * G_au * mCloud_Msun ** 2 / rCloud_pc


# ======================================================================
# Outcome classification
# ======================================================================

# Outcome labels
EXPAND = "expand"
COLLAPSE = "collapse"
STALLED = "stalled"


def classify_outcome(data_path: Path, t_end: float = None) -> Optional[Dict]:
    """
    Load a TRINITY run and classify its outcome.

    Parameters
    ----------
    data_path : Path
        Path to dictionary.jsonl for the run.
    t_end : float, optional
        If given, evaluate the outcome at this time [Myr] instead of the
        final snapshot.

    Returns
    -------
    dict or None
        Keys: outcome, rCloud, mCloud_snap.
        None if the run cannot be loaded.
    """
    try:
        output = load_output(data_path)
    except Exception as e:
        logger.warning("Could not load %s: %s", data_path, e)
        return None

    if len(output) < 2:
        logger.warning("Fewer than 2 snapshots in %s — skipping", data_path)
        return None

    first = output[0]

    # Read rCloud from first snapshot (stored in pc)
    rCloud = first.get("rCloud", None)

    # Read mCloud from first snapshot (stored in Msun)
    mCloud_snap = first.get("mCloud", None)

    # If t_end is set, evaluate outcome at t_end instead of the final snapshot
    if t_end is not None:
        t = output.get("t_now")
        v2 = output.get("v2")
        v2 = np.nan_to_num(v2, nan=0.0)
        valid = t <= t_end
        if valid.sum() < 2:
            logger.info("Fewer than 2 snapshots within t_end=%.3f in %s — skip",
                        t_end, data_path.parent.name)
            return None
        # Check if the simulation already ended (collapse/expand) before t_end
        last_full = output[-1]
        is_collapse_full = last_full.get("isCollapse", False)
        is_dissolved_full = last_full.get("isDissolved", False)
        end_reason_full = str(last_full.get("SimulationEndReason", "")).lower()
        t_final = float(t[-1])
        if t_final <= t_end:
            # Simulation ended before t_end — use original outcome
            last = last_full
            is_collapse = is_collapse_full
            is_dissolved = is_dissolved_full
            end_reason = end_reason_full
        else:
            # Simulation still running at t_end — classify by velocity
            i_end = int(np.sum(valid)) - 1
            v_at_end = float(v2[i_end])
            outcome = EXPAND if v_at_end > 0 else COLLAPSE
            return {
                "outcome": outcome,
                "rCloud": rCloud,
                "mCloud_snap": mCloud_snap,
            }
    else:
        last = output[-1]
        is_collapse = last.get("isCollapse", False)
        is_dissolved = last.get("isDissolved", False)
        end_reason = str(last.get("SimulationEndReason", "")).lower()

    # Classify outcome from last snapshot
    if is_dissolved or "dissolved" in end_reason or "large radius" in end_reason:
        outcome = EXPAND
    elif is_collapse or "small radius" in end_reason or "collapsed" in end_reason:
        outcome = COLLAPSE
    elif "stopping time" in end_reason or "max time" in end_reason:
        outcome = STALLED
    else:
        # Fallback: check velocity of last snapshot
        v2 = last.get("v2", 0.0)
        outcome = EXPAND if v2 > 0 else COLLAPSE

    return {
        "outcome": outcome,
        "rCloud": rCloud,
        "mCloud_snap": mCloud_snap,
    }


# ======================================================================
# Data collection
# ======================================================================

def collect_data(folder_path: Path, t_end: float = None) -> List[Dict]:
    """
    Walk a sweep output folder and collect (params, outcome) for every run.

    Returns
    -------
    list of dict
        Each dict has keys: nCore, mCloud, sfe, rCloud, Sigma, t_ff,
        E_bind, outcome, folder.
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

        nCore_val = float(parsed["ndens"])        # cm^-3
        mCloud_val = float(parsed["mCloud"])       # Msun
        sfe_val = int(parsed["sfe"]) / 100.0       # fraction

        info = classify_outcome(data_path, t_end=t_end)
        if info is None:
            continue

        # Cloud radius: prefer snapshot value, fall back to analytic
        rCloud_pc = info["rCloud"]
        if rCloud_pc is None or rCloud_pc <= 0:
            rCloud_cm = cloud_radius_cgs(mCloud_val, nCore_val)
            rCloud_pc = rCloud_cm * CONV.cm2pc

        Sigma = surface_density(mCloud_val, rCloud_pc)
        t_ff = freefall_time_Myr(nCore_val)
        E_b = binding_energy(mCloud_val, rCloud_pc)

        rec = {
            "nCore": nCore_val,
            "mCloud": mCloud_val,
            "sfe": sfe_val,
            "rCloud": rCloud_pc,
            "Sigma": Sigma,
            "t_ff": t_ff,
            "E_bind": E_b,
            "outcome": info["outcome"],
            "folder": folder_name,
        }
        records.append(rec)

    n_expand = sum(1 for r in records if r["outcome"] == EXPAND)
    n_collapse = sum(1 for r in records if r["outcome"] == COLLAPSE)
    n_stalled = sum(1 for r in records if r["outcome"] == STALLED)
    logger.info(
        "Collected %d runs: %d expand, %d collapse, %d stalled",
        len(records), n_expand, n_collapse, n_stalled,
    )
    return records


# ======================================================================
# Determine epsilon_min
# ======================================================================

# Flag for limit types
EXACT = "exact"
LOWER_LIMIT = "lower_limit"   # eps_min > max sampled SFE (all collapse)
UPPER_LIMIT = "upper_limit"   # eps_min < min sampled SFE (all expand)


def find_epsilon_min(records: List[Dict]) -> List[Dict]:
    """
    For each unique (nCore, mCloud) pair, find the minimum SFE for dispersal.

    Parameters
    ----------
    records : list of dict
        Output of :func:`collect_data`.

    Returns
    -------
    list of dict
        Each dict has keys: nCore, mCloud, rCloud, Sigma, t_ff, E_bind,
        eps_min, flag (EXACT / LOWER_LIMIT / UPPER_LIMIT).
    """
    # Group by (nCore, mCloud)
    groups: Dict[Tuple[float, float], List[Dict]] = {}
    for rec in records:
        key = (rec["nCore"], rec["mCloud"])
        groups.setdefault(key, []).append(rec)

    results: List[Dict] = []
    for (nc, mc), runs in sorted(groups.items()):
        # Sort by SFE
        runs_sorted = sorted(runs, key=lambda r: r["sfe"])

        # Classify each SFE (stalled counts as expand for eps_min)
        sfe_vals = np.array([r["sfe"] for r in runs_sorted])
        is_expand = np.array([r["outcome"] in (EXPAND, STALLED) for r in runs_sorted])

        # Take derived quantities from first run in group
        ref = runs_sorted[0]
        base = {
            "nCore": nc,
            "mCloud": mc,
            "rCloud": ref["rCloud"],
            "Sigma": ref["Sigma"],
            "t_ff": ref["t_ff"],
            "E_bind": ref["E_bind"],
        }

        if not is_expand.any():
            # All collapse → eps_min is a lower limit
            base["eps_min"] = sfe_vals.max()
            base["flag"] = LOWER_LIMIT
            logger.info(
                "  (n=%.0e, M=%.0e): all collapse → eps_min > %.3f",
                nc, mc, sfe_vals.max(),
            )
        elif is_expand.all():
            # All expand → eps_min is an upper limit
            base["eps_min"] = sfe_vals.min()
            base["flag"] = UPPER_LIMIT
            logger.info(
                "  (n=%.0e, M=%.0e): all expand → eps_min < %.3f",
                nc, mc, sfe_vals.min(),
            )
        else:
            # Mixed: find the boundary
            # Highest SFE that collapses
            collapse_sfes = sfe_vals[~is_expand]
            expand_sfes = sfe_vals[is_expand]
            highest_collapse = collapse_sfes.max()
            lowest_expand = expand_sfes.min()

            if lowest_expand > highest_collapse:
                # Clean boundary — midpoint
                eps_min = 0.5 * (highest_collapse + lowest_expand)
            else:
                # Noisy boundary — use lowest expanding SFE
                eps_min = lowest_expand

            base["eps_min"] = eps_min
            base["flag"] = EXACT
            logger.info(
                "  (n=%.0e, M=%.0e): eps_min = %.4f (collapse<%.3f, expand>%.3f)",
                nc, mc, eps_min, highest_collapse, lowest_expand,
            )

        results.append(base)

    return results


# ======================================================================
# Fitting
# ======================================================================

def _ols_sigma_clip(
    X: np.ndarray,
    y: np.ndarray,
    sigma_clip: float,
    max_iter: int = 10,
) -> Optional[Dict]:
    """
    OLS with iterative sigma-clipping. Returns coefficients and stats.
    """
    n_total = len(y)
    mask = np.ones(n_total, dtype=bool)

    for _ in range(max_iter):
        X_use = X[mask]
        y_use = y[mask]
        n_use = mask.sum()
        if n_use < X.shape[1]:
            return None

        XtX = X_use.T @ X_use
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return None
        beta = XtX_inv @ (X_use.T @ y_use)

        residuals = y - X @ beta
        rms = np.std(residuals[mask], ddof=X.shape[1])
        if rms == 0:
            break

        new_mask = np.abs(residuals) <= sigma_clip * rms
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
    cov = s2 * XtX_inv
    unc = np.sqrt(np.diag(cov))

    return {
        "beta": beta,
        "unc": unc,
        "R2": R2,
        "rms_dex": rms_dex,
        "n_used": n_used,
        "n_rejected": n_total - n_used,
        "mask": mask,
        "y_pred": y_pred,
    }


def fit_eps_min_nM(
    eps_data: List[Dict],
    nCore_ref: float,
    mCloud_ref: float,
    sigma_clip: float,
) -> Optional[Dict]:
    """
    Fit: log10(eps_min) = log10(A) + alpha*log10(nCore/n0) + beta*log10(mCloud/M0).

    Only uses EXACT data points.
    """
    pts = [d for d in eps_data if d["flag"] == EXACT]
    if len(pts) < 2:
        logger.warning("Too few exact eps_min points (%d) for nM fit", len(pts))
        return None

    nC = np.array([d["nCore"] for d in pts])
    mC = np.array([d["mCloud"] for d in pts])
    eps = np.array([d["eps_min"] for d in pts])
    log_eps = np.log10(eps)

    # Check which parameters vary
    cols = [np.ones(len(pts))]
    names = ["intercept"]
    if len(np.unique(nC)) >= 2:
        cols.append(np.log10(nC / nCore_ref))
        names.append("nCore")
    if len(np.unique(mC)) >= 2:
        cols.append(np.log10(mC / mCloud_ref))
        names.append("mCloud")

    X = np.column_stack(cols)
    result = _ols_sigma_clip(X, log_eps, sigma_clip)
    if result is None:
        return None

    result["param_names"] = names
    result["nCore"] = nC
    result["mCloud"] = mC
    result["eps_actual"] = eps
    result["eps_predicted"] = 10.0 ** result["y_pred"]
    result["refs"] = {"nCore": nCore_ref, "mCloud": mCloud_ref}
    return result


def fit_eps_min_sigma(
    eps_data: List[Dict],
    sigma_clip: float,
) -> Optional[Dict]:
    """
    Fit: log10(eps_min) = log10(A') + delta*log10(Sigma/Sigma_ref).

    Only uses EXACT data points.
    """
    pts = [d for d in eps_data if d["flag"] == EXACT]
    if len(pts) < 2:
        logger.warning("Too few exact eps_min points (%d) for Sigma fit", len(pts))
        return None

    Sigma = np.array([d["Sigma"] for d in pts])
    eps = np.array([d["eps_min"] for d in pts])
    log_eps = np.log10(eps)

    X = np.column_stack([np.ones(len(pts)), np.log10(Sigma / SIGMA_REF)])
    result = _ols_sigma_clip(X, log_eps, sigma_clip)
    if result is None:
        return None

    result["param_names"] = ["intercept", "Sigma"]
    result["Sigma"] = Sigma
    result["eps_actual"] = eps
    result["eps_predicted"] = 10.0 ** result["y_pred"]
    return result


# ======================================================================
# Plotting
# ======================================================================

_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

OUTCOME_COLORS = {
    EXPAND: "C0",
    COLLAPSE: "C3",
    STALLED: "0.55",
}
OUTCOME_LABELS = {
    EXPAND: "Dispersal",
    COLLAPSE: "Collapse",
    STALLED: "Stalled",
}


def plot_phase_diagram(
    records: List[Dict],
    eps_data: List[Dict],
    fit_nM: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 1: Phase diagram in (mCloud, SFE) space, one panel per nCore."""
    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = len(unique_nCore)

    fig, axes = plt.subplots(
        1, max(n_panels, 1),
        figsize=(5.0 * max(n_panels, 1), 4.5),
        squeeze=False,
        dpi=150,
    )
    axes = axes.ravel()

    for i, nc in enumerate(unique_nCore):
        ax = axes[i]
        subset = [r for r in records if r["nCore"] == nc]

        for outcome in [COLLAPSE, STALLED, EXPAND]:
            pts = [r for r in subset if r["outcome"] == outcome]
            if not pts:
                continue
            ax.scatter(
                [r["mCloud"] for r in pts],
                [r["sfe"] for r in pts],
                c=OUTCOME_COLORS[outcome],
                marker="o",
                s=40,
                edgecolors="k",
                linewidths=0.3,
                label=OUTCOME_LABELS[outcome],
                zorder=5,
            )

        # Overlay eps_min boundary from data
        eps_subset = [
            d for d in eps_data
            if d["nCore"] == nc and d["flag"] == EXACT
        ]
        if eps_subset:
            eps_sorted = sorted(eps_subset, key=lambda d: d["mCloud"])
            ax.plot(
                [d["mCloud"] for d in eps_sorted],
                [d["eps_min"] for d in eps_sorted],
                "k-",
                lw=1.8,
                alpha=0.7,
                label=r"$\varepsilon_{\min}$ boundary",
                zorder=6,
            )

        # Overlay constant Sigma contours
        mCloud_range = sorted(set(r["mCloud"] for r in subset))
        if len(mCloud_range) >= 2:
            m_arr = np.geomspace(min(mCloud_range) * 0.8, max(mCloud_range) * 1.2, 100)
            for Sig_val in [30, 100, 300, 1000]:
                # Sigma = M / (pi R^2); R = (3M / (4pi rho))^(1/3)
                # For a given nc: rho = mu * m_H * nc
                rho_cgs = MU_MOL * CGS.m_H * nc
                R_cm = (3.0 * m_arr / CONV.g2Msun / (4.0 * np.pi * rho_cgs)) ** (1.0 / 3.0)
                R_pc = R_cm * CONV.cm2pc
                Sig_arr = m_arr / (np.pi * R_pc ** 2)
                # These are actual Sigma values for each mCloud at this nCore.
                # We want to show where Sigma = Sig_val, which is just a
                # horizontal/vertical reference — but Sigma depends on mCloud
                # only, so the contour is a single mCloud value.
                # Find the mCloud where Sigma = Sig_val via interpolation.
                if Sig_arr.min() < Sig_val < Sig_arr.max():
                    m_cross = np.interp(Sig_val, Sig_arr, m_arr)
                    ax.axvline(
                        m_cross,
                        color="0.6",
                        ls=":",
                        lw=0.8,
                        alpha=0.6,
                        zorder=1,
                    )
                    ax.text(
                        m_cross, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5,
                        rf"$\Sigma={Sig_val}$",
                        fontsize=6,
                        color="0.5",
                        ha="center",
                        va="bottom",
                        rotation=90,
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm cloud}$ [$M_\odot$]")
        if i == 0:
            ax.set_ylabel(r"SFE ($\varepsilon$)")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    # Hide unused panels
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / f"collapse_criterion_phase_diagram.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_eps_vs_sigma(
    eps_data: List[Dict],
    fit_sigma: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 2: eps_min vs. surface density."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    unique_nCore = sorted(set(d["nCore"] for d in eps_data))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    for nc in unique_nCore:
        marker = nc_to_marker[nc]

        # Exact points
        exact = [d for d in eps_data if d["nCore"] == nc and d["flag"] == EXACT]
        if exact:
            ax.scatter(
                [d["Sigma"] for d in exact],
                [d["eps_min"] for d in exact],
                marker=marker,
                s=50,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
            )

        # Lower limits (arrows up)
        lower = [d for d in eps_data if d["nCore"] == nc and d["flag"] == LOWER_LIMIT]
        if lower:
            ax.scatter(
                [d["Sigma"] for d in lower],
                [d["eps_min"] for d in lower],
                marker=marker,
                s=50,
                facecolors="none",
                edgecolors="k",
                linewidths=0.8,
                zorder=4,
            )
            for d in lower:
                ax.annotate(
                    "", xy=(d["Sigma"], d["eps_min"] * 1.5),
                    xytext=(d["Sigma"], d["eps_min"]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1),
                )

        # Upper limits (arrows down)
        upper = [d for d in eps_data if d["nCore"] == nc and d["flag"] == UPPER_LIMIT]
        if upper:
            ax.scatter(
                [d["Sigma"] for d in upper],
                [d["eps_min"] for d in upper],
                marker=marker,
                s=50,
                facecolors="none",
                edgecolors="k",
                linewidths=0.8,
                zorder=4,
            )
            for d in upper:
                ax.annotate(
                    "", xy=(d["Sigma"], d["eps_min"] / 1.5),
                    xytext=(d["Sigma"], d["eps_min"]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1),
                )

    # Overlay fit line
    if fit_sigma is not None:
        beta = fit_sigma["beta"]
        Sigma_plot = np.geomspace(1, 1e5, 200)
        log_eps_pred = beta[0] + beta[1] * np.log10(Sigma_plot / SIGMA_REF)
        ax.plot(
            Sigma_plot,
            10.0 ** log_eps_pred,
            "k--",
            lw=1.5,
            alpha=0.7,
            label="Power-law fit",
        )
        # Annotation
        A_sig = 10.0 ** beta[0]
        delta = beta[1]
        delta_unc = fit_sigma["unc"][1]
        ann = (
            rf"$\varepsilon_{{\min}} \approx {A_sig:.3g}"
            rf"\,(\Sigma / {SIGMA_REF:.0f})^{{{delta:+.2f} \pm {delta_unc:.2f}}}$"
            + "\n"
            + rf"$R^2 = {fit_sigma['R2']:.3f}$, "
            + rf"RMS $= {fit_sigma['rms_dex']:.3f}$ dex"
        )
        ax.text(
            0.04, 0.04, ann,
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

    # Observed GMC surface densities: Sigma ~ 30-300 Msun/pc^2
    ax.axvspan(30, 300, color="green", alpha=0.08, zorder=0,
               label=r"Observed GMCs ($\Sigma \sim 30$-$300$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Sigma_{\rm cl}$ [$M_\odot\,$pc$^{-2}$]")
    ax.set_ylabel(r"$\varepsilon_{\min}$")
    ax.set_title("Minimum SFE for dispersal", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / f"collapse_criterion_epsilon_min_vs_sigma.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_parity(
    fit_nM: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Optional[Path]:
    """Figure 3: Parity plot for eps_min(nCore, mCloud) fit."""
    if fit_nM is None:
        return None

    eps_act = fit_nM["eps_actual"]
    eps_pred = fit_nM["eps_predicted"]
    mask = fit_nM["mask"]
    mCloud = fit_nM["mCloud"]
    nCore = fit_nM["nCore"]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    log_mCloud = np.log10(mCloud)
    unique_nCore = sorted(set(nCore))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    all_vals = np.concatenate([eps_act, eps_pred])
    lo, hi = all_vals[all_vals > 0].min() * 0.5, all_vals.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

    for nc in unique_nCore:
        nc_mask = nCore == nc
        marker = nc_to_marker[nc]

        sel = nc_mask & mask
        if sel.any():
            ax.scatter(
                eps_act[sel], eps_pred[sel],
                c=log_mCloud[sel],
                marker=marker,
                s=50,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
            )

        sel_out = nc_mask & ~mask
        if sel_out.any():
            ax.scatter(
                eps_act[sel_out], eps_pred[sel_out],
                c=log_mCloud[sel_out],
                marker=marker,
                s=50,
                alpha=0.3,
                edgecolors="grey",
                linewidths=0.8,
                zorder=3,
            )

    # Colourbar
    vmin, vmax = log_mCloud.min(), log_mCloud.max()
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

    ann = (
        rf"$R^2 = {fit_nM['R2']:.3f}$"
        + "\n"
        + rf"RMS $= {fit_nM['rms_dex']:.3f}$ dex"
        + "\n"
        + rf"$N = {fit_nM['n_used']}$ (rejected {fit_nM['n_rejected']})"
    )
    ax.text(
        0.04, 0.96, ann,
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\varepsilon_{\min}$ from TRINITY")
    ax.set_ylabel(r"$\varepsilon_{\min}$ from power-law fit")
    ax.set_title(r"Parity: $\varepsilon_{\min}(n_c, M_{\rm cloud})$", fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / f"collapse_criterion_parity.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


def plot_outcome_fraction(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 4: Expansion fraction vs SFE for each (nCore, mCloud) bin."""
    groups: Dict[Tuple[float, float], List[Dict]] = {}
    for rec in records:
        key = (rec["nCore"], rec["mCloud"])
        groups.setdefault(key, []).append(rec)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    unique_nCore = sorted(set(k[0] for k in groups))
    nc_to_marker = {
        nc: _MARKERS[i % len(_MARKERS)] for i, nc in enumerate(unique_nCore)
    }

    cmap = plt.cm.viridis
    all_mCloud = sorted(set(k[1] for k in groups))
    if len(all_mCloud) > 1:
        log_m_min = np.log10(min(all_mCloud))
        log_m_max = np.log10(max(all_mCloud))
    else:
        log_m_min = np.log10(all_mCloud[0]) - 0.5
        log_m_max = np.log10(all_mCloud[0]) + 0.5
    norm = matplotlib.colors.Normalize(vmin=log_m_min, vmax=log_m_max)

    for (nc, mc), runs in sorted(groups.items()):
        runs_sorted = sorted(runs, key=lambda r: r["sfe"])
        sfe_vals = [r["sfe"] for r in runs_sorted]
        frac_expand = []
        for r in runs_sorted:
            count_expand = sum(
                1 for rr in runs_sorted
                if rr["sfe"] <= r["sfe"] and rr["outcome"] == EXPAND
            )
            count_total = sum(1 for rr in runs_sorted if rr["sfe"] <= r["sfe"])
            frac_expand.append(count_expand / count_total if count_total > 0 else 0)

        # Simpler: for each SFE bin, just binary outcome
        sfe_arr = np.array([r["sfe"] for r in runs_sorted])
        expand_arr = np.array([1.0 if r["outcome"] == EXPAND else 0.0
                               for r in runs_sorted])

        color = cmap(norm(np.log10(mc)))
        marker = nc_to_marker[nc]
        ax.plot(
            sfe_arr, expand_arr,
            marker=marker,
            ms=5,
            color=color,
            lw=1,
            alpha=0.8,
        )

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cloud}\;/\;M_\odot)$")

    ax.set_xscale("log")
    ax.set_xlabel(r"SFE ($\varepsilon$)")
    ax.set_ylabel("Outcome (0=collapse, 1=expand)")
    ax.set_title("Outcome vs. SFE", fontsize=10)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Collapse", "Expand"])

    fig.tight_layout()
    out_path = output_dir / f"collapse_criterion_outcome_fraction.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(eps_data: List[Dict], output_dir: Path) -> Path:
    """Write eps_min results table."""
    csv_path = output_dir / "collapse_criterion_results.csv"
    header = ["nCore", "mCloud", "Sigma", "rCloud", "t_ff", "E_bind",
              "eps_min", "flag"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for d in eps_data:
            writer.writerow([
                f"{d['nCore']:.4e}",
                f"{d['mCloud']:.4e}",
                f"{d['Sigma']:.2f}",
                f"{d['rCloud']:.4f}",
                f"{d['t_ff']:.4f}",
                f"{d['E_bind']:.6e}",
                f"{d['eps_min']:.6f}",
                d["flag"],
            ])
    logger.info("Saved: %s", csv_path)
    return csv_path


def write_fits_csv(
    fit_nM: Optional[Dict],
    fit_sigma: Optional[Dict],
    output_dir: Path,
) -> Path:
    """Write fit coefficients table."""
    csv_path = output_dir / "collapse_criterion_fits.csv"
    header = ["fit_version", "param", "coefficient", "uncertainty", "R2",
              "RMS_dex", "N_used", "N_rejected"]
    rows = []

    if fit_nM is not None:
        for i, name in enumerate(fit_nM["param_names"]):
            rows.append([
                "eps_min(nCore,mCloud)",
                name,
                f"{fit_nM['beta'][i]:.6f}",
                f"{fit_nM['unc'][i]:.6f}",
                f"{fit_nM['R2']:.4f}" if i == 0 else "",
                f"{fit_nM['rms_dex']:.4f}" if i == 0 else "",
                str(fit_nM["n_used"]) if i == 0 else "",
                str(fit_nM["n_rejected"]) if i == 0 else "",
            ])

    if fit_sigma is not None:
        for i, name in enumerate(fit_sigma["param_names"]):
            rows.append([
                "eps_min(Sigma)",
                name,
                f"{fit_sigma['beta'][i]:.6f}",
                f"{fit_sigma['unc'][i]:.6f}",
                f"{fit_sigma['R2']:.4f}" if i == 0 else "",
                f"{fit_sigma['rms_dex']:.4f}" if i == 0 else "",
                str(fit_sigma["n_used"]) if i == 0 else "",
                str(fit_sigma["n_rejected"]) if i == 0 else "",
            ])

    with open(csv_path, "w", newline="") as fh:
        refs = fit_nM["refs"] if fit_nM is not None else {}
        fh.write(f"# Normalizations: nCore_ref={refs.get('nCore',0):.0e} cm^-3"
                 f", mCloud_ref={refs.get('mCloud',0):.0e} Msun"
                 f", Sigma_ref=100 Msun/pc^2\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("Saved: %s", csv_path)
    return csv_path


def print_summary(
    eps_data: List[Dict],
    fit_nM: Optional[Dict],
    fit_sigma: Optional[Dict],
) -> None:
    """Print summary to stdout."""
    print()
    print("=" * 80)
    print("COLLAPSE CRITERION SUMMARY")
    print("=" * 80)

    # eps_min table
    print()
    fmt = "{:<12} {:<12} {:>10} {:>10} {:>12} {:>12}"
    print(fmt.format("nCore", "mCloud", "Sigma", "eps_min", "flag", "t_ff [Myr]"))
    print("-" * 80)
    for d in eps_data:
        print(fmt.format(
            f"{d['nCore']:.1e}",
            f"{d['mCloud']:.1e}",
            f"{d['Sigma']:.1f}",
            f"{d['eps_min']:.4f}",
            d["flag"],
            f"{d['t_ff']:.3f}",
        ))

    # Fit results
    if fit_nM is not None:
        print()
        print("-" * 80)
        print("FIT 1: eps_min(nCore, mCloud)")
        print("-" * 80)
        A_nM = 10.0 ** fit_nM["beta"][0]
        eq_parts = [f"{A_nM:.3g}"]
        ref = fit_nM["refs"]
        for i, name in enumerate(fit_nM["param_names"][1:], start=1):
            b = fit_nM["beta"][i]
            u = fit_nM["unc"][i]
            r = ref.get(name, 1.0)
            eq_parts.append(f"({name}/{r:.0e})^{{{b:+.2f}+/-{u:.2f}}}")
            print(f"  {name}: exponent = {b:+.4f} +/- {u:.4f}")
        print(f"  A = {A_nM:.4g}")
        print(f"  R^2 = {fit_nM['R2']:.4f}")
        print(f"  RMS = {fit_nM['rms_dex']:.4f} dex")
        print(f"  eps_min ~ {' * '.join(eq_parts)}")

    if fit_sigma is not None:
        print()
        print("-" * 80)
        print("FIT 2: eps_min(Sigma)")
        print("-" * 80)
        A_sig = 10.0 ** fit_sigma["beta"][0]
        delta = fit_sigma["beta"][1]
        delta_unc = fit_sigma["unc"][1]
        print(f"  A' = {A_sig:.4g}")
        print(f"  delta = {delta:+.4f} +/- {delta_unc:.4f}")
        print(f"  R^2 = {fit_sigma['R2']:.4f}")
        print(f"  RMS = {fit_sigma['rms_dex']:.4f} dex")
        print(f"  eps_min ~ {A_sig:.3g} * (Sigma/{SIGMA_REF:.0f})^{{{delta:+.2f}}}")

    # Comparison
    if fit_nM is not None and fit_sigma is not None:
        print()
        print("-" * 80)
        r2_nM = fit_nM["R2"]
        r2_sig = fit_sigma["R2"]
        better = "(nCore, mCloud)" if r2_nM > r2_sig else "Sigma"
        print(f"R^2 comparison: (nCore,mCloud) = {r2_nM:.4f}, "
              f"Sigma = {r2_sig:.4f}  →  {better} is better correlated")

    print()
    print("=" * 80)


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
            for k in ("nCore", "mCloud", "sfe", "Sigma"):
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
    fit_nM: Optional[Dict],
    fit_sigma: Optional[Dict],
    output_dir: Path,
) -> Path:
    """Write equation data for the run_all summary PDF."""
    entries = []
    if fit_nM is not None:
        A = 10.0 ** fit_nM["beta"][0]
        refs = fit_nM.get("refs", {})
        names = fit_nM["param_names"]
        exponents = {}
        exponent_unc = {}
        for i, name in enumerate(names[1:], 1):
            exponents[name] = float(fit_nM["beta"][i])
            exponent_unc[name] = float(fit_nM["unc"][i])
        entries.append({
            "script": "collapse_criterion",
            "label": "eps_min(nCore, mCloud)",
            "A": float(A),
            "exponents": exponents,
            "exponent_unc": exponent_unc,
            "refs": {k: float(v) for k, v in refs.items()},
            "R2": float(fit_nM["R2"]),
            "rms_dex": float(fit_nM["rms_dex"]),
            "n_used": int(fit_nM["n_used"]),
            "n_rejected": int(fit_nM.get("n_rejected", 0)),
            "rejected": _extract_rejected(fit_nM),
        })
    if fit_sigma is not None:
        A = 10.0 ** fit_sigma["beta"][0]
        names = fit_sigma["param_names"]
        exponents = {}
        exponent_unc = {}
        for i, name in enumerate(names[1:], 1):
            exponents[name] = float(fit_sigma["beta"][i])
            exponent_unc[name] = float(fit_sigma["unc"][i])
        entries.append({
            "script": "collapse_criterion",
            "label": "eps_min(Sigma)",
            "A": float(A),
            "exponents": exponents,
            "exponent_unc": exponent_unc,
            "refs": {"Sigma": 100.0},
            "R2": float(fit_sigma["R2"]),
            "rms_dex": float(fit_sigma["rms_dex"]),
            "n_used": int(fit_sigma["n_used"]),
            "n_rejected": int(fit_sigma.get("n_rejected", 0)),
            "rejected": _extract_rejected(fit_sigma),
        })
    path = output_dir / "collapse_criterion_equations.json"
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)
    logger.info("Saved: %s", path)
    return path


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collapse criterion and minimum SFE for TRINITY parameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collapse_criterion.py -F /path/to/sweep_output
  python collapse_criterion.py -F /path/to/sweep_output --sigma-clip 2.5
  python collapse_criterion.py -F /path/to/sweep_output --fmt png
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
        "--t-end", type=float, default=None,
        help="Maximum time [Myr] to consider in calculations.",
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
    records = collect_data(folder_path, t_end=args.t_end)
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Step 2: determine eps_min
    logger.info("--- Determining eps_min for each (nCore, mCloud) pair ---")
    eps_data = find_epsilon_min(records)
    if not eps_data:
        logger.error("No eps_min values could be determined.")
        return 1

    # Step 3: fit scaling relations
    logger.info("--- Fitting eps_min(nCore, mCloud) ---")
    fit_nM = fit_eps_min_nM(
        eps_data,
        nCore_ref=args.nCore_ref,
        mCloud_ref=args.mCloud_ref,
        sigma_clip=args.sigma_clip,
    )

    logger.info("--- Fitting eps_min(Sigma) ---")
    fit_sigma = fit_eps_min_sigma(eps_data, sigma_clip=args.sigma_clip)

    # Step 4: figures
    plot_phase_diagram(records, eps_data, fit_nM, output_dir, args.fmt)
    plot_eps_vs_sigma(eps_data, fit_sigma, output_dir, args.fmt)
    plot_parity(fit_nM, output_dir, args.fmt)
    plot_outcome_fraction(records, output_dir, args.fmt)

    # Step 5: CSV and summary
    write_results_csv(eps_data, output_dir)
    write_fits_csv(fit_nM, fit_sigma, output_dir)
    print_summary(eps_data, fit_nM, fit_sigma)

    # Equation JSON for run_all summary
    _write_equation_json(fit_nM, fit_sigma, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
