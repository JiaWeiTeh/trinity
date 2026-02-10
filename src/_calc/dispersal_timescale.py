#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud dispersal timescale scaling relations from TRINITY parameter sweeps.

Physics background
------------------
The **dispersal time** t_disp is the time at which the expanding shell
first reaches the cloud boundary R_cloud with positive velocity,
physically marking the moment when feedback has swept the entire cloud
mass and begins releasing gas into the ISM.  For collapsing runs the
analogous quantity is t_collapse, the time at which the shell
decelerates to v = 0 and begins re-collapsing.

These timescales set the **molecular-cloud lifetime** under the
influence of internal stellar feedback — a quantity directly
constrained by observations of nearby galaxies.

Observational comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^
* **Chevance et al. (2020)** measured cloud-scale feedback timescales
  of 1–5 Myr and feedback velocities v_fb = 7–21 km/s across nine
  nearby galaxies using CO–Hα de-correlation.  The feedback velocity
  is defined here as v_fb = R_cloud / t_disp.

* **Rahner et al. (2017)** found with WARPFIELD that collapsing
  clouds require t_collapse ≈ 2–4 t_ff, where t_ff is the cloud
  free-fall time t_ff = √(3π / 32 G ρ).

Derived quantities
------------------
* **t_disp / t_ff** — dispersal time normalised by the free-fall time.
  Values ≫ 1 imply that feedback takes many free-fall times to disrupt
  the cloud; values ∼ 1 indicate rapid disruption within a single
  dynamical time.

* **v_fb = R_cloud / t_disp** [km/s] — feedback velocity.  This is
  the effective speed at which feedback information propagates outward
  through the cloud and is directly comparable to Chevance et al.

* **ε_ff = ε × t_ff / t_disp** — star-formation efficiency per free-
  fall time.  In turbulence-regulated theories of star formation,
  ε_ff ≈ 0.003–0.01 (Krumholz & McKee 2005; Padoan & Nordlund 2011).
  Here it is measured *a posteriori* from the simulation outcome rather
  than assumed.

Fitted power laws
-----------------
Five quantities are fitted as power laws of (n_c, M_cloud, ε):

1. t_disp (expanding runs only)
2. t_disp / t_ff (expanding runs only)
3. t_collapse / t_ff (collapsing runs only)
4. v_fb (expanding runs only)
5. ε_ff (expanding runs only)

All fits use sigma-clipping OLS in log₁₀ space.

References
----------
* Chevance, M. et al. (2020), MNRAS, 493, 2872 — feedback timescales.
* Rahner, D. et al. (2017), MNRAS, 470, 4453 — WARPFIELD collapse times.
* Krumholz, M. R. & McKee, C. F. (2005), ApJ, 630, 250 — ε_ff theory.
* Padoan, P. & Nordlund, Å. (2011), ApJ, 730, 40 — ε_ff in turbulence.

CLI usage
---------
    python dispersal_timescale.py -F /path/to/sweep_output
    python dispersal_timescale.py -F /path/to/sweep_output --fmt png
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

# Output directory: ./fig/ at project root
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Physical constants and helpers
# ======================================================================

MU_MOL = 1.4              # mean molecular weight per H nucleus (molecular gas)
V_AU2KMS = INV_CONV.v_au2kms   # pc/Myr → km/s  (~0.978)

# Outcome labels
EXPAND = "expand"
COLLAPSE = "collapse"
STALLED = "stalled"


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
    """Free-fall time t_ff = sqrt(3 pi / (32 G rho)) [Myr]."""
    rho = MU_MOL * CGS.m_H * nCore_cm3
    t_ff_s = np.sqrt(3.0 * np.pi / (32.0 * CGS.G * rho))
    return t_ff_s * CONV.s2Myr


# ======================================================================
# Data extraction
# ======================================================================

def extract_run(data_path: Path) -> Optional[Dict]:
    """
    Load one TRINITY run and extract timescale information.

    Parameters
    ----------
    data_path : Path
        Path to dictionary.jsonl.

    Returns
    -------
    dict or None
        Keys: outcome, rCloud, mCloud_snap, t_disp, t_collapse, t_stall,
        t_peak_R, R_disp, v_disp_kms, R_max.
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
    R2 = output.get("R2")
    v2 = output.get("v2")         # pc/Myr

    # Replace NaN
    v2 = np.nan_to_num(v2, nan=0.0)
    R2 = np.nan_to_num(R2, nan=0.0)

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

    # --- Timescale extraction ---

    t_disp = np.nan
    R_disp = np.nan
    v_disp_kms = np.nan
    t_collapse = np.nan
    t_stall = np.nan
    t_peak_R = np.nan
    R_max = np.nan

    # Peak radius
    i_peak = int(np.argmax(R2))
    R_max = float(R2[i_peak])
    t_peak_R = float(t[i_peak])

    if outcome == EXPAND:
        # Dispersal time: first time R2 >= rCloud with v2 > 0
        if rCloud is not None and rCloud > 0:
            dispersal_mask = (R2 >= rCloud) & (v2 > 0)
            if dispersal_mask.any():
                i_disp = int(np.argmax(dispersal_mask))
                t_disp = float(t[i_disp])
                R_disp = float(R2[i_disp])
                v_disp_kms = float(v2[i_disp] * V_AU2KMS)
            else:
                # Shell never reaches rCloud but is expanding — use final time
                t_disp = float(t[-1])
                R_disp = float(R2[-1])
                v_disp_kms = float(v2[-1] * V_AU2KMS)
        else:
            # No rCloud stored — use final time as fallback
            t_disp = float(t[-1])
            R_disp = float(R2[-1])
            v_disp_kms = float(v2[-1] * V_AU2KMS)

        # Stall time: if velocity ever crosses zero during expansion
        # (some runs decelerate then re-accelerate before dispersal)
        sign_changes = np.diff(np.sign(v2))
        zero_crossings = np.where(sign_changes != 0)[0]
        if len(zero_crossings) > 0 and t[zero_crossings[0]] < t_disp:
            t_stall = float(t[zero_crossings[0]])

    elif outcome == COLLAPSE:
        # Collapse time: when v2 first goes negative after expansion
        # (shell has stalled and is recollapsing)
        v_negative = v2 < 0
        if v_negative.any():
            i_collapse = int(np.argmax(v_negative))
            t_collapse = float(t[i_collapse])
        else:
            t_collapse = float(t[-1])

        # Stall time = time at peak R (velocity ≈ 0)
        t_stall = t_peak_R

    elif outcome == STALLED:
        # For stalled runs, use peak R as stall time
        t_stall = t_peak_R

    return {
        "outcome": outcome,
        "rCloud": rCloud,
        "mCloud_snap": mCloud_snap,
        "t_disp": t_disp,
        "t_collapse": t_collapse,
        "t_stall": t_stall,
        "t_peak_R": t_peak_R,
        "R_disp": R_disp,
        "v_disp_kms": v_disp_kms,
        "R_max": R_max,
    }


def collect_data(folder_path: Path) -> List[Dict]:
    """
    Walk sweep output and collect timescale data for every run.
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

        info = extract_run(data_path)
        if info is None:
            continue

        rCloud = info["rCloud"]
        if rCloud is None or rCloud <= 0:
            rCloud = _cloud_radius_pc(mCloud, nCore)

        Sigma = _surface_density(mCloud, rCloud)
        t_ff = _freefall_time_Myr(nCore)

        # Derived quantities
        t_disp = info["t_disp"]
        t_collapse = info["t_collapse"]

        t_disp_over_tff = t_disp / t_ff if np.isfinite(t_disp) and t_ff > 0 else np.nan
        t_collapse_over_tff = t_collapse / t_ff if np.isfinite(t_collapse) and t_ff > 0 else np.nan

        # Feedback velocity: v_fb = R_cl / t_disp [km/s]
        if np.isfinite(t_disp) and t_disp > 0:
            v_fb = (rCloud / t_disp) * V_AU2KMS   # rCloud in pc, t_disp in Myr → pc/Myr → km/s
        else:
            v_fb = np.nan

        # SFE per free-fall time: eps_ff = SFE * t_ff / t_disp
        if np.isfinite(t_disp) and t_disp > 0 and t_ff > 0:
            eps_ff = sfe * t_ff / t_disp
        else:
            eps_ff = np.nan

        rec = {
            "nCore": nCore,
            "mCloud": mCloud,
            "sfe": sfe,
            "rCloud": rCloud,
            "Sigma": Sigma,
            "t_ff": t_ff,
            "outcome": info["outcome"],
            "t_disp": t_disp,
            "t_collapse": t_collapse,
            "t_stall": info["t_stall"],
            "t_peak_R": info["t_peak_R"],
            "R_disp": info["R_disp"],
            "v_disp_kms": info["v_disp_kms"],
            "R_max": info["R_max"],
            "t_disp_over_tff": t_disp_over_tff,
            "t_collapse_over_tff": t_collapse_over_tff,
            "v_fb": v_fb,
            "eps_ff": eps_ff,
            "folder": folder_name,
        }
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
# Fitting
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


def fit_scaling(
    records: List[Dict],
    quantity_key: str,
    nCore_ref: float,
    mCloud_ref: float,
    sfe_ref: float,
    sigma_clip: float,
    outcome_filter: Optional[str] = None,
    log_y: bool = True,
) -> Optional[Dict]:
    """
    Fit a power-law scaling for a given quantity.

    Parameters
    ----------
    quantity_key : str
        Key in records to fit (e.g. "t_disp", "v_fb", "eps_ff").
    outcome_filter : str or None
        If set, only include runs with this outcome.
    log_y : bool
        If True, fit in log10 space (power-law).
    """
    pts = [
        r for r in records
        if (outcome_filter is None or r["outcome"] == outcome_filter)
        and np.isfinite(r[quantity_key]) and r[quantity_key] > 0
    ]
    if len(pts) < 2:
        logger.warning("Too few points (%d) for fit (key=%s, filter=%s)",
                        len(pts), quantity_key, outcome_filter)
        return None

    nC = np.array([r["nCore"] for r in pts])
    mC = np.array([r["mCloud"] for r in pts])
    sfe = np.array([r["sfe"] for r in pts])
    val = np.array([r[quantity_key] for r in pts])
    y = np.log10(val) if log_y else val

    # Build design matrix — only include varying parameters
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
    result["predicted"] = 10.0 ** result["y_pred"] if log_y else result["y_pred"]
    result["refs"] = refs
    result["outcome_filter"] = outcome_filter
    result["quantity_key"] = quantity_key
    result["log_y"] = log_y

    # Human-readable equation
    A = 10.0 ** result["beta"][0] if log_y else result["beta"][0]
    parts = [f"{A:.3g}"]
    for i, name in enumerate(names[1:], 1):
        b = result["beta"][i]
        u = result["unc"][i]
        r = refs.get(name, 1.0)
        parts.append(f"({name}/{r:.0e})^{{{b:+.2f}+/-{u:.2f}}}")
    result["equation_str"] = " * ".join(parts)

    return result


# ======================================================================
# Plotting
# ======================================================================

_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

OUTCOME_COLORS = {EXPAND: "C0", COLLAPSE: "C3", STALLED: "0.55"}
OUTCOME_LABELS = {EXPAND: "Dispersal", COLLAPSE: "Collapse", STALLED: "Stalled"}
OUTCOME_MARKERS = {EXPAND: "o", COLLAPSE: "x", STALLED: "+"}


def plot_t_disp_vs_sfe(
    records: List[Dict],
    fit_disp: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 1: t_disp vs SFE."""
    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = max(len(unique_nCore), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5),
                             squeeze=False, dpi=150)
    axes = axes.ravel()

    for pi, nc in enumerate(unique_nCore):
        ax = axes[pi]
        subset = [r for r in records if r["nCore"] == nc]

        unique_mCloud = sorted(set(r["mCloud"] for r in subset))
        cmap = plt.cm.viridis
        if len(unique_mCloud) > 1:
            norm = matplotlib.colors.LogNorm(
                vmin=min(unique_mCloud), vmax=max(unique_mCloud))
        else:
            norm = matplotlib.colors.LogNorm(
                vmin=unique_mCloud[0] * 0.5, vmax=unique_mCloud[0] * 2.0)

        # Plot expanding runs (t_disp)
        exp_pts = [r for r in subset if r["outcome"] == EXPAND
                   and np.isfinite(r["t_disp"]) and r["t_disp"] > 0]
        if exp_pts:
            colors = [cmap(norm(r["mCloud"])) for r in exp_pts]
            ax.scatter(
                [r["sfe"] for r in exp_pts],
                [r["t_disp"] for r in exp_pts],
                c=colors, marker="o", s=40, edgecolors="k",
                linewidths=0.3, zorder=5, label="Dispersal",
            )

        # Plot collapsing runs (t_collapse)
        col_pts = [r for r in subset if r["outcome"] == COLLAPSE
                   and np.isfinite(r["t_collapse"]) and r["t_collapse"] > 0]
        if col_pts:
            colors = [cmap(norm(r["mCloud"])) for r in col_pts]
            ax.scatter(
                [r["sfe"] for r in col_pts],
                [r["t_collapse"] for r in col_pts],
                c=colors, marker="x", s=40, linewidths=1.2,
                zorder=4, label="Collapse",
            )

        # Chevance+ 2020 band: 1–5 Myr
        ax.axhspan(1, 5, color="green", alpha=0.08, zorder=0,
                   label="Chevance+ 2020 (1–5 Myr)")

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02,
                     label=r"$M_{\rm cloud}$ [$M_\odot$]")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"SFE ($\varepsilon$)")
        if pi == 0:
            ax.set_ylabel(r"$t$ [Myr]")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)
        ax.legend(fontsize=7, loc="best", framealpha=0.8)

    for j in range(len(unique_nCore), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_vs_sfe.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_t_disp_normalized(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 2: t_disp / t_ff as a function of parameters, separated by density."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

    exp_pts = [r for r in records if r["outcome"] == EXPAND
               and np.isfinite(r["t_disp_over_tff"]) and r["t_disp_over_tff"] > 0]
    col_pts = [r for r in records if r["outcome"] == COLLAPSE
               and np.isfinite(r["t_collapse_over_tff"])
               and r["t_collapse_over_tff"] > 0]

    # Color by SFE
    all_sfe = [r["sfe"] for r in records if r["sfe"] > 0]
    if len(set(all_sfe)) > 1:
        sfe_norm = matplotlib.colors.LogNorm(vmin=min(all_sfe), vmax=max(all_sfe))
    else:
        sfe_norm = matplotlib.colors.LogNorm(vmin=all_sfe[0] * 0.5,
                                              vmax=all_sfe[0] * 2.0)
    cmap = plt.cm.coolwarm

    # Separate by density — different markers per nCore
    unique_nCore = sorted(set(r["nCore"] for r in records))
    nc_to_marker = {nc: _MARKERS[i % len(_MARKERS)]
                    for i, nc in enumerate(unique_nCore)}

    # Plot expanding runs grouped by nCore
    for nc in unique_nCore:
        sub = [r for r in exp_pts if r["nCore"] == nc]
        if not sub:
            continue
        colors = [cmap(sfe_norm(r["sfe"])) for r in sub]
        nlog = int(np.log10(nc))
        ax.scatter(
            [r["Sigma"] for r in sub],
            [r["t_disp_over_tff"] for r in sub],
            c=colors, marker=nc_to_marker[nc], s=40, edgecolors="k",
            linewidths=0.3, zorder=5,
            label=rf"Dispersal ($n_c = 10^{{{nlog}}}$)",
        )

    # Plot collapsing runs grouped by nCore
    for nc in unique_nCore:
        sub = [r for r in col_pts if r["nCore"] == nc]
        if not sub:
            continue
        colors = [cmap(sfe_norm(r["sfe"])) for r in sub]
        nlog = int(np.log10(nc))
        ax.scatter(
            [r["Sigma"] for r in sub],
            [r["t_collapse_over_tff"] for r in sub],
            c=colors, marker=nc_to_marker[nc], s=40, linewidths=1.2,
            facecolors="none", zorder=4,
            label=rf"Collapse ($n_c = 10^{{{nlog}}}$)",
        )

    # Reference lines
    for val, label in [(1, r"$t/t_{\rm ff} = 1$"),
                       (2, r"$t/t_{\rm ff} = 2$"),
                       (4, r"$t/t_{\rm ff} = 4$")]:
        ax.axhline(val, color="grey", ls=":", lw=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 10 else 1e3,
                val * 1.05, label, fontsize=6, color="0.5",
                ha="right", va="bottom")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=sfe_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"SFE ($\varepsilon$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Sigma_{\rm cl}$ [$M_\odot\,$pc$^{-2}$]")
    ax.set_ylabel(r"$t / t_{\rm ff}$")
    ax.set_title("Dispersal / collapse time normalized by free-fall time",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_normalized.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_feedback_velocity(
    records: List[Dict],
    fit_vfb: Optional[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 3: Feedback velocity v_fb vs surface density."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    pts = [r for r in records if r["outcome"] == EXPAND
           and np.isfinite(r["v_fb"]) and r["v_fb"] > 0]

    all_sfe = [r["sfe"] for r in pts] if pts else [0.01]
    if len(set(all_sfe)) > 1:
        sfe_norm = matplotlib.colors.LogNorm(vmin=min(all_sfe), vmax=max(all_sfe))
    else:
        sfe_norm = matplotlib.colors.LogNorm(vmin=all_sfe[0] * 0.5,
                                              vmax=all_sfe[0] * 2.0)
    cmap = plt.cm.viridis

    unique_nCore = sorted(set(r["nCore"] for r in pts)) if pts else []
    nc_to_marker = {nc: _MARKERS[i % len(_MARKERS)]
                    for i, nc in enumerate(unique_nCore)}

    for nc in unique_nCore:
        sub = [r for r in pts if r["nCore"] == nc]
        if not sub:
            continue
        colors = [cmap(sfe_norm(r["sfe"])) for r in sub]
        ax.scatter(
            [r["Sigma"] for r in sub],
            [r["v_fb"] for r in sub],
            c=colors, marker=nc_to_marker[nc], s=40,
            edgecolors="k", linewidths=0.3, zorder=5,
            label=rf"$n_c = {nc:.0e}$ cm$^{{-3}}$",
        )

    # Chevance+ 2020 band: 7–21 km/s
    ax.axhspan(7, 21, color="green", alpha=0.08, zorder=0,
               label="Chevance+ 2020 (7–21 km/s)")

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=sfe_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"SFE ($\varepsilon$)")

    # Fit line overlay
    if fit_vfb is not None:
        beta = fit_vfb["beta"]
        names = fit_vfb["param_names"]
        refs = fit_vfb["refs"]
        ann_parts = [f"$A = {10.0 ** beta[0]:.2f}$ km/s"]
        for i, name in enumerate(names[1:], 1):
            ann_parts.append(
                rf"${name}: {beta[i]:+.2f} \pm {fit_vfb['unc'][i]:.2f}$")
        ann_parts.append(rf"$R^2 = {fit_vfb['R2']:.3f}$")
        ax.text(
            0.04, 0.04, "\n".join(ann_parts),
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Sigma_{\rm cl}$ [$M_\odot\,$pc$^{-2}$]")
    ax.set_ylabel(r"$v_{\rm fb} = R_{\rm cl} / t_{\rm disp}$ [km/s]")
    ax.set_title("Feedback velocity", fontsize=10)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_feedback_velocity.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_epsilon_ff(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 4: eps_ff distribution."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    pts = [r for r in records if r["outcome"] == EXPAND
           and np.isfinite(r["eps_ff"]) and r["eps_ff"] > 0]
    if not pts:
        ax.text(0.5, 0.5, "No expanding runs with valid eps_ff",
                transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        out = output_dir / f"dispersal_timescale_epsilon_ff.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    vals = np.array([r["eps_ff"] for r in pts])

    # Color by mCloud
    unique_mCloud = sorted(set(r["mCloud"] for r in pts))
    if len(unique_mCloud) > 1:
        mc_norm = matplotlib.colors.LogNorm(
            vmin=min(unique_mCloud), vmax=max(unique_mCloud))
    else:
        mc_norm = matplotlib.colors.LogNorm(
            vmin=unique_mCloud[0] * 0.5, vmax=unique_mCloud[0] * 2.0)
    cmap = plt.cm.viridis

    # Histogram
    log_vals = np.log10(vals)
    bins = np.linspace(log_vals.min() - 0.2, log_vals.max() + 0.2,
                       max(8, len(vals) // 3))
    ax.hist(log_vals, bins=bins, color="C0", alpha=0.7,
            edgecolor="k", linewidth=0.4)

    # Reference lines
    for val, label in [(0.01, r"$\varepsilon_{\rm ff} = 0.01$"),
                       (0.003, r"$\varepsilon_{\rm ff} = 0.003$")]:
        ax.axvline(np.log10(val), color="grey", ls="--", lw=1, alpha=0.6)
        ax.text(np.log10(val) + 0.05, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 1,
                label, fontsize=7, color="0.5", va="top", rotation=90)

    ax.set_xlabel(r"$\log_{10}(\varepsilon_{\rm ff})$")
    ax.set_ylabel("Count")
    ax.set_title(r"SFE per free-fall time $\varepsilon_{\rm ff} = \varepsilon \times t_{\rm ff} / t_{\rm disp}$",
                 fontsize=9)

    # Annotation
    med = np.median(vals)
    ax.text(0.96, 0.96,
            rf"median $= {med:.4f}$" + "\n"
            + rf"range $= [{vals.min():.4f}, {vals.max():.4f}]$" + "\n"
            + rf"$N = {len(vals)}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"))

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_epsilon_ff.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_parity(
    fits: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 5: Parity plots (one subplot per fit)."""
    valid_fits = [(label, f) for label, f in fits if f is not None]
    n_fits = len(valid_fits)
    if n_fits == 0:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.text(0.5, 0.5, "No valid fits", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        out = output_dir / f"dispersal_timescale_parity.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    ncols = min(n_fits, 3)
    nrows = (n_fits + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.5 * nrows),
                             squeeze=False, dpi=150)
    axes_flat = axes.ravel()

    for idx, (label, fit) in enumerate(valid_fits):
        ax = axes_flat[idx]
        actual = fit["actual"]
        predicted = fit["predicted"]
        mask = fit["mask"]
        mCloud = fit["mCloud"]

        log_mCloud = np.log10(mCloud)
        vmin, vmax = log_mCloud.min(), log_mCloud.max()
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        vals = np.concatenate([actual, predicted])
        lo, hi = vals[vals > 0].min() * 0.5, vals.max() * 2.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="1:1")

        sel = mask
        if sel.any():
            sc = ax.scatter(
                actual[sel], predicted[sel],
                c=log_mCloud[sel], s=30, edgecolors="k",
                linewidths=0.3, zorder=5, cmap="viridis",
                vmin=vmin, vmax=vmax,
            )
        sel_out = ~mask
        if sel_out.any():
            ax.scatter(
                actual[sel_out], predicted[sel_out],
                c=log_mCloud[sel_out], s=30, alpha=0.3,
                edgecolors="grey", linewidths=0.6, zorder=3,
                cmap="viridis", vmin=vmin, vmax=vmax,
            )

        ax.text(
            0.04, 0.96,
            rf"$R^2 = {fit['R2']:.3f}$" + "\n"
            + rf"RMS $= {fit['rms_dex']:.3f}$ dex" + "\n"
            + rf"$N = {fit['n_used']}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("TRINITY")
        ax.set_ylabel("Power-law fit")
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")

    # Hide unused axes
    for j in range(n_fits, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_parity.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


def plot_collapse_map(
    records: List[Dict],
    output_dir: Path,
    fmt: str,
) -> Path:
    """Figure 6: t_collapse / t_ff as color map in (mCloud, SFE) space."""
    col_pts = [r for r in records if r["outcome"] == COLLAPSE
               and np.isfinite(r["t_collapse_over_tff"])
               and r["t_collapse_over_tff"] > 0]

    unique_nCore = sorted(set(r["nCore"] for r in records))
    n_panels = max(len(unique_nCore), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             squeeze=False, dpi=150)
    axes = axes.ravel()

    for pi, nc in enumerate(unique_nCore):
        ax = axes[pi]
        subset = [r for r in records if r["nCore"] == nc]
        col_sub = [r for r in col_pts if r["nCore"] == nc]

        # Plot all runs as background
        for outcome in [EXPAND, STALLED]:
            opts = [r for r in subset if r["outcome"] == outcome]
            if opts:
                ax.scatter(
                    [r["mCloud"] for r in opts],
                    [r["sfe"] for r in opts],
                    c=OUTCOME_COLORS[outcome],
                    marker=OUTCOME_MARKERS[outcome],
                    s=30, alpha=0.4, zorder=3,
                    label=OUTCOME_LABELS[outcome],
                )

        # Collapsing runs: color by t_collapse / t_ff
        if col_sub:
            tc_tff = np.array([r["t_collapse_over_tff"] for r in col_sub])
            sc = ax.scatter(
                [r["mCloud"] for r in col_sub],
                [r["sfe"] for r in col_sub],
                c=tc_tff,
                cmap="plasma",
                marker="s",
                s=50,
                edgecolors="k",
                linewidths=0.4,
                zorder=5,
                label="Collapse",
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(r"$t_{\rm collapse} / t_{\rm ff}$")

        # Reference: Rahner+ 2017 range (2–4 t_ff)
        ax.text(
            0.96, 0.04,
            "Rahner+ 2017:\n" + r"$t_{\rm coll} = 2$–$4\,t_{\rm ff}$",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm cloud}$ [$M_\odot$]")
        if pi == 0:
            ax.set_ylabel(r"SFE ($\varepsilon$)")
        ax.set_title(rf"$n_c = {nc:.0e}$ cm$^{{-3}}$", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    for j in range(len(unique_nCore), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out = output_dir / f"dispersal_timescale_collapse_map.{fmt}"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
    return out


# ======================================================================
# Summary output
# ======================================================================

def write_results_csv(records: List[Dict], output_dir: Path) -> Path:
    """Write per-run results table."""
    csv_path = output_dir / "dispersal_timescale_results.csv"
    header = [
        "nCore", "mCloud", "SFE", "Sigma", "rCloud", "t_ff", "outcome",
        "t_disp", "t_collapse", "t_stall", "t_peak_R",
        "t_disp_over_tff", "t_collapse_over_tff",
        "R_disp", "v_disp_kms", "R_max",
        "v_fb_kms", "eps_ff",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in records:
            def fmt_val(v, f=".4f"):
                return f"{v:{f}}" if np.isfinite(v) else "N/A"

            writer.writerow([
                f"{r['nCore']:.4e}", f"{r['mCloud']:.4e}",
                f"{r['sfe']:.4f}", f"{r['Sigma']:.2f}",
                f"{r['rCloud']:.4f}", f"{r['t_ff']:.4f}",
                r["outcome"],
                fmt_val(r["t_disp"]), fmt_val(r["t_collapse"]),
                fmt_val(r["t_stall"]), fmt_val(r["t_peak_R"]),
                fmt_val(r["t_disp_over_tff"]),
                fmt_val(r["t_collapse_over_tff"]),
                fmt_val(r["R_disp"]), fmt_val(r["v_disp_kms"]),
                fmt_val(r["R_max"]),
                fmt_val(r["v_fb"]), fmt_val(r["eps_ff"], ".6f"),
            ])

    logger.info("Saved: %s", csv_path)
    return csv_path


def write_fits_csv(
    fits: List[Tuple[str, Optional[Dict]]],
    output_dir: Path,
) -> Path:
    """Write fit coefficients table."""
    csv_path = output_dir / "dispersal_timescale_fits.csv"
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
    """Print summary to stdout."""
    print()
    print("=" * 90)
    print("DISPERSAL TIMESCALE SUMMARY")
    print("=" * 90)

    # Dispersal runs
    exp = [r for r in records if r["outcome"] == EXPAND
           and np.isfinite(r["t_disp"]) and r["t_disp"] > 0]
    if exp:
        td = np.array([r["t_disp"] for r in exp])
        vfb = np.array([r["v_fb"] for r in exp if np.isfinite(r["v_fb"])])
        eff = np.array([r["eps_ff"] for r in exp if np.isfinite(r["eps_ff"])])
        td_tff = np.array([r["t_disp_over_tff"] for r in exp
                           if np.isfinite(r["t_disp_over_tff"])])

        print(f"\nExpanding runs ({len(exp)}):")
        print(f"  t_disp      : median = {np.median(td):.3f} Myr, "
              f"range = [{td.min():.3f}, {td.max():.3f}] Myr")
        if len(td_tff) > 0:
            print(f"  t_disp/t_ff : median = {np.median(td_tff):.3f}, "
                  f"range = [{td_tff.min():.3f}, {td_tff.max():.3f}]")
        if len(vfb) > 0:
            print(f"  v_fb        : median = {np.median(vfb):.1f} km/s, "
                  f"range = [{vfb.min():.1f}, {vfb.max():.1f}] km/s")
            in_chevance = np.sum((vfb >= 7) & (vfb <= 21))
            print(f"                {in_chevance}/{len(vfb)} within Chevance+ 2020 "
                  f"range (7–21 km/s)")
        if len(eff) > 0:
            print(f"  eps_ff      : median = {np.median(eff):.5f}, "
                  f"range = [{eff.min():.5f}, {eff.max():.5f}]")

    # Collapse runs
    col = [r for r in records if r["outcome"] == COLLAPSE
           and np.isfinite(r["t_collapse"]) and r["t_collapse"] > 0]
    if col:
        tc = np.array([r["t_collapse"] for r in col])
        tc_tff = np.array([r["t_collapse_over_tff"] for r in col
                           if np.isfinite(r["t_collapse_over_tff"])])

        print(f"\nCollapsing runs ({len(col)}):")
        print(f"  t_collapse      : median = {np.median(tc):.3f} Myr, "
              f"range = [{tc.min():.3f}, {tc.max():.3f}] Myr")
        if len(tc_tff) > 0:
            print(f"  t_collapse/t_ff : median = {np.median(tc_tff):.3f}, "
                  f"range = [{tc_tff.min():.3f}, {tc_tff.max():.3f}]")
            in_rahner = np.sum((tc_tff >= 2) & (tc_tff <= 4))
            print(f"                    {in_rahner}/{len(tc_tff)} within Rahner+ 2017 "
                  f"range (2–4 t_ff)")

    # Fit results
    for label, fit in fits:
        if fit is None:
            continue
        print(f"\n--- {label} ---")
        A = 10.0 ** fit["beta"][0] if fit.get("log_y", True) else fit["beta"][0]
        print(f"  A = {A:.3g}")
        for i, name in enumerate(fit["param_names"][1:], 1):
            print(f"  {name}: {fit['beta'][i]:+.4f} +/- {fit['unc'][i]:.4f}")
        print(f"  R^2 = {fit['R2']:.4f}, RMS = {fit['rms_dex']:.4f} dex")
        print(f"  Scaling: {fit['equation_str']}")

    print()
    print("=" * 90)


# ======================================================================
# Equation JSON (for run_all summary)
# ======================================================================

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
        description="Cloud dispersal timescale scaling relations from TRINITY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dispersal_timescale.py -F /path/to/sweep_output
  python dispersal_timescale.py -F /path/to/sweep_output --fmt png
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

    # Fit 1: t_disp (expanding only)
    logger.info("--- Fit: t_disp (expanding) ---")
    fit_disp = fit_scaling(records, "t_disp", outcome_filter=EXPAND,
                           **fit_kwargs)
    fits.append(("t_disp [Myr]", fit_disp))

    # Fit 2: t_disp / t_ff (expanding only)
    logger.info("--- Fit: t_disp / t_ff (expanding) ---")
    fit_disp_norm = fit_scaling(records, "t_disp_over_tff",
                                outcome_filter=EXPAND, **fit_kwargs)
    fits.append(("t_disp / t_ff", fit_disp_norm))

    # Fit 3: t_collapse / t_ff (collapsing only)
    logger.info("--- Fit: t_collapse / t_ff (collapsing) ---")
    fit_collapse = fit_scaling(records, "t_collapse_over_tff",
                               outcome_filter=COLLAPSE, **fit_kwargs)
    fits.append(("t_collapse / t_ff", fit_collapse))

    # Fit 4: feedback velocity (expanding only)
    logger.info("--- Fit: v_fb (expanding) ---")
    fit_vfb = fit_scaling(records, "v_fb", outcome_filter=EXPAND,
                          **fit_kwargs)
    fits.append(("v_fb [km/s]", fit_vfb))

    # Fit 5: eps_ff (expanding only)
    logger.info("--- Fit: eps_ff (expanding) ---")
    fit_eff = fit_scaling(records, "eps_ff", outcome_filter=EXPAND,
                          **fit_kwargs)
    fits.append(("eps_ff", fit_eff))

    # Step 3: figures
    plot_t_disp_vs_sfe(records, fit_disp, output_dir, args.fmt)
    plot_t_disp_normalized(records, output_dir, args.fmt)
    plot_feedback_velocity(records, fit_vfb, output_dir, args.fmt)
    plot_epsilon_ff(records, output_dir, args.fmt)
    plot_parity(fits, output_dir, args.fmt)
    plot_collapse_map(records, output_dir, args.fmt)

    # Step 4: output
    write_results_csv(records, output_dir)
    write_fits_csv(fits, output_dir)
    print_summary(records, fits)

    # Equation JSON for run_all summary
    _write_equation_json(fits, output_dir, "dispersal_timescale")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
