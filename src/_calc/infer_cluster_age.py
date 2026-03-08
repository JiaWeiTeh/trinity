#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian inference of bubble age from observed bubble properties.

Physics background
------------------
Given an observed bubble with radius R, expansion velocity v_exp,
and/or density at the shell edge n_edge, this tool recovers the most
likely bubble age t by computing a posterior over the TRINITY simulation
grid.

Each grid model theta_i = (M_cloud, sfe, n_core) predicts R(t), v(t),
and n_edge(R) from the stored trajectory and the cloud density profile.
For every model, the likelihood is evaluated at each snapshot time,
scoring how well R_model(t), v_model(t), and n_edge(R(t)) match the
observations.  A cloud-mass-function prior weights the posterior.

Progressive degeneracy breaking is demonstrated by running inference
with increasing observable sets: (R), (R,v), (R,n), (R,v,n).

How to run
----------
This script requires a TRINITY parameter sweep directory (the same kind
used by ``infer_cluster_mass.py``).

**Required argument:**

  -F / --folder   Path to the sweep output directory tree.

**Specifying observables -- two modes:**

1. *Pre-stored system* -- use ``--system`` to select a known bubble.
   The script ships with the same targets as ``infer_cluster_mass.py``.

2. *Manual observables* -- supply ``--R-obs``, ``--sigma-R``
   (required), plus optionally ``--v-obs``, ``--sigma-v``,
   ``--n-edge-obs``, ``--sigma-n-edge``.

**Outputs (saved to fig/infer_cluster_age/<folder>/ by default):**

  * ``infer_age_posterior_{system}.pdf``  -- overlaid marginal PDFs
  * ``infer_age_Rt_tracks_{system}.pdf`` -- top R(t) tracks
  * Console summary table

Examples
--------
.. code-block:: bash

    python src/_calc/infer_cluster_age.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system RCW120

    python src/_calc/infer_cluster_age.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --R-obs 2.25 --sigma-R 0.3 \\
        --v-obs 15.0 --sigma-v 2.0
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Reuse grid loading and density helpers from infer_cluster_mass
from src._calc.infer_cluster_mass import (
    collect_grid,
    cloud_radius_uniform,
    compute_n_edge,
    check_density_coverage,
    interpolate_density_grid,
    check_mass_coverage,
    interpolate_mass_grid,
    OBSERVED_SYSTEMS,
    PROFILE_DENSITY,
    V_AU2KMS,
    N_ISM_FLOOR,
    MIN_PTS,
    C_BLUE,
    C_VERMILLION,
    C_GREEN,
    C_PURPLE,
    C_ORANGE,
    C_SKY,
    C_BLACK,
    MU_H,
)

logger = logging.getLogger(__name__)

# Output directory: ./fig/ at project root
FIG_DIR = Path(__file__).parent.parent.parent / "fig"

# Apply trinity plot style if available
_style_path = Path(__file__).parent.parent / "_plots" / "trinity.mplstyle"
if _style_path.exists():
    plt.style.use(str(_style_path))


# ======================================================================
# Step 1: Posterior computation -- marginalise over t
# ======================================================================

def _logsumexp(log_vals: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(log_vals)
    if not np.isfinite(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(log_vals - a_max)))


def compute_age_posterior(
    records: List[Dict],
    R_obs: float,
    sigma_R: float,
    v_obs: float = None,
    sigma_v: float = None,
    n_edge_obs: float = None,
    sigma_n_edge: float = None,
    cmf_slope: float = -1.8,
    profile_assumed: str = "uniform",
    t_min: float = 0.0,
    t_max: float = None,
) -> Optional[Dict]:
    """
    Compute posterior over bubble age from TRINITY grid trajectories.

    For every grid model and every snapshot time t_j, evaluate the
    Gaussian log-likelihood of the observables (R and optionally v,
    n_edge).  The prior includes a CMF term and log-uniform priors on
    sfe and n_core (identical to ``infer_cluster_mass``).

    The marginal posterior over age is obtained by summing weights in
    time bins.

    Parameters
    ----------
    records : list of dict
        Output of collect_grid().
    R_obs : float
        Observed bubble radius [pc].
    sigma_R : float
        Uncertainty in R [pc].
    v_obs : float, optional
        Observed expansion velocity [km/s].
    sigma_v : float, optional
        Uncertainty in v [km/s].
    n_edge_obs : float, optional
        Observed density at shell edge [cm^-3].
    sigma_n_edge : float, optional
        Uncertainty in n_edge [cm^-3].
    cmf_slope : float
        Cloud mass function slope dN/dM ~ M^beta (default -1.8).
    profile_assumed : str
        Density profile assumption for n_edge.
    t_min : float
        Minimum age to consider [Myr] (default 0).
    t_max : float, optional
        Maximum age to consider [Myr].

    Returns
    -------
    dict or None
        Posterior results with keys: t_bins, pdf_age, median_age,
        lo_age, hi_age, observables_used, N_eff, etc.
    """
    if not records:
        return None

    # Build observable description
    obs_parts = [r"$R$"]
    if v_obs is not None and sigma_v is not None:
        obs_parts.append(r"$v_{\rm exp}$")
    if n_edge_obs is not None and sigma_n_edge is not None:
        obs_parts.append(r"$n_{\rm edge}$")
    observables_used = ", ".join(obs_parts)

    # Collect (time, log_weight) pairs across all models and snapshots
    all_t = []
    all_log_w = []

    for rec in records:
        t_arr = rec["t"]
        R_arr = rec["R"]
        v_kms_arr = rec["v_kms"]

        R_cloud = rec["rCloud"]
        if not np.isfinite(R_cloud) or R_cloud <= 0:
            R_cloud = cloud_radius_uniform(rec["mCloud"], rec["nCore"])

        # CMF + Jeffreys prior (same as infer_cluster_mass)
        log_prior = ((cmf_slope + 1.0) * np.log(rec["mCloud"])
                     - np.log(rec["sfe"])
                     - np.log(rec["nCore"]))

        for j in range(len(t_arr)):
            t_j = t_arr[j]
            if t_j < t_min:
                continue
            if t_max is not None and t_j > t_max:
                continue

            R_model = R_arr[j]
            if R_model <= 0 or not np.isfinite(R_model):
                continue

            # Radius likelihood
            log_L = -0.5 * ((R_obs - R_model) / sigma_R) ** 2

            # Velocity likelihood
            if v_obs is not None and sigma_v is not None:
                v_model = v_kms_arr[j]
                if np.isfinite(v_model):
                    log_L += -0.5 * ((v_obs - v_model) / sigma_v) ** 2

            # Density likelihood (log-space)
            if n_edge_obs is not None and sigma_n_edge is not None:
                n_model = compute_n_edge(
                    R_model, rec["nCore"], R_cloud,
                    profile=profile_assumed,
                )
                if n_model > 0 and np.isfinite(n_model):
                    log10_n_obs = np.log10(n_edge_obs)
                    log10_n_model = np.log10(n_model)
                    sigma_logn = sigma_n_edge / (n_edge_obs * np.log(10.0))
                    log_L += -0.5 * ((log10_n_obs - log10_n_model)
                                     / sigma_logn) ** 2

            all_t.append(t_j)
            all_log_w.append(log_L + log_prior)

    if not all_t:
        logger.warning("No valid (model, snapshot) pairs — posterior is empty")
        return None

    all_t = np.asarray(all_t)
    all_log_w = np.asarray(all_log_w)

    # Normalise weights
    log_Z = _logsumexp(all_log_w)
    weights = np.exp(all_log_w - log_Z)

    # Effective sample size
    N_eff = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0.0

    # Marginal PDF over age via histogram
    n_bins = min(60, max(10, int(np.sqrt(len(all_t)))))
    t_lo_edge = max(all_t.min() - 0.05, 0.0)
    t_hi_edge = all_t.max() + 0.05
    bin_edges = np.linspace(t_lo_edge, t_hi_edge, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pdf_age, _ = np.histogram(all_t, bins=bin_edges, weights=weights)
    dbin = np.diff(bin_edges)
    total = np.sum(pdf_age * dbin)
    if total > 0:
        pdf_age = pdf_age / total

    # Weighted percentiles
    sorted_idx = np.argsort(all_t)
    cum_w = np.cumsum(weights[sorted_idx])
    if cum_w[-1] > 0:
        cum_w /= cum_w[-1]
        median_age_raw = float(np.interp(0.50, cum_w, all_t[sorted_idx]))
        lo_age_raw = float(np.interp(0.16, cum_w, all_t[sorted_idx]))
        hi_age_raw = float(np.interp(0.84, cum_w, all_t[sorted_idx]))
    else:
        median_age_raw = lo_age_raw = hi_age_raw = np.nan

    # KDE smoothing
    pdf_kde = None
    x_kde = None
    try:
        from scipy.stats import gaussian_kde
        if len(all_t) >= 5 and weights.sum() > 0:
            # Use adaptive bandwidth relative to data range
            bw = 0.05 * (all_t.max() - all_t.min()) if all_t.max() > all_t.min() else 0.1
            kde = gaussian_kde(all_t, weights=weights, bw_method=bw)
            x_kde = np.linspace(t_lo_edge, t_hi_edge, 300)
            pdf_kde = kde(x_kde)
    except ImportError:
        pass

    # Prefer KDE percentiles when available
    if pdf_kde is not None and x_kde is not None:
        cdf_kde = np.cumsum(pdf_kde)
        cdf_kde /= cdf_kde[-1]
        median_age = float(np.interp(0.50, cdf_kde, x_kde))
        lo_age = float(np.interp(0.16, cdf_kde, x_kde))
        hi_age = float(np.interp(0.84, cdf_kde, x_kde))
    else:
        median_age = median_age_raw
        lo_age = lo_age_raw
        hi_age = hi_age_raw

    return {
        "t_bins": bin_centres,
        "pdf_age": pdf_age,
        "median_age": median_age,
        "lo_age": lo_age,
        "hi_age": hi_age,
        "median_age_raw": median_age_raw,
        "lo_age_raw": lo_age_raw,
        "hi_age_raw": hi_age_raw,
        "observables_used": observables_used,
        "pdf_kde": pdf_kde,
        "x_kde": x_kde,
        "N_eff": N_eff,
        # Keep per-sample data for diagnostics
        "all_t": all_t,
        "all_weights": weights,
    }


# ======================================================================
# Step 2: Progressive degeneracy breaking
# ======================================================================

def run_progressive_inference(
    records: List[Dict],
    R_obs: float, sigma_R: float,
    v_obs: float = None, sigma_v: float = None,
    n_edge_obs: float = None, sigma_n_edge: float = None,
    cmf_slope: float = -1.8,
    profile_assumed: str = "uniform",
    t_min: float = 0.0,
    t_max: float = None,
) -> List[Dict]:
    """
    Run age inference with progressively more observables.

    Returns
    -------
    list of dict
        Each dict is a posterior result with an added 'label' key.
    """
    stages = []

    # Stage 1: R only
    res = compute_age_posterior(
        records, R_obs, sigma_R,
        cmf_slope=cmf_slope, profile_assumed=profile_assumed,
        t_min=t_min, t_max=t_max)
    if res is not None:
        res["label"] = r"$R$"
        stages.append(res)

    # Stage 2: R + v_exp
    if v_obs is not None and sigma_v is not None:
        res = compute_age_posterior(
            records, R_obs, sigma_R,
            v_obs=v_obs, sigma_v=sigma_v,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed,
            t_min=t_min, t_max=t_max)
        if res is not None:
            res["label"] = r"$R, v_{\rm exp}$"
            stages.append(res)

    # Stage 3: R + n_edge
    if n_edge_obs is not None and sigma_n_edge is not None:
        res = compute_age_posterior(
            records, R_obs, sigma_R,
            n_edge_obs=n_edge_obs, sigma_n_edge=sigma_n_edge,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed,
            t_min=t_min, t_max=t_max)
        if res is not None:
            res["label"] = r"$R, n_{\rm edge}$"
            stages.append(res)

    # Stage 4: R + v + n_edge
    if (v_obs is not None and sigma_v is not None
            and n_edge_obs is not None and sigma_n_edge is not None):
        res = compute_age_posterior(
            records, R_obs, sigma_R,
            v_obs=v_obs, sigma_v=sigma_v,
            n_edge_obs=n_edge_obs, sigma_n_edge=sigma_n_edge,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed,
            t_min=t_min, t_max=t_max)
        if res is not None:
            res["label"] = r"$R, v_{\rm exp}, n_{\rm edge}$"
            stages.append(res)

    return stages


# ======================================================================
# Step 3: Plotting
# ======================================================================

def plot_posterior(stages: List[Dict], system_name: str,
                  obs_dict: Dict, output_dir: Path,
                  fmt: str = "pdf", tag: str = "") -> None:
    """
    Marginal age posteriors from progressive inference stages.

    Layout mirrors infer_cluster_mass: 2x2 panels when multiple stages
    are present, single panel otherwise.
    """
    if not stages:
        return

    _PANEL_DEFS = [
        {"missing": None,
         "panel_label": "(a)", "color": C_BLUE},
        {"missing": r"No $v_{\rm exp}$ available",
         "panel_label": "(b)", "color": C_VERMILLION},
        {"missing": r"No $n_{\rm edge}$ available",
         "panel_label": "(c)", "color": C_GREEN},
        {"missing": r"No $v_{\rm exp}$ + $n_{\rm edge}$",
         "panel_label": "(d)", "color": C_PURPLE},
    ]

    # Literature age overlay
    _has_lit_t = obs_dict.get("t_obs") is not None
    if _has_lit_t:
        _t_true = obs_dict["t_obs"]
        _sigma_t = obs_dict.get("sigma_t", 0.0)
    _lit_label_used = False

    # Map stages to fixed slots
    slot_stage = [None] * 4
    for stage in stages:
        lbl = stage.get("label", "")
        if "v_{\\rm exp}, n" in lbl or "v_{\\rm exp}, n_{\\rm edge}" in lbl:
            slot_stage[3] = stage
        elif "n_{\\rm edge}" in lbl and "v" not in lbl:
            slot_stage[2] = stage
        elif "v_{\\rm exp}" in lbl and "n" not in lbl:
            slot_stage[1] = stage
        else:
            slot_stage[0] = stage

    # ------------------------------------------------------------------
    # Single-panel fallback
    # ------------------------------------------------------------------
    if len(stages) == 1:
        stage = stages[0]
        fig, ax = plt.subplots(figsize=(7, 5))
        c = C_BLUE
        bins = stage["t_bins"]
        pdf = stage["pdf_age"]
        med = stage["median_age"]
        lo = stage["lo_age"]
        hi = stage["hi_age"]
        n_eff = stage["N_eff"]

        ax.fill_between(bins, 0, pdf, color=c, alpha=0.25, step="mid")
        ax.step(bins, pdf, where="mid", color=c, lw=1.8)
        ci_mask = (bins >= lo) & (bins <= hi)
        if ci_mask.any():
            ax.fill_between(bins[ci_mask], 0, pdf[ci_mask],
                            color=c, alpha=0.15, step="mid")
        ax.axvline(med, color=c, ls="--", lw=1.2, alpha=0.7)
        if stage.get("pdf_kde") is not None:
            ax.plot(stage["x_kde"], stage["pdf_kde"],
                    color=c, lw=0.8, alpha=0.5)

        info = (f"{stage['label']}\n"
                f"$t = {med:.3f}"
                f"_{{-{med - lo:.3f}}}^{{+{hi - med:.3f}}}$ Myr\n"
                f"$N_{{\\rm eff}} = {n_eff:.1f}$")
        ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))

        if _has_lit_t and _sigma_t > 0:
            ax.axvline(_t_true, color=C_ORANGE, ls="-", lw=2.0,
                       alpha=0.8, label=r"Observed $t$", zorder=5)
            ax.axvspan(_t_true - _sigma_t, _t_true + _sigma_t,
                       color=C_ORANGE, alpha=0.12, zorder=1)
            ax.legend(fontsize=8, framealpha=0.7, loc="best")

        ax.set_xlabel(r"$t$ [Myr]")
        ax.set_ylabel(r"$p(t \mid \mathrm{data})$")

        fig.tight_layout()
        suffix = f"_{tag}" if tag else ""
        path = output_dir / f"infer_age_posterior_{system_name}{suffix}.{fmt}"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return

    # ------------------------------------------------------------------
    # 2x2 subplot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    # Shared x-axis range
    x_lo = min(s["t_bins"].min() for s in stages) - 0.05
    x_hi = max(s["t_bins"].max() for s in stages) + 0.05
    x_lo = max(x_lo, 0.0)

    base_stage = slot_stage[0]

    for slot_idx, (pdef, ax) in enumerate(zip(_PANEL_DEFS, axes.flat)):
        stage = slot_stage[slot_idx]
        c = pdef["color"]

        ax.set_xlim(x_lo, x_hi)

        ax.text(0.04, 0.96, pdef["panel_label"], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

        if stage is None:
            ax.text(0.5, 0.5, pdef["missing"], transform=ax.transAxes,
                    fontsize=10, color="0.5", style="italic",
                    ha="center", va="center")
        else:
            bins = stage["t_bins"]
            pdf = stage["pdf_age"]
            med = stage["median_age"]
            lo = stage["lo_age"]
            hi = stage["hi_age"]
            n_eff = stage["N_eff"]

            ax.fill_between(bins, 0, pdf, color=c, alpha=0.25, step="mid")
            ax.step(bins, pdf, where="mid", color=c, lw=1.8)

            ci_mask = (bins >= lo) & (bins <= hi)
            if ci_mask.any():
                ax.fill_between(bins[ci_mask], 0, pdf[ci_mask],
                                color=c, alpha=0.15, step="mid")

            ax.axvline(med, color=c, ls="--", lw=1.2, alpha=0.7)
            ax.annotate(f"{med:.3f}", xy=(med, 0.92),
                        xycoords=("data", "axes fraction"),
                        fontsize=7, color=c, ha="center", va="top")

            if stage.get("pdf_kde") is not None:
                ax.plot(stage["x_kde"], stage["pdf_kde"],
                        color=c, lw=0.8, alpha=0.5)

            info_lines = [stage["label"]]
            if np.isfinite(med):
                info_lines.append(
                    f"$t = {med:.3f}"
                    f"_{{-{med - lo:.3f}}}^{{+{hi - med:.3f}}}$ Myr")
            info_lines.append(f"$N_{{\\rm eff}} = {n_eff:.1f}$")
            ax.text(0.95, 0.95, "\n".join(info_lines),
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="grey", alpha=0.8))

            # Observed age overlay
            if _has_lit_t and _sigma_t > 0:
                _lbl = r"Observed $t$" if not _lit_label_used else None
                ax.axvline(_t_true, color=C_ORANGE, ls="-", lw=2.0,
                           alpha=0.8, label=_lbl, zorder=5)
                ax.axvspan(_t_true - _sigma_t, _t_true + _sigma_t,
                           color=C_ORANGE, alpha=0.12, zorder=1)
                if not _lit_label_used:
                    ax.legend(fontsize=7, framealpha=0.7, loc="upper left")
                _lit_label_used = True

        # Per-panel y-limit
        if stage is not None:
            y_max_hist = stage["pdf_age"].max() if len(stage["pdf_age"]) > 0 else 0
            y_max_kde = 0
            if stage.get("pdf_kde") is not None:
                y_max_kde = stage["pdf_kde"].max()
            y_hi_panel = max(y_max_hist, y_max_kde) * 1.15
            if slot_idx > 0 and base_stage is not None:
                y_max_ghost = base_stage["pdf_age"].max()
                y_hi_panel = max(y_hi_panel, y_max_ghost * 1.15)
            ax.set_ylim(0, max(y_hi_panel, 0.1))

        row, col = divmod(slot_idx, 2)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$t$ [Myr]")
        if col == 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$p(t \mid \mathrm{data})$")

    # System info in panel (a)
    obs_lines = [system_name]
    obs_lines.append(
        f"$R = {obs_dict['R_obs']:.2f} \\pm {obs_dict['sigma_R']:.2f}$ pc")
    if obs_dict.get("v_obs") is not None:
        obs_lines.append(
            f"$v = {obs_dict['v_obs']:.1f} \\pm "
            f"{obs_dict['sigma_v']:.1f}$ km/s")
    if obs_dict.get("n_edge_obs") is not None:
        obs_lines.append(
            f"$n_{{\\rm edge}} = {obs_dict['n_edge_obs']:.0f}"
            f" \\pm {obs_dict['sigma_n_edge']:.0f}$ cm$^{{-3}}$")
    ax_a = axes[0, 0]
    ax_a.text(0.95, 0.60, "\n".join(obs_lines), transform=ax_a.transAxes,
              fontsize=7, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                        edgecolor="grey", alpha=0.7))

    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"infer_age_posterior_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_Rt_tracks(stage: Dict, records: List[Dict],
                   obs_dict: Dict, system_name: str,
                   output_dir: Path,
                   fmt: str = "pdf", tag: str = "",
                   n_top: int = 20) -> None:
    """
    Top-weighted R(t) tracks, with inferred age highlighted.
    """
    if stage is None:
        return

    all_t = stage["all_t"]
    all_w = stage["all_weights"]
    med = stage["median_age"]
    lo = stage["lo_age"]
    hi = stage["hi_age"]

    # Find which records contribute most weight
    # Aggregate weight per record
    rec_weight = np.zeros(len(records))
    idx_ptr = 0
    for i, rec in enumerate(records):
        t_arr = rec["t"]
        n_snap = len(t_arr)
        # We need to re-derive per-record weight from all_weights
        # For simplicity, use sum of weights where time falls in this record
    # Instead, compute per-record total weight by re-running a lightweight pass
    for i, rec in enumerate(records):
        t_arr = rec["t"]
        R_arr = rec["R"]
        R_obs = obs_dict["R_obs"]
        sigma_R = obs_dict["sigma_R"]
        for j in range(len(t_arr)):
            R_model = R_arr[j]
            if R_model > 0 and np.isfinite(R_model):
                log_L = -0.5 * ((R_obs - R_model) / sigma_R) ** 2
                rec_weight[i] += np.exp(log_L)

    # Select top N by aggregated weight
    top_idx = np.argsort(rec_weight)[-n_top:]

    fig, ax = plt.subplots(figsize=(7, 5))

    w_max = rec_weight[top_idx].max() if rec_weight[top_idx].max() > 0 else 1.0
    log_Mcl = np.array([np.log10(r["M_star"]) for r in records])

    norm = plt.Normalize(
        vmin=log_Mcl[top_idx].min(),
        vmax=log_Mcl[top_idx].max(),
    )
    cmap = plt.cm.viridis

    for i in top_idx:
        rec = records[i]
        c = cmap(norm(log_Mcl[i]))
        lw = 0.5 + 2.5 * (rec_weight[i] / w_max)
        ax.plot(rec["t"], rec["R"], color=c, lw=lw, alpha=0.7)

    # Observed radius as horizontal band
    ax.axhspan(R_obs - sigma_R, R_obs + sigma_R,
               color=C_VERMILLION, alpha=0.15, zorder=1)
    ax.axhline(R_obs, color=C_VERMILLION, ls="--", lw=1.0, alpha=0.6)

    # Inferred age as vertical band
    ax.axvspan(lo, hi, color=C_BLUE, alpha=0.12, zorder=1,
               label=f"68% CI: [{lo:.3f}, {hi:.3f}] Myr")
    ax.axvline(med, color=C_BLUE, ls="--", lw=1.5, alpha=0.7,
               label=f"Median age = {med:.3f} Myr")

    # Observed t if available
    if obs_dict.get("t_obs") is not None:
        t_true = obs_dict["t_obs"]
        ax.axvline(t_true, color=C_ORANGE, ls="-", lw=2.0, alpha=0.8,
                   label=f"Observed t = {t_true:.3f} Myr", zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}\,M_{\rm cl}$", fontsize=10)

    ax.set_xlabel(r"$t$ [Myr]")
    ax.set_ylabel(r"$R$ [pc]")
    ax.legend(fontsize=8, framealpha=0.7, loc="best")

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"infer_age_Rt_tracks_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ======================================================================
# Step 4: Console summary
# ======================================================================

def print_inference_summary(system_name: str, stages: List[Dict],
                            obs_dict: Dict) -> None:
    """Print age inference results table."""
    print()
    print("=" * 76)
    print(f"  AGE INFERENCE SUMMARY -- System: {system_name}")
    print("=" * 76)

    print(f"  R = {obs_dict['R_obs']:.2f} +/- {obs_dict['sigma_R']:.2f} pc")
    if obs_dict.get("v_obs") is not None:
        print(f"  v = {obs_dict['v_obs']:.1f} +/- {obs_dict['sigma_v']:.1f} km/s")
    if obs_dict.get("n_edge_obs") is not None:
        print(f"  n_edge = {obs_dict['n_edge_obs']:.0f} +/- "
              f"{obs_dict['sigma_n_edge']:.0f} cm^-3")
    if obs_dict.get("t_obs") is not None:
        print(f"  Observed t = {obs_dict['t_obs']:.3f} +/- "
              f"{obs_dict.get('sigma_t', 0):.3f} Myr")
    if obs_dict.get("ref"):
        print(f"  Ref: {obs_dict['ref']}")

    print()
    print(f"  {'Observables':<35s} {'median t [Myr]':>16s} "
          f"{'68% CI [Myr]':>18s} {'N_eff':>7s}")
    print("  " + "-" * 78)
    for stage in stages:
        med = stage["median_age"]
        lo = stage["lo_age"]
        hi = stage["hi_age"]
        n_eff = stage["N_eff"]
        label = stage.get("label", "?")
        label_clean = label.replace(r"$", "").replace(r"\rm ", "")
        label_clean = label_clean.replace(r"_{\rm exp}", "_exp")
        label_clean = label_clean.replace(r"_{\rm edge}", "_edge")
        label_clean = label_clean.replace("\\", "")

        if np.isfinite(med):
            ci_str = f"[{lo:.3f}, {hi:.3f}]"
            print(f"  {label_clean:<35s} {med:>16.3f} {ci_str:>18s} "
                  f"{n_eff:>7.1f}")
        else:
            print(f"  {label_clean:<35s} {'N/A':>16s} {'N/A':>18s} "
                  f"{'N/A':>7s}")

    print("=" * 76)
    print()


# ======================================================================
# Step 5: Synthetic observation builder
# ======================================================================

def _build_synthetic_obs(
    records: List[Dict],
    frac_sigma: float = 0.10,
    t_min: float = 0.1,
    rng_seed: int = None,
) -> Dict:
    """
    Draw a random grid model and build a synthetic observation for age
    inference (R, v, n_edge — but NOT t, since t is what we're inferring).

    The true age is stored as ``t_true`` so the posterior can be compared.
    """
    rng = np.random.default_rng(rng_seed)

    # Filter to records with expanding snapshots past t_min
    eligible = []
    for r in records:
        mask = (r["t"] > t_min) & (r["v_kms"] > 0) & (r["R"] > 0)
        if mask.any():
            eligible.append(r)
    if not eligible:
        raise ValueError(
            f"No grid model has expanding snapshots beyond t_min = {t_min} Myr"
        )

    rec = eligible[rng.integers(len(eligible))]

    valid_idx = np.where(
        (rec["t"] > t_min) & (rec["v_kms"] > 0) & (rec["R"] > 0)
    )[0]
    idx = rng.choice(valid_idx)

    t_sample = float(rec["t"][idx])
    R_sample = float(rec["R"][idx])
    v_sample = float(rec["v_kms"][idx])

    R_cloud = rec.get("rCloud", cloud_radius_uniform(rec["mCloud"], rec["nCore"]))
    if not np.isfinite(R_cloud) or R_cloud <= 0:
        R_cloud = cloud_radius_uniform(rec["mCloud"], rec["nCore"])
    profile = rec.get("profile", "uniform")
    n_edge_sample = float(compute_n_edge(R_sample, rec["nCore"], R_cloud, profile))

    sigma_R = max(frac_sigma * R_sample, 1e-3)
    sigma_v = max(frac_sigma * v_sample, 0.5) if v_sample > 0 else None
    sigma_n = max(frac_sigma * n_edge_sample, 1.0) if n_edge_sample > N_ISM_FLOOR else None

    logger.info(
        "SYNTHETIC age test -- true answer: t = %.3f Myr "
        "(M_cloud = %.0f, sfe = %.2f, n_cl = %.0f cm^-3, folder = %s)",
        t_sample, rec["mCloud"], rec["sfe"], rec["nCore"], rec["folder"],
    )
    logger.info(
        "  Sampled: R = %.2f pc, v = %.1f km/s, n_edge = %.0f cm^-3",
        R_sample, v_sample, n_edge_sample,
    )

    obs = {
        "R_obs": R_sample,
        "sigma_R": sigma_R,
        "t_true": t_sample,            # ground truth (not used in inference)
        "t_obs": t_sample,              # for overlay on plots
        "sigma_t": frac_sigma * t_sample,
        "ref": "synthetic (self-test)",
        "note": (f"Drawn from {rec['folder']}; "
                 f"M_cloud={rec['mCloud']:.0f}, sfe={rec['sfe']:.2f}, "
                 f"n_cl={rec['nCore']:.0f}"),
    }

    if v_sample > 0 and sigma_v is not None:
        obs["v_obs"] = v_sample
        obs["sigma_v"] = sigma_v

    if n_edge_sample > N_ISM_FLOOR and sigma_n is not None:
        obs["n_edge_obs"] = n_edge_sample
        obs["sigma_n_edge"] = sigma_n

    return obs


# ======================================================================
# Step 6: CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Bayesian inference of bubble age from observed properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_cluster_age.py -F /path/to/sweep --system RCW120
  python infer_cluster_age.py -F /path/to/sweep \\
      --R-obs 2.25 --sigma-R 0.3 --v-obs 15.0 --sigma-v 2.0
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True, nargs="+",
        help="Path(s) to one or more sweep output directory trees.",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        choices=list(OBSERVED_SYSTEMS.keys()) + ["synthetic"],
        help="Pre-stored observational target, or 'synthetic' for "
             "a random grid model as a validation test.",
    )
    parser.add_argument(
        "--R-obs", type=float, default=None,
        help="Observed bubble radius [pc].",
    )
    parser.add_argument(
        "--sigma-R", type=float, default=None,
        help="Uncertainty in R [pc].",
    )
    parser.add_argument(
        "--v-obs", type=float, default=None,
        help="Observed expansion velocity [km/s].",
    )
    parser.add_argument(
        "--sigma-v", type=float, default=None,
        help="Uncertainty in v [km/s].",
    )
    parser.add_argument(
        "--n-edge-obs", type=float, default=None,
        help="Observed density at shell edge [cm^-3].",
    )
    parser.add_argument(
        "--sigma-n-edge", type=float, default=None,
        help="Uncertainty in n_edge [cm^-3].",
    )
    parser.add_argument(
        "--cmf-slope", type=float, default=-1.8,
        help="Cloud mass function slope dN/dM ~ M^beta (default: -1.8).",
    )
    parser.add_argument(
        "--profile", type=str, default="uniform",
        choices=list(PROFILE_DENSITY.keys()),
        help="Density profile assumption (default: uniform).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Tag appended to output filenames.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Truncate tracks at this time [Myr].",
    )
    parser.add_argument(
        "--t-min", type=float, default=0.0,
        help="Minimum age to consider [Myr] (default: 0).",
    )
    parser.add_argument(
        "--interp-density", action="store_true", default=False,
        help="Interpolate virtual models at intermediate densities.",
    )
    parser.add_argument(
        "--n-per-decade", type=int, default=3,
        help="Interpolation points per decade in n_cl (default: 3).",
    )
    parser.add_argument(
        "--interp-mass", action="store_true", default=False,
        help="Interpolate virtual models at intermediate cloud masses.",
    )
    parser.add_argument(
        "--m-per-decade", type=int, default=3,
        help="Interpolation points per decade in M_cloud (default: 3).",
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

    folder_paths = [Path(f) for f in args.folder]
    for fp in folder_paths:
        if not fp.is_dir():
            logger.error("Folder does not exist: %s", fp)
            return 1

    folder_name = "+".join(fp.name for fp in folder_paths)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else FIG_DIR / "infer_cluster_age" / folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect grid
    records: List[Dict] = []
    for fp in folder_paths:
        records.extend(collect_grid(fp, t_end=args.t_end))
    if not records:
        logger.error("No valid grid data collected -- aborting.")
        return 1

    # Resolve observables
    obs = {}
    if args.system == "synthetic":
        obs = _build_synthetic_obs(records)
        system_name = "synthetic"
    elif args.system is not None:
        obs = dict(OBSERVED_SYSTEMS[args.system])
        system_name = args.system
    else:
        system_name = "custom"

    # Manual overrides
    if args.R_obs is not None:
        obs["R_obs"] = args.R_obs
    if args.sigma_R is not None:
        obs["sigma_R"] = args.sigma_R
    if args.v_obs is not None:
        obs["v_obs"] = args.v_obs
    if args.sigma_v is not None:
        obs["sigma_v"] = args.sigma_v
    if args.n_edge_obs is not None:
        obs["n_edge_obs"] = args.n_edge_obs
    if args.sigma_n_edge is not None:
        obs["sigma_n_edge"] = args.sigma_n_edge

    # Validate required observables
    if "R_obs" not in obs or "sigma_R" not in obs:
        logger.error("R_obs and sigma_R are required (use --system or --R-obs)")
        return 1

    # Check grid coverage
    density_ok = check_density_coverage(
        records, n_edge_obs=obs.get("n_edge_obs"), max_gap_dex=0.5)

    if args.interp_density:
        records = interpolate_density_grid(records, n_per_decade=args.n_per_decade)
    elif not density_ok:
        logger.warning(
            ">>> Rerun with --interp-density to improve density coverage.")

    mass_ok = check_mass_coverage(records, max_gap_dex=0.5)

    if args.interp_mass:
        records = interpolate_mass_grid(records, m_per_decade=args.m_per_decade)
    elif not mass_ok:
        logger.warning(
            ">>> Rerun with --interp-mass to improve mass coverage.")

    # Run progressive inference
    stages = run_progressive_inference(
        records,
        R_obs=obs["R_obs"], sigma_R=obs["sigma_R"],
        v_obs=obs.get("v_obs"), sigma_v=obs.get("sigma_v"),
        n_edge_obs=obs.get("n_edge_obs"),
        sigma_n_edge=obs.get("sigma_n_edge"),
        cmf_slope=args.cmf_slope,
        profile_assumed=args.profile,
        t_min=args.t_min,
        t_max=args.t_end,
    )

    if not stages:
        logger.error("All inference stages returned None -- aborting.")
        return 1

    # Figures
    plot_posterior(stages, system_name, obs, output_dir,
                   fmt=args.fmt, tag=args.tag)

    best_stage = stages[-1]
    plot_Rt_tracks(best_stage, records, obs, system_name, output_dir,
                   fmt=args.fmt, tag=args.tag)

    # Console summary
    print_inference_summary(system_name, stages, obs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
