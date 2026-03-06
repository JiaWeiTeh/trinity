#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic diagrams for TRINITY Paper IV (Method 1).

Produces four diagnostic figures from a TRINITY parameter sweep:

1. **Radius–age fan plot** — R(t) tracks at a fixed ambient density,
   colour-coded by cluster mass, with Weaver (1977) reference lines
   and observed systems overlaid.

2. **Mass–radius isochrones** — (log M_cl, log R) at fixed ages,
   revealing the mapping between cluster mass and bubble size at each
   epoch.

3. **Regime map** — scatter of (log M_cl, log R) at a single snapshot
   time, colour-coded by the TRINITY simulation phase (energy-driven,
   implicit, transition, momentum).

4. **Multi-panel density-sliced (t, R)** — one panel per ambient
   density, tracks coloured by expansion velocity at a reference time.

CLI usage
---------
    python src/_calc/diagnostic_diagrams.py -F output/sweep
    python src/_calc/diagnostic_diagrams.py -F output/sweep --ncl 1000 --figures 1,2,3,4
    python src/_calc/diagnostic_diagrams.py -F output/sweep --t-regime 1.0 --fmt png

References
----------
* Weaver, R. et al. (1977), ApJ, 218, 377 — wind-bubble expansion.
* Luisi+2021, Sci. Adv. 7, eabe9511 — RCW 120.
* Pabst+2019, Nature 565, 618 — Orion Veil.
* Harper-Clark & Murray 2009, ApJ 693, 1696 — Carina.
"""

import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

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
    iter_progress,
)
from src._functions.unit_conversions import INV_CONV, CONV, CGS

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

V_AU2KMS = INV_CONV.v_au2kms          # pc/Myr -> km/s

# Minimum expanding points required
MIN_PTS = 5

# Mean molecular weight [m_H units]
MU_H = 1.4

# Colourblind-safe palette (Wong 2011)
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_BLACK = "#000000"

LINESTYLES = ["-", "--", "-.", ":"]


# ======================================================================
# Observed systems
# ======================================================================

OBSERVED_SYSTEMS = {
    "RCW120": {
        "R_obs": 2.25, "sigma_R": 0.3,
        "t_obs": 0.15, "sigma_t": 0.05,
        "v_obs": 15.0, "sigma_v": 2.0,
        "n_edge_obs": 3000.0, "sigma_n_edge": 500.0,
        "Mcl_lit": 1000.0, "sigma_Mcl_lit_dex": 0.3,
        "ref": "Luisi+2021; Deharveng+2009; Martins+2010",
        "marker": "*", "ms": 14,
    },
    "Orion_Veil": {
        "R_obs": 2.0, "sigma_R": 0.3,
        "t_obs": 0.20, "sigma_t": 0.05,
        "v_obs": 13.0, "sigma_v": 2.0,
        "n_edge_obs": 1500.0, "sigma_n_edge": 500.0,
        "Mcl_lit": 1800.0, "sigma_Mcl_lit_dex": 0.15,
        "ref": "Pabst+2019 Nature; Pabst+2020 A&A; Da Rio+2014",
        "marker": "D", "ms": 10,
    },
    "Carina": {
        "R_obs": 10.0, "sigma_R": 2.0,
        "t_obs": 3.6, "sigma_t": 0.5,
        "v_obs": None, "sigma_v": None,
        "n_edge_obs": 100.0, "sigma_n_edge": 30.0,
        "Mcl_lit": 10000.0, "sigma_Mcl_lit_dex": 0.2,
        "ref": "Harper-Clark & Murray 2009; Smith 2006",
        "marker": "s", "ms": 10,
    },
}


# ======================================================================
# Weaver reference
# ======================================================================

def weaver_radius(Mcl, ncl, t_Myr, alpha=0.76):
    """
    Weaver (1977) energy-driven bubble radius.

    Parameters
    ----------
    Mcl : float
        Cluster mass [Msun].
    ncl : float
        Ambient density [cm^-3].
    t_Myr : float or array
        Time [Myr].

    Returns
    -------
    R : float or array
        Bubble radius [pc].
    """
    # Wind luminosity: Lw ~ 1e34 * (Mcl / 1e4) erg/s (pre-SN average)
    Lw = 1e34 * (Mcl / 1e4)  # erg/s
    rho0 = ncl * MU_H * CGS.m_H  # g/cm^3
    t_s = t_Myr * 3.156e13  # seconds
    R_cm = alpha * (Lw / rho0) ** 0.2 * t_s ** 0.6
    return R_cm / CGS.pc  # pc


def cloud_radius_uniform(M_cloud, n_cl, mu=MU_H):
    """
    Cloud radius for a uniform sphere [pc].

    R_cloud = (3 M / (4 pi rho))^(1/3).
    """
    n_au = n_cl * CONV.ndens_cgs2au           # 1/pc^3
    mu_au = mu * CGS.m_H * CONV.g2Msun        # Msun
    rho = n_au * mu_au                         # Msun/pc^3
    return (3.0 * M_cloud / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


# ======================================================================
# Data loading — follows collect_grid() from infer_cluster_mass.py
# ======================================================================

def collect_grid(folder_path: Path, t_end: float = None) -> List[Dict]:
    """
    Walk sweep directory and extract grid tracks for diagnostics.

    Parameters
    ----------
    folder_path : Path
        Top-level sweep output directory.
    t_end : float, optional
        Maximum time [Myr] to consider.

    Returns
    -------
    list of dict
        One record per valid simulation run with keys:
        mCloud, sfe, nCore, M_star, folder, profile,
        t, R, v_au, v_kms, phase, rCloud.
    """
    sim_files = find_all_simulations(folder_path)
    if not sim_files:
        logger.error("No simulation files under %s", folder_path)
        return []

    logger.info("Found %d simulation(s) in %s", len(sim_files), folder_path)

    records: List[Dict] = []
    for data_path in iter_progress(sim_files, "Loading simulations"):
        folder_name = data_path.parent.name
        parsed = parse_simulation_params(folder_name)
        if parsed is None:
            logger.warning("Cannot parse '%s' — skipping", folder_name)
            continue

        nCore = float(parsed["ndens"])
        mCloud = float(parsed["mCloud"])
        sfe = int(parsed["sfe"]) / 100.0

        try:
            output = load_output(data_path)
        except Exception as e:
            logger.warning("Could not load %s: %s", data_path, e)
            continue

        if len(output) < MIN_PTS:
            logger.warning("Too few snapshots (%d) in %s — skipping",
                           len(output), data_path)
            continue

        t = output.get("t_now")
        R = output.get("R2")
        v_au = output.get("v2")                  # pc/Myr
        phase = np.array(output.get("current_phase", as_array=False))

        # Replace NaN
        v_au = np.nan_to_num(v_au, nan=0.0)
        R = np.nan_to_num(R, nan=0.0)

        # Get rCloud (constant per run — take first valid value)
        try:
            rCloud_arr = output.get("rCloud")
            rCloud_val = float(rCloud_arr[0]) if len(rCloud_arr) > 0 else np.nan
        except Exception:
            rCloud_val = np.nan

        # Truncate at t_end if requested
        if t_end is not None and t[-1] > t_end:
            mask_t = t <= t_end
            if mask_t.sum() < MIN_PTS:
                continue
            t = t[mask_t]
            R = R[mask_t]
            v_au = v_au[mask_t]
            phase = phase[mask_t]

        # Deduplicate timestamps
        _, unique_idx = np.unique(t, return_index=True)
        t = t[unique_idx]
        R = R[unique_idx]
        v_au = v_au[unique_idx]
        phase = phase[unique_idx]

        # Detect profile type from folder name
        profile = "powerlaw" if "_PL" in folder_name and "_PL0" not in folder_name else "uniform"

        M_star = sfe * mCloud

        records.append({
            "mCloud": mCloud,
            "sfe": sfe,
            "nCore": nCore,
            "M_star": M_star,
            "folder": folder_name,
            "profile": profile,
            "t": t,
            "R": R,
            "v_au": v_au,
            "v_kms": v_au * V_AU2KMS,
            "phase": phase,
            "rCloud": rCloud_val,
        })

    logger.info("Collected %d valid grid runs", len(records))
    return records


# ======================================================================
# Grouping helpers
# ======================================================================

def group_by_density(records):
    """Group records by nCore.  Returns dict: {nCore: [records]}."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec["nCore"]].append(rec)
    return dict(groups)


def find_nearest_density(groups, target_ncl):
    """Find the nCore key closest (in log-space) to target_ncl and warn if not exact."""
    available = sorted(groups.keys())
    if not available:
        return None
    idx = np.argmin(np.abs(np.log10(available) - np.log10(target_ncl)))
    best = available[idx]
    if best != target_ncl:
        logger.warning("Requested ncl=%.1e not in grid; using nearest ncl=%.1e "
                        "(%.2f dex away)", target_ncl, best,
                        abs(np.log10(best) - np.log10(target_ncl)))
    return best


def interp_at_time(t_arr, y_arr, t_target):
    """Linearly interpolate y(t_target) from a (t, y) track.

    Returns NaN if t_target is outside the track range.
    """
    if t_target < t_arr[0] or t_target > t_arr[-1]:
        return np.nan
    return float(np.interp(t_target, t_arr, y_arr))


def phase_at_time(t_arr, phase_arr, t_target):
    """Return the simulation phase at t_target (nearest-neighbour)."""
    if t_target < t_arr[0] or t_target > t_arr[-1]:
        return "unknown"
    idx = np.argmin(np.abs(t_arr - t_target))
    return str(phase_arr[idx])


# ======================================================================
# Phase colour mapping
# ======================================================================

PHASE_COLOURS = {
    "energy":     C_BLUE,
    "implicit":   C_SKY,
    "transition": C_GREEN,
    "momentum":   C_VERMILLION,
    "unknown":    C_PURPLE,
}

PHASE_LABELS = {
    "energy":     "Energy-driven",
    "implicit":   "Implicit energy",
    "transition": "Transition",
    "momentum":   "Momentum-driven",
    "unknown":    "Unknown",
}


# ======================================================================
# Observed-system plotting helper
# ======================================================================

def _obs_within_dex(obs, panel_ncl, dex_tol=0.5):
    """Return True if the system's n_edge_obs is within dex_tol of panel_ncl."""
    n = obs.get("n_edge_obs")
    if n is None or n <= 0:
        return False
    return abs(np.log10(n) - np.log10(panel_ncl)) <= dex_tol


def _plot_obs_Rt(ax, obs_systems, panel_ncl, dex_tol=0.5):
    """Overlay observed systems on a (t, R) panel."""
    for name, obs in obs_systems.items():
        if not _obs_within_dex(obs, panel_ncl, dex_tol):
            continue
        ax.errorbar(
            obs["t_obs"], obs["R_obs"],
            xerr=obs["sigma_t"], yerr=obs["sigma_R"],
            marker=obs["marker"], ms=obs["ms"],
            color=C_VERMILLION, markeredgecolor="k", markeredgewidth=0.8,
            capsize=3, lw=1.2, zorder=10,
            label=name.replace("_", " "),
        )


def _plot_obs_MR(ax, obs_systems):
    """Overlay observed systems on a (log Mcl, log R) panel."""
    for name, obs in obs_systems.items():
        Mcl = obs.get("Mcl_lit")
        if Mcl is None or Mcl <= 0:
            continue
        logMcl = np.log10(Mcl)
        # Asymmetric dex error -> log-space symmetric
        dex = obs.get("sigma_Mcl_lit_dex", 0.0) or 0.0
        logR = np.log10(obs["R_obs"])
        sigma_logR = obs["sigma_R"] / (obs["R_obs"] * np.log(10))
        ax.errorbar(
            logMcl, logR,
            xerr=dex, yerr=sigma_logR,
            marker=obs["marker"], ms=obs["ms"],
            color=C_VERMILLION, markeredgecolor="k", markeredgewidth=0.8,
            capsize=3, lw=1.2, zorder=10,
            label=name.replace("_", " "),
        )


# ======================================================================
# Figure 1: Radius–age fan plot
# ======================================================================

def plot_fig1(records, ncl, obs_systems, output_dir, fmt="pdf", tag=""):
    """
    Radius–age fan plot at a fixed density.

    R(t) tracks colour-coded by log10(M_star), with Weaver reference
    lines and observed systems overlaid.
    """
    groups = group_by_density(records)
    ncl_actual = find_nearest_density(groups, ncl)
    if ncl_actual is None:
        logger.error("No density groups available for Fig 1.")
        return
    recs = groups[ncl_actual]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Colour by log10(M_star)
    Mstar_vals = np.array([r["M_star"] for r in recs])
    logMstar = np.log10(np.clip(Mstar_vals, 1.0, None))
    vmin, vmax = logMstar.min(), logMstar.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    for rec, lm in zip(recs, logMstar):
        ax.plot(rec["t"], rec["R"], color=cmap(norm(lm)), lw=1.0, alpha=0.8)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$")

    # Weaver reference lines
    t_ref = np.linspace(0.01, max(r["t"][-1] for r in recs), 200)
    for exp in [2, 3, 4, 5]:
        Mcl_ref = 10.0 ** exp
        R_w = weaver_radius(Mcl_ref, ncl_actual, t_ref)
        ax.plot(t_ref, R_w, color="grey", ls="--", lw=0.8, alpha=0.6)
        # Label at right end
        idx = -1
        if np.isfinite(R_w[idx]) and R_w[idx] > 0:
            ax.text(t_ref[idx], R_w[idx],
                    rf"$M_{{\rm cl}}=10^{exp}$",
                    fontsize=7, color="grey", va="bottom", ha="right")

    # Cloud radius lines (median per unique mCloud at this ncl)
    unique_mc = sorted(set(r["mCloud"] for r in recs))
    for mc in unique_mc:
        Rc = cloud_radius_uniform(mc, ncl_actual)
        ax.axhline(Rc, color="grey", ls=":", lw=0.6, alpha=0.5)

    # Observed systems
    _plot_obs_Rt(ax, obs_systems, ncl_actual)

    ax.set_yscale("log")
    ax.set_ylim(0.1, 500)
    ax.set_xlabel(r"$t$ [Myr]")
    ax.set_ylabel(r"$R$ [pc]")
    ncl_log = int(round(np.log10(ncl_actual)))
    ax.set_title(rf"Radius–age fan plot ($n_{{\rm cl}}=10^{ncl_log}\;\rm cm^{{-3}}$)")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left",
                  fontsize=8, framealpha=0.8)

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    fname = output_dir / f"diag_Rt_fan_{ncl_actual:.0e}{suffix}.{fmt}"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ======================================================================
# Figure 2: Mass–radius isochrones
# ======================================================================

def plot_fig2(records, ncl, obs_systems, output_dir, fmt="pdf", tag="",
              isochrone_ages=None):
    """
    Mass–radius isochrones at a fixed density.

    At each isochrone age, interpolate R(t) for every run at this density
    and plot (log M_cl, log R).
    """
    if isochrone_ages is None:
        isochrone_ages = [0.1, 0.3, 1.0, 3.0, 5.0, 10.0]

    groups = group_by_density(records)
    ncl_actual = find_nearest_density(groups, ncl)
    if ncl_actual is None:
        logger.error("No density groups available for Fig 2.")
        return
    recs = groups[ncl_actual]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Age colourmap
    age_arr = np.array(isochrone_ages)
    norm = matplotlib.colors.LogNorm(vmin=age_arr.min(), vmax=age_arr.max())
    cmap = plt.cm.plasma

    for age in isochrone_ages:
        logM_vals = []
        logR_vals = []
        for rec in recs:
            R_interp = interp_at_time(rec["t"], rec["R"], age)
            if np.isfinite(R_interp) and R_interp > 0 and rec["M_star"] > 0:
                logM_vals.append(np.log10(rec["M_star"]))
                logR_vals.append(np.log10(R_interp))
        if not logM_vals:
            continue
        # Sort by mass for connected line
        order = np.argsort(logM_vals)
        logM_sorted = np.array(logM_vals)[order]
        logR_sorted = np.array(logR_vals)[order]
        colour = cmap(norm(age))
        ax.plot(logM_sorted, logR_sorted, color=colour, lw=1.5, alpha=0.9,
                marker="o", ms=3, zorder=5)
        # Label at right end
        ax.text(logM_sorted[-1] + 0.05, logR_sorted[-1],
                f"{age} Myr", fontsize=7, color=colour, va="center")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Isochrone age [Myr]")

    # Analytic scaling reference lines (guides in lower-right)
    x0, x1 = 2.0, 5.5
    y_base = -0.5
    # Weaver slope 1/5: R ~ M^{1/5}
    ax.plot([x0, x1], [y_base, y_base + (x1 - x0) * 0.2],
            color="grey", ls=":", lw=0.8, alpha=0.7)
    ax.text(x1 + 0.05, y_base + (x1 - x0) * 0.2,
            r"$\propto M^{1/5}$ (Weaver)", fontsize=6, color="grey")
    # Momentum slope 1/4
    ax.plot([x0, x1], [y_base - 0.3, y_base - 0.3 + (x1 - x0) * 0.25],
            color="grey", ls="-.", lw=0.8, alpha=0.7)
    ax.text(x1 + 0.05, y_base - 0.3 + (x1 - x0) * 0.25,
            r"$\propto M^{1/4}$ (momentum)", fontsize=6, color="grey")
    # Stromgren slope 1/3
    ax.plot([x0, x1], [y_base - 0.6, y_base - 0.6 + (x1 - x0) / 3.0],
            color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.text(x1 + 0.05, y_base - 0.6 + (x1 - x0) / 3.0,
            r"$\propto M^{1/3}$ (Str\"omgren)", fontsize=6, color="grey")

    # Observed systems
    _plot_obs_MR(ax, obs_systems)

    ax.set_xlim(1, 7)
    ax.set_ylim(-1, 3)
    ax.set_xlabel(r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R\;/\;\mathrm{pc})$")
    ncl_log = int(round(np.log10(ncl_actual)))
    ax.set_title(rf"Mass–radius isochrones ($n_{{\rm cl}}=10^{ncl_log}\;\rm cm^{{-3}}$)")

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left",
                  fontsize=8, framealpha=0.8)

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    fname = output_dir / f"diag_MR_isochrones_{ncl_actual:.0e}{suffix}.{fmt}"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ======================================================================
# Figure 3: Regime map
# ======================================================================

def plot_fig3(records, ncl, t_regime, obs_systems, output_dir,
              fmt="pdf", tag=""):
    """
    Regime map: scatter of (log M_cl, log R) at a single time,
    colour-coded by TRINITY phase.
    """
    groups = group_by_density(records)
    ncl_actual = find_nearest_density(groups, ncl)
    if ncl_actual is None:
        logger.error("No density groups available for Fig 3.")
        return
    recs = groups[ncl_actual]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect per-phase data
    phase_data = defaultdict(lambda: {"logM": [], "logR": []})

    for rec in recs:
        R_interp = interp_at_time(rec["t"], rec["R"], t_regime)
        ph = phase_at_time(rec["t"], rec["phase"], t_regime)
        if not np.isfinite(R_interp) or R_interp <= 0 or rec["M_star"] <= 0:
            continue
        phase_data[ph]["logM"].append(np.log10(rec["M_star"]))
        phase_data[ph]["logR"].append(np.log10(R_interp))

    # Plot each phase
    legend_handles = []
    for ph in ["energy", "implicit", "transition", "momentum", "unknown"]:
        if ph not in phase_data:
            continue
        d = phase_data[ph]
        colour = PHASE_COLOURS.get(ph, C_BLACK)
        label = PHASE_LABELS.get(ph, ph)
        ax.scatter(d["logM"], d["logR"], c=colour, s=40, alpha=0.8,
                   edgecolors="k", linewidths=0.3, zorder=5, label=label)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colour,
                   markeredgecolor="k", markersize=8, label=label))

    # Observed systems
    _plot_obs_MR(ax, obs_systems)

    ax.set_xlabel(r"$\log_{10}(M_{\rm cl}\;/\;M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R\;/\;\mathrm{pc})$")
    ncl_log = int(round(np.log10(ncl_actual)))
    ax.set_title(rf"Regime map at $t={t_regime}$ Myr "
                 rf"($n_{{\rm cl}}=10^{ncl_log}\;\rm cm^{{-3}}$)")

    # Combined legend
    obs_handles, obs_labels = ax.get_legend_handles_labels()
    # Keep only observed-system entries (those not in legend_handles labels)
    phase_label_set = {h.get_label() for h in legend_handles}
    obs_only = [h for h, l in zip(obs_handles, obs_labels) if l not in phase_label_set]
    all_handles = legend_handles + obs_only
    ax.legend(handles=all_handles, loc="upper left", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    fname = (output_dir /
             f"diag_regime_map_{ncl_actual:.0e}_t{t_regime:.1f}{suffix}.{fmt}")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ======================================================================
# Figure 4: Multi-panel density-sliced (t, R)
# ======================================================================

def plot_fig4(records, obs_systems, output_dir, fmt="pdf", tag="",
              t_ref_vel=1.0):
    """
    One panel per unique nCore; tracks coloured by v_exp at a reference time.
    """
    groups = group_by_density(records)
    densities = sorted(groups.keys())

    if not densities:
        logger.error("No density groups for Fig 4.")
        return

    # Layout: up to 3 columns; add rows as needed
    n_panels = len(densities)
    if n_panels <= 3:
        ncols = n_panels
        nrows = 1
        figsize = (5 * ncols, 5)
    else:
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Pre-compute global v range for consistent colourbar
    v_all = []
    for ncl_val in densities:
        for rec in groups[ncl_val]:
            v = interp_at_time(rec["t"], rec["v_kms"], t_ref_vel)
            if np.isnan(v):
                # Fallback: last available velocity
                valid = rec["v_kms"][rec["v_kms"] > 0]
                v = valid[-1] if len(valid) > 0 else np.nan
            if np.isfinite(v) and v > 0:
                v_all.append(v)

    if not v_all:
        logger.error("No valid velocities for Fig 4 colour mapping.")
        plt.close(fig)
        return

    v_min, v_max = min(v_all), max(v_all)
    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.cm.RdYlBu_r

    # Find which density is closest to each observed system
    obs_best_ncl = {}
    for name, obs in obs_systems.items():
        n = obs.get("n_edge_obs")
        if n is not None and n > 0:
            idx = np.argmin(np.abs(np.log10(densities) - np.log10(n)))
            obs_best_ncl[name] = densities[idx]

    for panel_idx, ncl_val in enumerate(densities):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row, col]

        recs = groups[ncl_val]
        for rec in recs:
            v = interp_at_time(rec["t"], rec["v_kms"], t_ref_vel)
            if np.isnan(v):
                valid = rec["v_kms"][rec["v_kms"] > 0]
                v = valid[-1] if len(valid) > 0 else 0.0
            ax.plot(rec["t"], rec["R"], color=cmap(norm(v)), lw=0.8, alpha=0.8)

        # Observed systems on all panels
        for name, obs in obs_systems.items():
            n = obs.get("n_edge_obs")
            if n is None or n <= 0:
                continue
            is_best = (obs_best_ncl.get(name) == ncl_val)
            alpha_val = 1.0 if is_best else 0.3
            ms_val = obs["ms"] * (1.3 if is_best else 1.0)
            lw_edge = 1.2 if is_best else 0.5
            ax.errorbar(
                obs["t_obs"], obs["R_obs"],
                xerr=obs["sigma_t"], yerr=obs["sigma_R"],
                marker=obs["marker"], ms=ms_val,
                color=C_VERMILLION, markeredgecolor="k",
                markeredgewidth=lw_edge,
                capsize=3, lw=1.0, alpha=alpha_val, zorder=10,
                label=name.replace("_", " ") if is_best else None,
            )

        ax.set_yscale("log")
        ax.set_ylim(0.1, 500)
        ncl_log = np.log10(ncl_val)
        ax.text(0.05, 0.95,
                rf"$n_{{\rm cl}} = 10^{{{ncl_log:.0f}}}\;\rm cm^{{-3}}$",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        if row == nrows - 1:
            ax.set_xlabel(r"$t$ [Myr]")
        if col == 0:
            ax.set_ylabel(r"$R$ [pc]")

    # Turn off unused axes
    for panel_idx in range(n_panels, nrows * ncols):
        row = panel_idx // ncols
        col = panel_idx % ncols
        axes[row, col].set_visible(False)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, shrink=0.8)
    cbar.set_label(rf"$v_{{\rm exp}}(t={t_ref_vel:.0f}\;\rm Myr)$ [km/s]")

    # De-duplicate legend
    handles, labels = [], []
    for a in axes.ravel():
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if handles:
        fig.legend(handles, labels, loc="lower right", fontsize=8, framealpha=0.8)

    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    suffix = f"_{tag}" if tag else ""
    fname = output_dir / f"diag_Rt_density_panels{suffix}.{fmt}"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Diagnostic diagrams for TRINITY Paper IV (Method 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/_calc/diagnostic_diagrams.py -F output/sweep
  python src/_calc/diagnostic_diagrams.py -F output/sweep --ncl 1000 --figures 1,2,3,4
  python src/_calc/diagnostic_diagrams.py -F output/sweep --t-regime 1.0 --fmt png
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True, nargs="+",
        help="Path(s) to one or more sweep output directory trees.",
    )
    parser.add_argument(
        "--ncl", type=float, default=None,
        help=("Core density [cm^-3] for single-density figures (1, 2, 3). "
              "Default: auto-select the most populated density bin."),
    )
    parser.add_argument(
        "--t-regime", type=float, default=1.0,
        help="Snapshot age [Myr] for the regime map (default: 1.0).",
    )
    parser.add_argument(
        "--isochrone-ages", type=str, default="0.1,0.3,1,3,5,10",
        help="Comma-separated isochrone ages [Myr] for Fig 2 (default: 0.1,0.3,1,3,5,10).",
    )
    parser.add_argument(
        "--figures", type=str, default="1,2,3,4",
        help="Comma-separated list of figures to produce (default: 1,2,3,4).",
    )
    parser.add_argument(
        "--fmt", type=str, default="pdf",
        help="Output figure format (default: pdf).",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="String appended to output filenames.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory override (default: fig/diagnostic_diagrams/<folder>/).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Truncate tracks at this time [Myr].",
    )
    return parser


def main(argv=None) -> int:
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
                  else FIG_DIR / "diagnostic_diagrams" / folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse figure selection
    fig_set = set(int(x.strip()) for x in args.figures.split(","))

    # Parse isochrone ages
    iso_ages = [float(x.strip()) for x in args.isochrone_ages.split(",")]

    # Load grid
    records: List[Dict] = []
    for fp in folder_paths:
        records.extend(collect_grid(fp, t_end=args.t_end))
    if not records:
        logger.error("No valid data collected — aborting.")
        return 1

    # Determine default ncl: most populated density bin
    groups = group_by_density(records)
    if args.ncl is not None:
        ncl = args.ncl
    else:
        ncl = max(groups, key=lambda k: len(groups[k]))
        logger.info("Auto-selected ncl=%.1e (%d runs)", ncl, len(groups[ncl]))

    # Produce figures
    if 1 in fig_set:
        logger.info("Generating Figure 1: radius–age fan plot ...")
        plot_fig1(records, ncl, OBSERVED_SYSTEMS, output_dir,
                  fmt=args.fmt, tag=args.tag)

    if 2 in fig_set:
        logger.info("Generating Figure 2: mass–radius isochrones ...")
        plot_fig2(records, ncl, OBSERVED_SYSTEMS, output_dir,
                  fmt=args.fmt, tag=args.tag, isochrone_ages=iso_ages)

    if 3 in fig_set:
        logger.info("Generating Figure 3: regime map ...")
        plot_fig3(records, ncl, args.t_regime, OBSERVED_SYSTEMS, output_dir,
                  fmt=args.fmt, tag=args.tag)

    if 4 in fig_set:
        logger.info("Generating Figure 4: multi-panel density slices ...")
        plot_fig4(records, OBSERVED_SYSTEMS, output_dir,
                  fmt=args.fmt, tag=args.tag)

    logger.info("Done. Output in %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
