#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian inference of stellar cluster mass from observed bubble properties.

Physics background
------------------
Given an observed bubble with radius R, age t, expansion velocity v_exp,
and/or density at the shell edge n_edge, this tool recovers the most likely
stellar cluster mass M_cl (= sfe * M_cloud) by computing a posterior over
the TRINITY simulation grid.

Each grid model θ_i = (M_cloud, sfe, n_core) predicts R(t), v(t), and
n_edge(R) from the stored trajectory and the cloud density profile.
A Gaussian log-likelihood scores each model against the observations,
and a cloud-mass-function prior weights the posterior.

Progressive degeneracy breaking is demonstrated by running inference
with increasing observable sets: (R,t), (R,t,v), (R,t,n), (R,t,v,n).

How to run
----------
This script requires a TRINITY parameter sweep directory (the same kind
used by ``bubble_distribution.py``).  The sweep directory must contain
subfolders with names like ``1e5_sfe010_n1e3/`` or ``m1e7_sfe020_n1e4_PL0/``,
each holding a ``dictionary.jsonl`` output file produced by TRINITY.

**Required argument:**

  -F / --folder   Path to the sweep output directory tree.

**Specifying observables — two modes:**

1. *Pre-stored system* — use ``--system`` to select a known bubble.  The
   script ships with five targets: ``RCW120``, ``Orion_Veil``, ``Carina``,
   ``Rosette``, and ``N49``.  Their observed R, t, v_exp, and n_edge values (with
   uncertainties and literature references) are built in.

2. *Manual observables* — supply ``--R-obs``, ``--sigma-R``, ``--t-obs``,
   ``--sigma-t`` (all four required), plus optionally ``--v-obs``,
   ``--sigma-v``, ``--n-edge-obs``, ``--sigma-n-edge``.  If ``--system``
   is also given, the manual flags override the corresponding system
   defaults, so you can e.g. tweak RCW120's velocity without re-entering
   everything.

**Optional arguments:**

  --cmf-slope FLOAT     Cloud mass function slope dN/dM ~ M^beta
                        (default: -1.8).
  --profile {uniform,sis,steep}
                        Density profile assumption for the n_edge likelihood.
                        ``uniform`` = constant density inside R_cloud,
                        ``sis``     = isothermal sphere n ~ r^{-2},
                        ``steep``   = n ~ r^{-3} (default: uniform).
  --fmt FMT             Figure format: pdf (default), png, svg, etc.
  --tag TAG             String appended to every output filename — useful
                        for distinguishing multiple runs.
  --output-dir DIR      Override the output directory
                        (default: fig/infer_cluster_mass/<folder>/).
  --t-end FLOAT         Truncate all TRINITY tracks at this time [Myr].

**Outputs (saved to fig/infer_cluster_mass/<folder>/ by default):**

  * ``infer_Mcl_posterior_{system}.pdf``  — overlaid marginal PDFs from
    progressive inference stages (R,t → R,t,v → R,t,n → R,t,v,n),
    shaded 68% credible intervals, and optional KDE overlay.
  * ``infer_Mcl_2d_{system}.pdf``        — 2D scatter of log M_cl vs
    log n_core with marker size/colour proportional to posterior weight.
  * ``infer_Mcl_Rt_tracks_{system}.pdf`` — top ~20 highest-weight R(t)
    grid tracks coloured by log M_cl, with the observed (R, t) point.
  * ``infer_Mcl_triangle_{system}.pdf``  — 3×3 corner plot of
    (log M_cl, log n_core, log ε) with weighted histograms and scatter.
  * Console summary table: median log M_cl, 68% CI, and effective sample
    size N_eff for each inference stage.

Examples
--------
**1. Run on a pre-stored system (simplest usage):**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system RCW120

This loads the RCW120 observables (R = 2.25 ± 0.3 pc, t = 0.15 ± 0.05 Myr,
v = 15 ± 2 km/s, n_edge = 3000 ± 500 cm⁻³; Luisi+2021), runs four
progressive inference stages, saves four PDF figures, and prints a summary.

**2. Run on a different pre-stored system with PNG output:**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system Carina --fmt png

Carina has no velocity measurement, so only three stages run: (R,t),
(R,t,n_edge), and the console summary shows N/A for the velocity stage.

**3. Fully manual observables (no pre-stored system):**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --R-obs 10.0 --sigma-R 2.0 \\
        --t-obs 1.0  --sigma-t 0.3 \\
        --v-obs 8.0  --sigma-v 2.0 \\
        --n-edge-obs 500 --sigma-n-edge 100

All four observables are specified, so all four stages run.  The system
name defaults to "custom".

**4. Override a pre-stored system's velocity:**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system RCW120 \\
        --v-obs 20.0 --sigma-v 3.0

Uses RCW120 defaults for R, t, n_edge, but overrides v_exp to 20 ± 3 km/s.

**5. Change the mass-function prior and density profile:**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system Orion_Veil \\
        --cmf-slope -1.7 \\
        --profile sis \\
        --tag cmf17_sis

This uses a shallower CMF slope (-1.7 instead of -1.8) and an isothermal-
sphere density profile for the n_edge likelihood.  The ``--tag`` appends
``_cmf17_sis`` to every output filename so results are not overwritten.

**6. Truncate tracks and redirect output:**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --system N49 \\
        --t-end 5.0 \\
        --output-dir results/N49_test

All grid tracks are truncated at 5 Myr, and figures are saved to
``results/N49_test/`` instead of the default ``fig/`` tree.

**7. Minimal run — radius and age only (no velocity or density):**

.. code-block:: bash

    python src/_calc/infer_cluster_mass.py \\
        -F output/sweep_mCloud_sfe_nCore \\
        --R-obs 5.0 --sigma-R 1.0 \\
        --t-obs 0.5 --sigma-t 0.1

Only the (R, t) stage runs.  The posterior will be broad — this is the
baseline showing the degeneracy that additional observables break.

References
----------
* Luisi+2021, Sci. Adv. 7, eabe9511 — RCW 120
* Harper-Clark & Murray 2009, ApJ 693, 1696 — Carina
* Pabst+2019, Nature 565, 618 — Orion Veil
* Watkins+2023, ApJS 264, 16 — NGC 628 / PHANGS-JWST
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
from matplotlib.lines import Line2D

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src._output.trinity_reader import (
    load_output,
    find_all_simulations,
    parse_simulation_params,
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

# ISM floor density [cm^-3]
N_ISM_FLOOR = 1.0

# Minimum expanding points required
MIN_PTS = 5

# Colourblind-safe palette (Wong 2011)
C_BLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"
C_BLACK = "#000000"

LINESTYLES = ["-", "--", "-.", ":"]

# Mean molecular weight [m_H units] — same as TRINITY default
MU_H = 1.4


# ======================================================================
# Pre-stored observational targets
# ======================================================================

OBSERVED_SYSTEMS = {
    "RCW120": {
        "R_obs": 2.25, "sigma_R": 0.3,
        "t_obs": 0.15, "sigma_t": 0.05,
        "v_obs": 15.0, "sigma_v": 2.0,
        "n_edge_obs": 3000.0, "sigma_n_edge": 500.0,
        "Mcl_lit": 1000.0,
        "sigma_Mcl_lit_dex": 0.3,
        "ref": "Luisi+2021; Deharveng+2009; Martins+2010",
        "note": "Single O8V star CD-38 11636 (~30 Msun); Mcl from IMF scaling",
    },
    "Orion_Veil": {
        "R_obs": 2.0, "sigma_R": 0.3,
        "t_obs": 0.20, "sigma_t": 0.05,
        "v_obs": 13.0, "sigma_v": 2.0,
        "n_edge_obs": 1500.0, "sigma_n_edge": 500.0,
        "Mcl_lit": 1800.0,
        "sigma_Mcl_lit_dex": 0.15,
        "ref": "Pabst+2019 Nature; Pabst+2020 A&A; Da Rio+2014",
        "note": "ONC total stellar mass from census; Veil is half-shell",
    },
    "Carina": {
        "R_obs": 10.0, "sigma_R": 2.0,
        "t_obs": 3.6, "sigma_t": 0.5,
        "v_obs": None, "sigma_v": None,
        "n_edge_obs": 100.0, "sigma_n_edge": 30.0,
        "Mcl_lit": 10000.0,
        "sigma_Mcl_lit_dex": 0.2,
        "ref": "Harper-Clark & Murray 2009; Smith 2006",
        "note": "Trumpler 16; ~70 O stars; age from eta Car evolution",
    },
    "Rosette": {
        "R_obs": 20.0, "sigma_R": 3.0,
        "t_obs": 2.0, "sigma_t": 0.5,
        "v_obs": 15.0, "sigma_v": 3.0,
        "n_edge_obs": 20.0, "sigma_n_edge": 10.0,
        "Mcl_lit": 2000.0,
        "sigma_Mcl_lit_dex": 0.2,
        "ref": "Celnik 1985; Wang+2008; Roman-Zuniga+2008",
        "note": "NGC 2244 cluster; well-studied wind bubble",
    },
    "N49": {
        "R_obs": 55.0, "sigma_R": 10.0,
        "t_obs": 4.0, "sigma_t": 1.0,
        "v_obs": 12.0, "sigma_v": 3.0,
        "n_edge_obs": None, "sigma_n_edge": None,
        "Mcl_lit": None,
        "sigma_Mcl_lit_dex": None,
        "ref": "Watkins+2023; approximate median PHANGS-JWST bubble",
        "note": "Extragalactic; no individual cluster mass available",
    },
}


# ======================================================================
# Step 1: Data loading
# ======================================================================

def collect_grid(folder_path: Path, t_end: float = None) -> List[Dict]:
    """
    Walk sweep directory and extract grid tracks for inference.

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
        t, R, v_au, v_kms, rCloud.
    """
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

        # Deduplicate timestamps
        _, unique_idx = np.unique(t, return_index=True)
        t = t[unique_idx]
        R = R[unique_idx]
        v_au = v_au[unique_idx]

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
            "rCloud": rCloud_val,
        })

    logger.info("Collected %d valid grid runs", len(records))
    return records


# ======================================================================
# Step 2: Density at shell edge
# ======================================================================

def cloud_radius_uniform(M_cloud: float, n_cl: float,
                         mu: float = MU_H) -> float:
    """
    Compute cloud radius for a uniform sphere.

    R_cloud = (3 M / (4 pi rho))^(1/3)
    where rho = n_cl [cm^-3] * mu * m_H [g] converted to Msun/pc^3.

    Parameters
    ----------
    M_cloud : float
        Cloud mass [Msun].
    n_cl : float
        Core number density [cm^-3].
    mu : float
        Mean molecular weight in m_H units.

    Returns
    -------
    float
        Cloud radius [pc].
    """
    # Convert n_cl [cm^-3] to [pc^-3] then to mass density [Msun/pc^3]
    n_au = n_cl * CONV.ndens_cgs2au        # 1/pc^3
    mu_au = mu * CGS.m_H * CONV.g2Msun     # Msun
    rho = n_au * mu_au                      # Msun/pc^3
    return (3.0 * M_cloud / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def density_at_radius_uniform(R: float, n_cl: float,
                              R_cloud: float) -> float:
    """
    Uniform cloud density at radius R.

    Returns n_cl if R <= R_cloud, else N_ISM_FLOOR.
    """
    if R <= R_cloud:
        return n_cl
    return N_ISM_FLOOR


def density_at_radius_powerlaw(R: float, n_cl: float, R_cloud: float,
                               alpha: float = -2.0,
                               r_core: float = None) -> float:
    """
    Power-law cloud density at radius R.

    n(r) = n_cl                           for r <= r_core
    n(r) = n_cl * (r / r_core)^alpha      for r_core < r <= R_cloud
    n(r) = N_ISM_FLOOR                    for r > R_cloud

    Parameters
    ----------
    R : float
        Radius [pc].
    n_cl : float
        Core number density [cm^-3].
    R_cloud : float
        Cloud outer radius [pc].
    alpha : float
        Power-law exponent (default -2, isothermal sphere).
    r_core : float or None
        Core radius [pc]. If None, uses 0.1 * R_cloud
        (matching TRINITY default rCore_fraction=0.1).
    """
    if r_core is None:
        r_core = 0.1 * R_cloud

    if R <= r_core:
        return n_cl
    elif R <= R_cloud:
        return n_cl * (R / r_core) ** alpha
    else:
        return N_ISM_FLOOR


# Registry for profile selection by name
PROFILE_DENSITY = {
    "uniform": density_at_radius_uniform,
    "sis": lambda R, n_cl, R_cloud: density_at_radius_powerlaw(
        R, n_cl, R_cloud, alpha=-2.0),
    "steep": lambda R, n_cl, R_cloud: density_at_radius_powerlaw(
        R, n_cl, R_cloud, alpha=-3.0),
}


def compute_n_edge(R: float, n_cl: float, R_cloud: float,
                   profile: str = "uniform") -> float:
    """
    Evaluate density at the shell edge for a given profile.

    Parameters
    ----------
    R : float
        Shell radius [pc].
    n_cl : float
        Core number density [cm^-3].
    R_cloud : float
        Cloud outer radius [pc].
    profile : str
        Profile name: 'uniform', 'sis', or 'steep'.

    Returns
    -------
    float
        Number density at R [cm^-3].
    """
    fn = PROFILE_DENSITY.get(profile)
    if fn is None:
        logger.warning("Unknown profile '%s', falling back to uniform", profile)
        fn = PROFILE_DENSITY["uniform"]
    return fn(R, n_cl, R_cloud)


# ======================================================================
# Step 3: Posterior computation
# ======================================================================

def _logsumexp(log_vals: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(log_vals)
    if not np.isfinite(a_max):
        return -np.inf
    return a_max + np.log(np.sum(np.exp(log_vals - a_max)))


def compute_posterior_grid(
    records: List[Dict],
    R_obs: float,
    t_obs: float,
    sigma_R: float,
    sigma_t: float,
    v_obs: float = None,
    sigma_v: float = None,
    n_edge_obs: float = None,
    sigma_n_edge: float = None,
    cmf_slope: float = -1.8,
    profile_assumed: str = "uniform",
) -> Optional[Dict]:
    """
    Compute posterior over the TRINITY grid for given observables.

    Parameters
    ----------
    records : list of dict
        Output of collect_grid().
    R_obs : float
        Observed bubble radius [pc].
    t_obs : float
        Observed bubble age [Myr].
    sigma_R : float
        Uncertainty in R [pc].
    sigma_t : float
        Uncertainty in t [Myr].
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
        Density profile assumption for n_edge: 'uniform', 'sis', 'steep'.

    Returns
    -------
    dict or None
        Posterior results with keys: log_Mcl_bins, pdf_Mcl, weights,
        records_eval, median_Mcl, lo_Mcl, hi_Mcl, observables_used,
        pdf_kde, x_kde, N_eff.
    """
    if not records:
        return None

    # Time points for marginalisation over t uncertainty
    t_points = np.array([t_obs - sigma_t, t_obs, t_obs + sigma_t])
    t_weights = np.array([0.25, 0.50, 0.25])
    # Ensure t >= 0
    t_points = np.maximum(t_points, 1e-6)

    n_models = len(records)
    log_weights = np.full(n_models, -np.inf)

    # Build observable description
    obs_parts = [r"$R, t$"]
    if v_obs is not None and sigma_v is not None:
        obs_parts.append(r"$v_{\rm exp}$")
    if n_edge_obs is not None and sigma_n_edge is not None:
        obs_parts.append(r"$n_{\rm edge}$")
    observables_used = ", ".join(obs_parts)

    records_eval = []

    for idx, rec in enumerate(records):
        t_arr = rec["t"]
        R_arr = rec["R"]
        v_kms_arr = rec["v_kms"]

        # Evaluate R_cloud for density computation
        R_cloud = rec["rCloud"]
        if not np.isfinite(R_cloud) or R_cloud <= 0:
            R_cloud = cloud_radius_uniform(rec["mCloud"], rec["nCore"])

        # Marginalise over time uncertainty
        log_L_t = np.full(len(t_points), -np.inf)

        for k, t_eval in enumerate(t_points):
            # Clamp to track range
            if t_eval < t_arr.min() or t_eval > t_arr.max():
                continue

            R_model = np.interp(t_eval, t_arr, R_arr)
            if R_model <= 0 or not np.isfinite(R_model):
                continue

            # Radius likelihood
            log_L = -0.5 * ((R_obs - R_model) / sigma_R) ** 2

            # Velocity likelihood
            if v_obs is not None and sigma_v is not None:
                v_model_au = np.interp(t_eval, t_arr, rec["v_au"])
                v_model = v_model_au * V_AU2KMS
                if np.isfinite(v_model) and v_model > 0:
                    log_L += -0.5 * ((v_obs - v_model) / sigma_v) ** 2

            # Density likelihood (log-space)
            if n_edge_obs is not None and sigma_n_edge is not None:
                n_model = compute_n_edge(R_model, rec["nCore"], R_cloud,
                                         profile=profile_assumed)
                if n_model > 0 and np.isfinite(n_model):
                    log10_n_obs = np.log10(n_edge_obs)
                    log10_n_model = np.log10(n_model)
                    sigma_logn = sigma_n_edge / (n_edge_obs * np.log(10.0))
                    log_L += -0.5 * ((log10_n_obs - log10_n_model) / sigma_logn) ** 2

            log_L_t[k] = log_L

        # Combine time marginalisation via log-sum-exp
        valid_t = np.isfinite(log_L_t)
        if not valid_t.any():
            records_eval.append({
                "idx": idx, "R_model": np.nan, "v_model": np.nan,
                "n_edge_model": np.nan, "log_L": -np.inf,
            })
            continue

        log_L_marginal = _logsumexp(
            log_L_t[valid_t] + np.log(t_weights[valid_t])
        )

        # Prior: cloud mass function dN/dM ~ M^beta => ln pi = (beta+1)*ln(M)
        # SFE and n_cl: log-uniform (flat prior in log space => no contribution)
        log_prior = (cmf_slope + 1.0) * np.log(rec["mCloud"])

        log_weights[idx] = log_L_marginal + log_prior

        # Store model evaluation at central t for diagnostics
        R_model_c = np.interp(t_obs, t_arr, R_arr) if t_arr.min() <= t_obs <= t_arr.max() else np.nan
        v_model_c = np.interp(t_obs, t_arr, rec["v_au"]) * V_AU2KMS if t_arr.min() <= t_obs <= t_arr.max() else np.nan
        n_model_c = compute_n_edge(R_model_c, rec["nCore"], R_cloud,
                                   profile=profile_assumed) if np.isfinite(R_model_c) and R_model_c > 0 else np.nan

        records_eval.append({
            "idx": idx,
            "R_model": R_model_c,
            "v_model": v_model_c,
            "n_edge_model": n_model_c,
            "log_L": log_L_marginal,
        })

    # Normalise weights
    valid = np.isfinite(log_weights) & (log_weights > -1e30)
    if valid.sum() == 0:
        logger.warning("No valid grid models — posterior is empty")
        return None

    log_Z = _logsumexp(log_weights[valid])
    weights = np.zeros(n_models)
    weights[valid] = np.exp(log_weights[valid] - log_Z)

    # Effective sample size
    N_eff = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0.0

    # Marginal PDF over log10(M_cl)
    log_Mcl = np.array([np.log10(rec["M_star"]) for rec in records])

    n_bins = min(40, max(10, int(np.sqrt(valid.sum()))))
    log_Mcl_range = log_Mcl[valid]
    bin_edges = np.linspace(log_Mcl_range.min() - 0.2,
                            log_Mcl_range.max() + 0.2, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pdf_Mcl, _ = np.histogram(log_Mcl, bins=bin_edges, weights=weights)
    dbin = np.diff(bin_edges)
    pdf_Mcl = pdf_Mcl / (dbin * pdf_Mcl.sum() * dbin[0]) if pdf_Mcl.sum() > 0 else pdf_Mcl

    # Normalise so integral over bins = 1
    total = np.sum(pdf_Mcl * dbin)
    if total > 0:
        pdf_Mcl = pdf_Mcl / total

    # Weighted percentiles via cumulative weight interpolation
    sorted_idx = np.argsort(log_Mcl)
    cum_w = np.cumsum(weights[sorted_idx])
    if cum_w[-1] > 0:
        cum_w /= cum_w[-1]
        median_Mcl = np.interp(0.50, cum_w, log_Mcl[sorted_idx])
        lo_Mcl = np.interp(0.16, cum_w, log_Mcl[sorted_idx])
        hi_Mcl = np.interp(0.84, cum_w, log_Mcl[sorted_idx])
    else:
        median_Mcl = lo_Mcl = hi_Mcl = np.nan

    # KDE smoothing (optional)
    pdf_kde = None
    x_kde = None
    try:
        from scipy.stats import gaussian_kde
        w_valid = weights[valid]
        lm_valid = log_Mcl[valid]
        if len(lm_valid) >= 3 and w_valid.sum() > 0:
            kde = gaussian_kde(lm_valid, weights=w_valid, bw_method=0.35)
            x_kde = np.linspace(lm_valid.min() - 0.5, lm_valid.max() + 0.5, 200)
            pdf_kde = kde(x_kde)
    except ImportError:
        pass

    return {
        "log_Mcl_bins": bin_centres,
        "pdf_Mcl": pdf_Mcl,
        "weights": weights,
        "records_eval": records_eval,
        "median_Mcl": median_Mcl,
        "lo_Mcl": lo_Mcl,
        "hi_Mcl": hi_Mcl,
        "observables_used": observables_used,
        "pdf_kde": pdf_kde,
        "x_kde": x_kde,
        "N_eff": N_eff,
    }


# ======================================================================
# Step 4: Progressive degeneracy breaking
# ======================================================================

def run_progressive_inference(
    records: List[Dict],
    R_obs: float, sigma_R: float,
    t_obs: float, sigma_t: float,
    v_obs: float = None, sigma_v: float = None,
    n_edge_obs: float = None, sigma_n_edge: float = None,
    cmf_slope: float = -1.8,
    profile_assumed: str = "uniform",
) -> List[Dict]:
    """
    Run inference with progressively more observables.

    Returns
    -------
    list of dict
        Each dict is a posterior result with an added 'label' key.
    """
    stages = []

    # Stage 1: (R, t) only
    res = compute_posterior_grid(
        records, R_obs, t_obs, sigma_R, sigma_t,
        cmf_slope=cmf_slope, profile_assumed=profile_assumed)
    if res is not None:
        res["label"] = r"$R, t$"
        stages.append(res)

    # Stage 2: (R, t, v_exp)
    if v_obs is not None and sigma_v is not None:
        res = compute_posterior_grid(
            records, R_obs, t_obs, sigma_R, sigma_t,
            v_obs=v_obs, sigma_v=sigma_v,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed)
        if res is not None:
            res["label"] = r"$R, t, v_{\rm exp}$"
            stages.append(res)

    # Stage 3: (R, t, n_edge)
    if n_edge_obs is not None and sigma_n_edge is not None:
        res = compute_posterior_grid(
            records, R_obs, t_obs, sigma_R, sigma_t,
            n_edge_obs=n_edge_obs, sigma_n_edge=sigma_n_edge,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed)
        if res is not None:
            res["label"] = r"$R, t, n_{\rm edge}$"
            stages.append(res)

    # Stage 4: (R, t, v_exp, n_edge)
    if (v_obs is not None and sigma_v is not None
            and n_edge_obs is not None and sigma_n_edge is not None):
        res = compute_posterior_grid(
            records, R_obs, t_obs, sigma_R, sigma_t,
            v_obs=v_obs, sigma_v=sigma_v,
            n_edge_obs=n_edge_obs, sigma_n_edge=sigma_n_edge,
            cmf_slope=cmf_slope, profile_assumed=profile_assumed)
        if res is not None:
            res["label"] = r"$R, t, v_{\rm exp}, n_{\rm edge}$"
            stages.append(res)

    return stages


# ======================================================================
# Step 6: Plotting
# ======================================================================

def plot_posterior(stages: List[Dict], system_name: str,
                  obs_dict: Dict, output_dir: Path,
                  fmt: str = "pdf", tag: str = "") -> None:
    """
    Fig 1: 2x2 subplot of marginal PDFs from progressive inference stages.

    Panel layout (fixed positions):
        (a) R, t           (b) R, t, v_exp
        (c) R, t, n_edge   (d) R, t, v_exp, n_edge

    Panels without data show a centred 'not available' message.
    Falls back to a single-panel figure if only one stage is present.
    """
    if not stages:
        return

    # Panel configuration: fixed slot definitions
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

    # Pre-compute literature mass overlay values
    _has_lit = obs_dict.get("Mcl_lit") is not None
    if _has_lit:
        _log_Mcl_true = np.log10(obs_dict["Mcl_lit"])
        _sigma_lit_dex = obs_dict.get("sigma_Mcl_lit_dex") or 0.3
    _lit_label_used = False

    # Map stages to fixed slots by matching label substrings
    slot_stage = [None] * 4
    for stage in stages:
        lbl = stage.get("label", "")
        # Stage 4 (has both v and n) — check first to avoid false match
        if "v_{\\rm exp}, n" in lbl or "v_{\\rm exp}, n_{\\rm edge}" in lbl:
            slot_stage[3] = stage
        elif "n_{\\rm edge}" in lbl and "v" not in lbl:
            slot_stage[2] = stage
        elif "v_{\\rm exp}" in lbl and "n" not in lbl:
            slot_stage[1] = stage
        else:
            slot_stage[0] = stage

    # ------------------------------------------------------------------
    # Fallback: single-panel figure if only one stage
    # ------------------------------------------------------------------
    if len(stages) == 1:
        stage = stages[0]
        fig, ax = plt.subplots(figsize=(7, 5))
        c = C_BLUE
        bins = stage["log_Mcl_bins"]
        pdf = stage["pdf_Mcl"]
        med, lo, hi = stage["median_Mcl"], stage["lo_Mcl"], stage["hi_Mcl"]
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
                f"$\\log M_{{\\rm cl}} = {med:.2f}"
                f"_{{-{med - lo:.2f}}}^{{+{hi - med:.2f}}}$\n"
                f"$N_{{\\rm eff}} = {n_eff:.1f}$")
        ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="grey", alpha=0.8))

        if obs_dict.get("Mcl_lit") is not None:
            log_Mcl_true = np.log10(obs_dict["Mcl_lit"])
            sigma_dex = obs_dict.get("sigma_Mcl_lit_dex") or 0.3
            ax.axvline(log_Mcl_true, color=C_GREEN, ls="-", lw=2.0,
                       alpha=0.8, label=r"Literature $M_{\rm cl}$", zorder=5)
            ax.axvspan(log_Mcl_true - sigma_dex, log_Mcl_true + sigma_dex,
                       color=C_GREEN, alpha=0.12, zorder=1)
            ax.legend(fontsize=8, framealpha=0.7, loc="best")

        ax.set_xlabel(r"$\log_{10}\,(M_{\rm cl}\;/\;M_\odot)$")
        ax.set_ylabel(r"$p(\log M_{\rm cl} \mid \mathrm{data})$")

        fig.tight_layout()
        suffix = f"_{tag}" if tag else ""
        path = output_dir / f"infer_Mcl_posterior_{system_name}{suffix}.{fmt}"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return

    # ------------------------------------------------------------------
    # Main path: 2x2 subplot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    # Shared axis ranges across all available stages
    x_lo = min(0.0, min(s["log_Mcl_bins"].min() for s in stages))
    x_hi = max(s["log_Mcl_bins"].max() for s in stages)
    y_hi = max(s["pdf_Mcl"].max() for s in stages) * 1.10

    # Reference (stage-1) posterior for ghost overlay
    base_stage = slot_stage[0]

    for slot_idx, (pdef, ax) in enumerate(zip(_PANEL_DEFS, axes.flat)):
        stage = slot_stage[slot_idx]
        c = pdef["color"]

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, y_hi)

        # Panel letter
        ax.text(0.04, 0.96, pdef["panel_label"], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

        if stage is None:
            # Empty panel — centred "not available" message
            ax.text(0.5, 0.5, pdef["missing"], transform=ax.transAxes,
                    fontsize=10, color="0.5", style="italic",
                    ha="center", va="center")
        else:
            bins = stage["log_Mcl_bins"]
            pdf = stage["pdf_Mcl"]
            med = stage["median_Mcl"]
            lo = stage["lo_Mcl"]
            hi = stage["hi_Mcl"]
            n_eff = stage["N_eff"]

            # Ghost: base-stage posterior (panels b, c, d only)
            if slot_idx > 0 and base_stage is not None:
                b_bins = base_stage["log_Mcl_bins"]
                b_pdf = base_stage["pdf_Mcl"]
                ax.fill_between(b_bins, 0, b_pdf, color="0.7",
                                alpha=0.30, step="mid")
                ax.step(b_bins, b_pdf, where="mid", color="0.7",
                        lw=1.0, alpha=0.5,
                        label=r"$(R,\,t)$ only" if slot_idx == 1 else None)

            # Main PDF: filled curve + solid outline
            ax.fill_between(bins, 0, pdf, color=c, alpha=0.25, step="mid")
            ax.step(bins, pdf, where="mid", color=c, lw=1.8)

            # 68% CI shading
            ci_mask = (bins >= lo) & (bins <= hi)
            if ci_mask.any():
                ax.fill_between(bins[ci_mask], 0, pdf[ci_mask],
                                color=c, alpha=0.15, step="mid")

            # Median line + annotation
            ax.axvline(med, color=c, ls="--", lw=1.2, alpha=0.7)
            ax.annotate(f"{med:.2f}", xy=(med, y_hi * 0.92),
                        fontsize=7, color=c, ha="center", va="top")

            # KDE overlay
            if stage.get("pdf_kde") is not None:
                ax.plot(stage["x_kde"], stage["pdf_kde"],
                        color=c, lw=0.8, alpha=0.5)

            # Info box
            info_lines = [stage["label"]]
            if np.isfinite(med):
                info_lines.append(
                    f"$\\log M_{{\\rm cl}} = {med:.2f}"
                    f"_{{-{med - lo:.2f}}}^{{+{hi - med:.2f}}}$")
            info_lines.append(f"$N_{{\\rm eff}} = {n_eff:.1f}$")
            ax.text(0.95, 0.95, "\n".join(info_lines),
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="grey", alpha=0.8))

            # Literature mass overlay
            if _has_lit:
                _lbl = r"Literature $M_{\rm cl}$" if not _lit_label_used else None
                ax.axvline(_log_Mcl_true, color=C_GREEN, ls="-", lw=2.0,
                           alpha=0.8, label=_lbl, zorder=5)
                ax.axvspan(_log_Mcl_true - _sigma_lit_dex,
                           _log_Mcl_true + _sigma_lit_dex,
                           color=C_GREEN, alpha=0.12, zorder=1)
                _lit_label_used = True

            # Legend only where ghost first appears
            if slot_idx == 1 or (slot_idx == 0 and _has_lit):
                ax.legend(fontsize=7, framealpha=0.7, loc="upper left")

        # Shared axis labels / tick-label hiding
        row, col = divmod(slot_idx, 2)
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\log_{10}\,(M_{\rm cl}\;/\;M_\odot)$")
        if col == 1:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"$p(\log M_{\rm cl} \mid \mathrm{data})$")

    # System info box in panel (a)
    obs_lines = [system_name]
    obs_lines.append(
        f"$R = {obs_dict['R_obs']:.2f} \\pm {obs_dict['sigma_R']:.2f}$ pc")
    obs_lines.append(
        f"$t = {obs_dict['t_obs']:.3f} \\pm {obs_dict['sigma_t']:.3f}$ Myr")
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
    path = output_dir / f"infer_Mcl_posterior_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_2d_scatter(stage: Dict, records: List[Dict],
                    system_name: str, output_dir: Path,
                    fmt: str = "pdf", tag: str = "") -> None:
    """
    Fig 2: 2D scatter of log M_cl vs log n_cl, coloured by posterior weight.
    """
    if stage is None:
        return

    weights = stage["weights"]
    med = stage["median_Mcl"]
    lo = stage["lo_Mcl"]
    hi = stage["hi_Mcl"]

    log_Mcl = np.array([np.log10(r["M_star"]) for r in records])
    log_ncl = np.array([np.log10(r["nCore"]) for r in records])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Marker size proportional to weight
    w_norm = weights / weights.max() if weights.max() > 0 else weights
    sizes = 10 + 200 * w_norm

    sc = ax.scatter(log_Mcl, log_ncl, c=w_norm, s=sizes,
                    cmap="inferno", alpha=0.8, edgecolors="grey",
                    linewidths=0.3, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Posterior weight (normalized)", fontsize=10)

    # Median + 68% CI
    ax.axvline(med, color=C_BLUE, ls="--", lw=1.5, alpha=0.7,
               label=f"Median $\\log M_{{\\rm cl}} = {med:.2f}$")
    ax.axvspan(lo, hi, color=C_BLUE, alpha=0.08,
               label=f"68% CI [{lo:.2f}, {hi:.2f}]")

    ax.set_xlabel(r"$\log_{10}\,(M_{\rm cl}\;/\;M_\odot)$")
    ax.set_ylabel(r"$\log_{10}\,(n_{\rm core}\;/\;\mathrm{cm}^{-3})$")
    ax.legend(fontsize=8, framealpha=0.7, loc="best")

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"infer_Mcl_2d_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_Rt_tracks(stage: Dict, records: List[Dict],
                   obs_dict: Dict, system_name: str,
                   output_dir: Path,
                   fmt: str = "pdf", tag: str = "",
                   n_top: int = 20) -> None:
    """
    Fig 3: Top-weighted R(t) tracks, coloured by log M_cl.
    """
    if stage is None:
        return

    weights = stage["weights"]
    log_Mcl = np.array([np.log10(r["M_star"]) for r in records])

    # Select top N by weight
    top_idx = np.argsort(weights)[-n_top:]

    fig, ax = plt.subplots(figsize=(7, 5))

    w_top = weights[top_idx]
    w_max = w_top.max() if w_top.max() > 0 else 1.0

    norm = plt.Normalize(vmin=log_Mcl[top_idx].min(),
                         vmax=log_Mcl[top_idx].max())
    cmap = plt.cm.viridis

    for i in top_idx:
        rec = records[i]
        c = cmap(norm(log_Mcl[i]))
        lw = 0.5 + 2.5 * (weights[i] / w_max)
        ax.plot(rec["t"], rec["R"], color=c, lw=lw, alpha=0.7)

    # Observed point
    ax.errorbar(obs_dict["t_obs"], obs_dict["R_obs"],
                xerr=obs_dict["sigma_t"], yerr=obs_dict["sigma_R"],
                marker="*", ms=15, color=C_VERMILLION, zorder=10,
                capsize=3, label=f"Observed ({system_name})")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}\,M_{\rm cl}$", fontsize=10)

    if obs_dict.get("Mcl_lit") is not None:
        ax.annotate(
            rf"$M_{{\rm cl, lit}} = {obs_dict['Mcl_lit']:.0f}\;M_\odot$",
            xy=(obs_dict["t_obs"], obs_dict["R_obs"]),
            xytext=(0.35, 0.15), textcoords="axes fraction",
            fontsize=8, color=C_GREEN,
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.0),
        )

    ax.set_xlabel(r"$t$ [Myr]")
    ax.set_ylabel(r"$R$ [pc]")
    ax.legend(fontsize=9, framealpha=0.7)

    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"infer_Mcl_Rt_tracks_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_triangle(stage: Dict, records: List[Dict],
                  system_name: str, output_dir: Path,
                  fmt: str = "pdf", tag: str = "") -> None:
    """
    Fig 4: 3x3 corner plot of (log M_cl, log n_cl, log sfe).
    """
    if stage is None:
        return

    weights = stage["weights"]
    log_Mcl = np.array([np.log10(r["M_star"]) for r in records])
    log_ncl = np.array([np.log10(r["nCore"]) for r in records])
    log_sfe = np.array([np.log10(r["sfe"]) for r in records])

    params = [log_Mcl, log_ncl, log_sfe]
    labels = [r"$\log M_{\rm cl}$", r"$\log n_{\rm core}$",
              r"$\log \varepsilon$"]
    n_params = len(params)

    fig, axes = plt.subplots(n_params, n_params, figsize=(8, 8))

    w_norm = weights / weights.max() if weights.max() > 0 else weights

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: 1D weighted histogram
                n_bins = min(30, max(8, int(np.sqrt(len(params[i])))))
                ax.hist(params[i], bins=n_bins, weights=weights,
                        color=C_BLUE, alpha=0.7, edgecolor=C_BLACK,
                        linewidth=0.3, density=True)
            else:
                # Off-diagonal: 2D scatter
                sizes = 5 + 80 * w_norm
                sc = ax.scatter(params[j], params[i], c=w_norm, s=sizes,
                                cmap="inferno", alpha=0.6,
                                edgecolors="none")

            if i == n_params - 1:
                ax.set_xlabel(labels[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=10)
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"{system_name} — {stage.get('label', 'all obs')}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"infer_Mcl_triangle_{system_name}{suffix}.{fmt}"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ======================================================================
# Step 7: Console summary
# ======================================================================

def print_inference_summary(system_name: str, stages: List[Dict],
                            obs_dict: Dict) -> None:
    """Print inference results table."""
    print()
    print("=" * 76)
    print(f"  INFERENCE SUMMARY — System: {system_name}")
    print("=" * 76)

    # Print observables
    print(f"  R = {obs_dict['R_obs']:.2f} +/- {obs_dict['sigma_R']:.2f} pc")
    print(f"  t = {obs_dict['t_obs']:.3f} +/- {obs_dict['sigma_t']:.3f} Myr")
    if obs_dict.get("v_obs") is not None:
        print(f"  v = {obs_dict['v_obs']:.1f} +/- {obs_dict['sigma_v']:.1f} km/s")
    if obs_dict.get("n_edge_obs") is not None:
        print(f"  n_edge = {obs_dict['n_edge_obs']:.0f} +/- {obs_dict['sigma_n_edge']:.0f} cm^-3")
    if obs_dict.get("Mcl_lit") is not None:
        log_lit = np.log10(obs_dict["Mcl_lit"])
        sig_dex = obs_dict.get("sigma_Mcl_lit_dex")
        if sig_dex is not None:
            print(f"  Literature M_cl = {obs_dict['Mcl_lit']:.0f} Msun"
                  f"  (log = {log_lit:.2f} +/- {sig_dex:.2f} dex)")
        else:
            print(f"  Literature M_cl = {obs_dict['Mcl_lit']:.0f} Msun"
                  f"  (log = {log_lit:.2f})")
    if obs_dict.get("ref"):
        print(f"  Ref: {obs_dict['ref']}")

    print()
    print(f"  {'Observables':<35s} {'median log M_cl':>16s} "
          f"{'68% CI (dex)':>14s} {'N_eff':>7s}")
    print("  " + "-" * 74)
    for stage in stages:
        med = stage["median_Mcl"]
        lo = stage["lo_Mcl"]
        hi = stage["hi_Mcl"]
        n_eff = stage["N_eff"]
        label = stage.get("label", "?")
        # Strip LaTeX for console
        label_clean = label.replace(r"$", "").replace(r"\rm ", "")
        label_clean = label_clean.replace(r"_{\rm exp}", "_exp")
        label_clean = label_clean.replace(r"_{\rm edge}", "_edge")
        label_clean = label_clean.replace("\\", "")

        if np.isfinite(med):
            ci_str = f"+{hi - med:.2f} / -{med - lo:.2f}"
            print(f"  {label_clean:<35s} {med:>16.3f} {ci_str:>14s} "
                  f"{n_eff:>7.1f}")
        else:
            print(f"  {label_clean:<35s} {'N/A':>16s} {'N/A':>14s} "
                  f"{'N/A':>7s}")

    print("=" * 76)
    print()


# ======================================================================
# Step 8: CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Bayesian inference of cluster mass from bubble observables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python infer_cluster_mass.py -F /path/to/sweep --system RCW120
  python infer_cluster_mass.py -F /path/to/sweep \\
      --R-obs 10.0 --sigma-R 2.0 --t-obs 1.0 --sigma-t 0.3
  python infer_cluster_mass.py -F /path/to/sweep --system Carina \\
      --cmf-slope -1.7 --profile sis --fmt png --tag test1
        """,
    )
    parser.add_argument(
        "-F", "--folder", required=True,
        help="Path to the sweep output directory tree (required).",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        choices=list(OBSERVED_SYSTEMS.keys()),
        help="Pre-stored observational target.",
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
        "--t-obs", type=float, default=None,
        help="Observed bubble age [Myr].",
    )
    parser.add_argument(
        "--sigma-t", type=float, default=None,
        help="Uncertainty in t [Myr].",
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
        "--Mcl-lit", type=float, default=None,
        help="Literature cluster mass [Msun] for validation overlay.",
    )
    parser.add_argument(
        "--sigma-Mcl-lit-dex", type=float, default=None,
        help="Uncertainty in log10(Mcl_lit) [dex].",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory override (default: fig/infer_cluster_mass/<folder>/).",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Truncate tracks at this time [Myr].",
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
    output_dir = Path(args.output_dir) if args.output_dir else FIG_DIR / "infer_cluster_mass" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve observables: system defaults + manual overrides
    obs = {}
    if args.system is not None:
        obs = dict(OBSERVED_SYSTEMS[args.system])
    system_name = args.system or "custom"

    # Manual overrides
    if args.R_obs is not None:
        obs["R_obs"] = args.R_obs
    if args.sigma_R is not None:
        obs["sigma_R"] = args.sigma_R
    if args.t_obs is not None:
        obs["t_obs"] = args.t_obs
    if args.sigma_t is not None:
        obs["sigma_t"] = args.sigma_t
    if args.v_obs is not None:
        obs["v_obs"] = args.v_obs
    if args.sigma_v is not None:
        obs["sigma_v"] = args.sigma_v
    if args.n_edge_obs is not None:
        obs["n_edge_obs"] = args.n_edge_obs
    if args.sigma_n_edge is not None:
        obs["sigma_n_edge"] = args.sigma_n_edge
    if args.Mcl_lit is not None:
        obs["Mcl_lit"] = args.Mcl_lit
    if args.sigma_Mcl_lit_dex is not None:
        obs["sigma_Mcl_lit_dex"] = args.sigma_Mcl_lit_dex

    # Validate required observables
    if "R_obs" not in obs or "sigma_R" not in obs:
        logger.error("R_obs and sigma_R are required (use --system or --R-obs)")
        return 1
    if "t_obs" not in obs or "sigma_t" not in obs:
        logger.error("t_obs and sigma_t are required (use --system or --t-obs)")
        return 1

    # Collect grid
    records = collect_grid(folder_path, t_end=args.t_end)
    if not records:
        logger.error("No valid grid data collected — aborting.")
        return 1

    # Run progressive inference
    stages = run_progressive_inference(
        records,
        R_obs=obs["R_obs"], sigma_R=obs["sigma_R"],
        t_obs=obs["t_obs"], sigma_t=obs["sigma_t"],
        v_obs=obs.get("v_obs"), sigma_v=obs.get("sigma_v"),
        n_edge_obs=obs.get("n_edge_obs"),
        sigma_n_edge=obs.get("sigma_n_edge"),
        cmf_slope=args.cmf_slope,
        profile_assumed=args.profile,
    )

    if not stages:
        logger.error("All inference stages returned None — aborting.")
        return 1

    # Figures
    plot_posterior(stages, system_name, obs, output_dir,
                   fmt=args.fmt, tag=args.tag)

    # Use the most constrained stage for 2D, tracks, and triangle
    best_stage = stages[-1]
    plot_2d_scatter(best_stage, records, system_name, output_dir,
                    fmt=args.fmt, tag=args.tag)
    plot_Rt_tracks(best_stage, records, obs, system_name, output_dir,
                   fmt=args.fmt, tag=args.tag)
    plot_triangle(best_stage, records, system_name, output_dir,
                  fmt=args.fmt, tag=args.tag)

    # Console summary
    print_inference_summary(system_name, stages, obs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
