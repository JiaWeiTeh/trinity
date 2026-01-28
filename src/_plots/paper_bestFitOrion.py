#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best-fit analysis for Orion Nebula (M42) parameter sweep.

This script analyzes TRINITY simulation parameter sweeps to find models
that best match M42 observables.

Observational Constraints for M42 (Orion Nebula)
------------------------------------------------
Directly Observed (Primary Constraints):
- v_expansion: 13 +/- 2 km/s (Pabst et al. 2020, [CII] observations)
- M_shell: 2000 +/- 500 M_sun (Pabst et al. 2019, range: 1300-2600)
- Age (dynamical): 0.2 +/- 0.05 Myr (Pabst et al. 2019, 2020)
- R_shell: 4 +/- 0.5 pc (Pabst et al. 2019)

Stellar Constraint (Critical):
- M_star: 34 +/- 5 M_sun (theta^1 Ori C, O7V spectral type)
- Q(H0): ~10^49 s^-1 (Martins et al. 2005 calibration)

The stellar mass constrains the (mCloud, sfe) relationship via:
    M_star = sfe * mCloud / (1 - sfe)

Analysis Modes
--------------
Mode A: Visualization Dimensionality
  - 2D Mode (--mode 2d or --nCore <value>):
    Creates separate plots for each nCore value
    Each plot is a (mCloud x sfe) heatmap

  - 3D Mode (--mode 3d):
    Treats nCore as a full third parameter
    Creates: faceted heatmaps, 3D scatter, marginal projections

Mode B: Free Parameter Analysis
  - Standard mode: Minimize total chi^2 across all observables
  - Free parameter mode (--free-param <param>):
    Fix some observables, report range/distribution of "free" parameter

Produces:
- Chi-squared heatmaps (mCloud x sfe grid)
- Trajectory comparisons (v(t), M(t) with observation point marked)
- Residual contours with 1-sigma, 2-sigma, 3-sigma levels
- Ranking table (top N best-fit combinations)
- 3D scatter plots and marginal projections (3D mode)

@author: Jia Wei Teh
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal, Tuple, Any
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src._output.trinity_reader import (
    load_output, find_all_simulations, organize_simulations_for_grid,
    get_unique_ndens, parse_simulation_params, resolve_data_input
)

print("...analyzing best-fit models for Orion Nebula (M42)")

# Unit conversion: 1 km/s ~ 1.0227 pc/Myr
PC_MYR_TO_KM_S = 1.0 / 1.0227

# --- Output directory
FIG_DIR = Path(__file__).parent.parent.parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load matplotlib style if available
try:
    plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trinity.mplstyle'))
except:
    pass


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class ObservationalConstraints:
    """M42/Orion Nebula observational constraints."""
    # Expansion velocity
    v_obs: float = 13.0          # km/s
    v_err: float = 2.0           # km/s

    # Shell mass
    M_shell_obs: float = 2000.0  # M_sun
    M_shell_err: float = 500.0   # M_sun

    # Dynamical age
    t_obs: float = 0.2           # Myr
    t_err: float = 0.05          # Myr

    # Shell radius
    R_obs: float = 4.0           # pc
    R_err: float = 0.5           # pc

    # Stellar mass (derived constraint)
    Mstar_obs: float = 34.0      # M_sun
    Mstar_err: float = 5.0       # M_sun


@dataclass
class AnalysisConfig:
    """Configuration for best-fit analysis."""
    # Visualization mode
    mode: Literal['2d', '3d'] = '2d'

    # Which observables to constrain (include in chi^2)
    constrain_v: bool = True
    constrain_M_shell: bool = True
    constrain_t: bool = True
    constrain_R: bool = False
    constrain_Mstar: bool = True

    # Free parameter (excluded from chi^2, reported as output)
    free_param: Optional[Literal['v', 'M_shell', 't', 'R']] = None

    # Filter by nCore (for 2D mode)
    nCore_filter: Optional[str] = None

    # Observational constraints
    obs: ObservationalConstraints = field(default_factory=ObservationalConstraints)

    def get_constraint_string(self) -> str:
        """Build a string describing active constraints."""
        constraints = []
        if self.constrain_v and self.free_param != 'v':
            constraints.append(f"v={self.obs.v_obs:.0f}+/-{self.obs.v_err:.0f} km/s")
        if self.constrain_M_shell and self.free_param != 'M_shell':
            constraints.append(f"M_shell={self.obs.M_shell_obs:.0f}+/-{self.obs.M_shell_err:.0f} M_sun")
        if self.constrain_t and self.free_param != 't':
            constraints.append(f"t={self.obs.t_obs:.2f}+/-{self.obs.t_err:.2f} Myr")
        if self.constrain_R and self.free_param != 'R':
            constraints.append(f"R={self.obs.R_obs:.1f}+/-{self.obs.R_err:.1f} pc")
        if self.constrain_Mstar:
            constraints.append(f"M_star={self.obs.Mstar_obs:.0f}+/-{self.obs.Mstar_err:.0f} M_sun")
        return ", ".join(constraints)

    def count_dof(self) -> int:
        """Count degrees of freedom (number of active constraints)."""
        dof = 0
        if self.constrain_v and self.free_param != 'v':
            dof += 1
        if self.constrain_M_shell and self.free_param != 'M_shell':
            dof += 1
        if self.constrain_t and self.free_param != 't':
            dof += 1
        if self.constrain_R and self.free_param != 'R':
            dof += 1
        if self.constrain_Mstar:
            dof += 1
        return dof


@dataclass
class SimulationResult:
    """Results from a single simulation."""
    # Input parameters
    path: str
    folder: str
    mCloud: str
    sfe: str
    nCore: str
    mCloud_float: float
    sfe_float: float
    nCore_float: float

    # Derived stellar mass
    Mstar: float

    # Simulation outputs at t_obs
    t_actual: float
    v_kms: float
    M_shell: float
    R2: float

    # Chi^2 components
    chi2_v: float
    chi2_M: float
    chi2_t: float
    chi2_R: float
    chi2_Mstar: float
    chi2_total: float

    # Residuals (in units of sigma)
    delta_v: float
    delta_M: float
    delta_t: float
    delta_R: float
    delta_Mstar: float

    # Full time series (for trajectory plots)
    t_full: Optional[np.ndarray] = None
    v_full_kms: Optional[np.ndarray] = None
    M_shell_full: Optional[np.ndarray] = None
    R_full: Optional[np.ndarray] = None

    # Free parameter value (if applicable)
    free_value: Optional[float] = None


# =============================================================================
# Delta chi^2 Thresholds for Confidence Regions
# =============================================================================

def get_delta_chi2_thresholds(n_dof: int) -> Dict[str, float]:
    """
    Get delta chi^2 thresholds for confidence regions.

    Parameters
    ----------
    n_dof : int
        Number of degrees of freedom

    Returns
    -------
    dict
        Dictionary with '1sigma', '2sigma', '3sigma' thresholds
    """
    # Delta chi^2 for p=0.683 (1sigma), p=0.954 (2sigma), p=0.997 (3sigma)
    thresholds = {
        1: {'1sigma': 1.00, '2sigma': 4.00, '3sigma': 9.00},
        2: {'1sigma': 2.30, '2sigma': 6.18, '3sigma': 11.83},
        3: {'1sigma': 3.53, '2sigma': 8.02, '3sigma': 14.16},
        4: {'1sigma': 4.72, '2sigma': 9.72, '3sigma': 16.25},
        5: {'1sigma': 5.89, '2sigma': 11.31, '3sigma': 18.21},
    }
    return thresholds.get(n_dof, thresholds[2])


# =============================================================================
# Core Functions
# =============================================================================

def compute_stellar_mass(mCloud, sfe):
    """
    Compute stellar mass from cloud parameters.

    M_star = sfe * mCloud / (1 - sfe)

    Parameters
    ----------
    mCloud : float or array-like
        Cloud mass in M_sun
    sfe : float or array-like
        Star formation efficiency (0-1)

    Returns
    -------
    float or ndarray
        Stellar mass in M_sun
    """
    mCloud = np.asarray(mCloud)
    sfe = np.asarray(sfe)

    # Use np.where to handle both scalar and array inputs
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(sfe >= 1.0, np.inf, sfe * mCloud / (1.0 - sfe))

    # Return scalar if inputs were scalar
    if result.ndim == 0:
        return float(result)
    return result


def compute_chi2(sim_values: dict, config: AnalysisConfig) -> dict:
    """
    Compute chi^2 with configurable constraints.

    Parameters
    ----------
    sim_values : dict
        Must contain: v_kms, M_shell, t_actual, R2, mCloud, sfe
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    dict
        Contains chi2_total, individual chi2 terms, residuals, Mstar, free_value
    """
    obs = config.obs
    chi2_terms = {}
    residuals = {}

    # Velocity
    if np.isfinite(sim_values['v_kms']) and obs.v_err > 0:
        delta_v = (sim_values['v_kms'] - obs.v_obs) / obs.v_err
    else:
        delta_v = np.nan
    residuals['delta_v'] = delta_v
    if config.constrain_v and config.free_param != 'v' and np.isfinite(delta_v):
        chi2_terms['chi2_v'] = delta_v ** 2
    else:
        chi2_terms['chi2_v'] = 0.0

    # Shell mass
    if np.isfinite(sim_values['M_shell']) and obs.M_shell_err > 0:
        delta_M = (sim_values['M_shell'] - obs.M_shell_obs) / obs.M_shell_err
    else:
        delta_M = np.nan
    residuals['delta_M'] = delta_M
    if config.constrain_M_shell and config.free_param != 'M_shell' and np.isfinite(delta_M):
        chi2_terms['chi2_M'] = delta_M ** 2
    else:
        chi2_terms['chi2_M'] = 0.0

    # Age
    if np.isfinite(sim_values['t_actual']) and obs.t_err > 0:
        delta_t = (sim_values['t_actual'] - obs.t_obs) / obs.t_err
    else:
        delta_t = np.nan
    residuals['delta_t'] = delta_t
    if config.constrain_t and config.free_param != 't' and np.isfinite(delta_t):
        chi2_terms['chi2_t'] = delta_t ** 2
    else:
        chi2_terms['chi2_t'] = 0.0

    # Radius
    if np.isfinite(sim_values['R2']) and obs.R_err > 0:
        delta_R = (sim_values['R2'] - obs.R_obs) / obs.R_err
    else:
        delta_R = np.nan
    residuals['delta_R'] = delta_R
    if config.constrain_R and config.free_param != 'R' and np.isfinite(delta_R):
        chi2_terms['chi2_R'] = delta_R ** 2
    else:
        chi2_terms['chi2_R'] = 0.0

    # Stellar mass (derived from mCloud and sfe)
    Mstar = compute_stellar_mass(sim_values['mCloud'], sim_values['sfe'])
    if np.isfinite(Mstar) and obs.Mstar_err > 0:
        delta_Mstar = (Mstar - obs.Mstar_obs) / obs.Mstar_err
    else:
        delta_Mstar = np.nan
    residuals['delta_Mstar'] = delta_Mstar
    if config.constrain_Mstar and np.isfinite(delta_Mstar):
        chi2_terms['chi2_Mstar'] = delta_Mstar ** 2
    else:
        chi2_terms['chi2_Mstar'] = 0.0

    chi2_total = sum(chi2_terms.values())

    # Extract free parameter value if specified
    free_value = None
    if config.free_param == 'v':
        free_value = sim_values['v_kms']
    elif config.free_param == 'M_shell':
        free_value = sim_values['M_shell']
    elif config.free_param == 't':
        free_value = sim_values['t_actual']
    elif config.free_param == 'R':
        free_value = sim_values['R2']

    return {
        'chi2_total': chi2_total,
        **chi2_terms,
        **residuals,
        'Mstar': Mstar,
        'free_value': free_value,
    }


def find_matching_time(output, config: AnalysisConfig) -> dict:
    """
    Find time(s) when simulation best matches constrained observables.

    Use when 't' is the free parameter to find optimal age.

    Parameters
    ----------
    output : TrinityOutput
        Loaded simulation output
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    dict
        Contains t_best, t_range_1sigma, chi2_min, values at t_best
    """
    obs = config.obs

    t_arr = output.get('t_now')
    v_arr = output.get('v2') * PC_MYR_TO_KM_S  # Convert to km/s
    M_arr = output.get('shell_mass')
    R_arr = output.get('R2')

    if t_arr is None or len(t_arr) == 0:
        return None

    # Compute chi^2 at each timestep (excluding time constraint)
    chi2_arr = np.zeros_like(t_arr, dtype=float)

    if config.constrain_v and v_arr is not None:
        chi2_arr += ((v_arr - obs.v_obs) / obs.v_err) ** 2
    if config.constrain_M_shell and M_arr is not None:
        chi2_arr += ((M_arr - obs.M_shell_obs) / obs.M_shell_err) ** 2
    if config.constrain_R and R_arr is not None:
        chi2_arr += ((R_arr - obs.R_obs) / obs.R_err) ** 2

    # Handle NaN/inf
    chi2_arr = np.where(np.isfinite(chi2_arr), chi2_arr, np.inf)

    # Find minimum
    idx_min = np.argmin(chi2_arr)
    t_best = t_arr[idx_min]
    chi2_min = chi2_arr[idx_min]

    # 1-sigma range depends on number of constrained DOF
    n_constrained = sum([config.constrain_v, config.constrain_M_shell, config.constrain_R])
    thresholds = get_delta_chi2_thresholds(max(1, n_constrained))
    delta_chi2_1sigma = thresholds['1sigma']

    mask_1sigma = chi2_arr < chi2_min + delta_chi2_1sigma
    if np.any(mask_1sigma):
        t_range_1sigma = (t_arr[mask_1sigma].min(), t_arr[mask_1sigma].max())
    else:
        t_range_1sigma = (t_best, t_best)

    return {
        't_best': t_best,
        't_range_1sigma': t_range_1sigma,
        'chi2_min': chi2_min,
        'v_at_best': v_arr[idx_min] if v_arr is not None else np.nan,
        'M_at_best': M_arr[idx_min] if M_arr is not None else np.nan,
        'R_at_best': R_arr[idx_min] if R_arr is not None else np.nan,
    }


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_simulation_at_time(data_path: Path, config: AnalysisConfig) -> Optional[SimulationResult]:
    """
    Load simulation and extract observables at specified time.

    Parameters
    ----------
    data_path : Path
        Path to simulation data file
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    SimulationResult or None
        SimulationResult object or None if loading fails
    """
    try:
        output = load_output(data_path)

        if len(output) == 0:
            return None

        # Parse folder name for parameters
        folder_name = data_path.parent.name
        params = parse_simulation_params(folder_name)

        if params is None:
            return None

        mCloud_str = params['mCloud']
        sfe_str = params['sfe']
        nCore_str = params['ndens']

        # Convert to floats
        mCloud_float = float(mCloud_str)
        sfe_float = int(sfe_str) / 100.0  # sfe is stored as percentage (e.g., "020" -> 0.20)
        nCore_float = float(nCore_str)

        # Get snapshot closest to observation time
        t_obs = config.obs.t_obs
        snap = output.get_at_time(t_obs, mode='closest', quiet=True)

        if snap is None:
            return None

        t_actual = snap.get('t_now', np.nan)
        v2_pcMyr = snap.get('v2', np.nan)
        shell_mass = snap.get('shell_mass', np.nan)
        R2 = snap.get('R2', np.nan)

        # Convert velocity to km/s
        v_kms = v2_pcMyr * PC_MYR_TO_KM_S if np.isfinite(v2_pcMyr) else np.nan

        # Build sim_values dict for chi2 calculation
        sim_values = {
            'v_kms': v_kms,
            'M_shell': shell_mass,
            't_actual': t_actual,
            'R2': R2,
            'mCloud': mCloud_float,
            'sfe': sfe_float,
        }

        # Compute chi2 and residuals
        chi2_result = compute_chi2(sim_values, config)

        # Get full time series for trajectory plots
        t_full = output.get('t_now')
        v2_full = output.get('v2')
        M_shell_full = output.get('shell_mass')
        R_full = output.get('R2')

        v_full_kms = v2_full * PC_MYR_TO_KM_S if v2_full is not None else None

        return SimulationResult(
            path=str(data_path),
            folder=folder_name,
            mCloud=mCloud_str,
            sfe=sfe_str,
            nCore=nCore_str,
            mCloud_float=mCloud_float,
            sfe_float=sfe_float,
            nCore_float=nCore_float,
            Mstar=chi2_result['Mstar'],
            t_actual=t_actual,
            v_kms=v_kms,
            M_shell=shell_mass,
            R2=R2,
            chi2_v=chi2_result['chi2_v'],
            chi2_M=chi2_result['chi2_M'],
            chi2_t=chi2_result['chi2_t'],
            chi2_R=chi2_result['chi2_R'],
            chi2_Mstar=chi2_result['chi2_Mstar'],
            chi2_total=chi2_result['chi2_total'],
            delta_v=chi2_result['delta_v'],
            delta_M=chi2_result['delta_M'],
            delta_t=chi2_result['delta_t'],
            delta_R=chi2_result['delta_R'],
            delta_Mstar=chi2_result['delta_Mstar'],
            t_full=t_full,
            v_full_kms=v_full_kms,
            M_shell_full=M_shell_full,
            R_full=R_full,
            free_value=chi2_result['free_value'],
        )

    except Exception as e:
        print(f"  Error loading {data_path}: {e}")
        return None


def load_sweep_results(folder_path: Path, config: AnalysisConfig) -> List[SimulationResult]:
    """
    Load all simulations from a sweep folder and compute chi2 values.

    Parameters
    ----------
    folder_path : Path
        Path to sweep folder
    config : AnalysisConfig
        Analysis configuration

    Returns
    -------
    list
        List of SimulationResult objects sorted by chi2
    """
    folder_path = Path(folder_path)
    sim_files = find_all_simulations(folder_path)

    if not sim_files:
        print(f"No simulation files found in {folder_path}")
        return []

    results = []

    for sim_path in sim_files:
        # Parse simulation parameters from folder name
        folder_name = sim_path.parent.name
        params = parse_simulation_params(folder_name)

        if params is None:
            continue

        ndens = params['ndens']

        # Apply nCore filter if specified
        if config.nCore_filter and ndens != config.nCore_filter:
            continue

        # Load simulation data
        result = load_simulation_at_time(sim_path, config)

        if result is not None:
            results.append(result)

    # Sort by chi2
    results.sort(key=lambda x: x.chi2_total)

    return results


# =============================================================================
# 2D MODE: Separate plots for each nCore
# =============================================================================

def plot_chi2_heatmap_2d(results: List[SimulationResult], config: AnalysisConfig,
                         output_dir: Path, nCore_value: str):
    """
    Create chi^2 heatmap for a single nCore value.

    Produces (mCloud x sfe) grid with:
    - Color: chi^2 value
    - Star marker: best-fit cell
    - Contours: M_star = 30, 34, 40 M_sun lines
    - Annotations: chi^2 values in cells

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory for figures
    nCore_value : str
        nCore value for filtering
    """
    # Filter for this nCore
    data = [r for r in results if r.nCore == nCore_value]

    if not data:
        print(f"  No results for nCore = {nCore_value}")
        return

    # Get unique values
    mCloud_list = sorted(set(r.mCloud for r in data), key=float)
    sfe_list = sorted(set(r.sfe for r in data), key=lambda x: int(x))

    nrows, ncols = len(mCloud_list), len(sfe_list)

    # Build grids
    chi2_grid = np.full((nrows, ncols), np.nan)
    v_grid = np.full_like(chi2_grid, np.nan)
    M_grid = np.full_like(chi2_grid, np.nan)
    Mstar_grid = np.full_like(chi2_grid, np.nan)

    # Build lookup
    lookup = {(r.mCloud, r.sfe): r for r in data}

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            if (mCloud, sfe) in lookup:
                r = lookup[(mCloud, sfe)]
                chi2_grid[i, j] = r.chi2_total
                v_grid[i, j] = r.v_kms
                M_grid[i, j] = r.M_shell
                Mstar_grid[i, j] = r.Mstar

    # Find best fit for this nCore
    best_result = min(data, key=lambda x: x.chi2_total)
    best_i = mCloud_list.index(best_result.mCloud)
    best_j = sfe_list.index(best_result.sfe)

    # Get chi2 thresholds
    n_dof = config.count_dof()
    thresholds = get_delta_chi2_thresholds(n_dof)

    # Create figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # --- Panel 1: Chi^2 heatmap ---
    ax1 = axes[0]
    chi2_min = np.nanmin(chi2_grid)
    chi2_max = np.nanmax(chi2_grid)

    cmap = plt.cm.RdYlGn_r
    im1 = ax1.imshow(chi2_grid, cmap=cmap, aspect='auto',
                     norm=mcolors.LogNorm(vmin=max(0.1, chi2_min), vmax=min(1000, chi2_max)))

    cbar1 = plt.colorbar(im1, ax=ax1, label=r'$\chi^2_{\rm total}$')

    # Add confidence level lines to colorbar
    for level_name, chi2_level in [('1sigma', thresholds['1sigma']),
                                    ('2sigma', thresholds['2sigma']),
                                    ('3sigma', thresholds['3sigma'])]:
        if chi2_min < chi2_level < chi2_max:
            cbar1.ax.axhline(y=chi2_level, color='k', linestyle='--', linewidth=0.8)

    # Mark best-fit cell with gold star
    ax1.plot(best_j, best_i, marker='*', markersize=20, color='gold',
             markeredgecolor='k', markeredgewidth=1.5, zorder=10)

    # Add chi2 values as text
    for i in range(nrows):
        for j in range(ncols):
            chi2_val = chi2_grid[i, j]
            if np.isfinite(chi2_val):
                text_color = 'white' if chi2_val > 10 else 'black'
                ax1.text(j, i, f'{chi2_val:.1f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')
            else:
                ax1.text(j, i, 'X', ha='center', va='center',
                        fontsize=10, color='gray', alpha=0.5)

    ax1.set_xticks(range(ncols))
    ax1.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=9)
    ax1.set_yticks(range(nrows))
    ax1.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=9)
    ax1.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
    ax1.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
    ax1.set_title(r'$\chi^2$ Heatmap')

    # --- Panel 2: Velocity heatmap ---
    ax2 = axes[1]
    v_min = np.nanmin(v_grid)
    v_max = np.nanmax(v_grid)

    im2 = ax2.imshow(v_grid, cmap='coolwarm', aspect='auto',
                     vmin=min(v_min, config.obs.v_obs - 3*config.obs.v_err),
                     vmax=max(v_max, config.obs.v_obs + 3*config.obs.v_err))
    cbar2 = plt.colorbar(im2, ax=ax2, label='v [km/s]')

    # Add observed velocity lines
    cbar2.ax.axhline(y=config.obs.v_obs, color='k', linestyle='-', linewidth=2)
    cbar2.ax.axhline(y=config.obs.v_obs - config.obs.v_err, color='k', linestyle='--', linewidth=1)
    cbar2.ax.axhline(y=config.obs.v_obs + config.obs.v_err, color='k', linestyle='--', linewidth=1)

    # Add velocity values as text
    for i in range(nrows):
        for j in range(ncols):
            v_val = v_grid[i, j]
            if np.isfinite(v_val):
                ax2.text(j, i, f'{v_val:.1f}', ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')

    ax2.set_xticks(range(ncols))
    ax2.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=9)
    ax2.set_yticks(range(nrows))
    ax2.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=9)
    ax2.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
    ax2.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
    ax2.set_title(f'Velocity (obs: {config.obs.v_obs:.0f} km/s)')

    # Mark best-fit
    ax2.plot(best_j, best_i, marker='*', markersize=15, color='gold',
             markeredgecolor='k', markeredgewidth=1, zorder=10)

    # --- Panel 3: Shell mass heatmap ---
    ax3 = axes[2]
    M_min = np.nanmin(M_grid)
    M_max = np.nanmax(M_grid)

    im3 = ax3.imshow(M_grid, cmap='coolwarm', aspect='auto',
                     vmin=min(M_min, config.obs.M_shell_obs - 3*config.obs.M_shell_err),
                     vmax=max(M_max, config.obs.M_shell_obs + 3*config.obs.M_shell_err))
    cbar3 = plt.colorbar(im3, ax=ax3, label=r'$M_{\rm shell}$ [$M_\odot$]')

    # Add observed shell mass lines
    cbar3.ax.axhline(y=config.obs.M_shell_obs, color='k', linestyle='-', linewidth=2)
    cbar3.ax.axhline(y=config.obs.M_shell_obs - config.obs.M_shell_err, color='k', linestyle='--', linewidth=1)
    cbar3.ax.axhline(y=config.obs.M_shell_obs + config.obs.M_shell_err, color='k', linestyle='--', linewidth=1)

    # Add mass values as text
    for i in range(nrows):
        for j in range(ncols):
            M_val = M_grid[i, j]
            if np.isfinite(M_val):
                ax3.text(j, i, f'{M_val:.0f}', ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')

    ax3.set_xticks(range(ncols))
    ax3.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=9)
    ax3.set_yticks(range(nrows))
    ax3.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=9)
    ax3.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
    ax3.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
    ax3.set_title(f'Shell Mass (obs: {config.obs.M_shell_obs:.0f} M$_\\odot$)')

    # Mark best-fit
    ax3.plot(best_j, best_i, marker='*', markersize=15, color='gold',
             markeredgecolor='k', markeredgewidth=1, zorder=10)

    fig.suptitle(f'M42 Best-Fit Analysis: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$\n'
                 f'Best: mCloud={best_result.mCloud}, sfe={best_result.sfe_float:.2f}, '
                 f'$M_\\star$={best_result.Mstar:.1f} M$_\\odot$, $\\chi^2$={best_result.chi2_total:.2f}',
                 fontsize=12, y=1.02)

    plt.tight_layout()

    out_pdf = output_dir / f'bestfit_n{nCore_value}_heatmap.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_trajectory_comparison_2d(results: List[SimulationResult], config: AnalysisConfig,
                                   output_dir: Path, nCore_value: str, top_n: int = 5):
    """
    Overlay v(t) and M_shell(t) trajectories for a single nCore.

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    nCore_value : str
        nCore value for filtering
    top_n : int
        Number of best-fit models to show
    """
    # Filter for this nCore
    data = [r for r in results if r.nCore == nCore_value]

    if not data:
        return

    # Sort by chi2 and take top_n
    data_sorted = sorted(data, key=lambda x: x.chi2_total)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    ax_v, ax_m = axes

    # Color map for different simulations
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_sorted)))

    for i, r in enumerate(data_sorted):
        if r.t_full is None:
            continue

        t = r.t_full
        v = r.v_full_kms
        M = r.M_shell_full
        label = f"{r.mCloud}_sfe{r.sfe} (M$_\\star$={r.Mstar:.0f}, $\\chi^2$={r.chi2_total:.1f})"

        # Velocity trajectory
        if v is not None:
            ax_v.plot(t, v, color=colors[i], lw=1.5, label=label, alpha=0.8)

        # Mass trajectory
        if M is not None:
            ax_m.plot(t, M, color=colors[i], lw=1.5, label=label, alpha=0.8)

    obs = config.obs

    # Mark observation point with error bars
    ax_v.errorbar(obs.t_obs, obs.v_obs, xerr=obs.t_err, yerr=obs.v_err,
                  fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                  label='M42 Observed', zorder=10, markeredgecolor='k')

    ax_m.errorbar(obs.t_obs, obs.M_shell_obs, xerr=obs.t_err, yerr=obs.M_shell_err,
                  fmt='s', color='red', markersize=12, capsize=5, capthick=2,
                  label='M42 Observed', zorder=10, markeredgecolor='k')

    # Shade observation uncertainty region
    ax_v.axhspan(obs.v_obs - obs.v_err, obs.v_obs + obs.v_err,
                 alpha=0.2, color='red', zorder=1)
    ax_v.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.2, color='blue', zorder=1)

    ax_m.axhspan(obs.M_shell_obs - obs.M_shell_err, obs.M_shell_obs + obs.M_shell_err,
                 alpha=0.2, color='red', zorder=1)
    ax_m.axvspan(obs.t_obs - obs.t_err, obs.t_obs + obs.t_err,
                 alpha=0.2, color='blue', zorder=1)

    # Labels and formatting
    ax_v.set_xlabel('Time [Myr]')
    ax_v.set_ylabel('Shell Velocity [km/s]')
    ax_v.set_title('Velocity Evolution')
    ax_v.legend(loc='upper right', fontsize=8)
    ax_v.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_v.set_ylim(0, None)
    ax_v.grid(True, alpha=0.3)

    ax_m.set_xlabel('Time [Myr]')
    ax_m.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax_m.set_title('Shell Mass Evolution')
    ax_m.legend(loc='upper right', fontsize=8)
    ax_m.set_xlim(0, max(0.5, obs.t_obs * 2.5))
    ax_m.set_ylim(0, None)
    ax_m.grid(True, alpha=0.3)

    fig.suptitle(f'M42 Trajectory Comparison: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$',
                 fontsize=14, y=1.02)

    plt.tight_layout()

    out_pdf = output_dir / f'bestfit_n{nCore_value}_trajectories.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_residual_contours_2d(results: List[SimulationResult], config: AnalysisConfig,
                               output_dir: Path, nCore_value: str):
    """
    Plot residual contours in velocity-mass space with confidence regions.

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    nCore_value : str
        nCore value for filtering
    """
    # Filter for this nCore
    data = [r for r in results if r.nCore == nCore_value]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot each simulation as a point
    v_vals = [r.v_kms for r in data if np.isfinite(r.v_kms)]
    M_vals = [r.M_shell for r in data if np.isfinite(r.M_shell)]
    chi2_vals = [r.chi2_total for r in data if np.isfinite(r.chi2_total)]
    Mstar_vals = [r.Mstar for r in data if np.isfinite(r.Mstar)]

    if not v_vals:
        print(f"  No valid data points for residual plot (nCore={nCore_value})")
        return

    # Scatter plot colored by chi2
    scatter = ax.scatter(v_vals, M_vals, c=chi2_vals, cmap='RdYlGn_r',
                         norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                         s=100, edgecolors='k', linewidths=0.5, zorder=5)

    # Draw confidence ellipses centered on observation
    obs = config.obs
    n_dof = config.count_dof()
    thresholds = get_delta_chi2_thresholds(n_dof)
    theta = np.linspace(0, 2 * np.pi, 100)

    for level_name, color, label in [('1sigma', 'green', r'1$\sigma$'),
                                      ('2sigma', 'orange', r'2$\sigma$'),
                                      ('3sigma', 'red', r'3$\sigma$')]:
        delta_chi2 = thresholds[level_name]
        scale = np.sqrt(delta_chi2)
        v_ellipse = obs.v_obs + scale * obs.v_err * np.cos(theta)
        M_ellipse = obs.M_shell_obs + scale * obs.M_shell_err * np.sin(theta)
        ax.plot(v_ellipse, M_ellipse, color=color, lw=2, linestyle='--',
                label=f'{label} ($\\Delta\\chi^2={delta_chi2:.2f}$)')

    # Mark observation point
    ax.errorbar(obs.v_obs, obs.M_shell_obs,
                xerr=obs.v_err, yerr=obs.M_shell_err,
                fmt='s', color='red', markersize=15, capsize=5, capthick=2,
                label='M42 Observed', zorder=10, markeredgecolor='k', markeredgewidth=2)

    # Mark best-fit
    best = min(data, key=lambda r: r.chi2_total)
    if np.isfinite(best.v_kms) and np.isfinite(best.M_shell):
        ax.plot(best.v_kms, best.M_shell, marker='*', markersize=25,
                color='gold', markeredgecolor='k', markeredgewidth=1.5, zorder=15,
                label=f'Best fit ($\\chi^2={best.chi2_total:.2f}$)')

    # Add annotations for top 3
    data_sorted = sorted(data, key=lambda r: r.chi2_total)
    for i, r in enumerate(data_sorted[:3]):
        if np.isfinite(r.v_kms) and np.isfinite(r.M_shell):
            ax.annotate(f"{r.mCloud}\nsfe{r.sfe}\nM$_\\star$={r.Mstar:.0f}",
                        (r.v_kms, r.M_shell),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=8, alpha=0.8)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label=r'$\chi^2$')

    # Labels
    ax.set_xlabel('Shell Velocity [km/s]')
    ax.set_ylabel(r'Shell Mass [$M_\odot$]')
    ax.set_title(f'M42 Parameter Space ($n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$)\n'
                 f'at t = {obs.t_obs} Myr')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_pdf = output_dir / f'bestfit_n{nCore_value}_residuals.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_Mstar_constraint_2d(results: List[SimulationResult], config: AnalysisConfig,
                              output_dir: Path, nCore_value: str):
    """
    Plot stellar mass constraint diagram showing M_star contours.

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    nCore_value : str
        nCore value for filtering
    """
    # Filter for this nCore
    data = [r for r in results if r.nCore == nCore_value]

    if not data:
        return

    # Get unique values
    mCloud_list = sorted(set(r.mCloud for r in data), key=float)
    sfe_list = sorted(set(r.sfe for r in data), key=lambda x: int(x))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Create mesh for contour plotting
    mCloud_vals = np.array([float(m) for m in mCloud_list])
    sfe_vals = np.array([int(s)/100.0 for s in sfe_list])

    mCloud_mesh, sfe_mesh = np.meshgrid(mCloud_vals, sfe_vals)
    Mstar_mesh = compute_stellar_mass(mCloud_mesh, sfe_mesh)

    # Plot chi2 as background
    nrows, ncols = len(mCloud_list), len(sfe_list)
    chi2_grid = np.full((nrows, ncols), np.nan)
    lookup = {(r.mCloud, r.sfe): r for r in data}

    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            if (mCloud, sfe) in lookup:
                chi2_grid[i, j] = lookup[(mCloud, sfe)].chi2_total

    # Plot heatmap
    im = ax.imshow(chi2_grid, cmap='RdYlGn_r', aspect='auto',
                   norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                   extent=[-0.5, ncols-0.5, nrows-0.5, -0.5])

    # Add M_star contours
    # Note: need to transpose because imshow has different axis order
    obs = config.obs
    Mstar_levels = [obs.Mstar_obs - 2*obs.Mstar_err,
                    obs.Mstar_obs - obs.Mstar_err,
                    obs.Mstar_obs,
                    obs.Mstar_obs + obs.Mstar_err,
                    obs.Mstar_obs + 2*obs.Mstar_err]

    # Create contour grid aligned with heatmap
    j_coords, i_coords = np.meshgrid(range(ncols), range(nrows))
    Mstar_grid = np.full((nrows, ncols), np.nan)
    for i, mCloud in enumerate(mCloud_list):
        for j, sfe in enumerate(sfe_list):
            mCloud_f = float(mCloud)
            sfe_f = int(sfe) / 100.0
            Mstar_grid[i, j] = compute_stellar_mass(mCloud_f, sfe_f)

    contour = ax.contour(j_coords, i_coords, Mstar_grid, levels=Mstar_levels,
                         colors=['gray', 'blue', 'black', 'blue', 'gray'],
                         linestyles=['--', '--', '-', '--', '--'],
                         linewidths=[1, 1.5, 2, 1.5, 1])
    ax.clabel(contour, inline=True, fontsize=9, fmt='M$_\\star$=%.0f')

    # Mark best-fit
    best = min(data, key=lambda r: r.chi2_total)
    best_i = mCloud_list.index(best.mCloud)
    best_j = sfe_list.index(best.sfe)
    ax.plot(best_j, best_i, marker='*', markersize=25, color='gold',
            markeredgecolor='k', markeredgewidth=1.5, zorder=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$\chi^2_{\rm total}$')

    # Labels
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list])
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list])
    ax.set_xlabel(r'Star Formation Efficiency ($\epsilon$)')
    ax.set_ylabel(r'Cloud Mass ($M_{\rm cloud}$ [$M_\odot$])')
    ax.set_title(f'M42 Stellar Mass Constraint: $n_{{\\rm core}}$ = {nCore_value} cm$^{{-3}}$\n'
                 f'M$_\\star$(obs) = {obs.Mstar_obs:.0f} +/- {obs.Mstar_err:.0f} M$_\\odot$ '
                 f'(black line = {obs.Mstar_obs:.0f} M$_\\odot$)')

    plt.tight_layout()

    out_pdf = output_dir / f'bestfit_n{nCore_value}_Mstar.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# 3D MODE: Full parameter space visualization
# =============================================================================

def plot_chi2_heatmap_3d_faceted(results: List[SimulationResult], config: AnalysisConfig,
                                  output_dir: Path):
    """
    Create side-by-side heatmaps for all nCore values.

    Panels arranged horizontally, sharing colorbar.
    Global best-fit marked across all panels.

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    """
    nCore_list = sorted(set(r.nCore for r in results), key=float)
    n_panels = len(nCore_list)

    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 5), dpi=150,
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    # Find global best
    best = min(results, key=lambda r: r.chi2_total)

    # Shared colorbar normalization
    chi2_all = [r.chi2_total for r in results if np.isfinite(r.chi2_total)]
    if not chi2_all:
        return
    vmin, vmax = max(0.1, min(chi2_all)), min(1000, max(chi2_all))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    for ax, nCore in zip(axes, nCore_list):
        # Filter for this nCore
        data = [r for r in results if r.nCore == nCore]

        if not data:
            continue

        # Get unique values
        mCloud_list = sorted(set(r.mCloud for r in data), key=float)
        sfe_list = sorted(set(r.sfe for r in data), key=lambda x: int(x))

        nrows, ncols = len(mCloud_list), len(sfe_list)
        chi2_grid = np.full((nrows, ncols), np.nan)

        lookup = {(r.mCloud, r.sfe): r for r in data}
        for i, mCloud in enumerate(mCloud_list):
            for j, sfe in enumerate(sfe_list):
                if (mCloud, sfe) in lookup:
                    chi2_grid[i, j] = lookup[(mCloud, sfe)].chi2_total

        im = ax.imshow(chi2_grid, cmap=cmap, aspect='auto', norm=norm)

        # Mark if global best is in this panel
        if best.nCore == nCore:
            best_i = mCloud_list.index(best.mCloud)
            best_j = sfe_list.index(best.sfe)
            ax.plot(best_j, best_i, marker='*', markersize=20, color='gold',
                    markeredgecolor='k', markeredgewidth=1.5, zorder=10)

        # Add chi2 values
        for i in range(nrows):
            for j in range(ncols):
                chi2_val = chi2_grid[i, j]
                if np.isfinite(chi2_val):
                    text_color = 'white' if chi2_val > 10 else 'black'
                    ax.text(j, i, f'{chi2_val:.1f}', ha='center', va='center',
                            fontsize=7, color=text_color)

        ax.set_xticks(range(ncols))
        ax.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_list], fontsize=8, rotation=45)
        ax.set_yticks(range(nrows))
        ax.set_yticklabels([f'{float(m):.0e}' for m in mCloud_list], fontsize=8)
        ax.set_xlabel(r'SFE ($\epsilon$)', fontsize=10)
        ax.set_title(f'$n_{{\\rm core}}$ = {nCore}', fontsize=11)

    axes[0].set_ylabel(r'$M_{\rm cloud}$ [$M_\odot$]', fontsize=10)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax,
                        label=r'$\chi^2_{\rm total}$')

    fig.suptitle(f'M42 Best-Fit Analysis (3D Mode)\n'
                 f'Global Best: mCloud={best.mCloud}, sfe={best.sfe_float:.2f}, '
                 f'nCore={best.nCore}, $\\chi^2$={best.chi2_total:.2f}',
                 fontsize=12, y=1.02)

    out_pdf = output_dir / 'bestfit_3d_faceted.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_3d_scatter(results: List[SimulationResult], config: AnalysisConfig,
                    output_dir: Path):
    """
    3D scatter plot in (log mCloud, sfe, log nCore) space.

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Extract data
    mCloud = np.array([np.log10(r.mCloud_float) for r in results])
    sfe = np.array([r.sfe_float for r in results])
    nCore = np.array([np.log10(r.nCore_float) for r in results])
    chi2 = np.array([r.chi2_total for r in results])

    # Filter finite values
    mask = np.isfinite(chi2)
    mCloud, sfe, nCore, chi2 = mCloud[mask], sfe[mask], nCore[mask], chi2[mask]

    if len(chi2) == 0:
        return

    # Size inversely proportional to chi2
    sizes = 300 / (chi2 + 1)
    sizes = np.clip(sizes, 20, 300)

    sc = ax.scatter(mCloud, sfe, nCore, c=chi2, s=sizes,
                    cmap='viridis_r', alpha=0.7, edgecolors='k', linewidths=0.5,
                    norm=mcolors.LogNorm(vmin=max(0.1, chi2.min()), vmax=min(100, chi2.max())))

    # Mark best
    best = min(results, key=lambda r: r.chi2_total)
    ax.scatter([np.log10(best.mCloud_float)], [best.sfe_float],
               [np.log10(best.nCore_float)], c='red', s=400, marker='*',
               edgecolors='black', linewidths=2, zorder=10, label='Best fit')

    ax.set_xlabel(r'log$_{10}$(M$_{\rm cloud}$/M$_\odot$)', fontsize=11)
    ax.set_ylabel(r'$\epsilon$ (SFE)', fontsize=11)
    ax.set_zlabel(r'log$_{10}$(n$_{\rm core}$/cm$^{-3}$)', fontsize=11)

    plt.colorbar(sc, ax=ax, label=r'$\chi^2_{\rm total}$', shrink=0.6, pad=0.1)

    ax.set_title(f'M42 Parameter Space (3D)\n'
                 f'Best: mCloud={best.mCloud}, sfe={best.sfe_float:.2f}, '
                 f'nCore={best.nCore}, $\\chi^2$={best.chi2_total:.2f}',
                 fontsize=12)

    ax.legend(loc='upper left')

    out_pdf = output_dir / 'bestfit_3d_scatter.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


def plot_marginal_projections(results: List[SimulationResult], config: AnalysisConfig,
                               output_dir: Path):
    """
    Show 2D projections with third parameter marginalized (min chi^2).

    Three panels:
    1. mCloud x sfe (min over nCore)
    2. mCloud x nCore (min over sfe)
    3. sfe x nCore (min over mCloud)

    Parameters
    ----------
    results : List[SimulationResult]
        List of simulation results
    config : AnalysisConfig
        Analysis configuration
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)

    # Get unique values
    mCloud_vals = sorted(set(r.mCloud for r in results), key=float)
    sfe_vals = sorted(set(r.sfe for r in results), key=lambda x: int(x))
    nCore_vals = sorted(set(r.nCore for r in results), key=float)

    cmap = plt.cm.RdYlGn_r
    chi2_all = [r.chi2_total for r in results if np.isfinite(r.chi2_total)]
    if not chi2_all:
        return
    vmin, vmax = max(0.1, min(chi2_all)), min(100, max(chi2_all))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # --- Panel 1: mCloud x sfe (min over nCore) ---
    ax1 = axes[0]
    grid1 = np.full((len(mCloud_vals), len(sfe_vals)), np.nan)
    for i, mCloud in enumerate(mCloud_vals):
        for j, sfe in enumerate(sfe_vals):
            matches = [r for r in results if r.mCloud == mCloud and r.sfe == sfe]
            if matches:
                grid1[i, j] = min(r.chi2_total for r in matches)

    im1 = ax1.imshow(grid1, cmap=cmap, aspect='auto', norm=norm)
    ax1.set_xticks(range(len(sfe_vals)))
    ax1.set_xticklabels([f'{int(s)/100:.2f}' for s in sfe_vals], fontsize=8, rotation=45)
    ax1.set_yticks(range(len(mCloud_vals)))
    ax1.set_yticklabels([f'{float(m):.0e}' for m in mCloud_vals], fontsize=8)
    ax1.set_xlabel(r'SFE ($\epsilon$)')
    ax1.set_ylabel(r'$M_{\rm cloud}$ [$M_\odot$]')
    ax1.set_title('min over $n_{\\rm core}$')
    plt.colorbar(im1, ax=ax1, label=r'$\chi^2$')

    # --- Panel 2: mCloud x nCore (min over sfe) ---
    ax2 = axes[1]
    grid2 = np.full((len(mCloud_vals), len(nCore_vals)), np.nan)
    for i, mCloud in enumerate(mCloud_vals):
        for j, nCore in enumerate(nCore_vals):
            matches = [r for r in results if r.mCloud == mCloud and r.nCore == nCore]
            if matches:
                grid2[i, j] = min(r.chi2_total for r in matches)

    im2 = ax2.imshow(grid2, cmap=cmap, aspect='auto', norm=norm)
    ax2.set_xticks(range(len(nCore_vals)))
    ax2.set_xticklabels([f'{float(n):.0e}' for n in nCore_vals], fontsize=8, rotation=45)
    ax2.set_yticks(range(len(mCloud_vals)))
    ax2.set_yticklabels([f'{float(m):.0e}' for m in mCloud_vals], fontsize=8)
    ax2.set_xlabel(r'$n_{\rm core}$ [cm$^{-3}$]')
    ax2.set_ylabel(r'$M_{\rm cloud}$ [$M_\odot$]')
    ax2.set_title('min over SFE')
    plt.colorbar(im2, ax=ax2, label=r'$\chi^2$')

    # --- Panel 3: sfe x nCore (min over mCloud) ---
    ax3 = axes[2]
    grid3 = np.full((len(sfe_vals), len(nCore_vals)), np.nan)
    for i, sfe in enumerate(sfe_vals):
        for j, nCore in enumerate(nCore_vals):
            matches = [r for r in results if r.sfe == sfe and r.nCore == nCore]
            if matches:
                grid3[i, j] = min(r.chi2_total for r in matches)

    im3 = ax3.imshow(grid3, cmap=cmap, aspect='auto', norm=norm)
    ax3.set_xticks(range(len(nCore_vals)))
    ax3.set_xticklabels([f'{float(n):.0e}' for n in nCore_vals], fontsize=8, rotation=45)
    ax3.set_yticks(range(len(sfe_vals)))
    ax3.set_yticklabels([f'{int(s)/100:.2f}' for s in sfe_vals], fontsize=8)
    ax3.set_xlabel(r'$n_{\rm core}$ [cm$^{-3}$]')
    ax3.set_ylabel(r'SFE ($\epsilon$)')
    ax3.set_title('min over $M_{\\rm cloud}$')
    plt.colorbar(im3, ax=ax3, label=r'$\chi^2$')

    fig.suptitle('M42 Marginal Projections (minimum $\\chi^2$ over marginalized parameter)',
                 fontsize=12, y=1.02)

    plt.tight_layout()

    out_pdf = output_dir / 'bestfit_marginal_projections.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# =============================================================================
# Output Functions
# =============================================================================

def print_ranking_table(results: List[SimulationResult], config: AnalysisConfig,
                        top_n: int = 10):
    """Print ranked table of best-fit parameter combinations."""
    sorted_results = sorted(results, key=lambda r: r.chi2_total)[:top_n]

    # Get thresholds
    n_dof = config.count_dof()
    thresholds = get_delta_chi2_thresholds(n_dof)

    free_str = f"FREE: {config.free_param}" if config.free_param else "None"

    print("\n" + "=" * 100)
    print(f"TOP {top_n} BEST-FIT PARAMETER COMBINATIONS")
    print("=" * 100)
    print(f"Constraints: {config.get_constraint_string()}")
    print(f"Free parameter: {free_str}")
    print(f"DOF: {n_dof}")
    print("-" * 100)
    print(f"{'Rank':>4} {'mCloud':>10} {'sfe':>6} {'nCore':>8} {'M_star':>8} "
          f"{'v_sim':>8} {'M_sim':>8} {'t_sim':>8} {'chi2':>8} {'Sig':>5}")
    print(f"{'':>4} {'[M_sun]':>10} {'':>6} {'[cm-3]':>8} {'[M_sun]':>8} "
          f"{'[km/s]':>8} {'[M_sun]':>8} {'[Myr]':>8} {'':>8} {'':>5}")
    print("-" * 100)

    for i, r in enumerate(sorted_results, 1):
        # Determine significance level
        if r.chi2_total < thresholds['1sigma']:
            sig = "***"
        elif r.chi2_total < thresholds['2sigma']:
            sig = "**"
        elif r.chi2_total < thresholds['3sigma']:
            sig = "*"
        else:
            sig = ""

        print(f"{i:>4} {r.mCloud:>10} {r.sfe_float:>6.2f} {r.nCore:>8} "
              f"{r.Mstar:>8.1f} {r.v_kms:>8.1f} {r.M_shell:>8.0f} "
              f"{r.t_actual:>8.3f} {r.chi2_total:>8.2f} {sig:>5}")

    print("-" * 100)
    print(f"Legend: *** = 1-sigma (dchi2<{thresholds['1sigma']:.2f}), "
          f"** = 2-sigma (dchi2<{thresholds['2sigma']:.2f}), "
          f"* = 3-sigma (dchi2<{thresholds['3sigma']:.2f})")
    print("=" * 100)

    # Best fit summary
    best = sorted_results[0]
    obs = config.obs
    print(f"\n** BEST FIT:")
    print(f"   mCloud = {best.mCloud} M_sun")
    print(f"   sfe = {best.sfe_float:.2f}")
    print(f"   nCore = {best.nCore} cm^-3")
    print(f"   M_star = {best.Mstar:.1f} M_sun (obs: {obs.Mstar_obs:.0f}+/-{obs.Mstar_err:.0f})")
    print(f"   v = {best.v_kms:.1f} km/s (obs: {obs.v_obs:.0f}+/-{obs.v_err:.0f})")
    print(f"   M_shell = {best.M_shell:.0f} M_sun (obs: {obs.M_shell_obs:.0f}+/-{obs.M_shell_err:.0f})")
    print(f"   t = {best.t_actual:.3f} Myr (target: {obs.t_obs:.2f}+/-{obs.t_err:.2f})")
    print(f"   R = {best.R2:.2f} pc (obs: {obs.R_obs:.1f}+/-{obs.R_err:.1f})")
    print(f"   chi2 = {best.chi2_total:.2f}")

    # Residuals breakdown
    print(f"\n   Residuals (in sigma):")
    print(f"     delta_v = {best.delta_v:+.2f} sigma")
    print(f"     delta_M_shell = {best.delta_M:+.2f} sigma")
    print(f"     delta_t = {best.delta_t:+.2f} sigma")
    print(f"     delta_M_star = {best.delta_Mstar:+.2f} sigma")

    if config.free_param:
        free_values = [r.free_value for r in sorted_results if r.free_value is not None]
        if free_values:
            print(f"\n   {config.free_param} range (top {top_n}): "
                  f"{min(free_values):.2f} - {max(free_values):.2f}")


def print_parameter_sensitivity(results: List[SimulationResult], top_n: int = 10):
    """Analyze which parameter values dominate the top fits."""
    sorted_results = sorted(results, key=lambda r: r.chi2_total)[:top_n]

    from collections import Counter

    mCloud_counts = Counter(r.mCloud for r in sorted_results)
    sfe_counts = Counter(r.sfe for r in sorted_results)
    nCore_counts = Counter(r.nCore for r in sorted_results)

    print(f"\nPARAMETER SENSITIVITY (top {top_n} fits):")

    if mCloud_counts:
        best_mCloud = mCloud_counts.most_common(1)[0]
        print(f"   Best mCloud: {best_mCloud[0]} M_sun ({best_mCloud[1]}/{top_n} of top fits)")

    if sfe_counts:
        best_sfe = sfe_counts.most_common(1)[0]
        print(f"   Best sfe: {int(best_sfe[0])/100:.2f} ({best_sfe[1]}/{top_n} of top fits)")

    if nCore_counts:
        best_nCore = nCore_counts.most_common(1)[0]
        print(f"   Best nCore: {best_nCore[0]} cm^-3 ({best_nCore[1]}/{top_n} of top fits)")

    # Stellar mass distribution
    Mstar_vals = [r.Mstar for r in sorted_results if np.isfinite(r.Mstar)]
    if Mstar_vals:
        print(f"   M_star range: {min(Mstar_vals):.1f} - {max(Mstar_vals):.1f} M_sun")
        print(f"   M_star mean: {np.mean(Mstar_vals):.1f} M_sun")


# =============================================================================
# Main Function
# =============================================================================

def main(folder_path: str, output_dir: str = None, config: AnalysisConfig = None):
    """
    Run best-fit analysis.

    Parameters
    ----------
    folder_path : str
        Path to sweep folder
    output_dir : str, optional
        Output directory (default: {folder}/analysis/)
    config : AnalysisConfig, optional
        Analysis configuration
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir) if output_dir else folder_path / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AnalysisConfig()

    print(f"\nLoading sweep results from: {folder_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {config.mode.upper()}")
    if config.nCore_filter:
        print(f"nCore filter: {config.nCore_filter}")
    if config.free_param:
        print(f"Free parameter: {config.free_param}")
    print(f"Constraints: {config.get_constraint_string()}")

    # Load all simulations
    results = load_sweep_results(folder_path, config)

    print(f"\nLoaded {len(results)} simulations")

    if not results:
        print("No valid simulations found!")
        return

    # Print ranking
    print_ranking_table(results, config, top_n=10)
    print_parameter_sensitivity(results, top_n=10)

    # Generate plots based on mode
    print("\nGenerating plots...")

    if config.mode == '2d':
        # 2D mode: separate plots per nCore
        if config.nCore_filter:
            nCore_list = [config.nCore_filter]
        else:
            nCore_list = sorted(set(r.nCore for r in results), key=float)

        for nCore in nCore_list:
            print(f"\n  Processing nCore = {nCore}...")
            plot_chi2_heatmap_2d(results, config, output_dir, nCore)
            plot_trajectory_comparison_2d(results, config, output_dir, nCore)
            plot_residual_contours_2d(results, config, output_dir, nCore)
            plot_Mstar_constraint_2d(results, config, output_dir, nCore)

    else:  # 3D mode
        print("\n  Creating 3D visualizations...")
        plot_chi2_heatmap_3d_faceted(results, config, output_dir)
        plot_3d_scatter(results, config, output_dir)
        plot_marginal_projections(results, config, output_dir)

    print(f"\n** All plots saved to: {output_dir}")


# =============================================================================
# Command-line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINITY parameter sweep best-fit analysis for M42 (Orion Nebula)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2D mode (separate plots per nCore)
  python paper_bestFitOrion.py --folder sweep_orion/ --mode 2d

  # 2D mode with specific nCore
  python paper_bestFitOrion.py --folder sweep_orion/ --nCore 1e4

  # 3D mode (full parameter space)
  python paper_bestFitOrion.py --folder sweep_orion/ --mode 3d

  # Free parameter: find optimal age
  python paper_bestFitOrion.py --folder sweep_orion/ --free-param t

  # Custom stellar mass constraint
  python paper_bestFitOrion.py --folder sweep_orion/ --Mstar 40 --Mstar-err 8

  # Without stellar mass constraint
  python paper_bestFitOrion.py --folder sweep_orion/ --no-Mstar

  # Without age constraint (if your simulations run longer)
  python paper_bestFitOrion.py --folder sweep_orion/ --no-t

  # Also constrain radius
  python paper_bestFitOrion.py --folder sweep_orion/ --constrain-R

Observational Constraints (default values):
  v_expansion:  13 +/- 2 km/s      (Pabst et al. 2020)
  M_shell:      2000 +/- 500 M_sun (Pabst et al. 2019)
  Age:          0.2 +/- 0.05 Myr   (Pabst et al. 2019, 2020)
  R_shell:      4 +/- 0.5 pc       (Pabst et al. 2019) - not constrained by default
  M_star:       34 +/- 5 M_sun     (theta^1 Ori C)

Stellar Mass Constraint:
  M_star = sfe * mCloud / (1 - sfe)
  This derived constraint helps identify physically plausible (mCloud, sfe) combinations.
        """
    )

    # Required
    parser.add_argument('--folder', '-F', required=True,
                        help='Path to sweep output folder')

    # Output
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: {folder}/analysis/)')

    # Mode selection
    parser.add_argument('--mode', '-m', choices=['2d', '3d'], default='2d',
                        help='Visualization mode: 2d (separate plots per nCore) or 3d (full space)')
    parser.add_argument('--nCore', '-n', default=None,
                        help='Filter by nCore value (e.g., "1e4"). Implies 2D mode.')

    # Constraint configuration
    parser.add_argument('--free-param', choices=['v', 'M_shell', 't', 'R'],
                        default=None, help='Parameter to leave free (not constrained)')
    parser.add_argument('--no-Mstar', action='store_true',
                        help='Do not constrain stellar mass')
    parser.add_argument('--no-v', action='store_true',
                        help='Do not constrain velocity')
    parser.add_argument('--no-M', action='store_true',
                        help='Do not constrain shell mass')
    parser.add_argument('--no-t', action='store_true',
                        help='Do not constrain age')
    parser.add_argument('--constrain-R', action='store_true',
                        help='Also constrain shell radius (not constrained by default)')

    # Custom observational values
    parser.add_argument('--v-obs', type=float, default=13.0,
                        help='Observed velocity [km/s] (default: 13.0)')
    parser.add_argument('--v-err', type=float, default=2.0,
                        help='Velocity uncertainty [km/s] (default: 2.0)')
    parser.add_argument('--M-obs', type=float, default=2000.0,
                        help='Observed shell mass [M_sun] (default: 2000.0)')
    parser.add_argument('--M-err', type=float, default=500.0,
                        help='Shell mass uncertainty [M_sun] (default: 500.0)')
    parser.add_argument('--t-obs', type=float, default=0.2,
                        help='Observed age [Myr] (default: 0.2)')
    parser.add_argument('--t-err', type=float, default=0.05,
                        help='Age uncertainty [Myr] (default: 0.05)')
    parser.add_argument('--R-obs', type=float, default=4.0,
                        help='Observed radius [pc] (default: 4.0)')
    parser.add_argument('--R-err', type=float, default=0.5,
                        help='Radius uncertainty [pc] (default: 0.5)')
    parser.add_argument('--Mstar', type=float, default=34.0,
                        help='Target stellar mass [M_sun] (default: 34.0)')
    parser.add_argument('--Mstar-err', type=float, default=5.0,
                        help='Stellar mass uncertainty [M_sun] (default: 5.0)')

    args = parser.parse_args()

    # Build observational constraints
    obs = ObservationalConstraints(
        v_obs=args.v_obs, v_err=args.v_err,
        M_shell_obs=args.M_obs, M_shell_err=args.M_err,
        t_obs=args.t_obs, t_err=args.t_err,
        R_obs=args.R_obs, R_err=args.R_err,
        Mstar_obs=args.Mstar, Mstar_err=args.Mstar_err,
    )

    # If nCore specified, force 2D mode
    mode = '2d' if args.nCore else args.mode

    config = AnalysisConfig(
        mode=mode,
        constrain_v=not args.no_v,
        constrain_M_shell=not args.no_M,
        constrain_t=not args.no_t,
        constrain_R=args.constrain_R,
        constrain_Mstar=not args.no_Mstar,
        free_param=args.free_param,
        nCore_filter=args.nCore,
        obs=obs,
    )

    main(args.folder, args.output_dir, config)
